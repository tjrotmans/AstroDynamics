//! Generic 13-state truth propagator — gravity, atmosphere, SRP, and third-body
//! perturbations dispatched from `body_models::TargetBody`, plus rigid-body
//! attitude dynamics with reaction-wheel momentum coupling.
//!
//! State layout mirrors `GNC/AutonomousNavigation/src/dynamics/mod.rs`'s `TruthState`
//! (r, v, q, omega, wheel speeds, C_R) but every physical parameter (mu, J2-J4,
//! inertia, SRP area/Cr) is passed in via [`SpacecraftProperties`]/[`Environment`]
//! instead of imported from a mission `config.rs` — this is the generic
//! equivalent referenced in the Phase 4 sim_engine plan.
//!
//! Wheel speed integration is NOT done here (see `actuators.rs`) — wheel speed
//! dynamics under a zero-order-hold motor torque are exactly linear over one
//! truth step, so they are integrated directly rather than folded into this RK4.

use nalgebra::{Vector3, Vector4};

use body_models::{AtmosphereModel as BodyAtmosphereModel, GravityModel as BodyGravityModel, TargetBody};
use orbital_models::attitude::{omega_dot, qdot, qnorm};
use orbital_models::{cannonball, flat_plate_accel, flat_plate_torque_body, gravity_gradient, pressure_at, Plate};
use orbital_models::GravityModel as AccelGravityModel;

/// Spacecraft SRP model resolved for truth dynamics (built by the caller from
/// mission config — this crate has no knowledge of TOML/hardware-list shapes).
#[derive(Clone, Debug)]
pub enum SrpTruthModel {
    /// Constant effective area, scalar reflectivity, no attitude dependence.
    Cannonball { c_r: f64, area_m2: f64 },
    /// Per-face optical properties; force/torque depend on attitude.
    FlatPlate { plates: Vec<Plate> },
}

/// Spacecraft physical properties needed by the truth propagator.
///
/// Resolved once (per mission) by the caller from `MissionConfig` — this crate
/// never references TOML field names or hardware-list enums.
#[derive(Clone, Debug)]
pub struct SpacecraftProperties {
    pub mass_kg: f64,
    pub inertia_diag_kgm2: Vector3<f64>,
    pub srp: SrpTruthModel,
    /// Cross-sectional area used for atmospheric drag [m^2]. Only consulted
    /// when the target body has `AtmosphereModel::Exponential`.
    pub drag_area_m2: f64,
}

/// Environment inputs that change with time but are not part of the
/// spacecraft's own state — body model, Sun direction, and any additional
/// third-body perturbers (Phase 5 extension point for ANISE-ephemeris
/// Earth/Moon/Mars missions with multiple perturbing bodies).
#[derive(Clone, Debug)]
pub struct Environment {
    pub body: TargetBody,
    /// Body's heliocentric position (Sun -> body vector) [m]. Spacecraft offset
    /// from the body is assumed negligible against this distance (valid for any
    /// proximity-ops regime — km-scale orbit vs. AU-scale heliocentric distance),
    /// the same approximation used by the Bennu reference implementation.
    pub sun_pos_from_body_m: Vector3<f64>,
    pub mu_sun_m3s2: f64,
    /// Additional third-body perturbers as (mu [m^3/s^2], position relative to
    /// the central body [m]). Empty for the validated Keplerian/Hill-frame case.
    pub extra_perturbers: Vec<(f64, Vector3<f64>)>,
}

/// 17-component truth state (r, v, q, omega, 4 wheel speeds, C_R) — same
/// layout as the Bennu reference, generalized to any target body.
#[derive(Clone, Debug)]
pub struct TruthState {
    pub t_s: f64,
    pub r_m: Vector3<f64>,
    pub v_mps: Vector3<f64>,
    /// Attitude quaternion [w, x, y, z], body -> inertial/Hill.
    pub q: Vector4<f64>,
    pub omega_radps: Vector3<f64>,
    pub wheel_speeds_radps: [f64; 4],
    /// True SRP reflectivity coefficient (the quantity the EKF estimates).
    pub c_r: f64,
}

// ── Acceleration dispatch ────────────────────────────────────────────────────

#[inline]
fn point_mass_accel(r: &Vector3<f64>, mu: f64) -> Vector3<f64> {
    let r_norm = r.norm();
    -(mu / (r_norm * r_norm * r_norm)) * r
}

fn zonal_accel(r: &Vector3<f64>, body: &TargetBody) -> Vector3<f64> {
    match &body.gravity {
        BodyGravityModel::PointMass => Vector3::zeros(),
        BodyGravityModel::J2 { j2 } => {
            AccelGravityModel::zonal_harmonics_body(r, body.mu_m3s2, body.radius_m, *j2, 0.0, 0.0)
        }
        BodyGravityModel::J2J3J4 { j2, j3, j4 } => {
            AccelGravityModel::zonal_harmonics_body(r, body.mu_m3s2, body.radius_m, *j2, *j3, *j4)
        }
    }
}

/// Standard satellite drag coefficient for a convex bus shape — Vallado,
/// *Fundamentals of Astrodynamics and Applications*, typical C_D ~2.2 for
/// spacecraft without detailed panel-by-panel aerodynamic modeling.
const DRAG_CD: f64 = 2.2;

fn drag_accel(r: &Vector3<f64>, v: &Vector3<f64>, body: &TargetBody, sc: &SpacecraftProperties) -> Vector3<f64> {
    match &body.atmosphere {
        BodyAtmosphereModel::None => Vector3::zeros(),
        BodyAtmosphereModel::Exponential { scale_height_m, rho0_kg_m3 } => {
            let alt = r.norm() - body.radius_m;
            let rho = rho0_kg_m3 * (-alt / scale_height_m).exp();
            // Co-rotating atmosphere about the body's spin axis (+z by convention).
            let omega_body = Vector3::new(0.0, 0.0, body.spin_rate_rads);
            let v_rel = v - omega_body.cross(r);
            let v_rel_norm = v_rel.norm();
            if v_rel_norm < 1e-9 {
                return Vector3::zeros();
            }
            -0.5 * rho * DRAG_CD * sc.drag_area_m2 / sc.mass_kg * v_rel_norm * v_rel
        }
    }
}

/// Sun direction unit vector (spacecraft -> Sun) and local radiation pressure,
/// from the body's heliocentric position.
fn sun_hat_and_pressure(env: &Environment) -> (Vector3<f64>, f64) {
    let sun_hat = -env.sun_pos_from_body_m.normalize();
    let p_srp = pressure_at(env.sun_pos_from_body_m.norm());
    (sun_hat, p_srp)
}

/// Net translational acceleration in the body-centered inertial (Hill) frame [m/s^2].
pub fn translational_accel(
    r: &Vector3<f64>,
    v: &Vector3<f64>,
    q: &Vector4<f64>,
    sc: &SpacecraftProperties,
    env: &Environment,
) -> Vector3<f64> {
    let mut a = point_mass_accel(r, env.body.mu_m3s2);
    a += zonal_accel(r, &env.body);
    a += AccelGravityModel::tidal(r, &env.sun_pos_from_body_m, env.mu_sun_m3s2);
    for &(mu, pos) in &env.extra_perturbers {
        a += AccelGravityModel::third_body(r, &pos, mu);
    }

    // sun_pos_from_body_m = body heliocentric position = FROM Sun TO body (anti-sun vector).
    // cannonball() takes the anti-sun direction; flat_plate_accel() takes the spacecraft→Sun
    // (toward-sun) direction.  These are opposites — derive each from the same stored vector.
    let body_helio = env.sun_pos_from_body_m;
    let p_srp = pressure_at(body_helio.norm());
    a += match &sc.srp {
        SrpTruthModel::Cannonball { c_r, area_m2 } => {
            let anti_sun = body_helio.normalize();
            cannonball(&anti_sun, p_srp, *c_r, *area_m2, sc.mass_kg)
        }
        SrpTruthModel::FlatPlate { plates } => {
            let toward_sun = -body_helio.normalize();
            flat_plate_accel(plates, q, &toward_sun, p_srp, sc.mass_kg)
        }
    };

    a += drag_accel(r, v, &env.body, sc);
    a
}

/// Disturbance torque in the body frame [N*m]: gravity-gradient + SRP (flat-plate
/// only — the cannonball model has no plate geometry, hence no valid SRP torque).
/// Control torque (wheel reaction, RCS) is added by the caller, not here.
pub fn disturbance_torque_body(
    state: &TruthState,
    sc: &SpacecraftProperties,
    env: &Environment,
) -> Vector3<f64> {
    let tau_gg = gravity_gradient(&state.q, &state.r_m, env.body.mu_m3s2, &sc.inertia_diag_kgm2);
    let tau_srp = match &sc.srp {
        SrpTruthModel::Cannonball { .. } => Vector3::zeros(),
        SrpTruthModel::FlatPlate { plates } => {
            let (sun_hat, p_srp) = sun_hat_and_pressure(env);
            let sun_hat_body = orbital_models::attitude::inertial_to_body(&state.q, &sun_hat);
            flat_plate_torque_body(plates, &sun_hat_body, p_srp)
        }
    };
    tau_gg + tau_srp
}

// ── Coupled RK4 step (r, v, q, omega) ────────────────────────────────────────

type Deriv = (Vector3<f64>, Vector3<f64>, Vector4<f64>, Vector3<f64>);

fn derivative(
    r: &Vector3<f64>,
    v: &Vector3<f64>,
    q: &Vector4<f64>,
    omega: &Vector3<f64>,
    sc: &SpacecraftProperties,
    env: &Environment,
    torque_body: &Vector3<f64>,
    h_wheel_body: &Vector3<f64>,
) -> Deriv {
    let a = translational_accel(r, v, q, sc, env);
    let qd = qdot(q, omega);
    let od = omega_dot(omega, torque_body, h_wheel_body, &sc.inertia_diag_kgm2);
    (*v, a, qd, od)
}

/// Advance the truth state by `dt` seconds using fixed-step RK4 on the coupled
/// (r, v, q, omega) dynamics. `torque_body` (control + reaction-wheel torque)
/// and `h_wheel_body` (wheel angular momentum) are held fixed over the step —
/// a zero-order-hold approximation standard for a step this short (~10 s)
/// relative to attitude/orbit timescales. Wheel speeds and C_R are carried
/// through unchanged; the caller integrates wheel speeds separately
/// (see `actuators.rs`) and C_R has no truth dynamics (kept for EKF reference).
pub fn propagate_step(
    state: &TruthState,
    dt: f64,
    sc: &SpacecraftProperties,
    env: &Environment,
    torque_body: &Vector3<f64>,
    h_wheel_body: &Vector3<f64>,
) -> TruthState {
    let f = |r: &Vector3<f64>, v: &Vector3<f64>, q: &Vector4<f64>, omega: &Vector3<f64>| {
        derivative(r, v, q, omega, sc, env, torque_body, h_wheel_body)
    };

    let (r0, v0, q0, w0) = (state.r_m, state.v_mps, state.q, state.omega_radps);

    let k1 = f(&r0, &v0, &q0, &w0);
    let (r1, v1, q1, w1) = (
        r0 + k1.0 * (dt / 2.0), v0 + k1.1 * (dt / 2.0),
        q0 + k1.2 * (dt / 2.0), w0 + k1.3 * (dt / 2.0),
    );
    let k2 = f(&r1, &v1, &q1, &w1);
    let (r2, v2, q2, w2) = (
        r0 + k2.0 * (dt / 2.0), v0 + k2.1 * (dt / 2.0),
        q0 + k2.2 * (dt / 2.0), w0 + k2.3 * (dt / 2.0),
    );
    let k3 = f(&r2, &v2, &q2, &w2);
    let (r3, v3, q3, w3) = (r0 + k3.0 * dt, v0 + k3.1 * dt, q0 + k3.2 * dt, w0 + k3.3 * dt);
    let k4 = f(&r3, &v3, &q3, &w3);

    let r_new = r0 + (dt / 6.0) * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
    let v_new = v0 + (dt / 6.0) * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
    let q_new = qnorm(&(q0 + (dt / 6.0) * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2)));
    let w_new = w0 + (dt / 6.0) * (k1.3 + 2.0 * k2.3 + 2.0 * k3.3 + k4.3);

    TruthState {
        t_s: state.t_s + dt,
        r_m: r_new,
        v_mps: v_new,
        q: q_new,
        omega_radps: w_new,
        wheel_speeds_radps: state.wheel_speeds_radps,
        c_r: state.c_r,
    }
}
