//! Truth-model state propagator — 6DOF + reaction wheels, RK4 integration.
//!
//! State vector: [r(3), v(3), q(4), ω(3), Ω_wheel(4)] = 17 states.
//!
//! Attitude control: continuous reaction wheel PD (primary).
//! Desaturation:     bang-bang RCS couples (secondary, on-demand).
//! Translation:      dedicated thrusters at CoM via step_with_force().

pub mod bennu;
pub mod hill;
pub mod attitude;
pub mod rcs;
pub mod perturbations;
pub mod srp;

use nalgebra::{Vector3, Vector4};
use orbital_models::Plate;
use crate::config::{C_R_NOMINAL, RCS_PULSE_DURATION_S};
use self::attitude::{qdot, omega_dot, qnorm, body_to_inertial};
use self::hill::accel_truth;
use self::bennu::bennu_heliocentric_pos;
use self::rcs::{Thruster, PdGains, pd_torque, thruster_selection, rcs_thrusters};
use self::perturbations::gravity_gradient_torque;
use self::srp::{spacecraft_plates, srp_torque_panels};
use crate::actuators::reaction_wheels as rw;
use crate::guidance::pointing::{PointingMode, desired_quaternion};

// ── Full 6DOF + wheel truth state ─────────────────────────────────────────────

/// Complete spacecraft truth state at one instant.
#[derive(Clone, Debug)]
pub struct TruthState {
    /// Time [s]
    pub t: f64,
    /// Position w.r.t. Bennu in Hill frame [m]
    pub r: Vector3<f64>,
    /// Velocity w.r.t. Bennu in Hill frame [m/s]
    pub v: Vector3<f64>,
    /// Attitude quaternion [w, x, y, z]  (body → Hill/inertial)
    pub q: Vector4<f64>,
    /// Angular rate in body frame [rad/s]
    pub omega: Vector3<f64>,
    /// True C_SRP (reflectivity)
    pub c_r: f64,
    /// Camera boresight (+x body axis) expressed in Hill/inertial frame
    pub boresight: Vector3<f64>,
    /// Reaction wheel speeds [rad/s], signed (positive = CCW along ŵ_i)
    pub wheel_speeds: [f64; 4],
}

impl TruthState {
    /// Build initial state from config defaults.
    pub fn from_config() -> Self {
        use crate::config::{sc_r0, sc_v0, sc_q0, sc_omega0};
        let q0 = sc_q0();
        let r0 = sc_r0();
        let q  = Vector4::new(q0[0], q0[1], q0[2], q0[3]);
        let boresight = body_to_inertial(&q, &Vector3::new(1.0, 0.0, 0.0));
        Self {
            t: 0.0, r: r0, v: sc_v0(), q, omega: sc_omega0(),
            c_r: C_R_NOMINAL, boresight, wheel_speeds: [0.0; 4],
        }
    }

    /// Build initial state from a cruise-to-proximity handoff.
    pub fn from_handoff(r: Vector3<f64>, v: Vector3<f64>, t_arr: f64) -> Self {
        use crate::config::{sc_q0, sc_omega0};
        let q0 = sc_q0();
        let q  = Vector4::new(q0[0], q0[1], q0[2], q0[3]);
        let boresight = body_to_inertial(&q, &Vector3::new(1.0, 0.0, 0.0));
        Self {
            t: t_arr, r, v, q, omega: sc_omega0(),
            c_r: C_R_NOMINAL, boresight, wheel_speeds: [0.0; 4],
        }
    }

    /// Build initial state with an explicit attitude quaternion and wheel speeds.
    pub fn from_state(
        r: Vector3<f64>, v: Vector3<f64>,
        q: Vector4<f64>, omega: Vector3<f64>,
        t: f64,
    ) -> Self {
        let boresight = body_to_inertial(&q, &Vector3::new(1.0, 0.0, 0.0));
        Self {
            t, r, v, q, omega,
            c_r: C_R_NOMINAL, boresight, wheel_speeds: [0.0; 4],
        }
    }

    /// Total reaction wheel angular momentum in body frame [N·m·s].
    pub fn wheel_momentum(&self) -> Vector3<f64> {
        rw::total_momentum(&self.wheel_speeds)
    }

    /// Speed of the fastest wheel as a fraction of `WHEEL_MAX_SPEED_RADS`.
    pub fn wheel_saturation_fraction(&self) -> f64 {
        use crate::config::WHEEL_MAX_SPEED_RADS;
        self.wheel_speeds.iter()
            .map(|s| s.abs() / WHEEL_MAX_SPEED_RADS)
            .fold(0.0_f64, f64::max)
    }
}

// ── RCS step output ───────────────────────────────────────────────────────────

/// Actuator outputs sampled at k1 of the propagation step (for logging).
pub struct RcsStep {
    /// Net torque from all firing attitude thrusters [N·m, body frame]
    pub torque_body:      Vector3<f64>,
    /// Torque actually delivered to the body from the wheels [N·m, body frame]
    pub wheel_torque:     Vector3<f64>,
    /// Sum of all firing thruster forces (unsigned) [N] — for ΔV accounting.
    pub total_thrust_n:   f64,
    /// Translational ΔV applied this step [m/s, Hill frame]
    pub dv_trans_ms:      Vector3<f64>,
    /// Whether desaturation RCS fired this step
    pub desat_active:     bool,
    /// PD commanded torque to reaction wheels [N·m, body frame]
    pub tau_pd:           Vector3<f64>,
    /// SRP perturbation torque [N·m, body frame]
    pub tau_srp:          Vector3<f64>,
    /// Gravity gradient perturbation torque [N·m, body frame]
    pub tau_gg:           Vector3<f64>,
}

// ── PD gains for reaction wheel control ──────────────────────────────────────

/// PD gains tuned for continuous reaction wheel torque output.
///
/// Lower kp than the bang-bang RCS gains because the wheels can deliver
/// fine continuous torque; high kd ensures overdamping for proximity science.
fn wheel_pd_gains() -> PdGains {
    // Dead-bands are zero so wheels run continuously (no bang-bang chatter suppression).
    PdGains { kp: 0.05, kd: 0.6, pointing_db_rad: 0.0, rate_db_rads: 0.0 }
}

// ── Propagator ────────────────────────────────────────────────────────────────

/// Holds runtime state for truth propagation.
pub struct Propagator {
    att_thrusters: Vec<Thruster>,
    wheel_gains:   PdGains,
    /// Spacecraft flat-plate model for attitude-dependent truth SRP.
    plates:        Vec<Plate>,
    /// Integration step size [s].
    pub dt:        f64,
}

impl Propagator {
    /// Create a propagator with the given truth integration step size [s].
    pub fn new(dt: f64) -> Self {
        Self {
            att_thrusters: rcs_thrusters(),
            wheel_gains:   wheel_pd_gains(),
            plates:        spacecraft_plates(),
            dt,
        }
    }

    // ── Attitude-only step (nadir by default) ─────────────────────────────────

    /// Advance by one `self.dt` step with nadir pointing and no translation.
    pub fn step(&self, s: &TruthState) -> (TruthState, RcsStep) {
        self.step_with_force(s, None, PointingMode::Nadir)
    }

    // ── Step with optional Hill-frame translational force ────────────────────

    /// Advance by `self.dt` with optional external translational force [N]
    /// in the Hill (inertial) frame, and a commanded pointing mode.
    ///
    /// The translational force acts through the CoM (no attitude torque).
    /// Attitude control runs concurrently via the reaction wheels.
    pub fn step_with_force(
        &self,
        s:           &TruthState,
        force_hill:  Option<Vector3<f64>>,
        mode:        PointingMode,
    ) -> (TruthState, RcsStep) {
        let dt     = self.dt;
        let f_hill = force_hill.unwrap_or(Vector3::zeros());
        let q_cmd  = desired_quaternion(mode, &s.r, &s.v);

        // Pre-compute Bennu position once — it barely moves within one step.
        // Sun direction changes ~1e-7 rad/s; error over 60 s is < 0.1 μrad.
        let bennu_pos = bennu_heliocentric_pos(s.t);

        // ── Actuator sample at k1 for logging ─────────────────────────────────
        let tau_pd_k1   = pd_torque(&s.q, &q_cmd, &s.omega, &self.wheel_gains);
        let tau_motor_k1 = rw::allocate(&tau_pd_k1);
        let wheel_torque_k1 = rw::body_torque(&tau_motor_k1);

        let (desat_fire, desat_torque_k1) = if rw::needs_desat(&s.wheel_speeds) {
            let td = rw::desat_torque(&s.wheel_speeds).unwrap_or(Vector3::zeros());
            let (_, tau_rcs, _) = thruster_selection(&td, &self.att_thrusters);
            (true, tau_rcs * (RCS_PULSE_DURATION_S / dt))
        } else {
            (false, Vector3::zeros())
        };

        let _total_torque_k1 = wheel_torque_k1 + desat_torque_k1;
        let pulse_scale = RCS_PULSE_DURATION_S / dt;
        let (_force_att_k1, _, thrust_k1) = thruster_selection(&desat_torque_k1, &self.att_thrusters);
        let total_thrust_k1 = thrust_k1 * pulse_scale;

        // Perturbation torques at k1 (for logging only)
        let tau_srp_k1 = srp_torque_panels(&self.plates, &s.q, &bennu_pos);
        let tau_gg_k1  = gravity_gradient_torque(&s.q, &s.r);

        // ── RK4 ───────────────────────────────────────────────────────────────
        let k1 = self.derivatives(s, &q_cmd, &f_hill, &bennu_pos);
        let s2 = apply_derivs(s, &k1, dt * 0.5);
        let k2 = self.derivatives(&s2, &q_cmd, &f_hill, &bennu_pos);
        let s3 = apply_derivs(s, &k2, dt * 0.5);
        let k3 = self.derivatives(&s3, &q_cmd, &f_hill, &bennu_pos);
        let s4 = apply_derivs(s, &k3, dt);
        let k4 = self.derivatives(&s4, &q_cmd, &f_hill, &bennu_pos);

        let r     = s.r     + dt / 6.0 * (k1.0 + 2.0*k2.0 + 2.0*k3.0 + k4.0);
        let v     = s.v     + dt / 6.0 * (k1.1 + 2.0*k2.1 + 2.0*k3.1 + k4.1);
        let q     = qnorm(&(s.q + dt / 6.0 * (k1.2 + 2.0*k2.2 + 2.0*k3.2 + k4.2)));
        let omega = s.omega + dt / 6.0 * (k1.3 + 2.0*k2.3 + 2.0*k3.3 + k4.3);

        // Wheel speed integration (simple RK4 slice: dΩ/dt terms in k.4)
        let mut wheel_speeds = [0.0_f64; 4];
        for i in 0..4 {
            wheel_speeds[i] = s.wheel_speeds[i]
                + dt / 6.0 * (k1.4[i] + 2.0*k2.4[i] + 2.0*k3.4[i] + k4.4[i]);
        }

        let boresight = body_to_inertial(&q, &Vector3::new(1.0, 0.0, 0.0));
        let dv_hill   = f_hill * dt / crate::config::SC_MASS;

        (
            TruthState { t: s.t + dt, r, v, q, omega, c_r: s.c_r, boresight, wheel_speeds },
            RcsStep {
                torque_body:  desat_torque_k1,
                wheel_torque: wheel_torque_k1,
                total_thrust_n: total_thrust_k1,
                dv_trans_ms:  dv_hill,
                desat_active: desat_fire,
                tau_pd:   tau_pd_k1,
                tau_srp:  tau_srp_k1,
                tau_gg:   tau_gg_k1,
            },
        )
    }

    /// Compute derivatives (ṙ, v̇, q̇, ω̇, Ω̇_wheels).
    fn derivatives(
        &self,
        s:          &TruthState,
        q_cmd:      &Vector4<f64>,
        force_hill: &Vector3<f64>,
        bennu_pos:  &Vector3<f64>,
    ) -> Derivs {

        // ── Reaction wheel control torque ─────────────────────────────────────
        let tau_pd      = pd_torque(&s.q, q_cmd, &s.omega, &self.wheel_gains);
        let tau_motor   = rw::allocate(&tau_pd);
        let tau_rw_body = rw::body_torque(&tau_motor);
        let wheel_sdot  = rw::speed_dots(&tau_motor);

        // ── Desaturation RCS ──────────────────────────────────────────────────
        let pulse_scale = RCS_PULSE_DURATION_S / self.dt;
        let (force_desat_body, tau_desat_body, _) =
            if rw::needs_desat(&s.wheel_speeds) {
                let td = rw::desat_torque(&s.wheel_speeds).unwrap_or(Vector3::zeros());
                thruster_selection(&td, &self.att_thrusters)
            } else {
                (Vector3::zeros(), Vector3::zeros(), 0.0)
            };
        let tau_desat_body  = tau_desat_body  * pulse_scale;
        let force_desat_hill = body_to_inertial(&s.q, &(force_desat_body * pulse_scale));

        // ── Perturbation torques ───────────────────────────────────────────────
        let tau_srp = srp_torque_panels(&self.plates, &s.q, &bennu_pos);
        let tau_gg  = gravity_gradient_torque(&s.q, &s.r);

        // ── Total torque on body ───────────────────────────────────────────────
        let h_w         = rw::total_momentum(&s.wheel_speeds);
        let total_torque = tau_rw_body + tau_desat_body + tau_srp + tau_gg;

        // ── Translational equations ───────────────────────────────────────────
        let total_force_hill = force_hill + force_desat_hill;

        let rdot  = s.v;
        let vdot  = accel_truth(&s.r, &bennu_pos, &s.q, &self.plates, &total_force_hill);
        let qdot_ = qdot(&s.q, &s.omega);
        let omdot = omega_dot(&s.omega, &total_torque, &h_w);

        (rdot, vdot, qdot_, omdot, wheel_sdot)
    }
}

// ── Derivative type alias ─────────────────────────────────────────────────────

type Derivs = (Vector3<f64>, Vector3<f64>, Vector4<f64>, Vector3<f64>, [f64; 4]);

// ── Helper: Euler half-step for RK4 stages ───────────────────────────────────

fn apply_derivs(s: &TruthState, d: &Derivs, dt: f64) -> TruthState {
    let q         = qnorm(&(s.q + dt * d.2));
    let boresight = body_to_inertial(&q, &Vector3::new(1.0, 0.0, 0.0));
    let mut wheel_speeds = [0.0_f64; 4];
    for i in 0..4 {
        wheel_speeds[i] = s.wheel_speeds[i] + dt * d.4[i];
    }
    TruthState {
        t: s.t + dt,
        r: s.r + dt * d.0,
        v: s.v + dt * d.1,
        q,
        omega: s.omega + dt * d.3,
        c_r: s.c_r,
        boresight,
        wheel_speeds,
    }
}
