//! Translational equations of motion in the Hill (orbit-fixed) frame centred on Bennu.
//!
//! Dominant terms:
//!   - Solar tidal / Hill acceleration (Sun gravity gradient relative to Bennu)
//!   - Bennu zonal gravity through degree 4 (J2, J3, J4) — filter and truth alike
//!   - Solar radiation pressure (cannonball)
//!   - Thrust force from RCS (in body frame, rotated to Hill frame)

use nalgebra::{Vector3, Vector4};
use orbital_models::constants::MU_SUN;
use orbital_models::{GravityModel, Plate};
use orbital_models::acceleration::{pressure_at, cannonball};
use crate::config::{SC_MASS, SC_AREA_CANNONBALL};
use super::bennu::zonal_gravity;
use super::srp::srp_accel_panels;

// ── Filter-model EOM (point mass + cannonball SRP) ────────────────────────────

/// Translational acceleration in Hill frame for the EKF predict step.
///
/// `r_sc` = spacecraft position w.r.t. Bennu [m],
/// `v_sc` = spacecraft velocity w.r.t. Bennu [m/s],
/// `bennu_pos` = Bennu heliocentric position [m],
/// `c_r` = estimated reflectivity coefficient.
pub fn accel_filter(
    r_sc:      &Vector3<f64>,
    bennu_pos: &Vector3<f64>,
    c_r:        f64,
) -> Vector3<f64> {
    let a_grav  = zonal_gravity(r_sc);
    let a_tidal = tidal_accel(r_sc, bennu_pos);
    let a_srp   = srp_cannonball_local(bennu_pos, c_r, SC_AREA_CANNONBALL, SC_MASS);
    a_grav + a_tidal + a_srp
}

// ── Truth-model EOM (J2 gravity + flat-plate SRP + RCS thrust) ───────────────

/// Translational acceleration for truth propagation.
///
/// Uses the attitude-dependent multi-panel SRP model — the realistic force the
/// filter's cannonball + Gauss-Markov stochastic accel must track.
///
/// `q`               = attitude quaternion (body → Hill), drives panel SRP,
/// `plates`          = spacecraft flat-plate model,
/// `thrust_inertial` = net RCS force vector in inertial/Hill frame [N].
pub fn accel_truth(
    r_sc:           &Vector3<f64>,
    bennu_pos:      &Vector3<f64>,
    q:              &Vector4<f64>,
    plates:         &[Plate],
    thrust_inertial: &Vector3<f64>,
) -> Vector3<f64> {
    let a_grav  = zonal_gravity(r_sc);
    let a_tidal = tidal_accel(r_sc, bennu_pos);
    let a_srp   = srp_accel_panels(plates, q, bennu_pos, SC_MASS);
    let a_thrust = thrust_inertial / SC_MASS;
    a_grav + a_tidal + a_srp + a_thrust
}

// ── Private helpers ───────────────────────────────────────────────────────────

#[inline]
fn tidal_accel(r_sc: &Vector3<f64>, bennu_pos: &Vector3<f64>) -> Vector3<f64> {
    GravityModel::tidal(r_sc, bennu_pos, MU_SUN)
}

fn srp_cannonball_local(bennu_pos: &Vector3<f64>, c_r: f64, area: f64, mass: f64) -> Vector3<f64> {
    // Sun→spacecraft direction ≈ Sun→Bennu (error < 10 km / 1 AU ≈ 1e-7 rad)
    let sun_hat = bennu_pos / bennu_pos.norm();
    let p_srp   = pressure_at(bennu_pos.norm());
    cannonball(&sun_hat, p_srp, c_r, area, mass)
}
