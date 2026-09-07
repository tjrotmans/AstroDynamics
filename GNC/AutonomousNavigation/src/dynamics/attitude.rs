//! Attitude dynamics — Euler's rotational equations and quaternion kinematics.
//!
//! Generic quaternion and attitude functions are re-exported from `orbital_models::attitude`.
//! Only the spacecraft-specific wrapper (`omega_dot` with hard-wired `SC_INERTIA_DIAG`)
//! lives here.

use nalgebra::Vector3;
use crate::config::SC_INERTIA_DIAG;

pub use orbital_models::attitude::{
    qdot, qnorm, quat_multiply,
    body_to_inertial, inertial_to_body,
    align_x_with as nadir_pointing_quat,
    omega_dot as omega_dot_generic,
    rot_to_quat,
};

/// Spacecraft inertia tensor (diagonal) [kg·m²].
#[inline]
pub fn inertia() -> Vector3<f64> {
    Vector3::new(SC_INERTIA_DIAG[0], SC_INERTIA_DIAG[1], SC_INERTIA_DIAG[2])
}

/// Angular acceleration ω̇ using the spacecraft's inertia tensor [rad/s²].
///
/// Thin mission-specific wrapper around `orbital_models::attitude::omega_dot`
/// that supplies `SC_INERTIA_DIAG` from config so callers don't have to pass it.
pub fn omega_dot(
    omega:   &Vector3<f64>,
    torque:  &Vector3<f64>,
    h_wheel: &Vector3<f64>,
) -> Vector3<f64> {
    let i = inertia();
    omega_dot_generic(omega, torque, h_wheel, &i)
}
