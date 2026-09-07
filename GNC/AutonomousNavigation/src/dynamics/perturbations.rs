//! Rotational perturbation torques acting on the spacecraft body.
//!
//! All torques are computed in the body frame and added to the Euler equations
//! alongside the reaction wheel and RCS control torques.
//!
//! SRP torque now comes from the flat-plate model (see `srp.rs::srp_torque_panels`),
//! which uses the per-plate lever arm r_i × F_i for physically correct
//! attitude-dependent torque.  The old cannonball SRP torque (using COP_OFFSET_BODY)
//! has been removed — it was physically inconsistent with the flat-plate force model.

use nalgebra::{Vector3, Vector4};
use orbital_models::torque::gravity_gradient;
use crate::config::{MU_BENNU, SC_INERTIA_DIAG};

/// Gravity gradient torque on the spacecraft body [N·m].
///
/// τ_gg = (3μ/r³) · (r̂_body × I · r̂_body)
///
/// At r = 1.5 km from Bennu this is ~1.3 μN·m — small but non-zero.
/// Delegates to the generic `orbital_models::torque::gravity_gradient`.
///
/// `q`      = attitude quaternion (body → Hill/inertial),
/// `r_hill` = spacecraft position w.r.t. Bennu in Hill frame [m].
pub fn gravity_gradient_torque(
    q:      &Vector4<f64>,
    r_hill: &Vector3<f64>,
) -> Vector3<f64> {
    let inertia = Vector3::new(SC_INERTIA_DIAG[0], SC_INERTIA_DIAG[1], SC_INERTIA_DIAG[2]);
    gravity_gradient(q, r_hill, MU_BENNU, &inertia)
}
