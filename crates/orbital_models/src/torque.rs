//! Attitude perturbation torques — generic forms parameterized over spacecraft configuration.
//!
//! These functions carry no spacecraft-specific constants.  Configuration values
//! (inertia tensor, gravitational parameters) are passed as arguments so that any
//! mission can call them without specialization.

use nalgebra::{Vector3, Vector4};
use crate::attitude::inertial_to_body;

/// Gravity gradient torque in the body frame [N·m].
///
/// τ_gg = (3μ/r³) · (r̂_body × I · r̂_body)
///
/// At 1.5 km from Bennu this is ~1.3 μN·m — small but present.  At lower altitudes
/// or around bodies with stronger gravity it grows as 1/r³.
///
/// # Arguments
/// * `q`          – Attitude quaternion [w,x,y,z] (body → inertial/Hill)
/// * `r_hill`     – Spacecraft position in Hill frame (relative to central body) [m]
/// * `mu`         – Gravitational parameter of the central body [m³/s²]
/// * `inertia`    – Diagonal inertia tensor `[Ixx, Iyy, Izz]` [kg·m²]
pub fn gravity_gradient(
    q:       &Vector4<f64>,
    r_hill:  &Vector3<f64>,
    mu:      f64,
    inertia: &Vector3<f64>,
) -> Vector3<f64> {
    let r2 = r_hill.norm_squared();
    let r  = r2.sqrt();
    let r3 = r2 * r;

    let r_hat_body = inertial_to_body(q, &(r_hill / r));
    let ir = Vector3::new(
        inertia[0] * r_hat_body[0],
        inertia[1] * r_hat_body[1],
        inertia[2] * r_hat_body[2],
    );
    (3.0 * mu / r3) * r_hat_body.cross(&ir)
}

/// Worst-case gravity-gradient disturbance torque magnitude [N·m], independent
/// of attitude — for preliminary actuator sizing before an attitude history exists.
///
/// τ_max = (3μ)/(2r³) · (I_max − I_min)
///
/// Reached when the body's extreme-inertia principal axis sits at 45° to the
/// local vertical (Wertz, *Spacecraft Attitude Determination and Control*, ch. 17).
///
/// # Arguments
/// * `mu`      – Gravitational parameter of the central body [m³/s²]
/// * `r`       – Orbit radius [m]
/// * `inertia` – Principal moments of inertia `[Ixx, Iyy, Izz]` [kg·m²]
pub fn gravity_gradient_max(mu: f64, r: f64, inertia: &Vector3<f64>) -> f64 {
    let i_max = inertia.iter().cloned().fold(f64::MIN, f64::max);
    let i_min = inertia.iter().cloned().fold(f64::MAX, f64::min);
    (3.0 * mu / (2.0 * r * r * r)) * (i_max - i_min)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Apophis 500 m orbit, Bennu-class bus inertia — hand-computed reference:
    /// (3·2.646e-3)/(2·500³)·(666.67−366.67) = 9.5256e-9 N·m.
    #[test]
    fn gravity_gradient_max_apophis_500m_orbit() {
        let mu = 2.646e-3;
        let r = 500.0;
        let inertia = Vector3::new(366.67, 366.67, 666.67);
        let tau = gravity_gradient_max(mu, r, &inertia);
        assert!(
            (tau - 9.5256e-9).abs() < 1.0e-12,
            "expected ~9.5256e-9 N·m, got {tau:e}"
        );
    }

    #[test]
    fn gravity_gradient_max_zero_for_spherical_inertia() {
        let inertia = Vector3::new(100.0, 100.0, 100.0);
        assert_eq!(gravity_gradient_max(1.0, 1000.0, &inertia), 0.0);
    }
}
