//! Generic IMU delta-V measurement: true delta-V rotated into the body frame
//! plus per-axis Gaussian noise.

use nalgebra::{Vector3, Vector4};
use orbital_models::attitude::inertial_to_body;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// `dv_inertial` = true delta-V in the inertial/Hill frame [m/s] over the step.
/// `q_true` = attitude at the maneuver epoch. `sigma_mps` = 1-sigma noise per
/// axis per step.
pub fn measure_dv<R: Rng>(
    dv_inertial: &Vector3<f64>,
    q_true: &Vector4<f64>,
    sigma_mps: f64,
    rng: &mut R,
) -> Vector3<f64> {
    let dv_body = inertial_to_body(q_true, dv_inertial);
    let dist = Normal::new(0.0, sigma_mps).unwrap();
    dv_body + Vector3::new(dist.sample(rng), dist.sample(rng), dist.sample(rng))
}
