//! Generic star-tracker attitude measurement: small-angle Gaussian noise per axis.

use nalgebra::Vector4;
use orbital_models::attitude::quat_multiply;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// `q_true` = true attitude quaternion [w,x,y,z]. `sigma_rad` = 1-sigma noise
/// per axis (`hardware_catalog::StarTrackerSpec::noise_rad`).
pub fn measure<R: Rng>(q_true: &Vector4<f64>, sigma_rad: f64, rng: &mut R) -> Vector4<f64> {
    let dist = Normal::new(0.0, sigma_rad).unwrap();
    let (droll, dpitch, dyaw) = (dist.sample(rng), dist.sample(rng), dist.sample(rng));
    // Small-angle quaternion error: q_err ~= [1, delta/2], renormalized.
    let q_err = Vector4::new(1.0, 0.5 * droll, 0.5 * dpitch, 0.5 * dyaw).normalize();
    quat_multiply(q_true, &q_err)
}
