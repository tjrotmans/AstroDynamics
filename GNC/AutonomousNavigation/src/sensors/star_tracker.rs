//! Star tracker measurement model.
//!
//! Outputs a noisy quaternion representing the spacecraft attitude.
//! Noise is modelled as a small rotation error about each body axis.

use nalgebra::Vector4;
use rand_distr::{Distribution, Normal};
use crate::config::STAR_TRACKER_SIGMA_RAD;
use crate::dynamics::attitude::quat_multiply;

/// Noisy attitude quaternion from the star tracker.
///
/// Adds independent Gaussian angle errors about each body axis
/// then composes them onto the true quaternion.
pub fn measure<R: rand::Rng>(
    q_true: &Vector4<f64>,
    rng:    &mut R,
) -> Vector4<f64> {
    let dist = Normal::new(0.0, STAR_TRACKER_SIGMA_RAD).unwrap();
    let droll  = dist.sample(rng);
    let dpitch = dist.sample(rng);
    let dyaw   = dist.sample(rng);

    // Small-angle error quaternion: q_err ≈ [1, δ/2] for tiny δ
    let q_err = Vector4::new(
        1.0,
        0.5 * droll,
        0.5 * dpitch,
        0.5 * dyaw,
    );
    let n = q_err.norm();
    let q_err = q_err / n;

    // q_meas = q_true ⊗ q_err
    quat_multiply(q_true, &q_err)
}
