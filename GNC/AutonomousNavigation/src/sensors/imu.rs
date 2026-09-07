//! IMU / accelerometer measurement model.
//!
//! During an impulsive manoeuvre the IMU integrates specific force over the
//! burn duration to produce ΔV in the spacecraft body frame.
//! We model this as the true ΔV vector (in Hill/inertial frame) rotated to
//! body frame, then corrupted by Gaussian noise on each axis.

use nalgebra::Vector3;
use rand_distr::{Distribution, Normal};
use crate::config::IMU_DV_SIGMA_MPS;
use crate::dynamics::attitude::inertial_to_body;
use nalgebra::Vector4;

/// Simulate an IMU ΔV measurement.
///
/// `dv_inertial` = true ΔV in Hill/inertial frame [m/s].
/// `q_true`      = true spacecraft attitude at manoeuvre epoch.
///
/// Returns the measured ΔV in the **body** frame.
pub fn measure_dv<R: rand::Rng>(
    dv_inertial: &Vector3<f64>,
    q_true:      &Vector4<f64>,
    rng:         &mut R,
) -> Vector3<f64> {
    let dist = Normal::new(0.0, IMU_DV_SIGMA_MPS).unwrap();

    // True ΔV in body frame
    let dv_body = inertial_to_body(q_true, dv_inertial);

    // Add noise per axis
    dv_body + Vector3::new(
        dist.sample(rng),
        dist.sample(rng),
        dist.sample(rng),
    )
}

/// Convert a body-frame ΔV measurement back to inertial frame using a
/// (possibly noisy) attitude estimate.
///
/// The EKF calls this to get the inertial-frame residual during an update.
pub fn dv_body_to_inertial(
    dv_body: &Vector3<f64>,
    q_est:   &Vector4<f64>,
) -> Vector3<f64> {
    use crate::dynamics::attitude::body_to_inertial;
    body_to_inertial(q_est, dv_body)
}
