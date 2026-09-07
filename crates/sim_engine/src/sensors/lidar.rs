//! Generic LIDAR altimeter: slant range to the body surface, with a maximum
//! operating range cutoff.

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Returns `Some(measured_range_m)` (surface-relative slant range) when within
/// `max_range_m`, else `None` (sensor has no return).
pub fn measure<R: Rng>(
    r_sc: &Vector3<f64>,
    body_radius_m: f64,
    sigma_m: f64,
    max_range_m: f64,
    rng: &mut R,
) -> Option<f64> {
    let true_range = r_sc.norm() - body_radius_m;
    if true_range > max_range_m {
        return None;
    }
    let dist = Normal::new(0.0, sigma_m).unwrap();
    Some(true_range + dist.sample(rng))
}
