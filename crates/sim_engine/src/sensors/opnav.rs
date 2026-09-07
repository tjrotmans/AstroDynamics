//! Generic optical-navigation measurement: line-of-sight bearing + apparent
//! angular size of the target body.
//!
//! Simplified relative to `GNC/AutonomousNavigation/src/sensors/opnav.rs`'s
//! pixel-rendering centroid model (shot noise vs. angular radius, phase-angle
//! bias correction) — that fidelity is an explicitly deferred fast-follow, not
//! Phase 4 MVP scope. Here bearing noise is an isotropic small-angle
//! perturbation of the true LOS, sized directly from the sensor's bearing
//! noise spec.

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Copy, Debug)]
pub struct OpNavMeas {
    /// Noisy unit line-of-sight, spacecraft -> body center, in the same frame as `r_sc`.
    pub los: Vector3<f64>,
    /// Noisy apparent angular radius of the body disk [rad].
    pub angular_size_rad: f64,
}

/// `r_sc` = true spacecraft position relative to the body center [m].
/// `sigma_bearing_rad` / `sigma_size_rad` = 1-sigma sensor noise
/// (`hardware_catalog::OpNavCameraSpec::bearing_noise_rad` / `angular_size_noise_rad`).
pub fn measure<R: Rng>(
    r_sc: &Vector3<f64>,
    body_radius_m: f64,
    sigma_bearing_rad: f64,
    sigma_size_rad: f64,
    rng: &mut R,
) -> OpNavMeas {
    let range = r_sc.norm();
    let los_true = -r_sc / range;

    let bearing_dist = Normal::new(0.0, sigma_bearing_rad).unwrap();
    let perturb_mag = bearing_dist.sample(rng);
    let perturb = random_perpendicular(&los_true, rng) * perturb_mag;
    let los_meas = (los_true + perturb).normalize();

    let size_dist = Normal::new(0.0, sigma_size_rad).unwrap();
    let angular_size_true = (body_radius_m / range).clamp(-1.0, 1.0).asin();
    let angular_size_meas = angular_size_true + size_dist.sample(rng);

    OpNavMeas { los: los_meas, angular_size_rad: angular_size_meas }
}

/// A unit vector perpendicular to `axis`, at a uniformly random azimuth about it.
fn random_perpendicular<R: Rng>(axis: &Vector3<f64>, rng: &mut R) -> Vector3<f64> {
    let arbitrary = if axis.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
    let perp1 = axis.cross(&arbitrary).normalize();
    let perp2 = axis.cross(&perp1);
    let theta = rng.gen_range(0.0..std::f64::consts::TAU);
    perp1 * theta.cos() + perp2 * theta.sin()
}
