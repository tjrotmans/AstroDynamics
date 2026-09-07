//! Generic Deep Space Network (DSN) range and range-rate (Doppler)
//! measurement models — pure measurement functions only, per the review in
//! `docs/MP/MANUAL.md` §11.5.
//!
//! Deliberately scoped down from the Bennu reference
//! (`GNC/AutonomousNavigation/src/sensors/dsn.rs`'s `GroundOdEkf`): no
//! ground/onboard orbit-determination filter and no uplink mechanism live
//! here. The reference's `inject_uplink` hard-state-reset pattern, its
//! finite-difference STM, and its unconditional `clamp_cr` have
//! produced unrealistic behavior before — §11.5 documents
//! what was found and why none of it was ported. This module is only the
//! sensor/measurement layer a future cruise navigation filter (Phase 13h)
//! will consume as proper Kalman measurement updates.
//!
//! Delta-DOR is intentionally NOT modeled here yet: the reference's
//! `update_ddor`/`sim_ddor_inner` construct the interferometric transverse
//! basis from the *filter's own estimated* line of sight rather than the
//! true line of sight or a fixed celestial frame — see §11.5 for why that is
//! a real (if subtle) truth/estimate coupling bug that should not be
//! repeated. A DDOR measurement model can be added later using a basis
//! derived only from truth (or a fixed frame), independent of any filter
//! state.

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use hardware_catalog::DsnLinkSpec;

/// A two-way range measurement: true predicted value + a noisy sample.
#[derive(Clone, Copy, Debug)]
pub struct RangeMeas {
    /// True range at the truth state [m].
    pub true_range_m: f64,
    /// Noisy sampled range [m].
    pub measured_range_m: f64,
}

/// A range-rate (Doppler) measurement: true predicted value + a noisy sample.
#[derive(Clone, Copy, Debug)]
pub struct RangeRateMeas {
    /// True range rate at the truth state [m/s].
    pub true_range_rate_mps: f64,
    /// Noisy sampled range rate [m/s].
    pub measured_range_rate_mps: f64,
}

/// Two-way range measurement: `z = |r_sc - r_earth| + noise`.
///
/// `r_sc_m` and `r_earth_m` must be expressed in the same inertial frame
/// (e.g. heliocentric ecliptic J2000 for cruise). `sigma_m` = 1-sigma range
/// noise (`DsnLinkSpec::range_noise_m`).
pub fn measure_range<R: Rng>(
    r_sc_m: &Vector3<f64>,
    r_earth_m: &Vector3<f64>,
    sigma_m: f64,
    rng: &mut R,
) -> RangeMeas {
    let true_range_m = (r_sc_m - r_earth_m).norm();
    let dist = Normal::new(0.0, sigma_m).unwrap();
    RangeMeas { true_range_m, measured_range_m: true_range_m + dist.sample(rng) }
}

/// Range-rate (Doppler) measurement:
/// `z = (r_sc - r_earth)·(v_sc - v_earth) / |r_sc - r_earth| + noise`.
///
/// `sigma_mps` = 1-sigma range-rate noise (`DsnLinkSpec::range_rate_noise_mps`).
/// Returns `0.0` for the true range rate (not `NaN`) in the degenerate
/// zero-range case, matching the reference's own guard.
pub fn measure_range_rate<R: Rng>(
    r_sc_m: &Vector3<f64>,
    v_sc_mps: &Vector3<f64>,
    r_earth_m: &Vector3<f64>,
    v_earth_mps: &Vector3<f64>,
    sigma_mps: f64,
    rng: &mut R,
) -> RangeRateMeas {
    let dr = r_sc_m - r_earth_m;
    let dv = v_sc_mps - v_earth_mps;
    let range_m = dr.norm();
    let true_range_rate_mps = if range_m > 1.0 { dr.dot(&dv) / range_m } else { 0.0 };
    let dist = Normal::new(0.0, sigma_mps).unwrap();
    RangeRateMeas {
        true_range_rate_mps,
        measured_range_rate_mps: true_range_rate_mps + dist.sample(rng),
    }
}

/// Convenience wrapper: two-way range using a `hardware_catalog::DsnLinkSpec`
/// directly for the noise parameter.
pub fn measure_range_from_spec<R: Rng>(
    r_sc_m: &Vector3<f64>,
    r_earth_m: &Vector3<f64>,
    spec: &DsnLinkSpec,
    rng: &mut R,
) -> RangeMeas {
    measure_range(r_sc_m, r_earth_m, spec.range_noise_m, rng)
}

/// Convenience wrapper: range-rate using a `hardware_catalog::DsnLinkSpec`
/// directly for the noise parameter.
pub fn measure_range_rate_from_spec<R: Rng>(
    r_sc_m: &Vector3<f64>,
    v_sc_mps: &Vector3<f64>,
    r_earth_m: &Vector3<f64>,
    v_earth_mps: &Vector3<f64>,
    spec: &DsnLinkSpec,
    rng: &mut R,
) -> RangeRateMeas {
    measure_range_rate(r_sc_m, v_sc_mps, r_earth_m, v_earth_mps, spec.range_rate_noise_mps, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn range_geometry_is_correct() {
        let r_sc = Vector3::new(3.0e11, 0.0, 0.0);
        let r_earth = Vector3::new(1.5e11, 0.0, 0.0);
        let mut rng = StdRng::seed_from_u64(1);
        let m = measure_range(&r_sc, &r_earth, 0.0, &mut rng);
        assert!((m.true_range_m - 1.5e11).abs() < 1e-3);
        assert!((m.measured_range_m - m.true_range_m).abs() < 1e-9);
    }

    #[test]
    fn range_rate_geometry_matches_closing_speed() {
        // Spacecraft directly along +x from Earth, receding at 1000 m/s in +x.
        let r_sc = Vector3::new(2.0e11, 0.0, 0.0);
        let v_sc = Vector3::new(1000.0, 0.0, 0.0);
        let r_earth = Vector3::new(1.5e11, 0.0, 0.0);
        let v_earth = Vector3::zeros();
        let mut rng = StdRng::seed_from_u64(2);
        let m = measure_range_rate(&r_sc, &v_sc, &r_earth, &v_earth, 0.0, &mut rng);
        assert!((m.true_range_rate_mps - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn range_rate_is_zero_for_pure_transverse_motion() {
        // Velocity purely perpendicular to the line of sight -> zero range rate.
        let r_sc = Vector3::new(2.0e11, 0.0, 0.0);
        let v_sc = Vector3::new(0.0, 500.0, 0.0);
        let r_earth = Vector3::zeros();
        let v_earth = Vector3::zeros();
        let mut rng = StdRng::seed_from_u64(3);
        let m = measure_range_rate(&r_sc, &v_sc, &r_earth, &v_earth, 0.0, &mut rng);
        assert!(m.true_range_rate_mps.abs() < 1e-6, "got {}", m.true_range_rate_mps);
    }

    #[test]
    fn range_noise_statistics_match_sigma() {
        let r_sc = Vector3::new(2.0e11, 0.0, 0.0);
        let r_earth = Vector3::zeros();
        let sigma = 2.0;
        let mut rng = StdRng::seed_from_u64(4);
        let n = 20_000;
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for _ in 0..n {
            let m = measure_range(&r_sc, &r_earth, sigma, &mut rng);
            let e = m.measured_range_m - m.true_range_m;
            sum += e;
            sumsq += e * e;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        let sample_sigma = var.sqrt();
        assert!(mean.abs() < 0.1, "mean error {mean} should be near zero");
        assert!((sample_sigma - sigma).abs() / sigma < 0.05, "sample sigma {sample_sigma}");
    }

    #[test]
    fn range_rate_noise_statistics_match_sigma() {
        let r_sc = Vector3::new(2.0e11, 0.0, 0.0);
        let v_sc = Vector3::new(500.0, 0.0, 0.0);
        let r_earth = Vector3::zeros();
        let v_earth = Vector3::zeros();
        let sigma = 1.0e-4;
        let mut rng = StdRng::seed_from_u64(5);
        let n = 20_000;
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for _ in 0..n {
            let m = measure_range_rate(&r_sc, &v_sc, &r_earth, &v_earth, sigma, &mut rng);
            let e = m.measured_range_rate_mps - m.true_range_rate_mps;
            sum += e;
            sumsq += e * e;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        let sample_sigma = var.sqrt();
        assert!(mean.abs() < sigma * 0.1, "mean error {mean} should be near zero");
        assert!((sample_sigma - sigma).abs() / sigma < 0.05, "sample sigma {sample_sigma}");
    }

    #[test]
    fn spec_wrappers_use_spec_noise_values() {
        let spec = DsnLinkSpec::medium();
        let r_sc = Vector3::new(2.0e11, 0.0, 0.0);
        let v_sc = Vector3::new(500.0, 0.0, 0.0);
        let r_earth = Vector3::zeros();
        let v_earth = Vector3::zeros();
        let mut rng = StdRng::seed_from_u64(6);
        let rm = measure_range_from_spec(&r_sc, &r_earth, &spec, &mut rng);
        let rrm = measure_range_rate_from_spec(&r_sc, &v_sc, &r_earth, &v_earth, &spec, &mut rng);
        assert!((rm.true_range_m - 2.0e11).abs() < 1e-3);
        assert!((rrm.true_range_rate_mps - 500.0).abs() < 1e-6);
    }
}
