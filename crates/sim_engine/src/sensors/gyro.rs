//! Generic rate-gyro measurement: two-state error model (bias random walk +
//! angle random walk on the rate) on top of the true body-frame angular
//! rate. See `docs/MP/MANUAL.md` §11.2 for the governing equations and
//! citation (Farrenkopf 1978; Markley & Crassidis §4). Feeds the attitude
//! MEKF (Phase 13h, MANUAL.md §12.3) — no estimator lives in this crate,
//! this module is the honest measurement model only.
//!
//! New model: the existing `imu` module in this crate only ever covered the
//! accelerometer/ΔV side (ported from the Bennu reference); no rate-gyro
//! model previously existed anywhere in this repo.

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Persistent gyro bias state, carried forward step-to-step by the caller —
/// bias is NOT resampled each call, it evolves as a random walk (mirrors how
/// `sim_engine`'s truth state carries reaction-wheel speeds forward rather
/// than deriving them fresh each step).
#[derive(Clone, Copy, Debug)]
pub struct GyroBiasState {
    /// Current per-axis bias estimate, body frame [rad/s].
    pub bias_rad_s: Vector3<f64>,
}

impl GyroBiasState {
    /// Zero-bias initial state (e.g. a freshly calibrated unit at power-on).
    pub fn zero() -> Self {
        Self { bias_rad_s: Vector3::zeros() }
    }
}

/// Propagate the gyro bias by one random-walk step and return a noisy rate
/// measurement, per MANUAL.md §11.2:
///
/// ```text
/// b_k = b_{k-1} + sigma_bias * sqrt(dt) * N(0,1)            (bias random walk / RRW)
/// omega_meas = omega_true + b_k + N(0, sigma_arw / sqrt(dt)) (angle random walk -> rate noise)
/// ```
///
/// `omega_true_rad_s` = true angular rate, body frame [rad/s].
/// `bias` = persistent bias state, updated in place.
/// `bias_walk_sigma_rad_s_sqrt_s` = bias (rate) random-walk 1-sigma coefficient
///   [rad/s/sqrt(s)] (`hardware_catalog::GyroSpec::bias_walk_sigma_rad_s_sqrt_s`).
/// `arw_sigma_rad_sqrt_s` = angle random walk 1-sigma coefficient [rad/sqrt(s)]
///   (`hardware_catalog::GyroSpec::arw_sigma_rad_sqrt_s`).
/// `dt_s` = time since the previous call [s].
///
/// Note the scaling direction: the bias step scales *up* with `sqrt(dt)` (a
/// random walk's variance grows linearly with elapsed time), while the ARW
/// rate-noise sample scales *down* with `1/sqrt(dt)` (a fixed-PSD white noise
/// on the rate, sampled at interval `dt` — finer sampling sees more noise
/// power per sample). Swapping these two scalings is the classic bug in this
/// kind of model; the two named parameters and their distinct dt-scalings are
/// kept explicit here for that reason.
pub fn measure<R: Rng>(
    omega_true_rad_s: &Vector3<f64>,
    bias: &mut GyroBiasState,
    bias_walk_sigma_rad_s_sqrt_s: f64,
    arw_sigma_rad_sqrt_s: f64,
    dt_s: f64,
    rng: &mut R,
) -> Vector3<f64> {
    let sqrt_dt = dt_s.sqrt();

    if bias_walk_sigma_rad_s_sqrt_s > 0.0 && sqrt_dt > 0.0 {
        let bias_dist = Normal::new(0.0, bias_walk_sigma_rad_s_sqrt_s * sqrt_dt).unwrap();
        bias.bias_rad_s += Vector3::new(
            bias_dist.sample(rng),
            bias_dist.sample(rng),
            bias_dist.sample(rng),
        );
    }

    let rate_noise = if arw_sigma_rad_sqrt_s > 0.0 && sqrt_dt > 0.0 {
        let rate_dist = Normal::new(0.0, arw_sigma_rad_sqrt_s / sqrt_dt).unwrap();
        Vector3::new(rate_dist.sample(rng), rate_dist.sample(rng), rate_dist.sample(rng))
    } else {
        Vector3::zeros()
    };

    omega_true_rad_s + bias.bias_rad_s + rate_noise
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    /// Bias random walk: variance of the accumulated bias should scale
    /// linearly with elapsed time (standard random-walk property), checked
    /// by Monte Carlo over many independent walk realizations.
    #[test]
    fn bias_random_walk_variance_scales_with_time() {
        let sigma_bias = 1.0e-6_f64; // rad/s/sqrt(s)
        let dt = 10.0;
        let n_trials = 20_000;

        let sample_var_after = |n_steps: usize| -> f64 {
            let mut rng = StdRng::seed_from_u64(42);
            let mut vals = Vec::with_capacity(n_trials);
            for _ in 0..n_trials {
                let mut bias = GyroBiasState::zero();
                for _ in 0..n_steps {
                    let _ = measure(&Vector3::zeros(), &mut bias, sigma_bias, 0.0, dt, &mut rng);
                }
                vals.push(bias.bias_rad_s.x);
            }
            let mean: f64 = vals.iter().sum::<f64>() / n_trials as f64;
            vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n_trials as f64 - 1.0)
        };

        let var_10 = sample_var_after(10);
        let var_40 = sample_var_after(40);

        // Expected: Var(t) = sigma_bias^2 * t, so var_40/var_10 should be ~4.0
        // (elapsed time ratio), well clear of Monte Carlo noise at 20k trials.
        let ratio = var_40 / var_10;
        assert!(
            (ratio - 4.0).abs() < 0.5,
            "variance ratio {ratio} should be close to 4.0 (elapsed-time ratio)"
        );

        // Sanity: absolute variance matches the closed-form sigma_bias^2 * t.
        let expected_var_10 = sigma_bias * sigma_bias * (10.0 * dt);
        assert!(
            (var_10 - expected_var_10).abs() / expected_var_10 < 0.15,
            "var_10={var_10} expected~{expected_var_10}"
        );
    }

    /// With bias walk disabled, the measurement is pure zero-mean white noise
    /// (ARW-driven rate noise only) — sample mean should be near zero.
    #[test]
    fn white_noise_only_case_has_zero_mean() {
        let arw = 1.0e-4_f64; // rad/sqrt(s)
        let dt = 1.0;
        let n = 50_000;
        let mut rng = StdRng::seed_from_u64(7);
        let mut bias = GyroBiasState::zero();

        let mut sum = 0.0;
        for _ in 0..n {
            let m = measure(&Vector3::zeros(), &mut bias, 0.0, arw, dt, &mut rng);
            sum += m.x;
        }
        let mean = sum / n as f64;
        let expected_sigma_of_mean = (arw / dt.sqrt()) / (n as f64).sqrt();

        assert!(
            mean.abs() < 5.0 * expected_sigma_of_mean,
            "sample mean {mean} should be within 5-sigma of zero (sigma_of_mean={expected_sigma_of_mean})"
        );
    }

    /// A stationary spacecraft with a low-bias (fine-grade) gyro should
    /// produce readings within a few sigma of zero.
    #[test]
    fn stationary_low_bias_gyro_reads_near_zero() {
        // Fine-grade-class figures (see hardware_catalog::GyroSpec::fine()).
        let bias_walk = 8.08e-10_f64;
        let arw = 5.82e-7_f64;
        let dt = 1.0_f64;
        let mut rng = StdRng::seed_from_u64(99);
        let mut bias = GyroBiasState::zero();

        let expected_sigma = arw / dt.sqrt(); // dominates over one small step
        let mut max_abs = 0.0_f64;
        for _ in 0..1000 {
            let m = measure(&Vector3::zeros(), &mut bias, bias_walk, arw, dt, &mut rng);
            max_abs = max_abs.max(m.x.abs()).max(m.y.abs()).max(m.z.abs());
        }
        assert!(
            max_abs < 6.0 * expected_sigma,
            "max reading {max_abs} should stay within ~6-sigma ({expected_sigma}) over 1000 steps"
        );
    }

    /// Zero-bias initial state should really be zero.
    #[test]
    fn zero_bias_state_is_zero() {
        let b = GyroBiasState::zero();
        assert_eq!(b.bias_rad_s, Vector3::zeros());
    }
}
