//! Radial-basis-function (RBF) surrogate model for surrogate-assisted
//! sampling in expensive black-box searches.
//!
//! A surrogate model is fitted to already-evaluated (point, cost) pairs and
//! then used to RANK candidate points before spending real evaluations —
//! reusing information an optimizer would otherwise throw away (e.g. the
//! ~97% of candidates an incremental-pruning level discards). This module
//! deliberately implements only the fit/predict core: acquisition policy
//! (what to do with a prediction) belongs to the caller.
//!
//! Design notes:
//! - Inputs are normalized to `[0,1]^n` via caller-supplied bounds before
//!   any distance computation — the same mixed-unit scaling lesson as
//!   `nelder_mead.rs` / `lbfgs.rs` / the multiple-shooting solver.
//! - Gaussian kernel with a FIXED caller-chosen length scale and ridge
//!   regularization; no hyperparameter optimization. This is a ranking
//!   model, not a full Gaussian process — deterministic, dependency-free,
//!   and cheap enough to refit on the fly.
//! - The fit cost is O(k³) in the number of retained samples; callers cap
//!   `max_samples` (most-recent-wins) to bound it.
//!
//! # References
//! - Jones, D. R., Schonlau, M. & Welch, W. J. (1998), "Efficient Global
//!   Optimization of Expensive Black-Box Functions", J. Global Optimization
//!   13(4):455–492 (the surrogate-assisted-optimization paradigm; EGO).
//! - Regis, R. G. & Shoemaker, C. A. (2007), "A Stochastic Radial Basis
//!   Function Method for the Global Optimization of Expensive Functions",
//!   INFORMS J. Computing 19(4):497–509 (RBF surrogates specifically, in
//!   place of full kriging).

use nalgebra::{DMatrix, DVector};

/// A fitted RBF interpolant over `[0,1]^n`-normalized inputs.
pub struct RbfSurrogate {
    /// Normalized sample points, one row per retained sample.
    centers: Vec<Vec<f64>>,
    /// Solved RBF weights, parallel to `centers`.
    weights: Vec<f64>,
    /// Gaussian kernel length scale, in normalized units.
    length_scale: f64,
    /// Mean of the training costs — predictions are `mean + Σ wᵢ·φ(r)`, so
    /// the model falls back to the mean far from all samples instead of 0.
    mean: f64,
}

impl RbfSurrogate {
    /// Fit an RBF interpolant to `(xs, fs)` pairs. `xs` are raw
    /// (unnormalized) points; `bounds` normalizes them. Returns `None` when
    /// fewer than 2 usable samples exist, when the linear solve fails, or
    /// when every cost is non-finite. When more than `max_samples` pairs
    /// are given, the LAST `max_samples` are kept (most recent wins — in a
    /// sequential search, recent samples cluster where the search actually
    /// is).
    pub fn fit(
        bounds: &[(f64, f64)],
        xs: &[Vec<f64>],
        fs: &[f64],
        length_scale: f64,
        ridge: f64,
        max_samples: usize,
    ) -> Option<Self> {
        let usable: Vec<(&Vec<f64>, f64)> = xs
            .iter()
            .zip(fs.iter().copied())
            .filter(|(x, f)| f.is_finite() && x.len() == bounds.len())
            .collect();
        if usable.len() < 2 || length_scale <= 0.0 {
            return None;
        }
        let start = usable.len().saturating_sub(max_samples.max(2));
        let kept = &usable[start..];
        let k = kept.len();

        let centers: Vec<Vec<f64>> = kept.iter().map(|(x, _)| normalize(bounds, x)).collect();
        let mean = kept.iter().map(|(_, f)| f).sum::<f64>() / k as f64;
        let rhs = DVector::from_iterator(k, kept.iter().map(|(_, f)| f - mean));

        let mut phi = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                phi[(i, j)] = gaussian(sq_dist(&centers[i], &centers[j]), length_scale);
            }
            phi[(i, i)] += ridge;
        }
        let weights = phi.lu().solve(&rhs)?;

        Some(RbfSurrogate {
            centers,
            weights: weights.iter().copied().collect(),
            length_scale,
            mean,
        })
    }

    /// Predict the cost at a raw (unnormalized) point.
    pub fn predict(&self, bounds: &[(f64, f64)], x: &[f64]) -> f64 {
        let xn = normalize(bounds, x);
        let mut acc = self.mean;
        for (c, w) in self.centers.iter().zip(&self.weights) {
            acc += w * gaussian(sq_dist(c, &xn), self.length_scale);
        }
        acc
    }

    /// Normalized Euclidean distance from `x` to the nearest training
    /// sample — a crude predictive-variance proxy (far from all data ⇒
    /// prediction is just the mean and shouldn't be trusted for fine
    /// ranking). Standard practice in RBF-assisted search (Regis &
    /// Shoemaker 2007 use exactly this distance term).
    pub fn min_dist_to_sample(&self, bounds: &[(f64, f64)], x: &[f64]) -> f64 {
        let xn = normalize(bounds, x);
        self.centers
            .iter()
            .map(|c| sq_dist(c, &xn).sqrt())
            .fold(f64::MAX, f64::min)
    }

    /// Number of retained training samples.
    pub fn n_samples(&self) -> usize {
        self.centers.len()
    }
}

fn normalize(bounds: &[(f64, f64)], x: &[f64]) -> Vec<f64> {
    bounds
        .iter()
        .zip(x)
        .map(|(&(lo, hi), &v)| if hi > lo { (v - lo) / (hi - lo) } else { 0.0 })
        .collect()
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn gaussian(sq_r: f64, length_scale: f64) -> f64 {
    (-sq_r / (2.0 * length_scale * length_scale)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::SplitMix64;

    fn quad(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    /// The interpolant must reproduce a smooth quadratic well enough for
    /// RANKING (its actual job) on held-out points inside the sampled
    /// region.
    #[test]
    fn reproduces_smooth_quadratic_ranking() {
        let bounds = vec![(-2.0_f64, 2.0), (-2.0, 2.0)];
        let mut rng = SplitMix64::new(7);
        let xs: Vec<Vec<f64>> = (0..120)
            .map(|_| bounds.iter().map(|(lo, hi)| lo + rng.next_f64() * (hi - lo)).collect())
            .collect();
        let fs: Vec<f64> = xs.iter().map(|x| quad(x)).collect();
        let s = RbfSurrogate::fit(&bounds, &xs, &fs, 0.3, 1e-8, 400).expect("fit");

        // A point near the origin must be predicted cheaper than one near a
        // corner — pairwise ranking over a spread of held-out points.
        let mut correct = 0;
        let mut total = 0;
        for i in 0..15 {
            for j in (i + 1)..15 {
                let a = vec![-1.8 + 0.24 * i as f64, 1.7 - 0.22 * i as f64];
                let b = vec![-1.8 + 0.24 * j as f64, 1.7 - 0.22 * j as f64];
                if (quad(&a) - quad(&b)).abs() < 0.2 {
                    continue; // too close to call — skip ties
                }
                total += 1;
                let pred_says_a = s.predict(&bounds, &a) < s.predict(&bounds, &b);
                let true_says_a = quad(&a) < quad(&b);
                if pred_says_a == true_says_a {
                    correct += 1;
                }
            }
        }
        assert!(
            correct as f64 >= 0.9 * total as f64,
            "ranking accuracy too low: {correct}/{total}"
        );
    }

    /// Degenerate inputs must not panic and must return None where a fit is
    /// meaningless.
    #[test]
    fn degenerate_inputs_return_none() {
        let bounds = vec![(0.0_f64, 1.0)];
        assert!(RbfSurrogate::fit(&bounds, &[], &[], 0.3, 1e-8, 100).is_none());
        assert!(RbfSurrogate::fit(&bounds, &[vec![0.5]], &[1.0], 0.3, 1e-8, 100).is_none());
        assert!(RbfSurrogate::fit(
            &bounds,
            &[vec![0.1], vec![0.9]],
            &[f64::NAN, f64::INFINITY],
            0.3,
            1e-8,
            100
        )
        .is_none());
        // Zero/negative length scale is a caller error, reported as None.
        assert!(RbfSurrogate::fit(&bounds, &[vec![0.1], vec![0.9]], &[1.0, 2.0], 0.0, 1e-8, 100).is_none());
    }

    /// `max_samples` must actually cap the retained sample count (fit cost
    /// is cubic in it).
    #[test]
    fn max_samples_caps_retained_set() {
        let bounds = vec![(0.0_f64, 1.0)];
        let xs: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 / 50.0]).collect();
        let fs: Vec<f64> = xs.iter().map(|x| quad(x)).collect();
        let s = RbfSurrogate::fit(&bounds, &xs, &fs, 0.2, 1e-8, 10).expect("fit");
        assert_eq!(s.n_samples(), 10);
    }

    /// Far from every sample, the prediction must relax to the training
    /// mean (not to 0 or something wild), and the variance proxy must grow.
    #[test]
    fn falls_back_to_mean_far_from_data() {
        let bounds = vec![(-10.0_f64, 10.0), (-10.0, 10.0)];
        // All samples clustered near (-9, -9).
        let xs: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![-9.0 + 0.01 * i as f64, -9.0 + 0.013 * i as f64])
            .collect();
        let fs: Vec<f64> = xs.iter().map(|x| quad(x)).collect();
        let mean = fs.iter().sum::<f64>() / fs.len() as f64;
        let s = RbfSurrogate::fit(&bounds, &xs, &fs, 0.1, 1e-8, 400).expect("fit");

        let far = vec![9.0, 9.0];
        assert!(
            (s.predict(&bounds, &far) - mean).abs() < 1e-6,
            "far prediction should be the training mean"
        );
        let near = vec![-9.0, -9.0];
        assert!(s.min_dist_to_sample(&bounds, &near) < s.min_dist_to_sample(&bounds, &far));
    }
}
