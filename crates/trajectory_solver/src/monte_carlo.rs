//! Monte Carlo random seeding near a reference point.
//!
//! Phase-2 scope: scatter samples around a reference parameter vector to
//! probe the local neighbourhood of a porkchop/DiffCorrection solution
//! (e.g. for seed generation or local robustness checks before committing
//! to a refined solution). This is distinct from the Phase-4 dispersion/
//! robustness Monte Carlo, which will disperse full mission initial
//! conditions, hardware noise, and mass properties through the sim engine.
//!
//! Self-contained PRNG (SplitMix64 + Box-Muller) avoids adding `rand`/
//! `rand_distr` as new workspace dependencies for this single use case.

/// SplitMix64 — fast, dependency-free, deterministic PRNG seeded by a `u64`.
/// Shared by every solver in this crate (and exported for callers with the
/// same need) rather than duplicated — avoids adding `rand`/
/// `rand_distr` as a workspace dependency for this one need.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform sample in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal sample via the Box-Muller transform.
    pub(crate) fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE); // avoid ln(0)
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// One scattered sample: perturbed parameters plus the caller-supplied evaluation.
#[derive(Clone, Debug)]
pub struct MonteCarloSample<T> {
    pub params: Vec<f64>,
    pub value: T,
}

/// Gaussian scatter around a reference parameter vector.
pub struct MonteCarloSolver {
    /// 1-sigma dispersion, one per parameter.
    pub sigma: Vec<f64>,
    /// Number of samples to draw.
    pub n_samples: usize,
    /// PRNG seed — fixed for reproducible runs.
    pub seed: u64,
}

impl MonteCarloSolver {
    /// Draw `n_samples` Gaussian-scattered parameter vectors around `reference`
    /// and evaluate each with the caller-supplied `eval`. Samples are returned
    /// in draw order (unranked) — the caller ranks/filters as needed.
    pub fn run<T, F>(&self, reference: &[f64], mut eval: F) -> Vec<MonteCarloSample<T>>
    where
        F: FnMut(&[f64]) -> T,
    {
        let mut rng = SplitMix64::new(self.seed);
        (0..self.n_samples)
            .map(|_| {
                let params: Vec<f64> = reference
                    .iter()
                    .zip(&self.sigma)
                    .map(|(p, s)| p + s * rng.next_gaussian())
                    .collect();
                let value = eval(&params);
                MonteCarloSample { params, value }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scatter around the minimum of a synthetic paraboloid and confirm the
    /// best-scoring sample lands close to the true minimum — no orbital
    /// mechanics involved, just validating the scatter + selection mechanics.
    #[test]
    fn recovers_paraboloid_minimum_from_scatter() {
        let solver = MonteCarloSolver { sigma: vec![1.0, 1.0], n_samples: 2000, seed: 42 };

        let true_min = [3.0, -2.0];
        let samples = solver.run(&true_min, |p| {
            (p[0] - true_min[0]).powi(2) + (p[1] - true_min[1]).powi(2)
        });

        let best = samples
            .iter()
            .min_by(|a, b| a.value.partial_cmp(&b.value).unwrap())
            .unwrap();

        assert!(best.value < 0.05, "expected a close sample, best value = {}", best.value);
    }

    /// Two runs with the same seed must be bit-identical — reproducibility
    /// matters for debugging and for regenerating a reported "best" solution.
    #[test]
    fn same_seed_is_reproducible() {
        let solver_a = MonteCarloSolver { sigma: vec![0.5], n_samples: 10, seed: 7 };
        let solver_b = MonteCarloSolver { sigma: vec![0.5], n_samples: 10, seed: 7 };

        let a = solver_a.run(&[0.0], |p| p[0]);
        let b = solver_b.run(&[0.0], |p| p[0]);

        for (sa, sb) in a.iter().zip(b.iter()) {
            assert_eq!(sa.params, sb.params);
        }
    }
}
