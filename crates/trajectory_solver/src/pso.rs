//! Particle swarm optimization for bounded continuous optimization.
//!
//! Phase 8j follow-up: a fresh, dependency-free PSO matching `ga.rs`'s
//! conventions (same closure-based fitness, same `SplitMix64` PRNG, same
//! bounded real-valued parameter vector) — a second global/heuristic method
//! alongside the GA, so `mars_pso.toml` converging to the same point as
//! `mars_ga.toml`/`mars_flyby.toml` is a cross-check between two unrelated
//! algorithm families (swarm vs. population+crossover), not a coincidence
//! of one algorithm's quirks.

use crate::monte_carlo::SplitMix64;

/// Result of a `PsoSolver::run()` call.
#[derive(Clone, Debug)]
pub struct PsoResult {
    pub best_params: Vec<f64>,
    pub best_fitness: f64,
    /// Best fitness found so far, one entry per iteration — same convention
    /// as `GaResult::history`, for the convergence-history plot.
    pub history: Vec<f64>,
}

/// Velocity is clamped to this fraction of each parameter's bound span —
/// standard PSO practice, prevents particles overshooting the search box
/// before inertia decay settles them.
const VELOCITY_CLAMP_FRACTION: f64 = 0.2;

/// Particle swarm optimization minimizing a fitness function over bounded
/// real parameters.
pub struct PsoSolver {
    pub swarm_size: usize,
    pub iterations: usize,
    /// Inertia weight at iteration 0 — decays linearly to `inertia_min` by
    /// the final iteration (broad exploration early, refinement late, same
    /// rationale as the GA's adaptive mutation decay).
    pub inertia_max: f64,
    pub inertia_min: f64,
    /// Cognitive coefficient — pull toward each particle's own best.
    pub cognitive_coeff: f64,
    /// Social coefficient — pull toward the swarm's global best.
    pub social_coeff: f64,
    pub seed: u64,
}

impl PsoSolver {
    /// Minimize `fitness` over `bounds` (one `(min, max)` pair per
    /// parameter). `fitness` returning `None` is treated as infeasible,
    /// penalized to `f64::MAX` — same convention as `GaSolver::run`.
    pub fn run<F>(&self, bounds: &[(f64, f64)], fitness: F) -> PsoResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
    {
        self.run_with_progress(bounds, fitness, |_iter, _best| true)
    }

    /// Same as [`Self::run`], but calls `on_iteration(iteration_index,
    /// best_fitness_so_far)` after each iteration — see
    /// `GaSolver::run_with_progress` for the rationale (Phase 9f).
    ///
    /// `on_iteration` returns `true` to continue, `false` to stop early
    /// (job cancellation, Phase 9k task 4) — the result reflects whichever
    /// iteration ran last.
    ///
    /// Delegates to [`Self::run_with_progress_params`] (
    /// additive refactor — same loop, same RNG stream, bit-identical
    /// results; verified by `params_variant_matches_plain_progress`).
    pub fn run_with_progress<F, P>(&self, bounds: &[(f64, f64)], fitness: F, mut on_iteration: P) -> PsoResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
        P: FnMut(usize, f64) -> bool,
    {
        self.run_with_progress_params(bounds, fitness, |iter, best, _params| on_iteration(iter, best))
    }

    /// Same as [`Self::run_with_progress`], but the callback also receives
    /// the global-best PARAMETER VECTOR so far — added so the
    /// live optimizer stream can carry real decision variables for PSO the
    /// way it already does for GA/MGA (the frontend's new search-variable
    /// convergence plot consumes them; previously GA/PSO streamed
    /// `best_params: null` and only MGA had live chromosomes).
    pub fn run_with_progress_params<F, P>(&self, bounds: &[(f64, f64)], mut fitness: F, mut on_iteration: P) -> PsoResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
        P: FnMut(usize, f64, &[f64]) -> bool,
    {
        let mut rng = SplitMix64::new(self.seed);
        let n_dim = bounds.len();
        let v_max: Vec<f64> = bounds.iter().map(|&(lo, hi)| (hi - lo) * VELOCITY_CLAMP_FRACTION).collect();

        let mut positions: Vec<Vec<f64>> =
            (0..self.swarm_size).map(|_| random_position(bounds, &mut rng)).collect();
        let mut velocities: Vec<Vec<f64>> = vec![vec![0.0; n_dim]; self.swarm_size];
        let fitnesses: Vec<f64> = positions.iter().map(|p| fitness(p).unwrap_or(f64::MAX)).collect();

        let mut personal_best_pos = positions.clone();
        let mut personal_best_fit = fitnesses;

        let mut global_best_fit = f64::MAX;
        let mut global_best_pos = personal_best_pos[0].clone();
        for i in 0..self.swarm_size {
            if personal_best_fit[i] < global_best_fit {
                global_best_fit = personal_best_fit[i];
                global_best_pos = personal_best_pos[i].clone();
            }
        }

        let mut history = Vec::with_capacity(self.iterations);

        for iter in 0..self.iterations {
            let progress = iter as f64 / self.iterations.max(1) as f64;
            let inertia = self.inertia_max + (self.inertia_min - self.inertia_max) * progress;

            for i in 0..self.swarm_size {
                for d in 0..n_dim {
                    let r1 = rng.next_f64();
                    let r2 = rng.next_f64();
                    let cognitive = self.cognitive_coeff * r1 * (personal_best_pos[i][d] - positions[i][d]);
                    let social = self.social_coeff * r2 * (global_best_pos[d] - positions[i][d]);
                    let v = inertia * velocities[i][d] + cognitive + social;
                    velocities[i][d] = v.clamp(-v_max[d], v_max[d]);
                    positions[i][d] = (positions[i][d] + velocities[i][d]).clamp(bounds[d].0, bounds[d].1);
                }

                let fit = fitness(&positions[i]).unwrap_or(f64::MAX);
                if fit < personal_best_fit[i] {
                    personal_best_fit[i] = fit;
                    personal_best_pos[i] = positions[i].clone();
                    if fit < global_best_fit {
                        global_best_fit = fit;
                        global_best_pos = positions[i].clone();
                    }
                }
            }

            history.push(global_best_fit);
            if !on_iteration(iter, global_best_fit, &global_best_pos) {
                break;
            }
        }

        PsoResult { best_params: global_best_pos, best_fitness: global_best_fit, history }
    }
}

fn random_position(bounds: &[(f64, f64)], rng: &mut SplitMix64) -> Vec<f64> {
    bounds.iter().map(|&(lo, hi)| lo + rng.next_f64() * (hi - lo)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimize a synthetic paraboloid from a wide bound — validates the
    /// swarm mechanics in isolation, no orbital mechanics involved.
    #[test]
    fn recovers_paraboloid_minimum() {
        let solver = PsoSolver {
            swarm_size: 40,
            iterations: 100,
            inertia_max: 0.9,
            inertia_min: 0.4,
            cognitive_coeff: 1.5,
            social_coeff: 1.5,
            seed: 42,
        };

        let true_min = [3.0, -2.0];
        let result = solver.run(&[(-50.0, 50.0), (-50.0, 50.0)], |p| {
            Some((p[0] - true_min[0]).powi(2) + (p[1] - true_min[1]).powi(2))
        });

        assert!(
            result.best_fitness < 0.1,
            "expected convergence near the true minimum, best_fitness = {}",
            result.best_fitness
        );
        assert_eq!(result.history.len(), 100);
        for i in 1..result.history.len() {
            assert!(result.history[i] <= result.history[i - 1]);
        }
    }

    /// Two runs with the same seed must be bit-identical.
    #[test]
    fn same_seed_is_reproducible() {
        let make = || PsoSolver {
            swarm_size: 20,
            iterations: 10,
            inertia_max: 0.9,
            inertia_min: 0.4,
            cognitive_coeff: 1.5,
            social_coeff: 1.5,
            seed: 7,
        };
        let a = make().run(&[(-10.0, 10.0)], |p| Some(p[0] * p[0]));
        let b = make().run(&[(-10.0, 10.0)], |p| Some(p[0] * p[0]));
        assert_eq!(a.best_params, b.best_params);
        assert_eq!(a.history, b.history);
    }

    /// The params-streaming variant must be bit-identical to the plain
    /// progress variant (it IS the same loop -- the plain one delegates),
    /// and the streamed params at the final iteration must equal the
    /// returned best_params.
    #[test]
    fn params_variant_matches_plain_progress() {
        let make = || PsoSolver {
            swarm_size: 20,
            iterations: 15,
            inertia_max: 0.9,
            inertia_min: 0.4,
            cognitive_coeff: 1.5,
            social_coeff: 1.5,
            seed: 11,
        };
        let f = |p: &[f64]| Some((p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2));
        let plain = make().run_with_progress(&[(-10.0, 10.0), (-10.0, 10.0)], f, |_i, _b| true);
        let mut last_streamed: Vec<f64> = Vec::new();
        let with_params = make().run_with_progress_params(&[(-10.0, 10.0), (-10.0, 10.0)], f, |_i, _b, params| {
            last_streamed = params.to_vec();
            true
        });
        assert_eq!(plain.best_params, with_params.best_params);
        assert_eq!(plain.history, with_params.history);
        assert_eq!(last_streamed, with_params.best_params);
    }

    /// `None`-returning fitness (infeasible point) must not panic and must
    /// not be selected as "best".
    #[test]
    fn infeasible_points_are_penalized_not_panicking() {
        let solver = PsoSolver {
            swarm_size: 30,
            iterations: 20,
            inertia_max: 0.9,
            inertia_min: 0.4,
            cognitive_coeff: 1.5,
            social_coeff: 1.5,
            seed: 1,
        };
        let result = solver.run(&[(-10.0, 10.0)], |p| {
            if p[0] < 0.0 { None } else { Some(p[0]) }
        });
        assert!(result.best_fitness >= 0.0, "PSO should avoid the infeasible region");
    }
}
