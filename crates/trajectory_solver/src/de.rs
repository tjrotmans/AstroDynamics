//! Differential Evolution optimizer for bounded continuous parameter spaces.
//!
//! Implements the DE/rand/1/bin variant — the most common and well-studied DE
//! strategy. For each individual, a mutant is formed from three randomly
//! chosen population members; trial parameters are created by binomial
//! crossover with the current individual; greedy selection keeps whichever
//! has lower fitness.
//!
//! DE consistently outperforms GA and PSO on MGA-DSM trajectory optimisation
//! problems (Izzo & Vinkó 2010, GTOP benchmark suite) because its differential
//! mutation naturally steps *along* the feasible manifold in parameter space
//! rather than recombining across it — critical when the feasible region is a
//! thin tunnel (valid Lambert arcs) surrounded by completely infeasible space.
//!
//! # References
//! - Storn & Price (1997), "Differential Evolution — A Simple and Efficient
//!   Heuristic for Global Optimization over Continuous Spaces", J. Global
//!   Optimization 11:341–359.
//! - Izzo & Vinkó (2010), "Global Optimisation Heuristics and Test Problems
//!   for Preliminary Spacecraft Trajectory Design", ESA Technical Report.

use crate::monte_carlo::SplitMix64;

/// Result of a [`DeSolver::run`] call.
#[derive(Clone, Debug)]
pub struct DeResult {
    pub best_params: Vec<f64>,
    pub best_fitness: f64,
    /// Best fitness found so far, one entry per generation — for convergence
    /// history plots (best-fitness-so-far vs. generation index).
    pub history: Vec<f64>,
    /// Best PARAMETER vector found so far, one entry per generation, parallel
    /// to `history` (for per-parameter convergence plots). Always
    /// the running best-so-far, matching `history`'s own semantics.
    pub param_history: Vec<Vec<f64>>,
}

/// Differential Evolution minimiser for bounded real-valued parameters.
///
/// Variant: DE/rand/1/bin (Storn & Price 1997).
/// - Mutation scale `f_weight` ∈ (0, 2]: controls the step size of the
///   differential perturbation. Recommended range: 0.4–1.0.
/// - Crossover probability `cr` ∈ [0, 1]: fraction of genes taken from the
///   mutant. High values (> 0.9) work well for MGA-DSM problems because the
///   leg parameters are tightly correlated — partial crossover destroys that
///   correlation. Recommended: 0.8–0.95.
/// - Population size `population_size`: recommended ≥ 10 × (number of
///   parameters). Smaller populations converge faster but risk premature
///   convergence on multi-modal landscapes.
pub struct DeSolver {
    pub population_size: usize,
    pub generations: usize,
    /// Differential weight F (mutation scale factor). Storn & Price recommend
    /// F ∈ [0.4, 1.0] for most problems; values above 1.0 rarely help.
    pub f_weight: f64,
    /// Crossover probability CR ∈ [0, 1]. Higher values (0.8–0.95) are
    /// typically better for tightly-coupled parameter spaces like MGA-DSM.
    pub cr: f64,
    pub seed: u64,
}

impl DeSolver {
    /// Minimise `fitness` over `bounds` (one `(min, max)` pair per parameter).
    ///
    /// `fitness` returning `None` is treated as infeasible, penalised to
    /// `f64::MAX` so the population steers away from it without panicking.
    pub fn run<F>(&self, bounds: &[(f64, f64)], fitness: F) -> DeResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
    {
        self.run_with_progress(bounds, fitness, |_gen, _best, _params| {})
    }

    /// Same as [`DeSolver::run`] but calls `on_generation(generation_index,
    /// best_fitness_so_far, best_params_so_far)` after each generation — for
    /// live progress streams.
    pub fn run_with_progress<F, G>(
        &self,
        bounds: &[(f64, f64)],
        fitness: F,
        on_generation: G,
    ) -> DeResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
        G: FnMut(usize, f64, &[f64]),
    {
        self.run_seeded_with_progress(bounds, &[], fitness, on_generation).0
    }

    /// Same as [`DeSolver::run_with_progress`], but part of the initial
    /// population is taken from `seeds` (clamped to bounds; the remainder is
    /// random), and the final population is returned alongside the result as
    /// `(params, fitness)` pairs so callers can harvest elites for a later
    /// refinement stage.
    pub fn run_seeded_with_progress<F, G>(
        &self,
        bounds: &[(f64, f64)],
        seeds: &[Vec<f64>],
        mut fitness: F,
        mut on_generation: G,
    ) -> (DeResult, Vec<(Vec<f64>, f64)>)
    where
        F: FnMut(&[f64]) -> Option<f64>,
        G: FnMut(usize, f64, &[f64]),
    {
        let mut rng = SplitMix64::new(self.seed);
        let n_dim = bounds.len();
        let np    = self.population_size.max(4); // DE requires NP ≥ 4

        // Initial population: seeds first (clamped), random fill after.
        let mut pop: Vec<Vec<f64>> = Vec::with_capacity(np);
        for s in seeds.iter().take(np) {
            pop.push((0..n_dim)
                .map(|j| s.get(j).copied().unwrap_or(0.5 * (bounds[j].0 + bounds[j].1))
                    .clamp(bounds[j].0, bounds[j].1))
                .collect());
        }
        while pop.len() < np {
            pop.push(bounds
                .iter()
                .map(|(lo, hi)| lo + rng.next_f64() * (hi - lo))
                .collect());
        }

        let mut fits: Vec<f64> = pop
            .iter()
            .map(|ind| fitness(ind).unwrap_or(f64::MAX))
            .collect();

        let mut best_idx = fits
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut history = Vec::with_capacity(self.generations);
        let mut param_history = Vec::with_capacity(self.generations);

        for gen in 0..self.generations {
            for i in 0..np {
                // Select three distinct random individuals r1, r2, r3 ≠ i.
                let (r1, r2, r3) = pick3_distinct(&mut rng, np, i);

                // Mutation: v = pop[r1] + F*(pop[r2] - pop[r3]), clipped to bounds.
                let mutant: Vec<f64> = (0..n_dim)
                    .map(|j| {
                        let v = pop[r1][j] + self.f_weight * (pop[r2][j] - pop[r3][j]);
                        v.clamp(bounds[j].0, bounds[j].1)
                    })
                    .collect();

                // Binomial crossover: each gene comes from the mutant with
                // probability CR; at least one gene (rand_j) always comes
                // from the mutant (ensures the trial is not identical to the
                // current individual).
                let rand_j = (rng.next_f64() * n_dim as f64) as usize;
                let trial: Vec<f64> = (0..n_dim)
                    .map(|j| {
                        if j == rand_j || rng.next_f64() < self.cr {
                            mutant[j]
                        } else {
                            pop[i][j]
                        }
                    })
                    .collect();

                // Greedy selection.
                let trial_fit = fitness(&trial).unwrap_or(f64::MAX);
                if trial_fit < fits[i] {
                    pop[i]  = trial;
                    fits[i] = trial_fit;
                    if trial_fit < fits[best_idx] {
                        best_idx = i;
                    }
                }
            }

            history.push(fits[best_idx]);
            param_history.push(pop[best_idx].clone());
            on_generation(gen, fits[best_idx], &pop[best_idx]);
        }

        let result = DeResult {
            best_params:  pop[best_idx].clone(),
            best_fitness: fits[best_idx],
            history,
            param_history,
        };
        let population = pop.into_iter().zip(fits).collect();
        (result, population)
    }
}

/// Pick three indices that are all distinct and all ≠ `exclude` from `[0, n)`.
fn pick3_distinct(rng: &mut SplitMix64, n: usize, exclude: usize) -> (usize, usize, usize) {
    let rand_ne = |rng: &mut SplitMix64, bad: &[usize]| loop {
        let v = (rng.next_f64() * n as f64) as usize % n;
        if !bad.contains(&v) { return v; }
    };
    let r1 = rand_ne(rng, &[exclude]);
    let r2 = rand_ne(rng, &[exclude, r1]);
    let r3 = rand_ne(rng, &[exclude, r1, r2]);
    (r1, r2, r3)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rosenbrock function: global minimum at (1, 1) with f=0.
    /// A classic non-convex test for global optimisers.
    #[test]
    fn minimises_rosenbrock() {
        let de = DeSolver {
            population_size: 50,
            generations:     500,
            f_weight:        0.8,
            cr:              0.9,
            seed:            42,
        };
        let bounds = vec![(-2.0_f64, 2.0), (-2.0, 2.0)];
        let result = de.run(&bounds, |x| {
            let f = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
            Some(f)
        });
        assert!(result.best_fitness < 1e-6,
            "Rosenbrock minimum not found: f={:.3e} at ({:.4}, {:.4})",
            result.best_fitness, result.best_params[0], result.best_params[1]);
        assert!((result.best_params[0] - 1.0).abs() < 1e-3);
        assert!((result.best_params[1] - 1.0).abs() < 1e-3);
    }

    /// Infeasible fitness (`None`) must not panic and must be avoided.
    #[test]
    fn handles_infeasible_fitness() {
        let de = DeSolver {
            population_size: 20,
            generations:     50,
            f_weight:        0.8,
            cr:              0.9,
            seed:            1,
        };
        let bounds = vec![(0.0_f64, 1.0)];
        // All evaluations infeasible — should not panic, result is f64::MAX.
        let result = de.run(&bounds, |_| None);
        assert_eq!(result.best_fitness, f64::MAX);
    }
}
