//! Real-valued genetic algorithm for bounded continuous optimization.
//!
//! Phase 8j: a fresh, dependency-free GA matching the algorithmic choices
//! already proven in `OptimizationProblems/src/ga.rs` (tournament selection,
//! adaptive cosine-decay mutation, elitism) — but without that crate's
//! SQLite run-logging coupling (`OptimizationContext`/`EvaluationContext`),
//! which doesn't fit this crate's simple closure-based solver pattern
//! (`DiffCorrectionSolver`, `MonteCarloSolver`). Operates on a bounded
//! real-valued parameter vector, the same convention as those solvers.

use crate::monte_carlo::SplitMix64;

/// Result of a `GaSolver::run()` call.
#[derive(Clone, Debug)]
pub struct GaResult {
    pub best_params: Vec<f64>,
    pub best_fitness: f64,
    /// Best fitness found so far, one entry per generation — for plotting
    /// convergence history (best-fitness-so-far vs. generation).
    pub history: Vec<f64>,
}

/// Genetic algorithm minimizing a fitness function over bounded real
/// parameters.
pub struct GaSolver {
    pub population_size: usize,
    pub generations: usize,
    pub crossover_rate: f64,
    /// Peak per-gene mutation probability — decays via cosine schedule to
    /// 10% of this value by the final generation (same schedule as
    /// `OptimizationProblems`'s GA: broad exploration early, refinement late).
    pub mutation_rate: f64,
    pub elitism_count: usize,
    pub tournament_size: usize,
    pub seed: u64,
}

impl GaSolver {
    /// Minimize `fitness` over `bounds` (one `(min, max)` pair per
    /// parameter). `fitness` returning `None` is treated as infeasible,
    /// penalized to `f64::MAX` so the GA steers away from it without
    /// panicking on a missing ephemeris/Lambert-solution point.
    pub fn run<F>(&self, bounds: &[(f64, f64)], fitness: F) -> GaResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
    {
        self.run_with_progress(bounds, fitness, |_gen, _best| true)
    }

    /// Same as [`Self::run_with_progress`], but `on_population(generation_index,
    /// &population, &fitnesses)` is called with the *entire* evaluated
    /// population for every generation (including the initial random
    /// population, logged as generation `0`; the loop's generations follow
    /// as `1..=generations`) -- not just the best-so-far scalar. Lets a
    /// caller log/plot the full search distribution (e.g. an x/y scatter of
    /// two search parameters colored by generation), which the single
    /// best-fitness-per-generation callback can't support. A separate
    /// method rather than a parameter on `run_with_progress` to avoid
    /// touching that already-tested method's signature/behavior.
    ///
    /// `on_population` returns `true` to continue, `false` to stop early
    /// (job cancellation, Phase 9k task 4) — the result returned reflects
    /// whichever generation ran last, not necessarily `self.generations`.
    pub fn run_with_population_progress<F, P>(&self, bounds: &[(f64, f64)], fitness: F, on_population: P) -> GaResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
        P: FnMut(usize, &[Vec<f64>], &[f64]) -> bool,
    {
        // Delegates with no seeds (additive refactor for
        // `run_seeded_with_population_progress` below) -- the seedless path
        // generates the exact same initial population from the exact same
        // RNG stream as before, so existing callers are bit-identical.
        self.run_seeded_with_population_progress(bounds, &[], fitness, on_population)
    }

    /// Same as [`Self::run_with_population_progress`], but the supplied
    /// seed chromosomes (clamped to `bounds`) are injected into GENERATION
    /// 1's population, replacing its first `seeds.len()` offspring — added
    /// (revised same day,: "we want to
    /// start purely random, but we add the 8 hohman seeds in gen 1") so
    /// the Phase 9 optimizer can inject analytically known-good candidates
    /// (e.g. the Hohmann-energy departure burn) without contaminating
    /// generation 0 — the initial population stays a genuinely uniform
    /// random scatter over the whole search space, so its logged/plotted
    /// cloud honestly shows the space, and the seeds' effect is cleanly
    /// separable from it. Seeds beyond `population_size` are ignored;
    /// wrong-dimension seeds are skipped (defensive — a truncated
    /// individual would panic deep inside crossover), not clamped-and-
    /// padded.
    pub fn run_seeded_with_population_progress<F, P>(
        &self,
        bounds: &[(f64, f64)],
        seeds: &[Vec<f64>],
        mut fitness: F,
        mut on_population: P,
    ) -> GaResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
        P: FnMut(usize, &[Vec<f64>], &[f64]) -> bool,
    {
        let mut rng = SplitMix64::new(self.seed);
        let n_dim = bounds.len();

        let mut population: Vec<Vec<f64>> = (0..self.population_size)
            .map(|_| random_individual(bounds, &mut rng))
            .collect();
        let clamped_seeds: Vec<Vec<f64>> = seeds
            .iter()
            .filter(|s| s.len() == n_dim)
            .map(|seed| seed.iter().zip(bounds).map(|(&v, &(lo, hi))| v.clamp(lo, hi)).collect())
            .collect();
        let mut fitnesses: Vec<f64> =
            population.iter().map(|ind| fitness(ind).unwrap_or(f64::MAX)).collect();
        if !on_population(0, &population, &fitnesses) {
            return ga_result_from(&population, &fitnesses, Vec::new());
        }

        let mut history = Vec::with_capacity(self.generations);

        for gen in 0..self.generations {
            let progress = gen as f64 / self.generations.max(1) as f64;
            let decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            let min_rate = self.mutation_rate * 0.1;
            let adaptive_rate = min_rate + (self.mutation_rate - min_rate) * decay;

            let mut indexed: Vec<usize> = (0..population.len()).collect();
            indexed.sort_by(|&a, &b| fitnesses[a].partial_cmp(&fitnesses[b]).unwrap());

            let elitism_count = self.elitism_count.min(population.len());
            let mut new_population: Vec<Vec<f64>> = indexed[..elitism_count]
                .iter()
                .map(|&i| population[i].clone())
                .collect();

            while new_population.len() < self.population_size {
                let p1 = tournament_select(&population, &fitnesses, self.tournament_size, &mut rng);
                let p2 = tournament_select(&population, &fitnesses, self.tournament_size, &mut rng);

                let child = if rng.next_f64() < self.crossover_rate {
                    (0..n_dim)
                        .map(|i| {
                            let t = rng.next_f64();
                            t * p1[i] + (1.0 - t) * p2[i]
                        })
                        .collect::<Vec<f64>>()
                } else {
                    p1.clone()
                };

                let mutated: Vec<f64> = child
                    .iter()
                    .zip(bounds)
                    .map(|(&v, &(lo, hi))| {
                        if rng.next_f64() < adaptive_rate {
                            let span = hi - lo;
                            (v + rng.next_gaussian() * span * 0.1).clamp(lo, hi)
                        } else {
                            v
                        }
                    })
                    .collect();

                new_population.push(mutated);
            }

            // Seed injection at GENERATION 1 (see this method's doc
            // comment): overwrite the TAIL of the first evolved
            // population, so elites carried from the random generation 0
            // survive alongside the seeds.
            if gen == 0 && !clamped_seeds.is_empty() {
                let n = new_population.len();
                for (i, seed) in clamped_seeds.iter().take(n).enumerate() {
                    new_population[n - 1 - i] = seed.clone();
                }
            }

            population = new_population;
            fitnesses = population.iter().map(|ind| fitness(ind).unwrap_or(f64::MAX)).collect();

            let best = fitnesses.iter().cloned().fold(f64::MAX, f64::min);
            history.push(best);
            if !on_population(gen + 1, &population, &fitnesses) {
                return ga_result_from(&population, &fitnesses, history);
            }
        }

        ga_result_from(&population, &fitnesses, history)
    }

    /// Same as [`Self::run`], but calls `on_generation(generation_index,
    /// best_fitness_so_far)` after each generation — for streaming live
    /// progress (Phase 9f) when the fitness function is expensive (e.g.
    /// real propagation per evaluation, not a closed-form Lambert solve).
    ///
    /// `on_generation` returns `true` to continue, `false` to stop early
    /// (job cancellation, Phase 9k task 4).
    pub fn run_with_progress<F, P>(&self, bounds: &[(f64, f64)], mut fitness: F, mut on_generation: P) -> GaResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
        P: FnMut(usize, f64) -> bool,
    {
        let mut rng = SplitMix64::new(self.seed);
        let n_dim = bounds.len();

        let mut population: Vec<Vec<f64>> = (0..self.population_size)
            .map(|_| random_individual(bounds, &mut rng))
            .collect();
        let mut fitnesses: Vec<f64> =
            population.iter().map(|ind| fitness(ind).unwrap_or(f64::MAX)).collect();

        let mut history = Vec::with_capacity(self.generations);

        for gen in 0..self.generations {
            // Adaptive mutation: cosine decay from mutation_rate to 10% of
            // it — early generations explore broadly, late generations
            // refine (same schedule as OptimizationProblems's GA).
            let progress = gen as f64 / self.generations.max(1) as f64;
            let decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            let min_rate = self.mutation_rate * 0.1;
            let adaptive_rate = min_rate + (self.mutation_rate - min_rate) * decay;

            let mut indexed: Vec<usize> = (0..population.len()).collect();
            indexed.sort_by(|&a, &b| fitnesses[a].partial_cmp(&fitnesses[b]).unwrap());

            let elitism_count = self.elitism_count.min(population.len());
            let mut new_population: Vec<Vec<f64>> = indexed[..elitism_count]
                .iter()
                .map(|&i| population[i].clone())
                .collect();

            while new_population.len() < self.population_size {
                let p1 = tournament_select(&population, &fitnesses, self.tournament_size, &mut rng);
                let p2 = tournament_select(&population, &fitnesses, self.tournament_size, &mut rng);

                let child = if rng.next_f64() < self.crossover_rate {
                    // Blend (arithmetic) crossover: one random weight per gene.
                    (0..n_dim)
                        .map(|i| {
                            let t = rng.next_f64();
                            t * p1[i] + (1.0 - t) * p2[i]
                        })
                        .collect::<Vec<f64>>()
                } else {
                    p1.clone()
                };

                let mutated: Vec<f64> = child
                    .iter()
                    .zip(bounds)
                    .map(|(&v, &(lo, hi))| {
                        if rng.next_f64() < adaptive_rate {
                            let span = hi - lo;
                            (v + rng.next_gaussian() * span * 0.1).clamp(lo, hi)
                        } else {
                            v
                        }
                    })
                    .collect();

                new_population.push(mutated);
            }

            population = new_population;
            fitnesses = population.iter().map(|ind| fitness(ind).unwrap_or(f64::MAX)).collect();

            let best = fitnesses.iter().cloned().fold(f64::MAX, f64::min);
            history.push(best);
            if !on_generation(gen, best) {
                return ga_result_from(&population, &fitnesses, history);
            }
        }

        ga_result_from(&population, &fitnesses, history)
    }
}

fn ga_result_from(population: &[Vec<f64>], fitnesses: &[f64], history: Vec<f64>) -> GaResult {
    let best_idx = fitnesses
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    GaResult {
        best_params: population[best_idx].clone(),
        best_fitness: fitnesses[best_idx],
        history,
    }
}

fn random_individual(bounds: &[(f64, f64)], rng: &mut SplitMix64) -> Vec<f64> {
    bounds.iter().map(|&(lo, hi)| lo + rng.next_f64() * (hi - lo)).collect()
}

/// Tournament selection: pick `k` random individuals, return the best.
fn tournament_select(
    population: &[Vec<f64>],
    fitnesses: &[f64],
    k: usize,
    rng: &mut SplitMix64,
) -> Vec<f64> {
    let n = population.len();
    let mut best_idx = ((rng.next_f64() * n as f64) as usize).min(n - 1);
    for _ in 1..k {
        let idx = ((rng.next_f64() * n as f64) as usize).min(n - 1);
        if fitnesses[idx] < fitnesses[best_idx] {
            best_idx = idx;
        }
    }
    population[best_idx].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimize a synthetic paraboloid from a wide bound and confirm the GA
    /// converges close to the true minimum — no orbital mechanics involved,
    /// just validating the algorithm mechanics in isolation.
    #[test]
    fn recovers_paraboloid_minimum() {
        let solver = GaSolver {
            population_size: 60,
            generations: 80,
            crossover_rate: 0.8,
            mutation_rate: 0.3,
            elitism_count: 2,
            tournament_size: 3,
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
        assert_eq!(result.history.len(), 80);
        // Convergence history should be monotonically non-increasing — the
        // best-so-far metric, not a true generation-by-generation value.
        for i in 1..result.history.len() {
            assert!(result.history[i] <= result.history[i - 1]);
        }
    }

    /// Two runs with the same seed must be bit-identical — reproducibility
    /// matters for debugging and for regenerating a reported "best" solution.
    #[test]
    fn same_seed_is_reproducible() {
        let make = || GaSolver {
            population_size: 20,
            generations: 10,
            crossover_rate: 0.8,
            mutation_rate: 0.3,
            elitism_count: 1,
            tournament_size: 3,
            seed: 7,
        };
        let a = make().run(&[(-10.0, 10.0)], |p| Some(p[0] * p[0]));
        let b = make().run(&[(-10.0, 10.0)], |p| Some(p[0] * p[0]));
        assert_eq!(a.best_params, b.best_params);
        assert_eq!(a.history, b.history);
    }

    /// A seeded run must (a) keep GENERATION 0 purely random (
    /// revision -- seeds inject at generation 1, so the initial cloud
    /// honestly samples the whole space), (b) actually evaluate the seeds
    /// at generation 1 (an exact-optimum seed makes gen-1 best 0,
    /// something a random 20-individual population in a (-50,50)^2 box
    /// essentially never achieves by chance), (c) clamp an out-of-bounds
    /// seed instead of letting it escape the search box, and (d) skip a
    /// wrong-dimension seed instead of panicking.
    #[test]
    fn seeded_run_injects_at_gen1_clamps_and_dimension_checks_seeds() {
        let solver = GaSolver {
            population_size: 20,
            generations: 2,
            crossover_rate: 0.8,
            mutation_rate: 0.3,
            elitism_count: 2,
            tournament_size: 3,
            seed: 42,
        };
        let true_min = [3.0, -2.0];
        let f = |p: &[f64]| Some((p[0] - true_min[0]).powi(2) + (p[1] - true_min[1]).powi(2));
        let bounds = [(-50.0, 50.0), (-50.0, 50.0)];

        let mut gen0_best = f64::MAX;
        let mut gen1_best = f64::MAX;
        let seeds = vec![
            vec![3.0, -2.0],       // the exact optimum
            vec![999.0, -999.0],   // out of bounds -- must clamp, not escape
            vec![1.0],             // wrong dimension -- must be skipped, not panic
        ];
        let result = solver.run_seeded_with_population_progress(&bounds, &seeds, f, |gen, _pop, fits| {
            let best = fits.iter().cloned().fold(f64::MAX, f64::min);
            if gen == 0 { gen0_best = best; }
            if gen == 1 { gen1_best = best; }
            true
        });
        assert!(gen0_best > 1e-6, "generation 0 must stay purely random (no seed injected), gen0_best={gen0_best}");
        assert!(gen1_best < 1e-12, "the exact-optimum seed must appear in generation 1, gen1_best={gen1_best}");
        assert!(result.best_fitness < 1e-12);
        for &v in &result.best_params {
            assert!((-50.0..=50.0).contains(&v));
        }
    }

    /// An `None`-returning fitness (infeasible point) must not panic and
    /// must be steered away from by the GA, not selected as "best".
    #[test]
    fn infeasible_points_are_penalized_not_panicking() {
        let solver = GaSolver {
            population_size: 30,
            generations: 20,
            crossover_rate: 0.8,
            mutation_rate: 0.3,
            elitism_count: 2,
            tournament_size: 3,
            seed: 1,
        };
        let result = solver.run(&[(-10.0, 10.0)], |p| {
            if p[0] < 0.0 { None } else { Some(p[0]) }
        });
        assert!(result.best_fitness >= 0.0, "GA should avoid the infeasible region");
    }
}
