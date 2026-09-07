//! SHADE — Success-History based Adaptive Differential Evolution.
//!
//! Self-adaptive DE variant: instead of fixed mutation scale F and crossover
//! probability CR (which must be hand-tuned per problem), SHADE maintains a
//! small success-history memory of (F, CR) pairs that recently produced
//! improvements, and samples each individual's parameters from that memory.
//! Mutation is current-to-pbest/1 with an external archive of replaced
//! parents (inherited from JADE), which balances exploitation of the current
//! best region against the diversity stored in the archive.
//!
//! On multimodal, narrow-valley landscapes (MGA-DSM trajectory problems are
//! the canonical case) success-history adaptation consistently outperforms
//! fixed-parameter DE/rand/1/bin at equal evaluation budget — the adapted
//! parameters are problem-independent by construction, which is exactly the
//! property wanted for a generic mission-design tool (no per-mission tuning).
//!
//! # References
//! - Tanabe & Fukunaga (2013), "Success-History Based Parameter Adaptation
//!   for Differential Evolution", IEEE CEC 2013, pp. 71–78.
//! - Zhang & Sanderson (2009), "JADE: Adaptive Differential Evolution with
//!   Optional External Archive", IEEE Trans. Evol. Comput. 13(5):945–958
//!   (current-to-pbest/1 mutation and the external archive).
//! - Storn & Price (1997), J. Global Optimization 11:341–359 (base DE).

use crate::de::DeResult;
use crate::monte_carlo::SplitMix64;

/// Number of success-history memory slots (H in Tanabe & Fukunaga 2013;
/// values 5–10 are standard and insensitive).
const MEMORY_SIZE: usize = 6;

/// Scale of the Cauchy (F) and normal (CR) perturbations around the memory
/// means (0.1 in both papers).
const PARAM_SIGMA: f64 = 0.1;

/// p-best fraction bounds: each individual draws p ∈ [2/NP, P_BEST_MAX] and
/// targets a random member of the top p fraction (Tanabe & Fukunaga 2013 §III).
const P_BEST_MAX: f64 = 0.2;

/// SHADE minimiser for bounded real-valued parameters.
///
/// `f_init`/`cr_init` seed the success-history memory (a reasonable classic-DE
/// setting such as F=0.8, CR=0.9 works; the memory adapts away from it within
/// a few generations). Infeasible fitness (`None`) is penalised to `f64::MAX`,
/// same convention as [`crate::DeSolver`].
pub struct ShadeSolver {
    pub population_size: usize,
    pub generations: usize,
    pub seed: u64,
    /// Initial mean for the F success-history memory.
    pub f_init: f64,
    /// Initial mean for the CR success-history memory.
    pub cr_init: f64,
}

impl ShadeSolver {
    /// Minimise `fitness` over `bounds`, optionally seeding part of the
    /// initial population with `seeds` (clamped to bounds; the remainder is
    /// initialised uniformly at random). Returns the usual [`DeResult`] plus
    /// the final population as `(params, fitness)` pairs, so callers can
    /// harvest elites for a later refinement stage.
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
        let np    = self.population_size.max(4);

        // Initial population: seeds first (clamped), random fill after.
        let mut pop: Vec<Vec<f64>> = Vec::with_capacity(np);
        for s in seeds.iter().take(np) {
            let clamped: Vec<f64> = (0..n_dim)
                .map(|j| s.get(j).copied().unwrap_or(0.5 * (bounds[j].0 + bounds[j].1))
                    .clamp(bounds[j].0, bounds[j].1))
                .collect();
            pop.push(clamped);
        }
        while pop.len() < np {
            pop.push(bounds.iter()
                .map(|(lo, hi)| lo + rng.next_f64() * (hi - lo))
                .collect());
        }

        let mut fits: Vec<f64> = pop.iter()
            .map(|ind| fitness(ind).unwrap_or(f64::MAX))
            .collect();

        let mut best_idx = argmin(&fits);

        // Success-history memories and the external archive of replaced parents.
        let mut m_f  = vec![self.f_init.clamp(0.05, 1.0); MEMORY_SIZE];
        let mut m_cr = vec![self.cr_init.clamp(0.0, 1.0); MEMORY_SIZE];
        let mut mem_idx = 0usize;
        let mut archive: Vec<Vec<f64>> = Vec::with_capacity(np);

        let mut history = Vec::with_capacity(self.generations);
        let mut param_history = Vec::with_capacity(self.generations);

        for gen in 0..self.generations {
            // Fitness-sorted index for p-best selection.
            let mut order: Vec<usize> = (0..np).collect();
            order.sort_by(|&a, &b| fits[a].partial_cmp(&fits[b]).unwrap());

            let mut s_f:  Vec<f64> = Vec::new();
            let mut s_cr: Vec<f64> = Vec::new();
            let mut s_w:  Vec<f64> = Vec::new();

            for i in 0..np {
                // Sample F (Cauchy) and CR (normal) from a random memory slot.
                let slot = (rng.next_f64() * MEMORY_SIZE as f64) as usize % MEMORY_SIZE;
                let f_i = loop {
                    let c = m_f[slot]
                        + PARAM_SIGMA * (std::f64::consts::PI * (rng.next_f64() - 0.5)).tan();
                    if c > 0.0 { break c.min(1.0); }
                };
                let cr_i = (m_cr[slot] + PARAM_SIGMA * rng.next_gaussian()).clamp(0.0, 1.0);

                // current-to-pbest/1: v = x_i + F·(x_pbest − x_i) + F·(x_r1 − x_r2),
                // r1 from the population, r2 from population ∪ archive.
                let p_i = 2.0 / np as f64
                    + rng.next_f64() * (P_BEST_MAX - 2.0 / np as f64).max(0.0);
                let n_pbest = ((p_i * np as f64).ceil() as usize).max(1);
                let pbest = order[(rng.next_f64() * n_pbest as f64) as usize % n_pbest];

                let r1 = loop {
                    let v = (rng.next_f64() * np as f64) as usize % np;
                    if v != i { break v; }
                };
                let pool = np + archive.len();
                let (r2_pop, r2_arc) = loop {
                    let v = (rng.next_f64() * pool as f64) as usize % pool;
                    if v != i && v != r1 { break (v < np, if v < np { v } else { v - np }); }
                };
                let x_r2: &[f64] = if r2_pop { &pop[r2_arc] } else { &archive[r2_arc] };

                let trial: Vec<f64> = {
                    let jrand = (rng.next_f64() * n_dim as f64) as usize % n_dim;
                    (0..n_dim).map(|j| {
                        if j == jrand || rng.next_f64() < cr_i {
                            let v = pop[i][j]
                                + f_i * (pop[pbest][j] - pop[i][j])
                                + f_i * (pop[r1][j] - x_r2[j]);
                            // Midpoint repair toward the parent (SHADE §III-B):
                            // keeps repaired genes interior instead of piling
                            // the population up on the box faces.
                            if v < bounds[j].0 {
                                0.5 * (bounds[j].0 + pop[i][j])
                            } else if v > bounds[j].1 {
                                0.5 * (bounds[j].1 + pop[i][j])
                            } else {
                                v
                            }
                        } else {
                            pop[i][j]
                        }
                    }).collect()
                };

                let trial_fit = fitness(&trial).unwrap_or(f64::MAX);
                if trial_fit < fits[i] {
                    // Improvement weight for the memory update; guard the
                    // f64::MAX sentinel producing inf/NaN weights.
                    let w = (fits[i] - trial_fit).min(1e12);
                    s_f.push(f_i);
                    s_cr.push(cr_i);
                    s_w.push(if w.is_finite() && w > 0.0 { w } else { 1.0 });

                    if archive.len() >= np {
                        let evict = (rng.next_f64() * archive.len() as f64) as usize
                            % archive.len();
                        archive.swap_remove(evict);
                    }
                    archive.push(std::mem::replace(&mut pop[i], trial));
                    fits[i] = trial_fit;
                    if trial_fit < fits[best_idx] {
                        best_idx = i;
                    }
                }
            }

            // Memory update: weighted Lehmer mean for F, weighted arithmetic
            // mean for CR (Tanabe & Fukunaga 2013, eqs. 4–6).
            if !s_f.is_empty() {
                let wsum: f64 = s_w.iter().sum();
                let lehmer_num: f64 = s_w.iter().zip(&s_f).map(|(w, f)| w * f * f).sum();
                let lehmer_den: f64 = s_w.iter().zip(&s_f).map(|(w, f)| w * f).sum();
                if lehmer_den > 0.0 {
                    m_f[mem_idx] = (lehmer_num / lehmer_den).clamp(0.05, 1.0);
                }
                if wsum > 0.0 {
                    m_cr[mem_idx] = (s_w.iter().zip(&s_cr).map(|(w, c)| w * c).sum::<f64>()
                        / wsum).clamp(0.0, 1.0);
                }
                mem_idx = (mem_idx + 1) % MEMORY_SIZE;
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

fn argmin(fits: &[f64]) -> usize {
    fits.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rosenbrock: global minimum at (1, 1), f = 0 — same acceptance bar as
    /// the classic DeSolver test so the two solvers are directly comparable.
    #[test]
    fn minimises_rosenbrock() {
        let shade = ShadeSolver {
            population_size: 50,
            generations:     500,
            seed:            42,
            f_init:          0.8,
            cr_init:         0.9,
        };
        let bounds = vec![(-2.0_f64, 2.0), (-2.0, 2.0)];
        let (result, _) = shade.run_seeded_with_progress(&bounds, &[], |x| {
            Some((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
        }, |_, _, _| {});
        assert!(result.best_fitness < 1e-6,
            "Rosenbrock minimum not found: f={:.3e} at ({:.4}, {:.4})",
            result.best_fitness, result.best_params[0], result.best_params[1]);
        assert!((result.best_params[0] - 1.0).abs() < 1e-3);
        assert!((result.best_params[1] - 1.0).abs() < 1e-3);
    }

    /// All-infeasible fitness must not panic and must return the sentinel.
    #[test]
    fn handles_infeasible_fitness() {
        let shade = ShadeSolver {
            population_size: 20,
            generations:     50,
            seed:            1,
            f_init:          0.8,
            cr_init:         0.9,
        };
        let bounds = vec![(0.0_f64, 1.0)];
        let (result, _) = shade.run_seeded_with_progress(&bounds, &[], |_| None, |_, _, _| {});
        assert_eq!(result.best_fitness, f64::MAX);
    }

    /// A seed placed at the optimum must never be lost: the result can only
    /// be at least as good as the best injected seed (greedy selection).
    #[test]
    fn seeded_run_preserves_seed_quality() {
        let shade = ShadeSolver {
            population_size: 20,
            generations:     10,
            seed:            7,
            f_init:          0.8,
            cr_init:         0.9,
        };
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let sphere = |x: &[f64]| Some(x[0] * x[0] + x[1] * x[1]);
        let (result, population) = shade.run_seeded_with_progress(
            &bounds, &[vec![0.0, 0.0]], sphere, |_, _, _| {});
        assert!(result.best_fitness <= 1e-12,
            "seed at optimum lost: f={:.3e}", result.best_fitness);
        assert_eq!(population.len(), 20);
    }
}
