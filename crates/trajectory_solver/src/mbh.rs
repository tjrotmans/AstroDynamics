//! Monotonic Basin Hopping (MBH) global optimizer for bounded continuous
//! parameter spaces.
//!
//! Unlike population-based methods ([`crate::DeSolver`], [`crate::ShadeSolver`]),
//! MBH tracks exactly one current candidate per chain: perturb it (a "kick"),
//! run a local descent ([`crate::NelderMead`]) from the perturbed point, and
//! accept the result only if it is strictly better than the current candidate
//! — "monotonic" in the name. This sidesteps a real failure mode of
//! population-based search on tightly-coupled, multi-modal landscapes (MGA-DSM
//! trajectory problems are the motivating case here): a genuinely good
//! candidate injected as a seed can still be discarded by a population's own
//! crossover/selection dynamics through no fault of its own quality, because a
//! single injected individual reshapes every subsequent generation's mutation-
//! parent selection. A basin-hopping chain has no population to lose a
//! competition against — a candidate is only ever replaced by something
//! strictly better than itself.
//!
//! MBH is the search method behind ESA's own GTOP reference solutions and
//! NASA Goddard's EMTG trajectory design tool.
//!
//! # References
//! - Wales, D. J. & Doye, J. P. K. (1997), "Global Optimization by
//!   Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters
//!   Containing up to 110 Atoms", J. Phys. Chem. A 101(28):5111–5116
//!   (basin hopping, originally for molecular potential-energy landscapes).
//! - Yam, C. H., Di Lorenzo, D. & Izzo, D. (2011), "Low-Thrust Trajectory
//!   Design as a Constrained Global Optimization Problem", Proc. IMechE Part
//!   G 225(11):1243–1251 (MBH applied to spacecraft trajectory optimization;
//!   basis for ESA GTOP's reference solutions).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::compass_search::CompassSearch;
use crate::hooke_jeeves::HookeJeevesSearch;
use crate::monte_carlo::SplitMix64;
use crate::nelder_mead::{NelderMead, NmResult};

/// Selectable inner local descent for [`MbhSolver`] — [`NelderMead`] (this
/// crate's prior sole option), [`CompassSearch`] (a best-of-all-directions
/// pattern search, added), or [`HookeJeevesSearch`] (a greedy
/// accept-first-improving pattern search with an evaluation-budget
/// termination rather than an iteration-count one, added to
/// match the shallow-per-hop-descent cost allocation real MBH reference
/// implementations use by default — see `HookeJeevesSearch`'s own doc
/// comment). All three share the same `run(bounds, x0, fitness) ->
/// NmResult` shape, so this is a plain dispatch wrapper, not a trait
/// object — avoids `dyn`/boxing on a hot path called once per hop.
pub enum LocalOptimizer {
    NelderMead(NelderMead),
    CompassSearch(CompassSearch),
    HookeJeeves(HookeJeevesSearch),
}

impl LocalOptimizer {
    fn run<F>(&self, bounds: &[(f64, f64)], x0: &[f64], fitness: F) -> NmResult
    where
        F: FnMut(&[f64]) -> Option<f64>,
    {
        match self {
            LocalOptimizer::NelderMead(nm) => nm.run(bounds, x0, fitness),
            LocalOptimizer::CompassSearch(cs) => cs.run(bounds, x0, fitness),
            LocalOptimizer::HookeJeeves(hj) => hj.run(bounds, x0, fitness),
        }
    }
}

impl From<NelderMead> for LocalOptimizer {
    fn from(nm: NelderMead) -> Self {
        LocalOptimizer::NelderMead(nm)
    }
}

impl From<CompassSearch> for LocalOptimizer {
    fn from(cs: CompassSearch) -> Self {
        LocalOptimizer::CompassSearch(cs)
    }
}

impl From<HookeJeevesSearch> for LocalOptimizer {
    fn from(hj: HookeJeevesSearch) -> Self {
        LocalOptimizer::HookeJeeves(hj)
    }
}

/// Result of an [`MbhSolver::run`] call.
#[derive(Clone, Debug)]
pub struct MbhResult {
    pub best_params: Vec<f64>,
    pub best_fitness: f64,
    /// Total number of hops actually run, summed across every chain.
    pub hops: usize,
    /// Global-best-so-far fitness, one entry per hop (all chains
    /// concatenated) — for convergence history plots.
    pub history: Vec<f64>,
    /// Global-best-so-far PARAMETER vector, one entry per hop, parallel to
    /// `history` (for per-parameter convergence plots).
    pub param_history: Vec<Vec<f64>>,
    /// Each chain's own final incumbent fitness, in chain-index order
    /// — the direct observable for "how many chains actually
    /// contributed," which previously had to be reconstructed by hand from
    /// concatenated history row counts.
    pub chain_bests: Vec<f64>,
}

/// Monotonic Basin Hopping minimiser for bounded real-valued parameters.
///
/// One hop chain runs per seed in [`MbhSolver::run_seeded_with_progress`]'s
/// `seeds` argument (or a single randomly-initialised chain if `seeds` is
/// empty). Each chain starts with a local descent from its own start point,
/// then repeats `hops` times: perturb a random subset of variables (a
/// "kick", magnitude scaled from each variable's own bounds width), run a
/// local descent from the kicked point, and keep the result only if it beats
/// the chain's current candidate. The global best across every chain is
/// returned — this is what gives MBH the multi-start diversity a DE
/// population provides implicitly.
pub struct MbhSolver {
    /// Number of hops per chain (after the initial descent).
    pub hops: usize,
    /// Fraction of variables perturbed on each kick, drawn independently
    /// per variable. `1.0` perturbs every variable every hop.
    pub perturb_fraction: f64,
    /// Kick magnitude as a fraction of each variable's bounds width
    /// (uniform in `[-kick_scale, +kick_scale] * (hi - lo)`).
    pub kick_scale: f64,
    /// Local descent configuration used for both the initial descent and
    /// every post-kick refinement.
    pub local: LocalOptimizer,
    pub seed: u64,
    /// Early-stop-on-stagnation (pagmo's `m_stop`): a chain terminates after
    /// this many consecutive non-improving hops instead of always burning
    /// its full `hops` budget. `None` (default) disables early stop —
    /// byte-identical to the pre-behaviour. Added per the
    /// handoff's Priority 1: real pagmo2 MBH (`src/algorithms/
    /// mbh.cpp`) defaults `stop = 5`, a very aggressive early-stop that lets
    /// its reference runs spend a fixed budget on many short, cheaply-
    /// abandoned chains rather than one long chain forced to keep
    /// perturbing a basin it has already exhausted.
    pub stop_after: Option<usize>,
    /// Global-relative early stop: `stop_after` alone only
    /// measures a chain's stagnation relative to ITS OWN incumbent — a
    /// chain can keep finding tiny improvements over itself indefinitely
    /// (dodging `stop_after`'s consecutive-failure counter) while remaining
    /// hopelessly worse than what a better-seeded chain already found
    /// elsewhere, burning its full `hops` budget for nothing. Found via a
    /// real run: a 12-chain/500-hop MBH search where every chain used its
    /// complete budget (`stop_after` never fired once) yet 6 of the 12
    /// chains never beat a result found within the first 6 chains — half
    /// the compute was wasted with no mechanism to reallocate it. `None`
    /// (default) disables this check, byte-identical to prior behaviour.
    pub global_stall: Option<GlobalStallConfig>,
    /// Extra purely-randomly-initialised chains added ALONGSIDE whatever
    /// `seeds` are passed to `run`/`run_seeded_with_progress`.
    /// When `seeds` is non-empty (the normal case for a real MGA search,
    /// seeded from pruning/bidirectional-backfit candidates), the ENTIRE
    /// search was previously 100% dependent on those informed seeds having
    /// found the right basin — no blind exploration ever happened. `0`
    /// (default) is byte-identical to prior behaviour (still exactly one
    /// random chain when `seeds` is empty, unchanged).
    pub extra_random_chains: usize,
    /// Archipelago-style migration: periodically inject the
    /// live global best solution INTO chains that are clearly losing, while
    /// they are still running — pagmo's archipelago/island model's core
    /// idea, applied to this solver's existing multi-chain phase rather
    /// than as a separate architectural layer. Motivated by a measured
    /// waste: in a real 12-chain/500-hop overnight run, 6 of the 12 chains
    /// burned their full budget without ever contributing to the final
    /// result. `global_stall` reallocates that waste by KILLING a losing
    /// chain; migration instead REPURPOSES it to explore around the best
    /// known basin with its own independent kick stream. `None` (default)
    /// disables migration — byte-identical to prior behaviour.
    pub migration: Option<MigrationConfig>,
}

/// Configuration for [`MbhSolver::global_stall`] — see its doc comment for
/// the motivation.
#[derive(Clone, Copy, Debug)]
pub struct GlobalStallConfig {
    /// Hops a chain is given before its incumbent is checked against the
    /// live global best-so-far (observed across all chains, read from the
    /// same shared progress state `on_hop` already reports from).
    pub patience: usize,
    /// A chain is abandoned once past `patience` hops if its own incumbent
    /// fitness exceeds `global_best * (1.0 + margin_frac)` — i.e. it is not
    /// within `margin_frac` (a fraction, e.g. `0.5` = within 50%) of the
    /// best result any chain has found so far. Assumes fitness values are
    /// non-negative (true for every ΔV-based objective in this codebase).
    pub margin_frac: f64,
}

/// Configuration for [`MbhSolver::migration`] — see its doc comment for the
/// motivation.
#[derive(Clone, Copy, Debug)]
pub struct MigrationConfig {
    /// Chain-local hop interval between migration checks (a chain considers
    /// receiving a migrant every `interval` of its OWN hops — no cross-chain
    /// synchronisation, chains stay fully independent workers).
    pub interval: usize,
    /// A chain receives the global best only if its own incumbent fitness
    /// exceeds `global_best * (1.0 + margin_frac)` — same semantics as
    /// [`GlobalStallConfig::margin_frac`]. Chains already competitive within
    /// the margin are left alone (they may be exploring a genuinely
    /// different, promising basin). Assumes non-negative fitness values
    /// (true for every ΔV-based objective in this codebase).
    pub margin_frac: f64,
}

impl MigrationConfig {
    /// Pure decision logic, factored out for direct deterministic unit
    /// testing (same pattern as [`GlobalStallConfig::triggered`]).
    fn due(&self, hop_idx: usize, current_fitness: f64, global_best: f64) -> bool {
        self.interval > 0
            && (hop_idx + 1) % self.interval == 0
            && global_best.is_finite()
            && current_fitness > global_best * (1.0 + self.margin_frac)
    }
}

impl GlobalStallConfig {
    /// Pure decision logic, factored out so it's directly unit-testable
    /// without depending on real thread timing (the live multi-chain
    /// behaviour is a thin, non-decisional wrapper around this).
    fn triggered(&self, hop_idx: usize, current_fitness: f64, global_best: f64) -> bool {
        hop_idx + 1 >= self.patience
            && global_best.is_finite()
            && current_fitness > global_best * (1.0 + self.margin_frac)
    }
}

impl MbhSolver {
    /// Minimise `fitness` over `bounds`, one chain per entry in `seeds`
    /// (clamped to bounds; a single randomly-initialised chain if `seeds`
    /// is empty).
    pub fn run<F>(&self, bounds: &[(f64, f64)], seeds: &[Vec<f64>], fitness: F) -> MbhResult
    where
        F: Fn(&[f64]) -> Option<f64> + Sync,
    {
        self.run_seeded_with_progress(bounds, seeds, fitness, |_hop, _best, _params| {})
    }

    /// Same as [`MbhSolver::run`] but calls `on_hop(hop_index,
    /// global_best_fitness_so_far, global_best_params_so_far)` as hops
    /// complete — for live progress streams.
    ///
    /// # Parallelism
    ///
    /// Chains run in parallel across `std::thread::available_parallelism()`
    /// workers — they are fully independent by construction (MBH's whole
    /// design is one isolated candidate per chain), so this is a pure
    /// wall-clock optimization. `fitness` must therefore be `Fn + Sync`
    /// (previously `FnMut`): it is shared read-only across workers.
    ///
    /// Determinism: each chain owns its own [`SplitMix64`] stream derived
    /// from `self.seed` and the CHAIN INDEX, so the final result is
    /// bit-identical regardless of worker count or scheduling order — a
    /// stronger property than the previous sequential implementation, whose
    /// single shared RNG stream made every chain's kicks depend on how many
    /// hops earlier chains had consumed. (Consequence: results differ from
    /// the pre-parallel version's for the same `seed` value — per-seed
    /// reproducibility across code versions is already a non-goal in this
    /// codebase; compare quality distributions, not per-seed numbers.)
    /// `MbhResult::history` is likewise deterministic: per-chain histories
    /// are merged in chain order after completion, as a running global
    /// minimum. Only the LIVE `on_hop` stream is scheduling-dependent (it
    /// reports real-time progress from a shared counter, polled by the
    /// calling thread — which also keeps `on_hop` free of any `Send`
    /// bound, so no caller signature changes).
    ///
    /// Caveat (applies only when configured): `global_stall` and
    /// `migration` both read the LIVE global best across chains, which is
    /// inherently scheduling-dependent — runs with either enabled are not
    /// bit-reproducible across machines/loads. With both `None` (the
    /// defaults) the bit-determinism above holds unchanged.
    pub fn run_seeded_with_progress<F, G>(
        &self,
        bounds: &[(f64, f64)],
        seeds: &[Vec<f64>],
        fitness: F,
        mut on_hop: G,
    ) -> MbhResult
    where
        F: Fn(&[f64]) -> Option<f64> + Sync,
        G: FnMut(usize, f64, &[f64]),
    {
        let n = bounds.len();

        // NOTE: the "no seeds" branch below MUST keep using `self.seed`
        // directly, unmodified -- it is the pre-existing (pre-)
        // single-random-chain fallback, and several real configs (e.g.
        // `evj_flyby.toml`, which has no `[optimization.mga.pruning]`
        // section) rely on `elite_seeds` being empty for MBH, landing here
        // every run. A prior version of this change reseeded this branch
        // and broke `mga_evj_smoke_mbh` (a 15-hop smoke budget is marginal
        // enough that a different starting point flipped it to total
        // infeasibility) -- the extra-random-chains stream below is
        // deliberately a SEPARATE RNG, used only additively, so it can
        // never perturb this pre-existing path.
        let mut starts: Vec<Vec<f64>> = if seeds.is_empty() {
            let mut rng = SplitMix64::new(self.seed);
            vec![bounds.iter().map(|(lo, hi)| lo + rng.next_f64() * (hi - lo)).collect()]
        } else {
            seeds
                .iter()
                .map(|s| {
                    (0..n)
                        .map(|j| {
                            s.get(j)
                                .copied()
                                .unwrap_or(0.5 * (bounds[j].0 + bounds[j].1))
                                .clamp(bounds[j].0, bounds[j].1)
                        })
                        .collect()
                })
                .collect()
        };
        // Dedicated stream for the extra random chains, distinct from the
        // "no seeds" fallback above and the per-chain kick streams below —
        // only ever drawn from when `extra_random_chains > 0`.
        let mut rng_extra = SplitMix64::new(self.seed.wrapping_add(0xC0FF_EE00_C0FF_EE00));
        for _ in 0..self.extra_random_chains {
            starts.push(bounds.iter().map(|(lo, hi)| lo + rng_extra.next_f64() * (hi - lo)).collect());
        }
        let n_chains = starts.len();

        /// Everything one chain produces; merged deterministically after join.
        struct ChainOutcome {
            best_params: Vec<f64>,
            best_fitness: f64,
            /// Chain-local best-so-far, one entry per hop (initial descent
            /// included).
            history: Vec<f64>,
            /// Chain-local best-so-far PARAMETER vector, parallel to `history`.
            param_history: Vec<Vec<f64>>,
        }

        // Shared live-progress state: (hops completed, best fitness seen,
        // best params seen). Workers update it; the calling thread polls it
        // for `on_hop`.
        let progress = Mutex::new((0usize, f64::MAX, Vec::<f64>::new()));
        let next_chain = AtomicUsize::new(0);
        let outcomes: Vec<Mutex<Option<ChainOutcome>>> =
            (0..n_chains).map(|_| Mutex::new(None)).collect();
        let chains_done = AtomicUsize::new(0);

        let workers = std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1)
            .min(n_chains);

        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| {
                    loop {
                        let chain_idx = next_chain.fetch_add(1, Ordering::Relaxed);
                        if chain_idx >= n_chains { break; }

                        // Per-chain RNG stream: seed mixed with the chain
                        // index by a 64-bit odd constant (SplitMix64's own
                        // gamma), so chains are decorrelated and each is
                        // deterministic in isolation.
                        let mut rng = SplitMix64::new(
                            self.seed.wrapping_add((chain_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                        );

                        let start = &starts[chain_idx];
                        let mut history = Vec::with_capacity(self.hops + 1);
                        let mut param_history = Vec::with_capacity(self.hops + 1);

                        let descended = self.local.run(bounds, start, |p| fitness(p));
                        let mut current_params = descended.best_params;
                        let mut current_fitness = descended.best_fitness;
                        history.push(current_fitness);
                        param_history.push(current_params.clone());
                        {
                            let mut pr = progress.lock().unwrap();
                            pr.0 += 1;
                            if current_fitness < pr.1 { pr.1 = current_fitness; pr.2 = current_params.clone(); }
                        }

                        let mut stagnant_hops = 0usize;
                        for hop_idx in 0..self.hops {
                            let mut kicked = current_params.clone();
                            for j in 0..n {
                                if rng.next_f64() < self.perturb_fraction {
                                    let width = bounds[j].1 - bounds[j].0;
                                    let delta = (rng.next_f64() * 2.0 - 1.0) * self.kick_scale * width;
                                    kicked[j] = (kicked[j] + delta).clamp(bounds[j].0, bounds[j].1);
                                }
                            }

                            let descended = self.local.run(bounds, &kicked, |p| fitness(p));
                            if descended.best_fitness < current_fitness {
                                current_fitness = descended.best_fitness;
                                current_params = descended.best_params;
                                stagnant_hops = 0;
                            } else {
                                stagnant_hops += 1;
                            }

                            history.push(current_fitness);
                            param_history.push(current_params.clone());
                            let mut pr = progress.lock().unwrap();
                            pr.0 += 1;
                            if current_fitness < pr.1 { pr.1 = current_fitness; pr.2 = current_params.clone(); }
                            let global_best_snapshot = pr.1;
                            // Migration decision made while the lock is held so
                            // the (fitness, params) pair is a consistent
                            // snapshot; the clone only happens when a migration
                            // actually fires.
                            let migrant: Option<(f64, Vec<f64>)> = match &self.migration {
                                Some(m) if m.due(hop_idx, current_fitness, pr.1) => {
                                    Some((pr.1, pr.2.clone()))
                                }
                                _ => None,
                            };
                            drop(pr);

                            if let Some((mig_f, mig_p)) = migrant {
                                // Inject the global best verbatim as this
                                // chain's new incumbent (fitness already known
                                // — no extra evaluation). Collapse-onto-one-
                                // point is not a real risk: the very next hop
                                // kicks the migrant with this chain's OWN RNG
                                // stream, so exploration diverges immediately
                                // even from identical incumbents. The stagnant
                                // counter resets (fresh material to work on),
                                // and the stall/stop checks are skipped this
                                // hop — migration is a rescue, and takes
                                // precedence over `global_stall`'s kill.
                                current_fitness = mig_f;
                                current_params = mig_p;
                                stagnant_hops = 0;
                                continue;
                            }

                            if let Some(stop_after) = self.stop_after {
                                if stagnant_hops >= stop_after {
                                    break;
                                }
                            }
                            if let Some(gs) = &self.global_stall {
                                if gs.triggered(hop_idx, current_fitness, global_best_snapshot) {
                                    break;
                                }
                            }
                        }

                        *outcomes[chain_idx].lock().unwrap() = Some(ChainOutcome {
                            best_params: current_params,
                            best_fitness: current_fitness,
                            history,
                            param_history,
                        });
                        chains_done.fetch_add(1, Ordering::Release);
                    }
                });
            }

            // Live progress from the calling thread: poll the shared counter
            // and forward NEW hop completions to `on_hop`. Keeps `on_hop`
            // un-Send'd and un-locked — it never leaves this thread.
            let mut reported = 0usize;
            while chains_done.load(Ordering::Acquire) < n_chains {
                std::thread::sleep(std::time::Duration::from_millis(50));
                let (done, best, best_params) = progress.lock().unwrap().clone();
                while reported < done {
                    on_hop(reported, best, &best_params);
                    reported += 1;
                }
            }
            let (done, best, best_params) = progress.lock().unwrap().clone();
            while reported < done {
                on_hop(reported, best, &best_params);
                reported += 1;
            }
        });

        // Deterministic merge, in chain-index order (independent of which
        // worker ran which chain, and of scheduling): global best is the min
        // over chains; the history is each chain's own trace re-scanned as a
        // running global minimum, matching the sequential implementation's
        // best-so-far semantics.
        let mut global_best_params: Vec<f64> = Vec::new();
        let mut global_best_fitness = f64::MAX;
        let mut history = Vec::new();
        let mut param_history: Vec<Vec<f64>> = Vec::new();
        let mut chain_bests = Vec::with_capacity(n_chains);
        for outcome in &outcomes {
            let Some(oc) = outcome.lock().unwrap().take() else { continue };
            chain_bests.push(oc.best_fitness);
            for (f, p) in oc.history.iter().zip(oc.param_history.iter()) {
                if *f < global_best_fitness {
                    global_best_fitness = *f;
                    global_best_params = p.clone();
                }
                history.push(global_best_fitness);
                param_history.push(global_best_params.clone());
            }
            if oc.best_fitness <= global_best_fitness {
                global_best_fitness = oc.best_fitness;
                global_best_params = oc.best_params;
            }
        }

        MbhResult {
            best_params: global_best_params,
            best_fitness: global_best_fitness,
            hops: history.len(),
            param_history,
            history,
            chain_bests,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_local() -> NelderMead {
        NelderMead {
            max_iter: 200,
            ..Default::default()
        }
    }

    /// Sphere function from a bad start: a single local descent already
    /// solves this (convex), so MBH must match it.
    #[test]
    fn minimises_sphere_from_bad_start() {
        let mbh = MbhSolver {
            hops: 5,
            perturb_fraction: 0.5,
            kick_scale: 0.2,
            local: default_local().into(),
            seed: 42,
            stop_after: None,
            global_stall: None,
            extra_random_chains: 0,
            migration: None,
        };
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let result = mbh.run(&bounds, &[vec![4.0, -3.0]], |x| Some(x[0].powi(2) + x[1].powi(2)));
        assert!(
            result.best_fitness < 1e-6,
            "sphere minimum not found: f={:.3e}",
            result.best_fitness
        );
    }

    /// Rastrigin function: many regularly-spaced local minima, global
    /// minimum at the origin with f=0. A lone Nelder-Mead descent from a
    /// distant start gets trapped in the nearest local minimum; MBH's kicks
    /// must escape it and find a materially better point.
    #[test]
    fn basin_hopping_beats_single_local_descent_on_rastrigin() {
        let rastrigin = |x: &[f64]| -> Option<f64> {
            let n = x.len() as f64;
            Some(10.0 * n + x.iter().map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>())
        };
        let bounds = vec![(-5.12_f64, 5.12), (-5.12, 5.12)];
        let start = vec![3.7, -4.1];

        let local_only = default_local().run(&bounds, &start, rastrigin);

        let mbh = MbhSolver {
            hops: 40,
            perturb_fraction: 0.8,
            kick_scale: 0.4,
            local: default_local().into(),
            seed: 7,
            stop_after: None,
            global_stall: None,
            extra_random_chains: 0,
            migration: None,
        };
        let hopped = mbh.run(&bounds, &[start], rastrigin);

        assert!(
            hopped.best_fitness < local_only.best_fitness - 1.0,
            "MBH did not improve materially over a single local descent: \
             local={:.4}, mbh={:.4}",
            local_only.best_fitness,
            hopped.best_fitness
        );
        // Global optimum is f=0; a good basin-hopping run should land close.
        assert!(
            hopped.best_fitness < 5.0,
            "MBH did not approach the global optimum: f={:.4}",
            hopped.best_fitness
        );
    }

    /// Infeasible fitness (`None`) must not panic and must be avoided.
    #[test]
    fn handles_infeasible_fitness() {
        let mbh = MbhSolver {
            hops: 5,
            perturb_fraction: 0.5,
            kick_scale: 0.2,
            local: default_local().into(),
            seed: 1,
            stop_after: None,
            global_stall: None,
            extra_random_chains: 0,
            migration: None,
        };
        let bounds = vec![(0.0_f64, 1.0)];
        let result = mbh.run(&bounds, &[vec![0.5]], |_| None);
        assert_eq!(result.best_fitness, f64::MAX);
    }

    /// A seed already at the optimum must never be made worse — MBH only
    /// ever replaces the current candidate with something strictly better.
    #[test]
    fn seeded_run_preserves_seed_quality() {
        let mbh = MbhSolver {
            hops: 10,
            perturb_fraction: 0.5,
            kick_scale: 0.3,
            local: default_local().into(),
            seed: 3,
            stop_after: None,
            global_stall: None,
            extra_random_chains: 0,
            migration: None,
        };
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let sphere = |x: &[f64]| Some(x[0] * x[0] + x[1] * x[1]);
        let result = mbh.run(&bounds, &[vec![0.0, 0.0]], sphere);
        assert!(
            result.best_fitness <= 1e-12,
            "seed at optimum lost: f={:.3e}",
            result.best_fitness
        );
    }

    /// `stop_after` (early-stop-on-stagnation) must terminate a chain before
    /// its full `hops` budget once no improvement has been found for that
    /// many consecutive hops — the actual number of hops run (reflected in
    /// `MbhResult::hops`, since chains run in parallel and stop
    /// independently) must be strictly less than the unbounded case on a
    /// convex function, where a single descent already finds the optimum
    /// and every subsequent kick is guaranteed non-improving.
    #[test]
    fn stop_after_terminates_a_stagnant_chain_early() {
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let sphere = |x: &[f64]| Some(x[0] * x[0] + x[1] * x[1]);

        let unbounded = MbhSolver {
            hops: 200,
            perturb_fraction: 0.5,
            kick_scale: 0.3,
            local: default_local().into(),
            seed: 11,
            stop_after: None,
            global_stall: None,
            extra_random_chains: 0,
            migration: None,
        };
        let early_stop = MbhSolver {
            hops: 200,
            perturb_fraction: 0.5,
            kick_scale: 0.3,
            local: default_local().into(),
            seed: 11,
            stop_after: Some(5),
            global_stall: None,
            extra_random_chains: 0,
            migration: None,
        };

        let full = unbounded.run(&bounds, &[vec![4.0, -3.0]], sphere);
        let stopped = early_stop.run(&bounds, &[vec![4.0, -3.0]], sphere);

        assert_eq!(full.hops, 201, "unbounded chain should run its full hop budget (+1 initial descent)");
        assert!(
            stopped.hops < full.hops,
            "early-stop chain should terminate before the full budget: stopped={}, full={}",
            stopped.hops, full.hops
        );
        // Early stop must not sacrifice solution quality on a convex function.
        assert!(stopped.best_fitness < 1e-6, "early-stopped chain lost solution quality: f={:.3e}", stopped.best_fitness);
    }

    /// `LocalOptimizer::CompassSearch` must be usable as MBH's inner descent
    /// exactly like `NelderMead` — same interface, same result shape.
    #[test]
    fn compass_search_local_optimizer_wires_into_mbh() {
        let mbh = MbhSolver {
            hops: 20,
            perturb_fraction: 0.5,
            kick_scale: 0.2,
            local: CompassSearch { max_iter: 200, ..Default::default() }.into(),
            seed: 5,
            stop_after: None,
            global_stall: None,
            extra_random_chains: 0,
            migration: None,
        };
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let result = mbh.run(&bounds, &[vec![4.0, -3.0]], |x| Some(x[0].powi(2) + x[1].powi(2)));
        assert!(
            result.best_fitness < 1e-4,
            "sphere minimum not found via CompassSearch: f={:.3e}",
            result.best_fitness
        );
    }

    /// `LocalOptimizer::HookeJeeves` must be usable as MBH's inner descent
    /// too — with pagmo-faithful defaults (`max_fevals=1`, a very shallow
    /// per-hop descent), MBH's own many-hops-plus-early-stop mechanism must
    /// still be able to reach the optimum given enough hops, since each hop
    /// contributes only a small nudge rather than a full local solve.
    #[test]
    fn hooke_jeeves_local_optimizer_wires_into_mbh() {
        let mbh = MbhSolver {
            hops: 300,
            perturb_fraction: 1.0,
            kick_scale: 0.2,
            local: HookeJeevesSearch::default().into(),
            seed: 9,
            stop_after: None,
            global_stall: None,
            extra_random_chains: 0,
            migration: None,
        };
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let result = mbh.run(&bounds, &[vec![4.0, -3.0]], |x| Some(x[0].powi(2) + x[1].powi(2)));
        assert!(
            result.best_fitness < 1e-2,
            "sphere minimum not approached via HookeJeeves: f={:.3e}",
            result.best_fitness
        );
    }

    /// `GlobalStallConfig::triggered`'s decision logic, tested directly and
    /// deterministically (no thread timing involved) — this is the actual
    /// arithmetic that matters; the live multi-chain wiring around it is a
    /// thin, non-decisional wrapper.
    #[test]
    fn global_stall_triggered_pure_logic() {
        let gs = GlobalStallConfig { patience: 5, margin_frac: 0.5 };
        // Before patience is reached: never triggers, no matter how bad.
        assert!(!gs.triggered(2, 1000.0, 10.0), "must not trigger before patience hops");
        // At/past patience, within the margin (threshold = 10 * 1.5 = 15): not triggered.
        assert!(!gs.triggered(10, 12.0, 10.0), "must not trigger within margin_frac of global best");
        // At/past patience, beyond the margin: triggered.
        assert!(gs.triggered(10, 20.0, 10.0), "must trigger once past patience AND beyond margin_frac");
        // Sentinel "no chain has reported a real result yet" (progress starts
        // at f64::MAX) must never spuriously trigger, regardless of hop_idx.
        assert!(!gs.triggered(1000, 5.0, f64::MAX), "must not trigger against the f64::MAX sentinel");
    }

    /// `extra_random_chains` must add chains ALONGSIDE the seeded ones, not
    /// replace them — verified deterministically via total row count (no
    /// early-stop configured, so every chain runs its full `hops` budget):
    /// (1 seeded + 3 extra) chains x (5 hops + 1 initial descent) = 24 rows.
    #[test]
    fn extra_random_chains_adds_chains_alongside_seeds() {
        let mbh = MbhSolver {
            hops: 5,
            perturb_fraction: 0.5,
            kick_scale: 0.2,
            local: default_local().into(),
            seed: 21,
            stop_after: None,
            global_stall: None,
            extra_random_chains: 3,
            migration: None,
        };
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let result = mbh.run(&bounds, &[vec![4.0, -3.0]], |x| Some(x[0].powi(2) + x[1].powi(2)));
        assert_eq!(result.hops, 24, "expected 4 chains (1 seeded + 3 extra random) x 6 rows each");
    }

    /// `global_stall` must actually cut a hopelessly-worse chain short in a
    /// real multi-chain run: one chain seeded exactly at the optimum (so it
    /// reports a near-zero global best almost immediately), one chain
    /// seeded in a region with NO usable gradient at all (a flat plateau,
    /// so its own local descent can never improve it, guaranteeing it would
    /// otherwise burn its entire `hops` budget). With `global_stall`
    /// configured, the plateau chain must be abandoned well before its full
    /// budget once the optimum chain's result becomes visible.
    #[test]
    fn global_stall_cuts_short_a_hopelessly_worse_chain() {
        // f is the sphere near the origin, but perfectly FLAT (constant)
        // for |x| large -- a chain stuck out there has no gradient signal
        // to ever improve on its own starting value.
        let f = |x: &[f64]| -> Option<f64> {
            if x[0].abs() > 50.0 || x[1].abs() > 50.0 {
                Some(9999.0)
            } else {
                Some(x[0] * x[0] + x[1] * x[1])
            }
        };
        let bounds = vec![(-1000.0_f64, 1000.0), (-1000.0, 1000.0)];
        let seeds = vec![vec![0.0, 0.0], vec![500.0, 500.0]];

        let unbounded = MbhSolver {
            hops: 200,
            perturb_fraction: 0.5,
            kick_scale: 0.05,
            local: default_local().into(),
            seed: 3,
            stop_after: None,
            global_stall: None,
            extra_random_chains: 0,
            migration: None,
        };
        let stalled = MbhSolver {
            global_stall: Some(GlobalStallConfig { patience: 5, margin_frac: 0.1 }),
            ..MbhSolver {
                hops: 200,
                perturb_fraction: 0.5,
                kick_scale: 0.05,
                local: default_local().into(),
                seed: 3,
                stop_after: None,
                global_stall: None,
                extra_random_chains: 0,
                migration: None,
            }
        };

        let full = unbounded.run(&bounds, &seeds, f);
        let cut_short = stalled.run(&bounds, &seeds, f);

        assert_eq!(full.hops, 2 * 201, "unbounded: both chains should run their full budget");
        assert!(
            cut_short.hops < full.hops,
            "global_stall should terminate the hopeless plateau chain early: cut_short={}, full={}",
            cut_short.hops, full.hops
        );
        // Solution quality must not suffer -- the optimum is still found via the other chain.
        assert!(cut_short.best_fitness < 1e-6, "global_stall must not sacrifice the good chain's result: f={:.3e}", cut_short.best_fitness);
    }

    /// `MigrationConfig::due`'s decision logic, tested directly and
    /// deterministically (no thread timing involved) — same pattern as
    /// `global_stall_triggered_pure_logic`.
    #[test]
    fn migration_due_pure_logic() {
        let m = MigrationConfig { interval: 10, margin_frac: 0.5 };
        // Only at interval multiples (hop_idx is 0-based; hop 10 completes at hop_idx 9).
        assert!(!m.due(3, 1000.0, 10.0), "must not fire off the interval");
        assert!(m.due(9, 1000.0, 10.0), "must fire at the interval when beyond margin");
        assert!(m.due(19, 1000.0, 10.0), "must fire at every interval multiple");
        // Within the margin (threshold = 10 * 1.5 = 15): left alone.
        assert!(!m.due(9, 12.0, 10.0), "must not migrate into a competitive chain");
        // f64::MAX sentinel (no chain has reported yet): never fires.
        assert!(!m.due(9, 5.0, f64::MAX), "must not fire against the f64::MAX sentinel");
        // Degenerate interval 0 must never fire (and never divide by zero).
        let m0 = MigrationConfig { interval: 0, margin_frac: 0.5 };
        assert!(!m0.due(9, 1000.0, 10.0), "interval 0 must disable migration");
    }

    /// Migration must actually rescue a losing chain in a real multi-chain
    /// run: one chain seeded at the optimum, one on a gradient-free plateau
    /// (same construction as the `global_stall` live test — its own local
    /// descent can never improve it). With migration configured, the plateau
    /// chain must END UP with a near-zero incumbent (visible directly in the
    /// new `chain_bests` observable) because it received the good chain's
    /// solution; without migration it must finish exactly where it started.
    #[test]
    fn migration_rescues_a_losing_chain() {
        let f = |x: &[f64]| -> Option<f64> {
            if x[0].abs() > 50.0 || x[1].abs() > 50.0 {
                Some(9999.0)
            } else {
                Some(x[0] * x[0] + x[1] * x[1])
            }
        };
        let bounds = vec![(-1000.0_f64, 1000.0), (-1000.0, 1000.0)];
        let seeds = vec![vec![0.0, 0.0], vec![500.0, 500.0]];

        let no_migration = MbhSolver {
            hops: 200,
            perturb_fraction: 0.5,
            kick_scale: 0.05,
            local: default_local().into(),
            seed: 3,
            stop_after: None,
            global_stall: None,
            extra_random_chains: 0,
            migration: None,
        };
        let with_migration = MbhSolver {
            migration: Some(MigrationConfig { interval: 5, margin_frac: 0.5 }),
            ..MbhSolver {
                hops: 200,
                perturb_fraction: 0.5,
                kick_scale: 0.05,
                local: default_local().into(),
                seed: 3,
                stop_after: None,
                global_stall: None,
                extra_random_chains: 0,
                migration: None,
            }
        };

        let isolated = no_migration.run(&bounds, &seeds, f);
        let migrated = with_migration.run(&bounds, &seeds, f);

        assert_eq!(isolated.chain_bests.len(), 2);
        assert_eq!(migrated.chain_bests.len(), 2);
        // Without migration the plateau chain never improves off 9999.
        assert_eq!(
            isolated.chain_bests[1], 9999.0,
            "plateau chain should be unable to improve on its own"
        );
        // With migration it must have received the good chain's solution.
        // The plateau chain runs 200 hops with a checkpoint every 5, so the
        // optimum chain's initial descent has ample real time to post its
        // result — same implicit-timing robustness argument as the
        // `global_stall` live test above.
        assert!(
            migrated.chain_bests[1] < 1.0,
            "plateau chain should have been rescued by migration: f={:.3e}",
            migrated.chain_bests[1]
        );
        // And the global result must never be hurt by migration.
        assert!(migrated.best_fitness < 1e-6, "migration must not hurt the global result: f={:.3e}", migrated.best_fitness);
    }
}
