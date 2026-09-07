//! Generic incremental pruning search — Ceriotti (2010) Ch. 3.
//!
//! Decomposes a search over a high-dimensional box `D` into a cascade of
//! smaller sub-problems ("levels"), each introducing a subset of the
//! variables. A partial objective `f_i`, evaluable from only the variables
//! introduced at levels `1..=i`, is used to prune the level-`i` search space
//! down to a "feasible set" — candidates whose partial cost is within a
//! generous multiple of the best partial cost found — before moving on to
//! level `i+1`. Because the partial objective for MGA-1DSM is a running sum
//! of non-negative per-leg ΔVs (Bellman's principle of optimality: a prefix
//! that is already expensive cannot lead to a cheap complete trajectory),
//! discarding high-cost prefixes is safe and does not risk losing the true
//! optimum, as long as the threshold is generous enough.
//!
//! This module implements the search as a **sample tree**, not Ceriotti's
//! illustrative grid-search test case (§3.3) — a full grid is combinatorial
//! in dimension, whereas per-parent random continuation sampling keeps total
//! evaluations polynomial in the number of levels (his own recommendation
//! for a first, practical implementation is plain multi-start, not a fancy
//! search — §3.2.1). Back-pruning (§3.2.2 — retroactively discarding a
//! level-`i-1` survivor once it's known to have no feasible level-`i`
//! continuation) falls out for free from this tree construction: a parent
//! with zero surviving children simply does not appear in the next
//! generation, no separate bookkeeping needed.
//!
//! # References
//! - Ceriotti, M. (2010), *Global Optimisation of Multiple Gravity Assist
//!   Trajectories*, PhD Thesis, University of Strathclyde, Ch. 3
//!   (§3.2 Incremental Approach, §3.2.2 Back Pruning).
//! - Bellman, R. (1957), *Dynamic Programming*, Princeton University Press
//!   (principle of optimality — the theoretical basis for why pruning a
//!   partial-cost prefix cannot discard the global optimum).

use crate::monte_carlo::SplitMix64;

/// One candidate carried between levels: the concatenated variable vector
/// (levels `0..=level_idx`, in the order the levels were defined) and its
/// cumulative partial-objective value.
#[derive(Clone, Debug)]
pub struct PruningCandidate {
    /// Concatenated variables from every level processed so far, in level
    /// order (each level's own local variable order, caller-defined).
    pub vars: Vec<f64>,
    /// Cumulative partial objective value at the level this candidate
    /// belongs to (the value returned by the evaluator for this level).
    pub partial_cost: f64,
}

/// Configuration for [`run_incremental_pruning`].
#[derive(Clone, Debug)]
pub struct PruningConfig {
    /// Number of random samples drawn at level 0 (no parent to extend).
    pub samples_level0: usize,
    /// Number of random continuations sampled per surviving parent at every
    /// level after 0.
    pub children_per_survivor: usize,
    /// Hard cap on how many candidates are carried forward into the next
    /// level — keeps the branching factor (and therefore total evaluation
    /// count) polynomial in the number of levels even if the threshold
    /// admits more. When more than this many pass the threshold, the
    /// cheapest `max_survivors_per_level` are kept.
    pub max_survivors_per_level: usize,
    /// Pruning threshold, expressed as a multiple of the best partial cost
    /// found at that level: `threshold = best_cost * threshold_factor`.
    /// "Generous" (Ceriotti's own term) is intentional — the goal is to
    /// discard clearly-bad prefixes, not to greedily commit to the
    /// single best one, since the best full trajectory may not need the
    /// best possible early leg (§3.2.1).
    pub threshold_factor: f64,
    /// PRNG seed — fixed for reproducible runs. Each level draws from a
    /// distinctly-seeded stream (`seed + 1_000_003 * level_idx`) so results
    /// don't silently correlate across levels.
    pub seed: u64,
}

/// Run incremental pruning over `level_bounds.len()` levels.
///
/// `level_bounds[i]` gives the `(lo, hi)` bounds for each variable
/// *introduced at* level `i` (not the full concatenated vector — just the
/// new ones). `evaluate(level_idx, prefix)` receives the FULL concatenated
/// variable vector through `level_idx` (inclusive) and must return the
/// cumulative partial objective (`None` if that particular combination is
/// infeasible — e.g. no Lambert solution for that leg — which simply drops
/// the sample, exactly like a threshold failure).
///
/// Returns the surviving candidates from the LAST level processed, sorted
/// by ascending partial cost. Total evaluation count is bounded by
/// `samples_level0 + (num_levels - 1) * max_survivors_per_level *
/// children_per_survivor` — polynomial in the number of levels, per
/// Ceriotti §3.2.1's requirement that avoiding the grid's exponential blowup
/// is what makes the approach practical.
pub fn run_incremental_pruning<E>(
    level_bounds: &[Vec<(f64, f64)>],
    evaluate: E,
    cfg: &PruningConfig,
) -> Vec<PruningCandidate>
where
    E: FnMut(usize, &[f64]) -> Option<f64>,
{
    run_incremental_pruning_with_sampler(
        level_bounds, evaluate, cfg,
        |_level_idx, bounds, next| sample_box_uniform(bounds, next),
    )
}

/// Same as [`run_incremental_pruning`], but draws each level's new variables
/// via a caller-supplied `sampler` instead of always sampling uniformly.
///
/// `sampler(level_idx, bounds, next_uniform)` must return one point inside
/// `bounds` (same length), using `next_uniform` (repeated `[0,1)` draws) as
/// its only source of randomness — this keeps the engine's seeded RNG as
/// the single source of entropy (reproducible runs), even when a caller
/// wants to bias sampling toward a sub-region it has domain knowledge
/// about. This crate has no notion of what that bias should be (e.g.
/// orbital-resonance windows for a same-body return leg, Phase 9x-iv) —
/// that logic belongs entirely in the caller's `sampler` closure. Use
/// [`sample_box_uniform`] inside `sampler` for any variable that shouldn't
/// be biased.
pub fn run_incremental_pruning_with_sampler<E, S>(
    level_bounds: &[Vec<(f64, f64)>],
    evaluate: E,
    cfg: &PruningConfig,
    sampler: S,
) -> Vec<PruningCandidate>
where
    E: FnMut(usize, &[f64]) -> Option<f64>,
    S: FnMut(usize, &[(f64, f64)], &mut dyn FnMut() -> f64) -> Vec<f64>,
{
    run_incremental_pruning_all_levels(level_bounds, evaluate, cfg, sampler)
        .pop()
        .unwrap_or_default()
}

/// Same as [`run_incremental_pruning_with_sampler`], but returns the pruned
/// survivor set at EVERY level, not just the last: element `i` of the result
/// is the candidate set that survived pruning after level `i` was processed
/// (each candidate's `vars` covering levels `0..=i`). The last element is
/// exactly what [`run_incremental_pruning_with_sampler`] returns.
///
/// Needed by bidirectional ("meet in the middle") search schemes that match
/// forward partial solutions against independently-found backward suffixes
/// at every intermediate interface, not only at the final level — the
/// caller decides what an "interface" means; this engine just exposes the
/// per-level snapshots it already computes internally.
pub fn run_incremental_pruning_all_levels<E, S>(
    level_bounds: &[Vec<(f64, f64)>],
    mut evaluate: E,
    cfg: &PruningConfig,
    mut sampler: S,
) -> Vec<Vec<PruningCandidate>>
where
    E: FnMut(usize, &[f64]) -> Option<f64>,
    S: FnMut(usize, &[(f64, f64)], &mut dyn FnMut() -> f64) -> Vec<f64>,
{
    if level_bounds.is_empty() {
        return Vec::new();
    }
    let mut snapshots: Vec<Vec<PruningCandidate>> = Vec::with_capacity(level_bounds.len());

    // ── Level 0: no parent, sample the full level-0 box directly ─────────────
    let mut rng = SplitMix64::new(cfg.seed);
    let mut candidates: Vec<PruningCandidate> = Vec::with_capacity(cfg.samples_level0);
    for _ in 0..cfg.samples_level0 {
        let vars = {
            let mut next = || rng.next_f64();
            sampler(0, &level_bounds[0], &mut next)
        };
        if let Some(cost) = evaluate(0, &vars) {
            candidates.push(PruningCandidate { vars, partial_cost: cost });
        }
    }
    candidates = prune(candidates, cfg);
    snapshots.push(candidates.clone());

    // ── Levels 1..N: extend each surviving parent with new random samples ───
    for level_idx in 1..level_bounds.len() {
        if candidates.is_empty() {
            // Every level-(i-1) survivor dead-ended — nothing to extend.
            // Reported by returning the snapshots so far (last one empty);
            // the caller (which has domain knowledge — e.g. widen bounds,
            // raise threshold_factor) decides how to react. Not treated as
            // an error here: the engine is generic and has no notion of what
            // "recoverable" means for the caller's problem.
            return snapshots;
        }
        let mut level_rng = SplitMix64::new(cfg.seed.wrapping_add(1_000_003u64.wrapping_mul(level_idx as u64)));
        let mut children: Vec<PruningCandidate> = Vec::with_capacity(candidates.len() * cfg.children_per_survivor);
        for parent in &candidates {
            for _ in 0..cfg.children_per_survivor {
                let new_vars = {
                    let mut next = || level_rng.next_f64();
                    sampler(level_idx, &level_bounds[level_idx], &mut next)
                };
                let mut vars = parent.vars.clone();
                vars.extend_from_slice(&new_vars);
                if let Some(cost) = evaluate(level_idx, &vars) {
                    children.push(PruningCandidate { vars, partial_cost: cost });
                }
            }
        }
        // Back pruning (§3.2.2) is implicit here: a `parent` whose every
        // sampled continuation returned `None` or failed the threshold
        // contributes nothing to `children`, so it silently vanishes from
        // the next generation — no separate retroactive step needed.
        candidates = prune(children, cfg);
        snapshots.push(candidates.clone());
    }

    snapshots
}

/// Keep only candidates within `threshold_factor × best_cost`, capped at
/// `max_survivors_per_level` (cheapest kept when over the cap).
fn prune(mut candidates: Vec<PruningCandidate>, cfg: &PruningConfig) -> Vec<PruningCandidate> {
    if candidates.is_empty() {
        return candidates;
    }
    candidates.sort_by(|a, b| a.partial_cost.total_cmp(&b.partial_cost));
    let best = candidates[0].partial_cost;
    let threshold = best * cfg.threshold_factor;
    candidates.retain(|c| c.partial_cost <= threshold);
    candidates.truncate(cfg.max_survivors_per_level);
    candidates
}

/// Draw one uniform-random point inside a hyper-rectangle, using `next` as
/// a `[0,1)` random source. Exposed so a custom `sampler` (see
/// [`run_incremental_pruning_with_sampler`]) can fall back to plain uniform
/// sampling for any variable it doesn't want to bias.
pub fn sample_box_uniform(bounds: &[(f64, f64)], next: &mut dyn FnMut() -> f64) -> Vec<f64> {
    bounds.iter().map(|&(lo, hi)| lo + next() * (hi - lo)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check on a toy 2-level separable quadratic:
    /// `f(x0, x1) = x0^2 + x1^2`, minimum at the origin. Confirms pruning
    /// converges toward the known minimum and does not lose it.
    #[test]
    fn separable_quadratic_finds_near_origin_minimum() {
        let level_bounds = vec![
            vec![(-10.0, 10.0)],
            vec![(-10.0, 10.0)],
        ];
        let cfg = PruningConfig {
            samples_level0: 200,
            children_per_survivor: 20,
            max_survivors_per_level: 20,
            threshold_factor: 3.0,
            seed: 42,
        };
        let survivors = run_incremental_pruning(
            &level_bounds,
            |level_idx, prefix| {
                let partial: f64 = prefix.iter().take(level_idx + 1).map(|v| v * v).sum();
                Some(partial)
            },
            &cfg,
        );
        assert!(!survivors.is_empty());
        let best = survivors.iter().map(|c| c.partial_cost).fold(f64::MAX, f64::min);
        assert!(best < 1.0, "expected near-origin minimum, got cost {best}");
    }

    /// An evaluator that always returns `None` at level 0 must yield an
    /// empty result without panicking (dead-end / total infeasibility case).
    #[test]
    fn all_infeasible_level0_returns_empty_without_panicking() {
        let level_bounds = vec![vec![(-1.0, 1.0)], vec![(-1.0, 1.0)]];
        let cfg = PruningConfig {
            samples_level0: 10,
            children_per_survivor: 5,
            max_survivors_per_level: 5,
            threshold_factor: 2.0,
            seed: 7,
        };
        let survivors = run_incremental_pruning(&level_bounds, |_, _| None, &cfg);
        assert!(survivors.is_empty());
    }

    /// `max_survivors_per_level` must actually bound the branching factor —
    /// otherwise total evaluations grow exponentially with level count,
    /// defeating the whole point (Ceriotti §3.2.1's polynomial-complexity
    /// requirement).
    #[test]
    fn survivor_cap_bounds_branching_factor() {
        let level_bounds = vec![
            vec![(0.0, 1.0)],
            vec![(0.0, 1.0)],
            vec![(0.0, 1.0)],
        ];
        let cfg = PruningConfig {
            samples_level0: 500,
            children_per_survivor: 50,
            max_survivors_per_level: 10,
            threshold_factor: 100.0, // admit almost everything
            seed: 1,
        };
        let survivors = run_incremental_pruning(&level_bounds, |_, _| Some(0.0), &cfg);
        assert!(survivors.len() <= cfg.max_survivors_per_level);
    }

    /// The all-levels variant must return one survivor snapshot per level,
    /// with candidate `vars` lengths matching the cumulative variable count
    /// through that level, and its last snapshot must equal what the
    /// single-result function returns (same seed, same sampler → identical
    /// RNG stream, bit-identical results).
    #[test]
    fn all_levels_snapshots_match_single_result_variant() {
        let level_bounds = vec![
            vec![(-5.0, 5.0), (-5.0, 5.0)],
            vec![(-5.0, 5.0)],
            vec![(-5.0, 5.0)],
        ];
        let cfg = PruningConfig {
            samples_level0: 100,
            children_per_survivor: 10,
            max_survivors_per_level: 10,
            threshold_factor: 3.0,
            seed: 42,
        };
        let eval = |level_idx: usize, prefix: &[f64]| {
            // Cumulative var count: 2 at level 0, then +1 per level.
            let n_vars = 2 + level_idx;
            Some(prefix.iter().take(n_vars).map(|v| v * v).sum::<f64>() + 1.0)
        };
        let snapshots = run_incremental_pruning_all_levels(
            &level_bounds, eval, &cfg,
            |_l, b, next| sample_box_uniform(b, next),
        );
        assert_eq!(snapshots.len(), 3);
        for (i, snap) in snapshots.iter().enumerate() {
            assert!(!snap.is_empty(), "level {i} snapshot empty");
            for c in snap {
                assert_eq!(c.vars.len(), 2 + i, "level {i} var length");
            }
        }
        let single = run_incremental_pruning(&level_bounds, eval, &cfg);
        assert_eq!(single.len(), snapshots[2].len());
        for (a, b) in single.iter().zip(&snapshots[2]) {
            assert_eq!(a.vars, b.vars);
            assert_eq!(a.partial_cost, b.partial_cost);
        }
    }
}
