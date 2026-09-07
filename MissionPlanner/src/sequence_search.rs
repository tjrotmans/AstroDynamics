//! Tisserand beam search for automatic MGA flyby-sequence discovery (Phase 9j-B).
//!
//! Given only a departure body, a target body, and a pool of candidate intermediate
//! bodies, finds the best flyby sequences using the Tisserand graph as a fast
//! pruning criterion — no propagation or Lambert solve needed per candidate.
//!
//! The outer discrete beam search retains the top-`beam_width` partial sequences
//! at each depth level; the inner Stage-A MGA optimizer (`mga::run_mga`) runs
//! on the top-`max_sequences_to_optimize` complete sequences found.
//!
//! # Reference
//! Strange & Longuski (2002), "Graphical Method for Gravity-Assist Trajectory
//!   Design", JGCD 25(6):1154–1159.
//! Campagnola & Russell (2010), J. Guid. Control Dyn. 33(2):476–487.

use crate::config::SequenceSearchConfig;
use trajectory_solver::{find_tisserand_link, tisserand_feasibility_score};

const MU_SUN: f64 = 1.327_124_400_18e20; // m³/s² (IAU 2012)

/// One candidate flyby sequence returned by the beam search.
#[derive(Debug, Clone)]
pub struct RankedSequence {
    /// Intermediate bodies only (excludes departure and target body).
    pub flyby_bodies: Vec<String>,
    /// Estimated v∞ at the target body from the Tisserand graph walk [m/s].
    pub estimated_vinf_arr_ms: f64,
    /// Cumulative Tisserand feasibility score (lower = better; used for ranking).
    pub tisserand_score: f64,
}

/// A node in the beam search tree.
#[derive(Debug, Clone)]
struct BeamState {
    sequence: Vec<String>,   // all bodies visited so far (includes departure at [0])
    current_vinf_ms: f64,    // estimated v∞ arriving at `current_body`
    score: f64,              // cumulative Tisserand feasibility score
}

/// Run the Tisserand beam search.
///
/// Returns sequences sorted by `tisserand_score` ascending (lowest = best).
/// Repeated visits are allowed — each candidate body can appear more than
/// once in a sequence (e.g. Venus→Earth→Earth→Jupiter).
pub fn run_sequence_search(
    cfg: &SequenceSearchConfig,
    departure_body: &str,
    target_body: &str,
) -> Vec<RankedSequence> {
    // Resolve heliocentric SMA and orbital speed for each body in the candidate pool
    // plus the departure and target bodies.
    let mut all_bodies = cfg.candidate_bodies.clone();
    if !all_bodies.contains(&departure_body.to_string()) {
        all_bodies.insert(0, departure_body.to_string());
    }
    if !all_bodies.contains(&target_body.to_string()) {
        all_bodies.push(target_body.to_string());
    }

    // Precompute (sma_m, v_circular_ms) for every body we'll reference.
    let mut body_sma: std::collections::HashMap<String, f64> = Default::default();
    for name in &all_bodies {
        if let Some(tb) = body_models::TargetBody::by_name(name) {
            if let Some(sma) = tb.sma_m {
                let v_circ = (MU_SUN / sma).sqrt();
                body_sma.insert(name.clone(), sma);
                let _ = v_circ; // stored implicitly via sma
            }
        }
    }

    let v_circular = |name: &str| -> Option<(f64, f64)> {
        body_sma.get(name).map(|&sma| (sma, (MU_SUN / sma).sqrt()))
    };

    // Require departure and target to be resolvable.
    if v_circular(departure_body).is_none() || v_circular(target_body).is_none() {
        eprintln!("[sequence_search] departure or target body has no known SMA — cannot search");
        return vec![];
    }

    let (_, _v_dep) = v_circular(departure_body).unwrap();
    let candidates: Vec<String> = cfg.candidate_bodies.clone();

    // Seed the beam with the initial state at the departure body.
    let mut beam: Vec<BeamState> = vec![BeamState {
        sequence: vec![departure_body.to_string()],
        current_vinf_ms: cfg.vinf_departure_estimate_ms,
        score: 0.0,
    }];

    let mut complete: Vec<RankedSequence> = vec![];

    for _depth in 0..cfg.max_legs {
        let mut next_beam: Vec<BeamState> = vec![];

        for state in &beam {
            let current_body = state.sequence.last().unwrap();
            let (a_cur, v_cur) = match v_circular(current_body) {
                Some(x) => x,
                None => continue,
            };

            // Try connecting to the target body directly.
            {
                let (a_tgt, v_tgt) = v_circular(target_body).unwrap();
                let link = find_tisserand_link(state.current_vinf_ms, v_cur, a_cur, a_tgt, MU_SUN);
                if let Some(vinf_arr) = link {
                    let seq_score = state.score + vinf_arr;
                    let mut full_seq = state.sequence.clone();
                    full_seq.push(target_body.to_string());
                    let flyby_bodies: Vec<String> = full_seq[1..full_seq.len() - 1].to_vec();
                    complete.push(RankedSequence {
                        flyby_bodies,
                        estimated_vinf_arr_ms: vinf_arr,
                        tisserand_score: seq_score,
                    });
                    let _ = v_tgt;
                }
            }

            // Try connecting to each candidate intermediate body.
            if _depth + 1 < cfg.max_legs {
                for next_body in &candidates {
                    let (a_next, v_next) = match v_circular(next_body) {
                        Some(x) => x,
                        None => continue,
                    };
                    let score_increment = tisserand_feasibility_score(
                        state.current_vinf_ms, v_cur, a_cur, a_next, MU_SUN,
                    );
                    // Estimate v∞ at next body if the link is feasible.
                    let vinf_next = match find_tisserand_link(state.current_vinf_ms, v_cur, a_cur, a_next, MU_SUN) {
                        Some(v) => v,
                        None => {
                            // Near-feasible: estimate from score (already a v-units penalty).
                            // Include it so the beam doesn't prune near-feasible paths too early.
                            score_increment
                        }
                    };
                    // Prune: skip if cumulative score exceeds a large multiple of the
                    // departure v∞ estimate (avoids exploding the search space with
                    // clearly infeasible partial paths).
                    let cum_score = state.score + score_increment;
                    if cum_score > cfg.vinf_departure_estimate_ms * 20.0 {
                        continue;
                    }
                    let mut new_seq = state.sequence.clone();
                    new_seq.push(next_body.clone());
                    next_beam.push(BeamState {
                        sequence: new_seq,
                        current_vinf_ms: vinf_next,
                        score: cum_score,
                    });
                    let _ = v_next;
                }
            }
        }

        // Keep only the top-`beam_width` partial sequences by score.
        next_beam.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        next_beam.truncate(cfg.beam_width);
        beam = next_beam;
        if beam.is_empty() { break; }
    }

    // Sort complete sequences by Tisserand score.
    complete.sort_by(|a, b| a.tisserand_score.partial_cmp(&b.tisserand_score).unwrap_or(std::cmp::Ordering::Equal));
    // Deduplicate (same flyby_bodies list can appear from different paths).
    complete.dedup_by(|a, b| a.flyby_bodies == b.flyby_bodies);

    complete
}

/// Direct (zero-flyby) baseline candidate: the departure→target
/// transfer with no gravity assists. The beam search only emits this on its
/// own when the tangential Tisserand link test passes at the departure v∞
/// estimate — but with the hyperbolic Lambert branch and DSMs the direct
/// transfer is usually still flyable (just more expensive), and optimizing
/// it alongside the beam's candidates makes the ranking show the ΔV the
/// gravity assists are actually saving, not just how sequences compare to
/// each other.
pub fn direct_baseline(
    cfg: &SequenceSearchConfig,
    departure_body: &str,
    target_body: &str,
) -> RankedSequence {
    let sma = |name: &str| body_models::TargetBody::by_name(name).and_then(|tb| tb.sma_m);
    let (score, vinf) = match (sma(departure_body), sma(target_body)) {
        (Some(a_dep), Some(a_tgt)) => {
            let v_dep = (MU_SUN / a_dep).sqrt();
            // Same convention as the beam search: the score IS the estimated
            // arrival v∞ when the link is feasible, and a v-scale penalty
            // when it isn't (see tisserand_feasibility_score).
            let s = tisserand_feasibility_score(
                cfg.vinf_departure_estimate_ms, v_dep, a_dep, a_tgt, MU_SUN,
            );
            (s, s)
        }
        _ => (f64::MAX, f64::NAN),
    };
    RankedSequence {
        flyby_bodies: vec![],
        estimated_vinf_arr_ms: vinf,
        tisserand_score: score,
    }
}

/// Pretty-print a ranked sequence table to stdout.
pub fn print_sequence_table(sequences: &[RankedSequence], departure: &str, target: &str) {
    if sequences.is_empty() {
        println!("  [no sequences found — try relaxing max_legs or beam_width]");
        return;
    }
    println!("\n{:<4}  {:<50}  {:>16}  {:>14}",
        "Rank", "Sequence", "Tisserand score", "Est. arr. v∞");
    println!("{}", "─".repeat(90));
    for (i, seq) in sequences.iter().enumerate() {
        let body_chain: Vec<&str> = {
            let mut chain = vec![departure];
            for fb in &seq.flyby_bodies { chain.push(fb.as_str()); }
            chain.push(target);
            chain
        };
        let chain_str = body_chain.join(" → ");
        println!("{:<4}  {:<50}  {:>16.2}  {:>11.2} m/s",
            i + 1,
            chain_str,
            seq.tisserand_score,
            seq.estimated_vinf_arr_ms);
    }
    println!();
}
