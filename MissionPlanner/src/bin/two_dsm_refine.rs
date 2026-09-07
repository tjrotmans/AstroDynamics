//! Two-DSM local refinement for primer-vector-flagged legs (Phase 9x,
//!).
//!
//! For each leg of an already-converged MGA chromosome whose Lawden/Olympio
//! primer-vector diagnostic (`primer_vector_check`, `trajectory_solver::
//! primer_vector`) reports `max||lambdaV|| > 1`, this seeds a second free
//! impulse at the diagnostic's own peak-violation time and locally refines
//! it (`trajectory_solver::refine_leg_two_dsm`, Nelder-Mead) against that
//! leg's existing one-DSM cost. This is READ-ONLY: it does not touch the
//! chromosome, the CSV, or any optimizer — it only reports whether a real
//! improvement exists and how large it is, for the user to judge before any
//! model change is made permanent (see the design notes
//! scoping section for the full design rationale).
//!
//! Run: `cargo run -p mission_planner --bin two_dsm_refine --release -- <config.toml>`

use ephemeris::Almanac;
use nalgebra::Vector3;
use mission_planner::config::{MissionConfig, MissionObjective};
use mission_planner::mga::{evaluate_chromosome_detailed, load_best_chromosome_csv, n_legs};
use trajectory_solver::{
    evaluate_mga_leg_n, keplerian::MU_SUN_M3S2, refine_leg_two_dsm, solve_leg_primer,
    LegBoundary, NelderMead, SubArc,
};

/// Minimum improvement to report as real rather than noise — matches the
/// magnitude other local-optimizer comparisons here (MBH vs. DE)
/// treated as decisive, not an arbitrary pick.
const MIN_REAL_IMPROVEMENT_MS: f64 = 5.0;

/// Best achievable ONE-DSM cost for this leg at ANY eta (n_rev fixed to 0,
/// matching what the 2-DSM model's own final segment uses) — a GLOBAL
/// multi-start search (coarse grid, then 1-D Nelder-Mead polish from the
/// grid's best point). This is the fair baseline for judging whether a
/// SECOND impulse genuinely helps: comparing against the chromosome's own
/// (possibly search-suboptimal) eta, or against a single-seed local search,
/// conflates "the outer DE/MBH search (or this baseline itself) left eta
/// short of its own one-DSM optimum" with "this leg's one-DSM model is
/// structurally insufficient" — two very different findings a naive
/// before/after comparison cannot tell apart. A single-seed version of this
/// function was tried first and found to get trapped in a worse local
/// optimum than the 2-DSM search's own (primer-vector-informed) seeding
/// reached — exactly the kind of false-positive this diagnostic exists to
/// rule out, so the grid pass is required, not optional polish.
const ETA_GRID_POINTS: usize = 41;

fn best_one_dsm_cost(
    r_sc_start: Vector3<f64>, v_sc_start: Vector3<f64>, tof_s: f64,
    r_next: Vector3<f64>, v_next: Vector3<f64>,
) -> Option<(f64, f64)> {
    let eval_eta = |eta: f64| -> Option<f64> {
        evaluate_mga_leg_n(r_sc_start, v_sc_start, eta, tof_s, r_next, v_next, MU_SUN_M3S2, 0)
            .map(|leg| leg.dv_dsm_ms)
    };
    let (grid_best_eta, _) = (1..ETA_GRID_POINTS).map(|i| i as f64 / ETA_GRID_POINTS as f64)
        .filter_map(|eta| eval_eta(eta).map(|dv| (eta, dv)))
        .fold((0.5_f64, f64::MAX), |acc, (eta, dv)| if dv < acc.1 { (eta, dv) } else { acc });

    let nm = NelderMead { max_iter: 200, ..Default::default() };
    let result = nm.run(&[(0.001, 0.999)], &[grid_best_eta], |x| eval_eta(x[0]));
    if result.best_fitness >= f64::MAX { return None; }
    Some((result.best_params[0], result.best_fitness))
}

fn main() {
    let config_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: two_dsm_refine <config.toml>");
        std::process::exit(1);
    });

    let cfg = MissionConfig::from_file(&config_path)
        .unwrap_or_else(|e| panic!("failed to load {config_path}: {e}"));

    let almanac = Almanac::new("kernels/de440s.bsp")
        .or_else(|_| Almanac::new("../kernels/de440s.bsp"))
        .expect("de440s.bsp not found — download from https://public-data.nyxspace.com/anise/de440s.bsp");

    let opt = cfg.optimization.as_ref()
        .expect("two_dsm_refine requires an [optimization] section");

    let out_dir = cfg.simulation.output_dir.trim_end_matches('/');
    let csv_path = format!("{out_dir}/mga_best_chromosome.csv");
    let (flyby_bodies, params) = load_best_chromosome_csv(&csv_path)
        .unwrap_or_else(|e| panic!("{e}"));
    let n = n_legs(&flyby_bodies);

    let dep_epoch_str = opt.departure_epoch.as_deref()
        .expect("optimization.departure_epoch is required");
    let dep_epoch = mission_planner::design::parse_epoch(dep_epoch_str)
        .unwrap_or_else(|e| panic!("departure_epoch parse: {e}"));
    let dep_jd_base = mission_planner::design::epoch_to_jd(dep_epoch);

    let ev = evaluate_chromosome_detailed(&params, &cfg, &almanac, dep_jd_base, &flyby_bodies)
        .expect("chromosome is infeasible — re-run the optimizer first");

    println!("=== Two-DSM local refinement: {csv_path} ===");
    println!("Read-only: diagnoses primer-vector-flagged legs, reports what a second free");
    println!("impulse would save. Does not modify the chromosome, CSV, or any search.\n");

    let nm = NelderMead { max_iter: 500, ..Default::default() };
    let mut any_flagged = false;

    for k in 0..n {
        let leg = &ev.legs[k];
        let tof_s = mission_planner::mga::tof_days(&params, k) * 86_400.0;
        let eta_k = mission_planner::mga::eta(&params, k);
        let dt1 = eta_k * tof_s;
        let dt2 = tof_s - dt1;

        let dv_vec = leg.leg.v_dsm_after_mps - leg.leg.v_dsm_before_mps;
        let dv_mag = dv_vec.norm();
        if dv_mag < 1.0 {
            println!("Leg {k}: DSM ~0 ({dv_mag:.3} m/s, ballistic) — nothing to refine.");
            continue;
        }
        let dv_dsm_unit = dv_vec / dv_mag;

        let sub_arc_a = SubArc { r0: leg.r_sc_start, v0: leg.v_sc_start, dt_s: dt1 };
        let sub_arc_b = SubArc { r0: leg.leg.r_dsm_m, v0: leg.leg.v_dsm_after_mps, dt_s: dt2 };

        let start_boundary = if k == 0 {
            let dv0_dir = (leg.v_sc_start - ev.body_rvs[0].1).normalize();
            LegBoundary::FixedImpulse(dv0_dir)
        } else {
            let vinf_out_dir = (leg.v_sc_start - ev.body_rvs[k].1).normalize();
            LegBoundary::FlybyDirection(vinf_out_dir)
        };
        let is_true_arrival = k == n - 1 && !matches!(cfg.mission.objective, MissionObjective::Flyby);
        let vinf_in_dir = leg.leg.v_inf_arr_mps.normalize();
        let end_boundary = if is_true_arrival {
            LegBoundary::FixedImpulse(vinf_in_dir)
        } else {
            LegBoundary::FlybyDirection(vinf_in_dir)
        };

        let primer = match solve_leg_primer(sub_arc_a, sub_arc_b, MU_SUN_M3S2, start_boundary, end_boundary, dv_dsm_unit) {
            Some(p) => p,
            None => {
                println!("Leg {k}: primer vector solve failed (degenerate sub-arc?) — skipped.");
                continue;
            }
        };

        if primer.max_primer_norm <= 1.0 {
            println!("Leg {k}: DSM={dv_mag:8.1} m/s  max||lambdaV||={:.4} [ok] — no refinement attempted.", primer.max_primer_norm);
            continue;
        }
        any_flagged = true;

        // Peak-violation time, in leg-elapsed-time fraction (eta), from
        // whichever sub-arc holds the maximum sample — the physically
        // meaningful place to try inserting the extra impulse.
        let (peak_a_idx, peak_a_val) = primer.samples_a.iter().enumerate()
            .filter(|(_, v)| v.is_finite())
            .fold((0usize, f64::MIN), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
        let (peak_b_idx, peak_b_val) = primer.samples_b.iter().enumerate()
            .filter(|(_, v)| v.is_finite())
            .fold((0usize, f64::MIN), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });

        let na = (primer.samples_a.len() - 1).max(1) as f64;
        let nb = (primer.samples_b.len() - 1).max(1) as f64;
        let peak_eta = if peak_a_val >= peak_b_val {
            eta_k * (peak_a_idx as f64 / na)
        } else {
            eta_k + (1.0 - eta_k) * (peak_b_idx as f64 / nb)
        };

        // Seed the two free eta positions from (peak time, original DSM
        // time), sorted — covers both "peak before the original DSM" and
        // "peak after" without assuming which.
        let mut seeds = [peak_eta, eta_k];
        seeds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Nudge apart if the peak landed exactly on the original DSM time.
        if (seeds[1] - seeds[0]).abs() < 1e-4 {
            seeds[0] = (seeds[0] - 0.02).max(0.005);
            seeds[1] = (seeds[1] + 0.02).min(0.995);
        }

        let (r_next, v_next) = ev.body_rvs[k + 1];

        // Fair baseline: best achievable ONE-DSM cost at any eta, not the
        // chromosome's own (possibly search-suboptimal) eta.
        let best_1dsm = best_one_dsm_cost(leg.r_sc_start, leg.v_sc_start, tof_s, r_next, v_next);

        let refined_from_peak = refine_leg_two_dsm(
            leg.r_sc_start, leg.v_sc_start, tof_s, r_next, v_next,
            MU_SUN_M3S2, seeds[0], seeds[1], dv_mag * 3.0, &nm,
        );
        // Second seed, from the 1-DSM baseline's own grid optimum with
        // dv1~0 (this exactly reproduces the best one-DSM leg, since a
        // zero first impulse makes segments A+B one continuous coast) —
        // guarantees the 2-DSM search can never report worse than the fair
        // baseline purely from bad seeding, by construction.
        let refined_from_1dsm = best_1dsm.and_then(|(eta_best, _)| {
            let (e1, e2) = ((eta_best - 0.03).max(0.005), eta_best.min(0.995));
            refine_leg_two_dsm(
                leg.r_sc_start, leg.v_sc_start, tof_s, r_next, v_next,
                MU_SUN_M3S2, e1, e2, dv_mag * 3.0, &nm,
            )
        });
        let refined = match (refined_from_peak, refined_from_1dsm) {
            (Some(a), Some(b)) => Some(if a.total_dv_ms <= b.total_dv_ms { a } else { b }),
            (a, None) => a,
            (None, b) => b,
        };

        match (refined, best_1dsm) {
            (Some(r), Some((eta_best, dv_best))) => {
                let vs_chromosome = dv_mag - r.total_dv_ms;
                let vs_best_1dsm   = dv_best - r.total_dv_ms;
                let verdict = if vs_best_1dsm > MIN_REAL_IMPROVEMENT_MS {
                    "GENUINE 2-DSM GAIN"
                } else if r.leg.dv1_ms < 1.0 {
                    "no gain — just eta re-optimization"
                } else {
                    "no meaningful gain"
                };
                println!(
                    "Leg {k}: DSM={dv_mag:8.1} m/s (chromosome eta={eta_k:.3})  max||lambdaV||={:.4} [VIOLATION]",
                    primer.max_primer_norm,
                );
                println!(
                    "         best 1-DSM (any eta)={dv_best:8.1} m/s @ eta={eta_best:.3}  \
                     [vs. chromosome: {vs_chromosome:+.1} m/s]",
                );
                println!(
                    "         2-DSM total={:8.1} m/s (dv1={:.1} @ eta1={:.3}, dv2={:.1} @ eta2={:.3})  \
                     [vs. best 1-DSM: {vs_best_1dsm:+.1} m/s] -> {verdict}",
                    r.total_dv_ms, r.leg.dv1_ms, r.leg.eta1, r.leg.leg.dv_dsm_ms, r.leg.eta2,
                );
            }
            _ => println!("Leg {k}: DSM={dv_mag:8.1} m/s  max||lambdaV||={:.4} [VIOLATION]  -> refine or baseline search failed.", primer.max_primer_norm),
        }
    }

    if !any_flagged {
        println!("\nNo legs flagged — this chromosome's one-DSM-per-leg model is already primer-vector-optimal (leg-isolated approximation).");
    }
}
