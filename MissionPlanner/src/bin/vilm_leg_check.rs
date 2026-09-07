//! VILM leg-model diagnostic for a saved MGA-1DSM chromosome
//! (Phase 9x-v Stage 5).
//!
//! For every leg whose start and end body match (a resonant/repeat-flyby
//! leg — same detection `resonance_bias_windows` already uses), re-models
//! sub-arc 2 (DSM point → next body) as a tangent VILT
//! (`trajectory_solver::vilm`) instead of the existing free-position-DSM
//! Lambert arc, sweeping a small fixed grid of domain/revolution-count
//! choices (NOT wired into any live search — greedy in-evaluator branch
//! selection is exactly the bug already found and fixed once for Lambert's
//! N-rev choice; see `evaluate_vilm_leg`'s own doc warning). Reports:
//!
//! 1. The resonant leg's own cost under VILM (departure DSM + internal
//!    leveraging burn) vs. the existing Lambert-based DSM.
//! 2. The DOWNSTREAM effect: re-applies the SAME flyby (rp, β) genes from
//!    the converged chromosome to VILM's new arrival v∞, then re-evaluates
//!    the NEXT leg (still plain Lambert, unchanged genes) with that new
//!    departure state — this is the actual test of the Stage 5 hypothesis
//!    (does a better-shaped resonant leg hand off a v∞ the next leg can
//!    reach more cheaply), not just whether VILM is cheap on its own leg.
//!
//! Run: `cargo run -p mission_planner --bin vilm_leg_check --release -- <config.toml>`

use ephemeris::Almanac;
use nalgebra::Vector3;
use mission_planner::config::MissionConfig;
use mission_planner::mga::{
    beta, eta, evaluate_chromosome_detailed, load_best_chromosome_csv, n_legs, rp_norm, tof_days,
};
use trajectory_solver::{
    evaluate_mga_leg_n, evaluate_vilm_leg, flyby_turn, keplerian::MU_SUN_M3S2, VilmDomain,
    VilmSolution,
};

fn main() {
    let config_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: vilm_leg_check <config.toml>");
        std::process::exit(1);
    });

    let cfg = MissionConfig::from_file(&config_path)
        .unwrap_or_else(|e| panic!("failed to load {config_path}: {e}"));

    let almanac = Almanac::new("kernels/de440s.bsp")
        .or_else(|_| Almanac::new("../kernels/de440s.bsp"))
        .expect("de440s.bsp not found — download from https://public-data.nyxspace.com/anise/de440s.bsp");

    let opt = cfg.optimization.as_ref()
        .expect("vilm_leg_check requires an [optimization] section");
    let mga = opt.mga.as_ref()
        .expect("vilm_leg_check requires [optimization.mga]");

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

    let mut body_names: Vec<&str> = Vec::with_capacity(n + 1);
    body_names.push(opt.departure_body.as_str());
    for fb in &flyby_bodies { body_names.push(fb.as_str()); }
    body_names.push(opt.target_body.as_str());

    println!("=== VILM leg diagnostic: {csv_path} ===");
    println!("Baseline (Lambert) total DSM sum: {:.1} m/s\n",
        ev.legs.iter().map(|l| l.leg.dv_dsm_ms).sum::<f64>());

    const DOMAINS: [VilmDomain; 2] = [VilmDomain::Interior, VilmDomain::Exterior];
    const K_RANGE: [u32; 2] = [0, 1];
    const SOLUTIONS: [VilmSolution; 2] = [VilmSolution::Lower, VilmSolution::Upper];

    let mut found_any = false;
    for k in 0..n {
        if !body_names[k].eq_ignore_ascii_case(body_names[k + 1]) {
            continue;
        }
        found_any = true;
        let leg = &ev.legs[k];
        let tof_s = tof_days(&params, k) * 86_400.0;
        let eta_k = eta(&params, k);
        let (r_next, v_next) = ev.body_rvs[k + 1];
        let baseline_dv = leg.leg.dv_dsm_ms;

        println!("Leg {k} ({}->{}): baseline Lambert DSM = {baseline_dv:.1} m/s",
            body_names[k], body_names[k + 1]);

        let mut best: Option<(f64, VilmDomain, u32, u32, VilmSolution, Vector3<f64>)> = None;
        for &domain in &DOMAINS {
            for &k_low in &K_RANGE {
                for &k_high in &K_RANGE {
                    for &solution in &SOLUTIONS {
                        let Some(result) = evaluate_vilm_leg(
                            leg.r_sc_start, leg.v_sc_start, eta_k, tof_s,
                            r_next, v_next, MU_SUN_M3S2, domain, true, k_low, k_high, solution,
                        ) else { continue };
                        let total = result.leg.dv_dsm_ms + result.dv_leverage_ms;
                        println!(
                            "    domain={domain:?} k_low={k_low} k_high={k_high} solution={solution:?}: \
                             dep_dsm={:.1} leverage={:.1} total={total:.1} m/s (r_c={:.3e} m)",
                            result.leg.dv_dsm_ms, result.dv_leverage_ms, result.r_c_m,
                        );
                        if best.as_ref().map(|b| total < b.0).unwrap_or(true) {
                            best = Some((total, domain, k_low, k_high, solution, result.leg.v_inf_arr_mps));
                        }
                    }
                }
            }
        }

        let Some((best_total, best_domain, best_klow, best_khigh, best_solution, vinf_arr_vilm)) = best else {
            println!("    no feasible VILT found for this leg in the swept grid — skipped.\n");
            continue;
        };

        let delta = best_total - baseline_dv;
        println!(
            "  BEST: domain={best_domain:?} k_low={best_klow} k_high={best_khigh} solution={best_solution:?} \
             total={best_total:.1} m/s (Lambert baseline {baseline_dv:.1} m/s, Δ={delta:+.1} m/s)"
        );

        // Downstream effect: only meaningful if there IS a next leg.
        if k + 1 >= n {
            println!("  (final leg — no downstream leg to re-evaluate)\n");
            continue;
        }
        let next_leg = &ev.legs[k + 1];
        let baseline_next_dv = next_leg.leg.dv_dsm_ms;

        let rp_raw = rp_norm(&params, n, k) * ev.body_radii[k + 1];
        let rp_used = rp_raw.max(mga.flyby_min_periapsis_m);
        let beta_k = beta(&params, n, k);
        let v_inf_out_vilm = flyby_turn(vinf_arr_vilm, rp_used, beta_k, ev.body_mus[k + 1]);

        let r_sc_start_next = r_next;
        let v_sc_start_next = v_next + v_inf_out_vilm;
        let tof_next_s = tof_days(&params, k + 1) * 86_400.0;
        let eta_next = eta(&params, k + 1);
        let (r_next2, v_next2) = ev.body_rvs[k + 2];
        // n_rev for the downstream leg: reuse whatever the converged
        // chromosome used (greedy-search convenience for this one-off
        // diagnostic only — matches this leg's own already-converged gene,
        // read back the same way run_mga's evaluators do).
        let n_rev_next = mission_planner::mga::n_rev_gene(&params, n, k + 1);

        match evaluate_mga_leg_n(
            r_sc_start_next, v_sc_start_next, eta_next, tof_next_s,
            r_next2, v_next2, MU_SUN_M3S2, n_rev_next,
        ) {
            Some(recomputed_next) => {
                let delta_next = recomputed_next.dv_dsm_ms - baseline_next_dv;
                println!(
                    "  Downstream leg {}: baseline DSM {baseline_next_dv:.1} m/s -> \
                     recomputed {:.1} m/s (Δ={delta_next:+.1} m/s) with VILM's new hand-off v∞",
                    k + 1, recomputed_next.dv_dsm_ms,
                );
                let combined_baseline = baseline_dv + baseline_next_dv;
                let combined_new = best_total + recomputed_next.dv_dsm_ms;
                println!(
                    "  COMBINED (leg {k} + leg {}): baseline {combined_baseline:.1} m/s -> \
                     VILM {combined_new:.1} m/s (Δ={:+.1} m/s)\n",
                    k + 1, combined_new - combined_baseline,
                );
            }
            None => println!("  Downstream leg {}: recomputation infeasible with VILM's hand-off v∞\n", k + 1),
        }
    }

    if !found_any {
        println!("No same-body-return legs found in this chromosome — nothing to check.");
    }
}
