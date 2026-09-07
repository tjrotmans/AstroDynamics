//! Primer vector optimality diagnostic for a saved MGA-1DSM chromosome
//! (Phase 9x Stage 1).
//!
//! Reads `<out_dir>/mga_best_chromosome.csv` (written by the `optimize`/
//! `mga-geometry` commands), re-evaluates every leg, and checks Lawden's
//! primer vector necessary condition (`‖λV(t)‖ ≤ 1` everywhere) along each
//! leg's two Keplerian sub-arcs — see `trajectory_solver::primer_vector`
//! for the full theory and the explicit scope note on the leg-isolated
//! approximation this diagnostic uses (no full cross-leg λR/ν coupling).
//!
//! A violation (`max ‖λV‖ > 1`) on a leg is a necessary-condition proof
//! that a second, smaller impulse there would reduce that leg's own DSM
//! cost — i.e. this leg genuinely needs the model's "one DSM per leg"
//! assumption relaxed, not just a better search. No violation is
//! INCONCLUSIVE under the leg-isolated approximation (see `fit_residual`
//! in the printed output), not a proof of optimality.
//!
//! Run: `cargo run -p mission_planner --bin primer_vector_check --release -- <config.toml>`

use ephemeris::Almanac;
use mission_planner::config::{MissionConfig, MissionObjective};
use mission_planner::mga::{evaluate_chromosome_detailed, load_best_chromosome_csv, n_legs};
use trajectory_solver::{keplerian::MU_SUN_M3S2, solve_leg_primer, LegBoundary, SubArc};

fn main() {
    let config_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: primer_vector_check <config.toml>");
        std::process::exit(1);
    });

    let cfg = MissionConfig::from_file(&config_path)
        .unwrap_or_else(|e| panic!("failed to load {config_path}: {e}"));

    let almanac = Almanac::new("kernels/de440s.bsp")
        .or_else(|_| Almanac::new("../kernels/de440s.bsp"))
        .expect("de440s.bsp not found — download from https://public-data.nyxspace.com/anise/de440s.bsp");

    let opt = cfg.optimization.as_ref()
        .expect("primer_vector_check requires an [optimization] section");

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

    println!("=== Primer vector diagnostic: {csv_path} ===");
    println!("Leg-isolated approximation (see trajectory_solver::primer_vector doc comment) —");
    println!("max||lambdaV|| > 1 is a necessary-condition VIOLATION (2nd impulse would help);");
    println!("<= 1 is INCONCLUSIVE (not a proof of optimality), read alongside fit_residual.\n");

    for k in 0..n {
        let leg = &ev.legs[k];
        let tof_s = mission_planner::mga::tof_days(&params, k) * 86_400.0;
        let eta_k = mission_planner::mga::eta(&params, k);
        let dt1 = eta_k * tof_s;
        let dt2 = tof_s - dt1;

        let dv_vec = leg.leg.v_dsm_after_mps - leg.leg.v_dsm_before_mps;
        let dv_mag = dv_vec.norm();
        if dv_mag < 1.0 {
            println!("Leg {k}: DSM ~0 ({dv_mag:.3} m/s, ballistic) — skipping, no impulse to anchor against.");
            continue;
        }
        let dv_dsm_unit = dv_vec / dv_mag;

        let sub_arc_a = SubArc { r0: leg.r_sc_start, v0: leg.v_sc_start, dt_s: dt1 };
        let sub_arc_b = SubArc { r0: leg.leg.r_dsm_m, v0: leg.leg.v_dsm_after_mps, dt_s: dt2 };

        // Start boundary: true departure (k=0, Olympio Eq. 19, FULLY known
        // direction) or a flyby's outgoing v-infinity direction (Eq. 21,
        // unknown scalar magnitude) otherwise.
        let start_boundary = if k == 0 {
            let dv0_dir = (leg.v_sc_start - ev.body_rvs[0].1).normalize();
            LegBoundary::FixedImpulse(dv0_dir)
        } else {
            let vinf_out_dir = (leg.v_sc_start - ev.body_rvs[k].1).normalize();
            LegBoundary::FlybyDirection(vinf_out_dir)
        };
        // End boundary: true arrival impulse (k=n-1, only when a real
        // capture/insertion burn exists — Flyby objective has NO arrival
        // impulse at all, so it's treated the same as an interior flyby's
        // incoming v-infinity direction, Eq. 20 — outside Olympio's own
        // stated framework, which assumes nonzero terminal impulses, but
        // the closest well-defined analogue for a diagnostic like this).
        let is_true_arrival = k == n - 1 && !matches!(cfg.mission.objective, MissionObjective::Flyby);
        let vinf_in_dir = leg.leg.v_inf_arr_mps.normalize();
        let end_boundary = if is_true_arrival {
            LegBoundary::FixedImpulse(vinf_in_dir)
        } else {
            LegBoundary::FlybyDirection(vinf_in_dir)
        };

        match solve_leg_primer(sub_arc_a, sub_arc_b, MU_SUN_M3S2, start_boundary, end_boundary, dv_dsm_unit) {
            Some(result) => {
                let flag = if result.max_primer_norm > 1.0 { "VIOLATION" } else { "ok" };
                println!(
                    "Leg {k}: DSM={dv_mag:8.1} m/s  max||lambdaV||={:.4} [{flag}]  fit_residual={:.4}",
                    result.max_primer_norm, result.fit_residual,
                );
            }
            None => println!("Leg {k}: primer vector solve failed (degenerate sub-arc?) — skipped."),
        }
    }
}
