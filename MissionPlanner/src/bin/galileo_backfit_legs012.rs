//! Backward-fit of VEEGA legs 0-2 to leg 3's known-good EGA2 departure v-inf
//! (Phase 9x-iv follow-up -- step 2 of the decoupled approach).
//!
//! `galileo_leg3_search` established that leg 3 (EGA2 1992-12-08 -> Jupiter/
//! JOI 1995-12-07, TOF fixed at the real 1094 days) is FULLY BALLISTIC
//! (DSM = 0.0 m/s) at departure v-inf = 8,918.9 m/s from EGA2 -- so the only
//! remaining question is whether a Venus-Earth-Earth opening exists that
//! arrives at EGA2 (on the real date) with a v-inf the EGA2 flyby can turn
//! into exactly that outgoing vector. This is the reverse of Ceriotti
//! incremental pruning: instead of pruning forward from departure, fit the
//! opening legs backward against the known-good final leg.
//!
//! Free variables (16): dep_offset_days, dep_vinf, theta, phi,
//! (tof_k, eta_k) for legs 0-1, eta_2, and (rp_norm, beta) for all three
//! flybys (Venus, EGA1, EGA2). Leg 2's TOF is NOT free -- it is computed as
//! the remainder so the chain arrives at EGA2 on exactly the real date
//! (leg 3's solution is pinned to that date, so the handoff epoch must
//! match, not merely be near).
//!
//! Fitness = sum(DSM ΔV, legs 0-2) + |v_inf_out(EGA2) - v_inf_target|.
//! The mismatch term is in m/s and is physically the extra impulse leg 3
//! would need at the handoff, so weighting it 1:1 against real DSM ΔV is
//! meaningful, not an arbitrary penalty scale. The EGA2 flyby's own
//! (rp_norm, beta) are searched, so the mismatch measures the FULL vector
//! difference after the best achievable turn -- both magnitude error (which
//! no unpowered flyby can remove) and any un-turnable direction error
//! (turn-angle limit at the minimum safe periapsis).
//!
//! The target v-inf vector is re-derived internally from the same pure
//! Lambert solve (EGA2 -> Jupiter at the fixed real dates) rather than
//! hardcoding the printed 4-decimal values from galileo_leg3_search.
//!
//! Run: `cargo run -p mission_planner --bin galileo_backfit_legs012 --release`

use ephemeris::{Almanac, Body, Epoch};
use nalgebra::Vector3;
use trajectory_solver::{
    evaluate_mga_leg, flyby_turn, keplerian::MU_SUN_M3S2, lambert, DeSolver,
};

fn jd_to_epoch(jd: f64) -> Epoch {
    Epoch::from_unix_seconds((jd - 2_440_587.5) * 86_400.0)
}

fn body_state(almanac: &Almanac, body: Body, jd: f64) -> (Vector3<f64>, Vector3<f64>) {
    let epoch = jd_to_epoch(jd);
    let state = almanac
        .body_state_heliocentric(body, epoch)
        .unwrap_or_else(|e| panic!("ANISE query failed for {body:?} at JD {jd:.1}: {e}"));
    let r = state.position.inner;
    let v = state.velocity.inner;
    (Vector3::new(r[0], r[1], r[2]), Vector3::new(v[0], v[1], v[2]))
}

// Body constants (same values as galileo_real_check.rs; sources cited in
// crates/body_models -- these are local copies for a self-contained debug binary).
const MU_VENUS: f64 = 3.248_599e14;
const R_VENUS:  f64 = 6_051_800.0;
const MU_EARTH: f64 = 3.986_004_418e14;
const R_EARTH:  f64 = 6_371_000.0;

/// Minimum safe flyby periapsis, normalized by body radius. Galileo's real
/// EGA2 pass was 303 km altitude (rp_norm ~ 1.048), so allow down to 1.03.
const RP_NORM_MIN: f64 = 1.03;
const RP_NORM_MAX: f64 = 50.0;

fn main() {
    let almanac = Almanac::new("kernels/de440s.bsp")
        .or_else(|_| Almanac::new("../kernels/de440s.bsp"))
        .expect("de440s.bsp not found");

    // Real historical dates (JD) -- see galileo_real_check.rs for sourcing.
    let jd_dep_base = 2_447_817.5; // 1989-10-18 (launch)
    let jd_ega2     = 2_448_964.5; // 1992-12-08 (2nd Earth flyby -- FIXED handoff epoch)
    let jd_joi      = 2_450_058.5; // 1995-12-07 (Jupiter arrival)

    // ── Derive the leg-3 target departure v-inf at EGA2 (pure Lambert) ──────
    let (r_ega2, v_earth_ega2) = body_state(&almanac, Body::Earth,   jd_ega2);
    let (r_jup,  _v_jup)       = body_state(&almanac, Body::Jupiter, jd_joi);
    let tof3_s = (jd_joi - jd_ega2) * 86_400.0;

    let sols = lambert(
        [r_ega2.x, r_ega2.y, r_ega2.z],
        [r_jup.x,  r_jup.y,  r_jup.z],
        tof3_s, true, MU_SUN_M3S2,
    );
    let (v3_dep_arr, _) = *sols.first().expect("leg-3 prograde Lambert solution must exist");
    let v3_dep = Vector3::new(v3_dep_arr[0], v3_dep_arr[1], v3_dep_arr[2]);
    let v_inf_target = v3_dep - v_earth_ega2;
    println!("=== Leg-3 target departure v-inf at EGA2 (pure Lambert, real dates) ===");
    println!("|v_inf_target| = {:.1} m/s  (galileo_leg3_search found 8918.9 -- should match)", v_inf_target.norm());
    println!("v_inf_target = [{:.1}, {:.1}, {:.1}] m/s\n", v_inf_target.x, v_inf_target.y, v_inf_target.z);

    // ── Chromosome layout (16 params) ────────────────────────────────────────
    //  0: dep_offset_days   [-30, +30] around the real launch date
    //  1: dep_vinf_ms       [2500, 6000]  (real C3=13 km2/s2 -> 3605.6 m/s)
    //  2: theta_dep_rad     [0, 2pi]
    //  3: phi_dep_rad       [-pi/2, pi/2]
    //  4: tof0_days         [90, 150]   (real 115d, Earth->Venus)
    //  5: eta0              [0.01, 0.99]
    //  6: tof1_days         [260, 350]  (real 301d, Venus->EGA1)
    //  7: eta1              [0.01, 0.99]
    //  8: eta2              [0.01, 0.99] (leg-2 TOF is the REMAINDER, not free)
    //  9: rp_norm_venus     [1.03, 50]
    // 10: beta_venus        [-pi, pi]
    // 11: rp_norm_ega1      [1.03, 50]
    // 12: beta_ega1         [-pi, pi]
    // 13: rp_norm_ega2      [1.03, 50]
    // 14: beta_ega2         [-pi, pi]
    // 15: (unused padding? no -- keep exactly 15) -- see bounds below.
    // Bounds tightened around the KNOWN real history (user
    // direction): launch date fixed (dep_offset pinned to ~0 -- DeSolver has
    // no fixed-parameter concept, so a numerically-tiny interval stands in),
    // launch v-inf near the real 3,605.6 m/s (C3=13), leg TOFs near the real
    // 115 d / 301 d. Every removed/tightened dimension raises sampling
    // density -- the same lesson as the earlier tight-bounds joint runs --
    // letting a minutes-scale budget do what the wide 15-dim search needed
    // ~1 h/seed for. The DSMs and flyby parameters stay free to absorb the
    // small differences between our ephemeris/model and the real mission.
    use std::f64::consts::PI;
    let bounds: Vec<(f64, f64)> = vec![
        (-1e-6, 1e-6),          // 0 dep_offset_days (FIXED: real launch 1989-10-18)
        (3_300.0, 3_900.0),     // 1 dep_vinf_ms (real: 3,605.6)
        (0.0, 2.0 * PI),        // 2 theta
        (-PI / 2.0, PI / 2.0),  // 3 phi
        (105.0, 125.0),         // 4 tof0 (real: 115 d)
        (0.01, 0.99),           // 5 eta0
        (285.0, 315.0),         // 6 tof1 (real: 301 d)
        (0.01, 0.99),           // 7 eta1
        (0.01, 0.99),           // 8 eta2
        (RP_NORM_MIN, RP_NORM_MAX), // 9 rp_norm_venus
        (-PI, PI),              // 10 beta_venus
        (RP_NORM_MIN, RP_NORM_MAX), // 11 rp_norm_ega1
        (-PI, PI),              // 12 beta_ega1
        (RP_NORM_MIN, RP_NORM_MAX), // 13 rp_norm_ega2
        (-PI, PI),              // 14 beta_ega2
    ];

    // Detailed evaluation shared by the fitness closure and final reporting.
    // Returns (dsm_total, mismatch, per-leg DSMs, tof2, vinf_in_ega2_mag).
    let eval = |p: &[f64]| -> Option<(f64, f64, [f64; 3], f64, f64)> {
        let dep_jd = jd_dep_base + p[0];
        let tof0   = p[4];
        let tof1   = p[6];
        // Leg 2's TOF is the remainder to hit EGA2 on the real date exactly.
        let tof2 = (jd_ega2 - dep_jd) - tof0 - tof1;
        if tof2 < 400.0 || tof2 > 1000.0 { return None; } // sanity window (real: 731d)

        let (r_dep, v_dep) = body_state(&almanac, Body::Earth, dep_jd);
        let vinf  = p[1];
        let theta = p[2];
        let phi   = p[3];
        let v_inf_vec = Vector3::new(
            vinf * phi.cos() * theta.cos(),
            vinf * phi.cos() * theta.sin(),
            vinf * phi.sin(),
        );
        let mut r_sc = r_dep;
        let mut v_sc = v_dep + v_inf_vec;

        let mut dsm = [0.0_f64; 3];
        let mut t_days = 0.0;

        // Leg 0: Earth -> Venus
        let jd_ven = dep_jd + tof0;
        let (r_ven, v_ven) = body_state(&almanac, Body::Venus, jd_ven);
        let leg0 = evaluate_mga_leg(r_sc, v_sc, p[5], tof0 * 86_400.0, r_ven, v_ven, MU_SUN_M3S2)?;
        dsm[0] = leg0.dv_dsm_ms;
        let v_out_ven = flyby_turn(leg0.v_inf_arr_mps, p[9] * R_VENUS, p[10], MU_VENUS);
        r_sc = r_ven;
        v_sc = v_ven + v_out_ven;
        t_days += tof0;

        // Leg 1: Venus -> Earth (EGA1)
        let jd_ega1 = dep_jd + t_days + tof1;
        let (r_e1, v_e1) = body_state(&almanac, Body::Earth, jd_ega1);
        let leg1 = evaluate_mga_leg(r_sc, v_sc, p[7], tof1 * 86_400.0, r_e1, v_e1, MU_SUN_M3S2)?;
        dsm[1] = leg1.dv_dsm_ms;
        let v_out_e1 = flyby_turn(leg1.v_inf_arr_mps, p[11] * R_EARTH, p[12], MU_EARTH);
        r_sc = r_e1;
        v_sc = v_e1 + v_out_e1;

        // Leg 2: Earth (EGA1) -> Earth (EGA2), resonant return -- multi-rev
        // Lambert inside evaluate_mga_leg handles the N=2 branch.
        let leg2 = evaluate_mga_leg(r_sc, v_sc, p[8], tof2 * 86_400.0, r_ega2, v_earth_ega2, MU_SUN_M3S2)?;
        dsm[2] = leg2.dv_dsm_ms;

        // EGA2 flyby: turn the incoming v-inf, compare against the leg-3 target.
        let vinf_in_ega2 = leg2.v_inf_arr_mps;
        let v_out_e2 = flyby_turn(vinf_in_ega2, p[13] * R_EARTH, p[14], MU_EARTH);
        let mismatch = (v_out_e2 - v_inf_target).norm();

        Some((dsm[0] + dsm[1] + dsm[2], mismatch, dsm, tof2, vinf_in_ega2.norm()))
    };

    let fitness = |p: &[f64]| -> Option<f64> {
        let (dsm_total, mismatch, _, _, _) = eval(p)?;
        Some(dsm_total + mismatch)
    };

    println!("=== Backward-fit DE search: legs 0-2 (E-V-E-E opening), EGA2 date fixed ===");
    let mut best_overall: Option<(f64, Vec<f64>)> = None;
    // Small budget: with the history-pinned bounds above, the searched space
    // is tiny compared to the original wide 15-dim box (whose 4-seed pop-400
    // x gen-800 run cost ~1 h/seed; only seed 9042 found the real family at
    // fitness 274.5 m/s -- seeds 42/1042 landed in a ~6.7 km/s wrong
    // resonant branch). Multi-seed retained as the basin check.
    for seed in [42u64, 1042, 9042] {
        let solver = DeSolver {
            population_size: 150,
            generations: 300,
            f_weight: 0.6,
            cr: 0.9,
            seed,
        };
        let result = solver.run(&bounds, fitness);
        let detail = eval(&result.best_params);
        match detail {
            Some((dsm_total, mismatch, dsms, tof2, vinf_in)) => {
                println!(
                    "seed={seed:>10}  fitness={:>9.1}  DSMs=[{:.1}, {:.1}, {:.1}]  mismatch@EGA2={:>8.1} m/s  tof2={:.1}d  |vinf_in(EGA2)|={:.1} m/s",
                    result.best_fitness, dsms[0], dsms[1], dsms[2], mismatch, tof2, vinf_in,
                );
                let p = &result.best_params;
                println!("  chromosome: dep_offset={:+.3}d  dep_vinf={:.1} m/s  theta={:.4}  phi={:.4}", p[0], p[1], p[2], p[3]);
                println!("              tof0={:.2}d eta0={:.3}  tof1={:.2}d eta1={:.3}  eta2={:.3}", p[4], p[5], p[6], p[7], p[8]);
                println!("              rp/beta: Venus {:.3}/{:.4}  EGA1 {:.3}/{:.4}  EGA2 {:.3}/{:.4}",
                    p[9], p[10], p[11], p[12], p[13], p[14]);
                println!("              raw params: {:?}", p);
                let _ = dsm_total;
            }
            None => println!("seed={seed:>10}  best infeasible (fitness={:.1})", result.best_fitness),
        }
        match &best_overall {
            Some((f, _)) if *f <= result.best_fitness => {}
            _ => best_overall = Some((result.best_fitness, result.best_params.clone())),
        }
    }

    let (best_fit, p) = best_overall.expect("at least one seed ran");
    let (dsm_total, mismatch, dsms, tof2, vinf_in) = eval(&p).expect("best must be feasible");

    println!("\n=== Best across all seeds ===");
    println!("Total fitness            = {best_fit:.1} m/s  (DSM total {dsm_total:.1} + EGA2 handoff mismatch {mismatch:.1})");
    println!("Per-leg DSMs             = [{:.1}, {:.1}, {:.1}] m/s  (real Galileo TCMs were ~1-11 m/s each)", dsms[0], dsms[1], dsms[2]);
    println!("dep date offset          = {:+.2} days from 1989-10-18", p[0]);
    println!("dep v_inf                = {:.1} m/s  (real: 3605.6 from C3=13 km2/s2)", p[1]);
    println!("TOFs                     = {:.1}d, {:.1}d, {:.1}d  (real: 115, 301, 731)", p[4], p[6], tof2);
    println!("|v_inf_in| at EGA2       = {:.1} m/s  (target |v_inf_out| = {:.1} -- unpowered flyby needs equal magnitudes)",
        vinf_in, v_inf_target.norm());
    println!("Flyby rp_norm            = Venus {:.3}, EGA1 {:.3}, EGA2 {:.3}  (real: ~4.64, ~1.151, ~1.048)",
        p[9], p[11], p[13]);

    println!("\n=== Interpretation ===");
    println!("If DSM total AND mismatch are both order-10s-of-m/s, the full 4-leg VEEGA closes");
    println!("near-ballistically in our model at the real dates: total mission cost = departure");
    println!("escape burn (from v_inf ~3.6 km/s) + this small correction budget, matching the");
    println!("real Galileo. The combined chromosome (these legs 0-2 + the ballistic leg 3) can");
    println!("then seed the joint DE / multiple-shooting refiner. If mismatch stays large, check");
    println!("whether it is magnitude error (legs 0-2 arrive with wrong |v_inf| -- geometry");
    println!("problem) or direction error (turn-angle limit at min periapsis).");
}
