//! Isolated leg-3 (EGA2 -> Jupiter/JOI) departure-v-infinity search
//! (Phase 9x-iv follow-up).
//!
//! Question: does a genuinely cheap (order-100s-of-m/s, matching real TCM
//! scale) leg 3 exist AT ALL for the real historical dates (EGA2:
//! 1992-12-08, JOI: 1995-12-07, TOF = 1094 days FIXED, not a bound) --
//! independent of how the spacecraft actually arrives at EGA2? If yes, this
//! pins down the departure v-infinity (magnitude + direction) that a
//! backward-fit of legs 0-2 (Venus-Earth-Earth) needs to hand off at EGA2.
//! If no cheap departure v-infinity exists at any direction, leg 3 itself has
//! a deeper issue (also needs multi-rev on ITS side, or the real JOI
//! approach used a powered correction our Flyby-only objective doesn't
//! model) and the "backward-fit legs 0-2" plan is not worth pursuing.
//!
//! Method: DE search over (v_inf_ms, theta_rad, phi_rad, eta) -- departure
//! velocity at EGA2 is v_body(EGA2) + v_inf_vec(theta, phi), completely
//! decoupled from any incoming leg. TOF and both endpoint epochs are FIXED
//! at the real historical values (not free variables) -- unlike the
//! chromosome's dep_offset/leg_tof_days, which are searched. Multi-rev
//! (n_rev = 0..=2) is handled automatically by `evaluate_mga_leg`'s internal
//! call to `lambert_min_dv_multi_rev` (Phase 9x-iv), so a resonant leg-3
//! solution (1094 days is close to 3 Earth years) would already be found
//! without any special-casing here.
//!
//! Fitness = DSM ΔV only (`leg.dv_dsm_ms`). The departure v-infinity itself
//! costs nothing in this isolated search -- it is a free parameter standing
//! in for "whatever legs 0-2 hand off at EGA2", not a burn. This directly
//! answers the question above: the DE's best DSM ΔV IS the minimum cost of
//! a real-date, real-arrival leg 3, as a function of the (unconstrained)
//! departure v-infinity.
//!
//! Run: `cargo run -p mission_planner --bin galileo_leg3_search --release`

use ephemeris::{Almanac, Body, Epoch};
use nalgebra::Vector3;
use trajectory_solver::{evaluate_mga_leg, keplerian::MU_SUN_M3S2, DeSolver};

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

fn main() {
    let almanac = Almanac::new("kernels/de440s.bsp")
        .or_else(|_| Almanac::new("../kernels/de440s.bsp"))
        .expect("de440s.bsp not found");

    // Real historical dates (JD, per NASA/JPL press materials -- see
    // the design notes Phase 9x-iv and galileo_real_check.rs for sourcing).
    let jd_ega2 = 2_448_964.5; // 1992-12-08 (2nd Earth flyby, altitude 303 km)
    let jd_joi  = 2_450_058.5; // 1995-12-07 (Jupiter arrival / JOI)
    let tof_s   = (jd_joi - jd_ega2) * 86_400.0;

    println!("=== Leg 3 isolated search: EGA2 (1992-12-08) -> Jupiter/JOI (1995-12-07) ===");
    println!("Fixed TOF = {:.1} days ({:.3} Earth years)\n", tof_s / 86_400.0, (jd_joi - jd_ega2) / 365.25);

    let (r_ega2, v_ega2) = body_state(&almanac, Body::Earth,   jd_ega2);
    let (r_jup,  v_jup)  = body_state(&almanac, Body::Jupiter, jd_joi);

    // Free variables: [v_inf_ms, theta_rad, phi_rad, eta]. Departure velocity
    // at EGA2 is v_ega2 + v_inf_vec(theta, phi) -- completely decoupled from
    // any incoming leg, per this binary's whole point (see module doc).
    let bounds = vec![
        (0.0, 20_000.0),                     // v_inf_ms -- generous upper bound
        (0.0, 2.0 * std::f64::consts::PI),   // theta_rad
        (-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2), // phi_rad
        (0.01, 0.99),                        // eta
    ];

    let fitness = |p: &[f64]| -> Option<f64> {
        let vinf  = p[0];
        let theta = p[1];
        let phi   = p[2];
        let eta   = p[3];
        let v_inf_vec = Vector3::new(
            vinf * phi.cos() * theta.cos(),
            vinf * phi.cos() * theta.sin(),
            vinf * phi.sin(),
        );
        let v_sc_start = v_ega2 + v_inf_vec;
        let leg = evaluate_mga_leg(r_ega2, v_sc_start, eta, tof_s, r_jup, v_jup, MU_SUN_M3S2)?;
        Some(leg.dv_dsm_ms)
    };

    // Multi-seed sweep (a single DE seed can
    // land in a locally-good but globally-suboptimal basin -- see the
    // Benchmark Regression Policy in the design notes).
    let mut best_overall: Option<(f64, Vec<f64>)> = None;
    for seed in [42u64, 1042, 9042, 20260710] {
        let solver = DeSolver {
            population_size: 300,
            generations: 400,
            f_weight: 0.6,
            cr: 0.9,
            seed,
        };
        let result = solver.run(&bounds, fitness);
        println!(
            "seed={seed:>10}  best_dsm={:>10.1} m/s  v_inf={:>8.1} m/s  theta={:>7.4} rad  phi={:>7.4} rad  eta={:.4}",
            result.best_fitness, result.best_params[0], result.best_params[1], result.best_params[2], result.best_params[3],
        );
        match &best_overall {
            Some((f, _)) if *f <= result.best_fitness => {}
            _ => best_overall = Some((result.best_fitness, result.best_params.clone())),
        }
    }

    let (best_dsm, best_p) = best_overall.expect("at least one seed ran");
    println!("\n=== Best across all seeds ===");
    println!("DSM DeltaV = {best_dsm:.1} m/s");
    println!("Departure v_inf @ EGA2 = {:.1} m/s, theta = {:.4} rad, phi = {:.4} rad, eta = {:.4}",
        best_p[0], best_p[1], best_p[2], best_p[3]);

    let vinf  = best_p[0];
    let theta = best_p[1];
    let phi   = best_p[2];
    let eta   = best_p[3];
    let v_inf_vec = Vector3::new(
        vinf * phi.cos() * theta.cos(),
        vinf * phi.cos() * theta.sin(),
        vinf * phi.sin(),
    );
    let v_sc_start = v_ega2 + v_inf_vec;
    let leg = evaluate_mga_leg(r_ega2, v_sc_start, eta, tof_s, r_jup, v_jup, MU_SUN_M3S2)
        .expect("best params must be feasible");
    println!("Arrival v_inf at Jupiter = {:.1} m/s", leg.v_inf_arr_mps.norm());

    println!("\n=== Interpretation ===");
    println!("If best DSM DeltaV is order-100s-of-m/s (matching real TCM scale, ~11 m/s per");
    println!("maneuver historically), leg 3 IS cheap for the real dates at SOME departure v_inf --");
    println!("confirming the backward-fit plan (search legs 0-2 for a VEEGA opening that hands");
    println!("off a matching v_inf at EGA2) is worth pursuing. If it stays in the multi-km/s range");
    println!("across all seeds/directions, leg 3 itself has an unresolved issue independent of");
    println!("how the spacecraft reaches EGA2 (e.g. needs its own multi-rev variant beyond what is");
    println!("already tried, or the real JOI approach used a powered correction this Flyby-only");
    println!("objective does not model).");
}
