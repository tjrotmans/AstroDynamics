//! GTOP Cassini-1 published-solution cross-check (Phase 9w-ii).
//!
//! Evaluates the PUBLISHED best-known GTOP Cassini-1 decision vector with the
//! Phase 9w-i ballistic powered-flyby evaluator (`orbital_math::lambert` legs
//! + `trajectory_solver::mga_scan::powered_flyby` matching burns) — Cassini-1
//! is exactly the no-DSM powered-flyby MGA formulation `mga_scan` implements,
//! so this is the validation benchmark for that module the same way
//! `gtop_cassini2_check` validates the MGA-1DSM leg evaluator.
//!
//! Problem definition (ESA GTOP archive + GTOPX solution file, retrieved
//!):
//!   https://www.esa.int/gsp/ACT/projects/gtop/cassini1/
//!   https://www.midaco-solver.com/data/gtopx/solutions/cassini1.txt
//!   Sequence: Earth → Venus → Venus → Earth → Jupiter → Saturn.
//!   Decision vector: [t0 (MJD2000), T1..T5 (days)] — 6 variables, pure
//!   ballistic Lambert legs, no DSMs, no pointing variables.
//!   GTOPX best known: f = 4930.708734 m/s at
//!     x = [-789.7839433782882, 158.323732108266285, 449.385881146596375,
//!          54.716205454453835, 1024.69612346664735, 4552.813287140832472]
//!   Constraint values in the solution file show the Venus-1 and Earth flyby
//!   periapses are pinned EXACTLY at their floors (margins ≈ 5e-5 km and
//!   2e-4 km) — active constraints, which matters for reproduction (below).
//!
//! Objective (GTOP `mga.m` / GTOPX, "orbit insertion" type):
//!   ΔV_total = |v∞_dep|                       (raw launch excess — NOT a
//!                                              parking-orbit escape burn; see
//! the design notes 9v-v conventions)
//!            + Σ powered-flyby burns          (periapsis burn, r_p solved
//!                                              UNCONSTRAINED from the turn
//!                                              equation; the floor is a
//!                                              separately-reported constraint,
//!                                              not an evaluation failure)
//!            + ΔV_insertion at Saturn         (into r_p = 108 950 km, e = 0.98)
//!   Flyby periapsis floors: Venus 6 351.8 km (both), Earth 6 778.1 km,
//!   Jupiter 671 492 km.
//!
//! Lambert transfer sense: GTOP's `mga.m` picks the long/short way from the
//! sign of (r1 × r2)·ẑ — equivalent to always-prograde heliocentric motion,
//! `lambert(..., prograde = true)` here. The Venus→Venus leg is a 1:2
//! resonant return (Δν ≈ 359.93°), which needs the relaxed near-0°/360°
//! transfer-angle guard (`lambert_with_min_transfer_angle`).
//!
//! # Reproduction caveat — why a refine mode exists
//! On the resonant V-V leg the departure v∞ DIRECTION at Venus is set by the
//! tiny 449-day Venus-to-Venus chord (~130 000 km), and the lever arm from
//! heliocentric velocity (~41 km/s) to v∞ (~6 km/s) amplifies chord-direction
//! differences ~7×. Different analytic-ephemeris element sets (our Standish
//! table vs. GTOP's own) therefore shift the required turn at the Venus-1
//! flyby by tens of degrees, and with two flyby constraints ACTIVE the
//! published vector evaluates measurably differently under any ephemeris that
//! isn't bit-identical to GTOP's. So, like `gtop_cassini2_check`'s
//! GTOP_BEST_CHAIN mode, the meaningful evaluator validation is:
//! `GTOP_REFINE=1` locally re-optimizes the SAME 6 continuous variables from
//! the published starting point under OUR evaluator/ephemeris — the optimum
//! should come back to ≈4.93 km/s with the same leg structure if the
//! evaluator (Lambert + powered flyby + insertion) is correct.
//!
//! Run:
//!   cargo run -p mission_planner --bin gtop_cassini1_check --release
//!   GTOP_LP=1 cargo run -p mission_planner --bin gtop_cassini1_check --release
//!   GTOP_LP=1 GTOP_REFINE=1 cargo run -p mission_planner --bin gtop_cassini1_check --release

use ephemeris::{Almanac, Body, Epoch};
use nalgebra::Vector3;
use trajectory_solver::{lambert_with_min_transfer_angle, mga_scan::powered_flyby, DeSolver};

/// Relaxed near-0°/360° transfer-angle guard for resonant return legs —
/// Cassini-1's V-V leg is a 1:2 resonant re-encounter at Δν ≈ 359.93°
/// (sin Δν ≈ 1.2e-3), which the default guard rejects. See
/// `lambert_with_min_transfer_angle`'s docs for why the near-180° band
/// stays guarded regardless.
const MIN_SIN_DNU_RESONANT: f64 = 1.0e-4;

/// GTOPX published best-known decision vector and objective (see module docs).
const X_PUBLISHED: [f64; 6] = [
    -789.783_943_378_288_2,
    158.323_732_108_266_285,
    449.385_881_146_596_375,
    54.716_205_454_453_835,
    1_024.696_123_466_647_35,
    4_552.813_287_140_832_472,
];
const F_PUBLISHED_MS: f64 = 4_930.708_733_982_513;

const BODIES: [Body; 6] = [Body::Earth, Body::Venus, Body::Venus, Body::Earth, Body::Jupiter, Body::Saturn];
/// Flyby-body μ [m³/s²] — GTOP's own `MU[]` table (gtopx.cpp), NOT this
/// repo's catalog values: Venus 324 860, Earth 398 601.19, Jupiter 126.7e6
/// km³/s². Venus, Venus, Earth, Jupiter.
const FLYBY_MU: [f64; 4] = [3.248_60e14, 3.248_60e14, 3.986_011_9e14, 1.267e17];
/// GTOP's periapsis penalty floors [m] (their `penalty[]` table — note
/// Jupiter is 600 000 km in the CODE, with 671 492 km commented out; the
/// archive page documents the commented-out value).
const RP_FLOOR_M: [f64; 4] = [6_351_800.0, 6_351_800.0, 6_778_100.0, 600_000_000.0];
/// GTOP's penalty coefficients converted to [m/s per m below the floor]
/// (their 0.01 / 0.001 km/s-per-km values).
const RP_PENALTY_MS_PER_M: [f64; 4] = [0.01, 0.01, 0.01, 0.001];

/// Saturn capture ellipse (GTOP problem spec): r_p = 108 950 km, e = 0.98,
/// μ_Saturn = 37.9e6 km³/s² (GTOP's value).
const MU_SATURN: f64 = 3.79e16;
const R_CAP_M: f64 = 108_950_000.0;
const E_CAP: f64 = 0.98;

/// GTOP's Sun μ [m³/s²] — all Lambert legs use this, not DE430's value.
const MU_SUN_GTOP: f64 = mission_planner::gtop_lp::GTOP_MU_SUN_M3S2;

/// JD to Epoch using the same formula as `design.rs::jd_to_epoch`.
fn jd_to_epoch(jd: f64) -> Epoch {
    Epoch::from_unix_seconds((jd - 2_440_587.5) * 86_400.0)
}

struct FlybyDetail {
    dv_ms: f64,
    rp_m: f64,
    turn_deg: f64,
    vinf_in_ms: f64,
    vinf_out_ms: f64,
}

struct ChainEval {
    vinf_dep_ms: f64,
    vinf_arr_ms: f64,
    flybys: Vec<FlybyDetail>,
    dv_insertion_ms: f64,
    /// Raw ΔV sum: |v∞_dep| + Σ flyby burns + insertion [m/s].
    total_ms: f64,
    /// GTOP's graded floor penalty Σ coeff·(floor − rp)⁺ [m/s] — the GTOP
    /// objective is `total_ms + penalty_ms` (zero for feasible solutions).
    penalty_ms: f64,
}

/// Evaluate the full ballistic E-V-V-E-J-S chain for decision vector
/// `x = [t0_mjd2000, T1..T5]` against the supplied ephemeris. Returns `None`
/// only when a Lambert leg has no solution at all.
fn evaluate_chain(x: &[f64], bstate: &dyn Fn(Body, f64) -> (Vector3<f64>, Vector3<f64>)) -> Option<ChainEval> {
    // Encounter epochs in MJD2000 (GTOP's native time variable).
    let mut encounter = vec![x[0]];
    for t in &x[1..6] {
        encounter.push(encounter.last().unwrap() + t);
    }

    let mut vinf_dep_ms = 0.0_f64;
    let mut flybys: Vec<FlybyDetail> = Vec::new();
    let mut vinf_in: Option<Vector3<f64>> = None;
    let mut vinf_arr = Vector3::zeros();
    let mut penalty_ms = 0.0_f64;

    for k in 0..5 {
        let (r_a, v_a) = bstate(BODIES[k], encounter[k]);
        let (r_b, v_b) = bstate(BODIES[k + 1], encounter[k + 1]);
        let tof_s = x[1 + k] * 86_400.0;
        if tof_s <= 0.0 {
            return None;
        }

        // Prograde Lambert (GTOP's lw rule — see module docs); on multiple
        // roots take the one with the cheapest local connection.
        let sols = lambert_with_min_transfer_angle(
            [r_a.x, r_a.y, r_a.z],
            [r_b.x, r_b.y, r_b.z],
            tof_s,
            true,
            MU_SUN_GTOP,
            MIN_SIN_DNU_RESONANT,
        );
        if sols.is_empty() {
            return None;
        }

        let mut best: Option<(f64, Vector3<f64>, Option<FlybyDetail>)> = None;
        for (v1a, v2a) in &sols {
            let v1 = Vector3::new(v1a[0], v1a[1], v1a[2]);
            let v2 = Vector3::new(v2a[0], v2a[1], v2a[2]);
            let vinf_out = v1 - v_a;
            let (cost, detail) = match vinf_in {
                None => (vinf_out.norm(), None),
                Some(vin) => {
                    let alpha = (vin.dot(&vinf_out) / (vin.norm() * vinf_out.norm()))
                        .clamp(-1.0, 1.0)
                        .acos();
                    // GTOP convention: r_p solved UNCONSTRAINED (floor is a
                    // reported constraint, not an eval failure) — pass an
                    // effectively-zero floor; alpha < π is always solvable.
                    let fb = powered_flyby(vin.norm(), vinf_out.norm(), alpha, FLYBY_MU[k - 1], 1.0)?;
                    (
                        fb.dv_ms,
                        Some(FlybyDetail {
                            dv_ms: fb.dv_ms,
                            rp_m: fb.rp_m,
                            turn_deg: alpha.to_degrees(),
                            vinf_in_ms: vin.norm(),
                            vinf_out_ms: vinf_out.norm(),
                        }),
                    )
                }
            };
            if best.as_ref().map_or(true, |(c, ..)| cost < *c) {
                best = Some((cost, v2, detail));
            }
        }
        let (cost, v2, detail) = best?;

        match detail {
            None => vinf_dep_ms = cost,
            Some(d) => {
                // GTOP's graded periapsis-floor penalty (their `penalty_coeffs`).
                penalty_ms += RP_PENALTY_MS_PER_M[k - 1] * (RP_FLOOR_M[k - 1] - d.rp_m).max(0.0);
                flybys.push(d);
            }
        }
        vinf_arr = v2 - v_b;
        vinf_in = Some(vinf_arr);
    }

    let vinf_arr_ms = vinf_arr.norm();
    let v_hyp = (vinf_arr_ms * vinf_arr_ms + 2.0 * MU_SATURN / R_CAP_M).sqrt();
    let v_peri = (MU_SATURN * (1.0 + E_CAP) / R_CAP_M).sqrt();
    let dv_insertion_ms = v_hyp - v_peri;

    let total_ms = vinf_dep_ms + flybys.iter().map(|f| f.dv_ms).sum::<f64>() + dv_insertion_ms;
    Some(ChainEval {
        vinf_dep_ms,
        vinf_arr_ms,
        flybys,
        dv_insertion_ms,
        total_ms,
        penalty_ms,
    })
}

fn print_eval(x: &[f64], ev: &ChainEval) {
    println!(
        "  x = [t0 = {:.4} MJD2000, T = {:.3} / {:.3} / {:.3} / {:.3} / {:.3} d]",
        x[0], x[1], x[2], x[3], x[4], x[5]
    );
    println!("  Launch v∞:        {:9.2} m/s", ev.vinf_dep_ms);
    let flyby_names = ["Venus-1", "Venus-2", "Earth", "Jupiter"];
    for (i, f) in ev.flybys.iter().enumerate() {
        let floor_km = RP_FLOOR_M[i] / 1e3;
        let marker = if f.rp_m >= RP_FLOOR_M[i] { "ok" } else { "VIOLATED" };
        println!(
            "  {:8} flyby:   {:9.2} m/s   v∞ {:7.1}→{:7.1} m/s  turn {:6.2}°  rp {:9.0} km (floor {:7.0} km, {marker})",
            flyby_names[i], f.dv_ms, f.vinf_in_ms, f.vinf_out_ms, f.turn_deg, f.rp_m / 1e3, floor_km
        );
    }
    println!(
        "  Saturn insertion: {:9.2} m/s   (v∞_arr = {:.1} m/s)",
        ev.dv_insertion_ms, ev.vinf_arr_ms
    );
    println!("  TOTAL (raw ΔV):   {:9.2} m/s", ev.total_ms);
    if ev.penalty_ms > 0.0 {
        println!("  Floor penalty:    {:9.2} m/s", ev.penalty_ms);
    }
    let objective = ev.total_ms + ev.penalty_ms;
    println!("  GTOP objective:   {objective:9.2} m/s");
    println!(
        "  Δ vs published:   {:+9.2} m/s  ({:+.3}%)",
        objective - F_PUBLISHED_MS,
        (objective - F_PUBLISHED_MS) / F_PUBLISHED_MS * 100.0
    );
}

fn main() {
    let almanac = Almanac::new("kernels/de440s.bsp")
        .or_else(|_| Almanac::new("../kernels/de440s.bsp"))
        .expect("de440s.bsp not found — download from https://public-data.nyxspace.com/anise/de440s.bsp");

    // Ephemeris selection: GTOP's own analytic elements by default (the
    // benchmark objective is only reproducible against those — see module
    // docs); GTOP_EPH=lp → Standish table; GTOP_EPH=de440 → ANISE/DE440s.
    let eph_mode = std::env::var("GTOP_EPH").unwrap_or_else(|_| "gtop".to_string());
    let eph_name = match eph_mode.as_str() {
        "lp" => "Standish analytic elements",
        "de440" => "DE440s (ANISE)",
        _ => "GTOP's own analytic elements (exact reproduction)",
    };
    let bstate = move |b: Body, mjd2000: f64| -> (Vector3<f64>, Vector3<f64>) {
        let jd = 2_451_544.5 + mjd2000;
        match eph_mode.as_str() {
            "lp" => mission_planner::gtop_lp::lp_state_icrf(b, jd),
            "de440" => {
                let state = almanac
                    .body_state_heliocentric(b, jd_to_epoch(jd))
                    .unwrap_or_else(|e| panic!("ANISE query failed for {b:?} at JD {jd:.1}: {e}"));
                let r = state.position.inner;
                let v = state.velocity.inner;
                (Vector3::new(r[0], r[1], r[2]), Vector3::new(v[0], v[1], v[2]))
            }
            _ => mission_planner::gtop_lp::gtop_state_icrf(b, mjd2000),
        }
    };

    println!("=== GTOP Cassini-1 published-solution cross-check (9w-ii) ===");
    println!("Ephemeris: {eph_name}\n");

    println!("── Published GTOPX vector, evaluated directly ──");
    match evaluate_chain(&X_PUBLISHED, &bstate) {
        Some(ev) => print_eval(&X_PUBLISHED, &ev),
        None => println!("  [infeasible under this ephemeris — see module docs on resonant-leg sensitivity]"),
    }

    // ── GTOP_REFINE=1: local re-optimization under OUR evaluator ────────────
    // Same 6 continuous variables, bounds = published point ± a local window
    // (t0 ±30 d, each TOF ±8%); DE/rand/1/bin, penalty for floor violations
    // (the published optimum pins Venus-1 and Earth exactly at the floor, so
    // the refined one is expected to sit on that boundary too).
    if std::env::var("GTOP_REFINE").is_ok() {
        println!("\n── GTOP_REFINE: local DE re-optimization under our evaluator/ephemeris ──");
        let mut bounds = [(0.0_f64, 0.0_f64); 6];
        bounds[0] = (X_PUBLISHED[0] - 60.0, X_PUBLISHED[0] + 60.0);
        for i in 1..6 {
            bounds[i] = (X_PUBLISHED[i] * 0.80, X_PUBLISHED[i] * 1.20);
        }
        let fitness = |x: &[f64]| -> Option<f64> {
            // GTOP's own objective: raw ΔV + graded floor penalty.
            evaluate_chain(x, &bstate).map(|ev| ev.total_ms + ev.penalty_ms)
        };
        let solver = DeSolver {
            population_size: 100,
            generations: 1500,
            f_weight: 0.7,
            cr: 0.9,
            seed: 42,
        };
        let (result, _) = solver.run_seeded_with_progress(
            &bounds,
            &[X_PUBLISHED.to_vec()],
            fitness,
            |_gen, _best, _params| {},
        );
        println!(
            "  refined fitness: {:.2} m/s (incl. any floor penalty)\n",
            result.best_fitness
        );
        match evaluate_chain(&result.best_params, &bstate) {
            Some(ev) => print_eval(&result.best_params, &ev),
            None => println!("  [refined point infeasible?!]"),
        }
    }
}
