//! MGA-1DSM standalone evaluator demo (Phase 9f-i).
//!
//! Evaluates a hard-coded Earth → Venus → Jupiter chromosome to verify that
//! the MGA-1DSM leg evaluator (`evaluate_mga_leg` + `flyby_turn`) is
//! numerically correct before committing to a full optimizer run.
//!
//! Run:
//!   cargo run -p mission_planner --bin mga_demo --release
//!
//! Expected output (approximate, with ANISE ephemeris):
//!   - Leg 0 (Earth→Venus): DSM ΔV finite and ≥ 0 m/s
//!   - Leg 1 (Venus→Jupiter): DSM ΔV finite and ≥ 0 m/s
//!   - Flyby turn conserves |v∞| across Venus
//!
//! A physical plausibility check, not a tight numerical test — the exact values
//! depend on the ANISE ephemeris state at the queried Julian dates.

use ephemeris::{Almanac, Body, Epoch};
use nalgebra::Vector3;
use trajectory_solver::{evaluate_mga_leg, flyby_turn, keplerian::MU_SUN_M3S2};

// JD to Epoch using the same formula as `design.rs::jd_to_epoch`.
fn jd_to_epoch(jd: f64) -> Epoch {
    Epoch::from_unix_seconds((jd - 2_440_587.5) * 86_400.0)
}

fn body_state_vec3(almanac: &Almanac, body: Body, jd: f64) -> (Vector3<f64>, Vector3<f64>) {
    let epoch = jd_to_epoch(jd);
    let state = almanac
        .body_state_heliocentric(body, epoch)
        .unwrap_or_else(|e| panic!("ANISE query failed for {body:?} at JD {jd:.1}: {e}"));
    // ephemeris::Almanac already converts ANISE's km to metres internally
    // (crates/ephemeris/src/almanac.rs::km_to_m_helio). This demo previously
    // multiplied by 1e3 on top — a real units bug (1000 AU "planets") that
    // its loose finite/non-negative sanity checks never caught; found
    // while building gtop_cassini2_check on the same template.
    let r = state.position.inner;
    let v = state.velocity.inner;
    (
        Vector3::new(r[0], r[1], r[2]),
        Vector3::new(v[0], v[1], v[2]),
    )
}

fn main() {
    // Load ANISE ephemeris kernel (de440s.bsp).
    let almanac = Almanac::new("kernels/de440s.bsp")
        .or_else(|_| Almanac::new("../kernels/de440s.bsp"))
        .expect("de440s.bsp not found — download from https://public-data.nyxspace.com/anise/de440s.bsp");

    // ── Hard-coded E→V→J candidate ─────────────────────────────────────────
    // 2030-01-01 ≈ JD 2462867.5
    let dep_jd    : f64 = 2_462_867.5;
    let dep_vinf  : f64 = 4_200.0;   // m/s (C3 ≈ 17.6 km²/s²)
    let theta_dep : f64 = 1.2;       // rad
    let phi_dep   : f64 = 0.15;      // rad
    let tof_0_days: f64 = 145.0;     // Earth→Venus
    let eta_0     : f64 = 0.55;
    let tof_1_days: f64 = 820.0;     // Venus→Jupiter
    let eta_1     : f64 = 0.40;
    let rp_norm_0 : f64 = 1.8;       // rp / R_Venus
    let beta_0    : f64 = 0.9;       // rad

    let mu_venus = 3.248_599e14_f64;  // m³/s² (Williams 2021)
    let r_venus  = 6_051_800.0_f64;   // m     (IAU 2015)

    let t1_jd = dep_jd + tof_0_days;
    let t2_jd = t1_jd  + tof_1_days;

    let (r_earth, v_earth) = body_state_vec3(&almanac, Body::Earth,   dep_jd);
    let (r_venus_t1, v_venus_t1) = body_state_vec3(&almanac, Body::Venus,   t1_jd);
    let (r_jupit_t2, v_jupit_t2) = body_state_vec3(&almanac, Body::Jupiter, t2_jd);

    // Departure state.
    let v_inf_vec = Vector3::new(
        dep_vinf * phi_dep.cos() * theta_dep.cos(),
        dep_vinf * phi_dep.cos() * theta_dep.sin(),
        dep_vinf * phi_dep.sin(),
    );
    let r_sc0 = r_earth;
    let v_sc0 = v_earth + v_inf_vec;

    println!("=== MGA-1DSM Evaluator Demo: Earth → Venus → Jupiter ===\n");
    println!("Departure JD:   {dep_jd:.1}  (~2030-01-01)");
    println!("Dep v∞:         {dep_vinf:.1} m/s  (C3 = {:.2} km²/s²)", dep_vinf * dep_vinf * 1e-6);
    println!("Venus arrival:  JD {t1_jd:.1}  (day {tof_0_days:.0})");
    println!("Jupiter arrival:JD {t2_jd:.1}  (day {:.0})\n", tof_0_days + tof_1_days);

    // Leg 0: Earth → Venus.
    let leg0 = evaluate_mga_leg(
        r_sc0, v_sc0, eta_0, tof_0_days * 86_400.0,
        r_venus_t1, v_venus_t1, MU_SUN_M3S2,
    );
    let leg0 = match leg0 {
        Some(l) => {
            println!("Leg 0 (Earth → Venus):");
            println!("  DSM ΔV:     {:.2} m/s", l.dv_dsm_ms);
            println!("  DSM r:      [{:.3e}, {:.3e}, {:.3e}] m",
                l.r_dsm_m.x, l.r_dsm_m.y, l.r_dsm_m.z);
            println!("  Arrival v∞: {:.2} m/s\n", l.v_inf_arr_mps.norm());
            l
        }
        None => {
            eprintln!("Leg 0 INFEASIBLE — try a different theta/phi or tof");
            std::process::exit(1);
        }
    };

    // Venus flyby turn.
    let r_p_venus  = rp_norm_0 * r_venus;
    let v_inf_in   = leg0.v_inf_arr_mps;
    let v_inf_out  = flyby_turn(v_inf_in, r_p_venus, beta_0, mu_venus);
    let delta_deg  = (v_inf_out.dot(&v_inf_in) / (v_inf_out.norm() * v_inf_in.norm()))
        .acos().to_degrees();
    let v_inf_cons = (v_inf_in.norm() - v_inf_out.norm()).abs() / v_inf_in.norm().max(1.0);

    println!("Venus flyby (rp_norm = {rp_norm_0:.2}):");
    println!("  Periapsis:    {:.1} km", r_p_venus * 1e-3);
    println!("  v∞ in:        {:.2} m/s", v_inf_in.norm());
    println!("  v∞ out:       {:.2} m/s", v_inf_out.norm());
    println!("  |Δ|v∞||/|v∞|: {v_inf_cons:.2e}  (should be < 1e-12)");
    println!("  Turn angle:   {delta_deg:.2}°\n");

    // Leg 1: Venus → Jupiter.
    let r_sc1 = r_venus_t1;
    let v_sc1 = v_venus_t1 + v_inf_out;
    let leg1 = evaluate_mga_leg(
        r_sc1, v_sc1, eta_1, tof_1_days * 86_400.0,
        r_jupit_t2, v_jupit_t2, MU_SUN_M3S2,
    );
    match leg1 {
        Some(l) => {
            println!("Leg 1 (Venus → Jupiter):");
            println!("  DSM ΔV:     {:.2} m/s", l.dv_dsm_ms);
            println!("  Arrival v∞: {:.2} m/s\n", l.v_inf_arr_mps.norm());

            println!("=== Summary ===");
            println!("  Dep v∞:       {dep_vinf:.2} m/s");
            println!("  Leg 0 DSM ΔV: {:.2} m/s", leg0.dv_dsm_ms);
            println!("  Leg 1 DSM ΔV: {:.2} m/s", l.dv_dsm_ms);
            println!("  Total DSM ΔV: {:.2} m/s", leg0.dv_dsm_ms + l.dv_dsm_ms);
            println!("  Jupiter v∞:   {:.2} m/s", l.v_inf_arr_mps.norm());
            println!("  TOF total:    {:.1} days", tof_0_days + tof_1_days);

            // Plausibility assertions — not tight bounds, just sanity.
            assert!(leg0.dv_dsm_ms.is_finite() && leg0.dv_dsm_ms >= 0.0,
                "Leg 0 DSM ΔV is not finite/non-negative");
            assert!(l.dv_dsm_ms.is_finite() && l.dv_dsm_ms >= 0.0,
                "Leg 1 DSM ΔV is not finite/non-negative");
            assert!(v_inf_cons < 1e-10, "flyby_turn did not conserve |v∞|");
            println!("\nSanity checks: PASSED ✓");
        }
        None => {
            eprintln!("Leg 1 INFEASIBLE — try a different tof_1 or eta_1");
            std::process::exit(1);
        }
    }
}
