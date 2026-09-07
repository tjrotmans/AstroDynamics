//! VEEGA resonant-leg diagnostic (Phase 9x-iv follow-up).
//!
//! Question: is the ~731-day Earth-Earth resonant leg (the real Galileo
//! mission's actual choice) even ΔV-competitive in our MGA-1DSM model for
//! the real 1989 departure window — decoupled entirely from whether the
//! search (pruned or unpruned DE) can find it? Four independent optimizer
//! runs (2025/1989 windows × pruned/unpruned) all converged to
//! Earth-Earth legs in the 369-546 day range, never near 731 days. This
//! binary answers whether that's a search-difficulty problem or whether the
//! resonant leg genuinely isn't the ΔV-cheapest choice in this model.
//!
//! Method: fix legs 0-1 (Earth→Venus→Earth) at the actual winning chromosome
//! from `out/veega_1989_nopruning_scratch/mga_best_chromosome.csv` — this is
//! the REAL state the spacecraft has when it returns to Earth after the
//! first Venus+Earth flyby sequence, not a synthetic one — then sweep ONLY
//! leg 2's (Earth→Earth) time-of-flight across its full configured bound
//! [250, 900] days, holding eta at the value that minimises DSM cost for
//! each TOF (small local sweep). Plots DSM ΔV vs TOF: if there's a genuine
//! low-cost dip near 731 days, the resonant leg is real and the search
//! missed it; if the curve is flat or has its minimum elsewhere, the
//! optimizer's 369-546 day answers are legitimately better in this model.
//!
//! Run: `cargo run -p mission_planner --bin veega_resonance_diag --release`

use ephemeris::{Almanac, Body, Epoch};
use nalgebra::Vector3;
use trajectory_solver::{evaluate_mga_leg, flyby_turn, keplerian::MU_SUN_M3S2};

fn jd_to_epoch(jd: f64) -> Epoch {
    Epoch::from_unix_seconds((jd - 2_440_587.5) * 86_400.0)
}

fn body_state_vec3(almanac: &Almanac, body: Body, jd: f64) -> (Vector3<f64>, Vector3<f64>) {
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

    // ── Fixed legs 0-1, verbatim from the real winning 1989-window chromosome ──
    // (out/veega_1989_nopruning_scratch/mga_best_chromosome.csv run)
    let dep_jd_base: f64 = 2_447_817.5; // 1989-10-18T00:00:00 UTC
    let dep_offset:  f64 = -29.96952245604668;
    let dep_vinf:    f64 = 3425.306609588566;
    let theta:       f64 = 5.486965213057247;
    let phi:         f64 = -0.5790219322559174;
    let tof_0_days:  f64 = 190.7312874182714;
    let eta_0:       f64 = 0.05749896817017047;
    let tof_1_days:  f64 = 299.2888168543622;
    let eta_1:       f64 = 0.06495702684999453;
    let rp_norm_0:   f64 = 8.468510375523906;   // Venus flyby
    let beta_0:      f64 = -0.5592046449979599;
    let rp_norm_1:   f64 = 14.88114710422582;   // Earth flyby #1
    let beta_1:      f64 = 2.943989134595002;

    let mu_venus = 3.248_599e14_f64; // m3/s2 (Williams 2021)
    let r_venus  = 6_051_800.0_f64;  // m (IAU 2015)
    let mu_earth = 3.986_004_418e14_f64; // m3/s2 (IAU/DE440)
    let r_earth_body = 6_371_000.0_f64;  // m (mean radius)

    let dep_jd = dep_jd_base + dep_offset;
    let t1_jd  = dep_jd + tof_0_days; // Venus arrival
    let t2_jd  = t1_jd  + tof_1_days; // Earth arrival (start of leg 2)

    let (r_earth_dep, v_earth_dep) = body_state_vec3(&almanac, Body::Earth, dep_jd);
    let (r_venus_t1,  v_venus_t1)  = body_state_vec3(&almanac, Body::Venus, t1_jd);
    let (r_earth_t2,  v_earth_t2)  = body_state_vec3(&almanac, Body::Earth, t2_jd);

    let v_inf_vec = Vector3::new(
        dep_vinf * phi.cos() * theta.cos(),
        dep_vinf * phi.cos() * theta.sin(),
        dep_vinf * phi.sin(),
    );
    let r_sc0 = r_earth_dep;
    let v_sc0 = v_earth_dep + v_inf_vec;

    let leg0 = evaluate_mga_leg(r_sc0, v_sc0, eta_0, tof_0_days * 86_400.0, r_venus_t1, v_venus_t1, MU_SUN_M3S2)
        .expect("leg 0 infeasible — chromosome mismatch?");
    let v_inf_out_0 = flyby_turn(leg0.v_inf_arr_mps, rp_norm_0 * r_venus, beta_0, mu_venus);
    let r_sc1 = r_venus_t1;
    let v_sc1 = v_venus_t1 + v_inf_out_0;

    let leg1 = evaluate_mga_leg(r_sc1, v_sc1, eta_1, tof_1_days * 86_400.0, r_earth_t2, v_earth_t2, MU_SUN_M3S2)
        .expect("leg 1 infeasible — chromosome mismatch?");
    let v_inf_out_1 = flyby_turn(leg1.v_inf_arr_mps, rp_norm_1 * r_earth_body, beta_1, mu_earth);
    let r_sc2 = r_earth_t2;
    let v_sc2 = v_earth_t2 + v_inf_out_1;

    println!("=== VEEGA resonant-leg diagnostic: leg 2 (Earth->Earth) TOF sweep ===");
    println!("Fixed legs 0-1 DSM: leg0={:.1} m/s, leg1={:.1} m/s (from real 1989-window winner)", leg0.dv_dsm_ms, leg1.dv_dsm_ms);
    println!("Leg 2 starts at JD {t2_jd:.3} (Earth departure after first flyby)\n");

    // ── Sweep leg 2's TOF across its full configured bound [250, 900] days ────
    // For each TOF, sweep eta over a coarse grid and keep the best (min DSM).
    println!("{:>8}  {:>10}  {:>12}", "TOF[d]", "best_eta", "dv_dsm[m/s]");

    let mut rows: Vec<(f64, f64, f64)> = Vec::new(); // (tof, eta, dv_dsm)
    let tof_lo = 250.0_f64;
    let tof_hi = 900.0_f64;
    let tof_step = 2.0_f64;
    let mut tof = tof_lo;
    while tof <= tof_hi {
        let arr_jd = t2_jd + tof;
        let (r_next, v_next) = body_state_vec3(&almanac, Body::Earth, arr_jd);

        let mut best: Option<(f64, f64)> = None; // (eta, dv_dsm)
        let n_eta = 40;
        for i in 0..=n_eta {
            let eta = 0.02 + (0.96) * (i as f64 / n_eta as f64);
            if let Some(leg) = evaluate_mga_leg(r_sc2, v_sc2, eta, tof * 86_400.0, r_next, v_next, MU_SUN_M3S2) {
                if best.map(|(_, d)| leg.dv_dsm_ms < d).unwrap_or(true) {
                    best = Some((eta, leg.dv_dsm_ms));
                }
            }
        }

        if let Some((eta, dv)) = best {
            rows.push((tof, eta, dv));
        }
        tof += tof_step;
    }

    // Print every 10th row for readability, plus a highlighted band around 731d.
    for (i, &(tof, eta, dv)) in rows.iter().enumerate() {
        if i % 10 == 0 || (tof - 731.0).abs() < 15.0 {
            let marker = if (tof - 731.0).abs() < 15.0 { "  <-- near real Galileo EGA2 TOF" } else { "" };
            println!("{tof:>8.1}  {eta:>10.3}  {dv:>12.1}{marker}");
        }
    }

    // ── Summary: global minimum vs. the 731-day-band minimum vs. what the ────
    // optimizer actually found (545.6 days).
    let global_min = rows.iter().cloned().fold((0.0, 0.0, f64::MAX), |acc, r| if r.2 < acc.2 { r } else { acc });
    let band_min = rows.iter().filter(|r| (r.0 - 731.0).abs() < 30.0).cloned()
        .fold((0.0, 0.0, f64::MAX), |acc, r| if r.2 < acc.2 { r } else { acc });
    let near_optimizer = rows.iter().filter(|r| (r.0 - 545.6).abs() < 5.0).cloned()
        .fold((0.0, 0.0, f64::MAX), |acc, r| if r.2 < acc.2 { r } else { acc });

    println!("\n=== Summary ===");
    println!("Global minimum over [250,900]d:      TOF={:.1}d  eta={:.3}  dv_dsm={:.1} m/s", global_min.0, global_min.1, global_min.2);
    println!("Best within +/-30d of 731d (Galileo): TOF={:.1}d  eta={:.3}  dv_dsm={:.1} m/s", band_min.0, band_min.1, band_min.2);
    println!("Best near optimizer's 545.6d answer:  TOF={:.1}d  eta={:.3}  dv_dsm={:.1} m/s", near_optimizer.0, near_optimizer.1, near_optimizer.2);
    println!("\nDelta (731d-band minus global minimum): {:+.1} m/s", band_min.2 - global_min.2);

    // Write CSV for plotting.
    let out_dir = "out/veega_1989_nopruning_scratch";
    let _ = std::fs::create_dir_all(out_dir);
    let mut csv = vec!["tof_days,best_eta,dv_dsm_ms".to_string()];
    for (tof, eta, dv) in &rows {
        csv.push(format!("{tof},{eta},{dv}"));
    }
    let path = format!("{out_dir}/resonance_diag.csv");
    match std::fs::write(&path, csv.join("\n") + "\n") {
        Ok(()) => println!("\nWrote {path}"),
        Err(e) => eprintln!("Warning: could not write {path}: {e}"),
    }
}
