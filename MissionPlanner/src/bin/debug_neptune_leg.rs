//! Scratch diagnostic: checking whether the earth_neptune_auto
//! all-infeasible finding is a real energy/TOF constraint (Hohmann-class
//! Jupiter->Neptune transfer needs ~13,500 days; even a Voyager-2-class
//! gravity-assist-boosted transfer needs ~3650 days) rather than a solver bug.

use ephemeris::{Almanac, Body, Epoch};
use nalgebra::Vector3;
use trajectory_solver::{evaluate_mga_leg, keplerian::MU_SUN_M3S2};

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
    (
        Vector3::new(r[0] * 1e3, r[1] * 1e3, r[2] * 1e3),
        Vector3::new(v[0] * 1e3, v[1] * 1e3, v[2] * 1e3),
    )
}

fn main() {
    let almanac = Almanac::new("kernels/de440s.bsp")
        .or_else(|_| Almanac::new("../kernels/de440s.bsp"))
        .expect("de440s.bsp not found");

    let dep_jd_base: f64 = 2_463_749.5;
    let thetas: Vec<f64> = (0..8).map(|i| i as f64 * std::f64::consts::TAU / 8.0).collect();
    let phis: Vec<f64> = [-0.8, 0.0, 0.8].to_vec();

    println!("=== Leg 1: Jupiter -> Neptune, sweeping TOF up to 8000 days ===");
    let jup_arr_jd = dep_jd_base + 900.0;
    let (r_jup, v_jup) = body_state_vec3(&almanac, Body::Jupiter, jup_arr_jd);
    for &tof_days in &[2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 8000.0] {
        let mut total = 0usize;
        let mut ok = 0usize;
        for &eta in &[0.05, 0.2, 0.4] {  // low eta -> most of TOF is the Lambert leg
            for &vinf_out in &[5000.0, 15000.0, 30000.0] {
                for &theta in &thetas {
                    for &phi in &phis {
                        let v_inf_vec = Vector3::new(
                            vinf_out * phi.cos() * theta.cos(),
                            vinf_out * phi.cos() * theta.sin(),
                            vinf_out * phi.sin(),
                        );
                        let r_sc0 = r_jup;
                        let v_sc0 = v_jup + v_inf_vec;
                        let t2_jd = jup_arr_jd + tof_days;
                        let (r_nep, v_nep) = body_state_vec3(&almanac, Body::Neptune, t2_jd);
                        let leg = evaluate_mga_leg(
                            r_sc0, v_sc0, eta, tof_days * 86_400.0,
                            r_nep, v_nep, MU_SUN_M3S2,
                        );
                        total += 1;
                        if leg.is_some() { ok += 1; }
                    }
                }
            }
        }
        println!("  tof={tof_days:>6.0}d  feasible: {ok}/{total} ({:.1}%)", 100.0 * ok as f64 / total as f64);
    }

    println!("\n=== Leg 0: Earth -> Jupiter, sweeping TOF (low eta) ===");
    for &tof_days in &[500.0, 800.0, 1200.0, 1800.0, 2500.0] {
        let mut total = 0usize;
        let mut ok = 0usize;
        for &dep_offset in &[-900.0, -300.0, 0.0, 300.0, 900.0] {
            let dep_jd = dep_jd_base + dep_offset;
            let (r_earth, v_earth) = body_state_vec3(&almanac, Body::Earth, dep_jd);
            for &eta in &[0.05, 0.2, 0.4] {
                for &dep_vinf in &[5000.0, 10000.0] {
                    for &theta in &thetas {
                        for &phi in &phis {
                            let v_inf_vec = Vector3::new(
                                dep_vinf * phi.cos() * theta.cos(),
                                dep_vinf * phi.cos() * theta.sin(),
                                dep_vinf * phi.sin(),
                            );
                            let r_sc0 = r_earth;
                            let v_sc0 = v_earth + v_inf_vec;
                            let t1_jd = dep_jd + tof_days;
                            let (r_jup, v_jup) = body_state_vec3(&almanac, Body::Jupiter, t1_jd);
                            let leg = evaluate_mga_leg(
                                r_sc0, v_sc0, eta, tof_days * 86_400.0,
                                r_jup, v_jup, MU_SUN_M3S2,
                            );
                            total += 1;
                            if leg.is_some() { ok += 1; }
                        }
                    }
                }
            }
        }
        println!("  tof={tof_days:>6.0}d  feasible: {ok}/{total} ({:.1}%)", 100.0 * ok as f64 / total as f64);
    }
}
