//! Earth → Bennu cruise trajectory design.
//!
//! 1. Lambert porkchop over (departure epoch, TOF) grid — parallel via rayon.
//! 2. Best grid point selected by minimum total ΔV.
//! 3. Local refinement: alternating golden-section search on departure epoch
//!    and TOF to converge to a local minimum ΔV around the best grid point.
//!
//! Outputs (written to `out/`):
//!   porkchop.csv      — full grid: dep_day, tof_day, dv_dep, dv_arr, dv_total  [km/s]
//!   best_transfer.csv — propagated transfer trajectory (r, v vs time)
//!
//! Run:  `cargo run -p autonomous_navigation --bin cruise_design --release`
//! Plot: `python plot/plot_cruise.py`

use autonomous_navigation::bennu_ephem::BennuEphem;
use orbital_models::constants::{MU_SUN, AU};
use orbital_math::lambert::lambert_min_dv;
use rayon::prelude::*;

// ── Earth circular orbit ──────────────────────────────────────────────────────

const R_E: f64 = AU;

// ── Porkchop grid ─────────────────────────────────────────────────────────────

const N_DEP: usize = 200;
const N_TOF: usize = 110;
/// Departure window anchored to the Horizons file coverage (2010-Apr-30 ≈ day 3771).
/// DEP_MIN_DAYS + TOF_MAX_DAYS must stay below the file's last epoch (~day 11082).
const DEP_MIN_DAYS: f64 = 3_900.0;   // 2010-Aug ≈ just after file start
const DEP_MAX_DAYS: f64 = 8_900.0;   // 2024-May — DEP+TOF_MAX < 9861 (file end)
/// TOF range: 200 – 950 days (Hohmann ~200 d minimum, practical 300-700 d)
const TOF_MIN_DAYS: f64 = 200.0;
const TOF_MAX_DAYS: f64 = 950.0;

// ── Local refinement ──────────────────────────────────────────────────────────

/// Golden-section tolerance [days] for both departure epoch and TOF.
const GS_TOL_DAYS: f64 = 0.1;
/// Search bracket half-width around best grid point [days].
const GS_BRACKET_DEP: f64 = 15.0;
const GS_BRACKET_TOF: f64 = 15.0;
/// Max alternating coordinate-descent iterations.
const REFINE_ITER: usize = 20;

// ── Types ─────────────────────────────────────────────────────────────────────

type V3 = [f64; 3];

// ── Ephemerides ───────────────────────────────────────────────────────────────

fn earth_rv(t_s: f64) -> (V3, V3) {
    let v_e  = (MU_SUN / R_E).sqrt();
    let om_e = v_e / R_E;
    let th   = om_e * t_s;
    (
        [R_E * th.cos(), R_E * th.sin(), 0.0],
        [v_e * (-th.sin()), v_e * th.cos(), 0.0],
    )
}

// ── RK4 2-body propagator ─────────────────────────────────────────────────────

fn rk4_step(r: V3, v: V3, dt: f64) -> (V3, V3) {
    let f = |rv: [f64; 6]| -> [f64; 6] {
        let rr = (rv[0]*rv[0] + rv[1]*rv[1] + rv[2]*rv[2]).sqrt();
        let a  = -MU_SUN / rr.powi(3);
        [rv[3], rv[4], rv[5], a*rv[0], a*rv[1], a*rv[2]]
    };
    let s0 = [r[0], r[1], r[2], v[0], v[1], v[2]];
    let k1 = f(s0);
    let k2 = f(arr_add(s0, arr_scale(k1, dt*0.5)));
    let k3 = f(arr_add(s0, arr_scale(k2, dt*0.5)));
    let k4 = f(arr_add(s0, arr_scale(k3, dt)));
    let s = arr_add(s0, arr_scale(arr_add(arr_add(arr_add(k1, arr_scale(k2,2.0)), arr_scale(k3,2.0)), k4), dt/6.0));
    ([s[0],s[1],s[2]], [s[3],s[4],s[5]])
}

fn propagate(r0: V3, v0: V3, total_t: f64, n_steps: usize) -> Vec<[f64; 7]> {
    let dt = total_t / n_steps as f64;
    let mut r = r0;
    let mut v = v0;
    let mut path = Vec::with_capacity(n_steps + 1);
    path.push([0.0, r[0], r[1], r[2], v[0], v[1], v[2]]);
    for k in 1..=n_steps {
        (r, v) = rk4_step(r, v, dt);
        path.push([k as f64 * dt, r[0], r[1], r[2], v[0], v[1], v[2]]);
    }
    path
}

// ── Golden-section minimum search ────────────────────────────────────────────

fn golden_min<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, tol: f64) -> f64 {
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    while (b - a).abs() > tol {
        if f(c) < f(d) { b = d; } else { a = c; }
        c = b - phi * (b - a);
        d = a + phi * (b - a);
    }
    (a + b) / 2.0
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    std::fs::create_dir_all("out/cruise").unwrap();

    println!("=== Earth → Bennu Cruise Design ===");
    println!("  Loading Bennu ephemeris...");
    let ephem = BennuEphem::load("horizons_results_bennu.txt");

    let dep_days: Vec<f64> = (0..N_DEP)
        .map(|j| DEP_MIN_DAYS + (DEP_MAX_DAYS - DEP_MIN_DAYS) * j as f64 / (N_DEP - 1) as f64)
        .collect();
    let tof_days: Vec<f64> = (0..N_TOF)
        .map(|i| TOF_MIN_DAYS + (TOF_MAX_DAYS - TOF_MIN_DAYS) * i as f64 / (N_TOF - 1) as f64)
        .collect();

    println!("  Grid: {}×{} = {} Lambert calls", N_DEP, N_TOF, N_DEP * N_TOF * 2);

    // Parallel porkchop — ephem is Sync so shared across rayon threads via &ref
    let rows: Vec<(f64, f64, f64, f64, f64)> = dep_days.par_iter().flat_map(|&td_d| {
        let td_s  = td_d * 86400.0;
        let (r_e, v_e) = earth_rv(td_s);
        let ephem_ref  = &ephem;
        tof_days.iter().filter_map(move |&tf_d| {
            let tf_s = tf_d * 86400.0;
            let (r_b, v_b) = ephem_ref.query(td_s + tf_s);
            lambert_min_dv(r_e, r_b, tf_s, MU_SUN, v_e, v_b)
                .map(|(v1, v2)| {
                    let dv1 = norm(sub(v1, v_e));
                    let dv2 = norm(sub(v2, v_b));
                    (td_d, tf_d, dv1/1e3, dv2/1e3, (dv1+dv2)/1e3)
                })
        }).collect::<Vec<_>>()
    }).collect();

    println!("  Valid solutions: {}/{}", rows.len(), N_DEP * N_TOF);

    // Best grid point
    let best = rows.iter()
        .min_by(|a, b| a.4.partial_cmp(&b.4).unwrap())
        .expect("no Lambert solutions found — check ephemeris");

    println!("\n── Best grid solution ──");
    println!("  Departure : J2000 + {:.0} d  ({:.2} yr)", best.0, best.0/365.25);
    println!("  TOF       : {:.0} d", best.1);
    println!("  ΔV_dep    : {:.3} km/s", best.2);
    println!("  ΔV_arr    : {:.3} km/s", best.3);
    println!("  ΔV_total  : {:.3} km/s", best.4);

    // ── Local refinement: alternating golden section on dep and tof ──────────
    let dv_fn = |dep_d: f64, tof_d: f64| -> f64 {
        let dep_s = dep_d * 86400.0;
        let tof_s = tof_d * 86400.0;
        let (r_e, v_e) = earth_rv(dep_s);
        let (r_b, v_b) = ephem.query(dep_s + tof_s);
        lambert_min_dv(r_e, r_b, tof_s, MU_SUN, v_e, v_b)
            .map(|(v1, v2)| (norm(sub(v1, v_e)) + norm(sub(v2, v_b))) / 1e3)
            .unwrap_or(99.0)
    };

    let mut rdep = best.0;
    let mut rtof = best.1;

    for _ in 0..REFINE_ITER {
        let prev_dep = rdep;
        let cap_tof  = rtof;
        rdep = golden_min(
            |d| dv_fn(d, cap_tof),
            (rdep - GS_BRACKET_DEP).max(0.0),
            rdep + GS_BRACKET_DEP,
            GS_TOL_DAYS,
        );
        let cap_dep = rdep;
        rtof = golden_min(
            |t| dv_fn(cap_dep, t),
            (rtof - GS_BRACKET_TOF).max(TOF_MIN_DAYS),
            rtof + GS_BRACKET_TOF,
            GS_TOL_DAYS,
        );
        if (rdep - prev_dep).abs() < 0.01 { break; }
    }

    let (rdv1, rdv2, rdvt) = {
        let dep_s = rdep * 86400.0;
        let tof_s = rtof * 86400.0;
        let (r_e, v_e) = earth_rv(dep_s);
        let (r_b, v_b) = ephem.query(dep_s + tof_s);
        let (v1, v2)   = lambert_min_dv(r_e, r_b, tof_s, MU_SUN, v_e, v_b).unwrap();
        let d1 = norm(sub(v1, v_e));
        let d2 = norm(sub(v2, v_b));
        (d1/1e3, d2/1e3, (d1+d2)/1e3)
    };

    println!("\n── Refined solution (golden-section) ──");
    println!("  Departure : J2000 + {:.1} d  ({:.2} yr)", rdep, rdep/365.25);
    println!("  TOF       : {:.1} d  ({:.2} yr)", rtof, rtof/365.25);
    println!("  Arrival   : J2000 + {:.0} d", rdep + rtof);
    println!("  ΔV_dep    : {:.4} km/s   (C3 = {:.3} km²/s²)",
             rdv1, rdv1 * rdv1);
    println!("  ΔV_arr    : {:.4} km/s", rdv2);
    println!("  ΔV_total  : {:.4} km/s", rdvt);

    // ── Save porkchop CSV ─────────────────────────────────────────────────────
    {
        use std::fmt::Write as W;
        let mut out = String::with_capacity(rows.len() * 50);
        writeln!(out, "dep_day,tof_day,dv_dep_kms,dv_arr_kms,dv_total_kms").unwrap();
        for &(td, tf, d1, d2, tot) in &rows {
            writeln!(out, "{:.2},{:.2},{:.4},{:.4},{:.4}", td, tf, d1, d2, tot).unwrap();
        }
        std::fs::write("out/cruise/porkchop.csv", &out).unwrap();
        println!("\n  Saved out/porkchop.csv ({} rows)", rows.len());
    }

    // ── Propagate best transfer trajectory ───────────────────────────────────
    let dep_s = rdep * 86400.0;
    let tof_s = rtof * 86400.0;
    let (r_e_dep, v_e_dep) = earth_rv(dep_s);
    let (r_b_arr, v_b_arr) = ephem.query(dep_s + tof_s);
    let (v1_best, v2_best) = lambert_min_dv(r_e_dep, r_b_arr, tof_s, MU_SUN, v_e_dep, v_b_arr)
        .expect("refined solution Lambert failed");

    let n_prop = 600;
    let traj   = propagate(r_e_dep, v1_best, tof_s, n_prop);

    // Also sample Bennu and Earth tracks over the transfer
    {
        use std::fmt::Write as W;
        let mut out = String::with_capacity((n_prop + 1) * 100);
        writeln!(out,
            "time_s,sc_x_m,sc_y_m,sc_z_m,sc_vx,sc_vy,sc_vz,\
             bennu_x_m,bennu_y_m,bennu_z_m,earth_x_m,earth_y_m,earth_z_m").unwrap();
        for row in &traj {
            let t_abs = dep_s + row[0];
            let (rb, _) = ephem.query(t_abs);
            let (re, _) = earth_rv(t_abs);
            writeln!(out,
                "{:.1},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},\
                 {:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e}",
                row[0],
                row[1], row[2], row[3], row[4], row[5], row[6],
                rb[0], rb[1], rb[2],
                re[0], re[1], re[2]).unwrap();
        }
        std::fs::write("out/cruise/best_transfer.csv", &out).unwrap();
        println!("  Saved out/best_transfer.csv ({} rows)", traj.len());
    }

    // Save refined solution summary for the plotter
    {
        use std::fmt::Write as W;
        let mut out = String::new();
        writeln!(out, "dep_day,tof_day,dv_dep_kms,dv_arr_kms,dv_total_kms,dep_year").unwrap();
        writeln!(out, "{:.2},{:.2},{:.4},{:.4},{:.4},{:.3}",
                 rdep, rtof, rdv1, rdv2, rdvt, rdep/365.25).unwrap();
        std::fs::write("out/cruise/best_solution.csv", &out).unwrap();
        writeln!(out).ok();

        // Departure/arrival position vectors for plotter arrows
        let mut meta = String::new();
        writeln!(meta, "dep_x_au,dep_y_au,arr_x_au,arr_y_au,\
                        dv_dep_x,dv_dep_y,dv_arr_x,dv_arr_y").unwrap();
        let dv_dep_vec = sub(v1_best, v_e_dep);
        let dv_arr_vec = sub(v2_best, v_b_arr);
        writeln!(meta, "{:.6},{:.6},{:.6},{:.6},{:.4e},{:.4e},{:.4e},{:.4e}",
                 r_e_dep[0]/AU, r_e_dep[1]/AU,
                 r_b_arr[0]/AU, r_b_arr[1]/AU,
                 dv_dep_vec[0], dv_dep_vec[1],
                 dv_arr_vec[0], dv_arr_vec[1]).unwrap();
        std::fs::write("out/cruise/transfer_meta.csv", &meta).unwrap();
        println!("  Saved out/transfer_meta.csv");
    }

    println!("\nTo visualise: python plot/plot_cruise.py");
    println!("Outputs in out/cruise/");
}

// ── Vec-3 helpers ─────────────────────────────────────────────────────────────

#[inline] fn dot(a: V3, b: V3) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
#[inline] fn norm(a: V3) -> f64 { dot(a, a).sqrt() }
#[inline] fn sub(a: V3, b: V3) -> V3 { [a[0]-b[0], a[1]-b[1], a[2]-b[2] ]}

type Arr6 = [f64; 6];
#[inline] fn arr_add(a: Arr6, b: Arr6) -> Arr6 {
    [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4], a[5]+b[5]]
}
#[inline] fn arr_scale(a: Arr6, s: f64) -> Arr6 {
    [a[0]*s, a[1]*s, a[2]*s, a[3]*s, a[4]*s, a[5]*s]
}
