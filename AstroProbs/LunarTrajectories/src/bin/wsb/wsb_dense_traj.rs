//! wsb_dense_traj — re-propagate the 10 diverse WSB solutions with dense output.
//!
//! Uses the adaptive-step integrator in sparse mode (OutputType::Sparse): every
//! accepted internal step is logged.  Near lunar periapsis the step is naturally
//! small (high forces), giving smooth arcs without Moon-crossing artefacts.  A
//! maximum step cap (H_MAX) keeps the cruise segment fine enough for plotting.
//!
//! Usage:
//!   cargo run -p lunar_trajectories --bin wsb_dense_traj --release

use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;
use std::fs;

use lunar_trajectories::crtbp::CrtbpParams;
use lunar_trajectories::propagator::{propagate_bcr4bp_raw, Bcr4bpParams};
use lunar_trajectories::transfers::{
    lunar_hill_radius, detect_capture, tli_injection_ic, tli_dv, min_loi_dv,
};

// ── Propagation parameters ────────────────────────────────────────────────────

/// Maximum adaptive step [nd].  0.02 nd ≈ 1.7 h keeps cruise segments smooth.
/// Inside the Hill sphere the integrator automatically uses much smaller steps.
const H_MAX: f64 = 0.02;
const RTOL:  f64 = 1e-8;   // match MC-search tolerance so chaotic trajectories reproduce
const ATOL:  f64 = 1e-10;

const T_PROP:  f64 = 20.0 * PI;   // same budget as wsb_refine
const R_PARK:  f64 = (6_371.0 + 378.0) / 384_400.0;

const LUNAR_PERIOD_ND: f64 = 2.0 * PI;
const MIN_CAPTURE_TIME: f64 = 0.15;

const OUT_DIR: &str = "out/wsb";
const OUT_CSV: &str = "out/wsb/dense_traj.csv";

// ── The 10 selected diverse solutions ────────────────────────────────────────
// Columns: (run_id, seed_id, theta_deg, theta_sun_deg, r_apogee_nd)
// Extracted from all_refinements.csv — same ICs as the MC refinement used.

// theta_deg/theta_sun_deg/r_apogee_nd are the actual ICs used in the MC search
// (verified against the initial positions in all_refinements.csv).
// dtheta_deg/dsun_deg in that CSV are deltas from the GA seed, NOT from these values.
const SOLUTIONS: &[(u64, u64, f64, f64, f64)] = &[
    (1300001, 13, 205.5490,  27.0190, 3.9000),
    (1800001, 18,  43.9970, 262.2990, 2.8000),
    ( 100001,  1,  76.8740, 115.8740, 3.5000),
    (2100001, 21, 129.3260, 282.8810, 2.8000),
    ( 400001,  4, 297.5540,  35.3950, 3.9000),
    (1400001, 14, 266.5510, 359.2860, 4.6000),
    ( 600001,  6, 109.3690, 224.0100, 4.3000),
    (1200001, 12, 172.6140, 209.5650, 4.1000),
    (1700001, 17, 150.7540, 301.5090, 4.6000),
    (2300001, 23,  24.6570,  67.0480, 3.5000),
];

// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    let params   = CrtbpParams::earth_moon();
    let mu       = params.mu;
    let v_km_s   = params.v_star / 1e3;
    let r_hill   = lunar_hill_radius(mu);
    let moon_x   = 1.0 - mu;

    fs::create_dir_all(OUT_DIR).expect("cannot create out/wsb");

    let mut csv = String::from(
        "time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,\
         run_id,seed_id,dtheta_deg,dsun_deg,dr_apogee_nd,\
         theta_deg,theta_sun_deg,r_apogee_nd,\
         dv_kms,entered_hill,max_hill_dwell_nd,est_capture_orbits\n"
    );

    for &(run_id, seed_id, theta_deg, theta_sun_deg, r_apogee_nd) in SOLUTIONS {
        eprint!("  run_id={run_id}  θ={theta_deg:.3}°  θ_sun={theta_sun_deg:.3}°  r_apo={r_apogee_nd:.2} … ");

        let theta = theta_deg.to_radians();
        let t_sun = theta_sun_deg.to_radians();

        let Some(ic) = tli_injection_ic(mu, R_PARK, r_apogee_nd, theta) else {
            eprintln!("IC failed — skipped");
            continue;
        };

        let bcr  = Bcr4bpParams::earth_moon_sun(t_sun);
        let traj = propagate_bcr4bp_raw(mu, bcr, ic, T_PROP, H_MAX, RTOL, ATOL);

        let cap         = detect_capture(&traj, mu, MIN_CAPTURE_TIME);
        let est_orbits  = cap.max_capture_interval / LUNAR_PERIOD_ND;
        let entered     = cap.n_entries > 0;
        let hill_dwell  = cap.max_capture_interval;

        // ΔV — same calculation as wsb_refine
        let dv_tli_nd   = tli_dv(mu, R_PARK, r_apogee_nd);
        let hill_entry_t = traj.iter()
            .find(|s| {
                let dx = s.x - moon_x;
                (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
            })
            .map(|s| s.time);
        let dv_loi_kms  = min_loi_dv(&traj, mu)
            .map(|dv| dv * v_km_s)
            .unwrap_or(0.0);
        let dv_kms = dv_tli_nd * v_km_s + dv_loi_kms;

        // Truncate: Hill entry time + 4 lunar periods (same as save_mc_traj_csv)
        let t_stop = hill_entry_t
            .map(|t| t + 4.0 * LUNAR_PERIOD_ND)
            .unwrap_or(T_PROP);

        let mut n_pts = 0usize;
        for s in traj.iter().take_while(|s| s.time <= t_stop) {
            writeln!(csv,
                "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},\
                 {},{},0.0,0.0,0.0,\
                 {:.4},{:.4},{:.4},\
                 {:.5},{},{:.4},{:.3}",
                s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz,
                run_id, seed_id,
                theta_deg, theta_sun_deg, r_apogee_nd,
                dv_kms,
                if entered { 1 } else { 0 },
                hill_dwell, est_orbits,
            ).unwrap();
            n_pts += 1;
        }

        // Metadata-only row (NaN trajectory) — used by the plot script for
        // groupby aggregation of theta_sun_deg / est_capture_orbits per run_id.
        writeln!(csv,
            "NaN,NaN,NaN,NaN,NaN,NaN,NaN,\
             {},{},0.0,0.0,0.0,\
             {:.4},{:.4},{:.4},\
             {:.5},{},{:.4},{:.3}",
            run_id, seed_id,
            theta_deg, theta_sun_deg, r_apogee_nd,
            dv_kms,
            if entered { 1 } else { 0 },
            hill_dwell, est_orbits,
        ).unwrap();

        eprintln!("{n_pts} pts  orbits={est_orbits:.2}  ΔV={dv_kms:.4} km/s");
    }

    fs::write(OUT_CSV, &csv).expect("dense_traj csv write failed");
    eprintln!("Saved {OUT_CSV}");
}
