//! wsb_basin — capture basin sweep across multiple r_apogee values.
//!
//! Sweeps a 3-D grid of (r_apogee, θ_inject, θ_sun) and classifies each
//! trajectory as "captured", "moon_crash", "capturable", or "escaped".
//!
//! "Capturable" = entered Hill sphere but est_orbits < MIN_CAPTURE_ORBITS,
//! yet min LOI ΔV inside Hill < LOI_CAPTURABLE_KMS.  These are the basin-edge
//! trajectories that a timely burn could have saved.
//!
//! Output columns:
//!   r_apogee_nd, theta_deg, theta_sun_deg, outcome,
//!   est_capture_orbits, min_loi_kms
//!
//! Usage:
//!   cargo run -p lunar_trajectories --bin wsb_basin --release
//!
//! Output: out/wsb/basin_sweep.csv

use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::fs;
use std::io::{BufWriter, Write};

use rayon::prelude::*;

use lunar_trajectories::crtbp::CrtbpParams;
use lunar_trajectories::propagator::{propagate_bcr4bp, Bcr4bpParams};
use lunar_trajectories::transfers::{
    lunar_hill_radius, detect_capture, tli_injection_ic, min_loi_dv,
};

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                         USER CONFIGURATION                                  ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

/// Injection angle grid [deg] — 1° spacing gives a sharp basin map.
/// At 360×360×N_APO the total is ~130k propagations per apogee slice.
const THETA_STEPS: usize = 1500;

/// Sun phase grid [deg].
const SUN_STEPS: usize = 1500;

/// Apogee distances to sweep [nd].  Add/remove values freely.
/// Single value example: &[3.9]
/// Multi-slice example:  &[2.8, 3.2, 3.5, 3.7, 3.9, 4.1, 4.3, 4.6]
const APO_SLICES: &[f64] = &[3.9];

/// Propagation time [nd].  8π ≈ 112 days — enough to confirm Hill capture
/// without the full 220-day search budget.
const T_PROP: f64 = 8.0 * PI;

/// Integrator — relaxed but sufficient to resolve the WSB boundary.
const LOG_DT: f64 = 0.01;
const RTOL:   f64 = 1e-8;
const ATOL:   f64 = 1e-10;

/// Minimum Hill sphere dwell to count as a capture entry [nd].
const MIN_CAPTURE_TIME: f64 = 0.15;

/// Minimum estimated orbits to label a trajectory "captured".
const MIN_CAPTURE_ORBITS: f64 = 0.5;

/// Trajectories with min LOI ΔV < this threshold inside the Hill sphere
/// are labelled "capturable" rather than "escaped" or "moon_crash".
const LOI_CAPTURABLE_KMS: f64 = 0.4;

const LUNAR_PERIOD_ND: f64 = 2.0 * PI;

/// Earth parking orbit [nd].  378 km altitude (Artemis II perigee).
const R_PARK: f64 = (6_371.0 + 378.0) / 384_400.0;

const OUT_DIR: &str = "out/wsb";

// ════════════════════════════════════════════════════════════════════════════════

fn main() {
    let params    = CrtbpParams::earth_moon();
    let mu        = params.mu;
    let moon_x    = 1.0 - mu;
    let r_hill    = lunar_hill_radius(mu);
    let l_km      = params.l_star / 1e3;
    let v_km_s    = params.v_star / 1e3;
    let r_moon_nd = 1_737.4_f64 / l_km;

    fs::create_dir_all(OUT_DIR).unwrap();

    // Build grids
    let theta_grid: Vec<f64> = (0..THETA_STEPS)
        .map(|i| 360.0 * i as f64 / THETA_STEPS as f64)
        .collect();
    let sun_grid: Vec<f64> = (0..SUN_STEPS)
        .map(|i| 360.0 * i as f64 / SUN_STEPS as f64)
        .collect();

    let total_per_slice = THETA_STEPS * SUN_STEPS;
    let total           = APO_SLICES.len() * total_per_slice;

    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!("║        WSB Capture Basin Sweep                       ║");
    eprintln!("╚══════════════════════════════════════════════════════╝");
    eprintln!("  μ        = {:.6}", mu);
    eprintln!("  r_Hill   = {:.5} nd  = {:.0} km", r_hill, r_hill * l_km);
    eprintln!("  Grid     : {} apogee × {}θ × {}θ_sun = {} trajectories",
        APO_SLICES.len(), THETA_STEPS, SUN_STEPS, total);
    eprintln!("  T_prop   = {:.1} nd  ({:.0} days)",
        T_PROP, T_PROP * params.t_star / 86_400.0);
    eprintln!("  Capturable threshold: LOI < {LOI_CAPTURABLE_KMS} km/s inside Hill");
    eprintln!();

    let csv_path = format!("{OUT_DIR}/basin_sweep.csv");
    let file     = fs::File::create(&csv_path).expect("cannot create basin_sweep.csv");
    let mut out  = BufWriter::new(file);
    writeln!(out, "r_apogee_nd,theta_deg,theta_sun_deg,outcome,est_capture_orbits,min_loi_kms")
        .unwrap();

    let n_captured   = AtomicUsize::new(0);
    let n_capturable = AtomicUsize::new(0);
    let n_crash      = AtomicUsize::new(0);

    for &r_apogee in APO_SLICES {
        eprintln!("  r_apogee = {r_apogee:.2} nd  ({:.0} km) …",
            r_apogee * 384_400.0);

        // Parallelise over theta; collect rows as strings to preserve order.
        let rows: Vec<String> = theta_grid
            .par_iter()
            .flat_map(|&theta_deg| {
                let theta = theta_deg.to_radians();
                let Some(ic) = tli_injection_ic(mu, R_PARK, r_apogee, theta)
                    else { return vec![]; };

                sun_grid.iter().map(|&sun_deg| {
                    let theta_sun = sun_deg.to_radians();
                    let bcr  = Bcr4bpParams::earth_moon_sun(theta_sun);
                    let traj = propagate_bcr4bp(mu, bcr, ic, T_PROP, LOG_DT, RTOL, ATOL);

                    // Moon surface check
                    let min_moon_dist = traj.iter().map(|s| {
                        let dx = s.x - moon_x;
                        (dx * dx + s.y * s.y + s.z * s.z).sqrt()
                    }).fold(f64::MAX, f64::min);
                    let moon_crash = min_moon_dist < r_moon_nd;

                    // Capture metrics
                    let cap        = detect_capture(&traj, mu, MIN_CAPTURE_TIME);
                    let est_orbits = cap.max_capture_interval / LUNAR_PERIOD_ND;

                    // LOI ΔV — only meaningful inside Hill sphere
                    let min_loi_kms = match min_loi_dv(&traj, mu) {
                        Some(dv) => dv * v_km_s,
                        None     => f64::NAN,
                    };

                    let outcome = if moon_crash {
                        n_crash.fetch_add(1, Ordering::Relaxed);
                        "moon_crash"
                    } else if est_orbits >= MIN_CAPTURE_ORBITS && cap.n_entries > 0 {
                        n_captured.fetch_add(1, Ordering::Relaxed);
                        "captured"
                    } else if cap.n_entries > 0
                        && min_loi_kms.is_finite()
                        && min_loi_kms < LOI_CAPTURABLE_KMS
                    {
                        // Entered Hill sphere but didn't stay — however a small
                        // LOI burn would have captured it.
                        n_capturable.fetch_add(1, Ordering::Relaxed);
                        "capturable"
                    } else {
                        "escaped"
                    };

                    let loi_str = if min_loi_kms.is_finite() {
                        format!("{:.4}", min_loi_kms)
                    } else {
                        "NaN".to_string()
                    };

                    format!("{r_apogee:.4},{theta_deg:.3},{sun_deg:.3},{outcome},{est_orbits:.4},{loi_str}\n")
                }).collect()
            })
            .collect();

        for row in &rows { out.write_all(row.as_bytes()).unwrap(); }

        let nc  = n_captured.load(Ordering::Relaxed);
        let nca = n_capturable.load(Ordering::Relaxed);
        let ncr = n_crash.load(Ordering::Relaxed);
        let ne  = total_per_slice - nc - nca - ncr;
        eprintln!("    captured={nc}  capturable={nca}  crash={ncr}  escaped={ne}");
    }

    out.flush().unwrap();

    let nc  = n_captured.load(Ordering::Relaxed);
    let nca = n_capturable.load(Ordering::Relaxed);
    let ncr = n_crash.load(Ordering::Relaxed);
    eprintln!();
    eprintln!("  ── Results ({total} total) ─────────────────────────────");
    eprintln!("  Captured   : {nc}  ({:.1}%)",   100.0 * nc  as f64 / total as f64);
    eprintln!("  Capturable : {nca}  ({:.1}%)",  100.0 * nca as f64 / total as f64);
    eprintln!("  Moon crash : {ncr}  ({:.1}%)",  100.0 * ncr as f64 / total as f64);
    eprintln!("  Escaped    : {}  ({:.1}%)",
        total - nc - nca - ncr,
        100.0 * (total - nc - nca - ncr) as f64 / total as f64);
    eprintln!();
    eprintln!("  Saved {csv_path}");
    eprintln!("  Plot: python plot/plot_wsb_basin_anim.py");
}