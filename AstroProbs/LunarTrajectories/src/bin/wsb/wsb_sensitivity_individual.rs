//! wsb_sensitivity_individual — one-at-a-time (OAT) grid sensitivity around a single
//! refined WSB hit.
//!
//! Varies each of the four error parameters independently while holding the others
//! at their nominal values.  Each parameter is swept uniformly from −1σ to +1σ in
//! N_PER_PARAM steps.  The nominal trajectory (all perturbations zero) is always
//! included as run_id = 0.
//!
//! Groups by run_id:
//!   0                      → nominal
//!   1  …  N_PER_PARAM      → ΔV magnitude sweep
//!   N_PER_PARAM+1 … 2N     → pointing (pitch) sweep
//!   2N+1 …          3N     → burn-timing (θ_inject) sweep
//!   3N+1 …          4N     → launch-window (θ_sun) sweep
//!
//! Output format is identical to sensitivity_ensemble.csv so the same Python
//! plotting scripts (plot_wsb_sensitivity_anim.py, plot_wsb_sensitivity_png.py)
//! work without modification.
//!
//! Usage:
//!   cargo run -p lunar_trajectories --bin wsb_sensitivity_individual --release -- --hifi
//!   cargo run -p lunar_trajectories --bin wsb_sensitivity_individual --release -- \
//!       --theta 205.549 --r-apogee 3.9 --theta-sun 27.019
//!   cargo run -p lunar_trajectories --bin wsb_sensitivity_individual --release -- \
//!       --hifi --tag oat
//!
//! Outputs (all in out/wsb/):
//!   sensitivity_ensemble[_TAG].csv   — trajectories + perturbation columns
//!   sensitivity_summary[_TAG].txt    — human-readable breakdown

use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;
use std::fs;

use lunar_trajectories::crtbp::CrtbpParams;
use lunar_trajectories::propagator::{propagate_bcr4bp, propagate_3d_stm_full,
                                      Bcr4bpParams, Step3d};
use lunar_trajectories::transfers::{lunar_hill_radius, detect_capture, tli_injection_ic};

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                         USER CONFIGURATION                                  ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

/// Number of steps per parameter swept from −σ to +σ (exclusive of the nominal).
/// Total trajectories = 4 × N_PER_PARAM + 1 (nominal).
const N_PER_PARAM: usize = 50;

/// Perturbation sigma values — identical to wsb_sensitivity.rs for direct comparison.
/// TLI ΔV magnitude error (fractional, 1-sigma).  1e-3 = ±0.1% thrust error.
const DV_MAG_SIGMA: f64 = 1e-3;
/// TLI pointing error (radians, 1-sigma) — pitch (in-plane) swept here; yaw held at 0.
/// 3.5e-3 rad ≈ ±0.2°.
const DV_DIR_SIGMA: f64 = 1.75e-3;
/// Burn timing error (degrees, 1-sigma).  0.2° ≈ 3 s on a ~92-min LEO parking orbit
/// (ω ≈ 0.065°/s).  Represents ground-commanded burn execution uncertainty.
const THETA_SIGMA_DEG: f64 = 0.2;
/// Launch-window error (degrees, 1-sigma).  Sun moves at 360°/29.53 days ≈ 0.51°/hr,
/// so 3° ≈ ±5.9 hr of launch window — a realistic operational window width.
const THETA_SUN_SIGMA_DEG: f64 = 3.0;

/// Propagation time budget [nd].
const T_PROP: f64 = 35.0 * PI;

/// Trajectory logging step [nd].  ~7 h for smooth animation.
const LOG_DT: f64 = 0.002;
const RTOL:   f64 = 1e-10;
const ATOL:   f64 = 1e-12;

const MIN_CAPTURE_TIME: f64 = 0.15;
const LUNAR_PERIOD_ND:  f64 = 2.0 * PI;
const R_MOON_KM:        f64 = 1_737.4;
const _T_STAR:          f64 = 375_700.0;

const OUT_DIR:     &str = "out/wsb";
const SUMMARY_CSV: &str = "out/wsb/refine_summary.csv";
const HIFI_CSV:    &str = "out/wsb/solution_hifi.csv";

// ════════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
#[allow(dead_code)]
struct HitSeed {
    hit_id:        usize,
    seed_id:       usize,
    theta_deg:     f64,
    theta_sun_deg: f64,
    r_apogee_nd:   f64,
    est_orbits:    f64,
    dtheta_deg:    f64,
    dsun_deg:      f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Outcome { Captured, MoonCrash, Escaped }

impl Outcome {
    fn label(self) -> &'static str {
        match self {
            Outcome::Captured  => "captured",
            Outcome::MoonCrash => "moon_crash",
            Outcome::Escaped   => "escaped",
        }
    }
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 { return vec![]; }
    if n == 1 { return vec![start]; }
    (0..n).map(|i| start + (end - start) * i as f64 / (n - 1) as f64).collect()
}

// ════════════════════════════════════════════════════════════════════════════════

fn main() {
    let params    = CrtbpParams::earth_moon();
    let mu        = params.mu;
    let l_km      = params.l_star / 1e3;
    let _v_km_s   = params.v_star / 1e3;
    let moon_x    = 1.0 - mu;
    let r_hill    = lunar_hill_radius(mu);
    let r_moon_nd = R_MOON_KM / l_km;
    let mu_earth  = 1.0 - mu;

    fs::create_dir_all(OUT_DIR).unwrap();

    // ── CLI (identical to wsb_sensitivity) ────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let hit_id: usize = args.iter()
        .position(|a| a == "--hit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let tag: String = args.iter()
        .position(|a| a == "--tag")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_default();
    let tag_suffix = if tag.is_empty() { String::new() } else { format!("_{tag}") };
    let use_hifi: bool = args.iter().any(|a| a == "--hifi");
    let ic_str: Option<String> = args.iter()
        .position(|a| a == "--ic")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let theta_sun_override: Option<f64> = args.iter()
        .position(|a| a == "--theta-sun")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let tli_theta_deg: Option<f64> = args.iter()
        .position(|a| a == "--theta")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let tli_r_apogee: Option<f64> = args.iter()
        .position(|a| a == "--r-apogee")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    let n_total = 4 * N_PER_PARAM + 1;
    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!("║   WSB Sensitivity — One-at-a-Time (OAT) Grid         ║");
    eprintln!("╚══════════════════════════════════════════════════════╝");
    eprintln!("  Tag          : {}", if tag.is_empty() { "(none)" } else { &tag });
    eprintln!("  N_PER_PARAM  : {N_PER_PARAM}  (× 4 params + 1 nominal = {n_total} total)");
    eprintln!("  Sweep range  : −1σ … +1σ per parameter (others held at nominal)");
    eprintln!("  ΔV mag  σ    : ±{DV_MAG_SIGMA:.0e}  (fractional)");
    eprintln!("  ΔV dir  σ    : ±{:.4}°  (pitch only; yaw = 0)", DV_DIR_SIGMA.to_degrees());
    eprintln!("  θ_inject σ   : ±{THETA_SIGMA_DEG:.4}°  (burn timing)");
    eprintln!("  θ_sun σ      : ±{THETA_SUN_SIGMA_DEG:.4}°  (launch window)");
    eprintln!("  Integrator   : dt={LOG_DT}  RTOL={RTOL}  ATOL={ATOL}");
    eprintln!();

    // ── Build nominal IC (identical path to wsb_sensitivity) ─────────────────
    let (nominal_ic, theta_sun, theta_nom_rad, r_apogee_nd) = if use_hifi {
        let (h_ic, h_theta, h_theta_sun, h_r_apo, h_orbits) = load_hifi().unwrap_or_else(|| {
            eprintln!("  [error] could not read {HIFI_CSV}");
            std::process::exit(1);
        });
        let ts = theta_sun_override.map(|d| d.to_radians()).unwrap_or(h_theta_sun.to_radians());
        eprintln!("  IC source    : --hifi  (θ={:.4}°  θ_sun={:.4}°  r_apo={:.4} nd  est_orbits={:.3})",
            h_theta, ts.to_degrees(), h_r_apo, h_orbits);
        (h_ic, ts, h_theta.to_radians(), h_r_apo)
    } else if let (Some(theta_deg), Some(r_apo)) = (tli_theta_deg, tli_r_apogee) {
        let ts = theta_sun_override.unwrap_or(0.0).to_radians();
        let ic = tli_injection_ic(mu, R_PARK, r_apo, theta_deg.to_radians())
            .expect("Bad TLI IC");
        eprintln!("  IC source    : --theta/--r-apogee");
        (ic, ts, theta_deg.to_radians(), r_apo)
    } else if let Some(ref ic_s) = ic_str {
        let vals: Vec<f64> = ic_s.split_whitespace()
            .filter_map(|s| s.parse().ok()).collect();
        if vals.len() != 6 {
            eprintln!("  [error] --ic requires exactly 6 values"); std::process::exit(1);
        }
        let ic: [f64; 6] = [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]];
        let ts = theta_sun_override.unwrap_or(0.0).to_radians();
        let r_apo = tli_r_apogee.unwrap_or_else(|| {
            eprintln!("  [error] --ic requires --r-apogee"); std::process::exit(1);
        });
        let theta_d = ic[1].atan2(ic[0]);
        eprintln!("  IC source    : --ic");
        (ic, ts, theta_d, r_apo)
    } else {
        let seed = load_hit(hit_id).unwrap_or_else(|| {
            eprintln!("  [error] hit {hit_id} not found in {SUMMARY_CSV}"); std::process::exit(1);
        });
        eprintln!("  IC source    : hit {hit_id}");
        let theta = seed.theta_deg.to_radians();
        let ts    = seed.theta_sun_deg.to_radians();
        let ic    = tli_injection_ic(mu, R_PARK, seed.r_apogee_nd, theta).expect("Bad IC");
        (ic, ts, theta, seed.r_apogee_nd)
    };

    // ── Reference orbital parameters ─────────────────────────────────────────
    let x_earth       = -mu;
    let dx_earth      = nominal_ic[0] - x_earth;
    let dy_earth      = nominal_ic[1];
    let r_park_ref    = (dx_earth*dx_earth + dy_earth*dy_earth + nominal_ic[2]*nominal_ic[2]).sqrt();
    let theta_nom_ref = dy_earth.atan2(dx_earth);
    let v_circ_ref    = (mu_earth / r_park_ref).sqrt();
    let a_tr          = (r_park_ref + r_apogee_nd) / 2.0;
    let dv_nom_mag    = (mu_earth * (2.0 / r_park_ref - 1.0 / a_tr)).sqrt() - v_circ_ref;
    eprintln!("  r_park_ref   : {:.6} nd", r_park_ref);
    eprintln!("  θ_nom_ref    : {:.4}°", theta_nom_ref.to_degrees());
    eprintln!("  TLI ΔV nom   : {:.6} nd  ({:.1} m/s)",
        dv_nom_mag, dv_nom_mag * params.v_star);
    eprintln!();
    let _ = theta_nom_rad;

    // ── Build OAT perturbation table ──────────────────────────────────────────
    // (ic, theta_sun_p, run_id, dmag, dpitch, dyaw, dtheta_deg, dtsun_deg)
    let mut ics: Vec<([f64; 6], f64, usize, f64, f64, f64, f64, f64)> = Vec::with_capacity(n_total);

    // Nominal
    ics.push((nominal_ic, theta_sun, 0, 0.0, 0.0, 0.0, 0.0, 0.0));

    // Helper: build perturbed IC from (dmag, dpitch, dyaw, dtheta_rad, dtsun_rad)
    let build_ic = |dmag: f64, dpitch: f64, dyaw: f64, dtheta: f64| -> [f64; 6] {
        let theta_p = theta_nom_ref + dtheta;
        let px = x_earth + r_park_ref * theta_p.cos();
        let py =           r_park_ref * theta_p.sin();
        let vel_pre = [
            -v_circ_ref * theta_p.sin() + py,
             v_circ_ref * theta_p.cos() - px,
            0.0_f64,
        ];
        let dv_dir   = [-theta_p.sin(), theta_p.cos(), 0.0_f64];
        let pitch_perp = [-theta_p.cos(), -theta_p.sin(), 0.0_f64];
        let dv_dir_p = [
            dv_dir[0] + dpitch * pitch_perp[0],
            dv_dir[1] + dpitch * pitch_perp[1],
            dyaw,
        ];
        let dir_norm = (dv_dir_p[0]*dv_dir_p[0]
            + dv_dir_p[1]*dv_dir_p[1]
            + dv_dir_p[2]*dv_dir_p[2]).sqrt();
        let dv_mag_p = dv_nom_mag * (1.0 + dmag);
        [
            px, py, 0.0,
            vel_pre[0] + dv_mag_p * dv_dir_p[0] / dir_norm,
            vel_pre[1] + dv_mag_p * dv_dir_p[1] / dir_norm,
            vel_pre[2] + dv_mag_p * dv_dir_p[2] / dir_norm,
        ]
    };

    // Group 1 — ΔV magnitude sweep (run_ids 1..=N_PER_PARAM)
    for (i, &dmag) in linspace(-DV_MAG_SIGMA, DV_MAG_SIGMA, N_PER_PARAM).iter().enumerate() {
        let run_id = 1 + i;
        let ic = build_ic(dmag, 0.0, 0.0, 0.0);
        ics.push((ic, theta_sun, run_id, dmag, 0.0, 0.0, 0.0, 0.0));
    }

    // Group 2 — pointing (pitch) sweep (run_ids N_PER_PARAM+1..=2*N_PER_PARAM)
    for (i, &dpitch) in linspace(-DV_DIR_SIGMA, DV_DIR_SIGMA, N_PER_PARAM).iter().enumerate() {
        let run_id = N_PER_PARAM + 1 + i;
        let ic = build_ic(0.0, dpitch, 0.0, 0.0);
        ics.push((ic, theta_sun, run_id, 0.0, dpitch, 0.0, 0.0, 0.0));
    }

    // Group 3 — burn-timing (θ_inject) sweep (run_ids 2*N_PER_PARAM+1..=3*N_PER_PARAM)
    for (i, &dtheta_deg) in linspace(-THETA_SIGMA_DEG, THETA_SIGMA_DEG, N_PER_PARAM).iter().enumerate() {
        let run_id = 2 * N_PER_PARAM + 1 + i;
        let ic = build_ic(0.0, 0.0, 0.0, dtheta_deg.to_radians());
        ics.push((ic, theta_sun, run_id, 0.0, 0.0, 0.0, dtheta_deg, 0.0));
    }

    // Group 4 — launch-window (θ_sun) sweep (run_ids 3*N_PER_PARAM+1..=4*N_PER_PARAM)
    for (i, &dtsun_deg) in linspace(-THETA_SUN_SIGMA_DEG, THETA_SUN_SIGMA_DEG, N_PER_PARAM).iter().enumerate() {
        let run_id = 3 * N_PER_PARAM + 1 + i;
        let ic = build_ic(0.0, 0.0, 0.0, 0.0);
        let theta_sun_p = theta_sun + dtsun_deg.to_radians();
        ics.push((ic, theta_sun_p, run_id, 0.0, 0.0, 0.0, 0.0, dtsun_deg));
    }

    // ── Propagate ────────────────────────────────────────────────────────────
    eprintln!("  Propagating {} trajectories …", ics.len());
    let mut ensemble: Vec<(usize, Vec<Step3d>, Outcome, usize, f64, f64, f64, f64, f64, f64, f64)> = Vec::new();
    // (run_id, traj, outcome, n_orbits, est_orbits, min_alt_km, dmag, dpitch, dyaw, dtheta_deg, dtsun_deg)

    for (idx, (ic, theta_sun_p, run_id, dmag_p, dpitch_p, dyaw_p, dtheta_p, dtsun_p)) in ics.iter().enumerate() {
        if idx % 10 == 0 {
            eprintln!("  [{}/{}] …", idx, ics.len());
        }

        let bcr  = Bcr4bpParams::earth_moon_sun(*theta_sun_p);
        let traj = propagate_bcr4bp(mu, bcr, *ic, T_PROP, LOG_DT, RTOL, ATOL);

        let min_moon_dist_nd = traj.iter().map(|s| {
            let dx = s.x - moon_x;
            (dx*dx + s.y*s.y + s.z*s.z).sqrt()
        }).fold(f64::MAX, f64::min);

        let moon_crash = min_moon_dist_nd < r_moon_nd || {
            traj.windows(2).any(|w| {
                let (a, b) = (&w[0], &w[1]);
                let ra = { let dx=a.x-moon_x; (dx*dx+a.y*a.y+a.z*a.z).sqrt() };
                let rb = { let dx=b.x-moon_x; (dx*dx+b.y*b.y+b.z*b.z).sqrt() };
                if ra > 3.0*r_moon_nd && rb > 3.0*r_moon_nd { return false; }
                let dx = b.x-a.x; let dy = b.y-a.y;
                let ax = a.x-moon_x; let ay = a.y;
                let t  = -(ax*dx + ay*dy) / (dx*dx + dy*dy + 1e-30);
                let t  = t.clamp(0.0, 1.0);
                let cx = ax + t*dx; let cy = ay + t*dy;
                (cx*cx + cy*cy).sqrt() < r_moon_nd
            })
        };

        let capture    = detect_capture(&traj, mu, MIN_CAPTURE_TIME);
        let est_orbits = capture.max_capture_interval / LUNAR_PERIOD_ND;
        let n_orbits   = capture.n_periapsis;
        let min_alt_km = min_moon_dist_nd * l_km - R_MOON_KM;

        let outcome = if moon_crash {
            Outcome::MoonCrash
        } else if n_orbits >= 1 && capture.n_entries > 0 {
            Outcome::Captured
        } else {
            Outcome::Escaped
        };

        if *run_id == 0 {
            eprintln!("  Nominal → {:?}  n_orbits={n_orbits}  min_alt={min_alt_km:.0} km", outcome);
        }

        ensemble.push((*run_id, traj, outcome, n_orbits, est_orbits, min_alt_km,
                       *dmag_p, *dpitch_p, *dyaw_p, *dtheta_p, *dtsun_p));
    }

    // ── Outcome summary ───────────────────────────────────────────────────────
    let n_captured = ensemble.iter().filter(|e| e.2 == Outcome::Captured).count();
    let n_crash    = ensemble.iter().filter(|e| e.2 == Outcome::MoonCrash).count();
    let n_escaped  = ensemble.iter().filter(|e| e.2 == Outcome::Escaped).count();
    eprintln!();
    eprintln!("  ── Outcome breakdown ─────────────────────────────────");
    eprintln!("  Captured   : {n_captured}");
    eprintln!("  Moon crash : {n_crash}");
    eprintln!("  Escaped    : {n_escaped}");
    eprintln!();

    // ── STM for nominal ───────────────────────────────────────────────────────
    let nominal_traj_ref = &ensemble[0].1;
    let t_hill_nom = nominal_traj_ref.iter()
        .find(|s| {
            let dx = s.x - moon_x;
            (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
        })
        .map(|s| s.time)
        .unwrap_or(T_PROP * 0.5);
    eprintln!("  CRTBP+STM: propagating nominal to Hill entry t={:.2} nd …", t_hill_nom);
    let stm_csv_path = format!("{OUT_DIR}/sensitivity_stm{tag_suffix}.csv");
    match propagate_3d_stm_full(mu, nominal_ic, t_hill_nom, LOG_DT, RTOL, ATOL) {
        Ok((stm_traj, stm_series)) => {
            let mut stm_csv = String::from(
                "time_nd,\
                 phi_0_0,phi_0_1,phi_0_2,phi_0_3,phi_0_4,phi_0_5,\
                 phi_1_0,phi_1_1,phi_1_2,phi_1_3,phi_1_4,phi_1_5\n"
            );
            for (step, phi) in stm_traj.iter().zip(stm_series.iter()) {
                writeln!(stm_csv,
                    "{:.6},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},\
                     {:.10},{:.10},{:.10},{:.10},{:.10},{:.10}",
                    step.time,
                    phi[0][0], phi[0][1], phi[0][2], phi[0][3], phi[0][4], phi[0][5],
                    phi[1][0], phi[1][1], phi[1][2], phi[1][3], phi[1][4], phi[1][5],
                ).unwrap();
            }
            fs::write(&stm_csv_path, &stm_csv).expect("STM csv write failed");
        }
        Err(e) => {
            eprintln!("  WARNING: CRTBP STM failed ({e}) — writing empty file.");
            fs::write(&stm_csv_path,
                "time_nd,phi_0_0,phi_0_1,phi_0_2,phi_0_3,phi_0_4,phi_0_5,\
                 phi_1_0,phi_1_1,phi_1_2,phi_1_3,phi_1_4,phi_1_5\n"
            ).expect("STM csv write failed");
        }
    }
    eprintln!("  Saved {stm_csv_path}");
    eprintln!();

    // ── Save ensemble CSV ─────────────────────────────────────────────────────
    let _save_window = 5.0 * LUNAR_PERIOD_ND;
    let csv_path    = format!("{OUT_DIR}/sensitivity_ensemble{tag_suffix}.csv");
    let mut csv = String::from(
        "run_id,time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,\
         outcome,n_orbits,est_orbits,min_alt_km,is_nominal,\
         dmag_frac,dpitch_rad,dyaw_rad,dtheta_deg,dtsun_deg\n"
    );

    for (run_id, traj, outcome, n_orbits, est_orbits, min_alt_km,
         dmag_p, dpitch_p, dyaw_p, dtheta_p, dtsun_p) in &ensemble
    {
        let is_nominal = if *run_id == 0 { 1 } else { 0 };

        // Moon-crash clip (same interpolation as wsb_sensitivity)
        let t_crash = if *outcome == Outcome::MoonCrash {
            let step_crash = traj.iter().find(|s| {
                let dx = s.x - moon_x;
                (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_moon_nd
            }).map(|s| s.time);
            if let Some(t) = step_crash {
                t
            } else {
                traj.windows(2).find_map(|w| {
                    let (a, b) = (&w[0], &w[1]);
                    let ra = { let dx=a.x-moon_x; (dx*dx+a.y*a.y+a.z*a.z).sqrt() };
                    let rb = { let dx=b.x-moon_x; (dx*dx+b.y*b.y+b.z*b.z).sqrt() };
                    if ra > 3.0*r_moon_nd && rb > 3.0*r_moon_nd { return None; }
                    let dx = b.x-a.x; let dy = b.y-a.y; let dz = b.z-a.z;
                    let ax = a.x-moon_x; let ay = a.y; let az = a.z;
                    let ca = dx*dx + dy*dy + dz*dz;
                    let cb = 2.0*(ax*dx + ay*dy + az*dz);
                    let cc = ax*ax + ay*ay + az*az - r_moon_nd*r_moon_nd;
                    let disc = cb*cb - 4.0*ca*cc;
                    if disc < 0.0 || ca < 1e-30 { return None; }
                    let t1 = (-cb - disc.sqrt()) / (2.0*ca);
                    let t2 = (-cb + disc.sqrt()) / (2.0*ca);
                    let t_hit = if t1 >= 0.0 && t1 <= 1.0 { t1 }
                                else if t2 >= 0.0 && t2 <= 1.0 { t2 }
                                else { return None; };
                    Some(a.time + t_hit * (b.time - a.time))
                }).unwrap_or(f64::MAX)
            }
        } else {
            f64::MAX
        };

        let _hill_entry_t = traj.iter().find(|s| {
            let dx = s.x - moon_x;
            (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
        }).map(|s| s.time).unwrap_or(f64::MAX);

        // let t_stop = if hill_entry_t < f64::MAX {
        //     hill_entry_t + save_window
        // } else {
        //     T_PROP
        // }.min(t_crash);
        let t_stop = T_PROP.min(t_crash);


        for s in traj.iter().take_while(|s| s.time <= t_stop) {
            writeln!(csv,
                "{run_id},{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{},{n_orbits},{:.3},{:.1},{is_nominal},{:.8},{:.8},{:.8},{:.6},{:.6}",
                s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz,
                outcome.label(), est_orbits, min_alt_km,
                dmag_p, dpitch_p, dyaw_p, dtheta_p, dtsun_p,
            ).unwrap();
        }
        writeln!(csv,
            "{run_id},NaN,NaN,NaN,NaN,NaN,NaN,NaN,{},{n_orbits},{:.3},{:.1},{is_nominal},{:.8},{:.8},{:.8},{:.6},{:.6}",
            outcome.label(), est_orbits, min_alt_km,
            dmag_p, dpitch_p, dyaw_p, dtheta_p, dtsun_p,
        ).unwrap();
    }

    fs::write(&csv_path, &csv).expect("ensemble csv write failed");
    eprintln!("  Saved {csv_path}");

    // ── Summary ───────────────────────────────────────────────────────────────
    let nom_n_orbits   = ensemble[0].3;
    let nom_est_orbits = ensemble[0].4;
    let mut info = String::new();
    writeln!(info, "=== WSB Sensitivity Ensemble (OAT grid) ===\n").unwrap();
    writeln!(info, "Mode           : one-at-a-time — each parameter swept from −1σ to +1σ").unwrap();
    writeln!(info, "Tag            : {}", if tag.is_empty() { "(default)" } else { &tag }).unwrap();
    writeln!(info, "N_PER_PARAM    : {N_PER_PARAM}  (× 4 + 1 nominal = {n_total} total)").unwrap();
    writeln!(info, "θ_sun          : {:.4}°", theta_sun.to_degrees()).unwrap();
    writeln!(info, "Nominal IC     : x={:.8} y={:.8} z={:.8} vx={:.8} vy={:.8} vz={:.8}",
        nominal_ic[0], nominal_ic[1], nominal_ic[2],
        nominal_ic[3], nominal_ic[4], nominal_ic[5]).unwrap();
    writeln!(info, "Nominal orbits : {nom_n_orbits}  /  {nom_est_orbits:.3} EM-period hill dwell").unwrap();
    writeln!(info).unwrap();
    writeln!(info, "ΔV mag  σ      : ±{DV_MAG_SIGMA:.0e}  (fractional)").unwrap();
    writeln!(info, "ΔV dir  σ      : ±{:.4}°  (pitch sweep; yaw = 0)", DV_DIR_SIGMA.to_degrees()).unwrap();
    writeln!(info, "θ_inject σ     : ±{THETA_SIGMA_DEG:.4}°").unwrap();
    writeln!(info, "θ_sun σ        : ±{THETA_SUN_SIGMA_DEG:.4}°").unwrap();
    writeln!(info).unwrap();
    writeln!(info, "── Outcome breakdown ──────────────────────────────────").unwrap();
    writeln!(info, "  Captured   : {n_captured} / {n_total}  ({:.0}%)",
        100.0 * n_captured as f64 / n_total as f64).unwrap();
    writeln!(info, "  Moon crash : {n_crash} / {n_total}  ({:.0}%)",
        100.0 * n_crash as f64 / n_total as f64).unwrap();
    writeln!(info, "  Escaped    : {n_escaped} / {n_total}  ({:.0}%)",
        100.0 * n_escaped as f64 / n_total as f64).unwrap();

    let info_path = format!("{OUT_DIR}/sensitivity_summary{tag_suffix}.txt");
    fs::write(&info_path, &info).expect("summary write failed");
    eprintln!("  Saved {info_path}");
    print!("{info}");
}

// ════════════════════════════════════════════════════════════════════════════════
// Helpers (identical to wsb_sensitivity.rs)
// ════════════════════════════════════════════════════════════════════════════════

const R_PARK: f64 = (6_371.0 + 378.0) / 384_400.0;

fn load_hit(target_hit_id: usize) -> Option<HitSeed> {
    let content = fs::read_to_string(SUMMARY_CSV).ok()?;
    let mut hits: Vec<HitSeed> = Vec::new();
    let mut counter = 0usize;
    for line in content.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 13 { continue; }
        if c[1].trim() != "refined" { continue; }
        let seed_id       = c[0].parse().unwrap_or(0);
        let theta_deg:f64 = c[2].parse().unwrap_or(f64::NAN);
        let theta_sun_deg = c[3].parse().unwrap_or(f64::NAN);
        let r_apogee_nd   = c[4].parse().unwrap_or(f64::NAN);
        let est_orbits    = c[7].parse().unwrap_or(0.0);
        let dtheta_deg    = c[9].parse().unwrap_or(f64::NAN);
        let dsun_deg      = c[10].parse().unwrap_or(f64::NAN);
        if theta_deg.is_nan() { continue; }
        counter += 1;
        hits.push(HitSeed { hit_id: counter, seed_id, theta_deg, theta_sun_deg,
                             r_apogee_nd, est_orbits, dtheta_deg, dsun_deg });
    }
    hits.sort_by(|a, b| b.est_orbits.partial_cmp(&a.est_orbits).unwrap());
    for (i, h) in hits.iter_mut().enumerate() { h.hit_id = i + 1; }
    hits.into_iter().find(|h| h.hit_id == target_hit_id)
}

fn load_hifi() -> Option<([f64; 6], f64, f64, f64, f64)> {
    let content = fs::read_to_string(HIFI_CSV).ok()?;
    for line in content.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 13 { continue; }
        let x  = c[1].parse::<f64>().ok()?;
        let y  = c[2].parse::<f64>().ok()?;
        let z  = c[3].parse::<f64>().ok()?;
        let vx = c[4].parse::<f64>().ok()?;
        let vy = c[5].parse::<f64>().ok()?;
        let vz = c[6].parse::<f64>().ok()?;
        let theta_deg     = c[10].parse().unwrap_or(f64::NAN);
        let theta_sun_deg = c[11].parse().ok()?;
        let r_apogee_nd   = c[12].parse().unwrap_or(f64::NAN);
        let est_orbits    = c[9].parse().unwrap_or(f64::NAN);
        return Some(([x, y, z, vx, vy, vz], theta_deg, theta_sun_deg, r_apogee_nd, est_orbits));
    }
    None
}
