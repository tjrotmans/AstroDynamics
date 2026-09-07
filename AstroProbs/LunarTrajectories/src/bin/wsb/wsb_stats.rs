//! wsb_stats — Sigma-sweep Monte Carlo sensitivity analysis.
//!
//! Propagates N_PER_LEVEL trajectories at each of several sigma-scale levels
//! without saving trajectory data.  Uses Gaussian (Box-Muller) perturbations
//! rather than the uniform ±σ used in wsb_sensitivity.
//!
//! Outputs (all in out/wsb/):
//!   stats_samples.csv  — one row per trajectory (actual perturbations + outcome)
//!   stats_sweep.csv    — outcome fractions aggregated per sigma-scale level
//!
//! Usage:
//!   cargo run -p lunar_trajectories --bin wsb_stats --release -- --hifi
//!   cargo run -p lunar_trajectories --bin wsb_stats --release -- \
//!       --theta 205.549 --r-apogee 3.9 --theta-sun 27.019

use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;
use std::fs;

use lunar_trajectories::crtbp::CrtbpParams;
use lunar_trajectories::propagator::{propagate_bcr4bp, Bcr4bpParams, Step3d};
use lunar_trajectories::transfers::{lunar_hill_radius, detect_capture, tli_injection_ic};

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                         CONFIGURATION                                        ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

/// Trajectories at σ×1.0.
const N_PER_LEVEL: usize = 20_000;

/// Single sigma level: realistic 1σ Monte Carlo only.
const SIGMA_SCALES: &[f64] = &[1.0];

/// Nominal 1-sigma values (must match wsb_sensitivity.rs for consistency).
const DV_MAG_SIGMA:        f64 = 1e-3;     // fractional thrust magnitude error
const DV_DIR_SIGMA:        f64 = 1.75e-3;  // rad — pitch and yaw pointing error
const THETA_SIGMA_DEG:     f64 = 0.2;     // burn timing error within parking orbit
const THETA_SUN_SIGMA_DEG: f64 = 3.0;     // launch window / day shift

/// Maximum propagation time [nd].
const T_PROP: f64 = 25.0 * PI;

/// Output step for statistics.  0.005 nd ≈ 44 min: ~5–8 pts per low Hill-sphere orbit
/// (T_min ≈ 3–5 h), sufficient for reliable periapsis counting via detect_capture.
const STATS_LOG_DT: f64 = 0.005;
const RTOL:         f64 = 1e-10;
const ATOL:         f64 = 1e-12;

/// Minimum Hill-sphere dwell to count as a capture [nd].
const MIN_CAPTURE_TIME: f64 = 0.15;

/// Flyby threshold: spacecraft periapsis passages inside the Hill sphere.
/// n_orbits counts local r_moon minima (each ≈ one spacecraft orbit, period hours–days).

const LUNAR_PERIOD_ND: f64 = 2.0 * PI;
const R_MOON_KM:       f64 = 1_737.4;
const T_STAR:          f64 = 375_700.0;   // [s] EM time unit

/// Captures with Hill-sphere entry more than this many days after the nominal's
/// first Hill entry are classified as LateTransfer rather than Captured.
const LATE_TRANSFER_DAYS: f64 = 90.0;
const LATE_TRANSFER_ND:   f64 = LATE_TRANSFER_DAYS * 86_400.0 / T_STAR;

const OUT_DIR:    &str = "out/wsb";
const HIFI_CSV:   &str = "out/wsb/solution_hifi.csv";
const SUMMARY_CSV: &str = "out/wsb/refine_summary.csv";

// ── Animation ensemble export ─────────────────────────────────────────────────
/// Stratified trajectory counts written to sensitivity_ensemble.csv.
/// Drawn from the full N_PER_LEVEL run, so diversity is guaranteed.
const EXPORT_N_CRASH: usize = 3;
const EXPORT_N_FLYBY: usize = 10;  // captured with est_orbits < capture_est_thresh()
const EXPORT_N_CAP:   usize = 3;  // captured with est_orbits >= capture_est_thresh()
const EXPORT_N_MISS:  usize = 184; // escaped

/// Minimum est_orbits (max_hill_dwell / LUNAR_PERIOD_ND) to classify as captured.
/// 5 days expressed as a fraction of LUNAR_PERIOD_ND = 2π × T_STAR.
fn capture_est_thresh(_mu: f64) -> f64 {
    7.0 * 86_400.0 / (T_STAR * 2.0 * PI)   // 3.5 days ≈ 0.128
}
/// Output step for animation — finer than STATS_LOG_DT for smooth visuals.
const ANIM_LOG_DT:    f64   = 0.002;  // ~7 h (matches wsb_sensitivity LOG_DT)
const ANIM_SAVE_WIN:  f64   = 6.0 * LUNAR_PERIOD_ND; // Hill entry + 5 orbits

// ════════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
enum Outcome { Captured, LateTransfer, MoonCrash, EarthCrash, Escaped }

impl Outcome {
    fn label(self) -> &'static str {
        match self {
            Outcome::Captured     => "captured",
            Outcome::LateTransfer => "late_transfer",
            Outcome::MoonCrash    => "moon_crash",
            Outcome::EarthCrash   => "earth_crash",
            Outcome::Escaped      => "escaped",
        }
    }
}

const R_EARTH_KM: f64 = 6_371.0;

struct SampleRecord {
    sigma_scale:          f64,
    run_id:               usize,
    dv_mag_actual:        f64,  // fractional (signed)
    dv_dir_actual_rad:    f64,  // magnitude = sqrt(pitch²+yaw²)
    theta_actual_deg:     f64,  // signed
    theta_sun_actual_deg: f64,  // signed
    outcome:              Outcome,
    n_orbits:             usize,
    est_orbits:           f64,
    min_alt_km:           f64,
    hill_entry_nd:        f64,  // first Hill-sphere entry time [nd]; NaN if never entered
    // Stored for animation re-propagation — avoids reconstructing from perturbations.
    ic:          [f64; 6],
    theta_sun_p: f64,           // absolute perturbed Sun angle [rad]
}

// ════════════════════════════════════════════════════════════════════════════════

fn main() {
    let params    = CrtbpParams::earth_moon();
    let mu        = params.mu;
    let l_km      = params.l_star / 1e3;
    let moon_x    = 1.0 - mu;
    let r_hill    = lunar_hill_radius(mu);
    let r_moon_nd = R_MOON_KM / l_km;
    let mu_earth  = 1.0 - mu;

    fs::create_dir_all(OUT_DIR).unwrap();

    // ── CLI (same interface as wsb_sensitivity) ───────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let use_hifi = args.iter().any(|a| a == "--hifi");
    let ic_str: Option<String> = args.iter()
        .position(|a| a == "--ic")
        .and_then(|i| args.get(i + 1)).cloned();
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
    let base_seed: u64 = args.iter()
        .position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xDEAD_BEEF_0000_0000_u64);

    let n_total = SIGMA_SCALES.len() * N_PER_LEVEL;
    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!("║        WSB Stats — Sigma-Sweep Monte Carlo           ║");
    eprintln!("╚══════════════════════════════════════════════════════╝");
    eprintln!("  N_PER_LEVEL   : {N_PER_LEVEL}  ({} levels × {N_PER_LEVEL} = {n_total} total)",
        SIGMA_SCALES.len());
    eprintln!("  Sigma scales  : {:?}", SIGMA_SCALES);
    eprintln!("  STATS_LOG_DT  : {STATS_LOG_DT} nd ({:.1} h)", STATS_LOG_DT * T_STAR / 3600.0);
    eprintln!("  DV_MAG σ      : {DV_MAG_SIGMA:.0e}  (Gaussian, fractional)");
    eprintln!("  DV_DIR σ      : {:.4}° (Gaussian, pitch & yaw)", DV_DIR_SIGMA.to_degrees());
    eprintln!("  θ_inject σ    : {THETA_SIGMA_DEG:.4}°  (Gaussian)");
    eprintln!("  θ_sun σ       : {THETA_SUN_SIGMA_DEG:.4}°  (Gaussian)");
    eprintln!();

    // ── Build nominal IC ──────────────────────────────────────────────────────
    let (nominal_ic, theta_sun, _theta_nom_ref, r_apogee_nd) = if use_hifi {
        let (h_ic, h_theta, h_theta_sun, h_r_apo, _) = load_hifi().unwrap_or_else(|| {
            eprintln!("  [error] could not read {HIFI_CSV} — run wsb_optimize first.");
            std::process::exit(1);
        });
        let ts = theta_sun_override.map(|d| d.to_radians()).unwrap_or(h_theta_sun.to_radians());
        eprintln!("  IC source : --hifi  (θ={:.4}°  θ_sun={:.4}°  r_apo={:.4} nd)",
            h_theta, ts.to_degrees(), h_r_apo);
        (h_ic, ts, h_theta.to_radians(), h_r_apo)
    } else if let (Some(theta_deg), Some(r_apo)) = (tli_theta_deg, tli_r_apogee) {
        let ts = theta_sun_override.unwrap_or(0.0).to_radians();
        let ic = tli_injection_ic(mu, R_PARK, r_apo, theta_deg.to_radians())
            .expect("tli_injection_ic failed");
        eprintln!("  IC source : --theta/--r-apogee");
        (ic, ts, theta_deg.to_radians(), r_apo)
    } else if let Some(ref ic_s) = ic_str {
        let vals: Vec<f64> = ic_s.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if vals.len() != 6 {
            eprintln!("  [error] --ic requires exactly 6 values"); std::process::exit(1);
        }
        let ic: [f64; 6] = [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]];
        let ts = theta_sun_override.unwrap_or(0.0).to_radians();
        let r_apo = tli_r_apogee.unwrap_or_else(|| {
            eprintln!("  [error] --ic requires --r-apogee"); std::process::exit(1);
        });
        let theta_d = ic[1].atan2(ic[0]);
        eprintln!("  IC source : --ic (direct state vector)");
        (ic, ts, theta_d, r_apo)
    } else {
        let seed = load_hit(hit_id).unwrap_or_else(|| {
            eprintln!("  [error] hit {hit_id} not found in {SUMMARY_CSV}");
            std::process::exit(1);
        });
        let theta = seed.theta_deg.to_radians();
        let ts    = seed.theta_sun_deg.to_radians();
        let ic    = tli_injection_ic(mu, R_PARK, seed.r_apogee_nd, theta)
            .expect("Bad nominal IC");
        eprintln!("  IC source : legacy hit {hit_id}");
        (ic, ts, theta, seed.r_apogee_nd)
    };

    // ── Reference orbital parameters for perturbations ────────────────────────
    let x_earth    = -mu;
    let dx_earth   = nominal_ic[0] - x_earth;
    let dy_earth   = nominal_ic[1];
    let r_park_ref = (dx_earth*dx_earth + dy_earth*dy_earth + nominal_ic[2]*nominal_ic[2]).sqrt();
    let theta_ref  = dy_earth.atan2(dx_earth);
    let v_circ_ref = (mu_earth / r_park_ref).sqrt();
    let a_tr       = (r_park_ref + r_apogee_nd) / 2.0;
    let dv_nom_mag = (mu_earth * (2.0 / r_park_ref - 1.0 / a_tr)).sqrt() - v_circ_ref;
    eprintln!("  TLI ΔV nom : {:.6} nd ({:.1} m/s)",
        dv_nom_mag, dv_nom_mag * params.v_star);
    eprintln!();

    // ── Propagate nominal once to obtain Hill-sphere entry time reference ─────
    let bcr_nom   = Bcr4bpParams::earth_moon_sun(theta_sun);
    let traj_nom  = propagate_bcr4bp(mu, bcr_nom, nominal_ic, T_PROP, STATS_LOG_DT, RTOL, ATOL);
    let t_hill_nominal: f64 = traj_nom.iter().find(|s| {
        let dx = s.x - moon_x;
        (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
    }).map(|s| s.time).unwrap_or(f64::NAN);
    if t_hill_nominal.is_nan() {
        eprintln!("  [warn] nominal trajectory never enters Hill sphere — LateTransfer will not trigger");
    } else {
        eprintln!("  Nominal Hill entry : {:.4} nd  ({:.1} days)",
            t_hill_nominal, t_hill_nominal * T_STAR / 86_400.0);
        eprintln!("  LateTransfer threshold: >{:.0} days past nominal ({:.4} nd)",
            LATE_TRANSFER_DAYS, LATE_TRANSFER_ND);
    }
    eprintln!();

    // ── Sigma sweep ───────────────────────────────────────────────────────────
    let mut all_samples: Vec<SampleRecord> = Vec::with_capacity(n_total);

    for (level_idx, &sigma_scale) in SIGMA_SCALES.iter().enumerate() {
        eprint!("  σ×{:.2}  [{:2}/{}]  propagating {N_PER_LEVEL} … ",
            sigma_scale, level_idx + 1, SIGMA_SCALES.len());

        let mut rng = SimpleLcg::new(base_seed
            .wrapping_add((level_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));

        let mut level_samples: Vec<SampleRecord> = Vec::with_capacity(N_PER_LEVEL);

        for run_id in 0..N_PER_LEVEL {
            // Gaussian perturbations via Box-Muller transform.
            let (g0, g1) = box_muller(rng.next_f64_nonzero(), rng.next_f64());
            let (g2, g3) = box_muller(rng.next_f64_nonzero(), rng.next_f64());
            let (g4, g5) = box_muller(rng.next_f64_nonzero(), rng.next_f64());
            let _ = g5;

            let dmag:   f64 = g0 * DV_MAG_SIGMA * sigma_scale;
            let dpitch: f64 = g1 * DV_DIR_SIGMA  * sigma_scale;
            let dyaw:   f64 = g2 * DV_DIR_SIGMA  * sigma_scale;
            let dtheta: f64 = g3 * THETA_SIGMA_DEG.to_radians() * sigma_scale;
            let dtsun:  f64 = g4 * THETA_SUN_SIGMA_DEG.to_radians() * sigma_scale;

            let ic = if sigma_scale == 0.0 {
                // Exact nominal — no floating-point reconstruction noise.
                nominal_ic
            } else {
                let theta_p     = theta_ref + dtheta;
                let theta_sun_p = theta_sun + dtsun;
                let _ = theta_sun_p;   // used in bcr below

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

            let theta_sun_p = theta_sun + dtsun;
            let bcr  = Bcr4bpParams::earth_moon_sun(theta_sun_p);
            let traj = propagate_bcr4bp(mu, bcr, ic, T_PROP, STATS_LOG_DT, RTOL, ATOL);

            // ── Hill entry time (computed first — used to window crash detection) ─
            let hill_entry_nd: f64 = traj.iter().find(|s| {
                let dx = s.x - moon_x;
                (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
            }).map(|s| s.time).unwrap_or(f64::NAN);

            // ── Outcome detection ─────────────────────────────────────────────
            // Moon crash is only checked within the animation show window
            // (hill_entry + ANIM_SAVE_WIN).  Without this, a trajectory that
            // does a clean flyby at t≈90 d and then re-encounters the Moon at
            // t≈250 d gets labelled MoonCrash even though the animation only
            // ever shows the first flyby.
            let show_end = if hill_entry_nd.is_nan() { T_PROP }
                           else { hill_entry_nd + ANIM_SAVE_WIN };

            let min_moon_dist = traj.iter()
                .filter(|s| s.time <= show_end)
                .map(|s| { let dx = s.x - moon_x; (dx*dx + s.y*s.y + s.z*s.z).sqrt() })
                .fold(f64::MAX, f64::min);

            // Earth crash checked over full trajectory (happens during outbound leg)
            let r_earth_nd  = R_EARTH_KM / l_km;
            let earth_crash = traj.iter().any(|s| {
                let dx = s.x - x_earth;
                (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_earth_nd
            });
            let moon_crash  = min_moon_dist < r_moon_nd;

            // Restrict capture detection to the same show window so that a
            // late re-encounter (t > show_end) doesn't inflate n_periapsis and
            // cause a flyby to be mis-labelled as captured.
            let show_end_idx = traj.partition_point(|s| s.time <= show_end);
            let capture    = detect_capture(&traj[..show_end_idx], mu, MIN_CAPTURE_TIME);
            let est_orbits = capture.max_capture_interval / LUNAR_PERIOD_ND;
            let n_orbits   = capture.n_periapsis;
            let min_alt_km = min_moon_dist * l_km - R_MOON_KM;

            let outcome = if earth_crash {
                Outcome::EarthCrash          // re-impact before reaching Moon
            } else if moon_crash {
                Outcome::MoonCrash
            } else if n_orbits >= 1 && capture.n_entries > 0 {
                if !t_hill_nominal.is_nan()
                    && !hill_entry_nd.is_nan()
                    && hill_entry_nd > t_hill_nominal + LATE_TRANSFER_ND
                {
                    Outcome::LateTransfer
                } else {
                    Outcome::Captured
                }
            } else {
                Outcome::Escaped
            };

            level_samples.push(SampleRecord {
                sigma_scale,
                run_id,
                dv_mag_actual:        dmag,
                dv_dir_actual_rad:    (dpitch*dpitch + dyaw*dyaw).sqrt(),
                theta_actual_deg:     dtheta.to_degrees(),
                theta_sun_actual_deg: dtsun.to_degrees(),
                outcome,
                n_orbits,
                est_orbits,
                min_alt_km,
                hill_entry_nd,
                ic,
                theta_sun_p,
            });
        }

        let n       = level_samples.len();
        let n_cap   = level_samples.iter().filter(|s| s.outcome == Outcome::Captured).count();
        let n_late  = level_samples.iter().filter(|s| s.outcome == Outcome::LateTransfer).count();
        let n_cra   = level_samples.iter().filter(|s| s.outcome == Outcome::MoonCrash).count();
        let n_earth = level_samples.iter().filter(|s| s.outcome == Outcome::EarthCrash).count();
        let n_esc   = level_samples.iter().filter(|s| s.outcome == Outcome::Escaped).count();
        eprintln!("done  cap={n_cap} ({:.0}%)  late={n_late} ({:.0}%)  moon_crash={n_cra} ({:.0}%)  earth_crash={n_earth} ({:.0}%)  esc={n_esc} ({:.0}%)",
            100.0*n_cap   as f64/n as f64, 100.0*n_late  as f64/n as f64,
            100.0*n_cra   as f64/n as f64, 100.0*n_earth as f64/n as f64,
            100.0*n_esc   as f64/n as f64);

        all_samples.extend(level_samples);
    }

    // ── Write stats_samples.csv ───────────────────────────────────────────────
    let samples_path = format!("{OUT_DIR}/stats_samples{tag_suffix}.csv");
    let mut csv = String::from(
        "sigma_scale,run_id,dv_mag_actual,dv_dir_actual_rad,\
         theta_actual_deg,theta_sun_actual_deg,\
         outcome,n_orbits,est_orbits,min_alt_km,hill_entry_nd\n"
    );
    for s in &all_samples {
        writeln!(csv,
            "{:.4},{},{:.8},{:.8},{:.6},{:.6},{},{},{:.4},{:.1},{:.6}",
            s.sigma_scale, s.run_id,
            s.dv_mag_actual, s.dv_dir_actual_rad,
            s.theta_actual_deg, s.theta_sun_actual_deg,
            s.outcome.label(), s.n_orbits, s.est_orbits, s.min_alt_km,
            s.hill_entry_nd,
        ).unwrap();
    }
    fs::write(&samples_path, &csv).expect("write stats_samples.csv failed");
    eprintln!("\n  Saved {samples_path}  ({} rows)", all_samples.len());

    // ── Write stats_sweep.csv (aggregated per sigma level) ────────────────────
    let sweep_path = format!("{OUT_DIR}/stats_sweep{tag_suffix}.csv");
    let mut sweep = String::from(
        "sigma_scale,n_total,n_captured,n_late_transfer,n_flyby,n_moon_crash,n_earth_crash,n_escaped,\
         frac_captured,frac_late_transfer,frac_flyby,frac_moon_crash,frac_earth_crash,frac_escaped\n"
    );
    for &sigma_scale in SIGMA_SCALES {
        let level: Vec<_> = all_samples.iter()
            .filter(|s| (s.sigma_scale - sigma_scale).abs() < 1e-9)
            .collect();
        let n = level.len();
        if n == 0 { continue; }

        let n_crash  = level.iter().filter(|s| s.outcome == Outcome::MoonCrash).count();
        let n_earth  = level.iter().filter(|s| s.outcome == Outcome::EarthCrash).count();
        let n_esc    = level.iter().filter(|s| s.outcome == Outcome::Escaped).count();
        let n_late   = level.iter().filter(|s| s.outcome == Outcome::LateTransfer).count();
        let est_thresh = capture_est_thresh(mu);
        let n_flyby  = level.iter().filter(|s|
            s.outcome == Outcome::Captured && s.est_orbits < est_thresh
        ).count();
        let n_cap    = level.iter().filter(|s|
            s.outcome == Outcome::Captured && s.est_orbits >= est_thresh
        ).count();

        writeln!(sweep,
            "{:.4},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            sigma_scale, n, n_cap, n_late, n_flyby, n_crash, n_earth, n_esc,
            n_cap   as f64 / n as f64,
            n_late  as f64 / n as f64,
            n_flyby as f64 / n as f64,
            n_crash as f64 / n as f64,
            n_earth as f64 / n as f64,
            n_esc   as f64 / n as f64,
        ).unwrap();
    }
    fs::write(&sweep_path, &sweep).expect("write stats_sweep.csv failed");
    eprintln!("  Saved {sweep_path}");

    // ── Write animation ensemble (stratified subset) ──────────────────────────
    write_anim_ensemble(
        &all_samples, nominal_ic, theta_sun,
        mu, moon_x, r_hill, r_moon_nd, l_km, &tag_suffix,
    );
}

// ════════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════════

const R_PARK: f64 = (6_371.0 + 378.0) / 384_400.0;

/// Box-Muller transform: two uniform [0,1] samples → two standard-normal samples.
fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    let r     = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// LCG RNG — deterministic, portable, no external crates.
struct SimpleLcg { state: u64 }
impl SimpleLcg {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Returns a value strictly in (0, 1) — avoids ln(0) in Box-Muller.
    fn next_f64_nonzero(&mut self) -> f64 {
        let v = self.next_f64();
        if v < 1e-15 { 1e-15 } else { v }
    }
}

// ── IC loaders (identical to wsb_sensitivity.rs) ─────────────────────────────

struct HitSeed {
    theta_deg:     f64,
    theta_sun_deg: f64,
    r_apogee_nd:   f64,
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
        let theta_deg:     f64 = c[10].parse().unwrap_or(f64::NAN);
        let theta_sun_deg: f64 = c[11].parse().ok()?;
        let r_apogee_nd:   f64 = c[12].parse().unwrap_or(f64::NAN);
        let est_orbits:    f64 = c[9].parse().unwrap_or(f64::NAN);
        return Some(([x, y, z, vx, vy, vz], theta_deg, theta_sun_deg, r_apogee_nd, est_orbits));
    }
    None
}

/// Matches wsb_sensitivity.rs: read all "refined" rows, sort by est_orbits
/// descending, return the target_id-th best (1-indexed).
fn load_hit(target_id: usize) -> Option<HitSeed> {
    let content = fs::read_to_string(SUMMARY_CSV).ok()?;
    let mut hits: Vec<(f64, f64, f64, f64)> = Vec::new(); // (est_orbits, theta, theta_sun, r_apo)
    for line in content.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 13 { continue; }
        if c[1].trim() != "refined" { continue; }
        let theta_deg:     f64 = c[2].parse().unwrap_or(f64::NAN);
        let theta_sun_deg: f64 = c[3].parse().unwrap_or(f64::NAN);
        let r_apogee_nd:   f64 = c[4].parse().unwrap_or(f64::NAN);
        let est_orbits:    f64 = c[7].parse().unwrap_or(0.0);
        if theta_deg.is_nan() { continue; }
        hits.push((est_orbits, theta_deg, theta_sun_deg, r_apogee_nd));
    }
    hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let (_, theta_deg, theta_sun_deg, r_apogee_nd) = *hits.get(target_id - 1)?;
    Some(HitSeed { theta_deg, theta_sun_deg, r_apogee_nd })
}

// ════════════════════════════════════════════════════════════════════════════════
// Animation ensemble export
// ════════════════════════════════════════════════════════════════════════════════

/// Evenly-spaced selection of up to `n` entries from `pool`.
fn anim_pick_n<'a>(pool: &[&'a SampleRecord], n: usize) -> Vec<&'a SampleRecord> {
    if pool.is_empty() || n == 0 { return Vec::new(); }
    let take = n.min(pool.len());
    if take == 1 { return vec![pool[0]]; }
    if take >= pool.len() { return pool.to_vec(); }
    (0..take).map(|i| pool[i * (pool.len() - 1) / (take - 1)]).collect()
}

/// Clip, format, and append one trajectory to `csv` in sensitivity_ensemble format.
fn anim_write_traj(
    csv:        &mut String,
    run_id:     usize,
    traj:       &[Step3d],
    outcome_str: &str,
    n_orbits:   usize,
    est_orbits: f64,
    min_alt_km: f64,
    is_nominal: u8,
    moon_x:     f64,
    r_hill:     f64,
    r_moon_nd:  f64,
    dmag:       f64,
    dtheta_deg: f64,
    dtsun_deg:  f64,
) {
    // Truncation gate
    let hill_t   = traj.iter().find(|s| {
        let dx = s.x - moon_x;
        (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
    }).map(|s| s.time);
    let t_save   = hill_t.map(|t| t + ANIM_SAVE_WIN).unwrap_or(T_PROP);

    // Moon-crash surface clip (same logic as wsb_sensitivity)
    let t_crash = if outcome_str == "moon_crash" {
        traj.iter().find(|s| {
            let dx = s.x - moon_x;
            (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_moon_nd
        }).map(|s| s.time).unwrap_or_else(|| {
            traj.windows(2).find_map(|w| {
                let (a, b) = (&w[0], &w[1]);
                let dx = b.x-a.x; let dy = b.y-a.y; let dz = b.z-a.z;
                let ax = a.x-moon_x; let ay = a.y; let az = a.z;
                let ca = dx*dx+dy*dy+dz*dz;
                let cb = 2.0*(ax*dx+ay*dy+az*dz);
                let cc = ax*ax+ay*ay+az*az - r_moon_nd*r_moon_nd;
                let disc = cb*cb - 4.0*ca*cc;
                if disc < 0.0 || ca < 1e-30 { return None; }
                let t1 = (-cb - disc.sqrt()) / (2.0*ca);
                let t2 = (-cb + disc.sqrt()) / (2.0*ca);
                let th = if t1 >= 0.0 && t1 <= 1.0 { t1 }
                         else if t2 >= 0.0 && t2 <= 1.0 { t2 }
                         else { return None; };
                Some(a.time + th * (b.time - a.time))
            }).unwrap_or(f64::MAX)
        })
    } else {
        f64::MAX
    };

    let t_stop = t_save.min(t_crash);

    for s in traj.iter().take_while(|s| s.time <= t_stop) {
        writeln!(csv,
            "{run_id},{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},\
             {outcome_str},{n_orbits},{:.3},{:.1},{is_nominal},\
             {:.8},0.00000000,0.00000000,{:.6},{:.6}",
            s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz,
            est_orbits, min_alt_km,
            dmag, dtheta_deg, dtsun_deg,
        ).unwrap();
    }
    // NaN metadata row (read by Python as df_meta)
    writeln!(csv,
        "{run_id},NaN,NaN,NaN,NaN,NaN,NaN,NaN,\
         {outcome_str},{n_orbits},{:.3},{:.1},{is_nominal},\
         {:.8},0.00000000,0.00000000,{:.6},{:.6}",
        est_orbits, min_alt_km,
        dmag, dtheta_deg, dtsun_deg,
    ).unwrap();
}

/// Re-propagates a stratified subset of the 20 k MC samples at animation resolution
/// and writes `sensitivity_ensemble<tag>.csv` in the format expected by
/// `plot_wsb_sensitivity_anim.py`.
fn write_anim_ensemble(
    samples:           &[SampleRecord],
    nominal_ic:        [f64; 6],
    nominal_theta_sun: f64,
    mu:                f64,
    moon_x:            f64,
    r_hill:            f64,
    r_moon_nd:         f64,
    l_km:              f64,
    tag_suffix:        &str,
) {
    // ── Propagate nominal first (needed for crash-time filter) ───────────────
    let nom_bcr  = Bcr4bpParams::earth_moon_sun(nominal_theta_sun);
    let nom_traj = propagate_bcr4bp(mu, nom_bcr, nominal_ic, T_PROP, ANIM_LOG_DT, RTOL, ATOL);
    let nom_cap  = detect_capture(&nom_traj, mu, MIN_CAPTURE_TIME);
    let nom_min  = nom_traj.iter().map(|s| {
        let dx = s.x - moon_x; (dx*dx + s.y*s.y + s.z*s.z).sqrt()
    }).fold(f64::MAX, f64::min);

    // Nominal Hill entry time — used to filter crash trajectories that arrive
    // on a similar timeline to the nominal (~90 days).
    let t_hill_nom: f64 = nom_traj.iter().find(|s| {
        let dx = s.x - moon_x;
        (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
    }).map(|s| s.time).unwrap_or(f64::NAN);

    // Tolerance: accept crashes arriving up to 10 days after nominal Hill entry.
    let crash_deadline = if t_hill_nom.is_nan() { f64::MAX }
                         else { t_hill_nom + 10.0 * 86_400.0 / T_STAR };

    // ── Stratify σ=1.0 samples by display outcome ─────────────────────────────
    let base: Vec<&SampleRecord> = samples.iter()
        .filter(|s| (s.sigma_scale - 1.0).abs() < 1e-9)
        .collect();

    // Crash pool: only trajectories that arrive at the Moon on a similar
    // timeline to the nominal (hill_entry_nd ≤ nominal + 10 days).
    let crash_pool: Vec<&SampleRecord> = base.iter().copied()
        .filter(|s| {
            s.outcome == Outcome::MoonCrash
                && !s.hill_entry_nd.is_nan()
                && s.hill_entry_nd <= crash_deadline
        })
        .collect();
    // Flyby / capture pools: same timing gate — only trajectories that enter
    // the Hill sphere within 10 days of nominal are shown in the animation.
    // Split by est_orbits vs the circular-orbit-at-r_hill threshold (≈ 0.577).
    let est_thresh = capture_est_thresh(mu);
    let flyby_pool: Vec<&SampleRecord> = base.iter().copied()
        .filter(|s| {
            s.outcome == Outcome::Captured && s.est_orbits < est_thresh
                && !s.hill_entry_nd.is_nan()
                && s.hill_entry_nd <= crash_deadline
        })
        .collect();
    let cap_pool: Vec<&SampleRecord> = base.iter().copied()
        .filter(|s| {
            s.outcome == Outcome::Captured && s.est_orbits >= est_thresh
                && !s.hill_entry_nd.is_nan()
                && s.hill_entry_nd <= crash_deadline
        })
        .collect();
    // Miss pool: pure escaped trajectories only — EarthCrash and LateTransfer excluded.
    let miss_pool: Vec<&SampleRecord> = base.iter().copied()
        .filter(|s| s.outcome == Outcome::Escaped)
        .collect();

    let sel_crash = anim_pick_n(&crash_pool, EXPORT_N_CRASH);
    let sel_flyby = anim_pick_n(&flyby_pool, EXPORT_N_FLYBY);
    let sel_cap   = anim_pick_n(&cap_pool,   EXPORT_N_CAP);
    let sel_miss  = anim_pick_n(&miss_pool,  EXPORT_N_MISS);

    eprintln!(
        "\n  Anim ensemble — crash {}/{} flyby {}/{} cap {}/{} miss {}/{}  (re-propagating ...)",
        sel_crash.len(), crash_pool.len(),
        sel_flyby.len(), flyby_pool.len(),
        sel_cap.len(),   cap_pool.len(),
        sel_miss.len(),  miss_pool.len(),
    );

    // ── Build CSV ─────────────────────────────────────────────────────────────
    let csv_path = format!("{OUT_DIR}/sensitivity_ensemble{tag_suffix}.csv");
    let mut csv = String::from(
        "run_id,time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,\
         outcome,n_orbits,est_orbits,min_alt_km,is_nominal,\
         dmag_frac,dpitch_rad,dyaw_rad,dtheta_deg,dtsun_deg\n"
    );

    // Nominal (run_id = 0, is_nominal = 1)
    anim_write_traj(
        &mut csv, 0, &nom_traj, "captured",
        nom_cap.n_periapsis,
        nom_cap.max_capture_interval / LUNAR_PERIOD_ND,
        nom_min * l_km - R_MOON_KM, 1,
        moon_x, r_hill, r_moon_nd,
        0.0, 0.0, 0.0,
    );

    // ── Write perturbed trajectories ─────────────────────────────────────────
    // Crash bucket is written separately so we can re-verify at ANIM resolution.
    // The stats run uses coarse LOG_DT output; the internal integrator has no
    // collision detection and propagates through the Moon as a point mass.  A
    // trajectory can pass THROUGH the Moon mathematically and re-emerge on the
    // other side, then be captured or escape.  The stats output may catch one
    // point inside r_moon (real dip OR dense-polynomial artifact) without the
    // ANIM output reproducing it.  We confirm via step-level AND segment-level
    // check on the fine ANIM trajectory before writing as moon_crash.
    let mut run_id = 1usize;
    let mut n_crash_skipped = 0usize;

    for rec in sel_crash.iter() {
        let bcr  = Bcr4bpParams::earth_moon_sun(rec.theta_sun_p);
        let traj = propagate_bcr4bp(mu, bcr, rec.ic, T_PROP, ANIM_LOG_DT, RTOL, ATOL);

        // Step-level confirmation
        let step_hit = traj.iter().any(|s| {
            let dx = s.x - moon_x;
            (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_moon_nd
        });
        // Segment-level confirmation (valid at ANIM_LOG_DT ≈ 7 h — much less
        // prone to false positives than at STATS_LOG_DT ≈ 44 h)
        let seg_hit = !step_hit && traj.windows(2).any(|w| {
            let (a, b) = (&w[0], &w[1]);
            let ra = { let dx=a.x-moon_x; (dx*dx+a.y*a.y+a.z*a.z).sqrt() };
            let rb = { let dx=b.x-moon_x; (dx*dx+b.y*b.y+b.z*b.z).sqrt() };
            if ra > 2.0*r_moon_nd && rb > 2.0*r_moon_nd { return false; }
            let dx=b.x-a.x; let dy=b.y-a.y; let dz=b.z-a.z;
            let ax=a.x-moon_x; let ay=a.y; let az=a.z;
            let ca=dx*dx+dy*dy+dz*dz; let cb=2.0*(ax*dx+ay*dy+az*dz);
            let cc=ax*ax+ay*ay+az*az-r_moon_nd*r_moon_nd;
            let disc=cb*cb-4.0*ca*cc;
            if disc < 0.0 || ca < 1e-30 { return false; }
            let t1=(-cb-disc.sqrt())/(2.0*ca); let t2=(-cb+disc.sqrt())/(2.0*ca);
            (t1>=0.0&&t1<=1.0)||(t2>=0.0&&t2<=1.0)
        });

        if !step_hit && !seg_hit {
            n_crash_skipped += 1;
            eprintln!("  [skip] crash stats_run={} not confirmed at ANIM res \
                       (min_alt_km_stats={:.0}) — likely ODE pass-through artifact",
                rec.run_id, rec.min_alt_km);
            continue;
        }

        anim_write_traj(
            &mut csv, run_id, &traj, "moon_crash",
            rec.n_orbits, rec.est_orbits, rec.min_alt_km, 0,
            moon_x, r_hill, r_moon_nd,
            rec.dv_mag_actual, rec.theta_actual_deg, rec.theta_sun_actual_deg,
        );
        run_id += 1;
    }
    if n_crash_skipped > 0 {
        eprintln!("  [{n_crash_skipped} crash(es) skipped — ODE pass-through artifacts]");
    }

    // Flyby, captured, miss — no confirmation needed (outcome is dwell-time based)
    for &(sel, out_str) in &[
        (sel_flyby.as_slice(), "captured"),
        (sel_cap.as_slice(),   "captured"),
        (sel_miss.as_slice(),  "escaped"),
    ] {
        for rec in sel.iter() {
            let bcr  = Bcr4bpParams::earth_moon_sun(rec.theta_sun_p);
            let traj = propagate_bcr4bp(mu, bcr, rec.ic, T_PROP, ANIM_LOG_DT, RTOL, ATOL);
            anim_write_traj(
                &mut csv, run_id, &traj, out_str,
                rec.n_orbits, rec.est_orbits, rec.min_alt_km, 0,
                moon_x, r_hill, r_moon_nd,
                rec.dv_mag_actual, rec.theta_actual_deg, rec.theta_sun_actual_deg,
            );
            run_id += 1;
        }
    }

    fs::write(&csv_path, &csv).expect("write sensitivity_ensemble.csv failed");
    eprintln!("  Saved {csv_path}  ({} trajectories total)", run_id - 1);
}
