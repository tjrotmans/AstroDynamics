//! wsb_sensitivity — micro-perturbation ensemble around a single refined hit.
//!
//! Loads one hit from refine_summary.csv, then propagates N_PERTURB trajectories
//! with tiny random perturbations applied to the injection state vector.  All
//! trajectories are saved to out/wsb_sensitivity/sensitivity_ensemble.csv so
//! the Python plotting script can colour them by outcome (capture / crash / escape).
//!
//! Usage:
//!   # Best natural-orbit solution (from maxhifi optimisation phase):
//!   cargo run -p lunar_trajectories --bin wsb_sensitivity --release -- --hifi --tag best_orbits
//!
//!   # Manual TLI parameters:
//!   cargo run -p lunar_trajectories --bin wsb_sensitivity --release -- \
//!       --theta 205.549 --r-apogee 3.9 --theta-sun 27.019 --tag best_orbits
//!
//!   # Legacy: by hit rank from refine_summary.csv:
//!   cargo run -p lunar_trajectories --bin wsb_sensitivity --release -- --hit 2
//!
//! Outputs (all in out/wsb_sensitivity/):
//!   sensitivity_ensemble.csv   — all trajectories + outcome labels
//!   sensitivity_summary.txt    — human-readable breakdown

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

/// Number of perturbed trajectories to propagate (plus 1 nominal = N_PERTURB+1 total).
/// Lower values → smaller CSV/HTML.  100 is a good default for visualisation.
const N_PERTURB: usize = 200;

/// TLI ΔV magnitude error (fractional, 1-sigma).  1e-3 = ±0.1% thrust error.
const DV_MAG_SIGMA: f64 = 1e-3;

/// TLI pointing error (radians, 1-sigma) applied independently to pitch (in-plane)
/// and yaw (out-of-plane).  1.75e-3 rad ≈ ±0.1°.
const DV_DIR_SIGMA: f64 = 1.75e-3;

/// Burn timing error (degrees, 1-sigma) — shifts the parking-orbit angle at ignition.
/// On a ~92-min LEO parking orbit (ω ≈ 0.065°/s), 0.2° ≈ 3 s of burn timing error.
/// Represents ground-commanded burn execution uncertainty.
const THETA_SIGMA_DEG: f64 = 0.2;

/// Launch-window error (degrees, 1-sigma) — shifts the Sun angle at departure.
/// Independent of THETA_SIGMA_DEG: represents choosing a different launch time
/// while hitting the same parking-orbit burn point.
/// Sun moves at 360°/29.53 days ≈ 0.51°/hr, so 3° ≈ ±5.9 hr of launch window.
const THETA_SUN_SIGMA_DEG: f64 = 3.0;

/// Propagation time budget [nd].  ~350 days — longer than refine (20π≈273 d) to
/// capture long-transfer BCR4BP trajectories where Hill entry can be at 40-50 nd.
const T_PROP: f64 = 25.0 * PI;

/// Integrator tolerances — tight, same as the hifi reprop pass.
const LOG_DT: f64 = 0.002;   // ~7 h — finer output for smooth animation (larger CSV)
const RTOL:   f64 = 1e-10;
const ATOL:   f64 = 1e-12;

/// Minimum capture dwell to count as a capture [nd].
const MIN_CAPTURE_TIME: f64 = 0.15;

const LUNAR_PERIOD_ND: f64 = 2.0 * PI;
const R_MOON_KM: f64 = 1_737.4;

const T_STAR: f64 = 375_700.0;   // [s] — EM non-dimensional time unit

const OUT_DIR:      &str = "out/wsb";
const SUMMARY_CSV:  &str = "out/wsb/refine_summary.csv";
const HIFI_CSV:     &str = "out/wsb/solution_hifi.csv";

// ════════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
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

/// Outcome of a single trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Outcome {
    Captured,    // ≥ MIN_CAPTURE_ORBITS inside Hill sphere
    MoonCrash,   // got inside Hill sphere but hit the Moon surface
    Escaped,     // never stayed in Hill sphere long enough
}

impl Outcome {
    fn label(self) -> &'static str {
        match self {
            Outcome::Captured  => "captured",
            Outcome::MoonCrash => "moon_crash",
            Outcome::Escaped   => "escaped",
        }
    }
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

    // ── Parse CLI arguments ───────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let hit_id: usize = args.iter()
        .position(|a| a == "--hit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);   // default: hit 2
    // --tag <label>  →  outputs sensitivity_ensemble_<label>.csv etc.
    //                   Use this to avoid overwriting a previous sweep.
    let tag: String = args.iter()
        .position(|a| a == "--tag")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_default();
    let tag_suffix = if tag.is_empty() {
        String::new()
    } else {
        format!("_{tag}")
    };
    // --hifi               →  use solution_hifi.csv (the maxhifi-optimised solution)
    //                         This is the recommended way to target the best natural-orbit solution.
    // --theta DEG          →  manual injection angle [deg] + --r-apogee ND; uses tli_injection_ic
    // --r-apogee ND        →  apogee radius [nd] for the TLI IC (used with --theta)
    // --theta-sun DEG      →  BCR4BP Sun angle (used with --hifi / --theta / --ic)
    // --ic "x y z vx vy vz"  →  direct state vector (low-precision fallback; prefer --hifi)
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

    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!("║     WSB Sensitivity — Micro-perturbation Ensemble    ║");
    eprintln!("╚══════════════════════════════════════════════════════╝");
    eprintln!("  Tag          : {}", if tag.is_empty() { "(none — default filenames)".to_string() } else { tag.clone() });
    eprintln!("  N_PERTURB    : {N_PERTURB}  (+1 nominal = {} total)", N_PERTURB + 1);
    eprintln!("  ΔV mag  σ    : {DV_MAG_SIGMA:.0e}  (fractional engine thrust error)");
    eprintln!("  ΔV dir  σ    : {:.4}°  (pitch and yaw pointing error)", DV_DIR_SIGMA.to_degrees());
    eprintln!("  θ_inject σ   : {THETA_SIGMA_DEG:.4}°  (burn timing within parking orbit)");
    eprintln!("  θ_sun σ      : {THETA_SUN_SIGMA_DEG:.4}°  (launch window / day shift)");
    eprintln!("  Integrator   : dt={LOG_DT}  RTOL={RTOL}  ATOL={ATOL}");
    eprintln!();

    // ── Build nominal IC ──────────────────────────────────────────────────────
    let (nominal_ic, theta_sun, theta_nom_rad, r_apogee_nd) = if use_hifi {
        // Preferred path: read solution_hifi.csv (the maxhifi-optimised result).
        // Uses the stored IC at t=0 directly — do NOT reconstruct via tli_injection_ic,
        // because WSB trajectories are chaotic and the Keplerian approximation changes
        // the capture outcome entirely.
        let (h_ic, h_theta, h_theta_sun, h_r_apo, h_orbits) = load_hifi().unwrap_or_else(|| {
            eprintln!("  [error] could not read {HIFI_CSV}");
            eprintln!("  Run wsb_optimize (maxhifi phase) first.");
            std::process::exit(1);
        });
        let ts = theta_sun_override.map(|d| d.to_radians()).unwrap_or(h_theta_sun.to_radians());
        eprintln!("  IC source    : --hifi  (solution_hifi.csv, stored t=0 state)");
        eprintln!("  θ_inject     : {:.4}°", h_theta);
        eprintln!("  r_apogee     : {:.4} nd", h_r_apo);
        eprintln!("  θ_sun        : {:.4}°", ts.to_degrees());
        eprintln!("  est_orbits   : {:.3}", h_orbits);
        (h_ic, ts, h_theta.to_radians(), h_r_apo)
    } else if let (Some(theta_deg), Some(r_apo)) = (tli_theta_deg, tli_r_apogee) {
        // Manual TLI mode: --theta DEG --r-apogee ND --theta-sun DEG
        let ts = theta_sun_override.unwrap_or(0.0).to_radians();
        let ic = tli_injection_ic(mu, R_PARK, r_apo, theta_deg.to_radians())
            .expect("Bad TLI IC from --theta/--r-apogee");
        eprintln!("  IC source    : --theta/--r-apogee (tli_injection_ic)");
        eprintln!("  θ_inject     : {:.4}°", theta_deg);
        eprintln!("  r_apogee     : {:.4} nd", r_apo);
        eprintln!("  θ_sun        : {:.4}°", ts.to_degrees());
        (ic, ts, theta_deg.to_radians(), r_apo)
    } else if let Some(ref ic_s) = ic_str {
        // Low-precision fallback: --ic "x y z vx vy vz" --theta-sun DEG
        let vals: Vec<f64> = ic_s.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() != 6 {
            eprintln!("  [error] --ic requires exactly 6 values: x y z vx vy vz");
            std::process::exit(1);
        }
        let ic: [f64; 6] = [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]];
        let ts = theta_sun_override.unwrap_or(0.0).to_radians();
        eprintln!("  IC source    : --ic (direct state vector — low precision)");
        eprintln!("  θ_sun        : {:.3}°", ts.to_degrees());
        let r_apo = tli_r_apogee.unwrap_or_else(|| {
            eprintln!("  [error] --ic mode requires --r-apogee ND for ΔV perturbation");
            std::process::exit(1);
        });
        let theta_d = ic[1].atan2(ic[0]);
        (ic, ts, theta_d, r_apo)
    } else {
        // Legacy mode: load from refine_summary.csv
        eprintln!("  Hit ID       : {hit_id}");
        let seed = load_hit(hit_id).unwrap_or_else(|| {
            eprintln!("  [error] hit {hit_id} not found in {SUMMARY_CSV}");
            eprintln!("  Run wsb_refine first, then check available hits with --reprop");
            std::process::exit(1);
        });
        eprintln!("  Loaded hit {}: seed={} θ={:.3}° θ_sun={:.3}° r_apo={:.3} nd  \
                   orbits={:.2}  dθ={:+.4}° dθ_sun={:+.4}°",
            seed.hit_id, seed.seed_id,
            seed.theta_deg, seed.theta_sun_deg, seed.r_apogee_nd,
            seed.est_orbits, seed.dtheta_deg, seed.dsun_deg);
        eprintln!();
        let theta     = seed.theta_deg.to_radians();
        let ts        = seed.theta_sun_deg.to_radians();
        let r_apogee  = seed.r_apogee_nd;
        let ic = tli_injection_ic(mu, R_PARK, r_apogee, theta)
            .expect("Bad nominal IC");
        (ic, ts, theta, r_apogee)
    };

    eprintln!("  Nominal IC: x={:.6}  y={:.6}  z={:.6}  vx={:.6}  vy={:.6}  vz={:.6}",
        nominal_ic[0], nominal_ic[1], nominal_ic[2], nominal_ic[3], nominal_ic[4], nominal_ic[5]);
    eprintln!();

    // ── Reference orbital parameters for perturbations ───────────────────────
    // Earth sits at x = −mu in the rotating frame (not at the barycenter origin).
    // All parking-orbit geometry must be measured from Earth, not from the barycenter.
    // We derive r_park_ref and theta_nom_ref directly from the nominal IC so that
    // the perturbed ICs are exactly consistent with the nominal for every IC source
    // (--hifi loads a stored IC that may differ slightly from R_PARK; --hit and
    // --theta produce ICs from tli_injection_ic which always use R_PARK exactly).
    let x_earth = -mu;
    let dx_earth = nominal_ic[0] - x_earth;
    let dy_earth = nominal_ic[1];
    let r_park_ref    = (dx_earth*dx_earth + dy_earth*dy_earth + nominal_ic[2]*nominal_ic[2]).sqrt();
    let theta_nom_ref = dy_earth.atan2(dx_earth);
    let v_circ_ref    = (mu_earth / r_park_ref).sqrt();

    // ΔV magnitude from vis-viva (parking orbit → transfer apogee).
    // r_apogee_nd is always a stored design variable in the WSB pipeline.
    let a_tr = (r_park_ref + r_apogee_nd) / 2.0;
    let dv_nom_mag = (mu_earth * (2.0 / r_park_ref - 1.0 / a_tr)).sqrt() - v_circ_ref;
    eprintln!("  r_park_ref   : {:.6} nd  (from IC; R_PARK design = {R_PARK:.6})", r_park_ref);
    eprintln!("  θ_nom_ref    : {:.4}°  (from Earth center; design = {:.4}°)",
        theta_nom_ref.to_degrees(), theta_nom_rad.to_degrees());
    eprintln!("  TLI ΔV nom   : {:.6} nd  ({:.1} m/s)",
        dv_nom_mag, dv_nom_mag * params.v_star);
    eprintln!();

    // ── Build perturbation ensemble ───────────────────────────────────────────
    let mut rng = SimpleLcg::new(0xDEAD_BEEF_1234_5678);

    // (ic, theta_sun_p, run_id) — sun angle is per-trajectory for launch-window sweep
    let mut ics: Vec<([f64; 6], f64, usize)> = Vec::with_capacity(N_PERTURB + 1);
    // (dmag_frac, dpitch_rad, dyaw_rad, dtheta_deg, dtsun_deg) — for CSV + group colouring
    let mut perturbs: Vec<(f64, f64, f64, f64, f64)> = Vec::with_capacity(N_PERTURB + 1);
    ics.push((nominal_ic, theta_sun, 0));
    perturbs.push((0.0, 0.0, 0.0, 0.0, 0.0));

    for run_id in 1..=N_PERTURB {
        let (g0, g1) = box_muller(rng.next_f64_nonzero(), rng.next_f64());
        let (g2, g3) = box_muller(rng.next_f64_nonzero(), rng.next_f64());
        let (g4, g5) = box_muller(rng.next_f64_nonzero(), rng.next_f64());
        let _ = g5;
        let dmag   = g0 * DV_MAG_SIGMA;
        let dpitch = g1 * DV_DIR_SIGMA;
        let dyaw   = g2 * DV_DIR_SIGMA;
        let dtheta = g3 * THETA_SIGMA_DEG.to_radians();
        let dtsun  = g4 * THETA_SUN_SIGMA_DEG.to_radians();

        let theta_p     = theta_nom_ref + dtheta;
        let theta_sun_p = theta_sun + dtsun;

        // Injection position in rotating frame — Earth-centred parking orbit.
        // px, py are measured from the barycenter (as the integrator expects).
        let px = x_earth + r_park_ref * theta_p.cos();
        let py =           r_park_ref * theta_p.sin();

        // Pre-burn rotating-frame velocity (circular orbit around Earth).
        // Rotating-frame: v_rot = v_inert − ω×r_bary  →  add [+py, −px, 0] (ω=1 nd).
        let vel_pre = [
            -v_circ_ref * theta_p.sin() + py,
            v_circ_ref * theta_p.cos() - px,
            0.0,
        ];

        // Nominal ΔV direction: prograde (tangent to orbit around Earth at theta_p)
        let dv_dir = [-theta_p.sin(), theta_p.cos(), 0.0];

        // Pitch axis: in-plane perpendicular to prograde (radially inward from Earth)
        let pitch_perp = [-theta_p.cos(), -theta_p.sin(), 0.0];

        // Perturbed ΔV direction: pitch rotates in-plane, yaw lifts out-of-plane
        let dv_dir_p = [
            dv_dir[0] + dpitch * pitch_perp[0],
            dv_dir[1] + dpitch * pitch_perp[1],
            dyaw,
        ];
        let dir_norm = (dv_dir_p[0]*dv_dir_p[0] + dv_dir_p[1]*dv_dir_p[1] + dv_dir_p[2]*dv_dir_p[2]).sqrt();
        let dv_mag_p = dv_nom_mag * (1.0 + dmag);

        // ΔV is an impulsive inertial-frame thrust: rotating-frame velocity changes
        // by exactly the inertial ΔV vector (position is unchanged during the burn).
        let ic = [
            px, py, 0.0,
            vel_pre[0] + dv_mag_p * dv_dir_p[0] / dir_norm,
            vel_pre[1] + dv_mag_p * dv_dir_p[1] / dir_norm,
            vel_pre[2] + dv_mag_p * dv_dir_p[2] / dir_norm,
        ];
        ics.push((ic, theta_sun_p, run_id));
        perturbs.push((dmag, dpitch, dyaw, dtheta.to_degrees(), dtsun.to_degrees()));
    }

    // ── Propagate all trajectories ────────────────────────────────────────────
    eprintln!("  Propagating {} trajectories …", ics.len());

    let mut ensemble: Vec<(usize, Vec<Step3d>, Outcome, usize, f64, f64)> = Vec::new();
    // Fields per entry: (run_id, traj, outcome, n_orbits, est_orbits, min_alt_km)

    for (idx, (ic, theta_sun_p, run_id)) in ics.iter().enumerate() {
        if idx % 10 == 0 {
            eprintln!("  [{}/{}] …", idx, ics.len());
        }

        let bcr = Bcr4bpParams::earth_moon_sun(*theta_sun_p);
        let traj = propagate_bcr4bp(mu, bcr, *ic, T_PROP, LOG_DT, RTOL, ATOL);

        // Moon crash check: did any point go below the Moon's surface inside Hill?

        // Minimum distance to Moon centre across all steps
        let min_moon_dist_nd = traj.iter().map(|s| {
            let dx = s.x - moon_x;
            (dx*dx + s.y*s.y + s.z*s.z).sqrt()
        }).fold(f64::MAX, f64::min);

        // Step-level only: segment (chord) check is unreliable at LOG_DT ≈ 12 min
        // because hyperbolic flyby chords cut inside the Moon sphere even when
        // the actual orbit never touches it.
        let moon_crash = min_moon_dist_nd < r_moon_nd;

        let capture    = detect_capture(&traj, mu, MIN_CAPTURE_TIME);
        let est_orbits = capture.max_capture_interval / LUNAR_PERIOD_ND; // Hill dwell / Earth-Moon period
        let n_orbits   = capture.n_periapsis;  // actual periapsis passages ≈ lunar orbits
        let min_alt_km = min_moon_dist_nd * l_km - R_MOON_KM;

        let outcome = if moon_crash {
            Outcome::MoonCrash
        } else if n_orbits >= 1 && capture.n_entries > 0 {
            Outcome::Captured
        } else {
            Outcome::Escaped
        };

        if *run_id == 0 {
            eprintln!("  Nominal → {:?}  n_orbits={n_orbits}  (hill_dwell={est_orbits:.3} EM-periods)  min_alt={min_alt_km:.0} km",
                outcome);
        }

        ensemble.push((*run_id, traj, outcome, n_orbits, est_orbits, min_alt_km));
    }

    // ── Outcome summary ───────────────────────────────────────────────────────
    let n_captured  = ensemble.iter().filter(|e| e.2 == Outcome::Captured).count();
    let n_crash     = ensemble.iter().filter(|e| e.2 == Outcome::MoonCrash).count();
    let n_escaped   = ensemble.iter().filter(|e| e.2 == Outcome::Escaped).count();
    eprintln!();
    eprintln!("  ── Outcome breakdown ─────────────────────────────────");
    eprintln!("  Captured   : {n_captured}");
    eprintln!("  Moon crash : {n_crash}");
    eprintln!("  Escaped    : {n_escaped}");
    eprintln!();

    // ── CRTBP+STM propagation for the nominal trajectory ────────────────────
    let nominal_traj_ref = &ensemble[0].1;
    let t_hill_nom = nominal_traj_ref.iter()
        .find(|s| {
            let dx = s.x - moon_x;
            (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
        })
        .map(|s| s.time)
        .unwrap_or(T_PROP * 0.5);
    eprintln!("  CRTBP+STM: propagating nominal to Hill entry \
               t={:.2} nd  ({:.1} d) …",
        t_hill_nom, t_hill_nom * T_STAR / 86400.0);
    let stm_result = propagate_3d_stm_full(
        mu, nominal_ic, t_hill_nom, LOG_DT, RTOL, ATOL,
    );
    let stm_csv_path = format!("{OUT_DIR}/sensitivity_stm{tag_suffix}.csv");
    match stm_result {
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
            // Close lunar approaches can cause step-size underflow in the CRTBP STM.
            // Write an empty file so downstream scripts don't crash, and continue.
            eprintln!("  WARNING: CRTBP STM integration failed ({e})");
            eprintln!("  Writing empty STM file — covariance plot will be unavailable.");
            fs::write(&stm_csv_path,
                "time_nd,phi_0_0,phi_0_1,phi_0_2,phi_0_3,phi_0_4,phi_0_5,\
                 phi_1_0,phi_1_1,phi_1_2,phi_1_3,phi_1_4,phi_1_5\n"
            ).expect("STM csv write failed");
        }
    }
    eprintln!("  Saved {stm_csv_path}");
    eprintln!();

    // ── Save ensemble CSV ─────────────────────────────────────────────────────
    // Truncate trajectories: keep up to Hill entry + 5 lunar periods inside,
    // to keep file size manageable.
    let save_window = 5.0 * LUNAR_PERIOD_ND;

    let csv_path = format!("{OUT_DIR}/sensitivity_ensemble{tag_suffix}.csv");
    let mut csv = String::from(
        "run_id,time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,\
         outcome,n_orbits,est_orbits,min_alt_km,is_nominal,\
         dmag_frac,dpitch_rad,dyaw_rad,dtheta_deg,dtsun_deg\n"
    );

    for (run_id, traj, outcome, n_orbits, est_orbits, min_alt_km) in &ensemble {
        let is_nominal = if *run_id == 0 { 1 } else { 0 };
        let (dmag_p, dpitch_p, dyaw_p, dtheta_p, dtsun_p) = perturbs[*run_id];

        // Crash clip: stop at Moon surface impact for MoonCrash trajectories.
        // The crash DETECTION above uses segment-level interpolation, so a crash can
        // occur between two saved LOG_DT steps with neither endpoint inside the Moon.
        // We therefore first look for a step-level crossing, then fall back to
        // quadratic interpolation on each segment near the Moon.
        let t_crash = if *outcome == Outcome::MoonCrash {
            // Step-level: first saved point inside the Moon surface
            let step_crash = traj.iter().find(|s| {
                let dx = s.x - moon_x;
                (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_moon_nd
            }).map(|s| s.time);

            if let Some(t) = step_crash {
                t
            } else {
                // Segment-level: quadratic (sphere–line) intersection to find
                // the exact time the trajectory crosses the Moon surface.
                traj.windows(2).find_map(|w| {
                    let (a, b) = (&w[0], &w[1]);
                    let ra = { let dx=a.x-moon_x; (dx*dx+a.y*a.y+a.z*a.z).sqrt() };
                    let rb = { let dx=b.x-moon_x; (dx*dx+b.y*b.y+b.z*b.z).sqrt() };
                    // Only check segments where at least one end is within 3×r_moon
                    if ra > 3.0*r_moon_nd && rb > 3.0*r_moon_nd { return None; }
                    // Direction vector of segment
                    let dx = b.x-a.x; let dy = b.y-a.y; let dz = b.z-a.z;
                    // Vector from Moon centre to start of segment
                    let ax = a.x-moon_x; let ay = a.y; let az = a.z;
                    // Solve |ax+t*dx|^2 = r_moon_nd^2  →  ca*t^2 + cb*t + cc = 0
                    let ca = dx*dx + dy*dy + dz*dz;
                    let cb = 2.0*(ax*dx + ay*dy + az*dz);
                    let cc = ax*ax + ay*ay + az*az - r_moon_nd*r_moon_nd;
                    let disc = cb*cb - 4.0*ca*cc;
                    if disc < 0.0 || ca < 1e-30 { return None; }
                    // Two roots — take the first one in [0, 1] (entry crossing)
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

        // Find Hill entry time for truncation
        let hill_entry_t = traj.iter().find(|s| {
            let dx = s.x - moon_x;
            (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
        }).map(|s| s.time).unwrap_or(f64::MAX);

        let t_stop = if hill_entry_t < f64::MAX {
            hill_entry_t + save_window
        } else {
            // Never entered Hill sphere: save full propagation so the animation
            // can show the complete miss trajectory without abrupt disappearance.
            T_PROP
        };

        let t_stop = t_stop.min(t_crash);

        for s in traj.iter().take_while(|s| s.time <= t_stop) {
            writeln!(csv,
                "{run_id},{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{},{n_orbits},{:.3},{:.1},{is_nominal},{:.8},{:.8},{:.8},{:.6},{:.6}",
                s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz,
                outcome.label(), est_orbits, min_alt_km,
                dmag_p, dpitch_p, dyaw_p, dtheta_p, dtsun_p,
            ).unwrap();
        }
        // NaN separator
        writeln!(csv,
            "{run_id},NaN,NaN,NaN,NaN,NaN,NaN,NaN,{},{n_orbits},{:.3},{:.1},{is_nominal},{:.8},{:.8},{:.8},{:.6},{:.6}",
            outcome.label(), est_orbits, min_alt_km,
            dmag_p, dpitch_p, dyaw_p, dtheta_p, dtsun_p,
        ).unwrap();
    }

    fs::write(&csv_path, &csv).expect("ensemble csv write failed");
    eprintln!("  Saved {csv_path}");

    // ── Save summary text ─────────────────────────────────────────────────────
    let nom_n_orbits   = ensemble[0].3;
    let nom_est_orbits = ensemble[0].4;
    let mut info = String::new();
    writeln!(info, "=== WSB Sensitivity Ensemble ===\n").unwrap();
    if use_hifi {
        writeln!(info, "IC source      : --hifi  (solution_hifi.csv — stored t=0 state)").unwrap();
    } else if tli_theta_deg.is_some() {
        writeln!(info, "IC source      : --theta/--r-apogee  (tli_injection_ic)").unwrap();
        writeln!(info, "θ_inject       : {:.4}°", tli_theta_deg.unwrap()).unwrap();
        writeln!(info, "r_apogee       : {:.4} nd", tli_r_apogee.unwrap()).unwrap();
    } else if ic_str.is_some() {
        writeln!(info, "IC source      : --ic  (direct state vector — low precision)").unwrap();
    } else {
        writeln!(info, "IC source      : --hit {hit_id}  (refine_summary.csv)").unwrap();
    }
    writeln!(info, "θ_sun          : {:.4}°", theta_sun.to_degrees()).unwrap();
    writeln!(info, "Nominal IC     : x={:.8} y={:.8} z={:.8} vx={:.8} vy={:.8} vz={:.8}",
        nominal_ic[0], nominal_ic[1], nominal_ic[2],
        nominal_ic[3], nominal_ic[4], nominal_ic[5]).unwrap();
    writeln!(info, "Tag            : {}", if tag.is_empty() { "(default)" } else { &tag }).unwrap();
    writeln!(info, "Nominal orbits : {} (periapsis passages)  /  {:.3} EM-periods hill dwell",
        nom_n_orbits, nom_est_orbits).unwrap();
    writeln!(info).unwrap();
    writeln!(info, "ΔV mag  σ      : ±{DV_MAG_SIGMA:.0e} fractional (engine thrust error)").unwrap();
    writeln!(info, "ΔV dir  σ      : ±{:.4}° (pitch and yaw pointing error)", DV_DIR_SIGMA.to_degrees()).unwrap();
    writeln!(info, "θ_inject σ     : ±{THETA_SIGMA_DEG:.4}°  (burn timing within parking orbit)").unwrap();
    writeln!(info, "θ_sun σ        : ±{THETA_SUN_SIGMA_DEG:.4}°  (launch window / day shift)").unwrap();
    writeln!(info, "N trajectories : {} ({N_PERTURB} perturbed + 1 nominal)", N_PERTURB + 1).unwrap();
    writeln!(info).unwrap();
    writeln!(info, "── Outcome breakdown ─────────────────────────────────").unwrap();
    writeln!(info, "  Captured   : {n_captured} / {}  ({:.0}%)",
        ensemble.len(), 100.0 * n_captured as f64 / ensemble.len() as f64).unwrap();
    writeln!(info, "  Moon crash : {n_crash} / {}  ({:.0}%)",
        ensemble.len(), 100.0 * n_crash as f64 / ensemble.len() as f64).unwrap();
    writeln!(info, "  Escaped    : {n_escaped} / {}  ({:.0}%)",
        ensemble.len(), 100.0 * n_escaped as f64 / ensemble.len() as f64).unwrap();

    let nominal_outcome = ensemble[0].2;
    writeln!(info).unwrap();
    writeln!(info, "Nominal outcome: {:?}", nominal_outcome).unwrap();
    if nominal_outcome == Outcome::MoonCrash {
        writeln!(info, "  ☠ Nominal trajectory impacts the Moon surface.").unwrap();
        writeln!(info, "  The ensemble shows whether nearby ICs thread the needle.").unwrap();
    }

    let info_path = format!("{OUT_DIR}/sensitivity_summary{tag_suffix}.txt");
    fs::write(&info_path, &info).expect("summary write failed");
    eprintln!("  Saved {info_path}");
    print!("{info}");
}

// ════════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════════

const R_PARK: f64 = (6_371.0 + 378.0) / 384_400.0;

fn load_hit(target_hit_id: usize) -> Option<HitSeed> {
    let content = fs::read_to_string(SUMMARY_CSV).ok()?;
    let mut hits: Vec<HitSeed> = Vec::new();
    let mut counter = 0usize;

    for line in content.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 13 { continue; }
        let source = c[1].trim();
        if source != "refined" { continue; }

        let seed_id:       usize = c[0].parse().unwrap_or(0);
        let theta_deg:     f64   = c[2].parse().unwrap_or(f64::NAN);
        let theta_sun_deg: f64   = c[3].parse().unwrap_or(f64::NAN);
        let r_apogee_nd:   f64   = c[4].parse().unwrap_or(f64::NAN);
        let est_orbits:    f64   = c[7].parse().unwrap_or(0.0);
        let dtheta_deg:    f64   = c[9].parse().unwrap_or(f64::NAN);
        let dsun_deg:      f64   = c[10].parse().unwrap_or(f64::NAN);
        if theta_deg.is_nan() { continue; }

        counter += 1;
        hits.push(HitSeed {
            hit_id: counter, seed_id,
            theta_deg, theta_sun_deg, r_apogee_nd, est_orbits,
            dtheta_deg, dsun_deg,
        });
    }

    // Sort by est_orbits descending (same as reprop mode)
    hits.sort_by(|a, b| b.est_orbits.partial_cmp(&a.est_orbits).unwrap());
    for (i, h) in hits.iter_mut().enumerate() { h.hit_id = i + 1; }

    hits.into_iter().find(|h| h.hit_id == target_hit_id)
}

/// Read the first data row (t=0) of solution_hifi.csv and return the exact
/// stored IC plus metadata.
///
/// Columns (0-indexed): time_nd=0, x_nd=1..6, hit_id=7, seed_id=8,
///   est_capture_orbits=9, theta_deg=10, theta_sun_deg=11, r_apogee_nd=12
///
/// Returns (ic_6dof, theta_deg, theta_sun_deg, r_apogee_nd, est_orbits).
/// Reading the IC directly from the file avoids re-deriving it from
/// tli_injection_ic — WSB trajectories are chaotic enough that the tiny
/// difference between the Keplerian approximation and the stored refined IC
/// completely changes the capture outcome.
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

// ── Simple deterministic LCG RNG ─────────────────────────────────────────────
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
    fn next_f64_nonzero(&mut self) -> f64 {
        let v = self.next_f64();
        if v < 1e-15 { 1e-15 } else { v }
    }
}

fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    let r     = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;
    (r * theta.cos(), r * theta.sin())
}