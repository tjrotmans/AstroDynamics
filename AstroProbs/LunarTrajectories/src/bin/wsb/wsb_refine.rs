//! wsb_refine — tight Monte Carlo refinement around GA-found solutions.
//!
//! # Strategy
//!
//! Runs AFTER wsb_optimize (GA global search).  Loads the top solutions from
//! ga_best.csv and runs a tight 3-D Monte Carlo search (θ, θ_sun, r_apogee)
//! in a small neighbourhood around each GA solution to polish and verify them.
//!
//! Objective:  ΔV_total = ΔV_TLI + ΔV_LOI_min
//!
//! Hill-entering solutions with LOI_ΔV < LOI_DV_MAX_KMS are kept, regardless
//! of capture orbit count.
//!
//! # Outputs  (all in out/wsb/)
//!
//!   mc_solutions.csv   — all captured solutions ranked by dv_total_kms
//!   refine_summary.csv — wsb_maxhifi-compatible format (source=refined rows)
//!   mc_traj.csv        — trajectories of top N_TOP_TRAJ solutions
//!
//! Usage:  cargo run -p lunar_trajectories --bin wsb_refine --release
//!         cargo run … -- --reprop [HIT_ID]   (re-propagate a specific hit)

use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::sync::OnceLock;

use rand::Rng;
use rayon::prelude::*;

use lunar_trajectories::crtbp::CrtbpParams;
use lunar_trajectories::propagator::{propagate_bcr4bp, Bcr4bpParams};
use lunar_trajectories::transfers::{
    lunar_hill_radius, detect_capture, tli_injection_ic, tli_dv, min_loi_dv,
};

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                         USER CONFIGURATION                                  ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ── Seed selection ────────────────────────────────────────────────────────────

/// Maximum GA solutions to load from ga_best.csv.
const N_SEEDS_MAX: usize = 30;

/// Minimum LOI ΔV [km/s] to accept a seed (filters out seeds that never captured).
const MIN_SEED_SCORE: f64 = 0.0;   // unused but kept for load_seeds signature

/// Minimum angular separation between seeds [deg] — diversity filter.
const SEED_MIN_SEP_DEG: f64 = 1.0;

// ── Monte Carlo search ────────────────────────────────────────────────────────

/// Tight 3-D samples per seed: (θ, θ_sun, r_apogee) — local polishing.
const N_MC_PER_SEED: usize = 1000;

/// Half-width of the θ search window around each GA seed [deg].
const DTHETA_MC: f64 = 3.0;

/// Half-width of the θ_sun search window around each GA seed [deg].
const DSUN_MC: f64 = 3.0;

/// Half-width of r_apogee perturbation around each GA seed's value [nd].
const DR_APOGEE_MC: f64 = 0.15;

/// LOI penalty applied when the spacecraft never enters the Hill sphere [nd].
/// Set high enough to rank uncaptured runs last.
const LOI_PENALTY: f64 = 2.0;

/// Maximum LOI ΔV [km/s] to include in mc_solutions.csv.
/// Replaces orbit-count filter: any Hill entry with feasible LOI is kept.
const LOI_DV_MAX_KMS: f64 = 1.0;

// ── Capture metrics (still computed, used for est_capture_orbits output) ──────

const LUNAR_PERIOD_ND: f64 = 2.0 * PI;
const MIN_CAPTURE_TIME: f64 = 0.15;

// ── Propagation ───────────────────────────────────────────────────────────────

const T_PROP:   f64 = 20.0 * PI;
const LOG_DT:   f64 = 0.06;
const RTOL:     f64 = 1e-8;
const ATOL:     f64 = 1e-10;

const LOG_DT_OUT: f64 = 0.001;
const RTOL_OUT:   f64 = 1e-12;
const ATOL_OUT:   f64 = 1e-12;

// ── Output ────────────────────────────────────────────────────────────────────

/// Number of top solutions to save with full trajectories in mc_traj.csv.
const N_TOP_TRAJ: usize = 10;

const R_PARK: f64 = (6_371.0 + 378.0) / 384_400.0;
const OUT_DIR:   &str = "out/wsb";
const SEEDS_CSV: &str = "out/wsb/ga_best.csv";

// ════════════════════════════════════════════════════════════════════════════════

static _CFG_R_PARK: OnceLock<f64> = OnceLock::new();
fn cfg_r_park() -> f64 {
    *_CFG_R_PARK.get_or_init(|| {
        std::env::var("WSB_R_PARK_ND").ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(R_PARK)
    })
}

// ════════════════════════════════════════════════════════════════════════════════
// Comparison helpers — load earlier pipeline stages for the summary table
// ════════════════════════════════════════════════════════════════════════════════

const FAMILY_CSV:     &str = "out/wsb/family_analysis.csv";
const GA_SOL_CSV:     &str = "out/wsb/ga_solutions.csv";

struct GridRow {
    theta:      f64,
    theta_sun:  f64,
    r_apo:      f64,
    dv_tli_kms: f64,
    alpha_deg:  f64,
    t_days:     f64,
    family:     String,
}

struct GaRow {
    theta:       f64,
    theta_sun:   f64,
    r_apo:       f64,
    dv_tli_kms:  f64,
    dv_loi_kms:  f64,
    dv_total_kms: f64,
    alpha_deg:   f64,
    t_days:      f64,
    is_pareto:   bool,
}

fn load_grid_best(n: usize) -> Vec<GridRow> {
    let content = match fs::read_to_string(FAMILY_CSV) {
        Ok(s)  => s,
        Err(_) => return Vec::new(),
    };
    // columns: 0:theta  1:theta_sun  2:dv_kms(TLI)  3:t_days  4:capture_dur
    //          5:min_alt  6:alpha_deg  7:apogee_nd  8:r_apo_target  9:n_entries
    //          10:est_orbits  11:family  12:pareto_rank  13:score  14:crashed_moon
    let mut rows: Vec<GridRow> = Vec::new();
    for line in content.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 15 { continue; }
        if c[14].trim() != "0" { continue; }
        let dv:    f64 = c[2].parse().unwrap_or(f64::NAN);
        let theta: f64 = c[0].parse().unwrap_or(f64::NAN);
        if dv.is_nan() || theta.is_nan() { continue; }
        rows.push(GridRow {
            theta,
            theta_sun:  c[1].parse().unwrap_or(f64::NAN),
            r_apo:      c[8].parse().unwrap_or(f64::NAN),
            dv_tli_kms: dv,
            alpha_deg:  c[6].parse().unwrap_or(f64::NAN),
            t_days:     c[3].parse().unwrap_or(f64::NAN),
            family:     c[11].trim().to_string(),
        });
    }
    rows.sort_by(|a, b| a.dv_tli_kms.partial_cmp(&b.dv_tli_kms)
        .unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<GridRow> = Vec::new();
    'outer: for row in rows {
        for d in &out {
            let dt = (row.theta - d.theta).rem_euclid(360.0);
            if dt.min(360.0 - dt) < 2.0 { continue 'outer; }
        }
        out.push(row);
        if out.len() >= n { break; }
    }
    out
}

fn load_ga_best(n: usize) -> Vec<GaRow> {
    let content = match fs::read_to_string(GA_SOL_CSV) {
        Ok(s)  => s,
        Err(_) => return Vec::new(),
    };
    // columns: 0:rank  1:theta  2:theta_sun  3:r_apo  4:dv_tli  5:dv_loi
    //          6:dv_total  7:est_orbits  8:t_days  9:alpha_deg  10:pareto_optimal
    let mut rows: Vec<GaRow> = Vec::new();
    for line in content.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 10 { continue; }
        let dv: f64 = c[6].parse().unwrap_or(f64::NAN);
        if dv.is_nan() { continue; }
        rows.push(GaRow {
            theta:        c[1].parse().unwrap_or(f64::NAN),
            theta_sun:    c[2].parse().unwrap_or(f64::NAN),
            r_apo:        c[3].parse().unwrap_or(f64::NAN),
            dv_tli_kms:   c[4].parse().unwrap_or(f64::NAN),
            dv_loi_kms:   c[5].parse().unwrap_or(f64::NAN),
            dv_total_kms: dv,
            alpha_deg:    c[9].parse().unwrap_or(f64::NAN),
            t_days:       c[8].parse().unwrap_or(f64::NAN),
            is_pareto:    c[10].trim() == "1",
        });
    }
    rows.truncate(n);
    rows
}

// ════════════════════════════════════════════════════════════════════════════════
// Data structures
// ════════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct Seed {
    seed_id:       usize,
    theta_deg:     f64,
    theta_sun_deg: f64,
    r_apogee_nd:   f64,
    est_orbits:    f64,
}

#[derive(Clone)]
struct McSolution {
    seed_id:         usize,
    theta_deg:       f64,
    theta_sun_deg:   f64,
    r_apogee_nd:     f64,
    dv_total_nd:     f64,
    dv_tli_kms:      f64,
    dv_loi_kms:      f64,
    dv_total_kms:    f64,
    est_capture_orbits: f64,
    t_transfer_days: f64,
    alpha_deg:       f64,
}

// ════════════════════════════════════════════════════════════════════════════════
// --reprop mode (re-propagate a specific hit at high fidelity)
// ════════════════════════════════════════════════════════════════════════════════

struct HitRow {
    hit_id:        usize,
    seed_id:       usize,
    theta_deg:     f64,
    theta_sun_deg: f64,
    r_apogee_nd:   f64,
    est_orbits:    f64,
}

fn load_hit_rows() -> Vec<HitRow> {
    let path = format!("{OUT_DIR}/refine_summary.csv");
    let content = match fs::read_to_string(&path) {
        Ok(s)  => s,
        Err(e) => { eprintln!("  Cannot read {path}: {e}"); return Vec::new(); }
    };
    let mut rows = Vec::new();
    let mut counter = 0usize;
    for line in content.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 12 { continue; }
        if c[1].trim() != "refined" { continue; }
        let seed_id:       usize = c[0].parse().unwrap_or(0);
        let theta_deg:     f64   = c[2].parse().unwrap_or(f64::NAN);
        let theta_sun_deg: f64   = c[3].parse().unwrap_or(f64::NAN);
        let r_apogee_nd:   f64   = c[4].parse().unwrap_or(f64::NAN);
        let est_orbits:    f64   = c[7].parse().unwrap_or(0.0);
        if theta_deg.is_nan() { continue; }
        counter += 1;
        rows.push(HitRow { hit_id: counter, seed_id, theta_deg, theta_sun_deg, r_apogee_nd, est_orbits });
    }
    rows.sort_by(|a, b| b.est_orbits.partial_cmp(&a.est_orbits).unwrap());
    for (i, r) in rows.iter_mut().enumerate() { r.hit_id = i + 1; }
    rows
}

fn reprop_mode(_params: &CrtbpParams, mu: f64, hit_id: Option<usize>) {
    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!("║     WSB Refinement — Re-propagate Solution           ║");
    eprintln!("╚══════════════════════════════════════════════════════╝");

    let rows = load_hit_rows();
    if rows.is_empty() {
        eprintln!("  No refined hits in {OUT_DIR}/refine_summary.csv");
        eprintln!("  Run wsb_refine without --reprop first.");
        return;
    }

    eprintln!("  {:>4}  {:>4}  {:>8}  {:>10}  {:>8}  {:>8}",
        "hit", "seed", "θ (°)", "θ_sun (°)", "r_apo nd", "orbits");
    for r in &rows {
        eprintln!("  {:>4}  {:>4}  {:>8.3}  {:>10.3}  {:>8.3}  {:>8.2}",
            r.hit_id, r.seed_id, r.theta_deg, r.theta_sun_deg, r.r_apogee_nd, r.est_orbits);
    }

    let chosen = match hit_id {
        None    => &rows[0],
        Some(n) => rows.iter().find(|r| r.hit_id == n).unwrap_or(&rows[0]),
    };
    eprintln!("  Selected hit {}: θ={:.3}° θ_sun={:.3}° r_apo={:.3}nd",
        chosen.hit_id, chosen.theta_deg, chosen.theta_sun_deg, chosen.r_apogee_nd);

    let r_park = cfg_r_park();
    let theta  = chosen.theta_deg.to_radians();
    let t_sun  = chosen.theta_sun_deg.to_radians();
    let ic = match tli_injection_ic(mu, r_park, chosen.r_apogee_nd, theta) {
        Some(ic) => ic,
        None     => { eprintln!("  Bad IC."); return; }
    };
    let bcr  = Bcr4bpParams::earth_moon_sun(t_sun);
    let traj = propagate_bcr4bp(mu, bcr, ic, T_PROP, LOG_DT_OUT, RTOL_OUT, ATOL_OUT);
    eprintln!("  Done — {} pts", traj.len());

    let out_path = format!("{OUT_DIR}/solution_hifi.csv");
    let mut csv = String::from(
        "time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,\
         hit_id,seed_id,est_capture_orbits,\
         theta_deg,theta_sun_deg,r_apogee_nd,\
         dtheta_deg,dsun_deg,dr_apogee_nd\n"
    );
    for s in &traj {
        writeln!(csv,
            "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{},{},{:.3},{:.4},{:.4},{:.4},0.0,0.0,0.0",
            s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz,
            chosen.hit_id, chosen.seed_id, chosen.est_orbits,
            chosen.theta_deg, chosen.theta_sun_deg, chosen.r_apogee_nd,
        ).unwrap();
    }
    fs::write(&out_path, &csv).expect("solution_hifi csv write failed");
    eprintln!("  Saved {out_path}");
    eprintln!("  Run: python plot/plot_wsb_solution_anim.py");
}

// ════════════════════════════════════════════════════════════════════════════════
// main
// ════════════════════════════════════════════════════════════════════════════════

fn main() {
    let r_park = cfg_r_park();
    let params = CrtbpParams::earth_moon();
    let mu     = params.mu;
    let v_km_s = params.v_star / 1e3;
    let moon_x = 1.0 - mu;
    let r_hill = lunar_hill_radius(mu);

    fs::create_dir_all(OUT_DIR).unwrap();

    // --reprop mode
    let args: Vec<String> = std::env::args().collect();
    if let Some(ri) = args.iter().position(|a| a == "--reprop") {
        let hit_id: Option<usize> = args.get(ri + 1).and_then(|s| s.parse().ok());
        reprop_mode(&params, mu, hit_id);
        return;
    }

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║     WSB Refinement — Monte Carlo (ΔV objective)              ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!("  Seeds   : up to {N_SEEDS_MAX} GA solutions from {SEEDS_CSV}");
    eprintln!("  MC      : {N_MC_PER_SEED} tight samples/seed  (3-D: θ, θ_sun, r_apo)");
    eprintln!("  Window  : ±{DTHETA_MC}° θ  ±{DSUN_MC}° θ_sun  ±{DR_APOGEE_MC} nd r_apo");
    eprintln!("  Filter  : entered_hill AND LOI_ΔV < {LOI_DV_MAX_KMS} km/s");
    eprintln!("  Fitness : ΔV_TLI + ΔV_LOI_min");
    eprintln!();

    // ── Load seeds ────────────────────────────────────────────────────────────
    let seeds = load_seeds(N_SEEDS_MAX, MIN_SEED_SCORE, SEED_MIN_SEP_DEG);
    if seeds.is_empty() {
        eprintln!("  No seeds — run wsb_search first.");
        return;
    }
    eprintln!("  Loaded {} GA seeds:", seeds.len());
    for s in &seeds {
        eprintln!("    {:3}: θ={:.1}°  θ_sun={:.1}°  r_apo={:.2}  orbits={:.2}",
            s.seed_id, s.theta_deg, s.theta_sun_deg, s.r_apogee_nd, s.est_orbits);
    }
    eprintln!();

    // ── Monte Carlo per seed ──────────────────────────────────────────────────
    let total_evals = seeds.len() * N_MC_PER_SEED;
    eprintln!("  Running {total_evals} propagations (rayon parallel) …");

    let all_solutions: Vec<McSolution> = seeds.par_iter().flat_map(|seed| {
        let mut rng = rand::thread_rng();

        // Build 3-D sample list: (θ, θ_sun, r_apogee), seed centre first
        let mut samples: Vec<(f64, f64, f64)> = Vec::with_capacity(N_MC_PER_SEED);
        samples.push((seed.theta_deg, seed.theta_sun_deg, seed.r_apogee_nd));
        for _ in 1..N_MC_PER_SEED {
            let dtheta = rng.gen_range(-DTHETA_MC..=DTHETA_MC);
            let dsun   = rng.gen_range(-DSUN_MC..=DSUN_MC);
            let dapo   = rng.gen_range(-DR_APOGEE_MC..=DR_APOGEE_MC);
            samples.push((
                seed.theta_deg     + dtheta,
                seed.theta_sun_deg + dsun,
                seed.r_apogee_nd   + dapo,
            ));
        }

        let mut seed_solutions: Vec<McSolution> = Vec::new();

        for (theta_deg, theta_sun_deg, r_apo) in &samples {
            let r_apo = *r_apo;
            if r_apo <= r_park { continue; }

            let theta = theta_deg.to_radians();
            let t_sun = theta_sun_deg.to_radians();

            let Some(ic) = tli_injection_ic(mu, r_park, r_apo, theta) else { continue };
            let dv_tli = tli_dv(mu, r_park, r_apo);

            let bcr  = Bcr4bpParams::earth_moon_sun(t_sun);
            let traj = propagate_bcr4bp(mu, bcr, ic, T_PROP, LOG_DT, RTOL, ATOL);

            let cap = detect_capture(&traj, mu, MIN_CAPTURE_TIME);
            let est_orbits = cap.max_capture_interval / LUNAR_PERIOD_ND;

            let entered_hill = cap.n_entries > 0;

            // Hill-entry time (for t_transfer metric)
            let hill_t = traj.iter().find(|s| {
                let dx = s.x - moon_x;
                (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
            }).map(|s| s.time).unwrap_or(0.0);

            let (dv_loi, penalized) = match min_loi_dv(&traj, mu) {
                Some(d) => (d, false),
                None    => (LOI_PENALTY, true),
            };
            let dv_loi_kms = if penalized { LOI_PENALTY * v_km_s } else { dv_loi * v_km_s };

            // Keep Hill-entering solutions with a feasible LOI burn
            if !entered_hill || dv_loi_kms > LOI_DV_MAX_KMS { continue; }

            let dv_total = dv_tli + dv_loi;

            let x_earth = -mu;
            let omega_s: f64 = 27.321_661 / 365.25 - 1.0;   // ≈ -0.9252 nd/nd
            let alpha_deg = traj.iter()
                .max_by(|a, b| {
                    let ra = ((a.x - x_earth).powi(2) + a.y.powi(2) + a.z.powi(2)).sqrt();
                    let rb = ((b.x - x_earth).powi(2) + b.y.powi(2) + b.z.powi(2)).sqrt();
                    ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|pt| {
                    let theta_sun_apo = t_sun + omega_s * pt.time;
                    let ang = f64::atan2(pt.y, pt.x - x_earth).rem_euclid(2.0 * PI);
                    (ang - theta_sun_apo).rem_euclid(2.0 * PI).to_degrees()
                })
                .unwrap_or(f64::NAN);

            seed_solutions.push(McSolution {
                seed_id:            seed.seed_id,
                theta_deg:          *theta_deg,
                theta_sun_deg:      *theta_sun_deg,
                r_apogee_nd:        r_apo,
                dv_total_nd:        dv_total,
                dv_tli_kms:         dv_tli * v_km_s,
                dv_loi_kms:         dv_loi_kms,
                dv_total_kms:       dv_total * v_km_s,
                est_capture_orbits: est_orbits,
                t_transfer_days:    hill_t * params.t_star / 86_400.0,
                alpha_deg,
            });
        }
        seed_solutions
    }).collect();

    let mut solutions = all_solutions;
    solutions.sort_by(|a, b| a.dv_total_nd.partial_cmp(&b.dv_total_nd).unwrap());

    // Deduplicate: drop solutions within 0.1° θ and 0.2° θ_sun of a better one
    let mut deduped: Vec<McSolution> = Vec::new();
    'outer: for s in &solutions {
        for d in &deduped {
            let dt = (s.theta_deg - d.theta_deg).abs().rem_euclid(360.0);
            let ds = (s.theta_sun_deg - d.theta_sun_deg).abs().rem_euclid(360.0);
            if dt.min(360.0 - dt) < 0.1 && ds.min(360.0 - ds) < 0.2
                && (s.r_apogee_nd - d.r_apogee_nd).abs() < 0.05 {
                continue 'outer;
            }
        }
        deduped.push(s.clone());
    }

    eprintln!("  Found {} solutions ({} after dedup)", solutions.len(), deduped.len());
    if !deduped.is_empty() {
        let best = &deduped[0];
        eprintln!("  Best: θ={:.3}°  θ_sun={:.3}°  r_apo={:.3}  ΔV_total={:.4} km/s  ({:.4} + {:.4})",
            best.theta_deg, best.theta_sun_deg, best.r_apogee_nd,
            best.dv_total_kms, best.dv_tli_kms, best.dv_loi_kms);
    }
    eprintln!();

    // ── Save mc_solutions.csv ─────────────────────────────────────────────────
    save_mc_solutions_csv(&deduped);

    // ── Save refine_summary.csv (wsb_maxhifi-compatible) ─────────────────────
    save_refine_summary_csv(&seeds, &deduped, &params);

    // ── Save top trajectories ─────────────────────────────────────────────────
    let top_n = deduped.len().min(N_TOP_TRAJ);
    if top_n > 0 {
        eprintln!("  Re-propagating top {top_n} solutions at medium fidelity for trajectory output …");
        save_mc_traj_csv(&deduped[..top_n], mu, &params);
    }

    // ── Three-way pipeline comparison ─────────────────────────────────────────
    let grid_best = load_grid_best(5);
    let ga_best   = load_ga_best(5);

    eprintln!();
    eprintln!("  ╔{}╗", "═".repeat(88));
    eprintln!("  ║{:^88}║", "  Pipeline comparison — Grid Search  →  GA  →  MC Refined  ");
    eprintln!("  ╚{}╝", "═".repeat(88));

    // ── Grid search ───────────────────────────────────────────────────────────
    eprintln!();
    eprintln!("  ── 1. Grid search  (ΔV_TLI only — LOI not evaluated) {}", "─".repeat(35));
    eprintln!("  {:>3}  {:>12}  {:>8}  {:>10}  {:>7}  {:>10}  {:>8}  {:>9}",
        "#", "family", "θ [°]", "θ_sun [°]", "r_apo", "ΔV_TLI", "α [°]", "t [days]");
    for (i, g) in grid_best.iter().enumerate() {
        eprintln!("  {:>3}  {:>12}  {:>8.3}  {:>10.3}  {:>7.3}  {:>10.5}  {:>8.2}  {:>9.1}",
            i + 1, g.family, g.theta, g.theta_sun, g.r_apo,
            g.dv_tli_kms, g.alpha_deg, g.t_days);
    }
    if grid_best.is_empty() { eprintln!("  (no data — run wsb_search first)"); }

    // ── GA global ─────────────────────────────────────────────────────────────
    eprintln!();
    eprintln!("  ── 2. GA global  (ΔV_total = TLI + LOI) {}", "─".repeat(48));
    eprintln!("  {:>3}  {:>8}  {:>10}  {:>7}  {:>10}  {:>10}  {:>10}  {:>8}  {:>9}  {}",
        "#", "θ [°]", "θ_sun [°]", "r_apo", "ΔV_TLI", "ΔV_LOI", "ΔV_tot", "α [°]", "t [days]", "Pareto");
    for (i, g) in ga_best.iter().enumerate() {
        eprintln!("  {:>3}  {:>8.3}  {:>10.3}  {:>7.3}  {:>10.5}  {:>10.5}  {:>10.5}  {:>8.2}  {:>9.1}  {}",
            i + 1, g.theta, g.theta_sun, g.r_apo,
            g.dv_tli_kms, g.dv_loi_kms, g.dv_total_kms,
            g.alpha_deg, g.t_days,
            if g.is_pareto { "★" } else { "" });
    }
    if ga_best.is_empty() { eprintln!("  (no data — run wsb_optimize first)"); }

    // ── MC refined ────────────────────────────────────────────────────────────
    eprintln!();
    eprintln!("  ── 3. MC refined  (current run) {}", "─".repeat(57));
    eprintln!("  {:>3}  {:>8}  {:>10}  {:>7}  {:>10}  {:>10}  {:>10}  {:>8}  {:>9}",
        "#", "θ [°]", "θ_sun [°]", "r_apo", "ΔV_TLI", "ΔV_LOI", "ΔV_tot", "α [°]", "t [days]");
    for (i, s) in deduped.iter().take(5).enumerate() {
        eprintln!("  {:>3}  {:>8.3}  {:>10.3}  {:>7.3}  {:>10.5}  {:>10.5}  {:>10.5}  {:>8.2}  {:>9.1}",
            i + 1, s.theta_deg, s.theta_sun_deg, s.r_apogee_nd,
            s.dv_tli_kms, s.dv_loi_kms, s.dv_total_kms,
            s.alpha_deg, s.t_transfer_days);
    }
    if deduped.is_empty() { eprintln!("  (no MC solutions found)"); }

    // ── Improvement summary ───────────────────────────────────────────────────
    eprintln!();
    eprintln!("  ── Improvement {}", "─".repeat(72));
    if let (Some(g), Some(ga)) = (grid_best.first(), ga_best.first()) {
        let d = ga.dv_tli_kms - g.dv_tli_kms;
        eprintln!("  Grid → GA    ΔV_TLI :  {:.5} → {:.5} km/s  ({:+.3}%)",
            g.dv_tli_kms, ga.dv_tli_kms, d / g.dv_tli_kms * 100.0);
    }
    if let (Some(ga), Some(mc)) = (ga_best.first(), deduped.first()) {
        let d = mc.dv_total_kms - ga.dv_total_kms;
        eprintln!("  GA  → MC     ΔV_tot :  {:.5} → {:.5} km/s  ({:+.3}%)",
            ga.dv_total_kms, mc.dv_total_kms, d / ga.dv_total_kms * 100.0);
    }
    if let (Some(g), Some(mc)) = (grid_best.first(), deduped.first()) {
        let d = mc.dv_tli_kms - g.dv_tli_kms;
        eprintln!("  Grid → MC    ΔV_TLI :  {:.5} → {:.5} km/s  ({:+.3}%)",
            g.dv_tli_kms, mc.dv_tli_kms, d / g.dv_tli_kms * 100.0);
        eprintln!("  MC best      ΔV_tot :  {:.5} km/s  (TLI {:.5} + LOI {:.5})",
            mc.dv_total_kms, mc.dv_tli_kms, mc.dv_loi_kms);
    }
    eprintln!();
}

// ════════════════════════════════════════════════════════════════════════════════
// Seed loading
// ════════════════════════════════════════════════════════════════════════════════

fn load_seeds(n: usize, _min_score: f64, min_sep_deg: f64) -> Vec<Seed> {
    let content = match fs::read_to_string(SEEDS_CSV) {
        Ok(s)  => s,
        Err(e) => { eprintln!("  [error] Cannot read {SEEDS_CSV}: {e}"); return Vec::new(); }
    };
    // ga_best.csv columns:
    // 0:rank 1:theta_deg 2:theta_sun_deg 3:r_apogee_nd 4:dv_tli_kms
    // 5:dv_loi_kms 6:dv_total_kms 7:est_capture_orbits 8:t_transfer_days
    let mut rows: Vec<Seed> = Vec::new();
    for line in content.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 8 { continue; }
        let theta_deg:     f64 = c[1].parse().unwrap_or(f64::NAN);
        let theta_sun_deg: f64 = c[2].parse().unwrap_or(f64::NAN);
        let r_apogee_nd:   f64 = c[3].parse().unwrap_or(f64::NAN);
        let est_orbits:    f64 = c[7].parse().unwrap_or(0.0);
        if theta_deg.is_nan() { continue; }
        rows.push(Seed {
            seed_id: 0, theta_deg, theta_sun_deg, r_apogee_nd, est_orbits,
        });
    }
    // Already sorted by ΔV_total in ga_best.csv — apply diversity filter
    let mut out: Vec<Seed> = Vec::new();
    let mut id = 1usize;
    'outer: for mut s in rows {
        for o in &out {
            let d = (s.theta_deg - o.theta_deg).abs().rem_euclid(360.0);
            if d.min(360.0 - d) < min_sep_deg { continue 'outer; }
        }
        s.seed_id = id; id += 1;
        out.push(s);
        if out.len() >= n { break; }
    }
    out
}

// ════════════════════════════════════════════════════════════════════════════════
// Output writers
// ════════════════════════════════════════════════════════════════════════════════

fn save_mc_solutions_csv(solutions: &[McSolution]) {
    let path = format!("{OUT_DIR}/mc_solutions.csv");
    let mut csv = String::from(
        "rank,seed_id,theta_deg,theta_sun_deg,r_apogee_nd,\
         dv_tli_kms,dv_loi_kms,dv_total_kms,\
         est_capture_orbits,t_transfer_days,alpha_deg\n"
    );
    for (i, s) in solutions.iter().enumerate() {
        writeln!(csv,
            "{},{},{:.4},{:.4},{:.4},{:.5},{:.5},{:.5},{:.3},{:.2},{:.2}",
            i + 1, s.seed_id,
            s.theta_deg, s.theta_sun_deg, s.r_apogee_nd,
            s.dv_tli_kms, s.dv_loi_kms, s.dv_total_kms,
            s.est_capture_orbits, s.t_transfer_days, s.alpha_deg,
        ).unwrap();
    }
    fs::write(&path, &csv).expect("mc_solutions csv write failed");
    eprintln!("  Saved {path}  ({} solutions)", solutions.len());
}

fn save_refine_summary_csv(seeds: &[Seed], solutions: &[McSolution], _params: &CrtbpParams) {
    let path = format!("{OUT_DIR}/refine_summary.csv");
    // Columns must match what wsb_maxhifi expects:
    // seed_id,source,theta_deg,theta_sun_deg,r_apogee_nd,
    // dv_kms,t_transfer_days,est_capture_orbits,min_alt_km,
    // dtheta_deg,dsun_deg,dr_apogee_nd,...
    let mut csv = String::from(
        "seed_id,source,theta_deg,theta_sun_deg,r_apogee_nd,\
         dv_kms,t_transfer_days,est_capture_orbits,min_alt_km,\
         dtheta_deg,dsun_deg,dr_apogee_nd,entered_hill,max_hill_dwell_nd,score,crashed_seed\n"
    );
    // Seed rows
    for s in seeds {
        writeln!(csv,
            "{},seed,{:.3},{:.3},{:.4},{:.5},{:.2},{:.3},NaN,0.000,0.000,0.000,1,NaN,{:.4},0",
            s.seed_id, s.theta_deg, s.theta_sun_deg, s.r_apogee_nd,
            0.0_f64, 0.0_f64, s.est_orbits, s.est_orbits,
        ).unwrap();
    }
    // Solution rows (appear as "refined" so wsb_maxhifi picks them up)
    for (_, sol) in solutions.iter().enumerate() {
        let dtheta = sol.theta_deg - seeds.iter()
            .find(|s| s.seed_id == sol.seed_id)
            .map(|s| s.theta_deg)
            .unwrap_or(sol.theta_deg);
        let dsun = sol.theta_sun_deg - seeds.iter()
            .find(|s| s.seed_id == sol.seed_id)
            .map(|s| s.theta_sun_deg)
            .unwrap_or(sol.theta_sun_deg);
        let dr = sol.r_apogee_nd - seeds.iter()
            .find(|s| s.seed_id == sol.seed_id)
            .map(|s| s.r_apogee_nd)
            .unwrap_or(sol.r_apogee_nd);
        writeln!(csv,
            "{},refined,{:.4},{:.4},{:.4},{:.5},{:.2},{:.3},NaN,{:.4},{:.4},{:.4},1,NaN,{:.4},0",
            sol.seed_id,
            sol.theta_deg, sol.theta_sun_deg, sol.r_apogee_nd,
            sol.dv_total_kms, sol.t_transfer_days, sol.est_capture_orbits,
            dtheta, dsun, dr,
            sol.est_capture_orbits,
        ).unwrap();
    }
    fs::write(&path, &csv).expect("refine_summary csv write failed");
    eprintln!("  Saved {path}  ({} seeds + {} solutions)", seeds.len(), solutions.len());
}

fn save_mc_traj_csv(solutions: &[McSolution], mu: f64, _params: &CrtbpParams) {
    let path = format!("{OUT_DIR}/mc_traj.csv");
    let r_park = cfg_r_park();
    let moon_x = 1.0 - mu;
    let r_hill = lunar_hill_radius(mu);

    let mut csv = String::from(
        "time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,\
         rank,seed_id,dv_total_kms,est_capture_orbits,\
         theta_deg,theta_sun_deg,r_apogee_nd\n"
    );

    for (rank, sol) in solutions.iter().enumerate() {
        let theta = sol.theta_deg.to_radians();
        let t_sun = sol.theta_sun_deg.to_radians();
        let Some(ic) = tli_injection_ic(mu, r_park, sol.r_apogee_nd, theta) else { continue };

        let bcr  = Bcr4bpParams::earth_moon_sun(t_sun);
        let traj = propagate_bcr4bp(mu, bcr, ic, T_PROP, LOG_DT_OUT, RTOL_OUT, ATOL_OUT);

        // Truncate: keep up to Hill entry + 4 lunar periods
        let hill_t = traj.iter().find(|s| {
            let dx = s.x - moon_x;
            (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
        }).map(|s| s.time);
        let t_stop = hill_t.map(|t| t + 4.0 * 2.0 * PI).unwrap_or(T_PROP);

        for s in traj.iter().take_while(|s| s.time <= t_stop) {
            writeln!(csv,
                "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{},{},{:.5},{:.3},{:.4},{:.4},{:.4}",
                s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz,
                rank + 1, sol.seed_id, sol.dv_total_kms, sol.est_capture_orbits,
                sol.theta_deg, sol.theta_sun_deg, sol.r_apogee_nd,
            ).unwrap();
        }
        writeln!(csv,
            "NaN,NaN,NaN,NaN,NaN,NaN,NaN,{},{},{:.5},{:.3},{:.4},{:.4},{:.4}",
            rank + 1, sol.seed_id, sol.dv_total_kms, sol.est_capture_orbits,
            sol.theta_deg, sol.theta_sun_deg, sol.r_apogee_nd,
        ).unwrap();

        eprintln!("  traj rank {}: {} pts  ΔV_tot={:.4} km/s",
            rank + 1, traj.len(), sol.dv_total_kms);
    }

    fs::write(&path, &csv).expect("mc_traj csv write failed");
    eprintln!("  Saved {path}");
}
