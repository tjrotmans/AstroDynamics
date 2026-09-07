//! wsb_optimize — Island GA global search for WSB total-ΔV.
//!
//! Runs AFTER wsb_search and BEFORE wsb_refine in the pipeline.
//!
//! Architecture: one GA island per trajectory family.
//!
//! Phase 4 (wsb_search) produces family_analysis.csv — a dense 8×8 envelope scan
//! that reliably maps which θ window each family occupies.  wsb_optimize reads
//! that map, derives data-driven θ bounds per family, and runs one independent
//! GA island per family confined to those bounds.  Islands run in parallel.
//!
//! This separates roles cleanly:
//!   Phase 4 grid  →  exploration  (find where families live in θ)
//!   Island GA     →  exploitation  (find the minimum within each family)
//!
//! Objective:  ΔV_total = ΔV_TLI + ΔV_LOI_min
//!
//! Outputs (out/wsb/):
//!   ga_solutions.csv  — all Hill-entering solutions, LOI ΔV < threshold
//!   ga_best.csv       — top GA_TOP_N ranked by ΔV_total  (read by wsb_refine)
//!
//! Usage:  cargo run -p lunar_trajectories --bin wsb_optimize --release

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

/// Individuals per island.
const ISLAND_POP: usize = 150;

/// Generations per island.
const ISLAND_GENERATIONS: u64 = 150;

const CROSSOVER_RATE:  f64   = 0.80;
const MUTATION_RATE:   f64   = 0.20;
const ELITISM_COUNT:   usize = 4;
const TOURNAMENT_SIZE: usize = 5;

/// Top solutions saved to ga_best.csv (read by wsb_refine as seeds).
const GA_TOP_N: usize = 30;

/// Warm-start seeds loaded per island (top-N by score from that family).
const N_WARM_PER_ISLAND: usize = 30;

/// Island θ half-width = observed family θ spread × this factor.
const THETA_MARGIN: f64 = 1.5;

/// Minimum island θ half-width [deg] — guards against over-narrow windows
/// if Phase 4 seeds happen to cluster tightly.
const THETA_MIN_HALF: f64 = 20.0;

/// LOI penalty [nd] for trajectories that never enter the Hill sphere.
const LOI_PENALTY: f64 = 2.0;

/// Maximum LOI ΔV [km/s] kept in outputs.
const LOI_DV_MAX_KMS: f64 = 1.0;

// ── θ_sun and r_apo are unrestricted within every island ──────────────────────
const SUN_MIN:   f64 = 0.0;
const SUN_MAX:   f64 = 360.0;
const R_APO_MIN: f64 = 2.8;
const R_APO_MAX: f64 = 4.6;

// ── Propagation ───────────────────────────────────────────────────────────────
const T_PROP: f64 = 20.0 * PI;
const LOG_DT: f64 = 0.06;
const RTOL:   f64 = 1e-8;
const ATOL:   f64 = 1e-10;

// ── Capture criterion ─────────────────────────────────────────────────────────
const LUNAR_PERIOD_ND:  f64 = 2.0 * PI;
const MIN_CAPTURE_TIME: f64 = 0.15;

const R_PARK:     f64  = (6_371.0 + 378.0) / 384_400.0;
const OUT_DIR:    &str = "out/wsb";
const FAMILY_CSV: &str = "out/wsb/family_analysis.csv";

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
// Types
// ════════════════════════════════════════════════════════════════════════════════

/// GA individual: [theta_deg, theta_sun_deg, r_apogee_nd]
type Individual = [f64; 3];

/// Per-family island: data-derived θ window + warm-start seeds.
struct IslandConfig {
    family:       String,
    theta_center: f64,  // deg — circular mean of family θ values
    theta_half:   f64,  // deg — half-width of the search window
    seeds:        Vec<Individual>,
}

struct GridRow {
    theta:      f64,
    theta_sun:  f64,
    r_apo:      f64,
    dv_tli_kms: f64,
    alpha_deg:  f64,
    t_days:     f64,
    family:     String,
}

#[derive(Clone)]
struct EvalResult {
    individual:      Individual,
    fitness:         f64,  // ΔV_total [nd] — lower is better
    dv_tli_kms:      f64,
    dv_loi_kms:      f64,
    dv_total_kms:    f64,
    est_orbits:      f64,
    t_transfer_days: f64,
    entered_hill:    bool,
    alpha_deg:       f64,
}

// ════════════════════════════════════════════════════════════════════════════════
// Trajectory evaluation
// ════════════════════════════════════════════════════════════════════════════════

fn evaluate(ind: &Individual, mu: f64, r_park: f64, v_km_s: f64, t_star: f64) -> EvalResult {
    let theta  = ind[0].to_radians();
    let t_sun  = ind[1].to_radians();
    let r_apo  = ind[2];
    let moon_x = 1.0 - mu;
    let r_hill = lunar_hill_radius(mu);
    let dv_tli = tli_dv(mu, r_park, r_apo);

    let Some(ic) = tli_injection_ic(mu, r_park, r_apo, theta) else {
        return EvalResult {
            individual:      *ind,
            fitness:         dv_tli + LOI_PENALTY,
            dv_tli_kms:      dv_tli * v_km_s,
            dv_loi_kms:      0.0,
            dv_total_kms:    (dv_tli + LOI_PENALTY) * v_km_s,
            est_orbits:      0.0,
            t_transfer_days: 0.0,
            entered_hill:    false,
            alpha_deg:       f64::NAN,
        };
    };

    let bcr  = Bcr4bpParams::earth_moon_sun(t_sun);
    let traj = propagate_bcr4bp(mu, bcr, ic, T_PROP, LOG_DT, RTOL, ATOL);

    let cap        = detect_capture(&traj, mu, MIN_CAPTURE_TIME);
    let est_orbits = cap.max_capture_interval / LUNAR_PERIOD_ND;

    let hill_t = traj.iter().find(|s| {
        let dx = s.x - moon_x;
        (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
    }).map(|s| s.time).unwrap_or(0.0);

    let entered_hill = cap.n_entries > 0;
    let (dv_loi, penalized) = match min_loi_dv(&traj, mu) {
        Some(d) => (d, false),
        None    => (LOI_PENALTY, true),
    };

    let fitness    = dv_tli + dv_loi;
    let dv_loi_out = if penalized { 0.0 } else { dv_loi };

    let x_earth = -mu;
    // Sun's angular velocity in the EM synodic frame [nd/nd].
    // Must use the evolved Sun angle at apogee time, not the initial t_sun —
    // the Sun moves ~190° during a typical 43-day transfer, which would
    // otherwise completely scramble the computed alpha.
    let omega_s: f64 = 27.321_661 / 365.25 - 1.0;   // ≈ -0.9252
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

    EvalResult {
        individual:      *ind,
        fitness,
        dv_tli_kms:      dv_tli     * v_km_s,
        dv_loi_kms:      dv_loi_out * v_km_s,
        dv_total_kms:    fitness    * v_km_s,
        est_orbits,
        t_transfer_days: hill_t     * t_star / 86_400.0,
        entered_hill,
        alpha_deg,
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Island-aware GA primitives
// ════════════════════════════════════════════════════════════════════════════════

fn random_in_island(rng: &mut impl Rng, center: f64, half: f64) -> Individual {
    let offset = rng.gen_range(-half..=half);
    [
        (center + offset).rem_euclid(360.0),
        rng.gen_range(SUN_MIN..SUN_MAX),
        rng.gen_range(R_APO_MIN..R_APO_MAX),
    ]
}

fn crossover(rng: &mut impl Rng, p1: &Individual, p2: &Individual) -> Individual {
    [
        if rng.gen_bool(0.5) { p1[0] } else { p2[0] },
        if rng.gen_bool(0.5) { p1[1] } else { p2[1] },
        if rng.gen_bool(0.5) { p1[2] } else { p2[2] },
    ]
}

/// Mutate θ within the island window; θ_sun and r_apo mutate freely.
fn mutate_in_island(
    rng:    &mut impl Rng,
    ind:    &Individual,
    pm:     f64,
    center: f64,
    half:   f64,
) -> Individual {
    let sigma_t = half                    * pm * 0.3;
    let sigma_s = (SUN_MAX - SUN_MIN)     * pm * 0.15;
    let sigma_r = (R_APO_MAX - R_APO_MIN) * pm * 0.15;

    // Work in offset-from-center to handle the 0°/360° wrap correctly.
    let mut offset = ind[0] - center;
    while offset >  180.0 { offset -= 360.0; }
    while offset < -180.0 { offset += 360.0; }
    let new_offset = (offset + rng.gen_range(-sigma_t..=sigma_t))
        .clamp(-half, half);

    [
        (center + new_offset).rem_euclid(360.0),
        (ind[1] + rng.gen_range(-sigma_s..=sigma_s)).rem_euclid(360.0),
        (ind[2] + rng.gen_range(-sigma_r..=sigma_r)).clamp(R_APO_MIN, R_APO_MAX),
    ]
}

fn tournament_select<'a>(
    population: &'a [Individual],
    fitnesses:  &[f64],
    size:       usize,
    rng:        &mut impl Rng,
) -> &'a Individual {
    let n = population.len();
    let mut best_i = rng.gen_range(0..n);
    for _ in 1..size {
        let i = rng.gen_range(0..n);
        if fitnesses[i] < fitnesses[best_i] { best_i = i; }
    }
    &population[best_i]
}

// ════════════════════════════════════════════════════════════════════════════════
// Load top grid-search solutions from family_analysis.csv for comparison
// ════════════════════════════════════════════════════════════════════════════════

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
        if c[14].trim() != "0" { continue; }   // skip crashed
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
    // Deduplicate by θ proximity
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

// ════════════════════════════════════════════════════════════════════════════════
// Load per-family island configs from family_analysis.csv
// ════════════════════════════════════════════════════════════════════════════════

fn load_family_islands(n_seeds: usize) -> Vec<IslandConfig> {
    let content = match fs::read_to_string(FAMILY_CSV) {
        Ok(s)  => s,
        Err(_) => {
            eprintln!("  [warn] Cannot read {FAMILY_CSV} — falling back to one global island");
            return Vec::new();
        }
    };

    // family_analysis.csv columns:
    // 0:theta_deg  1:theta_sun_deg  2:dv_kms  3:t_transfer_days  4:capture_dur_days
    // 5:min_alt_km  6:alpha_deg  7:apogee_nd  8:r_apogee_target_nd  9:n_entries
    // 10:est_capture_orbits  11:family  12:pareto_rank  13:score  14:crashed_moon
    let mut by_family: std::collections::HashMap<String, Vec<(f64, Individual)>> =
        std::collections::HashMap::new();

    for line in content.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 14 { continue; }
        let theta_deg: f64 = c[0].parse().unwrap_or(f64::NAN);
        let theta_sun: f64 = c[1].parse().unwrap_or(f64::NAN);
        let r_apogee:  f64 = c[8].parse().unwrap_or(f64::NAN);
        let family:  String = c[11].trim().to_string();
        let score:     f64 = c[13].parse().unwrap_or(0.0);
        if theta_deg.is_nan() || r_apogee.is_nan() { continue; }
        by_family.entry(family).or_default()
            .push((score, [theta_deg, theta_sun, r_apogee]));
    }

    let mut islands = Vec::new();
    let mut families: Vec<String> = by_family.keys().cloned().collect();
    families.sort();

    for fam in &families {
        let entries = by_family.get_mut(fam).unwrap();
        entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let thetas: Vec<f64> = entries.iter().map(|(_, ind)| ind[0]).collect();

        // Circular mean — correct across the 0°/360° boundary.
        let sin_sum = thetas.iter().map(|t| t.to_radians().sin()).sum::<f64>();
        let cos_sum = thetas.iter().map(|t| t.to_radians().cos()).sum::<f64>();
        let theta_center = sin_sum.atan2(cos_sum).to_degrees().rem_euclid(360.0);

        // Max angular deviation from the circular mean.
        let max_dev = thetas.iter().map(|&t| {
            let mut d = (t - theta_center).rem_euclid(360.0);
            if d > 180.0 { d = 360.0 - d; }
            d
        }).fold(0.0_f64, f64::max);

        let theta_half = (max_dev * THETA_MARGIN).max(THETA_MIN_HALF);
        let seeds: Vec<Individual> = entries.iter().take(n_seeds).map(|(_, ind)| *ind).collect();

        islands.push(IslandConfig { family: fam.clone(), theta_center, theta_half, seeds });
    }

    islands
}

// ════════════════════════════════════════════════════════════════════════════════
// Single island GA — called in parallel for each family
// ════════════════════════════════════════════════════════════════════════════════

fn run_island(
    cfg:    &IslandConfig,
    mu:     f64,
    r_park: f64,
    v_km_s: f64,
    t_star: f64,
) -> Vec<EvalResult> {
    let mut rng    = rand::thread_rng();
    let center     = cfg.theta_center;
    let half       = cfg.theta_half;

    // Initial population: warm seeds first, then random within window.
    let mut population: Vec<Individual> = cfg.seeds.clone();
    while population.len() < ISLAND_POP {
        population.push(random_in_island(&mut rng, center, half));
    }
    population.truncate(ISLAND_POP);

    let mut evals: Vec<EvalResult> = population.iter()
        .map(|ind| evaluate(ind, mu, r_park, v_km_s, t_star))
        .collect();
    let mut fitnesses: Vec<f64> = evals.iter().map(|e| e.fitness).collect();
    let mut archive: Vec<EvalResult> = evals.clone();

    for gen in 0..ISLAND_GENERATIONS {
        let progress = gen as f64 / ISLAND_GENERATIONS as f64;
        let decay    = 0.5 * (1.0 + (PI * progress).cos());
        let pm       = MUTATION_RATE * 0.1 + (MUTATION_RATE * 0.9) * decay;

        let mut indexed: Vec<(usize, f64)> = fitnesses.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let sorted_idx: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();

        // Elitism
        let mut new_pop: Vec<Individual> = sorted_idx[..ELITISM_COUNT.min(ISLAND_POP)]
            .iter()
            .map(|&i| population[i])
            .collect();

        // Tournament + crossover + mutation, confined to island window
        while new_pop.len() < ISLAND_POP {
            let p1    = tournament_select(&population, &fitnesses, TOURNAMENT_SIZE, &mut rng);
            let p2    = tournament_select(&population, &fitnesses, TOURNAMENT_SIZE, &mut rng);
            let child = if rng.gen::<f64>() < CROSSOVER_RATE {
                crossover(&mut rng, p1, p2)
            } else {
                *p1
            };
            new_pop.push(mutate_in_island(&mut rng, &child, pm, center, half));
        }

        population = new_pop;
        evals     = population.iter()
            .map(|ind| evaluate(ind, mu, r_park, v_km_s, t_star))
            .collect();
        fitnesses = evals.iter().map(|e| e.fitness).collect();
        archive.extend(evals.clone());
    }

    let best = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!("    {:12}  θ={:6.1}°±{:5.1}°  best={:.4} km/s",
        cfg.family, center, half, best * v_km_s);

    archive
}

// ════════════════════════════════════════════════════════════════════════════════
// main
// ════════════════════════════════════════════════════════════════════════════════

fn main() {
    let r_park = cfg_r_park();
    let params = CrtbpParams::earth_moon();
    let mu     = params.mu;
    let v_km_s = params.v_star / 1e3;
    let t_star = params.t_star;

    fs::create_dir_all(OUT_DIR).unwrap();

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║   WSB Optimizer — Island GA (one island per family)          ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!("  Pop/island={ISLAND_POP}  Gen/island={ISLAND_GENERATIONS}");
    eprintln!("  θ window: data-derived per family  (margin={THETA_MARGIN}×  min={THETA_MIN_HALF}°)");
    eprintln!("  θ_sun ∈ [0°,360°]  r_apo ∈ [{R_APO_MIN},{R_APO_MAX}] nd  (unrestricted)");
    eprintln!();

    // ── Load island configs from Phase 4 family map ───────────────────────────
    let mut islands = load_family_islands(N_WARM_PER_ISLAND);

    if islands.is_empty() {
        // Fallback: one island spanning the full θ range
        eprintln!("  [warn] No family data — running one global island");
        islands.push(IslandConfig {
            family:       "global".to_string(),
            theta_center: 180.0,
            theta_half:   180.0,
            seeds:        Vec::new(),
        });
    }

    eprintln!("  {} islands (running in parallel):", islands.len());
    for cfg in &islands {
        eprintln!("    {:12}  θ = {:6.1}° ± {:5.1}°  ({} warm seeds)",
            cfg.family, cfg.theta_center, cfg.theta_half, cfg.seeds.len());
    }
    eprintln!();
    eprintln!("  Results per island:");

    // ── Run all islands in parallel ───────────────────────────────────────────
    // Each island runs sequentially internally; islands share the rayon pool.
    let island_archives: Vec<Vec<EvalResult>> = islands.par_iter()
        .map(|cfg| run_island(cfg, mu, r_park, v_km_s, t_star))
        .collect();
    eprintln!();

    // ── Merge, deduplicate, Pareto rank ───────────────────────────────────────
    let all: Vec<EvalResult> = island_archives.into_iter().flatten().collect();

    let mut captured: Vec<EvalResult> = all.into_iter()
        .filter(|e| e.entered_hill && e.dv_loi_kms < LOI_DV_MAX_KMS)
        .collect();

    captured.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
    let mut deduped: Vec<EvalResult> = Vec::new();
    'outer: for e in captured {
        for d in &deduped {
            let dt = (e.individual[0] - d.individual[0]).rem_euclid(360.0);
            let ds = (e.individual[1] - d.individual[1]).rem_euclid(360.0);
            if dt.min(360.0 - dt) < 0.1
                && ds.min(360.0 - ds) < 0.2
                && (e.individual[2] - d.individual[2]).abs() < 0.05 {
                continue 'outer;
            }
        }
        deduped.push(e);
    }

    let pareto: Vec<bool> = (0..deduped.len()).map(|i| {
        let a = &deduped[i];
        !deduped.iter().enumerate().any(|(j, b)| {
            j != i
                && b.dv_total_kms  <= a.dv_total_kms
                && b.t_transfer_days <= a.t_transfer_days
                && (b.dv_total_kms < a.dv_total_kms || b.t_transfer_days < a.t_transfer_days)
        })
    }).collect();

    let n_pareto = pareto.iter().filter(|&&p| p).count();
    eprintln!("  Merged: {} unique captured solutions  ({} Pareto-optimal)",
        deduped.len(), n_pareto);

    // ── Grid search → GA comparison ───────────────────────────────────────────
    let grid_best = load_grid_best(5);

    eprintln!();
    eprintln!("  ╔{}╗", "═".repeat(88));
    eprintln!("  ║{:^88}║", "  Grid Search  →  GA Global  comparison  ");
    eprintln!("  ╚{}╝", "═".repeat(88));

    eprintln!();
    eprintln!("  ── Grid search  (ΔV_TLI only — LOI not evaluated in search phase) {}", "─".repeat(20));
    eprintln!("  {:>3}  {:>12}  {:>8}  {:>10}  {:>7}  {:>10}  {:>8}  {:>9}",
        "#", "family", "θ [°]", "θ_sun [°]", "r_apo", "ΔV_TLI", "α [°]", "t [days]");
    for (i, g) in grid_best.iter().enumerate() {
        eprintln!("  {:>3}  {:>12}  {:>8.3}  {:>10.3}  {:>7.3}  {:>10.5}  {:>8.2}  {:>9.1}",
            i + 1, g.family, g.theta, g.theta_sun, g.r_apo,
            g.dv_tli_kms, g.alpha_deg, g.t_days);
    }
    if grid_best.is_empty() { eprintln!("  (no data — run wsb_search first)"); }

    eprintln!();
    eprintln!("  ── GA global  (ΔV_total = TLI + LOI) {}", "─".repeat(50));
    eprintln!("  {:>3}  {:>8}  {:>10}  {:>7}  {:>10}  {:>10}  {:>10}  {:>8}  {:>9}  {}",
        "#", "θ [°]", "θ_sun [°]", "r_apo", "ΔV_TLI", "ΔV_LOI", "ΔV_tot", "α [°]", "t [days]", "Pareto");
    for (i, (e, is_p)) in deduped.iter().zip(pareto.iter()).take(5).enumerate() {
        eprintln!("  {:>3}  {:>8.3}  {:>10.3}  {:>7.3}  {:>10.5}  {:>10.5}  {:>10.5}  {:>8.2}  {:>9.1}  {}",
            i + 1,
            e.individual[0], e.individual[1], e.individual[2],
            e.dv_tli_kms, e.dv_loi_kms, e.dv_total_kms,
            e.alpha_deg, e.t_transfer_days,
            if *is_p { "★" } else { "" });
    }
    if deduped.is_empty() { eprintln!("  (no GA solutions found)"); }

    if let (Some(g), Some(ga)) = (grid_best.first(), deduped.first()) {
        let tli_pct = (ga.dv_tli_kms - g.dv_tli_kms) / g.dv_tli_kms * 100.0;
        eprintln!();
        eprintln!("  ── Improvement {}", "─".repeat(72));
        eprintln!("  Grid → GA   ΔV_TLI :  {:.5} → {:.5} km/s  ({:+.3}%)",
            g.dv_tli_kms, ga.dv_tli_kms, tli_pct);
        eprintln!("  GA best     ΔV_tot :  {:.5} km/s  (TLI {:.5} + LOI {:.5})",
            ga.dv_total_kms, ga.dv_tli_kms, ga.dv_loi_kms);
    }
    eprintln!();

    save_ga_solutions(&deduped, &pareto);
    save_ga_best(&deduped);
}

// ════════════════════════════════════════════════════════════════════════════════
// Output
// ════════════════════════════════════════════════════════════════════════════════

fn save_ga_solutions(solutions: &[EvalResult], pareto: &[bool]) {
    let path = format!("{OUT_DIR}/ga_solutions.csv");
    let mut csv = String::from(
        "rank,theta_deg,theta_sun_deg,r_apogee_nd,\
         dv_tli_kms,dv_loi_kms,dv_total_kms,\
         est_capture_orbits,t_transfer_days,alpha_deg,pareto_optimal\n"
    );
    for (i, (e, &is_p)) in solutions.iter().zip(pareto.iter()).enumerate() {
        writeln!(csv,
            "{},{:.4},{:.4},{:.4},{:.5},{:.5},{:.5},{:.3},{:.2},{:.2},{}",
            i + 1,
            e.individual[0], e.individual[1], e.individual[2],
            e.dv_tli_kms, e.dv_loi_kms, e.dv_total_kms,
            e.est_orbits, e.t_transfer_days,
            e.alpha_deg, if is_p { 1 } else { 0 },
        ).unwrap();
    }
    fs::write(&path, &csv).expect("ga_solutions csv write failed");
    eprintln!("  Saved {path}  ({} solutions)", solutions.len());
}

fn save_ga_best(solutions: &[EvalResult]) {
    let n = solutions.len().min(GA_TOP_N);
    let path = format!("{OUT_DIR}/ga_best.csv");
    let mut csv = String::from(
        "rank,theta_deg,theta_sun_deg,r_apogee_nd,\
         dv_tli_kms,dv_loi_kms,dv_total_kms,\
         est_capture_orbits,t_transfer_days\n"
    );
    for (i, e) in solutions.iter().take(n).enumerate() {
        writeln!(csv,
            "{},{:.4},{:.4},{:.4},{:.5},{:.5},{:.5},{:.3},{:.2}",
            i + 1,
            e.individual[0], e.individual[1], e.individual[2],
            e.dv_tli_kms, e.dv_loi_kms, e.dv_total_kms,
            e.est_orbits, e.t_transfer_days,
        ).unwrap();
    }
    fs::write(&path, &csv).expect("ga_best csv write failed");
    eprintln!("  Saved {path}  (top {n})");
}
