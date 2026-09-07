//! TLI targeting via multi-start differential correction
//!
//! Finds burn pitch, yaw, and coast time satisfying three free-return conditions:
//!   1. Closest approach to Moon ≈ 8,250 km from center (6,513 km alt)
//!   2. Earth-return perigee altitude ≈ 40 km (reentry corridor)
//!   3. Flyby timing near the expected epoch (from cfg.flyby_epoch)
//!
//! # Free parameters
//! - `pitch`, `yaw`         — burn steering offsets from prograde
//! - `burn_time_offset_s`   — coast time along parking orbit before ignition
//!
//! # Strategy
//!   A small number of seeds are scattered over (pitch, yaw, time_offset) space.
//!   A full-accuracy Newton solve runs from each seed, printing every iteration
//!   so convergence is visible.  The best converged trailing solution is reported.
//!
//! # Free-return geometry — TRAILING encounter
//! The spacecraft must approach from BEHIND the Moon (leading_dot < 0).
//! Moon gravity then gives a retrograde kick that lowers the return perigee.
//!
//! # Usage
//! ```bash
//! cargo run -p artemis --bin target --release
//! ```
//! Paste the printed values into `MissionConfig::artemis2()` in `src/config.rs`.
//! 
//! TODO: Search algorithm not working great yet, to be updated

use nalgebra::{Matrix3, SVector};
use ephemeris::{Almanac, Body};
use hifitime::Duration;

use orbital_models::FiniteBurn;
use artemis::config::MissionConfig;
use artemis::orbit::{compute_tli_initial_state, compute_burn_direction};
use ephemeris::{MoonTrack, SunTrack};
use artemis::propagator::{TrajectoryStep, propagate_with_abort};
use orbital_models::constants::{EARTH_RADIUS, MOON_RADIUS, MU_EARTH};

// ── Constants ─────────────────────────────────────────────────────────────────

const TARGETING_DAYS: f64       = 7.0;
const MOON_TRACK_SAMPLES: usize = 10_000;

/// Target Moon CA distance [m] (6,513 km alt + 1,737.4 km radius).
const TARGET_CA_M: f64          = (6_513.0 + 1_737.4) * 1_000.0;

/// Target Earth-return perigee altitude [m].
const TARGET_PERIGEE_ALT_M: f64 = 40_000.0;

/// Convergence tolerances — all three must be satisfied simultaneously.
const TOL_CA_M: f64             = 5_000.0;   // 5 km
const TOL_PERIGEE_M: f64        = 5_000.0;   // 5 km
const TOL_CA_TIME_S: f64        = 3_600.0;   // 1 hour

/// Time after CA to sample for post-flyby perigee [s].
/// 1.5 days ≈ 130,000 km — safely outside Moon's SOI (~66,000 km).
const POST_CA_OFFSET_S: f64     = 1.5 * 86_400.0;

// Newton solver
const FD_STEP_ANGLE: f64        = 2e-3;   // rad — finite-difference stencil width for pitch/yaw
const FD_STEP_TIME: f64         = 5.0;    // s   — stencil width for time_offset
const DAMPING: f64              = 0.5;    // fraction of Newton step applied each iteration
const MAX_STEP_ANGLE: f64       = 3.0 * std::f64::consts::PI / 180.0;  // 3° cap per iter
const MAX_STEP_TIME_S: f64      = 120.0;  // s cap per iter
const MAX_ITER: usize           = 60;
const LOG_DT_S: f64             = 300.0;

// Multi-start seed grid
const N_SEEDS_PITCH: usize      = 2;
const N_SEEDS_YAW: usize        = 2;
const N_SEEDS_TIME: usize       = 2;

/// Pitch search range [degrees].
const SEED_PITCH_MIN_DEG: f64   = -10.0;
const SEED_PITCH_MAX_DEG: f64   =  10.0;

/// Yaw search range [degrees].
const SEED_YAW_MIN_DEG: f64     = -10.0;
const SEED_YAW_MAX_DEG: f64     =  10.0;

/// Burn time offset search range [s].
const SEED_TIME_MIN_S: f64      = 300.0;
const SEED_TIME_MAX_S: f64      = 450.0;

// ── Solution record ───────────────────────────────────────────────────────────

struct Solution {
    pitch:       f64,
    yaw:         f64,
    time_offset: f64,
    ca_dist:     f64,
    peri_alt:    f64,
    ca_time_s:   f64,
    is_trailing: bool,
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let cfg = MissionConfig::artemis2();

    println!("Loading ephemeris...");
    let almanac = Almanac::new(&ephemeris::find_kernel("de440s.bsp"))
        .expect("Failed to load de440s.bsp kernel");
    println!("Loaded.\n");

    let moon_flyby_pos = almanac.body_state_eci(Body::Moon, cfg.flyby_epoch)
        .expect("Moon ephemeris").position.inner;

    let duration_s = TARGETING_DAYS * 86_400.0;
    let sample_dt  = duration_s / (MOON_TRACK_SAMPLES as f64 - 1.0);
    let mut moon_positions: Vec<SVector<f64, 3>> = Vec::with_capacity(MOON_TRACK_SAMPLES);
    let mut sun_positions:  Vec<SVector<f64, 3>> = Vec::with_capacity(MOON_TRACK_SAMPLES);
    print!("Sampling Moon and Sun tracks ({} pts, {} days)...", MOON_TRACK_SAMPLES, TARGETING_DAYS as u32);
    for i in 0..MOON_TRACK_SAMPLES {
        let t = cfg.tli_epoch + Duration::from_seconds(i as f64 * sample_dt);
        moon_positions.push(almanac.body_state_eci(Body::Moon, t).expect("Moon track").position.inner);
        sun_positions.push( almanac.body_state_eci(Body::Sun,  t).expect("Sun track").position.inner);
    }
    println!(" done.\n");
    let moon_track = MoonTrack { positions: moon_positions, sample_dt_s: sample_dt };
    let sun_track  = SunTrack  { positions: sun_positions,  sample_dt_s: sample_dt };

    let (initial_pos, initial_vel) = compute_tli_initial_state(&cfg, moon_flyby_pos);
    let expected_ca_s = (cfg.flyby_epoch - cfg.tli_epoch).to_seconds();

    let n_seeds = N_SEEDS_PITCH * N_SEEDS_YAW * N_SEEDS_TIME;
    println!("Targets:  CA = {:.0} km alt,  perigee = {:.0} km alt,  CA time = T+{:.3} days",
        (TARGET_CA_M - MOON_RADIUS) / 1e3, TARGET_PERIGEE_ALT_M / 1e3, expected_ca_s / 86400.0);
    println!("Seeds:    {}×{}×{} = {}  pitch [{:.0}°,{:.0}°]  yaw [{:.0}°,{:.0}°]  t [{:.0},{:.0}] s\n",
        N_SEEDS_PITCH, N_SEEDS_YAW, N_SEEDS_TIME, n_seeds,
        SEED_PITCH_MIN_DEG, SEED_PITCH_MAX_DEG,
        SEED_YAW_MIN_DEG,   SEED_YAW_MAX_DEG,
        SEED_TIME_MIN_S,    SEED_TIME_MAX_S);

    // ── Multi-start Newton ────────────────────────────────────────────────────
    let pitch_step = if N_SEEDS_PITCH > 1 {
        (SEED_PITCH_MAX_DEG - SEED_PITCH_MIN_DEG) / (N_SEEDS_PITCH as f64 - 1.0)
    } else { 0.0 };
    let yaw_step = if N_SEEDS_YAW > 1 {
        (SEED_YAW_MAX_DEG - SEED_YAW_MIN_DEG) / (N_SEEDS_YAW as f64 - 1.0)
    } else { 0.0 };
    let time_step = if N_SEEDS_TIME > 1 {
        (SEED_TIME_MAX_S - SEED_TIME_MIN_S) / (N_SEEDS_TIME as f64 - 1.0)
    } else { 0.0 };

    let mut solutions: Vec<Solution> = Vec::new();
    let mut seed_num = 0usize;

    for ti in 0..N_SEEDS_TIME {
        for pi in 0..N_SEEDS_PITCH {
            for yi in 0..N_SEEDS_YAW {
                seed_num += 1;
                let seed_pitch = (SEED_PITCH_MIN_DEG + pi as f64 * pitch_step).to_radians();
                let seed_yaw   = (SEED_YAW_MIN_DEG   + yi as f64 * yaw_step  ).to_radians();
                let seed_time  =  SEED_TIME_MIN_S     + ti as f64 * time_step;

                println!("── Seed {}/{}: pitch={:+.1}°  yaw={:+.1}°  t_off={:.0} s ──",
                    seed_num, n_seeds,
                    seed_pitch.to_degrees(), seed_yaw.to_degrees(), seed_time);
                println!("{:>5}  {:>+9}  {:>+9}  {:>10}  {:>12}  {:>14}  {:>13}  {:>5}",
                    "iter", "pitch°", "yaw°", "t_off [s]", "CA alt [km]", "peri alt [km]", "CA time [d]", "geom");
                println!("{}", "─".repeat(83));

                if let Some(sol) = run_newton(
                    seed_pitch, seed_yaw, seed_time,
                    &cfg, &initial_pos, &initial_vel,
                    &moon_track, &sun_track,
                    duration_s, expected_ca_s, sample_dt,
                ) {
                    println!("  → converged\n");
                    solutions.push(sol);
                } else {
                    println!("  → no convergence after {} iterations\n", MAX_ITER);
                }
            }
        }
    }

    if solutions.is_empty() {
        eprintln!("ERROR: no seeds converged. Widen SEED_*_DEG ranges or increase MAX_ITER.");
        return;
    }

    // Prefer trailing; within trailing pick smallest total scaled residual norm
    let s0 = 1.0 / TARGET_CA_M.abs().max(1e3);
    let s1 = 1.0 / TARGET_PERIGEE_ALT_M.abs().max(1_000.0);
    let s2 = 1.0 / expected_ca_s.abs().max(1.0);
    let residual_norm = |sol: &Solution| {
        let r0 = (sol.ca_dist   - TARGET_CA_M)          * s0;
        let r1 = (sol.peri_alt  - TARGET_PERIGEE_ALT_M) * s1;
        let r2 = (sol.ca_time_s - expected_ca_s)         * s2;
        (r0*r0 + r1*r1 + r2*r2).sqrt()
    };

    let trailing: Vec<&Solution> = solutions.iter().filter(|s| s.is_trailing).collect();
    let best = if !trailing.is_empty() {
        trailing.into_iter()
            .min_by(|a, b| residual_norm(a).partial_cmp(&residual_norm(b)).unwrap())
            .unwrap()
    } else {
        println!("WARNING: no trailing solutions — using best overall.");
        solutions.iter()
            .min_by(|a, b| residual_norm(a).partial_cmp(&residual_norm(b)).unwrap())
            .unwrap()
    };

    let (pitch, yaw, time_offset) = (best.pitch, best.yaw, best.time_offset);
    let (ca_dist, peri_alt, ca_time_s) = (best.ca_dist, best.peri_alt, best.ca_time_s);
    let ca_alt = (ca_dist - MOON_RADIUS) / 1e3;

    // ── Final diagnostic ──────────────────────────────────────────────────────
    let steps_full = run_steps(pitch, yaw, time_offset, &cfg, &initial_pos, &initial_vel,
                               &moon_track, &sun_track, duration_s);
    let post_t    = ca_time_s + POST_CA_OFFSET_S;
    let post_idx  = steps_full.partition_point(|s| s.time_s <= post_t)
        .min(steps_full.len().saturating_sub(1));
    let post_s    = &steps_full[post_idx];
    let energy    = 0.5 * post_s.vel.norm_squared() - MU_EARTH / post_s.pos.norm();

    let ca_idx   = steps_full.iter().enumerate()
        .min_by(|(_, a), (_, b)|
            (a.pos - a.moon_pos).norm().partial_cmp(&(b.pos - b.moon_pos).norm()).unwrap())
        .map(|(i, _)| i).unwrap_or(0);
    let ca_step  = &steps_full[ca_idx];
    let lead_dot = (ca_step.pos - ca_step.moon_pos)
        .dot(&moon_vel_at(ca_step.time_s, &moon_track, sample_dt));

    println!("═══════════════════════════════════════════════════════════");
    println!("RESULT");
    println!("═══════════════════════════════════════════════════════════");
    println!("  burn_pitch_rad:     {:.8}   ({:+.4}°)", pitch, pitch.to_degrees());
    println!("  burn_yaw_rad:       {:.8}   ({:+.4}°)", yaw,   yaw.to_degrees());
    println!("  burn_time_offset_s: {:.1}", time_offset);
    println!("  delta_v_ms:         {:.1}  (unchanged)", cfg.delta_v_ms);
    println!();
    println!("  Moon CA:  {:.0} km dist  ({:.0} km alt)  [target {:.0} km]",
        ca_dist / 1e3, ca_alt, (TARGET_CA_M - MOON_RADIUS) / 1e3);
    println!("  Time:     T+{:.3} days", ca_time_s / 86400.0);
    println!("  Geometry: {}  (leading dot = {:.0} km)",
        if lead_dot < 0.0 { "TRAILING ✓" } else { "LEADING ✗" }, lead_dot / 1e3);
    println!();
    println!("  Post-flyby (T+{:.2} days):", post_s.time_s / 86400.0);
    println!("    Two-body perigee: {:.0} km alt  [target {:.0} km]",
        peri_alt / 1e3, TARGET_PERIGEE_ALT_M / 1e3);
    println!("    Orbital energy:   {:.0} J/kg  ({})",
        energy, if energy < 0.0 { "BOUND ✓" } else { "HYPERBOLIC ✗" });
    println!();
    println!("Paste into src/config.rs  MissionConfig::artemis2():");
    println!("  burn_pitch_rad:     {:.8},", pitch);
    println!("  burn_yaw_rad:       {:.8},", yaw);
    println!("  burn_time_offset_s: {:.1},", time_offset);
}

// ── Newton solver ─────────────────────────────────────────────────────────────

/// Run a 3D Newton solve from (`seed_pitch`, `seed_yaw`, `seed_time`).
/// Prints one line per iteration.  Returns `Some(Solution)` on convergence,
/// `None` if MAX_ITER is reached without satisfying all three tolerances.
fn run_newton(
    seed_pitch: f64, seed_yaw: f64, seed_time: f64,
    cfg: &MissionConfig,
    initial_pos: &SVector<f64, 3>,
    initial_vel: &SVector<f64, 3>,
    moon_track: &MoonTrack,
    sun_track: &SunTrack,
    duration_s: f64,
    expected_ca_s: f64,
    sample_dt: f64,
) -> Option<Solution> {
    let mut pitch       = seed_pitch;
    let mut yaw         = seed_yaw;
    let mut time_offset = seed_time;

    // Scale factors so all three residuals contribute equally to the Jacobian
    let s0 = 1.0 / TARGET_CA_M.abs().max(1e3);
    let s1 = 1.0 / TARGET_PERIGEE_ALT_M.abs().max(1_000.0);
    let s2 = 1.0 / expected_ca_s.abs().max(1.0);

    for iter in 0..MAX_ITER {
        let (ca_dist, peri_alt, ca_time_s, ca_step) = eval(
            pitch, yaw, time_offset,
            cfg, initial_pos, initial_vel, moon_track, sun_track, duration_s,
        );
        let r0 = ca_dist  - TARGET_CA_M;
        let r1 = peri_alt - TARGET_PERIGEE_ALT_M;
        let r2 = ca_time_s - expected_ca_s;

        let ca_alt = (ca_dist - MOON_RADIUS) / 1e3;
        let trail  = (ca_step.pos - ca_step.moon_pos)
            .dot(&moon_vel_at(ca_step.time_s, moon_track, sample_dt)) < 0.0;

        println!("{:>5}  {:>+9.4}  {:>+9.4}  {:>10.1}  {:>12.0}  {:>14.0}  {:>13.4}  {:>5}",
            iter,
            pitch.to_degrees(), yaw.to_degrees(), time_offset,
            ca_alt, peri_alt / 1e3, ca_time_s / 86400.0,
            if trail { "TRAIL" } else { "LEAD" });

        if r0.abs() < TOL_CA_M && r1.abs() < TOL_PERIGEE_M && r2.abs() < TOL_CA_TIME_S {
            return Some(Solution { pitch, yaw, time_offset, ca_dist, peri_alt, ca_time_s, is_trailing: trail });
        }

        // Central-difference Jacobian: 6 extra propagations
        let (ca_pp, pe_pp, ct_pp, _) = eval(pitch + FD_STEP_ANGLE, yaw,                 time_offset,             cfg, initial_pos, initial_vel, moon_track, sun_track, duration_s);
        let (ca_pm, pe_pm, ct_pm, _) = eval(pitch - FD_STEP_ANGLE, yaw,                 time_offset,             cfg, initial_pos, initial_vel, moon_track, sun_track, duration_s);
        let (ca_yp, pe_yp, ct_yp, _) = eval(pitch,                 yaw + FD_STEP_ANGLE, time_offset,             cfg, initial_pos, initial_vel, moon_track, sun_track, duration_s);
        let (ca_ym, pe_ym, ct_ym, _) = eval(pitch,                 yaw - FD_STEP_ANGLE, time_offset,             cfg, initial_pos, initial_vel, moon_track, sun_track, duration_s);
        let (ca_tp, pe_tp, ct_tp, _) = eval(pitch,                 yaw,                 time_offset + FD_STEP_TIME, cfg, initial_pos, initial_vel, moon_track, sun_track, duration_s);
        let (ca_tm, pe_tm, ct_tm, _) = eval(pitch,                 yaw,                 time_offset - FD_STEP_TIME, cfg, initial_pos, initial_vel, moon_track, sun_track, duration_s);

        let j = Matrix3::new(
            (ca_pp - ca_pm) / (2.0 * FD_STEP_ANGLE) * s0,
            (ca_yp - ca_ym) / (2.0 * FD_STEP_ANGLE) * s0,
            (ca_tp - ca_tm) / (2.0 * FD_STEP_TIME)   * s0,
            (pe_pp - pe_pm) / (2.0 * FD_STEP_ANGLE) * s1,
            (pe_yp - pe_ym) / (2.0 * FD_STEP_ANGLE) * s1,
            (pe_tp - pe_tm) / (2.0 * FD_STEP_TIME)   * s1,
            (ct_pp - ct_pm) / (2.0 * FD_STEP_ANGLE) * s2,
            (ct_yp - ct_ym) / (2.0 * FD_STEP_ANGLE) * s2,
            (ct_tp - ct_tm) / (2.0 * FD_STEP_TIME)   * s2,
        );

        let Some(j_inv) = j.try_inverse() else {
            println!("  singular Jacobian at iter {}, nudging yaw", iter);
            yaw += FD_STEP_ANGLE * 5.0;
            continue;
        };

        let delta      = -j_inv * nalgebra::Vector3::new(r0 * s0, r1 * s1, r2 * s2);
        let angle_mag  = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
        let angle_scale = if angle_mag > MAX_STEP_ANGLE { MAX_STEP_ANGLE / angle_mag } else { 1.0 };

        pitch       += DAMPING * angle_scale * delta[0];
        yaw         += DAMPING * angle_scale * delta[1];
        time_offset  = (time_offset + DAMPING * delta[2].clamp(-MAX_STEP_TIME_S, MAX_STEP_TIME_S)).max(0.0);
    }

    None
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Propagate and return `(ca_dist, peri_alt, ca_time_s, ca_step)`.
fn eval(
    pitch: f64, yaw: f64, time_offset: f64,
    cfg: &MissionConfig,
    initial_pos: &SVector<f64, 3>,
    initial_vel: &SVector<f64, 3>,
    moon_track: &MoonTrack,
    sun_track: &SunTrack,
    duration_s: f64,
) -> (f64, f64, f64, TrajectoryStep) {
    let steps = run_steps(pitch, yaw, time_offset, cfg, initial_pos, initial_vel,
                          moon_track, sun_track, duration_s);
    let mut ca_idx  = 0;
    let mut ca_dist = f64::INFINITY;
    for (i, s) in steps.iter().enumerate() {
        let d = (s.pos - s.moon_pos).norm();
        if d < ca_dist { ca_dist = d; ca_idx = i; }
    }
    let ca_step   = steps[ca_idx].clone();
    let ca_time_s = ca_step.time_s;
    let post_idx  = steps.partition_point(|s| s.time_s <= ca_time_s + POST_CA_OFFSET_S)
        .min(steps.len() - 1);
    let peri_alt  = earth_perigee_alt(&steps[post_idx].pos, &steps[post_idx].vel);
    (ca_dist, peri_alt, ca_time_s, ca_step)
}

/// Propagate using the mission integrator settings.
fn run_steps(
    pitch: f64, yaw: f64, time_offset: f64,
    cfg: &MissionConfig,
    initial_pos: &SVector<f64, 3>,
    initial_vel: &SVector<f64, 3>,
    moon_track: &MoonTrack,
    sun_track: &SunTrack,
    duration_s: f64,
) -> Vec<TrajectoryStep> {
    let burn_dir = compute_burn_direction(initial_pos, initial_vel, pitch, yaw);
    let burn = FiniteBurn::from_delta_v(
        cfg.thrust_n, cfg.isp_s, cfg.mass_tli_kg, cfg.delta_v_ms, time_offset, burn_dir,
    );
    propagate_with_abort(
        *initial_pos, *initial_vel, cfg.mass_tli_kg,
        &burn, moon_track, sun_track,
        cfg.srp_area_m2, cfg.reflectivity,
        duration_s + time_offset.max(0.0), LOG_DT_S, cfg.rtol, cfg.atol,
        0.0,
    )
}

/// Two-body Earth-return perigee altitude [m]. Returns −1e6 for hyperbolic.
fn earth_perigee_alt(pos: &SVector<f64, 3>, vel: &SVector<f64, 3>) -> f64 {
    let r      = pos.norm();
    let v2     = vel.norm_squared();
    let energy = 0.5 * v2 - MU_EARTH / r;
    if energy >= 0.0 { return -1_000_000.0; }
    let a  = -MU_EARTH / (2.0 * energy);
    let rv = pos.dot(vel);
    let e  = ((v2 - MU_EARTH / r) * pos - rv * vel).norm() / MU_EARTH;
    a * (1.0 - e.clamp(0.0, 0.9999)) - EARTH_RADIUS
}

/// Moon velocity unit vector at time `t` from the pre-sampled track.
fn moon_vel_at(t: f64, moon_track: &MoonTrack, dt: f64) -> SVector<f64, 3> {
    let v = (moon_track.position_at(t + dt) - moon_track.position_at(t - dt)) / (2.0 * dt);
    v / v.norm()
}
