//! Artemis 2 Monte Carlo dispersion analysis
//!
//! Runs N=1000 trajectory simulations with Gaussian-dispersed burn parameters
//! (pitch, yaw, mass, Isp, ΔV, burn time offset, SRP area/reflectivity).
//! Scores each run by proximity to the OEM Entry Interface (EI) reference point
//! and lunar flyby altitude error. Saves the top 100 solutions and their full
//! trajectories to `out/`.
//!
//! # Requirements
//! Place `kernels/de440s.bsp` in the working directory (same as main binary).

use rand::prelude::*;
use rand_distr::{Normal, Distribution};

use ephemeris::{Almanac, Body};
use hifitime::Duration;

use orbital_models::FiniteBurn;
use orbital_models::constants::G0;
use artemis::config::MissionConfig;
use artemis::orbit::{compute_tli_initial_state, compute_burn_direction, find_closest_approach};
use ephemeris::{MoonTrack, SunTrack};
use artemis::propagator::propagate;
use orbital_models::constants::{MOON_RADIUS, EARTH_RADIUS};

// ── Monte Carlo size ──────────────────────────────────────────────────────────
const N: usize = 1000;

// ── OEM Entry Interface (EI) target point ────────────────────────────────────
// Final state from Artemis_II_OEM_2026_04_04_to_EI.asc (2026-04-10T23:53:17.163), km → m.
// Used to score trajectories by proximity to the real re-entry corridor.
const OEM_EI_POS_M: [f64; 3] = [
    3_946_027.553,
    4_784_562.414,
    1_995_148.492,
];

// ── Std deviations (tune these!) ─────────────────────────────────────────────
// thr is not sampled independently — it's derived from fixed mdot and sampled isp
// so that thr/isp = mdot*G0 remains constant (known mass flow rate).
// dv has its own sigma to capture execution uncertainty (gravity losses, cutoff errors, etc.),
// which means prop_mass and duration will vary slightly around their nominal values.
struct Sigma {
    burn_pitch_rad:    f64,
    burn_yaw_rad:      f64,
    mass_tli_kg:       f64,
    srp_area_m2:       f64,
    reflectivity:      f64,
    isp_s:             f64,
    delta_v_ms:        f64,
    burn_time_offset_s: f64,  // ± shift in TLI ignition time [s]: rotates burn ECI direction
}

// ── Solution with parameters ──────────────────────────────────────────────────
#[derive(Clone)]
struct Solution {
    pitch: f64,
    yaw: f64,
    mass: f64,
    area: f64,
    refl: f64,
    thr: f64,
    isp: f64,
    dv: f64,
    burn_time_offset_s: f64,
    group: &'static str,      // Dominant perturbation group label
    lunar_ca_altitude: f64,   // Lunar CA altitude (m)
    lunar_ca_time_s:   f64,   // Time of lunar CA (s from TLI)
    earth_ca_altitude: f64,   // Earth CA altitude after 6 days (m)
    earth_ca_time_s: f64,     // Time of Earth CA (s)
    ei_dist_m: f64,           // Closest distance to OEM EI point [m]
    ei_time_s: f64,           // Time of closest approach to EI point [s]
    final_pos_x: f64,
    final_pos_y: f64,
    final_pos_z: f64,
    score: f64,               // Fitness score for filtering
    trajectory: Vec<artemis::propagator::TrajectoryStep>,
}

/// Assign each solution to the group whose normalised deviation dominates.
/// Ties go to the timing axis.
fn assign_group(dv_delta_ms: f64, time_offset_s: f64, sigma_dv: f64, sigma_time: f64) -> &'static str {
    let dv_n   = dv_delta_ms.abs()   / sigma_dv.max(1e-12);
    let time_n = time_offset_s.abs() / sigma_time.max(1e-12);
    if time_n >= dv_n {
        if time_offset_s < 0.0 { "early_burn" } else { "late_burn" }
    } else {
        if dv_delta_ms < 0.0 { "low_dv" } else { "high_dv" }
    }
}

fn main() {
    let mut rng = thread_rng();

    let base_cfg = MissionConfig::artemis2();

    let sigma = Sigma {
        burn_pitch_rad: 0.05_f64.to_radians(),
        burn_yaw_rad:   0.05_f64.to_radians(),
        mass_tli_kg:    0.0,
        srp_area_m2:    0.0,
        reflectivity:   0.0,
        isp_s:              5.0,
        delta_v_ms:         5.0,
        burn_time_offset_s: 120.0,  // ±10 min — tune to find inclination match
    };

    // ── Load ephemeris ────────────────────────────────────────────────────────
    let almanac = Almanac::new(&ephemeris::find_kernel("de440s.bsp"))
        .expect("Failed to load de440s.bsp kernel");

    let moon_flyby_pos = almanac.body_state_eci(Body::Moon, base_cfg.flyby_epoch)
        .unwrap().position.inner;

    // ── Pre-sample Moon and Sun tracks ───────────────────────────────────────
    let duration_s = base_cfg.mission_duration_days * 86_400.0;
    let samples = 10_000;
    let dt = duration_s / (samples as f64 - 1.0);

    let mut moon_positions = Vec::with_capacity(samples);
    let mut sun_positions  = Vec::with_capacity(samples);

    for i in 0..samples {
        let epoch = base_cfg.tli_epoch + Duration::from_seconds(i as f64 * dt);
        moon_positions.push(almanac.body_state_eci(Body::Moon, epoch).unwrap().position.inner);
        sun_positions.push( almanac.body_state_eci(Body::Sun,  epoch).unwrap().position.inner);
    }

    let moon_track = MoonTrack { positions: moon_positions, sample_dt_s: dt };
    let sun_track  = SunTrack  { positions: sun_positions,  sample_dt_s: dt };

    // ── Fixed mass flow rate (from known prop_mass and burn_duration) ─────────
    // mdot is fixed; thrust scales with sampled isp to keep mdot constant.
    let prop_mass_kg    = 550.0_f64;
    let burn_duration_s = 5.0 * 60.0 + 50.0;  // 350 s
    let mdot            = prop_mass_kg / burn_duration_s;

    // ── Distributions — all centered on config nominal values ────────────────
    let d_pitch       = Normal::new(base_cfg.burn_pitch_rad,     sigma.burn_pitch_rad).unwrap();
    let d_yaw         = Normal::new(base_cfg.burn_yaw_rad,       sigma.burn_yaw_rad).unwrap();
    let d_mass        = Normal::new(base_cfg.mass_tli_kg,        sigma.mass_tli_kg).unwrap();
    let d_area        = Normal::new(base_cfg.srp_area_m2,        sigma.srp_area_m2).unwrap();
    let d_refl        = Normal::new(base_cfg.reflectivity,       sigma.reflectivity).unwrap();
    let d_isp         = Normal::new(base_cfg.isp_s,              sigma.isp_s).unwrap();
    let d_dv          = Normal::new(base_cfg.delta_v_ms,         sigma.delta_v_ms).unwrap();
    let d_time_offset = Normal::new(base_cfg.burn_time_offset_s, sigma.burn_time_offset_s).unwrap();

    // ── Position snapshots for uncertainty-ellipse visualisation ─────────────
    // Interpolate each run's ECI position at N_SNAPSHOTS evenly-spaced epochs.
    // This keeps the snapshot file small (N × N_SNAPSHOTS rows) without storing
    // full trajectories for all runs.
    // 8 epochs from TLI to nominal Earth CA (~9 days).
    // At this spacing: T+0, 1.29, 2.57, 3.86 (≈lunar CA), 5.14, 6.43, 7.71, 9.0 days.
    const N_SNAPSHOTS: usize = 8;
    const SNAPSHOT_END_S: f64 = 9.0 * 86_400.0;
    let snapshot_times_s: Vec<f64> = (0..N_SNAPSHOTS)
        .map(|i| i as f64 * SNAPSHOT_END_S / (N_SNAPSHOTS - 1) as f64)
        .collect();
    // Each entry: [time_days, x_km, y_km, z_km]
    let mut snapshots: Vec<[f64; 4]> = Vec::with_capacity(N * N_SNAPSHOTS);

    let mut solutions = Vec::with_capacity(N);

    // ── Monte Carlo loop ──────────────────────────────────────────────────────
    for _ in 0..N {
        // Sample cfg
        let pitch = d_pitch.sample(&mut rng);
        let yaw   = d_yaw.sample(&mut rng);
        let mass  = d_mass.sample(&mut rng);
        let area  = d_area.sample(&mut rng);
        let refl  = d_refl.sample(&mut rng);
        let isp         = d_isp.sample(&mut rng);
        let dv          = d_dv.sample(&mut rng);
        let time_offset = d_time_offset.sample(&mut rng);

        let thr = mdot * isp * G0;

        // Initial state — always from Kepler elements (TLI ignition point)
        let (r0, v0) = compute_tli_initial_state(&base_cfg, moon_flyby_pos);

        // Burn ignites at ignition_time_s = time_offset: the propagator coasts
        // along the parking orbit for |time_offset| seconds before firing,
        // rotating the burn's ECI direction to explore timing uncertainty.
        let burn_dir = compute_burn_direction(&r0, &v0, pitch, yaw);
        let burn = FiniteBurn::from_delta_v(thr, isp, mass, dv, time_offset, burn_dir);

        // Extend duration to account for positive time offsets (late ignition)
        let total_duration = duration_s + time_offset.max(0.0);

        // Propagate
        let steps = propagate(
            r0, v0, mass,
            &burn,
            &moon_track,
            &sun_track,
            area,
            refl,
            total_duration,
            base_cfg.log_dt_s,
            base_cfg.rtol,
            base_cfg.atol,
        );

        // Position snapshots for uncertainty visualisation
        for &t_s in &snapshot_times_s {
            let pos = interp_pos(&steps, t_s);
            snapshots.push([t_s / 86_400.0, pos[0] / 1e3, pos[1] / 1e3, pos[2] / 1e3]);
        }

        // Lunar closest approach
        let (lunar_ca_idx, lunar_dist) = find_closest_approach(&steps);
        let lunar_altitude  = lunar_dist - MOON_RADIUS;
        let lunar_ca_time_s = steps[lunar_ca_idx].time_s;

        // Earth closest approach after 6 days (> 6*86400 s)
        let six_days_s = 6.0 * 86_400.0;
        let (_, earth_ca_dist, earth_ca_time) = find_earth_ca_after_time(&steps, six_days_s);
        let earth_altitude = earth_ca_dist - EARTH_RADIUS;

        // Get final position
        let (final_pos_x, final_pos_y, final_pos_z) = if steps.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let p = steps[steps.len() - 1].pos;
            (p[0], p[1], p[2])
        };

        // Closest passage to the OEM EI point (after 6 days, same window as Earth CA)
        let ei_target = nalgebra::SVector::<f64, 3>::new(
            OEM_EI_POS_M[0], OEM_EI_POS_M[1], OEM_EI_POS_M[2],
        );
        let (ei_dist_m, ei_time_s) = steps.iter()
            .filter(|s| s.time_s >= six_days_s)
            .fold((f64::INFINITY, 0.0), |(best_d, best_t), s| {
                let d = (s.pos - ei_target).norm();
                if d < best_d { (d, s.time_s) } else { (best_d, best_t) }
            });

        // Score: lunar CA altitude error + EI proximity (replaces Earth CA distance check)
        let lunar_target = 7000e3;
        let lunar_error  = (lunar_altitude - lunar_target).abs();
        let ei_error     = ei_dist_m;  // minimise distance to OEM EI point
        let score = lunar_error + ei_error * 0.5;  // weight tunable

        solutions.push(Solution {
            pitch,
            yaw,
            mass,
            area,
            refl,
            thr,
            isp,
            dv,
            burn_time_offset_s: time_offset,
            group: assign_group(dv - base_cfg.delta_v_ms, time_offset - base_cfg.burn_time_offset_s, sigma.delta_v_ms, sigma.burn_time_offset_s),
            lunar_ca_altitude: lunar_altitude,
            lunar_ca_time_s,
            earth_ca_altitude: earth_altitude,
            earth_ca_time_s: earth_ca_time,
            ei_dist_m,
            ei_time_s,
            final_pos_x,
            final_pos_y,
            final_pos_z,
            score,
            trajectory: steps.clone(),
        });
    }

    // Extract lunar altitudes for stats
    let lunar_alts: Vec<f64> = solutions.iter().map(|s| s.lunar_ca_altitude).collect();

    // ── Stats ─────────────────────────────────────────────────────────────────
    let mean = lunar_alts.iter().sum::<f64>() / N as f64;

    let var = lunar_alts.iter()
        .map(|d| (d - mean).powi(2))
        .sum::<f64>() / N as f64;

    let std = var.sqrt();

    let min = lunar_alts.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = lunar_alts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("Lunar CA altitude stats (km):");
    println!("mean: {:.1}", mean / 1e3);
    println!("std : {:.1}", std / 1e3);
    println!("min : {:.1}", min / 1e3);
    println!("max : {:.1}", max / 1e3);

    println!("\nEarth CA altitude stats (km):");
    let earth_alts: Vec<f64> = solutions.iter().map(|s| s.earth_ca_altitude).collect();
    let earth_mean = earth_alts.iter().sum::<f64>() / N as f64;
    let earth_min = earth_alts.iter().cloned().fold(f64::INFINITY, f64::min);
    let earth_max = earth_alts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("mean: {:.1}", earth_mean / 1e3);
    println!("min : {:.1}", earth_min / 1e3);
    println!("max : {:.1}", earth_max / 1e3);

    // Filter solutions: must pass within 10,000 km of the OEM EI point
    let ei_filter_km = 10_000e3;
    let mut filtered: Vec<_> = solutions.iter()
        .filter(|s| s.ei_dist_m < ei_filter_km)
        .collect();

    println!("\n=== Filtering: distance to OEM EI point < {:.0} km ===", ei_filter_km / 1e3);
    println!("Solutions passing EI constraint: {} / {}", filtered.len(), N);

    // Sort by score (best first)
    filtered.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    // Take top 100
    let top_100: Vec<_> = filtered.iter().take(1000).cloned().collect();
    
    println!("Top 100 solutions selected.");
    
    if !top_100.is_empty() {
        let first = top_100[0];
        println!("\n=== Best candidate (top of 100) ===");
        println!("Lunar altitude:  {:.1} km", first.lunar_ca_altitude / 1e3);
        println!("Earth altitude:  {:.1} km (at T+{:.2} days)", first.earth_ca_altitude / 1e3, first.earth_ca_time_s / 86_400.0);
        println!("EI distance:     {:.0} km (at T+{:.2} days)", first.ei_dist_m / 1e3, first.ei_time_s / 86_400.0);
        println!("Score: {:.0}", first.score);
        println!("\nParameters:");
        println!("  Burn pitch: {:.4} rad ({:.2}°)", first.pitch, first.pitch.to_degrees());
        println!("  Burn yaw:   {:.4} rad ({:.2}°)", first.yaw, first.yaw.to_degrees());
        println!("  Mass (TLI): {:.1} kg", first.mass);
        println!("  SRP area:   {:.2} m²", first.area);
        println!("  Reflectivity: {:.3}", first.refl);
        println!("  Thrust:     {:.1} N", first.thr);
        println!("  Isp:        {:.1} s", first.isp);
        println!("  Delta-V:    {:.1} m/s", first.dv);
        println!("  Burn time offset: {:.1} s", first.burn_time_offset_s);
    }

    // Save all solutions (for uncertainty propagation analysis) + top 100
    std::fs::create_dir_all("out").unwrap();
    save_position_snapshots(&snapshots, N_SNAPSHOTS, "out/mc_position_snapshots.csv");
    save_all_solutions(&solutions, "out/mc_all_solutions.csv");
    save_top_solutions(&top_100, "out/mc_top_100.csv");
    save_top_positions(&top_100, "out/mc_top_100_positions.csv");
    save_top_trajectories(&top_100, "out/mc_top_100_trajectories.csv");
    // All trajectories downsampled — used by --all flag in the plot script
    save_all_trajectories_downsampled(&solutions, "out/mc_all_trajectories.csv", 150);
}

/// Save all N runs (inputs + scalar outputs) for uncertainty propagation analysis.
/// Does not include full trajectories — keeps file size small.
fn save_all_solutions(solutions: &[Solution], path: &str) {
    use std::fmt::Write as FmtWrite;
    let mut out = String::with_capacity(solutions.len() * 200);
    writeln!(out,
        "pitch_rad,yaw_rad,mass_kg,area_m2,reflectivity,isp_s,dv_ms,burn_time_offset_s,\
         lunar_alt_km,lunar_ca_time_days,earth_alt_km,earth_ca_time_days,\
         ei_dist_km,ei_time_days,score"
    ).unwrap();
    for s in solutions {
        writeln!(out,
            "{:.6},{:.6},{:.1},{:.2},{:.3},{:.1},{:.1},{:.1},{:.1},{:.4},{:.1},{:.4},{:.1},{:.4},{:.0}",
            s.pitch, s.yaw,
            s.mass, s.area, s.refl,
            s.isp, s.dv, s.burn_time_offset_s,
            s.lunar_ca_altitude / 1e3,
            s.lunar_ca_time_s   / 86_400.0,
            s.earth_ca_altitude / 1e3,
            s.earth_ca_time_s   / 86_400.0,
            s.ei_dist_m         / 1e3,
            s.ei_time_s         / 86_400.0,
            s.score,
        ).unwrap();
    }
    std::fs::write(path, out).expect("Failed to write all-solutions CSV");
    println!("Saved all {} solutions to {}", solutions.len(), path);
}

fn save_top_solutions(solutions: &[&Solution], path: &str) {
    use std::fmt::Write as FmtWrite;
    let mut out = String::with_capacity(solutions.len() * 150);
    writeln!(out, "lunar_alt_km,earth_alt_km,earth_ca_time_days,ei_dist_km,ei_time_days,score,pitch_rad,yaw_rad,mass_kg,area_m2,reflectivity,thrust_n,isp_s,dv_ms,burn_time_offset_s").unwrap();
    for s in solutions {
        writeln!(out,
            "{:.1},{:.1},{:.2},{:.1},{:.4},{:.0},{:.6},{:.6},{:.1},{:.2},{:.3},{:.1},{:.1},{:.1},{:.1}",
            s.lunar_ca_altitude / 1e3,
            s.earth_ca_altitude / 1e3,
            s.earth_ca_time_s / 86_400.0,
            s.ei_dist_m / 1e3,
            s.ei_time_s / 86_400.0,
            s.score,
            s.pitch, s.yaw,
            s.mass, s.area, s.refl,
            s.thr, s.isp, s.dv,
            s.burn_time_offset_s,
        ).unwrap();
    }
    std::fs::write(path, out).expect("Failed to write top 100 solutions CSV");
    println!("\nSaved {} best solutions to {}", solutions.len(), path);
}

fn save_top_positions(solutions: &[&Solution], path: &str) {
    use std::fmt::Write as FmtWrite;
    let mut out = String::with_capacity(solutions.len() * 100);
    writeln!(out, "x_m,y_m,z_m,lunar_alt_km,earth_alt_km,score,group").unwrap();
    for s in solutions {
        writeln!(out,
            "{:.3},{:.3},{:.3},{:.1},{:.1},{:.0},{}",
            s.final_pos_x, s.final_pos_y, s.final_pos_z,
            s.lunar_ca_altitude / 1e3,
            s.earth_ca_altitude / 1e3,
            s.score,
            s.group,
        ).unwrap();
    }
    std::fs::write(path, out).expect("Failed to write top 100 positions CSV");
    println!("Saved {} final positions to {}", solutions.len(), path);
}

fn save_top_trajectories(solutions: &[&Solution], path: &str) {
    use std::fmt::Write as FmtWrite;
    let mut out = String::with_capacity(solutions.len() * 10_000);
    writeln!(out, "solution_idx,group,time_s,x_m,y_m,z_m,vx_ms,vy_ms,vz_ms,mass_kg,is_burn").unwrap();
    for (idx, s) in solutions.iter().enumerate() {
        for step in &s.trajectory {
            writeln!(out,
                "{},{},{:.3},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.3},{}",
                idx, s.group,
                step.time_s,
                step.pos[0], step.pos[1], step.pos[2],
                step.vel[0], step.vel[1], step.vel[2],
                step.mass_kg,
                step.is_burning as u8,
            ).unwrap();
        }
    }
    std::fs::write(path, out).expect("Failed to write top 100 trajectories CSV");
    println!("Saved {} trajectories ({} total steps) to {}",
             solutions.len(),
             solutions.iter().map(|s| s.trajectory.len()).sum::<usize>(),
             path);
}

/// Save all N trajectories downsampled to ~`steps_per_run` points each.
/// Used by the `--all` flag in the Python plot script.
fn save_all_trajectories_downsampled(solutions: &[Solution], path: &str, steps_per_run: usize) {
    use std::fmt::Write as FmtWrite;
    let mut out = String::with_capacity(solutions.len() * steps_per_run * 60);
    writeln!(out, "solution_idx,group,time_s,x_m,y_m,z_m,is_burn").unwrap();
    for (idx, s) in solutions.iter().enumerate() {
        let n      = s.trajectory.len();
        let stride = (n / steps_per_run).max(1);
        for (i, step) in s.trajectory.iter().enumerate() {
            if i % stride == 0 || i == n - 1 {
                writeln!(out,
                    "{},{},{:.3},{:.3},{:.3},{:.3},{}",
                    idx, s.group,
                    step.time_s,
                    step.pos[0], step.pos[1], step.pos[2],
                    step.is_burning as u8,
                ).unwrap();
            }
        }
    }
    std::fs::write(path, out).expect("Failed to write all trajectories CSV");
    println!("Saved {} downsampled trajectories to {}", solutions.len(), path);
}

/// Save per-run position snapshots for covariance-ellipse visualisation.
/// Layout: one row per (run × snapshot), ordered run-major.
fn save_position_snapshots(snapshots: &[[f64; 4]], n_snapshots: usize, path: &str) {
    use std::fmt::Write as FmtWrite;
    let n_runs = snapshots.len() / n_snapshots;
    let mut out = String::with_capacity(snapshots.len() * 60);
    writeln!(out, "run,time_days,x_km,y_km,z_km").unwrap();
    for (i, s) in snapshots.iter().enumerate() {
        let run = i / n_snapshots;
        writeln!(out, "{},{:.4},{:.3},{:.3},{:.3}", run, s[0], s[1], s[2], s[3]).unwrap();
    }
    std::fs::write(path, out).expect("Failed to write position snapshots CSV");
    println!("Saved {} position snapshots ({} runs × {} epochs) to {}",
             snapshots.len(), n_runs, n_snapshots, path);
}

/// Linear interpolation of ECI position at arbitrary time t_s.
/// Clamps to the first/last step if t_s is out of range.
fn interp_pos(steps: &[artemis::propagator::TrajectoryStep], t_s: f64) -> nalgebra::SVector<f64, 3> {
    if steps.is_empty() { return nalgebra::SVector::zeros(); }
    if t_s <= steps[0].time_s { return steps[0].pos; }
    let last = steps.last().unwrap();
    if t_s >= last.time_s { return last.pos; }
    let idx  = steps.partition_point(|s| s.time_s <= t_s).saturating_sub(1);
    let lo   = &steps[idx];
    let hi   = &steps[(idx + 1).min(steps.len() - 1)];
    let frac = if (hi.time_s - lo.time_s).abs() < 1e-12 { 0.0 }
               else { (t_s - lo.time_s) / (hi.time_s - lo.time_s) };
    lo.pos + frac * (hi.pos - lo.pos)
}

fn find_earth_ca_after_time(steps: &[artemis::propagator::TrajectoryStep], min_time_s: f64) -> (usize, f64, f64) {
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    let mut best_time = 0.0;
    
    for (i, step) in steps.iter().enumerate() {
        if step.time_s < min_time_s {
            continue;  // Skip before min time
        }
        
        let dist = step.pos.norm();  // Distance to Earth (at origin)
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
            best_time = step.time_s;
        }
    }
    
    (best_idx, best_dist, best_time)
}