//! Artemis 2 Grid Search — one-at-a-time sensitivity analysis
//!
//! Varies each key parameter (ΔV, burn timing) independently across its range
//! (250 samples each, negative and positive directions), keeping all others at nominal.
//! Shows isolated effect of each parameter without confounding interactions.
//!
//! Run from AstroProbs/Artemis after placing kernels/de440s.bsp:
//!   cargo run -p artemis --bin grid --release

use ephemeris::{Almanac, Body};
use hifitime::Duration;

use orbital_models::FiniteBurn;
use orbital_models::constants::G0;
use artemis::config::MissionConfig;
use artemis::orbit::{compute_tli_initial_state, compute_burn_direction, find_closest_approach};
use ephemeris::{MoonTrack, SunTrack};
use artemis::propagator::propagate;
use orbital_models::constants::{MOON_RADIUS, EARTH_RADIUS};

const N_SAMPLES_PER_GROUP: usize = 200;  // 200 negative + 200 positive = 400 per parameter = 800 total

// ── OEM Entry Interface target point ──────────────────────────────────────────
const OEM_EI_POS_M: [f64; 3] = [
    3_946_027.553,
    4_784_562.414,
    1_995_148.492,
];

// ── Solution struct — same as MC ──────────────────────────────────────────────
#[derive(Clone)]
struct Solution {
    dv: f64,
    burn_time_offset_s: f64,
    group: &'static str,
    lunar_ca_altitude: f64,
    lunar_ca_time_s: f64,
    earth_ca_altitude: f64,
    earth_ca_time_s: f64,
    ei_dist_m: f64,
    ei_time_s: f64,
    final_pos_x: f64,
    final_pos_y: f64,
    final_pos_z: f64,
    score: f64,
    trajectory: Vec<artemis::propagator::TrajectoryStep>,
}

fn main() {
    let base_cfg = MissionConfig::artemis2();

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

    // ── Fixed mass flow rate (same as MC) ──────────────────────────────────────
    let prop_mass_kg    = 550.0_f64;
    let burn_duration_s = 5.0 * 60.0 + 50.0;
    let mdot            = prop_mass_kg / burn_duration_s;

    let mut solutions = Vec::with_capacity(N_SAMPLES_PER_GROUP * 4);

    println!("=== Grid Search: One-at-a-time Sensitivity ===");
    println!("Nominal ΔV: {:.1} m/s", base_cfg.delta_v_ms);
    println!("Nominal burn time offset: {:.1} s", base_cfg.burn_time_offset_s);

    // ── Grid 1: Low ΔV (negative perturbation) ────────────────────────────────
    {
        let sigma_dv = 5.0;
        let dv_min = (base_cfg.delta_v_ms - 3.0 * sigma_dv).max(0.1);
        let dv_max = base_cfg.delta_v_ms - 0.1 * sigma_dv;
        println!("\nGroup 'low_dv': varying ΔV from {:.1} to {:.1} m/s", dv_min, dv_max);

        for i in 0..N_SAMPLES_PER_GROUP {
            let frac = i as f64 / (N_SAMPLES_PER_GROUP - 1) as f64;
            let dv = dv_min + frac * (dv_max - dv_min);

            let sol = run_trajectory(
                &base_cfg,
                &moon_flyby_pos,
                &moon_track,
                &sun_track,
                mdot,
                dv,
                base_cfg.burn_time_offset_s,
                "low_dv",
            );
            solutions.push(sol);
        }
    }

    // ── Grid 2: High ΔV (positive perturbation) ───────────────────────────────
    {
        let sigma_dv = 5.0;
        let dv_min = base_cfg.delta_v_ms + 0.1 * sigma_dv;
        let dv_max = base_cfg.delta_v_ms + 3.0 * sigma_dv;
        println!("Group 'high_dv': varying ΔV from {:.1} to {:.1} m/s", dv_min, dv_max);

        for i in 0..N_SAMPLES_PER_GROUP {
            let frac = i as f64 / (N_SAMPLES_PER_GROUP - 1) as f64;
            let dv = dv_min + frac * (dv_max - dv_min);

            let sol = run_trajectory(
                &base_cfg,
                &moon_flyby_pos,
                &moon_track,
                &sun_track,
                mdot,
                dv,
                base_cfg.burn_time_offset_s,
                "high_dv",
            );
            solutions.push(sol);
        }
    }

    // ── Grid 3: Early burn (timing offset LESS than nominal — earlier ignition) ──
    {
        let t_min = 50.0;  // Start from zero (or could go negative for even earlier ignition)
        let t_max = base_cfg.burn_time_offset_s;  // Up to nominal
        println!("Group 'early_burn': varying offset from {:.1} to {:.1} s (earlier than nominal)", t_min, t_max);

        for i in 0..N_SAMPLES_PER_GROUP {
            let frac = i as f64 / (N_SAMPLES_PER_GROUP - 1) as f64;
            let time_offset = t_min + frac * (t_max - t_min);

            let sol = run_trajectory(
                &base_cfg,
                &moon_flyby_pos,
                &moon_track,
                &sun_track,
                mdot,
                base_cfg.delta_v_ms,
                time_offset,
                "early_burn",
            );
            solutions.push(sol);
        }
    }

    // ── Grid 4: Late burn (timing offset MORE than nominal — later ignition) ────
    {
        let sigma_time = 120.0;
        let t_min = base_cfg.burn_time_offset_s+30.0;  // Start from nominal
        let t_max = base_cfg.burn_time_offset_s + 3.0 * sigma_time;  // Up to 3σ beyond nominal
        println!("Group 'late_burn': varying offset from {:.1} to {:.1} s (later than nominal)", t_min, t_max);

        for i in 0..N_SAMPLES_PER_GROUP {
            let frac = i as f64 / (N_SAMPLES_PER_GROUP - 1) as f64;
            let time_offset = t_min + frac * (t_max - t_min);

            let sol = run_trajectory(
                &base_cfg,
                &moon_flyby_pos,
                &moon_track,
                &sun_track,
                mdot,
                base_cfg.delta_v_ms,
                time_offset,
                "late_burn",
            );
            solutions.push(sol);
        }
    }

    // ── Save outputs ──────────────────────────────────────────────────────────
    std::fs::create_dir_all("out").unwrap();
    save_trajectories(&solutions, "out/grid_trajectories.csv");
    save_positions(&solutions, "out/grid_positions.csv");

    println!("\n=== Complete ===");
    println!("Saved {} grid trajectories", solutions.len());
}

/// Run one trajectory with fixed parameters
fn run_trajectory(
    base_cfg: &artemis::config::MissionConfig,
    moon_flyby_pos: &nalgebra::SVector<f64, 3>,
    moon_track: &ephemeris::MoonTrack,
    sun_track: &ephemeris::SunTrack,
    mdot: f64,
    dv: f64,
    time_offset: f64,
    group_label: &'static str,
) -> Solution {
    // Grid samples use nominal pitch/yaw (no angular dispersion)
    let mass = base_cfg.mass_tli_kg;
    let thr = mdot * base_cfg.isp_s * G0;

    // Propagate
    let (r0, v0) = compute_tli_initial_state(base_cfg, *moon_flyby_pos);
    let burn_dir = compute_burn_direction(&r0, &v0, base_cfg.burn_pitch_rad, base_cfg.burn_yaw_rad);
    let burn = FiniteBurn::from_delta_v(thr, base_cfg.isp_s, mass, dv, time_offset, burn_dir);

    let duration_s = base_cfg.mission_duration_days * 86_400.0;
    let total_duration = duration_s + time_offset.max(0.0);

    let steps = propagate(
        r0, v0, mass,
        &burn,
        moon_track,
        sun_track,
        base_cfg.srp_area_m2,
        base_cfg.reflectivity,
        total_duration,
        base_cfg.log_dt_s,
        base_cfg.rtol,
        base_cfg.atol,
    );

    // Lunar CA
    let (lunar_ca_idx, lunar_dist) = find_closest_approach(&steps);
    let lunar_altitude = lunar_dist - MOON_RADIUS;
    let lunar_ca_time_s = steps[lunar_ca_idx].time_s;

    // Earth CA after 6 days
    let six_days_s = 6.0 * 86_400.0;
    let (_, earth_ca_dist, earth_ca_time) = find_earth_ca_after_time(&steps, six_days_s);
    let earth_altitude = earth_ca_dist - EARTH_RADIUS;

    // Final position
    let (final_pos_x, final_pos_y, final_pos_z) = if steps.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let p = steps[steps.len() - 1].pos;
        (p[0], p[1], p[2])
    };

    // EI distance (after 6 days)
    let ei_target = nalgebra::SVector::<f64, 3>::new(
        OEM_EI_POS_M[0], OEM_EI_POS_M[1], OEM_EI_POS_M[2],
    );
    let (ei_dist_m, ei_time_s) = steps.iter()
        .filter(|s| s.time_s >= six_days_s)
        .fold((f64::INFINITY, 0.0), |(best_d, best_t), s| {
            let d = (s.pos - ei_target).norm();
            if d < best_d { (d, s.time_s) } else { (best_d, best_t) }
        });

    // Score
    let lunar_target = 7000e3;
    let lunar_error = (lunar_altitude - lunar_target).abs();
    let ei_error = ei_dist_m;
    let score = lunar_error + ei_error * 0.5;

    Solution {
        dv,
        burn_time_offset_s: time_offset,
        group: group_label,
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
        trajectory: steps,
    }
}

fn save_trajectories(solutions: &[Solution], path: &str) {
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

    std::fs::write(path, out).expect("Failed to write trajectories CSV");
    println!("Saved {} trajectories ({} total steps) to {}",
             solutions.len(),
             solutions.iter().map(|s| s.trajectory.len()).sum::<usize>(),
             path);
}

fn save_positions(solutions: &[Solution], path: &str) {
    use std::fmt::Write as FmtWrite;
    let mut out = String::with_capacity(solutions.len() * 100);
    writeln!(out, "group,dv_ms,burn_offset_s,lunar_alt_km,lunar_ca_time_s,earth_alt_km,earth_ca_time_s,ei_dist_km,ei_time_s,x_m,y_m,z_m,score").unwrap();

    for s in solutions {
        writeln!(out,
            "{},{:.3},{:.3},{:.1},{:.1},{:.1},{:.1},{:.3},{:.1},{:.3},{:.3},{:.3},{:.0}",
            s.group,
            s.dv, s.burn_time_offset_s,
            s.lunar_ca_altitude / 1e3, s.lunar_ca_time_s,
            s.earth_ca_altitude / 1e3, s.earth_ca_time_s,
            s.ei_dist_m / 1e3, s.ei_time_s,
            s.final_pos_x, s.final_pos_y, s.final_pos_z,
            s.score,
        ).unwrap();
    }

    std::fs::write(path, out).expect("Failed to write positions CSV");
    println!("Saved {} final positions to {}", solutions.len(), path);
}

fn find_earth_ca_after_time(steps: &[artemis::propagator::TrajectoryStep], min_time_s: f64) -> (usize, f64, f64) {
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    let mut best_time = 0.0;

    for (i, step) in steps.iter().enumerate() {
        if step.time_s < min_time_s {
            continue;
        }
        let dist = step.pos.norm();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
            best_time = step.time_s;
        }
    }

    (best_idx, best_dist, best_time)
}
