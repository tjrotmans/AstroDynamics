//! Artemis 2 TLI + Free-Return Trajectory Simulation
//!
//! Simulates the Trans-Lunar Injection (TLI) burn and subsequent free-return
//! coast of Orion in the ECI (J2000) frame, starting from the perigee of the
//! pre-TLI parking orbit on.
//!
//! Uses burn angles from MissionConfig. Run `cargo run -p artemis --bin target`
//! first to find the angles that achieve the 6,513 km lunar flyby altitude.
//!
//! # Requirements
//! Ensure `kernels/de440s.bsp` is in the root directory.

use nalgebra::SVector;

use ephemeris::{Almanac, Body};
use hifitime::Duration;

use orbital_models::FiniteBurn;
use artemis::config::MissionConfig;
use artemis::orbit::{compute_tli_initial_state, compute_burn_direction, find_closest_approach};
use ephemeris::{MoonTrack, SunTrack};
use artemis::propagator::propagate;
use orbital_models::constants::{EARTH_RADIUS, MOON_RADIUS, MU_EARTH};

fn main() {
    const OUT_CSV: &str = "out/artemis2_trajectory.csv";
    const MOON_TRACK_SAMPLES: usize = 10_000;

    let cfg = MissionConfig::artemis2();

    // ── Load ephemeris ────────────────────────────────────────────────────────
    println!("Loading ephemeris kernel...");
    let almanac = Almanac::new(&ephemeris::find_kernel("de440s.bsp"))
        .expect("Failed to load de440s.bsp kernel");
    println!("Ephemeris loaded.\n");

    // ── Query key ephemeris states ────────────────────────────────────────────
    let moon_flyby_pos = almanac.body_state_eci(Body::Moon, cfg.flyby_epoch)
        .expect("Failed to query Moon state at flyby epoch").position.inner;
    let sun_pos_tli = almanac.body_state_eci(Body::Sun, cfg.tli_epoch)
        .expect("Failed to query Sun state at TLI epoch").position.inner;

    println!("Moon at flyby (April 6):  r = {:.0} km ({:.3} Moon-distances)",
        moon_flyby_pos.norm() / 1e3,
        moon_flyby_pos.norm() / 384_400_000.0);
    println!("Sun at TLI:               r = {:.3e} m ({:.3} AU)",
        sun_pos_tli.norm(), sun_pos_tli.norm() / 1.496e11);

    // ── Orbital period of parking orbit ──────────────────────────────────────
    let semi_major_axis = cfg.initial_kepler_elements
        .map(|e| e.a)
        .unwrap_or(EARTH_RADIUS + cfg.parking_perigee_alt_m);
    let orbit_period_s = 2.0 * std::f64::consts::PI * (semi_major_axis.powi(3) / MU_EARTH).sqrt();
    println!("Parking orbit period:     {:.1} s  ({:.2} h)", orbit_period_s, orbit_period_s / 3600.0);

    // ── Pre-sample Moon and Sun tracks (anchored to TLI epoch) ───────────────
    let duration_s = cfg.mission_duration_days * 86_400.0;
    let sample_dt  = duration_s / (MOON_TRACK_SAMPLES as f64 - 1.0);
    let mut moon_positions: Vec<SVector<f64, 3>> = Vec::with_capacity(MOON_TRACK_SAMPLES);
    let mut sun_positions:  Vec<SVector<f64, 3>> = Vec::with_capacity(MOON_TRACK_SAMPLES);

    print!("Sampling Moon and Sun tracks ({} points)...", MOON_TRACK_SAMPLES);
    for i in 0..MOON_TRACK_SAMPLES {
        let epoch = cfg.tli_epoch + Duration::from_seconds(i as f64 * sample_dt);
        moon_positions.push(
            almanac.body_state_eci(Body::Moon, epoch)
                .expect("Moon ephemeris query failed").position.inner
        );
        sun_positions.push(
            almanac.body_state_eci(Body::Sun, epoch)
                .expect("Sun ephemeris query failed").position.inner
        );
    }
    println!(" done.");

    let moon_track = MoonTrack { positions: moon_positions, sample_dt_s: sample_dt };
    let sun_track  = SunTrack  { positions: sun_positions,  sample_dt_s: sample_dt };

    // ── Initial state (always from Kepler elements — identical to MC) ─────────
    let (initial_pos, initial_vel) = compute_tli_initial_state(&cfg, moon_flyby_pos);
    let alt_km  = (initial_pos.norm() - EARTH_RADIUS) / 1e3;
    println!("\nTLI ignition point:");
    println!("  Altitude:    {:.1} km  (target: {:.1} km)", alt_km, cfg.parking_perigee_alt_m / 1e3);
    println!("  Speed:       {:.3} km/s", initial_vel.norm() / 1e3);
    println!("  ECI pos:     [{:.0}, {:.0}, {:.0}] km",
        initial_pos[0]/1e3, initial_pos[1]/1e3, initial_pos[2]/1e3);

    // ── TLI burn ──────────────────────────────────────────────────────────────
    let burn_dir = compute_burn_direction(&initial_pos, &initial_vel, cfg.burn_pitch_rad, cfg.burn_yaw_rad);
    // ignition_time_s = burn_time_offset_s: matches the MC exactly (coast along parking orbit first).
    let burn = FiniteBurn::from_delta_v(cfg.thrust_n, cfg.isp_s, cfg.mass_tli_kg, cfg.delta_v_ms, cfg.burn_time_offset_s, burn_dir);
    let prop_mass = burn.propellant_mass_kg(cfg.mass_tli_kg);

    println!("\nTLI burn:");
    println!("  ΔV:          {:.1} m/s", cfg.delta_v_ms);
    println!("  Duration:    {:.1} s  ({:.0} min {:.0} s)",
        burn.duration_s(), (burn.duration_s() / 60.0).floor(), burn.duration_s() % 60.0);
    println!("  Thrust:      {:.1} kN, Isp: {:.0} s", cfg.thrust_n / 1e3, cfg.isp_s);
    println!("  Propellant:  {:.0} kg  (mass after burn: {:.0} kg)", prop_mass, cfg.mass_tli_kg - prop_mass);
    println!("  Direction:   prograde + pitch={:.3}° yaw={:.3}°",
        cfg.burn_pitch_rad.to_degrees(), cfg.burn_yaw_rad.to_degrees());

    // ── Pre-burn parking orbit coast (for plotting only) ──────────────────────
    // Propagate one orbit from the TLI state with a dummy never-firing burn.
    // This traces the parking orbit without affecting the main simulation state.
    // Moon positions are slightly offset in time (~23.5 h) but acceptable for a plot.
    let dummy_burn = FiniteBurn {
        thrust_n:        0.0,
        isp_s:           1.0,
        ignition_time_s: f64::INFINITY,
        cutoff_time_s:   f64::INFINITY,
        direction:       SVector::zeros(),
    };
    let pre_burn_steps: Vec<_> = propagate(
        initial_pos, initial_vel, cfg.mass_tli_kg,
        &dummy_burn, &moon_track, &sun_track,
        cfg.srp_area_m2, cfg.reflectivity,
        orbit_period_s, cfg.log_dt_s, cfg.rtol, cfg.atol,
    ).into_iter().map(|mut s| { s.time_s -= orbit_period_s; s }).collect();

    // ── Main propagation (exact TLI state, identical to MC) ───────────────────
    println!("\nPropagating {:.0}-day trajectory...", cfg.mission_duration_days);
    let total_duration_s = duration_s + cfg.burn_time_offset_s.max(0.0);
    let main_steps = propagate(
        initial_pos, initial_vel, cfg.mass_tli_kg,
        &burn, &moon_track, &sun_track,
        cfg.srp_area_m2, cfg.reflectivity,
        total_duration_s, cfg.log_dt_s, cfg.rtol, cfg.atol,
    );
    println!("  {} states logged.", main_steps.len());

    // ── Mass during TLI burn (fine-grained re-propagation) ───────────────────
    let burn_log_dt_s = 5.0; // 5-second steps during burn
    let burn_fine_steps = propagate(
        initial_pos, initial_vel, cfg.mass_tli_kg,
        &burn, &moon_track, &sun_track,
        cfg.srp_area_m2, cfg.reflectivity,
        burn.cutoff_time_s + burn_log_dt_s,
        burn_log_dt_s,
        cfg.rtol, cfg.atol,
    );
    let burn_steps: Vec<_> = burn_fine_steps.iter().filter(|s| s.is_burning).collect();
    println!("\nMass during TLI burn ({}s steps):", burn_log_dt_s as i32);
    println!("  {:>10}  {:>12}", "t_burn (s)", "mass (kg)");
    for s in &burn_steps {
        println!("  {:>10.1}  {:>12.1}", s.time_s - burn.ignition_time_s, s.mass_kg);
    }
    if let (Some(first), Some(last)) = (burn_steps.first(), burn_steps.last()) {
        println!("  Δmass consumed: {:.1} kg", first.mass_kg - last.mass_kg);
    }

    // ── Closest approach ──────────────────────────────────────────────────────
    let (ca_idx, ca_dist) = find_closest_approach(&main_steps);
    let ca_step = &main_steps[ca_idx];
    let flyby_alt_km = (ca_dist - MOON_RADIUS) / 1e3;
    let ca_day = ca_step.time_s / 86_400.0;
    println!("\nClosest approach to Moon:");
    println!("  Distance:    {:.0} km from Moon center", ca_dist / 1e3);
    println!("  Altitude:    {:.0} km  (target: 6,513 km)", flyby_alt_km);
    println!("  Time:        T+{:.2} days (April {:.1})", ca_day, 2.0 + ca_day);

    // ── Save CSV (pre-burn orbit prepended with negative time_s, then main) ───
    std::fs::create_dir_all("out").unwrap();
    save_csv(&pre_burn_steps, &main_steps, OUT_CSV);
    println!("\nTrajectory saved to {}", OUT_CSV);
    println!("Plot ECI frame:      python plot/plot_eci.py");
    println!("Plot rotating frame: python plot/plot_rotating.py");
}

fn save_csv(pre_burn: &[artemis::propagator::TrajectoryStep], main: &[artemis::propagator::TrajectoryStep], path: &str) {
    use std::fmt::Write as FmtWrite;
    let total = pre_burn.len() + main.len();
    let mut out = String::with_capacity(total * 120);
    writeln!(out, "time_s,x_m,y_m,z_m,vx_ms,vy_ms,vz_ms,mass_kg,moon_x_m,moon_y_m,moon_z_m,sun_x_m,sun_y_m,sun_z_m,is_burn").unwrap();
    for s in pre_burn.iter().chain(main.iter()) {
        writeln!(out,
            "{:.3},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}",
            s.time_s, s.pos[0], s.pos[1], s.pos[2],
            s.vel[0], s.vel[1], s.vel[2], s.mass_kg,
            s.moon_pos[0], s.moon_pos[1], s.moon_pos[2],
            s.sun_pos[0],  s.sun_pos[1],  s.sun_pos[2],
            s.is_burning as u8,
        ).unwrap();
    }
    std::fs::write(path, out).expect("Failed to write trajectory CSV");
    println!("  {} rows written ({} pre-burn + {} main).", total, pre_burn.len(), main.len());
}
