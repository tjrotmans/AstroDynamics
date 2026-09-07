//! Deterministic trajectory regression test for the Artemis 2 simulation.
//!
//! Because the propagator is deterministic (fixed config + fixed ephemeris → identical
//! IEEE 754 results on the same platform), this test catches any unintended change to
//! physics models, integrator parameters, or the force loop.
//!
//! # Usage
//!
//! Run locally before any push (requires kernels/de440s.bsp):
//!
//!   cargo test -p artemis --test regression -- --ignored --nocapture
//!
//! # First-time setup
//!
//! 1. Set `GOLDEN_VALUES_SET = false` (already the default).
//! 2. Run the test — it will print actual values and then panic.
//! 3. Copy the printed values into the `GOLDEN_*` constants below.
//! 4. Set `GOLDEN_VALUES_SET = true`.
//! 5. Re-run — test now passes and is locked to those values.

use ephemeris::{Almanac, Body};
use hifitime::Duration;
use orbital_models::constants::MOON_RADIUS;

use orbital_models::FiniteBurn;
use artemis::config::MissionConfig;
use artemis::orbit::{compute_burn_direction, compute_tli_initial_state, find_closest_approach};
use ephemeris::{MoonTrack, SunTrack};
use artemis::propagator::propagate;

// ── Golden values ─────────────────────────────────────────────────────────────
// Set GOLDEN_VALUES_SET to false on first run to generate these, then fill them
// in and flip the flag to true.

/// Set to true after filling in the GOLDEN_* constants from a first run.
const GOLDEN_VALUES_SET: bool = true;

/// Final ECI position [m] at end of simulation.
const GOLDEN_FINAL_POS: [f64; 3] = [-1.61829103446048021e8, -2.24165728856852859e8, -1.01203811557630032e8];
/// Final ECI velocity [m/s] at end of simulation.
const GOLDEN_FINAL_VEL: [f64; 3] = [-3.27597120937076852e2, -8.00187071462241988e2, -4.46230942627789716e2];
/// Final spacecraft mass [kg].
const GOLDEN_FINAL_MASS: f64 = 2.57547310633443230e4;
/// Moon-center distance at closest approach [m].
const GOLDEN_LUNAR_CA_M: f64 = 8.84927022072117217e6;

// ── Tolerances ────────────────────────────────────────────────────────────────
// Matched to the integrator atol (1e-8 m) — should never be hit on the same machine.
// Loosened slightly only if this test is ever moved to cross-platform CI.

/// Position error tolerance [m].
const TOL_POS_M: f64 = 1e-3;
/// Velocity error tolerance [m/s].
const TOL_VEL_MS: f64 = 1e-6;
/// Mass error tolerance [kg].
const TOL_MASS_KG: f64 = 1e-9;
/// Closest-approach distance tolerance [m].
const TOL_CA_M: f64 = 1e-3;

// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires kernels/de440s.bsp — run locally before pushing, not in CI"]
fn artemis2_full_trajectory_regression() {
    const MOON_TRACK_SAMPLES: usize = 10_000;

    let cfg = MissionConfig::artemis2();

    let almanac = Almanac::new(&ephemeris::find_kernel("de440s.bsp"))
        .expect("Failed to load de440s.bsp kernel");

    // ── Pre-sample Moon and Sun tracks (identical to main.rs) ────────────────
    let duration_s = cfg.mission_duration_days * 86_400.0;
    let total_duration_s = duration_s + cfg.burn_time_offset_s.max(0.0);
    let sample_dt = duration_s / (MOON_TRACK_SAMPLES as f64 - 1.0);

    let moon_flyby_pos = almanac
        .body_state_eci(Body::Moon, cfg.flyby_epoch)
        .expect("Moon ephemeris query failed")
        .position
        .inner;

    let mut moon_positions = Vec::with_capacity(MOON_TRACK_SAMPLES);
    let mut sun_positions = Vec::with_capacity(MOON_TRACK_SAMPLES);
    for i in 0..MOON_TRACK_SAMPLES {
        let epoch = cfg.tli_epoch + Duration::from_seconds(i as f64 * sample_dt);
        moon_positions.push(
            almanac
                .body_state_eci(Body::Moon, epoch)
                .expect("Moon ephemeris query failed")
                .position
                .inner,
        );
        sun_positions.push(
            almanac
                .body_state_eci(Body::Sun, epoch)
                .expect("Sun ephemeris query failed")
                .position
                .inner,
        );
    }

    let moon_track = MoonTrack { positions: moon_positions, sample_dt_s: sample_dt };
    let sun_track = SunTrack { positions: sun_positions, sample_dt_s: sample_dt };

    // ── Initial state and burn (identical to main.rs) ─────────────────────────
    let (initial_pos, initial_vel) = compute_tli_initial_state(&cfg, moon_flyby_pos);
    let burn_dir = compute_burn_direction(
        &initial_pos, &initial_vel, cfg.burn_pitch_rad, cfg.burn_yaw_rad,
    );
    let burn = FiniteBurn::from_delta_v(
        cfg.thrust_n, cfg.isp_s, cfg.mass_tli_kg,
        cfg.delta_v_ms, cfg.burn_time_offset_s, burn_dir,
    );

    // ── Propagate ─────────────────────────────────────────────────────────────
    let steps = propagate(
        initial_pos, initial_vel, cfg.mass_tli_kg,
        &burn, &moon_track, &sun_track,
        cfg.srp_area_m2, cfg.reflectivity,
        total_duration_s, cfg.log_dt_s, cfg.rtol, cfg.atol,
    );
    assert!(!steps.is_empty(), "propagation returned no steps");

    let last = steps.last().unwrap();
    let (_, lunar_ca_m) = find_closest_approach(&steps);
    let flyby_alt_km = (lunar_ca_m - MOON_RADIUS) / 1e3;

    // ── Print actual values (always, for easy copy-paste on first run) ────────
    println!("Final state at T+{:.4} days:", last.time_s / 86_400.0);
    println!("  GOLDEN_FINAL_POS:  [{:.17e}, {:.17e}, {:.17e}]", last.pos[0], last.pos[1], last.pos[2]);
    println!("  GOLDEN_FINAL_VEL:  [{:.17e}, {:.17e}, {:.17e}]", last.vel[0], last.vel[1], last.vel[2]);
    println!("  GOLDEN_FINAL_MASS: {:.17e}", last.mass_kg);
    println!("  GOLDEN_LUNAR_CA_M: {:.17e}  ({:.1} km altitude)", lunar_ca_m, flyby_alt_km);

    if !GOLDEN_VALUES_SET {
        panic!(
            "Golden values not set — see instructions at the top of this file.\n\
             Copy the GOLDEN_* values printed above into the constants, \
             then set GOLDEN_VALUES_SET = true."
        );
    }

    // ── Assert against golden values ──────────────────────────────────────────
    use nalgebra::SVector;
    let golden_pos = SVector::<f64, 3>::from_column_slice(&GOLDEN_FINAL_POS);
    let golden_vel = SVector::<f64, 3>::from_column_slice(&GOLDEN_FINAL_VEL);

    let pos_err = (last.pos - golden_pos).norm();
    let vel_err = (last.vel - golden_vel).norm();
    let mass_err = (last.mass_kg - GOLDEN_FINAL_MASS).abs();
    let ca_err = (lunar_ca_m - GOLDEN_LUNAR_CA_M).abs();

    assert!(
        pos_err < TOL_POS_M,
        "Position regression: {pos_err:.6} m (tolerance {TOL_POS_M} m)"
    );
    assert!(
        vel_err < TOL_VEL_MS,
        "Velocity regression: {vel_err:.9} m/s (tolerance {TOL_VEL_MS} m/s)"
    );
    assert!(
        mass_err < TOL_MASS_KG,
        "Mass regression: {mass_err:.12} kg (tolerance {TOL_MASS_KG} kg)"
    );
    assert!(
        ca_err < TOL_CA_M,
        "Lunar closest-approach regression: {ca_err:.6} m (tolerance {TOL_CA_M} m)"
    );
}
