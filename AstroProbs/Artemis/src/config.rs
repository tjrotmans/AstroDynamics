//! Artemis 2 mission configuration
//!
//! Spacecraft parameters, ICPS engine specs, and initial orbital conditions
//! derived from NASA's public Artemis II Reference Guide and mission blogs.
//!
//! # Data sources
//! - NASA Artemis II Reference Guide (January 2026)
//! - NASA flight day blogs (April 2026)
//!
//! # Known inconsistencies in public data
//! The reported TLI burn duration (5:50) and ΔV (388 m/s) are inconsistent with
//! ICPS thrust (110.1 kN) and Isp (465 s): those specs imply ~89 s for 388 m/s.
//! The 5:50 duration appears to be from Artemis 1 (TLI from 200 km orbit, ~3 km/s).
//! This simulation uses ΔV as the primary parameter and derives burn duration from it.

use hifitime::Epoch;
use orbital_models::OrbitalElements;
use orbital_models::constants::EARTH_RADIUS;

/// Earth mean radius [m] — re-exported from orbital_models::constants::EARTH_RADIUS.
pub const EARTH_RADIUS_M: f64 = EARTH_RADIUS;

/// Artemis 2 mission parameters.
pub struct MissionConfig {
    // ── Spacecraft ───────────────────────────────────────────────────────────
    /// Mass at TLI ignition [kg]. Source: NASA flight day 2 blog.
    pub mass_tli_kg: f64,
    /// Spacecraft projected area for solar radiation pressure [m²].
    /// Estimated for Orion CM + SM combined face area.
    pub srp_area_m2: f64,
    /// SRP reflectivity coefficient (0 = absorber, 1 = perfect reflector).
    pub reflectivity: f64,

    // ── ICPS engine (RL10C-2) ────────────────────────────────────────────────
    /// Engine vacuum thrust [N]. Source: NASA ICPS reference.
    pub thrust_n: f64,
    /// Engine specific impulse [s]. Source: NASA RL10 reference.
    pub isp_s: f64,
    /// TLI delta-V [m/s]. Source: NASA flight day 2 blog ("1,274 ft/s").
    /// Burn duration is derived from this and the ICPS specs via Tsiolkovsky.
    pub delta_v_ms: f64,

    // ── Parking orbit (before TLI) ───────────────────────────────────────────
    /// Perigee altitude above Earth surface [m]. Source: NASA mission blog.
    pub parking_perigee_alt_m: f64,
    /// Apogee altitude above Earth surface [m]. Source: NASA mission blog.
    pub parking_apogee_alt_m: f64,
    /// Orbital inclination [rad]. Constrained by KSC launch latitude.
    pub inclination_rad: f64,
    /// Optional override for the parking orbit Kepler elements at TLI ignition.
    /// If `None`, the code computes the parking orbit from Moon geometry.
    pub initial_kepler_elements: Option<OrbitalElements>,

    // ── Burn steering offset from prograde ───────────────────────────────────
    /// Pitch offset from prograde [rad]. Tune to target Moon flyby altitude.
    /// Positive = pitch up (away from equatorial plane).
    pub burn_pitch_rad: f64,
    /// Yaw offset from prograde [rad]. Tune to correct Moon arrival azimuth.
    pub burn_yaw_rad: f64,
    /// Coast time before TLI ignition [s]. Propagator drifts along parking orbit
    /// for this duration before firing. Paste `burn_time_offset_s` from MC output.
    pub burn_time_offset_s: f64,

    // ── Epochs ───────────────────────────────────────────────────────────────
    /// TLI ignition epoch. Source: NASA flight day 2 blog (7:49 PM EDT = 23:49 UTC).
    pub tli_epoch: Epoch,
    /// Expected lunar flyby epoch. Source: NASA mission timeline (April 6, ~7 PM EDT).
    pub flyby_epoch: Epoch,
    /// Total simulation duration [days]. Full free-return is ~10 days.
    pub mission_duration_days: f64,

    // ── Integrator ───────────────────────────────────────────────────────────
    /// Relative tolerance for the Dormand-Prince RK45 solver.
    pub rtol: f64,
    /// Absolute tolerance [m / m/s / kg depending on component].
    pub atol: f64,
    /// Logging interval [s] — one output point per this many simulated seconds.
    pub log_dt_s: f64,
}

impl MissionConfig {
    /// Real Artemis 2 parameters as reported by NASA during the mission (April 2026).
    pub fn artemis2() -> Self {
        Self {
            // Spacecraft
            mass_tli_kg:  26_362.4,
            srp_area_m2:  11.51,
            reflectivity: 0.381,

            // ICPS engine
            thrust_n:    28_462.7,
            isp_s:       1847.0,
            delta_v_ms:  422.4,

            // Parking orbit — 378 km × 109,400 km, i ≈ 28.5°
            parking_perigee_alt_m: 378_000.0,
            parking_apogee_alt_m:  109_400_000.0,
            inclination_rad:       28.5_f64.to_radians(),
            initial_kepler_elements: Some(OrbitalElements {
                a: 41649.837856e3, // meters
                e: 0.842401928,
                i: 28.30247959_f64.to_radians(),
                o: 359.28462570_f64.to_radians(),
                w: 77.68094215_f64.to_radians(),
                nu: 352.62086874_f64.to_radians(),
            }),

            // Burn steering: start prograde, tune these to match flyby altitude
            burn_pitch_rad:     -0.1053,
            burn_yaw_rad:       -0.0952,
            burn_time_offset_s: 393.6,

            // TLI: April 2, 2026 23:49 UTC
            tli_epoch:   Epoch::from_gregorian_utc(2026, 4, 2, 23, 49, 0, 0),
            // Flyby: April 6, 2026 19:05 UTC
            flyby_epoch: Epoch::from_gregorian_utc(2026, 4, 6, 19, 5, 0, 0),

            mission_duration_days: 10.0,

            // Integrator tolerances — tight enough for sub-km accuracy over 10 days
            rtol:     1e-10,
            atol:     1e-8,   // 1 mm position, 1 mm/s velocity, ~0 kg mass
            log_dt_s: 300.0,  // one output point per 5 min → ~2,880 points over 10 days
        }
    }
}
