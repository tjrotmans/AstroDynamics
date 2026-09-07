//! Shared configuration defaults for solar sail scenarios.

use orbital_models::OrbitalElements;

/// Mission-level configuration for orbit raising scenarios.
#[derive(Clone, Copy, Debug)]
pub struct MissionConfig {
    pub initial_altitude: f64,
    pub target_altitude: f64,
    pub sail_area: f64,
    pub mass: f64,
    pub reflectivity: f64,
    pub drag_coefficient: f64,
    pub target_time: f64,
    pub max_slew_rate: f64,
    pub inclination: f64,
    pub argument_of_periapsis: f64,
    pub raan: f64,
    pub true_anomaly: f64,
}

impl MissionConfig {
    /// Defaults used for the main optimization binary.
    pub fn optimization_defaults() -> Self {
        MissionConfig {
            initial_altitude: 2000e3,
            target_altitude: 2500e3,
            sail_area: 800.0,
            mass: 10.0,
            reflectivity: 1.6,
            drag_coefficient: 2.2,
            target_time: 1.0 * 7600.0,
            max_slew_rate: 360.0 * (std::f64::consts::PI / 180.0) / 60.0,
            inclination: 98.0f64.to_radians(),
            argument_of_periapsis: 0.0,
            raan: 0.0,
            true_anomaly: 0.0,
        }
    }

    pub fn initial_sma(&self) -> f64 {
        self.initial_altitude + OrbitalElements::EARTH_RADIUS
    }

    pub fn target_sma(&self) -> f64 {
        self.target_altitude + OrbitalElements::EARTH_RADIUS
    }

    pub fn initial_orbit(&self) -> OrbitalElements {
        OrbitalElements {
            a: self.initial_sma(),
            e: 0.0,
            i: self.inclination,
            w: self.argument_of_periapsis,
            o: self.raan,
            nu: self.true_anomaly,
        }
    }
}
