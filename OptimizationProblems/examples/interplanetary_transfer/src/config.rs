//! Mission configuration for the Earth-Mars interplanetary transfer

use ephemeris::Epoch;

/// Mission and spacecraft parameters for the solar sail Earth-Mars transfer.
pub struct MissionConfig {
    /// Sail area [m²]
    pub sail_area: f64,
    /// Spacecraft total mass [kg]
    pub mass: f64,
    /// Sail reflectivity coefficient (0 = perfect absorber, 1 = perfect reflector)
    pub reflectivity: f64,
    /// Departure epoch (spacecraft leaves Earth vicinity at this time)
    pub departure_epoch: Epoch,
    /// Minimum time-of-flight [days]
    pub min_tof_days: f64,
    /// Maximum time-of-flight [days]
    pub max_tof_days: f64,
    /// Weight of velocity error in the rendezvous energy function.
    ///
    /// Energy = pos_err_normalized + vel_weight * vel_err_normalized + tof_weight * tof_normalized.
    /// All terms are normalized to roughly [0, 1] scale.
    pub vel_weight: f64,
    /// Weight of time-of-flight in the energy function.
    ///
    /// `tof_normalized = (tof_days - min_tof_days) / (max_tof_days - min_tof_days)` ∈ [0, 1].
    /// Set to 0.0 to optimize purely for rendezvous accuracy regardless of travel time.
    pub tof_weight: f64,
    /// Half-window for departure epoch search [days].
    ///
    /// The optimizer may shift the departure by ±`departure_window_days/2` days from
    /// `departure_epoch`. Set to 0.0 to fix the departure epoch.
    pub departure_window_days: f64,
}

impl MissionConfig {
    /// Default parameters suitable for a large interplanetary solar sail.
    pub fn optimization_defaults() -> Self {
        Self {
            sail_area: 350.0,
            mass: 10.0,
            reflectivity: 0.9,
            // UTC departure
            departure_epoch: Epoch::from_gregorian_utc(2026, 1, 1, 0, 0, 0, 0),
            min_tof_days: 150.0,
            max_tof_days: 1000.0,
            // Velocity matching is critical for rendezvous — higher weight drives optimizer
            // toward vel_err < 1 km/s (the threshold set by VEL_PENALTY_NORM in problem.rs)
            vel_weight: 2.0,
            // Penalize longer flights: at full weight, a 700-day trip costs ~1 extra energy unit
            tof_weight: 0.2,
            // Fixed departure epoch by default — set to e.g. 180.0 to search ±90 days
            departure_window_days: 0.0,
        }
    }
}
