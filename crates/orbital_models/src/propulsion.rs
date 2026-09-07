//! Propulsion models for finite-burn maneuvers.
//!
//! Provides [`FiniteBurn`] for constant-thrust engine firings with variable mass
//! (Tsiolkovsky rocket equation). Suitable for chemical propulsion stages where the
//! burn direction is fixed in the inertial frame at ignition.

use nalgebra::SVector;
use crate::constants::G0;

/// Constant-thrust engine firing over a time window with variable mass.
///
/// The thrust direction is fixed in the inertial frame at ignition — a good
/// approximation for short burns (< ~10 min) where attitude changes are small.
/// Mass decreases at a constant rate determined by thrust and Isp.
#[derive(Clone, Copy, Debug)]
pub struct FiniteBurn {
    /// Thrust magnitude [N]
    pub thrust_n: f64,
    /// Specific impulse [s]
    pub isp_s: f64,
    /// Burn ignition time [s from simulation start]
    pub ignition_time_s: f64,
    /// Burn cutoff time [s from simulation start]
    pub cutoff_time_s: f64,
    /// Thrust direction unit vector in the inertial frame — fixed at ignition
    pub direction: SVector<f64, 3>,
}

impl FiniteBurn {
    /// Construct a burn from a target ΔV and spacecraft initial mass.
    ///
    /// Burn duration is derived from Tsiolkovsky: `Δm = m0 * (1 − exp(−ΔV / (Isp·g0)))`,
    /// then `t_burn = Δm / ṁ` where `ṁ = F / (Isp·g0)`.
    ///
    /// # Arguments
    /// * `thrust_n`       – Engine thrust [N]
    /// * `isp_s`          – Specific impulse [s]
    /// * `m0_kg`          – Spacecraft mass at ignition [kg]
    /// * `delta_v_ms`     – Target ΔV magnitude [m/s]
    /// * `ignition_time_s` – Simulation time at ignition [s]
    /// * `direction`      – Unit thrust vector in the inertial frame
    pub fn from_delta_v(
        thrust_n: f64,
        isp_s: f64,
        m0_kg: f64,
        delta_v_ms: f64,
        ignition_time_s: f64,
        direction: SVector<f64, 3>,
    ) -> Self {
        let exhaust_vel = isp_s * G0;
        let mass_ratio  = (-delta_v_ms / exhaust_vel).exp();
        let prop_mass   = m0_kg * (1.0 - mass_ratio);
        let mass_flow   = thrust_n / exhaust_vel;
        let duration    = prop_mass / mass_flow;

        Self {
            thrust_n,
            isp_s,
            ignition_time_s,
            cutoff_time_s: ignition_time_s + duration,
            direction,
        }
    }

    /// Mass flow rate [kg/s] (negative — mass decreases). Zero outside burn window.
    pub fn mass_rate(&self, time_s: f64) -> f64 {
        if time_s >= self.ignition_time_s && time_s < self.cutoff_time_s {
            -(self.thrust_n / (self.isp_s * G0))
        } else {
            0.0
        }
    }

    /// Thrust acceleration [m/s²] in the inertial frame. Zero outside burn window.
    pub fn thrust_acceleration(&self, time_s: f64, mass_kg: f64) -> SVector<f64, 3> {
        if time_s >= self.ignition_time_s && time_s < self.cutoff_time_s && mass_kg > 0.0 {
            (self.thrust_n / mass_kg) * self.direction
        } else {
            SVector::zeros()
        }
    }

    /// Propellant mass consumed [kg].
    pub fn propellant_mass_kg(&self, m0_kg: f64) -> f64 {
        let exhaust_vel = self.isp_s * G0;
        let duration    = self.cutoff_time_s - self.ignition_time_s;
        let mass_flow   = self.thrust_n / exhaust_vel;
        (mass_flow * duration).min(m0_kg)
    }

    /// Burn duration [s].
    pub fn duration_s(&self) -> f64 {
        self.cutoff_time_s - self.ignition_time_s
    }
}
