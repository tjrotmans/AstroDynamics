//! Pre-sampled body position tracks for fast interpolation inside ODE solvers.
//!
//! Querying an ephemeris at every ODE step is expensive. [`BodyTrack`] pre-samples
//! body positions at a fixed interval and linearly interpolates during integration,
//! giving accurate results at a fraction of the cost.

use nalgebra::SVector;

/// Pre-sampled body ECI positions with linear interpolation.
///
/// Sample the body at a regular interval before integration, then call
/// [`BodyTrack::position_at`] inside the ODE to get the position at any time.
pub struct BodyTrack {
    /// Pre-sampled positions at uniform time intervals.
    pub positions: Vec<SVector<f64, 3>>,
    /// Time between consecutive samples [s].
    pub sample_dt_s: f64,
}

impl BodyTrack {
    /// Interpolated position at time `t` [m].
    ///
    /// Uses linear interpolation between the two nearest samples.
    /// Clamps to the last interval if `t` exceeds the sampled range.
    pub fn position_at(&self, t: f64) -> SVector<f64, 3> {
        let idx  = (t / self.sample_dt_s).floor() as usize;
        let idx  = idx.min(self.positions.len() - 2);
        let frac = t / self.sample_dt_s - idx as f64;
        self.positions[idx] + frac * (self.positions[idx + 1] - self.positions[idx])
    }
}

/// Pre-sampled Moon track. Type alias of [`BodyTrack`] for call-site clarity.
pub type MoonTrack = BodyTrack;
/// Pre-sampled Sun track. Type alias of [`BodyTrack`] for call-site clarity.
pub type SunTrack = BodyTrack;
