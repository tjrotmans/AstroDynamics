//! Control input types and utilities
//! 
//! This module defines control input representations (angles, control points)
//! and related mathematical utilities for interpolation and normalization.

use rand::Rng;

/// Sail control angles in the LVLH (Local Vertical Local Horizontal) / RSW frame
///
/// These angles define the orientation of the solar sail relative to the spacecraft's
/// local horizon. They can be extended in the future to support sun-relative angles
/// (cone angle α and clock angle δ).
#[derive(Clone, Copy, Debug)]
pub struct Angles {
    // Future extensions:
    // /// Cone angle with the sun
    // pub alpha: f64,
    // /// Clock angle with the sun
    // pub delta: f64,

    /// Elevation angle: Nadir = 0, Zenith = π
    /// Valid range: [-π/2, π/2]
    pub elevation: f64,

    /// Direction angle: Straight to velocity = 0, full left = π/2, full right = -π/2
    /// Valid range: [-π, π]
    pub direction: f64,
}

impl Angles {
    /// Generate random angles within valid ranges
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        Angles {
            elevation: rng.gen_range(-std::f64::consts::PI / 2.0..std::f64::consts::PI / 2.0),
            direction: rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
        }
    }
}

pub use orbital_math::normalize;

/// Normalize elevation angle to [-π/2, π/2]
pub fn normalize_elevation(elev: f64) -> f64 {
    normalize(
        elev,
        -std::f64::consts::PI / 2.0,
        std::f64::consts::PI / 2.0,
    )
}

/// Normalize direction angle to [-π, π]
pub fn normalize_direction(dir: f64) -> f64 {
    normalize(dir, -std::f64::consts::PI, std::f64::consts::PI)
}

/// Linear interpolation for Angles, accounting for cyclic nature
impl interpolation::Lerp for Angles {
    type Scalar = f64;
    
    fn lerp(&self, other: &Self, progress: &Self::Scalar) -> Self {
        // For cyclic angles, interpolate along the shortest path
        // This prevents linear sweeps when angles wrap around the boundary
        
        // Elevation: compute shortest angular distance
        let mut elev_diff = other.elevation - self.elevation;
        let elev_range = std::f64::consts::PI; // range is [-π/2, π/2], so span is π
        if elev_diff > elev_range / 2.0 {
            elev_diff -= elev_range;
        } else if elev_diff < -elev_range / 2.0 {
            elev_diff += elev_range;
        }
        let elevation = normalize_elevation(self.elevation + progress * elev_diff);
        
        // Direction: compute shortest angular distance
        let mut dir_diff = other.direction - self.direction;
        let dir_range = 2.0 * std::f64::consts::PI; // range is [-π, π], so span is 2π
        if dir_diff > dir_range / 2.0 {
            dir_diff -= dir_range;
        } else if dir_diff < -dir_range / 2.0 {
            dir_diff += dir_range;
        }
        let direction = normalize_direction(self.direction + progress * dir_diff);
        
        Angles {
            elevation,
            direction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};
    use interpolation::Lerp;

    #[test]
    fn lerp_shortest_path_elevation() {
        // Elevation wraps at [-π/2, π/2] (range π)
        let a = Angles { elevation: FRAC_PI_2 - 0.1, direction: 0.0 };
        let b = Angles { elevation: -FRAC_PI_2 + 0.1, direction: 0.0 };
        let mid = a.lerp(&b, &0.5);
        // Shortest path goes through ±π/2 boundary, not through 0
        assert!(mid.elevation.abs() > 1.0, "Expected wrap-around lerp, got elevation={}", mid.elevation);
    }

    #[test]
    fn lerp_shortest_path_direction() {
        // Direction wraps at [-π, π] (range 2π)
        let a = Angles { elevation: 0.0, direction: PI - 0.1 };
        let b = Angles { elevation: 0.0, direction: -PI + 0.1 };
        let mid = a.lerp(&b, &0.5);
        // Shortest path goes through ±π boundary, not through 0
        assert!(mid.direction.abs() > 2.5, "Expected wrap-around lerp, got direction={}", mid.direction);
    }
}

