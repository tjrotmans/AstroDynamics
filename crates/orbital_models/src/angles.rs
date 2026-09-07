//! Sail control angles for the sun-pointing (cone-clock) parameterization.
//!
//! Used by any problem where sail orientation is defined relative to the Sun direction:
//! solar sail orbit raising, interplanetary transfers, and any future sun-pointing problem.
//!
//! # Convention (McInnes)
//! - **Cone** (α): angle between sail normal and the Sun→SC direction. Range: [-π/2, π/2]
//!   - α = 0: sail faces the Sun directly → maximum SRP
//!   - α = ±π/2: edge-on → zero SRP
//! - **Clock** (δ): rotation around the Sun→SC axis. Range: [0, π]

use rand::Rng;
use orbital_math::normalize;

/// Sail control angles in the sun-pointing (cone-clock) frame.
#[derive(Clone, Copy, Debug)]
pub struct Angles {
    /// Cone angle (α): angle between sail normal and Sun direction. Range: [-π/2, π/2]
    pub cone: f64,
    /// Clock angle (δ): rotation around Sun vector. Range: [0, π]
    pub clock: f64,
}

impl Angles {
    /// Generate random angles within valid ranges.
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        Angles {
            cone:  rng.gen_range(-std::f64::consts::FRAC_PI_2..std::f64::consts::FRAC_PI_2),
            clock: rng.gen_range(0.0..std::f64::consts::PI),
        }
    }
}

/// Normalize cone angle to [-π/2, π/2] with wrapping.
pub fn normalize_cone(cone: f64) -> f64 {
    normalize(cone, -std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2)
}

/// Normalize clock angle to [0, π] with wrapping.
pub fn normalize_clock(clock: f64) -> f64 {
    normalize(clock, 0.0, std::f64::consts::PI)
}

/// Shortest-path linear interpolation for `Angles`.
///
/// Interpolates each angle along the shortest path within its valid range,
/// preventing linear sweeps when angles wrap around a boundary.
impl interpolation::Lerp for Angles {
    type Scalar = f64;

    fn lerp(&self, other: &Self, progress: &Self::Scalar) -> Self {
        let mut cone_diff = other.cone - self.cone;
        let cone_range = std::f64::consts::PI;
        if cone_diff > cone_range / 2.0 {
            cone_diff -= cone_range;
        } else if cone_diff < -cone_range / 2.0 {
            cone_diff += cone_range;
        }
        let cone = normalize_cone(self.cone + progress * cone_diff);

        let mut clock_diff = other.clock - self.clock;
        let clock_range = std::f64::consts::PI;
        if clock_diff > clock_range / 2.0 {
            clock_diff -= clock_range;
        } else if clock_diff < -clock_range / 2.0 {
            clock_diff += clock_range;
        }
        let clock = normalize_clock(self.clock + progress * clock_diff);

        Angles { cone, clock }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use interpolation::Lerp;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn lerp_midpoint_same_angles() {
        let a = Angles { cone: 0.3, clock: 0.5 };
        let mid = a.lerp(&a, &0.5);
        assert!((mid.cone - 0.3).abs() < 1e-12);
        assert!((mid.clock - 0.5).abs() < 1e-12);
    }

    #[test]
    fn lerp_shortest_path_cone() {
        let a = Angles { cone: FRAC_PI_2 - 0.1, clock: 0.0 };
        let b = Angles { cone: -FRAC_PI_2 + 0.1, clock: 0.0 };
        let mid = a.lerp(&b, &0.5);
        assert!(mid.cone.abs() > 1.0, "Expected wrap-around lerp, got cone={}", mid.cone);
    }

    #[test]
    fn normalize_cone_wraps() {
        let wrapped = normalize_cone(FRAC_PI_2 + 0.1);
        assert!(wrapped >= -FRAC_PI_2 && wrapped <= FRAC_PI_2);
    }

    #[test]
    fn normalize_clock_wraps() {
        let wrapped = normalize_clock(std::f64::consts::PI + 0.1);
        assert!(wrapped >= 0.0 && wrapped <= std::f64::consts::PI);
    }
}
