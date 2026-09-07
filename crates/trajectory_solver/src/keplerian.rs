//! Keplerian state propagation for bodies not covered by DE440S.
//!
//! Given classical orbital elements at a reference epoch, propagates to any
//! query epoch by solving Kepler's equation. Used for small bodies (asteroids,
//! comets) whose well-characterised heliocentric orbits are available from
//! JPL Horizons or similar catalogues.
//!
//! The output frame is heliocentric ecliptic J2000 — the same axes used by
//! ANISE's `body_state_heliocentric` calls, so Lambert inputs can be mixed
//! without any additional rotation.

use nalgebra::Vector3;
use orbital_math::kepler::{perifocal_to_ecliptic, solve_kepler};
use orbital_math::lambert::V3;

/// Sun's gravitational parameter [m³/s²] — JPL DE430
pub const MU_SUN_M3S2: f64 = 1.327_124_400_18e20;

/// 1 AU in metres — IAU 2012 exact definition
pub const AU_M: f64 = 1.496e11;

/// Classical orbital elements for a heliocentric elliptic orbit.
///
/// Elements are in the heliocentric ecliptic J2000 frame, consistent with
/// the output of JPL Horizons' "Elements" query with `ECLIPTIC` frame.
#[derive(Clone, Debug)]
pub struct KeplerianElements {
    /// Semi-major axis [m]
    pub sma_m: f64,
    /// Eccentricity (0 ≤ e < 1)
    pub eccentricity: f64,
    /// Inclination [rad]
    pub inclination_rad: f64,
    /// Right ascension of ascending node [rad]
    pub raan_rad: f64,
    /// Argument of periapsis [rad]
    pub aop_rad: f64,
    /// Mean anomaly at reference epoch [rad]
    pub mean_anomaly_0_rad: f64,
    /// Reference epoch [Julian Date, TDB]
    pub epoch_jd: f64,
    /// Gravitational parameter of central body [m³/s²]
    pub mu_central: f64,
}

impl KeplerianElements {
    /// Heliocentric ecliptic J2000 position [m] and velocity [m/s] at `query_jd`.
    pub fn state_at_jd(&self, query_jd: f64) -> (V3, V3) {
        let dt_s = (query_jd - self.epoch_jd) * 86_400.0;
        let n    = (self.mu_central / self.sma_m.powi(3)).sqrt(); // mean motion [rad/s]
        let m    = self.mean_anomaly_0_rad + n * dt_s;
        let ea   = solve_kepler(m, self.eccentricity, 1e-12);

        let nu = 2.0 * f64::atan2(
            ((1.0 + self.eccentricity) / (1.0 - self.eccentricity)).sqrt()
                * (ea / 2.0).sin(),
            (ea / 2.0).cos(),
        );

        let p  = self.sma_m * (1.0 - self.eccentricity * self.eccentricity);
        let r  = p / (1.0 + self.eccentricity * nu.cos());
        let h  = (self.mu_central * p).sqrt(); // specific angular momentum

        // Perifocal frame
        let x_pf  =  r * nu.cos();
        let y_pf  =  r * nu.sin();
        let vx_pf = -self.mu_central / h * nu.sin();
        let vy_pf =  self.mu_central / h * (self.eccentricity + nu.cos());

        let pos: Vector3<f64> =
            perifocal_to_ecliptic(x_pf, y_pf, self.inclination_rad, self.raan_rad, self.aop_rad);
        let vel: Vector3<f64> =
            perifocal_to_ecliptic(vx_pf, vy_pf, self.inclination_rad, self.raan_rad, self.aop_rad);

        ([pos.x, pos.y, pos.z], [vel.x, vel.y, vel.z])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Earth's heliocentric distance at J2000 should be ≈ 1 AU.
    #[test]
    fn earth_at_j2000() {
        use std::f64::consts::PI;
        let earth = KeplerianElements {
            sma_m:               1.000_000_11 * AU_M,
            eccentricity:        0.016_708_6,
            inclination_rad:     0.0,                          // reference plane
            raan_rad:            0.0,
            aop_rad:             102.94719_f64.to_radians(),
            mean_anomaly_0_rad:  100.46435_f64.to_radians(),   // J2000
            epoch_jd:            2_451_545.0,                  // J2000.0
            mu_central:          MU_SUN_M3S2,
        };
        let (r, _) = earth.state_at_jd(2_451_545.0);
        let dist_au = (r[0]*r[0] + r[1]*r[1] + r[2]*r[2]).sqrt() / AU_M;
        // At M=100°, Earth is near perihelion-side; distance ≈ 0.98-1.02 AU
        assert!(dist_au > 0.95 && dist_au < 1.05,
            "Earth should be ~1 AU from Sun, got {dist_au:.3} AU");
        let _ = PI; // suppress unused import
    }
}
