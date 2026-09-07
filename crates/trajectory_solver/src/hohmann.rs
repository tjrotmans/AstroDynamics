//! Analytical Hohmann transfer between two coplanar circular orbits.
//!
//! Used for fast ΔV sizing during preliminary mission design. The result
//! gives a lower bound on the true ΔV for any transfer between the same
//! two orbital radii (real transfers are generally more expensive due to
//! inclination, phasing, and departure/arrival hyperbolic losses).

use crate::solution::{SolverError, TrajectorySolution};

/// Analytical Hohmann transfer solver.
///
/// Both orbits are assumed circular and coplanar. The two impulsive burns
/// are purely tangential (prograde at departure, retrograde at arrival for
/// inner-to-outer, and the reverse for outer-to-inner).
pub struct HohmannSolver {
    /// Gravitational parameter of central body [m³/s²]
    pub mu_m3s2: f64,
    /// Departure orbit radius [m]
    pub r1_m: f64,
    /// Arrival orbit radius [m]
    pub r2_m: f64,
}

impl HohmannSolver {
    pub fn solve(&self) -> Result<TrajectorySolution, SolverError> {
        let mu = self.mu_m3s2;
        let r1 = self.r1_m;
        let r2 = self.r2_m;

        if r1 <= 0.0 || r2 <= 0.0 || mu <= 0.0 {
            return Err(SolverError::InvalidInput(
                "mu, r1, r2 must all be > 0".into(),
            ));
        }

        let a_t = 0.5 * (r1 + r2);                          // transfer semi-major axis
        let tof_s = std::f64::consts::PI * (a_t.powi(3) / mu).sqrt(); // half-period

        // Velocities on the transfer ellipse (vis-viva)
        let vt1 = (mu * (2.0 / r1 - 1.0 / a_t)).sqrt();    // at periapsis (dep)
        let vt2 = (mu * (2.0 / r2 - 1.0 / a_t)).sqrt();    // at apoapsis  (arr)

        // Circular orbit velocities
        let vc1 = (mu / r1).sqrt();
        let vc2 = (mu / r2).sqrt();

        let dv_dep = (vt1 - vc1).abs();
        let dv_arr = (vc2 - vt2).abs();

        // Transfer velocities expressed in the tangential direction only.
        // The caller must rotate into the actual frame; these scalar magnitudes
        // are what matters for ΔV budgeting.
        let v_inf_dep = (vt1 - vc1).abs();
        let c3_km2s2  = (v_inf_dep / 1_000.0).powi(2);

        Ok(TrajectorySolution {
            tof_s,
            v_transfer_dep:  [vt1, 0.0, 0.0], // tangential frame placeholder
            v_transfer_arr:  [vt2, 0.0, 0.0],
            dv_departure_ms: dv_dep,
            dv_arrival_ms:   dv_arr,
            dv_total_ms:     dv_dep + dv_arr,
            c3_km2s2,
            v_inf_arr_ms:    (vc2 - vt2).abs(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Earth LEO (200 km) → GEO (35 786 km). Textbook result: ΔV ≈ 3.9 km/s total.
    #[test]
    fn leo_to_geo() {
        const MU: f64 = 3.986_004_418e14;
        const R_EARTH: f64 = 6_371_000.0;
        let solver = HohmannSolver {
            mu_m3s2: MU,
            r1_m: R_EARTH + 200_000.0,
            r2_m: R_EARTH + 35_786_000.0,
        };
        let sol = solver.solve().unwrap();
        let dv_kms = sol.dv_total_ms / 1_000.0;
        assert!(
            (dv_kms - 3.9).abs() < 0.1,
            "LEO→GEO ΔV should be ~3.9 km/s, got {dv_kms:.3}"
        );
    }

    /// Earth (1 AU) → Mars (~1.52 AU). Textbook: ~2.9 km/s departure.
    #[test]
    fn earth_to_mars_hohmann() {
        const MU_SUN: f64 = 1.327_124_400_18e20;
        const AU: f64 = 1.496e11;
        let solver = HohmannSolver {
            mu_m3s2: MU_SUN,
            r1_m: 1.0 * AU,
            r2_m: 1.524 * AU,
        };
        let sol = solver.solve().unwrap();
        let dv_dep_kms = sol.dv_departure_ms / 1_000.0;
        let tof_days   = sol.tof_s / 86_400.0;
        // Earth→Mars Hohmann: dep ΔV ≈ 2.94 km/s, TOF ≈ 259 days
        assert!((dv_dep_kms - 2.94).abs() < 0.05, "dep ΔV: {dv_dep_kms:.3}");
        assert!((tof_days - 259.0).abs() < 2.0,   "TOF: {tof_days:.1} days");
    }
}
