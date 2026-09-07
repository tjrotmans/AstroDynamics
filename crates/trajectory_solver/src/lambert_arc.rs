//! Single Lambert arc: position-fixed, time-of-flight-fixed two-body transfer.
//!
//! Wraps `orbital_math::lambert::lambert_min_dv` and computes the ΔV quantities
//! relative to the body velocities at departure and arrival.

use orbital_math::lambert::{lambert_min_dv, V3};

use crate::solution::{SolverError, TrajectorySolution};

#[inline]
fn norm(v: V3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Single Lambert arc between two body positions at a fixed time-of-flight.
///
/// Call [`LambertArc::solve`] to get a [`TrajectorySolution`], or
/// [`SolverError::NoSolution`] if no elliptic transfer exists for this
/// geometry and TOF.
pub struct LambertArc {
    /// Departure body position [m]
    pub r_dep: V3,
    /// Departure body velocity [m/s] — used only to compute ΔV
    pub v_dep: V3,
    /// Arrival body position [m]
    pub r_arr: V3,
    /// Arrival body velocity [m/s] — used only to compute ΔV
    pub v_arr: V3,
    /// Time of flight [s]
    pub tof_s: f64,
    /// Central body gravitational parameter [m³/s²]
    pub mu: f64,
}

impl LambertArc {
    pub fn solve(&self) -> Result<TrajectorySolution, SolverError> {
        if self.tof_s <= 0.0 {
            return Err(SolverError::InvalidInput("tof_s must be > 0".into()));
        }

        let Some((vt_dep, vt_arr)) = lambert_min_dv(
            self.r_dep, self.r_arr, self.tof_s, self.mu,
            self.v_dep, self.v_arr,
        ) else {
            return Err(SolverError::NoSolution);
        };

        let dv_dep    = norm(sub(vt_dep, self.v_dep));
        let dv_arr    = norm(sub(vt_arr, self.v_arr));
        let c3_km2s2  = (dv_dep / 1_000.0).powi(2);
        let v_inf_arr = norm(sub(vt_arr, self.v_arr));

        Ok(TrajectorySolution {
            tof_s:           self.tof_s,
            v_transfer_dep:  vt_dep,
            v_transfer_arr:  vt_arr,
            dv_departure_ms: dv_dep,
            dv_arrival_ms:   dv_arr,
            dv_total_ms:     dv_dep + dv_arr,
            c3_km2s2,
            v_inf_arr_ms:    v_inf_arr,
        })
    }
}
