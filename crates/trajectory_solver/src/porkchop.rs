//! Porkchop diagram: sweep departure epoch × TOF and evaluate Lambert arcs.
//!
//! The grid itself has no ephemeris dependency. Body states at each grid node
//! are provided by a caller-supplied callback so this struct compiles against
//! stable types only. The [`MissionPlanner`] design command wires in the
//! ANISE/Keplerian state functions.

use orbital_math::lambert::V3;

use crate::lambert_arc::LambertArc;

/// A single evaluated point on the porkchop diagram.
#[derive(Clone, Debug)]
pub struct PorkchopPoint {
    /// Departure epoch offset from the reference epoch [days]
    pub dep_offset_days: f64,
    /// Time of flight [days]
    pub tof_days: f64,
    /// C3 at departure [km²/s²]
    pub c3_km2s2: f64,
    /// Arrival hyperbolic excess speed [m/s]
    pub v_inf_arr_ms: f64,
    /// Departure ΔV [m/s]
    pub dv_dep_ms: f64,
    /// Arrival ΔV [m/s]
    pub dv_arr_ms: f64,
    /// Total mission ΔV [m/s]
    pub dv_total_ms: f64,
}

impl PorkchopPoint {
    pub fn csv_header() -> &'static str {
        "dep_offset_days,tof_days,c3_km2s2,v_inf_arr_ms,dv_dep_ms,dv_arr_ms,dv_total_ms"
    }

    pub fn to_csv_row(&self) -> String {
        format!(
            "{:.4},{:.4},{:.6},{:.2},{:.2},{:.2},{:.2}",
            self.dep_offset_days, self.tof_days,
            self.c3_km2s2, self.v_inf_arr_ms,
            self.dv_dep_ms, self.dv_arr_ms, self.dv_total_ms,
        )
    }
}

/// Evaluates Lambert arcs over a departure-offset × TOF grid.
///
/// `dep_offsets_days` — departure epoch offsets to sweep [days from reference].
/// `tof_days_grid`    — TOF values to sweep [days].
///
/// Body states are provided by the callback `body_state`:
/// ```text
/// body_state(dep_offset_days, tof_days) -> Option<(r_dep, v_dep, r_arr, v_arr)>
/// ```
/// All positions in metres, velocities in m/s, heliocentric. Returning `None`
/// skips that grid point silently (e.g. body not visible, ephemeris gap).
pub struct PorkchopGrid {
    /// Central body gravitational parameter [m³/s²]
    pub mu: f64,
    /// Departure epoch offsets to sweep [days from reference epoch]
    pub dep_offsets_days: Vec<f64>,
    /// TOF values to sweep [days]
    pub tof_days_grid: Vec<f64>,
}

impl PorkchopGrid {
    pub fn evaluate<F>(&self, mut body_state: F) -> Vec<PorkchopPoint>
    where
        F: FnMut(f64, f64) -> Option<(V3, V3, V3, V3)>,
    {
        let mut points =
            Vec::with_capacity(self.dep_offsets_days.len() * self.tof_days_grid.len());

        for &dep_off in &self.dep_offsets_days {
            for &tof_d in &self.tof_days_grid {
                let Some((r_dep, v_dep, r_arr, v_arr)) = body_state(dep_off, tof_d) else {
                    continue;
                };

                let arc = LambertArc {
                    r_dep,
                    v_dep,
                    r_arr,
                    v_arr,
                    tof_s: tof_d * 86_400.0,
                    mu: self.mu,
                };

                if let Ok(sol) = arc.solve() {
                    points.push(PorkchopPoint {
                        dep_offset_days: dep_off,
                        tof_days:        tof_d,
                        c3_km2s2:        sol.c3_km2s2,
                        v_inf_arr_ms:    sol.v_inf_arr_ms,
                        dv_dep_ms:       sol.dv_departure_ms,
                        dv_arr_ms:       sol.dv_arrival_ms,
                        dv_total_ms:     sol.dv_total_ms,
                    });
                }
            }
        }

        points
    }

    /// Best (minimum total ΔV) point from the evaluated grid.
    pub fn best(points: &[PorkchopPoint]) -> Option<&PorkchopPoint> {
        points
            .iter()
            .min_by(|a, b| a.dv_total_ms.partial_cmp(&b.dv_total_ms).unwrap())
    }
}
