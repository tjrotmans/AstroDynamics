//! Result type returned by all trajectory solvers.

/// The result of a trajectory solver run.
///
/// All quantities are in SI units (m, m/s) unless otherwise noted.
#[derive(Clone, Debug)]
pub struct TrajectorySolution {
    /// Time of flight [s]
    pub tof_s: f64,
    /// Transfer orbit velocity at departure [m/s]
    pub v_transfer_dep: [f64; 3],
    /// Transfer orbit velocity at arrival [m/s]
    pub v_transfer_arr: [f64; 3],
    /// Departure ΔV = |v_transfer_dep − v_body_dep| [m/s]
    pub dv_departure_ms: f64,
    /// Arrival ΔV = |v_transfer_arr − v_body_arr| [m/s]
    pub dv_arrival_ms: f64,
    /// Total mission ΔV [m/s]
    pub dv_total_ms: f64,
    /// Departure hyperbolic excess speed squared [km²/s²]
    pub c3_km2s2: f64,
    /// Arrival hyperbolic excess speed [m/s]
    pub v_inf_arr_ms: f64,
}

/// Error type for trajectory solver failures.
#[derive(Debug)]
pub enum SolverError {
    /// Lambert solver found no real solution for this geometry / TOF.
    NoSolution,
    /// Input parameters are physically invalid.
    InvalidInput(String),
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSolution => write!(f, "no Lambert solution for this geometry/TOF"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for SolverError {}
