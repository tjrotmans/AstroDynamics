//! Generic trajectory solver abstraction for multi-body mission design.
//!
//! Provides solver primitives that work with `[f64; 3]` position/velocity arrays
//! (SI units, metres and m/s). Ephemeris queries stay in the caller — this crate
//! is pure orbital math with no ANISE dependency.
//!
//! # Solvers
//! - [`HohmannSolver`]         — analytical two-impulse circular transfer (sizing)
//! - [`LambertArc`]            — single Lambert arc at a fixed TOF
//! - [`PorkchopGrid`]          — sweep departure × TOF grid of Lambert arcs
//! - [`keplerian`]             — Keplerian state propagation for bodies without ANISE coverage
//! - [`DiffCorrectionSolver`]  — damped multi-start Newton targeting
//! - [`MonteCarloSolver`]      — Gaussian scatter around a reference solution
//! - [`GaSolver`]              — real-valued genetic algorithm, bounded global search (Phase 8j)
//! - [`incremental_pruning`]   — Ceriotti (2010) Ch. 3 leg-partitioned pruning search, precedes DE (Phase 9x)
//! - [`DeSolver`]              — Differential Evolution DE/rand/1/bin (Phase 9d-ii; preferred for MGA-DSM)
//! - [`ShadeSolver`]           — SHADE self-adaptive DE, success-history F/CR + current-to-pbest/1 (Phase 9v)
//! - [`PsoSolver`]             — particle swarm optimization, bounded global search (Phase 8j follow-up)
//! - [`mga_leg`]               — MGA-1DSM per-leg evaluator and flyby turn geometry (Phase 9d-iii)
//! - [`mga_scan`]              — ballistic powered-flyby MGA launch-window grid scan, STOUR-class (Phase 9w-i)
//! - [`tisserand`]             — Tisserand parameter, contours, and body-to-body feasibility links (Phase 9j)
//! - [`propagator::propagate`] — SOI-patched multi-body numerical propagation (Phase 7, Layer 1's real ephemeris)
//! - [`sims_flanagan`]         — Sims-Flanagan direct transcription for low-thrust optimization (Phase 9h)
//! - [`primer_vector`] — Lawden/Olympio primer vector optimality diagnostic for MGA-DSM legs (Stage 1)
//! - [`nelder_mead`] — bounds-normalised Nelder-Mead simplex local optimizer (Stage 2; MBH's descent step)
//! - [`compass_search`] — bounds-normalised compass/pattern search local optimizer (best-of-all-directions variant, selectable alternative to Nelder-Mead)
//! - [`hooke_jeeves`] — bounds-normalised greedy accept-first-improving pattern search (matches real MBH reference implementations' shallow per-hop descent, eval-budget termination)
//! - [`mbh`] — Monotonic Basin Hopping global optimizer (Stage 3; alternative to DE/SHADE for MGA)
//! - [`vilm`] — V∞ leveraging transfer (VILT) boundary-value solver, tangent case (Stage 5)
//! - [`launch_geometry`] — closed-form launch geometry from a departure asymptote: RLA/DLA, site-feasible plane, azimuth, injection state (Phase 14c)

pub mod compass_search;
pub mod de;
pub mod departure;
pub mod diff_correction;
pub mod ga;
pub mod hohmann;
pub mod hooke_jeeves;
pub mod incremental_pruning;
pub mod keplerian;
pub mod lambert_arc;
pub mod launch_geometry;
pub mod lbfgs;
pub mod mbh;
pub mod mga_leg;
pub mod mga_scan;
pub mod monte_carlo;
pub mod nelder_mead;
pub mod porkchop;
pub mod primer_vector;
pub mod propagator;
pub mod pso;
pub mod shade;
pub mod sims_flanagan;
pub mod solution;
pub mod surrogate;
pub mod tisserand;
pub mod vilm;

pub use compass_search::CompassSearch;
pub use de::{DeResult, DeSolver};
pub use departure::{circular_orbit_burn_state, hyperbolic_departure_state, CircularOrbitBurn, HyperbolicDeparture};
pub use diff_correction::{DiffCorrectionResult, DiffCorrectionSolver};
pub use ga::{GaResult, GaSolver};
pub use hohmann::HohmannSolver;
pub use hooke_jeeves::HookeJeevesSearch;
pub use incremental_pruning::{
    run_incremental_pruning, run_incremental_pruning_all_levels,
    run_incremental_pruning_with_sampler, sample_box_uniform,
    PruningCandidate, PruningConfig,
};
pub use lambert_arc::LambertArc;
pub use launch_geometry::{launch_geometry, EquatorialFrame, LaunchGeometry};
pub use lbfgs::{LbfgsResult, LbfgsSolver};
pub use mbh::{GlobalStallConfig, LocalOptimizer, MbhResult, MbhSolver, MigrationConfig};
pub use mga_leg::{
    evaluate_mga_leg, evaluate_mga_leg_2dsm, evaluate_mga_leg_n, evaluate_vilm_leg, flyby_turn,
    refine_leg_two_dsm, MgaLegResult, TwoDsmLegResult, TwoDsmRefineResult, VilmLegResult,
};
pub use mga_scan::{
    max_turn_angle_rad, powered_flyby, run_mga_scan, MgaScanConfig, MgaScanOutput, PoweredFlyby,
    ScanBody, ScanRecord,
};
pub use monte_carlo::{MonteCarloSample, MonteCarloSolver};
pub use nelder_mead::{NelderMead, NmResult};
pub use primer_vector::{
    costate_stm, gravity_gradient, sample_primer_magnitude, solve_leg_primer,
    LegBoundary, LegPrimerResult, SubArc,
};
pub use orbital_math::lambert::{
    lambert, lambert_min_dv_at_n_rev, lambert_min_dv_multi_rev, lambert_n_rev,
    lambert_with_min_transfer_angle, transfer_angle_rad,
};
pub use orbital_math::{eccentricity, orbital_period_s, propagate_kepler, semi_major_axis_m};
pub use porkchop::{PorkchopGrid, PorkchopPoint};
pub use propagator::{
    find_inbound_radius_crossing, laplace_soi_radius_m, propagate, propagate_departure_and_cruise,
    propagate_escape_leg, resolve_central_body, resolve_collision, DepartureLegResult, EscapeLegResult,
    PropagatedPoint, PropagatorBody, RadiusCrossing, ZonalFidelity,
};
pub use pso::{PsoResult, PsoSolver};
pub use shade::ShadeSolver;
pub use sims_flanagan::{ArcHalf, SFPoint, SFResult, SimsFlanagan};
pub use solution::{SolverError, TrajectorySolution};
pub use monte_carlo::SplitMix64;
pub use surrogate::RbfSurrogate;
pub use tisserand::{find_tisserand_link, tisserand_feasibility_score, TisserandContour};
pub use vilm::{solve_tangent_vilt, VilmDomain, VilmResult, VilmSolution};
