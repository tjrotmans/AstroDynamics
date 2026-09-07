//! Generic config-driven high-fidelity simulation engine — Phase 4 of the
//! Mission Planner roadmap (see `the design notes`).
//!
//! Depends only on the confirmed-generic crates (`orbital_models`,
//! `attitude_control`, `ephemeris`, `body_models`, `hardware_catalog`,
//! `trajectory_solver`). `GNC/AutonomousNavigation` is a read-only reference
//! pattern for this crate's structure — never a dependency, never edited.

pub mod actuators;
pub mod attitude_commander;
pub mod attitude_ekf;
pub mod control;
pub mod ekf;
pub mod engine;
pub mod guidance;
pub mod phase;
pub mod propagator6dof;
pub mod reference_guidance;
pub mod sensors;
pub mod truth;

pub use ekf::{EkfConfig, EkfState};
pub use engine::{SimEngine, StepTelemetry};
pub use guidance::PointingMode;
pub use phase::{FlybyPhase, HohmannTransferPhase, LandingPhase, OrbitInsertionPhase, OrbitPhase, Phase};
/// The single 6DOF propagator (Phase 13c) — see `propagator6dof`'s module
/// doc comment and `docs/MP/MANUAL.md` §7 for the governing design.
pub use propagator6dof::{
    boresight, disturbance_torque_breakdown, step_tick, step_tick_with_burn,
    translational_accel_breakdown, AccelBreakdown, BurnConfig, SixDofState, TorqueBreakdown,
};
/// Cascaded attitude control + allocation (Phase 13e) — see
/// `docs/MP/MANUAL.md` §10.
pub use control::{allocate, net_body_torque, AllocationOutput, AttitudeControlLaw, ControlMode, MomentumManagementLaw};
pub use control::{
    rcs_axis_authority_nm, rcs_worst_axis_authority_nm, small_angle_error_vec, tuned_law, Activity, ActivityTuning,
    AttitudeController, AttitudeLaw, PhasePlaneParams,
};
/// Reference-trajectory guidance + pointing-mode library (Phase 13d) — see
/// `docs/MP/MANUAL.md` §9.
pub use reference_guidance::{
    desired_quaternion_cruise, dispersion, tcm_lambert_correction, CruisePointingMode, Dispersion,
    ReferencePoint, ReferenceTrajectory,
};
pub use truth::{Environment, SpacecraftProperties, SrpTruthModel, TruthState};
/// Priority-ordered multi-rule attitude commander — see
/// `docs/MP/MANUAL.md` §9.4.
pub use attitude_commander::{solve_prioritized_attitude, ResolvedRule, RuleOutcome};
/// Attitude MEKF (Phase 13h) — see `docs/MP/MANUAL.md` §12.3.
pub use attitude_ekf::AttitudeEkfState;

/// Re-exported so callers (e.g. `MissionPlanner`) can construct `SimEngine`
/// without adding `attitude_control` as a separate direct dependency.
pub use attitude_control::{thruster_selection, PdGains, ReactionWheelCluster, Thruster};
/// Re-exported `Plate` for callers building a `SrpTruthModel::FlatPlate`.
pub use orbital_models::{flat_plate_force_per_plate, Plate};
