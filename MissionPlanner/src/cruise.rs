//! Phase 5.1) —
//! the core cruise mission-loop: a real composition of the Phase 13
//! translation+attitude propagator (`sim_engine::step_tick`), the Phase 13d
//! guidance library (`reference_guidance`), and the Phase 13e cascaded
//! controller (`sim_engine::control`) into an actual tick-by-tick loop that
//! flies a vehicle against a Layer-1 reference trajectory.
//!
//! Originally the SIMPLEST possible closed loop (one fixed attitude-hold
//! pointing mode for the whole leg) — extended
//! with [`GncCommander`], an optional priority-ordered multi-rule
//! attitude commander (see `docs/MP/MANUAL.md` §9.4 for the physics:
//! why only 2 rules per mode can ever control the attitude, and why lower-
//! priority rules are inherently "report what broke," not a bug). When no
//! commander is supplied, `run_cruise_leg` behaves exactly as before —
//! purely additive. Still no trajectory-correction burns (translation is
//! pure coast under the configured force model — TCM execution via
//! `reference_guidance::tcm_lambert_correction` is a deliberate fast-follow
//! once this coast-only loop is proven).
//!
//! `MissionPlanner/src/bin/cruise_demo.rs` exercises this against a real
//! ANISE Earth→Mars interplanetary leg; `plot/plot_cruise_demo.py` plots it.
//! See `docs/MP/MANUAL.md` §9/§10 for the governing guidance/control math
//! this loop composes.

use std::collections::HashMap;

use nalgebra::{Vector3, Vector4};

use orbital_models::attitude::{body_to_inertial, inertial_to_body};
use sim_engine::reference_guidance::{
    desired_quaternion_cruise, dispersion, tcm_lambert_correction, ReferencePoint, ReferenceTrajectory,
};
use sim_engine::{
    allocate, disturbance_torque_breakdown, net_body_torque, solve_prioritized_attitude, step_tick_with_burn,
    translational_accel_breakdown, Activity, AttitudeController, BurnConfig, ControlMode, CruisePointingMode,
    MomentumManagementLaw, ReactionWheelCluster, ResolvedRule, SixDofState,
    SpacecraftProperties, SrpTruthModel, Thruster,
};
use trajectory_solver::{orbital_period_s, propagate, semi_major_axis_m, PropagatorBody};

use crate::attitude_tuning::{AttitudeControlEffective, AttitudeControlSet, GainSchedulePoint};
use crate::config::{hardware_pointing_vector, CruiseSeedConfig, GncModeConfig, HardwareItem, MissionConfig, PointingTargetConfig};
use crate::simulate::{build_spacecraft_properties, rcs_from_hardware, wheel_cluster_from_hardware};

/// Build a [`ReferenceTrajectory`] sampled on a UNIFORM grid, `dt_s` apart.
///
/// `trajectory_solver::propagate`'s `OutputType::Sparse` integration mode
/// (found the hard way building `cruise_demo`) emits one point per accepted
/// ADAPTIVE step, not one per `sample_dt_s` — that parameter is only the
/// initial step-size hint. For a smooth coast the adaptive controller can
/// take very few, unevenly-spaced steps that are each individually accurate
/// but leave [`ReferenceTrajectory::state_at`]'s linear interpolation a poor
/// approximation of the true curved path between them (confirmed: a 6 h
/// natural step spacing produced a spurious ~10,800 km interpolation
/// artifact on a real Earth-departure leg). This function does NOT change
/// `propagate`'s behavior (a shared, heavily-verified core other callers
/// depend on) — it calls it repeatedly over short, fixed sub-intervals and
/// keeps only the state at each boundary, which is exactly as accurate
/// (still the same real adaptive Dopri5 physics per sub-call) but
/// guaranteed evenly spaced.
pub fn sample_reference_trajectory_uniform(
    r0: Vector3<f64>,
    v0: Vector3<f64>,
    mu_central_m3s2: f64,
    bodies: &[PropagatorBody],
    duration_s: f64,
    dt_s: f64,
    rtol: f64,
    atol: f64,
) -> ReferenceTrajectory {
    let mut points = vec![ReferencePoint { t_s: 0.0, r_m: r0, v_mps: v0 }];
    let (mut r, mut v, mut t) = (r0, v0, 0.0_f64);
    while t < duration_s - 1e-9 {
        let step = dt_s.min(duration_s - t);
        let leg = propagate(r, v, t, step, mu_central_m3s2, bodies, step, rtol, atol);
        let last = leg.last().expect("propagate always returns at least one point");
        r = last.r_m;
        v = last.v_mps;
        t += step;
        points.push(ReferencePoint { t_s: t, r_m: r, v_mps: v });
    }
    ReferenceTrajectory::new(points)
}

/// One tick's worth of cruise-loop telemetry — deliberately flat/plain (no
/// serde derives here; `MissionPlanner::server` routes convert to whatever
/// wire shape the API needs, per this crate's existing convention of
/// keeping physics-adjacent structs free of API concerns).
#[derive(Clone, Debug)]
pub struct CruiseTickRow {
    pub t_s: f64,
    pub r_m: Vector3<f64>,
    pub v_mps: Vector3<f64>,
    pub q: Vector4<f64>,
    /// The commanded/target attitude quaternion this tick — whatever
    /// `pointing_mode` (or, if `commander` is set, the resolved priority-
    /// ordered rule solve) decided the spacecraft SHOULD be pointed at.
    /// `q` above is what it actually achieved; the two are what the
    /// quaternion-PD control law is trying to drive together.
    pub q_cmd: Vector4<f64>,
    pub omega_radps: Vector3<f64>,
    pub wheel_speeds_radps: [f64; 4],
    pub wheel_momentum_nms: f64,
    /// Angle [deg] between the current and commanded attitude.
    pub pointing_error_deg: f64,
    /// Body-frame torque [N*m] the quaternion-PD law computed from
    /// `(state.q, q_cmd, omega)` — BEFORE wheel-speed-saturation zeroing
    /// (`sim_engine::control::allocate`'s `WheelsPrimary` fix, see that
    /// module's doc comment). This is what the controller WANTED to apply.
    pub torque_cmd_body_nm: Vector3<f64>,
    /// Body-frame torque [N*m] actually applied to the spacecraft this tick
    /// (`net_body_torque`, post-saturation) — what the controller COULD
    /// deliver given any wheel already pinned at its speed limit. Differs
    /// from `torque_cmd_body_nm` exactly when a wheel is saturated (see the
    /// real Newton's-third-law bug this pair of fields exists to make
    /// visible, documented in `sim_engine::control`'s WheelsPrimary branch).
    pub torque_delivered_body_nm: Vector3<f64>,
    /// `|actual_r - reference_r|` at this tick's time, from
    /// `reference_guidance::dispersion` — NOT corrected by this loop (no TCM
    /// execution yet), purely reported so tracking quality can be verified.
    pub dr_m: f64,
    /// `|actual_v - reference_v|` at this tick's time.
    pub dv_mps: f64,
    pub rcs_propellant_kg_cum: f64,
    /// 0-1, fraction of max wheel speed (most-saturated wheel) — same
    /// formula `sim_engine::engine::StepTelemetry::wheel_sat_frac` uses.
    pub wheel_sat_frac: f64,
    /// Review E3: the UNCLAMPED per-wheel motor-torque
    /// command [N·m] — see `sim_engine::control::AllocationOutput::
    /// wheel_motor_torque_cmd_nm`.
    pub wheel_motor_torque_cmd_nm: [f64; 4],
    /// Review E3: the DELIVERED per-wheel motor torque [N·m] (post
    /// ±max_torque clamp and speed-limit zeroing) — paired with the
    /// commanded value above so torque saturation's direction distortion
    /// is observable.
    pub wheel_motor_torque_nm: [f64; 4],
    /// Review E3: `max_i |cmd_i| / max_torque` — the torque-authority twin
    /// of the speed-based `wheel_sat_frac`; EXCEEDS 1.0 when the demand
    /// overloads the wheels.
    pub wheel_torque_sat_frac: f64,
    /// Which `planned_burns` entry the current `Slewing`/`Burning` phase
    /// belongs to (burn-executive ask) — `None` while coasting,
    /// RCS-correcting, or in a REACTIVE maneuver. The raw data
    /// [`detect_planned_burn_reports`] derives per-burn epochs from.
    pub planned_burn_idx: Option<usize>,
    /// Set on exactly the tick a planned burn is declared missed
    /// (`"MissedBurnTimeout"`), `None` otherwise.
    pub planned_burn_fault: Option<&'static str>,
    /// `propellant_mass_kg - rcs_propellant_kg_cum - tcm_propellant_kg_cum`,
    /// clamped to `>= 0.0` (item #3 — previously unclamped, so
    /// a hypothetical over-consumption could report a negative value; the
    /// executive itself is also gated on this reaching zero, see
    /// `run_cruise_leg`'s doc comment).
    pub propellant_remaining_kg: f64,
    /// Disturbance torque source magnitudes [N*m] at this tick's
    /// environment (`propagator6dof::disturbance_torque_breakdown`,
    /// Phase 13b) — gravity-gradient and SRP, reported separately so a
    /// GNC-manual-style "which source dominates" plot is possible.
    pub torque_gravity_gradient_nm: f64,
    pub torque_srp_nm: f64,
    /// Translational acceleration source magnitudes [m/s^2] at this tick's
    /// environment (`propagator6dof::translational_accel_breakdown`).
    /// `accel_srp_mps2` is always 0.0 in this first cut — SRP is not wired
    /// into translation for the decoupled `step_tick` path (see this
    /// module's doc comment and the design notes SRP-on-translation gap note);
    /// reported anyway (not omitted) so the field is honest about being a
    /// real, currently-zero measurement rather than absent.
    pub accel_central_gravity_mps2: f64,
    pub accel_third_body_mps2: f64,
    pub accel_srp_mps2: f64,
    /// Which named `GncModeConfig` was active this tick — `None` when no
    /// `GncCommander` was supplied (legacy fixed-`CruisePointingMode`
    /// behavior, unchanged) or when the commander itself had no mode to
    /// select (no schedule entry covers `t_s` and no `safe_mode` is set —
    /// falls back to the caller's `pointing_mode` in that case, reported
    /// as `None` since no named mode actually applied).
    pub active_mode: Option<String>,
    /// Worst (largest) angular error [deg] among this mode's rules beyond
    /// the top 2 (which cannot control the attitude — see `docs/MP/
    /// MANUAL.md` §9.4) — `None` when no commander is active, or the
    /// active mode has 2 or fewer rules (nothing left to violate).
    pub max_rule_violation_deg: Option<f64>,
    /// Label of the rule achieving `max_rule_violation_deg`, for "which
    /// lower-priority rule broke" reporting per the original ask.
    pub worst_violated_rule_label: Option<String>,
    /// TCM executive state this tick (added, item 5; extended
    /// with `RcsCorrecting`, item #4) — `None` when TCM is
    /// disabled (`tcm` argument to [`run_cruise_leg`] is `None`) or the
    /// loop is quietly coasting; `Some("Slewing")` while reorienting to
    /// burn attitude; `Some("Burning")` while the main-engine finite burn
    /// is actually firing; `Some("RcsCorrecting")` while an RCS-only
    /// correction is firing (no reorientation — see [`choose_tcm_actuator`]).
    /// See [`TcmPhase`].
    pub tcm_phase: Option<&'static str>,
    /// Cumulative TCM propellant consumed [kg] — main-engine burns
    /// (`TcmPhase::Burning`) AND RCS-only corrections (`TcmPhase::
    /// RcsCorrecting`, item #4) share this pool, since both
    /// draw from the same `spacecraft.propellant_mass_kg` budget in this
    /// model. Separate from `rcs_propellant_kg_cum` (which tracks RCS used
    /// for ATTITUDE control/desaturation, not translation).
    pub tcm_propellant_kg_cum: f64,
    /// Cumulative REAL delivered TCM/burn translational ΔV [m/s] (frontend
    /// ask 2d) — Tsiolkovsky per burning tick, projected RCS
    /// impulse per `RcsCorrecting` tick. Accumulated after the row is
    /// built, so a row's value covers everything through the previous tick
    /// (same convention as the burning-tick half of
    /// `tcm_propellant_kg_cum`).
    pub tcm_dv_mps_cum: f64,
    /// The reactive-executive correction solve that ran THIS tick, if any
    /// — see [`TcmSolveInfo`]. `None` on the vast
    /// majority of ticks.
    pub tcm_solve: Option<TcmSolveInfo>,
    /// Three-layer attitude control (`MANUAL.md` §10.5):
    /// the allocation mode actually used this tick (`"WheelsPrimary"`,
    /// `"ThrustersPrimary"`, `"ThrustersOnly"`), the layer-1 law that
    /// produced `torque_cmd_body_nm` (`"Pd"`/`"Pid"`/`"PhasePlane"`), and
    /// the layer-2 activity (`"Hold"`/`"Slew"`/`"BurnHold"`/`"Coast"`).
    pub control_mode: &'static str,
    pub controller_law: &'static str,
    pub control_activity: &'static str,
    /// Effective `(k_p, k_d)` of the law in force this tick — after
    /// activity tuning and gain scheduling; `None` for phase-plane.
    pub controller_kp: Option<f64>,
    pub controller_kd: Option<f64>,
    /// Layer 3: set on exactly the ticks the active law was (re)derived —
    /// mode/activity change or a scheduled mass trigger — `None` otherwise.
    /// `run_cruise_streaming` collects these into
    /// `CruiseResult.gain_schedule_points`.
    pub gain_schedule_point: Option<GainSchedulePoint>,
}

/// One threshold-triggered trajectory-correction maneuver's configuration
/// — `run_cruise_leg`'s `tcm: Option<TcmConfig>` parameter. `None` (every
/// call site before and every demo binary still) preserves
/// exactly the old coast-only behavior.
#[derive(Debug, Clone, Copy)]
pub struct TcmConfig {
    /// Trigger a correction the first tick `dr_m` (position dispersion
    /// against `reference`) exceeds this [m], while `Coast`ing.
    /// `f64::INFINITY` (see [`build_tcm_config`]) disables reactive
    /// triggering entirely — used when only Phase 13n's planned burns need
    /// this struct's `thrust_n`/`isp_s`, with no reactive threshold set.
    pub dr_threshold_m: f64,
    /// Main-engine thrust [N] — from `spacecraft.propulsion`.
    pub thrust_n: f64,
    /// Main-engine specific impulse [s] — from `spacecraft.propulsion`.
    pub isp_s: f64,
}

/// The TCM executive's own state machine (item 5) — tracked
/// across ticks inside [`run_cruise_leg`], never exposed outside it
/// (`CruiseTickRow::tcm_phase` reports a plain string view instead).
///
/// `Coast` → (trigger: `dr_m > dr_threshold_m`, cooldown elapsed, enough TOF
/// remains — see the trigger-time guards documented on the constants below)
/// → EITHER `Slewing` → (converged: `pointing_error_deg <=
/// DEFAULT_SETTLE_THRESHOLD_DEG`) → `Burning` → (delivered ΔV >= the
/// target, tracked via the REAL Tsiolkovsky mass loss each burning tick
/// actually produced, not an idealized instantaneous impulse) → `Coast`,
/// OR (item #4 — see [`choose_tcm_actuator`]) `RcsCorrecting`
/// directly (no slew) → (delivered ΔV >= the target, or no further progress
/// possible) → `Coast`. A main-engine burn (`Slewing`/`Burning`) takes
/// priority over `pointing_mode`/`commander` for its whole duration — real
/// spacecraft ops don't run a comm pass mid-maneuver. `RcsCorrecting`
/// deliberately does NOT override pointing — that is the entire point of
/// preferring it for a small correction or a pointing-locked mode.
#[derive(Debug, Clone, Copy)]
enum TcmPhase {
    Coast,
    /// `is_planned` (Phase 13n) — `true` when this maneuver
    /// came from `CruiseSeedConfig::planned_burns` rather than the
    /// reactive dispersion trigger. Gates the item-13p continuous
    /// re-targeting fix below (`solve_tcm_correction`) OFF for a planned
    /// burn — that re-targeting solves a DIFFERENT problem (null current
    /// dispersion against the reference) than what a planned burn's fixed
    /// `dv_inertial_mps` represents, and must not overwrite it.
    Slewing { thrust_dir_inertial: Vector3<f64>, dv_target_mps: f64, is_planned: bool, planned: Option<PlannedSlew> },
    Burning { thrust_dir_inertial: Vector3<f64>, dv_remaining_mps: f64, is_planned: bool, planned: Option<PlannedSlew> },
    /// Item #4 — an RCS-only correction, no slew: the
    /// spacecraft's attitude stays under `pointing_mode`/`commander`
    /// control the whole time, and every tick fires whichever configured
    /// `RcsThruster`s currently have positive alignment with
    /// `thrust_dir_inertial` (re-resolved each tick against the CURRENT
    /// attitude, since it is free to move under normal pointing control).
    RcsCorrecting { thrust_dir_inertial: Vector3<f64>, dv_remaining_mps: f64 },
}

/// Scheduling state a PLANNED maneuver carries through `Slewing`/`Burning`
/// (burn-executive lead-time/timeout ask). `idx` indexes
/// `planned_burns` (so the ignition-time re-solve and the per-burn report
/// can find the entry); `ignition_epoch_s` is the configured epoch the
/// burn must not fire BEFORE (the slew starts early and the attitude is
/// HELD — inertially fixed — until then); `deadline_s` is the go/no-go
/// timeout after which an unsettled slew is declared a missed-burn fault
/// instead of waiting indefinitely.
#[derive(Debug, Clone, Copy)]
struct PlannedSlew {
    idx: usize,
    ignition_epoch_s: f64,
    deadline_s: f64,
}

/// Per-planned-burn execution report (asks: "log planned-burn
/// trigger epoch vs. configured `epoch_s` per burn" + the missed-burn
/// fault as a reported finding). Derived post-hoc from the tick rows'
/// `planned_burn_idx`/`planned_burn_fault` by [`detect_planned_burn_reports`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct PlannedBurnReport {
    pub index: usize,
    pub label: String,
    /// The configured `epoch_s`.
    pub configured_epoch_s: f64,
    /// First tick the executive entered `Slewing` for this burn — BEFORE
    /// `configured_epoch_s` by the computed lead time when the executive
    /// scheduled it in time; `None` if the burn was never reached.
    pub slew_start_s: Option<f64>,
    /// First `Burning` tick (real ignition). Late relative to
    /// `configured_epoch_s` only when the attitude had not settled by then.
    pub ignition_s: Option<f64>,
    /// First tick back in `Coast` after the burn completed.
    pub completed_s: Option<f64>,
    /// `"Completed"`, `"MissedBurnTimeout"` (attitude never settled within
    /// the go/no-go window — burn skipped), or `"NotReached"` (the leg
    /// ended before the burn's lead window opened).
    pub status: String,
    /// Capture burns only: the orbit the burn
    /// ACTUALLY bought about its `capture_body`, from the truth's state at
    /// completion — compare against the Phase 01 result's
    /// `post_capture_orbit_*` fields to see achieved vs. planned. `None`
    /// for non-capture burns, faulted/unreached burns, or when the body
    /// track doesn't resolve.
    pub achieved_capture: Option<AchievedCaptureReport>,
}

/// See [`PlannedBurnReport::achieved_capture`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct AchievedCaptureReport {
    /// Specific orbital energy about the capture body is negative.
    pub bound: bool,
    /// Body-relative radius [m] at ignition — the single number that
    /// decides what orbit a velocity-matching burn can buy (see
    /// `time_to_periapsis_rel_s`'s doc comment).
    pub ignition_radius_m: Option<f64>,
    /// `None` when not bound (hyperbolic escape — the burn failed).
    pub sma_m: Option<f64>,
    pub eccentricity: f64,
    pub periapsis_m: Option<f64>,
    pub apoapsis_m: Option<f64>,
    pub period_s: Option<f64>,
}

/// One reactive-executive correction SOLVE, reported on the tick it ran
///: previously the only way to tell "solver
/// converged" from "returned the Lambert guess" was `TCM_DEBUG` on the
/// server console. Carried on `CruiseTickRow::tcm_solve` (and streamed via
/// `CruiseStepMsg`); [`detect_reactive_burn_reports`] folds these into the
/// per-maneuver [`ReactiveBurnReport`] list. Per-tick Slewing re-target
/// solves are deliberately NOT reported (one per tick — noise, and never
/// what actually fires); the trigger-time and ignition-time solves are.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TcmSolveInfo {
    /// `"Trigger"` (Coast-phase dispersion trigger), `"Ignition"` (the
    /// settled-attitude re-solve that decides what actually burns), or
    /// `"RcsTrigger"` (the RCS-path refine, which has no ignition step).
    pub context: &'static str,
    /// `"BPlane"` / `"Position"` when a shooting refinement ran,
    /// `"Lambert"` when only the two-body guess was solved (the direct
    /// trigger path — refinement happens at ignition there).
    pub targeting: &'static str,
    /// Epoch [s] the solve aimed at (SOI entry for a B-plane solve, the
    /// bounded Lambert horizon otherwise).
    pub aim_epoch_s: f64,
    /// |ΔV| [m/s] of the two-body Lambert guess the solve started from.
    pub lambert_dv_mps: Option<f64>,
    /// |ΔV| [m/s] of the solve's own result (what the decision was made on).
    pub solved_dv_mps: f64,
    /// Newton iterations the shooting refinement performed — 0 with a
    /// shooting `targeting` means it returned the guess untouched. `None`
    /// when no refinement ran (`targeting == "Lambert"`).
    pub iterations: Option<u32>,
    pub converged: Option<bool>,
    /// Final residual miss [m] at the aim point under the real force model.
    pub miss_m: Option<f64>,
    /// Predicted delivery miss [m] with NO burn (B-plane path only) — the
    /// (2b) trigger metric.
    pub no_burn_miss_m: Option<f64>,
    /// What the executive did with it: `"FireMainEngine"`, `"FireRcs"`,
    /// `"Ignite"`, `"ReSlew"` (refined direction moved > the re-slew gate),
    /// `"DeclinedDeliveryFine"` (no-burn miss already within threshold),
    /// `"DeclinedNotWorth"` (non-converged and not clearly helping), or
    /// `"AbortedAtIgnition"`.
    pub decision: &'static str,
}

/// Per-reactive-maneuver execution report —
/// the reactive twin of [`PlannedBurnReport`], derived post-hoc from the
/// tick rows by [`detect_reactive_burn_reports`]. Declined trigger solves
/// (no maneuver entered) get their own entry with zero executed ΔV, so a
/// run's full correction history — including the corrections the executive
/// chose NOT to make — is on the result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReactiveBurnReport {
    /// First tick of the maneuver (Slewing/RcsCorrecting entry), or the
    /// declined solve's own tick.
    pub trigger_epoch_s: f64,
    /// `"MainEngine"`, `"Rcs"`, or `"None"` (declined — nothing fired).
    pub actuator: String,
    /// From the governing solve — see [`TcmSolveInfo::targeting`].
    pub targeting: Option<String>,
    pub aim_epoch_s: Option<f64>,
    pub lambert_dv_mps: Option<f64>,
    pub solved_dv_mps: Option<f64>,
    pub shooting_iterations: Option<u32>,
    pub shooting_converged: Option<bool>,
    /// Final residual miss [m] of the governing solve at its aim point.
    pub predicted_miss_m: Option<f64>,
    pub no_burn_miss_m: Option<f64>,
    /// First `Burning`/`RcsCorrecting` tick — `None` if nothing fired.
    pub ignition_s: Option<f64>,
    /// First tick back in `Coast` — `None` if the leg ended mid-maneuver.
    pub completed_s: Option<f64>,
    /// Real delivered translational ΔV [m/s] over the maneuver (Tsiolkovsky
    /// per burning tick / projected RCS impulse per correcting tick).
    pub executed_dv_mps: f64,
    /// Propellant this maneuver consumed [kg] (main-engine + RCS-TCM pool).
    pub propellant_kg: f64,
    /// `"Completed"`, `"Declined"`, `"AbortedAtIgnition"` (slew ran but the
    /// ignition re-solve refused to fire), or `"Unfinished"` (leg ended
    /// mid-maneuver).
    pub status: String,
}

impl TcmPhase {
    fn label(&self) -> Option<&'static str> {
        match self {
            TcmPhase::Coast => None,
            TcmPhase::Slewing { .. } => Some("Slewing"),
            TcmPhase::Burning { .. } => Some("Burning"),
            TcmPhase::RcsCorrecting { .. } => Some("RcsCorrecting"),
        }
    }
}

// ── TCM executive constants (the design notes Phase 13m) ─────────────────────
//
// All hand-picked, self-scaling guard rails — same status as
// `DEFAULT_SETTLE_THRESHOLD_DEG`/`DEFAULT_MOMENTUM_DUMP_GAIN_PER_S` below
// (tunable, not derived from first principles). Deliberately NOT config
// fields: these are executive-robustness guards, not mission-design
// parameters, matching this module's existing convention for that
// distinction.

/// [fraction of `duration_s`] — item #2: minimum remaining time-to-leg-end
/// required before the TCM executive will attempt a NEW Lambert correction.
/// An endpoint-targeted fixed-time-of-arrival Lambert solve becomes
/// ill-conditioned as the remaining TOF shrinks toward zero (found
/// a real run showed near-zero dispersion for most of a
/// 117-day mission, then an exponential trajectory blowup in the final
/// ~10 days) — below this fraction of the leg's own duration, the
/// executive stops attempting new corrections entirely and simply reports
/// increased dispersion, matching real TCM ops practice (no correction is
/// planned so close to arrival that there is no time left to see whether
/// it worked).
const TCM_MIN_REMAINING_TOF_FRACTION: f64 = 0.02;

/// [fraction of the LOCAL osculating orbital period] — Phase 13p (found
/// root-causing the "corrections fire repeatedly but never
/// converge dispersion" bug): `tcm_lambert_correction` always targeted
/// `duration_s` (the fixed leg-end epoch) directly, with NO regard for how
/// many orbital revolutions separate "now" from then. Confirmed by direct
/// reproduction (`cruise::tests::diag_13p_many_corrections_over_a_long_leg`):
/// early in a mission, when nearly the FULL remaining duration lies ahead,
/// a single-revolution Lambert solve (`orbital_math::lambert::lambert`, not
/// the `_n_rev` family) forced to satisfy a multi-revolution time-of-flight
/// for what's really just a small residual position error is a genuinely
/// ill-posed problem — it still returns SOME solution, but a wildly wrong
/// one (measured: a 13 km position error produced a demanded 362 KM/S
/// correction). The fix is NOT a multi-rev-aware solver (that would still
/// hand back a real, huge re-targeting burn spanning multiple revolutions,
/// which is not what a TCM is for) — it's targeting a NEARER epoch, exactly
/// as `tcm_lambert_correction`'s own doc comment already anticipated
/// ("typically the next reference waypoint... the caller's choice"), which
/// this executive never actually did. The target epoch is capped at
/// `state.t_s + this_fraction × local_period` (via `semi_major_axis_m`/
/// `orbital_period_s` at the CURRENT state — vis-viva, generic/mu-
/// parameterized), falling back to `duration_s` whenever that's already
/// nearer (the common case for a realistic interplanetary cruise, where the
/// whole mission covers well under one heliocentric revolution) or the
/// current state is hyperbolic (`semi_major_axis_m <= 0`, no periodic bound
/// applies). A QUARTER period (not half) — found live-testing this exact
/// fix: `0.5` is the WORST possible choice, not a safe middle ground. For a
/// near-circular reference orbit, targeting exactly half a period ahead is
/// (by construction) close to a 180° transfer angle — the classical Lambert
/// geometric degeneracy (`sin(Δν) → 0`, the SAME failure mode
/// `MIN_SIN_TRANSFER_ANGLE` guards against, `orbital_math::lambert`), so
/// `lambert()` legitimately returns no solution and nothing fires for a
/// long stretch (confirmed: with `0.5`, the diagnostic test below went from
/// t=10 to t=31,510 without a single successful solve, despite dispersion
/// exceeding threshold every tick in between). A quarter period keeps every
/// correction inside a single, well-conditioned Lambert branch (safely
/// under one full revolution — no multi-rev ambiguity) while landing near a
/// 90° transfer angle for a circular-ish reference, comfortably clear of
/// BOTH degenerate extremes (near-0° from too-short a horizon, near-180°/
/// 360° from a horizon at or near a half/full period).
const TCM_HORIZON_PERIOD_FRACTION: f64 = 0.25;

/// [fraction of the ΔV deliverable from ALL remaining propellant] — item
/// #2's defense-in-depth magnitude cap, on top of the TOF floor above.
/// Computed from the real Tsiolkovsky-available ΔV at the trigger instant
/// (`isp_s * g0 * ln(mass / dry_mass)`) rather than a fixed absolute
/// number, so the cap scales with the actual vehicle/propellant state
/// instead of being simultaneously too loose for a small vehicle and too
/// tight for a large one. A single correction may never claim more than
/// half of whatever ΔV remains available — leaves margin for future
/// corrections, and bounds the worst case even if the TOF floor above is
/// somehow bypassed.
const TCM_DV_CAP_FRACTION_OF_AVAILABLE: f64 = 0.5;

/// [fraction of the REMAINING time-to-leg-end AT THE MOMENT A CORRECTION
/// ENDS] — item #1's fixed-cooldown guard: how long the executive waits
/// after a burn/RCS correction completes before it will even evaluate a
/// new trigger. A burn only changes VELOCITY — position error takes real
/// time to reflect the correction, so checking immediately after burnout
/// sees essentially the SAME dispersion that triggered the burn and
/// retriggers immediately (found: chaotic pointing/wheel
/// saturation for an entire 117-day mission, ~760 kg propellant burned
/// against a 600 kg tank). Scaled by the remaining time AT COMPLETION
/// (not a fixed duration) so a correction planned early in a long cruise
/// gets a proportionally longer observation window than one planned close
/// to arrival.
///
/// **A second, trend-based guard (`dr_now < dr_at_completion`) was tried
/// alongside this and REMOVED the same day, found by the frontend session
/// live-testing a real run** — it was a real, worse regression, not a
/// working belt-and-suspenders check: `dr_at_completion` is a single fixed
/// snapshot from the ONE most recent completed correction, never updated
/// unless another correction completes — which, once natural drift pushes
/// dispersion back above that snapshot (the whole reason a SECOND
/// correction is ever needed), it can't, because the guard that would
/// allow it is exactly the one requiring dispersion to be BELOW that
/// snapshot. The result is a permanent lockout after the very first
/// correction: live-verified against a real Mercury Orbiter run, one
/// Slewing→Burning transition at t=12,420s and then `Coast` for the
/// remaining ~11.1M s of an 11.15M s leg while `dr_m` grew to 6.4 BILLION
/// meters, completely unchecked. Whatever "confirm the burn actually
/// helped" value this guard was meant to add isn't worth reintroducing
/// without a real bounded-window design (e.g. only checked once, right as
/// the cooldown timer elapses, not evaluated forever after) — the fixed
/// cooldown alone already fixes the original immediate-retrigger bug this
/// was built for.
const TCM_COOLDOWN_FRACTION_OF_REMAINING_TOF: f64 = 0.05;

/// [m/s] — item #4: ΔV magnitude at or below which the TCM executive
/// prefers RCS (no slew) over the main engine, matching real ops practice
/// (main engine + slew is the DEFAULT; RCS is only for small corrections
/// or when pointing must not be disturbed — see [`choose_tcm_actuator`]).
/// Small relative to a nominal interplanetary TCM (typically single-digit
/// to tens of m/s) — RCS thrusters are attitude-control-sized (~N-class)
/// and would take impractically long to deliver anything larger.
const TCM_RCS_DV_THRESHOLD_MPS: f64 = 0.5;

/// [N] — item #4: minimum RCS thrust achievable along the needed
/// correction direction, from the CURRENT attitude (no slew), before the
/// executive commits to an RCS-only correction instead of falling back to
/// main-engine + slew. Below this, the placed RCS geometry is too poorly
/// aligned (or no thrusters are configured) for RCS to matter.
const TCM_RCS_MIN_ALIGNED_THRUST_N: f64 = 1e-6;

// Review E2: 13o's `SLEWING_GAIN_SCALE` gain-softening is
// SUPERSEDED by the rate-limited eigenaxis slew profile
// (`sim_engine::control::profiled_slew_target`, wired into the tick loop
// below) — the structural fix for exactly what the softening only
// palliated: raw PD on a 90–180° error with torque-limited wheels is
// prone to saturation/limit-cycling regardless of gain choice, because
// the proportional term commands torque the wheels can't deliver.
// Profiling keeps the tracked error (and therefore the demand) small BY
// CONSTRUCTION, for mode transitions AND TCM slews alike, so ordinary
// full gains apply throughout. The three constants below parameterize the
// profile.

/// [rad] — errors above this engage the slew profile; below it, ordinary
/// small-angle pointing-hold tracks the final target directly (the ask's
/// own "~10–20°" band).
const SLEW_PROFILE_ENGAGE_THRESHOLD_RAD: f64 = 15.0 * std::f64::consts::PI / 180.0;

/// [rad/s] — cruise slew-rate ceiling (~0.57°/s), a deliberately gentle
/// reorientation rate matching real ops practice (a maneuver slew takes
/// minutes to hours in real operations, unless it's an emergency).
const SLEW_PROFILE_CRUISE_RATE_RADPS: f64 = 0.01;

/// [ticks] — how far ahead of the current attitude the profiled target
/// leads, in units of the control tick: enough tracked error for the PD
/// to generate real torque, small enough that the demand stays well
/// inside wheel authority.
const SLEW_PROFILE_LEAD_TICKS: f64 = 3.0;

// Burn-executive lead time + go/no-go timeout (standard ops
// design): keep the 0.5° alignment gate, but START a planned burn's slew
// BEFORE its epoch by a planning-computed lead time so ignition happens AT
// the epoch instead of late by the slew duration; HOLD the (inertially
// fixed) burn attitude while coasting until the epoch; and if the attitude
// has not settled within a defined window after the epoch, declare a
// missed-burn fault instead of waiting indefinitely. The lead time is
// well-defined only because §10.4's profile bounds the slew rate — the
// two mechanisms belong together.

/// [s] — earliest the executive starts evaluating a planned burn's lead
/// time (bounds the per-tick fresh ΔV solve to a window before the epoch).
const PLANNED_BURN_MAX_LEAD_S: f64 = 4.0 * 3600.0;

/// [s] — settle margin added to the kinematic slew time
/// `θ / SLEW_PROFILE_CRUISE_RATE_RADPS` when computing the lead time.
const PLANNED_BURN_SLEW_MARGIN_S: f64 = 300.0;

/// [s] — go/no-go window after the configured epoch: an unsettled slew
/// past `epoch_s + this` is a missed-burn fault (reported, burn skipped).
/// Per-revolution retry for a parking-orbit injection would need
/// a central-body-aware orbital period the executive
/// doesn't track yet — a documented follow-up, not silently approximated
/// with the heliocentric period.
const PLANNED_BURN_TIMEOUT_S: f64 = 1800.0;

/// [m/s] — a TRACKING capture burn (see the `Burning` execution block) is
/// complete once the freshly-solved velocity-matching ΔV falls below this:
/// the remaining fraction of a m/s is cheaper to leave to the next
/// station-keeping pass than to chase with a 100+ N engine.
const CAPTURE_BURN_COMPLETE_DV_MPS: f64 = 0.5;

/// [deg] — ignition-time direction consistency: the vehicle
/// slews to and settles on the TRIGGER-time solve's thrust direction; the
/// ignition-time re-solve may move that direction (a little for a position
/// re-solve, a LOT when B-plane targeting replaces a Lambert guess). Firing
/// the refined MAGNITUDE along the stale HELD direction was a real, live
/// failure (Mars approach: every solve converged to <1.5 m yet
/// each burn manufactured the next dispersion — `dv_after` oscillated by
/// ±8 m/s). If the refined direction differs from the held one by more
/// than this, the executive RE-SLEWS to the refined direction instead of
/// igniting; the settle gate then re-applies.
const REFINED_DIRECTION_RESLEW_DEG: f64 = 2.0;

/// [deg] — burn-attitude abort: a main-engine burn whose
/// pointing error exceeds this for longer than `BURN_ABORT_HOLD_S` of burn
/// time is CUT OFF and the phase returns to `Coast` (fault
/// `BurnAttitudeAbort`). Found live: an engine-torque/RCS-authority
/// mismatch tumbled the vehicle during a capture burn and the burn kept
/// firing 116° off-axis until the tank was empty, because "ΔV remaining"
/// is measured from mass lost, not from ΔV delivered along the intended
/// direction. Real flight software cuts a burn on an attitude-error limit
/// for exactly this reason. 20° is far outside any sensible burn-hold
/// envelope yet well short of a wasted burn (cos 20° = 0.94).
const BURN_ABORT_POINTING_DEG: f64 = 20.0;
/// [s] — how long the pointing error must exceed `BURN_ABORT_POINTING_DEG`
/// before the burn is aborted (a one-tick ignition transient is not a
/// fault; a sustained tumble is).
const BURN_ABORT_HOLD_S: f64 = 30.0;

/// [s] — phase-adaptive reporting (`CruiseSeedConfig::
/// report_stride_maneuver`): after any `tcm_phase` or `active_mode`
/// transition, ticks keep relaying at the maneuver stride for this long,
/// so the settle/recovery right after a maneuver or mode change is
/// visible at full resolution, not just the maneuver itself.
const MANEUVER_REPORT_TRAIL_S: f64 = 600.0;

/// [kg] — item #3: propellant below this is treated as exhausted (a
/// nonzero epsilon rather than a bare `<= 0.0` check, since floating-point
/// accumulation of many small per-tick consumptions can leave a residual
/// value that is numerically nonzero but physically meaningless).
const PROPELLANT_EXHAUSTED_EPSILON_KG: f64 = 1e-9;

/// Which actuator delivers a TCM correction (item #4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TcmActuator {
    MainEngine,
    Rcs,
}

/// Review D5 — named `(track, mu)` lookup table for planned
/// capture burns, built once per run from `cruise_seed.body_tracks` (the
/// SAME tracks pointing/perturbation already consume, interpolated through
/// the same cubic Hermite `ReferenceTrajectory`). Entries whose μ doesn't
/// resolve (no explicit override, no catalog match) are excluded — the
/// runtime lookup then misses and the burn falls back to its stored
/// nominal vector (`check_config` already rejects a `capture_body`
/// pointing at such a track, so this is defense in depth, not the primary
/// gate).
fn build_capture_track_table(
    seed: &crate::config::CruiseSeedConfig,
) -> Vec<(String, ReferenceTrajectory, f64)> {
    seed.body_tracks
        .iter()
        .filter_map(|t| {
            if t.track.len() < 2 {
                return None;
            }
            let mu = t.mu_m3s2.or_else(|| body_models::TargetBody::by_name(&t.name).map(|b| b.mu_m3s2))?;
            let pts: Vec<ReferencePoint> = t
                .track
                .iter()
                .map(|p| ReferencePoint {
                    t_s: p.t_s,
                    r_m: Vector3::new(p.r_m[0], p.r_m[1], p.r_m[2]),
                    v_mps: Vector3::new(p.v_mps[0], p.v_mps[1], p.v_mps[2]),
                })
                .collect();
            Some((t.name.clone(), ReferenceTrajectory::new(pts), mu))
        })
        .collect()
}

/// The ΔV a planned burn should fire, solved from the CURRENT state:
/// capture (review D5, velocity-matching) if `capture_body` is set, else
/// DSM (nuance 2, Lambert position-shaping) if `target_epoch_s` is set,
/// else the stored nominal vector — each fresh solve falling back to the
/// stored vector rather than silently skipping. Shared by the lead-time
/// scheduling check (direction → slew time), and the ignition-time
/// re-solve (ask item 4: a burn that ignites late re-solves at
/// ACTUAL ignition).
fn planned_burn_dv(
    burn: &crate::config::PlannedBurnConfig,
    state: &SixDofState,
    reference: &ReferenceTrajectory,
    mu_central_m3s2: f64,
    // The closure's 4th element is the configured
    // `[trajectory.capture].capture_eccentricity` — the capture solve
    // targets the CONFIGURED orbit's periapsis speed (see
    // `solve_capture_burn_dv`), matching the Phase 01 pricing.
    capture_state_at: &dyn Fn(&str, f64) -> Option<(Vector3<f64>, Vector3<f64>, f64, f64)>,
    // the truth's own perturbers + tolerances, so a DSM's
    // fresh solve is refined by shooting under the REAL force model.
    bodies: &[PropagatorBody],
    rtol: f64,
    atol: f64,
) -> Vector3<f64> {
    let stored_dv = Vector3::new(burn.dv_inertial_mps[0], burn.dv_inertial_mps[1], burn.dv_inertial_mps[2]);
    burn.capture_body
        .as_deref()
        .and_then(|name| capture_state_at(name, state.t_s))
        .and_then(|(body_r, body_v, mu, e)| solve_capture_burn_dv(&state.r_m, &state.v_mps, &body_r, &body_v, mu, e))
        .or_else(|| {
            burn.target_epoch_s.and_then(|target_epoch_s| {
                let guess = tcm_lambert_correction(&state.r_m, &state.v_mps, state.t_s, target_epoch_s, reference, mu_central_m3s2, true)?;
                // A planned DSM fires regardless (it is not optional the
                // way a reactive TCM is); a non-converged refinement is
                // reported, and the best Δv found is used.
                let solve = tcm_shooting_correction(&state.r_m, &state.v_mps, state.t_s, target_epoch_s, reference, mu_central_m3s2, bodies, rtol, atol, guess);
                if !solve.converged {
                    eprintln!(
                        "Warning: planned DSM re-solve at t={:.0} s did not converge (miss {:.3e} m at target epoch {:.0} s); firing best Δv found",
                        state.t_s, solve.miss_m, target_epoch_s
                    );
                }
                Some(solve.dv)
            })
        })
        .unwrap_or(stored_dv)
}

#[cfg(test)]
mod shooting_tests {
    use super::*;

    /// §9.2: under a real third-body perturber the two-body Lambert
    /// correction misses the reference at the horizon by kilometres and
    /// leaves a velocity error; the shooting refinement (same perturber in
    /// the loop) hits it to within the tolerance. Synthetic heliocentric
    /// setup: Sun-central, one Earth-mass perturber parked ~2 million km
    /// off the path (a strong but realistic cruise-phase perturbation), a
    /// reference built by propagating WITH the perturber, a dispersed
    /// start 50 km off the reference.
    #[test]
    fn shooting_refinement_beats_lambert_under_a_perturber() {
        let mu_sun = orbital_models::constants::MU_SUN;
        let au = 1.495978707e11;
        let r0 = Vector3::new(au, 0.0, 0.0);
        let v0 = Vector3::new(0.0, (mu_sun / au).sqrt(), 0.0);
        let perturber_pos = r0 + Vector3::new(2.0e9, 1.0e9, 0.0);
        let perturber_state = move |_t: f64| (perturber_pos, Vector3::zeros());
        let bodies = vec![PropagatorBody {
            name: "Perturber",
            mu_m3s2: 3.986004418e14,
            soi_radius_m: None,
            state_at: &perturber_state,
            central_fidelity: None,
            radius_m: None,
        }];
        let horizon_s = 20.0 * 86400.0;
        let (rtol, atol) = (1e-9, 1e-3);
        // Reference: propagated with the perturber, sampled hourly.
        let reference = sample_reference_trajectory_uniform(r0, v0, mu_sun, &bodies, horizon_s, 3600.0, rtol, atol);
        // Dispersed start: 50 km cross-track.
        let r_disp = r0 + Vector3::new(0.0, 0.0, 5.0e4);
        let lambert = tcm_lambert_correction(&r_disp, &v0, 0.0, horizon_s, &reference, mu_sun, true).expect("lambert solves");
        let miss_at = |dv: &Vector3<f64>| {
            let leg = propagate(r_disp, v0 + dv, 0.0, horizon_s, mu_sun, &bodies, horizon_s, rtol, atol);
            (leg.last().unwrap().r_m - reference.state_at(horizon_s).0).norm()
        };
        let miss_lambert = miss_at(&lambert);
        let solve = tcm_shooting_correction(&r_disp, &v0, 0.0, horizon_s, &reference, mu_sun, &bodies, rtol, atol, lambert);
        assert!(solve.converged);
        let miss_shooting = miss_at(&solve.dv);
        assert!(miss_lambert > 10.0 * TCM_SHOOTING_TOL_M, "fixture should make Lambert miss by km, got {miss_lambert:.0} m");
        assert!(miss_shooting < TCM_SHOOTING_TOL_M, "shooting should converge: {miss_shooting:.0} m (lambert {miss_lambert:.0} m)");
        assert!(miss_shooting < 0.01 * miss_lambert);
    }

    /// §9.2: the reactive horizon may never reach a reference
    /// point inside a registered SOI-capture body's sphere. Synthetic
    /// reference: a straight heliocentric line that passes through a
    /// stationary "planet" with a 1e9 m SOI between t = 60 d and t = 80 d
    /// (entry at ~60 d), a planned capture burn at 70 d.
    #[test]
    fn reactive_horizon_stops_at_the_reference_soi_entry_before_the_capture_burn() {
        let day = 86_400.0;
        let planet_pos = Vector3::new(1.0e11, 0.0, 0.0);
        let planet_state = move |_t: f64| (planet_pos, Vector3::zeros());
        let bodies = vec![PropagatorBody {
            name: "Planet",
            mu_m3s2: 4.28e13,
            soi_radius_m: Some(1.0e9),
            state_at: &planet_state,
            central_fidelity: None,
            radius_m: None,
        }];
        // x(t) moves 1e9 m/day; the planet sits at x = 1e11 → |x − 1e11| <
        // 1e9 for 99 d < t < 101 d ... use a simpler parameterization:
        // reference x = 1e11 + (t − 70 d)·(1e9/10 d) → inside for |t−70d| < 10 d.
        let pts: Vec<ReferencePoint> = (0..=200)
            .map(|k| {
                let t = k as f64 * day; // 0..200 d
                ReferencePoint {
                    t_s: t,
                    r_m: Vector3::new(1.0e11 + (t - 70.0 * day) * (1.0e9 / (10.0 * day)), 0.0, 0.0),
                    v_mps: Vector3::new(1.0e9 / (10.0 * day), 0.0, 0.0),
                }
            })
            .collect();
        let reference = ReferenceTrajectory::new(pts);
        let entries = reference_soi_entry_epochs(&reference, &bodies);
        assert_eq!(entries.len(), 1, "one entry expected, got {entries:?}");
        // Inside for |t − 70 d| < 10 d → first inside sample is t = 61 d
        // (t = 60 d sits exactly on the boundary, |Δ| = 1e9 is not < 1e9).
        assert!((entries[0].t_s - 61.0 * day).abs() < 1e-6, "entry at {:.2} d", entries[0].t_s / day);
        assert_eq!(entries[0].body_idx, 0);
        // The approach test recognises the entry as the binding horizon.
        assert!(approach_soi_entry(10.0 * day, 61.0 * day, &entries).is_some());
        assert!(approach_soi_entry(10.0 * day, 50.0 * day, &entries).is_none());

        let burns = vec![crate::config::PlannedBurnConfig {
            epoch_s: 70.0 * day,
            dv_inertial_mps: [0.0; 3],
            target_epoch_s: None,
            capture_body: None,
            external_stage: false,
            label: String::new(),
        }];
        // Well before the approach: the horizon end is the SOI entry, not
        // the burn epoch and not the leg end.
        let end = reactive_tcm_horizon_end_s(10.0 * day, &burns, 0, 200.0 * day, &entries);
        assert!((end - 61.0 * day).abs() < 1e-6);
        // Past the entry (inside the well) with the burn still ahead: the
        // burn epoch caps it (the inside-SOI inhibit blocks triggering
        // there anyway).
        let end = reactive_tcm_horizon_end_s(65.0 * day, &burns, 0, 200.0 * day, &entries);
        assert!((end - 70.0 * day).abs() < 1e-6);
        // After the burn fired: no more burns, no more entries → leg end.
        let end = reactive_tcm_horizon_end_s(90.0 * day, &burns, 1, 200.0 * day, &entries);
        assert!((end - 200.0 * day).abs() < 1e-6);
    }

    /// B-plane invariants of a constructed hyperbola: |B| = h/v∞, the
    /// periapsis radius follows from r_p = √(B² + (μ/v∞²)²) − μ/v∞², and
    /// the time-to-periapsis integrates to the right epoch under two-body
    /// propagation. Uses `hyperbolic_departure_state` to build an exact
    /// hyperbola (its periapsis state), then evaluates the B-plane at a
    /// point propagated BACKWARD onto the incoming leg.
    #[test]
    fn bplane_invariants_match_the_hyperbola_they_were_computed_from() {
        let mu = 4.282837e13; // Mars
        let r_p = 3.79e6;
        let v_inf_vec = Vector3::new(0.3, -0.9, 0.3).normalize() * 3_000.0;
        let dep = trajectory_solver::hyperbolic_departure_state(mu, r_p, v_inf_vec).unwrap();
        // Propagate backwards 2 days along the hyperbola (time-reversed
        // velocity) to get an inbound state well outside periapsis.
        let back = propagate(dep.r0_m, -dep.v0_mps, 0.0, 2.0 * 86_400.0, mu, &[], 86_400.0, 1e-12, 1e-3);
        let end = back.last().unwrap();
        let (r_in, v_in) = (end.r_m, -end.v_mps); // inbound: velocity re-reversed
        assert!(r_in.dot(&v_in) < 0.0, "inbound state should be approaching");
        let bp = bplane_of_relative_state(&r_in, &v_in, mu, 0.0).unwrap();
        let v_inf = 3_000.0_f64;
        assert!((bp.v_inf_mps - v_inf).abs() < 1e-3, "v∞ {}", bp.v_inf_mps);
        let b = (bp.bt_m * bp.bt_m + bp.br_m * bp.br_m).sqrt();
        let h = r_in.cross(&v_in).norm();
        assert!((b - h / v_inf).abs() < 1.0, "|B| {b} vs h/v∞ {}", h / v_inf);
        let a_abs = mu / (v_inf * v_inf);
        let r_p_from_b = (b * b + a_abs * a_abs).sqrt() - a_abs;
        assert!((r_p_from_b - r_p).abs() < 1.0, "r_p from B {r_p_from_b} vs {r_p}");
        // Periapsis passage is 2 days ahead of this state.
        assert!((bp.tca_s - 2.0 * 86_400.0).abs() < 1.0, "tca {}", bp.tca_s);
    }

    /// §9.2: on an approach, B-plane shooting converges where the Lambert
    /// guess (aimed at the SOI-entry position) leaves a real B-plane miss.
    /// Synthetic: a heavy planet parked on the heliocentric path, a
    /// reference built by propagating WITH the planet through its SOI, a
    /// dispersed start; the solve is evaluated at the reference's SOI entry.
    #[test]
    fn bplane_correction_converges_on_a_dispersed_approach() {
        let mu_sun = orbital_models::constants::MU_SUN;
        let au = 1.495978707e11;
        let mu_planet = 4.282837e13;
        let r_soi = 5.77e8;
        let day = 86_400.0;
        // Heliocentric circular-ish start; planet placed ~15 days downstream,
        // slightly off the path so the reference flies a real hyperbola.
        let r0 = Vector3::new(au, 0.0, 0.0);
        let v0 = Vector3::new(0.0, (mu_sun / au).sqrt(), 0.0);
        let horizon_s = 15.0 * day;
        let (rtol, atol) = (1e-10, 1e-3);
        // Park the planet 100,000 km off the UNPERTURBED path's 15-day
        // point (the heliocentric arc curves — a straight-line extrapolation
        // would miss the SOI by millions of km).
        let plain = propagate(r0, v0, 0.0, horizon_s, mu_sun, &[], horizon_s, rtol, atol);
        let planet_pos = plain.last().unwrap().r_m + Vector3::new(-7.0e7, 0.0, 7.0e7);
        let planet_state = move |_t: f64| (planet_pos, Vector3::zeros());
        let bodies = vec![PropagatorBody {
            name: "Planet",
            mu_m3s2: mu_planet,
            soi_radius_m: Some(r_soi),
            state_at: &planet_state,
            central_fidelity: None,
            radius_m: None,
        }];
        let reference = sample_reference_trajectory_uniform(r0, v0, mu_sun, &bodies, 20.0 * day, 1800.0, rtol, atol);
        let entries = reference_soi_entry_epochs(&reference, &bodies);
        assert_eq!(entries.len(), 1, "reference must enter the planet's SOI once: {entries:?}");
        let t_e = entries[0].t_s;
        // Dispersed start: 300 km cross-track, 0.2 m/s.
        let r_disp = r0 + Vector3::new(0.0, 0.0, 3.0e5);
        let v_disp = v0 + Vector3::new(0.2, 0.0, 0.0);
        let guess = tcm_lambert_correction(&r_disp, &v_disp, 0.0, t_e, &reference, mu_sun, true).expect("lambert guess");
        let solve = tcm_bplane_correction(&r_disp, &v_disp, 0.0, t_e, &reference, &bodies[0], mu_sun, &bodies, rtol, atol, guess)
            .expect("reference state at SOI entry is hyperbolic");
        assert!(solve.converged, "B-plane solve should converge, residual {:.0} m", solve.miss_m);
        // Independent check: fly the corrected state and compare periapsis
        // radius against the reference's own (from its B-plane invariants).
        let (pr, pv) = (bodies[0].state_at)(t_e);
        let (rr, rv) = reference.state_at(t_e);
        let bp_ref = bplane_of_relative_state(&(rr - pr), &(rv - pv), mu_planet, t_e).unwrap();
        let leg = propagate(r_disp, v_disp + solve.dv, 0.0, t_e, mu_sun, &bodies, t_e, rtol, atol);
        let end = leg.last().unwrap();
        let bp = bplane_of_relative_state(&(end.r_m - pr), &(end.v_mps - pv), mu_planet, t_e).unwrap();
        let rp = |b: &BPlane| {
            let bb = (b.bt_m * b.bt_m + b.br_m * b.br_m).sqrt();
            let a = mu_planet / (b.v_inf_mps * b.v_inf_mps);
            (bb * bb + a * a).sqrt() - a
        };
        assert!((rp(&bp) - rp(&bp_ref)).abs() < 2.0e3, "periapsis {:.0} vs reference {:.0}", rp(&bp), rp(&bp_ref));
    }

    /// A non-converged solve fires only if it still halves the dispersion.
    #[test]
    fn non_converged_shooting_solve_is_fired_only_when_it_clearly_helps() {
        let dv = Vector3::new(1.0, 0.0, 0.0);
        let s = |miss_m: f64, converged: bool| ShootingSolve { dv, miss_m, converged, iterations: 0, method: "Position", no_burn_miss_m: f64::NAN };
        assert!(shooting_solve_worth_firing(&s(5.0e3, true), 100.0));
        assert!(shooting_solve_worth_firing(&s(4.0e5, false), 1.0e6));
        assert!(!shooting_solve_worth_firing(&s(6.0e5, false), 1.0e6));
        assert!(!shooting_solve_worth_firing(&s(f64::INFINITY, false), 1.0e6));
    }
}

/// Maximum Newton iterations of the shooting refinement (§9.2).
const TCM_SHOOTING_MAX_ITERS: usize = 6;
/// Miss-distance convergence tolerance [m] — well below any realistic
/// `tcm_dr_threshold_m`, well above the reference's interpolation floor.
const TCM_SHOOTING_TOL_M: f64 = 1000.0;

/// §9.2 — refine a correction ΔV by SHOOTING under the real force model
///: the "fidelity-agnostic differential correction" the
/// Layer-3 design calls for, applied to the executive. Starting from
/// `dv_guess` (normally the two-body Lambert solution, which is exact only
/// for Sun-only dynamics), propagate the corrected state to `arrival_t_s`
/// with the SAME perturbers the truth flies (`bodies`), measure the miss
/// against the reference position there, build the 3×3 sensitivity
/// `∂r(t_h)/∂Δv` by central-free forward differences (three extra
/// propagations), and Newton-iterate:
///
/// ```text
/// r_h(Δv)      = propagate(r, v + Δv, t → t_h ; bodies)
/// miss         = r_h(Δv) − r_ref(t_h)
/// J_ij         = [ r_h(Δv + h·e_j) − r_h(Δv) ]_i / h
/// Δv ← Δv − J⁻¹·miss           until |miss| < tol
/// ```
///
/// Why this matters: a third-body acceleration of a few 10⁻⁶ m/s² over a
/// multi-week horizon integrates to metres per second, so the Lambert arc
/// that "hits" `r_ref(t_h)` under two-body dynamics arrives kilometres
/// off under the real ones and departs with m/s-level velocity error.
/// Cost: ~4 propagations per iteration, a handful of iterations, once per
/// burn — the executive runs it at IGNITION (and for a DSM's fresh
/// solve), never per tick. Returns the best (smallest-miss) Δv seen,
/// which is the guess itself if no iteration improved on it — together
/// with that miss and whether the tolerance was met, so the caller can
/// decline to fire a solve that did NOT converge (found on a
/// Mars approach: with the horizon pinned onto the reference's hyperbolic
/// periapsis, successive solves disagreed by tens of m/s — a non-converged
/// Newton, not a real correction; firing it was the whole failure).
#[allow(clippy::too_many_arguments)]
fn tcm_shooting_correction(
    r_m: &Vector3<f64>,
    v_mps: &Vector3<f64>,
    t_s: f64,
    arrival_t_s: f64,
    reference: &ReferenceTrajectory,
    mu_central_m3s2: f64,
    bodies: &[PropagatorBody],
    rtol: f64,
    atol: f64,
    dv_guess: Vector3<f64>,
) -> ShootingSolve {
    let tof_s = arrival_t_s - t_s;
    if tof_s <= 0.0 {
        return ShootingSolve { dv: dv_guess, miss_m: f64::INFINITY, converged: false, iterations: 0, method: "Position", no_burn_miss_m: f64::NAN };
    }
    let (r_target, _) = reference.state_at(arrival_t_s);
    let shoot = |dv: &Vector3<f64>| -> Vector3<f64> {
        let leg = propagate(*r_m, v_mps + dv, t_s, tof_s, mu_central_m3s2, bodies, tof_s, rtol, atol);
        leg.last().map(|p| p.r_m).unwrap_or(*r_m)
    };
    let mut dv = dv_guess;
    let mut miss = shoot(&dv) - r_target;
    let (mut best_dv, mut best_miss) = (dv, miss.norm());
    let mut iterations = 0u32;
    for _ in 0..TCM_SHOOTING_MAX_ITERS {
        if best_miss < TCM_SHOOTING_TOL_M {
            break;
        }
        let h = (1e-3 * dv.norm()).max(1e-2);
        let mut j = nalgebra::Matrix3::<f64>::zeros();
        for k in 0..3 {
            let mut dvk = dv;
            dvk[k] += h;
            let col = (shoot(&dvk) - r_target - miss) / h;
            j.set_column(k, &col);
        }
        let Some(j_inv) = j.try_inverse() else { break };
        let step = j_inv * miss;
        // Damp absurd steps (a near-singular sensitivity, e.g. a horizon
        // close to a half-revolution): never move by more than 10× the
        // current correction or 100 m/s in one Newton step.
        let cap = (10.0 * dv.norm()).max(1.0).min(100.0);
        let step = if step.norm() > cap { step * (cap / step.norm()) } else { step };
        dv -= step;
        miss = shoot(&dv) - r_target;
        iterations += 1;
        if miss.norm() < best_miss {
            best_dv = dv;
            best_miss = miss.norm();
        } else {
            // No improvement: stop rather than wander.
            break;
        }
    }
    ShootingSolve { dv: best_dv, miss_m: best_miss, converged: best_miss < TCM_SHOOTING_TOL_M, iterations, method: "Position", no_burn_miss_m: f64::NAN }
}

/// Outcome of [`tcm_shooting_correction`]: the best Δv found, its miss at
/// the horizon under the real force model, and whether the solve met
/// `TCM_SHOOTING_TOL_M`. A non-converged solve is a FINDING, not a
/// correction: the executive fires it only if it still improves on the
/// current dispersion (`shooting_solve_worth_firing`).
#[derive(Clone, Copy, Debug)]
struct ShootingSolve {
    dv: Vector3<f64>,
    miss_m: f64,
    converged: bool,
    /// Newton iterations actually performed (
    /// "solver converged" vs. "returned the guess" was previously only
    /// distinguishable via `TCM_DEBUG` on the server console — 0 means the
    /// solve handed back its input guess untouched).
    iterations: u32,
    /// Which targeting formulation produced this solve: `"BPlane"`
    /// (`tcm_bplane_correction`) or `"Position"` (`tcm_shooting_correction`).
    method: &'static str,
    /// Residual of the CURRENT trajectory with NO burn [m] — the predicted
    /// delivery miss if the executive does nothing. Only computed by the
    /// B-plane path (one extra propagation per trigger); `NaN` from the
    /// position path. The approach trigger declines to correct at all when
    /// this is already below the TCM threshold — the frontend's (2b) ask:
    /// on an approach the instantaneous `dr_m` legitimately stays large
    /// after a correct burn (B-plane targeting does not null intermediate
    /// positions), so triggering on it re-fires forever.
    no_burn_miss_m: f64,
}

/// Fire-or-skip rule for a shooting solve (§9.2): converged → fire; not
/// converged → fire only if the residual miss at the horizon is still
/// clearly smaller than the dispersion the burn was meant to remove (a
/// half), otherwise the "correction" is noise from an ill-conditioned
/// solve and doing nothing is the better maneuver.
fn shooting_solve_worth_firing(solve: &ShootingSolve, disp_now_m: f64) -> bool {
    solve.converged || solve.miss_m < 0.5 * disp_now_m
}

/// One reference SOI entry: the epoch and which registered body's sphere
/// the reference enters (index into the executive's `bodies`).
#[derive(Clone, Copy, Debug)]
struct SoiEntry {
    t_s: f64,
    body_idx: usize,
}

/// Epochs at which the REFERENCE trajectory enters a registered SOI-
/// capture body's sphere of influence (outside → inside transitions,
/// scanned over the reference's own samples), ascending. A reactive TCM
/// must never aim at a reference point inside a planet's SOI: the point
/// moves under that body's gravity, the Sun-central Lambert guess is
/// meaningless there, and the hyperbolic periapsis is the stiffest possible
/// target for the shooting Newton (found live on a Mars
/// approach — the horizon collapsed onto the capture point and successive
/// solves disagreed wildly). Standard practice aims the last approach TCMs
/// at the B-plane instead (§9.2, `tcm_bplane_correction`); the reactive
/// horizon is capped at this entry epoch (`reactive_tcm_horizon_end_s`)
/// and any correction whose horizon IS an SOI entry is solved as a B-plane
/// correction at that body.
fn reference_soi_entry_epochs(reference: &ReferenceTrajectory, bodies: &[PropagatorBody]) -> Vec<SoiEntry> {
    let pts = reference.points();
    let mut entries = Vec::new();
    for (body_idx, b) in bodies.iter().enumerate() {
        let Some(r_soi) = b.soi_radius_m else { continue };
        let mut inside_prev: Option<bool> = None;
        for p in pts {
            let inside = (p.r_m - (b.state_at)(p.t_s).0).norm() < r_soi;
            if inside && inside_prev == Some(false) {
                entries.push(SoiEntry { t_s: p.t_s, body_idx });
            }
            inside_prev = Some(inside);
        }
    }
    entries.sort_by(|a, b| a.t_s.total_cmp(&b.t_s));
    entries
}

/// Latest epoch a reactive correction may target from `t_s`: the next
/// planned burn's epoch (the reference beyond a capture burn is the captured
/// orbit —), the reference's next SOI entry (see
/// `reference_soi_entry_epochs`), or the leg end, whichever comes first.
fn reactive_tcm_horizon_end_s(
    t_s: f64,
    planned_burns: &[crate::config::PlannedBurnConfig],
    next_planned_burn_idx: usize,
    duration_s: f64,
    soi_entries: &[SoiEntry],
) -> f64 {
    let next_burn = planned_burns.get(next_planned_burn_idx).map(|b| b.epoch_s).unwrap_or(duration_s);
    let next_soi_entry = next_soi_entry(t_s, soi_entries).map(|e| e.t_s).unwrap_or(duration_s);
    next_burn.min(next_soi_entry).min(duration_s)
}

/// The reference's next SOI entry after `t_s`, if any.
fn next_soi_entry(t_s: f64, soi_entries: &[SoiEntry]) -> Option<SoiEntry> {
    soi_entries.iter().copied().find(|e| e.t_s > t_s)
}

/// Approach-phase test: is the horizon `horizon_end_s` a reference SOI
/// entry (rather than a planned burn or the leg end)? Then the correction
/// is a B-plane correction at that entry's body.
fn approach_soi_entry(t_s: f64, horizon_end_s: f64, soi_entries: &[SoiEntry]) -> Option<SoiEntry> {
    next_soi_entry(t_s, soi_entries).filter(|e| (e.t_s - horizon_end_s).abs() < 1e-6)
}

// ── B-plane targeting (§9.2) ─────────────────────────────────────

/// B-plane parameters of a body-relative hyperbolic state (Kizner 1961;
/// Vallado §12.2 / Battin §9): with `Ŝ` the incoming asymptote direction,
/// the B-plane is the plane through the body's centre perpendicular to
/// `Ŝ`; `B` is where the undeflected asymptote pierces it, decomposed on
/// `T̂ = (Ŝ × N̂)/|Ŝ × N̂|` (here `N̂` = the inertial z-axis) and `R̂ = Ŝ × T̂`.
/// `|B| = b = h/v∞` is the hyperbola's semi-minor axis. `tca_s` is the
/// absolute epoch of periapsis passage from the hyperbolic Kepler equation.
/// These are asymptotic INVARIANTS of the approach hyperbola: nearly linear
/// in the approach state far from the body, which is exactly why they are
/// what approach TCMs target (a periapsis position is violently nonlinear
/// in the same state).
#[derive(Clone, Copy, Debug)]
struct BPlane {
    bt_m: f64,
    br_m: f64,
    tca_s: f64,
    v_inf_mps: f64,
}

fn bplane_of_relative_state(r_rel: &Vector3<f64>, v_rel: &Vector3<f64>, mu: f64, t_s: f64) -> Option<BPlane> {
    let r = r_rel.norm();
    let v2 = v_rel.norm_squared();
    let v_inf2 = v2 - 2.0 * mu / r;
    if r < 1.0 || v_inf2 <= 0.0 || mu <= 0.0 {
        return None; // not hyperbolic relative to this body
    }
    let v_inf = v_inf2.sqrt();
    let h_vec = r_rel.cross(v_rel);
    let h = h_vec.norm();
    if h < 1e-9 {
        return None;
    }
    let h_hat = h_vec / h;
    let e_vec = v_rel.cross(&h_vec) / mu - r_rel / r;
    let e = e_vec.norm();
    if e <= 1.0 + 1e-12 {
        return None;
    }
    let e_hat = e_vec / e;
    let p_hat = h_hat.cross(&e_hat);
    // Incoming asymptote velocity direction: −r̂(ν = −ν∞), cos ν∞ = −1/e.
    let s_hat = e_hat * (1.0 / e) + p_hat * (1.0 - 1.0 / (e * e)).sqrt();
    let b = h / v_inf;
    let b_vec = s_hat.cross(&h_hat) * b;
    let n_hat = Vector3::z();
    let t_raw = s_hat.cross(&n_hat);
    let t_hat = if t_raw.norm() < 1e-9 { Vector3::x() } else { t_raw.normalize() };
    let r_hat = s_hat.cross(&t_hat);
    // Time to periapsis: r = a(1 − e cosh H), a = −μ/v∞²; H < 0 inbound.
    let a_abs = mu / v_inf2;
    let cosh_h = ((1.0 + r / a_abs) / e).max(1.0);
    let h_anom = cosh_h.acosh() * if r_rel.dot(v_rel) < 0.0 { -1.0 } else { 1.0 };
    let mean_anom = e * h_anom.sinh() - h_anom;
    let n = (mu / (a_abs * a_abs * a_abs)).sqrt();
    let t_to_peri = -mean_anom / n;
    Some(BPlane { bt_m: b_vec.dot(&t_hat), br_m: b_vec.dot(&r_hat), tca_s: t_s + t_to_peri, v_inf_mps: v_inf })
}

/// Convergence tolerance for the B-plane residual [m] (each of B·T, B·R,
/// and v∞·ΔTCA).
const TCM_BPLANE_TOL_M: f64 = 1_000.0;

/// §9.2 — refine an approach correction by SHOOTING on B-PLANE residuals
/// (replacing position-targeting for any correction whose
/// horizon is a reference SOI entry). The reference B-plane is evaluated
/// from the reference's own body-relative state at the SOI-entry epoch
/// `t_eval_s`; a candidate Δv is propagated from now to `t_eval_s` under
/// the real force model and its body-relative state converted to the same
/// invariants:
///
/// ```text
/// res(Δv) = [ B·T − B·T_ref,  B·R − B·R_ref,  v∞·(TCA − TCA_ref) ]   [m]
/// J_ij    = ∂res_i/∂Δv_j  (forward differences, 3 propagations)
/// Δv ← Δv − J⁻¹·res            until |res| < tol
/// ```
///
/// The timing residual is scaled by v∞ so all three components are lengths
/// of comparable weight (a standard navigation choice). Same damping and
/// best-so-far bookkeeping as `tcm_shooting_correction`; returns a
/// `ShootingSolve` so the caller applies the same fire-or-skip rule.
#[allow(clippy::too_many_arguments)]
fn tcm_bplane_correction(
    r_m: &Vector3<f64>,
    v_mps: &Vector3<f64>,
    t_s: f64,
    t_eval_s: f64,
    reference: &ReferenceTrajectory,
    body: &PropagatorBody,
    mu_central_m3s2: f64,
    bodies: &[PropagatorBody],
    rtol: f64,
    atol: f64,
    dv_guess: Vector3<f64>,
) -> Option<ShootingSolve> {
    let tof_s = t_eval_s - t_s;
    if tof_s <= 0.0 {
        return None;
    }
    let (body_r, body_v) = (body.state_at)(t_eval_s);
    let (ref_r, ref_v) = reference.state_at(t_eval_s);
    let bp_ref = bplane_of_relative_state(&(ref_r - body_r), &(ref_v - body_v), body.mu_m3s2, t_eval_s)?;
    let residual = |dv: &Vector3<f64>| -> Option<Vector3<f64>> {
        let leg = propagate(*r_m, v_mps + dv, t_s, tof_s, mu_central_m3s2, bodies, tof_s, rtol, atol);
        let end = leg.last()?;
        let bp = bplane_of_relative_state(&(end.r_m - body_r), &(end.v_mps - body_v), body.mu_m3s2, t_eval_s)?;
        Some(Vector3::new(bp.bt_m - bp_ref.bt_m, bp.br_m - bp_ref.br_m, bp_ref.v_inf_mps * (bp.tca_s - bp_ref.tca_s)))
    };
    let debug = std::env::var("TCM_DEBUG").is_ok();
    let mut dv = dv_guess;
    let mut res = residual(&dv)?;
    let (mut best_dv, mut best_miss) = (dv, res.norm());
    // Residual of the CURRENT trajectory (no burn): the predicted delivery
    // miss if nothing is fired — see `ShootingSolve::no_burn_miss_m`.
    let res0 = residual(&Vector3::zeros()).map(|r| r.norm()).unwrap_or(f64::NAN);
    if debug {
        eprintln!(
            "[BPLANE t={:.0}] eval at {:.0} s (+{:.2} d) body '{}' ref B·T={:.3e} B·R={:.3e} v∞={:.1} TCA=+{:.2} d | no-burn |res|={:.3e} | guess |dv|={:.3} res={:?} |res|={:.3e}",
            t_s, t_eval_s, tof_s / 86_400.0, body.name, bp_ref.bt_m, bp_ref.br_m, bp_ref.v_inf_mps,
            (bp_ref.tca_s - t_eval_s) / 86_400.0, res0, dv.norm(), res, res.norm()
        );
    }
    let mut iterations = 0u32;
    for it in 0..TCM_SHOOTING_MAX_ITERS {
        if best_miss < TCM_BPLANE_TOL_M {
            break;
        }
        let h = (1e-3 * dv.norm()).max(1e-2);
        let mut j = nalgebra::Matrix3::<f64>::zeros();
        for k in 0..3 {
            let mut dvk = dv;
            dvk[k] += h;
            let Some(rk) = residual(&dvk) else { return Some(ShootingSolve { dv: best_dv, miss_m: best_miss, converged: false, iterations, method: "BPlane", no_burn_miss_m: res0 }) };
            j.set_column(k, &((rk - res) / h));
        }
        let Some(j_inv) = j.try_inverse() else { break };
        let step = j_inv * res;
        let cap = (10.0 * dv.norm()).max(1.0).min(100.0);
        let step = if step.norm() > cap { step * (cap / step.norm()) } else { step };
        dv -= step;
        let Some(r_new) = residual(&dv) else { break };
        res = r_new;
        iterations = it as u32 + 1;
        if debug {
            eprintln!("[BPLANE t={:.0}] it {it}: step={:.3e} |dv|={:.3} |res|={:.3e} res={:?}", t_s, step.norm(), dv.norm(), res.norm(), res);
        }
        if res.norm() < best_miss {
            best_dv = dv;
            best_miss = res.norm();
        } else {
            break;
        }
    }
    Some(ShootingSolve { dv: best_dv, miss_m: best_miss, converged: best_miss < TCM_BPLANE_TOL_M, iterations, method: "BPlane", no_burn_miss_m: res0 })
}

/// The shooting refinement for a reactive correction: B-plane targeting at
/// the approach body when the horizon is a reference SOI entry
/// (`approach_soi_entry`), fixed-time-of-arrival position targeting
/// otherwise. `disp_now_m` is what the fire-or-skip rule compares against.
#[allow(clippy::too_many_arguments)]
fn refine_reactive_correction(
    state: &SixDofState,
    arrival_t_s: f64,
    horizon_end_s: f64,
    reference: &ReferenceTrajectory,
    mu_central_m3s2: f64,
    bodies: &[PropagatorBody],
    soi_entries: &[SoiEntry],
    rtol: f64,
    atol: f64,
    dv_guess: Vector3<f64>,
) -> ShootingSolve {
    let debug = std::env::var("TCM_DEBUG").is_ok();
    if let Some(entry) = approach_soi_entry(state.t_s, horizon_end_s, soi_entries) {
        if let Some(solve) = tcm_bplane_correction(
            &state.r_m, &state.v_mps, state.t_s, entry.t_s, reference, &bodies[entry.body_idx], mu_central_m3s2,
            bodies, rtol, atol, dv_guess,
        ) {
            if debug {
                eprintln!("[REFINE t={:.0}] B-plane path: converged={} miss={:.3e} |dv|={:.3}", state.t_s, solve.converged, solve.miss_m, solve.dv.norm());
            }
            return solve;
        }
        if debug {
            eprintln!("[REFINE t={:.0}] B-plane path unavailable (reference not hyperbolic at entry?) — position targeting", state.t_s);
        }
    } else if debug {
        eprintln!("[REFINE t={:.0}] position path: horizon_end={:.0} arrival={:.0}", state.t_s, horizon_end_s, arrival_t_s);
    }
    tcm_shooting_correction(&state.r_m, &state.v_mps, state.t_s, arrival_t_s, reference, mu_central_m3s2, bodies, rtol, atol, dv_guess)
}

/// Time to periapsis [s] of a body-relative state (hyperbolic Kepler:
/// `r = a(1 − e cosh H)`, `M = e sinh H − H`, `t_p = −M/n`; negative =
/// periapsis already passed). `None` when the state isn't hyperbolic
/// relative to the body or is degenerate — the same construction
/// `bplane_of_relative_state`'s TCA uses, factored out because the CAPTURE
/// IGNITION GATE needs it on its own: a capture burn must
/// fire at the DISPERSED trajectory's true periapsis passage, not at the
/// plan's stored clock epoch — firing a velocity-matching burn wherever
/// the ship happens to be at the nominal epoch captures into whatever
/// orbit THAT radius implies (measured live: ignition at 13,998 km against
/// a planned 6,681 km periapsis bought a barely-bound a = 261,617 km
/// orbit instead of the planned 66,807 km ellipse).
fn time_to_periapsis_rel_s(r_rel: &Vector3<f64>, v_rel: &Vector3<f64>, mu: f64) -> Option<f64> {
    let r = r_rel.norm();
    let v2 = v_rel.norm_squared();
    if r < 1.0 || mu <= 0.0 {
        return None;
    }
    let v_inf2 = v2 - 2.0 * mu / r;
    if v_inf2 <= 0.0 {
        return None; // bound relative orbit — not the hyperbolic-approach case
    }
    let e_vec = v_rel.cross(&r_rel.cross(v_rel)) / mu - r_rel / r;
    let e = e_vec.norm();
    if e <= 1.0 + 1e-12 {
        return None;
    }
    let a_abs = mu / v_inf2;
    let cosh_h = ((1.0 + r / a_abs) / e).max(1.0);
    let h_anom = cosh_h.acosh() * if r_rel.dot(v_rel) < 0.0 { -1.0 } else { 1.0 };
    let mean_anom = e * h_anom.sinh() - h_anom;
    let n = (mu / (a_abs * a_abs * a_abs)).sqrt();
    Some(-mean_anom / n)
}

/// Review D5 — solve a planned CAPTURE burn's ΔV fresh from
/// the vehicle's real (dispersed) state at trigger time: the velocity-
/// matching solve 13n's own doc note said a capture burn needs, as opposed
/// to `tcm_lambert_correction`'s position-shaping (the wrong tool for a
/// burn whose intent is a specific relative-velocity state, MANUAL.md
/// §9.2). Construction: scale the body-relative speed to the local
/// circular speed along the CURRENT relative-velocity direction —
/// `Δv = v̂_rel·√(μ/|r_rel|) − v_rel` — provably always bound (e < 1)
/// regardless of how far the crossing is from periapsis, since specific
/// energy depends only on speed and radius (the same construction
/// `ArrivalCapture::dv_capture_ms`'s pricing and `propagate_captured_
/// orbit`'s real-state branch already use; regression-tested in
/// `orbital_math::kepler`). As a velocity difference the result needs no
/// body-relative→inertial frame conversion. `None` when the geometry is
/// degenerate (inside ~1 km of the body's center, or essentially zero
/// relative speed) — the caller falls back to the stored nominal vector.
/// `capture_eccentricity` (the executive half of Phase 14f):
/// the target speed is the CONFIGURED capture orbit's periapsis speed
/// `√(μ(1+e)/r)`, not the circular `√(μ/r)` — pricing (14f) and execution
/// must agree, and the difference is not small: at a Mars periapsis with
/// e = 0.9 the circular target demands ~2.0 km/s where the priced eccentric
/// capture needs ~740 m/s (measured live: the circular-targeting executive
/// burned 472 kg against a 740 m/s job). Still bound for any e < 1:
/// specific energy `μ(1+e)/2r − μ/r = −μ(1−e)/2r < 0` regardless of
/// direction.
fn solve_capture_burn_dv(
    r_m: &Vector3<f64>,
    v_mps: &Vector3<f64>,
    body_r_m: &Vector3<f64>,
    body_v_mps: &Vector3<f64>,
    mu_body_m3s2: f64,
    capture_eccentricity: f64,
) -> Option<Vector3<f64>> {
    let r_rel = r_m - body_r_m;
    let v_rel = v_mps - body_v_mps;
    let r_norm = r_rel.norm();
    let v_norm = v_rel.norm();
    if r_norm < 1.0e3 || v_norm < 1.0e-9 || mu_body_m3s2 <= 0.0 {
        return None;
    }
    let v_target = (mu_body_m3s2 * (1.0 + capture_eccentricity.clamp(0.0, 0.999)) / r_norm).sqrt();
    Some(v_rel * (v_target / v_norm) - v_rel)
}

/// Item 13p — solve a TCM correction from the CURRENT state,
/// returning `(thrust_dir_inertial, dv_target_mps)`. Shared by the initial
/// `Coast` trigger AND, critically, by continuous re-targeting while
/// `Slewing` (see [`run_cruise_leg`]'s call sites): `Slewing` can take real
/// time (wheel-torque-limited, entirely independent of main-engine thrust —
/// confirmed live: giving the main engine 1000x more thrust did NOT shorten
/// convergence time, it made overall dispersion WORSE, because a stronger
/// engine just fires a now-even-more-stale correction sooner), during which
/// the vehicle's true state keeps evolving. A correction solved once at
/// trigger time and carried verbatim into `Burning` is solving for where
/// the vehicle WAS, not where it IS by the time it can actually fire —
/// re-solving every tick while `Slewing` keeps the eventually-fired
/// correction matched to the vehicle's real state at ignition.
///
/// Targets a NEARER epoch than the fixed leg-end whenever that's more than
/// `TCM_HORIZON_PERIOD_FRACTION` of a local orbital period away (item 13p's
/// OTHER real bug, found first: a single-revolution Lambert solve forced to
/// satisfy a multi-revolution time-of-flight for a small residual position
/// error is genuinely ill-posed, not just imprecise — confirmed by direct
/// reproduction, `cruise::tests::diag_13p_many_corrections_over_a_long_leg`:
/// a 13 km position error produced a demanded 362 KM/S "correction" before
/// this fix). Applies item #2's magnitude cap
/// (`TCM_DV_CAP_FRACTION_OF_AVAILABLE`) against the real Tsiolkovsky-
/// available ΔV from current mass. Returns `None` when no correction is
/// currently warranted (Lambert solve failed/degenerate, or the resulting
/// capped ΔV is negligible) — the caller treats this as "close enough now"
/// during re-targeting, or "nothing to do yet" at the initial trigger.
fn solve_tcm_correction(
    state: &SixDofState,
    reference: &ReferenceTrajectory,
    duration_s: f64,
    mu_central_m3s2: f64,
    dry_mass_kg: f64,
    tcm: TcmConfig,
) -> Option<(Vector3<f64>, f64, f64)> {
    let remaining_tof_s = duration_s - state.t_s;
    let sma_m = semi_major_axis_m(&state.r_m, &state.v_mps, mu_central_m3s2);
    let horizon_s = if sma_m > 0.0 {
        TCM_HORIZON_PERIOD_FRACTION * orbital_period_s(sma_m, mu_central_m3s2)
    } else {
        remaining_tof_s
    };
    let arrival_t_s = (state.t_s + horizon_s).min(duration_s);
    let dv = tcm_lambert_correction(&state.r_m, &state.v_mps, state.t_s, arrival_t_s, reference, mu_central_m3s2, true)?;
    let dv_norm = dv.norm();
    if dv_norm <= 1e-9 {
        return None;
    }
    let dv_available_mps = if state.mass_kg > dry_mass_kg + 1e-9 {
        tcm.isp_s * orbital_models::constants::G0 * (state.mass_kg / dry_mass_kg).ln()
    } else {
        0.0
    };
    let dv_cap_mps = TCM_DV_CAP_FRACTION_OF_AVAILABLE * dv_available_mps;
    let dv_target_mps = dv_norm.min(dv_cap_mps);
    if std::env::var("TCM_DEBUG").is_ok() {
        eprintln!(
            "[TCM_DEBUG t={:.1}] r={:?} v={:?} arrival_t_s={:.1} remaining_tof_s={:.1} \
             dv_norm={:.3e} dv_available_mps={:.3e} dv_cap_mps={:.3e} dv_target_mps={:.3e}",
            state.t_s, state.r_m, state.v_mps, arrival_t_s, remaining_tof_s,
            dv_norm, dv_available_mps, dv_cap_mps, dv_target_mps,
        );
    }
    if dv_target_mps <= 1e-6 {
        return None;
    }
    // The aim epoch is returned alongside so the solve can be REPORTED
    // — previously internal-only.
    Some((dv / dv_norm, dv_target_mps, arrival_t_s))
}

/// Item #4 — decide which actuator delivers a TCM correction. Main engine
/// + slew is the DEFAULT (real missions don't give up
/// pointing lightly). RCS is only even CONSIDERED when the correction is
/// small (`dv_target_mps <= TCM_RCS_DV_THRESHOLD_MPS`) or the active mode
/// marks pointing non-negotiable (`pointing_locked`) — and even then, only
/// used if the placed RCS thrusters can genuinely deliver real net thrust
/// along the needed inertial direction FROM THE CURRENT ATTITUDE (no
/// slew): each thruster whose body-frame direction has positive alignment
/// with the target direction (expressed in body frame via the current
/// attitude) contributes its full thrust to a simple achievable-thrust
/// estimate — not an optimal allocation, just enough to decide feasibility
/// before [`run_cruise_leg`]'s `RcsCorrecting` phase actually fires them.
fn choose_tcm_actuator(
    thrust_dir_inertial: Vector3<f64>,
    dv_target_mps: f64,
    q: &Vector4<f64>,
    rcs_thrusters: &[Thruster],
    pointing_locked: bool,
) -> TcmActuator {
    let rcs_eligible = dv_target_mps <= TCM_RCS_DV_THRESHOLD_MPS || pointing_locked;
    if rcs_eligible && !rcs_thrusters.is_empty() {
        let dir_body = inertial_to_body(q, &thrust_dir_inertial);
        let aligned_thrust_n: f64 = rcs_thrusters
            .iter()
            .map(|t| t.dir.dot(&dir_body) * t.thrust_n)
            .filter(|contribution| *contribution > 0.0)
            .sum();
        if aligned_thrust_n > TCM_RCS_MIN_ALIGNED_THRUST_N {
            return TcmActuator::Rcs;
        }
    }
    TcmActuator::MainEngine
}

/// Fly a single coast leg: `initial` seeds the state at `t_s = 0`,
/// `reference` is the Layer-1 trajectory to track (report dispersion
/// against, not correct against — see module doc comment), `pointing_mode`
/// is held constant for the whole leg, `bodies`/`mu_central_m3s2` are the
/// SOI-candidate/third-body force model (pass `&[]` for a pure heliocentric
/// coast, matching whatever force model `reference` was itself propagated
/// under — using a DIFFERENT force model here than the reference used would
/// make the reported dispersion partly an artifact of that mismatch rather
/// than a real tracking-quality signal).
///
/// `sun_pos_at` supplies the Sun's position (same frame as `r_m`) as a
/// function of tick time — a closure rather than a fixed vector so this
/// works both for a heliocentric leg (Sun at the origin, `|_| Vector3::
/// zeros()`) and a future body-centric leg (real ephemeris lookup). Only
/// consulted when `pointing_mode` is `SunPointing` (or, when `commander` is
/// set, by any rule targeting `PointingTargetConfig::Sun`).
///
/// `commander` — when `Some`, OVERRIDES `pointing_mode`
/// for ticks its schedule/safe-mode actually cover, computing `q_cmd` via
/// the priority-ordered multi-rule solve instead of the fixed single mode.
/// When `None`, behavior is EXACTLY the pre-ask-#7 fixed-`pointing_mode`
/// path — this parameter is purely additive.
///
/// `on_row` is called once per tick with the row just computed; returning
/// `false` stops the loop early (mirrors the cancellation convention
/// already used by `simulate::run_streaming`/the `/api/optimize` job loop)
/// — whatever rows were produced up to that point are still returned.
///
/// `momentum_law` (found, fixing a real bug — see
/// `sim_engine::control::MomentumManagementLaw`'s own doc comment) governs
/// wheel desaturation: `MomentumManagementLaw::None` is the safe default
/// when no RCS is configured (`rcs_thrusters` empty) — wheels simply grow
/// to their physical `max_speed` clamp with no active unloading, matching
/// what a hardware config with wheels but no RCS should do. `ThresholdRcs`
/// actively unloads via RCS once any wheel crosses the desaturation
/// threshold, with RCS cancelling the resulting body reaction so attitude
/// tracking is undisturbed by the dump event.
///
/// `tcm` (hardened, the design notes Phase 13m
/// items) — `None` preserves the original coast-only
/// behavior (legacy call sites, and every demo binary
/// still). `Some(TcmConfig)` enables the threshold-triggered trajectory-
/// correction executive — see [`TcmPhase`] for the state machine and the
/// `TCM_*` constants above it for the guard rails: a new correction is
/// only attempted when a fixed cooldown AND a real dispersion-decreasing
/// trend both hold since the previous correction ended (item #1), when
/// enough leg time remains and the solved ΔV is capped against what
/// remaining propellant can actually deliver (item #2), and via RCS
/// (no slew) rather than the main engine when the correction is small or
/// the active mode locks pointing (item #4). Propellant exhaustion
/// (item #3) aborts any in-progress correction back to `Coast` and
/// degrades `momentum_law`/`control_mode` to their no-RCS equivalents for
/// that tick, rather than silently continuing to "fire" thrusters with
/// nothing left in the tank.
///
/// **A note on the shared propellant pool** — `spacecraft.propellant_mass_kg`
/// is one budget drawn from by RCS (attitude control/desaturation) AND TCM
/// (main-engine burns and, since RCS-only corrections) alike,
/// per this loop's existing model; a real vehicle with physically separate
/// RCS/main-engine tanks would need two independent budgets, which is out
/// of scope for this fix (unchanged behavior from before item #3).
///
/// `rtol`/`atol` (added, Phase 01/02 consistency ask) — the
/// integrator tolerances passed straight to `step_tick_with_burn`. Before
/// this, they were hardcoded to `1e-10`/`1e-12` here regardless of what
/// `simulation.rtol`/`atol` a mission actually configured in Phase 01 —
/// `run_cruise_streaming`/`run_cruise_mc_streaming` now pass `cfg.
/// simulation.rtol`/`atol` through, so Phase 02 integrates at the SAME
/// tolerance Phase 01's own trajectory design used, unless the caller
/// deliberately chooses otherwise.
///
/// No `rcs_isp_s` parameter here (
/// superseded) — `rcs_thrusters`
/// now carries each thruster's own real Isp, so `sim_engine::control::
/// allocate` prices propellant per-thruster instead of against one
/// aggregate scalar. Callers still get a real (possibly non-uniform) Isp
/// from `rcs_from_hardware`; they just no longer need to thread it through
/// separately.
///
/// `thrust_offset_body_m` — the main
/// engine's real mount-arm from the CoM, body frame [m], fed straight into
/// every `Burning`-phase `BurnConfig` below so `step_tick_with_burn`'s
/// already-real, already-tested `tau_misalign = thrust_offset_body_m.cross(
/// thrust_force_body)` (`propagator6dof.rs`) actually receives a nonzero
/// arm instead of the `Vector3::zeros()` every call site silently passed
/// before this — i.e. an off-CoM main engine now genuinely produces
/// disturbance torque during a burn, as it must. Callers pass
/// `Vector3::zeros()` for a demo/test with no real vehicle geometry behind
/// it (harmless: only consulted while `Burning`, and a `None` `tcm` never
/// reaches that phase at all) — `run_cruise_streaming`/
/// `run_cruise_mc_streaming` pass the real value, computed once (or once
/// per MC-dispersed vehicle) from the true derived CoM.
///
/// `planned_burns` (Phase 13n) -- scheduled burn EVENTS (a DSM,
/// an arrival/capture burn, a departure burn under GNC test), MUST be
/// sorted ascending by `epoch_s` (`check_config` enforces this on the
/// config-level list this is built from). Fires via the SAME `TcmPhase::
/// Slewing`/`Burning` machinery the reactive `tcm` executive uses, always
/// through the main engine (never `RcsCorrecting`) -- see
/// `CruiseSeedConfig::planned_burns`'s own doc comment for the exact
/// contract, including the documented "not re-solved fresh" limitation.
/// Takes priority over a reactive `tcm` trigger on any tick both would
/// fire (a scheduled maneuver is not optional the way a dispersion
/// correction is). Empty slice preserves the exact pre-
/// behavior for every existing call site.
#[allow(clippy::too_many_arguments)]
pub fn run_cruise_leg(
    initial: SixDofState,
    reference: &ReferenceTrajectory,
    bodies: &[PropagatorBody],
    mu_central_m3s2: f64,
    sc: &SpacecraftProperties,
    wheel_cluster: &ReactionWheelCluster,
    rcs_thrusters: &[Thruster],
    // Three-layer attitude-control registry (`attitude_tuning.
    // rs`, MANUAL.md §10.5): replaces the single `gains: PdGains` every
    // mode used to fly on. The loop asks it for the law to run each tick
    // from the ACTIVE control mode + activity + current mass.
    controls: &AttitudeControlSet,
    control_mode: ControlMode,
    momentum_law: MomentumManagementLaw,
    pointing_mode: CruisePointingMode,
    commander: Option<&GncCommander>,
    sun_pos_at: &dyn Fn(f64) -> Vector3<f64>,
    tick_s: f64,
    duration_s: f64,
    propellant_mass_kg: f64,
    tcm: Option<TcmConfig>,
    rtol: f64,
    atol: f64,
    thrust_offset_body_m: Vector3<f64>,
    planned_burns: &[crate::config::PlannedBurnConfig],
    // Review D5: `(body_name, t_s) -> (r_m, v_mps, mu_m3s2)`
    // lookup for a planned CAPTURE burn's fresh velocity-matching re-solve
    // (`PlannedBurnConfig::capture_body`) — a closure so this loop stays
    // ANISE-free, same convention as `sun_pos_at`. Callers with no capture
    // burns pass `&|_, _| None` (the stored-ΔV fallback then applies).
    capture_state_at: &dyn Fn(&str, f64) -> Option<(Vector3<f64>, Vector3<f64>, f64, f64)>,
    on_row: &mut dyn FnMut(&CruiseTickRow) -> bool,
) -> Vec<CruiseTickRow> {
    // Layer-3 scheduler state: the (mode, activity) the running controller
    // was built for, the mass it was derived at, and the controller itself
    // (carries PID integral / phase-plane latches between ticks; rebuilt —
    // state reset — whenever the law changes).
    let mut controller_key: Option<(ControlMode, Activity)> = None;
    let mut controller_mass_kg = initial.mass_kg;
    let mut controller = AttitudeController::new(controls.law_for(control_mode, Activity::Hold, initial.mass_kg));
    // Mode-scheduled tick (MANUAL.md §10.5.1): the tick
    // used while a main-engine burn is FIRING — `controls.thruster_tick_s`,
    // resolved by the caller from `cruise_seed.burn_tick_s`. Everything
    // per-tick below (control, allocation, propagation, propellant/wheel
    // integration) uses `tick_eff`, re-evaluated each iteration from the
    // CURRENT phase, so the loop transparently runs finer during burns.
    let burn_tick_s = controls.thruster_tick_s.min(tick_s).max(1e-6);
    let mut state = initial;
    let n_ticks = (duration_s / tick_s).ceil().max(1.0) as usize;
    // Structural (non-propellant) mass -- see `TcmConfig`'s doc comment on
    // the shared-pool model. `initial.mass_kg` is the caller's total wet
    // mass (`spacecraft.mass_kg = dry_mass_kg + propellant_mass_kg`, a
    // `check_config`-enforced invariant), so this recovers dry mass without
    // a separate parameter.
    let dry_mass_kg = initial.mass_kg - propellant_mass_kg;

    let mut rows: Vec<CruiseTickRow> = Vec::with_capacity(n_ticks);
    let mut rcs_propellant_kg_cum = 0.0_f64;
    let mut tcm_propellant_kg_cum = 0.0_f64;
    // cumulative REAL delivered TCM/burn ΔV
    // [m/s] — Tsiolkovsky per burning tick, projected RCS impulse per
    // RcsCorrecting tick. Accumulated after the row is built (same
    // convention as `tcm_propellant_kg_cum`'s burning-tick half), so a
    // row's value covers everything through the PREVIOUS tick.
    let mut tcm_dv_mps_cum = 0.0_f64;
    let mut tcm_phase = TcmPhase::Coast;
    // Item #1: `(t_end, cooldown_s)` from the most recently completed
    // correction -- `None` until the first one ends. The fixed-cooldown
    // timer reads this before allowing a new reactive trigger (a trend
    // guard was tried and reverted -- see this constant's own doc comment
    // on `TCM_COOLDOWN_FRACTION_OF_REMAINING_TOF`).
    let mut post_burn: Option<(f64, f64)> = None;
    // Phase 13n: index of the next not-yet-fired entry in `planned_burns`.
    let mut next_planned_burn_idx = 0usize;
    // reference SOI-entry epochs — a reactive correction never
    // targets past the next one (see `reference_soi_entry_epochs`).
    let ref_soi_entries = reference_soi_entry_epochs(reference, bodies);
    // Burn-attitude abort bookkeeping: consecutive burn time [s] spent
    // beyond `BURN_ABORT_POINTING_DEG`.
    let mut burn_misaligned_s = 0.0_f64;

    while state.t_s < duration_s - 1e-9 {
        let sun_pos = sun_pos_at(state.t_s);

        // Item #3: propellant remaining BEFORE this tick's own consumption
        // -- gates both new triggers and continuation of an in-progress
        // correction. Clamped defensively even though nothing below should
        // ever push the cumulative totals past `propellant_mass_kg`.
        let propellant_remaining_kg = (propellant_mass_kg - rcs_propellant_kg_cum - tcm_propellant_kg_cum).max(0.0);
        let propellant_exhausted = propellant_remaining_kg <= PROPELLANT_EXHAUSTED_EPSILON_KG;

        // Cannot continue any active maneuver without propellant -- abort to
        // Coast rather than silently "executing" a burn/RCS correction with
        // nothing left to fire (item #3).
        if propellant_exhausted && !matches!(tcm_phase, TcmPhase::Coast) {
            tcm_phase = TcmPhase::Coast;
        }

        // Phase 13n: planned-burn trigger -- takes priority over the
        // reactive TCM trigger below (a scheduled DSM/capture/departure
        // burn is not optional the way a dispersion correction is). Always
        // routes through the main engine (Slewing/Burning), never
        // RcsCorrecting -- see CruiseSeedConfig::planned_burns's own doc
        // comment.
        //
        // Nuance 2: when `target_epoch_s` is set, the fired
        // ΔV is re-solved FRESH from the vehicle's real current state via
        // the SAME `tcm_lambert_correction` mechanism the reactive
        // executive uses below -- targeting THIS burn's own known shaping
        // intent (`target_epoch_s`, `reference`'s own recorded position
        // there) instead of a reactive-trigger horizon. Falls back to the
        // stored `dv_inertial_mps` verbatim if the fresh solve fails
        // (degenerate Lambert geometry) or `target_epoch_s` is unset (the
        // only option for a velocity-shaping arrival/departure burn, which
        // has no position target to re-solve toward).
        // Lead-time scheduling (ask): within
        // PLANNED_BURN_MAX_LEAD_S of the epoch, solve the burn's ΔV
        // (`planned_burn_dv` — capture/DSM fresh solve or stored vector)
        // to get its DIRECTION, compute the kinematic slew time from the
        // current body +X to that direction at the §10.4 cruise rate plus
        // a settle margin, and enter `Slewing` once `t >= epoch − lead`.
        // The burn then HOLDS the (inertially fixed) burn attitude and
        // ignites AT the epoch (gate below), instead of late by the slew
        // duration. A zero-ΔV plan is a no-op (skipped, never slewed for).
        // External-stage burn (`PlannedBurnConfig::external_
        // stage`): delivered by a launcher upper stage, not the spacecraft
        // — an impulsive ΔV at the epoch, no slew, no propellant draw, no
        // TcmConfig needed. Tagged on this tick's row so the per-burn
        // report sees it as ignited+completed here.
        // the reactive solve (if any) this tick ran —
        // reset per tick, reported on the row.
        let mut tick_solve: Option<TcmSolveInfo> = None;
        let mut external_stage_fired: Option<usize> = None;
        if let (TcmPhase::Coast, Some(burn)) = (&tcm_phase, planned_burns.get(next_planned_burn_idx)) {
            if burn.external_stage && state.t_s >= burn.epoch_s {
                let dv = Vector3::new(burn.dv_inertial_mps[0], burn.dv_inertial_mps[1], burn.dv_inertial_mps[2]);
                state.v_mps += dv;
                external_stage_fired = Some(next_planned_burn_idx);
                next_planned_burn_idx += 1;
            }
        }

        if let (TcmPhase::Coast, Some(_tcm)) = (&tcm_phase, tcm) {
            if !propellant_exhausted {
                if let Some(burn) = planned_burns.get(next_planned_burn_idx).filter(|b| !b.external_stage) {
                    if state.t_s >= burn.epoch_s - PLANNED_BURN_MAX_LEAD_S {
                        let dv = planned_burn_dv(burn, &state, reference, mu_central_m3s2, capture_state_at, bodies, rtol, atol);
                        let dv_norm = dv.norm();
                        if dv_norm <= 1e-9 {
                            if state.t_s >= burn.epoch_s {
                                next_planned_burn_idx += 1;
                            }
                        } else {
                            let dir = dv / dv_norm;
                            // Slew angle = the FULL attitude error to the
                            // burn attitude (BurnAttitude fixes all three
                            // axes, not just the +X boresight): the
                            // boresight-only angle under-estimated a
                            // 165° reorientation as 40° and left ignition
                            // ~70 s late.
                            let q_burn = desired_quaternion_cruise(
                                CruisePointingMode::BurnAttitude { thrust_dir_inertial: dir }, &state.r_m, &sun_pos,
                            );
                            let theta = pointing_error_deg(&state.q, &q_burn).to_radians();
                            // Lead = kinematic slew at the profiled cruise
                            // rate + the wheel loop's REAL 2% settling time
                            // (4/(ζω_n), §10.5.1) — never less than the
                            // fixed margin. a fixed 300 s
                            // margin alone left a Mercury injection burn
                            // igniting 71 s late (job 4), since the
                            // settle after a 40° slew takes ~300 s by
                            // itself on that vehicle.
                            let settle_s = controls
                                .settling_time_s(control_mode)
                                .unwrap_or(0.0)
                                .max(PLANNED_BURN_SLEW_MARGIN_S);
                            let lead_s = theta / SLEW_PROFILE_CRUISE_RATE_RADPS + settle_s;
                            if state.t_s >= burn.epoch_s - lead_s {
                                tcm_phase = TcmPhase::Slewing {
                                    thrust_dir_inertial: dir,
                                    dv_target_mps: dv_norm,
                                    is_planned: true,
                                    planned: Some(PlannedSlew {
                                        idx: next_planned_burn_idx,
                                        ignition_epoch_s: burn.epoch_s,
                                        deadline_s: burn.epoch_s + PLANNED_BURN_TIMEOUT_S,
                                    }),
                                };
                                next_planned_burn_idx += 1;
                            }
                        }
                    }
                }
            }
        }

        // TCM trigger check -- only while quietly coasting, with propellant
        // remaining, past the item #1 cooldown guard, and with enough leg
        // time left (item #2's TOF floor). See TCM_COOLDOWN_FRACTION_OF_
        // REMAINING_TOF's own doc comment for why this is JUST the fixed
        // cooldown now, not also a permanent dr_now-vs-dr_at_completion
        // trend requirement -- that combination is what caused a real
        // regression (a single correction, then permanent lockout for the
        // rest of the mission, found live-testing).
        if let (TcmPhase::Coast, Some(tcm)) = (&tcm_phase, tcm) {
            if !propellant_exhausted {
                // never START a reactive correction while the
                // next PLANNED burn's evaluation window is open — real ops
                // don't fire a TCM minutes before a scheduled injection
                // (the scheduled burn re-solves at ignition and absorbs
                // the accumulated dispersion itself, 13n). Found live: a
                // reactive trigger firing inside a parking orbit ahead of
                // the departure burn ran PAST the planned epoch (a
                // non-Coast phase blocks the planned trigger), delaying
                // the injection and spending the correction budget on a
                // maneuver the injection re-solve would have absorbed.
                let planned_burn_imminent = planned_burns
                    .get(next_planned_burn_idx)
                    .is_some_and(|b| state.t_s >= b.epoch_s - PLANNED_BURN_MAX_LEAD_S);
                // never trigger a reactive correction while
                // INSIDE a registered SOI-capture body's sphere of
                // influence. `tcm_lambert_correction` is a two-body solve
                // about `mu_central_m3s2` (the Sun); inside a planet's SOI
                // that body's gravity dominates and the solve is simply
                // wrong — found live on an escape leg, where a 3 m/s
                // injection residual was "corrected" into a 200 m/s error
                // and half the tank. Real ops fly TCM-1 days after
                // injection, once clear of the planet, for the same reason.
                let inside_registered_soi = bodies.iter().any(|b| {
                    b.soi_radius_m
                        .is_some_and(|r_soi| (state.r_m - (b.state_at)(state.t_s).0).norm() < r_soi)
                });
                let disp_now_m = dispersion(&state.r_m, &state.v_mps, state.t_s, reference).dr_m.norm();
                let cooldown_ok = !planned_burn_imminent
                    && !inside_registered_soi
                    && match post_burn {
                        Some((t_end, cooldown_s)) => (state.t_s - t_end) >= cooldown_s,
                        None => true,
                    };
                // a reactive correction may never target past
                // the NEXT PLANNED BURN — the reference beyond a capture
                // burn is the captured orbit, and a heliocentric two-body
                // Lambert aimed at a point in orbit around the target body
                // returns an absurd "correction" (found live: 0.37 m/s of
                // dispersion → a 237 m/s burn, half the tank, as soon as
                // the horizon reached past the capture epoch). The planned
                // burn re-solves arrival itself. Nor past the reference's
                // next SOI ENTRY (same day, later): aiming at the capture
                // point itself — a hyperbolic periapsis deep in the target's
                // well — made the approach solves ill-conditioned.
                let tcm_horizon_end_s =
                    reactive_tcm_horizon_end_s(state.t_s, planned_burns, next_planned_burn_idx, duration_s, &ref_soi_entries);
                let remaining_tof_s = tcm_horizon_end_s - state.t_s;
                let min_remaining_tof_s = TCM_MIN_REMAINING_TOF_FRACTION * duration_s;
                if std::env::var("TCM_DEBUG").is_ok() && disp_now_m > tcm.dr_threshold_m {
                    eprintln!(
                        "[TCM_DEBUG t={:.1}] disp_now_m={:.3e} threshold={:.3e} cooldown_ok={} remaining_tof_s={:.1} min_remaining_tof_s={:.1}",
                        state.t_s, disp_now_m, tcm.dr_threshold_m, cooldown_ok, remaining_tof_s, min_remaining_tof_s,
                    );
                }
                if cooldown_ok && disp_now_m > tcm.dr_threshold_m && remaining_tof_s > min_remaining_tof_s {
                    if let Some((lambert_dir, lambert_mag, lambert_aim_s)) =
                        solve_tcm_correction(&state, reference, tcm_horizon_end_s, mu_central_m3s2, dry_mass_kg, tcm)
                    {
                        // Approach mode: solve the B-plane
                        // correction AT TRIGGER, so the slew targets the
                        // right attitude from the start — the Lambert and
                        // B-plane directions differ by construction on an
                        // approach, and slewing to Lambert's only to
                        // redirect at ignition wastes the whole slew.
                        let (thrust_dir_inertial, dv_target_mps) =
                            if approach_soi_entry(state.t_s, tcm_horizon_end_s, &ref_soi_entries).is_some() {
                                let solve = refine_reactive_correction(
                                    &state, tcm_horizon_end_s, tcm_horizon_end_s, reference, mu_central_m3s2,
                                    bodies, &ref_soi_entries, rtol, atol, lambert_dir * lambert_mag,
                                );
                                let n = solve.dv.norm();
                                // on an
                                // approach the trigger metric is the
                                // PREDICTED DELIVERY MISS (the no-burn
                                // B-plane residual), not the instantaneous
                                // dr — a correctly-corrected trajectory
                                // legitimately keeps a large intermediate
                                // dr, and re-firing on it forever was the
                                // 43-burn cadence measured live.
                                let delivery_fine = solve.no_burn_miss_m.is_finite()
                                    && solve.no_burn_miss_m < tcm.dr_threshold_m;
                                let declined = delivery_fine || !shooting_solve_worth_firing(&solve, disp_now_m) || n <= 1e-9;
                                tick_solve = Some(TcmSolveInfo {
                                    context: "Trigger",
                                    targeting: solve.method,
                                    aim_epoch_s: tcm_horizon_end_s,
                                    lambert_dv_mps: Some(lambert_mag),
                                    solved_dv_mps: n,
                                    iterations: Some(solve.iterations),
                                    converged: Some(solve.converged),
                                    miss_m: Some(solve.miss_m),
                                    no_burn_miss_m: solve.no_burn_miss_m.is_finite().then_some(solve.no_burn_miss_m),
                                    // Provisional — the actuator match below
                                    // overwrites the fire decision when RCS
                                    // is chosen.
                                    decision: if delivery_fine {
                                        "DeclinedDeliveryFine"
                                    } else if declined {
                                        "DeclinedNotWorth"
                                    } else {
                                        "FireMainEngine"
                                    },
                                });
                                if declined {
                                    (lambert_dir, 0.0) // below: a zero mag never enters Slewing
                                } else {
                                    (solve.dv / n, n)
                                }
                            } else {
                                tick_solve = Some(TcmSolveInfo {
                                    context: "Trigger",
                                    targeting: "Lambert",
                                    aim_epoch_s: lambert_aim_s,
                                    lambert_dv_mps: Some(lambert_mag),
                                    solved_dv_mps: lambert_mag,
                                    iterations: None,
                                    converged: None,
                                    miss_m: None,
                                    no_burn_miss_m: None,
                                    decision: "FireMainEngine",
                                });
                                (lambert_dir, lambert_mag)
                            };
                        let pointing_locked_now = commander.map(|c| c.pointing_locked_at(state.t_s)).unwrap_or(false);
                        if dv_target_mps <= 1e-9 {
                            // Approach solve declined to fire (not worth it).
                        } else {
                        tcm_phase = match choose_tcm_actuator(thrust_dir_inertial, dv_target_mps, &state.q, rcs_thrusters, pointing_locked_now) {
                            TcmActuator::MainEngine => TcmPhase::Slewing { thrust_dir_inertial, dv_target_mps, is_planned: false, planned: None },
                            TcmActuator::Rcs => {
                                // an RCS correction has no
                                // ignition step to refine at, so refine the
                                // Lambert guess by shooting HERE (§9.2) —
                                // once per trigger, same cost argument as
                                // the main-engine path. Magnitude capped by
                                // the same propellant guard as the trigger.
                                let sma_m = semi_major_axis_m(&state.r_m, &state.v_mps, mu_central_m3s2);
                                let horizon_s = if sma_m > 0.0 {
                                    TCM_HORIZON_PERIOD_FRACTION * orbital_period_s(sma_m, mu_central_m3s2)
                                } else {
                                    tcm_horizon_end_s - state.t_s
                                };
                                let arrival_t_s = (state.t_s + horizon_s).min(tcm_horizon_end_s);
                                let solve = refine_reactive_correction(
                                    &state, arrival_t_s, tcm_horizon_end_s, reference, mu_central_m3s2, bodies,
                                    &ref_soi_entries, rtol, atol, thrust_dir_inertial * dv_target_mps,
                                );
                                let n = solve.dv.norm();
                                let dv_cap_mps = if state.mass_kg > dry_mass_kg + 1e-9 {
                                    TCM_DV_CAP_FRACTION_OF_AVAILABLE * tcm.isp_s * orbital_models::constants::G0 * (state.mass_kg / dry_mass_kg).ln()
                                } else {
                                    dv_target_mps
                                };
                                let rcs_worth = shooting_solve_worth_firing(&solve, disp_now_m);
                                // Overwrites the provisional trigger entry —
                                // the RCS path refines HERE (no ignition
                                // step), so this IS the governing solve.
                                tick_solve = Some(TcmSolveInfo {
                                    context: "RcsTrigger",
                                    targeting: solve.method,
                                    aim_epoch_s: arrival_t_s,
                                    lambert_dv_mps: Some(lambert_mag),
                                    solved_dv_mps: solve.dv.norm(),
                                    iterations: Some(solve.iterations),
                                    converged: Some(solve.converged),
                                    miss_m: Some(solve.miss_m),
                                    no_burn_miss_m: solve.no_burn_miss_m.is_finite().then_some(solve.no_burn_miss_m),
                                    decision: if rcs_worth { "FireRcs" } else { "DeclinedNotWorth" },
                                });
                                if !rcs_worth {
                                    // Non-converged and not even halving the
                                    // dispersion: a finding, not a maneuver.
                                    eprintln!(
                                        "Warning: reactive RCS correction at t={:.0} s skipped — shooting solve did not converge (miss {:.3e} m vs dispersion {:.3e} m)",
                                        state.t_s, solve.miss_m, disp_now_m
                                    );
                                    TcmPhase::Coast
                                } else if n > 1e-9 {
                                    TcmPhase::RcsCorrecting { thrust_dir_inertial: solve.dv / n, dv_remaining_mps: n.min(dv_cap_mps) }
                                } else {
                                    TcmPhase::RcsCorrecting { thrust_dir_inertial, dv_remaining_mps: dv_target_mps }
                                }
                            }
                        };
                        }
                    }
                }
            }
        }

        // Item 13p: continuously re-target while Slewing -- the original
        // trigger-time solve above can go stale before the reorientation
        // (wheel-torque-limited, can take real time) actually completes.
        // See `solve_tcm_correction`'s own doc comment. Aborts cleanly to
        // `Coast` (rather than firing a stale command) if the correction is
        // no longer solvable/warranted. (Propellant exhaustion is already
        // handled above -- reaching here with `Slewing` still active means
        // propellant remains.) Gated to `is_planned: false` -- a planned
        // burn's fixed `dv_inertial_mps` (Phase 13n) must never be
        // overwritten by this reactive-dispersion solve.
        if let TcmPhase::Slewing { is_planned: false, .. } = &tcm_phase {
            if let Some(tcm) = tcm {
                // Same horizon cap as the trigger above: never target past
                // the next planned burn or the reference's next SOI entry.
                let horizon_end_s =
                    reactive_tcm_horizon_end_s(state.t_s, planned_burns, next_planned_burn_idx, duration_s, &ref_soi_entries);
                // Approach mode: the held direction is the
                // trigger-time B-PLANE solution — do NOT overwrite it with
                // the per-tick Lambert re-solve (whose direction is wrong
                // on an approach by construction); the ignition re-solve
                // (cheap once, full B-plane) handles staleness instead.
                if approach_soi_entry(state.t_s, horizon_end_s, &ref_soi_entries).is_none() {
                    tcm_phase = match solve_tcm_correction(&state, reference, horizon_end_s, mu_central_m3s2, dry_mass_kg, tcm) {
                        // Per-tick re-target — deliberately NOT reported on
                        // `tick_solve` (one per tick would be noise; the
                        // trigger and ignition solves are what matter).
                        Some((thrust_dir_inertial, dv_target_mps, _aim_s)) => TcmPhase::Slewing { thrust_dir_inertial, dv_target_mps, is_planned: false, planned: None },
                        None => TcmPhase::Coast,
                    };
                }
            }
        }

        // A main-engine burn (Slewing or Burning) takes priority over
        // pointing_mode/commander entirely -- real spacecraft ops don't run
        // a comm pass mid-maneuver. `RcsCorrecting` deliberately does NOT
        // override pointing (item #4's whole point), so it falls through to
        // the same commander/pointing_mode resolution as ordinary `Coast`.
        let (q_cmd, active_mode, max_rule_violation_deg, worst_violated_rule_label) = match &tcm_phase {
            TcmPhase::Slewing { thrust_dir_inertial, .. } | TcmPhase::Burning { thrust_dir_inertial, .. } => (
                desired_quaternion_cruise(CruisePointingMode::BurnAttitude { thrust_dir_inertial: *thrust_dir_inertial }, &state.r_m, &sun_pos),
                tcm_phase.label().map(str::to_string),
                None,
                None,
            ),
            TcmPhase::Coast | TcmPhase::RcsCorrecting { .. } => match commander.and_then(|c| c.resolve(state.t_s, &state.r_m, &state.v_mps, sun_pos)) {
                Some((mode_name, q, outcomes)) => {
                    let worst = outcomes
                        .iter()
                        .filter(|o| !o.is_controlling)
                        .max_by(|a, b| a.achieved_error_deg.total_cmp(&b.achieved_error_deg));
                    (
                        q,
                        Some(mode_name),
                        worst.map(|o| o.achieved_error_deg),
                        worst.map(|o| o.label.clone()),
                    )
                }
                None => (desired_quaternion_cruise(pointing_mode, &state.r_m, &sun_pos), None, None, None),
            },
        };
        let this_tick_pointing_error_deg = pointing_error_deg(&state.q, &q_cmd);

        // Slewing -> Burning once attitude has actually converged -- firing
        // while still slewing would waste propellant on a misaligned burn.
        // Planned burns (ask) additionally: (a) never ignite
        // BEFORE their configured epoch — the attitude is held (inertially
        // fixed) until then, so the gate is "settled AND epoch reached";
        // (b) re-solve the ΔV at ACTUAL ignition (item 4 — a late ignition
        // fires a fresh solve, not the lead-time one); (c) declare a
        // missed-burn fault and skip if still unsettled past the deadline
        // — never wait indefinitely.
        let mut planned_burn_fault: Option<&'static str> = None;
        if let TcmPhase::Slewing { thrust_dir_inertial, dv_target_mps, is_planned, planned } = tcm_phase {
            let settled = this_tick_pointing_error_deg <= DEFAULT_SETTLE_THRESHOLD_DEG;
            match planned {
                None => {
                    if settled {
                        // refine the Lambert correction by
                        // SHOOTING under the real force model at ignition
                        // (`tcm_shooting_correction`) — Lambert stays the
                        // trigger/slew-direction guess, the real dynamics
                        // decide what actually gets burned. Same horizon
                        // cap as the trigger (never past the next planned
                        // burn); the dv cap from the trigger-time solve is
                        // kept as an upper bound on the refined magnitude.
                        let horizon_end_s =
                            reactive_tcm_horizon_end_s(state.t_s, planned_burns, next_planned_burn_idx, duration_s, &ref_soi_entries);
                        let sma_m = semi_major_axis_m(&state.r_m, &state.v_mps, mu_central_m3s2);
                        let horizon_s = if sma_m > 0.0 {
                            TCM_HORIZON_PERIOD_FRACTION * orbital_period_s(sma_m, mu_central_m3s2)
                        } else {
                            horizon_end_s - state.t_s
                        };
                        let arrival_t_s = (state.t_s + horizon_s).min(horizon_end_s);
                        let solve = refine_reactive_correction(
                            &state, arrival_t_s, horizon_end_s, reference, mu_central_m3s2, bodies,
                            &ref_soi_entries, rtol, atol, thrust_dir_inertial * dv_target_mps,
                        );
                        let disp_now_m = dispersion(&state.r_m, &state.v_mps, state.t_s, reference).dr_m.norm();
                        // the ignition solve is the
                        // governing one for a main-engine correction —
                        // reported whichever way the decision goes (the
                        // `decision` field is finalized per branch below).
                        let mut ignition_solve_info = TcmSolveInfo {
                            context: "Ignition",
                            targeting: solve.method,
                            aim_epoch_s: arrival_t_s,
                            lambert_dv_mps: Some(dv_target_mps),
                            solved_dv_mps: solve.dv.norm(),
                            iterations: Some(solve.iterations),
                            converged: Some(solve.converged),
                            miss_m: Some(solve.miss_m),
                            no_burn_miss_m: solve.no_burn_miss_m.is_finite().then_some(solve.no_burn_miss_m),
                            decision: "Ignite",
                        };
                        if !shooting_solve_worth_firing(&solve, disp_now_m) {
                            // an ill-conditioned/non-converged
                            // solve is a finding, not a maneuver — abort to
                            // Coast rather than fire noise (the failure
                            // mode seen on a Mars approach).
                            eprintln!(
                                "Warning: reactive TCM at t={:.0} s aborted at ignition — shooting solve did not converge (miss {:.3e} m vs dispersion {:.3e} m)",
                                state.t_s, solve.miss_m, disp_now_m
                            );
                            ignition_solve_info.decision = "AbortedAtIgnition";
                            tick_solve = Some(ignition_solve_info);
                            tcm_phase = TcmPhase::Coast;
                        } else {
                            let n = solve.dv.norm();
                            // Same propellant-based cap the trigger applies
                            // (`TCM_DV_CAP_FRACTION_OF_AVAILABLE`), so a refined
                            // solve can legitimately exceed the Lambert guess
                            // but never the budget guard.
                            let dv_cap_mps = tcm
                                .filter(|_| state.mass_kg > dry_mass_kg + 1e-9)
                                .map(|t| TCM_DV_CAP_FRACTION_OF_AVAILABLE * t.isp_s * orbital_models::constants::G0 * (state.mass_kg / dry_mass_kg).ln())
                                .unwrap_or(dv_target_mps);
                            let (dir, mag) = if n > 1e-9 { (solve.dv / n, n.min(dv_cap_mps)) } else { (thrust_dir_inertial, dv_target_mps) };
                            // NEVER fire the refined magnitude
                            // along the stale held direction — if the
                            // re-solve moved the thrust direction, re-slew
                            // and let the settle gate re-apply (see
                            // REFINED_DIRECTION_RESLEW_DEG; the live
                            // oscillating-burns failure on the Mars approach).
                            let redirect_deg = dir.dot(&thrust_dir_inertial).clamp(-1.0, 1.0).acos().to_degrees();
                            tcm_phase = if redirect_deg > REFINED_DIRECTION_RESLEW_DEG {
                                ignition_solve_info.decision = "ReSlew";
                                TcmPhase::Slewing { thrust_dir_inertial: dir, dv_target_mps: mag, is_planned, planned }
                            } else {
                                TcmPhase::Burning { thrust_dir_inertial: dir, dv_remaining_mps: mag, is_planned, planned }
                            };
                            tick_solve = Some(ignition_solve_info);
                        }
                    }
                }
                Some(p) => {
                    // Capture-burn ignition gate: the burn
                    // fires at the DISPERSED trajectory's own periapsis
                    // passage relative to the capture body, not at the
                    // plan's stored epoch — the orbit a velocity-matching
                    // burn buys is set by the radius it fires at, so
                    // igniting wherever the ship happens to be at the
                    // nominal epoch captures into the wrong orbit
                    // entirely (see `time_to_periapsis_rel_s`). The stored
                    // epoch still OPENS the preparation window and anchors
                    // the go/no-go timeout; the periapsis estimate (from
                    // the hyperbolic Kepler equation, recomputed every
                    // tick from the real state) decides ignition: fire
                    // when within half a tick of periapsis, or once past
                    // it. Falls back to the stored epoch when the state
                    // isn't yet hyperbolic relative to the body (far out,
                    // or the track doesn't resolve).
                    let capture_tca_s: Option<f64> = planned_burns
                        .get(p.idx)
                        .and_then(|b| b.capture_body.as_deref())
                        .and_then(|name| capture_state_at(name, state.t_s))
                        .and_then(|(body_r, body_v, mu, _e)| {
                            time_to_periapsis_rel_s(&(state.r_m - body_r), &(state.v_mps - body_v), mu)
                        });
                    // Burn centering (standard MOI practice):
                    // start the burn ~half its estimated duration BEFORE
                    // periapsis so the burn arc straddles it — the classic
                    // gravity-loss halver (measured uncentered: 855 m/s
                    // delivered for a 642 m/s impulsive job). Duration
                    // estimated from the solved ΔV at the current mass and
                    // thrust; zero for a burn with no TcmConfig (can't
                    // happen for a planned burn, defensive only).
                    let ignition_due = match capture_tca_s {
                        Some(tca) => {
                            let t_burn_est_s = tcm
                                .map(|t| state.mass_kg * dv_target_mps / t.thrust_n.max(1e-3))
                                .unwrap_or(0.0);
                            tca <= 0.5 * t_burn_est_s + 0.5 * tick_s
                        }
                        None => state.t_s >= p.ignition_epoch_s,
                    };
                    // The timeout must track the REAL gate: a periapsis
                    // later than the stored epoch must not read as a
                    // missed burn.
                    let effective_deadline_s = match capture_tca_s {
                        Some(tca) => (state.t_s + tca.max(0.0) + PLANNED_BURN_TIMEOUT_S).max(p.deadline_s),
                        None => p.deadline_s,
                    };
                    if settled && ignition_due {
                        // A capture burn TRACKS its rotating solve during
                        // the burn itself (see the Burning execution
                        // block), so its ignition needs no direction-
                        // consistency re-slew — chasing the rotating
                        // direction with re-slews before ignition is a
                        // loop that can never latch (measured: 26 min of
                        // ignition delay ballooning the burn 772 → 1,154
                        // m/s).
                        let is_tracking = planned_burns.get(p.idx).is_some_and(|b| b.capture_body.is_some());
                        let (dir, mag) = match planned_burns.get(p.idx) {
                            Some(burn) => {
                                let dv = planned_burn_dv(burn, &state, reference, mu_central_m3s2, capture_state_at, bodies, rtol, atol);
                                let n = dv.norm();
                                if n > 1e-9 { (dv / n, n) } else { (thrust_dir_inertial, dv_target_mps) }
                            }
                            None => (thrust_dir_inertial, dv_target_mps),
                        };
                        // Same direction-consistency guard as the reactive
                        // path: a planned burn whose fresh
                        // solve moved the thrust direction re-slews (its
                        // ignition epoch has passed, so it ignites as soon
                        // as it settles on the new direction; the go/no-go
                        // deadline still bounds the total wait).
                        let redirect_deg = dir.dot(&thrust_dir_inertial).clamp(-1.0, 1.0).acos().to_degrees();
                        tcm_phase = if !is_tracking && redirect_deg > REFINED_DIRECTION_RESLEW_DEG {
                            TcmPhase::Slewing { thrust_dir_inertial: dir, dv_target_mps: mag, is_planned, planned }
                        } else {
                            TcmPhase::Burning { thrust_dir_inertial: dir, dv_remaining_mps: mag, is_planned, planned }
                        };
                    } else if !settled && state.t_s > effective_deadline_s {
                        planned_burn_fault = Some("MissedBurnTimeout");
                        eprintln!(
                            "Warning: planned burn #{} missed — attitude not settled ({:.2} deg) by t={:.0} s \
                             (epoch {:.0} s + {:.0} s go/no-go window); burn skipped",
                            p.idx, this_tick_pointing_error_deg, state.t_s, p.ignition_epoch_s, PLANNED_BURN_TIMEOUT_S
                        );
                        tcm_phase = TcmPhase::Coast;
                    }
                }
            }
        }

        // Burn-attitude abort: a burn that has been pointing
        // more than BURN_ABORT_POINTING_DEG off for BURN_ABORT_HOLD_S of
        // burn time is cut off before this tick fires the engine again.
        if let TcmPhase::Burning { planned, .. } = &tcm_phase {
            if this_tick_pointing_error_deg > BURN_ABORT_POINTING_DEG {
                burn_misaligned_s += burn_tick_s;
            } else {
                burn_misaligned_s = 0.0;
            }
            if burn_misaligned_s >= BURN_ABORT_HOLD_S {
                eprintln!(
                    "Warning: burn aborted at t={:.0} s — pointing error {:.1}° beyond {:.0}° for {:.0} s of burn \
                     (engine disturbance torque exceeding attitude-control authority?){}",
                    state.t_s, this_tick_pointing_error_deg, BURN_ABORT_POINTING_DEG, burn_misaligned_s,
                    planned.map(|p| format!("; planned burn #{}", p.idx)).unwrap_or_default()
                );
                if planned.is_some() {
                    planned_burn_fault = Some("BurnAttitudeAbort");
                }
                let cooldown_s = TCM_COOLDOWN_FRACTION_OF_REMAINING_TOF * (duration_s - state.t_s).max(0.0);
                post_burn = Some((state.t_s, cooldown_s));
                tcm_phase = TcmPhase::Coast;
                burn_misaligned_s = 0.0;
            }
        } else {
            burn_misaligned_s = 0.0;
        }

        // Real fix (
        // maneuver mode... wheels slowly slew the s/c to correct attitude...
        // then it hands off control to the RCS, and then the manoeuvre is
        // executed (while RCS keep attitude stable), then after manoeuvre,
        // we give back control to the wheels"). Before this, `control_mode`
        // was the caller's ONE fixed value (WheelsPrimary) for the entire
        // leg, `Burning` included -- confirmed live: introducing real
        // thrust-misalignment torque meant the wheels alone had
        // to absorb a continuous disturbance torque for the ENTIRE burn
        // duration with no relief, silently accumulating momentum until
        // they saturated and lost SunPointing control well after the burn
        // ended (matches the report exactly: fine before, saturates only
        // with corrective maneuvers enabled, during ordinary SunPointing
        // rather than during the maneuver itself). `ControlMode::
        // ThrustersPrimary` already exists for exactly this ("a maneuver:
        // wheel torque authority is typically far below what a burn-
        // attitude slew or large correction needs" -- control.rs's own doc
        // comment) but was never actually switched to. Slewing intentionally
        // stays on the caller's WheelsPrimary (the wheels do the slow
        // initial reorientation, as in real ops practice);
        // only `Burning` hands off to RCS. `Coast`/`RcsCorrecting` also stay
        // on the caller's mode -- RcsCorrecting deliberately never overrides
        // pointing (item #4), so ordinary wheel-held attitude control is
        // still correct there.
        let intended_control_mode = if matches!(tcm_phase, TcmPhase::Burning { .. }) {
            ControlMode::ThrustersPrimary
        } else {
            control_mode
        };
        // Mode-scheduled tick (§10.5.1): finer while the burn fires, so
        // the thruster loop's bandwidth cap rises exactly when the engine
        // disturbance is present. Clamped to the remaining leg so the
        // final tick lands on `duration_s` under either cadence.
        let mut tick_eff = if matches!(tcm_phase, TcmPhase::Burning { .. }) { burn_tick_s } else { tick_s }
            .min((duration_s - state.t_s).max(1e-9));
        // Land a tick boundary EXACTLY on the next planned burn's epoch
        //: an impulsive external-stage injection applied at
        // the first tick boundary past its epoch was up to one tick late —
        // at perigee (μ/r² ≈ 9 m/s²) even 0.7 s is a ~6 m/s, ~4 km error
        // that the executive then had to chase.
        if let Some(b) = planned_burns.get(next_planned_burn_idx) {
            let dt_to_epoch = b.epoch_s - state.t_s;
            if dt_to_epoch > 1e-6 && dt_to_epoch < tick_eff {
                tick_eff = dt_to_epoch;
            }
        }
        // Item #3: with propellant exhausted, RCS-dependent modes/laws
        // degrade to their no-RCS equivalents for this tick -- thrusters
        // physically cannot fire with nothing left in the tank, regardless
        // of what the caller (or the handoff above) configured.
        let (control_mode_eff, momentum_law_eff) = if propellant_exhausted {
            let cm = match intended_control_mode {
                ControlMode::ThrustersPrimary | ControlMode::ThrustersOnly => ControlMode::WheelsPrimary,
                other => other,
            };
            (cm, MomentumManagementLaw::None)
        } else {
            (intended_control_mode, momentum_law)
        };

        // Review E2: rate-limited eigenaxis slew profile for
        // LARGE attitude errors — mode transitions and TCM slews alike
        // (supersedes 13o's SLEWING_GAIN_SCALE gain-softening; see the
        // constants' own doc comments above). The PD tracks a profiled
        // intermediate target a bounded step ahead of the CURRENT attitude
        // along the error eigenaxis, with the allowed rate deceleration-
        // limited by the wheels' real worst-axis torque authority — so the
        // commanded torque stays inside what the wheels can deliver BY
        // CONSTRUCTION instead of saturating open-loop. Small errors pass
        // through untouched (ordinary pointing-hold, full gains).
        // `this_tick_pointing_error_deg` above deliberately measures
        // against the FINAL target — settle gating and telemetry keep
        // reporting real convergence, never "converged onto the moving
        // intermediate target."
        //
        // `α_max` comes from the ACTIVE mode's actuator (§10.5),
        // not unconditionally from the wheels — job 4 showed the wheel-
        // derived rate limit capping the proportional demand BELOW the
        // engine disturbance during a burn, so the vehicle could never
        // recover once its error exceeded the 15° engage threshold.
        let inertia_max = sc.inertia_diag_kgm2.x.max(sc.inertia_diag_kgm2.y).max(sc.inertia_diag_kgm2.z).max(1e-6);
        let alpha_max_radps2 = controls.authority_nm(control_mode_eff) / inertia_max;
        let q_cmd_ctrl = sim_engine::control::profiled_slew_target(
            &state.q, &q_cmd,
            SLEW_PROFILE_LEAD_TICKS * tick_eff,
            alpha_max_radps2,
            SLEW_PROFILE_CRUISE_RATE_RADPS,
            SLEW_PROFILE_ENGAGE_THRESHOLD_RAD,
        );

        // Layer 2 — activity (§10.5.4): BurnHold while the main engine
        // fires; Slew while reorienting (a maneuver slew, or the §10.4
        // profile engaged by a large error); Coast when a commander is
        // present but has NO named pointing requirement for this tick
        // (`active_mode` None — the caller's fallback pointing_mode is a
        // convenience, not a requirement); Hold otherwise, including every
        // fixed-`pointing_mode` run without a commander (that mode IS the
        // pointing requirement there, so the full-bandwidth law applies —
        // keeps every pre-run's behavior unchanged).
        let profile_engaged = this_tick_pointing_error_deg.to_radians() > SLEW_PROFILE_ENGAGE_THRESHOLD_RAD;
        let activity = match &tcm_phase {
            TcmPhase::Burning { .. } => Activity::BurnHold,
            TcmPhase::Slewing { .. } => Activity::Slew,
            _ if profile_engaged => Activity::Slew,
            _ if commander.is_some() && active_mode.is_none() => Activity::Coast,
            _ => Activity::Hold,
        };

        // Layer 3 — (re)derive the running law when the mode/activity
        // changes or the scheduled mass trigger fires; the controller's
        // integral/latch state resets with it (a new law starts clean).
        let key = (control_mode_eff, activity);
        let mut gain_schedule_point = None;
        if controller_key != Some(key) || controls.mass_trigger(controller_mass_kg, state.mass_kg) {
            let law = controls.law_for(control_mode_eff, activity, state.mass_kg);
            controller = AttitudeController::new(law);
            controller_key = Some(key);
            controller_mass_kg = state.mass_kg;
            gain_schedule_point = Some(controls.schedule_point(state.t_s, state.mass_kg, control_mode_eff, activity, &law));
        }
        let tau_cmd = controller.command_torque(&state.q, &q_cmd_ctrl, &state.omega_radps, state.t_s, tick_eff);
        let controller_law = controller.law().label();
        let (controller_kp, controller_kd) = controller.law().pd_gains().map(|(p, d)| (Some(p), Some(d))).unwrap_or((None, None));
        let control_mode_label: &'static str = match control_mode_eff {
            ControlMode::WheelsPrimary => "WheelsPrimary",
            ControlMode::ThrustersPrimary => "ThrustersPrimary",
            ControlMode::ThrustersOnly => "ThrustersOnly",
        };
        let alloc = allocate(
            control_mode_eff, tau_cmd, wheel_cluster, &state.wheel_speeds_radps,
            rcs_thrusters, tick_eff, momentum_law_eff,
        );
        let control_torque_body = net_body_torque(wheel_cluster, &alloc);
        rcs_propellant_kg_cum += alloc.rcs_propellant_kg;

        // Item #5: real RCS net translational force (from ordinary
        // attitude-hold/desaturation activity above), previously discarded
        // at every `thruster_selection` call site inside `allocate` --
        // fed into translation as a ZOH velocity kick below, same
        // approximation level as the already-ZOH control torque. Item #4's
        // `RcsCorrecting` phase adds its own translational-purpose thrust
        // to this same accumulator just below.
        let mut extra_force_body = alloc.rcs_net_force_body_avg;
        let q0 = state.q;

        // Item #4: RcsCorrecting's own translational thrust -- re-resolved
        // every tick against the CURRENT attitude (free to move under
        // normal pointing control, unlike a main-engine burn's fixed
        // BurnAttitude hold). Only thrusters with positive alignment to the
        // target inertial direction fire; the REAL summed vector is applied
        // (not an idealized pure-alignment assumption) -- any off-axis
        // component becomes real additional dispersion, picked up by the
        // next Coast-phase trigger check rather than silently discarded.
        let mut rcs_tcm_progress_mps = 0.0_f64;
        if let TcmPhase::RcsCorrecting { thrust_dir_inertial, .. } = &tcm_phase {
            if !propellant_exhausted {
                let dir_body = inertial_to_body(&state.q, thrust_dir_inertial);
                let mut force_body = Vector3::zeros();
                let mut prop_flow_kgps = 0.0_f64;
                for t in rcs_thrusters {
                    let alignment = t.dir.dot(&dir_body);
                    if alignment > 0.0 {
                        force_body += t.dir * t.thrust_n;
                        prop_flow_kgps += t.thrust_n / (t.isp_s * orbital_models::constants::G0);
                    }
                }
                if force_body.norm() > 1e-9 {
                    // Duty-scale down if the tank would run dry mid-tick --
                    // same PWM-duty philosophy §10.2 already uses for
                    // ordinary attitude RCS, applied here to translation.
                    let full_prop_use_kg = prop_flow_kgps * tick_eff;
                    let duty = if full_prop_use_kg > 1e-12 {
                        (propellant_remaining_kg / full_prop_use_kg).min(1.0)
                    } else {
                        1.0
                    };
                    let force_body_applied = force_body * duty;
                    extra_force_body += force_body_applied;
                    tcm_propellant_kg_cum += full_prop_use_kg * duty;
                    let force_inertial = body_to_inertial(&state.q, &force_body_applied);
                    rcs_tcm_progress_mps = (force_inertial / state.mass_kg * tick_eff).dot(thrust_dir_inertial).max(0.0);
                }
            }
        }

        let h_wheel = wheel_cluster.total_momentum(&state.wheel_speeds_radps);
        let disp = dispersion(&state.r_m, &state.v_mps, state.t_s, reference);
        let wheel_sat_frac = state.wheel_speeds_radps.iter().fold(0.0_f64, |m, s| m.max(s.abs()))
            / wheel_cluster.max_speed;
        let torque_bd = disturbance_torque_breakdown(&state, sc, bodies, mu_central_m3s2);
        let accel_bd = translational_accel_breakdown(&state, sc, bodies, mu_central_m3s2, false);

        let row = CruiseTickRow {
            t_s: state.t_s,
            r_m: state.r_m,
            v_mps: state.v_mps,
            q: state.q,
            q_cmd,
            omega_radps: state.omega_radps,
            wheel_speeds_radps: state.wheel_speeds_radps,
            wheel_momentum_nms: h_wheel.norm(),
            pointing_error_deg: this_tick_pointing_error_deg,
            torque_cmd_body_nm: tau_cmd,
            torque_delivered_body_nm: control_torque_body,
            dr_m: disp.dr_m.norm(),
            dv_mps: disp.dv_mps.norm(),
            rcs_propellant_kg_cum,
            wheel_sat_frac,
            wheel_motor_torque_cmd_nm: alloc.wheel_motor_torque_cmd_nm,
            wheel_motor_torque_nm: alloc.wheel_motor_torque_nm,
            wheel_torque_sat_frac: alloc.wheel_torque_sat_frac,
            propellant_remaining_kg: (propellant_mass_kg - rcs_propellant_kg_cum - tcm_propellant_kg_cum).max(0.0),
            torque_gravity_gradient_nm: torque_bd.gravity_gradient.norm(),
            torque_srp_nm: torque_bd.srp.norm(),
            accel_central_gravity_mps2: accel_bd.central_gravity.norm(),
            accel_third_body_mps2: accel_bd.third_body.norm(),
            accel_srp_mps2: accel_bd.srp.norm(),
            planned_burn_idx: match &tcm_phase {
                TcmPhase::Slewing { planned: Some(p), .. } | TcmPhase::Burning { planned: Some(p), .. } => Some(p.idx),
                _ => external_stage_fired,
            },
            planned_burn_fault,
            active_mode,
            max_rule_violation_deg,
            worst_violated_rule_label,
            tcm_phase: tcm_phase.label(),
            tcm_propellant_kg_cum,
            tcm_dv_mps_cum,
            tcm_solve: tick_solve,
            control_mode: control_mode_label,
            controller_law,
            control_activity: activity.label(),
            controller_kp,
            controller_kd,
            gain_schedule_point,
        };
        let keep_going = on_row(&row);
        rows.push(row);
        if !keep_going {
            break;
        }

        let mut new_speeds = state.wheel_speeds_radps;
        let speed_dots = wheel_cluster.speed_dots(&alloc.wheel_motor_torque_nm);
        for k in 0..4 {
            // Real motor current/torque saturates at the wheel's rated max
            // speed -- it physically cannot keep accelerating past that
            // point. Found via cruise_commander_demo's aggressive comm-pass
            // schedule: without this clamp, wheel momentum can run away
            // unbounded (observed >400 N*m*s for a cluster whose real
            // physical ceiling, from max_speed alone, is ~30 N*m*s) whenever
            // commanded torque persists in one direction faster than RCS
            // desaturation (a separate, threshold-triggered mechanism) can
            // unload it.
            new_speeds[k] = (new_speeds[k] + speed_dots[k] * tick_eff).clamp(-wheel_cluster.max_speed, wheel_cluster.max_speed);
        }

        let burn_cfg = match &tcm_phase {
            TcmPhase::Burning { .. } => tcm.map(|t| BurnConfig {
                thrust_n: t.thrust_n,
                isp_s: t.isp_s,
                // Body +x, per CruisePointingMode::BurnAttitude's own
                // convention -- q_cmd already aligns body +x with
                // thrust_dir_inertial, so the burn always fires along the
                // body's own +x boresight regardless of which inertial
                // direction that currently is.
                body_dir: Vector3::new(1.0, 0.0, 0.0),
                thrust_offset_body_m,
            }),
            _ => None,
        };
        let mass_before = state.mass_kg;

        state = step_tick_with_burn(
            &state, tick_eff, sc, bodies, mu_central_m3s2, burn_cfg.as_ref(),
            control_torque_body, h_wheel, rtol, atol,
        );
        state.wheel_speeds_radps = new_speeds;

        // Item #5/#4: apply the extra (attitude-hold RCS + RcsCorrecting)
        // force as a ZOH velocity kick, converted using the tick-START
        // attitude/mass (`q0`/`mass_before`) -- consistent with how
        // `control_torque_body` itself is frozen for the tick.
        if extra_force_body.norm() > 0.0 {
            let f_inertial = body_to_inertial(&q0, &extra_force_body);
            state.v_mps += f_inertial / mass_before * tick_eff;
        }

        if let TcmPhase::Burning { thrust_dir_inertial, dv_remaining_mps, is_planned, planned } = &tcm_phase {
            if let Some(t) = tcm {
                let mass_after = state.mass_kg;
                tcm_propellant_kg_cum += (mass_before - mass_after).max(0.0);
                // real delivered ΔV this burning tick,
                // Tsiolkovsky on the mass actually lost — accumulated
                // uniformly for both the tracking and fixed-impulse
                // branches below.
                if mass_after > 1e-6 && mass_before > mass_after {
                    tcm_dv_mps_cum += t.isp_s * orbital_models::constants::G0 * (mass_before / mass_after).ln();
                }
                // TRACKING capture burn: a velocity-matching
                // burn's correct thrust direction ROTATES near periapsis
                // (at ~v_rel/r_rel — measured ~0.035°/s on a Mars capture,
                // ~100° over the burn's own duration), so an inertially
                // fixed attitude is the wrong model for it entirely: held
                // fixed, a late one-hour capture burn ballooned from 772 to
                // 1,154 m/s and drained the tank. Real LOI burns steer.
                // Here the burn RE-SOLVES `solve_capture_burn_dv` from the
                // CURRENT state every burn tick: the attitude command
                // tracks the rotating direction (the thruster loop's
                // bandwidth is orders of magnitude above the rotation
                // rate), `dv_remaining` is the solve's own magnitude, and
                // completion is the solve falling below
                // `CAPTURE_BURN_COMPLETE_DV_MPS` — a closed-loop burn
                // cutoff on the achieved state, not a Tsiolkovsky
                // countdown of a stale impulse.
                let tracking_solve = planned
                    .and_then(|p| planned_burns.get(p.idx))
                    .filter(|b| b.capture_body.is_some())
                    .map(|burn| planned_burn_dv(burn, &state, reference, mu_central_m3s2, capture_state_at, bodies, rtol, atol));
                if let Some(dv) = tracking_solve {
                    let n = dv.norm();
                    // The engine delivers a quantized impulse per burn tick
                    // (`thrust·tick/m`); a remainder below that cannot be
                    // resolved — chasing it overshoots, the next solve
                    // flips direction, and the burn limit-cycles around
                    // the target (caught by the D5 regression test). The
                    // cutoff is therefore the LARGER of the fixed floor
                    // and ~1.2 tick-impulses.
                    let tick_impulse_mps = t.thrust_n / state.mass_kg.max(1.0) * burn_tick_s;
                    let cutoff_mps = CAPTURE_BURN_COMPLETE_DV_MPS.max(1.2 * tick_impulse_mps);
                    if n < cutoff_mps {
                        let cooldown_s = TCM_COOLDOWN_FRACTION_OF_REMAINING_TOF * (duration_s - state.t_s).max(0.0);
                        post_burn = Some((state.t_s, cooldown_s));
                        tcm_phase = TcmPhase::Coast;
                    } else {
                        tcm_phase = TcmPhase::Burning { thrust_dir_inertial: dv / n, dv_remaining_mps: n, is_planned: *is_planned, planned: *planned };
                    }
                } else {
                    // Real delivered DeltaV this tick, from the Tsiolkovsky
                    // relation applied to the mass this tick ACTUALLY lost --
                    // not an idealized instantaneous-impulse assumption.
                    let dv_delivered_mps = if mass_after > 1e-6 && mass_before > mass_after {
                        t.isp_s * orbital_models::constants::G0 * (mass_before / mass_after).ln()
                    } else {
                        0.0
                    };
                    let remaining = dv_remaining_mps - dv_delivered_mps;
                    if remaining <= 0.0 {
                        // Item #1: record completion state for the next
                        // reactive trigger's cooldown guard -- applied
                        // regardless of `is_planned`, a conservative default
                        // (don't let a reactive correction fire immediately
                        // after a planned burn either, before its effect on
                        // dispersion has had time to show).
                        let cooldown_s = TCM_COOLDOWN_FRACTION_OF_REMAINING_TOF * (duration_s - state.t_s).max(0.0);
                        post_burn = Some((state.t_s, cooldown_s));
                        tcm_phase = TcmPhase::Coast;
                    } else {
                        tcm_phase = TcmPhase::Burning { thrust_dir_inertial: *thrust_dir_inertial, dv_remaining_mps: remaining, is_planned: *is_planned, planned: *planned };
                    }
                }
            }
        }

        if let TcmPhase::RcsCorrecting { thrust_dir_inertial, dv_remaining_mps } = &tcm_phase {
            // the projected RCS impulse IS this phase's
            // delivered ΔV (same quantity `remaining` is decremented by).
            tcm_dv_mps_cum += rcs_tcm_progress_mps;
            let remaining = dv_remaining_mps - rcs_tcm_progress_mps;
            // Stalled (no progress this tick -- alignment lost as attitude
            // drifted, or propellant ran out) is treated the same as
            // "done": there is nothing more this correction can achieve,
            // so stop and let the next Coast-phase trigger check (which may
            // now choose the main engine instead, since RCS just proved
            // infeasible) decide whether more correction is still needed.
            let stalled = rcs_tcm_progress_mps <= 0.0;
            if remaining <= 0.0 || stalled {
                let cooldown_s = TCM_COOLDOWN_FRACTION_OF_REMAINING_TOF * (duration_s - state.t_s).max(0.0);
                post_burn = Some((state.t_s, cooldown_s));
                tcm_phase = TcmPhase::Coast;
            } else {
                tcm_phase = TcmPhase::RcsCorrecting { thrust_dir_inertial: *thrust_dir_inertial, dv_remaining_mps: remaining };
            }
        }
    }

    rows
}

fn pointing_error_deg(q_cur: &Vector4<f64>, q_cmd: &Vector4<f64>) -> f64 {
    let dot = q_cur.dot(q_cmd).clamp(-1.0, 1.0).abs();
    2.0 * dot.acos().to_degrees()
}

// ── Phase 5.2: /api/simulate wiring ─────────────────────────────────────────

/// One streamed cruise-loop tick, sent over `/api/simulate/:id/stream` when
/// the job's `MissionConfig` had `cruise_seed` set — a sibling of
/// `simulate::SimStepMsg`, not a reuse of it, since the shape genuinely
/// differs (dispersion against a reference instead of an EKF estimate; no
/// navigation-filter fields at all, since Phase 13h/navigation isn't built
/// yet — this first cut streams TRUTH state only).
///
/// `msg_type` (added, dispatch-hardening ask): a real client bug
/// sent `cruise_seed` + `monte_carlo_runs: 1`, which server-side dispatches
/// to the ENTIRELY DIFFERENT `CruiseMc` job kind (`server::routes::
/// simulate::start`'s `cfg.cruise_seed.is_some() && monte_carlo_runs > 0`
/// branch) — the client blindly `JSON.parse(...) as CruiseStepMsg`'d a
/// `CruiseMcRunMsg` instead and crashed reading `.r_m` off an object that
/// doesn't have it. The two message shapes were always distinguishable by
/// WHICH JOB you started (the request you made decides which one a job's
/// stream ever carries), but nothing on the wire let a consumer verify that
/// assumption at parse time. `msg_type` is always `"cruise_tick"` here —
/// same discriminator-field precedent already used by `server::routes::
/// optimize::MgaSequenceContextMsg` — so any consumer can check it before
/// trusting the rest of the shape, instead of relying on having gotten the
/// request right.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CruiseStepMsg {
    pub msg_type: &'static str,
    pub t_s: f64,
    pub r_m: [f64; 3],
    pub v_mps: [f64; 3],
    /// Attitude quaternion [w, x, y, z] — same layout convention as
    /// `simulate::SimStepMsg::q`.
    pub q: [f64; 4],
    /// The commanded/target attitude quaternion — see `CruiseTickRow::
    /// q_cmd`'s doc comment.
    pub q_cmd: [f64; 4],
    pub wheel_speeds_radps: [f64; 4],
    pub wheel_momentum_nms: f64,
    pub pointing_error_deg: f64,
    /// See `CruiseTickRow::torque_cmd_body_nm`/`torque_delivered_body_nm`.
    pub torque_cmd_body_nm: [f64; 3],
    pub torque_delivered_body_nm: [f64; 3],
    /// `|actual_r - reference_r|` [m] — reported, not corrected (no TCM
    /// execution in this first cut, see this module's doc comment).
    pub dr_m: f64,
    pub dv_mps: f64,
    pub rcs_propellant_kg_cum: f64,
    pub wheel_sat_frac: f64,
    /// Review E3 — wire names PINNED by the frontend's
    /// dormant Fig. 5 (`CruiseFigures.tsx`): commanded (unclamped) vs.
    /// delivered per-wheel motor torque [N·m], same wheel ordering as
    /// `wheel_speeds_radps` — see `CruiseTickRow::wheel_motor_torque_cmd_nm`
    /// / `wheel_motor_torque_nm` (the internal names) — and the torque-
    /// authority saturation fraction (may exceed 1.0).
    pub wheel_torque_cmd_nm: [f64; 4],
    pub wheel_torque_delivered_nm: [f64; 4],
    pub wheel_torque_sat_frac: f64,
    /// Body angular velocity [rad/s] (ask, bundled with E3) —
    /// the standard missing piece for diagnosing slew behavior; same
    /// serialization shape as the slew-test messages' `omega_radps`.
    pub omega_radps: [f64; 3],
    /// See `CruiseTickRow::planned_burn_idx` / `planned_burn_fault`.
    pub planned_burn_idx: Option<usize>,
    pub planned_burn_fault: Option<String>,
    pub propellant_remaining_kg: f64,
    pub torque_gravity_gradient_nm: f64,
    pub torque_srp_nm: f64,
    pub accel_central_gravity_mps2: f64,
    /// Real when `cruise_seed.body_tracks` resolves any perturber (see
    /// `build_body_track_perturbers`) — was structurally always 0.0 before
    /// since `run_cruise_leg`'s `bodies` parameter was always
    /// empty until that fix.
    pub accel_third_body_mps2: f64,
    /// Always 0.0 in this first cut — see `CruiseTickRow::accel_srp_mps2`'s
    /// doc comment.
    pub accel_srp_mps2: f64,
    /// See `CruiseTickRow::active_mode`.
    pub active_mode: Option<String>,
    pub max_rule_violation_deg: Option<f64>,
    pub worst_violated_rule_label: Option<String>,
    /// TCM executive state this tick — see `CruiseTickRow::tcm_phase`.
    pub tcm_phase: Option<String>,
    pub tcm_propellant_kg_cum: f64,
    /// See `CruiseTickRow::tcm_dv_mps_cum`.
    pub tcm_dv_mps_cum: f64,
    /// The reactive correction solve that ran this tick, if any — see
    /// [`TcmSolveInfo`]. A tick carrying one always relays regardless of
    /// `report_stride`, so declined solves are visible on the stream too.
    pub tcm_solve: Option<TcmSolveInfo>,
    /// Three-layer attitude control — see the same-named
    /// `CruiseTickRow` fields: allocation mode, layer-1 law, layer-2
    /// activity, effective PD gains (null for phase-plane), and the
    /// layer-3 schedule point on the ticks a law was (re)derived.
    pub control_mode: String,
    pub controller_law: String,
    pub control_activity: String,
    pub controller_kp: Option<f64>,
    pub controller_kd: Option<f64>,
    pub gain_schedule_point: Option<GainSchedulePoint>,
}

/// Final result for a cruise-seeded `/api/simulate` job — a sibling of
/// `simulate::SimResult`, analogous shape (final state + a CSV path) but
/// cruise-specific fields (final dispersion, peak wheel momentum) in place
/// of `SimResult`'s phase list (this loop has no phase concept, it flies
/// one continuous leg).
#[derive(Debug, Clone, serde::Serialize)]
pub struct CruiseResult {
    pub final_r_m: [f64; 3],
    pub final_v_mps: [f64; 3],
    pub final_dr_m: f64,
    pub final_dv_mps: f64,
    pub max_dr_m: f64,
    pub max_wheel_momentum_nms: f64,
    pub rcs_propellant_kg_used: f64,
    /// Cumulative propellant consumed by TCM corrections [kg] — item 5,
    /// extended (item #4) to also cover RCS-only
    /// corrections (`TcmPhase::RcsCorrecting`), not just main-engine burns.
    /// Always 0.0 when TCM wasn't configured/never triggered.
    pub tcm_propellant_kg_used: f64,
    pub final_propellant_remaining_kg: f64,
    pub cruise_csv_path: String,
    /// Empty when no `GncCommander` was active (legacy fixed-pointing-mode
    /// run) — see [`detect_mode_transitions`].
    pub mode_transitions: Vec<ModeTransitionReport>,
    /// Review C2: empirical interpolation floor of the
    /// submitted reference [m] — see [`reference_interpolation_floor_m`].
    /// `None` when the reference has too few samples to estimate one
    /// (fewer than 3).
    pub reference_interpolation_floor_m: Option<f64>,
    /// Review C2: human-readable warnings about this run's
    /// configuration that aren't hard `check_config` errors — currently:
    /// `tcm_dr_threshold_m` set below the reference's own interpolation
    /// floor (TCM would chase numbers below the reference's own resolution
    /// noise). Empty when nothing is worth flagging.
    pub warnings: Vec<String>,
    /// One entry per `cruise_seed.planned_burns` item, in config order —
    /// see [`PlannedBurnReport`] (burn-executive ask: real
    /// slew-start/ignition/completion epochs vs. the configured epoch, and
    /// missed-burn faults as reported findings).
    pub planned_burn_reports: Vec<PlannedBurnReport>,
    /// One entry per REACTIVE maneuver — trigger/ignition epochs, the
    /// governing solve's targeting/iterations/convergence/miss, executed
    /// ΔV and propellant — plus zero-ΔV entries for declined trigger
    /// solves. See [`ReactiveBurnReport`]
    /// and [`detect_reactive_burn_reports`].
    pub reactive_burn_reports: Vec<ReactiveBurnReport>,
    /// Three-layer attitude control (`MANUAL.md` §10.5):
    /// every derived/overridden law parameter the run actually used, per
    /// actuator class and activity, plus scheduling settings and any
    /// feasibility warnings — a run is reproducible from this.
    pub attitude_control_effective: AttitudeControlEffective,
    /// Layer 3: one entry per tick the active law was (re)derived (mode/
    /// activity change or mass trigger), in time order.
    pub gain_schedule_points: Vec<GainSchedulePoint>,
}

/// Derive per-planned-burn execution reports from the tick rows — see
/// [`PlannedBurnReport`]. Post-hoc over `rows` (same pattern as
/// [`detect_mode_transitions`]) so `run_cruise_leg`'s signature/return
/// stay unchanged: `slew_start_s` = first row tagged with the burn's
/// index, `ignition_s` = its first `Burning` row, `completed_s` = the
/// first row after those that is no longer tagged with it, the fault
/// from `planned_burn_fault`.
pub fn detect_planned_burn_reports(
    rows: &[CruiseTickRow],
    planned_burns: &[crate::config::PlannedBurnConfig],
    // 
    // body-state lookup so a capture burn's report can carry the ACHIEVED
    // orbit's elements about its body — same closure shape as
    // `run_cruise_leg`'s `capture_state_at`. Pass `&|_, _| None` when no
    // capture bodies exist (elements are then omitted).
    capture_state_at: &dyn Fn(&str, f64) -> Option<(Vector3<f64>, Vector3<f64>, f64, f64)>,
) -> Vec<PlannedBurnReport> {
    planned_burns
        .iter()
        .enumerate()
        .map(|(idx, burn)| {
            let mut slew_start_s = None;
            let mut ignition_s = None;
            let mut completed_s = None;
            let mut fault: Option<&str> = None;
            let mut seen = false;
            for row in rows {
                if row.planned_burn_idx == Some(idx) {
                    seen = true;
                    if slew_start_s.is_none() {
                        slew_start_s = Some(row.t_s);
                    }
                    if ignition_s.is_none() && row.tcm_phase == Some("Burning") {
                        ignition_s = Some(row.t_s);
                    }
                } else if seen && completed_s.is_none() {
                    completed_s = Some(row.t_s);
                    if row.planned_burn_fault.is_some() {
                        fault = row.planned_burn_fault;
                    }
                }
                // The fault tick itself is the first untagged tick after
                // the slew (phase already reset to Coast); it is caught by
                // the branch above. A fault while still tagged (can't
                // happen today) would be caught here.
                if row.planned_burn_idx == Some(idx) && row.planned_burn_fault.is_some() {
                    fault = row.planned_burn_fault;
                }
            }
            // External-stage burns are applied impulsively on
            // one tagged tick with no `Burning` phase: that tick is both
            // ignition and completion.
            if burn.external_stage && slew_start_s.is_some() && ignition_s.is_none() {
                ignition_s = slew_start_s;
            }
            let status = if let Some(f) = fault {
                f.to_string()
            } else if ignition_s.is_some() && burn.external_stage {
                "Completed (external stage)".to_string()
            } else if ignition_s.is_some() {
                "Completed".to_string()
            } else {
                "NotReached".to_string()
            };
            // Achieved capture orbit: elements about the
            // capture body from the truth's state at the completion tick —
            // what the burn actually bought, next to what was planned.
            let achieved_capture = burn.capture_body.as_deref().and_then(|name| {
                let done_t = if fault.is_some() { None } else { completed_s }?;
                let row = rows.iter().find(|r| r.t_s >= done_t)?;
                let (body_r, body_v, mu, _e) = capture_state_at(name, row.t_s)?;
                let r_rel = row.r_m - body_r;
                let v_rel = row.v_mps - body_v;
                let r = r_rel.norm();
                if r < 1.0 || mu <= 0.0 {
                    return None;
                }
                let energy = 0.5 * v_rel.norm_squared() - mu / r;
                let h2 = r_rel.cross(&v_rel).norm_squared();
                let ecc = (1.0 + 2.0 * energy * h2 / (mu * mu)).max(0.0).sqrt();
                let ignition_radius_m = ignition_s
                    .and_then(|t_ign| rows.iter().find(|r| r.t_s >= t_ign))
                    .and_then(|row_ign| capture_state_at(name, row_ign.t_s).map(|(br, ..)| (row_ign.r_m - br).norm()));
                let (sma_m, periapsis_m, apoapsis_m, period_s) = if energy < 0.0 {
                    let a = -mu / (2.0 * energy);
                    (Some(a), Some(a * (1.0 - ecc)), Some(a * (1.0 + ecc)), Some(2.0 * std::f64::consts::PI * (a * a * a / mu).sqrt()))
                } else {
                    (None, None, None, None)
                };
                Some(AchievedCaptureReport {
                    bound: energy < 0.0,
                    ignition_radius_m,
                    sma_m,
                    eccentricity: ecc,
                    periapsis_m,
                    apoapsis_m,
                    period_s,
                })
            });
            PlannedBurnReport {
                index: idx,
                label: burn.label.clone(),
                configured_epoch_s: burn.epoch_s,
                slew_start_s,
                ignition_s,
                completed_s: if fault.is_some() { None } else { completed_s },
                status,
                achieved_capture,
            }
        })
        .collect()
}

/// derive the reactive-maneuver execution
/// reports from the tick rows — the reactive twin of
/// [`detect_planned_burn_reports`], same post-hoc pattern so
/// `run_cruise_leg`'s signature stays unchanged. A reactive maneuver is a
/// maximal run of rows in a TCM phase with no `planned_burn_idx` tag; a
/// declined trigger solve (row still coasting, `tcm_solve` carrying a
/// `Declined*` decision) gets its own zero-ΔV entry, so the executive's
/// "looked, chose not to fire" decisions are on the record too.
///
/// The governing solve for a maneuver (whose targeting/iterations/miss the
/// report carries) is the LAST ignition-time (or RCS-trigger) solve seen
/// for it — including one on the first row AFTER the group, since an
/// `AbortedAtIgnition` decision resets the phase to `Coast` on the very
/// tick it solves. `lambert_dv_mps` always comes from the trigger-time
/// solve (the original two-body guess).
pub fn detect_reactive_burn_reports(rows: &[CruiseTickRow]) -> Vec<ReactiveBurnReport> {
    let is_reactive = |r: &CruiseTickRow| r.tcm_phase.is_some() && r.planned_burn_idx.is_none();
    let mut reports = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        if !is_reactive(&rows[i]) {
            if let Some(s) = rows[i].tcm_solve.as_ref() {
                if s.decision.starts_with("Declined") {
                    reports.push(ReactiveBurnReport {
                        trigger_epoch_s: rows[i].t_s,
                        actuator: "None".to_string(),
                        targeting: Some(s.targeting.to_string()),
                        aim_epoch_s: Some(s.aim_epoch_s),
                        lambert_dv_mps: s.lambert_dv_mps,
                        solved_dv_mps: Some(s.solved_dv_mps),
                        shooting_iterations: s.iterations,
                        shooting_converged: s.converged,
                        predicted_miss_m: s.miss_m,
                        no_burn_miss_m: s.no_burn_miss_m,
                        ignition_s: None,
                        completed_s: None,
                        executed_dv_mps: 0.0,
                        propellant_kg: 0.0,
                        status: "Declined".to_string(),
                    });
                }
            }
            i += 1;
            continue;
        }
        let start = i;
        while i < rows.len() && is_reactive(&rows[i]) {
            i += 1;
        }
        let group = &rows[start..i];
        // Cumulative-counter baselines: the row BEFORE the group covers
        // everything through the tick preceding it, and the first row
        // AFTER it covers the whole maneuver (accumulation is post-row) —
        // the leg ending mid-maneuver undercounts the final tick, which
        // "Unfinished" status makes honest.
        let dv0 = if start > 0 { rows[start - 1].tcm_dv_mps_cum } else { 0.0 };
        let prop0 = if start > 0 { rows[start - 1].tcm_propellant_kg_cum } else { 0.0 };
        let end_row = rows.get(i).unwrap_or(&rows[i - 1]);
        let after_group_ignition_solve = rows
            .get(i)
            .and_then(|r| r.tcm_solve.as_ref())
            .filter(|s| s.context == "Ignition");
        let trigger_solve = group.iter().filter_map(|r| r.tcm_solve.as_ref()).next();
        let governing = group
            .iter()
            .rev()
            .filter_map(|r| r.tcm_solve.as_ref())
            .find(|s| s.context != "Trigger")
            .or(after_group_ignition_solve)
            .or(trigger_solve);
        let ignition_s = group
            .iter()
            .find(|r| matches!(r.tcm_phase, Some("Burning") | Some("RcsCorrecting")))
            .map(|r| r.t_s);
        let completed_s = rows.get(i).map(|r| r.t_s);
        let aborted = after_group_ignition_solve.is_some_and(|s| s.decision == "AbortedAtIgnition")
            || governing.is_some_and(|s| s.decision == "AbortedAtIgnition");
        let status = if aborted && ignition_s.is_none() {
            "AbortedAtIgnition"
        } else if completed_s.is_none() {
            "Unfinished"
        } else {
            "Completed"
        };
        let actuator = if group.iter().any(|r| r.tcm_phase == Some("RcsCorrecting")) {
            "Rcs"
        } else {
            "MainEngine"
        };
        reports.push(ReactiveBurnReport {
            trigger_epoch_s: group[0].t_s,
            actuator: actuator.to_string(),
            targeting: governing.map(|s| s.targeting.to_string()),
            aim_epoch_s: governing.map(|s| s.aim_epoch_s),
            lambert_dv_mps: trigger_solve.and_then(|s| s.lambert_dv_mps),
            solved_dv_mps: governing.map(|s| s.solved_dv_mps),
            shooting_iterations: governing.and_then(|s| s.iterations),
            shooting_converged: governing.and_then(|s| s.converged),
            predicted_miss_m: governing.and_then(|s| s.miss_m),
            no_burn_miss_m: governing.and_then(|s| s.no_burn_miss_m),
            ignition_s,
            completed_s,
            executed_dv_mps: (end_row.tcm_dv_mps_cum - dv0).max(0.0),
            propellant_kg: (end_row.tcm_propellant_kg_cum - prop0).max(0.0),
            status: status.to_string(),
        });
    }
    reports
}

/// Review C2: empirical interpolation floor of a reference
/// trajectory [m] — leave-one-out cross-validation of the SAME cubic
/// Hermite interpolation the cruise loop uses in production (MANUAL.md
/// §9.1): reconstruct each interior sample from its two NEIGHBORS alone
/// and measure the position error against the real stored sample. The
/// reconstruction spans `2Δt` where production interpolation spans `Δt`;
/// Hermite error scales O(Δt⁴), so the production-spacing floor is
/// estimated as `max_err / 16`. The shape constant varies along the arc,
/// so this is an order-of-magnitude estimate for a WARNING threshold, not
/// a precise bound — which is all `tcm_dr_threshold_m` gating needs.
/// `None` for references with fewer than 3 samples (nothing to
/// leave out). Data-driven — no assumption about what force model
/// produced the reference.
pub fn reference_interpolation_floor_m(reference: &ReferenceTrajectory) -> Option<f64> {
    let pts = reference.points();
    if pts.len() < 3 {
        return None;
    }
    let mut max_err_m = 0.0_f64;
    for i in 1..pts.len() - 1 {
        let two_point = ReferenceTrajectory::new(vec![pts[i - 1].clone(), pts[i + 1].clone()]);
        let (r_pred, _) = two_point.state_at(pts[i].t_s);
        max_err_m = max_err_m.max((r_pred - pts[i].r_m).norm());
    }
    Some(max_err_m / 16.0)
}

/// Same settling-error threshold `/api/design/slew-test` (spacecraft-
/// builder) defaults to — reused here for mode-transition
/// settling-time detection so the two features report comparable numbers.
const DEFAULT_SETTLE_THRESHOLD_DEG: f64 = 0.5;

/// Momentum-dump gain [1/s] for `run_cruise_streaming`/`run_cruise_mc_streaming`'s
/// `MomentumManagementLaw::ThresholdRcs` — hand-picked, same status as
/// `PdGains`, tunable. Passed unconditionally: the law itself gracefully
/// degrades to a no-op when no RCS hardware is configured (see
/// `sim_engine::control::MomentumManagementLaw::ThresholdRcs`'s own doc
/// comment), so the caller doesn't need to branch on hardware presence.
const DEFAULT_MOMENTUM_DUMP_GAIN_PER_S: f64 = 0.02;

/// Null-motion (wheel-speed equalization) gain [1/s] — see
/// `sim_engine::control::MomentumManagementLaw::ThresholdRcs`'s own doc
/// comment for why this is a SEPARATE, always-active term from the
/// momentum-dump gain above (it damps the 4-wheel pyramid's redundant 4th
/// DOF, invisible to `needs_desat`-gated total-momentum control). Found
/// necessary via `cruise_commander_demo`'s real telemetry — a
/// real, secularly growing "internal momentum" component with the dump-only
/// law. Hand-picked, same status as `DEFAULT_MOMENTUM_DUMP_GAIN_PER_S`.
///
/// Value validated via `cruise_gain_sweep_demo` (a parametric
/// sweep against the SAME real ANISE Earth->Mars/comm-pass scenario
/// `cruise_commander_demo` uses — plot: `plot/plot_cruise_gain_sweep.py`,
/// data: `out/cruise_gain_sweep_demo/gain_sweep.csv`), which found the old
/// default of 0.05 was well BELOW a sharp settling threshold: 11 of 18 mode
/// transitions never settled (matching cruise_commander_demo's own earlier
/// finding), and `max_wheel_sat_frac` sat near 0.82 (an individual wheel
/// repeatedly pinning near its rated max_speed during the aggressive
/// slews). Full settling (18/18) first appears at null_motion_gain≈0.22,
/// where `max_wheel_sat_frac` drops to ≈0.19 in lock-step — confirming the
/// mechanism is saturation avoidance (per `control.rs`'s wheel-saturation-
/// zeroing fix, a pinned wheel loses real control authority in that axis),
/// not a direct pointing effect (this term is constructed to produce zero
/// net body torque). 0.3 is the minimum-mean-settling-time point in the
/// fully-settled region (926 s vs. never-settling before), with modest RCS
/// cost (≈0.32 kg over the 3-day demo) — kept well above the ≈0.22
/// threshold for margin, not pinned exactly at it. `DEFAULT_MOMENTUM_DUMP_
/// GAIN_PER_S` was swept over 0.005-0.16 in the same session and showed
/// NO measurable effect on settle rate — left unchanged; this gain governs
/// AGGREGATE momentum, not the per-wheel saturation this scenario turned
/// out to be gated on.
const DEFAULT_NULL_MOTION_GAIN_PER_S: f64 = 0.3;

// ── Priority-ordered multi-rule attitude commander ──────────────────────────

/// One mode's rules, pre-resolved as far as possible at build time: the
/// body-frame vector (from a placed hardware item's boresight/normal) is
/// fixed for the whole run, so it's resolved once here rather than on
/// every tick; the target is re-resolved per tick (`resolve` below) since
/// `Body`/`Velocity` targets move.
struct ModeRules {
    rules: Vec<(Vector3<f64>, PointingTargetConfig, String)>,
    /// Mirrors `GncModeConfig::pointing_locked` (Phase 13m item #4,
    ///) — see [`GncCommander::pointing_locked_at`].
    pointing_locked: bool,
}

/// Runtime-resolved commander, built once per run by [`build_gnc_commander`]
/// from a `MissionConfig`'s `cruise_seed.modes`/`mode_schedule`/`safe_mode`/
/// `body_tracks`. See `docs/MP/MANUAL.md` §9.4 for the governing
/// physics and this module's doc comment for how it plugs into
/// [`run_cruise_leg`].
pub struct GncCommander {
    modes: HashMap<String, ModeRules>,
    /// `(start_s, end_s, mode_name)`, in config/list order. Overlapping
    /// entries: the LAST matching entry wins (see `CruiseSeedConfig::
    /// mode_schedule`'s doc comment).
    schedule: Vec<(f64, f64, String)>,
    safe_mode: Option<String>,
    body_tracks: HashMap<String, ReferenceTrajectory>,
}

impl GncCommander {
    /// Which mode applies at `t_s` — the last schedule entry whose window
    /// contains it, falling back to `safe_mode`, falling back to `None`
    /// (caller then uses its own fixed `pointing_mode`).
    fn active_mode_name(&self, t_s: f64) -> Option<&str> {
        self.schedule
            .iter()
            .rev()
            .find(|(s, e, _)| t_s >= *s && t_s < *e)
            .map(|(_, _, m)| m.as_str())
            .or(self.safe_mode.as_deref())
    }

    /// Item #4 — whether the currently active mode (if any)
    /// marks its pointing non-negotiable, per `GncModeConfig::
    /// pointing_locked`. `false` when no mode is active (falls back to the
    /// caller's own fixed `pointing_mode`, which this executive never
    /// treats as locked).
    fn pointing_locked_at(&self, t_s: f64) -> bool {
        self.active_mode_name(t_s)
            .and_then(|name| self.modes.get(name))
            .map(|m| m.pointing_locked)
            .unwrap_or(false)
    }

    fn resolve_target(&self, target: &PointingTargetConfig, r_m: &Vector3<f64>, v_mps: &Vector3<f64>, sun_pos: Vector3<f64>, t_s: f64) -> Vector3<f64> {
        match target {
            PointingTargetConfig::Sun => sun_pos - r_m,
            PointingTargetConfig::Body { name } => match self.body_tracks.get(name) {
                Some(track) => track.state_at(t_s).0 - r_m,
                // check_config requires a matching body_tracks entry for
                // every referenced Body target; reaching here means this
                // commander was built from an unvalidated config. Fail soft
                // (hold current pointing) rather than panic in a hot loop.
                None => *r_m,
            },
            PointingTargetConfig::Velocity => *v_mps,
            PointingTargetConfig::Inertial { direction } => Vector3::new(direction[0], direction[1], direction[2]),
        }
    }

    /// Resolve the active mode (if any) at `t_s` into a commanded attitude
    /// quaternion and the full per-rule outcome list. `None` when no mode
    /// applies (no schedule coverage and no safe_mode) — caller falls back
    /// to its own fixed pointing mode in that case.
    fn resolve(&self, t_s: f64, r_m: &Vector3<f64>, v_mps: &Vector3<f64>, sun_pos: Vector3<f64>) -> Option<(String, Vector4<f64>, Vec<sim_engine::RuleOutcome>)> {
        let mode_name = self.active_mode_name(t_s)?;
        let mode = self.modes.get(mode_name)?;
        let resolved: Vec<ResolvedRule> = mode
            .rules
            .iter()
            .map(|(body_vec, target, label)| ResolvedRule {
                body_vec: *body_vec,
                target_vec: self.resolve_target(target, r_m, v_mps, sun_pos, t_s),
                label: label.clone(),
            })
            .collect();
        let (q, outcomes) = solve_prioritized_attitude(&resolved);
        Some((mode_name.to_string(), q, outcomes))
    }
}

/// Build a [`GncCommander`] from `cfg.cruise_seed`'s mode/schedule/
/// body-track fields — `None` when `cruise_seed` is absent or has no modes
/// configured (the legacy fixed-`pointing_mode` path applies). Assumes
/// `check_config` has already validated hardware indices/body-track
/// references — does not re-validate.
pub fn build_gnc_commander(cfg: &MissionConfig) -> Option<GncCommander> {
    let seed = cfg.cruise_seed.as_ref()?;
    if seed.modes.is_empty() {
        return None;
    }

    let mut modes = HashMap::new();
    for m in &seed.modes {
        let rules = m
            .rules
            .iter()
            .filter_map(|rule| {
                let item = cfg.spacecraft.hardware.get(rule.hardware_index)?;
                let bv = hardware_pointing_vector(item)?;
                let label = format!("hardware[{}]", rule.hardware_index);
                Some((Vector3::new(bv[0], bv[1], bv[2]), rule.target.clone(), label))
            })
            .collect();
        modes.insert(m.name.clone(), ModeRules { rules, pointing_locked: m.pointing_locked });
    }

    let schedule = seed.mode_schedule.iter().map(|e| (e.start_s, e.end_s, e.mode.clone())).collect();

    let body_tracks = seed
        .body_tracks
        .iter()
        .map(|t| {
            let points: Vec<ReferencePoint> = t
                .track
                .iter()
                .map(|p| ReferencePoint { t_s: p.t_s, r_m: Vector3::new(p.r_m[0], p.r_m[1], p.r_m[2]), v_mps: Vector3::zeros() })
                .collect();
            (t.name.clone(), ReferenceTrajectory::new(points))
        })
        .collect();

    Some(GncCommander { modes, schedule, safe_mode: seed.safe_mode.clone(), body_tracks })
}

/// Heuristic default mode set from placed hardware ("default
/// mode set is derivable from mission type + placed hardware"). Builds at
/// most 3 single-rule modes: `"Cruise"` (a placed, non-articulated
/// `SolarPanel`'s normal toward the Sun — an articulated (`TwoAxis`) panel
/// tracks independently and is correctly NOT given a body-attitude rule at
/// all, per `docs/MP/MANUAL.md` §9.5), `"Comm"` (a placed `CommAntenna`'s
/// boresight toward `Body("Earth")`), `"Science"` (a placed `OpNavCamera`
/// or `Lidar`'s boresight toward `Body(target_body.name)`). Any mode whose
/// one required hardware item isn't placed is simply omitted — the caller
/// gets whatever subset is derivable, not an error.
///
/// The `Body` targets this produces need a matching `body_tracks` entry to
/// actually resolve (see `CruiseSeedConfig::body_tracks`) — this function
/// only derives the RULE structure from hardware, it has no ephemeris
/// access to supply position tracks itself.
pub fn default_modes(cfg: &MissionConfig) -> Vec<GncModeConfig> {
    use crate::config::{HardwareItem, PanelArticulation, PointingRuleConfig};

    let mut modes = Vec::new();

    let cruise_rule = cfg.spacecraft.hardware.iter().enumerate().find_map(|(i, h)| match h {
        HardwareItem::SolarPanel { normal: Some(_), articulation, .. }
            if !matches!(articulation, Some(PanelArticulation::TwoAxis { .. })) =>
        {
            Some(PointingRuleConfig { hardware_index: i, target: PointingTargetConfig::Sun })
        }
        _ => None,
    });
    if let Some(rule) = cruise_rule {
        modes.push(GncModeConfig { name: "Cruise".to_string(), rules: vec![rule], pointing_locked: false });
    }

    let comm_rule = cfg.spacecraft.hardware.iter().enumerate().find_map(|(i, h)| match h {
        HardwareItem::CommAntenna { .. } => Some(PointingRuleConfig {
            hardware_index: i,
            target: PointingTargetConfig::Body { name: "Earth".to_string() },
        }),
        _ => None,
    });
    if let Some(rule) = comm_rule {
        modes.push(GncModeConfig { name: "Comm".to_string(), rules: vec![rule], pointing_locked: false });
    }

    let science_rule = cfg.spacecraft.hardware.iter().enumerate().find_map(|(i, h)| match h {
        HardwareItem::OpNavCamera { boresight: Some(_), .. } | HardwareItem::Lidar { boresight: Some(_), .. } => {
            Some(PointingRuleConfig {
                hardware_index: i,
                target: PointingTargetConfig::Body { name: cfg.target_body.name.clone() },
            })
        }
        _ => None,
    });
    if let Some(rule) = science_rule {
        modes.push(GncModeConfig { name: "Science".to_string(), rules: vec![rule], pointing_locked: false });
    }

    modes
}

/// One mode transition detected in a completed run's tick rows
/// ("mode transitions are real slews with reported time/propellant/
/// wheel cost"). `settling_time_s` is measured from the transition instant,
/// using the same threshold-crossing convention `/api/design/slew-test`
/// already established.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModeTransitionReport {
    pub t_s: f64,
    pub from_mode: Option<String>,
    pub to_mode: String,
    /// `None` if pointing error never dropped to/below `settle_threshold_deg`
    /// before the run ended or the next transition began.
    pub settling_time_s: Option<f64>,
    pub rcs_propellant_kg_used: f64,
    pub max_wheel_momentum_nms: f64,
}

/// Scan a completed run's rows for `active_mode` changes and report each
/// transition's settling time / propellant / wheel cost, measured over the
/// window from the transition up to (a) the next transition or (b) the end
/// of the run, whichever comes first.
pub fn detect_mode_transitions(rows: &[CruiseTickRow], settle_threshold_deg: f64) -> Vec<ModeTransitionReport> {
    let mut reports = Vec::new();
    let mut prev_mode: Option<&str> = None;

    for (i, row) in rows.iter().enumerate() {
        let cur = row.active_mode.as_deref();
        if cur != prev_mode && cur.is_some() {
            if let Some(to_mode) = cur {
                let window_end = rows[i + 1..]
                    .iter()
                    .position(|r| r.active_mode.as_deref() != cur)
                    .map(|k| i + 1 + k)
                    .unwrap_or(rows.len());
                let window = &rows[i..window_end];
                let settle_from = window[0].t_s;
                let settling_time_s = window
                    .iter()
                    .find(|r| r.pointing_error_deg <= settle_threshold_deg)
                    .map(|r| r.t_s - settle_from);
                let rcs_propellant_kg_used =
                    window.last().unwrap().rcs_propellant_kg_cum - window[0].rcs_propellant_kg_cum.max(0.0);
                let max_wheel_momentum_nms = window.iter().map(|r| r.wheel_momentum_nms).fold(0.0_f64, f64::max);
                reports.push(ModeTransitionReport {
                    t_s: row.t_s,
                    from_mode: prev_mode.map(|s| s.to_string()),
                    to_mode: to_mode.to_string(),
                    settling_time_s,
                    rcs_propellant_kg_used,
                    max_wheel_momentum_nms,
                });
            }
        }
        prev_mode = cur;
    }
    reports
}

fn cruise_ref_to_reference_trajectory(cfg: &MissionConfig) -> Result<ReferenceTrajectory, String> {
    let seed = cfg.cruise_seed.as_ref().ok_or("run_cruise_streaming called without cruise_seed set")?;
    let points: Vec<ReferencePoint> = seed
        .reference
        .iter()
        .map(|p| ReferencePoint {
            t_s: p.t_s,
            r_m: Vector3::new(p.r_m[0], p.r_m[1], p.r_m[2]),
            v_mps: Vector3::new(p.v_mps[0], p.v_mps[1], p.v_mps[2]),
        })
        .collect();
    Ok(ReferenceTrajectory::new(points))
}

/// Builds real gravitational-perturber entries from `cruise_seed.
/// body_tracks` — reuses the SAME precomputed position data the attitude
/// commander's `Body`-target pointing resolution already consumes (see
/// `BodyTrackConfig::mu_m3s2`'s own doc comment for why this is one
/// mechanism, not two), instead of a second, ANISE-querying path.
///
/// A track whose body name doesn't resolve a `mu_m3s2` (neither the
/// track's own explicit override nor a `body_models::TargetBody` catalog
/// match) is silently excluded from the returned perturber list — it's
/// still usable for pointing (the commander doesn't call this function),
/// just not as a gravity source. This is why `disperse_hardware_placement`-
/// style "documented, not silent" framing doesn't apply the same way here:
/// omission from THIS list is the expected, correct behavior for a
/// synthetic/non-physical track, not a gap.
///
/// `soi_capture` (fixing a real gap): each track defaults to
/// `soi_radius_m: None` — third-body-only, never a central-body candidate,
/// the ORIGINAL scope decision here (matching what Phase 7k's `design::
/// propagator_body_entries` also does for ITS third-body-only perturbers).
/// A track with `BodyTrackConfig::soi_capture: true` instead gets a real
/// Laplace SOI radius (`trajectory_solver::laplace_soi_radius_m`, the same
/// formula/convention `propagator_body_entries` already uses) and a real
/// `radius_m` — making it a genuine SOI-switching candidate that `step_tick`/
/// `step_tick_with_burn`'s own internal `resolve_central_body` (see
/// `crates/sim_engine::propagator6dof`) can select once the spacecraft
/// enters it, exactly the same way Layer 1's propagator already does. This
/// is what makes a real captured orbit near a body (e.g. Mercury) actually
/// fly as a capture in the cruise replay, instead of coasting straight
/// through under Sun-only gravity the whole time — confirmed as the real
/// root cause of a captured orbit reading as a flyby before this fix.
///
/// SOI sizing: `check_config` guarantees `soi_capture: true` only reaches
/// here with a resolvable `mu_m3s2`. The distance the Laplace formula needs
/// is this body's OWN distance from whatever it orbits (its "primary"),
/// evaluated at the track's first sample (`t_s`'s own t=0, matching the
/// convention `reference`/`r0_m` already use) — heliocentric distance
/// (`|track[0].r_m|`) when the catalog's `primary` is `None` (a body that
/// orbits the Sun directly, e.g. Mercury); distance from the PRIMARY's own
/// track when `primary` is `Some(name)` AND a body_track for that primary
/// is also present (e.g. the Moon needs distance from Earth, not the Sun —
/// same "wrong primary" bug Phase 8h already fixed once for Layer 1's own
/// propagator). If the primary's track isn't present, falls back to the
/// heliocentric distance with a printed warning — an approximation, not a
/// silent one, and correct for the common case this was built for (a
/// Sun-orbiting planet like Mercury has `primary: None` and needs no
/// fallback at all).
///
/// No central-body zonal-harmonic fidelity is applied here (`central_fidelity:
/// None`, point-mass only) — cruise.rs has no per-track gravity-model
/// config the way `target_body.gravity_model` gives Layer 1, and point-mass
/// is the correct, honest default absent one; a future fidelity dial would
/// be additive, not a correctness fix.
///
/// Returns owned `PropagatorBodyEntry`s (each boxes its own `move`d
/// `ReferenceTrajectory`, so the entries are `'static` — no borrow of
/// `seed` survives) — the caller converts to `&[PropagatorBody]` via
/// `design::as_propagator_bodies` immediately before passing to
/// `run_cruise_leg`, same pattern `design.rs`'s own callers use.
fn build_body_track_perturbers(seed: &CruiseSeedConfig) -> Vec<crate::design::PropagatorBodyEntry<'static>> {
    seed.body_tracks
        .iter()
        .filter_map(|t| {
            let catalog = body_models::TargetBody::by_name(&t.name);
            let mu_m3s2 = t.mu_m3s2.or_else(|| catalog.as_ref().map(|b| b.mu_m3s2))?;
            let radius_m = catalog.as_ref().map(|b| b.radius_m);
            let soi_radius_m = if t.soi_capture {
                t.track.first().map(|p0| {
                    let r0 = Vector3::new(p0.r_m[0], p0.r_m[1], p0.r_m[2]);
                    let primary = catalog.as_ref().and_then(|b| b.primary);
                    let (dist_from_primary_m, primary_mu_m3s2) = match primary {
                        Some(primary_name) => {
                            let primary_catalog = body_models::TargetBody::by_name(primary_name);
                            let primary_track_r0 = seed.body_tracks.iter().find(|other| other.name == primary_name).and_then(|other| other.track.first());
                            match (primary_catalog, primary_track_r0) {
                                (Some(pc), Some(p0_primary)) => {
                                    let r0_primary = Vector3::new(p0_primary.r_m[0], p0_primary.r_m[1], p0_primary.r_m[2]);
                                    ((r0 - r0_primary).norm(), pc.mu_m3s2)
                                }
                                _ => {
                                    eprintln!(
                                        "Warning: cruise_seed body_track '{}' has soi_capture=true and a real primary \
                                         ('{primary_name}') per the body_models catalog, but no body_track for that \
                                         primary was supplied -- falling back to heliocentric distance/mu for SOI \
                                         sizing, an approximation.",
                                        t.name
                                    );
                                    (r0.norm(), orbital_models::constants::MU_SUN)
                                }
                            }
                        }
                        None => (r0.norm(), orbital_models::constants::MU_SUN),
                    };
                    let mass_ratio = mu_m3s2 / primary_mu_m3s2;
                    trajectory_solver::laplace_soi_radius_m(dist_from_primary_m, mass_ratio)
                })
            } else {
                None
            };
            let points: Vec<ReferencePoint> = t
                .track
                .iter()
                .map(|p| ReferencePoint {
                    t_s: p.t_s,
                    r_m: Vector3::new(p.r_m[0], p.r_m[1], p.r_m[2]),
                    v_mps: Vector3::new(p.v_mps[0], p.v_mps[1], p.v_mps[2]),
                })
                .collect();
            let traj = ReferenceTrajectory::new(points);
            Some(crate::design::PropagatorBodyEntry {
                name: t.name.clone(),
                mu_m3s2,
                soi_radius_m,
                central_fidelity: None,
                state_at: Box::new(move |t_s: f64| traj.state_at(t_s)),
                radius_m,
            })
        })
        .collect()
}

/// Builds a [`TcmConfig`] from `cruise_seed.tcm_dr_threshold_m` and
/// `spacecraft.propulsion` — `None` unless BOTH are configured (a
/// threshold with no propulsion system has nothing to burn with; a
/// propulsion system with no threshold has nothing to trigger it — both
/// are required, matching the "opt-in, additive" convention every other
/// `cruise_seed` feature in this module follows).
/// Builds the shared main-engine propulsion source both the reactive TCM
/// executive AND Phase 13n's planned-burn executive draw from — `Some` when
/// EITHER `seed.tcm_dr_threshold_m` is set OR `seed.planned_burns` is
/// nonempty (both need `spacecraft.propulsion`; either alone is enough to
/// need this config). When only `planned_burns` is set (no reactive
/// threshold), `dr_threshold_m` is `f64::INFINITY` — dispersion can never
/// exceed it, so the reactive trigger check in `run_cruise_leg` never
/// fires, while the SAME `TcmConfig` still supplies `thrust_n`/`isp_s` for
/// planned burns to fire with.
fn build_tcm_config(cfg: &MissionConfig, seed: &CruiseSeedConfig) -> Option<TcmConfig> {
    if seed.tcm_dr_threshold_m.is_none() && seed.planned_burns.is_empty() {
        return None;
    }
    let dr_threshold_m = seed.tcm_dr_threshold_m.unwrap_or(f64::INFINITY);
    let propulsion = cfg.spacecraft.propulsion.as_ref()?;
    Some(TcmConfig { dr_threshold_m, thrust_n: propulsion.thrust_n, isp_s: propulsion.isp_s })
}

/// Real mount-offset arm for the main engine's thrust-misalignment torque
/// — found via `propagator6dof.rs`'s own
/// `BurnConfig::thrust_offset_body_m` / `RotOde`'s `tau_misalign` already
/// being real, tested physics that every TCM burn call site simply fed
/// `Vector3::zeros()`, silently flying every mission with a perfectly
/// CoM-aligned engine regardless of the vehicle's real mass distribution.
///
/// `PropulsionConfig` has no real mount-position field yet (that's the
/// still-open half of the builder design) — the best available
/// assumption today is the same one the frontend's own Phase 02 viewport
/// visual uses: the engine sits at the bus's geometric-center -X face
/// (body +X is the thrust/Δv direction, so the engine and its exhaust are
/// aft, at -X, per the real rocket convention -- matches `BurnConfig`'s
/// `body_dir: Vector3::new(1.0, 0.0, 0.0)` above). The arm is that mount
/// point minus the real derived CoM (`vehicle_properties::
/// compute_vehicle_properties`, the same function Table 2/CoM-marker
/// already use) -- NOT the geometric center itself, since the misalignment
/// torque this feeds depends on the true torque arm to the center of mass,
/// not to the bus's own geometric origin (the same origin-vs-CoM
/// distinction already flagged for placed hardware, now applying
/// here too). Update this once a real mount-position field exists.
/// `pub(crate)` since also consumed by `server::routes::
/// vehicle`'s review-E5 static feasibility check, which must use the SAME
/// torque arm the live simulation actually flies with.
pub(crate) fn main_engine_thrust_offset_body_m(cfg: &MissionConfig) -> Vector3<f64> {
    let vp = crate::vehicle_properties::compute_vehicle_properties(cfg);
    let mount = Vector3::new(-cfg.spacecraft.bus_dims_m[0] / 2.0, 0.0, 0.0);
    let com = Vector3::new(vp.com_m[0], vp.com_m[1], vp.com_m[2]);
    mount - com
}

/// Streaming entry point for a `cruise_seed`-bearing `/api/simulate` job —
/// the async-job counterpart of `run_cruise_leg`, following the exact
/// `Result<T, String>` + `on_row -> bool` (cancellation) conventions
/// `simulate::run_streaming` already established, so
/// `server/routes/simulate.rs` can dispatch to either with the same
/// pattern. Assumes `check_config` has already validated `cruise_seed`
/// (non-empty/sorted reference, matching r0/v0, positive duration/tick) —
/// does not re-validate here.
///
/// **Force model** (updated): third-body perturbers are now
/// real when `cruise_seed.body_tracks` resolves any (see
/// [`build_body_track_perturbers`]) — still no SOI-switching central-body
/// reassignment, by deliberate scope decision (see that function's own
/// doc comment), unlike Layer 1's `design::propagator_body_entries`, which
/// does support it. **Known first-cut limitation, still open**: no TCM
/// burn execution (dispersion is reported, never corrected) — flagged as
/// a fast-follow in the design notes Phase 5.2 entry, not a silent gap.
/// Pointing is `SunPointing` for the whole leg UNLESS `cruise_seed.modes`
/// is non-empty, in which case the attitude commander
/// drives pointing instead (see [`build_gnc_commander`]).
///
/// `cruise_seed.report_stride` decouples how often a row is
/// relayed to `on_row` (the routes-layer WS broadcast + `/steps` history)
/// from the real control tick — see that field's own doc comment. The
/// control loop, `max_dr_m`/`max_wheel_momentum_nms`, the on-disk CSV, and
/// `mode_transitions` are all unaffected by it; only the external relay is
/// decimated.
/// The mode-scheduled burn tick [s] for a seed (§10.5.1):
/// `cruise_seed.burn_tick_s` when set (validated `0 < bt <= tick_s`), else
/// `max(tick_s/10, 1 s)`, never longer than `tick_s`.
fn resolve_burn_tick_s(seed: &CruiseSeedConfig) -> f64 {
    seed.burn_tick_s
        .unwrap_or((seed.tick_s / 10.0).max(1.0))
        .min(seed.tick_s)
        .max(1e-6)
}

pub fn run_cruise_streaming(
    cfg: &MissionConfig,
    on_row: &mut dyn FnMut(&CruiseTickRow) -> bool,
) -> Result<CruiseResult, String> {
    let seed = cfg.cruise_seed.as_ref().ok_or("run_cruise_streaming called without cruise_seed set")?;
    let reference = cruise_ref_to_reference_trajectory(cfg)?;

    let sc = build_spacecraft_properties(cfg);
    let wheel_cluster = wheel_cluster_from_hardware(cfg);
    let (rcs_thrusters, _) = rcs_from_hardware(cfg);
    let controls = AttitudeControlSet::from_cfg(
        cfg, &sc, &wheel_cluster, &rcs_thrusters, seed.tick_s, resolve_burn_tick_s(seed),
    );
    let commander = build_gnc_commander(cfg);

    // Test window (`CruiseSeedConfig::window`): start at the
    // reference state at `start_s` (+ optional dispersion) and stop at
    // `end_s`; planned burns already behind `start_s` are dropped. Without
    // a window this is byte-identical to the whole-leg run.
    let (t_start_s, t_end_s) = seed.window.as_ref().map(|w| (w.start_s, w.end_s)).unwrap_or((0.0, seed.duration_s));
    let (r0, v0) = match seed.window.as_ref() {
        Some(w) => {
            let (r_ref, v_ref) = reference.state_at(w.start_s);
            let dr = w.initial_dr_m.map(|d| Vector3::new(d[0], d[1], d[2])).unwrap_or_else(Vector3::zeros);
            let dv = w.initial_dv_mps.map(|d| Vector3::new(d[0], d[1], d[2])).unwrap_or_else(Vector3::zeros);
            (r_ref + dr, v_ref + dv)
        }
        None => (
            Vector3::new(seed.r0_m[0], seed.r0_m[1], seed.r0_m[2]),
            Vector3::new(seed.v0_m[0], seed.v0_m[1], seed.v0_m[2]),
        ),
    };
    let planned_burns_active: Vec<crate::config::PlannedBurnConfig> =
        seed.planned_burns.iter().filter(|b| b.epoch_s >= t_start_s).cloned().collect();
    let initial = SixDofState {
        t_s: t_start_s,
        r_m: r0,
        v_mps: v0,
        q: desired_quaternion_cruise(CruisePointingMode::SunPointing, &r0, &Vector3::zeros()),
        omega_radps: Vector3::zeros(),
        wheel_speeds_radps: [0.0; 4],
        mass_kg: sc.mass_kg,
    };

    let mut max_dr_m = 0.0_f64;
    let mut max_wheel_momentum_nms = 0.0_f64;

    let perturber_entries = build_body_track_perturbers(seed);
    let bodies = crate::design::as_propagator_bodies(&perturber_entries);
    let tcm = build_tcm_config(cfg, seed);
    let thrust_offset_body_m = main_engine_thrust_offset_body_m(cfg);

    // Reporting-cadence decoupling -- gate the
    // relay to the ROUTES-layer `on_row` (WS broadcast + `/steps` history)
    // by `report_stride`, without touching `run_cruise_leg`'s own signature
    // or its per-tick fidelity at all: `row` is still computed by the real
    // control loop every `tick_s`, and `max_dr_m`/`max_wheel_momentum_nms`
    // below are still updated from EVERY tick (relay-gating only wraps the
    // call to `on_row`, not the max-tracking that precedes it) -- so this
    // field only shrinks what gets streamed/stored per-tick externally,
    // never the physics or the returned `rows` (which still backs the
    // full-fidelity on-disk CSV and `mode_transitions` below). Tick 0 and
    // the final tick are always relayed regardless of stride, so the
    // reported series still starts/ends exactly where the real one does.
    let report_stride = seed.report_stride.unwrap_or(1).max(1) as u64;
    // Phase-adaptive reporting: maneuver ticks
    // (any Slewing/Burning/RcsCorrecting phase, plus a trailing window
    // after every tcm_phase/active_mode transition) relay at their own
    // stride — default 1, full control-loop resolution — while coast keeps
    // the coarse budgeted stride. A whole slew used to fall between two
    // reported samples on a months-long mission (~20k-point budget at
    // tick 10 s ≈ one sample per several minutes).
    let maneuver_stride = seed.report_stride_maneuver.unwrap_or(1).max(1) as u64;
    let mut last_phase: Option<&'static str> = None;
    let mut last_mode: Option<String> = None;
    let mut dense_until_s = f64::NEG_INFINITY;
    let n_ticks = ((t_end_s - t_start_s) / seed.tick_s).ceil().max(1.0) as u64;
    let mut tick_idx: u64 = 0;

    // Review D5: body-state lookup for planned capture burns.
    let capture_tracks = build_capture_track_table(seed);
    // the capture solve targets the CONFIGURED capture orbit's
    // periapsis speed — thread `[trajectory.capture].capture_eccentricity`
    // through (the executive half of Phase 14f; a circular-targeting solve
    // burned 472 kg against a 740 m/s eccentric-capture job, measured live).
    let capture_ecc = cfg.trajectory.capture.as_ref().map(|c| c.capture_eccentricity).unwrap_or(0.0);
    let capture_state_at = move |name: &str, t_s: f64| {
        capture_tracks.iter().find(|(n, _, _)| n == name).map(|(_, tr, mu)| {
            let (r, v) = tr.state_at(t_s);
            (r, v, *mu, capture_ecc)
        })
    };

    let rows = run_cruise_leg(
        initial, &reference, &bodies, orbital_models::constants::MU_SUN, &sc, &wheel_cluster,
        &rcs_thrusters, &controls, ControlMode::WheelsPrimary,
        MomentumManagementLaw::ThresholdRcs {
            gain: DEFAULT_MOMENTUM_DUMP_GAIN_PER_S,
            null_motion_gain: DEFAULT_NULL_MOTION_GAIN_PER_S,
        },
        CruisePointingMode::SunPointing,
        commander.as_ref(),
        &|_t_s| Vector3::zeros(), seed.tick_s, t_end_s, cfg.spacecraft.propellant_mass_kg,
        tcm, cfg.simulation.rtol, cfg.simulation.atol, thrust_offset_body_m, &planned_burns_active,
        &capture_state_at,
        &mut |row| {
            max_dr_m = max_dr_m.max(row.dr_m);
            max_wheel_momentum_nms = max_wheel_momentum_nms.max(row.wheel_momentum_nms);
            // With the mode-scheduled burn tick the loop no longer runs a
            // fixed tick count, so "last tick" is detected from the row's
            // own time (within one nominal tick of the end), not from
            // tick_idx against a precomputed n_ticks.
            let is_last_tick = tick_idx + 1 >= n_ticks || row.t_s >= t_end_s - seed.tick_s;
            // Phase-adaptive stride: a transition tick always
            // relays and opens the trailing dense window.
            let transition = row.tcm_phase != last_phase || row.active_mode != last_mode;
            if transition {
                last_phase = row.tcm_phase;
                last_mode = row.active_mode.clone();
                dense_until_s = row.t_s + MANEUVER_REPORT_TRAIL_S;
            }
            // Inside a registered SOI-capture body's sphere (parking coast,
            // capture orbit) the dynamics are orders of magnitude faster
            // than cruise — a coarse cruise stride rendered a parking orbit
            // as a jagged polygon (~1 sample per 3-4 ORBITS) and made the
            // capture-orbit attitude replay swing 10-30° between samples
            // — so those ticks use the maneuver
            // stride too.
            let near_body = bodies.iter().any(|b| {
                b.soi_radius_m.is_some_and(|rs| (row.r_m - (b.state_at)(row.t_s).0).norm() < rs)
            });
            let in_maneuver = row.tcm_phase.is_some() || near_body || row.t_s <= dense_until_s;
            let stride_now = if in_maneuver { maneuver_stride } else { report_stride };
            // A tick carrying a gain-schedule point is always relayed
            // — the schedule is sparse and a stride must not
            // hide the tick a law changed on.
            // A tick carrying a reactive correction solve always relays
            // — a DECLINED solve leaves the phase in
            // Coast, so nothing else would force that tick through a
            // coarse cruise stride.
            let should_relay = tick_idx % stride_now == 0
                || transition
                || is_last_tick
                || row.gain_schedule_point.is_some()
                || row.tcm_solve.is_some();
            let keep_going = if should_relay { on_row(row) } else { true };
            tick_idx += 1;
            keep_going
        },
    );
    let mode_transitions = detect_mode_transitions(&rows, DEFAULT_SETTLE_THRESHOLD_DEG);
    let planned_burn_reports = detect_planned_burn_reports(&rows, &planned_burns_active, &capture_state_at);
    let reactive_burn_reports = detect_reactive_burn_reports(&rows);
    let gain_schedule_points: Vec<GainSchedulePoint> = rows.iter().filter_map(|r| r.gain_schedule_point.clone()).collect();
    let attitude_control_effective = controls.effective();

    // Review C2: never let TCM chase numbers below the
    // reference's own interpolation-noise floor — warn (not error: the run
    // itself is still valid, the threshold is just partly meaningless).
    let reference_interpolation_floor_m = reference_interpolation_floor_m(&reference);
    let mut warnings: Vec<String> = controls.warnings().to_vec();
    for w in &warnings {
        eprintln!("Warning: {w}");
    }
    if let (Some(threshold), Some(floor)) = (seed.tcm_dr_threshold_m, reference_interpolation_floor_m) {
        if threshold < floor {
            let w = format!(
                "tcm_dr_threshold_m ({threshold:.1} m) is below the submitted reference's own \
                 interpolation floor (~{floor:.1} m at its actual sample spacing) — TCM will partly \
                 chase interpolation noise rather than real dispersion; raise the threshold or \
                 supply a denser reference"
            );
            eprintln!("Warning: {w}");
            warnings.push(w);
        }
    }

    let last = rows.last().ok_or("cruise loop produced no ticks")?;

    let out_dir = format!("{}/cruise", cfg.simulation.output_dir.trim_end_matches('/'));
    let _ = std::fs::create_dir_all(&out_dir);
    let cruise_csv_path = format!("{out_dir}/cruise.csv");
    let mut csv_rows = vec![
        "t_s,x_m,y_m,z_m,dr_m,dv_mps,pointing_error_deg,wheel_momentum_nms,w1,w2,w3,w4,\
         rcs_propellant_kg_cum,wheel_sat_frac,propellant_remaining_kg,torque_gravity_gradient_nm,\
         torque_srp_nm,accel_central_gravity_mps2,accel_third_body_mps2,accel_srp_mps2,\
         tcm_phase,tcm_propellant_kg_cum,tcm_dv_mps_cum"
            .to_string(),
    ];
    for row in &rows {
        csv_rows.push(format!(
            "{:.3},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6},{:.6},{:.4},{:.4},{:.4},{:.4},{:.6e},\
             {:.6},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{},{:.6e},{:.6e}",
            row.t_s, row.r_m.x, row.r_m.y, row.r_m.z, row.dr_m, row.dv_mps, row.pointing_error_deg,
            row.wheel_momentum_nms, row.wheel_speeds_radps[0], row.wheel_speeds_radps[1],
            row.wheel_speeds_radps[2], row.wheel_speeds_radps[3], row.rcs_propellant_kg_cum,
            row.wheel_sat_frac, row.propellant_remaining_kg, row.torque_gravity_gradient_nm,
            row.torque_srp_nm, row.accel_central_gravity_mps2, row.accel_third_body_mps2,
            row.accel_srp_mps2, row.tcm_phase.unwrap_or(""), row.tcm_propellant_kg_cum,
            row.tcm_dv_mps_cum,
        ));
    }
    let _ = std::fs::write(&cruise_csv_path, csv_rows.join("\n") + "\n");

    Ok(CruiseResult {
        final_r_m: [last.r_m.x, last.r_m.y, last.r_m.z],
        final_v_mps: [last.v_mps.x, last.v_mps.y, last.v_mps.z],
        final_dr_m: last.dr_m,
        final_dv_mps: last.dv_mps,
        max_dr_m,
        max_wheel_momentum_nms,
        rcs_propellant_kg_used: last.rcs_propellant_kg_cum,
        tcm_propellant_kg_used: last.tcm_propellant_kg_cum,
        final_propellant_remaining_kg: last.propellant_remaining_kg,
        cruise_csv_path,
        mode_transitions,
        reference_interpolation_floor_m,
        warnings,
        planned_burn_reports,
        reactive_burn_reports,
        attitude_control_effective,
        gain_schedule_points,
    })
}

// ── Phase 5.2 follow-up): Monte Carlo over
// vehicle uncertainty ───────────────────────────────────────────────────────

/// One completed Monte Carlo run's result, sent over `/stream` when the job
/// was started with `cruise_seed` set AND `simulation.monte_carlo_runs > 0`
/// — mirrors `simulate::McRunMsg`'s "one message per completed run, never
/// per-tick" convention).
///
/// `msg_type` (added) — see `CruiseStepMsg::msg_type`'s doc
/// comment for why this exists: always `"cruise_mc_run"` here, so a client
/// can tell this apart from `CruiseStepMsg` at parse time instead of
/// trusting that it requested the right job kind.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CruiseMcRunMsg {
    pub msg_type: &'static str,
    pub run_index: usize,
    pub seed: u64,
    /// The dispersed mass actually used for this run [kg].
    pub mass_kg_used: f64,
    /// The dispersed SRP reflectivity actually used for this run — `None`
    /// when the vehicle's SRP model isn't `Cannonball` (reflectivity
    /// dispersion only applies to that model; see this module's doc
    /// comment on the first-cut scope).
    pub reflectivity_used: Option<f64>,
    /// The dispersed inertia diagonal actually used for this run
    /// [kg*m^2] — unlike mass/reflectivity, this has a REAL effect on
    /// every tracked attitude metric today (gravity-gradient torque and
    /// the attitude dynamics themselves both depend on inertia, not mass —
    /// see this module's doc comment for why mass/reflectivity dispersion
    /// alone was found to be a silent no-op).
    pub inertia_diag_kgm2_used: [f64; 3],
    pub final_dr_m: f64,
    pub max_dr_m: f64,
    pub final_pointing_error_deg: f64,
    pub max_wheel_momentum_nms: f64,
    pub propellant_remaining_kg: f64,
}

/// Final summary once all cruise Monte Carlo runs complete — analogous to
/// `simulate::McSummaryResult` for the body-centric MC path.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CruiseMcSummaryResult {
    pub n_runs: usize,
    pub mean_max_dr_m: f64,
    pub min_max_dr_m: f64,
    pub max_max_dr_m: f64,
    pub mc_summary_csv_path: String,
}

/// SplitMix64 + Box-Muller Gaussian — same RNG construction as
/// `simulate.rs::run_single_mc_with_traj` (no external RNG dependency,
/// deliberately duplicated in miniature rather than sharing code across
/// two independently-evolving MC implementations for two different mission
/// loops).
fn gaussian_from_seed(seed: u64) -> f64 {
    let mut s = seed.wrapping_add(0x9e3779b97f4a7c15);
    let rng_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        (z as i64 as f64) / (i64::MAX as f64)
    };
    let u = (rng_f64(&mut s) + 1.0) * 0.5;
    let v = rng_f64(&mut s);
    let r = (-2.0 * u.max(1e-15).ln()).sqrt();
    r * (2.0 * std::f64::consts::PI * v).cos()
}

/// 2% 1-sigma — a representative as-built manufacturing/measurement
/// uncertainty (not sourced from a specific mission spec; matches the
/// order of magnitude typical SMAD-class mass-property margins use).
const MC_SIGMA_MASS_FRAC: f64 = 0.02;
/// 2% 1-sigma on the Cannonball SRP reflectivity coefficient.
const MC_SIGMA_REFLECTIVITY_FRAC: f64 = 0.02;
/// 2% 1-sigma per inertia axis, independently.
const MC_SIGMA_INERTIA_FRAC: f64 = 0.02;
/// 5 mm 1-sigma placement uncertainty for RCS thrusters — a representative
/// spacecraft-integration mechanical tolerance (not sourced from a specific
/// mission spec, same hand-picked status as the sigmas above).
const MC_SIGMA_RCS_POSITION_M: f64 = 0.005;
/// 0.5 deg 1-sigma RCS thrust-vector misalignment — a real, operationally
/// meaningful GNC uncertainty source: a misaligned thruster couples torque
/// into axes the allocation law didn't intend, and can inject a small net
/// translational force during what's meant to be a pure attitude
/// correction. Typical thruster-alignment tolerance order of magnitude.
const MC_SIGMA_RCS_ALIGNMENT_DEG: f64 = 0.5;
/// 1 cm 1-sigma placement uncertainty for CustomPlate/placed-SolarPanel
/// geometry — same manufacturing-tolerance framing as
/// `MC_SIGMA_RCS_POSITION_M`, applied to SRP-relevant plates instead.
const MC_SIGMA_PLATE_POSITION_M: f64 = 0.01;
/// 1 deg 1-sigma plate-normal misalignment (deployment/mounting
/// tolerance) — affects SRP torque only under the FlatPlate model
/// (Cannonball is attitude-independent by construction, same caveat as
/// MC_SIGMA_REFLECTIVITY_FRAC).
const MC_SIGMA_PLATE_ALIGNMENT_DEG: f64 = 1.0;

/// Perturbs a body-frame direction vector by a small random rotation —
/// Gaussian noise added to two axes orthogonal to the input, then
/// renormalized to the input's own length. Correct to first order for
/// `sigma_rad` small (a few degrees or less, which is the regime every
/// caller here uses it in — real mechanical alignment tolerances, not
/// large reorientations).
fn perturb_direction(dir: Vector3<f64>, sigma_rad: f64, seed: u64) -> Vector3<f64> {
    let n = dir.norm();
    if n < 1e-12 {
        return dir;
    }
    let u = dir / n;
    let arbitrary = if u.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
    let e1 = u.cross(&arbitrary).normalize();
    let e2 = u.cross(&e1);
    let dtheta1 = sigma_rad * gaussian_from_seed(seed);
    let dtheta2 = sigma_rad * gaussian_from_seed(seed.wrapping_add(1));
    (u + dtheta1 * e1 + dtheta2 * e2).normalize() * n
}

/// Disperses `RcsThruster`/`CustomPlate`/placed-`SolarPanel` PLACEMENT
/// fields (position + orientation) in place — the follow-up this
/// module's own doc comment flagged as needing `MissionConfig: Clone`
/// (added specifically to unblock this). Every other hardware
/// item (sensors/antennas, unplaced entries) is left untouched: sensor
/// noise magnitudes don't depend on placement (see the design notes
/// note on this), and boresight/mount misalignment dispersion for
/// commander-referenced hardware is real future scope, documented but not
/// built here — kept to the two axes with an already-established real
/// effect on tracked outputs (RCS control authority/propellant efficiency;
/// SRP torque under FlatPlate), matching this module's existing "real
/// effect first" precedent for inertia over mass/reflectivity dispersion.
fn disperse_hardware_placement(hardware: &mut [HardwareItem], seed: u64) {
    for (i, item) in hardware.iter_mut().enumerate() {
        let base = seed.wrapping_add(100 + 10 * i as u64);
        match item {
            HardwareItem::RcsThruster { position_m, direction, .. } => {
                let mut pos = Vector3::new(position_m[0], position_m[1], position_m[2]);
                for (k, p) in pos.iter_mut().enumerate() {
                    *p += MC_SIGMA_RCS_POSITION_M * gaussian_from_seed(base.wrapping_add(k as u64));
                }
                *position_m = [pos.x, pos.y, pos.z];

                let dir = Vector3::new(direction[0], direction[1], direction[2]);
                let dir = perturb_direction(dir, MC_SIGMA_RCS_ALIGNMENT_DEG.to_radians(), base.wrapping_add(3));
                *direction = [dir.x, dir.y, dir.z];
            }
            HardwareItem::CustomPlate { center_offset_m, normal, .. } => {
                let mut pos = Vector3::new(center_offset_m[0], center_offset_m[1], center_offset_m[2]);
                for (k, p) in pos.iter_mut().enumerate() {
                    *p += MC_SIGMA_PLATE_POSITION_M * gaussian_from_seed(base.wrapping_add(k as u64));
                }
                *center_offset_m = [pos.x, pos.y, pos.z];

                let n = Vector3::new(normal[0], normal[1], normal[2]);
                let n = perturb_direction(n, MC_SIGMA_PLATE_ALIGNMENT_DEG.to_radians(), base.wrapping_add(3));
                *normal = [n.x, n.y, n.z];
            }
            HardwareItem::SolarPanel { position_m: Some(position_m), normal: Some(normal), .. } => {
                let mut pos = Vector3::new(position_m[0], position_m[1], position_m[2]);
                for (k, p) in pos.iter_mut().enumerate() {
                    *p += MC_SIGMA_PLATE_POSITION_M * gaussian_from_seed(base.wrapping_add(k as u64));
                }
                *position_m = [pos.x, pos.y, pos.z];

                let n = Vector3::new(normal[0], normal[1], normal[2]);
                let n = perturb_direction(n, MC_SIGMA_PLATE_ALIGNMENT_DEG.to_radians(), base.wrapping_add(3));
                *normal = [n.x, n.y, n.z];
            }
            // Unplaced SolarPanel, or any other hardware kind — no
            // placement to disperse.
            _ => {}
        }
    }
}

/// Monte Carlo over vehicle uncertainty for a `cruise_seed`-bearing run —
/// dispatched when `simulation.monte_carlo_runs > 0` is ALSO set alongside
/// `cruise_seed` (Phase 5.2 initially ignored `monte_carlo_runs` in that
/// combination; this is the follow-up that implements it).
///
/// **Scope, documented not silent**: disperses `spacecraft.mass_kg`, its
/// inertia diagonal, and (for a `Cannonball`-SRP vehicle) its reflectivity
/// coefficient — all applied directly to a cloned `SpacecraftProperties` —
/// PLUS (added, `MissionConfig` gained `Clone` specifically to
/// unblock this) real `HardwareItem` PLACEMENT dispersion via
/// [`disperse_hardware_placement`]: `RcsThruster` position + thrust-vector
/// alignment, and `CustomPlate`/placed-`SolarPanel` position + normal
/// alignment. Still NOT dispersed: sensor/antenna boresight/position
/// (`StarTracker`/`OpNavCamera`/`Lidar`/`CommAntenna`) — real future scope
/// (mount misalignment would move the attitude commander's resolved
/// pointing target for any mode referencing that hardware), but kept out
/// of this pass to match the same "real effect on tracked outputs first"
/// precedent that already governs the mass/inertia/reflectivity split
/// below — RCS/plate geometry has an already-established real effect
/// (control authority, SRP torque under FlatPlate); sensor boresight
/// dispersion would need a new tracked output (a pointing-vs-commanded
/// error attributable to mount misalignment specifically) to be similarly
/// verifiable, which doesn't exist yet.
///
/// **Real finding from live-verifying the first version of this function
/// (mass + reflectivity only, no inertia): `max_dr_m` came back BIT-FOR-BIT
/// IDENTICAL across every run, despite mass/reflectivity genuinely
/// dispersing.** Not a bug — a direct, correct consequence of the physics
/// this loop actually models: translation under `step_tick`'s decoupled
/// path is gravity-only (mass-independent by the equivalence principle,
/// same reason a feather and a hammer fall together) and has no burns
/// (mass would matter for ΔV/Tsiolkovsky, but none are executed here); SRP
/// is not wired into translation at all (documented gap); and — the sharper
/// point — `Cannonball` SRP produces ZERO torque BY DEFINITION (that's the
/// whole point of the cannonball simplification), so dispersing its
/// reflectivity can never show up in pointing error or wheel momentum
/// either, for ANY vehicle using that model. Mass/reflectivity dispersion
/// is real and correctly threaded through, but in THIS physics model
/// (no burns, decoupled attitude, Cannonball SRP) it is a structural
/// no-op on every currently-tracked output — it only becomes meaningful
/// once burns or SRP-on-translation land. Inertia dispersion was added
/// specifically to give this function a REAL, visible effect today: both
/// gravity-gradient torque and the attitude dynamics themselves (Euler's
/// equations) depend on inertia, not mass. **Confirmed live, precisely**:
/// a real server smoke test (8 runs, 1 AU heliocentric SunPointing coast)
/// showed genuine per-run variation in `max_wheel_momentum_nms` (~2.7e-12
/// to 3.3e-12 N*m*s — small, because gravity-gradient torque at 1 AU is
/// itself tiny, but real and inertia-correlated). `final_pointing_error_deg`
/// stayed exactly 0.0 across every run in that same test — NOT because
/// inertia dispersion has no effect, but because the quaternion-PD
/// controller fully rejects a disturbance this small within one control
/// tick, at the 4-decimal precision reported; the wheel-momentum field is
/// therefore the more sensitive signal for this dispersion axis at
/// quiescent-cruise disturbance scales, not pointing error.
///
/// Threading mirrors `simulate::run_monte_carlo_streaming` exactly
/// (`thread::scope`, each run built and joined independently, `on_run_done`
/// called from the calling thread in run order). Shared/borrowed across
/// threads: the reference trajectory, wheel cluster, commander, gains,
/// initial position/velocity — identical across runs by design. Rebuilt
/// fresh PER RUN instead: `SpacecraftProperties` (mass/inertia/reflectivity
/// AND, since real dispersed RCS-thruster/plate geometry via a
/// per-run cloned `MissionConfig`), the RCS thruster list, and the
/// third-body perturber list (not dispersed, but its `PropagatorBody::
/// state_at` closures aren't `Sync`, so a single shared instance can't
/// cross the `thread::scope` boundary — see [`build_body_track_perturbers`]).
pub fn run_cruise_mc_streaming(
    cfg: &MissionConfig,
    on_run_done: &mut dyn FnMut(CruiseMcRunMsg),
) -> Result<CruiseMcSummaryResult, String> {
    let n = cfg.simulation.monte_carlo_runs as usize;
    if n == 0 {
        return Err("monte_carlo_runs is 0 -- nothing to run".to_string());
    }
    let seed_cfg = cfg.cruise_seed.as_ref().ok_or("run_cruise_mc_streaming called without cruise_seed set")?;
    let reference = cruise_ref_to_reference_trajectory(cfg)?;

    let wheel_cluster = wheel_cluster_from_hardware(cfg);
    // Attitude-control registry is built PER RUN below from the dispersed
    // mass properties/RCS layout — a dispersed inertia must
    // re-derive the gains, exactly as a real vehicle's would be re-tuned.
    // Shared across every run -- neither depends on RcsThruster/CustomPlate/
    // SolarPanel placement (the two axes disperse_hardware_placement varies):
    // wheel sizing has no geometry concept at all (as its own
    // note), and the commander's referenced hardware here is sensors/
    // antennas' POINTING vectors, not dispersed this pass (see
    // disperse_hardware_placement's own doc comment on scope).
    let commander = build_gnc_commander(cfg);
    let tcm = build_tcm_config(cfg, seed_cfg);

    let r0 = Vector3::new(seed_cfg.r0_m[0], seed_cfg.r0_m[1], seed_cfg.r0_m[2]);
    let v0 = Vector3::new(seed_cfg.v0_m[0], seed_cfg.v0_m[1], seed_cfg.v0_m[2]);
    let q0 = desired_quaternion_cruise(CruisePointingMode::SunPointing, &r0, &Vector3::zeros());

    let run_one = |run_index: usize| -> CruiseMcRunMsg {
        let seed = run_index as u64;

        // Geometry dispersion (RCS placement/alignment, plate placement/
        // alignment) needs its own per-run MissionConfig, since it feeds
        // build_spacecraft_properties' SRP plate list and rcs_from_hardware's
        // thruster list -- both derived FROM hardware placement, not
        // overridable as a scalar the way mass/inertia/reflectivity are.
        let mut dispersed_cfg = cfg.clone();
        disperse_hardware_placement(&mut dispersed_cfg.spacecraft.hardware, seed);
        let mut sc = build_spacecraft_properties(&dispersed_cfg);
        let (rcs_thrusters, _) = rcs_from_hardware(&dispersed_cfg);
        // Real per-run CoM, since dispersed hardware placement shifts it --
        // see main_engine_thrust_offset_body_m's own doc comment.
        let thrust_offset_body_m = main_engine_thrust_offset_body_m(&dispersed_cfg);

        // Rebuilt per run (not hoisted, unlike commander/wheel_cluster/
        // reference) rather than shared by reference across threads --
        // `PropagatorBody`'s `state_at: &dyn Fn(...)` isn't declared
        // `+ Sync`, so a single shared instance can't cross the
        // thread::scope boundary below; body_tracks aren't dispersed
        // anyway, so each thread just redoes this cheap, small-trajectory
        // build independently.
        let perturber_entries = build_body_track_perturbers(seed_cfg);
        let bodies = crate::design::as_propagator_bodies(&perturber_entries);

        let mass_kg_used = (sc.mass_kg * (1.0 + MC_SIGMA_MASS_FRAC * gaussian_from_seed(seed))).max(1.0);
        sc.mass_kg = mass_kg_used;
        let reflectivity_used = if let SrpTruthModel::Cannonball { c_r, .. } = &mut sc.srp {
            *c_r = (*c_r * (1.0 + MC_SIGMA_REFLECTIVITY_FRAC * gaussian_from_seed(seed.wrapping_add(1)))).max(0.0);
            Some(*c_r)
        } else {
            None
        };
        for (axis, i) in sc.inertia_diag_kgm2.iter_mut().zip(0u64..) {
            *axis = (*axis * (1.0 + MC_SIGMA_INERTIA_FRAC * gaussian_from_seed(seed.wrapping_add(2 + i)))).max(1e-6);
        }
        let inertia_diag_kgm2_used = [sc.inertia_diag_kgm2.x, sc.inertia_diag_kgm2.y, sc.inertia_diag_kgm2.z];

        let initial = SixDofState {
            t_s: 0.0, r_m: r0, v_mps: v0, q: q0, omega_radps: Vector3::zeros(),
            wheel_speeds_radps: [0.0; 4], mass_kg: mass_kg_used,
        };

        let mut max_dr_m = 0.0_f64;
        let mut max_wheel_momentum_nms = 0.0_f64;
        let controls = AttitudeControlSet::from_cfg(
            cfg, &sc, &wheel_cluster, &rcs_thrusters, seed_cfg.tick_s, resolve_burn_tick_s(seed_cfg),
        );
        let rows = run_cruise_leg(
            initial, &reference, &bodies, orbital_models::constants::MU_SUN, &sc, &wheel_cluster,
            &rcs_thrusters, &controls, ControlMode::WheelsPrimary,
            MomentumManagementLaw::ThresholdRcs {
                gain: DEFAULT_MOMENTUM_DUMP_GAIN_PER_S,
                null_motion_gain: DEFAULT_NULL_MOTION_GAIN_PER_S,
            },
            CruisePointingMode::SunPointing,
            commander.as_ref(),
            &|_t_s| Vector3::zeros(), seed_cfg.tick_s, seed_cfg.duration_s, cfg.spacecraft.propellant_mass_kg,
            tcm, cfg.simulation.rtol, cfg.simulation.atol, thrust_offset_body_m, &seed_cfg.planned_burns,
            // Review D5: same capture-burn body-state lookup as the
            // single-run path (built per MC run — cheap relative to the
            // propagation itself, and keeps the closure thread-local).
            &{
                let capture_tracks = build_capture_track_table(seed_cfg);
                let capture_ecc = cfg.trajectory.capture.as_ref().map(|c| c.capture_eccentricity).unwrap_or(0.0);
                move |name: &str, t_s: f64| {
                    capture_tracks.iter().find(|(n, _, _)| n == name).map(|(_, tr, mu)| {
                        let (r, v) = tr.state_at(t_s);
                        (r, v, *mu, capture_ecc)
                    })
                }
            },
            &mut |row| {
                max_dr_m = max_dr_m.max(row.dr_m);
                max_wheel_momentum_nms = max_wheel_momentum_nms.max(row.wheel_momentum_nms);
                true
            },
        );
        let last = rows.last();

        CruiseMcRunMsg {
            msg_type: "cruise_mc_run",
            run_index,
            seed,
            mass_kg_used,
            reflectivity_used,
            inertia_diag_kgm2_used,
            final_dr_m: last.map(|r| r.dr_m).unwrap_or(0.0),
            max_dr_m,
            final_pointing_error_deg: last.map(|r| r.pointing_error_deg).unwrap_or(0.0),
            max_wheel_momentum_nms,
            propellant_remaining_kg: last.map(|r| r.propellant_remaining_kg).unwrap_or(cfg.spacecraft.propellant_mass_kg),
        }
    };

    let mut results: Vec<CruiseMcRunMsg> = Vec::with_capacity(n);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..n).map(|i| s.spawn(move || run_one(i))).collect();
        results = handles.into_iter().filter_map(|h| h.join().ok()).collect();
    });
    results.sort_by_key(|r| r.run_index);

    for r in &results {
        on_run_done(r.clone());
    }

    let max_dr_values: Vec<f64> = results.iter().map(|r| r.max_dr_m).collect();
    let mean_max_dr_m = if max_dr_values.is_empty() { 0.0 } else { max_dr_values.iter().sum::<f64>() / max_dr_values.len() as f64 };
    let min_max_dr_m = max_dr_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_max_dr_m = max_dr_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let out_dir = format!("{}/cruise_mc", cfg.simulation.output_dir.trim_end_matches('/'));
    let _ = std::fs::create_dir_all(&out_dir);
    let mc_summary_csv_path = format!("{out_dir}/cruise_mc_summary.csv");
    let mut csv_rows = vec![
        "run_index,seed,mass_kg_used,reflectivity_used,ixx_kgm2,iyy_kgm2,izz_kgm2,final_dr_m,max_dr_m,\
         final_pointing_error_deg,max_wheel_momentum_nms,propellant_remaining_kg"
            .to_string(),
    ];
    for r in &results {
        csv_rows.push(format!(
            "{},{},{:.6},{},{:.6},{:.6},{:.6},{:.6e},{:.6e},{:.6},{:.6e},{:.6}",
            r.run_index, r.seed, r.mass_kg_used,
            r.reflectivity_used.map(|v| format!("{v:.6}")).unwrap_or_default(),
            r.inertia_diag_kgm2_used[0], r.inertia_diag_kgm2_used[1], r.inertia_diag_kgm2_used[2],
            r.final_dr_m, r.max_dr_m, r.final_pointing_error_deg, r.max_wheel_momentum_nms,
            r.propellant_remaining_kg,
        ));
    }
    let _ = std::fs::write(&mc_summary_csv_path, csv_rows.join("\n") + "\n");

    Ok(CruiseMcSummaryResult {
        n_runs: results.len(),
        mean_max_dr_m,
        min_max_dr_m,
        max_max_dr_m,
        mc_summary_csv_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::check_config;

    /// Minimal, self-contained `MissionConfig` with a real `cruise_seed` —
    /// no ANISE/kernel dependency (pure heliocentric two-body physics), so
    /// this test runs everywhere `cargo test` runs, unlike `cruise_demo`.
    /// The reference trajectory is built with the SAME
    /// `sample_reference_trajectory_uniform` helper a real caller would
    /// use, over a short (1 h) heliocentric-scale (1 AU) circular-ish
    /// starting state, at the SAME `MU_SUN` `run_cruise_streaming` itself
    /// assumes (Phase 5.2's documented first-cut limitation).
    fn cruise_seeded_config() -> MissionConfig {
        let mu = orbital_models::constants::MU_SUN;
        let r = 1.495_98e11_f64;
        let v_circ = (mu / r).sqrt();
        let r0 = Vector3::new(r, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);

        let duration_s = 3600.0;
        let tick_s = 20.0;
        let reference = sample_reference_trajectory_uniform(r0, v0, mu, &[], duration_s, 300.0, 1e-10, 1e-3);

        let reference_json: Vec<serde_json::Value> = reference
            .points()
            .iter()
            .map(|p| {
                serde_json::json!({
                    "t_s": p.t_s,
                    "r_m": [p.r_m.x, p.r_m.y, p.r_m.z],
                    "v_mps": [p.v_mps.x, p.v_mps.y, p.v_mps.z],
                })
            })
            .collect();

        let v = serde_json::json!({
            "mission": { "name": "cruise_seed test", "objective": "Orbit" },
            "target_body": { "name": "Bennu", "ephemeris": "Keplerian" },
            "spacecraft": {
                "mass_kg": 400.0, "dry_mass_kg": 320.0, "propellant_mass_kg": 80.0,
                "bus_dims_m": [1.0, 1.0, 1.2], "inertia_diag_kgm2": [60.0, 60.0, 40.0],
                "srp_model": "Cannonball",
            },
            "trajectory": { "phases": ["Cruise"], "solver": "Hohmann", "departure_body": "Earth" },
            "gnc": { "navigation_filter": "EKF", "pointing_mode": "Nadir", "attitude_controller": "ReactionWheelPD" },
            "simulation": {
                "integrator": "DormandPrince45", "rtol": 1.0e-9, "atol": 1.0e-7,
                "dt_truth_s": 10.0, "dt_meas_s": 120.0, "monte_carlo_runs": 0,
                "output_dir": "out/cruise_seed_test/",
            },
            "cruise_seed": {
                "r0_m": [r0.x, r0.y, r0.z],
                "v0_m": [v0.x, v0.y, v0.z],
                "reference": reference_json,
                "duration_s": duration_s,
                "tick_s": tick_s,
            },
        });
        serde_json::from_value(v).expect("cruise_seeded_config should deserialize")
    }

    /// Same base scenario as [`cruise_seeded_config`], plus a real
    /// hardware list (a `CommAntenna` boresight +z, a `SolarPanel` normal
    /// +x, both at index-known positions) and a two-mode schedule:
    /// `"Cruise"` (panel -> Sun) for the first half of the run, `"Comm"`
    /// (antenna -> a fixed synthetic "Earth" track) for the second half —
    /// enough to exercise mode selection, target resolution (Sun AND
    /// Body), and transition detection all in one fixture.
    fn cruise_seeded_config_with_commander() -> MissionConfig {
        let mu = orbital_models::constants::MU_SUN;
        let r = 1.495_98e11_f64;
        let v_circ = (mu / r).sqrt();
        let r0 = Vector3::new(r, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);

        let duration_s = 3600.0;
        let tick_s = 20.0;
        let reference = sample_reference_trajectory_uniform(r0, v0, mu, &[], duration_s, 300.0, 1e-10, 1e-3);
        let reference_json: Vec<serde_json::Value> = reference
            .points()
            .iter()
            .map(|p| serde_json::json!({ "t_s": p.t_s, "r_m": [p.r_m.x, p.r_m.y, p.r_m.z], "v_mps": [p.v_mps.x, p.v_mps.y, p.v_mps.z] }))
            .collect();

        // A fixed synthetic "Earth" position, far off in +y -- real motion
        // doesn't matter for this test, only that Body-target resolution
        // and the resulting commanded attitude are correct.
        let earth_pos = Vector3::new(0.0, 1.0e11, 0.0);
        let earth_track = vec![
            serde_json::json!({ "t_s": 0.0, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
            serde_json::json!({ "t_s": duration_s, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
        ];

        let v = serde_json::json!({
            "mission": { "name": "cruise_seed commander test", "objective": "Orbit" },
            "target_body": { "name": "Bennu", "ephemeris": "Keplerian" },
            "spacecraft": {
                "mass_kg": 400.0, "dry_mass_kg": 320.0, "propellant_mass_kg": 80.0,
                "bus_dims_m": [1.0, 1.0, 1.2], "inertia_diag_kgm2": [60.0, 60.0, 40.0],
                "srp_model": "Cannonball",
                "hardware": [
                    { "type": "CommAntenna", "boresight": [0.0, 0.0, 1.0], "beamwidth_deg": 20.0 },
                    { "type": "SolarPanel", "area_m2": 4.0, "position_m": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0] },
                ],
            },
            "trajectory": { "phases": ["Cruise"], "solver": "Hohmann", "departure_body": "Earth" },
            "gnc": { "navigation_filter": "EKF", "pointing_mode": "Nadir", "attitude_controller": "ReactionWheelPD" },
            "simulation": {
                "integrator": "DormandPrince45", "rtol": 1.0e-9, "atol": 1.0e-7,
                "dt_truth_s": 10.0, "dt_meas_s": 120.0, "monte_carlo_runs": 0,
                "output_dir": "out/cruise_seed_commander_test/",
            },
            "cruise_seed": {
                "r0_m": [r0.x, r0.y, r0.z],
                "v0_m": [v0.x, v0.y, v0.z],
                "reference": reference_json,
                "duration_s": duration_s,
                "tick_s": tick_s,
                "modes": [
                    { "name": "Cruise", "rules": [{ "hardware_index": 1, "target": { "type": "Sun" } }] },
                    { "name": "Comm", "rules": [{ "hardware_index": 0, "target": { "type": "Body", "name": "Earth" } }] },
                ],
                "mode_schedule": [
                    { "start_s": 0.0, "end_s": duration_s / 2.0, "mode": "Cruise" },
                    { "start_s": duration_s / 2.0, "end_s": duration_s, "mode": "Comm" },
                ],
                "body_tracks": [{ "name": "Earth", "track": earth_track }],
            },
        });
        serde_json::from_value(v).expect("cruise_seeded_config_with_commander should deserialize")
    }

    #[test]
    fn commander_config_passes_check_config() {
        let cfg = cruise_seeded_config_with_commander();
        let errors = check_config(&cfg);
        assert!(errors.is_empty(), "unexpected validation errors: {errors:?}");
    }

    #[test]
    fn default_modes_derives_cruise_and_comm_from_fixed_panel_and_antenna() {
        let cfg = cruise_seeded_config_with_commander();
        let modes = default_modes(&cfg);
        let names: Vec<&str> = modes.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"Cruise"), "expected a Cruise mode from the fixed SolarPanel, got {names:?}");
        assert!(names.contains(&"Comm"), "expected a Comm mode from the CommAntenna, got {names:?}");
        assert!(!names.contains(&"Science"), "no OpNavCamera/Lidar in this fixture, should not derive Science");
    }

    /// Per `docs/MP/MANUAL.md` §9.5: a `TwoAxis`-articulated panel
    /// tracks the Sun independently of body attitude, so it must NOT be
    /// given a body-attitude rule the way a fixed panel is.
    #[test]
    fn default_modes_excludes_two_axis_articulated_panel_from_cruise_rule() {
        let mut cfg = cruise_seeded_config_with_commander();
        if let crate::config::HardwareItem::SolarPanel { articulation, .. } = &mut cfg.spacecraft.hardware[1] {
            *articulation = Some(crate::config::PanelArticulation::TwoAxis {
                axis1: [0.0, 1.0, 0.0],
                axis2: [0.0, 0.0, 1.0],
            });
        } else {
            panic!("fixture hardware[1] should be the SolarPanel");
        }
        let modes = default_modes(&cfg);
        let names: Vec<&str> = modes.iter().map(|m| m.name.as_str()).collect();
        assert!(!names.contains(&"Cruise"), "a TwoAxis panel should not get an automatic Cruise body-attitude rule, got {names:?}");
        assert!(names.contains(&"Comm"), "Comm should still be derived independently, got {names:?}");
    }

    #[test]
    fn commander_check_config_rejects_bad_hardware_index() {
        let mut cfg = cruise_seeded_config_with_commander();
        cfg.cruise_seed.as_mut().unwrap().modes[0].rules[0].hardware_index = 99;
        let errors = check_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("out of range")), "got: {errors:?}");
    }

    #[test]
    fn commander_check_config_rejects_unresolvable_body_target() {
        let mut cfg = cruise_seeded_config_with_commander();
        cfg.cruise_seed.as_mut().unwrap().body_tracks.clear();
        let errors = check_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("no matching")), "got: {errors:?}");
    }

    /// End-to-end: run the full cruise loop with a real two-mode schedule
    /// and confirm (1) the active_mode reported per tick matches the
    /// schedule, (2) the achieved attitude actually satisfies each mode's
    /// rule (panel normal -> Sun in the first half, antenna boresight ->
    /// Earth in the second), and (3) exactly one transition is detected at
    /// the schedule midpoint.
    #[test]
    fn run_cruise_streaming_with_commander_follows_the_schedule() {
        let cfg = cruise_seeded_config_with_commander();
        assert!(check_config(&cfg).is_empty());

        let mut rows: Vec<CruiseTickRow> = Vec::new();
        let result = run_cruise_streaming(&cfg, &mut |row| {
            rows.push(row.clone());
            true
        })
        .expect("run_cruise_streaming should succeed");

        let midpoint = 1800.0;
        let before: Vec<&CruiseTickRow> = rows.iter().filter(|r| r.t_s < midpoint - 1.0).collect();
        let after: Vec<&CruiseTickRow> = rows.iter().filter(|r| r.t_s > midpoint + 1.0).collect();
        assert!(!before.is_empty() && !after.is_empty());

        assert!(before.iter().all(|r| r.active_mode.as_deref() == Some("Cruise")));
        assert!(after.iter().all(|r| r.active_mode.as_deref() == Some("Comm")));

        // Panel normal (+x body) should end up close to the Sun direction
        // (-r_hat, since Sun sits at the heliocentric origin) during Cruise,
        // and antenna boresight (+z body) close to Earth during Comm.
        //
        // Tolerance note: this fixture reuses the config's real (low,
        // slew-test-tuned) PD gains -- the SAME gains #8's live server test
        // showed converging a 90 deg initial
        // error to only 2.3 deg after 1800 s. Switching from "panel at Sun"
        // to "antenna at Earth" at the schedule midpoint is a comparably
        // large reorientation, so a real, honest bound here is "clearly
        // converging, not stuck at the initial/uncommanded attitude" --
        // NOT sub-degree precision within the same window, which the first
        // version of this test wrongly assumed (a real, single-rule q_cmd
        // IS exact by construction -- verified separately in
        // `attitude_commander::tests::single_rule_is_exact_and_controlling`
        // -- the residual here is genuinely controller settling time, not
        // a commander bug).
        let last_cruise = before.last().unwrap();
        let panel_normal_inertial = orbital_models::attitude::body_to_inertial(&last_cruise.q, &Vector3::new(1.0, 0.0, 0.0));
        let sun_dir = -last_cruise.r_m.normalize();
        let panel_err_deg = panel_normal_inertial.dot(&sun_dir).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(panel_err_deg < 10.0, "panel should be converging toward the Sun during Cruise, err={panel_err_deg} deg");

        let last_comm = after.last().unwrap();
        let antenna_boresight_inertial = orbital_models::attitude::body_to_inertial(&last_comm.q, &Vector3::new(0.0, 0.0, 1.0));
        let earth_dir = (Vector3::new(0.0, 1.0e11, 0.0) - last_comm.r_m).normalize();
        let antenna_err_deg = antenna_boresight_inertial.dot(&earth_dir).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(antenna_err_deg < 10.0, "antenna should be converging toward Earth during Comm, err={antenna_err_deg} deg");

        // Real convergence, not just "small by luck": error should shrink
        // monotonically-ish across the Comm window (first vs. last row).
        let first_comm_err = after.first().unwrap().pointing_error_deg;
        let last_comm_err = after.last().unwrap().pointing_error_deg;
        assert!(
            last_comm_err < first_comm_err,
            "expected pointing error to decrease over the Comm window: first={first_comm_err} last={last_comm_err}"
        );

        // Two transitions, correctly: entering "Cruise" from "no mode" at
        // t=0 is itself a real transition, then the genuine Cruise->Comm
        // switch at the schedule midpoint.
        assert_eq!(result.mode_transitions.len(), 2, "got {:?}", result.mode_transitions);
        assert_eq!(result.mode_transitions[0].from_mode, None);
        assert_eq!(result.mode_transitions[0].to_mode, "Cruise");
        assert!(result.mode_transitions[0].t_s.abs() < 1e-6);

        let t = &result.mode_transitions[1];
        assert_eq!(t.from_mode.as_deref(), Some("Cruise"));
        assert_eq!(t.to_mode, "Comm");
        assert!((t.t_s - midpoint).abs() < tick_s_for_test());
    }

    fn tick_s_for_test() -> f64 { 20.0 }

    #[test]
    fn cruise_seed_passes_check_config() {
        let cfg = cruise_seeded_config();
        let errors = check_config(&cfg);
        assert!(errors.is_empty(), "unexpected validation errors: {errors:?}");
    }

    #[test]
    fn check_config_rejects_mismatched_seed_and_reference_start() {
        let mut cfg = cruise_seeded_config();
        cfg.cruise_seed.as_mut().unwrap().r0_m[0] += 1_000_000.0; // 1000 km off
        let errors = check_config(&cfg);
        assert!(
            errors.iter().any(|e| e.contains("must match r0_m/v0_m")),
            "expected a mismatched-start error, got: {errors:?}"
        );
    }

    #[test]
    fn check_config_rejects_duration_exceeding_reference_span() {
        let mut cfg = cruise_seeded_config();
        cfg.cruise_seed.as_mut().unwrap().duration_s = 1e9;
        let errors = check_config(&cfg);
        assert!(
            errors.iter().any(|e| e.contains("exceeds the reference trajectory's own span")),
            "expected a duration-exceeds-span error, got: {errors:?}"
        );
    }

    /// End-to-end: parse -> validate -> run -> result, the same path
    /// `server/routes/simulate.rs::start_cruise_job` exercises over HTTP.
    /// Since the reference was built from the SAME initial state the loop
    /// flies from, under the SAME force model (heliocentric gravity only,
    /// no burns), dispersion should stay small throughout — this is
    /// Review C2: the empirical interpolation floor must (a)
    /// be positive for a genuinely curved reference, (b) shrink with
    /// denser sampling (the whole point of a resolution floor), and (c)
    /// be None for a 2-sample reference (nothing to leave out).
    #[test]
    fn reference_interpolation_floor_shrinks_with_denser_sampling() {
        // Circular heliocentric orbit at 1 AU — real curved motion with
        // exact analytic samples.
        let mu = orbital_models::constants::MU_SUN;
        let r_au = 1.495_978_707e11_f64;
        let omega = (mu / r_au.powi(3)).sqrt();
        let make_ref = |n: usize, span_s: f64| {
            let pts: Vec<sim_engine::ReferencePoint> = (0..n)
                .map(|i| {
                    let t_s = span_s * i as f64 / (n - 1) as f64;
                    let th = omega * t_s;
                    sim_engine::ReferencePoint {
                        t_s,
                        r_m: Vector3::new(r_au * th.cos(), r_au * th.sin(), 0.0),
                        v_mps: Vector3::new(-r_au * omega * th.sin(), r_au * omega * th.cos(), 0.0),
                    }
                })
                .collect();
            ReferenceTrajectory::new(pts)
        };
        let span_s = 30.0 * 86_400.0; // 30 days
        let coarse = reference_interpolation_floor_m(&make_ref(11, span_s)).unwrap();
        let fine = reference_interpolation_floor_m(&make_ref(101, span_s)).unwrap();
        assert!(coarse > 0.0, "curved motion must have a positive floor");
        assert!(
            fine < coarse / 100.0,
            "10x denser sampling should shrink an O(dt^4) floor by ~10^4: coarse={coarse:.3e}, fine={fine:.3e}"
        );
        assert!(reference_interpolation_floor_m(&make_ref(2, span_s)).is_none());
    }

    /// exactly the composition-correctness signal `cruise_demo` found
    /// (a few meters over a real multi-day leg); here, over a much shorter
    /// synthetic leg, it should be smaller still.
    #[test]
    fn run_cruise_streaming_tracks_its_own_reference_closely() {
        let cfg = cruise_seeded_config();
        assert!(check_config(&cfg).is_empty());

        let mut n_rows = 0usize;
        let result = run_cruise_streaming(&cfg, &mut |_row| {
            n_rows += 1;
            true
        })
        .expect("run_cruise_streaming should succeed");

        assert_eq!(n_rows, (3600.0_f64 / 20.0).ceil() as usize);
        assert!(result.max_dr_m < 100.0, "unexpectedly large dispersion: {} m", result.max_dr_m);
        assert!(result.final_r_m.iter().all(|x| x.is_finite()));
        assert!(std::path::Path::new(&result.cruise_csv_path).exists());
    }

    /// `report_stride` — confirms the three-way
    /// split this field is supposed to produce: the EXTERNAL relay (what
    /// `on_row` here receives, i.e. what the routes layer would broadcast/
    /// store for `/steps`) shrinks by the stride factor (plus always tick 0
    /// and the final tick), while `max_dr_m` (computed every tick
    /// regardless, per `run_cruise_streaming`'s doc comment) and the
    /// on-disk CSV (built from the full-fidelity `rows` returned by
    /// `run_cruise_leg`, never decimated) both stay at full fidelity —
    /// verified by comparing against a stride=1 run of the identical
    /// scenario, not just asserting a shrunk count in isolation.
    #[test]
    fn run_cruise_streaming_report_stride_shrinks_relay_only() {
        // Unique output_dir per config -- cruise_seeded_config()'s shared
        // default "out/cruise_seed_test/" is written by several other
        // tests running concurrently in the same process; reusing it here
        // (twice, for the strided and full runs) would race on the same
        // CSV file and produce a nondeterministic line count unrelated to
        // report_stride itself.
        let mut cfg = cruise_seeded_config();
        cfg.cruise_seed.as_mut().unwrap().report_stride = Some(20);
        cfg.simulation.output_dir = "out/cruise_report_stride_test_strided/".to_string();
        assert!(check_config(&cfg).is_empty());

        let mut n_relayed = 0usize;
        let strided = run_cruise_streaming(&cfg, &mut |_row| {
            n_relayed += 1;
            true
        })
        .expect("run_cruise_streaming should succeed");

        let n_ticks = (3600.0_f64 / 20.0).ceil() as usize; // 180
        // Multiples of 20 in [0, 180): 0,20,...,160 -> 9, plus the final
        // tick (179, not itself a multiple of 20) -> 10.
        assert_eq!(n_relayed, 10, "expected the relay to shrink to 10 rows for stride=20 over {n_ticks} ticks");

        let mut cfg_full = cruise_seeded_config();
        cfg_full.cruise_seed.as_mut().unwrap().report_stride = None;
        cfg_full.simulation.output_dir = "out/cruise_report_stride_test_full/".to_string();
        let mut n_full = 0usize;
        let full = run_cruise_streaming(&cfg_full, &mut |_row| {
            n_full += 1;
            true
        })
        .expect("run_cruise_streaming should succeed");
        assert_eq!(n_full, n_ticks, "sanity: stride=None must still relay every tick");

        // max_dr_m/max_wheel_momentum_nms must be IDENTICAL between the two
        // runs -- the whole point of gating only the relay, not the
        // max-tracking closure, which sees every tick in both cases.
        assert_eq!(
            strided.max_dr_m, full.max_dr_m,
            "max_dr_m must not depend on report_stride -- it is computed from every tick regardless"
        );
        assert_eq!(strided.max_wheel_momentum_nms, full.max_wheel_momentum_nms);

        // The on-disk CSV must also be full-fidelity regardless of stride
        // (built from `run_cruise_leg`'s own returned `rows`, which
        // `report_stride` never touches) -- one header line + n_ticks data
        // lines.
        let csv = std::fs::read_to_string(&strided.cruise_csv_path).expect("cruise CSV should exist");
        let n_csv_lines = csv.lines().count();
        assert_eq!(n_csv_lines, n_ticks + 1, "CSV must stay full-fidelity ({n_ticks} rows + header) regardless of report_stride");
    }

    #[test]
    fn check_config_rejects_zero_report_stride() {
        let mut cfg = cruise_seeded_config();
        cfg.cruise_seed.as_mut().unwrap().report_stride = Some(0);
        let errors = check_config(&cfg);
        assert!(
            errors.iter().any(|e| e.contains("report_stride")),
            "expected a report_stride validation error, got: {errors:?}"
        );
    }

    /// The `on_row` cancellation contract (mirrors `simulate::run_streaming`
    /// and the `/cancel` endpoint's use of it) — returning `false` after a
    /// handful of rows must stop the loop early, not run to completion.
    #[test]
    fn run_cruise_streaming_stops_early_when_on_row_returns_false() {
        let cfg = cruise_seeded_config();
        let mut n_rows = 0usize;
        let _ = run_cruise_streaming(&cfg, &mut |_row| {
            n_rows += 1;
            n_rows < 5
        });
        assert_eq!(n_rows, 5);
    }

    /// the design notes backlog item #10: MC over vehicle mass/reflectivity for a
    /// cruise-seeded run. Confirms real dispersion (not every run getting
    /// the identical nominal mass), a real summary CSV, and that every
    /// run's own mass is genuinely used by the physics (not just recorded)
    /// — a Cannonball-SRP vehicle at 1 AU also gets real dispersion in its
    /// reflectivity coefficient.
    #[test]
    fn run_cruise_mc_streaming_disperses_mass_reflectivity_and_inertia() {
        let mut cfg = cruise_seeded_config();
        cfg.simulation.monte_carlo_runs = 8;

        let mut n_runs_seen = 0usize;
        let mut masses = Vec::new();
        let mut reflectivities = Vec::new();
        let mut inertias = Vec::new();
        let result = run_cruise_mc_streaming(&cfg, &mut |run| {
            n_runs_seen += 1;
            masses.push(run.mass_kg_used);
            reflectivities.push(run.reflectivity_used);
            inertias.push(run.inertia_diag_kgm2_used);
        })
        .expect("run_cruise_mc_streaming should succeed");

        assert_eq!(n_runs_seen, 8);
        assert_eq!(result.n_runs, 8);

        let nominal_mass = 400.0;
        assert!(
            masses.iter().any(|m| (*m - nominal_mass).abs() > 0.1),
            "expected real mass dispersion across runs, got: {masses:?}"
        );
        assert!(masses.iter().all(|m| (*m - nominal_mass).abs() < 0.2 * nominal_mass), "dispersion should be small relative to nominal");

        assert!(reflectivities.iter().all(|r| r.is_some()), "Cannonball SRP should always report a reflectivity_used");
        let refl_values: Vec<f64> = reflectivities.iter().map(|r| r.unwrap()).collect();
        assert!(
            refl_values.iter().any(|r| (*r - 1.4).abs() > 1e-6),
            "expected real reflectivity dispersion across runs, got: {refl_values:?}"
        );

        let nominal_inertia = [60.0, 60.0, 40.0];
        assert!(
            inertias.iter().any(|i| (0..3).any(|k| (i[k] - nominal_inertia[k]).abs() > 0.01)),
            "expected real inertia dispersion across runs, got: {inertias:?}"
        );

        assert!(std::path::Path::new(&result.mc_summary_csv_path).exists());
    }

    /// Direct unit test of the geometry-dispersion helper itself, rather
    /// than trying to detect it indirectly through tracked outputs —
    /// `CruiseMcRunMsg` doesn't report per-hardware-item geometry (only
    /// scalar summary fields), so this is the reliable way to confirm real
    /// per-seed variation, matching the same "expected real dispersion
    /// across runs" rigor as the mass/reflectivity/inertia test above.
    #[test]
    fn disperse_hardware_placement_varies_rcs_and_plate_geometry_across_seeds() {
        let make_hardware = || {
            vec![
                HardwareItem::RcsThruster {
                    thrust_n: 1.0,
                    position_m: [0.5, 0.0, 0.0],
                    direction: [0.0, 1.0, 0.0],
                    mass_kg: None,
                    isp_s: None,
                },
                HardwareItem::CustomPlate {
                    normal: [0.0, 0.0, 1.0],
                    area_m2: 2.0,
                    center_offset_m: [0.0, 0.5, 0.0],
                    rho_s: None,
                    rho_d: None,
                    double_sided: None,
                    mass_kg: None,
                },
                // Unplaced SolarPanel -- must be left untouched (no
                // position_m/normal to disperse).
                HardwareItem::SolarPanel {
                    area_m2: 4.0,
                    efficiency: None,
                    position_m: None,
                    normal: None,
                    width_m: None,
                    height_m: None,
                    articulation: None,
                    rho_s: None,
                    rho_d: None,
                    mass_kg: None,
                },
            ]
        };

        let mut hw_a = make_hardware();
        let mut hw_b = make_hardware();
        disperse_hardware_placement(&mut hw_a, 1);
        disperse_hardware_placement(&mut hw_b, 2);

        let (pos_a, dir_a) = match &hw_a[0] {
            HardwareItem::RcsThruster { position_m, direction, .. } => (*position_m, *direction),
            _ => panic!("expected RcsThruster"),
        };
        let (pos_b, dir_b) = match &hw_b[0] {
            HardwareItem::RcsThruster { position_m, direction, .. } => (*position_m, *direction),
            _ => panic!("expected RcsThruster"),
        };
        assert_ne!(pos_a, pos_b, "different seeds should disperse RCS position differently");
        assert_ne!(dir_a, dir_b, "different seeds should disperse RCS direction differently");
        // Direction must stay a real unit vector after perturbation.
        let dir_a_norm = (dir_a[0].powi(2) + dir_a[1].powi(2) + dir_a[2].powi(2)).sqrt();
        assert!((dir_a_norm - 1.0).abs() < 1e-9, "perturbed direction should stay unit length, got norm {dir_a_norm}");

        let (center_a, normal_a) = match &hw_a[1] {
            HardwareItem::CustomPlate { center_offset_m, normal, .. } => (*center_offset_m, *normal),
            _ => panic!("expected CustomPlate"),
        };
        let (center_b, normal_b) = match &hw_b[1] {
            HardwareItem::CustomPlate { center_offset_m, normal, .. } => (*center_offset_m, *normal),
            _ => panic!("expected CustomPlate"),
        };
        assert_ne!(center_a, center_b, "different seeds should disperse plate position differently");
        assert_ne!(normal_a, normal_b, "different seeds should disperse plate normal differently");

        // Unplaced SolarPanel must be left completely untouched.
        match &hw_a[2] {
            HardwareItem::SolarPanel { position_m, normal, .. } => {
                assert!(position_m.is_none());
                assert!(normal.is_none());
            }
            _ => panic!("expected SolarPanel"),
        }

        // Small dispersions only -- sigma is 5mm (RCS position), so a
        // perturbation should never move it by more than a generous
        // 10-sigma sanity bound from the nominal [0.5, 0.0, 0.0].
        let nominal_pos = [0.5_f64, 0.0, 0.0];
        for k in 0..3 {
            assert!(
                (pos_a[k] - nominal_pos[k]).abs() < 10.0 * MC_SIGMA_RCS_POSITION_M,
                "RCS position dispersion implausibly large on axis {k}: {pos_a:?}",
            );
        }
    }

    /// Integration-level smoke test: `run_cruise_mc_streaming` must run
    /// cleanly end-to-end when the vehicle has real placed `RcsThruster`/
    /// `CustomPlate` hardware for `disperse_hardware_placement` to act on
    /// (unlike the base `cruise_seeded_config()` fixture, which has none) —
    /// the geometry-dispersion unit test above confirms the dispersion
    /// itself varies correctly; this confirms the whole per-run rebuild
    /// path (`build_spacecraft_properties`/`rcs_from_hardware` against a
    /// cloned+dispersed config) doesn't break the loop.
    #[test]
    fn run_cruise_mc_streaming_runs_cleanly_with_placed_rcs_and_plate_hardware() {
        let mut cfg = cruise_seeded_config();
        cfg.simulation.monte_carlo_runs = 4;
        cfg.spacecraft.srp_model = crate::config::SrpModel::FlatPlate;
        cfg.spacecraft.hardware = vec![
            HardwareItem::RcsThruster {
                thrust_n: 1.0,
                position_m: [0.5, 0.0, 0.0],
                direction: [0.0, 1.0, 0.0],
                mass_kg: None,
                isp_s: None,
            },
            HardwareItem::CustomPlate {
                normal: [0.0, 0.0, 1.0],
                area_m2: 2.0,
                center_offset_m: [0.0, 0.5, 0.0],
                rho_s: None,
                rho_d: None,
                double_sided: None,
                mass_kg: None,
            },
        ];

        let mut n_runs_seen = 0usize;
        let result = run_cruise_mc_streaming(&cfg, &mut |_run| {
            n_runs_seen += 1;
        })
        .expect("run_cruise_mc_streaming should succeed with real RCS/plate hardware present");

        assert_eq!(n_runs_seen, 4);
        assert_eq!(result.n_runs, 4);
    }

    /// the design notes item 4 (third-body perturbers via `body_tracks`) end to
    /// end: a real `body_tracks` entry resolving a real catalog `mu_m3s2`
    /// must genuinely perturb the flown trajectory, not just sit there
    /// unused. Places a synthetic "Earth" track ~1e9 m from the
    /// spacecraft's own path (a real, working third-body distance, just
    /// closer than the genuine Earth-spacecraft separation would be on an
    /// actual interplanetary leg, so the effect is unambiguous within this
    /// test's short 1 h window) and confirms the flown trajectory diverges
    /// measurably from the perturber-free baseline the OTHER reference-
    /// tracking test asserts stays under 100 m.
    #[test]
    fn run_cruise_streaming_is_perturbed_by_a_real_body_track() {
        let baseline_cfg = cruise_seeded_config();
        let baseline = run_cruise_streaming(&baseline_cfg, &mut |_row| true).expect("baseline run should succeed");

        let mut cfg = cruise_seeded_config();
        let r0 = Vector3::new(baseline_cfg.cruise_seed.as_ref().unwrap().r0_m[0], 0.0, 0.0);
        let earth_pos = r0 + Vector3::new(1.0e9, 0.0, 0.0);
        let earth_track = vec![
            serde_json::json!({ "t_s": 0.0, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
            serde_json::json!({ "t_s": 3600.0, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
        ];
        cfg.cruise_seed.as_mut().unwrap().body_tracks = vec![
            serde_json::from_value(serde_json::json!({ "name": "Earth", "track": earth_track })).unwrap(),
        ];
        assert!(check_config(&cfg).is_empty());

        let perturbed = run_cruise_streaming(&cfg, &mut |_row| true).expect("perturbed run should succeed");

        assert!(
            perturbed.max_dr_m > 10.0 * baseline.max_dr_m,
            "a real nearby third-body perturber should measurably diverge the trajectory: baseline max_dr_m={}, perturbed max_dr_m={}",
            baseline.max_dr_m, perturbed.max_dr_m,
        );
        assert!(perturbed.final_r_m.iter().all(|x| x.is_finite()));
    }

    /// A `body_tracks` entry for a body NOT in the catalog (and with no
    /// explicit `mu_m3s2`) must be silently excluded from perturbation —
    /// still usable for pointing, per `build_body_track_perturbers`'s own
    /// doc comment — rather than erroring or panicking.
    #[test]
    fn build_body_track_perturbers_skips_unresolvable_bodies() {
        let seed_json = serde_json::json!({
            "r0_m": [1.0, 0.0, 0.0], "v0_m": [0.0, 1.0, 0.0],
            "reference": [
                { "t_s": 0.0, "r_m": [1.0, 0.0, 0.0], "v_mps": [0.0, 1.0, 0.0] },
                { "t_s": 10.0, "r_m": [2.0, 0.0, 0.0], "v_mps": [0.0, 1.0, 0.0] },
            ],
            "duration_s": 10.0, "tick_s": 1.0,
            "body_tracks": [
                { "name": "NotARealBody", "track": [
                    { "t_s": 0.0, "r_m": [5.0, 0.0, 0.0], "v_mps": [0.0, 0.0, 0.0] },
                    { "t_s": 10.0, "r_m": [5.0, 0.0, 0.0], "v_mps": [0.0, 0.0, 0.0] },
                ] },
                { "name": "Earth", "track": [
                    { "t_s": 0.0, "r_m": [6.0, 0.0, 0.0], "v_mps": [0.0, 0.0, 0.0] },
                    { "t_s": 10.0, "r_m": [6.0, 0.0, 0.0], "v_mps": [0.0, 0.0, 0.0] },
                ] },
            ],
        });
        let seed: CruiseSeedConfig = serde_json::from_value(seed_json).unwrap();
        let entries = build_body_track_perturbers(&seed);
        assert_eq!(entries.len(), 1, "only the catalog-resolvable body should produce a perturber entry");
        assert_eq!(entries[0].name, "Earth");
    }

    /// `soi_capture` — the flag that fixes "captured orbit
    /// reads as a flyby": a track with `soi_capture: false` (the default,
    /// pre-existing behavior) must get `soi_radius_m: None` (third-body-
    /// only, per `resolve_central_body`'s own contract that a `None` SOI
    /// radius means "never a central-body candidate"); one with
    /// `soi_capture: true` must get a real, correctly-sized SOI radius.
    /// Mercury (`primary: None` in the catalog -- orbits the Sun directly)
    /// is the concrete case that motivated this fix, so this test computes
    /// the expected radius the same way `build_body_track_perturbers`
    /// itself must (heliocentric distance, mass ratio against the Sun) and
    /// checks it independently, not just "is Some/is None".
    #[test]
    fn build_body_track_perturbers_sizes_a_real_soi_when_soi_capture_is_true() {
        let mercury = body_models::TargetBody::mercury();
        assert!(mercury.primary.is_none(), "test assumes Mercury orbits the Sun directly");
        let dist_from_sun_m = 6.98e10_f64; // a real Mercury-ish heliocentric distance
        let track = vec![
            serde_json::json!({ "t_s": 0.0, "r_m": [dist_from_sun_m, 0.0, 0.0], "v_mps": [0.0, 47_000.0, 0.0] }),
            serde_json::json!({ "t_s": 3600.0, "r_m": [dist_from_sun_m, 0.0, 0.0], "v_mps": [0.0, 47_000.0, 0.0] }),
        ];

        let seed_default_json = serde_json::json!({
            "r0_m": [1.0e11, 0.0, 0.0], "v0_m": [0.0, 30_000.0, 0.0],
            "reference": [
                { "t_s": 0.0, "r_m": [1.0e11, 0.0, 0.0], "v_mps": [0.0, 30_000.0, 0.0] },
                { "t_s": 10.0, "r_m": [1.0e11, 0.0, 0.0], "v_mps": [0.0, 30_000.0, 0.0] },
            ],
            "duration_s": 10.0, "tick_s": 1.0,
            "body_tracks": [{ "name": "Mercury", "track": track.clone() }],
        });
        let seed_default: CruiseSeedConfig = serde_json::from_value(seed_default_json).unwrap();
        let entries_default = build_body_track_perturbers(&seed_default);
        assert_eq!(entries_default.len(), 1);
        assert_eq!(entries_default[0].soi_radius_m, None, "soi_capture defaults to false -- must stay third-body-only");

        let mut seed_capture_json = serde_json::json!({
            "r0_m": [1.0e11, 0.0, 0.0], "v0_m": [0.0, 30_000.0, 0.0],
            "reference": [
                { "t_s": 0.0, "r_m": [1.0e11, 0.0, 0.0], "v_mps": [0.0, 30_000.0, 0.0] },
                { "t_s": 10.0, "r_m": [1.0e11, 0.0, 0.0], "v_mps": [0.0, 30_000.0, 0.0] },
            ],
            "duration_s": 10.0, "tick_s": 1.0,
            "body_tracks": [{ "name": "Mercury", "track": track, "soi_capture": true }],
        });
        assert_eq!(seed_capture_json["body_tracks"][0]["soi_capture"], true);
        let seed_capture: CruiseSeedConfig = serde_json::from_value(seed_capture_json.take()).unwrap();
        let entries_capture = build_body_track_perturbers(&seed_capture);
        assert_eq!(entries_capture.len(), 1);
        let soi_radius_m = entries_capture[0].soi_radius_m.expect("soi_capture: true must size a real SOI radius");

        let expected = trajectory_solver::laplace_soi_radius_m(dist_from_sun_m, mercury.mu_m3s2 / orbital_models::constants::MU_SUN);
        assert!(
            (soi_radius_m - expected).abs() / expected < 1e-9,
            "SOI radius should exactly match the independently-computed Laplace formula: got {soi_radius_m}, expected {expected}"
        );
        // Sanity: a real Mercury-scale SOI is on the order of 10^8-10^9 m,
        // not a body-relative-scale number and not the full heliocentric
        // distance either.
        assert!(soi_radius_m > 1.0e7 && soi_radius_m < dist_from_sun_m / 10.0);
        assert_eq!(entries_capture[0].radius_m, Some(mercury.radius_m));
    }

    /// The `primary: Some(...)` branch (a body that orbits something other
    /// than the Sun, e.g. the Moon around Earth) when the primary's own
    /// `body_track` is present: SOI must be sized from the real distance to
    /// the PRIMARY, not the Sun -- the same "wrong primary" class of bug
    /// Phase 8h already fixed once for Layer 1's own propagator.
    #[test]
    fn build_body_track_perturbers_sizes_soi_relative_to_the_real_primary_when_its_track_is_present() {
        let moon = body_models::TargetBody::moon();
        assert_eq!(moon.primary, Some("Earth"));
        let earth = body_models::TargetBody::by_name("Earth").unwrap();

        // Earth far from the Sun (1 AU on the x-axis); Moon offset from
        // Earth by a real lunar distance, NOT anywhere near 1 AU from the
        // Sun -- if the code used heliocentric distance instead of
        // distance-from-Earth, the resulting SOI would be wildly wrong.
        let earth_pos = [1.495_98e11_f64, 0.0, 0.0];
        let moon_dist_from_earth_m = 3.844e8_f64;
        let moon_pos = [earth_pos[0] + moon_dist_from_earth_m, 0.0, 0.0];

        let seed_json = serde_json::json!({
            "r0_m": [1.0e11, 0.0, 0.0], "v0_m": [0.0, 30_000.0, 0.0],
            "reference": [
                { "t_s": 0.0, "r_m": [1.0e11, 0.0, 0.0], "v_mps": [0.0, 30_000.0, 0.0] },
                { "t_s": 10.0, "r_m": [1.0e11, 0.0, 0.0], "v_mps": [0.0, 30_000.0, 0.0] },
            ],
            "duration_s": 10.0, "tick_s": 1.0,
            "body_tracks": [
                { "name": "Earth", "track": [
                    { "t_s": 0.0, "r_m": earth_pos, "v_mps": [0.0, 0.0, 0.0] },
                    { "t_s": 3600.0, "r_m": earth_pos, "v_mps": [0.0, 0.0, 0.0] },
                ] },
                { "name": "Moon", "track": [
                    { "t_s": 0.0, "r_m": moon_pos, "v_mps": [0.0, 0.0, 0.0] },
                    { "t_s": 3600.0, "r_m": moon_pos, "v_mps": [0.0, 0.0, 0.0] },
                ], "soi_capture": true },
            ],
        });
        let seed: CruiseSeedConfig = serde_json::from_value(seed_json).unwrap();
        let entries = build_body_track_perturbers(&seed);
        let moon_entry = entries.iter().find(|e| e.name == "Moon").expect("Moon entry should exist");
        let soi_radius_m = moon_entry.soi_radius_m.expect("soi_capture: true must size a real SOI radius");

        let expected = trajectory_solver::laplace_soi_radius_m(moon_dist_from_earth_m, moon.mu_m3s2 / earth.mu_m3s2);
        assert!(
            (soi_radius_m - expected).abs() / expected < 1e-9,
            "Moon's SOI must be sized from its real distance from Earth, not the Sun: got {soi_radius_m}, expected {expected}"
        );
        // A real lunar SOI is ~66,000 km -- nowhere close to what a
        // heliocentric-distance mistake would produce (many millions of km).
        assert!(soi_radius_m > 5.0e7 && soi_radius_m < 1.0e8);
    }

    /// End-to-end confirmation that this isn't just a sizing calculation:
    /// once a `soi_capture: true` body's real SOI radius is registered,
    /// `trajectory_solver::resolve_central_body` (the exact function
    /// `step_tick`/`step_tick_with_burn` call internally) actually selects
    /// it as central once the spacecraft's position falls inside — this is
    /// what makes the propagated TRUTH switch to the captured body's own
    /// gravity instead of coasting under Sun-only gravity through a real
    /// close approach.
    #[test]
    fn soi_capture_body_track_is_actually_selected_as_central_by_resolve_central_body() {
        let dist_from_sun_m = 6.98e10_f64;
        let track = vec![
            serde_json::json!({ "t_s": 0.0, "r_m": [dist_from_sun_m, 0.0, 0.0], "v_mps": [0.0, 47_000.0, 0.0] }),
            serde_json::json!({ "t_s": 3600.0, "r_m": [dist_from_sun_m, 0.0, 0.0], "v_mps": [0.0, 47_000.0, 0.0] }),
        ];
        let seed_json = serde_json::json!({
            "r0_m": [1.0e11, 0.0, 0.0], "v0_m": [0.0, 30_000.0, 0.0],
            "reference": [
                { "t_s": 0.0, "r_m": [1.0e11, 0.0, 0.0], "v_mps": [0.0, 30_000.0, 0.0] },
                { "t_s": 10.0, "r_m": [1.0e11, 0.0, 0.0], "v_mps": [0.0, 30_000.0, 0.0] },
            ],
            "duration_s": 10.0, "tick_s": 1.0,
            "body_tracks": [{ "name": "Mercury", "track": track, "soi_capture": true }],
        });
        let seed: CruiseSeedConfig = serde_json::from_value(seed_json).unwrap();
        let entries = build_body_track_perturbers(&seed);
        let bodies = crate::design::as_propagator_bodies(&entries);
        let soi_radius_m = entries[0].soi_radius_m.unwrap();

        // Well outside Mercury's SOI (10x its radius, same direction) --
        // must NOT resolve to Mercury.
        let far_pos = Vector3::new(dist_from_sun_m + 10.0 * soi_radius_m, 0.0, 0.0);
        assert_eq!(
            trajectory_solver::resolve_central_body(&far_pos, &bodies, 0.0), None,
            "outside the SOI, no body should be selected as central"
        );

        // Well inside Mercury's real SOI -- must resolve to Mercury (index 0).
        let near_pos = Vector3::new(dist_from_sun_m + 0.5 * soi_radius_m, 0.0, 0.0);
        assert_eq!(
            trajectory_solver::resolve_central_body(&near_pos, &bodies, 0.0), Some(0),
            "inside the SOI, Mercury (the only registered central-body candidate) must be selected"
        );
    }

    /// the design notes item 5 (TCM burn execution) end to end: a real
    /// threshold-triggered correction, under a real growing-dispersion
    /// scenario (a body-track perturber, same mechanism as
    /// `run_cruise_streaming_is_perturbed_by_a_real_body_track`), must
    /// (1) actually fire a burn (`Burning` seen on at least one tick, real
    /// propellant consumed) and (2) leave the flown trajectory measurably
    /// CLOSER to the reference than the identical scenario with TCM
    /// disabled — not just "doesn't crash," a real corrective effect.
    ///
    /// Does NOT verify the burn running to completion and returning to
    /// `Coast` (a known, honest gap, not silently assumed): with this
    /// fixture's real orbital geometry, once the position dispersion has
    /// grown enough to trigger, correcting all the way back to the
    /// ORIGINAL undisturbed reference's endpoint by the leg's own fixed
    /// end time demands more ΔV than a modest 5 N thruster delivers in the
    /// remaining window — genuine, physically consistent behavior (a real
    /// mission would either trigger TCM earlier, budget more thrust, or
    /// retarget a later arrival), not a bug in the state machine. Confirmed
    /// via debug instrumentation before finalizing this fixture: the burn
    /// visibly damps growth (pointing converges, dr_m's rate of increase
    /// drops) without a numerically implausible result.
    /// Review D5, pure-math half: the capture solve must
    /// scale the relative speed to exactly local circular speed (bound
    /// orbit by construction) and reject degenerate geometry.
    #[test]
    fn solve_capture_burn_dv_reaches_local_circular_speed_and_rejects_degenerate_input() {
        let mu = 2.5e11_f64;
        let body_r = Vector3::new(1.0e9, 2.0e9, 3.0e9);
        let body_v = Vector3::new(100.0, -50.0, 25.0);
        let r = body_r + Vector3::new(1.0e7, 0.0, 0.0);
        let v = body_v + Vector3::new(0.0, 250.0, 0.0);
        let dv = solve_capture_burn_dv(&r, &v, &body_r, &body_v, mu, 0.0).expect("well-posed geometry");
        let v_rel_post = (v + dv) - body_v;
        let v_circ = (mu / 1.0e7_f64).sqrt();
        assert!((v_rel_post.norm() - v_circ).abs() < 1e-9, "post-burn relative speed must equal local circular");
        // Bound: specific orbital energy about the body is negative.
        let energy = 0.5 * v_rel_post.norm_squared() - mu / 1.0e7;
        assert!(energy < 0.0, "capture must produce a bound orbit, got energy {energy}");
        // Retrograde along the relative velocity (was faster than circular).
        assert!(dv.dot(&Vector3::new(0.0, 1.0, 0.0)) < 0.0);

        // Degenerate: at the body's center, or zero relative speed.
        assert!(solve_capture_burn_dv(&body_r, &v, &body_r, &body_v, mu, 0.0).is_none());
        assert!(solve_capture_burn_dv(&r, &body_v, &body_r, &body_v, mu, 0.0).is_none());
        // Eccentric capture: the target speed is the configured
        // orbit's periapsis speed √(μ(1+e)/r) — a cheaper burn for the same
        // approach, matching `capture_target_speed_mps`'s pricing.
        let dv_ecc = solve_capture_burn_dv(&r, &v, &body_r, &body_v, mu, 0.9).expect("well-posed geometry");
        let v_rel_post_ecc = ((v + dv_ecc) - body_v).norm();
        assert!((v_rel_post_ecc - (mu * 1.9 / 1.0e7_f64).sqrt()).abs() < 1e-9);
        let energy_ecc = 0.5 * v_rel_post_ecc * v_rel_post_ecc - mu / 1.0e7;
        assert!(energy_ecc < 0.0, "eccentric capture must still be bound");
    }

    /// Phase-adaptive report stride: with a coarse coast
    /// `report_stride`, every maneuver-phase tick must still be relayed at
    /// the maneuver stride (default 1), transition ticks always relay, and
    /// coast must remain genuinely strided (the budget is not silently
    /// abandoned).
    #[test]
    fn report_stride_is_phase_adaptive_dense_in_maneuvers_coarse_in_coast() {
        let mu = orbital_models::constants::MU_SUN;
        let r = 1.0e9_f64;
        let v_circ_sun = (mu / r).sqrt();
        let r0 = Vector3::new(r, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ_sun, 0.0);
        let duration_s = 40_000.0;
        let tick_s = 10.0;
        let reference = sample_reference_trajectory_uniform(r0, v0, mu, &[], duration_s, 30.0, 1e-10, 1e-3);
        let reference_json: Vec<serde_json::Value> = reference
            .points()
            .iter()
            .map(|p| serde_json::json!({ "t_s": p.t_s, "r_m": [p.r_m.x, p.r_m.y, p.r_m.z], "v_mps": [p.v_mps.x, p.v_mps.y, p.v_mps.z] }))
            .collect();
        let v = serde_json::json!({
            "mission": { "name": "adaptive stride test", "objective": "Orbit" },
            "target_body": { "name": "Bennu", "ephemeris": "Keplerian" },
            "spacecraft": {
                "mass_kg": 400.0, "dry_mass_kg": 320.0, "propellant_mass_kg": 80.0,
                "bus_dims_m": [1.0, 1.0, 1.2], "inertia_diag_kgm2": [60.0, 60.0, 40.0],
                "srp_model": "Cannonball",
                "propulsion": { "type": "Monoprop", "isp_s": 220.0, "thrust_n": 100.0 },
            },
            "trajectory": { "phases": ["Cruise"], "solver": "Hohmann", "departure_body": "Earth" },
            "gnc": { "navigation_filter": "EKF", "pointing_mode": "Nadir", "attitude_controller": "ReactionWheelPD" },
            "simulation": {
                "integrator": "DormandPrince45", "rtol": 1.0e-9, "atol": 1.0e-7,
                "dt_truth_s": 10.0, "dt_meas_s": 120.0, "monte_carlo_runs": 0,
                "output_dir": "out/adaptive_stride_test/",
            },
            "cruise_seed": {
                "r0_m": [r0.x, r0.y, r0.z], "v0_m": [v0.x, v0.y, v0.z],
                "reference": reference_json, "duration_s": duration_s, "tick_s": tick_s,
                "report_stride": 40,
                // A mid-leg burn along +x (a "DSM"): a real Slewing +
                // Burning window far from both leg ends.
                "planned_burns": [{ "epoch_s": 20_000.0, "dv_inertial_mps": [5.0, 0.0, 0.0], "label": "dsm" }],
            },
        });
        let cfg: MissionConfig = serde_json::from_value(v).expect("adaptive-stride config should deserialize");
        assert!(check_config(&cfg).is_empty(), "{:?}", check_config(&cfg));

        let mut relayed: Vec<(f64, Option<&'static str>)> = Vec::new();
        let result = run_cruise_streaming(&cfg, &mut |row| {
            relayed.push((row.t_s, row.tcm_phase));
            true
        })
        .expect("adaptive-stride run should succeed");
        assert_eq!(result.planned_burn_reports[0].status, "Completed", "{:?}", result.planned_burn_reports);

        // (a) Maneuver density: every consecutive relayed pair where BOTH
        // ticks are in a maneuver phase must be one control tick apart (the
        // burn tick can be shorter than tick_s, never longer).
        let mut maneuver_pairs = 0;
        for w in relayed.windows(2) {
            if w[0].1.is_some() && w[1].1.is_some() {
                maneuver_pairs += 1;
                assert!(
                    w[1].0 - w[0].0 <= tick_s + 1e-6,
                    "maneuver ticks must relay densely: gap {:.1} s at t={:.0}",
                    w[1].0 - w[0].0, w[0].0
                );
            }
        }
        assert!(maneuver_pairs > 3, "the burn window should contribute multiple dense relayed pairs");

        // (b) Coast remains strided: before the maneuver's dense window
        // opens (first 15,000 s), gaps are the full stride.
        let early_coast: Vec<f64> = relayed.iter().filter(|(t, p)| *t < 15_000.0 && p.is_none()).map(|(t, _)| *t).collect();
        assert!(early_coast.len() >= 2);
        let max_gap = early_coast.windows(2).map(|w| w[1] - w[0]).fold(0.0_f64, f64::max);
        assert!(
            max_gap >= 40.0 * tick_s - 1e-6,
            "early coast must still be strided (expected ~{:.0} s gaps, got max {max_gap:.1} s)",
            40.0 * tick_s
        );
    }

    /// Review D5, end-to-end half: a planned burn with `capture_body` set
    /// must fire the FRESH velocity-matching solve from the real state, not
    /// the stored nominal vector — the stored vector here is deliberately
    /// absurd (4 km/s, which would drain far more than the whole tank), so
    /// the observed TCM propellant unambiguously identifies which ΔV fired
    /// (Tsiolkovsky for the ~expected fresh magnitude vs. tank exhaustion).
    #[test]
    fn run_cruise_streaming_capture_burn_re_solves_from_the_dispersed_state() {
        let mu = orbital_models::constants::MU_SUN;
        let r = 1.0e9_f64;
        let v_circ_sun = (mu / r).sqrt();
        let r0 = Vector3::new(r, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ_sun, 0.0);
        let duration_s = 1800.0;
        let tick_s = 10.0;
        let reference = sample_reference_trajectory_uniform(r0, v0, mu, &[], duration_s, 30.0, 1e-10, 1e-3);
        let reference_json: Vec<serde_json::Value> = reference
            .points()
            .iter()
            .map(|p| serde_json::json!({ "t_s": p.t_s, "r_m": [p.r_m.x, p.r_m.y, p.r_m.z], "v_mps": [p.v_mps.x, p.v_mps.y, p.v_mps.z] }))
            .collect();
        // Synthetic capture body riding alongside the reference: offset
        // 1e7 m out-of-plane, relative velocity ~150 m/s along +x. mu
        // chosen so v_circ at the offset distance is ~158 m/s — the fresh
        // solve's magnitude is |v_circ - v_rel| ≈ small tens of m/s.
        let mu_body = 2.5e11_f64;
        let v_rel = Vector3::new(150.0, 0.0, 0.0);
        let offset = Vector3::new(0.0, 0.0, 1.0e7);
        let body_track: Vec<serde_json::Value> = reference
            .points()
            .iter()
            .map(|p| {
                let rb = p.r_m + offset - v_rel * p.t_s;
                let vb = p.v_mps - v_rel;
                serde_json::json!({ "t_s": p.t_s, "r_m": [rb.x, rb.y, rb.z], "v_mps": [vb.x, vb.y, vb.z] })
            })
            .collect();
        let v = serde_json::json!({
            "mission": { "name": "d5 capture test", "objective": "Orbit" },
            "target_body": { "name": "Bennu", "ephemeris": "Keplerian" },
            "spacecraft": {
                "mass_kg": 400.0, "dry_mass_kg": 320.0, "propellant_mass_kg": 80.0,
                "bus_dims_m": [1.0, 1.0, 1.2], "inertia_diag_kgm2": [60.0, 60.0, 40.0],
                "srp_model": "Cannonball",
                "propulsion": { "type": "Monoprop", "isp_s": 220.0, "thrust_n": 500.0 },
            },
            "trajectory": { "phases": ["Cruise"], "solver": "Hohmann", "departure_body": "Earth" },
            "gnc": { "navigation_filter": "EKF", "pointing_mode": "Nadir", "attitude_controller": "ReactionWheelPD" },
            "simulation": {
                "integrator": "DormandPrince45", "rtol": 1.0e-9, "atol": 1.0e-7,
                "dt_truth_s": 10.0, "dt_meas_s": 120.0, "monte_carlo_runs": 0,
                "output_dir": "out/d5_capture_test/",
            },
            "cruise_seed": {
                "r0_m": [r0.x, r0.y, r0.z], "v0_m": [v0.x, v0.y, v0.z],
                "reference": reference_json, "duration_s": duration_s, "tick_s": tick_s,
                "body_tracks": [{ "name": "CaptureTest", "track": body_track, "mu_m3s2": mu_body }],
                "planned_burns": [{
                    "epoch_s": 100.0,
                    "dv_inertial_mps": [4000.0, 0.0, 0.0],
                    "capture_body": "CaptureTest",
                    "label": "capture at CaptureTest",
                }],
            },
        });
        let cfg: MissionConfig = serde_json::from_value(v).expect("d5 test config should deserialize");
        assert!(check_config(&cfg).is_empty(), "{:?}", check_config(&cfg));

        let result = run_cruise_streaming(&cfg, &mut |_row| true).expect("capture-burn run should succeed");

        // Expected fresh magnitude at trigger: |v_circ(offset) - |v_rel||.
        let dv_fresh_mps = ((mu_body / offset.norm()).sqrt() - v_rel.norm()).abs();
        let g0 = orbital_models::constants::G0;
        let expected_prop_kg = 400.0 * (1.0 - (-dv_fresh_mps / (220.0 * g0)).exp());
        assert!(
            result.tcm_propellant_kg_used > 0.0,
            "the capture burn must actually fire"
        );
        assert!(
            (result.tcm_propellant_kg_used - expected_prop_kg).abs() < 0.6 * expected_prop_kg,
            "TCM propellant {:.2} kg should match the FRESH solve's ~{:.2} kg (Tsiolkovsky at {:.1} m/s), \
             not the stored 4 km/s vector (which would exhaust the whole 80 kg tank)",
            result.tcm_propellant_kg_used, expected_prop_kg, dv_fresh_mps
        );
        assert!(
            result.final_propellant_remaining_kg > 40.0,
            "the stored 4 km/s vector would have drained the tank; fresh solve must not ({:.2} kg left)",
            result.final_propellant_remaining_kg
        );

        // Burn-executive lead time + report (ask): the slew must
        // START before the configured epoch, ignition must never precede
        // it, and the per-burn report must say so.
        assert_eq!(result.planned_burn_reports.len(), 1);
        let rep = &result.planned_burn_reports[0];
        assert_eq!(rep.status, "Completed", "{rep:?}");
        assert_eq!(rep.configured_epoch_s, 100.0);
        let slew_start = rep.slew_start_s.expect("slew must have started");
        let ignition = rep.ignition_s.expect("burn must have ignited");
        assert!(slew_start < 100.0, "slew should start BEFORE the epoch (lead time), started at {slew_start}");
        assert!(ignition >= 100.0, "ignition must never precede the configured epoch, got {ignition}");
        assert!(rep.completed_s.is_some());
    }

    /// Phase 13n regression test: a single scheduled `planned_burns` entry
    /// (no reactive `tcm_dr_threshold_m` at all) must actually fire at its
    /// epoch and deliver real, physically consistent ΔV via the SAME
    /// Slewing/Burning machinery the reactive executive uses.
    /// Nuance 2: a planned burn with `target_epoch_s` set must
    /// re-solve its fired ΔV from the vehicle's REAL current state, not
    /// fire the stored `dv_inertial_mps` verbatim -- concretely, it must
    /// absorb real accumulated dispersion the stored vector (computed
    /// under Phase 01's idealized, undispersed assumptions) knows nothing
    /// about. Uses a stored `dv_inertial_mps` of EXACTLY ZERO -- the
    /// idealized "no shaping needed, just follow the reference" case -- so
    /// the verbatim run is a clean no-op baseline (equivalent to no
    /// planned burn at all) and any dispersion reduction in the
    /// `target_epoch_s` run is unambiguously the fresh solve's own doing,
    /// not a shaping burn's incidental side effect.
    #[test]
    fn run_cruise_streaming_planned_burn_with_target_epoch_absorbs_real_dispersion() {
        let mu = orbital_models::constants::MU_SUN;
        let r = 1.0e9_f64;
        let v_circ = (mu / r).sqrt();
        let r0 = Vector3::new(r, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);
        let duration_s = 1800.0;
        let tick_s = 10.0;
        let reference = sample_reference_trajectory_uniform(r0, v0, mu, &[], duration_s, 30.0, 1e-10, 1e-3);
        let reference_json: Vec<serde_json::Value> = reference
            .points()
            .iter()
            .map(|p| serde_json::json!({ "t_s": p.t_s, "r_m": [p.r_m.x, p.r_m.y, p.r_m.z], "v_mps": [p.v_mps.x, p.v_mps.y, p.v_mps.z] }))
            .collect();
        // Same real, close-orbit "Earth" perturber the reactive-TCM tests
        // use -- causes genuine, measurable dispersion within this short
        // window.
        let earth_pos = r0 + Vector3::new(5.0e7, 0.0, 0.0);
        let earth_track = vec![
            serde_json::json!({ "t_s": 0.0, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
            serde_json::json!({ "t_s": duration_s, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
        ];
        let build_cfg = |target_epoch_s: Option<f64>| -> MissionConfig {
            let mut planned_burns_json = serde_json::json!([{
                "epoch_s": 200.0,
                "dv_inertial_mps": [0.0, 0.0, 0.0],
                "label": "absorb-drift checkpoint",
            }]);
            if let Some(t) = target_epoch_s {
                planned_burns_json[0]["target_epoch_s"] = serde_json::json!(t);
            }
            let v = serde_json::json!({
                "mission": { "name": "nuance2 test", "objective": "Orbit" },
                "target_body": { "name": "Bennu", "ephemeris": "Keplerian" },
                "spacecraft": {
                    "mass_kg": 400.0, "dry_mass_kg": 320.0, "propellant_mass_kg": 80.0,
                    "bus_dims_m": [1.0, 1.0, 1.2], "inertia_diag_kgm2": [60.0, 60.0, 40.0],
                    "srp_model": "Cannonball",
                    "propulsion": { "type": "Monoprop", "isp_s": 220.0, "thrust_n": 50.0 },
                },
                "trajectory": { "phases": ["Cruise"], "solver": "Hohmann", "departure_body": "Earth" },
                "gnc": { "navigation_filter": "EKF", "pointing_mode": "Nadir", "attitude_controller": "ReactionWheelPD" },
                "simulation": {
                    "integrator": "DormandPrince45", "rtol": 1.0e-9, "atol": 1.0e-7,
                    "dt_truth_s": 10.0, "dt_meas_s": 120.0, "monte_carlo_runs": 0,
                    "output_dir": "out/nuance2_test/",
                },
                "cruise_seed": {
                    "r0_m": [r0.x, r0.y, r0.z], "v0_m": [v0.x, v0.y, v0.z],
                    "reference": reference_json, "duration_s": duration_s, "tick_s": tick_s,
                    "body_tracks": [{ "name": "Earth", "track": earth_track }],
                    "planned_burns": planned_burns_json,
                },
            });
            serde_json::from_value(v).expect("nuance2 test config should deserialize")
        };

        let verbatim_cfg = build_cfg(None);
        assert!(check_config(&verbatim_cfg).is_empty(), "{:?}", check_config(&verbatim_cfg));
        let fresh_cfg = build_cfg(Some(1700.0));
        assert!(check_config(&fresh_cfg).is_empty(), "{:?}", check_config(&fresh_cfg));

        let verbatim_result = run_cruise_streaming(&verbatim_cfg, &mut |_row| true).expect("verbatim run should succeed");
        let mut saw_slewing_fresh = false;
        let fresh_result = run_cruise_streaming(&fresh_cfg, &mut |row| {
            if row.tcm_phase == Some("Slewing") {
                saw_slewing_fresh = true;
            }
            true
        })
        .expect("fresh-solve run should succeed");

        // The zero-vector "burn" in the verbatim run never actually fires
        // (dv_norm <= 1e-9 guard) -- confirms this really is a no-op
        // baseline, not a burn that happens to cancel out.
        assert_eq!(verbatim_result.tcm_propellant_kg_used, 0.0, "the verbatim zero-DV burn should never actually fire");
        assert!(saw_slewing_fresh, "the fresh-solve run should have found a real nonzero correction to fire");
        assert!(fresh_result.tcm_propellant_kg_used > 0.0, "the fresh-solve run should have consumed real propellant");

        assert!(
            fresh_result.final_dr_m < verbatim_result.final_dr_m,
            "re-solving fresh at the planned burn should leave the trajectory measurably closer to the \
             reference than the verbatim no-op: verbatim final_dr_m={}, fresh final_dr_m={}",
            verbatim_result.final_dr_m, fresh_result.final_dr_m,
        );
    }

    #[test]
    fn run_cruise_streaming_fires_a_planned_burn_at_its_scheduled_epoch() {
        let mut cfg = cruise_seeded_config();
        cfg.spacecraft.propulsion = Some(crate::config::PropulsionConfig {
            kind: crate::config::PropulsionType::Monoprop,
            isp_s: 220.0,
            thrust_n: 50.0,
        });
        // A small, well within the vehicle's Tsiolkovsky-available budget
        // (mass 400 kg, 80 kg propellant -> avail dv = 220*9.80665*ln(400/320)
        // =~ 481 m/s), so the burn should complete well inside the 3600 s
        // window at a real 50 N thrust (accel ~0.125 m/s^2 -> ~40 s to
        // deliver 5 m/s).
        let planned_dv_mps = Vector3::new(5.0, 0.0, 0.0);
        // epoch_s=10 (not, say, 600) is a deliberate test-fixture choice,
        // not a realism claim -- found the hard way: `run_cruise_streaming`
        // seeds initial attitude at SunPointing, and this direction's
        // required reorientation is close to the worst-case near-180 deg
        // quaternion-PD geometry (inherently slow to converge, a known
        // property of quaternion feedback, independent of any bug). Firing
        // as early as possible maximizes the Slewing window so the test
        // exercises the PLANNED-BURN MECHANISM itself, not this fixture's
        // incidental attitude-convergence time.
        cfg.cruise_seed.as_mut().unwrap().planned_burns = vec![crate::config::PlannedBurnConfig {
            epoch_s: 10.0,
            dv_inertial_mps: [planned_dv_mps.x, planned_dv_mps.y, planned_dv_mps.z],
            target_epoch_s: None,
            capture_body: None, external_stage: false,
            label: "test DSM".to_string(),
        }];
        assert!(check_config(&cfg).is_empty(), "{:?}", check_config(&cfg));

        let reference = cruise_ref_to_reference_trajectory(&cfg).expect("reference should build");
        let mut saw_burning = false;
        let result = run_cruise_streaming(&cfg, &mut |row| {
            if row.tcm_phase == Some("Burning") {
                saw_burning = true;
            }
            true
        })
        .expect("run should succeed");

        assert!(saw_burning, "the planned burn should have fired via Burning");
        assert!(result.tcm_propellant_kg_used > 0.0, "a real burn should consume real propellant");

        // The vehicle started exactly on the reference (r0/v0 match by
        // check_config's own invariant); after a 5 m/s burn along +x with
        // no other perturbation, its actual velocity should differ from
        // the (unperturbed, un-burned) reference's velocity at the final
        // tick by close to 5 m/s -- confirms the commanded DV was really
        // applied, not just that *some* burn happened.
        let (_, v_ref_final) = reference.state_at(cfg.cruise_seed.as_ref().unwrap().duration_s);
        let actual_v_final = Vector3::new(result.final_v_mps[0], result.final_v_mps[1], result.final_v_mps[2]);
        let delivered_dv_mps = (actual_v_final - v_ref_final).norm();
        assert!(
            (delivered_dv_mps - planned_dv_mps.norm()).abs() < 0.5,
            "expected ~5 m/s delivered relative to the un-burned reference, got {delivered_dv_mps:.3} m/s"
        );
    }

    #[test]
    fn run_cruise_streaming_tcm_reduces_dispersion_from_a_real_perturber() {
        // A much closer, faster orbit than cruise_seeded_config()'s 1 AU
        // circular reference -- deliberately, not a mistake. At 1 AU, this
        // test's 3600 s window sweeps ~0.03 deg of true anomaly; `tcm_
        // lambert_correction`'s re-solve (current position -> the leg's own
        // near-identical nearby endpoint) degenerates at that near-zero
        // transfer angle and orbital_math::lambert legitimately returns
        // None every tick (confirmed via debug instrumentation before
        // writing this fixture -- the trigger condition (dr_m > threshold)
        // WAS true from t~1150s onward, but no Lambert solution ever came
        // back to act on it). At r=1e9 m the same window sweeps ~35 deg,
        // a well-conditioned geometry -- this is a test-fixture scale
        // choice to exercise the TCM mechanism correctly, not a claim
        // about realistic mission distances.
        fn perturbed_cfg(tcm_threshold_m: Option<f64>) -> MissionConfig {
            let mu = orbital_models::constants::MU_SUN;
            let r = 1.0e9_f64;
            let v_circ = (mu / r).sqrt();
            let r0 = Vector3::new(r, 0.0, 0.0);
            let v0 = Vector3::new(0.0, v_circ, 0.0);
            let duration_s = 1800.0;
            let tick_s = 10.0;
            let reference = sample_reference_trajectory_uniform(r0, v0, mu, &[], duration_s, 30.0, 1e-10, 1e-3);
            let reference_json: Vec<serde_json::Value> = reference
                .points()
                .iter()
                .map(|p| serde_json::json!({ "t_s": p.t_s, "r_m": [p.r_m.x, p.r_m.y, p.r_m.z], "v_mps": [p.v_mps.x, p.v_mps.y, p.v_mps.z] }))
                .collect();

            // Weaker than the pure-perturbation test's ratio, deliberately
            // -- a low tcm_dr_threshold_m triggers a correction almost
            // immediately (small accumulated dispersion, small required
            // DeltaV, so a modest 5 N thruster can actually complete the
            // burn well inside the test window, rather than chasing an
            // ever-growing target it can never catch).
            let earth_pos = r0 + Vector3::new(5.0e7, 0.0, 0.0);
            let earth_track = vec![
                serde_json::json!({ "t_s": 0.0, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
                serde_json::json!({ "t_s": duration_s, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
            ];

            let mut v = serde_json::json!({
                "mission": { "name": "tcm test", "objective": "Orbit" },
                "target_body": { "name": "Bennu", "ephemeris": "Keplerian" },
                "spacecraft": {
                    "mass_kg": 400.0, "dry_mass_kg": 320.0, "propellant_mass_kg": 80.0,
                    "bus_dims_m": [1.0, 1.0, 1.2], "inertia_diag_kgm2": [60.0, 60.0, 40.0],
                    "srp_model": "Cannonball",
                },
                "trajectory": { "phases": ["Cruise"], "solver": "Hohmann", "departure_body": "Earth" },
                "gnc": { "navigation_filter": "EKF", "pointing_mode": "Nadir", "attitude_controller": "ReactionWheelPD" },
                "simulation": {
                    "integrator": "DormandPrince45", "rtol": 1.0e-9, "atol": 1.0e-7,
                    "dt_truth_s": 10.0, "dt_meas_s": 120.0, "monte_carlo_runs": 0,
                    "output_dir": "out/tcm_test/",
                },
                "cruise_seed": {
                    "r0_m": [r0.x, r0.y, r0.z], "v0_m": [v0.x, v0.y, v0.z],
                    "reference": reference_json, "duration_s": duration_s, "tick_s": tick_s,
                    "body_tracks": [{ "name": "Earth", "track": earth_track }],
                    "tcm_dr_threshold_m": tcm_threshold_m,
                },
            });
            if tcm_threshold_m.is_some() {
                v["spacecraft"]["propulsion"] = serde_json::json!({ "type": "Monoprop", "isp_s": 220.0, "thrust_n": 5.0 });
            }
            serde_json::from_value(v).expect("tcm test config should deserialize")
        }

        let no_tcm_cfg = perturbed_cfg(None);
        assert!(check_config(&no_tcm_cfg).is_empty());
        let no_tcm = run_cruise_streaming(&no_tcm_cfg, &mut |_row| true).expect("no-TCM run should succeed");

        // Threshold well below the no-TCM run's own observed max_dr_m, so a
        // correction is guaranteed to trigger.
        let tcm_threshold_m = (no_tcm.max_dr_m * 0.02).max(10.0);
        let tcm_cfg = perturbed_cfg(Some(tcm_threshold_m));
        assert!(check_config(&tcm_cfg).is_empty());

        let mut saw_burning = false;
        let tcm_result = run_cruise_streaming(&tcm_cfg, &mut |row| {
            if row.tcm_phase == Some("Burning") {
                saw_burning = true;
            }
            true
        })
        .expect("TCM run should succeed");

        assert!(saw_burning, "TCM should have actually fired a burn given a threshold well below observed dispersion");
        assert!(tcm_result.tcm_propellant_kg_used > 0.0, "a real burn should consume real propellant");
        assert!(
            tcm_result.max_dr_m < no_tcm.max_dr_m,
            "a real TCM correction should leave the trajectory measurably closer to the reference: no_tcm={}, tcm={}",
            no_tcm.max_dr_m, tcm_result.max_dr_m,
        );
        assert!(tcm_result.final_r_m.iter().all(|x| x.is_finite()));
    }

    /// a reactive correction produces a
    /// structured `ReactiveBurnReport` on the result (trigger/ignition
    /// epochs, the governing solve's convergence data, executed DV and
    /// propellant), and the tick that ran the solve reaches the stream
    /// even through a coarse `report_stride`. Same fixture family as
    /// `run_cruise_streaming_tcm_reduces_dispersion_from_a_real_perturber`.
    #[test]
    fn run_cruise_streaming_reports_reactive_burns_with_solve_telemetry() {
        let mu = orbital_models::constants::MU_SUN;
        let r = 1.0e9_f64;
        let v_circ = (mu / r).sqrt();
        let r0 = Vector3::new(r, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);
        let duration_s = 1800.0;
        let reference = sample_reference_trajectory_uniform(r0, v0, mu, &[], duration_s, 30.0, 1e-10, 1e-3);
        let reference_json: Vec<serde_json::Value> = reference
            .points()
            .iter()
            .map(|p| serde_json::json!({ "t_s": p.t_s, "r_m": [p.r_m.x, p.r_m.y, p.r_m.z], "v_mps": [p.v_mps.x, p.v_mps.y, p.v_mps.z] }))
            .collect();
        let earth_pos = r0 + Vector3::new(5.0e7, 0.0, 0.0);
        let earth_track = vec![
            serde_json::json!({ "t_s": 0.0, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
            serde_json::json!({ "t_s": duration_s, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
        ];
        let v = serde_json::json!({
            "mission": { "name": "reactive report test", "objective": "Orbit" },
            "target_body": { "name": "Bennu", "ephemeris": "Keplerian" },
            "spacecraft": {
                "mass_kg": 400.0, "dry_mass_kg": 320.0, "propellant_mass_kg": 80.0,
                "bus_dims_m": [1.0, 1.0, 1.2], "inertia_diag_kgm2": [60.0, 60.0, 40.0],
                "srp_model": "Cannonball",
                // 50 N (not the sibling test's 5 N): the early trigger's
                // ~40 m/s solved correction must finish INSIDE the 1800 s
                // leg for the "Completed" assertions below (measured at
                // 5 N: the leg ends still burning -- a legitimate
                // "Unfinished" report, but not the case under test).
                "propulsion": { "type": "Monoprop", "isp_s": 220.0, "thrust_n": 50.0 },
            },
            "trajectory": { "phases": ["Cruise"], "solver": "Hohmann", "departure_body": "Earth" },
            "gnc": { "navigation_filter": "EKF", "pointing_mode": "Nadir", "attitude_controller": "ReactionWheelPD" },
            "simulation": {
                "integrator": "DormandPrince45", "rtol": 1.0e-9, "atol": 1.0e-7,
                "dt_truth_s": 10.0, "dt_meas_s": 120.0, "monte_carlo_runs": 0,
                "output_dir": "out/tcm_test/",
            },
            "cruise_seed": {
                "r0_m": [r0.x, r0.y, r0.z], "v0_m": [v0.x, v0.y, v0.z],
                "reference": reference_json, "duration_s": duration_s, "tick_s": 10.0,
                "body_tracks": [{ "name": "Earth", "track": earth_track }],
                "tcm_dr_threshold_m": 10.0,
                // Deliberately absurdly coarse: only the always-relay rules
                // (transitions, solve ticks, maneuver stride) can get a
                // mid-leg tick through this.
                "report_stride": 100000,
            },
        });
        let cfg: MissionConfig = serde_json::from_value(v).expect("config should deserialize");
        assert!(check_config(&cfg).is_empty(), "{:?}", check_config(&cfg));

        let mut streamed_solves = 0usize;
        let result = run_cruise_streaming(&cfg, &mut |row| {
            if row.tcm_solve.is_some() {
                streamed_solves += 1;
            }
            true
        })
        .expect("run should succeed");

        assert!(
            !result.reactive_burn_reports.is_empty(),
            "a triggered reactive correction should produce a report"
        );
        assert!(streamed_solves > 0, "the solve tick should reach the stream despite report_stride=100000");

        let fired: Vec<_> = result
            .reactive_burn_reports
            .iter()
            .filter(|rep| rep.status == "Completed")
            .collect();
        assert!(!fired.is_empty(), "at least one correction should complete: {:?}", result.reactive_burn_reports);
        let rep = fired[0];
        assert_eq!(rep.actuator, "MainEngine", "no RCS thrusters are configured in this fixture");
        assert!(rep.executed_dv_mps > 0.0, "a completed burn should report real delivered DV");
        assert!(rep.propellant_kg > 0.0, "a completed burn should report real propellant");
        let ign = rep.ignition_s.expect("a completed burn has an ignition epoch");
        assert!(ign >= rep.trigger_epoch_s, "ignition {ign} should not precede the trigger {}", rep.trigger_epoch_s);
        assert!(rep.completed_s.expect("completed") > ign);
        // The governing solve's telemetry made it onto the report -- the
        // whole point of ask 2d ("solver converged" vs "returned the
        // guess" without TCM_DEBUG).
        assert!(rep.targeting.is_some(), "{rep:?}");
        assert!(rep.solved_dv_mps.unwrap_or(0.0) > 0.0, "{rep:?}");
        assert!(rep.lambert_dv_mps.unwrap_or(0.0) > 0.0, "{rep:?}");
        // Per-maneuver propellant must roughly account for the run's total
        // TCM propellant (single-tick accounting-boundary slack allowed).
        let sum_kg: f64 = result.reactive_burn_reports.iter().map(|r| r.propellant_kg).sum();
        assert!(
            sum_kg > 0.5 * result.tcm_propellant_kg_used && sum_kg <= result.tcm_propellant_kg_used + 1e-9,
            "per-maneuver propellant ({sum_kg}) should account for the total ({})",
            result.tcm_propellant_kg_used,
        );
    }

    /// Regression test for a KNOWN, DELIBERATELY UNRESOLVED limit case,
    /// found investigating Phase 13p -- kept as a documented
    /// boundary, not deleted, so a future change to the TCM executive that
    /// silently makes this worse (e.g. NaN/panic, or a regression severe
    /// enough to be worth re-investigating) doesn't go unnoticed.
    ///
    /// Same fixture family as the tests around it, but deliberately extreme:
    /// a duration spanning >2 local orbital periods (multi-revolution) with
    /// a thruster (5 N on ~400 kg) that ends up needing to burn for >60% of
    /// the mission timeline just to keep pace with an aggressively strong,
    /// close-orbit "Earth" perturber. Even after both real Phase 13p fixes
    /// (see `run_cruise_streaming_tcm_many_corrections_converges_within_
    /// one_orbital_period`'s doc comment for what those are), this specific
    /// scenario's `max_dr_m` still ends up WORSE than doing nothing at all
    /// (measured: TCM ~4.8e6 m vs. no-TCM ~2.0e6 m) -- investigated and
    /// judged, not confirmed, to be a thruster/disturbance mismatch no
    /// correction STRATEGY could fix (the vehicle is asked to out-thrust a
    /// disturbance far beyond its real propulsive capability), rather than
    /// a further code bug -- but this judgment call is exactly the kind of
    /// thing worth being able to revisit later, hence keeping the fixture
    /// here rather than discarding it once it stopped being the active
    /// investigation. Only asserts the run completes cleanly (no NaN/panic)
    /// -- NOT that it converges, which is the whole point.
    #[test]
    fn cruise_tcm_extreme_multi_revolution_perturber_completes_without_diverging_to_nan() {
        let mu = orbital_models::constants::MU_SUN;
        let r = 1.0e9_f64;
        let v_circ = (mu / r).sqrt();
        let r0 = Vector3::new(r, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);
        let duration_s = 40_000.0; // >2x the ~17,250 s local period
        let tick_s = 10.0;
        let reference = sample_reference_trajectory_uniform(r0, v0, mu, &[], duration_s, 30.0, 1e-10, 1e-3);
        let reference_json: Vec<serde_json::Value> = reference
            .points()
            .iter()
            .map(|p| serde_json::json!({ "t_s": p.t_s, "r_m": [p.r_m.x, p.r_m.y, p.r_m.z], "v_mps": [p.v_mps.x, p.v_mps.y, p.v_mps.z] }))
            .collect();
        let earth_pos = r0 + Vector3::new(5.0e7, 0.0, 0.0);
        let earth_track = vec![
            serde_json::json!({ "t_s": 0.0, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
            serde_json::json!({ "t_s": duration_s, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
        ];
        let v = serde_json::json!({
            "mission": { "name": "tcm-extreme", "objective": "Orbit" },
            "target_body": { "name": "Bennu", "ephemeris": "Keplerian" },
            "spacecraft": {
                "mass_kg": 400.0, "dry_mass_kg": 320.0, "propellant_mass_kg": 80.0,
                "bus_dims_m": [1.0, 1.0, 1.2], "inertia_diag_kgm2": [60.0, 60.0, 40.0],
                "srp_model": "Cannonball",
                "propulsion": { "type": "Monoprop", "isp_s": 220.0, "thrust_n": 5.0 },
            },
            "trajectory": { "phases": ["Cruise"], "solver": "Hohmann", "departure_body": "Earth" },
            "gnc": { "navigation_filter": "EKF", "pointing_mode": "Nadir", "attitude_controller": "ReactionWheelPD" },
            "simulation": {
                "integrator": "DormandPrince45", "rtol": 1.0e-9, "atol": 1.0e-7,
                "dt_truth_s": 10.0, "dt_meas_s": 120.0, "monte_carlo_runs": 0,
                "output_dir": "out/tcm_extreme_test/",
            },
            "cruise_seed": {
                "r0_m": [r0.x, r0.y, r0.z], "v0_m": [v0.x, v0.y, v0.z],
                "reference": reference_json, "duration_s": duration_s, "tick_s": tick_s,
                "body_tracks": [{ "name": "Earth", "track": earth_track }],
                "tcm_dr_threshold_m": 2000.0,
            },
        });
        let cfg: MissionConfig = serde_json::from_value(v).expect("config should deserialize");
        assert!(check_config(&cfg).is_empty(), "{:?}", check_config(&cfg));

        let result = run_cruise_streaming(&cfg, &mut |_row| true).expect("run should complete without erroring");
        assert!(result.final_r_m.iter().all(|x| x.is_finite()));
        assert!(result.final_v_mps.iter().all(|x| x.is_finite()));
        assert!(result.max_dr_m.is_finite());
    }

    /// Real regression test (not a diagnostic) for Phase 13p's two fixes,
    /// in the regime that actually matters: MANY corrections (not just one,
    /// unlike `run_cruise_streaming_tcm_reduces_dispersion_from_a_real_
    /// perturber` above), but a duration well under one local orbital
    /// period (unlike `diag_13p_many_corrections_over_a_long_leg`, which
    /// deliberately spans >2 periods with a thruster far too weak for its
    /// own aggressive perturber -- found, while investigating this fix, to
    /// remain divergent even after both real bugs below are fixed, most
    /// likely because no correction strategy can out-thrust a disturbance
    /// that outmatches available propulsion by orders of magnitude; NOT
    /// further evidence of a code bug, but flagged honestly rather than
    /// silently dropped -- see this test's own module-level context in
    /// the design notes Phase 13p for the full reasoning). Same fixture family,
    /// weaker perturber (Earth 4x farther -> ~16x weaker pull) and a
    /// duration at ~35% of the local period (no multi-revolution spans).
    ///
    /// The two real, confirmed bugs this exercises:
    /// (1) `tcm_lambert_correction` always targeted the fixed leg-end
    /// epoch, regardless of how many orbital revolutions separated "now"
    /// from then -- a single-revolution Lambert solve forced to satisfy a
    /// multi-rev time-of-flight for a small residual position error is
    /// genuinely ill-posed, not just imprecise (measured: a 13 km position
    /// error demanded a 362 KM/S "correction" before this fix).
    /// (2) `Slewing`'s reorientation is wheel-torque-limited and can take
    /// real time, independent of main-engine thrust, during which the
    /// original trigger-time solve goes stale -- fixed by continuously
    /// re-targeting every tick while `Slewing`.
    #[test]
    fn run_cruise_streaming_tcm_many_corrections_converges_within_one_orbital_period() {
        let mu = orbital_models::constants::MU_SUN;
        let r = 1.0e9_f64;
        let v_circ = (mu / r).sqrt();
        let r0 = Vector3::new(r, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);
        let duration_s = 6_000.0; // ~35% of the ~17,250 s local period -- no multi-rev spans
        let tick_s = 10.0;
        let reference = sample_reference_trajectory_uniform(r0, v0, mu, &[], duration_s, 30.0, 1e-10, 1e-3);
        let reference_json: Vec<serde_json::Value> = reference
            .points()
            .iter()
            .map(|p| serde_json::json!({ "t_s": p.t_s, "r_m": [p.r_m.x, p.r_m.y, p.r_m.z], "v_mps": [p.v_mps.x, p.v_mps.y, p.v_mps.z] }))
            .collect();
        let earth_pos = r0 + Vector3::new(2.0e8, 0.0, 0.0); // 4x farther than the diag stress-test
        let earth_track = vec![
            serde_json::json!({ "t_s": 0.0, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
            serde_json::json!({ "t_s": duration_s, "r_m": [earth_pos.x, earth_pos.y, earth_pos.z], "v_mps": [0.0, 0.0, 0.0] }),
        ];
        let v = serde_json::json!({
            "mission": { "name": "tcm-many-corrections", "objective": "Orbit" },
            "target_body": { "name": "Bennu", "ephemeris": "Keplerian" },
            "spacecraft": {
                "mass_kg": 400.0, "dry_mass_kg": 320.0, "propellant_mass_kg": 80.0,
                "bus_dims_m": [1.0, 1.0, 1.2], "inertia_diag_kgm2": [60.0, 60.0, 40.0],
                "srp_model": "Cannonball",
                "propulsion": { "type": "Monoprop", "isp_s": 220.0, "thrust_n": 5.0 },
            },
            "trajectory": { "phases": ["Cruise"], "solver": "Hohmann", "departure_body": "Earth" },
            "gnc": { "navigation_filter": "EKF", "pointing_mode": "Nadir", "attitude_controller": "ReactionWheelPD" },
            "simulation": {
                "integrator": "DormandPrince45", "rtol": 1.0e-9, "atol": 1.0e-7,
                "dt_truth_s": 10.0, "dt_meas_s": 120.0, "monte_carlo_runs": 0,
                "output_dir": "out/tcm_many_corrections_test/",
            },
            "cruise_seed": {
                "r0_m": [r0.x, r0.y, r0.z], "v0_m": [v0.x, v0.y, v0.z],
                "reference": reference_json, "duration_s": duration_s, "tick_s": tick_s,
                "body_tracks": [{ "name": "Earth", "track": earth_track }],
                "tcm_dr_threshold_m": 2000.0,
            },
        });
        let cfg: MissionConfig = serde_json::from_value(v).expect("config should deserialize");
        assert!(check_config(&cfg).is_empty(), "{:?}", check_config(&cfg));

        let mut n_corrections = 0u32;
        let mut prev_phase: Option<&'static str> = None;
        let result = run_cruise_streaming(&cfg, &mut |row| {
            if row.tcm_phase != prev_phase {
                if matches!(row.tcm_phase, Some("Slewing") | Some("RcsCorrecting")) && prev_phase != Some("Burning") {
                    n_corrections += 1;
                }
                prev_phase = row.tcm_phase;
            }
            true
        })
        .expect("run should succeed");

        let mut no_tcm_cfg = cfg.clone();
        no_tcm_cfg.cruise_seed.as_mut().unwrap().tcm_dr_threshold_m = None;
        let no_tcm_result = run_cruise_streaming(&no_tcm_cfg, &mut |_row| true).expect("no-TCM baseline should succeed");

        assert!(n_corrections >= 2, "expected multiple corrections over the leg, got {n_corrections}");
        assert!(
            result.max_dr_m < no_tcm_result.max_dr_m,
            "repeated TCM corrections should leave the trajectory measurably closer to the reference \
             than doing nothing: no_tcm max_dr_m={:.3e}, tcm max_dr_m={:.3e}",
            no_tcm_result.max_dr_m, result.max_dr_m,
        );
        assert!(result.final_r_m.iter().all(|x| x.is_finite()));
    }

    /// `check_config` must flag a `tcm_dr_threshold_m` with no
    /// `spacecraft.propulsion` — a real, easy-to-make mistake that would
    /// otherwise silently never fire (see `build_tcm_config`'s own doc
    /// comment on why this combination is a documented no-op, not an
    /// error, at the `run_cruise_leg` level — but a config author should
    /// still be told).
    #[test]
    fn check_config_flags_tcm_threshold_without_propulsion() {
        let mut cfg = cruise_seeded_config();
        cfg.cruise_seed.as_mut().unwrap().tcm_dr_threshold_m = Some(100.0);
        assert!(cfg.spacecraft.propulsion.is_none());
        let errors = check_config(&cfg);
        assert!(
            errors.iter().any(|e| e.contains("tcm_dr_threshold_m") && e.contains("propulsion")),
            "expected a propulsion-missing error, got: {errors:?}"
        );
    }

    /// Mirrors `check_config_flags_tcm_threshold_without_propulsion` for
    /// Phase 13n's `planned_burns` -- a nonempty list with no propulsion
    /// source has nothing to fire with.
    #[test]
    fn check_config_flags_planned_burns_without_propulsion() {
        let mut cfg = cruise_seeded_config();
        cfg.cruise_seed.as_mut().unwrap().planned_burns = vec![crate::config::PlannedBurnConfig {
            epoch_s: 10.0,
            dv_inertial_mps: [1.0, 0.0, 0.0],
            target_epoch_s: None,
            capture_body: None, external_stage: false,
            label: String::new(),
        }];
        assert!(cfg.spacecraft.propulsion.is_none());
        let errors = check_config(&cfg);
        assert!(
            errors.iter().any(|e| e.contains("planned_burns") && e.contains("propulsion")),
            "expected a propulsion-missing error, got: {errors:?}"
        );
    }

    /// `planned_burns` must be sorted ascending by `epoch_s`, matching
    /// `reference`/`body_tracks`' own convention.
    #[test]
    fn check_config_rejects_unsorted_planned_burns() {
        let mut cfg = cruise_seeded_config();
        cfg.spacecraft.propulsion = Some(crate::config::PropulsionConfig {
            kind: crate::config::PropulsionType::Monoprop,
            isp_s: 220.0,
            thrust_n: 5.0,
        });
        cfg.cruise_seed.as_mut().unwrap().planned_burns = vec![
            crate::config::PlannedBurnConfig { epoch_s: 200.0, dv_inertial_mps: [1.0, 0.0, 0.0], target_epoch_s: None, capture_body: None, external_stage: false, label: String::new() },
            crate::config::PlannedBurnConfig { epoch_s: 100.0, dv_inertial_mps: [1.0, 0.0, 0.0], target_epoch_s: None, capture_body: None, external_stage: false, label: String::new() },
        ];
        let errors = check_config(&cfg);
        assert!(
            errors.iter().any(|e| e.contains("planned_burns") && e.contains("sorted")),
            "expected a sort-order error, got: {errors:?}"
        );
    }

    #[test]
    fn run_cruise_mc_streaming_rejects_zero_runs() {
        let cfg = cruise_seeded_config(); // monte_carlo_runs defaults to 0
        let result = run_cruise_mc_streaming(&cfg, &mut |_| {});
        assert!(result.is_err());
    }

    /// `soi_capture: true` on a track whose body resolves no `mu_m3s2`
    /// (neither an explicit override nor a catalog match) must be a hard
    /// validation error, not a silent third-body-only fallback -- setting
    /// the flag is a clear signal the config author wants real SOI
    /// switching, so silently ignoring it would defeat the point without
    /// telling them why.
    #[test]
    fn check_config_rejects_soi_capture_with_no_resolvable_mu() {
        let mut cfg = cruise_seeded_config();
        cfg.cruise_seed.as_mut().unwrap().body_tracks = vec![serde_json::from_value(serde_json::json!({
            "name": "NotARealBody",
            "track": [
                { "t_s": 0.0, "r_m": [1.0, 0.0, 0.0], "v_mps": [0.0, 0.0, 0.0] },
                { "t_s": 10.0, "r_m": [1.0, 0.0, 0.0], "v_mps": [0.0, 0.0, 0.0] },
            ],
            "soi_capture": true,
        }))
        .unwrap()];
        let errors = check_config(&cfg);
        assert!(
            errors.iter().any(|e| e.contains("soi_capture") && e.contains("NotARealBody")),
            "expected a soi_capture-without-resolvable-mu error, got: {errors:?}"
        );
    }

    /// The same track, with an explicit `mu_m3s2` override instead of
    /// relying on a catalog match, must pass -- confirms the check reads
    /// BOTH resolution paths, not just the catalog one.
    #[test]
    fn check_config_accepts_soi_capture_with_an_explicit_mu_override() {
        let mut cfg = cruise_seeded_config();
        cfg.cruise_seed.as_mut().unwrap().body_tracks = vec![serde_json::from_value(serde_json::json!({
            "name": "SyntheticBody",
            "track": [
                { "t_s": 0.0, "r_m": [1.0e11, 0.0, 0.0], "v_mps": [0.0, 0.0, 0.0] },
                { "t_s": 3600.0, "r_m": [1.0e11, 0.0, 0.0], "v_mps": [0.0, 0.0, 0.0] },
            ],
            "mu_m3s2": 4.0e13,
            "soi_capture": true,
        }))
        .unwrap()];
        let errors = check_config(&cfg);
        assert!(
            !errors.iter().any(|e| e.contains("soi_capture")),
            "an explicit mu_m3s2 override should satisfy soi_capture's validation, got: {errors:?}"
        );
    }
}
