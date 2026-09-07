//! Cascaded attitude control (Phase 13e): a high-level controller (Euler-
//! equation-based, produces a commanded body torque from attitude error)
//! feeding a mode-dependent control allocation layer (distributes that
//! torque across whichever actuators the current mode allows). Thruster
//! on/off behavior is realized by this module's pulse-width-modulation
//! (PWM) duty-cycle computation, a property of the actuator (thrusters are
//! physically two-state devices). The continuous laws (PD/PID) know nothing
//! about that; the one deliberately thruster-native law, the phase-plane
//! controller (§10.5.3), encodes
//! the deadband / hysteresis / minimum-impulse-bit physics of a two-state
//! actuator in the law itself — it is selectable, never the wheel default.
//! See `docs/MP/MANUAL.md` §10.

use attitude_control::{pd_torque, select_firing_thrusters, thruster_selection, PdGains, ReactionWheelCluster, Thruster};
use nalgebra::{Vector3, Vector4};
use orbital_models::constants::G0;

// ── §10.1 High-level attitude control law ────────────────────────────────────

/// High-level attitude control law — produces a commanded body torque from
/// attitude error. An enum (not a trait object) matching this repo's
/// "concrete types dispatched directly" convention elsewhere (see
/// `the design notes`'s note on `sa.rs`/`ga.rs` dispatch). `QuaternionPd` is the
/// baseline; a second option (LQR or MPC) is planned once the baseline is
/// validated in real missions — this enum is the extension point for that,
/// not a premature abstraction (`docs/MP/MANUAL.md` §10.1 tracks this).
#[derive(Clone, Copy, Debug)]
pub enum AttitudeControlLaw {
    QuaternionPd(PdGains),
}

impl AttitudeControlLaw {
    /// Commanded body torque [N*m] to drive `q_cur` toward `q_cmd` given the
    /// current body rate `omega`. `attitude_control::pd_torque` already
    /// implements the quaternion PD law correctly (dead-banded quaternion
    /// feedback) — this wrapper exists only to dispatch on the law variant,
    /// not to re-derive control law math that's already implemented and
    /// tested elsewhere in the shared crates.
    pub fn command_torque(
        &self,
        q_cur: &Vector4<f64>,
        q_cmd: &Vector4<f64>,
        omega: &Vector3<f64>,
    ) -> Vector3<f64> {
        match self {
            AttitudeControlLaw::QuaternionPd(gains) => pd_torque(q_cur, q_cmd, omega, gains),
        }
    }
}

// ── §10.5 Attitude-controller registry: laws, activities, scheduling ────────
//
// Three-layer attitude-control architecture (
// the design notes Phase 13 "tumbling-during-burn" item — full derivations in
// `docs/MP/MANUAL.md` §10.5):
//
//   Layer 1 — per ACTUATOR: a control LAW (`AttitudeLaw`: Pd / Pid /
//             PhasePlane) with its own parameter set, selected per
//             `ControlMode` (wheels vs. thrusters), defaults derived from
//             that actuator's real torque authority.
//   Layer 2 — per ACTIVITY (`Activity`: Hold / Slew / BurnHold / Coast): a
//             tuning adjustment (`ActivityTuning`) applied on top of the
//             layer-1 law — bandwidth and deadband scaling.
//   Layer 3 — gain SCHEDULING: the caller re-derives the layer-1 law as
//             the vehicle's mass properties change (`MissionPlanner::
//             attitude_tuning`), reporting every schedule point.
//
// This module owns the actuator-agnostic pieces (the laws themselves, the
// stateful controller that runs them, the activity scaling rule); what is
// DERIVED from a specific vehicle (gain values from inertia + authority)
// lives in `MissionPlanner`, since this crate has no notion of a mission
// config. The pre-existing `AttitudeControlLaw::QuaternionPd` above is
// untouched — `AttitudeLaw::Pd` runs the same `pd_torque`.

/// Phase-plane (bang-off-bang with Schmitt-trigger deadband) parameters —
/// the classic thruster attitude-control law (Wie, *Space Vehicle Dynamics
/// and Control*, 2nd ed., Ch. 7.5; Bryson, *Control of Spacecraft and
/// Aircraft*, Ch. 11). Per body axis `i`, with the small-angle error
/// `θ_i` (§6.1: `θ ≈ 2·sign(q_err,w)·q_err,vec`) and rate `ω_i`:
///
/// ```text
/// s_i = θ_i + T·ω_i                     (switching function; T = lead_time_s)
/// on  when |s_i| > δ                    (δ = deadband_rad)
/// off when |s_i| < δ − h                (h = hysteresis_rad, Schmitt trigger)
/// τ_i = −sign(s_i) · min( τ_auth,i , I_i·|s_i| / (T·tick) )   while on
/// τ_i ≥ τ_auth,i · min_on_time_s / tick                        while on
/// ```
///
/// The torque magnitude is impulse-limited: full authority only when the
/// state is far from the switching line, otherwise the torque that would
/// null `s_i` within one lead time — so a coarse control tick (10 s here,
/// versus the ~10 ms pulses real thruster electronics use) doesn't
/// overshoot the deadband by a whole tick's worth of full-authority
/// impulse. The floor is the MINIMUM IMPULSE BIT expressed as a duty
/// fraction of the tick: once a thruster is on it cannot deliver less than
/// `τ_auth·min_on_time`, which is what produces the well-known limit cycle
/// inside the deadband and sets its propellant cost (§10.5.3). The
/// allocation layer's PWM (§10.2) turns this magnitude into a duty cycle.
#[derive(Clone, Copy, Debug)]
pub struct PhasePlaneParams {
    /// Position deadband δ [rad] on the switching function.
    pub deadband_rad: f64,
    /// Rate deadband [rad/s]: with `|θ_i| < δ` the axis also stays off
    /// unless `|ω_i|` exceeds this (prevents chasing sensor-level rate
    /// noise inside the deadband).
    pub rate_deadband_radps: f64,
    /// Schmitt-trigger hysteresis h [rad] (turn-off threshold `δ − h`).
    pub hysteresis_rad: f64,
    /// Minimum thruster on-time per pulse [s] — the minimum impulse bit,
    /// as a floor on the per-tick duty fraction.
    pub min_on_time_s: f64,
    /// Lead time T [s] weighting rate against position in `s_i` — plays
    /// the role `k_d/k_p` plays for a PD (§10.5.3).
    pub lead_time_s: f64,
    /// Principal inertia per body axis [kg·m²] (impulse limiting).
    pub inertia_diag_kgm2: Vector3<f64>,
    /// Torque authority per body axis [N·m] — the actuator's full-fire
    /// capability along ±that axis (thrusters: `rcs_worst_axis_authority_nm`
    /// or per-axis probes; wheels: the cluster `max_torque`).
    pub tau_authority_nm: Vector3<f64>,
}

/// Error magnitude above which the PID integrator is held at zero
/// (conditional integration, §10.5.2) — the same 15° at which §10.4's
/// slew profile engages: above it the loop is slewing, not holding, and
/// the error is a trajectory to follow rather than a bias to integrate
/// out. Integrating through a large traverse produces overshoot on
/// arrival proportional to the accumulated term.
pub const PID_INTEGRAL_ENGAGE_RAD: f64 = 15.0 * std::f64::consts::PI / 180.0;

/// Layer-1 attitude control law — one per actuator class, selected per
/// `ControlMode`. `Pd` is the pre-existing quaternion PD (`attitude_control::
/// pd_torque`, §10.1); `Pid` adds an anti-windup integral on the
/// small-angle error vector (§10.5.2 — removes the steady-state offset a
/// constant disturbance such as main-engine misalignment leaves under pure
/// PD); `PhasePlane` is the thruster-native bang-off-bang law (§10.5.3).
#[derive(Clone, Copy, Debug)]
pub enum AttitudeLaw {
    Pd(PdGains),
    Pid {
        gains: PdGains,
        /// Integral gain [N·m/(rad·s)].
        ki: f64,
        /// Anti-windup clamp on `|∫θ dt|` per axis [rad·s].
        integral_limit_rad_s: f64,
    },
    PhasePlane(PhasePlaneParams),
}

impl AttitudeLaw {
    /// Short wire label for telemetry (`"Pd"`, `"Pid"`, `"PhasePlane"`).
    pub fn label(&self) -> &'static str {
        match self {
            AttitudeLaw::Pd(_) => "Pd",
            AttitudeLaw::Pid { .. } => "Pid",
            AttitudeLaw::PhasePlane(_) => "PhasePlane",
        }
    }

    /// `(kp, kd)` for the PD/PID variants, `None` for phase-plane.
    pub fn pd_gains(&self) -> Option<(f64, f64)> {
        match self {
            AttitudeLaw::Pd(g) | AttitudeLaw::Pid { gains: g, .. } => Some((g.kp, g.kd)),
            AttitudeLaw::PhasePlane(_) => None,
        }
    }

    /// Closed-loop `(ω_n, ζ)` about one axis of inertia `I` for the PD/PID
    /// variants (§10.1: `ω_n = √(k_p/2I)`, `ζ = k_d/(2√(k_p·I/2))`), `None`
    /// for phase-plane (no linear closed loop to speak of).
    pub fn natural_frequency_and_damping(&self, inertia_kgm2: f64) -> Option<(f64, f64)> {
        let (kp, kd) = self.pd_gains()?;
        let i = inertia_kgm2.max(1e-12);
        if kp <= 0.0 {
            return None;
        }
        let omega_n = (kp / (2.0 * i)).sqrt();
        let zeta = kd / (2.0 * (kp * i / 2.0).sqrt());
        Some((omega_n, zeta))
    }

    /// 2%-settling-time estimate `≈ 4/(ζ·ω_n)` [s] for the PD/PID
    /// variants (standard second-order-system rule, Franklin–Powell–
    /// Emami-Naeini §3.4); for phase-plane, the time to traverse the
    /// switching line at the slew rate isn't a settling problem, so `None`.
    pub fn settling_time_s(&self, inertia_kgm2: f64) -> Option<f64> {
        let (omega_n, zeta) = self.natural_frequency_and_damping(inertia_kgm2)?;
        if omega_n <= 0.0 || zeta <= 0.0 {
            return None;
        }
        Some(4.0 / (zeta * omega_n))
    }
}

/// Layer 2 — what the vehicle is DOING this tick, which selects a tuning
/// adjustment on top of the actuator's layer-1 law. Resolved by the
/// caller's executive (`MissionPlanner::cruise`) from its own phase state;
/// this crate only defines the vocabulary and the scaling rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activity {
    /// Fine-pointing hold under a named pointing mode/rule.
    Hold,
    /// Large-angle reorientation (§10.4 profile engaged, or a maneuver slew).
    Slew,
    /// Holding burn attitude against the persistent main-engine
    /// disturbance while thrusting — tight deadband, full bandwidth.
    BurnHold,
    /// Quiescent coast with no named pointing requirement — wide
    /// deadband, reduced bandwidth (propellant/actuator-cycle saving).
    Coast,
}

impl Activity {
    pub fn label(&self) -> &'static str {
        match self {
            Activity::Hold => "Hold",
            Activity::Slew => "Slew",
            Activity::BurnHold => "BurnHold",
            Activity::Coast => "Coast",
        }
    }

    /// Default tuning per activity (§10.5.4). Overridable per mission via
    /// `MissionPlanner`'s `gnc.attitude_control.activities` config.
    pub fn default_tuning(&self) -> ActivityTuning {
        match self {
            Activity::Hold => ActivityTuning { bandwidth_scale: 1.0, deadband_scale: 1.0 },
            Activity::Slew => ActivityTuning { bandwidth_scale: 1.0, deadband_scale: 1.0 },
            Activity::BurnHold => ActivityTuning { bandwidth_scale: 1.0, deadband_scale: 0.5 },
            Activity::Coast => ActivityTuning { bandwidth_scale: 0.5, deadband_scale: 2.0 },
        }
    }
}

/// Layer-2 adjustment applied to a layer-1 law by [`tuned_law`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActivityTuning {
    /// Closed-loop bandwidth multiplier `s` for PD/PID: `k_p → s²·k_p`,
    /// `k_d → s·k_d`, `k_i → s³·k_i` — scales `ω_n` by `s` while leaving
    /// the damping ratio ζ INVARIANT (§10.5.4; the 13o lesson: scaling
    /// `k_p` and `k_d` by the same factor changes ζ by `√s`). For
    /// phase-plane the lead time scales as `1/s`.
    pub bandwidth_scale: f64,
    /// Multiplier on the phase-plane position/rate deadbands and on the
    /// PD/PID dead-bands (`PdGains::pointing_db_rad`/`rate_db_rads`).
    pub deadband_scale: f64,
}

/// Apply an activity's tuning to a law (§10.5.4). Pure function — the
/// caller decides when the activity changes and rebuilds the controller.
pub fn tuned_law(law: &AttitudeLaw, tuning: ActivityTuning) -> AttitudeLaw {
    let s = tuning.bandwidth_scale.max(1e-6);
    let d = tuning.deadband_scale.max(0.0);
    match *law {
        AttitudeLaw::Pd(g) => AttitudeLaw::Pd(PdGains {
            kp: g.kp * s * s,
            kd: g.kd * s,
            pointing_db_rad: g.pointing_db_rad * d,
            rate_db_rads: g.rate_db_rads * d,
        }),
        AttitudeLaw::Pid { gains: g, ki, integral_limit_rad_s } => AttitudeLaw::Pid {
            gains: PdGains {
                kp: g.kp * s * s,
                kd: g.kd * s,
                pointing_db_rad: g.pointing_db_rad * d,
                rate_db_rads: g.rate_db_rads * d,
            },
            ki: ki * s * s * s,
            integral_limit_rad_s,
        },
        AttitudeLaw::PhasePlane(p) => AttitudeLaw::PhasePlane(PhasePlaneParams {
            deadband_rad: p.deadband_rad * d,
            rate_deadband_radps: p.rate_deadband_radps * d,
            hysteresis_rad: p.hysteresis_rad * d,
            lead_time_s: p.lead_time_s / s,
            ..p
        }),
    }
}

/// Stateful runner for an [`AttitudeLaw`] — carries the PID integral and
/// the phase-plane per-axis Schmitt latches between ticks. Built fresh
/// (state reset) whenever the caller's scheduler changes the law.
#[derive(Clone, Debug)]
pub struct AttitudeController {
    law: AttitudeLaw,
    /// PID: `∫θ dt` per axis [rad·s], anti-windup-clamped.
    integral_rad_s: Vector3<f64>,
    /// Phase-plane: per-axis latch — `0` off, `±1` on with that torque sign.
    pp_on: [i8; 3],
    /// Phase-plane: epoch [s] each axis last switched on (min-on-time).
    pp_on_since_s: [f64; 3],
}

impl AttitudeController {
    pub fn new(law: AttitudeLaw) -> Self {
        Self { law, integral_rad_s: Vector3::zeros(), pp_on: [0; 3], pp_on_since_s: [f64::NEG_INFINITY; 3] }
    }

    pub fn law(&self) -> &AttitudeLaw {
        &self.law
    }

    /// Clear integral/latch state (e.g. on a target change).
    pub fn reset(&mut self) {
        self.integral_rad_s = Vector3::zeros();
        self.pp_on = [0; 3];
        self.pp_on_since_s = [f64::NEG_INFINITY; 3];
    }

    /// Commanded body torque [N·m] this tick. `t_s`/`tick_s` feed the
    /// integral and the phase-plane timing; PD ignores them.
    pub fn command_torque(
        &mut self,
        q_cur: &Vector4<f64>,
        q_cmd: &Vector4<f64>,
        omega: &Vector3<f64>,
        t_s: f64,
        tick_s: f64,
    ) -> Vector3<f64> {
        match self.law {
            AttitudeLaw::Pd(gains) => pd_torque(q_cur, q_cmd, omega, &gains),
            AttitudeLaw::Pid { gains, ki, integral_limit_rad_s } => {
                let theta = small_angle_error_vec(q_cur, q_cmd);
                let angle = theta.norm();
                // Same dual dead-band convention as `pd_torque`: inside it
                // the loop is idle and the integral is frozen (no windup
                // while nothing is being commanded).
                if angle < gains.pointing_db_rad && omega.norm() < gains.rate_db_rads {
                    return Vector3::zeros();
                }
                // Conditional integration (§10.5.2): the integrator exists
                // to null a small, persistent bias; during a large-angle
                // traverse the error is not a bias, and integrating it
                // charges the integrator with a term the loop must later
                // unwind as overshoot. Freeze it (and let it bleed off)
                // while the error is in the slew regime.
                if angle > PID_INTEGRAL_ENGAGE_RAD {
                    self.integral_rad_s = Vector3::zeros();
                } else {
                    let lim = integral_limit_rad_s.max(0.0);
                    for i in 0..3 {
                        self.integral_rad_s[i] = (self.integral_rad_s[i] + theta[i] * tick_s).clamp(-lim, lim);
                    }
                }
                pd_torque(q_cur, q_cmd, omega, &gains) - ki * self.integral_rad_s
            }
            AttitudeLaw::PhasePlane(p) => {
                let theta = small_angle_error_vec(q_cur, q_cmd);
                let mut tau = Vector3::zeros();
                let tick = tick_s.max(1e-9);
                for i in 0..3 {
                    let s = theta[i] + p.lead_time_s * omega[i];
                    let on_now = self.pp_on[i] != 0;
                    let inside_rate_db = theta[i].abs() < p.deadband_rad && omega[i].abs() < p.rate_deadband_radps;
                    let min_on_elapsed = t_s - self.pp_on_since_s[i] >= p.min_on_time_s;
                    // Torque sign that opposes the switching function NOW.
                    let opposing_sign = if s >= 0.0 { -1.0 } else { 1.0 };
                    let want_on = if on_now {
                        // Schmitt: stay on until |s| drops below δ − h, OR
                        // the state has crossed the switching line so the
                        // latched sign would now push AWAY from it (found
                        // by the unit test: without this release the latch
                        // rides the wrong side of the line forever) — both
                        // subject to the minimum on-time (impulse bit).
                        let crossed = (self.pp_on[i] as f64) != opposing_sign;
                        let inside = s.abs() < (p.deadband_rad - p.hysteresis_rad).max(0.0);
                        !((crossed || inside) && min_on_elapsed)
                    } else {
                        s.abs() > p.deadband_rad && !inside_rate_db
                    };
                    if want_on {
                        let sign = if on_now { self.pp_on[i] as f64 } else { opposing_sign };
                        if !on_now {
                            self.pp_on[i] = sign as i8;
                            self.pp_on_since_s[i] = t_s;
                        }
                        let auth = p.tau_authority_nm[i].abs();
                        let impulse_limited = p.inertia_diag_kgm2[i].abs() * s.abs() / (p.lead_time_s.max(1e-9) * tick);
                        let floor = auth * (p.min_on_time_s / tick).clamp(0.0, 1.0);
                        let mag = impulse_limited.min(auth).max(floor);
                        tau[i] = sign * mag;
                    } else {
                        self.pp_on[i] = 0;
                    }
                }
                tau
            }
        }
    }
}

/// Small-angle attitude-error vector `θ ≈ 2·sign(q_err,w)·q_err,vec` [rad]
/// (body frame) for `q_err = conj(q_cmd) ⊗ q_cur` — the same quantity
/// `pd_torque`'s proportional term acts on (up to the factor 2), exposed
/// for the PID/phase-plane laws. Exact to first order; the sign handles
/// the quaternion double cover (§6.1).
pub fn small_angle_error_vec(q_cur: &Vector4<f64>, q_cmd: &Vector4<f64>) -> Vector3<f64> {
    let conj = Vector4::new(q_cmd[0], -q_cmd[1], -q_cmd[2], -q_cmd[3]);
    let q_err = quat_hamilton(&conj, q_cur);
    let sign = if q_err[0] >= 0.0 { 1.0 } else { -1.0 };
    2.0 * sign * Vector3::new(q_err[1], q_err[2], q_err[3])
}

/// Full-fire RCS torque authority per body axis [N·m] — for each of ±x, ±y,
/// ±z, the aligned thruster set's net torque projected onto that probe
/// (exactly the allocator's own selection rule), taking the weaker of the
/// two signs per axis. The worst axis of this is the scalar `/api/design/
/// vehicle` reports as `rcs_authority_nm` when no disturbance is given.
pub fn rcs_axis_authority_nm(thrusters: &[Thruster]) -> Vector3<f64> {
    let mut out = Vector3::zeros();
    for (i, axis) in [Vector3::x(), Vector3::y(), Vector3::z()].iter().enumerate() {
        let plus = thruster_selection(axis, thrusters).1.dot(axis).max(0.0);
        let minus = thruster_selection(&(-axis), thrusters).1.dot(&(-axis)).max(0.0);
        out[i] = plus.min(minus);
    }
    out
}

/// Minimum over the three axes of [`rcs_axis_authority_nm`] [N·m] — the
/// scalar an RCS-driven gain derivation sizes against (§10.5.1). `0.0`
/// when no thrusters are configured.
pub fn rcs_worst_axis_authority_nm(thrusters: &[Thruster]) -> f64 {
    if thrusters.is_empty() {
        return 0.0;
    }
    let a = rcs_axis_authority_nm(thrusters);
    a.x.min(a.y).min(a.z)
}

// ── §10.3 Momentum management (desaturation + null-motion) ──────────────────

/// Momentum-management (desaturation) law — computes a desired wheel
/// "unload" torque request from the wheel state alone, INDEPENDENT of the
/// attitude control law (`AttitudeControlLaw` above knows nothing about
/// this, and vice versa — the two torque requests are combined and
/// resolved together in `allocate`). An extension point, same pattern as
/// `AttitudeControlLaw`: `None` and `ThresholdRcs` are implemented;
/// `Magnetorquer`/`SrpTrim` are documented future variants (need a local
/// B-field model / panel-articulation coupling, neither built yet — see
/// `docs/MP/MANUAL.md` §10.3).
///
/// Found and fixed: an earlier version of the WheelsPrimary
/// desat branch fired RCS whenever `needs_desat` was true, but the RCS
/// torque was computed from `-total_momentum` and applied ONLY to the
/// body — never paired with a wheel-unload command — so it injected a real
/// external disturbance every tick without ever reducing wheel momentum at
/// all (`needs_desat` stayed true indefinitely once triggered). This
/// self-reinforcing loop (spurious disturbance → bigger attitude
/// correction → more wheel torque → more saturation) is what produced the
/// escalating-then-erratic control effort seen in `cruise_commander_demo`.
/// The fix below makes wheel-unload and its RCS cancellation a hard PAIR,
/// derived from what the wheel ACTUALLY received (post every clamp), not
/// an independently-guessed RCS target.
#[derive(Clone, Copy, Debug)]
pub enum MomentumManagementLaw {
    /// No active unloading — wheel speed is bounded only by the existing
    /// physical `max_speed` clamp (`allocate`'s per-wheel saturation
    /// zeroing). The correct, safe default when no dump-capable actuator
    /// is configured (e.g. `rcs_thrusters` is empty) — desaturation simply
    /// cannot happen without SOME actuator to cancel the wheel's reaction,
    /// so this is what the system degrades to, not a crash or a silent
    /// no-op bug.
    None,
    /// Continuous proportional unloading via RCS, PLUS continuous
    /// null-motion (wheel-speed equalization). Two independent, additive
    /// requests, both resolved through the same wheel/RCS cancellation
    /// pairing in `allocate`:
    ///
    /// - `gain` [1/s]: whenever `ReactionWheelCluster::needs_desat` is
    ///   true, request wheel motor torque toward `dH/dt = -gain · H_wheel`
    ///   (driving the cluster's CONTROLLED (3-DOF) momentum toward zero).
    /// - `null_motion_gain` [1/s]: ALWAYS active (not gated on the desat
    ///   threshold — this is a continuous background correction, same
    ///   philosophy as real ADCS wheel-speed-equalization loops), damping
    ///   the cluster's one REDUNDANT (4th) wheel-speed DOF toward zero via
    ///   `ReactionWheelCluster::null_motion_torque`. Found necessary
    /// `gain` alone only controls `total_momentum` (the
    ///   controlled 3-DOF subspace) — the redundant DOF is completely
    ///   invisible to it, so individual wheels can drift arbitrarily far
    ///   apart even while the cluster's net momentum looks perfectly
    ///   healthy (confirmed via `cruise_commander_demo`'s real telemetry:
    ///   a real, secularly growing "internal momentum" component, and
    ///   `needs_desat` triggering increasingly often as individual wheels
    ///   crept toward the speed threshold even while total |H| plateaued).
    ///
    /// Both `gain`s are hand-tuned rate constants, same status as
    /// `PdGains` — not derived from anything, tune empirically. Set either
    /// to `0.0` to disable that term independently.
    ThresholdRcs { gain: f64, null_motion_gain: f64 },
}

// ── §10.2 Control allocation ──────────────────────────────────────────────────

/// Mode-dependent actuator distribution policy, explicit
/// design statement: cruise always uses wheels; maneuvers use thrusters
/// (with automatic wheel unloading, since the thrusters are already firing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlMode {
    /// Wheels deliver 100% of the commanded attitude torque. RCS fires ONLY
    /// when wheel momentum crosses the desaturation threshold
    /// (`ReactionWheelCluster::needs_desat`) — the normal cruise/quiescent
    /// mode.
    WheelsPrimary,
    /// RCS delivers 100% of the commanded attitude torque (a maneuver: wheel
    /// torque authority is typically far below what a burn-attitude slew or
    /// large correction needs). Wheel momentum is opportunistically unloaded
    /// in the SAME allocation call — not gated on the desaturation
    /// threshold, since the thrusters are already firing regardless, this
    /// is the cheapest possible time to unload them (avoids a second,
    /// separate desaturation burn later) — but RATE-LIMITED to a fraction
    /// of the RCS torque authority left over after the attitude command
    /// (`THRUSTERS_PRIMARY_UNLOAD_AUTHORITY_FRACTION`, MANUAL.md
    /// §10.2). The original one-tick full dump (tumbling-
    /// during-burn investigation, the design notes Phase 13) demanded
    /// `|H_w|/tick` of RCS cancellation in a single tick — far above any
    /// real layout's authority once the slew that precedes a burn has
    /// loaded the wheels — and the uncancelled remainder spun the body up.
    /// Whatever momentum is left when the maneuver ends is picked up by
    /// the ordinary coast-phase law (`WheelsPrimary` +
    /// `MomentumManagementLaw::ThresholdRcs`).
    ThrustersPrimary,
    /// RCS delivers 100% of the commanded attitude torque and the wheel
    /// cluster is EXCLUDED from this tick's control entirely — zero motor
    /// torque, zero reaction torque, zero momentum-dump activity, full
    /// stop. Neither `WheelsPrimary` (wheels do the work) nor
    /// `ThrustersPrimary` (wheels are still actively driven toward zero
    /// speed every tick — a real, non-zero reaction torque on the body) is
    /// a true RCS-only isolation test; this variant is. The real "can the
    /// placed thrusters alone hold this vehicle's pointing" check.
    ThrustersOnly,
}

/// One control-tick's allocation result — enough to compute both the net
/// body torque fed into `propagator6dof::step_tick` and the propellant/
/// wheel-speed bookkeeping the caller needs to carry forward.
///
/// No longer `Copy` — the new
/// `rcs_thruster_duty_cycles: Vec<f64>` field isn't `Copy`. Every existing
/// call site only ever reads this by reference or copies individual scalar
/// fields out of it, so this is a non-breaking change (verified by
/// inspection — no site moves an `AllocationOutput` by value more than
/// once).
#[derive(Clone, Debug)]
pub struct AllocationOutput {
    /// Per-wheel commanded motor torque [N*m] — feed into
    /// `attitude_control::ReactionWheelCluster`'s speed integration
    /// (`speed_dots`) exactly as the existing `actuators::wheel_step`
    /// pattern already does.
    pub wheel_motor_torque_nm: [f64; 4],
    /// Fraction of the tick the selected RCS thruster set fires [0, 1] —
    /// the PWM duty cycle. `0.0` means RCS did not fire this tick.
    pub rcs_duty_cycle: f64,
    /// RCS body torque, TIME-AVERAGED over the tick (`full_fire_torque *
    /// duty_cycle`) — this is the ZOH-consistent quantity to add into
    /// `step_tick`'s `control_torque_body`, valid under the standard PWM
    /// approximation that the pulse period is short relative to the control
    /// tick (so the tick-averaged torque is what the attitude dynamics
    /// actually see — see `docs/MP/MANUAL.md` §10.2).
    pub rcs_body_torque_avg: Vector3<f64>,
    /// Propellant consumed this tick at the computed duty cycle [kg].
    pub rcs_propellant_kg: f64,
    /// Per-thruster PWM duty cycle this tick [0, 1], indexed identically to
    /// the `rcs_thrusters` slice passed to `allocate` —
    /// surfaces `select_firing_thrusters`'s actual
    /// per-actuator decision instead of only the aggregate `rcs_duty_cycle`.
    /// A thruster's selection (torque aligned with this tick's demand or
    /// not) is binary; the PWM duty modulates the whole selected GROUP
    /// together, so every SELECTED thruster shares the same nonzero value
    /// (`rcs_duty_cycle`) — unselected thrusters get `0.0`. Empty iff
    /// `rcs_thrusters` was empty.
    pub rcs_thruster_duty_cycles: Vec<f64>,
    /// Net RCS translational force, TIME-AVERAGED over the tick (body
    /// frame) [N] — `thruster_selection`'s own `net_force` output (see its
    /// doc comment: real whenever the selected/torque-aligned thruster set
    /// doesn't happen to form exact couples), scaled by the same
    /// `rcs_duty_cycle` this call resolved, matching `rcs_body_torque_avg`'s
    /// own ZOH convention exactly. Previously computed and discarded at
    /// every call site below (`let (_net_force, ...) = thruster_selection(...)`)
    /// — real bug found): an
    /// arbitrary, potentially-imbalanced placed-thruster layout is not
    /// guaranteed to null this out, so a real translational disturbance was
    /// being silently thrown away instead of ever reaching the vehicle's
    /// translational EOM. The caller (`cruise::run_cruise_leg`) is
    /// responsible for actually applying this — `allocate` only surfaces
    /// it, since this module has no notion of vehicle mass or translation.
    pub rcs_net_force_body_avg: Vector3<f64>,
    /// Review E3: the UNCLAMPED per-wheel motor-torque command
    /// [N·m] this tick's attitude demand implied
    /// (`ReactionWheelCluster::allocate_unclamped`) — paired with
    /// `wheel_motor_torque_nm` (the delivered, post-clamp/post-zeroing
    /// value) so torque saturation is observable: when any wheel clamps,
    /// the delivered body torque differs from the commanded one in
    /// DIRECTION, and the speed-only `wheel_sat_frac` telemetry reports
    /// nothing. `WheelsPrimary`: the pure attitude command (momentum-
    /// management additions are reported through their own realized-delta
    /// mechanism, not here). `ThrustersPrimary`: the rate-limited unload
    /// command BEFORE its ±max_torque clamp (`wheel_motor_torque_nm` is
    /// the clamped, delivered value). `ThrustersOnly`: zeros.
    pub wheel_motor_torque_cmd_nm: [f64; 4],
    /// Review E3: `max_i |wheel_motor_torque_cmd_nm[i]| / max_torque` —
    /// the torque-authority twin of the speed-based `wheel_sat_frac`.
    /// EXCEEDS 1.0 when the demand overloads the wheels (deliberately
    /// uncapped: the overload ratio is the useful signal).
    pub wheel_torque_sat_frac: f64,
}

/// Real propellant mass [kg] consumed this tick, summed PER THRUSTER from
/// its own real `thrust_n`/`isp_s` and the duty cycle it individually fired
/// at (`duty_cycles`, indexed identically to `thrusters` — the same vector
/// `rcs_thruster_duty_cycles` reports). Reduces exactly to the old
/// aggregate formula (`total_thrust_n * rcs_duty_cycle * tick_s / (isp_s *
/// G0)`) when every thruster shares one Isp and `rcs_duty_cycle` is applied
/// uniformly to the firing set, since `total_thrust_n * rcs_duty_cycle =
/// Σ(thrust_n[i] * duty_cycle[i])` in that case — this is a strict
/// generalization, not a behavior change, for every existing single-Isp
/// layout.
fn sum_rcs_propellant_kg(thrusters: &[Thruster], duty_cycles: &[f64], tick_s: f64) -> f64 {
    thrusters
        .iter()
        .zip(duty_cycles)
        .map(|(t, &duty)| t.thrust_n * duty * tick_s / (t.isp_s * G0))
        .sum()
}

/// Fraction of the RCS torque authority that `ThrustersPrimary` may spend on
/// wheel unloading in any one tick — MANUAL.md §10.2. "Authority" here
/// is the placed layout's full-fire torque projected onto the unload
/// cancellation direction (`−Ĥ_w`), MINUS the attitude command's own
/// magnitude this tick (which already carries the main-engine disturbance
/// the PD loop is cancelling), so pointing always gets first call on the
/// thrusters and unloading only the remainder. The chosen
/// value: 10%. For the reference 24-thruster vehicle (3.54 N·m
/// authority) that is ~0.35 N·m, so 40 N·m·s of stored momentum unloads in
/// ~2 min — well inside a burn lasting hours. A hand-tuned rate constant
/// with the same status as `PdGains`, not derived from anything.
pub const THRUSTERS_PRIMARY_UNLOAD_AUTHORITY_FRACTION: f64 = 0.1;

/// Rate-limited per-wheel unload command for `ThrustersPrimary` — returns
/// `(unclamped, clamped)` motor torques [N·m], the latter bounded to
/// `±max_torque` per wheel (the old one-tick dump applied no clamp at all).
///
/// ```text
/// H_w      = Σᵢ I_w Ω_i ŵ_i                       (cluster momentum, §8.1)
/// τ_auth   = max( 0 , τ_full-fire(−Ĥ_w) · (−Ĥ_w) )   (RCS authority along the
///                                                   cancellation direction)
/// budget   = f · max( 0 , τ_auth − |τ_c| )         (f = THRUSTERS_PRIMARY_
///                                                   UNLOAD_AUTHORITY_FRACTION)
/// s        = min( 1 , budget / (|H_w| / tick) )    (fraction of the full
///                                                   one-tick dump allowed)
/// τ_m,i    = clamp( −s · I_w Ω_i / tick , ±τ_max )
/// ```
///
/// The per-wheel form (rather than `dump_motor_torque`'s pseudo-inverse)
/// keeps the redundant null-space component of the wheel speeds unloading
/// at full rate too — it produces zero body torque by construction
/// (§10.3), so it costs no RCS authority. With no thrusters configured
/// `τ_auth = 0` and the wheels are left alone (nothing could cancel the
/// reaction). A pure null-space spin (`|H_w| ≈ 0`, speeds nonzero) is
/// unloaded at full rate for the same reason.
fn thrusters_primary_unload_command(
    tau_cmd_body: &Vector3<f64>,
    wheel_cluster: &ReactionWheelCluster,
    wheel_speeds: &[f64; 4],
    rcs_thrusters: &[Thruster],
    tick_s: f64,
) -> ([f64; 4], [f64; 4]) {
    let h_wheel = wheel_cluster.total_momentum(wheel_speeds);
    let h_norm = h_wheel.norm();
    let full_rate_nm = h_norm / tick_s;
    let scale = if full_rate_nm < 1e-15 {
        1.0
    } else {
        let cancel_dir = -h_wheel / h_norm;
        let (_net_force, tau_full, _total_thrust) = thruster_selection(&cancel_dir, rcs_thrusters);
        let authority = tau_full.dot(&cancel_dir).max(0.0);
        let budget = THRUSTERS_PRIMARY_UNLOAD_AUTHORITY_FRACTION * (authority - tau_cmd_body.norm()).max(0.0);
        (budget / full_rate_nm).min(1.0)
    };
    let unclamped: [f64; 4] =
        std::array::from_fn(|i| -scale * wheel_speeds[i] * wheel_cluster.wheel_inertia / tick_s);
    let clamped: [f64; 4] =
        std::array::from_fn(|i| unclamped[i].clamp(-wheel_cluster.max_torque, wheel_cluster.max_torque));
    (unclamped, clamped)
}

/// Allocate a commanded attitude torque across the spacecraft's actuators,
/// per `mode`'s policy. `wheel_speeds`/`tick_s` are needed for desaturation
/// sizing and PWM/propellant bookkeeping. `momentum_law` only has any
/// effect in `WheelsPrimary` mode — `ThrustersPrimary` already
/// opportunistically unloads the wheels on every call regardless (rate-
/// limited, see that branch's own doc comment), and `ThrustersOnly`
/// excludes the wheel cluster from control entirely, so a separate
/// momentum-management law would be redundant (or, for `ThrustersOnly`,
/// meaningless) there.
///
/// Propellant is priced per-thruster (each `Thruster`'s own `isp_s`), not
/// against one aggregate scalar
/// — so a mixed
/// layout (different thruster classes at different Isp) is accounted
/// honestly. There is no longer an `rcs_isp_s` parameter here; it was
/// removed rather than left unused once every `Thruster` carried its own
/// value.
#[allow(clippy::too_many_arguments)]
pub fn allocate(
    mode: ControlMode,
    tau_cmd_body: Vector3<f64>,
    wheel_cluster: &ReactionWheelCluster,
    wheel_speeds: &[f64; 4],
    rcs_thrusters: &[Thruster],
    tick_s: f64,
    momentum_law: MomentumManagementLaw,
) -> AllocationOutput {
    // Review E3: the unclamped attitude-demand command +
    // torque-saturation fraction, computed once for every mode's telemetry
    // (ThrustersOnly deliberately reports zeros — the wheels are excluded
    // from control there, so "wheel torque demand" has no meaning).
    let thrusters_primary_unload = match mode {
        ControlMode::ThrustersPrimary => Some(thrusters_primary_unload_command(
            &tau_cmd_body, wheel_cluster, wheel_speeds, rcs_thrusters, tick_s,
        )),
        _ => None,
    };
    let wheel_motor_torque_cmd_nm = match mode {
        ControlMode::WheelsPrimary => wheel_cluster.allocate_unclamped(&tau_cmd_body),
        ControlMode::ThrustersPrimary => thrusters_primary_unload.as_ref().map(|u| u.0).unwrap_or([0.0; 4]),
        ControlMode::ThrustersOnly => [0.0; 4],
    };
    let wheel_torque_sat_frac = wheel_motor_torque_cmd_nm
        .iter()
        .fold(0.0_f64, |m, t| m.max(t.abs()))
        / wheel_cluster.max_torque.max(1e-12);

    match mode {
        ControlMode::WheelsPrimary => {
            let mut wheel_motor_torque_nm = wheel_cluster.allocate(&tau_cmd_body);

            // A wheel already at its rated max speed cannot accelerate
            // further in that direction -- the motor physically saturates
            // there (a real, hard ceiling, distinct from `desat_fraction`'s
            // much lower RCS-unloading trigger below). `wheel_cluster.
            // allocate` has no notion of current wheel speed at all, so
            // without this, the torque used both for the body's reaction
            // torque (below) and for integrating the wheel's own speed
            // (in the caller) would assume the wheel keeps absorbing
            // momentum it physically cannot once saturated -- a real
            // violation of Newton's third law (found via
            // `cruise_commander_demo`'s real periodic comm-pass schedule:
            // the body was being torqued as if the wheel responded, long
            // after the wheel could no longer actually do so, producing
            // erratic non-convergent pointing error once wheels first hit
            // this ceiling repeatedly). Zeroing the torque component that
            // would push an already-saturated wheel further past its
            // limit keeps both effects consistent with what the wheel can
            // actually deliver -- torque in the OPPOSITE direction (to
            // slow the wheel back down) is untouched.
            for i in 0..4 {
                let at_positive_limit = wheel_speeds[i] >= wheel_cluster.max_speed && wheel_motor_torque_nm[i] > 0.0;
                let at_negative_limit = wheel_speeds[i] <= -wheel_cluster.max_speed && wheel_motor_torque_nm[i] < 0.0;
                if at_positive_limit || at_negative_limit {
                    wheel_motor_torque_nm[i] = 0.0;
                }
            }
            // Snapshot the attitude-only command (post its own clamp/zero)
            // BEFORE layering a momentum-management request on top — needed
            // below to isolate exactly what the dump component realized.
            let attitude_only_motor_torque_nm = wheel_motor_torque_nm;

            let (rcs_body_torque_avg, rcs_duty_cycle, rcs_propellant_kg, rcs_thruster_duty_cycles, rcs_net_force_body_avg) = match momentum_law {
                MomentumManagementLaw::None => (Vector3::zeros(), 0.0, 0.0, vec![0.0; rcs_thrusters.len()], Vector3::zeros()),
                MomentumManagementLaw::ThresholdRcs { gain, null_motion_gain } => {
                    if rcs_thrusters.is_empty() {
                        (Vector3::zeros(), 0.0, 0.0, Vec::new(), Vector3::zeros())
                    } else {
                        // Two independent, additive requests -- gated
                        // separately: `dump_request` only when
                        // needs_desat() trips (the controlled 3-DOF
                        // subspace); `null_request` ALWAYS (a continuous
                        // background correction on the redundant 4th DOF,
                        // which needs_desat() cannot see at all -- see
                        // MomentumManagementLaw's own doc comment).
                        let h_wheel = wheel_cluster.total_momentum(wheel_speeds);
                        let dump_request = if wheel_cluster.needs_desat(wheel_speeds) {
                            wheel_cluster.dump_motor_torque(&(-gain * h_wheel))
                        } else {
                            [0.0_f64; 4]
                        };
                        let null_request = wheel_cluster.null_motion_torque(wheel_speeds, null_motion_gain);

                        // Layer BOTH on top of the attitude-only command,
                        // then re-apply BOTH physical constraints (motor
                        // torque limit, speed-limit zeroing) to the TOTAL —
                        // a wheel at its speed limit cannot accept further
                        // torque in that direction regardless of which
                        // logical component asked for it.
                        let mut combined = [0.0_f64; 4];
                        for i in 0..4 {
                            combined[i] = (attitude_only_motor_torque_nm[i] + dump_request[i] + null_request[i])
                                .clamp(-wheel_cluster.max_torque, wheel_cluster.max_torque);
                            let at_pos = wheel_speeds[i] >= wheel_cluster.max_speed && combined[i] > 0.0;
                            let at_neg = wheel_speeds[i] <= -wheel_cluster.max_speed && combined[i] < 0.0;
                            if at_pos || at_neg {
                                combined[i] = 0.0;
                            }
                        }

                        // The combined law's REALIZED contribution —
                        // whatever the wheel actually received beyond the
                        // attitude-only command, after every clamp. This is
                        // what must be cancelled on the body, not the raw
                        // pre-clamp request (which the wheel may not have
                        // been able to fully deliver). Note: null_request
                        // alone produces EXACTLY zero net body torque
                        // before clamping (Σ null_dir[i]·axes[i] = 0) — it
                        // only needs cancelling in the rare case joint
                        // clamping distorted it asymmetrically, which this
                        // realized-difference approach handles for free,
                        // same as it does for dump_request.
                        let mut realized_extra = [0.0_f64; 4];
                        for i in 0..4 {
                            realized_extra[i] = combined[i] - attitude_only_motor_torque_nm[i];
                        }
                        wheel_motor_torque_nm = combined;

                        // Cancel target: the body reaction due to
                        // realized_extra ALONE, negated. If RCS realized
                        // this exactly, net_body_torque (attitude-only
                        // reaction + realized-extra reaction + this
                        // cancellation) would reduce to exactly the
                        // attitude-only torque -- attitude tracking
                        // undisturbed BY DESIGN, not by luck (unlike the
                        // old code, which never even tried).
                        //
                        // In practice this codebase's 12-thruster RCS
                        // layout (attitude_control::rcs_thrusters) is 6
                        // purely axis-aligned couples -- it can only
                        // produce "cube-corner" torque directions, not an
                        // arbitrary continuous one, so an off-axis
                        // cancel_target is realized approximately, not
                        // exactly. Match `ThrustersPrimary`'s own
                        // magnitude-matching convention (duty-cycle scaled
                        // to the TARGET's norm, not a blind full-duty
                        // fire) rather than the old code's disconnected
                        // duty_cycle=1.0 -- direction error from the
                        // discrete thruster geometry is real and expected,
                        // but magnitude is no longer arbitrarily wrong on
                        // top of it.
                        let cancel_target = -wheel_cluster.body_torque(&realized_extra);
                        let (net_force_full, net_torque_full, _total_thrust_n) =
                            thruster_selection(&cancel_target, rcs_thrusters);
                        let rcs_duty_cycle = if net_torque_full.norm() > 1e-12 {
                            (cancel_target.norm() / net_torque_full.norm()).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let net_torque = net_torque_full * rcs_duty_cycle;
                        let net_force = net_force_full * rcs_duty_cycle;
                        let thruster_duty_cycles: Vec<f64> = select_firing_thrusters(&cancel_target, rcs_thrusters)
                            .into_iter()
                            .map(|fire| if fire { rcs_duty_cycle } else { 0.0 })
                            .collect();
                        let propellant = sum_rcs_propellant_kg(rcs_thrusters, &thruster_duty_cycles, tick_s);
                        (net_torque, rcs_duty_cycle, propellant, thruster_duty_cycles, net_force)
                    }
                }
            };

            AllocationOutput {
                wheel_motor_torque_nm, rcs_duty_cycle, rcs_body_torque_avg, rcs_propellant_kg,
                rcs_thruster_duty_cycles, rcs_net_force_body_avg,
                wheel_motor_torque_cmd_nm, wheel_torque_sat_frac,
            }
        }

        ControlMode::ThrustersPrimary => {
            // Unload wheel momentum opportunistically (not gated on the
            // desaturation threshold -- see this variant's doc comment),
            // RATE-LIMITED to what the RCS can actually cancel this tick
            // on top of the attitude command (`thrusters_primary_unload_
            // command`, MANUAL.md §10.2). `.1` is the delivered,
            // ±max_torque-clamped per-wheel command.
            let (_, wheel_motor_torque_nm) = thrusters_primary_unload.expect("computed above for ThrustersPrimary");

            // The reaction the body feels from the REALIZED (post-clamp)
            // unload: decelerating the wheels removes momentum from them,
            // and an equal, opposite change appears on the body --
            // `body_torque(τ_motor) = −A·τ_motor = +ΔH_w/tick`. RCS must
            // cancel that reaction (fire −reaction, exactly §10.3's
            // `τ_cancel = −body_torque(realized)` convention) AND deliver
            // the commanded attitude torque in the same combined demand.
            //
            // Sign bug fixed (tumbling-during-burn
            // investigation): this used to demand `τ_c + H_w/tick`, i.e.
            // the RCS fired WITH the unload reaction instead of against
            // it, doubling the body spin-up rather than cancelling it.
            let reaction_on_body = wheel_cluster.body_torque(&wheel_motor_torque_nm);
            let tau_rcs_demand = tau_cmd_body - reaction_on_body;

            let (net_force_full, net_torque_full, _total_thrust_n) =
                thruster_selection(&tau_rcs_demand, rcs_thrusters);
            let rcs_duty_cycle = if net_torque_full.norm() > 1e-12 {
                (tau_rcs_demand.norm() / net_torque_full.norm()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let rcs_body_torque_avg = net_torque_full * rcs_duty_cycle;
            let rcs_net_force_body_avg = net_force_full * rcs_duty_cycle;
            let rcs_thruster_duty_cycles: Vec<f64> = select_firing_thrusters(&tau_rcs_demand, rcs_thrusters)
                .into_iter()
                .map(|fire| if fire { rcs_duty_cycle } else { 0.0 })
                .collect();
            let rcs_propellant_kg = sum_rcs_propellant_kg(rcs_thrusters, &rcs_thruster_duty_cycles, tick_s);

            AllocationOutput {
                wheel_motor_torque_nm, rcs_duty_cycle, rcs_body_torque_avg, rcs_propellant_kg,
                rcs_thruster_duty_cycles, rcs_net_force_body_avg,
                wheel_motor_torque_cmd_nm, wheel_torque_sat_frac,
            }
        }

        ControlMode::ThrustersOnly => {
            // Wheel cluster fully excluded from this tick's control: zero
            // motor torque, zero reaction torque (`net_body_torque` reads
            // this array through `wheel_cluster.body_torque`, so all-zero
            // here means the wheels contribute nothing to the body), zero
            // momentum-dump activity -- not even the passive "decelerate
            // toward zero" reaction `ThrustersPrimary` still applies. RCS
            // alone must deliver the full commanded torque.
            let wheel_motor_torque_nm = [0.0_f64; 4];

            let (net_force_full, net_torque_full, _total_thrust_n) =
                thruster_selection(&tau_cmd_body, rcs_thrusters);
            let rcs_duty_cycle = if net_torque_full.norm() > 1e-12 {
                (tau_cmd_body.norm() / net_torque_full.norm()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let rcs_body_torque_avg = net_torque_full * rcs_duty_cycle;
            let rcs_net_force_body_avg = net_force_full * rcs_duty_cycle;
            let rcs_thruster_duty_cycles: Vec<f64> = select_firing_thrusters(&tau_cmd_body, rcs_thrusters)
                .into_iter()
                .map(|fire| if fire { rcs_duty_cycle } else { 0.0 })
                .collect();
            let rcs_propellant_kg = sum_rcs_propellant_kg(rcs_thrusters, &rcs_thruster_duty_cycles, tick_s);

            AllocationOutput {
                wheel_motor_torque_nm, rcs_duty_cycle, rcs_body_torque_avg, rcs_propellant_kg,
                rcs_thruster_duty_cycles, rcs_net_force_body_avg,
                wheel_motor_torque_cmd_nm, wheel_torque_sat_frac,
            }
        }
    }
}

// ── §10.4 Rate-limited eigenaxis slew profiling (review E2) ─────

/// Shape a large reorientation into a rate-limited eigenaxis slew — review
/// E2. Raw quaternion-PD on a 90–180° error with
/// torque-limited wheels is structurally prone to limit-cycling even with
/// correctly derived gains: the proportional term commands torque the
/// wheels cannot deliver, the loop runs saturated open-loop, and the
/// stored momentum overshoots. The standard fix (Wie, *Space Vehicle
/// Dynamics and Control*, 2nd ed., Ch. 7.4 — eigenaxis slew with rate
/// limiting) is to track a PROFILED intermediate target instead of the
/// final one, so the PD's error — and therefore its torque demand — stays
/// small by construction.
///
/// Per call (stateless, one tick): compute the eigenaxis rotation from
/// `q_cur` to `q_target` (shortest path, double-cover handled); if the
/// remaining angle `θ` is below `engage_threshold_rad`, pass `q_target`
/// through unchanged (ordinary small-error pointing is not this
/// function's business). Otherwise return an intermediate target
/// `lead_s`-worth of allowed rate ahead of the CURRENT attitude along the
/// eigenaxis, with
///
/// ```text
/// ω_allow = min( ω_cruise_max , √(2·α_max·θ) )        (deceleration-limited:
///                                                      can always stop within
///                                                      the remaining angle)
/// θ_step  = min( θ , ω_allow·lead_s )
/// ```
///
/// `α_max` is the wheels' worst-axis angular-acceleration authority
/// [rad/s²] (τ_authority / I_max — caller supplies it from the same
/// derived inertia the gain derivation uses). Quaternions are `[w,x,y,z]`,
/// matching this crate's existing convention.
pub fn profiled_slew_target(
    q_cur: &Vector4<f64>,
    q_target: &Vector4<f64>,
    lead_s: f64,
    alpha_max_radps2: f64,
    omega_cruise_max_radps: f64,
    engage_threshold_rad: f64,
) -> Vector4<f64> {
    // Body-frame rotation taking q_cur to q_target: q_d = conj(q_cur) ⊗ q_target.
    let conj = Vector4::new(q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]);
    let mut q_d = quat_hamilton(&conj, q_target);
    // Shortest path (quaternion double cover).
    if q_d[0] < 0.0 {
        q_d = -q_d;
    }
    let vec_norm = (q_d[1] * q_d[1] + q_d[2] * q_d[2] + q_d[3] * q_d[3]).sqrt();
    let theta = 2.0 * vec_norm.atan2(q_d[0].max(0.0));
    if theta <= engage_threshold_rad || vec_norm < 1e-12 {
        return *q_target;
    }
    let axis = Vector3::new(q_d[1], q_d[2], q_d[3]) / vec_norm;
    let omega_allow = omega_cruise_max_radps.min((2.0 * alpha_max_radps2.max(0.0) * theta).sqrt());
    let theta_step = theta.min(omega_allow * lead_s.max(0.0));
    let (s, c) = (0.5 * theta_step).sin_cos();
    let dq = Vector4::new(c, axis.x * s, axis.y * s, axis.z * s);
    let out = quat_hamilton(q_cur, &dq);
    out / out.norm().max(1e-12)
}

/// Hamilton product `q1 ⊗ q2`, `[w,x,y,z]` layout.
fn quat_hamilton(q1: &Vector4<f64>, q2: &Vector4<f64>) -> Vector4<f64> {
    let (w1, x1, y1, z1) = (q1[0], q1[1], q1[2], q1[3]);
    let (w2, x2, y2, z2) = (q2[0], q2[1], q2[2], q2[3]);
    Vector4::new(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
}

/// Net body torque this tick — wheel reaction torque plus the time-averaged
/// RCS torque — the quantity to pass as `propagator6dof::step_tick`'s
/// `control_torque_body`.
pub fn net_body_torque(wheel_cluster: &ReactionWheelCluster, out: &AllocationOutput) -> Vector3<f64> {
    wheel_cluster.body_torque(&out.wheel_motor_torque_nm) + out.rcs_body_torque_avg
}

#[cfg(test)]
mod tests {
    use super::*;
    use attitude_control::rcs_thrusters as build_rcs_thrusters;

    fn wheels() -> ReactionWheelCluster {
        ReactionWheelCluster::four_wheel_pyramid(0.012, 628.3, 0.12, 0.8)
    }

    fn thrusters() -> Vec<Thruster> {
        build_rcs_thrusters(1.0, 0.5)
    }

    /// Review E2: the slew profiler must (a) pass small
    /// errors through untouched, (b) command a bounded step along the real
    /// eigenaxis for large errors, and (c) shrink the allowed rate with
    /// the deceleration limit when torque authority is weak.
    #[test]
    fn profiled_slew_target_bounds_the_step_and_respects_the_deceleration_limit() {
        let q_id = Vector4::new(1.0, 0.0, 0.0, 0.0);
        // 90° about +z.
        let half = std::f64::consts::FRAC_PI_4;
        let q_90z = Vector4::new(half.cos(), 0.0, 0.0, half.sin());
        let threshold = 15.0_f64.to_radians();

        // (a) 5° error: passthrough exactly.
        let small_half = 2.5_f64.to_radians();
        let q_5z = Vector4::new(small_half.cos(), 0.0, 0.0, small_half.sin());
        let out = profiled_slew_target(&q_id, &q_5z, 30.0, 1.0e-3, 0.01, threshold);
        assert_eq!(out, q_5z, "below the engage threshold the target must pass through");

        // (b) 90° error, cruise-rate-limited: step = omega_max * lead =
        // 0.01 * 30 = 0.3 rad about +z, measured from the CURRENT attitude.
        let out = profiled_slew_target(&q_id, &q_90z, 30.0, 1.0e-3, 0.01, threshold);
        let step_angle = 2.0 * (out[1] * out[1] + out[2] * out[2] + out[3] * out[3]).sqrt().atan2(out[0]);
        assert!((step_angle - 0.3).abs() < 1e-9, "expected a 0.3 rad step, got {step_angle}");
        assert!(out[3] > 0.0 && out[1].abs() < 1e-12 && out[2].abs() < 1e-12, "step must be about the +z eigenaxis");

        // (c) weak wheels: alpha = 1e-5 rad/s² makes the deceleration
        // limit sqrt(2*alpha*theta) = sqrt(2e-5 * pi/2) ≈ 5.6e-3 rad/s the
        // binding constraint (below the 0.01 cruise cap).
        let out = profiled_slew_target(&q_id, &q_90z, 30.0, 1.0e-5, 0.01, threshold);
        let step_angle = 2.0 * (out[1] * out[1] + out[2] * out[2] + out[3] * out[3]).sqrt().atan2(out[0]);
        let expected = (2.0_f64 * 1.0e-5 * std::f64::consts::FRAC_PI_2).sqrt() * 30.0;
        assert!((step_angle - expected).abs() < 1e-9, "expected {expected}, got {step_angle}");
    }

    /// Review E3: torque-saturation telemetry. An overload
    /// demand must report `wheel_torque_sat_frac > 1` with an unclamped
    /// command exceeding `max_torque` while the DELIVERED torque stays
    /// clamped; a small demand must report `< 1` with commanded ==
    /// delivered exactly (no clamp engaged).
    #[test]
    fn torque_saturation_telemetry_reports_overload_and_clamped_delivery() {
        let cluster = wheels(); // max_torque = 0.12 N·m
        let overload = allocate(
            ControlMode::WheelsPrimary, Vector3::new(5.0, 0.0, 0.0), &cluster, &[0.0; 4],
            &[], 1.0, MomentumManagementLaw::None,
        );
        assert!(overload.wheel_torque_sat_frac > 1.0, "5 N·m demand must overload 0.12 N·m wheels");
        let max_cmd = overload.wheel_motor_torque_cmd_nm.iter().fold(0.0_f64, |m, t| m.max(t.abs()));
        assert!(max_cmd > cluster.max_torque, "unclamped command must exceed max_torque");
        for t in overload.wheel_motor_torque_nm {
            assert!(t.abs() <= cluster.max_torque + 1e-12, "delivered torque must stay clamped");
        }

        let small = allocate(
            ControlMode::WheelsPrimary, Vector3::new(1.0e-4, 0.0, 0.0), &cluster, &[0.0; 4],
            &[], 1.0, MomentumManagementLaw::None,
        );
        assert!(small.wheel_torque_sat_frac < 1.0);
        for i in 0..4 {
            assert!(
                (small.wheel_motor_torque_cmd_nm[i] - small.wheel_motor_torque_nm[i]).abs() < 1e-15,
                "below the clamp, commanded and delivered must be identical"
            );
        }
    }

    #[test]
    fn wheels_primary_uses_only_wheels_when_no_desat_needed() {
        let cluster = wheels();
        let speeds = [0.0; 4]; // far from saturation
        let out = allocate(
            ControlMode::WheelsPrimary, Vector3::new(0.01, 0.0, 0.0),
            &cluster, &speeds, &thrusters(), 10.0,
            MomentumManagementLaw::ThresholdRcs { gain: 0.02, null_motion_gain: 0.0 },
        );
        // needs_desat() is false at these speeds, so even with a
        // ThresholdRcs law configured, nothing should fire.
        assert_eq!(out.rcs_duty_cycle, 0.0);
        assert!(out.rcs_body_torque_avg.norm() < 1e-15);
        assert!(out.wheel_motor_torque_nm.iter().any(|t| t.abs() > 0.0));
    }

    /// A wheel already at its rated max speed, commanded to accelerate
    /// FURTHER in the same direction, must receive zero torque (real motor
    /// saturation — see this fix's own doc comment on `allocate`'s
    /// `WheelsPrimary` branch for the conservation-of-momentum bug this
    /// closes). A wheel at the SAME speed limit but commanded to
    /// DECELERATE (opposite-sign torque) must be unaffected — saturation
    /// only blocks further acceleration past the limit, not recovery from
    /// it.
    #[test]
    fn wheels_primary_zeroes_torque_for_a_wheel_already_at_its_speed_limit() {
        let cluster = wheels();
        let tau_cmd = Vector3::new(0.01, 0.0, 0.0);

        // Unsaturated baseline: which direction does each wheel want to
        // spin for this command? (MomentumManagementLaw::None -- this test
        // is about the speed-limit clamp, not desaturation.)
        let baseline = allocate(ControlMode::WheelsPrimary, tau_cmd, &cluster, &[0.0; 4], &thrusters(), 10.0, MomentumManagementLaw::None);

        // Pin every wheel at the speed limit IN the direction its own
        // commanded torque would push it further.
        let mut pinned_speeds = [0.0_f64; 4];
        for i in 0..4 {
            pinned_speeds[i] = if baseline.wheel_motor_torque_nm[i] >= 0.0 { cluster.max_speed } else { -cluster.max_speed };
        }
        let saturated = allocate(ControlMode::WheelsPrimary, tau_cmd, &cluster, &pinned_speeds, &thrusters(), 10.0, MomentumManagementLaw::None);
        assert!(
            saturated.wheel_motor_torque_nm.iter().all(|t| t.abs() < 1e-15),
            "expected all-zero torque once every wheel is pinned at its limit in the commanded direction, got {:?}",
            saturated.wheel_motor_torque_nm
        );

        // Now command the OPPOSITE torque (wants to decelerate the pinned
        // wheels back down) -- should pass through unaffected.
        let recovery = allocate(ControlMode::WheelsPrimary, -tau_cmd, &cluster, &pinned_speeds, &thrusters(), 10.0, MomentumManagementLaw::None);
        assert!(
            recovery.wheel_motor_torque_nm.iter().any(|t| t.abs() > 1e-9),
            "expected nonzero torque when commanding a saturated wheel to decelerate, got {:?}",
            recovery.wheel_motor_torque_nm
        );
    }

    #[test]
    fn wheels_primary_does_not_fire_rcs_when_momentum_law_is_none() {
        let cluster = wheels();
        // All four wheels above the 0.8 x max_speed desaturation threshold.
        let speeds = [600.0, 600.0, 600.0, 600.0];
        let out = allocate(
            ControlMode::WheelsPrimary, Vector3::zeros(),
            &cluster, &speeds, &thrusters(), 10.0,
            MomentumManagementLaw::None,
        );
        // The safe-default/no-dump-actuator-configured case: wheels simply
        // stay bounded by the existing max_speed clamp, nothing fires.
        assert_eq!(out.rcs_duty_cycle, 0.0);
        assert!(out.rcs_propellant_kg < 1e-15);
    }

    #[test]
    fn wheels_primary_fires_rcs_when_wheels_saturated_with_threshold_law() {
        let cluster = wheels();
        // All four wheels above the 0.8 x max_speed desaturation threshold.
        let speeds = [600.0, 600.0, 600.0, 600.0];
        let out = allocate(
            ControlMode::WheelsPrimary, Vector3::zeros(),
            &cluster, &speeds, &thrusters(), 10.0,
            MomentumManagementLaw::ThresholdRcs { gain: 0.02, null_motion_gain: 0.0 },
        );
        // Duty cycle is now magnitude-matched to the cancel target (same
        // convention as ThrustersPrimary), not blindly 1.0 -- confirm it
        // actually fired, in (0, 1], rather than asserting an exact value.
        assert!(out.rcs_duty_cycle > 0.0 && out.rcs_duty_cycle <= 1.0);
        assert!(out.rcs_body_torque_avg.norm() > 0.0);
        assert!(out.rcs_propellant_kg > 0.0);
    }

    #[test]
    fn wheels_primary_does_not_fire_rcs_when_no_thrusters_configured() {
        // "remove RCS from the hardware config, sim must still run" --
        // ThresholdRcs is configured but no thrusters exist, so it must
        // gracefully degrade to no-op (same output as MomentumManagementLaw::None).
        let cluster = wheels();
        let speeds = [600.0, 600.0, 600.0, 600.0];
        let out = allocate(
            ControlMode::WheelsPrimary, Vector3::zeros(),
            &cluster, &speeds, &[], 10.0,
            MomentumManagementLaw::ThresholdRcs { gain: 0.02, null_motion_gain: 0.0 },
        );
        assert_eq!(out.rcs_duty_cycle, 0.0);
        assert!(out.rcs_propellant_kg < 1e-15);
    }

    /// The real property this fix exists to guarantee: when the attitude
    /// law commands ZERO torque (nothing to track), an active
    /// desaturation event must leave net body torque approximately zero
    /// too -- i.e. RCS actually cancels the wheel-unload reaction instead
    /// of injecting an uncancelled disturbance (the pre-fix bug this
    /// module's doc comment describes: RCS fired from `-total_momentum`
    /// with NO wheel-unload command paired to it, and NO cancellation
    /// property at all).
    #[test]
    fn threshold_rcs_desat_does_not_disturb_a_zero_attitude_command() {
        let cluster = wheels();
        let speeds = [600.0, -550.0, 520.0, -610.0]; // all above the 0.8x threshold
        let out = allocate(
            ControlMode::WheelsPrimary, Vector3::zeros(),
            &cluster, &speeds, &thrusters(), 10.0,
            MomentumManagementLaw::ThresholdRcs { gain: 0.02, null_motion_gain: 0.0 },
        );
        // A real dump was actually commanded (not a no-op)...
        assert!(
            out.wheel_motor_torque_nm.iter().any(|t| t.abs() > 1e-9),
            "expected a nonzero unload torque request, got {:?}", out.wheel_motor_torque_nm
        );
        // ...and RCS meaningfully cancels its reaction, not just adds it
        // uncancelled to the body (the pre-fix bug). Real precision is
        // bounded by the axis-aligned 12-thruster RCS geometry (see the
        // implementation's own comment) -- exact cancellation isn't
        // guaranteed for an off-axis target, so this asserts the fix's
        // real, honest property: net body torque is substantially SMALLER
        // than the uncancelled wheel-reaction-from-dump-alone would have
        // been, not that it's driven to zero.
        let net = net_body_torque(&cluster, &out);
        // tau_cmd_body was zero above, so the wheel's full commanded torque
        // IS the dump component -- what "net" would equal without any RCS
        // cancellation at all.
        let uncancelled_reaction = cluster.body_torque(&out.wheel_motor_torque_nm);
        assert!(
            net.norm() < 0.5 * uncancelled_reaction.norm(),
            "expected RCS to meaningfully cancel the dump reaction, got net={:?} (norm {}) vs. uncancelled {:?} (norm {})",
            net, net.norm(), uncancelled_reaction, uncancelled_reaction.norm()
        );
    }

    /// Speeds chosen purely along `null_dir` ([1,-1,1,-1]): total_momentum
    /// is ~zero (all well under the 0.8x desat threshold, so `gain`'s term
    /// never fires here — isolates null-motion), but the individual wheels
    /// are badly imbalanced. Confirms null-motion (a) fires even with no
    /// desat trigger and zero attitude command, (b) actually damps the
    /// null-space projection (not a no-op), (c) net body torque stays
    /// extremely close to zero -- TIGHTER than the general dump-cancellation
    /// test's tolerance, since an unclamped null-motion request produces
    /// EXACTLY zero net torque by construction (Σ null_dir[i]·axes[i] = 0),
    /// not merely an approximately-cancelled one.
    #[test]
    fn null_motion_damps_the_internal_component_without_disturbing_the_body() {
        let cluster = wheels();
        let speeds = [100.0, -100.0, 100.0, -100.0]; // pure null_dir direction
        let h_wheel = cluster.total_momentum(&speeds);
        assert!(h_wheel.norm() < 1e-9, "test setup should be near-zero total momentum, got {:?}", h_wheel);
        assert!(!cluster.needs_desat(&speeds), "test setup should be well under the desat threshold");

        let out = allocate(
            ControlMode::WheelsPrimary, Vector3::zeros(),
            &cluster, &speeds, &thrusters(), 10.0,
            MomentumManagementLaw::ThresholdRcs { gain: 0.02, null_motion_gain: 0.05 },
        );

        // A real correction was commanded (not a no-op) -- and it opposes
        // the [1,-1,1,-1] pattern (each wheel's torque should push its own
        // speed back toward the OTHER wheels', i.e. w1/w3 negative, w2/w4
        // positive, since w1=w3=+100 and w2=w4=-100 here).
        assert!(out.wheel_motor_torque_nm[0] < -1e-9, "wheel 1 should be commanded to reduce its excess speed");
        assert!(out.wheel_motor_torque_nm[1] > 1e-9, "wheel 2 should be commanded to reduce its (negative) excess speed");

        let net = net_body_torque(&cluster, &out);
        assert!(
            net.norm() < 1e-6,
            "null-motion should leave net body torque extremely close to zero by construction, got {:?} (norm {})",
            net, net.norm()
        );
    }

    #[test]
    fn null_motion_gain_zero_means_no_correction_despite_internal_imbalance() {
        let cluster = wheels();
        let speeds = [100.0, -100.0, 100.0, -100.0];
        let out = allocate(
            ControlMode::WheelsPrimary, Vector3::zeros(),
            &cluster, &speeds, &thrusters(), 10.0,
            MomentumManagementLaw::ThresholdRcs { gain: 0.02, null_motion_gain: 0.0 },
        );
        assert!(out.wheel_motor_torque_nm.iter().all(|t| t.abs() < 1e-15));
        assert_eq!(out.rcs_duty_cycle, 0.0);
    }

    #[test]
    fn thrusters_primary_unloads_wheels_and_commands_attitude_torque() {
        let cluster = wheels();
        let speeds = [200.0, -150.0, 100.0, -50.0];
        let out = allocate(
            ControlMode::ThrustersPrimary, Vector3::new(0.0, 0.05, 0.0),
            &cluster, &speeds, &thrusters(), 5.0, MomentumManagementLaw::None,
        );
        // Wheels commanded toward zero (negative of their current sign).
        for (motor_tau, speed) in out.wheel_motor_torque_nm.iter().zip(speeds.iter()) {
            if *speed > 0.0 {
                assert!(*motor_tau < 0.0, "wheel spinning positive should be commanded to decelerate");
            } else if *speed < 0.0 {
                assert!(*motor_tau > 0.0, "wheel spinning negative should be commanded to decelerate");
            }
        }
        assert!(out.rcs_duty_cycle > 0.0, "RCS should fire to unload wheels + deliver commanded torque");
        assert!(out.rcs_propellant_kg > 0.0);
    }

    /// Tumbling-during-burn fix: the RCS must fire AGAINST the
    /// wheel-unload reaction, not with it. Equal speeds on all four pyramid
    /// wheels put `H_w` on the body +z axis, where the 12-thruster layout's
    /// axis-aligned couples cancel the reaction exactly — so with no
    /// attitude command the net body torque must vanish. The old
    /// `τ_c + H_w/tick` demand produced 2× the reaction here instead.
    #[test]
    fn thrusters_primary_rcs_cancels_the_unload_reaction_exactly_on_a_body_axis() {
        let cluster = wheels();
        // Small enough that the full one-tick dump fits inside the unload
        // budget, so the cancellation (not the rate limit) is what's tested.
        let speeds = [0.5; 4];
        let out = allocate(
            ControlMode::ThrustersPrimary, Vector3::zeros(),
            &cluster, &speeds, &thrusters(), 5.0, MomentumManagementLaw::None,
        );
        let reaction = cluster.body_torque(&out.wheel_motor_torque_nm);
        assert!(reaction.norm() > 1e-6, "expected a real unload reaction, got {reaction:?}");
        assert!(reaction.z > 0.0 && reaction.x.abs() < 1e-12 && reaction.y.abs() < 1e-12, "H_w should sit on +z, got {reaction:?}");
        assert!(out.rcs_body_torque_avg.z < 0.0, "RCS must oppose the reaction, got {:?}", out.rcs_body_torque_avg);
        let net = net_body_torque(&cluster, &out);
        assert!(net.norm() < 1e-3 * reaction.norm(), "net body torque should be ~0 with an exactly-cancelled unload, got {net:?} vs reaction {reaction:?}");
        // The whole (small) momentum unloads in one tick in this regime.
        assert!((reaction.norm() - cluster.total_momentum(&speeds).norm() / 5.0).abs() < 1e-9);
    }

    /// Tumbling-during-burn fix: with a lot of stored momentum
    /// the unload is rate-limited to `THRUSTERS_PRIMARY_UNLOAD_AUTHORITY_
    /// FRACTION` of the RCS authority along the cancellation direction —
    /// the demand never exceeds what the thrusters can cancel, the duty
    /// cycle never clamps at 1, and net body torque stays ~0.
    #[test]
    fn thrusters_primary_unload_is_rate_limited_to_a_fraction_of_rcs_authority() {
        let cluster = wheels();
        let t = thrusters();
        let speeds = [300.0; 4]; // |H_w| ≈ 11.8 N·m·s on +z; /tick would be 1.18 N·m vs 1 N·m of z authority
        let tick_s = 10.0;
        let out = allocate(
            ControlMode::ThrustersPrimary, Vector3::zeros(),
            &cluster, &speeds, &t, tick_s, MomentumManagementLaw::None,
        );
        let h = cluster.total_momentum(&speeds);
        let cancel_dir = -h / h.norm();
        let (_, tau_full, _) = attitude_control::thruster_selection(&cancel_dir, &t);
        let authority = tau_full.dot(&cancel_dir);
        let reaction = cluster.body_torque(&out.wheel_motor_torque_nm);
        assert!(reaction.norm() > 0.0);
        assert!(reaction.norm() <= THRUSTERS_PRIMARY_UNLOAD_AUTHORITY_FRACTION * authority + 1e-12,
            "unload reaction {} must not exceed {}% of authority {}", reaction.norm(), 100.0 * THRUSTERS_PRIMARY_UNLOAD_AUTHORITY_FRACTION, authority);
        assert!(reaction.norm() < h.norm() / tick_s, "must be slower than the old one-tick full dump");
        assert!(out.rcs_duty_cycle > 0.0 && out.rcs_duty_cycle < 1.0, "duty {} should be a real partial cycle, never clamped", out.rcs_duty_cycle);
        let net = net_body_torque(&cluster, &out);
        assert!(net.norm() < 1e-3 * reaction.norm(), "net {net:?} should be cancelled");
        // Attitude command gets first call: a command that saturates the
        // RCS by itself leaves no budget, so the wheels are left alone.
        let out_sat = allocate(
            ControlMode::ThrustersPrimary, Vector3::new(0.0, 0.0, -10.0 * authority),
            &cluster, &speeds, &t, tick_s, MomentumManagementLaw::None,
        );
        assert!(out_sat.wheel_motor_torque_nm.iter().all(|m| m.abs() < 1e-15), "no unload budget when attitude saturates the RCS, got {:?}", out_sat.wheel_motor_torque_nm);
    }

    /// The unload command is clamped to each wheel's `max_torque` — the
    /// old one-tick dump applied no clamp at all, so a short tick could
    /// command motor torque far beyond what the wheel can deliver.
    #[test]
    fn thrusters_primary_unload_respects_wheel_max_torque() {
        let cluster = wheels(); // max_torque = 0.12 N·m
        // Alternating signs put the momentum mostly in the null space, so
        // the RCS-authority rate limit doesn't engage and only the motor
        // clamp bounds the command.
        let speeds = [600.0, -600.0, 600.0, -600.0];
        let out = allocate(
            ControlMode::ThrustersPrimary, Vector3::zeros(),
            &cluster, &speeds, &thrusters(), 0.1, MomentumManagementLaw::None,
        );
        for (cmd, delivered) in out.wheel_motor_torque_cmd_nm.iter().zip(out.wheel_motor_torque_nm.iter()) {
            assert!(cmd.abs() > cluster.max_torque, "fixture should overload the motor, cmd {cmd}");
            assert!((delivered.abs() - cluster.max_torque).abs() < 1e-12, "delivered {delivered} should clamp to ±max_torque");
        }
        assert!(out.wheel_torque_sat_frac > 1.0);
    }

    /// Single-axis rigid-body closed loop under a constant disturbance
    /// torque — the burn-hold situation (§10.5.2). Returns the final
    /// angle error [rad] after `n` ticks.
    fn run_one_axis(law: AttitudeLaw, inertia: f64, tau_dist: f64, tick_s: f64, n: usize) -> (f64, f64) {
        let mut ctl = AttitudeController::new(law);
        let mut theta = 0.0_f64; // about +z
        let mut omega = 0.0_f64;
        let q_cmd = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let mut max_abs = 0.0_f64;
        for k in 0..n {
            let (s, c) = (0.5 * theta).sin_cos();
            let q = Vector4::new(c, 0.0, 0.0, s);
            let tau = ctl.command_torque(&q, &q_cmd, &Vector3::new(0.0, 0.0, omega), k as f64 * tick_s, tick_s);
            let alpha = (tau.z + tau_dist) / inertia;
            omega += alpha * tick_s;
            theta += omega * tick_s;
            max_abs = max_abs.max(theta.abs());
        }
        (theta, max_abs)
    }

    /// §10.5.2: pure PD leaves a steady-state offset `θ_ss = 2τ_dist/k_p`
    /// under a constant disturbance; PID drives it to zero.
    #[test]
    fn pid_removes_the_constant_disturbance_offset_that_pd_leaves() {
        let inertia = 500.0;
        let (kp, kd) = (1.76, 35.7); // ω_n ≈ 0.042, ζ ≈ 0.85 on I=500
        let gains = PdGains { kp, kd, pointing_db_rad: 0.0, rate_db_rads: 0.0 };
        let tau_dist = 0.059; // the reference vehicle's engine misalignment torque
        let tick = 10.0;
        let (pd_final, _) = run_one_axis(AttitudeLaw::Pd(gains), inertia, tau_dist, tick, 3000);
        let expected_offset = 2.0 * tau_dist / kp;
        assert!((pd_final.abs() - expected_offset).abs() < 0.1 * expected_offset,
            "PD offset {pd_final} should be ≈ 2τ/kp = {expected_offset}");
        let (pid_final, _) = run_one_axis(
            AttitudeLaw::Pid { gains, ki: 0.01, integral_limit_rad_s: 100.0 }, inertia, tau_dist, tick, 3000,
        );
        assert!(pid_final.abs() < 0.05 * expected_offset, "PID should null the offset, got {pid_final} vs PD {pd_final}");
    }

    /// §10.5.3: the phase-plane law (a) captures from 5° and holds within
    /// its deadband against a SMALL disturbance, (b) never commands more
    /// than the axis authority, (c) its Schmitt latch keeps the on/off
    /// decision from chattering every tick, and (d) against the reference
    /// vehicle's REAL engine-misalignment torque at a 10 s tick it settles
    /// at the derived proportional-law offset `s_ss ≈ τ_d·T·tick/I` (2.7°
    /// here) — the physics reason burn-hold defaults to PID, not phase-
    /// plane, at a coarse control tick (§10.5.3, "hold-quality bound").
    #[test]
    fn phase_plane_holds_within_deadband_and_obeys_the_coarse_tick_offset_bound() {
        let inertia = 500.0;
        let auth = 3.54;
        let db = 0.5_f64.to_radians();
        let lead = 40.0;
        let tick = 10.0;
        // Minimum impulse bit must fit inside the deadband: the limit-cycle
        // rate kick is Δω_min = τ_auth·t_min/I, which moves s by Δω_min·T —
        // at 0.5 s pulses that is 8° here (infeasible against a 0.5°
        // deadband, found by this test's first version); 20 ms pulses
        // (typical monoprop valve minimum) give 0.32°.
        let p = PhasePlaneParams {
            deadband_rad: db,
            rate_deadband_radps: 1e-4,
            hysteresis_rad: 0.2 * db,
            min_on_time_s: 0.02,
            lead_time_s: lead,
            inertia_diag_kgm2: Vector3::new(inertia, inertia, inertia),
            tau_authority_nm: Vector3::new(auth, auth, auth),
        };
        let run = |tau_dist: f64| {
            let mut ctl = AttitudeController::new(AttitudeLaw::PhasePlane(p));
            let (mut theta, mut omega) = (5.0_f64.to_radians(), 0.0_f64);
            let q_cmd = Vector4::new(1.0, 0.0, 0.0, 0.0);
            let (mut max_after_capture, mut max_tau, mut switches, mut prev_on) = (0.0_f64, 0.0_f64, 0usize, false);
            for k in 0..3000 {
                let (s, c) = (0.5 * theta).sin_cos();
                let q = Vector4::new(c, 0.0, 0.0, s);
                let tau = ctl.command_torque(&q, &q_cmd, &Vector3::new(0.0, 0.0, omega), k as f64 * tick, tick);
                max_tau = max_tau.max(tau.z.abs());
                let on = tau.z != 0.0;
                if on != prev_on { switches += 1; }
                prev_on = on;
                omega += (tau.z + tau_dist) / inertia * tick;
                theta += omega * tick;
                if k > 1000 { max_after_capture = max_after_capture.max(theta.abs()); }
            }
            (max_after_capture, max_tau, switches)
        };
        // (a)-(c): small disturbance, holds at the deadband.
        let (held, max_tau, switches) = run(0.002);
        assert!(max_tau <= auth + 1e-12, "never exceed authority: {max_tau}");
        assert!(held < 3.0 * db, "should hold near the deadband ({:.3}°), got {:.3}°", db.to_degrees(), held.to_degrees());
        assert!(switches < 2000, "Schmitt latch should prevent per-tick chatter, got {switches} switches in 3000 ticks");
        // (d): real-regime disturbance, settles at the derived offset bound.
        let tau_dist = 0.059;
        let s_ss = tau_dist * lead * tick / inertia;
        let (held_big, _, _) = run(tau_dist);
        assert!(held_big < 2.0 * s_ss + 3.0 * db,
            "offset {:.2}° should be within the bound 2·s_ss+3δ = {:.2}° (s_ss = {:.2}°)",
            held_big.to_degrees(), (2.0 * s_ss + 3.0 * db).to_degrees(), s_ss.to_degrees());
        assert!(held_big > 0.5 * s_ss, "the offset is real physics, not zero: got {:.2}° vs s_ss {:.2}°", held_big.to_degrees(), s_ss.to_degrees());
    }

    /// §10.5.4: activity bandwidth scaling moves ω_n by `s` and leaves ζ
    /// invariant — the 13o damping lesson, enforced.
    #[test]
    fn activity_bandwidth_scaling_preserves_damping_ratio() {
        let base = AttitudeLaw::Pd(PdGains { kp: 1.0, kd: 20.0, pointing_db_rad: 0.01, rate_db_rads: 0.001 });
        let (wn0, z0) = base.natural_frequency_and_damping(300.0).unwrap();
        let scaled = tuned_law(&base, ActivityTuning { bandwidth_scale: 0.5, deadband_scale: 2.0 });
        let (wn1, z1) = scaled.natural_frequency_and_damping(300.0).unwrap();
        assert!((wn1 / wn0 - 0.5).abs() < 1e-12);
        assert!((z1 - z0).abs() < 1e-12, "ζ must be invariant: {z0} vs {z1}");
        if let AttitudeLaw::Pd(g) = scaled {
            assert!((g.pointing_db_rad - 0.02).abs() < 1e-15);
        }
        // Settling time scales inversely with bandwidth.
        let t0 = base.settling_time_s(300.0).unwrap();
        let t1 = scaled.settling_time_s(300.0).unwrap();
        assert!((t1 / t0 - 2.0).abs() < 1e-9);
    }

    /// The 12-thruster reference layout is symmetric, so every axis probe
    /// returns the same authority and the worst-axis scalar equals it.
    #[test]
    fn rcs_axis_authority_is_symmetric_for_the_reference_layout() {
        let t = thrusters();
        let a = rcs_axis_authority_nm(&t);
        assert!(a.x > 0.0 && (a.x - a.y).abs() < 1e-12 && (a.y - a.z).abs() < 1e-12, "{a:?}");
        assert!((rcs_worst_axis_authority_nm(&t) - a.x).abs() < 1e-12);
        assert_eq!(rcs_worst_axis_authority_nm(&[]), 0.0);
    }

    #[test]
    fn thrusters_primary_idle_when_nothing_commanded_and_wheels_empty() {
        let cluster = wheels();
        let speeds = [0.0; 4];
        let out = allocate(
            ControlMode::ThrustersPrimary, Vector3::zeros(),
            &cluster, &speeds, &thrusters(), 5.0, MomentumManagementLaw::None,
        );
        assert_eq!(out.rcs_duty_cycle, 0.0);
        assert!(out.rcs_propellant_kg < 1e-15);
    }

    /// `ThrustersOnly` must exclude
    /// the wheel cluster from control entirely -- zero motor torque
    /// regardless of current wheel speed, so `net_body_torque` (which reads
    /// wheel reaction torque through `wheel_motor_torque_nm`) reflects RCS
    /// alone. This is the real distinction from `ThrustersPrimary`, which
    /// still actively drives the wheels toward zero (a real, nonzero
    /// reaction torque) even when RCS is doing the pointing.
    #[test]
    fn thrusters_only_excludes_the_wheel_cluster_entirely() {
        let cluster = wheels();
        let speeds = [200.0, -150.0, 100.0, -50.0]; // nonzero, would drive ThrustersPrimary's unload torque
        let out = allocate(
            ControlMode::ThrustersOnly, Vector3::new(0.0, 0.05, 0.0),
            &cluster, &speeds, &thrusters(), 5.0, MomentumManagementLaw::None,
        );
        assert_eq!(out.wheel_motor_torque_nm, [0.0; 4], "wheel cluster must be fully excluded, got {:?}", out.wheel_motor_torque_nm);
        assert!(out.rcs_duty_cycle > 0.0, "RCS alone must deliver the commanded torque");
        assert!(out.rcs_propellant_kg > 0.0);
        // With zero wheel motor torque, net body torque is RCS alone.
        let net = net_body_torque(&cluster, &out);
        assert!((net - out.rcs_body_torque_avg).norm() < 1e-12, "net torque should equal RCS's contribution alone when wheels are excluded");
    }

    #[test]
    fn thrusters_only_idle_when_nothing_commanded() {
        let cluster = wheels();
        let speeds = [200.0, -150.0, 100.0, -50.0];
        let out = allocate(
            ControlMode::ThrustersOnly, Vector3::zeros(),
            &cluster, &speeds, &thrusters(), 5.0, MomentumManagementLaw::None,
        );
        assert_eq!(out.wheel_motor_torque_nm, [0.0; 4]);
        assert_eq!(out.rcs_duty_cycle, 0.0);
        assert!(out.rcs_propellant_kg < 1e-15);
    }

    /// `rcs_thruster_duty_cycles`
    /// must be indexed identically to the input `rcs_thrusters` slice, with
    /// every SELECTED (torque-aligned) thruster carrying the same nonzero
    /// value as the aggregate `rcs_duty_cycle`, and every unselected
    /// thruster exactly `0.0`.
    #[test]
    fn per_thruster_duty_cycles_match_the_aggregate_selection() {
        let cluster = wheels();
        let t = thrusters();
        let out = allocate(
            ControlMode::ThrustersOnly, Vector3::new(0.0, 0.05, 0.0),
            &cluster, &[0.0; 4], &t, 5.0, MomentumManagementLaw::None,
        );
        assert_eq!(out.rcs_thruster_duty_cycles.len(), t.len());
        assert!(out.rcs_duty_cycle > 0.0);
        let selected = attitude_control::select_firing_thrusters(&Vector3::new(0.0, 0.05, 0.0), &t);
        for (i, (&fire, &duty)) in selected.iter().zip(out.rcs_thruster_duty_cycles.iter()).enumerate() {
            if fire {
                assert!((duty - out.rcs_duty_cycle).abs() < 1e-15, "thruster {i} should fire at the aggregate duty cycle");
            } else {
                assert_eq!(duty, 0.0, "thruster {i} should not fire");
            }
        }
        // At least one thruster actually selected -- otherwise this test
        // would pass vacuously.
        assert!(selected.iter().any(|&f| f));
    }

    /// Empty `rcs_thrusters` must produce an empty `rcs_thruster_duty_cycles`
    /// in every mode, not a panic or a mismatched-length vec.
    #[test]
    fn per_thruster_duty_cycles_empty_when_no_thrusters_configured() {
        let cluster = wheels();
        for mode in [ControlMode::WheelsPrimary, ControlMode::ThrustersPrimary, ControlMode::ThrustersOnly] {
            let out = allocate(
                mode, Vector3::new(0.0, 0.05, 0.0),
                &cluster, &[0.0; 4], &[], 5.0,
                MomentumManagementLaw::ThresholdRcs { gain: 0.02, null_motion_gain: 0.0 },
            );
            assert!(out.rcs_thruster_duty_cycles.is_empty(), "mode {mode:?} should produce an empty vec with no thrusters, got {:?}", out.rcs_thruster_duty_cycles);
        }
    }
}
