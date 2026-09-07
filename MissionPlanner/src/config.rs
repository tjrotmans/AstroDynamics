//! TOML-deserialized mission configuration types.
//!
//! A full mission is specified by a single `.toml` file. Call
//! `MissionConfig::from_file()` to parse it into these types.
//! The `validate` CLI command then checks consistency and prints a summary.
//!
//! Each top-level section (`[target_body]`, `[spacecraft]`, etc.) maps
//! directly to a struct here. All values are SI units unless the field name
//! carries an explicit suffix (e.g. `mu_m3s2`, `thrust_n`, `isp_s`).

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Deserializer};

// ── Top-level ────────────────────────────────────────────────────────────────

/// Complete mission configuration parsed from a `.toml` file.
#[derive(Debug, Clone, Deserialize)]
pub struct MissionConfig {
    pub mission: MissionMeta,
    pub target_body: TargetBodyConfig,
    pub spacecraft: SpacecraftConfig,
    pub trajectory: TrajectoryConfig,
    pub gnc: GncConfig,
    pub simulation: SimulationConfig,
    /// Phase 9 trajectory optimization stage (Layer 1b) — real propagated-dynamics
    /// optimization, distinct from `trajectory`'s closed-form narrowing-stage
    /// solvers. `None` when this mission doesn't use the optimization stage.
    pub optimization: Option<OptimizationConfig>,
    /// Phase 5.2 (roadmap item 13i) — when present, `/api/simulate` flies
    /// `cruise::run_cruise_leg` (the Phase 13 translation+attitude+guidance
    /// composition) seeded from a real Layer-1 design/optimize result,
    /// instead of `simulate::run_orbit_chain`'s synthesized body-centric
    /// orbit. `None` (the default, and the only behavior before this field
    /// existed) preserves the existing body-centric proximity-ops path
    /// completely unchanged. See `cruise.rs`'s module doc comment for the
    /// composition this seeds, and the design notes Phase 5.2 entry for the
    /// known first-cut limitations (fixed SunPointing mode, no SOI-
    /// candidate force-model bodies, no TCM burn execution).
    pub cruise_seed: Option<CruiseSeedConfig>,
}

/// One sample of a Layer-1 reference trajectory, as submitted by the
/// client — mirrors `sim_engine::reference_guidance::ReferencePoint`
/// exactly (kept as a separate `Deserialize` type so `config.rs` doesn't
/// need to depend on `sim_engine` for this one shape; `cruise_seed_to_*`
/// helpers in `cruise.rs` convert between them).
#[derive(Debug, Clone, Deserialize)]
pub struct CruiseReferencePointConfig {
    pub t_s: f64,
    pub r_m: [f64; 3],
    pub v_mps: [f64; 3],
}

/// Seeds a Phase 5.2 cruise-loop `/api/simulate` run. All positions/
/// velocities are heliocentric (the same frame Layer 1's `arc`/
/// `ArcApiPoint` results use) — the client is expected to pass through a
/// prior `/api/design/trajectory` or `/api/optimize` result's own departure
/// state and arc directly, not synthesize one.
#[derive(Debug, Clone, Deserialize)]
pub struct CruiseSeedConfig {
    /// Initial position [m], heliocentric, at `reference[0].t_s` (normally
    /// 0 — the leg's own departure instant).
    pub r0_m: [f64; 3],
    /// Initial velocity [m/s], heliocentric.
    pub v0_m: [f64; 3],
    /// The Layer-1 trajectory to track (report dispersion against — not
    /// corrected in this first cut, see `cruise.rs`). Must have at least 2
    /// points, sorted ascending by `t_s`, with `reference[0].t_s == 0.0`
    /// and `reference[0].r_m/v_mps` matching `r0_m`/`v0_m` (checked in
    /// `check_config` — a mismatched seed/reference start would silently
    /// report a nonzero dispersion at t=0 that has nothing to do with real
    /// tracking quality).
    pub reference: Vec<CruiseReferencePointConfig>,
    /// How long to fly the leg [s]. May be shorter than the reference's own
    /// span (e.g. to check just the leading segment) but not longer.
    pub duration_s: f64,
    /// Control-loop tick [s]. See `cruise_demo.rs`'s doc comment for why
    /// this must respect the vehicle/gains' closed-loop bandwidth, not just
    /// disturbance-torque frequency — `check_config` cannot verify this
    /// (it would need the full PD-gain/inertia natural-frequency
    /// computation `cruise_demo` did by hand), so an unstable choice here
    /// will surface at runtime (divergent pointing error, integrator
    /// warnings), not as a validation error.
    pub tick_s: f64,
    /// Named, prioritized pointing-rule sets. Empty (the
    /// default) preserves the existing behavior completely: a fixed
    /// `SunPointing` attitude hold for the whole leg, exactly as before
    /// this field existed. See `docs/MP/MANUAL.md` §9.4 for why only
    /// the top 2 rules in any one mode can ever control the attitude.
    #[serde(default)]
    pub modes: Vec<GncModeConfig>,
    /// Time-windowed assignment of `modes` (by name) to the leg's timeline.
    /// A tick not covered by any entry falls back to `safe_mode` if set,
    /// else to the legacy fixed `SunPointing` behavior. Overlapping entries
    /// are not rejected — the LAST matching entry (list order) wins, a
    /// simple, documented tie-break rather than an enforced non-overlap
    /// constraint.
    #[serde(default)]
    pub mode_schedule: Vec<ModeScheduleEntryConfig>,
    /// Name of a mode (must exist in `modes`) to use for any tick not
    /// covered by `mode_schedule` — the "exists outside the schedule as
    /// fault response" mode from the original ask, in this first cut
    /// entered by schedule-gap fallback rather than real fault detection
    /// (no fault-detection system exists anywhere in this codebase yet;
    /// building one is out of scope here — see the design notes).
    pub safe_mode: Option<String>,
    /// Position tracks for any `PointingTargetConfig::Body` target
    /// referenced by a rule in `modes` — the client supplies these
    /// precomputed (same reasoning as `reference` itself: `cruise.rs` is
    /// deliberately ANISE-free, taking precomputed position data from the
    /// caller rather than querying an ephemeris live).
    #[serde(default)]
    pub body_tracks: Vec<BodyTrackConfig>,
    /// Trigger a trajectory-correction maneuver (TCM) the first tick
    /// position dispersion against `reference` (`dr_m`) exceeds this [m] —
    /// item 5. `None` (default) preserves the exact pre-TCM
    /// coast-only behavior. Requires `spacecraft.propulsion` to also be
    /// set (the thrust/Isp source for the burn) — set without it is a
    /// no-op (nothing to trigger the correction WITH), not an error; see
    /// `cruise::build_tcm_config`.
    #[serde(default)]
    pub tcm_dr_threshold_m: Option<f64>,
    /// Mode-scheduled control tick (`MANUAL.md` §10.5.1):
    /// the tick used while a main-engine burn is FIRING
    /// (`TcmPhase::Burning`), in place of `tick_s`. The discrete-loop
    /// margin caps the attitude bandwidth at `ω_n ≤ 2π/(15·tick)`, so at a
    /// coarse cruise tick BOTH actuator classes derive the same
    /// tick-limited gains and the thrusters' much larger torque authority
    /// is unusable — a burn-attitude hold against a persistent
    /// engine-misalignment torque then degrades no matter the law. A
    /// shorter tick during burns lifts that cap exactly when it matters,
    /// at ~tick_s/burn_tick_s× the per-second compute cost for the burn's
    /// duration only. The thruster-mode law is derived at THIS tick.
    /// Default when unset: `max(tick_s/10, 1 s)` (never longer than
    /// `tick_s`). Must satisfy `0 < burn_tick_s <= tick_s`.
    #[serde(default)]
    pub burn_tick_s: Option<f64>,
    /// Test window (so a full mission run can be shortened for fast
    /// iteration during testing): fly only
    /// `[start_s, end_s]` of the leg. The truth starts at the REFERENCE
    /// state at `start_s` (plus the optional dispersion), `r0_m`/`v0_m` are
    /// ignored, planned burns before `start_s` are skipped (they are part of
    /// the reference already), the tank is assumed full at `start_s`. This
    /// is how an approach-phase change is tested without flying the whole
    /// cruise. `None` = the whole leg, byte-identical to before.
    #[serde(default)]
    pub window: Option<CruiseWindowConfig>,
    /// Decouples REPORTING cadence (the `/api/simulate/:id/stream` WS
    /// broadcast and the `/steps` REST history) from the real control-loop
    /// tick (`cruise_report_stride`). The
    /// control loop itself (PD solve, allocation, propagation) always runs
    /// at the full `tick_s` regardless of this field, and so do
    /// `max_dr_m`/`max_wheel_momentum_nms`/the on-disk CSV/
    /// `mode_transitions` — none of those are affected. Only the
    /// routes-layer relay (WS message + `/steps` push) is skipped on
    /// non-stride ticks — see `run_cruise_streaming`, which always
    /// reports tick 0 and the final tick regardless of stride. `None`/
    /// `Some(1)` both mean "report every tick" (the pre-existing
    /// behavior). Needed because a long leg reported at full `tick_s`
    /// cadence can overflow a JSON string's max length client-side
    /// (confirmed: `ERR_STRING_TOO_LONG` on a real 129-day/`tick_s=30`
    /// leg, ~371,500 ticks) — naively raising `tick_s` instead would also
    /// coarsen the CONTROL loop's own bandwidth, which is not this field's
    /// job (see `tick_s`'s own doc comment).
    ///
    /// Trade-off, stated plainly: `/cancel` is only checked on reported
    /// ticks too, so worst-case cancel latency becomes
    /// `report_stride * tick_s` seconds instead of ~1 tick.
    #[serde(default)]
    pub report_stride: Option<u32>,
    /// Phase-adaptive reporting (a fixed stride makes attitude plots too
    /// coarse during maneuvers and inside a body's SOI): the stride
    /// used INSTEAD of `report_stride` on FAST-DYNAMICS ticks — any tick in
    /// a `Slewing`/`Burning`/`RcsCorrecting` phase, any tick inside a
    /// registered SOI-capture body's sphere (parking coast, capture orbit —
    /// a coarse cruise stride rendered a parking orbit as a jagged polygon
    /// and made the capture-orbit attitude replay swing 10–30° between
    /// samples), and every tick within a trailing window
    /// (`cruise::MANEUVER_REPORT_TRAIL_S`, 600 s) after a `tcm_phase` or
    /// `active_mode` transition (the transition tick itself always relays).
    /// Default 1 = full control-loop resolution there, while cruise coast
    /// keeps the coarse budgeted stride. Those windows are a small share of
    /// a real mission, so the total message count stays near the coast
    /// budget; relay-layer only — control-loop fidelity, `max_dr_m`, the
    /// on-disk CSV and `mode_transitions` are unaffected, exactly like
    /// `report_stride`.
    #[serde(default)]
    pub report_stride_maneuver: Option<u32>,
    /// Phase 13n — scheduled burn EVENTS (a DSM on an MGA leg,
    /// an arrival/capture burn, a departure burn under GNC test) — as
    /// opposed to `tcm_dr_threshold_m`'s REACTIVE, dispersion-triggered
    /// executive. Before this field existed, `cruise_seed` had no way to
    /// fly a planned burn at all: an MGA leg's own `mga_dsm_positions_m`/
    /// `mga_dv_dsms_ms` were never executed, and an arrival/capture burn
    /// for an Orbit/Landing/Rendezvous/SampleReturn mission never fired
    /// either (the client's own reference-building silently dropped every
    /// sample past the patched-conic handoff, so the replay ended before
    /// any capture burn could occur). Executed by the SAME `BurnAttitude` +
    /// body-+X-fire mechanism `tcm_dr_threshold_m`'s reactive executive
    /// already implements (`TcmPhase::Slewing`/`Burning`) — always via the
    /// main engine (never `RcsCorrecting`; a scheduled DSM/capture burn is
    /// exactly the "big, deliberate maneuver" case item #4's own actuator
    /// choice already defaults to the main engine for). Requires
    /// `spacecraft.propulsion` (the same thrust/Isp source
    /// `tcm_dr_threshold_m` uses) — a nonempty list with no propulsion
    /// configured is a `check_config` error, matching that field's own
    /// no-propulsion-source validation.
    ///
    /// **Known, documented limitation (not silently skipped): the ΔV fired
    /// is exactly `dv_inertial_mps`, replayed verbatim from whatever the
    /// caller supplied (the Phase 01 arc's own DSM/capture ΔV) — it is NOT
    /// re-solved fresh from the vehicle's real (dispersed) state at the
    /// trigger epoch.** Real ops practice re-targets a scheduled maneuver
    /// through the shaping intent (the same `tcm_lambert_correction`-style
    /// solve the reactive executive already does) so it also absorbs
    /// whatever correction has accumulated since the last one — this first
    /// cut does not do that; a planned burn and a genuinely dispersed
    /// vehicle can therefore miss the intended shaping. Flagged in
    /// the design notes Phase 13n as real, scoped future work, not forgotten.
    #[serde(default)]
    pub planned_burns: Vec<PlannedBurnConfig>,
}

/// One scheduled burn event — see [`CruiseSeedConfig::planned_burns`].
/// `cruise_seed.window` — see `CruiseSeedConfig::window`.
#[derive(Debug, Clone, Deserialize)]
pub struct CruiseWindowConfig {
    /// Window start [s on the reference clock], `>= 0`.
    pub start_s: f64,
    /// Window end [s], `> start_s` and `<= duration_s`.
    pub end_s: f64,
    /// Optional initial position dispersion [m], added to the reference
    /// position at `start_s` (e.g. to test the executive against a known
    /// miss). Default none.
    #[serde(default)]
    pub initial_dr_m: Option<[f64; 3]>,
    /// Optional initial velocity dispersion [m/s]. Default none.
    #[serde(default)]
    pub initial_dv_mps: Option<[f64; 3]>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlannedBurnConfig {
    /// Mission-elapsed time to trigger this burn [s]. `planned_burns` must
    /// be sorted ascending by this field (checked in `check_config`,
    /// matching `reference`/`body_tracks`' own sorted-list convention).
    pub epoch_s: f64,
    /// The ΔV to deliver, heliocentric/inertial frame [m/s] — direction
    /// AND magnitude combined in one vector, straight from the adopted
    /// arc's own DSM/capture-burn fields. Fired VERBATIM unless
    /// `target_epoch_s` is set (see that field) — the DEFAULT behavior for
    /// every entry that predates the "solve fresh" fix (nuance
    /// 2), and the only option for a burn with no natural position-target
    /// (an arrival/capture or departure burn shapes VELOCITY, not
    /// position — re-targeting those needs different math, not yet built,
    /// see this struct's own module-level context in the design notes Phase 13n).
    pub dv_inertial_mps: [f64; 3],
    /// Nuance 2 — when set, this burn's ACTUAL fired ΔV is
    /// re-solved fresh at ignition from the vehicle's real (possibly
    /// dispersed) state, targeting the reference trajectory's own recorded
    /// position at this epoch [s since departure] — the exact same
    /// `tcm_lambert_correction` mechanism the reactive TCM executive
    /// (13m/13p) already uses, just pointed at this burn's own known
    /// shaping intent (typically the next flyby encounter, or the leg's
    /// own final arrival point for the last DSM) instead of a
    /// reactive-trigger-computed horizon. This is what lets a scheduled
    /// maneuver absorb whatever correction has accumulated since the last
    /// one, rather than treating shaping and correction as two
    /// disconnected ΔV pools. Falls back to firing `dv_inertial_mps`
    /// verbatim if the fresh solve fails (Lambert degenerate/no solution)
    /// — a scheduled maneuver still fires on its nominal plan rather than
    /// being silently skipped. Only meaningful for a DSM (a POSITION
    /// target) — leave unset for an arrival/capture or departure burn.
    /// `None` (default) preserves the exact pre-verbatim
    /// behavior.
    #[serde(default)]
    pub target_epoch_s: Option<f64>,
    /// Review D5 — when set, this burn is an ARRIVAL/CAPTURE
    /// burn at the named body, and its ACTUAL fired ΔV is re-solved fresh
    /// at trigger time from the vehicle's real (dispersed) state — a
    /// VELOCITY-matching solve, the genuinely different math 13n's own doc
    /// note said a capture burn needs (`tcm_lambert_correction` shapes
    /// POSITION and is the wrong tool entirely): scale the body-relative
    /// speed to local circular speed along the CURRENT relative-velocity
    /// direction (`cruise::solve_capture_burn_dv` — the same provably-
    /// always-bound construction `ArrivalCapture::dv_capture_ms`'s pricing
    /// and `propagate_captured_orbit`'s real-state branch already use).
    /// Requires a matching `body_tracks` entry with a resolvable
    /// `mu_m3s2`, and is mutually exclusive with `target_epoch_s` (both
    /// checked in `check_config`). Falls back to firing `dv_inertial_mps`
    /// verbatim if the track lookup fails at runtime — a scheduled capture
    /// still fires on its nominal plan rather than being silently skipped.
    /// Solved at TRIGGER time (not continuously re-solved through the
    /// slew like 13p's reactive corrections — a velocity-matching target
    /// drifts far more slowly than a shrinking-TOF Lambert one; a
    /// documented approximation, not an oversight).
    #[serde(default)]
    pub capture_body: Option<String>,
    /// this burn is delivered by an EXTERNAL stage (a launch
    /// vehicle's upper stage for an Earth departure — the launcher-provided
    /// ΔV pool of the Phase 8c two-pool model, `design.rs::
    /// compute_launch_vehicle_check`), not by the spacecraft's own engine:
    /// applied as an impulsive `dv_inertial_mps` at `epoch_s`, with no
    /// slew, no spacecraft-attitude requirement, and NO draw on
    /// `propellant_mass_kg`. Without this, a Phase-03 replay of a
    /// launcher-injected mission fires the multi-km/s injection through
    /// the onboard engine and drains the tank before cruise begins (found
    /// on a real replay: a 5.9 km/s injection vs. a tank good for ~0.5
    /// km/s). Default `false` = the onboard-engine behavior every existing
    /// entry has.
    #[serde(default)]
    pub external_stage: bool,
    /// Free-text label for telemetry/debugging only (e.g. `"DSM leg 2"`,
    /// `"Mercury capture"`) — never consulted by the executive itself.
    #[serde(default)]
    pub label: String,
}

/// Which inertial direction a [`PointingRuleConfig`] targets.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum PointingTargetConfig {
    /// Heliocentric origin — trivial to resolve (no track needed), matches
    /// the existing `CruisePointingMode::SunPointing` convention.
    Sun,
    /// Any body with a matching entry in `body_tracks` (Earth, the target
    /// body, a flyby body, ...) — deliberately one uniform variant rather
    /// than separate `Earth`/`TargetBody`/`FlybyBody` cases, since
    /// resolution is identical for all three (a name-keyed track lookup).
    Body { name: String },
    /// The spacecraft's own current velocity direction (prograde).
    Velocity,
    /// A fixed inertial direction, e.g. a specific burn attitude.
    Inertial { direction: [f64; 3] },
}

/// One pointing rule: a placed hardware item's boresight/normal should
/// point at `target`. `hardware_index` indexes `spacecraft.hardware`
/// directly (not by name — `HardwareItem` has no general name field) and
/// must resolve to a hardware type with a real pointing direction
/// (`StarTracker`/`OpNavCamera`/`Lidar`/`CommAntenna`/`SolarPanel`/
/// `CustomPlate` — checked in `check_config`; `ReactionWheelCluster`/`RCS`/
/// `RcsThruster`/`IMU` have no meaningful boresight and are rejected).
#[derive(Debug, Clone, Deserialize)]
pub struct PointingRuleConfig {
    pub hardware_index: usize,
    pub target: PointingTargetConfig,
}

/// A named, prioritized pointing-rule set — a GNC "mode." `rules`' LIST
/// ORDER is the priority order (index 0 = highest/primary) — no separate
/// priority field, so there's nothing that can drift out of sync with the
/// order itself.
#[derive(Debug, Clone, Deserialize)]
pub struct GncModeConfig {
    pub name: String,
    pub rules: Vec<PointingRuleConfig>,
    /// Marks this mode's pointing as non-negotiable — the design notes Phase 13m
    /// item #4: when this mode is active and a TCM correction
    /// is needed, the executive must not slew the spacecraft off this
    /// mode's rules to fire the main engine. `false` (default) preserves
    /// the pre-behavior for every existing config. See
    /// `cruise::choose_tcm_actuator`/`docs/MP/MANUAL.md` §9.2.
    #[serde(default)]
    pub pointing_locked: bool,
}

/// One timeline entry assigning a named mode to a `[start_s, end_s)`
/// window of the cruise leg.
#[derive(Debug, Clone, Deserialize)]
pub struct ModeScheduleEntryConfig {
    pub start_s: f64,
    pub end_s: f64,
    pub mode: String,
}

/// A precomputed heliocentric position track for one named body, for
/// `PointingTargetConfig::Body` resolution. Reuses
/// [`CruiseReferencePointConfig`]'s shape (only `r_m` is consulted; `v_mps`
/// is accepted but ignored — kept for schema uniformity with `reference`
/// rather than introducing a second, position-only point type).
#[derive(Debug, Clone, Deserialize)]
pub struct BodyTrackConfig {
    pub name: String,
    /// Precomputed samples — OR empty (the `#[serde(default)]`, review-B2
    /// server-resolved form): an empty `track` on a named body
    /// asks the SERVER to sample its own ANISE ephemeris over the leg's
    /// duration before the job starts (`design::resolve_named_body_tracks`,
    /// called from the `/api/simulate` route ahead of `check_config`, whose
    /// ≥2-points rule then doubles as the safety net). The backend owns the
    /// kernels; shipping ephemeris client→server through JSON at
    /// client-chosen density was an inversion of responsibility — this
    /// makes the client-side sampling (and its interpolation-density
    /// pitfalls) unnecessary for any ANISE-covered body. A body without
    /// ANISE coverage still requires an explicit client-supplied track.
    #[serde(default)]
    pub track: Vec<CruiseReferencePointConfig>,
    /// Epoch of `t_s = 0` [Julian Date] for the server-resolved form —
    /// only consulted when `track` is empty. Defaults to
    /// `trajectory.departure_epoch` when unset; an empty track with
    /// neither is a validation error (nothing anchors the sampling).
    #[serde(default)]
    pub epoch_jd: Option<f64>,
    /// Gravitational parameter [m³/s²] — added so the SAME
    /// track this struct already carries for pointing resolution can also
    /// drive real third-body gravitational perturbation in the cruise
    /// loop (`cruise::build_body_track_perturbers`), instead of a second
    /// mechanism. `None` falls back to a `body_models::TargetBody::
    /// by_name(&name)` catalog lookup; a track whose body isn't in the
    /// catalog AND has no explicit `mu_m3s2` is used for pointing only
    /// (silently excluded from third-body perturbation) — this field is
    /// therefore optional so a synthetic/test-only track (not a real
    /// gravitational body) doesn't need one. `#[serde(default)]` so every
    /// existing `body_tracks` entry (which predates this field) still
    /// deserializes without changes.
    #[serde(default)]
    pub mu_m3s2: Option<f64>,
    /// Registers this track as a real SOI-switching CENTRAL-BODY candidate
    /// in the cruise loop, not just a third-body perturber —.
    /// `false` (default) preserves the exact pre-existing behavior for
    /// every config written before this field existed: `soi_radius_m` is
    /// always `None` for every `body_track`, so `step_tick`'s internal
    /// `resolve_central_body` (see `crates/sim_engine::propagator6dof`) can
    /// never select it, and the cruise loop's propagated truth stays under
    /// Sun-only central gravity even during a real close approach — this
    /// was the confirmed root cause of a genuine captured orbit (around,
    /// e.g., Mercury) reading as a flyby in a mission-validation replay: a
    /// correct heliocentric reference showed the real orbit, but the
    /// propagated TRUTH never switched central body, so dispersion against
    /// that reference blew up as if the spacecraft had flown straight
    /// through instead of capturing.
    ///
    /// `true` sizes a real Laplace SOI radius (`trajectory_solver::
    /// laplace_soi_radius_m`, the same formula/convention Layer 1's
    /// `design::propagator_body_entries` already uses, just fed from this
    /// client-supplied track instead of a live ANISE query) via
    /// `cruise::build_body_track_perturbers` — see that function's doc
    /// comment for exactly how the primary-relative distance is resolved.
    /// Requires `mu_m3s2` to resolve (explicit override or catalog lookup)
    /// — `check_config` rejects `soi_capture: true` on a track that can't
    /// resolve one, since silently leaving it third-body-only would defeat
    /// the whole point of setting this flag without telling the config
    /// author why.
    #[serde(default)]
    pub soi_capture: bool,
}

impl MissionConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Self = toml::from_str(&text)?;
        Ok(cfg)
    }
}

// ── Mission metadata ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct MissionMeta {
    pub name: String,
    pub objective: MissionObjective,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub enum MissionObjective {
    Flyby,
    Orbit,
    Landing,
    Rendezvous,
    SampleReturn,
}

impl std::fmt::Display for MissionObjective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Flyby => write!(f, "Flyby"),
            Self::Orbit => write!(f, "Orbit"),
            Self::Landing => write!(f, "Landing"),
            Self::Rendezvous => write!(f, "Rendezvous"),
            Self::SampleReturn => write!(f, "Sample Return"),
        }
    }
}

// ── Target body ──────────────────────────────────────────────────────────────

/// `mu_m3s2`, `radius_m`, `gravity_model`, `atmosphere`, and `j2`/`j3`/`j4` may
/// all be omitted from the TOML and are then resolved from the
/// `body_models::TargetBody` catalog by `name` (case-insensitive) — see the
/// manual `Deserialize` impl below. Any field given explicitly overrides the
/// catalog value, so a mission can use a known body's name as a shorthand
/// and still tweak individual constants (e.g. a refined μ estimate).
#[derive(Debug, Clone)]
pub struct TargetBodyConfig {
    pub name: String,
    /// Gravitational parameter [m³/s²]
    pub mu_m3s2: f64,
    /// Mean equatorial radius [m]
    pub radius_m: f64,
    pub gravity_model: GravityModel,
    pub atmosphere: AtmosphereModel,
    /// J2 — required when gravity_model is J2 or J2J3J4
    pub j2: Option<f64>,
    /// J3 — required when gravity_model is J2J3J4
    pub j3: Option<f64>,
    /// J4 — required when gravity_model is J2J3J4
    pub j4: Option<f64>,
    pub ephemeris: EphemerisSource,
    /// Keplerian elements for bodies not covered by DE440S (e.g. small asteroids).
    /// Required when `ephemeris = "Keplerian"` and the design stage is run.
    /// Not resolvable from the body catalog — orbital elements are epoch-
    /// dependent and not a fixed physical constant of the body.
    #[allow(dead_code)]
    pub keplerian_orbit: Option<KeplerianOrbit>,
    /// Third-body gravitational perturbers active during simulation (e.g. `["Phobos", "Deimos"]`
    /// for a Mars orbiter, or `["Jupiter"]` for a Europa orbiter). Each name must resolve to an
    /// entry in the `body_models::TargetBody` catalog. Empty = no third-body perturbations.
    pub third_bodies: Vec<String>,
    /// North pole right ascension [deg], ICRF/J2000 constant term — needed to apply
    /// J2/J2J3J4 gravity fidelity correctly when this body is the propagator's central
    /// body (see `orbital_models::GravityModel::zonal_harmonics_body_oriented`).
    /// `None` when not in the catalog and not specified manually — callers must fall
    /// back to point-mass central fidelity and warn rather than guess an orientation.
    pub pole_ra_deg: Option<f64>,
    /// North pole declination [deg], ICRF/J2000 constant term. See `pole_ra_deg`.
    pub pole_dec_deg: Option<f64>,
}

impl<'de> Deserialize<'de> for TargetBodyConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawTargetBody {
            name: String,
            mu_m3s2: Option<f64>,
            radius_m: Option<f64>,
            gravity_model: Option<GravityModel>,
            atmosphere: Option<AtmosphereModel>,
            j2: Option<f64>,
            j3: Option<f64>,
            j4: Option<f64>,
            ephemeris: EphemerisSource,
            keplerian_orbit: Option<KeplerianOrbit>,
            #[serde(default)]
            third_bodies: Vec<String>,
            pole_ra_deg: Option<f64>,
            pole_dec_deg: Option<f64>,
        }

        let raw = RawTargetBody::deserialize(deserializer)?;
        let preset = body_models::TargetBody::by_name(&raw.name);
        let missing = |field: &str| {
            serde::de::Error::custom(format!(
                "target_body '{}': '{field}' not specified, and '{}' is not in the \
                 body_models catalog (known: Sun, Bennu, Earth, Moon, Mars, Apophis, Ryugu, \
                 Jupiter, Saturn, Europa, Titan, Phobos, Deimos, Eros, Didymos) — \
                 specify it manually",
                raw.name, raw.name,
            ))
        };

        let mu_m3s2 = match raw.mu_m3s2 {
            Some(v) => v,
            None => preset.as_ref().map(|p| p.mu_m3s2).ok_or_else(|| missing("mu_m3s2"))?,
        };
        let radius_m = match raw.radius_m {
            Some(v) => v,
            None => preset.as_ref().map(|p| p.radius_m).ok_or_else(|| missing("radius_m"))?,
        };
        let (gravity_model, j2, j3, j4) = match raw.gravity_model {
            Some(gm) => (gm, raw.j2, raw.j3, raw.j4),
            None => {
                let p = preset.as_ref().ok_or_else(|| missing("gravity_model"))?;
                match &p.gravity {
                    body_models::GravityModel::PointMass => (GravityModel::PointMass, None, None, None),
                    body_models::GravityModel::J2 { j2 } => (GravityModel::J2, Some(*j2), None, None),
                    body_models::GravityModel::J2J3J4 { j2, j3, j4 } => {
                        (GravityModel::J2J3J4, Some(*j2), Some(*j3), Some(*j4))
                    }
                }
            }
        };
        let atmosphere = match raw.atmosphere {
            Some(a) => a,
            None => {
                let p = preset.as_ref().ok_or_else(|| missing("atmosphere"))?;
                match &p.atmosphere {
                    body_models::AtmosphereModel::None => AtmosphereModel::None,
                    body_models::AtmosphereModel::Exponential { .. } => AtmosphereModel::Exponential,
                }
            }
        };

        // Pole orientation has no "missing" error — many bodies (every small
        // body in the catalog so far) legitimately have none yet. Callers
        // applying zonal-harmonic central fidelity must check for `None` and
        // fall back to point-mass with a warning (see the design notes).
        let pole_ra_deg = raw.pole_ra_deg.or_else(|| preset.as_ref().and_then(|p| p.pole_ra_deg));
        let pole_dec_deg = raw.pole_dec_deg.or_else(|| preset.as_ref().and_then(|p| p.pole_dec_deg));

        Ok(TargetBodyConfig {
            name: raw.name,
            mu_m3s2,
            radius_m,
            gravity_model,
            atmosphere,
            j2,
            j3,
            j4,
            ephemeris: raw.ephemeris,
            keplerian_orbit: raw.keplerian_orbit,
            third_bodies: raw.third_bodies,
            pole_ra_deg,
            pole_dec_deg,
        })
    }
}

/// Classical heliocentric orbital elements from JPL Horizons or equivalent.
///
/// All angles in degrees (converted to radians at use); SMA in AU.
/// Elements should be in the ecliptic J2000 frame (Horizons default: `ECLIPTIC`).
/// Fields are intentionally stored for use by the design stage.
#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct KeplerianOrbit {
    /// Semi-major axis [AU]
    pub sma_au: f64,
    /// Eccentricity (0 ≤ e < 1)
    pub eccentricity: f64,
    /// Inclination [deg]
    pub inclination_deg: f64,
    /// Right ascension of ascending node [deg]
    pub raan_deg: f64,
    /// Argument of periapsis [deg]
    pub aop_deg: f64,
    /// Mean anomaly at reference epoch [deg]
    pub mean_anomaly_deg: f64,
    /// Reference epoch as Julian Date (TDB) from JPL Horizons
    pub epoch_jd: f64,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub enum GravityModel {
    PointMass,
    J2,
    J2J3J4,
    SphericalHarmonic,
}

impl std::fmt::Display for GravityModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PointMass => write!(f, "Point Mass"),
            Self::J2 => write!(f, "J2"),
            Self::J2J3J4 => write!(f, "J2/J3/J4"),
            Self::SphericalHarmonic => write!(f, "Spherical Harmonic"),
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub enum AtmosphereModel {
    None,
    Exponential,
    NRLMSISE,
}

impl std::fmt::Display for AtmosphereModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Exponential => write!(f, "Exponential"),
            Self::NRLMSISE => write!(f, "NRLMSISE"),
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub enum EphemerisSource {
    Keplerian,
    Anise,
    Custom,
}

impl std::fmt::Display for EphemerisSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Keplerian => write!(f, "Keplerian"),
            Self::Anise => write!(f, "ANISE (DE440S)"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

// ── Spacecraft ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct SpacecraftConfig {
    /// Total wet mass [kg]
    pub mass_kg: f64,
    /// Dry mass (no propellant) [kg]
    pub dry_mass_kg: f64,
    /// Propellant mass [kg]
    pub propellant_mass_kg: f64,
    /// Bus dimensions [x, y, z] [m]
    pub bus_dims_m: [f64; 3],
    /// Principal moments of inertia [Ixx, Iyy, Izz] [kg·m²]
    pub inertia_diag_kgm2: [f64; 3],
    /// When `true`, the live simulation uses the diagonal of
    /// `vehicle_properties::compute_vehicle_properties(cfg)`'s real
    /// parallel-axis-derived inertia tensor (bus box + every placed/
    /// itemized `HardwareItem`, about the true center of mass) instead of
    /// this struct's own manually-typed `inertia_diag_kgm2` — spacecraft-
    /// builder wiring. `None`/`false` (every existing
    /// config) is byte-for-byte the old behavior: `inertia_diag_kgm2` stays
    /// authoritative. Deliberately opt-in, not automatic — the derived
    /// value can differ substantially from a manually-typed one (e.g. a
    /// uniform-box approximation vs. a real CAD-derived estimate), so
    /// flipping every existing config's rotational dynamics silently would
    /// be a real, undiscussed physics change. Off-diagonal (cross-coupling)
    /// terms from the full tensor are NOT used here — `SpacecraftProperties`
    /// is diagonal-only by signature (see `docs/MP/MANUAL.md` §6.4 and
    /// the note on why that's a separate, larger migration).
    ///
    /// Also gates `simulate::build_plates`'s torque-arm fix: every
    /// `Plate.center_body` (bus faces, panels, `CustomPlate` entries) is
    /// re-expressed relative to the same derived CoM before the SRP torque
    /// law uses it as an `r x F` arm, instead of the geometric-center
    /// origin every `HardwareItem` placement field is relative to. The two
    /// fixes share one flag deliberately — both are "trust
    /// `compute_vehicle_properties`'s real derived geometry" toggles, and a
    /// mission opting into one almost certainly wants the other too.
    pub derive_inertia_from_geometry: Option<bool>,
    pub srp_model: SrpModel,
    pub propulsion: Option<PropulsionConfig>,
    /// Center-of-pressure to center-of-mass offset [m], for worst-case SRP
    /// disturbance torque sizing. Defaults to 10% of the largest bus dimension
    /// (SMAD heuristic) when not specified — see `gnc_design::srp_torque_max()`.
    pub cp_cg_offset_m: Option<f64>,
    /// SRP reflectivity coefficient C_R used by the Phase 4 truth simulation's
    /// SRP acceleration (cannonball or flat-plate). Defaults to 1.4 — typical
    /// absorptive/reflective mixed spacecraft surface, SMAD (Wertz & Larson)
    /// range 1.2-1.5 — the same default the GNC design stage already assumes
    /// for torque sizing (`gnc_design::SRP_CR_DEFAULT`).
    pub reflectivity_cr: Option<f64>,
    /// Launch vehicle name, matched case-insensitively against
    /// `hardware_catalog::LaunchVehicleSpec::catalog()` — only meaningful
    /// when `[trajectory].departure_body` is Earth (or unset); a rocket
    /// provides the departure C3 for free there (checked against this
    /// vehicle's curve), whereas a non-Earth departure has no rocket and
    /// charges a real escape burn to the spacecraft's own propellant
    /// instead (Phase 8c — see the design notes).
    pub launch_vehicle: Option<String>,
    #[serde(default)]
    pub hardware: Vec<HardwareItem>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub enum SrpModel {
    Cannonball,
    FlatPlate,
    NPlate,
}

impl std::fmt::Display for SrpModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cannonball => write!(f, "Cannonball"),
            Self::FlatPlate => write!(f, "Flat-Plate"),
            Self::NPlate => write!(f, "N-Plate"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PropulsionConfig {
    #[serde(rename = "type")]
    pub kind: PropulsionType,
    /// Specific impulse [s]
    pub isp_s: f64,
    /// Thrust [N]
    pub thrust_n: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub enum PropulsionType {
    Monoprop,
    Biprop,
    Ion,
    HallEffect,
    ColdGas,
}

impl std::fmt::Display for PropulsionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Monoprop => write!(f, "Monoprop"),
            Self::Biprop => write!(f, "Biprop"),
            Self::Ion => write!(f, "Ion"),
            Self::HallEffect => write!(f, "Hall-Effect"),
            Self::ColdGas => write!(f, "Cold Gas"),
        }
    }
}

/// Hardware items carried on the spacecraft.
///
/// Each `[[spacecraft.hardware]]` TOML table entry deserialises into one of
/// these variants, discriminated by the `type` key.
/// Fields are intentionally stored for future use by the sim engine (Phase 4).
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum HardwareItem {
    ReactionWheelCluster {
        model: Option<String>,
        count: u32,
        /// Maximum wheel speed [rad/s]
        max_speed_rads: Option<f64>,
        /// Maximum output torque per wheel [N·m]
        max_torque_nm: Option<f64>,
        /// Wheel spin-axis inertia [kg·m²]
        inertia_kgm2: Option<f64>,
        /// PER-WHEEL mass [kg] (derived-mass-properties
        /// extension). `None` falls back to
        /// `hardware_catalog::ReactionWheelSpec::medium().mass_kg` (or the
        /// matching catalog grade when `model` names one). No placement
        /// field exists for this variant (see its own description
        /// elsewhere) — wheels are assumed to sit at the spacecraft's
        /// geometric-center origin for the parallel-axis rollup, so a
        /// wheel cluster's mass counts toward total mass/CoM but
        /// contributes (approximately) zero inertia arm of its own,
        /// consistent with wheels typically being mounted near the bus
        /// center in practice.
        mass_kg: Option<f64>,
    },
    RCS {
        /// Thrust per thruster [N]
        thrust_n: Option<f64>,
        /// Number of thrusters
        count: Option<u32>,
        /// Moment arm [m]
        moment_arm_m: Option<f64>,
        /// PER-THRUSTER mass [kg]. `None` falls back to
        /// `hardware_catalog::ThrusterSpec::monoprop().mass_kg`. No
        /// placement field (aggregate cluster, see this variant's own
        /// description) — same geometric-center-origin assumption as
        /// `ReactionWheelCluster::mass_kg`.
        mass_kg: Option<f64>,
    },
    /// One real, individually-placed RCS thruster (spacecraft-builder
    /// design) — for per-thruster torque-arm (r × F) and
    /// plume-impingement checks, AND real
    /// dynamics: `simulate.rs::rcs_from_hardware` builds the sim's actual
    /// `Vec<Thruster>` directly from any `RcsThruster` entries present,
    /// exactly as placed — real position/direction, no smoothing — falling
    /// back to the aggregate `RCS` variant's symmetric layout only when no
    /// `RcsThruster` entries exist.
    RcsThruster {
        /// Thrust magnitude [N].
        thrust_n: f64,
        /// Thruster position relative to spacecraft CoM, body frame [m].
        position_m: [f64; 3],
        /// Thrust direction — the direction of the FORCE applied to the
        /// spacecraft (same convention as `propagator6dof::BurnConfig::body_dir`,
        /// not the nozzle/exhaust direction, which is the opposite sense),
        /// body frame. Normalized on load.
        direction: [f64; 3],
        /// This thruster's own mass [kg]. `None` falls back to
        /// `hardware_catalog::ThrusterSpec::monoprop().mass_kg`.
        mass_kg: Option<f64>,
        /// This thruster's own specific impulse [s] — per-thruster
        /// so a layout mixing thruster classes (e.g. fine
        /// ColdGas thrusters for pointing alongside a higher-thrust
        /// Monoprop set for coarse slews) prices propellant against each
        /// thruster's own real Isp, not one aggregate value borrowed from
        /// whichever class happened to be assumed for the whole set.
        /// `None` falls back to `hardware_catalog::ThrusterSpec::
        /// monoprop().isp_s`, matching `mass_kg`'s own fallback
        /// convention.
        #[serde(default)]
        isp_s: Option<f64>,
    },
    StarTracker {
        model: Option<String>,
        /// 1-σ attitude noise per axis [rad]
        noise_rad: Option<f64>,
        /// Boresight direction, body frame (unit; normalized on load).
        /// `None` = unplaced (Phase 1: rendered generically, no real
        /// pointing geometry) — spacecraft-builder placement extension,
        ///.
        boresight: Option<[f64; 3]>,
        /// Sensor position relative to spacecraft CoM, body frame [m].
        /// Cosmetic only (star trackers have no meaningful parallax at
        /// spacecraft scale) — kept for a consistent per-item placement
        /// representation with the other placeable sensors.
        position_m: Option<[f64; 3]>,
        /// Full field-of-view half-angle [deg], for the FOV-cone render
        /// and occlusion warnings (e.g. "star tracker stares into the
        /// array"). `None` = no cone rendered.
        fov_deg: Option<f64>,
        /// Unit mass [kg]. `None` falls back to
        /// `hardware_catalog::StarTrackerSpec::medium().mass_kg` (or the
        /// matching catalog grade when `model` names one).
        mass_kg: Option<f64>,
    },
    IMU {
        /// 1-σ ΔV noise per axis per step [m/s]
        noise_sigma_mps: Option<f64>,
        /// Unit mass [kg]. `None` falls back to
        /// `hardware_catalog::ImuSpec::medium().mass_kg`.
        mass_kg: Option<f64>,
    },
    OpNavCamera {
        /// 1-σ bearing noise [mrad]
        bearing_noise_mrad: Option<f64>,
        /// 1-σ angular size noise [mrad]
        angular_size_noise_mrad: Option<f64>,
        /// Boresight direction, body frame (unit; normalized on load).
        /// See `StarTracker::boresight`.
        boresight: Option<[f64; 3]>,
        /// Camera position relative to spacecraft CoM, body frame [m].
        position_m: Option<[f64; 3]>,
        /// Full field-of-view half-angle [deg]. See `StarTracker::fov_deg`.
        fov_deg: Option<f64>,
        /// Unit mass [kg]. `None` falls back to
        /// `hardware_catalog::OpNavCameraSpec::medium().mass_kg`.
        mass_kg: Option<f64>,
    },
    Lidar {
        /// 1-σ range noise [m]
        range_noise_m: Option<f64>,
        /// Maximum operating range [m]
        max_range_m: Option<f64>,
        /// Boresight direction, body frame (unit; normalized on load).
        /// See `StarTracker::boresight`.
        boresight: Option<[f64; 3]>,
        /// Sensor position relative to spacecraft CoM, body frame [m].
        position_m: Option<[f64; 3]>,
        /// Full field-of-view half-angle [deg]. See `StarTracker::fov_deg`.
        fov_deg: Option<f64>,
        /// Unit mass [kg]. `None` falls back to
        /// `hardware_catalog::LidarSpec::medium().mass_kg`.
        mass_kg: Option<f64>,
    },
    /// Comm antenna (HGA/LGA). New
    /// variant (no prior representation existed); renders identically to a
    /// star-tracker/camera boresight cone. Sets up the classic simultaneous-
    /// pointing-constraint problem (panels at Sun + HGA at Earth + camera at
    /// target) for the Phase 03 pointing-mode/attitude-commander work.
    CommAntenna {
        /// Boresight direction, body frame (unit; normalized on load).
        boresight: [f64; 3],
        /// Full beamwidth [deg] (half-angle of the comm cone — same
        /// convention as `fov_deg` elsewhere in this enum, named
        /// differently here since "beamwidth" is the standard RF term).
        beamwidth_deg: f64,
        /// Antenna position relative to spacecraft CoM, body frame [m].
        /// Cosmetic only (no parallax effect modeled), kept for a
        /// consistent per-item placement representation.
        position_m: Option<[f64; 3]>,
        /// Unit mass [kg]. `None` falls back to a representative HGA-class
        /// default (`DEFAULT_COMM_ANTENNA_MASS_KG`, since no dedicated
        /// antenna catalog exists yet — `hardware_catalog::DsnLinkSpec` is
        /// the closest existing spec but describes the ground-link
        /// terminal's performance grade, not general antenna geometry).
        mass_kg: Option<f64>,
    },
    SolarPanel {
        /// Total panel area [m²] — used directly for the unplaced/aggregate
        /// case (existing behavior, unchanged); ignored in favor of
        /// `width_m * height_m` when a placed panel supplies both.
        area_m2: f64,
        /// Panel efficiency (0–1)
        efficiency: Option<f64>,
        // ── Spacecraft-builder placement extension, all
        // optional — an unplaced/aggregate SolarPanel entry (every existing
        // config) omits these and behaves exactly as before (contributes
        // no SRP plate of its own; area only feeds the automatic symmetric
        // ±y panel pair in `simulate.rs::build_plates`). Supplying
        // `position_m` + `normal` makes this a real placed,
        // SRP-force-contributing panel instead (own `Plate`, not folded
        // into the automatic pair) — the "own fields" choice from the
        // ask, over standardizing placed panels as `CustomPlate`, because
        // the mockup review wants independent live width_m/height_m
        // resize handles and a gimbal flag, neither of which `CustomPlate`
        // represents. ──────────────────────────────────────────────────
        /// Panel geometric centre relative to spacecraft CoM, body frame
        /// [m].
        position_m: Option<[f64; 3]>,
        /// Deployed outward normal, body frame (unit; normalized on load).
        normal: Option<[f64; 3]>,
        /// Panel width [m] (in-plane). With `height_m`, overrides
        /// `area_m2` for the plate this panel contributes.
        width_m: Option<f64>,
        /// Panel height [m] (in-plane). See `width_m`.
        height_m: Option<f64>,
        /// Articulation mechanism —
        /// the three real mechanism classes (see `docs/MP/MANUAL.md`
        /// §9.5): `None` = fixed (body-mounted, no articulation — the
        /// panel normal is a hard spacecraft-attitude constraint, same as
        /// today); `Some(OneAxis)` = single rotation axis (exact Sun-
        /// tracking only when the Sun lies in the plane perpendicular to
        /// that axis; a real cosine loss otherwise); `Some(TwoAxis)` =
        /// full independent Sun-tracking (a real Solar Array Drive
        /// Assembly — decouples panel pointing from spacecraft attitude
        /// entirely). Schema-only for now — not yet consumed by any
        /// solver; added now so it is present from the
        /// start of the Phase 03 pointing-mode work.
        articulation: Option<PanelArticulation>,
        /// Specular reflectivity ρ_s ∈ [0, 1] for a PLACED panel's own
        /// `Plate`. Defaults to the same 0.08 the automatic panel pair
        /// uses when omitted. Meaningless (ignored) for an unplaced entry.
        rho_s: Option<f64>,
        /// Diffuse reflectivity ρ_d ∈ [0, 1] for a PLACED panel. Defaults
        /// to 0.10 (matching the automatic pair). See `rho_s`.
        rho_d: Option<f64>,
        /// TOTAL panel mass [kg] (both wings, if this entry represents a
        /// pair — matches `area_m2`'s own "total" convention). `None`
        /// falls back to `area * hardware_catalog::PanelSpec::rigid().areal_density_kgm2`
        /// (area = `width_m*height_m` if both given, else `area_m2`) — a
        /// deliberately conservative (heavier) default absent a stated
        /// structural class.
        mass_kg: Option<f64>,
    },
    /// One arbitrary flat plate added to the spacecraft's SRP/`FlatPlate`
    /// (or `NPlate`) geometry, on top of the automatic 6 bus faces and any
    /// `SolarPanel` entries. Lets a mission describe shapes more complex
    /// than a box + symmetric panel pair — e.g. a dish antenna, an
    /// asymmetric deployable, an instrument boom cover — by adding one or
    /// more `[[spacecraft.hardware]]` entries of this type. Maps directly
    /// onto `orbital_models::acceleration::srp::Plate` (Phase 13a); see
    /// `docs/MP/MANUAL.md` §4.2 for the governing force/torque law.
    CustomPlate {
        /// Outward unit normal in the body frame [x, y, z] (dimensionless;
        /// normalized on load — need not be pre-normalized in the TOML).
        normal: [f64; 3],
        /// Plate area [m²].
        area_m2: f64,
        /// Plate geometric centre relative to spacecraft CoM, body frame
        /// [m]. Required for SRP torque: τ_i = center_i × F_i.
        center_offset_m: [f64; 3],
        /// Specular reflectivity ρ_s ∈ [0, 1]. Defaults to a representative
        /// MLI-blanket bus value (0.30, matching `build_plates`'s bus
        /// faces) when omitted.
        rho_s: Option<f64>,
        /// Diffuse reflectivity ρ_d ∈ [0, 1]. Defaults to 0.20 (matching
        /// `build_plates`'s bus faces) when omitted.
        rho_d: Option<f64>,
        /// Whether the plate is illuminated from either side (e.g. a
        /// deployable dish or panel-like plate) or only its outward face
        /// (e.g. a bus-mounted instrument cover). Defaults to `false`
        /// (single-sided) when omitted.
        double_sided: Option<bool>,
        /// Plate mass [kg]. `None` falls back to
        /// `area_m2 * GENERIC_PLATE_AREAL_DENSITY_KGM2` — a rough,
        /// explicitly-caveated placeholder (see that constant's own doc
        /// comment), since a `CustomPlate` represents an arbitrary
        /// user-defined shape with no natural structural class the way
        /// `SolarPanel`/`RcsThruster`/the sensor variants do.
        mass_kg: Option<f64>,
    },
}

/// Solar-panel articulation mechanism (`docs/MP/MANUAL.md`
/// §9.5) — the three real classes of solar-array drive mechanism, each with
/// a different pointing-constraint relaxation for the not-yet-built Phase 03
/// attitude commander. Internally tagged the same way as
/// `HardwareItem`, since this is itself a small discriminated union nested
/// inside `HardwareItem::SolarPanel`.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "kind")]
pub enum PanelArticulation {
    /// Single rotation axis, body frame (unit; normalized on load). Sweeps
    /// the panel normal through a circle about `axis`; the Sun can be
    /// tracked exactly only when it lies in the plane perpendicular to
    /// `axis` (§9.5's "Sun in the gimbal plane" condition) — otherwise
    /// there is a real, unavoidable cosine loss.
    OneAxis { axis: [f64; 3] },
    /// Two independent rotation axes, body frame (unit; normalized on
    /// load) — a real Solar Array Drive Assembly. In the idealized
    /// unconstrained-range case this fully decouples panel pointing from
    /// spacecraft attitude: the Sun can be tracked exactly regardless of
    /// body attitude, which REMOVES the panel's "normal → Sun" rule from
    /// the attitude commander's body-pointing constraint list entirely
    /// (see §9.5).
    TwoAxis { axis1: [f64; 3], axis2: [f64; 3] },
}

// ── Trajectory ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct TrajectoryConfig {
    pub phases: Vec<MissionPhase>,
    pub solver: TrajectorySolver,
    /// ISO 8601 departure epoch string, e.g. "2026-09-15T00:00:00 UTC"
    pub departure_epoch: Option<String>,
    /// Required, not defaulted to "Earth" -- departure economics (whether
    /// the escape burn is launch-vehicle-provided or self-funded) and the
    /// real parking-orbit departure leg both depend on which body this is,
    /// so an implicit default risks silently scoring the wrong physics.
    pub departure_body: String,
    pub departure: Option<DepartureConfig>,
    pub cruise: Option<CruiseConfig>,
    pub capture: Option<CaptureConfig>,
    pub landing: Option<LandingConfig>,
    /// Optional per-phase orbit radius overrides [m]. Keys are phase names
    /// ("CloseOrbit", "Flyover", etc.); any phase not listed falls back to
    /// `capture.target_orbit_radius_m`. Allows multi-altitude proximity missions
    /// without changing the capture orbit radius.
    ///
    /// Example TOML:
    /// ```toml
    /// [trajectory.per_phase_radii]
    /// CloseOrbit  = 900.0
    /// Flyover     = 500.0
    /// ScienceHold = 900.0
    /// ```
    #[serde(default)]
    pub per_phase_radii: HashMap<String, f64>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum MissionPhase {
    Cruise,
    Capture,
    Survey,
    CloseOrbit,
    Flyover,
    ScienceHold,
    RadioScience,
    Proximity,
    Descent,
    Landing,
    Departure,
}

impl std::fmt::Display for MissionPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cruise => write!(f, "Cruise"),
            Self::Capture => write!(f, "Capture"),
            Self::Survey => write!(f, "Survey"),
            Self::CloseOrbit => write!(f, "Close Orbit"),
            Self::Flyover => write!(f, "Flyover"),
            Self::ScienceHold => write!(f, "Science Hold"),
            Self::RadioScience => write!(f, "Radio Science"),
            Self::Proximity => write!(f, "Proximity"),
            Self::Descent => write!(f, "Descent"),
            Self::Landing => write!(f, "Landing"),
            Self::Departure => write!(f, "Departure"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub enum TrajectorySolver {
    Hohmann,
    Lambert,
    DiffCorrection,
    LambertThenDiffCorrect,
    GridSearch,
    MonteCarlo,
    SA,
    GA,
    PSO,
    ManifoldStitch,
    WSB,
}

impl std::fmt::Display for TrajectorySolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hohmann => write!(f, "Hohmann"),
            Self::Lambert => write!(f, "Lambert"),
            Self::DiffCorrection => write!(f, "Differential Correction"),
            Self::LambertThenDiffCorrect => write!(f, "Lambert + Differential Correction"),
            Self::GridSearch => write!(f, "Grid Search"),
            Self::MonteCarlo => write!(f, "Monte Carlo"),
            Self::SA => write!(f, "Simulated Annealing"),
            Self::GA => write!(f, "Genetic Algorithm"),
            Self::PSO => write!(f, "Particle Swarm Optimization"),
            Self::ManifoldStitch => write!(f, "Manifold Stitching"),
            Self::WSB => write!(f, "Weak Stability Boundary"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CruiseConfig {
    /// Minimum time-of-flight to search [days]
    pub tof_days_min: Option<f64>,
    /// Maximum time-of-flight to search [days]
    pub tof_days_max: Option<f64>,
    /// Number of grid points per axis for the porkchop scan
    pub grid_resolution: Option<u32>,
    /// Width of the departure window to sweep [days].
    /// The scan covers [departure_epoch − window/2, departure_epoch + window/2].
    /// Defaults to 60 days if not specified.
    pub departure_window_days: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LandingConfig {
    /// Altitude above body surface [m] at which the simulation exits and
    /// reports a powered-descent handoff. Defaults to 50 m if not set.
    pub terminal_altitude_m: Option<f64>,
    /// Orbit radius [m] from which the deorbit burn is fired. Defaults to
    /// `capture.target_orbit_radius_m` if not set.
    pub deorbit_radius_m: Option<f64>,
}

/// Departure-side mirror of `CaptureConfig` (Phase 8e) — feeds the
/// escape-burn sizing in `design.rs::departure_escape_dv_ms()` for a
/// non-Earth departure body. Only meaningful when `departure_body` is set
/// to something other than Earth; an Earth departure uses a launch vehicle
/// instead (Phase 8c), which has no parking orbit of its own to configure.
#[derive(Debug, Clone, Deserialize)]
pub struct DepartureConfig {
    /// Parking orbit radius at the departure body [m] the escape burn is
    /// sized from. Omit to fall back to the same atmospheric/airless
    /// heuristic `hohmann_api`/`run_hohmann` already use on the arrival
    /// side (`body_radius + 200 km` if atmospheric, `1.5 × body_radius` if
    /// airless) — see `design.rs::departure_escape_dv_ms()`. In `Launch`
    /// mode the default is `body_radius + parking_altitude_m` instead.
    pub parking_orbit_radius_m: Option<f64>,
    /// Phase 14a: how the mission leaves the departure body.
    /// Default `ParkingOrbit` — every existing config is unchanged.
    #[serde(default)]
    pub mode: DepartureMode,
    /// `Launch` mode: the launch site. Required in `Launch` mode
    /// (`check_config`); ignored otherwise. Only the latitude enters the
    /// physics (`trajectory_solver::launch_geometry`); longitude and name
    /// are for the frontend's schematic ascent and for daily launch-window
    /// work later.
    #[serde(default)]
    pub launch_site: Option<LaunchSiteConfig>,
    /// `Launch` mode: altitude of the parking orbit the launcher's ascent
    /// reaches before the upper-stage injection burn [m]. Default 185 km
    /// (the customary ~100 n.mi. injection altitude). Ignored when
    /// `parking_orbit_radius_m` is set.
    #[serde(default)]
    pub parking_altitude_m: Option<f64>,
    /// `Launch` mode: explicit parking/escape-plane inclination [deg] to
    /// the departure body's equator. Omit for the minimum feasible plane
    /// `max(|DLA|, |site latitude|)` — no plane change, no dogleg
    /// (`MANUAL.md` §13.6). A value below the site latitude is a
    /// dogleg and the result is flagged infeasible rather than priced.
    #[serde(default)]
    pub inclination_deg: Option<f64>,
}

/// `[trajectory.departure].mode` — Phase 14a. Mirrors the arrival choice
/// (flyby / orbit / land) on the departure side.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default)]
pub enum DepartureMode {
    /// The spacecraft starts in a given orbit around the departure body and
    /// the departure burn is its own (or a launcher lifted it to that orbit
    /// and the injection is still onboard). The single-leg search keeps its
    /// free burn genes (`theta_burn`, `dv`, `phi`).
    #[default]
    ParkingOrbit,
    /// The mission starts from a launch site; the launch vehicle's upper
    /// stage delivers the injection, so the departure ΔV is the LAUNCHER
    /// pool (capture + DSMs stay onboard). No ascent is propagated: the
    /// interplanetary solve returns the departure asymptote `(C3, RLA, DLA)`
    /// and the launch is closed-form geometry from it
    /// (`trajectory_solver::launch_geometry`, `MANUAL.md` §13.6). The
    /// single-leg search's departure genes become `(offset, v∞, RLA, DLA)`
    /// in the same four chromosome slots. Requires `[spacecraft].
    /// launch_vehicle`, a `launch_site`, and a departure body with a
    /// catalog pole.
    Launch,
}

/// `[trajectory.departure.launch_site]` — Phase 14a.
#[derive(Debug, Clone, Deserialize)]
pub struct LaunchSiteConfig {
    /// Display name, e.g. "Cape Canaveral".
    #[serde(default)]
    pub name: String,
    /// Geodetic latitude [deg], positive north. The only site quantity that
    /// enters the physics (minimum reachable inclination, launch azimuth).
    pub lat_deg: f64,
    /// Longitude [deg], positive east. Display / future daily-window use only.
    #[serde(default)]
    pub lon_deg: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CaptureConfig {
    /// Target orbit radius at arrival body [m]
    pub target_orbit_radius_m: Option<f64>,
    /// Hyperbolic excess speed at arrival [m/s]. When set, the simulate stage
    /// starts from a hyperbolic approach at 20× target radius and fires a
    /// single LOI burn at periapsis (`OrbitInsertionPhase`). Omit or set to
    /// 0 to start directly in the circular target orbit (default).
    pub approach_v_inf_mps: Option<f64>,
    /// Place the initial orbit in the terminator plane: orbital pole aligned
    /// with the body→Sun direction so the spacecraft is always at ~90° phase
    /// angle (flying the day/night boundary). Used by OSIRIS-REx at Bennu.
    /// Default: false (equatorial, along x-axis).
    #[serde(default)]
    pub terminator_orbit: bool,
    /// Capture orbit eccentricity at arrival (Phase 9v-ii). Default 0.0 =
    /// circular capture, the historical behaviour. When e > 0,
    /// `target_orbit_radius_m` is interpreted as the capture orbit's
    /// PERIAPSIS radius and the LOI burn is applied at periapsis:
    /// ΔV = √(v∞² + 2μ/r_p) − √(μ·(1+e)/r_p) (vis-viva at periapsis of an
    /// ellipse with a = r_p/(1−e)). Real missions capture eccentric first —
    /// the GTOP Cassini-2 benchmark uses r_p = 108,950 km, e = 0.98
    /// (Schlueter et al. 2017), ~600 m/s vs ~8–9 km/s circular.
    #[serde(default)]
    pub capture_eccentricity: f64,
}

// ── GNC ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct GncConfig {
    pub navigation_filter: NavigationFilter,
    pub pointing_mode: PointingMode,
    pub attitude_controller: AttitudeController,
    /// Required 1-σ position knowledge accuracy [m]. Drives sensor-grade
    /// selection in the GNC design stage. Defaults to 100 m if unspecified.
    pub position_accuracy_req_m: Option<f64>,
    /// Required 1-σ velocity knowledge accuracy [m/s]. Defaults to 0.01 m/s
    /// if unspecified.
    pub velocity_accuracy_req_mps: Option<f64>,
    /// Reaction-wheel PD proportional gain [N*m/rad]. Defaults to 0.05 —
    /// the value from `GNC/AutonomousNavigation`'s `wheel_pd_gains()`.
    /// Note: `ATTITUDE_KP=1.5` is the RCS bang-bang gain and must NOT be
    /// used here — it over-drives the wheels and saturates them in hours.
    pub reaction_wheel_kp: Option<f64>,
    /// Reaction-wheel PD derivative gain [N*m*s/rad]. Defaults to 0.6 —
    /// from `GNC/AutonomousNavigation`'s `wheel_pd_gains()`.
    pub reaction_wheel_kd: Option<f64>,
    /// Pointing-error dead-band [rad]. Defaults to 0.0 — dead-bands suppress
    /// bang-bang chatter for RCS thrusters but reaction wheels run
    /// continuously and do not need one.
    pub pointing_deadband_rad: Option<f64>,
    /// Angular-rate dead-band [rad/s]. Defaults to 0.0 for the same reason
    /// as `pointing_deadband_rad` (wheels are continuous, not bang-bang).
    pub rate_deadband_radps: Option<f64>,
    /// Three-layer attitude-control architecture (
    /// `docs/MP/MANUAL.md` §10.5): per-actuator control law + per-
    /// activity tuning + gain scheduling. Every field optional — omitted
    /// values are DERIVED from the vehicle's real inertia and each actuator
    /// class's real torque authority; the values actually used are reported
    /// back in `CruiseResult.attitude_control_effective` so a run is
    /// reproducible from what was used (frontend parity rule).
    /// `reaction_wheel_kp`/`reaction_wheel_kd` above remain the legacy
    /// explicit wheel-PD override, honored when `attitude_control.wheels`
    /// is absent.
    #[serde(default)]
    pub attitude_control: AttitudeControlConfig,
}

/// `[gnc.attitude_control]` — see `GncConfig::attitude_control`. Suggested
/// shape from the decision:
/// `gnc.attitude_control = { wheels: {law, params...}, thrusters: {law,
/// params...}, activities: {hold, slew, burn_hold, coast}, gain_scheduling:
/// {...} }`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AttitudeControlConfig {
    /// Layer 1 — law used whenever the wheels are the primary actuator
    /// (`ControlMode::WheelsPrimary`). Default: `Pd` derived from the wheel
    /// cluster's `max_torque` (the E1 derivation, MANUAL.md §10.1).
    pub wheels: Option<AttitudeLawConfig>,
    /// Layer 1 — law used whenever thrusters are primary
    /// (`ThrustersPrimary`/`ThrustersOnly`, i.e. every main-engine burn).
    /// Default: `Pid` derived from the placed RCS layout's worst-axis torque
    /// authority — integral action is what removes the steady-state offset
    /// a persistent engine-misalignment torque leaves under pure PD
    /// (§10.5.2); `PhasePlane` is the thruster-native alternative (§10.5.3).
    pub thrusters: Option<AttitudeLawConfig>,
    /// Layer 2 — per-activity tuning overrides (§10.5.4).
    #[serde(default)]
    pub activities: ActivitiesConfig,
    /// Layer 3 — gain scheduling (§10.5.5).
    #[serde(default)]
    pub gain_scheduling: GainSchedulingConfig,
}

/// One layer-1 law selection. Serde-tagged on `law`; every parameter is
/// optional and falls back to the derived default for that actuator class.
/// Angles in degrees here (user-facing), converted to radians internally.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "law")]
pub enum AttitudeLawConfig {
    /// Quaternion PD (§10.1): `kp` [N·m/rad], `kd` [N·m·s/rad].
    Pd { kp: Option<f64>, kd: Option<f64> },
    /// Quaternion PID (§10.5.2): PD plus `ki` [N·m/(rad·s)] on the
    /// small-angle error integral, anti-windup clamp `integral_limit`
    /// [rad·s] (default: `τ_authority / ki`, so the integral term alone can
    /// never exceed the actuator's authority).
    Pid { kp: Option<f64>, ki: Option<f64>, kd: Option<f64>, integral_limit: Option<f64> },
    /// Phase-plane bang-off-bang with Schmitt-trigger deadband (§10.5.3).
    /// `deadband_deg` (default 0.5°), `rate_deadband_degs` (default
    /// δ/T), `hysteresis_deg` (default 0.2·δ), `min_on_time_s` (minimum
    /// impulse bit, default 0.02 s), `lead_time_s` (default `2·k_d/k_p` of
    /// the derived PD — the equivalent rate weighting).
    PhasePlane {
        deadband_deg: Option<f64>,
        rate_deadband_degs: Option<f64>,
        hysteresis_deg: Option<f64>,
        min_on_time_s: Option<f64>,
        lead_time_s: Option<f64>,
    },
}

/// `[gnc.attitude_control.activities]` — layer-2 overrides, all optional
/// (defaults: `sim_engine::Activity::default_tuning`).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ActivitiesConfig {
    pub hold: Option<ActivityTuningConfig>,
    pub slew: Option<ActivityTuningConfig>,
    pub burn_hold: Option<ActivityTuningConfig>,
    pub coast: Option<ActivityTuningConfig>,
}

/// One activity's tuning: `bandwidth_scale` multiplies the closed-loop
/// natural frequency (ζ invariant — §10.5.4), `deadband_scale` multiplies
/// the phase-plane/PD dead-bands.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ActivityTuningConfig {
    pub bandwidth_scale: Option<f64>,
    pub deadband_scale: Option<f64>,
}

/// `[gnc.attitude_control.gain_scheduling]` (§10.5.5). `enabled` (default
/// true): re-derive the active law whenever the control mode or activity
/// changes, or the vehicle mass has moved by more than
/// `mass_change_fraction` (default 0.05) since the last derivation.
/// `inertia_scales_with_mass` (default FALSE): model the scheduling
/// inertia as `I_ref·m/m_ref`. Off by default because the truth propagator
/// (`sim_engine::propagator6dof`) currently holds inertia CONSTANT through
/// propellant depletion — enabling this makes the controller assume an
/// inertia change the simulated vehicle does not actually undergo; turn it
/// on only once the truth model carries a propellant-inertia term.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GainSchedulingConfig {
    pub enabled: Option<bool>,
    pub mass_change_fraction: Option<f64>,
    pub inertia_scales_with_mass: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub enum NavigationFilter {
    EKF,
    UKF,
    IEKF,
}

impl std::fmt::Display for NavigationFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EKF => write!(f, "Extended Kalman Filter (EKF)"),
            Self::UKF => write!(f, "Unscented Kalman Filter (UKF)"),
            Self::IEKF => write!(f, "Iterated EKF (IEKF)"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub enum PointingMode {
    Nadir,
    SunPointing,
    VelocityAligned,
    Custom,
}

impl std::fmt::Display for PointingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nadir => write!(f, "Nadir"),
            Self::SunPointing => write!(f, "Sun Pointing"),
            Self::VelocityAligned => write!(f, "Velocity Aligned"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub enum AttitudeController {
    ReactionWheelPD,
    RcsBangBang,
}

impl std::fmt::Display for AttitudeController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReactionWheelPD => write!(f, "Reaction Wheel PD"),
            Self::RcsBangBang => write!(f, "RCS Bang-Bang"),
        }
    }
}

// ── Simulation ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct SimulationConfig {
    pub integrator: Integrator,
    /// Relative tolerance for adaptive integrators
    pub rtol: f64,
    /// Absolute tolerance for adaptive integrators
    pub atol: f64,
    /// Truth propagation step size [s]
    pub dt_truth_s: f64,
    /// GNC measurement/update interval [s]
    pub dt_meas_s: f64,
    /// Number of Monte Carlo runs (0 = deterministic only)
    pub monte_carlo_runs: u32,
    pub output_dir: String,
}

#[derive(Debug, Clone, Deserialize)]
pub enum Integrator {
    RK4,
    DormandPrince45,
    Dopri5,
    Radau,
}

impl std::fmt::Display for Integrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RK4 => write!(f, "RK4 (fixed-step)"),
            Self::DormandPrince45 => write!(f, "Dormand-Prince RK45"),
            Self::Dopri5 => write!(f, "Dopri5"),
            Self::Radau => write!(f, "Radau (5th order implicit)"),
        }
    }
}

// ── Optimization (Phase 9, Layer 1b) ──────────────────────────────────────────

/// Real propagated-dynamics trajectory optimization, consuming the narrowing
/// stage's (`[trajectory]`) output design space but with its own independent
/// copy of the epoch/window fields — the frontend sets these per-stage, not
/// derived from `[trajectory]`'s.
///
/// The departure burn is *not* anchored to a Lambert-solved v-infinity (that
/// would make the GA/PSO search trivially converge to the analytic
/// Hohmann-like minimum — not a meaningful test of global search, and not
/// how a real free departure-burn search works). Instead it's described by
/// four independent, freely-searchable numbers: `departure_offset_days`
/// (which day — sets the Earth/target synodic alignment), and three burn
/// parameters at a circular parking orbit (`r_park`, resolved the same way
/// as the narrowing stage's `departure_escape_dv_ms` /
/// `resolve_parking_orbit_radius_m`): burn location (true anomaly), burn
/// magnitude (`dv_min_ms..dv_max_ms`), and an out-of-plane angle (handles
/// the real plane change to the target's orbit — full range, no separate
/// config bound needed). Reaching the target at all is therefore a genuine
/// search outcome, not a foregone conclusion baked into the construction.
#[derive(Debug, Clone, Deserialize)]
pub struct OptimizationConfig {
    pub objective: ObjectiveFunction,
    pub method: OptimizationMethod,
    /// Catalog name of the departure body (overall start; also the start of
    /// the leg chain when `method = "MGA"`).
    pub departure_body: String,
    /// Catalog name of the final target body (overall end; also the end of
    /// the leg chain when `method = "MGA"` — intermediate flyby bodies are
    /// configured separately under `[optimization.mga]`).
    pub target_body: String,
    /// ISO 8601 departure epoch string, e.g. "2026-09-15T00:00:00 UTC"
    pub departure_epoch: Option<String>,
    pub departure_window_days: Option<f64>,
    /// Minimum departure burn magnitude to search [m/s].
    pub dv_min_ms: f64,
    /// Maximum departure burn magnitude to search [m/s].
    pub dv_max_ms: f64,
    /// Burn-location angle theta search range [deg] — every search
    /// variable's range is configurable. Defaults to the full circle [0, 360)
    /// when unset, exactly the previous hardcoded behavior.
    #[serde(default)]
    pub theta_min_deg: Option<f64>,
    #[serde(default)]
    pub theta_max_deg: Option<f64>,
    /// Out-of-plane angle phi search range [deg] — defaults to the
    /// previous hardcoded [-45, +45] when unset (see bounds_from's own doc
    /// comment for why that default range is already generous).
    #[serde(default)]
    pub phi_min_deg: Option<f64>,
    #[serde(default)]
    pub phi_max_deg: Option<f64>,
    /// How long to coast and watch for a close approach to the target body
    /// before giving up [days] -- a ceiling, not a literal prescribed
    /// transfer duration (replaces the old `tof_days_min`/`tof_days_max`,
    /// which assumed a Lambert-prescribed TOF; the achieved time-to-target
    /// is now an output of the search, not an input).
    pub max_coast_days: f64,
    pub force_model: ForceModelConfig,
    /// Required when `method = "GA"`.
    pub ga: Option<GaParams>,
    /// Required when `method = "PSO"`.
    pub pso: Option<PsoParams>,
    /// Required when `method = "MultipleShooting"`.
    pub shooting: Option<ShootingParams>,
    /// Required when `method = "MGA"`.
    pub mga: Option<MgaParams>,
}

/// Optimization objective — distinct from `MissionObjective` (Flyby/Orbit/
/// Landing/etc., the *mission* objective). This is the scalar the Phase 9
/// optimizer is minimizing/maximizing under real propagated dynamics.
///
/// `MinC3` and `MaxMassMargin` (the original 4th/5th options) were dropped:
/// for a fixed spacecraft, `c3_km2s2` is a deterministic monotonic function
/// of departure ΔV alone (`c3 = (dv_departure/1000)^2` —
/// `crates/trajectory_solver/src/lambert_arc.rs`), and mass margin is a
/// deterministic monotonic (Tsiolkovsky) function of the *same* total ΔV
/// `MinDeltaV` already uses — both always rank candidates identically to
/// `MinDeltaV`, carrying zero independent optimization signal. They're
/// still reportable as alternate display units on a result, just not
/// separate selectable optimizer behaviors.
#[derive(Debug, Deserialize, Clone, Copy)]
pub enum ObjectiveFunction {
    MinDeltaV,
    MinTof,
    /// Minimize the *difference* between the real closest-approach distance
    /// to the target body and the mission's configured target distance
    /// (`[trajectory.capture].target_orbit_radius_m` — required when this
    /// objective is selected). The right objective for `Flyby`/`Orbit`,
    /// where the goal isn't to hit the target exactly but to pass it (or
    /// capture into orbit around it) at a *specific* distance.
    MatchTargetDistance,
}

impl std::fmt::Display for ObjectiveFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MinDeltaV => write!(f, "Minimize ΔV"),
            Self::MinTof => write!(f, "Minimize Time of Flight"),
            Self::MatchTargetDistance => write!(f, "Match Target Distance"),
        }
    }
}

/// Phase 9 optimization method. Deliberately a separate enum from
/// `TrajectorySolver` — `TrajectorySolver::GA`/`PSO` evaluate a closed-form
/// Lambert proxy (narrowing stage, Phase 7/8); these variants evaluate real
/// propagated dynamics inside the fitness/targeting loop, a materially
/// different cost profile despite the shared algorithm name.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
pub enum OptimizationMethod {
    GA,
    PSO,
    MultipleShooting,
    MGA,
}

impl std::fmt::Display for OptimizationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GA => write!(f, "Genetic Algorithm (real dynamics)"),
            Self::PSO => write!(f, "Particle Swarm Optimization (real dynamics)"),
            Self::MultipleShooting => write!(f, "Multiple Shooting"),
            Self::MGA => write!(f, "Multi-Gravity-Assist"),
        }
    }
}

/// Force-model configuration for the optimization stage's fitness evaluator.
/// `bodies` is the TOML-facing equivalent of `design.rs::propagator_body_entries()`'s
/// output — user-specified instead of hardcoded to "target body + configured
/// third_bodies" the way the narrowing-stage visualization arcs are.
#[derive(Debug, Clone, Deserialize)]
pub struct ForceModelConfig {
    #[serde(default)]
    pub srp: bool,
    pub integrator: Integrator,
    /// Relative tolerance for the adaptive propagator
    pub rtol: f64,
    /// Absolute tolerance for the adaptive propagator
    pub atol: f64,
    /// Additional third-body perturbers / SOI candidates beyond the
    /// departure and target bodies, which are always auto-registered
    /// regardless of this list (`optimize.rs::force_model_body_entries`).
    /// May be omitted or empty.
    #[serde(default)]
    pub bodies: Vec<OptimizationBody>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OptimizationBody {
    /// Catalog name, resolved via `body_models::TargetBody::by_name()`.
    pub name: String,
    pub role: BodyRole,
    /// Central-body gravity fidelity. Only meaningful when `role =
    /// "CentralWhenInSoi"` — ignored (and must be omitted) for
    /// `AlwaysThirdBody` perturbers, which are point-mass only by design
    /// (see the design notes...
    /// Third-body perturbers stay point-mass only").
    pub fidelity: Option<GravityModel>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum BodyRole {
    CentralWhenInSoi,
    AlwaysThirdBody,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GaParams {
    pub population_size: u32,
    pub generations: u32,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub elitism_count: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PsoParams {
    pub swarm_size: u32,
    pub iterations: u32,
    pub inertia_weight: f64,
    pub cognitive_weight: f64,
    pub social_weight: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ShootingParams {
    pub max_iterations: u32,
    pub tolerance: f64,
}

/// Multi-gravity-assist leg chain. The full sequence is
/// `[OptimizationConfig::departure_body] + flyby_bodies + [OptimizationConfig::target_body]`,
/// so `leg_tof_days.len()` must equal `flyby_bodies.len() + 1` — checked in
/// `check_config()`.
#[derive(Debug, Clone, Deserialize)]
pub struct MgaParams {
    /// Intermediate flyby bodies only, in visit order (excludes departure_body
    /// and target_body, which are the chain's overall endpoints). Optional —
    /// ignored when `sequence_search` is set (see below); `#[serde(default)]`
    /// so a request that omits it entirely (the natural shape for a pure
    /// auto-sequence-search request) doesn't 400 with an opaque JSON parse
    /// error (Phase 9k-v bug #2).
    #[serde(default)]
    pub flyby_bodies: Vec<String>,
    /// One `[min, max]` TOF range [days] per leg.
    pub leg_tof_days: Vec<[f64; 2]>,
    /// Maximum allowed departure v∞ from the departure body [m/s].
    #[serde(default = "default_departure_vinf_max")]
    pub departure_vinf_max_ms: f64,
    /// Minimum allowed departure v∞ from the departure body [m/s]
    /// (Phase 9v-iii). Default 0.0 = unconstrained, the historical
    /// behaviour. Without a floor the DE can converge to a degenerate
    /// near-zero departure burn with the DSMs and LOI paying for
    /// everything — the GTOP Cassini-2 benchmark bounds departure v∞ to
    /// [3, 5] km/s precisely to exclude that family.
    #[serde(default)]
    pub departure_vinf_min_ms: f64,
    /// Minimum periapsis radius [m] for any intermediate gravity-assist flyby.
    /// Must be > flyby body surface radius. Default: 6 371 000 m (Earth radius).
    #[serde(default = "default_flyby_min_periapsis")]
    pub flyby_min_periapsis_m: f64,
    /// Maximum acceptable arrival v∞ at the final body [m/s].
    /// Candidates exceeding this are penalised in the real-objective phase.
    #[serde(default = "default_arrival_vinf_max")]
    pub arrival_vinf_max_ms: f64,
    /// Per-leg `[min, max]` periapsis bound override (units of that flyby
    /// body's radius), one entry per intermediate flyby (`flyby_bodies.len()`
    /// entries, i.e. one fewer than `leg_tof_days`). Default: `None`, which
    /// keeps the historical uniform `(1.05, 300.0)` bound for every flyby
    /// regardless of body. Added to let a specific benchmark
    /// config (e.g. the real GTOP Cassini-2 problem, whose official bounds
    /// per Izzo & Vinkó 2008 Table IV differ sharply per body: Venus
    /// [1.05, 6], Earth [1.15, 6.5], Jupiter [1.7, 291]) reproduce the exact
    /// official search box instead of our own generic, much wider default —
    /// found to matter after the uniform default let the search exploit a
    /// near-zero-turn "flyby" at rp_norm ~ 193 (vs. the official max of 6)
    /// on the leg that produced the (invalid) 5,509 m/s Cassini-2 result.
    /// Scoped intentionally per-config, not a global default change.
    #[serde(default)]
    pub rp_norm_bounds: Option<Vec<[f64; 2]>>,
    /// Upper bound override for every leg's DSM-timing fraction η. Default:
    /// `None`, which keeps the historical `0.99` upper bound. The real GTOP
    /// Cassini-2 problem uses `0.9` (Table IV) — set this to `0.9` to match
    /// exactly when reproducing that benchmark. Added alongside
    /// `rp_norm_bounds` for the same reason.
    #[serde(default)]
    pub eta_max: Option<f64>,
    /// DE mutation scale factor F ∈ [0.4, 1.0] (Storn & Price 1997).
    /// Recommended: 0.8 for MGA-DSM problems.
    #[serde(default = "default_de_f_weight")]
    pub de_f_weight: f64,
    /// DE crossover probability CR ∈ [0.8, 0.95].
    /// High values work well because MGA-DSM leg parameters are tightly coupled.
    #[serde(default = "default_de_cr")]
    pub de_cr: f64,
    /// DE population size. Recommended ≥ 10 × chromosome dimension.
    #[serde(default = "default_de_population_size")]
    pub de_population_size: usize,
    /// DE generations per restart (Phase 1 and Phase 2 each get this budget).
    #[serde(default = "default_de_generations")]
    pub de_generations: usize,
    /// Number of independent DE restarts (multi-start for basin coverage).
    #[serde(default = "default_de_restarts")]
    pub de_restarts: usize,
    /// Base RNG seed for the DE search (Phase 9v-vii). Restart r uses
    /// `de_seed + r·137`; Phase 2 uses `de_seed + de_restarts·137`.
    /// Default 42 reproduces the historical hardcoded seeding exactly.
    /// Vary this across runs to measure seed-to-seed spread — a single
    /// run of a stochastic optimizer is not evidence of anything.
    #[serde(default = "default_de_seed")]
    pub de_seed: u64,
    /// Use SHADE self-adaptive DE (success-history F/CR adaptation +
    /// current-to-pbest/1 with archive; Tanabe & Fukunaga 2013) instead of
    /// classic fixed-parameter DE/rand/1/bin. When true, `de_f_weight`/
    /// `de_cr` seed the success-history memory rather than being fixed.
    /// Default true; set false to A/B against the classic solver.
    #[serde(default = "default_de_adaptive")]
    pub de_adaptive: bool,
    /// When set, automatically discovers intermediate flyby bodies via a
    /// Tisserand beam search rather than requiring the user to specify
    /// `flyby_bodies` explicitly. See [`SequenceSearchConfig`].
    pub sequence_search: Option<SequenceSearchConfig>,
    /// Minimum heliocentric perihelion [m] permitted for EITHER sub-arc of
    /// any leg (the departure Kepler sub-arc or the Lambert sub-arc) —
    /// rejects chromosomes whose orbit dives unrealistically close to the
    /// Sun. Unlike `flyby_min_periapsis_m` (which bounds closest approach to
    /// a *flyby body*), this bounds closest approach to the Sun itself,
    /// which nothing previously constrained: `rp_dep_m`/`rp_lambert_m` were
    /// already computed by `evaluate_mga_leg` but never checked against any
    /// floor, so the DE was free to "discover" physically implausible
    /// near-Sun-grazing solutions with no engineering basis (found
    /// via visibly degenerate tight loops near the Sun in a plotted result).
    /// Default: 0.2 AU.
    #[serde(default = "default_min_solar_perihelion")]
    pub min_solar_perihelion_m: f64,
    /// Phase 9w ballistic (no-DSM) powered-flyby grid scan — the middle tier
    /// between Tisserand sequence pruning and this DE-based MGA-1DSM search.
    /// Optional: when present, the `mga-scan` CLI subcommand can run a
    /// deterministic launch-window scan for this mission's flyby sequence
    /// before (or instead of) running the DE search.
    pub scan: Option<MgaScanConfigToml>,
    /// Phase 9x incremental pruning (Ceriotti 2010, Ch. 3) — an optional
    /// leg-by-leg pre-search that seeds Phase 1's DE restarts instead of (or
    /// alongside) the existing seeded/random restarts. Optional: when
    /// absent, `run_mga` behaves exactly as before (no pruning).
    pub pruning: Option<PruningConfigToml>,
    /// Phase 9x-v Stage 3: which global search algorithm runs Phase 2 (the
    /// real-objective refinement). `Mbh` (Monotonic Basin Hopping, default
    /// as of Stage 4) beat matched-budget DE by 40-57% on both
    /// head-to-head validation missions (VEEGA 1989, Cassini-2 GTOP) — see
    /// `docs/MP/BENCHMARKS.md` "Search method comparison" for the numbers
    /// and `trajectory_solver::mbh` for the motivation (a population's own
    /// crossover/selection dynamics can discard a genuinely good seeded
    /// candidate; MBH's single-chain-per-seed design cannot). `De` remains
    /// available as an explicit opt-in for A/B comparison.
    #[serde(default)]
    pub search_method: SearchMethod,
    /// MBH configuration, used only when `search_method = "Mbh"`.
    /// `#[serde(default)]` so an `Mbh` config can omit this and take the
    /// documented defaults.
    #[serde(default)]
    pub mbh: MbhConfigToml,
    /// Phase 9w-vi: when true, `run_mga` first runs the Phase 9w ballistic
    /// grid scan (`[optimization.mga.scan]` MUST also be set) for the
    /// resolved flyby sequence, then auto-derives the DE search's
    /// `departure_epoch`/`departure_window_days` and per-leg `leg_tof_days`
    /// bounds from the scan's best feasible branch(es), instead of trusting
    /// the config's own hand-written values. This exists because
    /// `leg_tof_days` is indexed by ARRAY POSITION, not leg role — a real
    /// bug class (see the design notes) where hand-written per-leg bounds
    /// have been wrong twice already, once making a whole scan's direct
    /// baseline infeasible. The scan is cheap, deterministic, and
    /// propagation-free, so it's a strictly safer source of bounds than a
    /// human's guess whenever it finds a feasible branch at all. Default
    /// false = today's exact behaviour (trust `departure_epoch`/
    /// `departure_window_days`/`leg_tof_days` as written).
    ///
    /// **Caveat, only useful on ballistic-ish missions**: the Phase 9w scan
    /// deliberately has NO DSM model (mixing DSMs in would blur the sharp
    /// departure-window structure the scan exists to reveal — see
    /// `crates/trajectory_solver/src/mga_scan.rs`'s module doc). On a
    /// mission whose real optimum leans on a large DSM, the scan can find
    /// ZERO feasible branches for a window the real DE search (which does
    /// model DSMs) would happily solve — confirmed on a `venus_saturn_auto`-
    /// class case where the DE's optimum used a ~5.3 km/s DSM, but the true
    /// ballistic-only minimum departure v∞ for that window was ~13.3 km/s,
    /// far above the ~8 km/s cap the DSM-equipped search stayed under. This
    /// function errors loudly rather than silently falling back to the raw
    /// config bounds in that case — do not "fix" that by adding a fallback;
    /// a scan-derived bound is only trustworthy when the scan actually found
    /// something, and a fallback would hide that it didn't. Leave this flag
    /// off (or use generously wide hand-written bounds instead) for
    /// DSM-heavy legs.
    #[serde(default)]
    pub scan_informed_window: bool,
}

/// Which global search algorithm runs MGA-1DSM Phase 2 (Phase 9x-v Stage 3).
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMethod {
    De,
    #[default]
    Mbh,
}

/// Configuration for the Phase 9x-v Monotonic Basin Hopping search method.
/// See `trajectory_solver::mbh::MbhSolver` for the algorithm itself.
#[derive(Debug, Deserialize, Clone)]
pub struct MbhConfigToml {
    /// Number of hops per chain (after each chain's initial local descent).
    #[serde(default = "default_mbh_hops")]
    pub hops: usize,
    /// Fraction of chromosome variables perturbed on each kick.
    #[serde(default = "default_mbh_perturb_fraction")]
    pub perturb_fraction: f64,
    /// Kick magnitude as a fraction of each variable's bounds width.
    #[serde(default = "default_mbh_kick_scale")]
    pub kick_scale: f64,
    /// Max local-descent iterations per hop (both the initial descent and
    /// every post-kick refinement), for whichever `local_optimizer` is
    /// selected.
    #[serde(default = "default_mbh_local_max_iter")]
    pub local_max_iter: usize,
    /// RNG seed for the kick sequence.
    #[serde(default = "default_de_seed")]
    pub seed: u64,
    /// Early-stop-on-stagnation: a chain terminates after this many
    /// consecutive non-improving hops instead of always burning its full
    /// `hops` budget. `None` (default) disables early stop. Added
    /// per the real pagmo2 MBH default (`stop = 5`,
    /// `src/algorithms/mbh.cpp`) — see the design notes
    /// Priority 1. Setting this lets a fixed compute budget spend itself on
    /// many short, cheaply-abandoned chains instead of one long chain
    /// forced to keep perturbing an already-exhausted basin.
    #[serde(default)]
    pub stop_after: Option<usize>,
    /// Which local optimizer runs MBH's inner descent step. `NelderMead`
    /// (default, unchanged prior behaviour), `CompassSearch` (a
    /// best-of-all-directions pattern search, added), or
    /// `HookeJeeves` (a greedy accept-first-improving pattern search with an
    /// evaluation-budget termination, added — see
    /// `trajectory_solver::hooke_jeeves` for why this exists as a distinct
    /// variant: real MBH reference implementations default to a very
    /// shallow per-hop descent of this kind, not a full local solve).
    #[serde(default)]
    pub local_optimizer: MbhLocalOptimizerToml,
    /// `HookeJeevesSearch` configuration, used only when
    /// `local_optimizer = "hooke_jeeves"`. Defaults reproduce a real,
    /// independently-verified MBH reference implementation's own default
    /// inner-descent behaviour exactly (see `trajectory_solver::hooke_jeeves`
    /// doc comment for how this was confirmed and why it's a fresh
    /// reimplementation, not a port).
    #[serde(default)]
    pub hooke_jeeves: HookeJeevesConfigToml,
    /// Extra purely-randomly-initialised chains added alongside the elite
    /// seeds (pruning survivors + bidirectional-backfit stitches). `0`
    /// (default) is byte-identical to prior behaviour. Added 
    /// when non-zero, the outer search is no longer 100% dependent on the
    /// pre-search having found the right basin — see
    /// `trajectory_solver::MbhSolver::extra_random_chains`.
    #[serde(default)]
    pub extra_random_chains: usize,
    /// Global-relative early stop: abandon a chain once past
    /// `global_stall.patience` hops if it's still worse than the best
    /// result ANY chain has found so far by more than `global_stall
    /// .margin_frac`. `None` (default) disables this — byte-identical to
    /// prior behaviour. Added after finding (via a real run)
    /// that `stop_after` alone lets a chain keep dodging its own
    /// consecutive-failure counter indefinitely while being globally
    /// irrelevant — see `trajectory_solver::MbhSolver::global_stall`.
    #[serde(default)]
    pub global_stall: Option<GlobalStallConfigToml>,
    /// Archipelago-style migration: every `migration.interval` of a chain's
    /// own hops, a chain whose incumbent is worse than the live global best
    /// by more than `migration.margin_frac` receives the global best as its
    /// new incumbent and keeps hopping from there with its own kick stream.
    /// Added per the measured waste that motivated `global_stall`
    /// (6 of 12 chains contributing nothing in a real overnight run): where
    /// `global_stall` kills a losing chain, migration repurposes it — see
    /// `trajectory_solver::MbhSolver::migration`. **Promoted to a STANDARD
    /// default (same day) after a same-seed A/B on Cassini-2
    /// showed it strictly helped two of three resonance-family branches
    /// (chain contribution 1/17→2/17 and 1/17→4/17 "within 1% of best",
    /// one branch −1,537.6 m/s) and never hurt the winning branch — see
    /// the design notes Phase 9x-v for the numbers.** `enabled = false` restores
    /// the pre-behaviour exactly.
    #[serde(default)]
    pub migration: MigrationConfigToml,
}

/// Configuration for [`trajectory_solver::MigrationConfig`]. See
/// `MbhConfigToml::migration`.
#[derive(Debug, Deserialize, Clone, Copy)]
pub struct MigrationConfigToml {
    /// `true` (default) = migration active with the parameters below.
    /// Set `false` to disable — reproduces pre-behaviour.
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_migration_interval")]
    pub interval: usize,
    #[serde(default = "default_migration_margin_frac")]
    pub margin_frac: f64,
}

fn default_true() -> bool { true }
fn default_migration_interval() -> usize { 10 }
fn default_migration_margin_frac() -> f64 { 0.3 }

impl Default for MigrationConfigToml {
    fn default() -> Self {
        MigrationConfigToml {
            enabled: default_true(),
            interval: default_migration_interval(),
            margin_frac: default_migration_margin_frac(),
        }
    }
}

/// Configuration for [`trajectory_solver::GlobalStallConfig`]. See
/// `MbhConfigToml::global_stall`.
#[derive(Debug, Deserialize, Clone, Copy)]
pub struct GlobalStallConfigToml {
    pub patience: usize,
    pub margin_frac: f64,
}

/// Which local optimizer runs [`trajectory_solver::MbhSolver`]'s inner
/// descent step. See `MbhConfigToml::local_optimizer`.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
pub enum MbhLocalOptimizerToml {
    #[default]
    NelderMead,
    CompassSearch,
    HookeJeeves,
}

/// Configuration for [`trajectory_solver::HookeJeevesSearch`]. See
/// `MbhConfigToml::hooke_jeeves`.
#[derive(Debug, Deserialize, Clone, Copy)]
pub struct HookeJeevesConfigToml {
    /// Fitness-evaluation budget per local descent (NOT an iteration
    /// count — see `HookeJeevesSearch`'s own doc comment). `1` is a real,
    /// independently-verified reference-implementation default, not a
    /// placeholder — it produces a near-single-probe descent, deliberately
    /// leaving nearly all exploration work to the outer MBH hop loop.
    #[serde(default = "default_hooke_jeeves_max_fevals")]
    pub max_fevals: usize,
    /// Initial step size, as a fraction of each parameter's bounds width.
    #[serde(default = "default_hooke_jeeves_start_range")]
    pub start_range: f64,
    /// Stop once the (normalised) step size drops below this.
    #[serde(default = "default_hooke_jeeves_stop_range")]
    pub stop_range: f64,
    /// Step shrink factor applied after a full sweep finds no improving
    /// move.
    #[serde(default = "default_hooke_jeeves_reduction_coeff")]
    pub reduction_coeff: f64,
}

impl Default for HookeJeevesConfigToml {
    fn default() -> Self {
        HookeJeevesConfigToml {
            max_fevals: default_hooke_jeeves_max_fevals(),
            start_range: default_hooke_jeeves_start_range(),
            stop_range: default_hooke_jeeves_stop_range(),
            reduction_coeff: default_hooke_jeeves_reduction_coeff(),
        }
    }
}

fn default_hooke_jeeves_max_fevals() -> usize { 1 }
fn default_hooke_jeeves_start_range() -> f64 { 0.1 }
fn default_hooke_jeeves_stop_range() -> f64 { 0.01 }
fn default_hooke_jeeves_reduction_coeff() -> f64 { 0.5 }

impl Default for MbhConfigToml {
    fn default() -> Self {
        MbhConfigToml {
            hops: default_mbh_hops(),
            perturb_fraction: default_mbh_perturb_fraction(),
            kick_scale: default_mbh_kick_scale(),
            local_max_iter: default_mbh_local_max_iter(),
            seed: default_de_seed(),
            stop_after: None,
            local_optimizer: MbhLocalOptimizerToml::default(),
            hooke_jeeves: HookeJeevesConfigToml::default(),
            extra_random_chains: 0,
            global_stall: None,
            migration: MigrationConfigToml::default(),
        }
    }
}

fn default_mbh_hops() -> usize { 200 }
fn default_mbh_perturb_fraction() -> f64 { 0.3 }
fn default_mbh_kick_scale() -> f64 { 0.15 }
fn default_mbh_local_max_iter() -> usize { 500 }

/// Configuration for the Phase 9x incremental pruning pre-search.
///
/// See `crates/trajectory_solver/src/incremental_pruning.rs` for the
/// generic engine and the design notes Phase 9x section for the motivation
/// (all-at-once DE over the full chromosome is, per Ceriotti Ch. 3, the
/// empirically weakest search strategy for multi-flyby MGA).
#[derive(Debug, Deserialize, Clone)]
pub struct PruningConfigToml {
    /// Number of random samples drawn at level 0 (departure + leg-0 vars).
    #[serde(default = "default_pruning_samples_level0")]
    pub samples_level0: usize,
    /// Number of random continuations sampled per surviving parent at every
    /// level after 0.
    #[serde(default = "default_pruning_children_per_survivor")]
    pub children_per_survivor: usize,
    /// Hard cap on survivors carried into the next level (bounds total
    /// evaluation count — keeps the search polynomial in level count).
    #[serde(default = "default_pruning_max_survivors")]
    pub max_survivors_per_level: usize,
    /// Pruning threshold as a multiple of the best partial cost found at
    /// that level. "Generous" is intentional (Ceriotti's own term) — this
    /// discards clearly-bad prefixes, not just the single best one.
    #[serde(default = "default_pruning_threshold_factor")]
    pub threshold_factor: f64,
    /// Number of best survivors (by partial cost) fed as elite seeds into
    /// the existing Phase 1 DE restarts.
    #[serde(default = "default_pruning_n_seeds")]
    pub n_seeds: usize,
    /// Bidirectional ("meet in the middle") extension (Phase 9x):
    /// alongside the forward leg-by-leg pruning, run a mirror-image backward
    /// pass from the target (suffix legs with a free handoff v∞ at their
    /// first body), match forward prefixes against backward suffixes at every
    /// intermediate flyby body (epoch agreement, |v∞| magnitude agreement —
    /// the quantity an unpowered flyby cannot fix — and turn-angle
    /// feasibility above the periapsis floor), and inject the best stitched
    /// full chromosomes as additional Phase 2 seeds. Generic promotion of the
    /// backward-fit decomposition that hand-solved Galileo's VEEGA
    /// (galileo_leg3_search / galileo_backfit_legs012). Default
    /// true — on problems without a rigid downstream constraint the backward
    /// pass merely confirms the forward pass at ~n× pruning cost (small vs.
    /// the DE); set false for A/B isolation runs.
    #[serde(default = "default_pruning_bidirectional")]
    pub bidirectional: bool,
    /// Admissible remaining-cost bound (branch-and-bound extension,
    ///): before the forward pruning pass, run a small backward
    /// suffix sampling per intermediate split and bin the cheapest suffix
    /// cost found over (interface epoch, |v∞|); the forward pass then
    /// prunes on `cost_so_far + bound` (A*'s f = g + h) instead of
    /// cost-so-far alone. The GASP mechanism (Myatt et al. 2004; Izzo et
    /// al. 2007, J. Global Opt. 38(2)) applied to this codebase's
    /// Ceriotti-style sampled pruning. `None` (default) = today's purely
    /// greedy behaviour, byte-identical. **Deliberately NOT promoted to a
    /// default, unlike `surrogate`/migration below/`MbhConfigToml
    /// ::migration`** — its own A/B was mixed (regressed one branch at the
    /// original budget; needed a 3.3× larger `samples_per_split` to reliably
    /// beat plain sampling) and a bound+surrogate STACK A/B regressed
    /// further as budget grew (+42% at 3.3× budget vs. either alone) — a
    /// real, unresolved interaction. Opt-in only until that's diagnosed —
    /// see the design notes.
    #[serde(default)]
    pub bound: Option<PruningBoundToml>,
    /// Surrogate-assisted child sampling: reuse the
    /// evaluations pruning already performs by fitting a per-level RBF
    /// model to (child vars → added cost) pairs and, once enough data
    /// exists, drawing `oversample_factor` candidate children per slot and
    /// keeping only the best-predicted one for real evaluation (Jones et
    /// al. 1998 EGO paradigm; Regis & Shoemaker 2007 for the RBF variant —
    /// see `trajectory_solver::RbfSurrogate`). **Promoted to a STANDARD
    /// default (same day it was built) after a same-seed A/B
    /// on Cassini-2 showed it beat plain sampling on every one of 3
    /// resonance-family branches, at both the original and a 3.3×-larger
    /// pruning budget (best overall winner both times: −25.2% then
    /// −5.0% further vs. plain sampling) — see the design notes
    /// the full numbers.** `enabled = false` restores pre-
    /// (plain-sampling) behaviour exactly. NOTE: enabling this changes the
    /// sampler's RNG-draw pattern, so results are not comparable draw-for-
    /// draw with a disabled run — compare outcome quality, not streams.
    #[serde(default)]
    pub surrogate: PruningSurrogateToml,
}

/// Configuration for the pruning remaining-cost bound. See
/// `PruningConfigToml::bound`.
#[derive(Debug, Deserialize, Clone, Copy)]
pub struct PruningBoundToml {
    /// Fraction of the sampled suffix-cost bound that is DISCOUNTED before
    /// use, hedging the gap between a sampled minimum (an overestimate of
    /// the true minimum) and an admissible bound: `h = bin_min × (1 −
    /// safety_frac)`. 0.0 trusts the sampled minimum outright; larger is
    /// safer but prunes less.
    #[serde(default = "default_bound_safety_frac")]
    pub safety_frac: f64,
    /// Bins over the interface-epoch axis of each split's bound table.
    #[serde(default = "default_bound_epoch_bins")]
    pub epoch_bins: usize,
    /// Bins over the interface-|v∞| axis of each split's bound table.
    #[serde(default = "default_bound_vinf_bins")]
    pub vinf_bins: usize,
    /// Backward samples drawn per split to build the table (full-length
    /// suffixes, cheapest-per-bin retained).
    #[serde(default = "default_bound_samples_per_split")]
    pub samples_per_split: usize,
}

fn default_bound_safety_frac() -> f64 { 0.5 }
fn default_bound_epoch_bins() -> usize { 24 }
fn default_bound_vinf_bins() -> usize { 12 }
fn default_bound_samples_per_split() -> usize { 3000 }

/// Configuration for surrogate-assisted pruning sampling. See
/// `PruningConfigToml::surrogate`.
#[derive(Debug, Deserialize, Clone, Copy)]
pub struct PruningSurrogateToml {
    /// `true` (default) = surrogate-assisted sampling active. Set `false`
    /// to disable — reproduces pre-(plain-sampling) behaviour.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Candidate children drawn per child slot; only the best-predicted one
    /// is really evaluated.
    #[serde(default = "default_surrogate_oversample")]
    pub oversample_factor: usize,
    /// Minimum recorded (vars, cost) pairs at a level before the surrogate
    /// is trusted; below this, sampling is plain (first candidate kept).
    #[serde(default = "default_surrogate_min_fit_samples")]
    pub min_fit_samples: usize,
    /// Gaussian-kernel length scale in normalized [0,1] units.
    #[serde(default = "default_surrogate_length_scale")]
    pub length_scale: f64,
}

fn default_surrogate_oversample() -> usize { 4 }
fn default_surrogate_min_fit_samples() -> usize { 60 }
fn default_surrogate_length_scale() -> f64 { 0.3 }

impl Default for PruningSurrogateToml {
    fn default() -> Self {
        PruningSurrogateToml {
            enabled: default_true(),
            oversample_factor: default_surrogate_oversample(),
            min_fit_samples: default_surrogate_min_fit_samples(),
            length_scale: default_surrogate_length_scale(),
        }
    }
}

fn default_pruning_samples_level0() -> usize { 2000 }
fn default_pruning_children_per_survivor() -> usize { 50 }
fn default_pruning_max_survivors() -> usize { 40 }
fn default_pruning_threshold_factor() -> f64 { 3.0 }
fn default_pruning_n_seeds() -> usize { 8 }
fn default_pruning_bidirectional() -> bool { true }

/// Configuration for the Phase 9w ballistic MGA grid scan.
///
/// Scans a departure-date x per-leg-TOF grid with pure Lambert legs and
/// analytic powered-flyby feasibility (no DSMs, no propagation, no
/// optimizer) — see the design notes
/// `trajectory_solver::mga_scan` for the model.
#[derive(Debug, Deserialize, Clone)]
pub struct MgaScanConfigToml {
    /// Scan horizon centered on `[optimization].departure_epoch`, spanning
    /// `± horizon_years/2` years.
    pub horizon_years: f64,
    /// Departure-date grid step [days].
    #[serde(default = "default_scan_departure_step_days")]
    pub departure_step_days: f64,
    /// Number of TOF grid points per leg, spanning that leg's
    /// `[optimization.mga].leg_tof_days` bounds.
    #[serde(default = "default_scan_tof_grid_points")]
    pub tof_grid_points_per_leg: usize,
    /// Prune branches whose per-flyby powered-flyby matching burn exceeds
    /// this [m/s] — the scan's "v∞-match tolerance" expressed as a ΔV cost.
    #[serde(default = "default_scan_flyby_dv_max_ms")]
    pub flyby_dv_max_ms: f64,
    /// Safety cap on stored feasible branches (see
    /// `trajectory_solver::mga_scan::MgaScanConfig::max_records`).
    #[serde(default = "default_scan_max_records")]
    pub max_records: usize,
}

fn default_scan_departure_step_days() -> f64 { 5.0 }
fn default_scan_tof_grid_points() -> usize { 40 }
fn default_scan_flyby_dv_max_ms() -> f64 { 3_000.0 }
fn default_scan_max_records() -> usize { 500_000 }

fn default_departure_vinf_max() -> f64 { 15_000.0 }
fn default_flyby_min_periapsis() -> f64 { 6_371_000.0 }
fn default_arrival_vinf_max() -> f64 { 25_000.0 }
fn default_de_f_weight() -> f64 { 0.8 }
fn default_de_cr() -> f64 { 0.9 }
fn default_de_population_size() -> usize { 300 }
fn default_de_generations() -> usize { 1000 }
fn default_de_restarts() -> usize { 3 }
fn default_de_seed() -> u64 { 42 }
fn default_de_adaptive() -> bool { true }
/// 0.2 AU (1 AU = 1.495_978_707e11 m, IAU 2012 nominal value).
fn default_min_solar_perihelion() -> f64 { 2.991_957_414e10 }

/// Configuration for the Tisserand beam-search sequence discovery (Phase 9j).
///
/// When present under `[optimization.mga.sequence_search]`, the optimizer
/// automatically discovers intermediate flyby bodies from `candidate_bodies`
/// rather than requiring the user to specify `flyby_bodies` explicitly.
/// The outer Tisserand beam search prunes the discrete sequence space; the
/// inner Stage-A DE optimizer runs on each top-ranked candidate sequence.
///
/// `flyby_bodies` in [`MgaParams`] is ignored when `sequence_search` is set.
#[derive(Debug, Clone, Deserialize)]
pub struct SequenceSearchConfig {
    /// Pool of body names to consider as intermediate flyby targets.
    /// Each must be a catalog body with a known `sma_m`. Repeated visits
    /// (e.g. Venus→Earth→Earth→Jupiter) are allowed — list each body once.
    pub candidate_bodies: Vec<String>,
    /// Maximum number of legs (= max intermediate flybys + 1). Keep ≤ 6 for
    /// reasonable run time; each level multiplies the beam by `candidate_bodies.len()`.
    pub max_legs: usize,
    /// Beam width: how many partial sequences to keep alive at each depth
    /// level of the Tisserand tree. Larger = more thorough outer search.
    /// Recommended: 10–50.
    #[serde(default = "default_beam_width")]
    pub beam_width: usize,
    /// How many top-ranked Tisserand sequences to run the full Stage-A inner
    /// DE optimizer on. Each is a complete `run_mga_fixed_sequence()` call.
    /// Recommended: 5–15.
    #[serde(default = "default_max_sequences_to_optimize")]
    pub max_sequences_to_optimize: usize,
    /// Approximate departure v∞ [m/s] used to seed the Tisserand graph walk.
    /// A rough estimate is fine; the inner optimizer finds the true value.
    #[serde(default = "default_vinf_departure_estimate")]
    pub vinf_departure_estimate_ms: f64,
}

fn default_beam_width() -> usize { 20 }
fn default_max_sequences_to_optimize() -> usize { 8 }
fn default_vinf_departure_estimate() -> f64 { 3_000.0 }

// ── Public helpers (also used by the HTTP server) ─────────────────────────────

/// True iff `v` is a finite, non-zero (normalizable) vector — the same
/// non-degenerate-direction check `CustomPlate`'s `normal` validation
/// already applies, shared here so every placement/boresight/direction
/// field added by the spacecraft-builder schema extension
/// uses the identical rule instead of a re-derived copy.
fn is_normalizable_direction(v: &[f64; 3]) -> bool {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    norm.is_finite() && norm >= 1e-9
}

/// The `mass_kg` override on any [`HardwareItem`] variant that has one
/// (every variant does, as of the derived-mass-properties extension,
///) — `None` if the variant's own override is unset (a caller
/// wanting the EFFECTIVE mass, including catalog fallback, should use
/// `vehicle_properties`'s resolver instead; this is only the raw override,
/// used here for validation and shared so a manually-supplied `mass_kg`
/// is checked identically regardless of which variant carries it).
pub fn hardware_item_mass_kg(h: &HardwareItem) -> Option<f64> {
    match h {
        HardwareItem::ReactionWheelCluster { mass_kg, .. }
        | HardwareItem::RCS { mass_kg, .. }
        | HardwareItem::RcsThruster { mass_kg, .. }
        | HardwareItem::StarTracker { mass_kg, .. }
        | HardwareItem::IMU { mass_kg, .. }
        | HardwareItem::OpNavCamera { mass_kg, .. }
        | HardwareItem::Lidar { mass_kg, .. }
        | HardwareItem::CommAntenna { mass_kg, .. }
        | HardwareItem::SolarPanel { mass_kg, .. }
        | HardwareItem::CustomPlate { mass_kg, .. } => *mass_kg,
    }
}

/// The body-frame direction a hardware item points, for an attitude-commander
/// `PointingRuleConfig` to target — `None` for an unplaced sensor
/// (`boresight`/`normal` is `None`) or a hardware type with no meaningful
/// pointing direction (`ReactionWheelCluster`/`RCS`/`RcsThruster`/`IMU`).
/// Shared between `check_config` (presence check) and `cruise.rs` (actual
/// resolution) so both read the exact same set of variants/fields.
pub fn hardware_pointing_vector(h: &HardwareItem) -> Option<[f64; 3]> {
    match h {
        HardwareItem::StarTracker { boresight, .. }
        | HardwareItem::OpNavCamera { boresight, .. }
        | HardwareItem::Lidar { boresight, .. } => *boresight,
        HardwareItem::CommAntenna { boresight, .. } => Some(*boresight),
        HardwareItem::SolarPanel { normal, .. } => *normal,
        HardwareItem::CustomPlate { normal, .. } => Some(*normal),
        HardwareItem::ReactionWheelCluster { .. }
        | HardwareItem::RCS { .. }
        | HardwareItem::RcsThruster { .. }
        | HardwareItem::IMU { .. } => None,
    }
}

/// Human-readable variant name, for error messages that need to name which
/// hardware entry failed a shared (variant-agnostic) check.
fn hardware_item_type_name(h: &HardwareItem) -> &'static str {
    match h {
        HardwareItem::ReactionWheelCluster { .. } => "ReactionWheelCluster",
        HardwareItem::RCS { .. } => "RCS",
        HardwareItem::RcsThruster { .. } => "RcsThruster",
        HardwareItem::StarTracker { .. } => "StarTracker",
        HardwareItem::IMU { .. } => "IMU",
        HardwareItem::OpNavCamera { .. } => "OpNavCamera",
        HardwareItem::Lidar { .. } => "Lidar",
        HardwareItem::CommAntenna { .. } => "CommAntenna",
        HardwareItem::SolarPanel { .. } => "SolarPanel",
        HardwareItem::CustomPlate { .. } => "CustomPlate",
    }
}

/// Validate a parsed [`MissionConfig`] and return all errors found.
///
/// Returns an empty `Vec` when the config is fully valid. Used by both the
/// CLI `validate` command and the HTTP `/api/validate` endpoint.
pub fn check_config(c: &MissionConfig) -> Vec<String> {
    let mut errors: Vec<String> = Vec::new();

    if c.target_body.mu_m3s2 <= 0.0 {
        errors.push("target_body.mu_m3s2 must be > 0".into());
    }
    for tb in &c.target_body.third_bodies {
        if body_models::TargetBody::by_name(tb).is_none() {
            errors.push(format!(
                "target_body.third_bodies: '{}' is not in the body catalog \
                 (known: Sun, Bennu, Earth, Moon, Mars, Apophis, Ryugu, Jupiter, Saturn, Europa, \
                 Titan, Phobos, Deimos, Eros, Didymos)",
                tb
            ));
        }
    }
    if c.target_body.radius_m <= 0.0 {
        errors.push("target_body.radius_m must be > 0".into());
    }
    match c.target_body.gravity_model {
        GravityModel::J2 => {
            if c.target_body.j2.is_none() {
                errors.push("target_body.j2 required when gravity_model = \"J2\"".into());
            }
        }
        GravityModel::J2J3J4 => {
            if c.target_body.j2.is_none() {
                errors.push("target_body.j2 required when gravity_model = \"J2J3J4\"".into());
            }
            if c.target_body.j3.is_none() {
                errors.push("target_body.j3 required when gravity_model = \"J2J3J4\"".into());
            }
            if c.target_body.j4.is_none() {
                errors.push("target_body.j4 required when gravity_model = \"J2J3J4\"".into());
            }
        }
        _ => {}
    }
    if c.spacecraft.mass_kg <= 0.0 {
        errors.push("spacecraft.mass_kg must be > 0".into());
    }
    if c.spacecraft.dry_mass_kg <= 0.0 {
        errors.push("spacecraft.dry_mass_kg must be > 0".into());
    }
    if c.spacecraft.propellant_mass_kg < 0.0 {
        errors.push("spacecraft.propellant_mass_kg must be >= 0".into());
    }
    let budget = c.spacecraft.dry_mass_kg + c.spacecraft.propellant_mass_kg;
    if (budget - c.spacecraft.mass_kg).abs() > 0.1 {
        errors.push(format!(
            "mass budget: dry ({:.1}) + propellant ({:.1}) = {:.1}, but mass_kg = {:.1}",
            c.spacecraft.dry_mass_kg, c.spacecraft.propellant_mass_kg, budget, c.spacecraft.mass_kg,
        ));
    }
    if c.spacecraft.inertia_diag_kgm2.iter().any(|&v| v <= 0.0) {
        errors.push("spacecraft.inertia_diag_kgm2 all components must be > 0".into());
    }
    if c.spacecraft.bus_dims_m.iter().any(|&v| v <= 0.0) {
        errors.push("spacecraft.bus_dims_m all components must be > 0".into());
    }
    if let Some(prop) = &c.spacecraft.propulsion {
        if prop.isp_s <= 0.0 {
            errors.push("spacecraft.propulsion.isp_s must be > 0".into());
        }
        if prop.thrust_n <= 0.0 {
            errors.push("spacecraft.propulsion.thrust_n must be > 0".into());
        }
    }
    for h in &c.spacecraft.hardware {
        if let Some(m) = hardware_item_mass_kg(h) {
            if !(m > 0.0) {
                errors.push(format!(
                    "spacecraft.hardware {}: mass_kg must be > 0 when set",
                    hardware_item_type_name(h)
                ));
            }
        }
        if let HardwareItem::CustomPlate { normal, area_m2, rho_s, rho_d, .. } = h {
            let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if !norm.is_finite() || norm < 1e-9 {
                errors.push(
                    "spacecraft.hardware CustomPlate: normal must be a non-zero, finite vector".into(),
                );
            }
            if !(*area_m2 > 0.0) {
                errors.push("spacecraft.hardware CustomPlate: area_m2 must be > 0".into());
            }
            if let Some(rs) = rho_s {
                if !(0.0..=1.0).contains(rs) {
                    errors.push("spacecraft.hardware CustomPlate: rho_s must be in [0, 1]".into());
                }
            }
            if let Some(rd) = rho_d {
                if !(0.0..=1.0).contains(rd) {
                    errors.push("spacecraft.hardware CustomPlate: rho_d must be in [0, 1]".into());
                }
            }
            if let (Some(rs), Some(rd)) = (rho_s, rho_d) {
                if rs + rd > 1.0 + 1e-9 {
                    errors.push(
                        "spacecraft.hardware CustomPlate: rho_s + rho_d must be <= 1 \
                         (reflected energy cannot exceed incident energy)"
                            .into(),
                    );
                }
            }
        }
        if let HardwareItem::RcsThruster { thrust_n, direction, isp_s, .. } = h {
            if !(*thrust_n > 0.0) {
                errors.push("spacecraft.hardware RcsThruster: thrust_n must be > 0".into());
            }
            if !is_normalizable_direction(direction) {
                errors.push(
                    "spacecraft.hardware RcsThruster: direction must be a non-zero, finite vector".into(),
                );
            }
            if let Some(isp) = isp_s {
                if !(*isp > 0.0) {
                    errors.push("spacecraft.hardware RcsThruster: isp_s must be > 0".into());
                }
            }
        }
        if let HardwareItem::StarTracker { boresight: Some(b), .. }
        | HardwareItem::OpNavCamera { boresight: Some(b), .. }
        | HardwareItem::Lidar { boresight: Some(b), .. } = h
        {
            if !is_normalizable_direction(b) {
                errors.push(
                    "spacecraft.hardware: boresight must be a non-zero, finite vector".into(),
                );
            }
        }
        if let HardwareItem::CommAntenna { boresight, beamwidth_deg, .. } = h {
            if !is_normalizable_direction(boresight) {
                errors.push(
                    "spacecraft.hardware CommAntenna: boresight must be a non-zero, finite vector".into(),
                );
            }
            if !(*beamwidth_deg > 0.0 && *beamwidth_deg <= 360.0) {
                errors.push("spacecraft.hardware CommAntenna: beamwidth_deg must be in (0, 360]".into());
            }
        }
        if let HardwareItem::SolarPanel { position_m, normal, width_m, height_m, rho_s, rho_d, articulation, .. } = h {
            let placed = position_m.is_some() || normal.is_some();
            if placed && (position_m.is_none() || normal.is_none()) {
                errors.push(
                    "spacecraft.hardware SolarPanel: position_m and normal must both be set together \
                     for a placed panel (a placed panel needs both a location and a deployed direction)"
                        .into(),
                );
            }
            match articulation {
                Some(PanelArticulation::OneAxis { axis }) => {
                    if !is_normalizable_direction(axis) {
                        errors.push(
                            "spacecraft.hardware SolarPanel articulation (OneAxis): axis must be a \
                             non-zero, finite vector"
                                .into(),
                        );
                    }
                }
                Some(PanelArticulation::TwoAxis { axis1, axis2 }) => {
                    if !is_normalizable_direction(axis1) {
                        errors.push(
                            "spacecraft.hardware SolarPanel articulation (TwoAxis): axis1 must be a \
                             non-zero, finite vector"
                                .into(),
                        );
                    }
                    if !is_normalizable_direction(axis2) {
                        errors.push(
                            "spacecraft.hardware SolarPanel articulation (TwoAxis): axis2 must be a \
                             non-zero, finite vector"
                                .into(),
                        );
                    }
                    if is_normalizable_direction(axis1) && is_normalizable_direction(axis2) {
                        let dot = axis1[0] * axis2[0] + axis1[1] * axis2[1] + axis1[2] * axis2[2];
                        let n1 = (axis1[0] * axis1[0] + axis1[1] * axis1[1] + axis1[2] * axis1[2]).sqrt();
                        let n2 = (axis2[0] * axis2[0] + axis2[1] * axis2[1] + axis2[2] * axis2[2]).sqrt();
                        let cos_angle = dot / (n1 * n2);
                        if 1.0 - cos_angle.abs() < 1e-6 {
                            errors.push(
                                "spacecraft.hardware SolarPanel articulation (TwoAxis): axis1 and axis2 \
                                 must not be (nearly) parallel — a two-axis mechanism needs two genuinely \
                                 independent rotation axes to provide 2 real degrees of freedom"
                                    .into(),
                            );
                        }
                    }
                }
                None => {}
            }
            if let Some(n) = normal {
                if !is_normalizable_direction(n) {
                    errors.push(
                        "spacecraft.hardware SolarPanel: normal must be a non-zero, finite vector".into(),
                    );
                }
            }
            if width_m.is_some() != height_m.is_some() {
                errors.push(
                    "spacecraft.hardware SolarPanel: width_m and height_m must both be set together \
                     (or both omitted)"
                        .into(),
                );
            }
            if let Some(w) = width_m {
                if !(*w > 0.0) {
                    errors.push("spacecraft.hardware SolarPanel: width_m must be > 0".into());
                }
            }
            if let Some(hh) = height_m {
                if !(*hh > 0.0) {
                    errors.push("spacecraft.hardware SolarPanel: height_m must be > 0".into());
                }
            }
            if let Some(rs) = rho_s {
                if !(0.0..=1.0).contains(rs) {
                    errors.push("spacecraft.hardware SolarPanel: rho_s must be in [0, 1]".into());
                }
            }
            if let Some(rd) = rho_d {
                if !(0.0..=1.0).contains(rd) {
                    errors.push("spacecraft.hardware SolarPanel: rho_d must be in [0, 1]".into());
                }
            }
            if let (Some(rs), Some(rd)) = (rho_s, rho_d) {
                if rs + rd > 1.0 + 1e-9 {
                    errors.push(
                        "spacecraft.hardware SolarPanel: rho_s + rho_d must be <= 1 \
                         (reflected energy cannot exceed incident energy)"
                            .into(),
                    );
                }
            }
        }
    }
    if c.trajectory.phases.is_empty() {
        errors.push("trajectory.phases must contain at least one phase".into());
    }
    if c.simulation.rtol <= 0.0 {
        errors.push("simulation.rtol must be > 0".into());
    }
    if c.simulation.atol <= 0.0 {
        errors.push("simulation.atol must be > 0".into());
    }
    if c.simulation.dt_truth_s <= 0.0 {
        errors.push("simulation.dt_truth_s must be > 0".into());
    }
    if c.simulation.dt_meas_s < c.simulation.dt_truth_s {
        errors.push("simulation.dt_meas_s must be >= dt_truth_s".into());
    }

    // Phase 14a: `Launch` departure mode prerequisites — the launcher curve
    // (departure ΔV is the launcher pool), a site latitude (the plane
    // feasibility rule), and a departure-body pole (RLA/DLA are quoted in
    // the body's equatorial frame) for BOTH departure-body fields.
    if let Some(dep) = &c.trajectory.departure {
        if dep.mode == DepartureMode::Launch {
            if c.spacecraft.launch_vehicle.is_none() {
                errors.push("trajectory.departure.mode = \"Launch\" requires spacecraft.launch_vehicle".into());
            }
            match &dep.launch_site {
                None => errors.push("trajectory.departure.mode = \"Launch\" requires trajectory.departure.launch_site".into()),
                Some(site) => {
                    if !(-90.0..=90.0).contains(&site.lat_deg) {
                        errors.push("trajectory.departure.launch_site.lat_deg must be in [-90, 90]".into());
                    }
                }
            }
            if let Some(alt) = dep.parking_altitude_m {
                if alt <= 0.0 {
                    errors.push("trajectory.departure.parking_altitude_m must be > 0".into());
                }
            }
            if let Some(i) = dep.inclination_deg {
                if !(0.0..=180.0).contains(&i) {
                    errors.push("trajectory.departure.inclination_deg must be in [0, 180]".into());
                }
            }
            let mut dep_bodies = vec![c.trajectory.departure_body.as_str()];
            if let Some(opt) = &c.optimization {
                if !opt.departure_body.eq_ignore_ascii_case(&c.trajectory.departure_body) {
                    dep_bodies.push(opt.departure_body.as_str());
                }
            }
            for name in dep_bodies {
                match body_models::TargetBody::by_name(name) {
                    Some(b) if b.pole_ra_deg.is_some() && b.pole_dec_deg.is_some() => {}
                    Some(_) => errors.push(format!(
                        "trajectory.departure.mode = \"Launch\": departure body '{name}' has no catalog pole (pole_ra_deg/pole_dec_deg), so RLA/DLA are undefined"
                    )),
                    None => errors.push(format!(
                        "trajectory.departure.mode = \"Launch\": departure body '{name}' is not in the body catalog"
                    )),
                }
            }
        }
    }

    if let Some(opt) = &c.optimization {
        if body_models::TargetBody::by_name(&opt.departure_body).is_none() {
            errors.push(format!(
                "optimization.departure_body: '{}' is not in the body catalog",
                opt.departure_body
            ));
        }
        if body_models::TargetBody::by_name(&opt.target_body).is_none() {
            errors.push(format!(
                "optimization.target_body: '{}' is not in the body catalog",
                opt.target_body
            ));
        }
        if opt.dv_min_ms < 0.0 || opt.dv_max_ms <= opt.dv_min_ms {
            errors.push("optimization.dv_min_ms must be >= 0 and < dv_max_ms".into());
        }
        if opt.max_coast_days <= 0.0 {
            errors.push("optimization.max_coast_days must be > 0".into());
        }
        // Configurable angle search ranges: each pair must be
        // a real, non-empty range when set; phi must stay within +/-90 deg
        // (past that, "out-of-plane share" stops meaning anything).
        let theta_min = opt.theta_min_deg.unwrap_or(0.0);
        let theta_max = opt.theta_max_deg.unwrap_or(360.0);
        if theta_max <= theta_min || theta_min < 0.0 || theta_max > 360.0 {
            errors.push("optimization.theta_min_deg/theta_max_deg must satisfy 0 <= min < max <= 360".into());
        }
        let phi_min = opt.phi_min_deg.unwrap_or(-45.0);
        let phi_max = opt.phi_max_deg.unwrap_or(45.0);
        if phi_max <= phi_min || phi_min < -90.0 || phi_max > 90.0 {
            errors.push("optimization.phi_min_deg/phi_max_deg must satisfy -90 <= min < max <= 90".into());
        }
        // A real capture burn (Orbit/Landing/SampleReturn) needs a target
        // distance to insert at; MatchTargetDistance already required this
        // separately, but now any capturing objective needs it to even
        // compute a meaningful fitness, regardless of which ObjectiveFunction
        // is picked. Flyby needs no arrival burn at all. Rendezvous
        // needs no capture radius either -- its arrival burn is the full
        // relative-velocity magnitude (`arrival_dv_ms` in mga.rs), not a
        // periapsis-radius-parameterized capture orbit.
        let needs_capture_radius = !matches!(
            c.mission.objective,
            MissionObjective::Flyby | MissionObjective::Rendezvous
        );
        if needs_capture_radius && c.trajectory.capture.as_ref().and_then(|cap| cap.target_orbit_radius_m).is_none() {
            errors.push(
                "optimization is configured with a capturing mission.objective (not Flyby) -- \
                 [trajectory.capture].target_orbit_radius_m is required to compute a real arrival/capture burn."
                    .into(),
            );
        }
        if let Some(cap) = c.trajectory.capture.as_ref() {
            if !(0.0..1.0).contains(&cap.capture_eccentricity) {
                errors.push(
                    "[trajectory.capture].capture_eccentricity must be in [0, 1) -- \
                     a closed (elliptic or circular) capture orbit."
                        .into(),
                );
            }
            // Phase 9y: a configured target/capture distance
            // below the target body's own physical radius is a real,
            // silent misconfiguration -- e.g. the frontend's capture-radius
            // slider previously defaulted to a flat 7,000 km regardless of
            // target body, which sits INSIDE Jupiter's real ~71,492 km
            // radius. For the single-leg GA/PSO/MBH path this doesn't cause
            // a literal body penetration (the propagator's collision-margin
            // stop already floors `closest_approach_m` at
            // `radius_m + COLLISION_MARGIN_M`), but the search would spend
            // its whole budget chasing an unreachable target and MGA's
            // capture-burn model (`arrival_dv_ms`, no propagation at all)
            // has no such floor -- catching this at config time is cheap
            // and applies uniformly to both paths, independent of either's
            // deeper fix.
            if let Some(r) = cap.target_orbit_radius_m {
                if r < c.target_body.radius_m {
                    errors.push(format!(
                        "[trajectory.capture].target_orbit_radius_m ({r:.0} m) is smaller than \
                         target_body.radius_m ({:.0} m) -- this targets a point inside the body's \
                         physical surface.",
                        c.target_body.radius_m
                    ));
                }
            }
        }
        if opt.force_model.rtol <= 0.0 {
            errors.push("optimization.force_model.rtol must be > 0".into());
        }
        if opt.force_model.atol <= 0.0 {
            errors.push("optimization.force_model.atol must be > 0".into());
        }
        // No "must be non-empty"/"must have a CentralWhenInSoi entry" check
        // here -- the departure and target bodies are now *always*
        // auto-registered as CentralWhenInSoi candidates regardless of this
        // list (`optimize.rs::force_model_body_entries`), so an empty list
        // is legitimate (no extra third-body perturbers configured).
        for b in &opt.force_model.bodies {
            if body_models::TargetBody::by_name(&b.name).is_none() {
                errors.push(format!(
                    "optimization.force_model.bodies: '{}' is not in the body catalog",
                    b.name
                ));
            }
            if let BodyRole::AlwaysThirdBody = b.role {
                if b.fidelity.is_some() {
                    errors.push(format!(
                        "optimization.force_model.bodies: '{}' has role = \"AlwaysThirdBody\" \
                         but specifies fidelity — third-body perturbers are point-mass only",
                        b.name
                    ));
                }
            }
        }
        match opt.method {
            OptimizationMethod::GA => {
                if opt.ga.is_none() {
                    errors.push("optimization.ga required when method = \"GA\"".into());
                }
            }
            OptimizationMethod::PSO => {
                if opt.pso.is_none() {
                    errors.push("optimization.pso required when method = \"PSO\"".into());
                }
            }
            OptimizationMethod::MultipleShooting => {
                if opt.shooting.is_none() {
                    errors.push(
                        "optimization.shooting required when method = \"MultipleShooting\""
                            .into(),
                    );
                }
            }
            OptimizationMethod::MGA => match &opt.mga {
                None => errors.push("optimization.mga required when method = \"MGA\"".into()),
                Some(mga) => {
                    // When sequence_search is set, flyby_bodies is auto-discovered;
                    // leg_tof_days length check is skipped (the search determines legs).
                    if mga.sequence_search.is_none() {
                        let expected_legs = mga.flyby_bodies.len() + 1;
                        if mga.leg_tof_days.len() != expected_legs {
                            errors.push(format!(
                                "optimization.mga.leg_tof_days has {} entries but \
                                 flyby_bodies has {} bodies — expected {} legs \
                                 (departure_body -> flyby[0] -> ... -> target_body)",
                                mga.leg_tof_days.len(),
                                mga.flyby_bodies.len(),
                                expected_legs
                            ));
                        }
                        for fb in &mga.flyby_bodies {
                            if body_models::TargetBody::by_name(fb).is_none() {
                                errors.push(format!(
                                    "optimization.mga.flyby_bodies: '{}' is not in the body catalog",
                                    fb
                                ));
                            }
                        }
                    } else if let Some(ss) = &mga.sequence_search {
                        if ss.candidate_bodies.is_empty() {
                            errors.push("optimization.mga.sequence_search.candidate_bodies must not be empty".into());
                        }
                        if ss.max_legs < 1 {
                            errors.push("optimization.mga.sequence_search.max_legs must be ≥ 1".into());
                        }
                        if ss.beam_width < 1 {
                            errors.push("optimization.mga.sequence_search.beam_width must be ≥ 1".into());
                        }
                        if ss.max_sequences_to_optimize < 1 {
                            errors.push("optimization.mga.sequence_search.max_sequences_to_optimize must be ≥ 1".into());
                        }
                        for b in &ss.candidate_bodies {
                            if body_models::TargetBody::by_name(b).is_none() {
                                errors.push(format!(
                                    "optimization.mga.sequence_search.candidate_bodies: '{}' is not in the body catalog",
                                    b
                                ));
                            }
                        }
                    }
                    for (i, [lo, hi]) in mga.leg_tof_days.iter().enumerate() {
                        if *lo <= 0.0 || hi <= lo {
                            errors.push(format!(
                                "optimization.mga.leg_tof_days[{}] must satisfy 0 < min < max",
                                i
                            ));
                        }
                    }
                    if mga.departure_vinf_min_ms < 0.0
                        || mga.departure_vinf_min_ms >= mga.departure_vinf_max_ms
                    {
                        errors.push(
                            "optimization.mga.departure_vinf_min_ms must satisfy \
                             0 <= min < departure_vinf_max_ms"
                                .into(),
                        );
                    }
                    if let Some(scan) = &mga.scan {
                        if scan.horizon_years <= 0.0 {
                            errors.push("optimization.mga.scan.horizon_years must be > 0".into());
                        }
                        if scan.departure_step_days <= 0.0 {
                            errors.push("optimization.mga.scan.departure_step_days must be > 0".into());
                        }
                        if scan.tof_grid_points_per_leg < 2 {
                            errors.push("optimization.mga.scan.tof_grid_points_per_leg must be >= 2".into());
                        }
                        if scan.flyby_dv_max_ms <= 0.0 {
                            errors.push("optimization.mga.scan.flyby_dv_max_ms must be > 0".into());
                        }
                    }
                    // Phase 9w-vi: scan_informed_window needs a real scan
                    // config to run — without one there's nothing to derive
                    // bounds from, and silently falling back to the raw
                    // config bounds would defeat the whole point of the flag
                    // (routing around exactly the hand-written-bounds bug
                    // class this feature exists to catch).
                    if mga.scan_informed_window && mga.scan.is_none() {
                        errors.push(
                            "optimization.mga.scan_informed_window is true but \
                             optimization.mga.scan is not configured — the scan-informed \
                             window derivation has no scan to run"
                                .into(),
                        );
                    }
                    if mga.search_method == SearchMethod::Mbh {
                        if !(0.0..=1.0).contains(&mga.mbh.perturb_fraction) {
                            errors.push("optimization.mga.mbh.perturb_fraction must be in [0, 1]".into());
                        }
                        if mga.mbh.kick_scale <= 0.0 {
                            errors.push("optimization.mga.mbh.kick_scale must be > 0".into());
                        }
                        if mga.mbh.local_max_iter == 0 {
                            errors.push("optimization.mga.mbh.local_max_iter must be >= 1".into());
                        }
                        if let Some(stop_after) = mga.mbh.stop_after {
                            if stop_after == 0 {
                                errors.push("optimization.mga.mbh.stop_after must be >= 1 when set".into());
                            }
                        }
                        if mga.mbh.local_optimizer == MbhLocalOptimizerToml::HookeJeeves {
                            let hj = &mga.mbh.hooke_jeeves;
                            if hj.max_fevals == 0 {
                                errors.push("optimization.mga.mbh.hooke_jeeves.max_fevals must be >= 1".into());
                            }
                            if !(0.0..=1.0).contains(&hj.start_range) {
                                errors.push("optimization.mga.mbh.hooke_jeeves.start_range must be in (0, 1]".into());
                            }
                            // NOTE: a third-party reference's own doc comment for this
                            // constraint reads "stop_range must be in (start_range, 1]",
                            // but that's backwards relative to what its own source
                            // actually enforces (confirmed directly against the real
                            // comparison logic) -- stop_range must be SMALLER than
                            // start_range, matching a shrinking search (defaults 0.1
                            // start / 0.01 stop only make sense this way around).
                            if hj.stop_range >= hj.start_range || hj.stop_range > 1.0 {
                                errors.push(
                                    "optimization.mga.mbh.hooke_jeeves.stop_range must be < \
                                     start_range and <= 1"
                                        .into(),
                                );
                            }
                            if !(0.0..1.0).contains(&hj.reduction_coeff) {
                                errors.push(
                                    "optimization.mga.mbh.hooke_jeeves.reduction_coeff must be in (0, 1)".into(),
                                );
                            }
                        }
                        if let Some(gs) = &mga.mbh.global_stall {
                            if gs.patience == 0 {
                                errors.push("optimization.mga.mbh.global_stall.patience must be >= 1".into());
                            }
                            if gs.margin_frac < 0.0 {
                                errors.push("optimization.mga.mbh.global_stall.margin_frac must be >= 0".into());
                            }
                        }
                        {
                            let m = &mga.mbh.migration;
                            if m.interval == 0 {
                                errors.push("optimization.mga.mbh.migration.interval must be >= 1".into());
                            }
                            if m.margin_frac < 0.0 {
                                errors.push("optimization.mga.mbh.migration.margin_frac must be >= 0".into());
                            }
                        }
                        if let Some(p) = &mga.pruning {
                            if let Some(b) = &p.bound {
                                if !(0.0..1.0).contains(&b.safety_frac) {
                                    errors.push("optimization.mga.pruning.bound.safety_frac must be in [0, 1)".into());
                                }
                                if b.epoch_bins == 0 || b.vinf_bins == 0 {
                                    errors.push("optimization.mga.pruning.bound epoch_bins/vinf_bins must be >= 1".into());
                                }
                                if b.samples_per_split == 0 {
                                    errors.push("optimization.mga.pruning.bound.samples_per_split must be >= 1".into());
                                }
                            }
                            {
                                let s = &p.surrogate;
                                if s.oversample_factor < 2 {
                                    errors.push("optimization.mga.pruning.surrogate.oversample_factor must be >= 2".into());
                                }
                                if s.min_fit_samples < 10 {
                                    errors.push("optimization.mga.pruning.surrogate.min_fit_samples must be >= 10".into());
                                }
                                if s.length_scale <= 0.0 {
                                    errors.push("optimization.mga.pruning.surrogate.length_scale must be > 0".into());
                                }
                            }
                        }
                    }
                }
            },
        }
    }

    if let Some(seed) = &c.cruise_seed {
        if seed.reference.len() < 2 {
            errors.push("cruise_seed.reference must have at least 2 points".into());
        } else {
            if (seed.reference[0].t_s).abs() > 1e-6 {
                errors.push("cruise_seed.reference[0].t_s must be 0.0".into());
            }
            let r0_matches = (0..3).all(|i| (seed.reference[0].r_m[i] - seed.r0_m[i]).abs() < 1.0);
            let v0_matches = (0..3).all(|i| (seed.reference[0].v_mps[i] - seed.v0_m[i]).abs() < 1e-3);
            // With a test window the truth starts at the reference state
            // at `window.start_s`; `r0_m`/`v0_m` are not used.
            if seed.window.is_none() && (!r0_matches || !v0_matches) {
                errors.push(
                    "cruise_seed.reference[0] must match r0_m/v0_m (within 1 m / 1 mm/s) -- a \
                     mismatched seed/reference start would silently report a nonzero dispersion \
                     at t=0 unrelated to real tracking quality"
                        .into(),
                );
            }
            if !seed.reference.windows(2).all(|w| w[1].t_s >= w[0].t_s) {
                errors.push("cruise_seed.reference must be sorted ascending by t_s".into());
            }
            let ref_span_s = seed.reference.last().unwrap().t_s - seed.reference[0].t_s;
            if seed.duration_s > ref_span_s + 1e-6 {
                errors.push(format!(
                    "cruise_seed.duration_s ({:.1} s) exceeds the reference trajectory's own span ({:.1} s)",
                    seed.duration_s, ref_span_s
                ));
            }
        }
        if seed.duration_s <= 0.0 {
            errors.push("cruise_seed.duration_s must be > 0".into());
        }
        if let Some(w) = &seed.window {
            if w.start_s < 0.0 || w.end_s <= w.start_s || w.end_s > seed.duration_s + 1e-6 {
                errors.push(format!(
                    "cruise_seed.window must satisfy 0 <= start_s < end_s <= duration_s (got start {:.1}, end {:.1}, duration {:.1})",
                    w.start_s, w.end_s, seed.duration_s
                ));
            }
        }
        if seed.tick_s <= 0.0 {
            errors.push("cruise_seed.tick_s must be > 0".into());
        }
        if let Some(bt) = seed.burn_tick_s {
            if bt <= 0.0 || bt > seed.tick_s {
                errors.push("cruise_seed.burn_tick_s must satisfy 0 < burn_tick_s <= tick_s".into());
            }
        }

        let mode_names: std::collections::HashSet<&str> = seed.modes.iter().map(|m| m.name.as_str()).collect();
        if mode_names.len() != seed.modes.len() {
            errors.push("cruise_seed.modes: mode names must be unique".into());
        }
        for m in &seed.modes {
            if m.rules.is_empty() {
                errors.push(format!("cruise_seed.modes['{}']: must have at least 1 rule", m.name));
            }
            for (i, rule) in m.rules.iter().enumerate() {
                match c.spacecraft.hardware.get(rule.hardware_index) {
                    None => errors.push(format!(
                        "cruise_seed.modes['{}'].rules[{}]: hardware_index {} out of range \
                         (spacecraft.hardware has {} items)",
                        m.name, i, rule.hardware_index, c.spacecraft.hardware.len()
                    )),
                    Some(item) => {
                        if hardware_pointing_vector(item).is_none() {
                            errors.push(format!(
                                "cruise_seed.modes['{}'].rules[{}]: hardware_index {} ({}) has no \
                                 usable boresight/normal -- either an unplaced sensor or a hardware \
                                 type with no pointing direction",
                                m.name, i, rule.hardware_index, hardware_item_type_name(item)
                            ));
                        }
                    }
                }
                if let PointingTargetConfig::Body { name } = &rule.target {
                    if !seed.body_tracks.iter().any(|t| &t.name == name) {
                        errors.push(format!(
                            "cruise_seed.modes['{}'].rules[{}]: target Body('{}') has no matching \
                             cruise_seed.body_tracks entry",
                            m.name, i, name
                        ));
                    }
                }
            }
        }
        for entry in &seed.mode_schedule {
            if entry.start_s >= entry.end_s {
                errors.push(format!(
                    "cruise_seed.mode_schedule: start_s ({:.1}) must be < end_s ({:.1})",
                    entry.start_s, entry.end_s
                ));
            }
            if !mode_names.contains(entry.mode.as_str()) {
                errors.push(format!("cruise_seed.mode_schedule: mode '{}' not found in cruise_seed.modes", entry.mode));
            }
        }
        if let Some(safe) = &seed.safe_mode {
            if !mode_names.contains(safe.as_str()) {
                errors.push(format!("cruise_seed.safe_mode '{}' not found in cruise_seed.modes", safe));
            }
        }
        for track in &seed.body_tracks {
            if track.track.len() < 2 {
                errors.push(format!("cruise_seed.body_tracks['{}']: must have at least 2 points", track.name));
            } else if !track.track.windows(2).all(|w| w[1].t_s >= w[0].t_s) {
                errors.push(format!("cruise_seed.body_tracks['{}']: must be sorted ascending by t_s", track.name));
            }
            if track.soi_capture {
                let mu_resolves = track.mu_m3s2.is_some() || body_models::TargetBody::by_name(&track.name).is_some();
                if !mu_resolves {
                    errors.push(format!(
                        "cruise_seed.body_tracks['{}']: soi_capture is true but no mu_m3s2 resolves (no explicit \
                         override and no body_models catalog match) -- this track would silently stay third-body-\
                         only, defeating the point of setting soi_capture",
                        track.name
                    ));
                }
            }
        }
        if let Some(threshold) = seed.tcm_dr_threshold_m {
            if threshold <= 0.0 {
                errors.push("cruise_seed.tcm_dr_threshold_m: must be positive".to_string());
            }
            // Not a hard error -- `cruise::build_tcm_config` already treats
            // this combination as a documented no-op (a threshold with no
            // propulsion has nothing to burn with) -- but silently doing
            // nothing is exactly the kind of mistake a config author should
            // be told about explicitly rather than discovering by noticing
            // TCM never fires.
            if c.spacecraft.propulsion.is_none() {
                errors.push(
                    "cruise_seed.tcm_dr_threshold_m is set but spacecraft.propulsion is not -- TCM has no thrust/Isp source and will never fire".to_string(),
                );
            }
        }
        if let Some(stride) = seed.report_stride {
            if stride == 0 {
                errors.push("cruise_seed.report_stride: must be >= 1 (0 is not a valid stride)".to_string());
            }
        }
        if !seed.planned_burns.is_empty() {
            if !seed.planned_burns.windows(2).all(|w| w[1].epoch_s >= w[0].epoch_s) {
                errors.push("cruise_seed.planned_burns: must be sorted ascending by epoch_s".to_string());
            }
            if c.spacecraft.propulsion.is_none() {
                errors.push(
                    "cruise_seed.planned_burns is set but spacecraft.propulsion is not -- planned burns have no thrust/Isp source and will never fire".to_string(),
                );
            }
            for (i, burn) in seed.planned_burns.iter().enumerate() {
                if let Some(target_epoch_s) = burn.target_epoch_s {
                    if target_epoch_s <= burn.epoch_s {
                        errors.push(format!(
                            "cruise_seed.planned_burns[{i}]: target_epoch_s ({target_epoch_s}) must be after \
                             epoch_s ({}) -- a burn re-targets toward a future encounter, not the past",
                            burn.epoch_s
                        ));
                    }
                    if target_epoch_s > seed.duration_s {
                        errors.push(format!(
                            "cruise_seed.planned_burns[{i}]: target_epoch_s ({target_epoch_s}) exceeds \
                             cruise_seed.duration_s ({}) -- the fresh re-targeting solve needs a reference \
                             sample at the target epoch, which won't exist past the leg's own end",
                            seed.duration_s
                        ));
                    }
                }
                // Review D5: capture_body needs a resolvable
                // body track (position AND mu), and makes no sense combined
                // with target_epoch_s (a burn is a position-shaping DSM or a
                // velocity-matching capture, never both).
                if let Some(capture_body) = &burn.capture_body {
                    if burn.target_epoch_s.is_some() {
                        errors.push(format!(
                            "cruise_seed.planned_burns[{i}]: capture_body and target_epoch_s are mutually \
                             exclusive -- a position-shaping DSM re-solve and a velocity-matching capture \
                             re-solve are different maneuvers"
                        ));
                    }
                    match seed.body_tracks.iter().find(|t| &t.name == capture_body) {
                        None => errors.push(format!(
                            "cruise_seed.planned_burns[{i}]: capture_body '{capture_body}' has no matching \
                             cruise_seed.body_tracks entry -- the fresh capture solve needs the body's real \
                             state at trigger time"
                        )),
                        Some(t) => {
                            let mu_resolves = t.mu_m3s2.is_some() || body_models::TargetBody::by_name(capture_body).is_some();
                            if !mu_resolves {
                                errors.push(format!(
                                    "cruise_seed.planned_burns[{i}]: capture_body '{capture_body}' resolves no \
                                     mu_m3s2 (no explicit override on its body_tracks entry and no catalog \
                                     match) -- the circularization speed sqrt(mu/r) is uncomputable"
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid mission TOML, parameterised on the `[target_body]` block
    /// so tests can exercise catalog resolution / override / error paths.
    fn mission_toml(target_body_block: &str) -> String {
        format!(
            r#"
[mission]
name = "Test"
objective = "Orbit"

{target_body_block}

[spacecraft]
mass_kg = 1000.0
dry_mass_kg = 800.0
propellant_mass_kg = 200.0
bus_dims_m = [2.0, 2.0, 0.63]
inertia_diag_kgm2 = [366.67, 366.67, 666.67]
srp_model = "Cannonball"

[trajectory]
phases = ["Cruise"]
solver = "Hohmann"
departure_body = "Earth"

[gnc]
navigation_filter = "EKF"
pointing_mode = "Nadir"
attitude_controller = "ReactionWheelPD"

[simulation]
integrator = "DormandPrince45"
rtol = 1.0e-9
atol = 1.0e-7
dt_truth_s = 10.0
dt_meas_s = 120.0
monte_carlo_runs = 0
output_dir = "out/test/"
"#
        )
    }

    #[test]
    fn target_body_resolves_omitted_fields_from_catalog() {
        let toml = mission_toml(
            r#"[target_body]
name = "Apophis"
ephemeris = "Keplerian""#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should resolve from catalog");
        assert_eq!(cfg.target_body.mu_m3s2, 2.646e-3);
        assert_eq!(cfg.target_body.radius_m, 185.0);
        assert!(matches!(cfg.target_body.gravity_model, GravityModel::PointMass));
        assert!(matches!(cfg.target_body.atmosphere, AtmosphereModel::None));
    }

    #[test]
    fn target_body_explicit_field_overrides_catalog() {
        let toml = mission_toml(
            r#"[target_body]
name = "Apophis"
mu_m3s2 = 1.234e-3
ephemeris = "Keplerian""#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should parse with override");
        // Explicit value wins...
        assert_eq!(cfg.target_body.mu_m3s2, 1.234e-3);
        // ...but fields not overridden still resolve from the catalog.
        assert_eq!(cfg.target_body.radius_m, 185.0);
    }

    #[test]
    fn target_body_unknown_name_without_fields_errors() {
        let toml = mission_toml(
            r#"[target_body]
name = "Planet Nine"
ephemeris = "Keplerian""#,
        );
        let result: Result<MissionConfig, _> = toml::from_str(&toml);
        assert!(result.is_err(), "unknown body with no explicit constants should fail to parse");
    }

    #[test]
    fn target_body_known_name_with_full_manual_fields_ignores_catalog() {
        // A body not in the catalog is fine as long as everything is given manually.
        let toml = mission_toml(
            r#"[target_body]
name = "CustomRock"
mu_m3s2 = 9.99e-4
radius_m = 100.0
gravity_model = "PointMass"
atmosphere = "None"
ephemeris = "Keplerian""#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("fully-specified custom body should parse");
        assert_eq!(cfg.target_body.mu_m3s2, 9.99e-4);
        assert_eq!(cfg.target_body.radius_m, 100.0);
    }

    // ── CustomPlate hardware (Phase 13a) ────────────────────────────────────

    #[test]
    fn custom_plate_hardware_parses_and_passes_check_config() {
        let toml = format!(
            "{}\n[[spacecraft.hardware]]\ntype = \"CustomPlate\"\nnormal = [0.0, 1.0, 0.0]\n\
             area_m2 = 2.5\ncenter_offset_m = [0.0, 1.5, 0.0]\nrho_s = 0.1\nrho_d = 0.2\n",
            mission_toml(
                r#"[target_body]
name = "Apophis"
ephemeris = "Keplerian""#,
            )
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("CustomPlate should parse");
        assert!(
            matches!(cfg.spacecraft.hardware.as_slice(), [HardwareItem::CustomPlate { area_m2, .. }] if *area_m2 == 2.5)
        );
        let errors = check_config(&cfg);
        assert!(errors.is_empty(), "unexpected check_config errors: {errors:?}");
    }

    #[test]
    fn custom_plate_zero_normal_fails_check_config() {
        let toml = format!(
            "{}\n[[spacecraft.hardware]]\ntype = \"CustomPlate\"\nnormal = [0.0, 0.0, 0.0]\n\
             area_m2 = 1.0\ncenter_offset_m = [0.0, 0.0, 0.0]\n",
            mission_toml(
                r#"[target_body]
name = "Apophis"
ephemeris = "Keplerian""#,
            )
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should still parse");
        let errors = check_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("normal")), "expected a normal-vector error, got: {errors:?}");
    }

    #[test]
    fn custom_plate_reflectivity_exceeding_unity_fails_check_config() {
        let toml = format!(
            "{}\n[[spacecraft.hardware]]\ntype = \"CustomPlate\"\nnormal = [1.0, 0.0, 0.0]\n\
             area_m2 = 1.0\ncenter_offset_m = [0.0, 0.0, 0.0]\nrho_s = 0.7\nrho_d = 0.6\n",
            mission_toml(
                r#"[target_body]
name = "Apophis"
ephemeris = "Keplerian""#,
            )
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should still parse");
        let errors = check_config(&cfg);
        assert!(
            errors.iter().any(|e| e.contains("rho_s + rho_d")),
            "expected a reflectivity-budget error, got: {errors:?}"
        );
    }

    // ── Spacecraft-builder placement extension ────────────────────

    #[test]
    fn rcs_thruster_parses_and_passes_check_config() {
        let toml = format!(
            "{}\n[[spacecraft.hardware]]\ntype = \"RcsThruster\"\nthrust_n = 1.0\n\
             position_m = [1.0, 0.0, 0.5]\ndirection = [0.0, 1.0, 0.0]\n",
            mission_toml(r#"[target_body]
name = "Apophis"
ephemeris = "Keplerian""#)
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("RcsThruster should parse");
        let errors = check_config(&cfg);
        assert!(errors.is_empty(), "unexpected check_config errors: {errors:?}");
    }

    #[test]
    fn rcs_thruster_zero_direction_fails_check_config() {
        let toml = format!(
            "{}\n[[spacecraft.hardware]]\ntype = \"RcsThruster\"\nthrust_n = 1.0\n\
             position_m = [1.0, 0.0, 0.5]\ndirection = [0.0, 0.0, 0.0]\n",
            mission_toml(r#"[target_body]
name = "Apophis"
ephemeris = "Keplerian""#)
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should still parse");
        let errors = check_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("direction")), "expected a direction error, got: {errors:?}");
    }

    #[test]
    fn comm_antenna_parses_and_passes_check_config() {
        let toml = format!(
            "{}\n[[spacecraft.hardware]]\ntype = \"CommAntenna\"\nboresight = [0.0, 0.0, 1.0]\nbeamwidth_deg = 20.0\n",
            mission_toml(r#"[target_body]
name = "Apophis"
ephemeris = "Keplerian""#)
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("CommAntenna should parse");
        let errors = check_config(&cfg);
        assert!(errors.is_empty(), "unexpected check_config errors: {errors:?}");
    }

    #[test]
    fn comm_antenna_zero_beamwidth_fails_check_config() {
        let toml = format!(
            "{}\n[[spacecraft.hardware]]\ntype = \"CommAntenna\"\nboresight = [0.0, 0.0, 1.0]\nbeamwidth_deg = 0.0\n",
            mission_toml(r#"[target_body]
name = "Apophis"
ephemeris = "Keplerian""#)
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should still parse");
        let errors = check_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("beamwidth_deg")), "expected a beamwidth error, got: {errors:?}");
    }

    #[test]
    fn placed_solar_panel_requires_both_position_and_normal() {
        let toml = format!(
            "{}\n[[spacecraft.hardware]]\ntype = \"SolarPanel\"\narea_m2 = 4.0\nposition_m = [0.0, 1.0, 0.0]\n",
            mission_toml(r#"[target_body]
name = "Apophis"
ephemeris = "Keplerian""#)
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should still parse");
        let errors = check_config(&cfg);
        assert!(
            errors.iter().any(|e| e.contains("position_m and normal must both be set")),
            "expected a partial-placement error, got: {errors:?}"
        );
    }

    #[test]
    fn placed_solar_panel_with_position_and_normal_passes_check_config() {
        let toml = format!(
            "{}\n[[spacecraft.hardware]]\ntype = \"SolarPanel\"\narea_m2 = 4.0\n\
             position_m = [0.0, 1.5, 0.0]\nnormal = [0.0, 1.0, 0.0]\nwidth_m = 2.0\nheight_m = 2.0\n",
            mission_toml(r#"[target_body]
name = "Apophis"
ephemeris = "Keplerian""#)
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should parse");
        let errors = check_config(&cfg);
        assert!(errors.is_empty(), "unexpected check_config errors: {errors:?}");
    }

    #[test]
    fn panel_one_axis_articulation_parses_and_passes_check_config() {
        let toml = format!(
            "{}\n[[spacecraft.hardware]]\ntype = \"SolarPanel\"\narea_m2 = 4.0\n\
             position_m = [0.0, 1.5, 0.0]\nnormal = [0.0, 1.0, 0.0]\n\
             articulation = {{ kind = \"OneAxis\", axis = [1.0, 0.0, 0.0] }}\n",
            mission_toml(r#"[target_body]
name = "Apophis"
ephemeris = "Keplerian""#)
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("OneAxis articulation should parse");
        let errors = check_config(&cfg);
        assert!(errors.is_empty(), "unexpected check_config errors: {errors:?}");
    }

    #[test]
    fn panel_two_axis_articulation_parallel_axes_fails_check_config() {
        let toml = format!(
            "{}\n[[spacecraft.hardware]]\ntype = \"SolarPanel\"\narea_m2 = 4.0\n\
             position_m = [0.0, 1.5, 0.0]\nnormal = [0.0, 1.0, 0.0]\n\
             articulation = {{ kind = \"TwoAxis\", axis1 = [1.0, 0.0, 0.0], axis2 = [2.0, 0.0, 0.0] }}\n",
            mission_toml(r#"[target_body]
name = "Apophis"
ephemeris = "Keplerian""#)
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should still parse");
        let errors = check_config(&cfg);
        assert!(
            errors.iter().any(|e| e.contains("must not be (nearly) parallel")),
            "expected a parallel-axes error, got: {errors:?}"
        );
    }

    #[test]
    fn panel_two_axis_articulation_independent_axes_passes_check_config() {
        let toml = format!(
            "{}\n[[spacecraft.hardware]]\ntype = \"SolarPanel\"\narea_m2 = 4.0\n\
             position_m = [0.0, 1.5, 0.0]\nnormal = [0.0, 1.0, 0.0]\n\
             articulation = {{ kind = \"TwoAxis\", axis1 = [1.0, 0.0, 0.0], axis2 = [0.0, 0.0, 1.0] }}\n",
            mission_toml(r#"[target_body]
name = "Apophis"
ephemeris = "Keplerian""#)
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("TwoAxis articulation should parse");
        let errors = check_config(&cfg);
        assert!(errors.is_empty(), "unexpected check_config errors: {errors:?}");
    }
}
