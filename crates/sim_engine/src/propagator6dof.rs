//! The single 6DOF propagator (Phase 13c) — one translation+attitude
//! propagator used for every mission mode/phase, cruise included, per
//! the design notes Phase 13 design statement ("i just want one 6dof propagator,
//! suited for any kind of mode/phase"). See `docs/MP/MANUAL.md` §7 for
//! the governing equations and the design rationale summarized below.
//!
//! ## Architecture
//!
//! Translation and attitude are advanced by **two separate adaptive Dopri5
//! integrations per control tick**, not one coupled 17-state ODE. This is a
//! deliberate simplification, not a shortcut:
//!
//! - Translation → attitude coupling is real (gravity-gradient torque needs
//!   the spacecraft's position; SRP torque needs the Sun direction) and is
//!   preserved — it just uses the position/Sun-direction *frozen at the
//!   tick's start* (a zero-order-hold, exactly matching the ZOH already
//!   applied to the commanded control torque over the same tick).
//! - Attitude → translation coupling does **not exist** in the decoupled
//!   path at all: translation (`trajectory_solver::propagate`) applies
//!   gravity only (point mass + optional zonal + third-body) — no SRP term
//!   of any kind, cannonball included (a real, more severe simplification
//!   than an earlier version of this doc comment claimed — see
//!   `docs/MP/MANUAL.md` §7's correction). Because no attitude-
//!   dependent (or even attitude-INdependent SRP) force acts on translation
//!   in that path, splitting the integration loses no physics there.
//! - **Phase 13f (done): finite-burn propulsion.** [`step_tick_with_burn`]
//!   adds a SECOND, fully-coupled integration path (one 14-state ODE: r, v,
//!   q, omega, mass together) used only for ticks with an active burn,
//!   where the thrust vector is fixed in the BODY frame (a real engine
//!   mounted along a fixed spacecraft axis) and its inertial direction
//!   therefore depends on the attitude achieved DURING the tick — exactly
//!   the coupling this module's original design flagged as unavoidable once
//!   finite-burn propulsion arrived. Burn-free ticks are UNCHANGED — they
//!   still use the decoupled path via [`step_tick`], with identical
//!   results (verified by regression test). This coupled path also
//!   includes cannonball SRP (on translation, correctly, unlike the
//!   decoupled path above) since it was cheap to add while building a
//!   from-scratch force assembly anyway — see `docs/MP/MANUAL.md` §8.3.
//!
//! Translation reuses Layer 1's SOI-patched propagator
//! (`trajectory_solver::propagator::propagate`) **unmodified** — this is the
//! same "no duplicate logic" reasoning that makes `resolve_central_body` a
//! shared, `pub` function between the two crates rather than re-derived here.
//! Rotation uses its own adaptive Dopri5 integration of the quaternion
//! kinematics + Euler's equations (§6), with real-time attitude-dependent
//! torque (gravity-gradient and flat-plate SRP torque both depend on the
//! *evolving* quaternion within the tick, not just the frozen tick-start
//! value — only the external environment, i.e. which body is central and
//! where the Sun is, is frozen).

use nalgebra::{Vector3, Vector4};
use ode_solvers::dopri5::Dopri5;
use ode_solvers::{SVector as OdeVec, System};

use orbital_models::attitude::{body_to_inertial, inertial_to_body, omega_dot, qdot, qnorm};
use orbital_models::{cannonball, flat_plate_torque_body, gravity_gradient, pressure_at};
use trajectory_solver::{
    propagate as propagate_translation, resolve_central_body, resolve_collision, PropagatorBody,
};

use crate::truth::{SpacecraftProperties, SrpTruthModel};

type RotState = OdeVec<f64, 7>;

/// Full 6DOF state carried between control ticks. Translational components
/// (`r_m`, `v_mps`) are always expressed in the propagator's reference frame
/// (heliocentric, matching `trajectory_solver::propagator`'s convention) —
/// central-body switching happens *inside* [`step_tick`] and is transparent
/// to the caller, exactly as it already is for the 3DOF Layer 1 propagator.
#[derive(Clone, Copy, Debug)]
pub struct SixDofState {
    pub t_s: f64,
    pub r_m: Vector3<f64>,
    pub v_mps: Vector3<f64>,
    /// Attitude quaternion `[w, x, y, z]`, body → reference/inertial frame.
    /// Frame-independent — attitude has no notion of "which body is
    /// central," unlike translation.
    pub q: Vector4<f64>,
    pub omega_radps: Vector3<f64>,
    /// Reaction wheel speeds [rad/s] — carried through unchanged by
    /// [`step_tick`]. Wheel-speed integration is exact under a zero-order-
    /// hold motor torque (a linear ODE — see `actuators::wheel_step`), so it
    /// is deliberately kept OUTSIDE this propagator as the controller/
    /// allocator layer's responsibility (Phase 13e), the same separation of
    /// concerns `engine.rs::SimEngine::step` already uses today.
    pub wheel_speeds_radps: [f64; 4],
    /// Total spacecraft mass [kg] (Phase 13f). Constant through
    /// [`step_tick`] (no propellant depletion in the decoupled/no-burn
    /// path); depleted via the Tsiolkovsky relation during an active burn
    /// in [`step_tick_with_burn`]. Kept in the state (not `sc.mass_kg`,
    /// which is the caller's fixed nominal value) because it is genuinely
    /// time-varying once burns are involved — the same reasoning
    /// `TruthState::c_r` in the legacy engine already follows for a
    /// quantity that starts at a config value but evolves at runtime.
    pub mass_kg: f64,
}

/// Rotational-dynamics ODE for one control tick: quaternion kinematics
/// (`q̇ = ½Ξ(q)ω`, MANUAL.md §6.2) coupled to Euler's equations with
/// reaction-wheel momentum (`Iω̇ = τ − ω×(Iω+H_w)`, §6.3). Torque sources
/// (gravity-gradient §5.2, SRP §4.1/4.2) are evaluated against the
/// *evolving* quaternion `y[0..4]` at every internal step the adaptive
/// integrator takes — only the external geometry (central-body offset, Sun
/// direction) is frozen at the tick's start, per this module's doc comment.
struct RotOde {
    inertia: Vector3<f64>,
    /// Central body's gravitational parameter for gravity-gradient torque
    /// [m^3/s^2] — the Sun's, if no body's SOI contains the spacecraft.
    mu_central: f64,
    /// Spacecraft position relative to the current central body, frozen at
    /// tick start [m]. See module doc comment for why freezing this (rather
    /// than re-resolving it from the coupled translational state) is exact
    /// at the current fidelity level.
    r_central_rel0: Vector3<f64>,
    /// Spacecraft → Sun unit vector, reference/inertial frame, frozen at
    /// tick start.
    sun_hat_ref0: Vector3<f64>,
    /// Local solar radiation pressure [N/m^2], frozen at tick start.
    p_srp0: f64,
    srp: SrpTruthModel,
    /// Commanded control torque (already allocated across actuators by the
    /// controller layer — Phase 13e), body frame, held fixed (ZOH) over the
    /// tick [N*m].
    control_torque_body: Vector3<f64>,
    /// Reaction-wheel angular momentum, body frame, held fixed over the tick
    /// [N*m*s] — the `H_w` cross-coupling term in Euler's equations.
    h_wheel_body: Vector3<f64>,
}

/// Disturbance-torque breakdown by source [N*m, body frame] — for telemetry/
/// verification (per the design notes "every new capability gets a plot" rule,
/// and the sizing stage, §13) as well as internal use by [`RotOde`]. Kept as separate
/// components rather than only the sum so a caller can plot/log which source
/// dominates at a given instant.
#[derive(Clone, Copy, Debug, Default)]
pub struct TorqueBreakdown {
    pub gravity_gradient: Vector3<f64>,
    pub srp: Vector3<f64>,
}

impl TorqueBreakdown {
    pub fn disturbance_total(&self) -> Vector3<f64> {
        self.gravity_gradient + self.srp
    }
}

/// Translational acceleration breakdown by source [m/s^2, reference/inertial
/// frame] — the translational analog of [`TorqueBreakdown`], for telemetry/
/// verification/plotting (Phase 13b). `central_gravity` folds the point-mass
/// term together with any zonal-harmonic (J2-J4) contribution the central
/// body carries — both act through the same physical channel ("the central
/// body's own field"), unlike SRP vs. gravity-gradient torque, so there is no
/// plotting value in separating them the way [`TorqueBreakdown`] separates
/// its two genuinely distinct sources. `srp` is Cannonball-only and only ever
/// nonzero when `include_coupled_srp` is set on the call to
/// [`translational_accel_breakdown`] — see this module's own doc comment for
/// why the decoupled [`step_tick`] path applies NO SRP force to translation
/// at all (flat-plate included): this struct reports the physical truth of
/// whichever path actually ran, it does not editorialize.
#[derive(Clone, Copy, Debug, Default)]
pub struct AccelBreakdown {
    pub central_gravity: Vector3<f64>,
    pub third_body: Vector3<f64>,
    pub srp: Vector3<f64>,
}

impl AccelBreakdown {
    pub fn total(&self) -> Vector3<f64> {
        self.central_gravity + self.third_body + self.srp
    }
}

/// Shared central-gravity (point-mass + optional zonal) / third-body
/// acceleration assembly, in the CURRENT central body's local frame (`r` is
/// central-relative, or heliocentric if `central_index` is `None`) — the
/// exact physics [`CoupledBurnLegOde::system`] already computes inline, and
/// what [`translational_accel_breakdown`] duplicates for telemetry. Factored
/// out once both needed it, so the two never drift apart (the "single source
/// of truth for every algorithm" rule in `the design notes`).
fn gravity_breakdown(
    r: &Vector3<f64>,
    t_abs: f64,
    mu_central: f64,
    central_index: Option<usize>,
    bodies: &[PropagatorBody],
) -> (Vector3<f64>, Vector3<f64>) {
    let mut central = orbital_models::GravityModel::point_mass(r, mu_central);
    if let Some(ci) = central_index {
        if let Some(zf) = &bodies[ci].central_fidelity {
            central += orbital_models::GravityModel::zonal_harmonics_body_oriented(
                r, mu_central, zf.r0_m, zf.j2, zf.j3, zf.j4, zf.pole_ra_rad, zf.pole_dec_rad,
            );
        }
    }
    let central_pos = central_index.map(|ci| (bodies[ci].state_at)(t_abs).0);
    let mut third_body = Vector3::zeros();
    for (i, b) in bodies.iter().enumerate() {
        if Some(i) == central_index {
            continue;
        }
        let (body_pos_ref, _) = (b.state_at)(t_abs);
        let body_pos_relative = match central_pos {
            Some(cp) => body_pos_ref - cp,
            None => body_pos_ref,
        };
        third_body += orbital_models::GravityModel::third_body(r, &body_pos_relative, b.mu_m3s2);
    }
    (central, third_body)
}

/// Translational force breakdown at the CURRENT state, for telemetry/
/// plotting (Phase 13b/13j). `include_coupled_srp` should be `true` only when
/// logging a run driven by [`step_tick_with_burn`] with an active burn (the
/// only path that ever applies Cannonball SRP to translation) — pass `false`
/// when logging a [`step_tick`]-driven (decoupled) run, so the reported
/// breakdown matches what that run actually integrated rather than what a
/// different path would have.
pub fn translational_accel_breakdown(
    state: &SixDofState,
    sc: &SpacecraftProperties,
    bodies: &[PropagatorBody],
    reference_mu_m3s2: f64,
    include_coupled_srp: bool,
) -> AccelBreakdown {
    let central_index = resolve_central_body(&state.r_m, bodies, state.t_s);
    let mu_central = match central_index {
        Some(i) => bodies[i].mu_m3s2,
        None => reference_mu_m3s2,
    };
    let central_pos = central_index.map(|i| (bodies[i].state_at)(state.t_s).0);
    let r_local = match central_pos {
        Some(cp) => state.r_m - cp,
        None => state.r_m,
    };
    let (central_gravity, third_body) =
        gravity_breakdown(&r_local, state.t_s, mu_central, central_index, bodies);

    let srp = if include_coupled_srp {
        match &sc.srp {
            SrpTruthModel::Cannonball { c_r, area_m2 } => {
                let dist_to_sun = state.r_m.norm(); // reference frame is heliocentric-origin (see SixDofState doc comment)
                if dist_to_sun > 1e-6 {
                    let anti_sun = state.r_m / dist_to_sun;
                    let p_srp = pressure_at(dist_to_sun);
                    cannonball(&anti_sun, p_srp, *c_r, *area_m2, state.mass_kg)
                } else {
                    Vector3::zeros()
                }
            }
            SrpTruthModel::FlatPlate { .. } => Vector3::zeros(),
        }
    } else {
        Vector3::zeros()
    };

    AccelBreakdown { central_gravity, third_body, srp }
}

impl RotOde {
    fn torque_breakdown(&self, q: &Vector4<f64>) -> TorqueBreakdown {
        let tau_gg = gravity_gradient(q, &self.r_central_rel0, self.mu_central, &self.inertia);
        let tau_srp = match &self.srp {
            SrpTruthModel::Cannonball { .. } => Vector3::zeros(),
            SrpTruthModel::FlatPlate { plates } => {
                let sun_hat_body = inertial_to_body(q, &self.sun_hat_ref0);
                flat_plate_torque_body(plates, &sun_hat_body, self.p_srp0)
            }
        };
        TorqueBreakdown { gravity_gradient: tau_gg, srp: tau_srp }
    }

    fn torque_body(&self, q: &Vector4<f64>) -> Vector3<f64> {
        self.torque_breakdown(q).disturbance_total() + self.control_torque_body
    }
}

/// Resolve the tick-start attitude-torque environment: which body is
/// central (for gravity-gradient's `mu`/`r`) and the Sun direction/pressure
/// (for SRP torque) — see the module doc comment for why this is frozen at
/// tick start rather than re-resolved continuously. Shared by [`step_tick`]
/// and [`disturbance_torque_breakdown`] so telemetry/diagnostics use exactly
/// the same environment resolution the propagator itself uses, never a
/// separate re-derivation that could silently drift out of sync.
fn resolve_tick_environment(
    r_m: &Vector3<f64>,
    t_s: f64,
    bodies: &[PropagatorBody],
    reference_mu_m3s2: f64,
) -> (f64, Vector3<f64>, Vector3<f64>, f64) {
    let central_index = resolve_central_body(r_m, bodies, t_s);
    let (mu_central, r_central_rel0) = match central_index {
        Some(i) => {
            let (central_pos, _central_vel) = (bodies[i].state_at)(t_s);
            (bodies[i].mu_m3s2, r_m - central_pos)
        }
        None => (reference_mu_m3s2, *r_m),
    };
    // The reference frame is heliocentric (`reference_mu_m3s2` is the Sun's
    // mu, per `trajectory_solver::propagator::PropagatorBody`'s own
    // convention), so the Sun sits at that frame's origin regardless of
    // which body is currently central for gravity purposes — the
    // spacecraft-to-Sun direction is simply -r̂ in that frame.
    let dist_to_sun = r_m.norm();
    let sun_hat_ref0 = if dist_to_sun > 1e-6 {
        -r_m / dist_to_sun
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let p_srp0 = pressure_at(dist_to_sun);
    (mu_central, r_central_rel0, sun_hat_ref0, p_srp0)
}

/// Disturbance-torque breakdown at the CURRENT state (not a tick-start
/// snapshot) — for telemetry/plotting. Uses the exact same environment
/// resolution [`step_tick`] uses internally (see [`resolve_tick_environment`]),
/// evaluated fresh against `state`'s own position/attitude rather than
/// whatever was frozen at the start of whichever tick produced it, so a
/// caller logging this after every tick sees the true instantaneous torque,
/// not a tick-stale one.
pub fn disturbance_torque_breakdown(
    state: &SixDofState,
    sc: &SpacecraftProperties,
    bodies: &[PropagatorBody],
    reference_mu_m3s2: f64,
) -> TorqueBreakdown {
    let (mu_central, r_central_rel0, sun_hat_ref0, p_srp0) =
        resolve_tick_environment(&state.r_m, state.t_s, bodies, reference_mu_m3s2);
    let ode = RotOde {
        inertia: sc.inertia_diag_kgm2,
        mu_central,
        r_central_rel0,
        sun_hat_ref0,
        p_srp0,
        srp: sc.srp.clone(),
        control_torque_body: Vector3::zeros(),
        h_wheel_body: Vector3::zeros(),
    };
    ode.torque_breakdown(&state.q)
}

impl System<f64, RotState> for RotOde {
    fn system(&self, _t: f64, y: &RotState, dy: &mut RotState) {
        let q = Vector4::new(y[0], y[1], y[2], y[3]);
        let omega = Vector3::new(y[4], y[5], y[6]);

        let torque = self.torque_body(&q);
        let qd = qdot(&q, &omega);
        let od = omega_dot(&omega, &torque, &self.h_wheel_body, &self.inertia);

        dy[0] = qd[0]; dy[1] = qd[1]; dy[2] = qd[2]; dy[3] = qd[3];
        dy[4] = od.x; dy[5] = od.y; dy[6] = od.z;
    }

    fn solout(&mut self, _t: f64, _y: &RotState, _dy: &RotState) -> bool {
        false
    }
}

/// Advance [`SixDofState`] by one control tick (`tick_s`), per the mode-
/// scheduled zero-order-hold tick scheme in `docs/MP/MANUAL.md` §2.4:
/// the controller's commanded torque and wheel momentum are held fixed for
/// the whole tick, and BOTH translation and rotation integrate adaptively
/// (Dopri5) within it — quiet cruise legs take one or two internal steps per
/// tick, dynamics near a body or during a maneuver refine automatically.
///
/// `bodies`/`reference_mu_m3s2` are exactly `trajectory_solver::propagator`'s
/// SOI-candidate list and reference-frame gravitational parameter (almost
/// always the Sun) — this function is a thin attitude-aware wrapper around
/// that existing, tested translational propagator, not a reimplementation.
///
/// `control_torque_body`/`h_wheel_body` are supplied by the caller (the
/// controller/allocation layer, Phase 13e — not yet built) — this function
/// only propagates dynamics given an already-decided control torque, it does
/// not compute one.
#[allow(clippy::too_many_arguments)]
pub fn step_tick(
    state: &SixDofState,
    tick_s: f64,
    sc: &SpacecraftProperties,
    bodies: &[PropagatorBody],
    reference_mu_m3s2: f64,
    control_torque_body: Vector3<f64>,
    h_wheel_body: Vector3<f64>,
    rtol: f64,
    atol: f64,
) -> SixDofState {
    // ── Translation: unmodified reuse of Layer 1's SOI-patched propagator ──
    let translated = propagate_translation(
        state.r_m, state.v_mps, state.t_s, tick_s, reference_mu_m3s2, bodies, tick_s.max(1.0), rtol, atol,
    );
    let (r_new, v_new) = match translated.last() {
        Some(p) => (p.r_m, p.v_mps),
        None => (state.r_m, state.v_mps),
    };

    // ── Attitude environment, frozen at tick start (see module doc comment) ──
    let (mu_central, r_central_rel0, sun_hat_ref0, p_srp0) =
        resolve_tick_environment(&state.r_m, state.t_s, bodies, reference_mu_m3s2);

    // ── Rotation: adaptive Dopri5 over the tick, attitude-dependent torque ──
    let y0 = RotState::from_column_slice(&[
        state.q[0], state.q[1], state.q[2], state.q[3],
        state.omega_radps.x, state.omega_radps.y, state.omega_radps.z,
    ]);
    let ode = RotOde {
        inertia: sc.inertia_diag_kgm2,
        mu_central,
        r_central_rel0,
        sun_hat_ref0,
        p_srp0,
        srp: sc.srp.clone(),
        control_torque_body,
        h_wheel_body,
    };
    let tick_s = tick_s.max(1e-6);
    let mut stepper = Dopri5::from_param(
        ode, 0.0, tick_s, tick_s, y0, rtol, atol,
        0.9, 0.04, 0.2, 10.0, tick_s, 0.0, 100_000, 1000,
        ode_solvers::dop_shared::OutputType::Sparse,
    );
    if let Err(e) = stepper.integrate() {
        eprintln!(
            "Warning: 6DOF propagator's rotational integration did not complete cleanly over a \
             {tick_s:.3e} s tick: {e:?} — attitude for this tick may be inaccurate."
        );
    }
    let y_final = stepper.y_out().last().copied().unwrap_or(y0);
    let q_new = qnorm(&Vector4::new(y_final[0], y_final[1], y_final[2], y_final[3]));
    let omega_new = Vector3::new(y_final[4], y_final[5], y_final[6]);

    SixDofState {
        t_s: state.t_s + tick_s,
        r_m: r_new,
        v_mps: v_new,
        q: q_new,
        omega_radps: omega_new,
        wheel_speeds_radps: state.wheel_speeds_radps,
        mass_kg: state.mass_kg,
    }
}

/// Body +x axis expressed in the reference/inertial frame — the boresight
/// convention already used throughout this repo (`GNC/AutonomousNavigation`,
/// `sim_engine::truth`). Convenience wrapper for telemetry/plotting.
pub fn boresight(state: &SixDofState) -> Vector3<f64> {
    body_to_inertial(&state.q, &Vector3::new(1.0, 0.0, 0.0))
}

// ── Phase 13f: finite-burn propulsion, fully coupled ────────────────────────

/// A body-frame-fixed finite burn: a real engine mounted along a fixed
/// spacecraft axis, firing for the ENTIRE duration of whichever tick(s) the
/// caller passes `Some(&BurnConfig)` for. Ignition/cutoff timing is a
/// tick-selection decision made by the caller (consistent with the
/// mode-scheduled tick-length design, `docs/MP/MANUAL.md` §2.4 — a
/// burn-phase mission mode is expected to select short ticks anyway, so
/// sub-tick ignition/cutoff timing isn't needed for reasonable fidelity),
/// not something this struct or [`step_tick_with_burn`] tracks internally.
#[derive(Clone, Copy, Debug)]
pub struct BurnConfig {
    /// Thrust magnitude [N].
    pub thrust_n: f64,
    /// Specific impulse [s].
    pub isp_s: f64,
    /// Thrust direction, BODY frame (need not be pre-normalized). Fixed
    /// relative to the spacecraft — its INERTIAL direction depends on the
    /// attitude achieved during the burn, which is exactly why this needs
    /// the coupled integrator rather than [`step_tick`]'s decoupled path.
    pub body_dir: Vector3<f64>,
    /// Thrust application point relative to the centre of mass, body frame
    /// [m]. Nonzero produces a real disturbance torque (`docs/MP/MANUAL.md`
    /// §5.3) — e.g. from CoM shift as propellant depletes, or a genuinely
    /// off-axis-mounted engine. Zero means perfectly aligned thrust (no
    /// misalignment torque).
    pub thrust_offset_body_m: Vector3<f64>,
}

type BurnState = OdeVec<f64, 14>;

/// Fully-coupled ODE for ONE LEG of a burn-active tick: translation,
/// rotation, AND mass depletion together (r, v, q, omega, mass — 14
/// states), under a FIXED central body for the leg. Mirrors
/// `trajectory_solver::propagator::LegOde` exactly in state-frame
/// convention (`r`/`v` are relative to `central_index`'s body, or the
/// reference/Sun frame if `None`) so the same `resolve_central_body`/
/// `resolve_collision` primitives that drive `propagate()`'s own per-leg
/// loop can drive this one too — see [`propagate_coupled_burn`], the only
/// caller, for the leg-switching loop itself. Built as a self-contained
/// force/torque assembly (reusing the same public `orbital_models`/
/// `orbital_math` primitives the decoupled path and
/// `trajectory_solver::propagator` already use) rather than modifying
/// `trajectory_solver::propagate()` itself, which every existing Layer 1
/// solver depends on and which is deliberately left untouched here (see
/// `docs/MP/MANUAL.md` §7's correction note for why).
struct CoupledBurnLegOde<'a, F: FnMut(f64, &Vector3<f64>) -> bool> {
    inertia: Vector3<f64>,
    mu_central: f64,
    bodies: &'a [PropagatorBody<'a>],
    central_index: Option<usize>,
    /// Absolute mission time at the start of THIS LEG — `system`/`solout`
    /// receive leg-local time `t`, matching `LegOde`'s own convention.
    t0_abs_s: f64,
    srp: SrpTruthModel,
    burn: BurnConfig,
    control_torque_body: Vector3<f64>,
    h_wheel_body: Vector3<f64>,
    should_stop: F,
    /// True last-visited `(t, state)` on every `solout` call — same
    /// staleness guard `LegOde` uses (see its own doc comment) for the case
    /// `should_stop` aborts before the first recorded sample.
    last_visited: std::rc::Rc<std::cell::Cell<(f64, BurnState)>>,
}

impl<'a, F: FnMut(f64, &Vector3<f64>) -> bool> System<f64, BurnState> for CoupledBurnLegOde<'a, F> {
    fn system(&self, t: f64, y: &BurnState, dy: &mut BurnState) {
        let r = Vector3::new(y[0], y[1], y[2]); // central-relative (or heliocentric if central_index is None)
        let v = Vector3::new(y[3], y[4], y[5]);
        let q = Vector4::new(y[6], y[7], y[8], y[9]);
        let omega = Vector3::new(y[10], y[11], y[12]);
        let mass = y[13].max(1.0); // floor avoids a divide-by-zero if a caller mis-sizes propellant

        let t_abs = self.t0_abs_s + t;

        // ── Gravity: point-mass central (+ optional zonal fidelity) +
        // third-body, evaluated at the TRUE integration time -- identical
        // assembly to `LegOde::system` (Layer 1), just with q/omega/mass
        // carried alongside. ──────────────────────────────────────────────
        let (grav_central, grav_third_body) =
            gravity_breakdown(&r, t_abs, self.mu_central, self.central_index, self.bodies);
        let mut a = grav_central + grav_third_body;
        let central_pos = self.central_index.map(|ci| (self.bodies[ci].state_at)(t_abs).0);

        // Heliocentric position -- needed for Sun direction/pressure, since
        // `r` above is central-relative whenever a body other than the Sun
        // is resolved as central this leg.
        let r_ref = central_pos.map(|cp| r + cp).unwrap_or(r);

        // ── SRP (cannonball only, on translation -- see this module's doc
        // comment for why this coupled path can afford it and the decoupled
        // path today cannot without touching trajectory_solver). ──────────
        let dist_to_sun = r_ref.norm();
        if dist_to_sun > 1e-6 {
            let anti_sun = r_ref / dist_to_sun;
            let p_srp = pressure_at(dist_to_sun);
            if let SrpTruthModel::Cannonball { c_r, area_m2 } = &self.srp {
                a += orbital_models::cannonball(&anti_sun, p_srp, *c_r, *area_m2, mass);
            }
            // FlatPlate SRP is intentionally NOT applied to translation here
            // either -- this coupled path adds cannonball (cheap, and closes
            // part of the §7 gap) but does not attempt attitude-dependent
            // translational SRP, which is a bigger step left for later.
        }

        // ── Thrust: body-frame-fixed, converted to inertial via the
        // EVOLVING q -- this is the real attitude->translation coupling
        // this whole path exists for. ───────────────────────────────────────
        let thrust_dir_body = self.burn.body_dir.normalize();
        let thrust_dir_inertial = body_to_inertial(&q, &thrust_dir_body);
        a += (self.burn.thrust_n / mass) * thrust_dir_inertial;

        // ── Torque: gravity-gradient (using the REAL evolving r, already
        // central-relative) + SRP torque (flat-plate only, else zero) +
        // thrust misalignment + commanded control torque. ──────────────────
        let tau_gg = gravity_gradient(&q, &r, self.mu_central, &self.inertia);
        let tau_srp = match &self.srp {
            SrpTruthModel::Cannonball { .. } => Vector3::zeros(),
            SrpTruthModel::FlatPlate { plates } => {
                let sun_hat_ref = if dist_to_sun > 1e-6 { -r_ref / dist_to_sun } else { Vector3::new(1.0, 0.0, 0.0) };
                let sun_hat_body = inertial_to_body(&q, &sun_hat_ref);
                let p_srp = pressure_at(dist_to_sun.max(1.0));
                flat_plate_torque_body(plates, &sun_hat_body, p_srp)
            }
        };
        let thrust_force_body = self.burn.thrust_n * thrust_dir_body;
        let tau_misalign = self.burn.thrust_offset_body_m.cross(&thrust_force_body);
        let total_torque = tau_gg + tau_srp + tau_misalign + self.control_torque_body;

        let qd = qdot(&q, &omega);
        let od = omega_dot(&omega, &total_torque, &self.h_wheel_body, &self.inertia);
        let mass_dot = -self.burn.thrust_n / (self.burn.isp_s * orbital_models::constants::G0);

        dy[0] = v.x; dy[1] = v.y; dy[2] = v.z;
        dy[3] = a.x; dy[4] = a.y; dy[5] = a.z;
        dy[6] = qd[0]; dy[7] = qd[1]; dy[8] = qd[2]; dy[9] = qd[3];
        dy[10] = od.x; dy[11] = od.y; dy[12] = od.z;
        dy[13] = mass_dot;
    }

    fn solout(&mut self, t: f64, y: &BurnState, _dy: &BurnState) -> bool {
        self.last_visited.set((t, *y));
        let r = Vector3::new(y[0], y[1], y[2]);
        (self.should_stop)(t, &r)
    }
}

/// Integrate the fully-coupled burn ODE across as many legs as needed to
/// cover `duration_s`, switching central body at each SOI crossing exactly
/// like `trajectory_solver::propagator::propagate` does for the decoupled
/// (3DOF) path — reusing that function's own `pub` `resolve_central_body`/
/// `resolve_collision` primitives rather than re-deriving SOI-membership
/// logic here (Phase 13f follow-up).
/// Unlike `propagate()` this returns only the final state (no sampled
/// trajectory) — [`step_tick_with_burn`], the only caller, needs just the
/// tick's end state.
///
/// A burn tick that never leaves its starting central body's SOI (the
/// common case — burns are short) takes exactly one leg and reproduces the
/// previous single-shot integration bit-for-bit; a burn that genuinely
/// crosses an SOI boundary now gets a real central-body switch mid-burn
/// instead of an unphysical frozen-central-body error.
#[allow(clippy::too_many_arguments)]
fn propagate_coupled_burn(
    r0_ref: Vector3<f64>,
    v0_ref: Vector3<f64>,
    q0: Vector4<f64>,
    omega0: Vector3<f64>,
    mass0: f64,
    t0_abs_s: f64,
    duration_s: f64,
    inertia: Vector3<f64>,
    srp: &SrpTruthModel,
    bodies: &[PropagatorBody],
    reference_mu_m3s2: f64,
    burn: &BurnConfig,
    control_torque_body: Vector3<f64>,
    h_wheel_body: Vector3<f64>,
    sample_dt_s: f64,
    rtol: f64,
    atol: f64,
) -> (Vector3<f64>, Vector3<f64>, Vector4<f64>, Vector3<f64>, f64) {
    let mut t_abs = t0_abs_s;
    let mut t_remaining = duration_s.max(1e-6);
    let mut r_ref = r0_ref;
    let mut v_ref = v0_ref;
    let mut q = q0;
    let mut omega = omega0;
    let mut mass = mass0;
    let sample_dt_s = sample_dt_s.max(1e-6);
    let mut nudged_after_last_degenerate_leg = false;
    // Same safety cap as `propagate()` — see its own comment for why a
    // low-v-infinity boundary case can genuinely need many legs; a burn
    // tick is short, so this is a generous ceiling, not an expected count.
    let max_legs = (bodies.len() * 4 + 4).max(200);

    for _ in 0..max_legs {
        if t_remaining <= 1e-6 {
            break;
        }

        let central_index = resolve_central_body(&r_ref, bodies, t_abs);
        let mu_central = match central_index {
            Some(i) => bodies[i].mu_m3s2,
            None => reference_mu_m3s2,
        };
        let (central_pos0, central_vel0) = match central_index {
            Some(i) => (bodies[i].state_at)(t_abs),
            None => (Vector3::zeros(), Vector3::zeros()),
        };
        let r_local0 = r_ref - central_pos0;
        let v_local0 = v_ref - central_vel0;

        let bodies_for_check = bodies;
        let t0_for_check = t_abs;
        let central_for_check = central_index;
        let should_stop = move |t_local: f64, r_local: &Vector3<f64>| {
            let t_now = t0_for_check + t_local;
            let central_pos_now = match central_for_check {
                Some(i) => (bodies_for_check[i].state_at)(t_now).0,
                None => Vector3::zeros(),
            };
            let r_ref_now = *r_local + central_pos_now;
            let resolved = resolve_central_body(&r_ref_now, bodies_for_check, t_now);
            resolved != central_for_check || resolve_collision(&r_ref_now, bodies_for_check, t_now).is_some()
        };

        let y0 = BurnState::from_column_slice(&[
            r_local0.x, r_local0.y, r_local0.z,
            v_local0.x, v_local0.y, v_local0.z,
            q[0], q[1], q[2], q[3],
            omega.x, omega.y, omega.z,
            mass,
        ]);
        let last_visited = std::rc::Rc::new(std::cell::Cell::new((0.0, y0)));
        let leg_duration = t_remaining.max(1e-6);
        let ode = CoupledBurnLegOde {
            inertia,
            mu_central,
            bodies,
            central_index,
            t0_abs_s: t_abs,
            srp: srp.clone(),
            burn: *burn,
            control_torque_body,
            h_wheel_body,
            should_stop,
            last_visited: last_visited.clone(),
        };
        let mut stepper = Dopri5::from_param(
            ode, 0.0, leg_duration, leg_duration, y0, rtol, atol,
            0.9, 0.04, 0.2, 10.0, leg_duration, 0.0, 100_000, 1000,
            ode_solvers::dop_shared::OutputType::Sparse,
        );
        if let Err(e) = stepper.integrate() {
            eprintln!(
                "Warning: 6DOF propagator's coupled burn leg did not complete cleanly (requested \
                 duration {leg_duration:.3e} s, mu_central {mu_central:.3e} m^3/s^2): {e:?} — burn \
                 state may be inaccurate."
            );
        }

        // Same dense-vs-solout reconciliation `integrate_leg` uses: prefer
        // the dense-output last sample unless the solout-recorded true last
        // point is strictly further along (e.g. `should_stop` fired before
        // the first dense sample).
        let (t_solout, y_solout) = last_visited.get();
        let (t_final_local, y_final) = match (stepper.x_out().last(), stepper.y_out().last()) {
            (Some(&t_dense), Some(y_dense)) if t_dense >= t_solout - 1e-6 => (t_dense, *y_dense),
            _ => (t_solout, y_solout),
        };

        let r_final_local = Vector3::new(y_final[0], y_final[1], y_final[2]);
        let v_final_local = Vector3::new(y_final[3], y_final[4], y_final[5]);
        q = qnorm(&Vector4::new(y_final[6], y_final[7], y_final[8], y_final[9]));
        omega = Vector3::new(y_final[10], y_final[11], y_final[12]);
        mass = y_final[13].max(0.0);

        t_abs += t_final_local;
        t_remaining -= t_final_local;

        let (cp_now, cv_now) = match central_index {
            Some(i) => (bodies[i].state_at)(t_abs),
            None => (Vector3::zeros(), Vector3::zeros()),
        };
        r_ref = r_final_local + cp_now;
        v_ref = v_final_local + cv_now;

        // A collision stop is terminal, same as `propagate()`'s own rule —
        // no sensible "next leg" from inside a body's surface.
        if resolve_collision(&r_ref, bodies, t_abs).is_some() {
            break;
        }

        // Degenerate (zero-progress) leg recovery, mirroring `propagate()`'s
        // one-time nudge — rare for a short burn tick, but the same noisy-
        // boundary re-trigger is possible in principle near an SOI edge.
        if t_final_local <= 1e-9 {
            if !nudged_after_last_degenerate_leg {
                let nudge_s = (sample_dt_s * 1e-3).min(t_remaining.max(0.0)).max(1e-9);
                r_ref += v_ref * nudge_s;
                t_abs += nudge_s;
                t_remaining -= nudge_s;
                nudged_after_last_degenerate_leg = true;
                continue;
            }
            break;
        }
        nudged_after_last_degenerate_leg = false;
    }

    (r_ref, v_ref, q, omega, mass)
}

/// [`step_tick`], extended with an optional active burn (Phase 13f). When
/// `burn` is `None` this delegates to [`step_tick`] UNCHANGED (byte-for-byte
/// identical results — verified by regression test) — the decoupled path is
/// exact whenever there's no attitude-dependent force acting on translation,
/// per this module's doc comment. When `burn` is `Some`, translation,
/// rotation, and mass depletion are integrated together via
/// [`propagate_coupled_burn`]'s own SOI-switching leg loop, because the
/// thrust direction now genuinely depends on the evolving attitude.
#[allow(clippy::too_many_arguments)]
pub fn step_tick_with_burn(
    state: &SixDofState,
    tick_s: f64,
    sc: &SpacecraftProperties,
    bodies: &[PropagatorBody],
    reference_mu_m3s2: f64,
    burn: Option<&BurnConfig>,
    control_torque_body: Vector3<f64>,
    h_wheel_body: Vector3<f64>,
    rtol: f64,
    atol: f64,
) -> SixDofState {
    let Some(burn) = burn else {
        return step_tick(state, tick_s, sc, bodies, reference_mu_m3s2, control_torque_body, h_wheel_body, rtol, atol);
    };

    let tick_s = tick_s.max(1e-6);
    let (r_final, v_final, q_final, omega_final, mass_final) = propagate_coupled_burn(
        state.r_m, state.v_mps, state.q, state.omega_radps, state.mass_kg,
        state.t_s, tick_s,
        sc.inertia_diag_kgm2, &sc.srp, bodies, reference_mu_m3s2, burn,
        control_torque_body, h_wheel_body,
        tick_s, rtol, atol,
    );

    SixDofState {
        t_s: state.t_s + tick_s,
        r_m: r_final,
        v_mps: v_final,
        q: q_final,
        omega_radps: omega_final,
        wheel_speeds_radps: state.wheel_speeds_radps,
        mass_kg: mass_final,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cannonball_sc(inertia: Vector3<f64>) -> SpacecraftProperties {
        SpacecraftProperties {
            mass_kg: 1000.0,
            inertia_diag_kgm2: inertia,
            srp: SrpTruthModel::Cannonball { c_r: 1.4, area_m2: 4.0 },
            drag_area_m2: 4.0,
        }
    }

    fn state_at_1au() -> SixDofState {
        SixDofState {
            t_s: 0.0,
            r_m: Vector3::new(1.495_98e11, 0.0, 0.0),
            v_mps: Vector3::new(0.0, 29_780.0, 0.0),
            q: Vector4::new(1.0, 0.0, 0.0, 0.0),
            omega_radps: Vector3::new(0.001, -0.002, 0.0005),
            wheel_speeds_radps: [0.0; 4],
            mass_kg: 1000.0,
        }
    }

    /// A torque-free, spherically symmetric body (`I = kI_3`) has zero
    /// gravity-gradient torque by construction (`r̂ × I·r̂ = 0` whenever `I`
    /// is isotropic, since `I·r̂ ∥ r̂`) and zero SRP torque (cannonball is
    /// attitude-independent, §4.1). With no commanded torque either, `ω`
    /// must be exactly conserved — Euler's equations reduce to `İω̇ = -ω×Iω`,
    /// which is identically zero for isotropic `I`.
    #[test]
    fn symmetric_body_free_rotation_conserves_omega() {
        let sc = cannonball_sc(Vector3::new(100.0, 100.0, 100.0));
        let bodies: Vec<PropagatorBody> = Vec::new();
        let state = state_at_1au();
        let next = step_tick(
            &state, 100.0, &sc, &bodies, orbital_models::constants::MU_SUN,
            Vector3::zeros(), Vector3::zeros(), 1e-10, 1e-12,
        );
        assert!(
            (next.omega_radps - state.omega_radps).norm() < 1e-9,
            "omega should be conserved for a symmetric torque-free body: {:?} vs {:?}",
            next.omega_radps, state.omega_radps
        );
        assert!((next.q.norm() - 1.0).abs() < 1e-9, "quaternion should stay normalized");
    }

    /// An asymmetric body in low orbit around a real (Earth-mass) central
    /// body should show a real gravity-gradient torque, i.e. omega must
    /// change. Registers an actual `PropagatorBody` (Earth-like mu, a large
    /// SOI, fixed heliocentric position) so `resolve_central_body` picks it
    /// as central for BOTH translation and the gravity-gradient torque's
    /// `mu_central` -- using the Sun's own mu at a LEO-scale radius (an
    /// earlier version of this test's mistake) is wildly unphysical and
    /// makes the integrator fail outright, not a real regression.
    #[test]
    fn asymmetric_body_near_massive_center_feels_gravity_gradient_torque() {
        let sc = cannonball_sc(Vector3::new(100.0, 100.0, 250.0));
        const EARTH_POS: Vector3<f64> = Vector3::new(1.495_98e11, 0.0, 0.0);
        const EARTH_MU: f64 = 3.986_004_418e14;
        let state_at = |_t: f64| (EARTH_POS, Vector3::zeros());
        let bodies = vec![PropagatorBody {
            name: "Earth",
            mu_m3s2: EARTH_MU,
            soi_radius_m: Some(9.24e8), // real Earth SOI, comfortably covers LEO
            state_at: &state_at,
            central_fidelity: None,
            radius_m: Some(6.378e6),
        }];

        // Offset has BOTH x and z components so the local-vertical direction
        // is not aligned with any principal axis -- with I = diag(100,100,250),
        // a purely x- or z-aligned r_hat gives I*r_hat parallel to r_hat
        // (zero cross product, zero torque) by construction, which was this
        // test's original bug. A mixed direction is required to see a
        // nonzero (r_hat x I*r_hat) term.
        let mut state = state_at_1au();
        state.r_m = EARTH_POS + Vector3::new(6_500_000.0, 0.0, 2_500_000.0);
        state.v_mps = Vector3::new(0.0, 7_500.0, 0.0);
        state.omega_radps = Vector3::zeros();
        let next = step_tick(
            &state, 50.0, &sc, &bodies, orbital_models::constants::MU_SUN,
            Vector3::zeros(), Vector3::zeros(), 1e-10, 1e-12,
        );
        assert!(
            next.omega_radps.norm() > 1e-12,
            "expected a nonzero gravity-gradient-induced omega change, got {:?}",
            next.omega_radps
        );
    }

    /// Splitting a translation into several ticks must reproduce the same
    /// endpoint as one direct call to `trajectory_solver::propagate` over
    /// the whole duration -- the core "tick composability" regression this
    /// module depends on (Phase 13c's key claim: many ticks == one leg).
    #[test]
    fn tick_splitting_matches_direct_propagate_call() {
        let sc = cannonball_sc(Vector3::new(100.0, 100.0, 100.0));
        let bodies: Vec<PropagatorBody> = Vec::new();
        let r0 = Vector3::new(1.495_98e11, 0.0, 0.0);
        let v0 = Vector3::new(0.0, 29_780.0, 0.0);
        let total_s = 5_000.0;
        let n_ticks = 5;
        let tick_s = total_s / n_ticks as f64;

        let mut state = SixDofState {
            t_s: 0.0, r_m: r0, v_mps: v0,
            q: Vector4::new(1.0, 0.0, 0.0, 0.0), omega_radps: Vector3::zeros(),
            wheel_speeds_radps: [0.0; 4], mass_kg: 1000.0,
        };
        for _ in 0..n_ticks {
            state = step_tick(
                &state, tick_s, &sc, &bodies, orbital_models::constants::MU_SUN,
                Vector3::zeros(), Vector3::zeros(), 1e-12, 1e-14,
            );
        }

        let direct = propagate_translation(
            r0, v0, 0.0, total_s, orbital_models::constants::MU_SUN, &bodies, total_s, 1e-12, 1e-14,
        );
        let direct_last = direct.last().expect("direct propagate should return points");

        let r_err = (state.r_m - direct_last.r_m).norm();
        let v_err = (state.v_mps - direct_last.v_mps).norm();
        assert!(r_err < 1.0, "position mismatch between ticked and direct propagation: {r_err:.3e} m");
        assert!(v_err < 1e-4, "velocity mismatch between ticked and direct propagation: {v_err:.3e} m/s");
    }

    // ── Phase 13f: finite-burn propulsion ───────────────────────────────────

    /// `step_tick_with_burn(..., burn: None, ...)` must be byte-for-byte
    /// identical to `step_tick` -- the whole point of keeping the decoupled
    /// path as the default is that burn-free ticks pay zero cost/risk from
    /// this phase's new coupled integrator existing at all.
    #[test]
    fn step_tick_with_burn_none_matches_step_tick_exactly() {
        let sc = cannonball_sc(Vector3::new(100.0, 100.0, 250.0));
        let bodies: Vec<PropagatorBody> = Vec::new();
        let state = state_at_1au();

        let a = step_tick(&state, 50.0, &sc, &bodies, orbital_models::constants::MU_SUN, Vector3::zeros(), Vector3::zeros(), 1e-10, 1e-12);
        let b = step_tick_with_burn(&state, 50.0, &sc, &bodies, orbital_models::constants::MU_SUN, None, Vector3::zeros(), Vector3::zeros(), 1e-10, 1e-12);

        assert!((a.r_m - b.r_m).norm() < 1e-9);
        assert!((a.v_mps - b.v_mps).norm() < 1e-9);
        assert!((a.q - b.q).norm() < 1e-12);
        assert!((a.omega_radps - b.omega_radps).norm() < 1e-12);
        assert_eq!(a.mass_kg, b.mass_kg);
    }

    /// Mass depletion during a burn must match the Tsiolkovsky mass-flow
    /// rate exactly: mdot = -F/(Isp*g0), independent of the surrounding
    /// orbital dynamics (a deterministic bookkeeping check, not a dynamics
    /// check).
    #[test]
    fn burn_depletes_mass_at_the_tsiolkovsky_rate() {
        let sc = cannonball_sc(Vector3::new(100.0, 100.0, 100.0));
        let bodies: Vec<PropagatorBody> = Vec::new();
        let mut state = state_at_1au();
        state.omega_radps = Vector3::zeros();

        let burn = BurnConfig {
            thrust_n: 400.0,
            isp_s: 300.0,
            body_dir: Vector3::new(1.0, 0.0, 0.0),
            thrust_offset_body_m: Vector3::zeros(),
        };
        let tick_s = 10.0;
        let next = step_tick_with_burn(
            &state, tick_s, &sc, &bodies, orbital_models::constants::MU_SUN, Some(&burn),
            Vector3::zeros(), Vector3::zeros(), 1e-11, 1e-13,
        );

        let expected_mdot = -burn.thrust_n / (burn.isp_s * orbital_models::constants::G0);
        let expected_mass = state.mass_kg + expected_mdot * tick_s;
        assert!(
            (next.mass_kg - expected_mass).abs() < 1e-6,
            "mass depletion mismatch: got {:.6}, expected {:.6}", next.mass_kg, expected_mass
        );
    }

    /// A zero-offset (perfectly aligned) burn along body +x, fired from
    /// identity attitude, must produce a purely +x delta-v with NO torque
    /// (misalignment term is exactly zero) -- a real, if degenerate, sanity
    /// case that the thrust force and misalignment torque are wired
    /// correctly before checking the nonzero-offset case.
    #[test]
    fn aligned_burn_produces_no_misalignment_torque() {
        let sc = cannonball_sc(Vector3::new(100.0, 100.0, 100.0));
        let bodies: Vec<PropagatorBody> = Vec::new();
        let mut state = state_at_1au();
        state.omega_radps = Vector3::zeros();

        let burn = BurnConfig {
            thrust_n: 100.0,
            isp_s: 300.0,
            body_dir: Vector3::new(1.0, 0.0, 0.0),
            thrust_offset_body_m: Vector3::zeros(), // perfectly aligned
        };
        let next = step_tick_with_burn(
            &state, 5.0, &sc, &bodies, orbital_models::constants::MU_SUN, Some(&burn),
            Vector3::zeros(), Vector3::zeros(), 1e-11, 1e-13,
        );
        assert!(
            next.omega_radps.norm() < 1e-9,
            "aligned thrust through the CoM should produce no rotation: {:?}", next.omega_radps
        );
    }

    /// A thrust application point offset from the CoM MUST produce a real
    /// misalignment torque -- confirms the attitude/thrust coupling this
    /// whole coupled path exists for is actually wired in, not just present
    /// in the doc comments.
    #[test]
    fn offset_burn_produces_real_misalignment_torque() {
        let sc = cannonball_sc(Vector3::new(100.0, 100.0, 100.0));
        let bodies: Vec<PropagatorBody> = Vec::new();
        let mut state = state_at_1au();
        state.omega_radps = Vector3::zeros();

        let burn = BurnConfig {
            thrust_n: 100.0,
            isp_s: 300.0,
            body_dir: Vector3::new(1.0, 0.0, 0.0),
            thrust_offset_body_m: Vector3::new(0.0, 0.5, 0.0), // 0.5 m off-axis in y
        };
        let next = step_tick_with_burn(
            &state, 5.0, &sc, &bodies, orbital_models::constants::MU_SUN, Some(&burn),
            Vector3::zeros(), Vector3::zeros(), 1e-11, 1e-13,
        );
        assert!(
            next.omega_radps.norm() > 1e-6,
            "offset thrust should produce a real misalignment-torque-induced spin-up: {:?}", next.omega_radps
        );
        // tau = offset x F = (0,0.5,0) x (100,0,0) = (0*0 - 0*0, 0*100 - 0.5*0, 0.5*0 - 0*100)...
        // compute directly: (0,0.5,0) x (100,0,0) = (0.5*0-0*0, 0*100-0*0, 0*0-0.5*100) = (0,0,-50)
        // so the induced omega should be dominantly about -z.
        assert!(
            next.omega_radps.z < -1e-7 && next.omega_radps.x.abs() < next.omega_radps.z.abs()
                && next.omega_radps.y.abs() < next.omega_radps.z.abs(),
            "misalignment torque direction should dominantly spin the body about -z: {:?}", next.omega_radps
        );
    }

    /// Thrust direction must rotate WITH the body -- a burn commanded along
    /// body +x from a spacecraft already rotated 90 degrees about z should
    /// accelerate along inertial +y, not inertial +x. This is the actual
    /// attitude->translation coupling claim this whole module exists to
    /// deliver; confirming it end-to-end (not just that a torque exists).
    #[test]
    fn thrust_direction_follows_the_rotated_body_frame() {
        let sc = cannonball_sc(Vector3::new(100.0, 100.0, 100.0));
        let bodies: Vec<PropagatorBody> = Vec::new();
        let mut state = state_at_1au();
        state.v_mps = Vector3::zeros(); // isolate the thrust-induced velocity change
        state.omega_radps = Vector3::zeros();
        // 90 deg rotation about z: body +x now points along inertial +y.
        let half = (std::f64::consts::FRAC_PI_4).sin();
        let cos_half = (std::f64::consts::FRAC_PI_4).cos();
        state.q = Vector4::new(cos_half, 0.0, 0.0, half);

        let burn = BurnConfig {
            thrust_n: 5000.0,
            isp_s: 300.0,
            body_dir: Vector3::new(1.0, 0.0, 0.0),
            thrust_offset_body_m: Vector3::zeros(),
        };
        let next = step_tick_with_burn(
            &state, 1.0, &sc, &bodies, orbital_models::constants::MU_SUN, Some(&burn),
            Vector3::zeros(), Vector3::zeros(), 1e-11, 1e-13,
        );
        let dv = next.v_mps - state.v_mps;
        assert!(dv.y > 0.0, "thrust along body +x from a 90deg-about-z attitude should push +y: {:?}", dv);
        assert!(dv.x.abs() < 0.01 * dv.y.abs(), "x-component should be negligible: {:?}", dv);
    }

    // ── Follow-up: coupled-burn leg loop ──

    /// A burn tick that never leaves its starting central body's SOI (the
    /// common case) should give the identical result whether or not other
    /// SOI-candidate bodies exist elsewhere in the system — confirms the new
    /// per-leg `resolve_central_body`/`resolve_collision` checks don't
    /// perturb a burn that never actually crosses a boundary. A body whose
    /// SOI never contains the spacecraft is registered specifically so this
    /// exercises the leg loop's real membership-checking machinery (not just
    /// the trivially-true empty-`bodies` case already covered by every other
    /// burn test in this module).
    #[test]
    fn coupled_burn_matches_empty_bodies_when_soi_never_contains_spacecraft() {
        let sc = cannonball_sc(Vector3::new(100.0, 100.0, 100.0));
        let far_body_pos = Vector3::new(-1.0e11, 5.0e11, 0.0);
        let state_at = move |_t: f64| (far_body_pos, Vector3::zeros());
        let bodies = vec![PropagatorBody {
            name: "FarBody",
            mu_m3s2: 3.986e14,
            soi_radius_m: Some(1.0e8),
            state_at: &state_at,
            central_fidelity: None,
            radius_m: None,
        }];
        let mut state = state_at_1au();
        state.omega_radps = Vector3::zeros();

        let burn = BurnConfig {
            thrust_n: 400.0,
            isp_s: 300.0,
            body_dir: Vector3::new(1.0, 0.0, 0.0),
            thrust_offset_body_m: Vector3::zeros(),
        };
        let tick_s = 10.0;
        let with_far_body = step_tick_with_burn(
            &state, tick_s, &sc, &bodies, orbital_models::constants::MU_SUN, Some(&burn),
            Vector3::zeros(), Vector3::zeros(), 1e-11, 1e-13,
        );
        let empty_bodies: Vec<PropagatorBody> = Vec::new();
        let without_far_body = step_tick_with_burn(
            &state, tick_s, &sc, &empty_bodies, orbital_models::constants::MU_SUN, Some(&burn),
            Vector3::zeros(), Vector3::zeros(), 1e-11, 1e-13,
        );

        let r_err = (with_far_body.r_m - without_far_body.r_m).norm();
        let v_err = (with_far_body.v_mps - without_far_body.v_mps).norm();
        assert!(r_err < 1e-3, "a non-contained SOI candidate should not change the translational result: {r_err:.3e} m");
        assert!(v_err < 1e-6, "a non-contained SOI candidate should not change the translational result: {v_err:.3e} m/s");
        assert!(
            (with_far_body.mass_kg - without_far_body.mass_kg).abs() < 1e-9,
            "mass depletion should be unaffected: {} vs {}", with_far_body.mass_kg, without_far_body.mass_kg
        );
    }

    /// A burn tick that genuinely crosses an SOI boundary must show a real
    /// central-body switch mid-burn — the actual gap previously
    /// flagged (today's `CoupledBurnOde` resolved the central body once,
    /// frozen for the whole tick). Mirrors
    /// `trajectory_solver::propagator::tests::switches_central_body_on_soi_entry`'s
    /// toy star+planet setup, run in reverse (starting just inside the
    /// planet's SOI, escaping outward) so the crossing happens promptly and
    /// the planet's own gravity over that short crossing window stays a
    /// small perturbation, not a dominant, hard-to-predict effect.
    #[test]
    fn coupled_burn_switches_central_body_mid_burn_on_soi_exit() {
        const MU_STAR: f64 = 1.327e20; // Sun-like
        const MU_PLANET: f64 = 3.986e14; // Earth-like
        let planet_a_m = 1.0e11;
        let mass_ratio = MU_PLANET / MU_STAR;
        let soi = trajectory_solver::laplace_soi_radius_m(planet_a_m, mass_ratio);

        let planet_pos = Vector3::new(planet_a_m, 0.0, 0.0);
        let state_at = move |_t: f64| (planet_pos, Vector3::zeros());
        let bodies = vec![PropagatorBody {
            name: "Planet",
            mu_m3s2: MU_PLANET,
            soi_radius_m: Some(soi),
            state_at: &state_at,
            central_fidelity: None,
            radius_m: None,
        }];

        let sc = cannonball_sc(Vector3::new(100.0, 100.0, 100.0));
        let mut state = SixDofState {
            t_s: 0.0,
            r_m: planet_pos + Vector3::new(soi * 0.98, 0.0, 0.0), // just inside the SOI
            v_mps: Vector3::new(2_000.0, 0.0, 0.0), // heading outward
            q: Vector4::new(1.0, 0.0, 0.0, 0.0),
            omega_radps: Vector3::zeros(),
            wheel_speeds_radps: [0.0; 4],
            mass_kg: 1000.0,
        };
        assert_eq!(
            resolve_central_body(&state.r_m, &bodies, state.t_s), Some(0),
            "test setup should start inside the planet's SOI"
        );

        let burn = BurnConfig {
            thrust_n: 50.0,
            isp_s: 300.0,
            body_dir: Vector3::new(1.0, 0.0, 0.0),
            thrust_offset_body_m: Vector3::zeros(),
        };
        // At 2000 m/s, crossing the remaining 2% of the SOI (~1.24e7 m for
        // this toy system) takes ~6,200 s; 8,000 s gives comfortable margin
        // while the outward-motion deceleration from the planet's own
        // gravity at this range (~1e-3 m/s^2) stays a small perturbation.
        let tick_s = 8_000.0;
        state.omega_radps = Vector3::zeros();
        let next = step_tick_with_burn(
            &state, tick_s, &sc, &bodies, MU_STAR, Some(&burn),
            Vector3::zeros(), Vector3::zeros(), 1e-9, 1e-11,
        );

        assert_eq!(
            resolve_central_body(&next.r_m, &bodies, next.t_s), None,
            "a burn tick that genuinely crosses the planet's SOI boundary should end heliocentric \
             (outside every candidate's SOI), confirming the leg loop switched central body mid-burn \
             instead of integrating the whole tick under a frozen central body"
        );
    }
}
