//! POST /api/design/slew-test — cheap synchronous canned attitude-step test
//!
//!
//! Reuses exactly the same closed-loop machinery `sixdof_control_demo`
//! (`src/bin/sixdof_control_demo.rs`) exercises by hand — quaternion PD +
//! mode-dependent control allocation, Phase 13d/13e — driven by the REAL
//! vehicle/actuator/gain values derived from the submitted `MissionConfig`
//! (`crate::simulate::build_spacecraft_properties`/`wheel_cluster_from_hardware`/
//! `rcs_from_hardware`/`pd_gains_from_cfg`, the same helpers `run_streaming`
//! itself uses). This is the fast inner-loop tool for tuning control gains
//! and checking actuator sizing before committing to a full mission
//! simulation — see `docs/MP/MANUAL.md` §10 for the governing control
//! law/allocation math.
//!
//! The reference scenario (a circular Earth orbit at `ORBIT_ALT_M`) is
//! fixed/"canned", not user-configurable — only the vehicle (via `config`),
//! gains, control mode, and initial pointing error vary. This keeps the
//! endpoint a pure actuator/gain check, not a mission-specific trajectory
//! tool (that's `/api/simulate`).
//!
//! Request: `{ config, initial_error_deg?, axis?, control_mode?, kp?, kd?,
//! tick_s?, duration_s?, settle_threshold_deg?, initial_state?,
//! initial_omega_radps? }`. `kp`/`kd` fall back to the config's own `[gnc]`
//! values (`pd_gains_from_cfg`) when omitted, so leaving them out tests the
//! gains the mission would actually fly with. `duration_s`/`tick_s` are
//! clamped server-side to keep this endpoint's cost bounded and synchronous
//! — no job/polling, unlike `/api/simulate`. `initial_state` (round-2 ask
//! #4) chains calls into a continuous animation — see
//! `SlewTestRequest::initial_state`'s own doc comment for the exact
//! semantics, including how it changes what `initial_error_deg`/`axis`
//! mean. `initial_omega_radps` seeds a FRESH
//! (non-chained) run with a synthetic initial spin to probe control
//! authority on a specific axis — see that field's own doc comment.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use nalgebra::{Vector3, Vector4};
use serde::Deserialize;
use serde_json::json;
use std::time::Instant;

use crate::config::MissionConfig;
use sim_engine::reference_guidance::desired_quaternion_cruise;
use sim_engine::{
    allocate, net_body_torque, step_tick, ControlMode, CruisePointingMode,
    PdGains, SixDofState,
};
use trajectory_solver::PropagatorBody;

const EARTH_MU_M3S2: f64 = 3.986_004_418e14;
const EARTH_R_M: f64 = 6.378e6;
const ORBIT_ALT_M: f64 = 700_000.0;
const EARTH_SOI_RADIUS_M: f64 = 9.24e8;
const EARTH_HELIOCENTRIC_X_M: f64 = 1.495_98e11;

const MIN_TICK_S: f64 = 0.1;
const MAX_TICK_S: f64 = 60.0;
/// 2 h cap on simulated duration — keeps this endpoint's cost bounded so it
/// can stay a plain synchronous request (no job/polling, unlike
/// `/api/simulate`/`/api/optimize`).
const MAX_DURATION_S: f64 = 7_200.0;
/// Decimation cap for the returned time series (plotting-sized, not a full
/// per-tick dump).
const MAX_SAMPLES: usize = 300;

/// Lets a caller chain slew-test calls into a
/// continuous "chase a moving target" animation — see `SlewTestRequest::
/// initial_state`'s own doc comment for the exact semantics.
#[derive(Deserialize)]
pub struct SlewTestInitialStateRequest {
    /// [w, x, y, z] — re-normalized server-side, same convention as every
    /// other quaternion field in this API.
    q: [f64; 4],
    omega_radps: [f64; 3],
}

#[derive(Deserialize)]
pub struct SlewTestRequest {
    config: MissionConfig,
    initial_error_deg: Option<f64>,
    axis: Option<[f64; 3]>,
    /// "WheelsPrimary" (default), "ThrustersPrimary", or "ThrustersOnly"
    /// —
    /// see `sim_engine::ControlMode`.
    control_mode: Option<String>,
    kp: Option<f64>,
    kd: Option<f64>,
    tick_s: Option<f64>,
    duration_s: Option<f64>,
    settle_threshold_deg: Option<f64>,
    /// Chained runs:
    /// seed this call's STARTING attitude/rate from a previous call's real
    /// ending state instead of the canned `q0` derivation — this is what
    /// makes a chained sequence of calls a genuinely continuous animation
    /// (the body's real motion carries over) rather than a jump-cut back to
    /// a fresh nominal start every call.
    ///
    /// **Semantic switch this triggers, worth being explicit about**: when
    /// `initial_state` is omitted (the original, unchanged behavior),
    /// `initial_error_deg`/`axis` define the STARTING attitude as an offset
    /// FROM the fixed commanded target (`q0 = dq * q_cmd`) — the target
    /// itself never moves. When `initial_state` IS given, `state.q`/
    /// `state.omega_radps` are seeded directly from it instead, so
    /// `initial_error_deg`/`axis` would have nothing left to define if they
    /// kept that same role — instead they're reinterpreted to offset the
    /// COMMANDED TARGET itself (`q_cmd = dq * <fixed nominal>`), which is
    /// what actually produces a "moving target" chase across chained calls
    /// (a fixed target the vehicle has already reached would otherwise make
    /// every subsequent call start at ~0 error, which isn't an interesting
    /// animation). Concretely: run one call, take its LAST sample's real
    /// `q`/`omega_radps` from the response, feed those as the next call's
    /// `initial_state` alongside a freshly chosen `axis`/`initial_error_deg`
    /// for that call's new target, repeat.
    initial_state: Option<SlewTestInitialStateRequest>,
    /// Initial spin:
    /// body-frame angular rate
    /// [rad/s] to start a FRESH (non-chained) run with, one independent
    /// value per axis — default `[0,0,0]` (the original behavior). A step
    /// response from rest can't surface a control gap on an axis whose
    /// initial ERROR happens to start near zero; a real, persistent initial
    /// SPIN on a specific axis is what actually probes whether the
    /// allocation has authority there (motivating case: a rank-deficient
    /// thruster layout where roll about the shared thrust direction is
    /// structurally uncontrollable, but a from-rest test never excites it).
    /// Only applied when `initial_state` is NOT set — when it IS set, its
    /// own real `omega_radps` is genuine continuity data from a previous
    /// call and takes priority outright (not summed with this field, to
    /// avoid the two silently interacting).
    initial_omega_radps: Option<[f64; 3]>,
}

pub async fn slew_test(
    body: Result<Json<SlewTestRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let req = match body {
        Ok(Json(r)) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("JSON parse error: {e}") })),
            )
                .into_response();
        }
    };

    let control_mode_label = req.control_mode.clone().unwrap_or_else(|| "WheelsPrimary".to_string());
    let control_mode = match control_mode_label.as_str() {
        "WheelsPrimary" => ControlMode::WheelsPrimary,
        "ThrustersPrimary" => ControlMode::ThrustersPrimary,
        // A true RCS-only isolation
        // mode -- wheels contribute zero torque/momentum, full stop. See
        // `ControlMode::ThrustersOnly`'s own doc comment for why neither
        // existing mode actually isolates RCS.
        "ThrustersOnly" => ControlMode::ThrustersOnly,
        other => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({
                    "error": format!(
                        "unknown control_mode '{other}' -- expected 'WheelsPrimary', \
                         'ThrustersPrimary', or 'ThrustersOnly'"
                    )
                })),
            )
                .into_response();
        }
    };

    let initial_error_deg = req.initial_error_deg.unwrap_or(90.0);
    if !(0.0..=180.0).contains(&initial_error_deg) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": "initial_error_deg must be within [0, 180]" })),
        )
            .into_response();
    }

    let axis = {
        let a = req.axis.unwrap_or([0.0, 0.0, 1.0]);
        let v = Vector3::new(a[0], a[1], a[2]);
        let n = v.norm();
        if n > 1e-9 { v / n } else { Vector3::new(0.0, 0.0, 1.0) }
    };

    let tick_s = req.tick_s.unwrap_or(5.0).clamp(MIN_TICK_S, MAX_TICK_S);
    let duration_s = req.duration_s.unwrap_or(1800.0).clamp(tick_s, MAX_DURATION_S);
    let settle_threshold_deg = req.settle_threshold_deg.unwrap_or(0.5).max(0.0);

    let sc = crate::simulate::build_spacecraft_properties(&req.config);
    let wheel_cluster = crate::simulate::wheel_cluster_from_hardware(&req.config);
    let (rcs_thrusters, _) = crate::simulate::rcs_from_hardware(&req.config);

    // Three-layer attitude control: the slew test flies the
    // SAME registry the cruise loop does, so a `ThrustersPrimary`/
    // `ThrustersOnly` test exercises the RCS-authority-derived thruster law
    // (previously every mode ran the wheel-sized PD). Explicit `kp`/`kd`
    // still force a plain PD override for hand-tuning. The activity is
    // `Slew` (a large-angle reorientation is what this endpoint tests).
    // Single-tick test: no mode-scheduled burn tick here, both classes at
    // the test's own tick.
    let controls = crate::attitude_tuning::AttitudeControlSet::from_cfg(&req.config, &sc, &wheel_cluster, &rcs_thrusters, tick_s, tick_s);
    let law = match (req.kp, req.kd) {
        (None, None) => controls.law_for(control_mode, sim_engine::Activity::Slew, sc.mass_kg),
        (kp, kd) => {
            let base = controls.law_for(control_mode, sim_engine::Activity::Slew, sc.mass_kg);
            let (bkp, bkd) = base.pd_gains().unwrap_or((0.0, 0.0));
            sim_engine::AttitudeLaw::Pd(PdGains {
                kp: kp.unwrap_or(bkp),
                kd: kd.unwrap_or(bkd),
                pointing_db_rad: req.config.gnc.pointing_deadband_rad.unwrap_or(0.0),
                rate_db_rads: req.config.gnc.rate_deadband_radps.unwrap_or(0.0),
            })
        }
    };
    let (gains_kp, gains_kd) = law.pd_gains().map(|(p, d)| (Some(p), Some(d))).unwrap_or((None, None));
    let mut controller = sim_engine::AttitudeController::new(law);

    let earth_pos = Vector3::new(EARTH_HELIOCENTRIC_X_M, 0.0, 0.0);
    let earth_state_at = move |_t: f64| (earth_pos, Vector3::zeros());
    let bodies = vec![PropagatorBody {
        name: "Earth",
        mu_m3s2: EARTH_MU_M3S2,
        soi_radius_m: Some(EARTH_SOI_RADIUS_M),
        state_at: &earth_state_at,
        central_fidelity: None,
        radius_m: Some(EARTH_R_M),
    }];
    let r_orbit = EARTH_R_M + ORBIT_ALT_M;
    let v_circ = (EARTH_MU_M3S2 / r_orbit).sqrt();
    let r0 = earth_pos + Vector3::new(r_orbit, 0.0, 0.0);

    // Commanded attitude: fixed inertial direction (the nominal reference
    // target for this canned scenario).
    let target_dir = Vector3::new(0.0, 1.0, 0.0);
    let pointing_mode = CruisePointingMode::BurnAttitude { thrust_dir_inertial: target_dir };
    let q_cmd_nominal = desired_quaternion_cruise(pointing_mode, &r0, &Vector3::zeros());

    let half = initial_error_deg.to_radians() / 2.0;
    let dq = Vector4::new(half.cos(), axis.x * half.sin(), axis.y * half.sin(), axis.z * half.sin());

    // See `SlewTestRequest::initial_state`'s doc comment for the full
    // rationale behind this branch.
    let (q_cmd, q0) = match &req.initial_state {
        None => {
            // Original behavior: q0 starts `initial_error_deg` away from
            // the fixed nominal target, via a half-angle offset composed
            // onto q_cmd_nominal. Target never moves across calls.
            let q0 = orbital_models::attitude::quat_multiply(&dq, &q_cmd_nominal);
            (q_cmd_nominal, q0)
        }
        Some(seed) => {
            // Continuity mode: the caller's real ending state IS q0 (no
            // offset applied to it). `axis`/`initial_error_deg` instead
            // offset the TARGET away from the fixed nominal, so a chained
            // sequence of calls can present a genuinely different target
            // each time while the vehicle's motion carries over unbroken.
            let q_cmd = orbital_models::attitude::quat_multiply(&dq, &q_cmd_nominal);
            let qv = Vector4::new(seed.q[0], seed.q[1], seed.q[2], seed.q[3]);
            let n = qv.norm();
            let q0 = if n > 1e-9 { qv / n } else { q_cmd_nominal };
            (q_cmd, q0)
        }
    };

    // `initial_state`'s real omega_radps (genuine continuity data) takes
    // priority outright; otherwise a fresh run's `initial_omega_radps`
    // (a synthetic disturbance to probe
    // control authority on a specific axis, not continuity data), defaulting
    // to zero -- the original behavior.
    let omega0 = match &req.initial_state {
        Some(seed) => Vector3::new(seed.omega_radps[0], seed.omega_radps[1], seed.omega_radps[2]),
        None => {
            let w = req.initial_omega_radps.unwrap_or([0.0, 0.0, 0.0]);
            Vector3::new(w[0], w[1], w[2])
        }
    };

    let mut state = SixDofState {
        t_s: 0.0,
        r_m: r0,
        v_mps: Vector3::new(0.0, v_circ, 0.0),
        q: q0,
        omega_radps: omega0,
        wheel_speeds_radps: [0.0; 4],
        mass_kg: sc.mass_kg,
    };

    let n_ticks = (duration_s / tick_s).ceil() as usize;
    let sample_stride = (n_ticks / MAX_SAMPLES).max(1);

    let mut samples = Vec::new();
    let mut max_wheel_momentum_nms = 0.0_f64;
    let mut rcs_propellant_kg_used = 0.0_f64;
    let mut settling_time_s: Option<f64> = None;
    let mut peak_error_after_settle_deg = 0.0_f64;

    let wall_start = Instant::now();
    for i in 0..n_ticks {
        let tau_cmd = controller.command_torque(&state.q, &q_cmd, &state.omega_radps, state.t_s, tick_s);
        let alloc = allocate(
            control_mode, tau_cmd, &wheel_cluster, &state.wheel_speeds_radps,
            &rcs_thrusters, tick_s,
            // A short slew test never runs long enough for desaturation to
            // matter -- no momentum-management law is exercised here.
            sim_engine::MomentumManagementLaw::None,
        );
        let control_torque_body = net_body_torque(&wheel_cluster, &alloc);
        rcs_propellant_kg_used += alloc.rcs_propellant_kg;

        let err_deg = pointing_error_deg(&state.q, &q_cmd);
        let h_wheel = wheel_cluster.total_momentum(&state.wheel_speeds_radps);
        max_wheel_momentum_nms = max_wheel_momentum_nms.max(h_wheel.norm());

        if settling_time_s.is_none() && err_deg <= settle_threshold_deg {
            settling_time_s = Some(state.t_s);
        }
        if settling_time_s.is_some() {
            peak_error_after_settle_deg = peak_error_after_settle_deg.max(err_deg);
        }

        if i % sample_stride == 0 || i == n_ticks - 1 {
            samples.push(json!({
                "t_s": state.t_s,
                "error_deg": err_deg,
                "wheel_momentum_nms": h_wheel.norm(),
                "rcs_duty_cycle": alloc.rcs_duty_cycle,
                // Real attitude samples:
                // real attitude/rate at this sample, for a non-approximated
                // attitude-replay animation. [w, x, y, z] Hamilton convention,
                // same as every other quaternion field in this API.
                "q": [state.q[0], state.q[1], state.q[2], state.q[3]],
                "omega_radps": [state.omega_radps.x, state.omega_radps.y, state.omega_radps.z],
                // Which physical thruster fired
                // this tick, indexed identically to rcs_from_hardware()'s
                // thruster list -- see AllocationOutput::rcs_thruster_duty_cycles.
                "thruster_duty_cycles": alloc.rcs_thruster_duty_cycles.clone(),
            }));
        }

        let mut new_speeds = state.wheel_speeds_radps;
        let speed_dots = wheel_cluster.speed_dots(&alloc.wheel_motor_torque_nm);
        for k in 0..4 {
            // Real motor torque saturates at the wheel's rated max speed —
            // see cruise.rs's identical clamp for the real runaway this
            // prevents (found via cruise_commander_demo).
            new_speeds[k] = (new_speeds[k] + speed_dots[k] * tick_s).clamp(-wheel_cluster.max_speed, wheel_cluster.max_speed);
        }

        state = step_tick(
            &state, tick_s, &sc, &bodies, orbital_models::constants::MU_SUN,
            control_torque_body, h_wheel, 1e-10, 1e-12,
        );
        state.wheel_speeds_radps = new_speeds;
    }
    let wall_clock_ms = wall_start.elapsed().as_secs_f64() * 1e3;

    let final_error_deg = pointing_error_deg(&state.q, &q_cmd);
    let final_wheel_momentum_nms = wheel_cluster.total_momentum(&state.wheel_speeds_radps).norm();
    // How far above the settle threshold the error climbed again after
    // first reaching it -- 0.0 if it stayed under (or never settled at all,
    // reported separately via `settling_time_s: null`).
    let overshoot_deg = settling_time_s.map(|_| (peak_error_after_settle_deg - settle_threshold_deg).max(0.0));

    Json(json!({
        "control_mode": control_mode_label,
        // Three-layer attitude control: which layer-1 law
        // this test flew (`Pd`/`Pid`/`PhasePlane`) and its effective PD
        // gains (null for phase-plane).
        "controller_law": law.label(),
        "kp": gains_kp,
        "kd": gains_kd,
        "tick_s": tick_s,
        "duration_s": duration_s,
        "n_ticks": n_ticks,
        "initial_error_deg": initial_error_deg,
        "settle_threshold_deg": settle_threshold_deg,
        "final_error_deg": final_error_deg,
        "settling_time_s": settling_time_s,
        "overshoot_deg": overshoot_deg,
        "final_wheel_momentum_nms": final_wheel_momentum_nms,
        "max_wheel_momentum_nms": max_wheel_momentum_nms,
        "rcs_propellant_kg_used": rcs_propellant_kg_used,
        "wall_clock_ms": wall_clock_ms,
        // The fixed target attitude every
        // sample's error_deg is measured against -- this is a canned FIXED
        // step test (see this module's own doc comment), so q_cmd is
        // constant for the whole run and belongs on the result, not
        // per-sample. [w, x, y, z], same convention as SlewTestSample.q.
        "q_command": [q_cmd[0], q_cmd[1], q_cmd[2], q_cmd[3]],
        "samples": samples,
    }))
    .into_response()
}

fn pointing_error_deg(q_cur: &Vector4<f64>, q_cmd: &Vector4<f64>) -> f64 {
    let dot = q_cur.dot(q_cmd).clamp(-1.0, 1.0).abs();
    2.0 * dot.acos().to_degrees()
}
