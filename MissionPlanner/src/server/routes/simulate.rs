//! POST /api/simulate                  — start an async simulation job, returns `{job_id}`
//! GET  /api/simulate/:job_id/status   — `{running, step, total_steps, error, cancelled}`
//! WS   /api/simulate/:job_id/stream   — live feed, shape depends on job kind (see below)
//! GET  /api/simulate/:job_id/steps    — full step-message history so far (partial if still running)
//! GET  /api/simulate/:job_id/result   — final result once done (shape depends on job kind)
//! POST /api/simulate/:job_id/cancel   — request early stop (Phase 9k task 4)
//!
//! Jobs run on `spawn_blocking` (the sim loop is synchronous/CPU-bound, same
//! pattern as `POST /api/design/trajectory`). The registry is an in-memory
//! process-wide singleton (`OnceLock`, not Axum `State` — the rest of the
//! router has no shared state, so adding one just for this would mean
//! threading a `State` type through every other handler). Jobs are lost on
//! server restart and never evicted — both fine for a local dev tool, not for
//! a long-running deployment.
//!
//! `total_steps` in the status response is always `null` in this v1 — an
//! accurate estimate would require duplicating `simulate.rs`'s phase-duration
//! construction just to predict step count ahead of time. The frontend should
//! treat `null` as "indeterminate progress" (step counter, not a percentage).
//!
//! The WebSocket stream ends with a literal `"__DONE__"` text frame (not a
//! channel close) — explicit sentinel because the broadcast sender is kept
//! alive in the job registry for the process lifetime, so relying on
//! channel-closed semantics would never fire. `handle_stream` checks job
//! state before subscribing and sends the sentinel immediately if the job
//! already finished (broadcast channels don't replay history to late
//! subscribers, and fast simulations routinely finish before a client's WS
//! handshake completes) — there's a narrow, unfixed TOCTOU race between that
//! check and `subscribe()` if the job finishes in between; acceptable for a
//! local dev tool, not for a high-concurrency deployment.
//!
//! `/steps` exists because that race is common enough to matter in practice:
//! this backend runs fast enough that a sim can finish before a client's WS
//! handshake completes, and the late-subscribe path above then delivers
//! nothing but "__DONE__" — real step data with nowhere to go. `/steps`
//! retrieves it after the fact instead of requiring the client to "catch"
//! the live stream.
//!
//! ## Two job kinds, one endpoint family
//!
//! `POST /api/simulate` dispatches on `cfg.simulation.monte_carlo_runs`:
//!   - `== 0` (default, unchanged from before this change): single-run job.
//!     `/stream` carries `SimStepMsg` per measurement step; `/result` returns
//!     `SimResult`. Identical behavior to before — this branch is untouched.
//! - `> 0`: Monte Carlo job. Per the design notes hard rule, this NEVER gets the
//!     dense/decimated per-step stream a single run gets — `/stream` instead
//!     carries one `McRunMsg` per *completed run* (not per truth step), with a
//!     sparse trajectory only on a stride-selected subset of runs (mirrors
//!     `design.rs`'s `MC_TRAJ_SAMPLE_COUNT` narrowing-stage Monte Carlo
//!     convention). `/result` returns `McSummaryResult` instead of `SimResult`.
//!     `/steps` likewise replays the run-completion messages rather than
//!     per-truth-step ones for this job kind.
//!
//! `JobState`/`Job` are generic over which kind of payload they carry
//! (`JobPayload::Sim`/`JobPayload::Mc`) so both kinds share one registry, one
//! `/status` shape, and one WS sentinel-handling path rather than duplicating
//! the whole module for Monte Carlo. The streamed message itself is plain
//! JSON text either way (`SimStepMsg` or `McRunMsg`); the client distinguishes
//! by whichever request configuration it sent (it already knows whether it
//! posted a `monte_carlo_runs > 0` config).
//!
//! ## Cancellation (Phase 9k task 4)
//!
//! Compute runs in `spawn_blocking`, so `JoinHandle::abort()` would not stop
//! it — a real `Arc<AtomicBool>` flag on `Job`, polled by the running
//! compute, is used instead. For the single-run path (`monte_carlo_runs ==
//! 0`), `run_streaming`'s `on_step` now returns `bool` (`false` = stop); the
//! truth-integration loop in `run_flyby`/`run_landing`/`run_orbit_chain`
//! (`simulate.rs`) checks it every truth step and breaks early, so a cancel
//! request takes effect within one `dt_truth_s` step. For the Monte Carlo
//! path (`monte_carlo_runs > 0`), this is a **known, documented gap**: all
//! `n` runs are dispatched onto OS threads via `thread::scope` up front and
//! joined together (see `run_monte_carlo_streaming`), so there is no
//! per-run checkpoint to interrupt mid-batch without restructuring that
//! function — cancelling an MC job sets the flag and `/status` reports it,
//! but the already-dispatched batch runs to completion regardless.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use tokio::sync::broadcast;

use sim_engine::StepTelemetry;

use crate::config::{check_config, MissionConfig};
use crate::cruise::{
    run_cruise_mc_streaming, run_cruise_streaming, CruiseMcRunMsg, CruiseMcSummaryResult, CruiseResult,
    CruiseStepMsg, CruiseTickRow,
};
use crate::simulate::{run_monte_carlo_streaming, run_streaming, McRunMsg, McSummaryResult, SimResult, SimStepMsg};

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

/// Which kind of result/step-history a job carries — see the module doc
/// comment's "Two job kinds" section (now four — `Cruise`/`CruiseMc` are
/// Phase 5.2 and its follow-up). `Cruise`/`CruiseMc` are
/// selected when the request's `MissionConfig.cruise_seed` is set —
/// dispatched ahead of the plain `monte_carlo_runs` check in `start()`,
/// since a cruise-seeded run is a different mission-loop entirely (Phase
/// 13's translation+attitude composition, not the body-centric
/// proximity-ops path `Sim`/`Mc` both use). `CruiseMc` disperses vehicle
/// mass/reflectivity across `monte_carlo_runs` runs (see `cruise::
/// run_cruise_mc_streaming`'s doc comment for the first-cut dispersion
/// scope) rather than the initial-state dispersion `Mc` uses.
enum JobPayload {
    Sim { steps: Mutex<Vec<SimStepMsg>> },
    Mc { runs: Mutex<Vec<McRunMsg>> },
    CruiseMc { runs: Mutex<Vec<CruiseMcRunMsg>> },
    Cruise { steps: Mutex<Vec<CruiseStepMsg>> },
}

enum JobState {
    Running { step: u64 },
    DoneSim(SimResult),
    DoneMc(McSummaryResult),
    DoneCruise(CruiseResult),
    DoneCruiseMc(CruiseMcSummaryResult),
    Error(String),
}

struct Job {
    state: Mutex<JobState>,
    tx: broadcast::Sender<String>,
    /// Full step/run history, appended to alongside broadcasting — lets a
    /// client that missed the live stream (e.g. its WS handshake didn't
    /// finish before a fast simulation already did) retrieve the run after
    /// the fact via GET .../steps, instead of only ever seeing "__DONE__"
    /// with no data. Retained for the job's lifetime, same as everything else
    /// in the registry — fine for local dev, not for long-running
    /// deployments with many large jobs.
    payload: JobPayload,
    /// Polled by the running compute — single-run jobs only; see the module
    /// doc comment's Cancellation section. Set by `POST .../cancel`.
    cancelled: Arc<AtomicBool>,
}

fn registry() -> &'static Mutex<HashMap<u64, Arc<Job>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<u64, Arc<Job>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub async fn start(
    body: Result<Json<MissionConfig>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let mut cfg = match parse_body(body) {
        Ok(c) => c,
        Err(r) => return r,
    };
    // Review B2: server-resolved body tracks — expand any
    // `{ name, epoch_jd }` entry with an empty `track` into a real
    // ANISE-sampled track BEFORE validation, so `check_config`'s
    // ≥2-points rule doubles as the safety net rather than rejecting the
    // server-resolved form outright. A hard error (not best-effort like
    // the third-body auto-populate below): an empty track the client
    // explicitly asked the server to fill must never quietly become "no
    // perturbation at all."
    let wants_server_resolved = cfg
        .cruise_seed
        .as_ref()
        .is_some_and(|s| s.body_tracks.iter().any(|t| t.track.is_empty()));
    if wants_server_resolved {
        match crate::design::load_almanac() {
            Some(almanac) => {
                if let Err(e) = crate::design::resolve_named_body_tracks(&mut cfg, &almanac) {
                    return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "errors": [e] }))).into_response();
                }
            }
            None => {
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(json!({ "errors": ["cruise_seed.body_tracks: an empty track requests server-side \
                        ephemeris resolution, but this server has no ANISE kernel loaded (de440s.bsp) — \
                        supply the track samples explicitly"] })),
                )
                    .into_response();
            }
        }
    }
    let errors = check_config(&cfg);
    if !errors.is_empty() {
        return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "errors": errors }))).into_response();
    }

    // Phase 01/02 consistency ask: a cruise job's third-body
    // perturbers should match Phase 01's `target_body.third_bodies`
    // config, not silently drop to whatever the client happened to also
    // put in `cruise_seed.body_tracks`. Best-effort (no departure_epoch,
    // no kernel, or a name ANISE doesn't cover all degrade to "nothing
    // added", never an error) — see `design::build_third_body_tracks`'s
    // own doc comment for the full reasoning.
    if let Some(seed) = &cfg.cruise_seed {
        if let Some(almanac) = crate::design::load_almanac() {
            let auto = crate::design::build_third_body_tracks(&cfg, &almanac, seed.duration_s, &seed.body_tracks);
            cfg.cruise_seed.as_mut().unwrap().body_tracks.extend(auto);
        }
    }

    if cfg.cruise_seed.is_some() && cfg.simulation.monte_carlo_runs > 0 {
        start_cruise_mc_job(cfg)
    } else if cfg.cruise_seed.is_some() {
        // Checked ahead of the plain monte_carlo_runs check -- a
        // cruise-seeded run is a different mission loop entirely (Phase
        // 13's translation+attitude composition), not a variant of the
        // body-centric Sim/Mc path.
        start_cruise_job(cfg)
    } else if cfg.simulation.monte_carlo_runs > 0 {
        start_mc_job(cfg)
    } else {
        start_single_run_job(cfg)
    }
}

/// `cruise_seed`-bearing path (Phase 5.2). Mirrors `start_single_run_job`'s
/// structure exactly (registry entry, spawn_blocking, cancellation via the
/// same `Arc<AtomicBool>` polled every tick) — the only real difference is
/// which loop function it calls and which message/result types it streams.
fn start_cruise_job(cfg: MissionConfig) -> Response {
    let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let (tx, _rx) = broadcast::channel::<String>(256);
    let cancelled = Arc::new(AtomicBool::new(false));
    let job = Arc::new(Job {
        state: Mutex::new(JobState::Running { step: 0 }),
        tx: tx.clone(),
        payload: JobPayload::Cruise { steps: Mutex::new(Vec::new()) },
        cancelled,
    });
    registry().lock().unwrap().insert(job_id, job.clone());

    tokio::task::spawn_blocking(move || {
        let tx_for_cb = tx.clone();
        let job_for_cb = job.clone();
        let step_count = AtomicU64::new(0);

        let mut on_row = move |row: &CruiseTickRow| -> bool {
            if job_for_cb.cancelled.load(Ordering::Relaxed) {
                return false;
            }
            let msg = CruiseStepMsg {
                msg_type: "cruise_tick",
                t_s: row.t_s,
                r_m: [row.r_m.x, row.r_m.y, row.r_m.z],
                v_mps: [row.v_mps.x, row.v_mps.y, row.v_mps.z],
                q: [row.q[0], row.q[1], row.q[2], row.q[3]],
                q_cmd: [row.q_cmd[0], row.q_cmd[1], row.q_cmd[2], row.q_cmd[3]],
                wheel_speeds_radps: row.wheel_speeds_radps,
                wheel_momentum_nms: row.wheel_momentum_nms,
                pointing_error_deg: row.pointing_error_deg,
                torque_cmd_body_nm: [row.torque_cmd_body_nm.x, row.torque_cmd_body_nm.y, row.torque_cmd_body_nm.z],
                torque_delivered_body_nm: [row.torque_delivered_body_nm.x, row.torque_delivered_body_nm.y, row.torque_delivered_body_nm.z],
                dr_m: row.dr_m,
                dv_mps: row.dv_mps,
                rcs_propellant_kg_cum: row.rcs_propellant_kg_cum,
                wheel_sat_frac: row.wheel_sat_frac,
                wheel_torque_cmd_nm: row.wheel_motor_torque_cmd_nm,
                wheel_torque_delivered_nm: row.wheel_motor_torque_nm,
                wheel_torque_sat_frac: row.wheel_torque_sat_frac,
                omega_radps: [row.omega_radps.x, row.omega_radps.y, row.omega_radps.z],
                planned_burn_idx: row.planned_burn_idx,
                planned_burn_fault: row.planned_burn_fault.map(str::to_string),
                propellant_remaining_kg: row.propellant_remaining_kg,
                torque_gravity_gradient_nm: row.torque_gravity_gradient_nm,
                torque_srp_nm: row.torque_srp_nm,
                accel_central_gravity_mps2: row.accel_central_gravity_mps2,
                accel_third_body_mps2: row.accel_third_body_mps2,
                accel_srp_mps2: row.accel_srp_mps2,
                active_mode: row.active_mode.clone(),
                max_rule_violation_deg: row.max_rule_violation_deg,
                worst_violated_rule_label: row.worst_violated_rule_label.clone(),
                tcm_phase: row.tcm_phase.map(str::to_string),
                tcm_propellant_kg_cum: row.tcm_propellant_kg_cum,
                tcm_dv_mps_cum: row.tcm_dv_mps_cum,
                tcm_solve: row.tcm_solve.clone(),
                control_mode: row.control_mode.to_string(),
                controller_law: row.controller_law.to_string(),
                control_activity: row.control_activity.to_string(),
                controller_kp: row.controller_kp,
                controller_kd: row.controller_kd,
                gain_schedule_point: row.gain_schedule_point.clone(),
            };
            if let JobPayload::Cruise { steps } = &job_for_cb.payload {
                steps.lock().unwrap().push(msg.clone());
            }
            if let Ok(text) = serde_json::to_string(&msg) {
                let _ = tx_for_cb.send(text);
            }
            let n = step_count.fetch_add(1, Ordering::Relaxed) + 1;
            *job_for_cb.state.lock().unwrap() = JobState::Running { step: n };
            true
        };

        let outcome = run_cruise_streaming(&cfg, &mut on_row);
        *job.state.lock().unwrap() = match outcome {
            Ok(result) => JobState::DoneCruise(result),
            Err(e) => JobState::Error(e),
        };
        let _ = tx.send("__DONE__".to_string());
    });

    (StatusCode::OK, Json(json!({ "job_id": job_id }))).into_response()
}

/// `cruise_seed` + `monte_carlo_runs > 0` path)
/// — mirrors `start_mc_job`'s structure exactly (one message per completed
/// run, never per-tick, per the Monte Carlo streaming hard rule); the
/// underlying dispersion is computed by `cruise::run_cruise_mc_streaming`,
/// not per-run in this route. `cancelled` is stored/settable via `/cancel`
/// for status-reporting consistency but NOT honored by the compute below —
/// same known gap `start_mc_job` already has (all `n` runs dispatched via
/// `thread::scope` up front, no per-run checkpoint to interrupt mid-batch).
fn start_cruise_mc_job(cfg: MissionConfig) -> Response {
    let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let (tx, _rx) = broadcast::channel::<String>(256);
    let cancelled = Arc::new(AtomicBool::new(false));
    let job = Arc::new(Job {
        state: Mutex::new(JobState::Running { step: 0 }),
        tx: tx.clone(),
        payload: JobPayload::CruiseMc { runs: Mutex::new(Vec::new()) },
        cancelled,
    });
    registry().lock().unwrap().insert(job_id, job.clone());

    tokio::task::spawn_blocking(move || {
        let tx_for_cb = tx.clone();
        let job_for_cb = job.clone();

        let mut on_run_done = move |msg: CruiseMcRunMsg| {
            let n = (msg.run_index + 1) as u64;
            if let JobPayload::CruiseMc { runs } = &job_for_cb.payload {
                runs.lock().unwrap().push(msg.clone());
            }
            if let Ok(text) = serde_json::to_string(&msg) {
                let _ = tx_for_cb.send(text);
            }
            *job_for_cb.state.lock().unwrap() = JobState::Running { step: n };
        };

        let outcome = run_cruise_mc_streaming(&cfg, &mut on_run_done);
        *job.state.lock().unwrap() = match outcome {
            Ok(result) => JobState::DoneCruiseMc(result),
            Err(e) => JobState::Error(e),
        };
        let _ = tx.send("__DONE__".to_string());
    });

    (StatusCode::OK, Json(json!({ "job_id": job_id }))).into_response()
}

/// `monte_carlo_runs == 0` path — the original single-run behavior.
fn start_single_run_job(cfg: MissionConfig) -> Response {
    let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let (tx, _rx) = broadcast::channel::<String>(256);
    let cancelled = Arc::new(AtomicBool::new(false));
    let job = Arc::new(Job {
        state: Mutex::new(JobState::Running { step: 0 }),
        tx: tx.clone(),
        payload: JobPayload::Sim { steps: Mutex::new(Vec::new()) },
        cancelled,
    });
    registry().lock().unwrap().insert(job_id, job.clone());

    tokio::task::spawn_blocking(move || {
        let tx_for_cb = tx.clone();
        let job_for_cb = job.clone();
        let step_count = AtomicU64::new(0);

        let mut on_step = move |telem: &StepTelemetry, q: &nalgebra::Vector4<f64>| -> bool {
            // Cancellation is checked on every truth step (dt_truth_s
            // granularity) regardless of the streaming throttle below, so a
            // cancel request takes effect promptly even between measurements.
            if job_for_cb.cancelled.load(Ordering::Relaxed) {
                return false;
            }
            // Throttle to measurement cadence (dt_meas_s) — streaming every
            // dense truth-integration step (dt_truth_s) would flood the socket.
            if !telem.measurement_taken {
                return true;
            }
            let msg = SimStepMsg {
                t_s: telem.t_s,
                phase_name: telem.phase_name.to_string(),
                r_truth_m: [telem.r_truth_m.x, telem.r_truth_m.y, telem.r_truth_m.z],
                v_truth_mps: [telem.v_truth_mps.x, telem.v_truth_mps.y, telem.v_truth_mps.z],
                r_ekf_m: [telem.r_ekf_m.x, telem.r_ekf_m.y, telem.r_ekf_m.z],
                v_ekf_mps: [telem.v_ekf_mps.x, telem.v_ekf_mps.y, telem.v_ekf_mps.z],
                q: [q[0], q[1], q[2], q[3]],
                sigma_pos_m: telem.sigma_pos_m,
                sigma_vel_mps: telem.sigma_vel_mps,
                wheel_speeds_radps: telem.wheel_speeds_radps,
                wheel_momentum_nms: telem.wheel_momentum_nms,
                wheel_sat_frac: telem.wheel_sat_frac,
                desat_fired: telem.desat_fired,
                bearing_residual_rad: telem.bearing_residual_rad,
                angular_size_residual_rad: telem.angular_size_residual_rad,
                lidar_residual_m: telem.lidar_residual_m,
            };
            if let JobPayload::Sim { steps } = &job_for_cb.payload {
                steps.lock().unwrap().push(msg.clone());
            }
            if let Ok(text) = serde_json::to_string(&msg) {
                let _ = tx_for_cb.send(text);
            }
            let n = step_count.fetch_add(1, Ordering::Relaxed) + 1;
            *job_for_cb.state.lock().unwrap() = JobState::Running { step: n };
            true
        };

        let outcome = run_streaming(&cfg, &mut on_step);
        *job.state.lock().unwrap() = match outcome {
            Ok(result) => JobState::DoneSim(result),
            Err(e) => JobState::Error(e),
        };
        let _ = tx.send("__DONE__".to_string());
    });

    (StatusCode::OK, Json(json!({ "job_id": job_id }))).into_response()
}

/// `monte_carlo_runs > 0` path. Streams one `McRunMsg` per
/// *completed run*, never the dense per-step feed — see the module doc
/// comment's "Two job kinds" section and the design notes Monte Carlo streaming
/// hard rule.
fn start_mc_job(cfg: MissionConfig) -> Response {
    let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let (tx, _rx) = broadcast::channel::<String>(256);
    // Stored and settable via /cancel for status-reporting consistency, but
    // NOT honored by the compute below — see the module doc comment's
    // Cancellation section for why MC can't be interrupted mid-batch today.
    let cancelled = Arc::new(AtomicBool::new(false));
    let job = Arc::new(Job {
        state: Mutex::new(JobState::Running { step: 0 }),
        tx: tx.clone(),
        payload: JobPayload::Mc { runs: Mutex::new(Vec::new()) },
        cancelled,
    });
    registry().lock().unwrap().insert(job_id, job.clone());

    tokio::task::spawn_blocking(move || {
        let tx_for_cb = tx.clone();
        let job_for_cb = job.clone();

        let mut on_run_done = move |msg: McRunMsg| {
            let n = (msg.run_index + 1) as u64;
            if let JobPayload::Mc { runs } = &job_for_cb.payload {
                runs.lock().unwrap().push(msg.clone());
            }
            if let Ok(text) = serde_json::to_string(&msg) {
                let _ = tx_for_cb.send(text);
            }
            *job_for_cb.state.lock().unwrap() = JobState::Running { step: n };
        };

        let outcome = run_monte_carlo_streaming(&cfg, &mut on_run_done);
        *job.state.lock().unwrap() = match outcome {
            Ok(result) => JobState::DoneMc(result),
            Err(e) => JobState::Error(e),
        };
        let _ = tx.send("__DONE__".to_string());
    });

    (StatusCode::OK, Json(json!({ "job_id": job_id }))).into_response()
}

pub async fn status(Path(job_id): Path<u64>) -> Response {
    let Some(job) = registry().lock().unwrap().get(&job_id).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "job not found" }))).into_response();
    };
    let cancelled = job.cancelled.load(Ordering::Relaxed);
    let state = job.state.lock().unwrap();
    match &*state {
        JobState::Running { step } => {
            Json(json!({ "running": true, "step": step, "total_steps": null, "error": null, "cancelled": cancelled })).into_response()
        }
        JobState::DoneSim(_) | JobState::DoneMc(_) | JobState::DoneCruise(_) | JobState::DoneCruiseMc(_) => {
            Json(json!({ "running": false, "step": null, "total_steps": null, "error": null, "cancelled": cancelled })).into_response()
        }
        JobState::Error(e) => {
            Json(json!({ "running": false, "step": null, "total_steps": null, "error": e, "cancelled": cancelled })).into_response()
        }
    }
}

/// Request early stop for a running job (Phase 9k task 4). Idempotent;
/// valid for any job state (cancelling an already-finished job is a
/// harmless no-op). See the module doc comment's Cancellation section — for
/// a Monte Carlo job (`monte_carlo_runs > 0`) this only affects `/status`
/// reporting, not the in-flight batch.
pub async fn cancel(Path(job_id): Path<u64>) -> Response {
    let Some(job) = registry().lock().unwrap().get(&job_id).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "job not found" }))).into_response();
    };
    job.cancelled.store(true, Ordering::Relaxed);
    (StatusCode::OK, Json(json!({ "ok": true, "cancelled": true }))).into_response()
}

pub async fn result(Path(job_id): Path<u64>) -> Response {
    let Some(job) = registry().lock().unwrap().get(&job_id).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "job not found" }))).into_response();
    };
    let state = job.state.lock().unwrap();
    match &*state {
        JobState::DoneSim(r) => Json(serde_json::to_value(r).unwrap()).into_response(),
        JobState::DoneMc(r) => Json(serde_json::to_value(r).unwrap()).into_response(),
        JobState::DoneCruise(r) => Json(serde_json::to_value(r).unwrap()).into_response(),
        JobState::DoneCruiseMc(r) => Json(serde_json::to_value(r).unwrap()).into_response(),
        JobState::Running { .. } => {
            (StatusCode::CONFLICT, Json(json!({ "error": "job still running" }))).into_response()
        }
        JobState::Error(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
    }
}

/// Full step/run history for a job, whether still running (partial so far)
/// or done. Same message shape as the WS stream (`SimStepMsg` for a
/// single-run job, `McRunMsg` for a Monte Carlo job) — for replaying a run
/// that finished before a client connected to `/stream`, or for any client
/// that just prefers polling over a WebSocket.
pub async fn steps(Path(job_id): Path<u64>) -> Response {
    let Some(job) = registry().lock().unwrap().get(&job_id).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "job not found" }))).into_response();
    };
    match &job.payload {
        JobPayload::Sim { steps } => {
            let steps = steps.lock().unwrap().clone();
            Json(json!({ "steps": steps })).into_response()
        }
        JobPayload::Mc { runs } => {
            let runs = runs.lock().unwrap().clone();
            Json(json!({ "runs": runs })).into_response()
        }
        JobPayload::Cruise { steps } => {
            let steps = steps.lock().unwrap().clone();
            Json(json!({ "steps": steps })).into_response()
        }
        JobPayload::CruiseMc { runs } => {
            let runs = runs.lock().unwrap().clone();
            Json(json!({ "runs": runs })).into_response()
        }
    }
}

pub async fn stream(ws: WebSocketUpgrade, Path(job_id): Path<u64>) -> Response {
    let Some(job) = registry().lock().unwrap().get(&job_id).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "job not found" }))).into_response();
    };
    ws.on_upgrade(move |socket| handle_stream(socket, job))
}

async fn handle_stream(mut socket: WebSocket, job: Arc<Job>) {
    // Broadcast channels don't replay history to late subscribers. If the job
    // already finished before this client connected (very likely for fast
    // simulations — the WS handshake can easily take longer than the sim
    // itself), subscribing now would wait forever for a message that already
    // went out. Check first and send the sentinel immediately in that case.
    let already_done = !matches!(*job.state.lock().unwrap(), JobState::Running { .. });
    if already_done {
        let _ = socket.send(Message::Text("__DONE__".to_string())).await;
        let _ = socket.close().await;
        return;
    }

    let mut rx = job.tx.subscribe();
    loop {
        match rx.recv().await {
            Ok(text) => {
                let is_done = text == "__DONE__";
                if socket.send(Message::Text(text)).await.is_err() {
                    break;
                }
                if is_done {
                    break;
                }
            }
            Err(broadcast::error::RecvError::Closed) => break,
            // Client too slow to keep up — skip ahead rather than disconnect.
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
        }
    }
    let _ = socket.close().await;
}

fn parse_body(
    body: Result<Json<MissionConfig>, axum::extract::rejection::JsonRejection>,
) -> Result<MissionConfig, Response> {
    body.map(|Json(c)| c).map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("JSON parse error: {e}") })))
            .into_response()
    })
}
