//! POST /api/optimize                  — start an async Phase 9 optimization job, returns `{job_id}`
//! GET  /api/optimize/:job_id/status   — `{running, step, total_steps, error, cancelled}`
//! WS   /api/optimize/:job_id/stream   — live `OptimizeStepMsg` feed, one message per generation/iteration
//! GET  /api/optimize/:job_id/result   — final `OptimizeApiResult` once done
//! POST /api/optimize/:job_id/cancel   — request early stop (Phase 9k task 4)
//!
//! Same job-registry/broadcast pattern as `routes::simulate` (Phase 9f) —
//! see that module's doc comment for the rationale (in-memory `OnceLock`
//! registry, `"__DONE__"` sentinel, the `/steps`-style late-subscriber race).
//! The one real difference: `total_steps` here is NOT `null` — generation/
//! iteration count is a config value (`[optimization.ga].generations` or
//! `[optimization.pso].iterations`), known before the job starts, unlike
//! `/api/simulate`'s phase-duration-dependent step count.
//!
//! Per the design notes Monte Carlo streaming rule (never the dense per-step
//! feed): this stream carries only `{step, best_fitness}` per generation/
//! iteration, never a propagated arc — re-propagating and serializing a full
//! arc on every generation would be needless cost for a value only the final
//! result needs. The arc is computed once, in `/result`.
//!
//! ## Cancellation (Phase 9k task 4)
//!
//! Compute runs in `spawn_blocking` (see `start`), so `JoinHandle::abort()`
//! would not interrupt it — the underlying OS thread keeps running a
//! CPU-bound loop regardless. Instead, each `Job` carries a
//! `cancelled: Arc<AtomicBool>` that the running compute polls: GA/PSO check
//! it after every generation/iteration and the post-search coordinate-descent
//! refinement checks it every pass (`GaSolver`/`PsoSolver::run_with_progress`
//! / `run_with_population_progress` now return `bool` from their progress
//! callback — `false` stops the solver's internal loop early and it returns
//! whatever candidate was best so far). `POST .../cancel` just sets the flag;
//! the job transitions to done (with a partial-but-valid result) on its own
//! once the running generation finishes.
//!
//! MGA (`OptimizationMethod::MGA`) does NOT honor cancellation yet — its DE
//! search has no polling hook wired in (see `optimize_api_with_progress`'s
//! doc comment). `POST .../cancel` still sets the flag and `/status` still
//! reports `cancelled: true`, but an MGA job's compute keeps running to
//! completion regardless. This is a known, documented gap, not an oversight.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use serde_json::json;
use tokio::sync::broadcast;

use crate::config::{check_config, MissionConfig, OptimizationMethod};
use crate::design::load_almanac;
use crate::mga::MgaLegStepInfo;
use crate::optimize::{optimize_api_with_progress, OptimizeApiResult};

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

enum JobState {
    Running { step: u64 },
    Done(OptimizeApiResult),
    Error(String),
}

struct Job {
    state: Mutex<JobState>,
    tx: broadcast::Sender<String>,
    total_steps: Option<u64>,
    /// Polled by the running compute (GA/PSO + refinement only — see the
    /// module doc comment's Cancellation section); set by `POST .../cancel`.
    cancelled: Arc<AtomicBool>,
}

#[derive(Serialize, Clone)]
struct OptimizeStepMsg {
    step: usize,
    /// `1` (flyby-only closest-approach search, value in km) or `2` (real
    /// configured objective, dimensionless normalized fitness) -- the two
    /// metrics are on completely different scales; a client must not plot
    /// them on one shared y-axis. See `run_optimization_with_progress`'s
    /// doc comment.
    phase: u8,
    best_fitness: f64,
    /// The generation's best chromosome (raw MGA parameter vector — see
    /// `mga.rs`'s chromosome layout doc comment), for MGA jobs only (Phase
    /// 9k task 1). `None` for GA/PSO, which don't stream decision variables
    /// live (only in the final result's `population_log`).
    best_params: Option<Vec<f64>>,
    /// Per-leg trajectory state (DSM position/velocity, arrival velocity,
    /// arrival v_∞) for `best_params`, MGA jobs only (backlog item #18,
    ///) — see `MgaLegStepInfo`'s doc comment. `None` for GA/PSO
    /// (no leg concept). For an MGA job this is `Some(vec![])` rather than
    /// `None` on a generation whose best chromosome happened not to be
    /// evaluable — a client should treat an empty list the same as "no leg
    /// detail this generation," not as a different job type.
    mga_legs: Option<Vec<MgaLegStepInfo>>,
    /// This generation's full evaluated population (each entry one
    /// parameter vector, same layout as `best_params`) — GA jobs only,
    /// (the live search-space scatter plots' data source; the
    /// stream used to carry only the best candidate, so full clouds only
    /// appeared after completion via `population_log`). FEASIBLE
    /// individuals only, parallel to `population_outcomes`. `None` for PSO
    /// (solver exposes only the global best per iteration) and MGA.
    population: Option<Vec<Vec<f64>>>,
    /// Per-individual outcome pairs `[dv_arrival_ms, tof_days]`, parallel
    /// to (same length/order as) `population` — GA jobs only:
    /// the outcome scatter plots' data source (departure vs arrival ΔV,
    /// departure offset vs coast time). Recorded as fitness-evaluation
    /// side-products, zero extra propagation cost.
    population_outcomes: Option<Vec<[f64; 2]>>,
    /// Best objective value among FEASIBLE-capture candidates seen so far
    /// (natural units, same as `best_fitness`) — `None` until
    /// the first candidate inside the target's SOI exists. The solid line
    /// on the convergence plot; `best_fitness` is the dashed
    /// may-be-infeasible lower bound.
    best_feasible_value: Option<f64>,
}

/// A separate announcement frame, NOT a step — sent over the same `/stream`
/// WebSocket connection (Phase 9k "step-stream context" ask).
/// Fires once per candidate flyby-body sequence for an MGA job, BEFORE the
/// `OptimizeStepMsg`s for that sequence's own generations arrive — see
/// `mga::run_mga`'s doc comment. Never sent for GA/PSO jobs. Distinguished
/// from `OptimizeStepMsg` on the wire by shape (`msg_type` is always
/// `"mga_sequence"`, a field `OptimizeStepMsg` never carries — no field is
/// ever added to `OptimizeStepMsg` itself, so existing consumers parsing it
/// directly are unaffected).
#[derive(Serialize, Clone)]
struct MgaSequenceContextMsg {
    msg_type: &'static str,
    /// 0-indexed position of this sequence among the candidates being
    /// optimized this run.
    seq_idx: usize,
    /// Total candidate sequences being optimized this run (>= 1; also 1 for
    /// the non-`sequence_search` fixed-sequence path).
    seq_count: usize,
    /// The real flyby-body sequence about to be optimized (excludes the
    /// overall `departure_body`/`target_body` — same convention as
    /// `MgaParams.flyby_bodies`/`OptimizeApiResult.mga_body_sequence`'s
    /// interior slice).
    flyby_bodies: Vec<String>,
    /// True for the always-included zero-flyby direct baseline the
    /// Tisserand beam search compares every gravity-assist candidate
    /// against (Phase 9j-B). False for every other sequence, including the
    /// fixed-sequence (non-auto) path.
    is_direct_baseline: bool,
}

fn registry() -> &'static Mutex<HashMap<u64, Arc<Job>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<u64, Arc<Job>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `[optimization].ga.generations` or `.pso.iterations`, when known —
/// `None` for methods without a config'd step count yet (MultipleShooting/MGA).
fn known_total_steps(cfg: &MissionConfig) -> Option<u64> {
    let opt = cfg.optimization.as_ref()?;
    match opt.method {
        OptimizationMethod::GA => opt.ga.as_ref().map(|p| p.generations as u64),
        OptimizationMethod::PSO => opt.pso.as_ref().map(|p| p.iterations as u64),
        OptimizationMethod::MultipleShooting | OptimizationMethod::MGA => None,
    }
}

pub async fn start(
    body: Result<Json<MissionConfig>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let cfg = match parse_body(body) {
        Ok(c) => c,
        Err(r) => return r,
    };
    let errors = check_config(&cfg);
    if !errors.is_empty() {
        return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "errors": errors }))).into_response();
    }
    if cfg.optimization.is_none() {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "errors": ["this config has no [optimization] section"] })),
        )
            .into_response();
    }

    let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let total_steps = known_total_steps(&cfg);
    let (tx, _rx) = broadcast::channel::<String>(256);
    let cancelled = Arc::new(AtomicBool::new(false));
    let job = Arc::new(Job { state: Mutex::new(JobState::Running { step: 0 }), tx: tx.clone(), total_steps, cancelled });
    registry().lock().unwrap().insert(job_id, job.clone());

    tokio::task::spawn_blocking(move || {
        let Some(almanac) = load_almanac() else {
            *job.state.lock().unwrap() = JobState::Error("could not load ANISE kernel (de440s.bsp not found)".into());
            let _ = tx.send("__DONE__".to_string());
            return;
        };

        let tx_for_cb = tx.clone();
        let job_for_cb = job.clone();
        let on_step = move |step: usize, phase: u8, best_fitness: f64, best_feasible_value: Option<f64>, best_params: Option<&[f64]>, legs: &[MgaLegStepInfo], population: &[Vec<f64>], outcomes: &[[f64; 2]]| {
            let msg = OptimizeStepMsg {
                step, phase, best_fitness,
                best_params: best_params.map(|p| p.to_vec()),
                mga_legs: best_params.map(|_| legs.to_vec()),
                population: if population.is_empty() { None } else { Some(population.to_vec()) },
                population_outcomes: if outcomes.is_empty() { None } else { Some(outcomes.to_vec()) },
                best_feasible_value,
            };
            if let Ok(text) = serde_json::to_string(&msg) {
                let _ = tx_for_cb.send(text);
            }
            *job_for_cb.state.lock().unwrap() = JobState::Running { step: step as u64 };
        };

        let tx_for_seq = tx.clone();
        let on_sequence = move |seq_idx: usize, seq_count: usize, flyby_bodies: &[String], is_direct_baseline: bool| {
            let msg = MgaSequenceContextMsg {
                msg_type: "mga_sequence",
                seq_idx,
                seq_count,
                flyby_bodies: flyby_bodies.to_vec(),
                is_direct_baseline,
            };
            if let Ok(text) = serde_json::to_string(&msg) {
                let _ = tx_for_seq.send(text);
            }
        };

        let outcome = optimize_api_with_progress(&cfg, &almanac, on_step, on_sequence, &job.cancelled);
        *job.state.lock().unwrap() = match outcome {
            Ok(result) => JobState::Done(result),
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
            Json(json!({ "running": true, "step": step, "total_steps": job.total_steps, "error": null, "cancelled": cancelled })).into_response()
        }
        JobState::Done(_) => {
            Json(json!({ "running": false, "step": null, "total_steps": job.total_steps, "error": null, "cancelled": cancelled })).into_response()
        }
        JobState::Error(e) => {
            Json(json!({ "running": false, "step": null, "total_steps": job.total_steps, "error": e, "cancelled": cancelled })).into_response()
        }
    }
}

/// Request early stop for a running job (Phase 9k task 4). Idempotent and
/// valid regardless of job state — cancelling an already-finished job is a
/// harmless no-op, not an error, since the client may race a fast job.
/// See the module doc comment's Cancellation section for what this does and
/// does not interrupt.
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
        JobState::Done(r) => Json(serde_json::to_value(r).unwrap()).into_response(),
        JobState::Running { .. } => {
            (StatusCode::CONFLICT, Json(json!({ "error": "job still running" }))).into_response()
        }
        JobState::Error(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
    }
}

pub async fn stream(ws: WebSocketUpgrade, Path(job_id): Path<u64>) -> Response {
    let Some(job) = registry().lock().unwrap().get(&job_id).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "job not found" }))).into_response();
    };
    ws.on_upgrade(move |socket| handle_stream(socket, job))
}

async fn handle_stream(mut socket: WebSocket, job: Arc<Job>) {
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
