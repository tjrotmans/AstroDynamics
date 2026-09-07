//! POST /api/mga-scan                  — start an async Phase 9w ballistic MGA
//!                                        window-scan job, returns `{job_id}`
//! GET  /api/mga-scan/:job_id/status   — `{running, error}`
//! GET  /api/mga-scan/:job_id/result   — final `MgaScanApiResult` once done
//!
//! Wraps the existing `mga-scan` CLI (`crate::mga_scan_run::mga_scan_api`,
//! Phase 9w) as an async job, same registry/`spawn_blocking` pattern as
//! `routes::optimize`/`routes::simulate` — a scan can take from seconds to
//! several minutes depending on grid density (a real 4-leg VEEGA scan at
//! 1,949 departure dates x 18 TOF points/leg took ~6 minutes, per the design notes
//! Phase 9w-i), so this must not block the request thread.
//!
//! No WebSocket stream: `run_mga_scan` has no per-evaluation progress hook
//! today (it's one blocking call, not a generation loop like GA/PSO/MGA-DE),
//! so there's nothing meaningful to stream. `/status` only reports
//! running/done/error, not a step counter. Adding real progress reporting
//! would mean threading a callback through `trajectory_solver::mga_scan` —
//! left for later if the frontend's explore-mode UI needs it (Phase 9w-vii).
//!
//! Request body: a full `MissionConfig` JSON with `[optimization.mga.scan]`
//! present (same shape the CLI reads via `mga_scan_run::run_mga_window_scan`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::config::{check_config, MissionConfig};
use crate::design::load_almanac;
use crate::mga_scan_run::{mga_scan_api, MgaScanApiResult};

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

enum JobState {
    Running,
    Done(MgaScanApiResult),
    Error(String),
}

struct Job {
    state: Mutex<JobState>,
}

fn registry() -> &'static Mutex<HashMap<u64, Arc<Job>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<u64, Arc<Job>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
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
    let mga_scan_configured = cfg.optimization.as_ref()
        .and_then(|o| o.mga.as_ref())
        .and_then(|m| m.scan.as_ref())
        .is_some();
    if !mga_scan_configured {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "errors": ["this config has no [optimization.mga.scan] section"] })),
        )
            .into_response();
    }

    let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let job = Arc::new(Job { state: Mutex::new(JobState::Running) });
    registry().lock().unwrap().insert(job_id, job.clone());

    tokio::task::spawn_blocking(move || {
        let Some(almanac) = load_almanac() else {
            *job.state.lock().unwrap() = JobState::Error("could not load ANISE kernel (de440s.bsp not found)".into());
            return;
        };
        let outcome = mga_scan_api(&cfg, &almanac);
        *job.state.lock().unwrap() = match outcome {
            Ok(result) => JobState::Done(result),
            Err(e) => JobState::Error(e),
        };
    });

    (StatusCode::OK, Json(json!({ "job_id": job_id }))).into_response()
}

pub async fn status(Path(job_id): Path<u64>) -> Response {
    let Some(job) = registry().lock().unwrap().get(&job_id).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "job not found" }))).into_response();
    };
    let state = job.state.lock().unwrap();
    match &*state {
        JobState::Running => Json(json!({ "running": true, "error": null })).into_response(),
        JobState::Done(_) => Json(json!({ "running": false, "error": null })).into_response(),
        JobState::Error(e) => Json(json!({ "running": false, "error": e })).into_response(),
    }
}

pub async fn result(Path(job_id): Path<u64>) -> Response {
    let Some(job) = registry().lock().unwrap().get(&job_id).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({ "error": "job not found" }))).into_response();
    };
    let state = job.state.lock().unwrap();
    match &*state {
        JobState::Done(r) => Json(serde_json::to_value(r).unwrap()).into_response(),
        JobState::Running => (StatusCode::CONFLICT, Json(json!({ "error": "job still running" }))).into_response(),
        JobState::Error(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
    }
}

fn parse_body(
    body: Result<Json<MissionConfig>, axum::extract::rejection::JsonRejection>,
) -> Result<MissionConfig, Response> {
    body.map(|Json(c)| c).map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("JSON parse error: {e}") })))
            .into_response()
    })
}
