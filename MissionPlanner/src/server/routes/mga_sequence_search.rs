//! POST /api/mga-sequence-search — run the Tisserand beam search only, and
//! return every ranked candidate flyby sequence.
//!
//! Plain synchronous handler, unlike `routes::mga_scan`/`routes::optimize`
//! (async job + polling) — `sequence_search::run_sequence_search` is
//! explicitly the cheap middle tier of the MGA pipeline (its own doc comment:
//! "the outer Tisserand beam search prunes the discrete sequence space" before
//! the expensive inner solve), no propagation or Lambert solve per candidate,
//! so there is nothing to poll for.
//!
//! Added 
//! `run_sequence_search` previously only ever ran as an internal step of
//! `POST /api/mga-scan` (always immediately followed by the full ballistic
//! scan), `mga::run_mga`'s Auto path (ditto, followed by the real DE/MBH
//! search), or the CLI-only `search-sequence` subcommand — there was no way
//! to see the ranked candidate list on its own before committing to either of
//! those. This endpoint is exactly that: "just run the Tisserand search".
//!
//! Request body: a full `MissionConfig` JSON with
//! `[optimization.mga.sequence_search]` present.

use axum::{http::StatusCode, response::{IntoResponse, Response}, Json};
use serde_json::json;

use crate::config::{check_config, MissionConfig};
use crate::sequence_search::{run_sequence_search, RankedSequence};

/// One ranked candidate sequence, JSON-shaped — mirrors
/// `sequence_search::RankedSequence` (kept as a dedicated API type rather
/// than deriving `Serialize` directly on the internal struct, matching this
/// crate's existing convention of `*Api` wrapper types for API-facing data,
/// e.g. `MgaScanRecordApi` in `mga_scan_run.rs`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RankedSequenceApi {
    /// Intermediate flyby bodies only, in visit order (excludes the
    /// departure and target bodies, which are the chain's overall endpoints).
    /// Empty for the direct (zero-flyby) case.
    pub flyby_bodies: Vec<String>,
    /// Estimated v∞ at the target body from the Tisserand graph walk [m/s].
    pub estimated_vinf_arr_ms: f64,
    /// Cumulative Tisserand feasibility score (lower = better; this is the
    /// list's sort key).
    pub tisserand_score: f64,
}

impl From<&RankedSequence> for RankedSequenceApi {
    fn from(r: &RankedSequence) -> Self {
        RankedSequenceApi {
            flyby_bodies: r.flyby_bodies.clone(),
            estimated_vinf_arr_ms: r.estimated_vinf_arr_ms,
            tisserand_score: r.tisserand_score,
        }
    }
}

/// `POST /api/mga-sequence-search` result: every candidate sequence the beam
/// search found, sorted by `tisserand_score` ascending (best first) — not
/// just the top pick, so the caller can see the full ranked field before
/// deciding what to scan/optimize further.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SequenceSearchApiResult {
    pub sequences: Vec<RankedSequenceApi>,
}

pub async fn search(
    body: Result<Json<MissionConfig>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let cfg = match body {
        Ok(Json(c)) => c,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("JSON parse error: {e}") })),
            )
                .into_response();
        }
    };

    let errors = check_config(&cfg);
    if !errors.is_empty() {
        return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "errors": errors }))).into_response();
    }

    let Some(opt) = cfg.optimization.as_ref() else {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "errors": ["this config has no [optimization] section"] })),
        )
            .into_response();
    };
    let Some(mga) = opt.mga.as_ref() else {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "errors": ["this config has no [optimization.mga] section"] })),
        )
            .into_response();
    };
    let Some(ss_cfg) = mga.sequence_search.as_ref() else {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "errors": ["this config has no [optimization.mga.sequence_search] section"] })),
        )
            .into_response();
    };

    let sequences = run_sequence_search(ss_cfg, &opt.departure_body, &opt.target_body);
    let result = SequenceSearchApiResult {
        sequences: sequences.iter().map(RankedSequenceApi::from).collect(),
    };

    (StatusCode::OK, Json(result)).into_response()
}
