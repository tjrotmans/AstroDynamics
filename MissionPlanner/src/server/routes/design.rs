//! POST /api/design/trajectory — trajectory solver; returns `TrajectoryApiResult`.
//! POST /api/design/gnc         — GNC sizing; returns `GncDesign`.
//!
//! Both accept JSON matching `MissionConfig`. The trajectory endpoint offloads
//! the porkchop scan to `spawn_blocking` so the async runtime is never stalled.

use axum::{http::StatusCode, response::{IntoResponse, Response}, Json};
use serde_json::json;

use crate::config::{check_config, MissionConfig};

pub async fn trajectory(
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

    match tokio::task::spawn_blocking(move || crate::design::compute_trajectory(&cfg)).await {
        Ok(Ok(result)) => Json(serde_json::to_value(result).unwrap()).into_response(),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    }
}

pub async fn gnc(
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

    match crate::gnc_design::compute(&cfg) {
        Ok(result) => Json(serde_json::to_value(result).unwrap()).into_response(),
        Err(e) => (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "error": e }))).into_response(),
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
