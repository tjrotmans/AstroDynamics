//! POST /api/dev/capture-preset — capture the CURRENT frontend mission
//! state as the app's fixed example (: "Can
//! you capture the current page state and make it the standard one when
//! clicking on the landing page on mercury orbit? And in addition can we
//! make a toml for it so we dont lose any of these inputs").
//!
//! A local development tool, not a product API: this server only ever runs
//! on the developer's own machine, and the whole point is writing REAL
//! files into both sibling repos so the captured example is versioned:
//!
//! - `MissionPlanner/config/{name}.toml` — the full `MissionConfig` as a
//!   permanent, human-readable TOML (nulls stripped: TOML has no null, and
//!   an absent key deserializes identically to an explicit null here).
//! - `../AstroDynamics-UI/src/data/presetSnapshots/{name}-config.json` —
//!   the same config verbatim, loaded by the landing chip so EVERY setting
//!   (bounds, budgets, capture radius, windows) is restored, not just the
//!   route/objective.
//! - `../AstroDynamics-UI/src/data/presetSnapshots/{name}-optimize.json` —
//!   the captured `OptimizeApiResult`, seeded as the chip's pre-baked
//!   optimizer result.
//!
//! `name` is restricted to `[a-z0-9_-]` — these become file names in two
//! repos; nothing else is sanitized because nothing else is a path.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
pub struct CapturePresetRequest {
    /// Snapshot base name, e.g. "mercury" — see module doc for the three
    /// files it becomes.
    pub name: String,
    /// The frontend's full current MissionConfig, as the same JSON it sends
    /// to /api/optimize.
    pub config: Value,
    /// The captured OptimizeApiResult (the frontend's persisted result).
    pub optimize_result: Value,
}

/// Recursively drop null values — TOML has no null, and for `MissionConfig`
/// an absent optional key deserializes identically to an explicit null.
fn strip_nulls(v: &Value) -> Value {
    match v {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(_, val)| !val.is_null())
                .map(|(k, val)| (k.clone(), strip_nulls(val)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(strip_nulls).collect()),
        other => other.clone(),
    }
}

pub async fn capture_preset(body: Json<CapturePresetRequest>) -> Response {
    let name = body.name.trim().to_lowercase();
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": "name must be non-empty [a-z0-9_-]" })),
        )
            .into_response();
    }

    // Round-trip the config through the real MissionConfig type first — a
    // capture that wouldn't load back is worse than an error now.
    let cleaned = strip_nulls(&body.config);
    if let Err(e) = serde_json::from_value::<crate::config::MissionConfig>(cleaned.clone()) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": format!("config does not round-trip as MissionConfig: {e}") })),
        )
            .into_response();
    }

    let toml_text = match toml::to_string_pretty(&cleaned) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": format!("config does not serialize to TOML: {e}") })),
            )
                .into_response();
        }
    };

    let toml_path = format!("MissionPlanner/config/{name}_orbiter_example.toml");
    let cfg_json_path = format!("../AstroDynamics-UI/src/data/presetSnapshots/{name}-config.json");
    let result_json_path = format!("../AstroDynamics-UI/src/data/presetSnapshots/{name}-optimize.json");

    let header = format!(
        "# Captured frontend example \"{name}\" (via POST /api/dev/capture-preset).\n\
         # The exact MissionConfig that produced the committed {name}-optimize.json\n\
         # preset snapshot -- regenerate that snapshot by POSTing this config to\n\
         # /api/optimize on a matching build.\n\n"
    );
    let writes = [
        (toml_path.clone(), format!("{header}{toml_text}")),
        (cfg_json_path.clone(), serde_json::to_string_pretty(&cleaned).unwrap_or_default()),
        (
            result_json_path.clone(),
            serde_json::to_string(&body.optimize_result).unwrap_or_default(),
        ),
    ];
    for (path, content) in &writes {
        if let Err(e) = std::fs::write(path, content) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("could not write {path}: {e}") })),
            )
                .into_response();
        }
    }

    Json(json!({ "written": [toml_path, cfg_json_path, result_json_path] })).into_response()
}
