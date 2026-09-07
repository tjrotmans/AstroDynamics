//! GET /api/objectives

use axum::Json;
use serde_json::{json, Value};

pub async fn objectives() -> Json<Value> {
    Json(json!({
        "objectives": [
            { "id": "Flyby",        "description": "Closest-approach trajectory past the body" },
            { "id": "Orbit",        "description": "Capture and station-keep in a closed orbit" },
            { "id": "Landing",      "description": "Descent from orbit to surface" },
            { "id": "Rendezvous",   "description": "Match the orbit of a target body or object" },
            { "id": "SampleReturn", "description": "Proximity ops + sampling + departure back to Earth" },
        ]
    }))
}
