//! POST /api/validate — parse and validate a JSON mission config.
//!
//! Request body: JSON object matching `MissionConfig` (same field names as TOML).
//! Response: `{ valid: bool, errors: string[], summary: { ... } }`

use axum::{http::StatusCode, response::{IntoResponse, Response}, Json};
use serde_json::{json, Value};

use crate::config::{check_config, MissionConfig};

pub async fn validate(
    body: Result<Json<MissionConfig>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let cfg = match body {
        Ok(Json(c)) => c,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "valid": false, "errors": [format!("JSON parse error: {e}")], "summary": null })),
            )
                .into_response();
        }
    };

    let errors = check_config(&cfg);
    let valid = errors.is_empty();
    let summary = config_summary(&cfg);

    Json(json!({ "valid": valid, "errors": errors, "summary": summary })).into_response()
}

fn config_summary(c: &MissionConfig) -> Value {
    json!({
        "name":      c.mission.name,
        "objective": c.mission.objective.to_string(),
        "body": {
            "name":          c.target_body.name,
            "mu_m3s2":       c.target_body.mu_m3s2,
            "radius_m":      c.target_body.radius_m,
            "gravity_model": c.target_body.gravity_model.to_string(),
            "atmosphere":    c.target_body.atmosphere.to_string(),
            "ephemeris":     c.target_body.ephemeris.to_string(),
            "third_bodies":  c.target_body.third_bodies,
        },
        "spacecraft": {
            "mass_kg":            c.spacecraft.mass_kg,
            "dry_mass_kg":        c.spacecraft.dry_mass_kg,
            "propellant_mass_kg": c.spacecraft.propellant_mass_kg,
            "srp_model":          c.spacecraft.srp_model.to_string(),
            "hardware_count":     c.spacecraft.hardware.len(),
        },
        "trajectory": {
            "solver": c.trajectory.solver.to_string(),
            "phases": c.trajectory.phases.iter().map(|p| p.to_string()).collect::<Vec<_>>(),
        },
        "gnc": {
            "navigation_filter":      c.gnc.navigation_filter.to_string(),
            "pointing_mode":          c.gnc.pointing_mode.to_string(),
            "position_accuracy_req_m": c.gnc.position_accuracy_req_m,
        },
        "simulation": {
            "integrator":       c.simulation.integrator.to_string(),
            "dt_truth_s":       c.simulation.dt_truth_s,
            "dt_meas_s":        c.simulation.dt_meas_s,
            "monte_carlo_runs": c.simulation.monte_carlo_runs,
            "output_dir":       c.simulation.output_dir,
        },
    })
}
