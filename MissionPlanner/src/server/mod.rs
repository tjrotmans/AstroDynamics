//! HTTP server for the Mission Planner — Axum router + CORS setup.
//!
//! Start with `mission-server` binary. Listens on `0.0.0.0:8000`.
//! All POST endpoints accept JSON matching [`crate::config::MissionConfig`].
//!
//! Routes:
//!   GET  /api/health
//!   GET  /api/bodies
//!   GET  /api/bodies/:name/state
//!   GET  /api/hardware
//!   GET  /api/objectives
//!   GET  /api/presets
//!   POST /api/validate
//!   POST /api/design/trajectory
//!   POST /api/design/gnc
//!   POST /api/simulate
//!   GET  /api/simulate/:job_id/status
//!   WS   /api/simulate/:job_id/stream
//!   GET  /api/simulate/:job_id/steps
//!   GET  /api/simulate/:job_id/result
//!   POST /api/simulate/:job_id/cancel
//!   POST /api/optimize
//!   GET  /api/optimize/:job_id/status
//!   WS   /api/optimize/:job_id/stream
//!   GET  /api/optimize/:job_id/result
//!   POST /api/optimize/:job_id/cancel
//!   POST /api/mga-sequence-search
//!   POST /api/mga-scan
//!   GET  /api/mga-scan/:job_id/status
//!   GET  /api/mga-scan/:job_id/result

pub mod routes;

use axum::{routing::{get, post}, Router};
use tower_http::cors::{CorsLayer, Any};

/// Build the Axum router with all API routes and CORS middleware.
pub fn create_router() -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/api/health",             get(routes::health::health))
        .route("/api/bodies",             get(routes::bodies::bodies))
        .route("/api/bodies/:name/state", get(routes::bodies::state))
        .route("/api/hardware",           get(routes::hardware::hardware))
        .route("/api/objectives",         get(routes::objectives::objectives))
        .route("/api/presets",            get(routes::presets::presets))
        .route("/api/validate",           post(routes::validate::validate))
        .route("/api/design/trajectory",  post(routes::design::trajectory))
        .route("/api/design/gnc",         post(routes::design::gnc))
        .route("/api/design/vehicle",     post(routes::vehicle::vehicle))
        .route("/api/design/slew-test",   post(routes::slew_test::slew_test))
        .route("/api/simulate",                post(routes::simulate::start))
        .route("/api/simulate/:job_id/status", get(routes::simulate::status))
        .route("/api/simulate/:job_id/stream", get(routes::simulate::stream))
        .route("/api/simulate/:job_id/steps",  get(routes::simulate::steps))
        .route("/api/simulate/:job_id/result", get(routes::simulate::result))
        .route("/api/simulate/:job_id/cancel", post(routes::simulate::cancel))
        .route("/api/optimize",                post(routes::optimize::start))
        .route("/api/optimize/:job_id/status", get(routes::optimize::status))
        .route("/api/optimize/:job_id/stream", get(routes::optimize::stream))
        .route("/api/optimize/:job_id/result", get(routes::optimize::result))
        .route("/api/optimize/:job_id/cancel", post(routes::optimize::cancel))
        .route("/api/mga-sequence-search",     post(routes::mga_sequence_search::search))
        .route("/api/mga-scan",                post(routes::mga_scan::start))
        .route("/api/mga-scan/:job_id/status", get(routes::mga_scan::status))
        .route("/api/mga-scan/:job_id/result", get(routes::mga_scan::result))
        .route("/api/dev/capture-preset",      post(routes::capture_preset::capture_preset))
        .layer(cors)
}
