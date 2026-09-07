//! Mission Planner HTTP server entry point.
//!
//! Listens on `127.0.0.1:8000` by default — loopback only, no Windows Firewall prompt.
//! Override: `MP_HOST=0.0.0.0` for remote access, `MP_PORT=<n>` for a different port.
//! The Vite dev server proxies from localhost:5173 to this server.
//!
//! Usage:
//!   cargo run --bin mission-server --release

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("MP_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8000);
    // Loopback by default — sufficient for local dev and avoids Windows Firewall prompts.
    // Set MP_HOST=0.0.0.0 only when remote access is needed.
    let host = std::env::var("MP_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let addr = format!("{host}:{port}");

    let app = mission_planner::server::create_router();
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind {addr}: {e}"));

    println!("Mission Planner server listening on http://{addr}");
    println!("  GET  /api/health");
    println!("  GET  /api/bodies");
    println!("  GET  /api/hardware");
    println!("  GET  /api/objectives");
    println!("  POST /api/validate");
    println!("  POST /api/design/trajectory");
    println!("  POST /api/design/gnc");
    println!("  POST /api/simulate");
    println!("  GET  /api/simulate/:job_id/status");
    println!("  WS   /api/simulate/:job_id/stream");
    println!("  GET  /api/simulate/:job_id/steps");
    println!("  GET  /api/simulate/:job_id/result");
    println!("  POST /api/optimize");
    println!("  GET  /api/optimize/:job_id/status");
    println!("  WS   /api/optimize/:job_id/stream");
    println!("  GET  /api/optimize/:job_id/result");

    axum::serve(listener, app)
        .await
        .unwrap_or_else(|e| panic!("Server error: {e}"));
}
