//! Full Bennu mission pipeline — runs all phases end to end.
//!
//! Sequence:
//!   1. cruise_design      — Lambert porkchop + optimal transfer
//!   2. cruise_operations  — heliocentric cruise, DSN OD, TCMs, arrival burn
//!   3. proximity_mission  — 5-phase proximity operations
//!
//! Run:  `cargo run -p autonomous_navigation --bin mission --release`
//!
//! Each sub-binary writes its own outputs under out/<phase>/.
//! Mission summary is printed at the end.

use std::process::Command;
use std::time::Instant;

fn run(binary: &str, label: &str) {
    println!("\n{}", "─".repeat(60));
    println!("  PHASE: {label}");
    println!("{}\n", "─".repeat(60));

    let t0 = Instant::now();

    // Detect the compiled binary path (release mode).
    // Works whether mission was invoked via `cargo run` or directly.
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    // On Windows executables have a .exe suffix; on Unix they don't.
    let exe_name = if cfg!(windows) {
        format!("{binary}.exe")
    } else {
        binary.to_string()
    };

    let exe_path = exe_dir.join(&exe_name);

    // All sub-binaries must run from the package root so that relative paths
    // like "out/mission/" and "horizons_results_bennu.txt" resolve correctly.
    // CARGO_MANIFEST_DIR is embedded at compile time = the directory that
    // contains this package's Cargo.toml — reliable regardless of CWD.
    let work_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let status = Command::new(&exe_path)
        .current_dir(&work_dir)
        .status()
        .unwrap_or_else(|e| {
            eprintln!("  [ERROR] Could not launch {}: {}", exe_path.display(), e);
            eprintln!("  Make sure all binaries are built: cargo build --release --bins");
            std::process::exit(1);
        });

    if !status.success() {
        eprintln!("  [ERROR] {label} failed (exit {:?})", status.code());
        std::process::exit(1);
    }

    println!("\n  [OK] {label} completed in {:.1} s", t0.elapsed().as_secs_f64());
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║          BENNU MISSION — FULL PIPELINE                  ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let t_mission = Instant::now();

    run("cruise_design",      "1/3  Cruise Design       (Lambert + porkchop)");
    run("cruise_operations",  "2/3  Cruise Operations   (DSN OD + TCMs + arrival)");
    run("proximity_mission",  "3/3  Proximity Mission   (5-phase Bennu proximity)");

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  MISSION COMPLETE  —  total wall time: {:.1} s          ",
             t_mission.elapsed().as_secs_f64());
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("Outputs:");
    println!("  out/cruise/          — Lambert porkchop, best transfer");
    println!("  out/cruise_ops/      — truth + OD + EKF, TCM log");
    println!("  out/mission/         — 5-phase proximity (nav + attitude)");
    println!();
    println!("Plots:");
    println!("  py plot/plot_cruise.py");
    println!("  py plot/plot_cruise_ops.py");
    println!("  py plot/plot_mission.py");
}
