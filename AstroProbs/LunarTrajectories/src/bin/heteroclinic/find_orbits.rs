//! find_orbits — find and save periodic orbits in the Earth-Moon CRTBP.
//!
//! Edit the `orbit_specs()` function below to define which orbits to compute,
//! then run:
//!
//!   cargo run -p lunar_trajectories --bin find_orbits
//!
//! Output: out/orbits/{label}.csv  +  out/orbits/manifest.csv
//! Plot:   auto-opens  out/orbits/orbits.html

use std::fmt::Write;

use lunar_trajectories::crtbp::{CrtbpParams, jacobi_constant};
use lunar_trajectories::periodic_orbits::{
    lyapunov, halo, dro, full_period_traj, FoundOrbit, OrbitFamily,
};
use lunar_trajectories::propagator::Step3d;

// ╔══════════════════════════════════════════════════════════════╗
// ║                    USER CONFIGURATION                        ║
// ║                                                              ║
// ║  Edit orbit_specs() to add / remove / change orbits.        ║
// ║  Each entry needs:                                           ║
// ║    label     — output filename stem (no extension)           ║
// ║    family    — Lyapunov / HaloNorth / HaloSouth / Dro        ║
// ║    amplitude — Ax [nd] for Lyapunov, Az [nd] for Halo,      ║
// ║                r [nd] for DRO                                ║
// ╚══════════════════════════════════════════════════════════════╝

struct OrbitSpec {
    label:     &'static str,
    family:    OrbitFamily,
    amplitude: f64,
}

fn orbit_specs() -> Vec<OrbitSpec> {
    vec![
        // ── Lyapunov orbits ──────────────────────────────────────────────
        OrbitSpec { label: "l2_lyapunov",   family: OrbitFamily::Lyapunov  { lagrange: 2 }, amplitude: 0.010 },
        OrbitSpec { label: "l1_lyapunov",   family: OrbitFamily::Lyapunov  { lagrange: 1 }, amplitude: 0.010 },

        // ── Halo orbits ──────────────────────────────────────────────────
        OrbitSpec { label: "l2_halo_north", family: OrbitFamily::HaloNorth { lagrange: 2 }, amplitude: 0.020 },
        // OrbitSpec { label: "l2_halo_south", family: OrbitFamily::HaloSouth { lagrange: 2 }, amplitude: 0.020 },
        // OrbitSpec { label: "l1_halo_north", family: OrbitFamily::HaloNorth { lagrange: 1 }, amplitude: 0.020 },

        // ── Distant Retrograde Orbit (distance from Moon) ────────────────
        OrbitSpec { label: "dro_015",       family: OrbitFamily::Dro,                       amplitude: 0.150 },
        // OrbitSpec { label: "dro_020",    family: OrbitFamily::Dro,                       amplitude: 0.200 },
    ]
}

// ── Corrector settings ───────────────────────────────────────────────────────
const TOL:    f64   = 1e-10;
const MAX_IT: usize = 40;

// ╚══════════════════════════════════════════════════════════════╝
//  END OF CONFIGURATION
// ╚══════════════════════════════════════════════════════════════╝

fn main() {
    let p  = CrtbpParams::earth_moon();
    let mu = p.mu;

    std::fs::create_dir_all("out/orbits").unwrap();

    let specs = orbit_specs();
    println!("=== Finding {} periodic orbit(s) ===", specs.len());
    println!("  μ   = {:.8}", mu);
    println!("  L*  = {:.0} km,  T* = {:.4} days", p.l_star / 1e3, p.t_star / 86_400.0);

    let mut manifest = String::new();
    writeln!(manifest, "label,family,lagrange,amplitude,period_nd,period_days,jacobi,ic_x,ic_y,ic_z,ic_vy").unwrap();

    let mut found: Vec<(String, FoundOrbit)> = Vec::new();

    for spec in &specs {
        println!("\n── {} ({:?}, amp = {:.4}) ──", spec.label, spec.family, spec.amplitude);

        let orbit = find_orbit(mu, &spec.family, spec.amplitude);

        println!("  IC:     x={:.8}  vy={:.8}  z={:.8}",
            orbit.ic[0], orbit.ic[4], orbit.ic[2]);
        println!("  Period: {:.6} nd  ({:.2} days)",
            orbit.period, p.dim_time_days(orbit.period));
        println!("  Jacobi: {:.8}", orbit.jacobi);

        let traj = full_period_traj(mu, &orbit);
        let csv  = format!("out/orbits/{}.csv", spec.label);
        save_traj_csv(&csv, &traj, spec.label);

        let (lagrange, fam_str) = family_info(&spec.family);
        writeln!(manifest,
            "{},{},{},{:.4},{:.8},{:.4},{:.8},{:.10},{:.10},{:.10},{:.10}",
            spec.label, fam_str, lagrange, spec.amplitude,
            orbit.period, p.dim_time_days(orbit.period), orbit.jacobi,
            orbit.ic[0], orbit.ic[1], orbit.ic[2], orbit.ic[4]).unwrap();

        found.push((spec.label.to_string(), orbit));
    }

    std::fs::write("out/orbits/manifest.csv", &manifest).expect("manifest write failed");
    println!("\n  Saved out/orbits/manifest.csv ({} entries)", found.len());

    // Jacobi drift check
    println!("\n── Jacobi conservation check ──");
    for (label, orbit) in &found {
        let traj = full_period_traj(mu, orbit);
        let drift = traj.iter()
            .map(|s| (jacobi_constant(mu, &[s.x, s.y, s.vx, s.vy]) - orbit.jacobi).abs())
            .fold(0.0_f64, f64::max);
        println!("  {label}: max |ΔC| = {drift:.2e}");
    }

    println!("\nOutput: out/orbits/");
    println!("Launching plot/plot_orbits.py ...");
    std::process::Command::new("python")
        .arg("plot/plot_orbits.py")
        .spawn()
        .ok();
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn find_orbit(mu: f64, family: &OrbitFamily, amplitude: f64) -> FoundOrbit {
    match family {
        OrbitFamily::Lyapunov  { lagrange } => lyapunov(mu, *lagrange, amplitude, TOL, MAX_IT),
        OrbitFamily::HaloNorth { lagrange } => halo(mu, *lagrange, amplitude, true,  TOL, MAX_IT),
        OrbitFamily::HaloSouth { lagrange } => halo(mu, *lagrange, amplitude, false, TOL, MAX_IT),
        OrbitFamily::Dro                   => dro(mu, amplitude, TOL, MAX_IT),
    }
}

fn family_info(family: &OrbitFamily) -> (u8, &'static str) {
    match family {
        OrbitFamily::Lyapunov  { lagrange } => (*lagrange, "Lyapunov"),
        OrbitFamily::HaloNorth { lagrange } => (*lagrange, "HaloNorth"),
        OrbitFamily::HaloSouth { lagrange } => (*lagrange, "HaloSouth"),
        OrbitFamily::Dro                   => (0,          "DRO"),
    }
}

fn save_traj_csv(path: &str, traj: &[Step3d], label: &str) {
    let mut out = String::with_capacity(traj.len() * 80);
    writeln!(out, "time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,label").unwrap();
    for s in traj {
        writeln!(out, "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{label}",
            s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz).unwrap();
    }
    std::fs::write(path, &out).expect("csv write failed");
    println!("  Saved {path} ({} rows)", traj.len());
}
