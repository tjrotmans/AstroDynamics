//! plot_manifolds — compute and visualise invariant manifolds of a periodic orbit.
//!
//! Edit the config block below to choose the orbit and manifold parameters, then:
//!
//!   cargo run -p lunar_trajectories --bin plot_manifolds
//!
//! Output: out/manifolds/orbit.csv, unstable.csv, stable.csv
//! Plot:   auto-opens  out/manifolds/manifolds.html

use std::fmt::Write;
use std::f64::consts::PI;

use lunar_trajectories::crtbp::CrtbpParams;
use lunar_trajectories::linalg::{mat_vec6, norm6};
use lunar_trajectories::manifolds::{
    monodromy, manifold_branches, unstable_eigenvec, stable_eigenvec,
    ManifoldBranch, ManifoldParams,
};
use lunar_trajectories::periodic_orbits::{
    lyapunov, halo, dro, full_period_traj, FoundOrbit, OrbitFamily,
};
use lunar_trajectories::propagator::Step3d;

// ╔══════════════════════════════════════════════════════════════╗
// ║                    USER CONFIGURATION                        ║
// ║                                                              ║
// ║  orbit_config() → which orbit to use                        ║
// ║  MANIFOLD_* constants → how many branches, how far, etc.    ║
// ╚══════════════════════════════════════════════════════════════╝

/// Return (family, amplitude) for the orbit whose manifolds to compute.
///
/// Examples:
///   Lyapunov L2 (Ax=0.010)  →  (OrbitFamily::Lyapunov { lagrange: 2 }, 0.010)
///   Halo L2 north (Az=0.020) →  (OrbitFamily::HaloNorth { lagrange: 2 }, 0.020)
///   DRO (r=0.15)             →  (OrbitFamily::Dro, 0.150)
fn orbit_config() -> (OrbitFamily, f64) {
    (OrbitFamily::Lyapunov { lagrange: 2 }, 0.010)
}

// ── Manifold parameters ──────────────────────────────────────────────────────

/// Number of sample points around the orbit.  Total branches = 2 × N_BRANCHES
/// (N forward + N backward propagations).
const N_BRANCHES: usize = 20;

/// Perturbation size along the eigenvector [nd].  Smaller = more accurate,
/// but don't go below ~1e-7 (numerical noise floor of the eigenvector).
const EPSILON: f64 = 1e-6;

/// Integration time for each manifold branch [nd].
/// 2.5π ≈ half a synodic revolution; increase to see longer branches.
const T_MAN: f64 = 6.0 * PI;

// ── Corrector tolerance ──────────────────────────────────────────────────────
const TOL:    f64   = 1e-10;
const MAX_IT: usize = 40;

// ╚══════════════════════════════════════════════════════════════╝
//  END OF CONFIGURATION
// ╚══════════════════════════════════════════════════════════════╝

fn main() {
    let p  = CrtbpParams::earth_moon();
    let mu = p.mu;

    std::fs::create_dir_all("out/manifolds").unwrap();

    let (family, amplitude) = orbit_config();
    println!("=== Manifolds: {:?}  (amp = {:.4}) ===", family, amplitude);

    // ── Find orbit ───────────────────────────────────────────────────────────
    let orbit = find_orbit(mu, &family, amplitude);
    println!("  Period: {:.6} nd  ({:.2} days)", orbit.period, p.dim_time_days(orbit.period));
    println!("  Jacobi: {:.8}", orbit.jacobi);
    println!("  IC:     x={:.8}  vy={:.8}  z={:.8}", orbit.ic[0], orbit.ic[4], orbit.ic[2]);

    let traj = full_period_traj(mu, &orbit);
    save_traj_csv("out/manifolds/orbit.csv", &traj, "orbit");

    // ── Monodromy matrix ─────────────────────────────────────────────────────
    let mono    = monodromy(mu, &orbit);
    let v_u     = unstable_eigenvec(&mono);
    let _v_s    = stable_eigenvec(&mono);
    let lam_u   = norm6(&mat_vec6(&mono, &v_u));
    println!("\n  λ_unstable ≈ {lam_u:.4}  (λ_stable ≈ {:.6})", 1.0 / lam_u);

    // ── Manifold branches ────────────────────────────────────────────────────
    let params = ManifoldParams {
        n_branches: N_BRANCHES,
        epsilon:    EPSILON,
        t_man:      T_MAN,
        log_dt:     0.005,
        rtol:       1e-10,
        atol:       1e-12,
    };
    println!("\n  Computing {N_BRANCHES}×2 branches  (t_man = {:.2}π) …", T_MAN / PI);
    let (unstable, stable) = manifold_branches(mu, &orbit, &mono, &params);
    println!("  Unstable branches: {}", unstable.len());
    println!("  Stable  branches:  {}", stable.len());

    save_manifold_csv("out/manifolds/unstable.csv", &unstable);
    save_manifold_csv("out/manifolds/stable.csv",   &stable);

    // Write a small metadata file so the Python script can display it
    let mut meta = String::new();
    writeln!(meta, "family={:?}", family).unwrap();
    writeln!(meta, "amplitude={amplitude}").unwrap();
    writeln!(meta, "period_nd={:.8}", orbit.period).unwrap();
    writeln!(meta, "period_days={:.4}", p.dim_time_days(orbit.period)).unwrap();
    writeln!(meta, "jacobi={:.8}", orbit.jacobi).unwrap();
    writeln!(meta, "lambda_unstable={lam_u:.6}").unwrap();
    writeln!(meta, "n_branches={N_BRANCHES}").unwrap();
    writeln!(meta, "t_man={T_MAN:.6}").unwrap();
    std::fs::write("out/manifolds/meta.txt", meta).expect("meta write failed");

    println!("\nOutput: out/manifolds/");
    println!("Launching plot/plot_manifolds.py ...");
    std::process::Command::new("python")
        .arg("plot/plot_manifolds.py")
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

fn save_manifold_csv(path: &str, branches: &[ManifoldBranch]) {
    let total: usize = branches.iter().map(|b| b.steps.len()).sum();
    let mut out = String::with_capacity(total * 80);
    writeln!(out, "time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,branch").unwrap();
    for (i, branch) in branches.iter().enumerate() {
        for s in &branch.steps {
            writeln!(out, "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{i}",
                s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz).unwrap();
        }
    }
    std::fs::write(path, &out).expect("csv write failed");
    println!("  Saved {path} ({total} rows, {} branches)", branches.len());
}
