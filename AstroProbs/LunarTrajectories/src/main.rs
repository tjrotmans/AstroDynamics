//! LunarTrajectories — Earth-Moon trajectory design library.
//!
//! This workspace contains **two independent branches** for Earth-Moon transfers:
//!
//! ═══════════════════════════════════════════════════════════════════════════════
//! BRANCH 1 — WSB (Weak-Stability-Boundary) ballistic lunar transfers
//! ═══════════════════════════════════════════════════════════════════════════════
//! Dynamics: BCR4BP (Earth-Moon-Sun, 4-body bicircular rotating frame).
//! Method:   Global search → GA optimisation → MC refinement → max-fidelity
//!           propagation → real-ephemeris continuation.
//!
//! **Entry point: `src/bin/wsb_pipeline.rs`**
//! Run with:  `cargo run -p lunar_trajectories --bin wsb_pipeline --release`
//!
//! Individual WSB binaries (src/bin/wsb/):
//!   wsb_search              Phase 1-4 global seed search (backward/forward screening)
//!   wsb_optimize            Island genetic algorithm over (θ, θ_sun, r_apogee)
//!   wsb_refine              Monte Carlo local polish (1 000 samples per seed)
//!   wsb_maxhifi             Maximum-fidelity Dopri5 repropagation
//!   wsb_circularize         Lunar orbit insertion (LOI) burn + ECI frame conversion
//!   wsb_continuation_corrected  Homotopy BCR4BP → real ANISE ephemeris, λ=0→1
//!   wsb_sensitivity         Micro-perturbation ensemble (200 trajectories)
//!   wsb_sensitivity_individual  One-at-a-time parameter sweep (4×50)
//!   wsb_stats               20 000-sample sigma-sweep Monte Carlo
//!   wsb_basin               Capture-basin grid sweep (θ × θ_sun)
//!   wsb_dense_traj          10 diverse solutions at maximum time resolution
//!
//! Python visualisation (plot/):
//!   `python plot/wsb_plots.py`  — master orchestrator for all WSB plots
//!   See individual plot_wsb_*.py scripts for specific outputs.
//!
//! ═══════════════════════════════════════════════════════════════════════════════
//! BRANCH 2 — Heteroclinic/homoclinic connections (invariant manifold transfers)
//! ═══════════════════════════════════════════════════════════════════════════════
//! Dynamics: Pure 3-D CRTBP (Earth-Moon two-body rotating frame, no Sun).
//! Method:   Differential correction for periodic orbits → monodromy matrix
//!           eigenanalysis → manifold branch shooting → Poincaré section matching.
//!
//! **Entry point: this binary (`lunar_traj`)** — reference/demonstration.
//! Run with:  `cargo run -p lunar_trajectories --bin lunar_traj`
//! Plot:      `python plot/plot_comparison.py`
//!
//! Individual heteroclinic binaries (src/bin/heteroclinic/):
//!   find_orbits             Compute and save periodic orbit families
//!   find_transfers          Design manifold intersection transfers via Poincaré sections
//!   improve_transfers       Newton-based refinement of transfer candidates
//!   plot_manifolds          Visualise invariant manifolds of a target orbit
//!
//! ═══════════════════════════════════════════════════════════════════════════════
//! Shared library (src/lib.rs):
//!   crtbp         CRTBP EoM, Jacobi constant, Lagrange points, normalisation
//!   propagator    RK45 integrators (2-D, 3-D, BCR4BP, with optional STM)
//!   periodic_orbits  Differential correctors for Lyapunov / Halo / DRO families
//!   manifolds     Monodromy matrix, unstable/stable eigenvectors, branch shooting
//!   transfers     Poincaré section matching, manifold stitching, WSB injection ICs
//!   linalg        6-D vector / matrix helpers
//! ═══════════════════════════════════════════════════════════════════════════════

use lunar_trajectories::crtbp::{CrtbpParams, jacobi_constant, lyapunov_frequency};
use lunar_trajectories::linalg::{mat_vec6, norm6};
use lunar_trajectories::manifolds::{monodromy, manifold_branches, ManifoldParams};
use lunar_trajectories::periodic_orbits::{lyapunov, halo, dro, full_period_traj};
use lunar_trajectories::propagator::Step3d;
use lunar_trajectories::transfers::{earth_to_orbit, c_lagrange, find_poincare_crossings};

fn main() {
    std::fs::create_dir_all("out").unwrap();

    let p  = CrtbpParams::earth_moon();
    let mu = p.mu;

    println!("=== Earth-Moon CRTBP (3-D) ===");
    println!("  μ     = {:.8}",  mu);
    println!("  L*    = {:.0} km",  p.l_star / 1e3);
    println!("  T*    = {:.4} days", p.t_star / 86_400.0);
    println!("  V*    = {:.4} km/s", p.v_star / 1e3);
    println!("  C_L1  = {:.6}", c_lagrange(mu, 1));
    println!("  C_L2  = {:.6}", c_lagrange(mu, 2));

    // ── 1. L2 Lyapunov orbit ─────────────────────────────────────────────────
    let ax      = 0.010;
    let omega_l2 = lyapunov_frequency(mu, lunar_trajectories::crtbp::lagrange_x(mu, 2));
    println!("\n── L2 Lyapunov orbit (Ax = {:.4}) ──", ax);
    println!("  Linear period ≈ {:.2} days",
        p.dim_time_days(2.0 * std::f64::consts::PI / omega_l2));

    let lyap_l2 = lyapunov(mu, 2, ax, 1e-10, 40);
    println!("  IC:     {:?}", &lyap_l2.ic);
    println!("  Period: {:.6} ({:.2} days)", lyap_l2.period, p.dim_time_days(lyap_l2.period));
    println!("  Jacobi: {:.8}", lyap_l2.jacobi);

    let lyap_traj = full_period_traj(mu, &lyap_l2);
    let dj_max = lyap_traj.iter()
        .map(|s| (jacobi_constant(mu, &[s.x, s.y, s.vx, s.vy]) - lyap_l2.jacobi).abs())
        .fold(0.0_f64, f64::max);
    println!("  Jacobi drift (max): {:.2e}", dj_max);
    save_csv("out/l2_lyapunov.csv", &lyap_traj);

    // ── 2. L1 Lyapunov orbit ─────────────────────────────────────────────────
    println!("\n── L1 Lyapunov orbit (Ax = {:.4}) ──", ax);
    let lyap_l1 = lyapunov(mu, 1, ax, 1e-10, 40);
    println!("  Period: {:.6} ({:.2} days)", lyap_l1.period, p.dim_time_days(lyap_l1.period));
    println!("  Jacobi: {:.8}", lyap_l1.jacobi);
    save_csv("out/l1_lyapunov.csv", &full_period_traj(mu, &lyap_l1));

    // ── 3. L2 Halo orbit ─────────────────────────────────────────────────────
    println!("\n── L2 Halo orbit (Az = 0.020, north) ──");
    let halo_l2 = halo(mu, 2, 0.020, true, 1e-10, 40);
    println!("  IC:     {:?}", &halo_l2.ic);
    println!("  Period: {:.6} ({:.2} days)", halo_l2.period, p.dim_time_days(halo_l2.period));
    println!("  Jacobi: {:.8}", halo_l2.jacobi);
    save_csv("out/l2_halo_north.csv", &full_period_traj(mu, &halo_l2));

    // ── 4. DRO ───────────────────────────────────────────────────────────────
    println!("\n── Distant Retrograde Orbit (r = 0.15 from Moon) ──");
    let dro_orb = dro(mu, 0.15, 1e-10, 40);
    println!("  IC:     {:?}", &dro_orb.ic);
    println!("  Period: {:.6} ({:.2} days)", dro_orb.period, p.dim_time_days(dro_orb.period));
    println!("  Jacobi: {:.8}", dro_orb.jacobi);
    save_csv("out/dro.csv", &full_period_traj(mu, &dro_orb));

    // ── 5. Manifolds of L2 Lyapunov ─────────────────────────────────────────
    println!("\n── L2 Lyapunov invariant manifolds ──");
    let mono = monodromy(mu, &lyap_l2);

    let v_u  = lunar_trajectories::manifolds::unstable_eigenvec(&mono);
    let _v_s = lunar_trajectories::manifolds::stable_eigenvec(&mono);
    let lambda_u = norm6(&mat_vec6(&mono, &v_u));
    println!("  λ_unstable ≈ {:.4}  (λ_stable ≈ {:.6})", lambda_u, 1.0/lambda_u);

    let man_params = ManifoldParams {
        n_branches: 15,
        epsilon:    1e-6,
        t_man:      2.5 * std::f64::consts::PI,
        log_dt:     0.005,
        rtol:       1e-10,
        atol:       1e-12,
    };
    let (unstable, stable) = manifold_branches(mu, &lyap_l2, &mono, &man_params);
    println!("  Unstable branches: {}", unstable.len());
    println!("  Stable  branches:  {}", stable.len());
    save_manifold_csv("out/unstable_manifold.csv", &unstable);
    save_manifold_csv("out/stable_manifold.csv",   &stable);

    // ── 6. Direct transfer ───────────────────────────────────────────────────
    println!("\n── Direct Earth → Moon transfer ──");
    let xfer = earth_to_orbit(mu, 0.15, 1, 0.005,
                              3.0 * std::f64::consts::PI, p.v_star);
    println!("  Jacobi     = {:.4}   (C_L1 = {:.4})", xfer.jacobi, c_lagrange(mu, 1));
    println!("  Inj. speed ≈ {:.3} km/s", xfer.v_inject_km_s);
    println!("  Arc length : {} steps", xfer.arc.len());
    save_csv("out/direct_transfer.csv", &xfer.arc);

    // ── 7. Poincaré crossings at x = x_L1 (manifold transfer setup) ─────────
    let x_l1 = lunar_trajectories::crtbp::lagrange_x(mu, 1);
    println!("\n── Poincaré section at x = {:.6} (L1) ──", x_l1);
    let u_cross = find_poincare_crossings(&unstable, x_l1, -1.0);  // approaching L1
    let s_cross = find_poincare_crossings(&stable,   x_l1,  1.0);  // leaving  L1
    println!("  Unstable crossings: {}", u_cross.len());
    println!("  Stable  crossings:  {}", s_cross.len());

    println!("\n=== Output files ===");
    for f in &[
        "out/l2_lyapunov.csv", "out/l1_lyapunov.csv",
        "out/l2_halo_north.csv", "out/dro.csv",
        "out/unstable_manifold.csv", "out/stable_manifold.csv",
        "out/direct_transfer.csv",
    ] {
        println!("  {}", f);
    }
    println!("\nTo visualise: python plot/plot_comparison.py");
}

// ─── CSV helpers ──────────────────────────────────────────────────────────────

fn save_csv(path: &str, traj: &[Step3d]) {
    use std::fmt::Write as W;
    let mut out = String::with_capacity(traj.len() * 80);
    writeln!(out, "time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,branch").unwrap();
    for s in traj {
        writeln!(out, "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},0",
            s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz).unwrap();
    }
    std::fs::write(path, &out).expect("write failed");
    println!("  Saved {path} ({} rows)", traj.len());
}

fn save_manifold_csv(path: &str, branches: &[lunar_trajectories::manifolds::ManifoldBranch]) {
    use std::fmt::Write as W;
    let total: usize = branches.iter().map(|b| b.steps.len()).sum();
    let mut out = String::with_capacity(total * 80);
    writeln!(out, "time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,branch").unwrap();
    for (i, branch) in branches.iter().enumerate() {
        for s in &branch.steps {
            writeln!(out, "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{i}",
                s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz).unwrap();
        }
    }
    std::fs::write(path, &out).expect("write failed");
    println!("  Saved {path} ({total} rows across {} branches)", branches.len());
}
