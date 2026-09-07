//! find_transfers — design transfers in the Earth-Moon CRTBP.
//!
//! Two modes (select in `transfer_config()` below):
//!
//!   A. EarthToOrbit      — optimise the injection angle on a circular parking
//!                          orbit to minimise the closest-approach distance to
//!                          the stable manifold of the target orbit.  A capture
//!                          ΔV is then applied at the patch point.
//!
//!   B. ManifoldIntersect — find the minimum-cost (|Δr_yz| + |ΔV|) connection
//!                          between the unstable manifold of a source orbit and
//!                          the stable manifold of a target orbit by sweeping
//!                          a range of Poincaré sections automatically.
//!
//! Usage:   cargo run -p lunar_trajectories --bin find_transfers
//! Output:  out/transfers/  (CSVs + info.txt)
//! Plot:    auto-opens  out/transfers/transfers.html

use std::fmt::Write;
use std::f64::consts::PI;

use lunar_trajectories::crtbp::CrtbpParams;
use lunar_trajectories::manifolds::{manifold_branches, ManifoldBranch, ManifoldParams};
use lunar_trajectories::periodic_orbits::{
    lyapunov, halo, dro, full_period_traj, FoundOrbit, OrbitFamily,
};
use lunar_trajectories::propagator::{Step3d, propagate_3d};
use lunar_trajectories::transfers::{
    c_lagrange, omega_nd,
    find_poincare_crossings, find_arc_poincare_crossings, injection_ic,
    optimize_injection_poincare,
    find_best_transfer_section, refine_manifold_crossing,
    refine_arc_free_section, RefinedArc,
};

// ╔══════════════════════════════════════════════════════════════╗
// ║                    USER CONFIGURATION                        ║
// ║                                                              ║
// ║  1. Choose a mode in transfer_config().                      ║
// ║  2. Uncomment the desired block and edit the parameters.     ║
// ╚══════════════════════════════════════════════════════════════╝

#[allow(dead_code)]
enum TransferMode {
    /// Optimised ballistic launch: Earth parking orbit → target orbit.
    ///
    /// Sweeps injection angles and uses a Poincaré section to find the pair
    /// (arc crossing, stable manifold crossing) with the smallest 6D gap.
    /// Newton refinement then drives both the position AND velocity gap to
    /// zero at the section — giving a physically smooth connection.
    EarthToOrbit {
        target_family: OrbitFamily,
        target_amp:    f64,
        /// Parking orbit radius from Earth centre [nd].
        /// Earth-Moon = 1.0 nd.  Practical range: 0.05–0.30.
        r_park:        f64,
        /// Which Lagrange gateway to open (1 = L1, 2 = L2).
        gate:          u8,
        /// Kept for documentation; arc Jacobi is set to exact target Jacobi.
        margin:        f64,
        /// Integration time for each candidate arc [nd].
        t_end:         f64,
        /// Number of injection angles in the coarse grid search (default: 36).
        n_grid:        usize,
        /// Number of branches in the target's stable manifold.
        n_branches:    usize,
        /// Integration time for the stable manifold branches [nd].
        t_man:         f64,
        /// x-coordinate of the Poincaré matching section [nd].
        /// Choose a value between the parking orbit and the Lagrange point.
        /// Good default for L1: 0.65 (inside the Earth-Moon neck).
        x_section:     f64,
        /// Maximum accepted position gap |Δy| at the Poincaré section [km].
        /// Transfers with a larger gap are flagged as physically unacceptable.
        /// Set to f64::INFINITY to accept all results regardless of gap.
        max_gap_km:    f64,
    },

    /// Manifold intersection: source unstable → target stable, minimum cost.
    ///
    /// Sweeps Poincaré sections from x_min to x_max and returns the crossing
    /// pair that minimises  |Δr_yz| + |ΔV|  across all sections.
    ManifoldIntersect {
        source_family: OrbitFamily,
        source_amp:    f64,
        target_family: OrbitFamily,
        target_amp:    f64,
        /// x-range of Poincaré sections to try [nd].
        x_min:         f64,
        x_max:         f64,
        /// How many sections to sweep (more → better coverage, slower).
        n_sections:    usize,
        /// vy direction filter for unstable crossings (+1 or −1).
        vy_sign_u:     f64,
        /// vy direction filter for stable crossings (+1 or −1).
        vy_sign_s:     f64,
        /// Number of manifold branches per orbit.
        n_branches:    usize,
        /// Integration time per manifold branch [nd].
        t_man:         f64,
    },
}

fn transfer_config() -> TransferMode {
    // ── Mode A: Optimised Earth launch → L1 Lyapunov orbit ──────────────────
    TransferMode::EarthToOrbit {
        target_family: OrbitFamily::Lyapunov { lagrange: 1 },
        target_amp:    0.060,    // larger amp → manifold extends further toward Earth
        r_park:        0.1,     // ≈ 57 700 km from Earth
        gate:          1,        // open the L1 gateway
        margin:        0.005,    // informational only; arc uses exact target Jacobi
        t_end:         5.0 * PI, // long enough for arc to cross the section
        n_grid:        40,       // 5° coarse grid
        n_branches:    40,       // more branches → better manifold coverage
        t_man:         2.0 * PI, // avoid primary close-approach; still reaches x=0.65
        x_section:     0.8,     // Poincaré section: between Earth and L1
        max_gap_km:    10_000.0,  // reject if position gap at section > 1 000 km
    }

    // ── Mode B: L1 Lyapunov → L2 Lyapunov via manifold intersection ─────────
    // Run plot_manifolds for each orbit first to see where their manifolds go,
    // then set x_min/x_max to bracket the region where they overlap.
    //
    // TransferMode::ManifoldIntersect {
    //     source_family: OrbitFamily::Lyapunov { lagrange: 1 },
    //     source_amp:    0.010,
    //     target_family: OrbitFamily::Lyapunov { lagrange: 2 },
    //     target_amp:    0.010,
    //     x_min:         0.84,   // just past L1
    //     x_max:         1.15,   // just before L2
    //     n_sections:    40,
    //     vy_sign_u:    -1.0,
    //     vy_sign_s:     1.0,
    //     n_branches:    30,
    //     t_man:         4.0 * PI,
    // }
}

// ── Corrector tolerance ──────────────────────────────────────────────────────
const TOL:    f64   = 1e-9;
const MAX_IT: usize = 50;

// ╚══════════════════════════════════════════════════════════════╝
//  END OF CONFIGURATION
// ╚══════════════════════════════════════════════════════════════╝

fn main() {
    let p  = CrtbpParams::earth_moon();
    let mu = p.mu;

    std::fs::create_dir_all("out/transfers").unwrap();

    match transfer_config() {
        TransferMode::EarthToOrbit {
            target_family, target_amp,
            r_park, gate, margin, t_end, n_grid,
            n_branches, t_man, x_section, max_gap_km,
        } => run_earth_to_orbit(
            mu, &p,
            target_family, target_amp,
            r_park, gate, margin, t_end, n_grid,
            n_branches, t_man, x_section, max_gap_km,
        ),

        TransferMode::ManifoldIntersect {
            source_family, source_amp,
            target_family, target_amp,
            x_min, x_max, n_sections,
            vy_sign_u, vy_sign_s,
            n_branches, t_man,
        } => run_manifold_intersect(
            mu, &p,
            source_family, source_amp,
            target_family, target_amp,
            x_min, x_max, n_sections,
            vy_sign_u, vy_sign_s,
            n_branches, t_man,
        ),
    }

    println!("\nOutput: out/transfers/");
    println!("Launching plot/plot_transfers.py ...");
    std::process::Command::new("python")
        .arg("plot/plot_transfers.py")
        .spawn()
        .ok();
}

// ─── Mode A ───────────────────────────────────────────────────────────────────

fn run_earth_to_orbit(
    mu: f64, p: &CrtbpParams,
    target_family: OrbitFamily, target_amp: f64,
    r_park: f64, gate: u8, _margin: f64, t_end: f64, n_grid: usize,
    n_branches: usize, t_man: f64, x_section: f64, max_gap_km: f64,
) {
    println!("=== Mode A: Earth → Orbit  (gate = L{gate}) ===");

    // ── Find target orbit ────────────────────────────────────────────────────
    let target = find_orbit(mu, &target_family, target_amp);
    let c_orbit = target.jacobi;
    let c_gate  = c_lagrange(mu, gate);
    println!("Target: {:?}  amp = {target_amp}", target_family);
    println!("  Period: {:.2} days    Jacobi: {:.6}",
        p.dim_time_days(target.period), c_orbit);
    println!("  C_L{gate} = {c_gate:.6}    C_arc = {c_orbit:.6}  (margin = {:.3})",
        c_gate - c_orbit);

    // ── Compute stable manifold ──────────────────────────────────────────────
    println!("\n  Computing stable manifold ({n_branches} branches, t_man = {:.2}π) ...",
        t_man / PI);
    let mono = lunar_trajectories::manifolds::monodromy(mu, &target);
    let man_params = ManifoldParams {
        n_branches, epsilon: 1e-6, t_man, log_dt: 0.005, rtol: 1e-9, atol: 1e-11,
    };
    let (_, stable) = manifold_branches(mu, &target, &mono, &man_params);
    println!("  Stable branches: {}", stable.len());

    // ── Coarse grid search: injection angle vs Poincaré section ─────────────
    // For each θ, propagate arc with C = c_orbit to x = x_section.
    // Find the (θ, manifold branch) pair minimising |Δr| + |ΔV| at the section.
    println!("\n  Optimising injection angle ({n_grid} grid points) ...");
    let coarse = match optimize_injection_poincare(
        mu, r_park, c_orbit, t_end, &stable, x_section, n_grid,
    ) {
        Some(r) => r,
        None => {
            println!("  [!] No crossings at x = {x_section:.3}. Try adjusting x_section or t_man.");
            return;
        }
    };

    println!("  Coarse best angle: {:.1}°", coarse.theta_rad.to_degrees());
    println!("  |Δr| at section:   {:.4} nd    |ΔV| = {:.4} nd",
        coarse.delta_r, coarse.delta_v);

    // ── Refinement: Mode B-style (arc treated as a ManifoldBranch) ─────────────
    // The transfer arc and the stable manifold branch are both pre-integrated
    // trajectories.  We use the same EOM-tangent corrector as Mode B, sliding
    // (t_arc, t_s) along their respective trajectories to minimise |ΔS|.
    // This gives a physically correct Jacobian and avoids the ill-conditioned
    // φ-index parameterisation.
    println!("\n  Mode B-style refinement (arc as trajectory, EOM-tangent corrector) ...");

    // Manifold crossings at x_section — one per branch.
    let man_crossings = find_poincare_crossings(&stable, x_section, 0.0);
    println!("  Manifold crossings at section: {}", man_crossings.len());

    // Build multi-start seeds.
    // Arcs from a ~0.1 nd parking orbit only reach the L1-side section for a
    // narrow band of injection angles near the coarse best (~279°).  Spreading
    // seeds uniformly over [0°,360°] wastes every start.  Instead we:
    //   • vary theta within ±60° of the coarse best  (N_THETA values)
    //   • vary r_park across physically meaningful values  (N_R seeds)
    //   • use the coarse t_s_cross so the section stays at x ≈ x_section
    //   • also try nearby manifold branches (top N_BRANCHES from man_crossings)
    const N_THETA:    usize = 7;  // injection angle spread around coarse best
    const N_R:        usize = 5;  // parking orbit radius grid
    const N_BRANCHES: usize = 5;  // top manifold branches from coarse search

    let r_park_seeds: [f64; N_R] = [0.05, 0.07, r_park, 0.13, 0.17];
    let theta_spread  = PI / 3.0; // ±60°

    // Best N_BRANCHES branches from man_crossings (already sorted by coarse gap).
    let mut seen = std::collections::HashSet::new();
    let top_branches: Vec<(usize, f64)> = man_crossings.iter()
        .filter(|mc| seen.insert(mc.branch_idx))
        .take(N_BRANCHES)
        .map(|mc| (mc.branch_idx, mc.time))
        .collect();

    let mut seeds: Vec<(f64, f64, usize, f64)> = vec![   // (theta, r_park, branch, t_s)
        (coarse.theta_rad, r_park, coarse.branch_idx, coarse.t_s_cross),
    ];
    for (branch_idx, t_s_cross) in &top_branches {
        for j in 0..N_THETA {
            let frac = j as f64 / (N_THETA - 1) as f64;
            let th = coarse.theta_rad - theta_spread + frac * 2.0 * theta_spread;
            for &rp in &r_park_seeds {
                seeds.push((th, rp, *branch_idx, *t_s_cross));
            }
        }
    }

    // ── Free-section corrector: free parameters (θ, r_park, t_s) ─────────────
    // The section x = x_stable(t_s) slides with t_s, so both the parking orbit
    // radius and the matching location are optimised simultaneously.
    // c_arc = c_orbit (no δc) → a converged result is a zero-ΔV patch point.
    let mut all_results: Vec<(usize, RefinedArc)> = Vec::new();
    let mut best_result:  Option<RefinedArc> = None;
    let mut best_branch_s = coarse.branch_idx;

    println!("\n  Corrector multi-start ({} seeds):", seeds.len());
    println!("  {:>4}  {:>8}  {:>8}  {:>6}  {:>8}  {:>8}  {:>7}  {:>10}  {:>10}  {}",
        "seed", "θ_in°", "θ_out°", "br", "r_in_nd", "r_out_nd", "Δt_s%", "res_nd", "res_km", "conv");

    for (seed_idx, (theta_seed, r_park_seed, branch_idx_s, t_s_seed)) in seeds.into_iter().enumerate() {
        if branch_idx_s >= stable.len() { continue; }

        let br = &stable[branch_idx_s];
        let t_s_a = br.steps.first().map(|s| s.time).unwrap_or(0.0);
        let t_s_b = br.steps.last() .map(|s| s.time).unwrap_or(0.0);
        let branch_span = (t_s_b - t_s_a).abs().max(1e-12);

        let result = refine_arc_free_section(
            mu, r_park_seed, theta_seed, c_orbit,
            br, t_s_seed,
            t_end, MAX_IT, TOL,
        );

        let ts_change_pct = (result.t_s - t_s_seed).abs() / branch_span * 100.0;
        println!("  {:>4}  {:>8.2}  {:>8.2}  {:>6}  {:>8.5}  {:>8.5}  {:>6.1}%  {:>10.4e}  {:>10.1}  {}",
            seed_idx,
            theta_seed.to_degrees(),
            result.theta.to_degrees(),
            branch_idx_s,
            r_park_seed,
            result.r_park,
            ts_change_pct,
            result.residual,
            result.residual * p.l_star,
            if result.converged { "YES" } else { "no" },
        );

        if best_result.as_ref().map_or(true, |b: &RefinedArc| result.residual < b.residual) {
            best_branch_s = branch_idx_s;
            best_result   = Some(result.clone());
        }
        all_results.push((branch_idx_s, result));
    }

    // Save top-5 *distinct* candidates for improve_transfers.
    // Two results are considered the same solution if they are within 15° in
    // theta AND within 0.015 nd in r_park — keep only the best (lowest residual)
    // from each cluster so improve_transfers works on truly different trajectories.
    all_results.sort_by(|a, b| a.1.residual.partial_cmp(&b.1.residual).unwrap());
    let mut unique: Vec<&(usize, RefinedArc)> = Vec::new();
    'outer: for entry in &all_results {
        let res = &entry.1;
        for kept in &unique {
            let dth = (res.theta - kept.1.theta).abs()
                .min((2.0 * PI - (res.theta - kept.1.theta).abs()).abs());
            let dr  = (res.r_park - kept.1.r_park).abs();
            if dth < 15.0_f64.to_radians() && dr < 0.015 {
                continue 'outer; // duplicate — skip
            }
        }
        unique.push(entry);
        if unique.len() == 5 { break; }
    }
    {
        let mut cands = String::new();
        writeln!(cands, "rank,theta_rad,r_park,branch_idx,t_s,residual_nd,x_patch,converged").unwrap();
        for (rank, (bi, res)) in unique.iter().enumerate() {
            writeln!(cands, "{},{:.8},{:.8},{},{:.8},{:.6e},{:.8},{}",
                rank+1, res.theta, res.r_park, bi,
                res.t_s, res.residual, res.state_stable[0], res.converged).unwrap();
        }
        std::fs::write("out/transfers/candidates.csv", &cands).ok();
    }

    // Fallback to coarse result if every seed failed.
    let result = best_result.unwrap_or_else(|| {
        let sa = coarse.state_arc;
        let ss = coarse.state_stable;
        let dv = [ss[3]-sa[3], ss[4]-sa[4], ss[5]-sa[5]];
        RefinedArc {
            theta:        coarse.theta_rad,
            delta_c:      0.0,
            r_park,
            t_arc:        coarse.t_arc_cross,
            t_s:          coarse.t_s_cross,
            state_arc:    sa,
            state_stable: ss,
            delta_v:      dv,
            delta_v_mag:  (dv[0]*dv[0]+dv[1]*dv[1]+dv[2]*dv[2]).sqrt(),
            residual:     coarse.delta_r,
            converged:    false,
        }
    });

    let best_theta  = result.theta;
    let best_r_park = result.r_park;   // converged parking orbit radius
    // x of the patch point is determined by t_s on the manifold branch
    let x_patch = result.state_stable[0];

    // Re-propagate full arc at converged (θ, r_park) and trim to the patch crossing.
    let arc_full = {
        let ic = injection_ic(mu, -mu, best_r_park, c_orbit, best_theta)
            .unwrap_or([-mu + best_r_park, 0.0, 0.0, 0.0, 0.0, 0.0]);
        propagate_3d(mu, ic, t_end, 0.005, 1e-9, 1e-9)
    };
    let arc_section_crossing = find_arc_poincare_crossings(&arc_full, x_patch, 0.0)
        .into_iter().next();
    let (t_trim, patch_arc_state) = match arc_section_crossing {
        Some(c) => (c.time, c.state),
        None    => {
            let t = if result.t_arc > 1e-6 { result.t_arc }
                    else { arc_full.last().map(|s| s.time).unwrap_or(t_end) };
            (t, result.state_arc)
        }
    };
    let arc_trimmed: Vec<Step3d> = arc_full.into_iter()
        .take_while(|s| s.time <= t_trim)
        .collect();

    // Stable arc: keep branch from patch-time to orbit, then reverse for forward display.
    let mut stable_arc: Vec<Step3d> = if best_branch_s < stable.len() {
        stable[best_branch_s].steps.iter()
            .filter(|s| s.time >= result.t_s)
            .cloned()
            .collect()
    } else { vec![] };
    stable_arc.reverse();   // forward-time order: patch point → orbit

    let theta_deg     = best_theta.to_degrees();
    let res_nd        = result.residual;
    let res_km        = res_nd * p.l_star / 1e3;
    // Recompute ΔV from the correctly-interpolated patch arc state.
    let ss_vel        = &result.state_stable;
    let dv_patch_vec  = [ss_vel[3]-patch_arc_state[3], ss_vel[4]-patch_arc_state[4], ss_vel[5]-patch_arc_state[5]];
    let dv_patch_nd   = (dv_patch_vec[0].powi(2)+dv_patch_vec[1].powi(2)+dv_patch_vec[2].powi(2)).sqrt();
    let dv_patch_km_s = dv_patch_nd * p.v_star / 1e3;

    // ── Physical gap check ───────────────────────────────────────────────────
    let gap_accepted = res_km <= max_gap_km;
    println!("  Injection angle θ (parking orbit, from Earth-Moon line): {theta_deg:.1}°");
    println!("  |ΔS| at patch point: {res_nd:.4e} nd  ({res_km:.0} km)  [limit: {max_gap_km:.0} km  →  {}]",
        if gap_accepted { "ACCEPTED" } else { "REJECTED — gap too large" });
    println!("  |ΔV| at patch point: {dv_patch_nd:.6} nd  ({dv_patch_km_s:.4} km/s)");
    println!("  Converged: {}", result.converged);
    println!("  δc applied:          {:.4e} nd", result.delta_c);
    println!("  Arc steps:           {}", arc_trimmed.len());
    println!("  Stable arc steps:    {}", stable_arc.len());
    if !gap_accepted {
        println!("\n  [!] Transfer REJECTED: state gap {res_km:.0} km > limit {max_gap_km:.0} km.");
        println!("  [!] Try: smaller target_amp, longer t_man, finer n_grid, or raise max_gap_km.");
    }

    // ── Injection ΔV: circular parking orbit → transfer arc ─────────────────
    let x0  = -mu + best_r_park * best_theta.cos();
    let y0  =        best_r_park * best_theta.sin();
    let v   = (2.0 * omega_nd(mu, x0, y0) - c_orbit).max(0.0).sqrt();
    let v_c = ((1.0 - mu) / best_r_park).sqrt();
    let dvx_inj = (-v * best_theta.sin()) - (-v_c * best_theta.sin() + y0);
    let dvy_inj = ( v * best_theta.cos()) - ( v_c * best_theta.cos() - x0);
    let dv_inj_nd   = (dvx_inj*dvx_inj + dvy_inj*dvy_inj).sqrt();
    let dv_inj_km_s = dv_inj_nd * p.v_star / 1e3;
    let r_park_km   = best_r_park * p.l_star / 1e3;
    println!("  Parking orbit radius: {best_r_park:.5} nd  ({r_park_km:.0} km)");
    println!("  ΔV at injection:     {dv_inj_nd:.4} nd  ({dv_inj_km_s:.4} km/s)");

    // ── Extended arc: propagate patch point forward to verify arrival ─────────
    {
        let r_earth = {
            let dx = patch_arc_state[0] + mu;
            let dy = patch_arc_state[1];
            (dx*dx + dy*dy).sqrt()
        };
        let r_moon = {
            let dx = patch_arc_state[0] - (1.0 - mu);
            let dy = patch_arc_state[1];
            (dx*dx + dy*dy).sqrt()
        };
        // Only propagate if patch state is safely away from both primaries.
        if r_earth > 0.05 && r_moon > 0.01 {
            let ext_t   = 3.0 * target.period;
            let patch_t = arc_trimmed.last().map(|s| s.time).unwrap_or(0.0);
            let mut ext = propagate_3d(mu, patch_arc_state, ext_t, 0.005, 1e-9, 1e-9);
            for s in &mut ext { s.time += patch_t; }
            save_traj_csv("out/transfers/extended_arc.csv", &ext, "extended_arc");
        } else {
            println!("  [!] Patch point near primary — skipping extended arc.");
        }
    }

    // ── Save ─────────────────────────────────────────────────────────────────
    let traj_target = full_period_traj(mu, &target);
    save_traj_csv("out/transfers/target_orbit.csv",  &traj_target,   "arrival_orbit");
    save_traj_csv("out/transfers/transfer_arc.csv",  &arc_trimmed,   "departure_arc");
    save_traj_csv("out/transfers/stable_arc.csv",    &stable_arc,    "stable_arc");
    save_manifold_csv("out/transfers/stable_manifold.csv", &stable);

    let sa = &patch_arc_state;
    let ss = &result.state_stable;
    let mut patch = String::new();
    writeln!(patch, "side,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd").unwrap();
    writeln!(patch, "arc,{:.8},{:.8},{:.8},{:.8},{:.8},{:.8}",
        sa[0], sa[1], sa[2], sa[3], sa[4], sa[5]).unwrap();
    writeln!(patch, "stable,{:.8},{:.8},{:.8},{:.8},{:.8},{:.8}",
        ss[0], ss[1], ss[2], ss[3], ss[4], ss[5]).unwrap();
    std::fs::write("out/transfers/patch_point.csv", patch).expect("patch write failed");

    // ── Debug: save every grid-angle arc that crosses the section ─────────────
    {
        let mut buf = String::new();
        writeln!(buf, "time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,arc_idx").unwrap();
        let mut n_saved = 0usize;
        for i in 0..n_grid {
            let theta_i = 2.0 * PI * i as f64 / n_grid as f64;
            let ic = match injection_ic(mu, -mu, r_park, c_orbit, theta_i) {
                Some(v) => v, None => continue,
            };
            let arc = propagate_3d(mu, ic, t_end, 0.005, 1e-9, 1e-9);
            let crossings = find_arc_poincare_crossings(&arc, x_section, 0.0);
            if crossings.is_empty() { continue; }   // doesn't reach section — skip
            let t_stop = crossings[0].time;
            for s in arc.iter().take_while(|s| s.time <= t_stop) {
                writeln!(buf, "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{i}",
                    s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz).unwrap();
            }
            n_saved += 1;
        }
        std::fs::write("out/transfers/debug_departure.csv", &buf).ok();
        println!("  Saved out/transfers/debug_departure.csv ({n_saved}/{n_grid} arcs reach section)");
    }
    {
        let mut buf = String::new();
        writeln!(buf, "time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,branch_idx").unwrap();
        // Only save crossings where vx > 0 at the section (arc approaching from Earth
        // side, x < x_section in forward time).  Moon-side crossings (vx < 0) produce
        // long looping arcs that are irrelevant for Earth → L1 transfers.
        // Also deduplicate per branch: keep only the outermost crossing (most negative
        // time = first encountered from the far end) to avoid multi-crossing clutter.
        let mut seen_branches = std::collections::HashSet::new();
        // Sort by time ascending (most negative first) so we take the outermost crossing.
        let mut sorted_crossings = man_crossings.iter().collect::<Vec<_>>();
        sorted_crossings.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        for mc in &sorted_crossings {
            if mc.state[3] <= 0.0 { continue; }          // skip Moon-side crossings
            if !seen_branches.insert(mc.branch_idx) { continue; } // one arc per branch
            if mc.branch_idx >= stable.len() { continue; }
            let branch = &stable[mc.branch_idx];
            // Outer portion: far_end → section in forward physical time
            for s in branch.steps.iter().filter(|s| s.time <= mc.time).rev() {
                writeln!(buf, "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{}",
                    s.time.abs(), s.x, s.y, s.z, s.vx, s.vy, s.vz, mc.branch_idx).unwrap();
            }
        }
        let n_arr = seen_branches.len();
        std::fs::write("out/transfers/debug_arrival.csv", &buf).ok();
        println!("  Saved out/transfers/debug_arrival.csv ({n_arr} Earth-side manifold arcs)");
    }

    let mut info = String::new();
    writeln!(info, "mode=EarthToOrbit").unwrap();
    writeln!(info, "gate={gate}").unwrap();
    writeln!(info, "r_park={r_park}").unwrap();
    writeln!(info, "r_park_converged={best_r_park:.8}").unwrap();
    writeln!(info, "r_park_converged_km={r_park_km:.2}").unwrap();
    writeln!(info, "c_gate={c_gate:.6}").unwrap();
    writeln!(info, "c_arc={c_orbit:.6}").unwrap();
    writeln!(info, "theta_deg={theta_deg:.4}").unwrap();
    writeln!(info, "x_section={x_section:.4}").unwrap();
    writeln!(info, "x_patch={x_patch:.6}").unwrap();
    writeln!(info, "converged={}", result.converged).unwrap();
    writeln!(info, "residual_nd={res_nd:.3e}").unwrap();
    writeln!(info, "residual_km={res_km:.4}").unwrap();
    writeln!(info, "dv_inject_nd={dv_inj_nd:.8}").unwrap();
    writeln!(info, "dv_inject_km_s={dv_inj_km_s:.4}").unwrap();
    writeln!(info, "dv_capture_km_s={dv_patch_km_s:.4}").unwrap();
    writeln!(info, "target_period_nd={:.8}", target.period).unwrap();
    writeln!(info, "max_gap_km={max_gap_km:.0}").unwrap();
    writeln!(info, "gap_accepted={gap_accepted}").unwrap();
    std::fs::write("out/transfers/info.txt", info).expect("info write failed");
    println!("  Saved out/transfers/info.txt");
}

// ─── Mode B ───────────────────────────────────────────────────────────────────

fn run_manifold_intersect(
    mu: f64, p: &CrtbpParams,
    source_family: OrbitFamily, source_amp: f64,
    target_family: OrbitFamily, target_amp: f64,
    x_min: f64, x_max: f64, n_sections: usize,
    vy_sign_u: f64, vy_sign_s: f64,
    n_branches: usize, t_man: f64,
) {
    println!("=== Mode B: Manifold intersection ===");
    println!("  Source: {:?}  amp = {source_amp}", source_family);
    println!("  Target: {:?}  amp = {target_amp}", target_family);
    println!("  Poincaré sweep: x ∈ [{x_min:.4}, {x_max:.4}]  n = {n_sections}");

    let man_params = ManifoldParams {
        n_branches, epsilon: 1e-6, t_man, log_dt: 0.005, rtol: 1e-9, atol: 1e-11,
    };

    // Source — unstable manifold
    let source    = find_orbit(mu, &source_family, source_amp);
    let mono_src  = lunar_trajectories::manifolds::monodromy(mu, &source);
    let (unstable, _) = manifold_branches(mu, &source, &mono_src, &man_params);
    println!("\n  Source: {:.2} days  C = {:.6}", p.dim_time_days(source.period), source.jacobi);
    println!("  Unstable branches: {}", unstable.len());

    // Target — stable manifold
    let target    = find_orbit(mu, &target_family, target_amp);
    let mono_tgt  = lunar_trajectories::manifolds::monodromy(mu, &target);
    let (_, stable) = manifold_branches(mu, &target, &mono_tgt, &man_params);
    println!("\n  Target: {:.2} days  C = {:.6}", p.dim_time_days(target.period), target.jacobi);
    println!("  Stable branches: {}", stable.len());

    // Multi-section sweep
    println!("\n  Sweeping {n_sections} Poincaré sections ...");
    let best = find_best_transfer_section(
        &unstable, &stable,
        x_min, x_max, n_sections,
        vy_sign_u, vy_sign_s,
    );

    // Save orbits and manifolds
    save_traj_csv("out/transfers/source_orbit.csv", &full_period_traj(mu, &source), "source");
    save_traj_csv("out/transfers/target_orbit.csv", &full_period_traj(mu, &target), "target");
    save_manifold_csv("out/transfers/unstable.csv", &unstable);
    save_manifold_csv("out/transfers/stable.csv",   &stable);

    let mut info = String::new();
    writeln!(info, "mode=ManifoldIntersect").unwrap();
    writeln!(info, "x_min={x_min}").unwrap();
    writeln!(info, "x_max={x_max}").unwrap();
    writeln!(info, "n_sections={n_sections}").unwrap();
    writeln!(info, "source_jacobi={:.6}", source.jacobi).unwrap();
    writeln!(info, "target_jacobi={:.6}", target.jacobi).unwrap();

    match best {
        None => {
            println!("  [!] No crossings found in sweep.");
            println!("  Try: longer t_man, wider x range, or flip vy_sign.");
            writeln!(info, "transfer_found=false").unwrap();
        }
        Some(xfer) => {
            let dv_coarse_km_s = xfer.dv_mag * p.v_star / 1e3;
            println!("\n  Coarse transfer:");
            println!("    |Δr_yz|  = {:.4} nd  ({:.0} km)",
                xfer.delta_pos, xfer.delta_pos * p.l_star / 1e3);
            println!("    |ΔV|     = {:.4} nd  ({dv_coarse_km_s:.4} km/s)", xfer.dv_mag);

            // ── Newton refinement: close full 6D state gap ───────────────
            println!("\n  Newton refinement (closing Δr and ΔV simultaneously) ...");
            let refined = refine_manifold_crossing(
                &unstable[xfer.branch_idx_u],
                &stable  [xfer.branch_idx_s],
                xfer.t_u, xfer.t_s,
                mu, MAX_IT, TOL,
            );

            let dv_km_s   = refined.delta_v_mag * p.v_star / 1e3;
            let res_km    = refined.residual * p.l_star / 1e3;
            println!("    |ΔV| refined: {:.6} nd  ({dv_km_s:.4} km/s)", refined.delta_v_mag);
            println!("    Residual |ΔS|: {:.2e} nd  ({res_km:.3} km)", refined.residual);
            println!("    Converged: {}", refined.converged);

            // Trim arcs at refined times
            let arc_u = unstable[xfer.branch_idx_u].steps.iter()
                .filter(|s| s.time <= refined.t_u).cloned().collect::<Vec<_>>();
            let arc_s = stable[xfer.branch_idx_s].steps.iter()
                .filter(|s| s.time >= refined.t_s).cloned().collect::<Vec<_>>();

            save_traj_csv("out/transfers/arc_unstable.csv", &arc_u, "unstable_arc");
            save_traj_csv("out/transfers/arc_stable.csv",   &arc_s, "stable_arc");

            writeln!(info, "transfer_found=true").unwrap();
            writeln!(info, "converged={}", refined.converged).unwrap();
            writeln!(info, "residual_nd={:.3e}", refined.residual).unwrap();
            writeln!(info, "residual_km={:.4}", res_km).unwrap();
            writeln!(info, "dv_mag_nd={:.8}", refined.delta_v_mag).unwrap();
            writeln!(info, "dv_mag_km_s={dv_km_s:.4}").unwrap();
        }
    }

    std::fs::write("out/transfers/info.txt", info).expect("info write failed");
    println!("  Saved out/transfers/info.txt");
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
