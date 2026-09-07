//! improve_transfers — refine the top-N transfer candidates produced by
//! find_transfers into fully continuous Earth → target-orbit trajectories.
//!
//! Strategy
//! ────────
//! find_transfers leaves a position/velocity gap at the patch point because its
//! 3×3 free-section LM corrector works in finite-difference Jacobian space and
//! can stall for poor seeds.
//!
//! This binary treats the pre-integrated departure arc as a "manifold branch"
//! (unstable side) and calls the Mode-B EOM-tangent corrector
//! (`refine_manifold_crossing`), which drives the full 6D state residual to
//! near machine precision for same-Jacobi trajectories.
//!
//! Usage:   cargo run -p lunar_trajectories --bin improve_transfers
//! Input:   out/transfers/candidates.csv   (written by find_transfers)
//! Output:  out/transfers/improved_*.csv   + info printed to stdout

use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;

use lunar_trajectories::crtbp::CrtbpParams;
use lunar_trajectories::manifolds::{manifold_branches, ManifoldBranch, ManifoldParams};
use lunar_trajectories::periodic_orbits::{lyapunov, halo, dro, full_period_traj, FoundOrbit, OrbitFamily};
use lunar_trajectories::propagator::{Step3d, propagate_3d};
use lunar_trajectories::transfers::{
    find_arc_poincare_crossings, injection_ic,
    refine_manifold_crossing_tracked, interp_branch_pub,
};

// ╔══════════════════════════════════════════════════════════════╗
// ║  CONFIG — keep in sync with find_transfers.rs               ║
// ╚══════════════════════════════════════════════════════════════╝

fn orbit_config() -> (OrbitFamily, f64, u8) {
    (
        OrbitFamily::Lyapunov { lagrange: 1 },
        0.060_f64,   // target_amp
        1_u8,        // gate (L1 = 1, L2 = 2)
    )
}

const N_BRANCHES: usize = 40;
const T_MAN:      f64   = 2.0 * PI;
const T_END:      f64   = 5.0 * PI;
const MAX_IT:     usize = 200;
const TOL:        f64   = 1e-10;
const TOP_N:      usize = 5;

// ─────────────────────────────────────────────────────────────────────────────

fn find_orbit(mu: f64, family: &OrbitFamily, amp: f64) -> FoundOrbit {
    match family {
        OrbitFamily::Lyapunov  { lagrange } => lyapunov(mu, *lagrange, amp, TOL, MAX_IT),
        OrbitFamily::HaloNorth { lagrange } => halo(mu, *lagrange, amp, true,  TOL, MAX_IT),
        OrbitFamily::HaloSouth { lagrange } => halo(mu, *lagrange, amp, false, TOL, MAX_IT),
        OrbitFamily::Dro                   => dro(mu, amp, TOL, MAX_IT),
    }
}

fn save_csv(path: &str, traj: &[Step3d], label: &str) {
    let mut out = String::with_capacity(traj.len() * 80);
    writeln!(out, "time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,label").unwrap();
    for s in traj {
        writeln!(out, "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{label}",
            s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz).unwrap();
    }
    std::fs::write(path, &out).expect("csv write failed");
    println!("    Saved {path} ({} rows)", traj.len());
}

// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    let mu = CrtbpParams::earth_moon().mu;
    let p  = CrtbpParams::earth_moon();

    let (target_family, target_amp, _gate) = orbit_config();

    // ── Recompute target orbit and stable manifold ────────────────────────────
    let target = find_orbit(mu, &target_family, target_amp);
    println!("Target orbit: {:?}  amp={target_amp}  C={:.6}  T={:.2} days",
        target_family, target.jacobi, p.dim_time_days(target.period));

    let man_params = ManifoldParams {
        n_branches: N_BRANCHES,
        epsilon:    1e-6,
        t_man:      T_MAN,
        log_dt:     0.005,
        rtol:       1e-9,
        atol:       1e-11,
    };
    let mono = lunar_trajectories::manifolds::monodromy(mu, &target);
    let (_, stable) = manifold_branches(mu, &target, &mono, &man_params);
    println!("Stable branches: {}", stable.len());

    // ── Load candidates ───────────────────────────────────────────────────────
    let cands_path = "out/transfers/candidates.csv";
    let cands_txt  = match std::fs::read_to_string(cands_path) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("[!] {cands_path} not found — run find_transfers first.");
            return;
        }
    };

    #[derive(Debug)]
    struct Candidate {
        rank:       usize,
        theta_rad:  f64,
        r_park:     f64,
        branch_idx: usize,
        t_s:        f64,
        residual:   f64,
        x_patch:    f64,
    }

    let mut candidates: Vec<Candidate> = Vec::new();
    for line in cands_txt.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 8 { continue; }
        candidates.push(Candidate {
            rank:       cols[0].trim().parse().unwrap_or(0),
            theta_rad:  cols[1].trim().parse().unwrap_or(0.0),
            r_park:     cols[2].trim().parse().unwrap_or(0.1),
            branch_idx: cols[3].trim().parse().unwrap_or(0),
            t_s:        cols[4].trim().parse().unwrap_or(0.0),
            residual:   cols[5].trim().parse().unwrap_or(f64::INFINITY),
            x_patch:    cols[7].trim().parse().unwrap_or(0.8),
        });
    }

    if candidates.is_empty() {
        eprintln!("[!] No candidates found in {cands_path}.");
        return;
    }

    println!("\nLoaded {} candidate(s) — refining top {}:\n", candidates.len(), TOP_N);
    println!("  {:>4}  {:>8}  {:>8}  {:>6}  {:>10}  {:>10}  {}",
        "rank", "θ_in°", "r_park", "br", "res_in_nd", "res_out_nd", "conv");

    std::fs::create_dir_all("out/transfers").ok();

    for cand in candidates.iter().take(TOP_N) {
        let branch_idx = cand.branch_idx;
        if branch_idx >= stable.len() {
            println!("  {:>4}  branch {} out of range — skip", cand.rank, branch_idx);
            continue;
        }
        let branch_s = &stable[branch_idx];

        // ── Pre-integrate departure arc ───────────────────────────────────────
        let c_orbit = target.jacobi;
        let ic = match injection_ic(mu, -mu, cand.r_park, c_orbit, cand.theta_rad) {
            Some(v) => v,
            None    => {
                println!("  {:>4}  injection_ic returned None — skip", cand.rank);
                continue;
            }
        };
        let arc_steps = propagate_3d(mu, ic, T_END, 0.005, 1e-9, 1e-9);

        // Wrap arc as a ManifoldBranch so we can pass it to refine_manifold_crossing.
        let arc_branch = ManifoldBranch { steps: arc_steps };

        // ── Seed t_u: first arc crossing of x = x_patch ──────────────────────
        let x_patch = cand.x_patch;
        let crossings = find_arc_poincare_crossings(&arc_branch.steps, x_patch, 0.0);
        let t_u0 = match crossings.first() {
            Some(c) => c.time,
            None    => {
                // Fall back: find the arc step closest to x_patch
                arc_branch.steps.iter()
                    .min_by(|a, b| (a.x - x_patch).abs().partial_cmp(&(b.x - x_patch).abs()).unwrap())
                    .map(|s| s.time)
                    .unwrap_or(T_END * 0.5)
            }
        };
        let t_s0 = cand.t_s;

        // ── Mode-B EOM-tangent corrector (with history for convergence plot) ──
        let mut history: Vec<(f64, f64, f64)> = Vec::new();
        let refined = refine_manifold_crossing_tracked(
            &arc_branch, branch_s,
            t_u0, t_s0, mu, MAX_IT, TOL,
            &mut history,
        );

        println!("  {:>4}  {:>8.2}  {:>8.5}  {:>6}  {:>10.4e}  {:>10.4e}  {}",
            cand.rank,
            cand.theta_rad.to_degrees(),
            cand.r_park,
            branch_idx,
            cand.residual,
            refined.residual,
            if refined.converged { "YES" } else { "no" },
        );

        if !refined.converged && refined.residual > 1.0 {
            println!("         [!] Residual {:.3e} nd — gap still large.", refined.residual);
        }

        // ── Save original (pre-improvement) arc trimmed to initial patch ─────
        let rank = cand.rank;
        let orig_trimmed: Vec<Step3d> = arc_branch.steps.iter()
            .take_while(|s| s.time <= t_u0)
            .cloned()
            .collect();
        save_csv(&format!("out/transfers/improved_arc_orig_{rank}.csv"), &orig_trimmed, "orig_arc");

        // ── Build improved arc trimmed to refined patch time ──────────────────
        let t_trim = refined.t_u;
        let arc_trimmed: Vec<Step3d> = arc_branch.steps.iter()
            .take_while(|s| s.time <= t_trim)
            .cloned()
            .collect();

        // ── Stable arc: manifold from patch (t_s) forward to orbit (t=0) ─────
        let mut stable_seg: Vec<Step3d> = branch_s.steps.iter()
            .filter(|s| s.time >= refined.t_s)
            .cloned()
            .collect();
        stable_seg.reverse();   // forward-time order: patch → orbit

        // ── Orbit continuation: propagate from end of stable arc for 4 periods
        let continuation: Vec<Step3d> = if let Some(end) = stable_seg.last() {
            let end_state = [end.x, end.y, end.z, end.vx, end.vy, end.vz];
            let t_cont    = 4.0 * target.period;
            let time_off  = end.time;
            let mut cont  = propagate_3d(mu, end_state, t_cont, 0.005, 1e-9, 1e-9);
            for s in &mut cont { s.time += time_off; }
            cont
        } else {
            vec![]
        };

        let target_traj = full_period_traj(mu, &target);

        save_csv(&format!("out/transfers/improved_arc_{rank}.csv"),          &arc_trimmed, "departure_arc");
        save_csv(&format!("out/transfers/improved_stable_{rank}.csv"),       &stable_seg,  "stable_arc");
        save_csv(&format!("out/transfers/improved_continuation_{rank}.csv"), &continuation,"continuation");
        save_csv(&format!("out/transfers/improved_orbit_{rank}.csv"),        &target_traj, "target_orbit");

        // ── Convergence history CSV ───────────────────────────────────────────
        {
            let mut conv_out = String::with_capacity(history.len() * 80);
            writeln!(conv_out, "iter,x_arc,y_arc,x_man,y_man,residual_nd").unwrap();
            for (i, &(t_u_i, t_s_i, res_i)) in history.iter().enumerate() {
                let arc_state = interp_branch_pub(&arc_branch, t_u_i).unwrap_or([0.0; 6]);
                let man_state = interp_branch_pub(branch_s, t_s_i).unwrap_or([0.0; 6]);
                writeln!(conv_out, "{},{:.8},{:.8},{:.8},{:.8},{:.6e}",
                    i, arc_state[0], arc_state[1], man_state[0], man_state[1], res_i).unwrap();
            }
            let conv_path = format!("out/transfers/improved_convergence_{rank}.csv");
            std::fs::write(&conv_path, &conv_out).expect("convergence csv write failed");
            println!("    Saved {conv_path} ({} iterations)", history.len());
        }

        // ── Summary info ──────────────────────────────────────────────────────
        let res_km      = refined.residual * p.l_star / 1e3;
        let res_km_orig = cand.residual    * p.l_star / 1e3;
        let dv_km       = refined.delta_v_mag * p.v_star / 1e3;
        let mut info = String::new();
        writeln!(info, "rank={rank}").unwrap();
        writeln!(info, "theta_deg={:.4}", cand.theta_rad.to_degrees()).unwrap();
        writeln!(info, "r_park={:.8}", cand.r_park).unwrap();
        writeln!(info, "branch_idx={branch_idx}").unwrap();
        writeln!(info, "t_u={:.8}", refined.t_u).unwrap();
        writeln!(info, "t_s={:.8}", refined.t_s).unwrap();
        writeln!(info, "residual_nd_orig={:.4e}", cand.residual).unwrap();
        writeln!(info, "residual_km_orig={:.2}",  res_km_orig).unwrap();
        writeln!(info, "residual_nd={:.4e}", refined.residual).unwrap();
        writeln!(info, "residual_km={:.4}",  res_km).unwrap();
        writeln!(info, "dv_patch_km_s={:.6}", dv_km).unwrap();
        writeln!(info, "converged={}", refined.converged).unwrap();
        std::fs::write(
            format!("out/transfers/improved_info_{rank}.txt"), info,
        ).ok();
    }

    println!("\nDone.  Results in out/transfers/improved_*");
    println!("Plot:  python plot/plot_improved.py");
}
