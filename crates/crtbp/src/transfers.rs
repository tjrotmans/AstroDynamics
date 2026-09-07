//! Transfer trajectory design for the Earth-Moon CRTBP.
//!
//! # Modes
//!
//! ## Baseline — `earth_to_orbit`
//! Single fixed injection from the x-axis.
//!
//! ## Injection-angle optimisation — `optimize_injection_angle`
//! Grid + ternary search over the parking-orbit injection angle θ to minimise
//! the closest-approach distance to the stable manifold.
//!
//! ## Newton injection refinement — `refine_injection`
//! After the coarse optimisation, refine (θ, t_arc, t_s) with a shooting
//! Newton step to drive both the position gap AND velocity gap to zero.
//! Convergence is guaranteed when C_arc = C_orbit exactly.
//!
//! ## Multi-section manifold search — `find_best_transfer_section`
//! Sweeps Poincaré sections and returns the minimum-cost (|Δr_yz| + |ΔV|)
//! manifold crossing pair.
//!
//! ## Newton manifold refinement — `refine_manifold_crossing`
//! Given two coarse crossing times (t_u, t_s), drives the full state residual
//! |S_u(t_u) − S_s(t_s)| to zero using Newton's method on the two time
//! parameters.  For same-Jacobi orbits this closes both position AND velocity
//! gaps simultaneously.

use std::f64::consts::PI;

use crate::crtbp::{lagrange_x, eom_3d};
use crate::linalg::{dot6, norm6, sub6};
use crate::manifolds::ManifoldBranch;
use crate::propagator::Step3d;

// ─── Effective potential & Jacobi helpers ─────────────────────────────────────

/// Ω(x, y) = (x²+y²)/2 + (1−μ)/R1 + μ/R2  (normalized units)
pub fn omega_nd(mu: f64, x: f64, y: f64) -> f64 {
    let r1 = ((x + mu).powi(2) + y * y).sqrt();
    let r2 = ((x - (1.0 - mu)).powi(2) + y * y).sqrt();
    0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2
}

/// Jacobi constant at a Lagrange point (C_Li = 2·Ω(x_Li, 0)).
pub fn c_lagrange(mu: f64, lagrange: u8) -> f64 {
    let x_l = lagrange_x(mu, lagrange);
    2.0 * omega_nd(mu, x_l, 0.0)
}

// ─── Baseline direct transfer ────────────────────────────────────────────────

/// Result of the fixed-axis direct transfer.
pub struct DirectTransfer {
    pub arc:           Vec<Step3d>,
    pub vy_inject:     f64,
    pub jacobi:        f64,
    pub v_inject_km_s: f64,
}

/// Compute a direct ballistic transfer, injecting on the x-axis with Jacobi
/// constant `c_lagrange(mu, gate) - margin`.
pub fn earth_to_orbit(
    mu:     f64,
    r_park: f64,
    gate:   u8,
    margin: f64,
    t_end:  f64,
    v_star: f64,
) -> DirectTransfer {
    use crate::propagator::propagate_3d;
    let x0     = -mu + r_park;
    let omega0 = omega_nd(mu, x0, 0.0);
    let c_tgt  = c_lagrange(mu, gate) - margin;
    let vy     = (2.0 * omega0 - c_tgt).max(0.0).sqrt();
    let ic: [f64; 6] = [x0, 0.0, 0.0, 0.0, vy, 0.0];
    let arc = propagate_3d(mu, ic, t_end, 0.01, 1e-9, 1e-11);
    DirectTransfer {
        arc,
        vy_inject: vy,
        jacobi:    2.0 * omega0 - vy * vy,
        v_inject_km_s: (vy + x0) * v_star / 1e3,
    }
}

// ─── Poincaré section ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct PoincareCrossing {
    pub branch_idx: usize,
    pub state:      [f64; 6],
    pub time:       f64,
}

pub fn find_poincare_crossings(
    branches:  &[ManifoldBranch],
    x_section: f64,
    vy_sign:   f64,
) -> Vec<PoincareCrossing> {
    let mut out = Vec::new();
    for (bi, branch) in branches.iter().enumerate() {
        for w in branch.steps.windows(2) {
            let s0 = &w[0];
            let s1 = &w[1];
            let dx0 = s0.x - x_section;
            let dx1 = s1.x - x_section;
            if dx0 * dx1 > 0.0 { continue; }
            let frac = dx0 / (dx0 - dx1);
            let interp = |a: f64, b: f64| a + frac * (b - a);
            let state = [
                x_section,
                interp(s0.y,  s1.y),
                interp(s0.z,  s1.z),
                interp(s0.vx, s1.vx),
                interp(s0.vy, s1.vy),
                interp(s0.vz, s1.vz),
            ];
            let time = interp(s0.time, s1.time);
            if state[4] * vy_sign >= 0.0 {
                out.push(PoincareCrossing { branch_idx: bi, state, time });
            }
        }
    }
    out
}

// ─── Manifold transfer (coarse) ───────────────────────────────────────────────

/// A transfer arc stitched from two manifold branches.
pub struct ManifoldTransfer {
    /// Unstable arc trimmed to patch point.
    pub arc_unstable:  Vec<Step3d>,
    /// Stable arc from patch point toward target orbit.
    pub arc_stable:    Vec<Step3d>,
    /// ΔV = v_stable − v_unstable [nd].
    pub dv:            [f64; 3],
    /// |ΔV| [nd].
    pub dv_mag:        f64,
    /// Position gap |(Δy, Δz)| at the section [nd].
    pub delta_pos:     f64,
    /// Branch index in the unstable manifold Vec.
    pub branch_idx_u:  usize,
    /// Branch index in the stable manifold Vec.
    pub branch_idx_s:  usize,
    /// Crossing time along the unstable branch [nd].
    pub t_u:           f64,
    /// Crossing time along the stable branch [nd].
    pub t_s:           f64,
}

/// Find the transfer minimising |Δr_yz| + |ΔV| at a single Poincaré section.
pub fn find_best_manifold_transfer(
    unstable_crossings: &[PoincareCrossing],
    stable_crossings:   &[PoincareCrossing],
    unstable_branches:  &[ManifoldBranch],
    stable_branches:    &[ManifoldBranch],
) -> Option<ManifoldTransfer> {
    best_crossing_pair(
        unstable_crossings, stable_crossings,
        unstable_branches,  stable_branches,
    )
}

/// Sweep Poincaré sections over [x_min, x_max] and return the globally
/// minimum-cost (|Δr_yz| + |ΔV|) manifold transfer.
pub fn find_best_transfer_section(
    unstable:   &[ManifoldBranch],
    stable:     &[ManifoldBranch],
    x_min:      f64,
    x_max:      f64,
    n_sections: usize,
    vy_sign_u:  f64,
    vy_sign_s:  f64,
) -> Option<ManifoldTransfer> {
    let mut best: Option<(f64, ManifoldTransfer)> = None;
    let dx = if n_sections > 1 {
        (x_max - x_min) / (n_sections - 1) as f64
    } else { 0.0 };

    for i in 0..n_sections {
        let x = x_min + i as f64 * dx;
        let u_cross = find_poincare_crossings(unstable, x, vy_sign_u);
        let s_cross = find_poincare_crossings(stable,   x, vy_sign_s);
        if u_cross.is_empty() || s_cross.is_empty() { continue; }

        if let Some(t) = best_crossing_pair(&u_cross, &s_cross, unstable, stable) {
            let cost = t.delta_pos + t.dv_mag;
            if best.as_ref().map_or(true, |(c, _)| cost < *c) {
                best = Some((cost, t));
            }
        }
    }
    best.map(|(_, t)| t)
}

fn best_crossing_pair(
    u_cross: &[PoincareCrossing],
    s_cross: &[PoincareCrossing],
    u_branches: &[ManifoldBranch],
    s_branches: &[ManifoldBranch],
) -> Option<ManifoldTransfer> {
    let mut best: Option<(f64, ManifoldTransfer)> = None;

    for uc in u_cross {
        for sc in s_cross {
            let dy = uc.state[1] - sc.state[1];
            let dz = uc.state[2] - sc.state[2];
            let dr_yz = (dy * dy + dz * dz).sqrt();

            let dv = [
                sc.state[3] - uc.state[3],
                sc.state[4] - uc.state[4],
                sc.state[5] - uc.state[5],
            ];
            let dv_mag = (dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2]).sqrt();
            let cost   = dr_yz + dv_mag;

            if best.as_ref().map_or(true, |(c, _)| cost < *c) {
                let arc_u = trim_branch(&u_branches[uc.branch_idx].steps, uc.time, false);
                let arc_s = trim_branch(&s_branches[sc.branch_idx].steps, sc.time, true);
                best = Some((cost, ManifoldTransfer {
                    arc_unstable: arc_u,
                    arc_stable:   arc_s,
                    dv,
                    dv_mag,
                    delta_pos:   dr_yz,
                    branch_idx_u: uc.branch_idx,
                    branch_idx_s: sc.branch_idx,
                    t_u:          uc.time,
                    t_s:          sc.time,
                }));
            }
        }
    }
    best.map(|(_, t)| t)
}

fn trim_branch(steps: &[Step3d], t_cross: f64, keep_after: bool) -> Vec<Step3d> {
    if keep_after {
        steps.iter().filter(|s| s.time >= t_cross).cloned().collect()
    } else {
        steps.iter().filter(|s| s.time <= t_cross).cloned().collect()
    }
}

// ─── Newton refinement: manifold crossing (Mode B) ───────────────────────────

/// Result of Newton refinement of a manifold crossing.
pub struct RefinedCrossing {
    /// Refined state on the unstable branch at the patch point [nd].
    pub state_u:     [f64; 6],
    /// Refined state on the stable  branch at the patch point [nd].
    pub state_s:     [f64; 6],
    /// Time along the unstable branch [nd].
    pub t_u:         f64,
    /// Time along the stable  branch [nd].
    pub t_s:         f64,
    /// Velocity gap ΔV = v_s − v_u [nd].
    pub delta_v:     [f64; 3],
    /// |ΔV| [nd].
    pub delta_v_mag: f64,
    /// Full residual |S_u − S_s| at convergence.  Zero for exact intersections.
    pub residual:    f64,
    /// Whether the corrector converged within `tol`.
    pub converged:   bool,
}

/// Refine a manifold crossing using Newton's method on the two branch times.
///
/// The Jacobian is `J = [ẋ_u | −ẋ_s]` (from the EOM), giving the 2×2 normal
/// equations  `(J^T J) [δt_u; δt_s] = −J^T R`.
///
/// For same-Jacobi orbits the two manifolds intersect and the full state
/// residual converges to numerical precision, closing both position AND
/// velocity gaps simultaneously.
pub fn refine_manifold_crossing(
    branch_u: &ManifoldBranch,
    branch_s: &ManifoldBranch,
    t_u0: f64, t_s0: f64,
    mu: f64, max_iter: usize, tol: f64,
) -> RefinedCrossing {
    refine_manifold_crossing_impl(branch_u, branch_s, t_u0, t_s0, mu, max_iter, tol, None)
}

/// Like `refine_manifold_crossing` but records `(t_u, t_s, residual)` at each
/// accepted iteration into `history` for visualisation.
pub fn refine_manifold_crossing_tracked(
    branch_u: &ManifoldBranch,
    branch_s: &ManifoldBranch,
    t_u0: f64, t_s0: f64,
    mu: f64, max_iter: usize, tol: f64,
    history: &mut Vec<(f64, f64, f64)>,
) -> RefinedCrossing {
    refine_manifold_crossing_impl(branch_u, branch_s, t_u0, t_s0, mu, max_iter, tol, Some(history))
}

fn refine_manifold_crossing_impl(
    branch_u: &ManifoldBranch,
    branch_s: &ManifoldBranch,
    t_u0:     f64,
    t_s0:     f64,
    mu:       f64,
    max_iter: usize,
    tol:      f64,
    mut history:  Option<&mut Vec<(f64, f64, f64)>>,
) -> RefinedCrossing {
    // Branches may be integrated forward OR backward, so first/last times may
    // be in either order.  Always produce (min, max) regardless of direction.
    let t_u_a = branch_u.steps.first().map(|s| s.time).unwrap_or(0.0);
    let t_u_b = branch_u.steps.last() .map(|s| s.time).unwrap_or(0.0);
    let t_u_min = t_u_a.min(t_u_b);
    let t_u_max = t_u_a.max(t_u_b);
    let t_s_a = branch_s.steps.first().map(|s| s.time).unwrap_or(0.0);
    let t_s_b = branch_s.steps.last() .map(|s| s.time).unwrap_or(0.0);
    let t_s_min = t_s_a.min(t_s_b);
    let t_s_max = t_s_a.max(t_s_b);

    let mut t_u = t_u0.clamp(t_u_min, t_u_max);
    let mut t_s = t_s0.clamp(t_s_min, t_s_max);
    let mut converged = false;

    // Record initial state before any correction.
    if let Some(ref mut h) = history.as_deref_mut() {
        if let (Some(su), Some(ss)) = (interp_branch(branch_u, t_u), interp_branch(branch_s, t_s)) {
            h.push((t_u, t_s, norm6(&sub6(&su, &ss))));
        }
    }

    for _ in 0..max_iter {
        let su = match interp_branch(branch_u, t_u) { Some(s) => s, None => break };
        let ss = match interp_branch(branch_s, t_s) { Some(s) => s, None => break };

        let r = sub6(&su, &ss);
        let res = norm6(&r);
        if res < tol { converged = true; break; }

        // EOM tangent vectors — J = [f_u | -f_s]
        let f_u = eom_3d(mu, &su);
        let f_s = eom_3d(mu, &ss);

        // Normal equations: (J^T J) δ = −J^T R
        let a00 =  dot6(&f_u, &f_u);
        let a01 = -dot6(&f_u, &f_s);
        let a11 =  dot6(&f_s, &f_s);
        let det  = a00 * a11 - a01 * a01;
        if det.abs() < 1e-20 { break; }

        let c0 = -dot6(&f_u, &r);
        let c1 =  dot6(&f_s, &r);

        let dt_u = (c0 * a11 - c1 * a01) / det;
        let dt_s = (c1 * a00 - c0 * a01) / det;

        // Step limiter: no more than 10% of each branch's time range
        let lim_u = 0.1 * (t_u_max - t_u_min).abs().max(1e-6);
        let lim_s = 0.1 * (t_s_max - t_s_min).abs().max(1e-6);
        let scale = (lim_u / dt_u.abs().max(1e-15))
                        .min(lim_s / dt_s.abs().max(1e-15))
                        .min(1.0);

        t_u = (t_u + scale * dt_u).clamp(t_u_min, t_u_max);
        t_s = (t_s + scale * dt_s).clamp(t_s_min, t_s_max);

        if let Some(ref mut h) = history.as_deref_mut() {
            let su2 = interp_branch(branch_u, t_u).unwrap_or([0.0;6]);
            let ss2 = interp_branch(branch_s, t_s).unwrap_or([0.0;6]);
            h.push((t_u, t_s, norm6(&sub6(&su2, &ss2))));
        }
    }

    let su = interp_branch(branch_u, t_u).unwrap_or([0.0; 6]);
    let ss = interp_branch(branch_s, t_s).unwrap_or([0.0; 6]);
    let r  = sub6(&su, &ss);
    let dv = [ss[3]-su[3], ss[4]-su[4], ss[5]-su[5]];

    RefinedCrossing {
        state_u:     su,
        state_s:     ss,
        t_u,
        t_s,
        delta_v:     dv,
        delta_v_mag: norm6(&[0.0, 0.0, 0.0, dv[0], dv[1], dv[2]]),
        residual:    norm6(&r),
        converged,
    }
}

// ─── Injection angle optimisation via Poincaré section (Mode A — coarse) ─────

/// A single crossing of the transfer arc with a Poincaré section.
#[derive(Clone, Debug)]
pub struct ArcCrossing {
    /// Interpolated 6-state at the section crossing [nd].
    pub state: [f64; 6],
    /// Arc time at the crossing [nd].
    pub time:  f64,
}

/// Find all crossings of `arc` with the plane x = `x_section`.
/// Optionally filter by sign of vy (`vy_sign` ≥ 0 keeps positive vy,
/// vy_sign = 0 keeps ALL crossings).
pub fn find_arc_poincare_crossings(
    arc:       &[Step3d],
    x_section: f64,
    vy_sign:   f64,   // 0 = any, +1 = vy≥0, −1 = vy≤0
) -> Vec<ArcCrossing> {
    let mut out = Vec::new();
    for w in arc.windows(2) {
        let s0 = &w[0];
        let s1 = &w[1];
        let dx0 = s0.x - x_section;
        let dx1 = s1.x - x_section;
        if dx0 * dx1 > 0.0 { continue; }
        let frac = dx0 / (dx0 - dx1);
        let interp = |a: f64, b: f64| a + frac * (b - a);
        let state = [
            x_section,
            interp(s0.y,  s1.y),
            interp(s0.z,  s1.z),
            interp(s0.vx, s1.vx),
            interp(s0.vy, s1.vy),
            interp(s0.vz, s1.vz),
        ];
        let time = interp(s0.time, s1.time);
        if vy_sign == 0.0 || state[4] * vy_sign >= 0.0 {
            out.push(ArcCrossing { state, time });
        }
    }
    out
}

/// Coarse result from the Poincaré-section injection search.
pub struct InjectionSectionResult {
    /// Best injection angle [rad].
    pub theta_rad:    f64,
    /// Arc time when it crosses the section [nd] — used as t_arc0 for Newton.
    pub t_arc_cross:  f64,
    /// Branch index in the stable manifold Vec.
    pub branch_idx:   usize,
    /// Manifold time at the section crossing [nd] — used as t_s0 for Newton.
    pub t_s_cross:    f64,
    /// Arc state at the section crossing [nd].
    pub state_arc:    [f64; 6],
    /// Manifold state at the section crossing [nd].
    pub state_stable: [f64; 6],
    /// Position gap |Δr| at the section [nd].
    pub delta_r:      f64,
    /// Velocity gap |ΔV| at the section [nd].
    pub delta_v:      f64,
    /// Combined cost (|Δr| + |ΔV|) at the section [nd].
    pub cost:         f64,
}

/// Grid search over injection angles θ, propagate each arc to a Poincaré
/// section x = `x_section`, and find the (θ, branch) pair that minimises
/// the position + velocity gap at the section.
///
/// This gives a much better starting point for Newton than the 3-D closest-
/// approach search, because both arcs are constrained to the same hyperplane.
pub fn optimize_injection_poincare(
    mu:        f64,
    r_park:    f64,
    c_target:  f64,
    t_end:     f64,
    stable:    &[ManifoldBranch],
    x_section: f64,
    n_grid:    usize,
) -> Option<InjectionSectionResult> {
    use crate::propagator::propagate_3d;
    let x_earth = -mu;

    // Pre-compute all stable manifold crossings at the section (any vy direction)
    let man_crossings = find_poincare_crossings(stable, x_section, 0.0);
    if man_crossings.is_empty() {
        return None;
    }

    let mut best: Option<(f64, InjectionSectionResult)> = None;

    for i in 0..n_grid {
        let theta = 2.0 * PI * i as f64 / n_grid as f64;
        let ic = match injection_ic(mu, x_earth, r_park, c_target, theta) {
            None    => continue,
            Some(v) => v,
        };
        let arc = propagate_3d(mu, ic, t_end, 0.005, 1e-10, 1e-12);
        let arc_crossings = find_arc_poincare_crossings(&arc, x_section, 0.0);

        for ac in &arc_crossings {
            for mc in &man_crossings {
                let dr  = sub6(&ac.state, &mc.state);
                let dr_pos = norm6(&[dr[0], dr[1], dr[2], 0., 0., 0.]);
                let dr_vel = norm6(&[0., 0., 0., dr[3], dr[4], dr[5]]);
                let cost   = dr_pos + dr_vel;
                if best.as_ref().map_or(true, |(c, _)| cost < *c) {
                    best = Some((cost, InjectionSectionResult {
                        theta_rad:    theta,
                        t_arc_cross:  ac.time,
                        branch_idx:   mc.branch_idx,
                        t_s_cross:    mc.time,
                        state_arc:    ac.state,
                        state_stable: mc.state,
                        delta_r:      dr_pos,
                        delta_v:      dr_vel,
                        cost,
                    }));
                }
            }
        }
    }

    best.map(|(_, r)| r)
}

// ─── Manifold tube injection: stable manifold ∩ parking orbit (Mode A) ───────

/// A point where the stable manifold crosses the parking orbit sphere.
///
/// The trajectory from this point, following the stable manifold forward in
/// time, reaches the target orbit with **zero ΔV** at the patch point.
/// The only maneuver needed is the injection ΔV at departure.
pub struct ManifoldInjectionPoint {
    /// Branch index in the stable manifold Vec.
    pub branch_idx:    usize,
    /// Time along the branch at the crossing [nd]  (negative for backward-
    /// integrated branches; corresponds to `orbit_time − |time|` in forward).
    pub time:          f64,
    /// 6-state at the crossing [nd].
    pub state:         [f64; 6],
    /// Injection angle θ = atan2(y, x − x_earth) on the parking orbit [rad].
    pub theta_rad:     f64,
    /// Injection ΔV vector (manifold vel − circular parking orbit vel) [nd].
    pub dv_inject:     [f64; 3],
    /// |injection ΔV| [nd].
    pub dv_inject_mag: f64,
}

/// Find all crossings of the stable manifold branches with a circular parking
/// orbit sphere of radius `r_park` centred on Earth.
///
/// Each crossing is an exact **zero-ΔV patch point**.  The injection ΔV is the
/// velocity difference between the manifold tangent and the prograde circular
/// orbit at that point.
pub fn find_manifold_injection_points(
    stable: &[ManifoldBranch],
    mu:     f64,
    r_park: f64,
) -> Vec<ManifoldInjectionPoint> {
    let x_earth = -mu;
    let v_circ  = ((1.0 - mu) / r_park).sqrt();   // inertial circular speed [nd]
    let mut out = Vec::new();

    for (bi, branch) in stable.iter().enumerate() {
        for w in branch.steps.windows(2) {
            let s0 = &w[0];
            let s1 = &w[1];
            let r0 = ((s0.x - x_earth).powi(2) + s0.y.powi(2) + s0.z.powi(2)).sqrt();
            let r1 = ((s1.x - x_earth).powi(2) + s1.y.powi(2) + s1.z.powi(2)).sqrt();
            if (r0 - r_park) * (r1 - r_park) > 0.0 { continue; }   // no crossing

            // Linear interpolation to the crossing
            let frac = (r_park - r0) / (r1 - r0);
            let interp = |a: f64, b: f64| a + frac * (b - a);
            let state = [
                interp(s0.x,  s1.x),  interp(s0.y,  s1.y),  interp(s0.z,  s1.z),
                interp(s0.vx, s1.vx), interp(s0.vy, s1.vy), interp(s0.vz, s1.vz),
            ];
            let t_cross = interp(s0.time, s1.time);

            let theta = state[1].atan2(state[0] - x_earth);   // angle on parking orbit

            // Prograde circular orbit velocity in the rotating frame at (x0, y0)
            // v_inertial = v_circ * (−sin θ, cos θ)
            // v_rotating = v_inertial − ω × r  where ω = ẑ
            //            = v_circ*(−sinθ, cosθ) − (−y0, x0)
            let x0   = state[0];
            let y0   = state[1];
            let vp_x = -v_circ * theta.sin() + y0;
            let vp_y =  v_circ * theta.cos() - x0;
            let vp_z = 0.0_f64;

            let dv = [state[3] - vp_x, state[4] - vp_y, state[5] - vp_z];
            let dv_mag = (dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2]).sqrt();

            out.push(ManifoldInjectionPoint {
                branch_idx:    bi,
                time:          t_cross,
                state,
                theta_rad:     theta,
                dv_inject:     dv,
                dv_inject_mag: dv_mag,
            });
        }
    }
    out
}

// ─── Newton injection refinement — Poincaré section (Mode A — fine) ─────────

/// Result of the Newton injection refinement.
pub struct RefinedInjection {
    /// Refined injection angle [rad].
    pub theta_rad:     f64,
    /// Actual arc Jacobi constant used (c_target + delta_c) [nd].
    pub c_arc:         f64,
    /// Energy perturbation applied by the corrector [nd].
    pub delta_c:       f64,
    /// Arc from injection to patch point.
    pub arc:           Vec<Step3d>,
    /// Stable branch from patch point toward target orbit.
    pub stable_arc:    Vec<Step3d>,
    /// State at patch point — transfer arc side [nd].
    pub state_arc:     [f64; 6],
    /// State at patch point — stable manifold side [nd].
    pub state_stable:  [f64; 6],
    /// ΔV = v_stable − v_arc [nd].  Ideally → 0.
    pub delta_v:       [f64; 3],
    /// |ΔV| [nd].
    pub delta_v_mag:   f64,
    /// 3D residual |(Δy, Δvx, Δvy)| at convergence [nd].
    pub residual:      f64,
    /// True if the corrector converged.
    pub converged:     bool,
}

/// Refine the injection using a **3×3 Poincaré-section Newton corrector**.
///
/// Three free parameters evaluated AT x = `x_section`:
///   - θ   : injection angle on the parking orbit → moves the arc crossing
///   - φ   : fractional index into `man_crossings` → moves the manifold crossing
///   - δc  : perturbation to the arc Jacobi constant → adjusts injection speed
///
/// `man_crossings` is the pre-computed, ordered list of stable-manifold crossings
/// of the section.  Interpolating between adjacent entries gives a continuous
/// mapping φ → (y_man, vx_man, vy_man) without re-propagating.
///
/// Residual:  R = [y_arc(θ,δc) − y_man(φ),  vx_arc(θ,δc) − vx_man(φ),  vy_arc(θ,δc) − vy_man(φ)]
///
/// Jacobian columns:
///   - ∂R/∂θ  : FD, one extra arc propagation per iteration.
///   - ∂R/∂φ  : free — just interpolation of the pre-computed crossings.
///   - ∂R/∂δc : FD, one extra arc propagation per iteration.
///
/// The 3×3 system is solved exactly each Newton step, closing the full
/// in-plane velocity gap (including vx) that the old 2×2 corrector left open.
pub fn refine_injection_poincare(
    mu:            f64,
    r_park:        f64,
    theta0:        f64,
    phi0:          f64,           // starting φ (fractional index into man_crossings)
    c_target:      f64,
    stable:        &[ManifoldBranch],
    man_crossings: &[PoincareCrossing],
    x_section:     f64,
    t_end:         f64,
    max_iter:      usize,
    tol:           f64,
) -> RefinedInjection {
    use crate::propagator::propagate_3d;
    let x_earth = -mu;
    let n_cross  = man_crossings.len();

    // Interpolate manifold crossing state at fractional index φ.
    let interp_man = |phi: f64| -> [f64; 6] {
        if n_cross == 0 { return [0.0; 6]; }
        let phi = phi.rem_euclid(n_cross as f64);
        let i0  = phi.floor() as usize;
        let i1  = (i0 + 1).min(n_cross - 1);
        let f   = phi.fract();
        let s0  = &man_crossings[i0].state;
        let s1  = &man_crossings[i1].state;
        let l   = |a: f64, b: f64| a + f * (b - a);
        [l(s0[0],s1[0]), l(s0[1],s1[1]), l(s0[2],s1[2]),
         l(s0[3],s1[3]), l(s0[4],s1[4]), l(s0[5],s1[5])]
    };

    // Arc: propagate from (θ, c), stopping as soon as x = x_section is crossed.
    // Uses propagate_3d_until_x to avoid integrating well past the section.
    let get_arc_c = |th: f64, c: f64| -> Option<(f64, [f64; 6])> {
        use crate::propagator::propagate_3d_until_x;
        let ic = injection_ic(mu, x_earth, r_park, c, th)?;
        propagate_3d_until_x(mu, ic, x_section, t_end, 1e-10, 1e-12)
    };

    let mut theta   = theta0;
    let mut phi     = phi0;
    let mut delta_c = 0.0_f64;
    let mut converged = false;

    let dth = 1e-4_f64;   // FD step for θ
    let dph = 1e-3_f64;   // FD step for φ (dimensionless index)
    let ddc = 1e-4_f64;   // FD step for δc

    // ── Levenberg-Marquardt ────────────────────────────────────────────────────
    // Each outer iteration: compute J and R at the current point (3 propagations),
    // then find an accepting λ by solving (JᵀJ + λI)δ = −JᵀR and checking whether
    // the residual decreases.  The Jacobian is reused across all λ trials for the
    // same outer step — no extra propagations on rejection.
    //
    // λ schedule: initialise as τ·max(diag(JᵀJ)), decrease on acceptance (→ Newton),
    // increase on rejection (→ gradient descent).
    let mut lambda = -1.0_f64;   // negative sentinel → initialised on first iteration
    const LM_UP:   f64 = 10.0;
    const LM_DOWN: f64 = 0.1;
    const LM_MAX:  f64 = 1e12;
    const LM_TAU:  f64 = 1e-3;  // initial λ = τ · max(diag(JᵀJ))

    for _ in 0..max_iter {
        let c_curr = c_target + delta_c;
        let (_, s_arc) = match get_arc_c(theta, c_curr) { None => break, Some(v) => v };
        let s_man      = interp_man(phi);

        // Residual R = [y_arc − y_man,  vx_arc − vx_man,  vy_arc − vy_man]
        let r0 = s_arc[1] - s_man[1];   // Δy
        let r1 = s_arc[3] - s_man[3];   // Δvx
        let r2 = s_arc[4] - s_man[4];   // Δvy
        let res = (r0*r0 + r1*r1 + r2*r2).sqrt();
        if res < tol { converged = true; break; }

        // Column 1 (θ): FD — one extra propagation
        let (_, s_th) = match get_arc_c(theta + dth, c_curr) { None => break, Some(v) => v };
        let j00 = (s_th[1] - s_arc[1]) / dth;   // ∂y/∂θ
        let j10 = (s_th[3] - s_arc[3]) / dth;   // ∂vx/∂θ
        let j20 = (s_th[4] - s_arc[4]) / dth;   // ∂vy/∂θ

        // Column 2 (φ): free — just interpolation, no propagation
        let s_mp = interp_man(phi + dph);
        let j01 = -(s_mp[1] - s_man[1]) / dph;  // −∂y_man/∂φ
        let j11 = -(s_mp[3] - s_man[3]) / dph;  // −∂vx_man/∂φ
        let j21 = -(s_mp[4] - s_man[4]) / dph;  // −∂vy_man/∂φ

        // Column 3 (δc): FD — one extra propagation
        let (_, s_dc) = match get_arc_c(theta, c_curr + ddc) { None => break, Some(v) => v };
        let j02 = (s_dc[1] - s_arc[1]) / ddc;   // ∂y/∂δc
        let j12 = (s_dc[3] - s_arc[3]) / ddc;   // ∂vx/∂δc
        let j22 = (s_dc[4] - s_arc[4]) / ddc;   // ∂vy/∂δc

        // Pre-compute JᵀJ (symmetric 3×3) and −JᵀR — reused for every λ trial.
        let jtj_00 = j00*j00 + j10*j10 + j20*j20;
        let jtj_11 = j01*j01 + j11*j11 + j21*j21;
        let jtj_22 = j02*j02 + j12*j12 + j22*j22;
        let jtj_01 = j00*j01 + j10*j11 + j20*j21;
        let jtj_02 = j00*j02 + j10*j12 + j20*j22;
        let jtj_12 = j01*j02 + j11*j12 + j21*j22;
        let neg_jtr = [
            -(j00*r0 + j10*r1 + j20*r2),
            -(j01*r0 + j11*r1 + j21*r2),
            -(j02*r0 + j12*r1 + j22*r2),
        ];

        // Initialise λ from the diagonal on the very first iteration.
        if lambda < 0.0 {
            lambda = LM_TAU * jtj_00.max(jtj_11).max(jtj_22).max(1e-10);
        }

        // Inner loop: try increasing λ until the step reduces the residual.
        let mut accepted = false;
        for _ in 0..20 {
            let a = [
                [jtj_00 + lambda, jtj_01,          jtj_02         ],
                [jtj_01,          jtj_11 + lambda,  jtj_12         ],
                [jtj_02,          jtj_12,           jtj_22 + lambda],
            ];
            let step = match solve_3x3(a, neg_jtr) {
                None => { lambda *= LM_UP; continue; }
                Some(d) => d,
            };

            let th_new = theta   + step[0];
            let ph_new = phi     + step[1];
            let dc_new = delta_c + step[2];

            let new_res = match get_arc_c(th_new, c_target + dc_new) {
                None => f64::INFINITY,
                Some((_, s_a)) => {
                    let s_m = interp_man(ph_new);
                    let e0 = s_a[1]-s_m[1]; let e1 = s_a[3]-s_m[3]; let e2 = s_a[4]-s_m[4];
                    (e0*e0 + e1*e1 + e2*e2).sqrt()
                }
            };

            if new_res < res {
                theta   = th_new;
                phi     = ph_new;
                delta_c = dc_new;
                lambda  = (lambda * LM_DOWN).max(1e-14);
                accepted = true;
                break;
            }
            lambda *= LM_UP;
            if lambda > LM_MAX { break; }
        }
        if !accepted { break; }
    }

    // ── Build final arcs at converged (θ, φ, δc) ─────────────────────────────
    // Use the full propagator here (not the early-stopping variant) so that we
    // always get a valid crossing state even when the LM did not converge.
    // This is only called once after the loop, so cost is one propagation.
    let c_final  = c_target + delta_c;
    let ic_final = injection_ic(mu, x_earth, r_park, c_final, theta)
        .unwrap_or([x_earth + r_park, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let full_arc = propagate_3d(mu, ic_final, t_end, 0.005, 1e-10, 1e-12);
    let (t_arc_final, s_arc) = find_arc_poincare_crossings(&full_arc, x_section, 0.0)
        .into_iter().next()
        .map(|c| (c.time, c.state))
        .unwrap_or_else(|| {
            // Arc never crossed the section — use the last integrated state.
            let last = full_arc.last()
                .map(|s| s.state())
                .unwrap_or([x_earth + r_park, 0.0, 0.0, 0.0, 0.0, 0.0]);
            (t_end, last)
        });
    // Trim to the crossing time.
    let arc_final: Vec<Step3d> = full_arc.into_iter()
        .take_while(|s| s.time <= t_arc_final)
        .collect();
    let s_man = interp_man(phi);

    // Stable arc: use the branch nearest to converged φ.
    let best_cross_idx = (phi.rem_euclid(n_cross as f64).round() as usize).min(n_cross.saturating_sub(1));
    let (branch_idx, t_s_cross) = if n_cross > 0 {
        (man_crossings[best_cross_idx].branch_idx,
         man_crossings[best_cross_idx].time)
    } else { (0, 0.0) };

    let stable_arc: Vec<Step3d> = if branch_idx < stable.len() {
        let branch = &stable[branch_idx];
        let mut v: Vec<Step3d> = branch.steps.iter()
            .filter(|s| s.time >= t_s_cross)
            .cloned()
            .collect();
        v.reverse();   // forward-time: patch point → orbit
        v
    } else { vec![] };

    let dv  = [s_man[3]-s_arc[3], s_man[4]-s_arc[4], s_man[5]-s_arc[5]];
    let res = ((s_arc[1]-s_man[1]).powi(2) + (s_arc[3]-s_man[3]).powi(2) + (s_arc[4]-s_man[4]).powi(2)).sqrt();

    RefinedInjection {
        theta_rad:    theta,
        c_arc:        c_final,
        delta_c,
        arc:          arc_final,
        stable_arc,
        state_arc:    s_arc,
        state_stable: s_man,
        delta_v:      dv,
        delta_v_mag:  norm6(&[0., 0., 0., dv[0], dv[1], dv[2]]),
        residual:     res,
        converged,
    }
}

// ─── Mode A refinement ────────────────────────────────────────────────────────

/// Result of the Mode A LM corrector.
#[derive(Clone)]
pub struct RefinedArc {
    /// Refined injection angle [rad].
    pub theta:        f64,
    /// Jacobi perturbation applied by the corrector [nd].
    pub delta_c:      f64,
    /// Converged parking orbit radius [nd].
    pub r_park:       f64,
    /// Arc time at the Poincaré section crossing [nd].
    pub t_arc:        f64,
    /// Time on the stable branch at the section crossing [nd].
    pub t_s:          f64,
    /// Arc state at the patch point [nd].
    pub state_arc:    [f64; 6],
    /// Manifold state at the patch point [nd].
    pub state_stable: [f64; 6],
    /// ΔV = v_stable − v_arc [nd].
    pub delta_v:      [f64; 3],
    /// |ΔV| [nd].
    pub delta_v_mag:  f64,
    /// Residual |(Δy, Δvx, Δvy)| at convergence [nd].
    pub residual:     f64,
    /// Whether the corrector converged.
    pub converged:    bool,
}

/// LM corrector for Mode A (Earth → orbit): free parameters (θ, δc).
///
/// The manifold crossing state `man_state` is fixed at x = `x_section`.
/// Only the injection angle θ and energy perturbation δc are varied to
/// bring the arc's section-crossing state to match the manifold state.
/// This guarantees the patch point stays on the Poincaré section.
///
/// Residual (overdetermined 3×2 system):
///   R = [y_arc − y_man,  vx_arc − vx_man,  vy_arc − vy_man]
///
/// Normal equations solved each step: (JᵀJ + λI) δ = −JᵀR  (2×2).
pub fn refine_arc_to_manifold(
    mu:        f64,
    r_park:    f64,
    c_target:  f64,
    theta0:    f64,
    man_state: [f64; 6],
    x_section: f64,
    t_end:     f64,
    max_iter:  usize,
    tol:       f64,
) -> RefinedArc {
    use crate::propagator::propagate_3d_until_x;

    let arc_at_section = |th: f64, c: f64| -> Option<(f64, [f64; 6])> {
        let ic = injection_ic(mu, -mu, r_park, c, th)?;
        propagate_3d_until_x(mu, ic, x_section, t_end, 1e-10, 1e-12)
    };

    let mut theta     = theta0;
    let mut delta_c   = 0.0_f64;
    let mut converged = false;

    let dth = 1e-4_f64;
    let ddc = 1e-4_f64;

    let mut lambda = -1.0_f64;
    const LM_UP:   f64 = 10.0;
    const LM_DOWN: f64 = 0.1;
    const LM_MAX:  f64 = 1e12;
    const LM_TAU:  f64 = 1e-3;

    let mut last_t_arc = 0.0_f64;
    let mut last_s_arc = [0.0_f64; 6];

    for _ in 0..max_iter {
        let c_curr = c_target + delta_c;
        let (t_arc, s_arc) = match arc_at_section(theta, c_curr) {
            None => break, Some(v) => v,
        };
        last_t_arc = t_arc;
        last_s_arc = s_arc;

        let r0 = s_arc[1] - man_state[1];
        let r1 = s_arc[3] - man_state[3];
        let r2 = s_arc[4] - man_state[4];
        let res = (r0*r0 + r1*r1 + r2*r2).sqrt();
        if res < tol { converged = true; break; }

        let (_, s_th) = match arc_at_section(theta + dth, c_curr) {
            None => break, Some(v) => v,
        };
        let j00 = (s_th[1] - s_arc[1]) / dth;
        let j10 = (s_th[3] - s_arc[3]) / dth;
        let j20 = (s_th[4] - s_arc[4]) / dth;

        let (_, s_dc) = match arc_at_section(theta, c_curr + ddc) {
            None => break, Some(v) => v,
        };
        let j01 = (s_dc[1] - s_arc[1]) / ddc;
        let j11 = (s_dc[3] - s_arc[3]) / ddc;
        let j21 = (s_dc[4] - s_arc[4]) / ddc;

        let jtj_00 = j00*j00 + j10*j10 + j20*j20;
        let jtj_11 = j01*j01 + j11*j11 + j21*j21;
        let jtj_01 = j00*j01 + j10*j11 + j20*j21;
        let neg_jtr = [
            -(j00*r0 + j10*r1 + j20*r2),
            -(j01*r0 + j11*r1 + j21*r2),
        ];

        if lambda < 0.0 {
            lambda = LM_TAU * jtj_00.max(jtj_11).max(1e-10);
        }

        let mut accepted = false;
        for _ in 0..20 {
            let a = [[jtj_00 + lambda, jtj_01], [jtj_01, jtj_11 + lambda]];
            let step = match solve_2x2(a, neg_jtr) {
                None => { lambda *= LM_UP; continue; }
                Some(d) => d,
            };
            let th_new = theta   + step[0];
            let dc_new = delta_c + step[1];

            let new_res = match arc_at_section(th_new, c_target + dc_new) {
                None => f64::INFINITY,
                Some((_, s_a)) => {
                    let e0 = s_a[1]-man_state[1];
                    let e1 = s_a[3]-man_state[3];
                    let e2 = s_a[4]-man_state[4];
                    (e0*e0 + e1*e1 + e2*e2).sqrt()
                }
            };

            if new_res < res {
                theta   = th_new;
                delta_c = dc_new;
                lambda  = (lambda * LM_DOWN).max(1e-14);
                accepted = true;
                break;
            }
            lambda *= LM_UP;
            if lambda > LM_MAX { break; }
        }
        if !accepted { break; }
    }

    let (t_arc_fin, s_arc_fin) = arc_at_section(theta, c_target + delta_c)
        .unwrap_or((last_t_arc, last_s_arc));

    let dv = [man_state[3]-s_arc_fin[3], man_state[4]-s_arc_fin[4], man_state[5]-s_arc_fin[5]];
    let dv_mag = (dv[0]*dv[0]+dv[1]*dv[1]+dv[2]*dv[2]).sqrt();
    let residual = {
        let e0 = s_arc_fin[1]-man_state[1];
        let e1 = s_arc_fin[3]-man_state[3];
        let e2 = s_arc_fin[4]-man_state[4];
        (e0*e0 + e1*e1 + e2*e2).sqrt()
    };

    RefinedArc {
        theta, delta_c, r_park,
        t_arc: t_arc_fin, t_s: 0.0,
        state_arc: s_arc_fin, state_stable: man_state,
        delta_v: dv, delta_v_mag: dv_mag,
        residual, converged,
    }
}

/// LM corrector for Mode A with a **free Poincaré section**: free parameters (θ, r_park, t_s).
///
/// Unlike `refine_arc_to_manifold`, the manifold crossing state is NOT fixed.
/// Instead, the manifold branch time `t_s` is a free parameter that slides the
/// meeting point along the branch — implicitly optimising the section position.
/// The arc Jacobi constant is held at `c_target` (no δc degree of freedom), so
/// a converged solution is a genuine zero-ΔV patch point.
///
/// Free parameters evaluated AT x = x_stable(t_s):
///   - θ     : injection angle on the parking orbit
///   - r_park: parking orbit radius [nd]
///   - t_s   : time along the stable branch (determines meeting x automatically)
///
/// Residual (3 equations, 3 unknowns → square Newton / LM):
///   R = [y_arc(θ,r) − y_man(t_s),  vx_arc − vx_man,  vy_arc − vy_man]
pub fn refine_arc_free_section(
    mu:       f64,
    r_park0:  f64,
    theta0:   f64,
    c_target: f64,
    branch:   &ManifoldBranch,
    t_s0:     f64,
    t_end:    f64,
    max_iter: usize,
    tol:      f64,
) -> RefinedArc {
    use crate::propagator::propagate_3d;

    let x_earth = -mu;

    // Branch time bounds (works for both ascending and descending time order).
    let t_s_a   = branch.steps.first().map(|s| s.time).unwrap_or(0.0);
    let t_s_b   = branch.steps.last() .map(|s| s.time).unwrap_or(0.0);
    let t_s_min = t_s_a.min(t_s_b);
    let t_s_max = t_s_a.max(t_s_b);

    // Evaluate stable state at time t_s, or return zeros on failure.
    let man_at = |ts: f64| -> [f64; 6] {
        interp_branch(branch, ts).unwrap_or([0.0; 6])
    };

    // Propagate arc (θ, r_park) to the section x = man_at(ts).x.
    // Uses the full logged propagator + interpolated crossing search rather than
    // propagate_3d_until_x: the early-stop variant can miss the crossing when the
    // crossing point falls between logged output intervals, causing the LM to see
    // None on its very first call and break before touching the parameters.
    let arc_at = |th: f64, r: f64, ts: f64| -> Option<(f64, [f64; 6])> {
        let x_sec = man_at(ts)[0];
        let ic    = injection_ic(mu, x_earth, r, c_target, th)?;
        let arc   = propagate_3d(mu, ic, t_end, 0.005, 1e-9, 1e-9);
        find_arc_poincare_crossings(&arc, x_sec, 0.0)
            .into_iter().next()
            .map(|c| (c.time, c.state))
    };

    // Residual R = [Δy, Δvx, Δvy] at the sliding section.
    let residual_vec = |th: f64, r: f64, ts: f64| -> Option<[f64; 3]> {
        let (_, sa) = arc_at(th, r, ts)?;
        let sm      = man_at(ts);
        Some([sa[1]-sm[1], sa[3]-sm[3], sa[4]-sm[4]])
    };

    let mut theta   = theta0;
    let mut r_park  = r_park0;
    let mut t_s     = t_s0.clamp(t_s_min, t_s_max);
    let mut converged = false;

    let dth = 1e-4_f64;
    let dr  = 1e-4_f64;   // 1e-5 was too small (~3.8 km) for reliable r_park Jacobian
    let dts = 1e-4_f64;

    // Per-step caps: limit each LM step to ≤10 % of the respective parameter range.
    let lim_th  = 0.10 * (2.0 * PI);
    let lim_r   = 0.10 * (0.20 - 0.04_f64).max(1e-6);
    let lim_ts  = 0.10 * (t_s_max - t_s_min).abs().max(1e-6);

    let mut lambda = -1.0_f64;
    const LM_UP:   f64 = 10.0;
    const LM_DOWN: f64 = 0.1;
    const LM_MAX:  f64 = 1e12;
    const LM_TAU:  f64 = 1e-3;

    let mut last_t_arc = 0.0_f64;
    let mut last_s_arc = [0.0_f64; 6];

    for _ in 0..max_iter {
        let (t_arc, s_arc) = match arc_at(theta, r_park, t_s) {
            None => break, Some(v) => v,
        };
        last_t_arc = t_arc;
        last_s_arc = s_arc;
        let s_man = man_at(t_s);

        let r0 = s_arc[1] - s_man[1];
        let r1 = s_arc[3] - s_man[3];
        let r2 = s_arc[4] - s_man[4];
        let res = (r0*r0 + r1*r1 + r2*r2).sqrt();
        if res < tol { converged = true; break; }

        // FD Jacobian — column θ (s_man unchanged, x_sec unchanged)
        let (_, s_th) = match arc_at(theta + dth, r_park, t_s) {
            None => break, Some(v) => v,
        };
        let j00 = (s_th[1] - s_arc[1]) / dth;
        let j10 = (s_th[3] - s_arc[3]) / dth;
        let j20 = (s_th[4] - s_arc[4]) / dth;

        // FD Jacobian — column r_park (s_man unchanged, x_sec unchanged)
        let (_, s_r) = match arc_at(theta, r_park + dr, t_s) {
            None => break, Some(v) => v,
        };
        let j01 = (s_r[1] - s_arc[1]) / dr;
        let j11 = (s_r[3] - s_arc[3]) / dr;
        let j21 = (s_r[4] - s_arc[4]) / dr;

        // FD Jacobian — column t_s (both arc x-stop AND s_man change)
        let r_pert = match residual_vec(theta, r_park, t_s + dts) {
            None => break, Some(v) => v,
        };
        let j02 = (r_pert[0] - r0) / dts;
        let j12 = (r_pert[1] - r1) / dts;
        let j22 = (r_pert[2] - r2) / dts;

        // JᵀJ (symmetric 3×3) and −JᵀR
        let jtj_00 = j00*j00 + j10*j10 + j20*j20;
        let jtj_11 = j01*j01 + j11*j11 + j21*j21;
        let jtj_22 = j02*j02 + j12*j12 + j22*j22;
        let jtj_01 = j00*j01 + j10*j11 + j20*j21;
        let jtj_02 = j00*j02 + j10*j12 + j20*j22;
        let jtj_12 = j01*j02 + j11*j12 + j21*j22;
        let neg_jtr = [
            -(j00*r0 + j10*r1 + j20*r2),
            -(j01*r0 + j11*r1 + j21*r2),
            -(j02*r0 + j12*r1 + j22*r2),
        ];

        if lambda < 0.0 {
            lambda = LM_TAU * jtj_00.max(jtj_11).max(jtj_22).max(1e-10);
        }

        let mut accepted = false;
        for _ in 0..20 {
            let a = [
                [jtj_00 + lambda, jtj_01,          jtj_02         ],
                [jtj_01,          jtj_11 + lambda,  jtj_12         ],
                [jtj_02,          jtj_12,           jtj_22 + lambda],
            ];
            let step = match solve_3x3(a, neg_jtr) {
                None => { lambda *= LM_UP; continue; }
                Some(d) => d,
            };

            let th_new = theta  + step[0].clamp(-lim_th, lim_th);
            let r_new  = (r_park + step[1].clamp(-lim_r, lim_r)).clamp(0.04, 0.20);
            let ts_new = (t_s   + step[2].clamp(-lim_ts, lim_ts)).clamp(t_s_min, t_s_max);

            let new_res = match residual_vec(th_new, r_new, ts_new) {
                None => f64::INFINITY,
                Some(rv) => (rv[0]*rv[0] + rv[1]*rv[1] + rv[2]*rv[2]).sqrt(),
            };

            if new_res < res {
                theta  = th_new;
                r_park = r_new;
                t_s    = ts_new;
                lambda = (lambda * LM_DOWN).max(1e-14);
                accepted = true;
                break;
            }
            lambda *= LM_UP;
            if lambda > LM_MAX { break; }
        }
        if !accepted { break; }
    }

    // Build final arc at converged (θ, r_park) and trim to section crossing.
    let s_man_fin   = man_at(t_s);
    let x_sec_fin   = s_man_fin[0];
    let ic_final    = injection_ic(mu, x_earth, r_park, c_target, theta)
        .unwrap_or([x_earth + r_park, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let full_arc    = propagate_3d(mu, ic_final, t_end, 0.005, 1e-10, 1e-12);
    let (t_arc_fin, s_arc_fin) = find_arc_poincare_crossings(&full_arc, x_sec_fin, 0.0)
        .into_iter().next()
        .map(|c| (c.time, c.state))
        .unwrap_or((last_t_arc, last_s_arc));

    let dv  = [s_man_fin[3]-s_arc_fin[3], s_man_fin[4]-s_arc_fin[4], s_man_fin[5]-s_arc_fin[5]];
    let res = ((s_arc_fin[1]-s_man_fin[1]).powi(2)
             + (s_arc_fin[3]-s_man_fin[3]).powi(2)
             + (s_arc_fin[4]-s_man_fin[4]).powi(2)).sqrt();

    RefinedArc {
        theta, delta_c: 0.0, r_park,
        t_arc: t_arc_fin, t_s,
        state_arc: s_arc_fin, state_stable: s_man_fin,
        delta_v: dv,
        delta_v_mag: norm6(&[0., 0., 0., dv[0], dv[1], dv[2]]),
        residual: res, converged,
    }
}

/// 2×2 linear solve Ax = b (Cramer's rule).
fn solve_2x2(a: [[f64; 2]; 2], b: [f64; 2]) -> Option<[f64; 2]> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-25 { return None; }
    Some([
        (b[0] * a[1][1] - b[1] * a[0][1]) / det,
        (a[0][0] * b[1] - a[1][0] * b[0]) / det,
    ])
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Rotating-frame initial conditions for a tangential injection at angle θ
/// on a circular parking orbit.  Returns `None` if the position is in a
/// forbidden zone for this Jacobi constant.
pub fn injection_ic(
    mu: f64, x_earth: f64, r_park: f64, c_target: f64, theta: f64,
) -> Option<[f64; 6]> {
    let x0 = x_earth + r_park * theta.cos();
    let y0 = r_park * theta.sin();
    let v2 = 2.0 * omega_nd(mu, x0, y0) - c_target;
    if v2 <= 0.0 { return None; }
    let v = v2.sqrt();
    Some([x0, y0, 0.0, -v * theta.sin(), v * theta.cos(), 0.0])
}

/// TLI injection initial condition in the Earth-Moon rotating frame.
///
/// Computes the state vector at the injection point on a circular parking orbit
/// of radius `r_park` [nd] at injection angle `theta` [rad], targeting an apogee
/// of `r_apogee` [nd] from Earth via a Hohmann-like transfer ellipse.
///
/// All distances in normalized (nd) units; velocities in nd/nd-time.
/// Earth is located at x = −μ in the rotating frame.
///
/// Returns `None` if `r_apogee <= r_park` (degenerate orbit).
pub fn tli_injection_ic(mu: f64, r_park: f64, r_apogee: f64, theta: f64) -> Option<[f64; 6]> {
    if r_apogee <= r_park { return None; }
    let x_earth  = -mu;
    let mu_earth = 1.0 - mu;
    let a        = (r_park + r_apogee) / 2.0;
    let v_inj    = (mu_earth * (2.0 / r_park - 1.0 / a)).sqrt();
    // Position on the parking orbit at angle theta
    let px = x_earth + r_park * theta.cos();
    let py =           r_park * theta.sin();
    // Rotating-frame velocity: subtract frame drag ω×r_bary (ω = 1 nd/nd-time)
    let vx = -v_inj * theta.sin() + py;
    let vy =  v_inj * theta.cos() - px;
    Some([px, py, 0.0, vx, vy, 0.0])
}

/// Linearly interpolate a 6-state from a manifold branch at time `t`.
///
/// Works for both ascending-time (forward) and descending-time (backward)
/// branches.  The stable manifold branches produced by `propagate_3d_backward`
/// store times as negative values (0 at the orbit, −t_man at the far end), so
/// a naive `partition_point(|s| s.time <= t)` gives wrong results; we handle
/// both orderings explicitly.
pub fn interp_branch_pub(branch: &ManifoldBranch, t: f64) -> Option<[f64; 6]> {
    interp_branch(branch, t)
}

fn interp_branch(branch: &ManifoldBranch, t: f64) -> Option<[f64; 6]> {
    let steps = &branch.steps;
    if steps.len() < 2 { return None; }
    let t0 = steps.first()?.time;
    let t1 = steps.last()?.time;
    if t < t0.min(t1) - 1e-12 || t > t0.max(t1) + 1e-12 { return None; }

    // Binary search — choose predicate direction based on sort order.
    let i = if t1 >= t0 {
        // Ascending (forward integration): standard partition
        steps.partition_point(|s| s.time <= t)
    } else {
        // Descending (backward integration, times 0 → −t_man):
        // find first index where s.time < t  (i.e., further from the orbit)
        steps.partition_point(|s| s.time >= t)
    }.clamp(1, steps.len() - 1);

    let s0 = &steps[i - 1];
    let s1 = &steps[i];
    let dt = s1.time - s0.time;
    let frac = if dt.abs() < 1e-15 { 0.0 } else { (t - s0.time) / dt };
    let lerp = |a: f64, b: f64| a + frac * (b - a);
    Some([
        lerp(s0.x, s1.x), lerp(s0.y, s1.y), lerp(s0.z, s1.z),
        lerp(s0.vx, s1.vx), lerp(s0.vy, s1.vy), lerp(s0.vz, s1.vz),
    ])
}

#[allow(dead_code)]
/// Minimum Euclidean distance from any arc step to any point in `pts`.
fn min_dist_to_pts(arc: &[Step3d], pts: &[[f64; 3]]) -> f64 {
    let mut best = f64::INFINITY;
    for s in arc {
        for p in pts {
            let d = (s.x-p[0]).powi(2) + (s.y-p[1]).powi(2) + (s.z-p[2]).powi(2);
            if d < best { best = d; }
        }
    }
    best.sqrt()
}

#[allow(dead_code)]
/// Return (arc_idx, branch_idx, step_idx, dist) for the closest approach
/// between the arc and all steps of all stable branches.
fn closest_approach_indices(
    arc:    &[Step3d],
    stable: &[ManifoldBranch],
) -> (usize, usize, usize, f64) {
    let mut best = f64::INFINITY;
    let (mut ai, mut bi, mut si) = (0, 0, 0);
    for (a, s) in arc.iter().enumerate() {
        for (b, branch) in stable.iter().enumerate() {
            for (k, ss) in branch.steps.iter().enumerate() {
                let d = (s.x-ss.x).powi(2) + (s.y-ss.y).powi(2) + (s.z-ss.z).powi(2);
                if d < best { best = d; ai = a; bi = b; si = k; }
            }
        }
    }
    (ai, bi, si, best.sqrt())
}

#[allow(dead_code)]
/// Ternary search for minimum of a unimodal function over [lo, hi].
fn ternary_min(mut lo: f64, mut hi: f64, n: usize, f: impl Fn(f64) -> f64) -> f64 {
    for _ in 0..n {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if f(m1) <= f(m2) { hi = m2; } else { lo = m1; }
    }
    (lo + hi) * 0.5
}

#[allow(dead_code)]
/// Solve a 3×3 linear system Ax = b via Gaussian elimination with partial pivot.
fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| {
            a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap()
        }).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        if a[col][col].abs() < 1e-25 { return None; }
        for row in (col + 1)..3 {
            let f = a[row][col] / a[col][col];
            for k in col..3 { a[row][k] -= f * a[col][k]; }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        x[i] = b[i];
        for j in (i + 1)..3 { x[i] -= a[i][j] * x[j]; }
        if a[i][i].abs() < 1e-25 { return None; }
        x[i] /= a[i][i];
    }
    Some(x)
}

// ─── Ballistic capture helpers ───────────────────────────────────────────────

/// TLI (Trans-Lunar Injection) ΔV = v_inject − v_circ at the parking orbit.
///
/// Uses vis-viva targeting a Hohmann apogee at `r_apogee`.  Returns ΔV in
/// normalized (nd) units.  Returns `f64::INFINITY` if `r_apogee ≤ r_park`.
pub fn tli_dv(mu: f64, r_park: f64, r_apogee: f64) -> f64 {
    if r_apogee <= r_park { return f64::INFINITY; }
    let mu_earth = 1.0 - mu;
    let a        = (r_park + r_apogee) / 2.0;
    let v_inject = (mu_earth * (2.0 / r_park - 1.0 / a)).sqrt();
    let v_circ   = (mu_earth / r_park).sqrt();
    v_inject - v_circ
}

/// LOI (Lunar Orbit Insertion) circularization ΔV at a single trajectory step.
///
/// Computes the burn magnitude to circularize at the current Moon-relative
/// distance in the orbital plane.  Converts rotating-frame velocities to
/// Moon-relative inertial with the Coriolis correction:
///   v_rel = (vx − y,  vy + x − moon_x,  vz)
///
/// Mirrors the formula in wsb_circularize.rs (`circularization_dv`).
/// Returns ΔV in normalized (nd) units; `f64::INFINITY` near zero distance.
pub fn loi_dv(mu: f64, state: &Step3d) -> f64 {
    let moon_x = 1.0 - mu;
    let rx = state.x - moon_x;
    let ry = state.y;
    let rz = state.z;
    let r_mag = (rx*rx + ry*ry + rz*rz).sqrt();
    if r_mag < 1e-10 { return f64::INFINITY; }
    let vrx = state.vx - state.y;
    let vry = state.vy + state.x - moon_x;
    let vrz = state.vz;
    let v_circ = (mu / r_mag).sqrt();
    let hx = ry*vrz - rz*vry;
    let hy = rz*vrx - rx*vrz;
    let hz = rx*vry - ry*vrx;
    let h_mag = (hx*hx + hy*hy + hz*hz).sqrt();
    if h_mag < 1e-14 {
        let v_rel = (vrx*vrx + vry*vry + vrz*vrz).sqrt();
        return (v_rel - v_circ).abs();
    }
    let (hxn, hyn, hzn) = (hx/h_mag, hy/h_mag, hz/h_mag);
    let (rxn, ryn, rzn) = (rx/r_mag, ry/r_mag, rz/r_mag);
    let tx = hyn*rzn - hzn*ryn;
    let ty = hzn*rxn - hxn*rzn;
    let tz = hxn*ryn - hyn*rxn;
    let dvx = v_circ*tx - vrx;
    let dvy = v_circ*ty - vry;
    let dvz = v_circ*tz - vrz;
    (dvx*dvx + dvy*dvy + dvz*dvz).sqrt()
}

/// Minimum LOI ΔV found inside the lunar Hill sphere over a trajectory.
///
/// Scans every logged step with r_moon < r_Hill and returns the minimum
/// `loi_dv()`.  Returns `None` if the spacecraft never enters the Hill sphere.
pub fn min_loi_dv(traj: &[Step3d], mu: f64) -> Option<f64> {
    let r_hill = lunar_hill_radius(mu);
    let moon_x = 1.0 - mu;
    let mut best: Option<f64> = None;
    for s in traj {
        let dx = s.x - moon_x;
        let r_moon = (dx*dx + s.y*s.y + s.z*s.z).sqrt();
        if r_moon < r_hill {
            let dv = loi_dv(mu, s);
            if dv.is_finite() {
                best = Some(best.map_or(dv, |b: f64| b.min(dv)));
            }
        }
    }
    best
}

/// Moon's Hill sphere radius in normalized CRTBP units: r_H = (μ/3)^(1/3).
pub fn lunar_hill_radius(mu: f64) -> f64 {
    (mu / 3.0_f64).cbrt()
}

/// Two-body specific orbital energy of the spacecraft relative to the Moon,
/// evaluated using inertial-frame velocities.
///
/// In normalized units the frame rotates at ω = 1, so the inertial velocity is
///   v_in = v_rot + ω × r   with ω = ẑ  →  ω × r = (−y, x, 0)
/// giving v_in = (vx − y,  vy + x,  vz).
pub fn two_body_energy_moon(state: &[f64; 6], mu: f64) -> f64 {
    let moon_x = 1.0 - mu;
    let dx = state[0] - moon_x;
    let r_moon = (dx * dx + state[1] * state[1] + state[2] * state[2]).sqrt();
    let vx_in = state[3] - state[1];
    let vy_in = state[4] + state[0];
    let v2 = vx_in * vx_in + vy_in * vy_in + state[5] * state[5];
    v2 / 2.0 - mu / r_moon
}

/// Result of a lunar capture detection scan.
pub struct CaptureResult {
    /// True if the total time captured meets the minimum threshold.
    pub is_captured:          bool,
    /// Total accumulated time inside the Hill sphere [nd].
    pub capture_duration:     f64,
    /// Number of distinct Hill-sphere entry events.
    pub n_entries:            usize,
    /// Minimum distance to Moon centre over the full trajectory [nd].
    pub min_lunar_dist:       f64,
    /// Duration of the single longest continuous capture interval [nd].
    pub max_capture_interval: f64,
    /// Number of periapsis passages (local r_moon minima) inside the Hill sphere.
    /// Each passage ≈ one orbit around the Moon.  More physically meaningful than
    /// max_capture_interval / LUNAR_PERIOD_ND, which counts Earth-Moon periods.
    pub n_periapsis: usize,
}

/// Scan a trajectory for temporary capture by the Moon.
///
/// A step is counted as "captured" when the spacecraft is inside the Hill
/// sphere (r < r_H).  The optional `min_duration` threshold determines
/// `is_captured`.
pub fn detect_capture(traj: &[Step3d], mu: f64, min_duration: f64) -> CaptureResult {
    let r_hill   = lunar_hill_radius(mu);
    let moon_x   = 1.0 - mu;
    let mut total     = 0.0_f64;
    let mut n_entries = 0_usize;
    let mut min_dist  = f64::MAX;
    let mut max_iv    = 0.0_f64;
    let mut in_cap    = false;
    let mut t_start   = 0.0_f64;

    // Periapsis counting: track local minima of r_moon inside the Hill sphere.
    let mut n_periapsis   = 0_usize;
    let mut prev_r_hill   = f64::MAX;  // r_moon at previous step (only updated inside Hill)
    let mut r_decreasing  = false;     // was r_moon decreasing at the previous Hill step?

    for step in traj {
        let s  = step.state();
        let dx = s[0] - moon_x;
        let r_moon = (dx * dx + s[1] * s[1] + s[2] * s[2]).sqrt();
        if r_moon < min_dist { min_dist = r_moon; }

        let cap = r_moon < r_hill;
        match (in_cap, cap) {
            (false, true) => {
                in_cap = true; t_start = step.time; n_entries += 1;
                prev_r_hill  = r_moon;
                r_decreasing = false;
            }
            (true, false) => {
                in_cap = false;
                let dur = (step.time - t_start).abs();
                total += dur;
                if dur > max_iv { max_iv = dur; }
                prev_r_hill  = f64::MAX;
                r_decreasing = false;
            }
            (true, true) => {
                // Detect periapsis: r_moon was decreasing and is now increasing.
                if r_moon > prev_r_hill && r_decreasing {
                    n_periapsis += 1;
                    r_decreasing = false;
                } else if r_moon < prev_r_hill {
                    r_decreasing = true;
                }
                prev_r_hill = r_moon;
            }
            _ => {}
        }
    }
    if in_cap {
        let dur = traj.last().map(|s| (s.time - t_start).abs()).unwrap_or(0.0);
        total += dur;
        if dur > max_iv { max_iv = dur; }
    }

    CaptureResult {
        is_captured:          total >= min_duration,
        capture_duration:     total,
        n_entries,
        min_lunar_dist:       if min_dist == f64::MAX { 0.0 } else { min_dist },
        max_capture_interval: max_iv,
        n_periapsis,
    }
}
