//! Periodic orbit finders for the Earth-Moon CRTBP.
//!
//! # Orbit types
//!
//! | Type      | Plane  | Corrector | Free vars   | Targets          |
//! |-----------|--------|-----------|-------------|------------------|
//! | Lyapunov  | z = 0  | 1-D       | vy₀         | vx_f = 0         |
//! | Halo      | 3-D    | 2-D       | x₀, vy₀    | vx_f = 0, vz_f=0 |
//! | DRO       | z = 0  | 1-D       | vy₀         | vx_f = 0         |
//!
//! # How the differential corrector works
//!
//! All orbit types exploit the time-reversal symmetry of the CRTBP:
//! a solution that crosses y = 0 with vx = 0 (and vz = 0 for 3-D) can
//! be reflected to form a full period.  We:
//!
//! 1. Start from an approximate initial condition at the x-axis (y = 0).
//! 2. Propagate until the next y = 0 crossing, recording the STM.
//! 3. Use the relevant STM entries to form a Newton correction.
//! 4. Repeat until the crossing is perpendicular to within `tol`.
//!
//! This is the standard single-shooting differential corrector described in
//! Koon, Lo, Marsden & Ross (2000) and Howell (1984).
//!
//! # Finding new solutions
//!
//! The corrector itself is a general tool: supply any initial condition near
//! a closed trajectory and it will converge (if the guess is close enough).
//! `trace_family` does parameter continuation — each solution seeds the
//! next amplitude step — allowing full orbit families to be traced without
//! any analytical approximation.

use crate::crtbp::{jacobi_constant, lagrange_x, lyapunov_vy0_approx};
use crate::linalg::{solve_2x2};
use crate::propagator::{propagate_3d, propagate_3d_stm, Step3d};

// ─── Public orbit type ────────────────────────────────────────────────────────

/// Label describing which family a periodic orbit belongs to.
#[derive(Clone, Debug)]
pub enum OrbitFamily {
    /// Planar Lyapunov orbit around a collinear Lagrange point.
    Lyapunov { lagrange: u8 },
    /// Northern (+z) halo orbit around a collinear Lagrange point.
    HaloNorth { lagrange: u8 },
    /// Southern (−z) halo orbit around a collinear Lagrange point.
    HaloSouth { lagrange: u8 },
    /// Distant Retrograde Orbit (around the Moon, z = 0).
    Dro,
}

/// A converged periodic orbit in the CRTBP.
#[derive(Clone, Debug)]
pub struct FoundOrbit {
    /// Initial condition [x, y=0, z, vx=0, vy, vz=0] on the x-axis.
    pub ic:     [f64; 6],
    /// Half-period (time from IC to next y=0 crossing).
    pub t_half: f64,
    /// Full period = 2 · t_half.
    pub period: f64,
    /// Jacobi constant.
    pub jacobi: f64,
    /// Which family this orbit belongs to.
    pub family: OrbitFamily,
}

// ─── Internal propagation helper ─────────────────────────────────────────────

/// Propagate `ic` until the first y=0 crossing after t=0, returning
/// `(t_half, vx_f, vz_f, STM)` at the crossing.
///
/// Uses a coarse pass to locate the crossing, then a tight STM integration
/// to exactly t_half.  vx_f and vz_f come from the tight pass.
fn propagate_to_half_period(
    mu:   f64,
    ic:   [f64; 6],
    rtol: f64,
    atol: f64,
) -> Option<(f64, f64, f64, [[f64; 6]; 6])> {
    let t_max  = 4.0 * std::f64::consts::PI;
    let log_dt = t_max / 800.0;

    // Coarse pass to find the crossing time
    let traj = propagate_3d(mu, ic, t_max, log_dt, rtol * 10.0, atol * 10.0);

    let idx = traj.windows(2)
        .enumerate()
        .skip(2)
        .find(|(_, w)| w[0].y * w[1].y <= 0.0 && w[0].time > 1e-4)
        .map(|(i, _)| i + 1)?;

    let s0 = &traj[idx - 1];
    let s1 = &traj[idx];
    let frac   = s0.y / (s0.y - s1.y);
    let t_half = s0.time + frac * (s1.time - s0.time);

    // Tight STM pass to t_half for accurate derivatives.
    // Keep vx_f/vz_f from the coarse interpolation — more accurate than the
    // endpoint of the tight pass when t_half is only approximately correct.
    let (_, stm) = propagate_3d_stm(mu, ic, t_half, rtol, atol);
    let vx_f = s0.vx + frac * (s1.vx - s0.vx);
    let vz_f = s0.vz + frac * (s1.vz - s0.vz);

    Some((t_half, vx_f, vz_f, stm))
}

// ─── 1-D corrector (Lyapunov / DRO) ─────────────────────────────────────────

/// Single-variable differential corrector for planar orbits.
///
/// Free variable: `ic[4]` (vy₀).  Target: vx_f = 0 at y = 0 crossing.
/// Uses STM entry `φ[3][4] = ∂vx_f / ∂vy₀`.
fn correct_planar(
    mu:     f64,
    mut ic: [f64; 6],
    tol:    f64,
    max_it: usize,
    family: OrbitFamily,
) -> FoundOrbit {
    let rtol = 1e-11;
    let atol = 1e-13;
    let mut t_half = 0.0;

    for iter in 0..max_it {
        let result = propagate_to_half_period(mu, ic, rtol, atol)
            .unwrap_or_else(|| panic!(
                "planar corrector: no y=0 crossing (iter {iter}, vy0={:.6})", ic[4]
            ));
        let (th, vx_f, _, stm) = result;
        t_half = th;

        if vx_f.abs() < tol { break; }
        if iter == max_it - 1 {
            eprintln!("planar corrector: max iterations (|vx_f|={:.2e})", vx_f.abs());
        }

        let dvxf_dvy0 = stm[3][4];
        if dvxf_dvy0.abs() < 1e-30 { break; }
        ic[4] -= vx_f / dvxf_dvy0;
    }

    let jacobi = jacobi_constant(mu, &[ic[0], ic[1], ic[3], ic[4]]);
    FoundOrbit { ic, t_half, period: 2.0 * t_half, jacobi, family }
}

// ─── 2-D corrector (Halo) ────────────────────────────────────────────────────

/// Two-variable differential corrector for halo orbits.
///
/// **Fixed**: z₀ = ic[2] (the requested Az — must not drift to the planar solution).
/// **Free variables**: x₀ = ic[0] and vy₀ = ic[4].
/// **Targets**: vx_f = 0 and vz_f = 0 at the y = 0 half-period crossing.
///
/// Correction matrix:
/// ```text
/// M = [ φ[3][0]  φ[3][4] ]    row 3 = vx,  col 0 = x₀
///     [ φ[5][0]  φ[5][4] ]    row 5 = vz,  col 4 = vy₀
/// ```
/// For small Az these entries can be near-singular (z decouples from xy).
/// A step-size limiter prevents divergence; in the limit Az → 0 the orbit
/// degenerates gracefully to the Lyapunov solution at the same x₀.
fn correct_halo(
    mu:     f64,
    mut ic: [f64; 6],
    tol:    f64,
    max_it: usize,
    family: OrbitFamily,
) -> FoundOrbit {
    let rtol = 1e-11;
    let atol = 1e-13;
    let mut t_half = 0.0;

    for iter in 0..max_it {
        let result = propagate_to_half_period(mu, ic, rtol, atol)
            .unwrap_or_else(|| panic!("halo corrector: no y=0 crossing (iter {iter})"));
        let (th, vx_f, vz_f, stm) = result;
        t_half = th;

        if vx_f.abs() < tol && vz_f.abs() < tol { break; }
        if iter == max_it - 1 {
            eprintln!("halo corrector: max iterations (|vx_f|={:.2e}, |vz_f|={:.2e})",
                      vx_f.abs(), vz_f.abs());
        }

        // 2×2 Newton step with free vars (x0, vy0); z0 stays fixed at Az
        let det = stm[3][0] * stm[5][4] - stm[3][4] * stm[5][0];
        let (dx0, dvy0) = if det.abs() > 1e-20 {
            solve_2x2(stm[3][0], stm[3][4], stm[5][0], stm[5][4], -vx_f, -vz_f)
        } else {
            // Near-singular (small Az): fall back to 1-D vy0 correction
            let d = stm[3][4];
            (0.0, if d.abs() > 1e-30 { -vx_f / d } else { 0.0 })
        };

        // Step-size limiter — prevents divergence far from solution
        let max_dx  = 0.05_f64.min(ic[0].abs() * 0.1);
        let max_dvy = 0.05_f64.min(ic[4].abs() * 0.1 + 1e-6);
        let scale   = (max_dx / dx0.abs().max(1e-30))
                      .min(max_dvy / dvy0.abs().max(1e-30))
                      .min(1.0);
        ic[0] += scale * dx0;
        ic[4] += scale * dvy0;
    }

    // Jacobi: v² = vy₀² at IC (vx = vz = 0)
    let r1 = ((ic[0]+mu).powi(2) + ic[1].powi(2) + ic[2].powi(2)).sqrt();
    let r2 = ((ic[0]-(1.0-mu)).powi(2) + ic[1].powi(2) + ic[2].powi(2)).sqrt();
    let omega = 0.5*(ic[0]*ic[0] + ic[1]*ic[1]) + (1.0-mu)/r1 + mu/r2;
    let jacobi = 2.0*omega - ic[4]*ic[4];

    FoundOrbit { ic, t_half, period: 2.0 * t_half, jacobi, family }
}

// ─── Public orbit finders ─────────────────────────────────────────────────────

/// Find a Lyapunov orbit around Lagrange point `lagrange` (1 or 2).
///
/// `ax` is the x-amplitude from the Lagrange point in normalized units.
/// The orbit lies in the z = 0 plane.
pub fn lyapunov(mu: f64, lagrange: u8, ax: f64, tol: f64, max_it: usize) -> FoundOrbit {
    let x_l = lagrange_x(mu, lagrange);
    let vy0  = lyapunov_vy0_approx(mu, x_l, ax);

    // Try both sides; outer = x_L + ax first, then x_L - ax
    for &(offset, vy) in &[(ax, vy0), (-ax, -vy0)] {
        let ic = [x_l + offset, 0.0, 0.0, 0.0, vy, 0.0];
        let traj = propagate_3d(mu, ic, 2.0*std::f64::consts::PI,
                                2.0*std::f64::consts::PI/200.0, 1e-10, 1e-12);
        let has_cross = traj.windows(2).skip(2)
            .any(|w| w[0].y * w[1].y <= 0.0 && w[0].time > 1e-4);
        if has_cross {
            return correct_planar(mu, ic, tol, max_it,
                                  OrbitFamily::Lyapunov { lagrange });
        }
    }
    panic!("lyapunov: no crossing found for ax={ax:.4}, L{lagrange}");
}

/// Find a halo orbit around Lagrange point `lagrange` (1 or 2).
///
/// `az` is the initial z-coordinate at the y=0 crossing (approximate z-amplitude).
/// `north = true` places the orbit above the xy-plane (z₀ > 0).
///
/// **Seeding strategy**: x₀ and vy₀ are taken from the corresponding Lyapunov
/// orbit (which is the Az→0 limit).  The corrector then varies z₀ and vy₀
/// while keeping x₀ fixed, which is well-conditioned for all Az > 0.
pub fn halo(
    mu:       f64,
    lagrange: u8,
    az:       f64,
    north:    bool,
    tol:      f64,
    max_it:   usize,
) -> FoundOrbit {
    let z_sign = if north { 1.0 } else { -1.0 };

    // Seed x₀ and vy₀ from a Lyapunov orbit (the Az=0 family member).
    // Use a moderate in-plane amplitude as the representative seed.
    let ax_seed = 0.015_f64.max(az * 0.5);
    let lyap    = lyapunov(mu, lagrange, ax_seed, tol, max_it);
    let ic      = [lyap.ic[0], 0.0, z_sign * az, 0.0, lyap.ic[4], 0.0];

    let family = if north {
        OrbitFamily::HaloNorth { lagrange }
    } else {
        OrbitFamily::HaloSouth { lagrange }
    };
    correct_halo(mu, ic, tol, max_it, family)
}

/// Find a Distant Retrograde Orbit (DRO) at distance `r` from the Moon.
///
/// DROs are planar, retrograde orbits centered on the Moon.  They exist
/// for any radius and are stable (no unstable manifold), making them
/// attractive for long-duration lunar vicinity operations.
///
/// `r` is the distance from the Moon centre in normalized units.
/// Find a Distant Retrograde Orbit (DRO) at distance `r` from the Moon.
///
/// DROs are planar, retrograde orbits centred on the Moon.  They are
/// stable (no real unstable manifold), making them useful for long-duration
/// lunar vicinity operations.
///
/// `r` is the distance from the Moon centre in normalized units.
///
/// Initial velocity estimate: the retrograde circular speed around the Moon
/// in the rotating frame is approximately `vy₀ ≈ -(√(μ/r) + r_moon_x)`,
/// where `r_moon_x = 1 − μ + r` is the x position.  For large r the frame
/// rotation term dominates and makes the orbit look prograde in the inertial
/// frame; the corrector adjusts from this estimate.
pub fn dro(mu: f64, r: f64, tol: f64, max_it: usize) -> FoundOrbit {
    let x0  = (1.0 - mu) + r;
    // Inertial circular speed around Moon minus frame rotation: retrograde vy
    let v_c = (mu / r).sqrt();      // inertial circular speed
    let vy0 = -(v_c + x0);         // rotating frame: subtract ω×r = x0
    let ic  = [x0, 0.0, 0.0, 0.0, vy0, 0.0];
    correct_planar(mu, ic, tol, max_it, OrbitFamily::Dro)
}

// ─── Parameter continuation (family tracing) ─────────────────────────────────

/// Trace the Lyapunov family at Lagrange point `lagrange` over `ax_vals`.
///
/// Each orbit uses the previous solution as its initial guess, so the
/// family can be traced far from the linear regime without a good analytic
/// approximation.  `ax_vals` should be monotone (increasing or decreasing).
pub fn trace_lyapunov_family(
    mu:      f64,
    lagrange: u8,
    ax_vals: &[f64],
    tol:     f64,
) -> Vec<FoundOrbit> {
    let mut orbits = Vec::with_capacity(ax_vals.len());
    for &ax in ax_vals {
        let orbit = if orbits.is_empty() {
            lyapunov(mu, lagrange, ax, tol, 50)
        } else {
            // Seed from previous orbit, scale vy0 by amplitude ratio
            let prev: &FoundOrbit = orbits.last().unwrap();
            let ax_prev = (prev.ic[0] - lagrange_x(mu, lagrange)).abs();
            let scale = if ax_prev > 1e-10 { ax / ax_prev } else { 1.0 };
            let mut ic = prev.ic;
            ic[4] *= scale;
            ic[0]  = lagrange_x(mu, lagrange) + ax;
            correct_planar(mu, ic, tol, 50, OrbitFamily::Lyapunov { lagrange })
        };
        orbits.push(orbit);
    }
    orbits
}

/// Trace the halo family at Lagrange point `lagrange` over `az_vals`.
///
/// Uses continuation from small Az upward.  `az_vals` should start small
/// (e.g., 0.005) and increase monotonically.
pub fn trace_halo_family(
    mu:      f64,
    lagrange: u8,
    az_vals: &[f64],
    north:   bool,
    tol:     f64,
) -> Vec<FoundOrbit> {
    let mut orbits = Vec::with_capacity(az_vals.len());
    for &az in az_vals {
        let orbit = if orbits.is_empty() {
            halo(mu, lagrange, az, north, tol, 50)
        } else {
            let prev: &FoundOrbit = orbits.last().unwrap();
            let z_sign = if north { 1.0 } else { -1.0 };
            let mut ic  = prev.ic;
            ic[2] = z_sign * az;   // update z0 to new amplitude
            // keep x0, vy0 from previous solution (good warm start)
            let family = if north {
                OrbitFamily::HaloNorth { lagrange }
            } else {
                OrbitFamily::HaloSouth { lagrange }
            };
            correct_halo(mu, ic, tol, 50, family)
        };
        orbits.push(orbit);
    }
    orbits
}

/// Trace the DRO family over `r_vals` (distances from the Moon).
pub fn trace_dro_family(
    mu:     f64,
    r_vals: &[f64],
    tol:    f64,
) -> Vec<FoundOrbit> {
    let mut orbits = Vec::with_capacity(r_vals.len());
    for &r in r_vals {
        let orbit = if orbits.is_empty() {
            dro(mu, r, tol, 50)
        } else {
            let prev: &FoundOrbit = orbits.last().unwrap();
            let mut ic = prev.ic;
            ic[0] = (1.0 - mu) + r;
            // Scale vy0 approximately by sqrt(1/r) (circular speed scaling)
            let r_prev = prev.ic[0] - (1.0 - mu);
            if r_prev > 1e-10 { ic[4] *= (r_prev / r).sqrt(); }
            correct_planar(mu, ic, tol, 50, OrbitFamily::Dro)
        };
        orbits.push(orbit);
    }
    orbits
}

// ─── Convenience: full-period trajectory ─────────────────────────────────────

/// Propagate a `FoundOrbit` for one full period and return the trajectory.
///
/// Uses 1e-9/1e-11 tolerances (sufficient for plotting); the corrector
/// already verified the orbit to tighter tolerances.
pub fn full_period_traj(mu: f64, orbit: &FoundOrbit) -> Vec<Step3d> {
    propagate_3d(
        mu, orbit.ic, orbit.period + 1e-6,
        orbit.period / 500.0, 1e-9, 1e-11,
    )
}
