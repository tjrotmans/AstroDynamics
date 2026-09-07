//! Invariant manifold computation for periodic orbits in the CRTBP.
//!
//! Given a `FoundOrbit` and its monodromy matrix (from `propagate_3d_stm`),
//! this module:
//!   1. Computes the unstable and stable eigenvectors of the 6×6 monodromy.
//!   2. Transports those eigenvectors to N points along the orbit via the STM.
//!   3. Perturbs ±ε along each direction and propagates to form manifold branches.
//!
//! Only manifold computation lives here.  Orbit finding lives in `periodic_orbits`.

use crate::linalg::{mat_vec6, norm6, scale6, add6, dominant_eigenvec6, mat_inv6};
use crate::periodic_orbits::FoundOrbit;
use crate::propagator::{propagate_3d, propagate_3d_backward, propagate_3d_stm_full, Step3d};

// ─── Eigenvectors of the monodromy ───────────────────────────────────────────

/// Dominant (unstable) eigenvector of the 6×6 monodromy matrix.
///
/// For a planar Lyapunov orbit the z–vz block decouples; the result has
/// near-zero z and vz components.  For a halo orbit, all six components
/// are generally non-zero.
pub fn unstable_eigenvec(mono: &[[f64; 6]; 6]) -> [f64; 6] {
    dominant_eigenvec6(mono, 200).1
}

/// Stable eigenvector of the monodromy matrix (dominant eigenvector of M⁻¹).
pub fn stable_eigenvec(mono: &[[f64; 6]; 6]) -> [f64; 6] {
    let inv = mat_inv6(mono);
    dominant_eigenvec6(&inv, 200).1
}

// ─── Branch types ─────────────────────────────────────────────────────────────

/// A single manifold trajectory (forward or backward propagation).
#[derive(Clone, Debug)]
pub struct ManifoldBranch {
    pub steps: Vec<Step3d>,
}

/// Parameters controlling manifold branch sampling.
pub struct ManifoldParams {
    /// Number of sample points around the orbit (total branches = n × 2).
    pub n_branches: usize,
    /// Perturbation magnitude in normalized units (typically 1e-6).
    pub epsilon:    f64,
    /// Integration time for each branch (normalized).
    pub t_man:      f64,
    /// Logging step size for each branch.
    pub log_dt:     f64,
    pub rtol:       f64,
    pub atol:       f64,
}

impl Default for ManifoldParams {
    fn default() -> Self {
        Self {
            n_branches: 20,
            epsilon:    1e-6,
            t_man:      3.0 * std::f64::consts::PI,
            log_dt:     0.005,
            rtol:       1e-10,
            atol:       1e-12,
        }
    }
}

// ─── Main computation ─────────────────────────────────────────────────────────

/// Compute the monodromy matrix for a `FoundOrbit` by propagating one
/// full period with the STM.
pub fn monodromy(mu: f64, orbit: &FoundOrbit) -> [[f64; 6]; 6] {
    use crate::propagator::propagate_3d_stm;
    propagate_3d_stm(mu, orbit.ic, orbit.period, 1e-10, 1e-12).1
}

/// Compute unstable (+ε) and stable (−ε backward) manifold branches.
///
/// Works for any `FoundOrbit` regardless of which Lagrange point it
/// encircles or whether it is planar or 3-D.
///
/// Returns `(unstable_branches, stable_branches)`.
pub fn manifold_branches(
    mu:     f64,
    orbit:  &FoundOrbit,
    mono:   &[[f64; 6]; 6],
    params: &ManifoldParams,
) -> (Vec<ManifoldBranch>, Vec<ManifoldBranch>) {
    let rtol = params.rtol;
    let atol = params.atol;

    let v_u0 = unstable_eigenvec(mono);
    let v_s0 = stable_eigenvec(mono);

    // Sample the full orbit + STM at fine steps
    let dt_log = orbit.period / (params.n_branches as f64 * 10.0);
    let (traj_full, stms_full) = propagate_3d_stm_full(
        mu, orbit.ic, orbit.period, dt_log, rtol, atol,
    ).expect("STM propagation failed in manifold computation");

    // Pick n_branches evenly spaced indices
    let n    = traj_full.len();
    let step = (n / params.n_branches).max(1);
    let indices: Vec<usize> = (0..params.n_branches)
        .map(|k| (k * step).min(n - 1))
        .collect();

    let mut unstable = Vec::with_capacity(params.n_branches * 2);
    let mut stable   = Vec::with_capacity(params.n_branches * 2);

    for &idx in &indices {
        let s   = &traj_full[idx];
        let phi = &stms_full[idx];
        let base: [f64; 6] = [s.x, s.y, s.z, s.vx, s.vy, s.vz];

        // Transport eigenvectors to this point via the STM and normalise
        let vu = { let v = mat_vec6(phi, &v_u0); scale6(&v, 1.0 / norm6(&v)) };
        let vs = { let v = mat_vec6(phi, &v_s0); scale6(&v, 1.0 / norm6(&v)) };

        for &sign in &[1.0_f64, -1.0] {
            // Unstable: perturb along v_u, propagate forward
            let ic_u = add6(&base, &scale6(&vu, sign * params.epsilon));
            unstable.push(ManifoldBranch {
                steps: propagate_3d(mu, ic_u, params.t_man, params.log_dt, rtol, atol),
            });

            // Stable: perturb along v_s, propagate backward
            let ic_s = add6(&base, &scale6(&vs, sign * params.epsilon));
            stable.push(ManifoldBranch {
                steps: propagate_3d_backward(mu, ic_s, params.t_man, params.log_dt, rtol, atol),
            });
        }
    }

    (unstable, stable)
}
