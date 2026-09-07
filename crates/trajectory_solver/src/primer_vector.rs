//! Primer vector optimality diagnostic for impulsive MGA-DSM legs.
//!
//! Implements Lawden's primer vector necessary conditions (Lawden 1963),
//! extended to multi-gravity-assist trajectories by Olympio (2009), as a
//! POST-HOC diagnostic on an already-converged MGA-1DSM chromosome — it
//! answers "does this leg's single-DSM structure leave ΔV on the table,"
//! not a search/optimization tool itself.
//!
//! # Theory (condensed from Olympio 2009, ESA ACT-RPR-MAD-2009)
//!
//! The primer vector `λV(t)` is the co-state (adjoint) of velocity in the
//! impulsive-trajectory optimal control problem. Four necessary conditions
//! for optimality:
//! 1. `λV(t)` and `λV̇(t)` are continuous everywhere, including through an
//!    impulse (only the swing-by jump conditions — not implemented here,
//!    see module-level scope note — break this).
//! 2. At an impulse, `λV = Δv/|Δv|` (unit vector along the impulse).
//! 3. `‖λV(t)‖ ≤ 1` everywhere — violation anywhere is a proof that adding
//!    an impulse there would reduce total ΔV.
//! 4. At an impulse whose *timing* is itself a free, optimized parameter
//!    (exactly our case — `η` is optimized to minimize that leg's own DSM
//!    magnitude), `d‖λV‖/dt = 0` there, equivalently `λV̇ ⊥ λV`.
//!
//! At a gravity-assist swing-by, Olympio's Eqs. (19)-(22) additionally show
//! `λV` is not free — it must be *parallel* to the incoming/outgoing v∞
//! direction, with only its scalar magnitude (`ν1`/`cν` in the paper) left
//! as an unknown to solve for.
//!
//! # Scope (explicitly reduced from the paper's full treatment)
//!
//! Olympio's full multi-leg solve also carries a position co-state (`λR`)
//! and an unknown Lagrange-multiplier jump (`νR2`, his Eq. 11) coupling
//! every leg's `λV̇` boundary value to its neighbors. None of that enters
//! the `‖λV‖ ≤ 1` test above, so this module deliberately omits it and
//! solves each leg IN ISOLATION: `λV̇` at both of a leg's true boundaries
//! (its own start/end, whether that's a flyby or the true mission
//! departure/arrival) is treated as a free unknown local to that leg, not
//! constrained by the neighboring legs' own co-states. This makes the
//! resulting boundary-value system over-determined (see [`solve_leg_primer`]
//! for the exact equation count) — solved via least squares, so the
//! reported result is the BEST ACHIEVABLE fit under this leg-isolated
//! approximation, not an exact global solution. Treat it as a genuine but
//! approximate signal, not a proof either way.
//!
//! # References
//! - Lawden, D.F. (1963), *Optimal Trajectories for Space Navigation*,
//!   Butterworths — the original primer vector necessary conditions.
//! - Olympio, J.T. (2009), "Designing Optimal Multi-Gravity-Assist
//!   Trajectories with Free Number of Impulses", ESA ACT-RPR-MAD-2009 —
//!   the MGA extension (swing-by jump conditions, Eqs. 4, 9-22) this module
//!   implements the `λV`-only subset of.
//! - Prussing, J.E. (1995), "Primer Vector Theory and Applications",
//!   in *Spacecraft Trajectory Optimization* — standard textbook treatment.

use nalgebra::{Matrix3, Matrix6, Vector3};
use orbital_math::kepler::propagate_kepler;

/// Gravity-gradient tensor `G = ∂/∂r (μ·r/|r|³) = (μ/|r|³)·(I − 3·r̂r̂ᵀ)`.
///
/// This is the SAME tensor used in the co-state ODE (Olympio Eq. 9) and
/// the standard two-body state-deviation ODE — but note the two ODEs use
/// it with OPPOSITE sign (see module scope note in [`propagate_costate`]).
/// Cross-checked two independent ways during derivation: matches both the
/// direct partial derivative of `μr/|r|³`, and the sign-flip of the
/// standard "tidal tensor" `∇g = -μ/r³·(I − 3r̂r̂ᵀ)` (g = -μr/|r|³ being
/// the acceleration, the negative of the quantity differentiated here).
pub fn gravity_gradient(r: Vector3<f64>, mu_m3s2: f64) -> Matrix3<f64> {
    let r_mag = r.norm();
    let r_hat = r / r_mag;
    let coeff = mu_m3s2 / (r_mag * r_mag * r_mag);
    coeff * (Matrix3::identity() - 3.0 * r_hat * r_hat.transpose())
}

/// Right-hand side of the co-state ODE, written in terms of `P := λV` and
/// `Q := λV̇` (so the second-order pair is `(Ṗ, Q̇) = (Q, G(r(τ))·P)`).
///
/// Derived from Olympio Eq. (9), `dΛ/dt = -[[0,G],[I,0]]Λ` with
/// `Λ = [λR; λV]`, expanding to `λṘ = -G·λV`, `λV̇ = -λR`. Substituting
/// `P := λV`, `Q := λV̇ = -λR` gives `Ṗ = Q` (definitional) and
/// `Q̇ = -λṘ = G·λV = G·P` — the `+G` sign (opposite the standard
/// state-deviation ODE `δv̇ = -G·δr`) is the one place this module's
/// derivation departs from the position/velocity STM, and is the reason
/// [`propagate_costate`] integrates its own ODE rather than reusing
/// `propagate_kepler`'s state STM directly. Verified in
/// `costate_stm_is_state_stm_with_flipped_gravity_sign` below: negating
/// `G` in this RHS reproduces the independently-validated state STM
/// (built via finite differences of `propagate_kepler`) to high precision,
/// isolating any residual risk to the sign choice alone, not the
/// integrator or tensor formula.
fn costate_rhs(r: Vector3<f64>, mu_m3s2: f64, p: Vector3<f64>, q: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    (q, gravity_gradient(r, mu_m3s2) * p)
}

/// RK4-integrate the co-state ODE from `(p0, q0)` at `t=0` to `t=dt_s`
/// along the Keplerian reference trajectory starting at `(r0, v0)`.
fn integrate_costate(
    r0: Vector3<f64>, v0: Vector3<f64>, mu_m3s2: f64,
    p0: Vector3<f64>, q0: Vector3<f64>,
    dt_s: f64, n_steps: usize,
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let h = dt_s / n_steps as f64;
    let (mut p, mut q) = (p0, q0);
    let mut t = 0.0_f64;
    for _ in 0..n_steps {
        let r_t   = propagate_kepler(r0, v0, t, mu_m3s2)?.0;
        let r_mid = propagate_kepler(r0, v0, t + 0.5 * h, mu_m3s2)?.0;
        let r_end = propagate_kepler(r0, v0, t + h, mu_m3s2)?.0;

        let (k1p, k1q) = costate_rhs(r_t, mu_m3s2, p, q);
        let (k2p, k2q) = costate_rhs(r_mid, mu_m3s2, p + 0.5 * h * k1p, q + 0.5 * h * k1q);
        let (k3p, k3q) = costate_rhs(r_mid, mu_m3s2, p + 0.5 * h * k2p, q + 0.5 * h * k2q);
        let (k4p, k4q) = costate_rhs(r_end, mu_m3s2, p + h * k3p, q + h * k3q);

        p += h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
        q += h / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q);
        t += h;
    }
    Some((p, q))
}

/// Number of RK4 steps used to build the co-state STM per sub-arc — matches
/// `ARC_SAMPLES_PER_LEG`'s order of magnitude in `MissionPlanner/src/mga.rs`
/// (plenty for a smooth Keplerian coast; no fast dynamics to resolve).
const COSTATE_STM_STEPS: usize = 200;

/// 6×6 state-transition matrix of the co-state pair `(P,Q) = (λV, λV̇)`
/// along a Keplerian coast from `(r0, v0)`, built by integrating
/// [`integrate_costate`] once per canonical basis vector of `(P0, Q0)`
/// (exploits linearity of the ODE — six integrations assemble the matrix
/// column by column). Returns `None` if the underlying Kepler propagation
/// fails anywhere along the arc.
pub fn costate_stm(r0: Vector3<f64>, v0: Vector3<f64>, mu_m3s2: f64, dt_s: f64) -> Option<Matrix6<f64>> {
    let mut phi = Matrix6::<f64>::zeros();
    for j in 0..6 {
        let mut p0 = Vector3::zeros();
        let mut q0 = Vector3::zeros();
        if j < 3 { p0[j] = 1.0; } else { q0[j - 3] = 1.0; }
        let (p, q) = integrate_costate(r0, v0, mu_m3s2, p0, q0, dt_s, COSTATE_STM_STEPS)?;
        phi.fixed_view_mut::<3, 1>(0, j).copy_from(&p);
        phi.fixed_view_mut::<3, 1>(3, j).copy_from(&q);
    }
    Some(phi)
}

/// Sample `‖λV(t)‖` at evenly-spaced points along a sub-arc, given the
/// boundary primer vector/rate at its start.
pub fn sample_primer_magnitude(
    r0: Vector3<f64>, v0: Vector3<f64>, mu_m3s2: f64, dt_s: f64,
    p0: Vector3<f64>, q0: Vector3<f64>, n_samples: usize,
) -> Vec<f64> {
    (0..=n_samples)
        .map(|i| {
            let t = dt_s * i as f64 / n_samples as f64;
            integrate_costate(r0, v0, mu_m3s2, p0, q0, t, (COSTATE_STM_STEPS / 4).max(20))
                .map(|(p, _)| p.norm())
                .unwrap_or(f64::NAN)
        })
        .collect()
}

/// One sub-arc's boundary Kepler states, as already computed by
/// `evaluate_mga_leg`/`evaluate_mga_leg_n` (`mga_leg.rs`) — the caller
/// extracts these from an already-converged chromosome, this module does
/// no ephemeris/Lambert work of its own.
#[derive(Clone, Copy, Debug)]
pub struct SubArc {
    pub r0: Vector3<f64>,
    pub v0: Vector3<f64>,
    pub dt_s: f64,
}

/// How a leg's true boundary (not the interior DSM) constrains `λV` there.
#[derive(Clone, Copy, Debug)]
pub enum LegBoundary {
    /// True mission departure/arrival impulse — `λV` is the FULLY KNOWN
    /// unit vector of that impulse (Olympio Eq. 19/22). No unknown scalar.
    FixedImpulse(Vector3<f64>),
    /// Flyby-adjacent boundary — `λV` is constrained to lie along the
    /// known (unit) incoming/outgoing v∞ direction, with an unknown scalar
    /// magnitude (Olympio Eq. 20/21, the `ν1`/`cν` Lagrange multipliers).
    FlybyDirection(Vector3<f64>),
}

/// Result of [`solve_leg_primer`].
#[derive(Clone, Debug)]
pub struct LegPrimerResult {
    /// Peak `‖λV(t)‖` found across both sub-arcs under the best-fit
    /// (least-squares) boundary solution. `> 1` is the necessary-condition
    /// violation signaling a second impulse would help — but see the
    /// module-level scope note: this is the leg-isolated approximation,
    /// not the full cross-leg-coupled answer.
    pub max_primer_norm: f64,
    /// L2 residual norm of the least-squares fit to the impulse/continuity
    /// equations — near zero means the leg-isolated boundary conditions
    /// were (numerically) exactly satisfiable; a large residual means even
    /// the best achievable choice under this approximation is a poor fit,
    /// so `max_primer_norm` should be read with more caution.
    pub fit_residual: f64,
    /// `‖λV(t)‖` samples along sub-arc A (leg-start → DSM), for plotting.
    pub samples_a: Vec<f64>,
    /// `‖λV(t)‖` samples along sub-arc B (DSM → leg-end), for plotting.
    pub samples_b: Vec<f64>,
}

/// Solve the leg-isolated primer-vector boundary value problem for one
/// MGA-1DSM leg and report whether `‖λV‖ > 1` anywhere.
///
/// # Equation count (see module scope note for why this is over-determined)
///
/// Unknowns (8, or fewer if a boundary is `FixedImpulse`): the two
/// boundary scalars `ν1`/`cν` (1 each, `FlybyDirection` only) and the two
/// free boundary rates `λV̇(start)`, `λV̇(end)` (3 each).
///
/// Equations (10): sub-arc A propagated forward must land exactly on
/// `λV(dsm) = unit(Δv_dsm)` (3); sub-arc B propagated backward must also
/// land exactly on `λV(dsm) = unit(Δv_dsm)` (3); the shared `λV̇(dsm)`
/// from both sub-arcs must agree (continuity, condition 1 — 3 equations);
/// and `λV̇(dsm) ⊥ λV(dsm)` (condition 4, impulse-timing optimality — 1
/// equation). Solved via ordinary least squares (`nalgebra`'s SVD-based
/// solve, the same numerical tool this codebase already uses for other
/// over/under-determined linear systems in `mga.rs`'s multiple shooting).
pub fn solve_leg_primer(
    sub_arc_a: SubArc, sub_arc_b: SubArc, mu_m3s2: f64,
    start_boundary: LegBoundary, end_boundary: LegBoundary,
    dv_dsm_unit: Vector3<f64>,
) -> Option<LegPrimerResult> {
    let phi_a = costate_stm(sub_arc_a.r0, sub_arc_a.v0, mu_m3s2, sub_arc_a.dt_s)?;
    // Sub-arc B is parametrized from ITS OWN start (the DSM, where λV is
    // fully known = dv_dsm_unit) rather than inverted from its end — the
    // unknown there is only λV̇(dsm), shared with sub-arc A's own endpoint
    // (continuity, condition 1), avoiding an explicit matrix inverse.
    let phi_b = costate_stm(sub_arc_b.r0, sub_arc_b.v0, mu_m3s2, sub_arc_b.dt_s)?;

    // Unknown vector x, laid out as: [nu1 (0/1), qa (3), cnu (0/1), q_dsm (3)]
    // where qa = λV̇(start) and q_dsm = λV̇(dsm) (shared by both sub-arcs).
    // The two scalar unknowns (nu1, cnu) are only present when that
    // boundary is a FlybyDirection — their index is unused (never written)
    // when absent, but MUST NOT alias qa/q_dsm's columns, so every write
    // to a scalar's column is guarded by its `_dim == 1` check below.
    let (nu1_dim, d_start) = match start_boundary {
        LegBoundary::FixedImpulse(v) => (0usize, v),
        LegBoundary::FlybyDirection(d) => (1usize, d),
    };
    let (cnu_dim, d_end) = match end_boundary {
        LegBoundary::FixedImpulse(v) => (0usize, v),
        LegBoundary::FlybyDirection(d) => (1usize, d),
    };
    let nu1_idx  = 0usize;
    let qa_idx   = nu1_dim;
    let cnu_idx  = nu1_dim + 3;
    let qdsm_idx = nu1_dim + 3 + cnu_dim;
    let n_unknown = qdsm_idx + 3;

    // Rows: (A: λV(dsm) = unit(dv))×3, (A: λV̇(dsm) = q_dsm)×3,
    //       (B, from the dsm: λV(end) = d_end·cnu)×3,
    //       (perpendicularity: q_dsm · unit(dv) = 0)×1.
    let n_eq = 3 + 3 + 3 + 1;
    let mut mat = nalgebra::DMatrix::<f64>::zeros(n_eq, n_unknown);
    let mut rhs = nalgebra::DVector::<f64>::zeros(n_eq);

    // Sub-arc A: [λV(dsm); λV̇(dsm)] = phi_a · [p_start; qa], where
    // p_start = nu1·d_start (unknown scalar) or d_start (fixed).
    for row in 0..3 {
        let known_contribution = (0..3).map(|k| phi_a[(row, k)] * d_start[k]).sum::<f64>();
        if nu1_dim == 1 {
            mat[(row, nu1_idx)] = known_contribution;
        }
        for k in 0..3 { mat[(row, qa_idx + k)] = phi_a[(row, 3 + k)]; }
        rhs[row] = dv_dsm_unit[row] - if nu1_dim == 0 { known_contribution } else { 0.0 };
    }
    // λV̇(dsm) computed from sub-arc A (its rows 3..6) must equal the
    // shared unknown q_dsm.
    for row in 0..3 {
        let known_contribution = (0..3).map(|k| phi_a[(3 + row, k)] * d_start[k]).sum::<f64>();
        if nu1_dim == 1 {
            mat[(3 + row, nu1_idx)] = known_contribution;
        }
        for k in 0..3 { mat[(3 + row, qa_idx + k)] = phi_a[(3 + row, 3 + k)]; }
        mat[(3 + row, qdsm_idx + row)] -= 1.0;
        rhs[3 + row] = if nu1_dim == 0 { -known_contribution } else { 0.0 };
    }
    // Sub-arc B, propagated FROM the dsm: [λV(end); λV̇(end)] =
    // phi_b · [dv_dsm_unit (known); q_dsm (unknown)]. Only the λV(end)
    // block is a real constraint — λV̇(end) is free (a leg boundary pins
    // λV there, not λV̇, per Olympio Eq. 20-22), so rows 3..6 of phi_b
    // are intentionally unused.
    for row in 0..3 {
        for k in 0..3 { mat[(6 + row, qdsm_idx + k)] = phi_b[(row, 3 + k)]; }
        let known_part = (0..3).map(|k| phi_b[(row, k)] * dv_dsm_unit[k]).sum::<f64>();
        if cnu_dim == 1 {
            mat[(6 + row, cnu_idx)] = -d_end[row];
            rhs[6 + row] = -known_part;
        } else {
            rhs[6 + row] = d_end[row] - known_part;
        }
    }
    // Perpendicularity: q_dsm · unit(dv) = 0 (condition 4).
    for k in 0..3 { mat[(9, qdsm_idx + k)] = dv_dsm_unit[k]; }
    rhs[9] = 0.0;

    let svd = mat.clone().svd(true, true);
    let x = svd.solve(&rhs, 1e-10).ok()?;
    let fit_residual = (&mat * &x - &rhs).norm();

    let qa = Vector3::new(x[qa_idx], x[qa_idx + 1], x[qa_idx + 2]);
    let q_dsm = Vector3::new(x[qdsm_idx], x[qdsm_idx + 1], x[qdsm_idx + 2]);
    let p_start = if nu1_dim == 1 { x[nu1_idx] * d_start } else { d_start };

    let samples_a = sample_primer_magnitude(sub_arc_a.r0, sub_arc_a.v0, mu_m3s2, sub_arc_a.dt_s, p_start, qa, 40);
    let samples_b = sample_primer_magnitude(sub_arc_b.r0, sub_arc_b.v0, mu_m3s2, sub_arc_b.dt_s, dv_dsm_unit, q_dsm, 40);

    let max_primer_norm = samples_a.iter().chain(samples_b.iter())
        .cloned().filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max);

    Some(LegPrimerResult { max_primer_norm, fit_residual, samples_a, samples_b })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU_SUN: f64 = 1.327_124_400_18e20;

    /// Cross-check: negating `G` in the co-state RHS must reproduce the
    /// independently-validated position/velocity STM (built via finite
    /// differences of `propagate_kepler`, which is already exercised by
    /// this codebase's own extensive Lambert/Kepler test suite). This
    /// isolates correctness of the RK4 integrator and the gravity-gradient
    /// tensor formula from the ONE remaining risk (the sign of `G` in the
    /// true co-state ODE, taken from Olympio Eq. 9) — a genuine, targeted
    /// verification rather than trusting the derivation blind.
    #[test]
    fn costate_integrator_matches_state_stm_with_flipped_gravity_sign() {
        // A representative heliocentric elliptical orbit (~1 AU, mildly eccentric).
        let r0 = Vector3::new(1.4e11, 0.2e11, 0.0);
        let v0 = Vector3::new(-3000.0, 29000.0, 500.0);
        let dt_s = 150.0 * 86_400.0;

        // Reference: finite-difference state STM of propagate_kepler.
        let eps_r = 1.0e4; // 10 km, matches this codebase's MS_FD_EPS_POS_M
        let eps_v = 0.05;  // m/s, matches MS_FD_EPS_VEL_MS
        let mut phi_state = Matrix6::<f64>::zeros();
        let (r_ref, v_ref) = propagate_kepler(r0, v0, dt_s, MU_SUN).unwrap();
        for j in 0..6 {
            let mut r_p = r0;
            let mut v_p = v0;
            if j < 3 { r_p[j] += eps_r; } else { v_p[j - 3] += eps_v; }
            let (r_p2, v_p2) = propagate_kepler(r_p, v_p, dt_s, MU_SUN).unwrap();
            let eps = if j < 3 { eps_r } else { eps_v };
            let dr = (r_p2 - r_ref) / eps;
            let dv = (v_p2 - v_ref) / eps;
            phi_state.fixed_view_mut::<3, 1>(0, j).copy_from(&dr);
            phi_state.fixed_view_mut::<3, 1>(3, j).copy_from(&dv);
        }

        // Sign-flipped co-state integration (using -G reproduces the state ODE).
        let mut phi_flipped = Matrix6::<f64>::zeros();
        for j in 0..6 {
            let mut p0 = Vector3::zeros();
            let mut q0 = Vector3::zeros();
            if j < 3 { p0[j] = 1.0; } else { q0[j - 3] = 1.0; }
            // Manually integrate with -G by negating mu in gravity_gradient's
            // effect: easiest is a local RK4 using -gravity_gradient.
            let n = COSTATE_STM_STEPS;
            let h = dt_s / n as f64;
            let (mut p, mut q) = (p0, q0);
            let mut t = 0.0_f64;
            for _ in 0..n {
                let rhs = |r: Vector3<f64>, p: Vector3<f64>, q: Vector3<f64>| {
                    (q, -gravity_gradient(r, MU_SUN) * p)
                };
                let r_t   = propagate_kepler(r0, v0, t, MU_SUN).unwrap().0;
                let r_mid = propagate_kepler(r0, v0, t + 0.5 * h, MU_SUN).unwrap().0;
                let r_end = propagate_kepler(r0, v0, t + h, MU_SUN).unwrap().0;
                let (k1p, k1q) = rhs(r_t, p, q);
                let (k2p, k2q) = rhs(r_mid, p + 0.5 * h * k1p, q + 0.5 * h * k1q);
                let (k3p, k3q) = rhs(r_mid, p + 0.5 * h * k2p, q + 0.5 * h * k2q);
                let (k4p, k4q) = rhs(r_end, p + h * k3p, q + h * k3q);
                p += h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
                q += h / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q);
                t += h;
            }
            phi_flipped.fixed_view_mut::<3, 1>(0, j).copy_from(&p);
            phi_flipped.fixed_view_mut::<3, 1>(3, j).copy_from(&q);
        }

        let diff = (phi_state - phi_flipped).norm();
        let scale = phi_state.norm();
        assert!(
            diff / scale < 1e-4,
            "sign-flipped co-state STM should match state STM: diff/scale={:.3e}",
            diff / scale
        );
    }

    /// The TRUE (unflipped) co-state STM must NOT match the state STM
    /// (confirms the sign genuinely matters and this isn't a degenerate
    /// test that would pass either way).
    #[test]
    fn true_costate_stm_differs_from_state_stm() {
        let r0 = Vector3::new(1.4e11, 0.2e11, 0.0);
        let v0 = Vector3::new(-3000.0, 29000.0, 500.0);
        let dt_s = 150.0 * 86_400.0;
        let phi_costate = costate_stm(r0, v0, MU_SUN, dt_s).unwrap();

        let eps_r = 1.0e4;
        let eps_v = 0.05;
        let mut phi_state = Matrix6::<f64>::zeros();
        let (r_ref, v_ref) = propagate_kepler(r0, v0, dt_s, MU_SUN).unwrap();
        for j in 0..6 {
            let mut r_p = r0;
            let mut v_p = v0;
            if j < 3 { r_p[j] += eps_r; } else { v_p[j - 3] += eps_v; }
            let (r_p2, v_p2) = propagate_kepler(r_p, v_p, dt_s, MU_SUN).unwrap();
            let eps = if j < 3 { eps_r } else { eps_v };
            let dr = (r_p2 - r_ref) / eps;
            let dv = (v_p2 - v_ref) / eps;
            phi_state.fixed_view_mut::<3, 1>(0, j).copy_from(&dr);
            phi_state.fixed_view_mut::<3, 1>(3, j).copy_from(&dv);
        }
        let diff = (phi_state - phi_costate).norm();
        assert!(diff / phi_state.norm() > 0.1, "sanity: true co-state STM should differ substantially from state STM");
    }

    #[test]
    fn gravity_gradient_is_symmetric_and_traceless() {
        let r = Vector3::new(1.4e11, 0.3e11, -0.1e11);
        let g = gravity_gradient(r, MU_SUN);
        assert!((g - g.transpose()).norm() < 1e-20 * g.norm().max(1.0), "G must be symmetric");
        // trace(I - 3 r_hat r_hat^T) = 3 - 3 = 0, so G is traceless (Laplace's equation for a point mass).
        assert!(g.trace().abs() < 1e-6 * g.norm(), "G must be traceless (Laplace eq.): trace={:.3e}", g.trace());
    }

    /// Solve a leg where the impulse direction happens to point exactly
    /// along both boundary directions (a degenerate but well-posed case) —
    /// confirms the linear system assembles and solves without panicking
    /// and gives a finite, sane result.
    #[test]
    fn solve_leg_primer_runs_and_gives_finite_result() {
        let r_start = Vector3::new(1.4e11, 0.0, 0.0);
        let v_start = Vector3::new(0.0, 29000.0, 0.0);
        let dt1 = 60.0 * 86_400.0;
        let (r_dsm, v_dsm) = propagate_kepler(r_start, v_start, dt1, MU_SUN).unwrap();

        // Fabricate a small DSM and a second sub-arc.
        let dv = Vector3::new(50.0, -20.0, 10.0);
        let v_after = v_dsm + dv;
        let dt2 = 90.0 * 86_400.0;
        let (r_end, _v_end) = propagate_kepler(r_dsm, v_after, dt2, MU_SUN).unwrap();
        let v_end_actual = propagate_kepler(r_dsm, v_after, dt2, MU_SUN).unwrap().1;

        let sub_a = SubArc { r0: r_start, v0: v_start, dt_s: dt1 };
        let sub_b = SubArc { r0: r_dsm, v0: v_after, dt_s: dt2 };
        let dv_unit = dv.normalize();

        let result = solve_leg_primer(
            sub_a, sub_b, MU_SUN,
            LegBoundary::FlybyDirection(v_start.normalize()),
            LegBoundary::FlybyDirection(v_end_actual.normalize()),
            dv_unit,
        ).expect("solve should succeed");

        assert!(result.max_primer_norm.is_finite());
        assert!(result.fit_residual.is_finite());
        assert_eq!(result.samples_a.len(), 41);
        assert_eq!(result.samples_b.len(), 41);
        let _ = r_end;
    }
}
