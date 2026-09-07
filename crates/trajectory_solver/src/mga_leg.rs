//! MGA-1DSM leg evaluator and flyby turn geometry.
//!
//! Implements the per-leg arithmetic of the MGA-1DSM velocity formulation:
//! one deep-space manoeuvre (DSM) per leg, freely positioned at fraction η of
//! the leg time-of-flight. Each leg is split into two sub-arcs:
//!
//! 1. **Sub-arc 1**: Keplerian propagation from the departure state
//!    `(r_sc, v_sc)` for `η·T` seconds under point-mass solar gravity →
//!    DSM position `(r_dsm, v_dsm)`.
//! 2. **Sub-arc 2**: Lambert arc from `r_dsm` to the next body's position at
//!    arrival epoch in the remaining `(1−η)·T` seconds → velocities `(v_L1, v_L2)`.
//! 3. DSM impulse: `ΔV_DSM = |v_L1 − v_dsm|`.
//! 4. Arrival hyperbolic excess: `v_∞_arr = v_L2 − v_body_next`.
//!
//! The unpowered gravity-assist flyby turn is implemented separately in
//! [`flyby_turn`] — it is applied *between* legs to connect the arrival v_∞
//! of one leg to the departure v_∞ of the next.
//!
//! This module contains no ephemeris queries and no mission-specific config.
//! It operates entirely on caller-supplied vectors and scalars (SI units).
//!
//! # References
//! - Ceriotti (2010), *Global Optimisation of Multiple Gravity Assist
//!   Trajectories*, PhD Thesis, University of Glasgow, §2.4.
//! - Izzo (2010), "Revisiting Lambert's Problem", Celest. Mech. Dyn. Astron.
//!   121:1–15 (the Lambert solver used in sub-arc 2).
//! - Strange & Longuski (2002), "Graphical Method for Gravity-Assist
//!   Trajectory Design", JGCD 25(6):1154–1159 (flyby geometry background).

use nalgebra::Vector3;
use orbital_math::{kepler::propagate_kepler, lambert::{lambert_min_dv, lambert_min_dv_multi_rev}};

use crate::nelder_mead::NelderMead;
use crate::vilm::{solve_tangent_vilt, VilmDomain, VilmSolution};

/// Maximum extra revolutions tried per leg (Phase 9x-iv) — real
/// resonant-return MGA legs (VEEGA's Earth->Earth hop, Cassini-class
/// Venus->Venus returns) need at most 1-2 extra loops; higher N transfers
/// are astronomically expensive in TOF for essentially no ΔV benefit, so
/// searching further is not worth the added evaluation cost.
const MAX_LEG_N_REV: u32 = 2;

/// [`MAX_LEG_N_REV`], overridable via the `MGA_MAX_LEG_N_REV` env var —
/// diagnostic knob for isolating multi-rev Lambert's effect on
/// a search without a rebuild per experiment: `MGA_MAX_LEG_N_REV=0`
/// reproduces the pre-multi-rev evaluator exactly (the n_rev=0 path is
/// byte-identical to the old `lambert_min_dv`, same transfer-angle guard).
fn max_leg_n_rev() -> u32 {
    std::env::var("MGA_MAX_LEG_N_REV")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(MAX_LEG_N_REV)
}

/// Result of evaluating one MGA-DSM leg.
///
/// All quantities are in heliocentric inertial coordinates (SI units).
#[derive(Clone, Debug)]
pub struct MgaLegResult {
    /// Deep-space manoeuvre impulse magnitude [m/s].
    pub dv_dsm_ms: f64,
    /// Spacecraft position at the DSM point [m], heliocentric.
    pub r_dsm_m: Vector3<f64>,
    /// Spacecraft velocity just *before* the DSM [m/s], heliocentric.
    /// `v_dsm_before + ΔV_DSM_vector = v_dsm_after = v_L1` (Lambert departure).
    pub v_dsm_before_mps: Vector3<f64>,
    /// Spacecraft velocity just *after* the DSM [m/s] = Lambert departure velocity
    /// `v_L1` at the DSM point. Used by the re-propagation step to seed segment B
    /// of each leg with the correct post-burn state under real dynamics.
    pub v_dsm_after_mps: Vector3<f64>,
    /// Spacecraft heliocentric velocity at arrival, as produced by the Lambert
    /// arc (`v_L2`) — subtract the next body's velocity to get `v_∞_arr`.
    pub v_arrival_helio_mps: Vector3<f64>,
    /// Hyperbolic excess velocity on arrival at the next body [m/s].
    /// `v_∞_arr = v_arrival_helio − v_body_next`. This is the incoming v_∞
    /// that feeds into [`flyby_turn`] for intermediate bodies, or into the
    /// LOI calculation for the final body.
    pub v_inf_arr_mps: Vector3<f64>,
    /// Perihelion [m] and aphelion [m] of the *departure* sub-arc (sub-arc 1:
    /// Keplerian propagation from the departure body to the DSM point).
    /// Used by the Tisserand graph overlay — this point lies on the departure
    /// body's Tisserand contour for the departure v∞.
    pub rp_dep_m: f64,
    pub ra_dep_m: f64,
    /// Perihelion [m] and aphelion [m] of the *Lambert* sub-arc (sub-arc 2:
    /// Lambert arc from the DSM point to the arrival body).
    /// Used by the Tisserand graph overlay — this point lies on the arrival
    /// body's Tisserand contour for the arrival v∞.
    pub rp_lambert_m: f64,
    pub ra_lambert_m: f64,
}

/// Evaluate one MGA-1DSM leg from a known spacecraft departure state to a
/// known next-body position/velocity.
///
/// # Arguments
/// * `r_sc_start` — spacecraft heliocentric position at leg start [m]
/// * `v_sc_start` — spacecraft heliocentric velocity at leg start [m/s]
///   (already includes the outgoing flyby v_∞ for intermediate legs)
/// * `eta` — fractional DSM position ∈ (0, 1)
/// * `tof_s` — total leg time-of-flight [s]
/// * `r_body_next` — next body's heliocentric position at arrival epoch [m]
/// * `v_body_next` — next body's heliocentric velocity at arrival epoch [m/s]
/// * `mu_sun` — Sun's gravitational parameter [m³/s²]
///
/// Returns `None` when no elliptic Lambert solution exists for the given
/// geometry (sub-arc 2 is geometrically infeasible), or when the Keplerian
/// propagation of sub-arc 1 fails to converge. The caller should treat `None`
/// as an infeasible chromosome and penalise accordingly.
pub fn evaluate_mga_leg(
    r_sc_start:   Vector3<f64>,
    v_sc_start:   Vector3<f64>,
    eta:          f64,
    tof_s:        f64,
    r_body_next:  Vector3<f64>,
    v_body_next:  Vector3<f64>,
    mu_sun:       f64,
) -> Option<MgaLegResult> {
    // ⚠ GREEDY branch selection — diagnostic/single-point use only. Inside
    // an optimizer's fitness loop the greedy leg-local pick makes fitness
    // discontinuous in the decision variables and judges the branch by one
    // leg's cost while its consequences land downstream (confirmed on GTOP
    // Cassini-2 — see the design notes). Search code must use
    // [`evaluate_mga_leg_n`] with an explicit chromosome-supplied `n_rev`.
    evaluate_mga_leg_inner(r_sc_start, v_sc_start, eta, tof_s, r_body_next, v_body_next, mu_sun, None)
}

/// [`evaluate_mga_leg`] with the Lambert revolution count fixed to `n_rev`
/// instead of greedily searched — the sub-arc-2 transfer uses exactly
/// `n_rev` extra revolutions (still min-ΔV over prograde/retrograde within
/// that N). `n_rev` comes from an explicit chromosome gene (
/// N-as-gene fix), so the optimizer's full-mission fitness — not a
/// leg-local rule — judges the branch choice, and the fitness stays
/// continuous within each fixed-N family. Returns `None` when no solution
/// with that revolution count geometrically exists for this (r1, r2, TOF).
pub fn evaluate_mga_leg_n(
    r_sc_start:   Vector3<f64>,
    v_sc_start:   Vector3<f64>,
    eta:          f64,
    tof_s:        f64,
    r_body_next:  Vector3<f64>,
    v_body_next:  Vector3<f64>,
    mu_sun:       f64,
    n_rev:        u32,
) -> Option<MgaLegResult> {
    evaluate_mga_leg_inner(r_sc_start, v_sc_start, eta, tof_s, r_body_next, v_body_next, mu_sun, Some(n_rev))
}

/// Shared body of [`evaluate_mga_leg`]/[`evaluate_mga_leg_n`]:
/// `n_rev = None` → greedy min-ΔV over N=0..=[`max_leg_n_rev`] (legacy /
/// diagnostic); `Some(n)` → exactly that revolution count.
fn evaluate_mga_leg_inner(
    r_sc_start:   Vector3<f64>,
    v_sc_start:   Vector3<f64>,
    eta:          f64,
    tof_s:        f64,
    r_body_next:  Vector3<f64>,
    v_body_next:  Vector3<f64>,
    mu_sun:       f64,
    n_rev:        Option<u32>,
) -> Option<MgaLegResult> {
    // Sub-arc 1: Keplerian propagation to DSM point.
    let dt1 = eta * tof_s;
    let dt2 = (1.0 - eta) * tof_s;
    if dt1 <= 0.0 || dt2 <= 0.0 { return None; }

    let (r_dsm, v_dsm) = propagate_kepler(r_sc_start, v_sc_start, dt1, mu_sun)?;

    // Sub-arc 2: Lambert arc from DSM to next body in remaining time.
    // Multi-revolution branches (N ≥ 1) are what make genuine resonant-
    // return legs representable at all (e.g. VEEGA's ~2-year Earth->Earth
    // hop sweeps the Sun ~2 extra times — plain N=0 Lambert structurally
    // cannot produce that family; Phase 9x-iv). WHICH branch is
    // used is the caller's decision — see the two public wrappers above for
    // why greedy in-evaluator selection is only safe outside search loops.
    let r_dsm_v3  = [r_dsm.x, r_dsm.y, r_dsm.z];
    let r_next_v3 = [r_body_next.x, r_body_next.y, r_body_next.z];
    let v_dsm_v3  = [v_dsm.x,  v_dsm.y,  v_dsm.z];
    let v_body_v3 = [v_body_next.x, v_body_next.y, v_body_next.z];

    // v_dsm is passed as v_dep so the min-ΔV pick (over prograde/retrograde,
    // and over N too in the greedy case) minimises
    // |v_L1 − v_dsm| + |v_L2 − v_body_next| — the combined DSM +
    // arrival-excess cost of this leg.
    let lambert_result = match n_rev {
        Some(nr) => orbital_math::lambert::lambert_min_dv_at_n_rev(
            r_dsm_v3, r_next_v3, dt2, mu_sun, v_dsm_v3, v_body_v3, nr,
        ).map(|(v1, v2)| (v1, v2, nr)),
        None => lambert_min_dv_multi_rev(
            r_dsm_v3, r_next_v3, dt2, mu_sun, v_dsm_v3, v_body_v3, max_leg_n_rev(),
        ),
    };
    if std::env::var("MGA_DEBUG").is_ok() {
        let mag1 = (r_dsm_v3[0].powi(2)+r_dsm_v3[1].powi(2)+r_dsm_v3[2].powi(2)).sqrt();
        let mag2 = (r_next_v3[0].powi(2)+r_next_v3[1].powi(2)+r_next_v3[2].powi(2)).sqrt();
        let dotp = r_dsm_v3[0]*r_next_v3[0]+r_dsm_v3[1]*r_next_v3[1]+r_dsm_v3[2]*r_next_v3[2];
        let cos_dnu = (dotp/(mag1*mag2)).clamp(-1.0,1.0);
        eprintln!(
            "[MGA_DEBUG] dt1={:.1}d dt2={:.1}d |r1|={:.3e} |r2|={:.3e} dnu_deg={:.2} lambert_ok={} n_rev={:?}",
            dt1/86400.0, dt2/86400.0, mag1, mag2, cos_dnu.acos().to_degrees(), lambert_result.is_some(),
            lambert_result.map(|(_, _, n)| n),
        );
    }
    let (v_L1_arr, v_L2_arr, _n_rev_used) = lambert_result?;

    let v_L1 = Vector3::new(v_L1_arr[0], v_L1_arr[1], v_L1_arr[2]);
    let v_L2 = Vector3::new(v_L2_arr[0], v_L2_arr[1], v_L2_arr[2]);

    let dv_dsm_ms    = (v_L1 - v_dsm).norm();
    let v_inf_arr    = v_L2 - v_body_next;

    let (rp_dep_m, ra_dep_m)         = orbit_apsides(r_sc_start, v_sc_start, mu_sun);
    let (rp_lambert_m, ra_lambert_m) = orbit_apsides(r_dsm,      v_L1,       mu_sun);

    Some(MgaLegResult {
        dv_dsm_ms,
        r_dsm_m:             r_dsm,
        v_dsm_before_mps:    v_dsm,
        v_dsm_after_mps:     v_L1,
        v_arrival_helio_mps: v_L2,
        v_inf_arr_mps:       v_inf_arr,
        rp_dep_m,
        ra_dep_m,
        rp_lambert_m,
        ra_lambert_m,
    })
}

/// Result of [`evaluate_vilm_leg`]: a leg evaluated with sub-arc 2 modelled
/// as a tangent VILT (`crate::vilm`) instead of a plain Lambert arc.
#[derive(Clone, Debug)]
pub struct VilmLegResult {
    /// Sub-arc 1 (departure Kepler coast) and the departure-side impulse
    /// only — `dv_dsm_ms` here is the mismatch between the natural coast
    /// arriving at the DSM point and the VILM low-arc's required departure
    /// velocity, NOT the leg's total cost (see `dv_leverage_ms` below).
    pub leg: MgaLegResult,
    /// The VILT's own internal leveraging maneuver at its shared apse
    /// `r_c_m` [m/s] — a SECOND impulse this leg spends beyond `leg.dv_dsm_ms`.
    /// A leg's true total DSM cost is `leg.dv_dsm_ms + dv_leverage_ms`.
    pub dv_leverage_ms: f64,
    /// Leveraging apse radius [m] (heliocentric distance of the internal burn).
    pub r_c_m: f64,
}

/// Evaluate one MGA-1DSM leg with sub-arc 2 (DSM point → next body) modelled
/// as a tangent VILT instead of a plain Lambert arc — see `crate::vilm` for
/// the model and motivation (this is the field's purpose-built technique for
/// a resonant/repeat-flyby leg, where a plain free-position-DSM Lambert arc
/// has no mechanism to shape the resulting v∞ for what the NEXT leg needs).
///
/// Sub-arc 1 (the pre-DSM Kepler coast) is UNCHANGED from
/// [`evaluate_mga_leg_n`] — same `eta`, same departure-coast physics; only
/// the arc from the DSM point onward differs. `domain`/`k_low`/`k_high` are
/// explicit caller-supplied choices (not greedily searched inside this
/// function) — greedy in-evaluator branch selection is exactly the class of
/// bug already found and fixed once for Lambert's N-rev choice (makes
/// fitness discontinuous in the chromosome; see `evaluate_mga_leg`'s own
/// doc warning). Callers needing to choose among domains/k should do so
/// as an explicit, fixed, outer decision — a chromosome gene if wired into
/// a live search, or a diagnostic sweep if not (as in
/// `MissionPlanner/src/bin/vilm_leg_check.rs`).
///
/// Returns `None` when sub-arc 1's Kepler propagation fails to converge, or
/// when no tangent VILT solution exists for the given `(r_dsm, r_body_next,
/// remaining TOF, domain)` — treat as infeasible, same convention as
/// [`evaluate_mga_leg_n`].
pub fn evaluate_vilm_leg(
    r_sc_start:   Vector3<f64>,
    v_sc_start:   Vector3<f64>,
    eta:          f64,
    tof_s:        f64,
    r_body_next:  Vector3<f64>,
    v_body_next:  Vector3<f64>,
    mu_sun:       f64,
    domain:       VilmDomain,
    prograde:     bool,
    k_low:        u32,
    k_high:       u32,
    solution:     VilmSolution,
) -> Option<VilmLegResult> {
    let dt1 = eta * tof_s;
    let dt2 = (1.0 - eta) * tof_s;
    if dt1 <= 0.0 || dt2 <= 0.0 { return None; }

    let (r_dsm, v_dsm) = propagate_kepler(r_sc_start, v_sc_start, dt1, mu_sun)?;

    let vilm = solve_tangent_vilt(r_dsm, r_body_next, dt2, mu_sun, domain, prograde, k_low, k_high, solution)?;

    let dv_dsm_ms = (vilm.v_low - v_dsm).norm();
    let v_inf_arr = vilm.v_high - v_body_next;

    let (rp_dep_m, ra_dep_m)         = orbit_apsides(r_sc_start, v_sc_start, mu_sun);
    let (rp_lambert_m, ra_lambert_m) = orbit_apsides(r_dsm,      vilm.v_low, mu_sun);

    Some(VilmLegResult {
        leg: MgaLegResult {
            dv_dsm_ms,
            r_dsm_m:             r_dsm,
            v_dsm_before_mps:    v_dsm,
            v_dsm_after_mps:     vilm.v_low,
            v_arrival_helio_mps: vilm.v_high,
            v_inf_arr_mps:       v_inf_arr,
            rp_dep_m,
            ra_dep_m,
            rp_lambert_m,
            ra_lambert_m,
        },
        dv_leverage_ms: vilm.dv_leverage_ms.abs(),
        r_c_m: vilm.r_c_m,
    })
}

/// Result of [`evaluate_mga_leg_2dsm`]: a leg refined with a SECOND free
/// impulse inserted somewhere along its length, on top of the existing
/// Lambert-anchored (implicit) DSM.
///
/// This is a LOCAL REFINEMENT model only (Phase 9x) — it is not
/// wired into the chromosome/DE-MBH search at all. See
/// [`refine_leg_two_dsm`]'s doc comment for the motivation (Lawden/Olympio
/// primer-vector necessary-condition violations found on already-converged
/// ordinary legs, `crate::primer_vector`) and why this stays outside the
/// main search's dimensionality rather than growing the chromosome.
#[derive(Clone, Debug)]
pub struct TwoDsmLegResult {
    /// The leg's Lambert-anchored second impulse and downstream quantities
    /// (arrival v∞, apsides of the post-impulse-2 sub-arc), in the SAME
    /// shape as a plain one-DSM leg's [`MgaLegResult`] — `leg.dv_dsm_ms` here
    /// is impulse 2's magnitude, `leg.r_dsm_m`/`v_dsm_before_mps`/
    /// `v_dsm_after_mps` describe impulse 2's location and pre/post state.
    /// `leg.rp_dep_m`/`ra_dep_m` describe the FIRST sub-arc (start → impulse
    /// 1), unchanged in meaning from the one-DSM model.
    pub leg: MgaLegResult,
    /// First (free) impulse magnitude [m/s], applied at `r1_m` (heliocentric
    /// position at fraction `eta1` of the leg TOF).
    pub dv1_ms: f64,
    /// Position of the first impulse [m], heliocentric.
    pub r1_m: Vector3<f64>,
    /// `eta1` actually used (echoed back for the caller's convenience).
    pub eta1: f64,
    /// `eta2` actually used (echoed back for the caller's convenience).
    pub eta2: f64,
}

/// Evaluate one MGA-DSM leg with a SECOND free impulse inserted between the
/// leg's start and its Lambert-anchored end point — three sub-arcs instead
/// of the one-DSM model's two:
///
/// 1. Kepler coast from `(r_sc_start, v_sc_start)` for `eta1·T` → impulse 1
///    point `(r1, v1_before)`.
/// 2. Impulse 1: a genuinely FREE 3-vector `dv1` (not solved for — this is
///    the model's one new degree of freedom vs. the one-DSM leg) →
///    `v1_after = v1_before + dv1`.
/// 3. Kepler coast from `(r1, v1_after)` for `(eta2 − eta1)·T` → impulse 2
///    point `(r2, v2_before)`.
/// 4. Impulse 2: Lambert-anchored (implicit, exactly like the one-DSM
///    model's single DSM) from `r2` to `r_body_next` in the remaining
///    `(1 − eta2)·T`.
///
/// Requires `0 < eta1 < eta2 < 1` (returns `None` otherwise, same
/// infeasibility convention as [`evaluate_mga_leg_n`]). Deliberately uses a
/// plain `n_rev = 0` Lambert solve for the anchored segment (this is a small
/// local patch on an already-converged leg, not a global search — no
/// resonant-length coast is expected in this final short segment).
pub fn evaluate_mga_leg_2dsm(
    r_sc_start:   Vector3<f64>,
    v_sc_start:   Vector3<f64>,
    eta1:         f64,
    eta2:         f64,
    dv1:          Vector3<f64>,
    tof_s:        f64,
    r_body_next:  Vector3<f64>,
    v_body_next:  Vector3<f64>,
    mu_sun:       f64,
) -> Option<TwoDsmLegResult> {
    let dt1 = eta1 * tof_s;
    let dt2 = (eta2 - eta1) * tof_s;
    let dt3 = (1.0 - eta2) * tof_s;
    if dt1 <= 0.0 || dt2 <= 0.0 || dt3 <= 0.0 { return None; }

    let (r1, v1_before) = propagate_kepler(r_sc_start, v_sc_start, dt1, mu_sun)?;
    let v1_after = v1_before + dv1;
    let (r2, v2_before) = propagate_kepler(r1, v1_after, dt2, mu_sun)?;

    let r2_v3     = [r2.x, r2.y, r2.z];
    let r_next_v3 = [r_body_next.x, r_body_next.y, r_body_next.z];
    let v2_v3     = [v2_before.x, v2_before.y, v2_before.z];
    let v_body_v3 = [v_body_next.x, v_body_next.y, v_body_next.z];

    let (v_L1_arr, v_L2_arr) = lambert_min_dv(r2_v3, r_next_v3, dt3, mu_sun, v2_v3, v_body_v3)?;
    let v_L1 = Vector3::new(v_L1_arr[0], v_L1_arr[1], v_L1_arr[2]);
    let v_L2 = Vector3::new(v_L2_arr[0], v_L2_arr[1], v_L2_arr[2]);

    let dv2_ms    = (v_L1 - v2_before).norm();
    let v_inf_arr = v_L2 - v_body_next;

    let (rp_dep_m, ra_dep_m)         = orbit_apsides(r_sc_start, v_sc_start, mu_sun);
    let (rp_lambert_m, ra_lambert_m) = orbit_apsides(r2, v_L1, mu_sun);

    Some(TwoDsmLegResult {
        leg: MgaLegResult {
            dv_dsm_ms:           dv2_ms,
            r_dsm_m:             r2,
            v_dsm_before_mps:    v2_before,
            v_dsm_after_mps:     v_L1,
            v_arrival_helio_mps: v_L2,
            v_inf_arr_mps:       v_inf_arr,
            rp_dep_m,
            ra_dep_m,
            rp_lambert_m,
            ra_lambert_m,
        },
        dv1_ms: dv1.norm(),
        r1_m: r1,
        eta1,
        eta2,
    })
}

/// Result of [`refine_leg_two_dsm`].
#[derive(Clone, Debug)]
pub struct TwoDsmRefineResult {
    /// Total leg ΔV under the best two-impulse solution found
    /// (`dv1_ms + leg.dv_dsm_ms`) — compare directly against the leg's
    /// original one-DSM `dv_dsm_ms` to judge whether the second impulse is
    /// a genuine improvement.
    pub total_dv_ms: f64,
    pub leg: TwoDsmLegResult,
    /// Underlying Nelder-Mead run info (iterations, convergence history).
    pub nm: crate::nelder_mead::NmResult,
}

/// Local-refinement search for [`evaluate_mga_leg_2dsm`]'s two free impulse
/// times and the first impulse's 3-vector, over an already-converged leg.
///
/// This is deliberately NOT part of the chromosome/DE-MBH search — Phase 9x
/// found that ordinary (non-resonant) legs in already-converged
/// MGA winners show real Lawden/Olympio primer-vector necessary-condition
/// violations (`crate::primer_vector::solve_leg_primer`, `‖λV‖ > 1`),
/// meaning a second impulse would provably reduce that leg's cost — but
/// this project has independently found (Ceriotti's own thesis data, and
/// this codebase's own all-at-once-DE experience) that blindly growing
/// chromosome dimensionality for every leg degrades search quality more
/// than it helps. So instead: run the main search once, diagnose which
/// specific legs actually need a second impulse, then locally polish just
/// those legs with a small dedicated Nelder-Mead — cheap, targeted, and
/// leaves the global search's dimensionality untouched.
///
/// `eta1_seed`/`eta2_seed` should bracket wherever the primer-vector
/// diagnostic found its peak `‖λV‖` violation (the physically meaningful
/// place to try inserting the extra impulse) — the caller is expected to
/// sort them so `eta1_seed < eta2_seed`, but this function does not require
/// it (both are free chromosome-style parameters; [`evaluate_mga_leg_2dsm`]'s
/// own `0 < eta1 < eta2 < 1` check rejects any infeasible ordering the
/// simplex wanders into, exactly like a normal infeasible chromosome).
///
/// `dv1_bound_ms` bounds each component of the free first impulse — set it
/// generously above the leg's own one-DSM cost (there's no reason a genuine
/// improvement would need a *larger* impulse than the one it replaces).
pub fn refine_leg_two_dsm(
    r_sc_start:    Vector3<f64>,
    v_sc_start:    Vector3<f64>,
    tof_s:         f64,
    r_body_next:   Vector3<f64>,
    v_body_next:   Vector3<f64>,
    mu_sun:        f64,
    eta1_seed:     f64,
    eta2_seed:     f64,
    dv1_bound_ms:  f64,
    nm:            &NelderMead,
) -> Option<TwoDsmRefineResult> {
    let bounds = vec![
        (0.001, 0.999),
        (0.001, 0.999),
        (-dv1_bound_ms, dv1_bound_ms),
        (-dv1_bound_ms, dv1_bound_ms),
        (-dv1_bound_ms, dv1_bound_ms),
    ];
    let x0 = [
        eta1_seed.clamp(0.002, 0.998),
        eta2_seed.clamp(0.002, 0.998),
        0.0, 0.0, 0.0,
    ];

    let fitness = |x: &[f64]| -> Option<f64> {
        let dv1 = Vector3::new(x[2], x[3], x[4]);
        let result = evaluate_mga_leg_2dsm(
            r_sc_start, v_sc_start, x[0], x[1], dv1, tof_s, r_body_next, v_body_next, mu_sun,
        )?;
        Some(dv1.norm() + result.leg.dv_dsm_ms)
    };

    let nm_result = nm.run(&bounds, &x0, fitness);
    if nm_result.best_fitness >= f64::MAX { return None; }

    let dv1 = Vector3::new(nm_result.best_params[2], nm_result.best_params[3], nm_result.best_params[4]);
    let leg = evaluate_mga_leg_2dsm(
        r_sc_start, v_sc_start, nm_result.best_params[0], nm_result.best_params[1],
        dv1, tof_s, r_body_next, v_body_next, mu_sun,
    )?;

    Some(TwoDsmRefineResult {
        total_dv_ms: nm_result.best_fitness,
        leg,
        nm: nm_result,
    })
}

/// Compute perihelion and aphelion of the heliocentric orbit defined by (r, v).
///
/// Returns `(Rp, Ra)` in metres.  For a hyperbolic orbit (energy ≥ 0),
/// returns `(r, f64::INFINITY)` as a sentinel — the caller should skip
/// Tisserand-graph plotting for such legs.
fn orbit_apsides(r: Vector3<f64>, v: Vector3<f64>, mu: f64) -> (f64, f64) {
    let r_mag  = r.norm();
    let energy = 0.5 * v.dot(&v) - mu / r_mag;
    if energy >= 0.0 {
        return (r_mag, f64::INFINITY);
    }
    let a   = -mu / (2.0 * energy);
    let h   = r.cross(&v).norm();
    let p   = h * h / mu;
    let e   = (1.0 - p / a).max(0.0).sqrt();
    (a * (1.0 - e), a * (1.0 + e))
}

/// Compute the outgoing hyperbolic excess velocity after an unpowered
/// gravity-assist flyby.
///
/// The flyby is modelled as an instantaneous rotation of the incoming v_∞
/// vector by the gravitational turn angle `δ`, about an axis `n̂` in the
/// B-plane (the plane perpendicular to the incoming v_∞). The B-plane axis
/// is parameterised by the crank angle `beta_rad`.
///
/// Energy is exactly conserved: `|v_∞_out| = |v_∞_in|`.
///
/// # Arguments
/// * `v_inf_in`  — incoming hyperbolic excess velocity [m/s] (body-centred)
/// * `r_p_m`     — flyby periapsis radius [m] (must be > body surface radius)
/// * `beta_rad`  — B-plane crank angle [rad] ∈ (−π, π)
/// * `mu_body`   — flyby body's gravitational parameter [m³/s²]
///
/// # References
/// - Ceriotti (2010), PhD Thesis §2.3 — B-plane parameterisation.
/// - Battin (1999), §6.3 — hyperbolic flyby turn angle formula.
/// - Rodrigues rotation formula applied to the B-plane: Vallado (2013), §A.5.
pub fn flyby_turn(
    v_inf_in: Vector3<f64>,
    r_p_m:    f64,
    beta_rad: f64,
    mu_body:  f64,
) -> Vector3<f64> {
    let v_inf_mag = v_inf_in.norm();
    if v_inf_mag < 1.0 { return v_inf_in; } // degenerate: no meaningful turn

    // Hyperbolic flyby turn angle.
    // From the conic section of a hyperbolic orbit:
    //   e = 1 + r_p · v_∞² / μ
    //   sin(δ/2) = 1/e   →   δ = 2·arcsin(μ / (μ + r_p·v_∞²))
    // Reference: Battin (1999) §6.3.
    let e     = 1.0 + r_p_m * v_inf_mag * v_inf_mag / mu_body;
    let delta = 2.0 * (1.0 / e).asin();

    // S-hat: unit vector along incoming v_∞ (the "velocity" direction in the
    // B-plane coordinate system).
    let s_hat = v_inf_in / v_inf_mag;

    // Build an orthonormal basis (T_hat, R_hat) in the B-plane, perpendicular
    // to s_hat. Choose a stable reference vector: use [1,0,0] unless s_hat is
    // nearly parallel to it, then fall back to [0,1,0].
    let ref_vec = if s_hat.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let t_hat = s_hat.cross(&ref_vec).normalize();
    let r_hat = s_hat.cross(&t_hat); // already unit length: s⊥t, both unit

    // Rotation axis in the B-plane, parameterised by the crank angle beta.
    // beta = 0 → axis along T_hat; beta = π/2 → axis along R_hat.
    let n_hat = beta_rad.cos() * t_hat + beta_rad.sin() * r_hat;

    // Rodrigues rotation of v_inf_in by angle delta about n_hat.
    // Since n_hat ⊥ v_inf_in (n_hat is in the B-plane, v_inf_in is normal
    // to it), the formula simplifies to:
    //   v_inf_out = v_inf_in·cos(δ) + (n_hat × v_inf_in)·sin(δ)
    // This exactly conserves |v_inf|. Reference: Vallado (2013) §A.5.
    v_inf_in * delta.cos() + n_hat.cross(&v_inf_in) * delta.sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU_SUN: f64 = 1.327_124_400_18e20;
    const MU_EARTH: f64 = 3.986_004_418e14;
    const AU: f64 = 1.496e11;

    /// flyby_turn must conserve |v_∞| exactly for any (beta, r_p) input.
    #[test]
    fn flyby_preserves_vinf_magnitude() {
        let v_inf_in = Vector3::new(3_000.0, 1_000.0, 500.0); // arbitrary [m/s]
        let r_p = 7_000e3_f64; // 7000 km periapsis
        for beta in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
            let v_out = flyby_turn(v_inf_in, r_p, beta, MU_EARTH);
            let ratio = v_out.norm() / v_inf_in.norm();
            assert!((ratio - 1.0).abs() < 1e-10,
                "beta={beta}: |v_out|/|v_in| = {ratio:.10}  (should be 1.0)");
        }
    }

    /// Very large r_p → near-zero turn angle → v_∞_out ≈ v_∞_in.
    #[test]
    fn flyby_large_periapsis_is_no_turn() {
        let v_inf_in = Vector3::new(4_000.0, 0.0, 0.0);
        let r_p      = 1e15_f64; // essentially infinite periapsis
        let v_out    = flyby_turn(v_inf_in, r_p, 0.0, MU_EARTH);
        let diff     = (v_out - v_inf_in).norm();
        assert!(diff < 1e-3, "large r_p should produce ~0 turn; diff = {diff:.3e} m/s");
    }

    /// Evaluate a near-Hohmann leg: Earth at 1 AU, Mars at ~1.52 AU,
    /// η=0.5, TOF = 259 days (Hohmann half-period). DSM ΔV should be tiny
    /// because the spacecraft is already on a near-Hohmann path.
    #[test]
    fn near_hohmann_leg_small_dsm() {
        // Circular Earth orbit departure state.
        let a_earth = 1.0 * AU;
        let v_earth = (MU_SUN / a_earth).sqrt();
        let r_earth = Vector3::new(a_earth, 0.0, 0.0);
        let v_earth_vec = Vector3::new(0.0, v_earth, 0.0);

        // Hohmann transfer to Mars orbit (1.524 AU).
        let a_mars   = 1.524 * AU;
        let a_xfer   = 0.5 * (a_earth + a_mars);
        let v_dep    = (MU_SUN * (2.0 / a_earth - 1.0 / a_xfer)).sqrt();
        let tof_s    = std::f64::consts::PI * (a_xfer.powi(3) / MU_SUN).sqrt();

        // After exactly half the transfer, the spacecraft is at Mars's orbit.
        let v_sc     = Vector3::new(0.0, v_dep, 0.0);
        let v_mars   = Vector3::new(0.0, (MU_SUN / a_mars).sqrt(), 0.0);

        // Find the expected arrival position by propagating.
        let (r_arr, _) = propagate_kepler(r_earth, v_sc, tof_s, MU_SUN)
            .expect("Kepler propagation failed");

        let leg = evaluate_mga_leg(
            r_earth, v_sc,
            0.5, tof_s,
            r_arr, v_mars,
            MU_SUN,
        ).expect("Lambert solution not found");

        // For a perfect Hohmann transfer with η=0.5, the DSM correction is
        // exactly zero (the spacecraft is already on the right path). In
        // practice floating-point rounding means it is merely very small.
        assert!(leg.dv_dsm_ms < 1.0,
            "near-Hohmann DSM ΔV should be < 1 m/s; got {:.3} m/s", leg.dv_dsm_ms);
    }

    /// Lambert failure case: TOF too short for any elliptic solution.
    #[test]
    fn too_short_tof_returns_none() {
        let r1 = Vector3::new(AU, 0.0, 0.0);
        let v1 = Vector3::new(0.0, (MU_SUN / AU).sqrt() + 5_000.0, 0.0);
        let r2 = Vector3::new(-AU, 0.0, 0.0); // opposite side of the Sun
        let v2 = Vector3::new(0.0, -(MU_SUN / AU).sqrt(), 0.0);
        // 1 second is geometrically impossible for a half-orbit transfer.
        let result = evaluate_mga_leg(r1, v1, 0.5, 1.0, r2, v2, MU_SUN);
        assert!(result.is_none(), "expected None for impossibly short TOF");
    }

    /// [`evaluate_mga_leg_2dsm`] with a zero first impulse must still return
    /// a finite, feasible result (impulse 1 becomes a no-op coast waypoint,
    /// impulse 2 remains Lambert-anchored to the target exactly as the
    /// one-DSM model) — basic sanity before trusting the refine driver.
    #[test]
    fn two_dsm_leg_zero_impulse_is_finite_and_feasible() {
        let a_earth = 1.0 * AU;
        let v_earth = (MU_SUN / a_earth).sqrt();
        let r_earth = Vector3::new(a_earth, 0.0, 0.0);
        let v_earth_vec = Vector3::new(0.0, v_earth, 0.0);
        let a_mars   = 1.524 * AU;
        let a_xfer   = 0.5 * (a_earth + a_mars);
        let v_dep    = (MU_SUN * (2.0 / a_earth - 1.0 / a_xfer)).sqrt();
        let tof_s    = std::f64::consts::PI * (a_xfer.powi(3) / MU_SUN).sqrt();
        let v_sc     = Vector3::new(0.0, v_dep, 0.0);
        let v_mars   = Vector3::new(0.0, (MU_SUN / a_mars).sqrt(), 0.0);
        let (r_arr, _) = propagate_kepler(r_earth, v_sc, tof_s, MU_SUN).unwrap();

        let result = evaluate_mga_leg_2dsm(
            r_earth, v_sc, 0.2, 0.7, Vector3::zeros(), tof_s, r_arr, v_mars, MU_SUN,
        ).expect("feasible two-DSM leg should evaluate");
        assert!(result.leg.dv_dsm_ms.is_finite());
        assert_eq!(result.dv1_ms, 0.0);
    }

    /// Infeasible ordering (eta1 >= eta2) must return `None`, the same
    /// convention as every other infeasible-chromosome case in this crate.
    #[test]
    fn two_dsm_leg_rejects_bad_eta_ordering() {
        let r1 = Vector3::new(AU, 0.0, 0.0);
        let v1 = Vector3::new(0.0, (MU_SUN / AU).sqrt(), 0.0);
        let r2 = Vector3::new(-AU, 0.0, 0.0);
        let v2 = Vector3::new(0.0, -(MU_SUN / AU).sqrt(), 0.0);
        let tof_s = 200.0 * 86_400.0;
        let result = evaluate_mga_leg_2dsm(r1, v1, 0.7, 0.3, Vector3::zeros(), tof_s, r2, v2, MU_SUN);
        assert!(result.is_none(), "eta1 > eta2 should be infeasible");
    }

    /// The real point of this whole model: seeded on a leg whose target is
    /// offset from the departure orbit's natural continuation (so a real
    /// DSM is unavoidable, as in a real chromosome where the outer search's
    /// departure v∞ direction rarely points exactly at the target),
    /// [`refine_leg_two_dsm`] must find a genuinely lower total ΔV than the
    /// one-DSM cost at the same fixed eta — proof the second free impulse
    /// captures real, usable slack, matching what a primer-vector
    /// `‖λV‖ > 1` violation on such a leg would predict.
    #[test]
    fn two_dsm_refine_improves_a_deliberately_suboptimal_leg() {
        let a_earth = 1.0 * AU;
        let v_earth = (MU_SUN / a_earth).sqrt();
        let r_earth = Vector3::new(a_earth, 0.0, 0.0);
        let a_mars   = 1.524 * AU;
        let a_xfer   = 0.5 * (a_earth + a_mars);
        let v_dep    = (MU_SUN * (2.0 / a_earth - 1.0 / a_xfer)).sqrt();
        let tof_s    = std::f64::consts::PI * (a_xfer.powi(3) / MU_SUN).sqrt();
        let v_sc     = Vector3::new(0.0, v_dep, 0.0);
        let v_mars   = Vector3::new(0.0, (MU_SUN / a_mars).sqrt(), 0.0);
        let (r_arr_natural, _) = propagate_kepler(r_earth, v_sc, tof_s, MU_SUN).unwrap();

        // Rotate the target 15 deg off the departure orbit's natural
        // continuation — a real transfer-angle mismatch a single DSM must
        // correct, and one whose cost genuinely depends on eta (unlike the
        // exact-Hohmann fixture above, where the target sits exactly on the
        // continuation and any eta gives ~0 DSM by construction).
        let angle = 15.0_f64.to_radians();
        let rot = |v: Vector3<f64>| Vector3::new(
            v.x * angle.cos() - v.y * angle.sin(),
            v.x * angle.sin() + v.y * angle.cos(),
            v.z,
        );
        let r_arr = rot(r_arr_natural);
        let v_mars_rot = rot(v_mars);

        let bad_eta = 0.1;
        let baseline = evaluate_mga_leg_n(r_earth, v_sc, bad_eta, tof_s, r_arr, v_mars_rot, MU_SUN, 0)
            .expect("baseline one-DSM leg should evaluate");
        assert!(baseline.dv_dsm_ms > 100.0, "fixture should be genuinely suboptimal; got {:.3} m/s", baseline.dv_dsm_ms);

        let nm = NelderMead { max_iter: 400, ..Default::default() };
        let refined = refine_leg_two_dsm(
            r_earth, v_sc, tof_s, r_arr, v_mars_rot, MU_SUN,
            bad_eta, 0.6, baseline.dv_dsm_ms * 2.0, &nm,
        ).expect("refine should find a feasible solution");

        assert!(
            refined.total_dv_ms < 0.9 * baseline.dv_dsm_ms,
            "expected a real improvement: baseline={:.3} m/s, refined={:.3} m/s",
            baseline.dv_dsm_ms, refined.total_dv_ms,
        );
        let _ = v_earth;
    }

    /// Seeded on an already near-optimal leg (the near-Hohmann fixture,
    /// one-DSM cost < 1 m/s), the refine driver must not report a spurious
    /// "improvement" of any real magnitude — confirms the model doesn't
    /// fabricate savings when the leg genuinely has none to give (no false
    /// positives feeding into the accept/reject decision downstream).
    #[test]
    fn two_dsm_refine_finds_no_fake_improvement_on_already_optimal_leg() {
        let a_earth = 1.0 * AU;
        let v_earth = (MU_SUN / a_earth).sqrt();
        let r_earth = Vector3::new(a_earth, 0.0, 0.0);
        let a_mars   = 1.524 * AU;
        let a_xfer   = 0.5 * (a_earth + a_mars);
        let v_dep    = (MU_SUN * (2.0 / a_earth - 1.0 / a_xfer)).sqrt();
        let tof_s    = std::f64::consts::PI * (a_xfer.powi(3) / MU_SUN).sqrt();
        let v_sc     = Vector3::new(0.0, v_dep, 0.0);
        let v_mars   = Vector3::new(0.0, (MU_SUN / a_mars).sqrt(), 0.0);
        let (r_arr, _) = propagate_kepler(r_earth, v_sc, tof_s, MU_SUN).unwrap();

        let nm = NelderMead { max_iter: 300, ..Default::default() };
        let refined = refine_leg_two_dsm(
            r_earth, v_sc, tof_s, r_arr, v_mars, MU_SUN,
            0.3, 0.5, 50.0, &nm,
        ).expect("refine should find a feasible solution");

        assert!(
            refined.total_dv_ms < 5.0,
            "already-optimal leg should refine to near-zero, not a fake improvement claim; got {:.3} m/s",
            refined.total_dv_ms,
        );
        let _ = v_earth;
    }
}
