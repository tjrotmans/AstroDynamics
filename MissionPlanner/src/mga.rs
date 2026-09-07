//! MGA-1DSM trajectory optimizer — Phase 9 Layer 1b (Stage A).
//!
//! Implements the fixed-sequence Multi-Gravity-Assist optimizer using
//! Differential Evolution (DE/rand/1/bin) as the inner search engine. The
//! MGA-1DSM model places one freely-positioned deep-space manoeuvre (DSM)
//! per leg; flyby turns are modelled as instantaneous Rodrigues rotations
//! parameterised by periapsis radius and B-plane crank angle β (Ceriotti 2010,
//! §2.3–2.4) — this analytic turn is used by the fast DE search only. The
//! multiple-shooting refinement (`run_multiple_shooting`) instead treats each
//! flyby body as a real N-body gravitational perturber with SOI-switching, so
//! the gravity-assist turn there emerges from the dynamics, not from the
//! Rodrigues approximation.
//!
//! ## Chromosome structure (5N + 2 parameters for N legs)
//!
//! | Index range     | Variable           | Bounds                                    |
//! |-----------------|--------------------|-------------------------------------------|
//! | 0               | dep_offset_days    | [-window/2, +window/2]                    |
//! | 1               | dep_vinf_ms        | [departure_vinf_min_ms, departure_vinf_max_ms] |
//! | 2               | theta_dep_rad      | [0, 2pi]                                  |
//! | 3               | phi_dep_rad        | [-pi/2, +pi/2]                            |
//! | 4 + 2k          | tof_days[k]        | leg_tof_days[k] bounds, per leg k         |
//! | 4 + 2k + 1      | eta[k]             | [0.05, 0.95]                              |
//! | 4 + 2N + 2j     | rp_norm[j]         | [1.1, 50]  (periapsis / body_radius)      |
//! | 4 + 2N + 2j + 1 | beta_rad[j]        | [-pi, +pi]                                |
//! | 2 + 4N + k      | n_rev/leg_model[k] | [0, 3) or [0, 7) -- per-leg discrete gene |
//!
//! **Final-leg target-distance handling (Phase 9y-h, revised)**:
//! an EARLIER version of this fix added an `rp_final_norm` chromosome gene,
//! searched by the global DE/MBH search alongside everything else. That was
//! wrong: unlike an intermediate flyby's `(rp_norm, beta)` pair -- which
//! genuinely determines the post-turn velocity feeding into the NEXT leg,
//! a real deltaV tradeoff -- the final leg has no downstream leg to redirect
//! into, so the "chosen" final periapsis has NO effect on deltaV or on any
//! other part of the trajectory in this analytic model. It is not a search
//! problem at all: the optimal choice is always the closed-form
//! `target_orbit_radius_m` itself (clamped to the body's own floor), so
//! [`arrival_dv_ms`]'s `Flyby` branch computes it directly from config,
//! with zero chromosome dimensions spent on it -- see that function's doc
//! comment. This also sidesteps the chromosome-length-change risk (bounds/
//! pruning/branching/backfit index audit) the original gene-based design
//! required, since the chromosome layout is unchanged from pre-9y-h.
//!
//! ## Two-phase search
//!
//! **Phase 1** (multi-start) — `de_restarts` independent DE runs, each
//! minimising the sum of per-leg DSM ΔVs only. This finds geometrically
//! feasible body-to-body paths without penalising arrival energy or departure
//! cost. Phase 1's best individual seeds Phase 2.
//!
//! **Phase 2** — single DE run seeded with Phase 1's top elites but searched
//! over the FULL chromosome bounds (not narrowed — narrowing permanently
//! committed the search to one basin, see Phase 9v-xi), minimising total ΔV
//! (departure escape burn + all DSMs + optional LOI).
//!
//! ## Multiple-shooting refinement (Phase 9v-ix)
//!
//! `run_multiple_shooting` refines the DE's analytic/Rodrigues-turn solution
//! into a real N-body trajectory via TRUE multiple shooting: free variables
//! are the spacecraft's heliocentric state `(r, v)` at each intermediate
//! flyby node PLUS each leg's DSM ΔV vector, and every leg propagates
//! independently from its own node — never chained through a previous leg's
//! propagated end state. Continuity (no positional gap, no impulsive
//! velocity jump) is enforced as a Newton constraint, not by construction.
//! This replaced an earlier single-shooting-in-disguise formulation that
//! chained state through every flyby SOI in one unbroken propagation, whose
//! finite-difference Jacobian became astronomically ill-conditioned beyond
//! one flyby (an upstream perturbation only reached a downstream constraint
//! by surviving amplification through each intervening hyperbolic SOI
//! passage). See `ms_constraint`'s doc comment for the free-variable layout.
//!
//! ## References
//! - Ceriotti (2010), *Global Optimisation of Multiple Gravity Assist
//!   Trajectories*, PhD Thesis, University of Glasgow.
//! - Storn & Price (1997), "Differential Evolution", J. Global Opt. 11:341–359.
//! - Izzo & Vinkó (2010), ESA Technical Report — GTOP benchmark evidence that
//!   DE outperforms GA/PSO on MGA-DSM problems.

use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;

use ephemeris::Almanac;
use nalgebra::{DMatrix, DVector, Vector3, SVD};
use serde::Serialize;
use trajectory_solver::{
    evaluate_mga_leg_n, evaluate_vilm_leg, flyby_turn, hyperbolic_departure_state,
    keplerian::MU_SUN_M3S2, laplace_soi_radius_m,
    mga_scan::ScanRecord, orbital_period_s, propagate, propagate_kepler, run_incremental_pruning_all_levels,
    sample_box_uniform, CompassSearch, DeResult, DeSolver, GlobalStallConfig, HookeJeevesSearch,
    LbfgsSolver, LocalOptimizer, MbhSolver, MgaLegResult, MigrationConfig, NelderMead, PruningCandidate,
    PruningConfig, RbfSurrogate, ShadeSolver, SplitMix64, VilmDomain, VilmSolution,
};

use crate::config::{
    EphemerisSource, MbhLocalOptimizerToml, MissionConfig, MissionObjective, PruningBoundToml,
    PruningConfigToml, PruningSurrogateToml, SearchMethod,
};
use crate::design::{
    anise_body, as_propagator_bodies, body_state, epoch_to_jd, parse_epoch,
    PropagatorBodyEntry,
};
use crate::sequence_search;

// ── Sampling constants ────────────────────────────────────────────────────────

/// Number of trajectory sample points output per leg arc for plotting.
/// Bumped from 40 — for multi-thousand-day legs, 40 points gave
/// visibly faceted/polygonal curves in the 3D plots instead of smooth arcs.
const ARC_SAMPLES_PER_LEG: usize = 200;

// ── Graded-penalty constants (Phase 9v-viii) ──────────────────────────────────

/// Penalty scale [m/s] per unit relative violation of a soft constraint
/// (solar-perihelion floor, flyby minimum periapsis). At full violation this
/// dwarfs any real total ΔV (~3–20 km/s) so violating chromosomes still rank
/// below feasible ones, but near the constraint boundary the gradient is
/// comparable to real ΔV differences — DE can follow the slope into the
/// feasible region instead of facing an f64::MAX cliff.
const CONSTRAINT_PENALTY_MS: f64 = 1.0e5;

/// Base fitness [m/s] for a chromosome that cannot be evaluated at all
/// (no Lambert solution, Kepler non-convergence, missing body state).
/// Far above any graded-feasible fitness so complete evaluations always
/// rank ahead of incomplete ones.
const INFEASIBLE_BASE_MS: f64 = 1.0e8;

/// Additional penalty [m/s] per leg left unevaluated when evaluation fails —
/// failing on leg 4 of 6 ranks better than failing on leg 0, giving the DE
/// population a slope even in the deep-infeasible region.
const INFEASIBLE_PER_LEG_MS: f64 = 1.0e6;

// ── Public result types ───────────────────────────────────────────────────────

/// One heliocentric trajectory sample for the MGA arc, written to
/// `mga_best.csv` for plotting.
#[derive(Debug, Clone)]
pub struct MgaArcPoint {
    pub t_days: f64,
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
    /// Heliocentric velocity [m/s] at this sample (closing the
    /// "MGA arc has no velocity" API gap). Every sampling path already had
    /// this in hand and was discarding it: `propagate_kepler` returns the
    /// velocity alongside the position, and `PropagatedPoint::v_mps` is
    /// always populated by the Dopri5 re-propagations.
    pub vx_mps: f64,
    pub vy_mps: f64,
    pub vz_mps: f64,
    pub leg_idx: usize,
    /// Which body was gravitationally central at this point (inside its SOI),
    /// or `None` for the Sun. Only populated by the N-body multiple-shooting
    /// re-propagation; analytic/Keplerian arcs always report `None`.
    pub central_body: Option<String>,
}

/// Detailed MGA result returned from [`run_mga`].
#[derive(Debug)]
pub struct MgaResult {
    pub dv_total_ms: f64,
    pub dv_departure_ms: f64,
    pub dv_dsms_ms: Vec<f64>,
    pub dv_arrival_ms: f64,
    pub tof_total_days: f64,
    /// Per-leg TOFs [days], one entry per leg — `leg_tofs_days[k]` is the
    /// time-of-flight for the k-th leg (from body k to body k+1).
    pub leg_tofs_days: Vec<f64>,
    /// DSM position [m] in heliocentric inertial frame for each leg.
    /// `dsm_positions_m[k]` is where the k-th leg's deep-space manoeuvre fires.
    pub dsm_positions_m: Vec<[f64; 3]>,
    /// DSM epoch [s since departure, on the same clock as the arc's
    /// `t_days`] for each leg: `(t_start_days + eta_k * tof_k) * 86_400`
    /// (Phase 03's `cruise_seed.planned_burns`
    /// needs a real epoch to fire `mga_dv_dsms_inertial_mps[k]` at, and
    /// only per-leg TOF sums were exposed before). Valid for the multiple-
    /// shooting-refined arc too, not just the search-time one: in
    /// `ms_leg_timing`, segment A starts `dt_a_extra` EARLIER but also
    /// lasts `dt_a_extra` LONGER, so the DSM instant `t0_abs_a + dt_a`
    /// algebraically reduces to the same expression (the transit offsets
    /// cancel exactly; MS's free variables are DSM ΔVs + node states,
    /// never the leg timing).
    pub dsm_epochs_s: Vec<f64>,
    /// Full body-name sequence: [departure_body, flyby0, ..., target_body].
    /// Length = number of legs + 1.
    pub body_sequence: Vec<String>,
    /// Actual departure JD (base epoch + dep_offset search variable).
    pub dep_jd: f64,
    /// Base departure JD (the configured epoch, without the offset variable).
    pub dep_jd_base: f64,
    pub best_params: Vec<f64>,
    pub convergence: Vec<f64>,
    /// Phase 1 (DSM-only) best-fitness history [m/s sum of DSM ΔVs].
    pub phase1_history: Vec<f64>,
    /// Best-so-far PARAMETER vector, one entry per Phase-2 history step,
    /// parallel to `convergence` (for per-parameter convergence
    /// plots).
    pub param_history: Vec<Vec<f64>>,
    /// Same, parallel to `phase1_history`.
    pub phase1_param_history: Vec<Vec<f64>>,
    /// Keplerian/Lambert arc — fast, written to `mga_best.csv`.
    pub arc: Vec<MgaArcPoint>,
    /// Real Dopri5-propagated arc, written to `mga_repropagated.csv`.
    /// Empty when re-propagation fails (e.g. integrator diverges on a
    /// degenerate chromosome — non-fatal, Keplerian arc still available).
    pub repropagated_arc: Vec<MgaArcPoint>,
}

/// Result returned from [`run_multiple_shooting`].
#[derive(Debug)]
pub struct MultipleShotResult {
    /// True when Newton–Raphson converged within `MAX_MS_ITER` iterations.
    pub converged: bool,
    /// Number of Newton iterations actually performed.
    pub iterations: usize,
    /// Position defect norm at each patch point after the final iteration [m].
    /// One entry per leg (flyby bodies + final target).
    pub residuals_m: Vec<f64>,
    /// Velocity defect norm at each intermediate flyby node after the final
    /// iteration [m/s] — near-zero confirms the flyby is a genuine unpowered
    /// gravity assist (no impulsive velocity jump smuggled into the patch).
    /// One entry per intermediate flyby node (length N-1 for N legs; empty
    /// for a direct, no-flyby leg).
    pub vel_residuals_ms: Vec<f64>,
    /// Corrected DSM ΔV vectors [m/s], one per leg.
    pub dv_dsm_corrected: Vec<[f64; 3]>,
    /// Fully re-propagated arc with the corrected DSM ΔVs.
    pub arc: Vec<MgaArcPoint>,
    /// Total ΔV after correction [m/s] (departure + corrected DSM magnitudes + arrival).
    pub dv_total_ms: f64,
    /// Real body-relative arrival position at the converged final leg's end
    /// [m] — target-body-centered, from re-propagating the final leg under
    /// the SAME real dynamics/corrected DSM the rest of `arc` uses (see
    /// [`ms_final_arrival_state`]). `None` when `converged` is `false` (no
    /// genuinely converged crossing to report) or the re-propagation/
    /// ephemeris query failed.
    pub arrival_r_rel_m: Option<[f64; 3]>,
    /// Real body-relative arrival velocity at the same point [m/s]. Paired
    /// with `arrival_r_rel_m` — together these give a real orbital plane to
    /// seed a captured-orbit visualization from, the same way the
    /// single-leg path's `eval.arrival.r_rel_m`/`v_rel_mps` already does.
    pub arrival_v_rel_mps: Option<[f64; 3]>,
    /// Real ΔV [m/s] each intermediate flyby actually delivered, measured
    /// directly from the converged, real-dynamics-propagated trajectory —
    /// `|v_helio_at_SOI_exit − v_helio_at_SOI_entry|` (see the post-check
    /// loop's own doc comment for why `propagate()`'s heliocentric frame
    /// makes this a direct measurement, not an analytic re-derivation).
    /// Length `n-1` (one per intermediate flyby body); empty when
    /// `converged` is `false` (matches `arrival_r_rel_m`'s own gating —
    /// no genuinely converged passage to measure).
    pub flyby_dv_gained_ms: Vec<f64>,
    /// Heliocentric speed [m/s] just before each flyby's real SOI entry —
    /// same length/gating as `flyby_dv_gained_ms`.
    pub flyby_speed_before_ms: Vec<f64>,
    /// Heliocentric speed [m/s] just after each flyby's real SOI exit —
    /// paired with `flyby_speed_before_ms`: `after > before` means the
    /// flyby ADDED heliocentric energy (taken from the planet's own orbit);
    /// `after < before` means it REMOVED energy (given back to the
    /// planet) — the real, checkable signature of an energy-shedding
    /// deceleration flyby (e.g. MESSENGER's resonant Mercury passes).
    pub flyby_speed_after_ms: Vec<f64>,
}

// ── Chromosome decode helpers ─────────────────────────────────────────────────

/// `pub` (Stage 1): needed by `src/bin/primer_vector_check.rs`.
pub fn n_legs(flyby_bodies: &[String]) -> usize {
    flyby_bodies.len() + 1
}

/// `pub` — see [`n_legs`].
pub fn chromosome_len(n: usize) -> usize {
    // 4 departure vars + (TOF, η) per leg + (rp, β) per flyby + n_rev gene
    // per leg (appended LAST N-as-gene fix, so every other
    // accessor's index is unchanged): 4 + 2n + 2(n-1) + n = 5n + 2.
    4 + 2 * n + 2 * (n - 1) + n
}

#[inline]
fn dep_offset(p: &[f64]) -> f64 { p[0] }
#[inline]
fn dep_vinf(p: &[f64])   -> f64 { p[1] }
#[inline]
fn theta_dep(p: &[f64])  -> f64 { p[2] }
#[inline]
fn phi_dep(p: &[f64])    -> f64 { p[3] }

/// Decode the departure-direction genes into (theta, phi) spherical angles
/// used to build the departure v_inf vector in our own (inertial/ecliptic)
/// frame. Under `MGA_GTOP_UNIFORM_DEPARTURE=1` (diagnostic-only toggle,
/// same established pattern as `MGA_DISABLE_VILM` — never used in a live/
/// production search), the genes are instead read as GTOP's own (u, v) ∈
/// [0,1] parameterization (Izzo & Vinkó 2008,
/// ACT-TNT-INF-2008-GOHTPPSTD): `theta = 2*pi*u`, `phi = acos(2*v-1) -
/// pi/2` — this gives a UNIFORM distribution of departure directions on
/// the sphere, unlike sampling theta/phi directly (which clusters samples
/// near the poles). Note this only fixes the sampling density, not the
/// reference frame: GTOP's own u,v decode into a frame built from the
/// departure body's local velocity/orbit-normal directions, whereas we
/// keep constructing v_inf in our existing global frame — a deliberate,
/// smaller-risk scope decision for a fair-sampling A/B
/// against the official GTOP box, not a full frame-match.
#[inline]
fn decode_theta_phi(raw_a: f64, raw_b: f64) -> (f64, f64) {
    if std::env::var("MGA_GTOP_UNIFORM_DEPARTURE").is_ok() {
        let u = raw_a.clamp(0.0, 1.0);
        let v = raw_b.clamp(0.0, 1.0);
        let theta = 2.0 * PI * u;
        let phi   = (2.0 * v - 1.0).clamp(-1.0, 1.0).acos() - PI / 2.0;
        (theta, phi)
    } else {
        (raw_a, raw_b)
    }
}
#[inline]
pub fn tof_days(p: &[f64], k: usize) -> f64 { p[4 + 2 * k] }
#[inline]
pub fn eta(p: &[f64], k: usize)      -> f64 { p[4 + 2 * k + 1] }
#[inline]
// `pub` (Stage 5 VILM diagnostic): needed by
// `src/bin/vilm_leg_check.rs`, same reason LegEval/ChromosomeEval were
// widened for primer_vector_check.rs — a separate `src/bin/*.rs` target
// only sees genuinely `pub` items of this library crate.
pub fn rp_norm(p: &[f64], n: usize, j: usize) -> f64 { p[4 + 2 * n + 2 * j] }
#[inline]
pub fn beta(p: &[f64], n: usize, j: usize)    -> f64 { p[4 + 2 * n + 2 * j + 1] }
/// Leg k's Lambert revolution-count gene (N-as-gene fix):
/// continuous-encoded in [0, 3), floored to {0, 1, 2} at evaluation so DE's
/// arithmetic mutation/crossover works unmodified. Genes sit at the END of
/// the chromosome (indices `2+4n .. 2+5n`) so every pre-existing accessor
/// above keeps its index.
// `pub` (Stage 5 VILM diagnostic) — same reason as rp_norm/beta above.
#[inline]
pub fn n_rev_gene(p: &[f64], n: usize, k: usize) -> u32 {
    (p[2 + 4 * n + k].floor() as i64).clamp(0, 2) as u32
}

/// Decoded sub-arc-2 model choice for a same-body (repeat-flyby) leg —
/// see [`leg_model_gene`].
#[derive(Clone, Copy, Debug)]
enum LegModel {
    /// Plain Lambert arc with this many extra revolutions ({0, 1, 2}).
    Lambert(u32),
    /// Tangent VILT (`trajectory_solver::vilm`) with the given domain and
    /// solution branch (`k_low`/`k_high` fixed to 0 — Stage 5's live runs
    /// found the k=1 variants near-dead weight: rarely a feasible bracket).
    Vilm(VilmDomain, VilmSolution),
}

/// Decode a same-body leg's MODEL-CHOICE gene (replacing the
/// forced-VILM routing that regressed the search — see the design notes
/// 9x-v "ESCALATION"): the tail gene now selects among 7 options —
/// {Lambert N=0/1/2, VILM Interior/Exterior × Lower/Upper} — so the SEARCH
/// decides per candidate whether the leg is a plain (possibly multi-rev)
/// Lambert arc or a v∞-leveraging VILT, instead of VILM being imposed by
/// body name alone. This restores the pre-VILM model space as a strict
/// subset of the new one (gene values 0-2 reproduce the old n_rev
/// behaviour exactly, including the resonance-bias sampler's `N + 0.5`
/// seeding), removes the infeasibility cliffs forced-VILM created (a
/// geometry with no tangent-VILT solution is now only infeasible for
/// candidates whose gene actually selects VILM), and shrinks the gene
/// bound from the old [0, 16) to [0, 7) — close to the ordinary [0, 3)
/// n_rev step scale that every other discrete gene uses.
///
/// Reuses the SAME chromosome slot as `n_rev_gene`; `build_bounds` widens
/// this gene's bound to [0, 7) ONLY for same-body legs — ordinary legs
/// keep the original [0, 3) n_rev bound and `n_rev_gene`'s {0,1,2} clamp,
/// unaffected.
#[inline]
fn leg_model_gene(p: &[f64], n: usize, k: usize) -> LegModel {
    match (p[2 + 4 * n + k].floor() as i64).clamp(0, 6) {
        0 => LegModel::Lambert(0),
        1 => LegModel::Lambert(1),
        2 => LegModel::Lambert(2),
        3 => LegModel::Vilm(VilmDomain::Interior, VilmSolution::Lower),
        4 => LegModel::Vilm(VilmDomain::Interior, VilmSolution::Upper),
        5 => LegModel::Vilm(VilmDomain::Exterior, VilmSolution::Lower),
        _ => LegModel::Vilm(VilmDomain::Exterior, VilmSolution::Upper),
    }
}

// ── Chromosome bounds ─────────────────────────────────────────────────────────

pub fn build_bounds(cfg: &MissionConfig, flyby_bodies: &[String]) -> Option<Vec<(f64, f64)>> {
    build_bounds_with_overrides(cfg, flyby_bodies, None, None)
}

/// Same as [`build_bounds`], but the departure-window width and per-leg TOF
/// ranges can be overridden instead of read straight from
/// `opt.departure_window_days`/`mga.leg_tof_days` (Phase 9w-vi:
/// scan-informed window derivation feeds its own auto-derived values in
/// here). `None` for either override reproduces `build_bounds`'s exact
/// historical behaviour — this is a strict superset, not a fork.
fn build_bounds_with_overrides(
    cfg: &MissionConfig,
    flyby_bodies: &[String],
    window_days_override: Option<f64>,
    leg_tof_days_override: Option<&[[f64; 2]]>,
) -> Option<Vec<(f64, f64)>> {
    let opt = cfg.optimization.as_ref()?;
    let mga = opt.mga.as_ref()?;
    let n   = n_legs(flyby_bodies);

    let mut body_names: Vec<&str> = Vec::with_capacity(n + 1);
    body_names.push(&opt.departure_body);
    for fb in flyby_bodies { body_names.push(fb.as_str()); }
    body_names.push(&opt.target_body);

    let window = window_days_override.unwrap_or_else(|| opt.departure_window_days.unwrap_or(0.0));
    let mut bounds = Vec::with_capacity(chromosome_len(n));

    bounds.push((-window / 2.0, window / 2.0));           // dep_offset_days
    bounds.push((mga.departure_vinf_min_ms, mga.departure_vinf_max_ms)); // dep_vinf_ms
    if std::env::var("MGA_GTOP_UNIFORM_DEPARTURE").is_ok() {
        // Diagnostic-only toggle (matches MGA_DISABLE_VILM's established
        // pattern): search these genes as GTOP's own (u, v) in‑[0,1]
        // parameterization instead of raw (theta, phi) — see
        // `decode_theta_phi` for why (uniform sphere sampling vs. plain
        // theta/phi's pole-clustering bias). Never used in a live/
        // production search; only for controlled A/B comparison against
        // the official GTOP problem definition.
        bounds.push((0.0, 1.0));                           // u
        bounds.push((0.0, 1.0));                           // v
    } else {
        bounds.push((0.0, 2.0 * PI));                      // theta_dep_rad
        bounds.push((-PI / 2.0, PI / 2.0));                // phi_dep_rad
    }

    for k in 0..n {
        // When sequence_search auto-discovers a sequence shorter than the
        // TOML's leg_tof_days list, use the last entry as a fallback. Same
        // fallback applies to `leg_tof_days_override` when present (Phase
        // 9w-vi's derived per-leg bounds are already sized to `n`, but keep
        // the same defensive pattern for consistency).
        let tof_range = match leg_tof_days_override {
            Some(ov) if !ov.is_empty() => *ov.get(k).unwrap_or_else(|| ov.last().unwrap()),
            _ => mga.leg_tof_days.get(k)
                .or_else(|| mga.leg_tof_days.last())
                .copied()
                .unwrap_or([30.0, 1500.0]),
        };
        let [lo, hi] = tof_range;
        bounds.push((lo, hi));                             // tof_days[k]
        // η bounds match the chromosome spec ([0.01, 0.99], the design notes 9d)
        // and cover GTOP's [0.01, 0.9] — the previous hardcoded [0.05, 0.95]
        // excluded the published Cassini-2 optimum's η3 = 0.027 (
        // bounds audit; the optimizer pinned η against the 0.05 wall).
        // `eta_max` lets a specific config tighten this to the
        // real official upper bound (0.9 for GTOP Cassini-2) instead.
        bounds.push((0.01, mga.eta_max.unwrap_or(0.99)));  // eta[k]
    }
    for j in 0..(n - 1) {
        // rp_norm bound: per-leg override via `rp_norm_bounds` (
        // for exactly reproducing a benchmark's official per-body bounds —
        // e.g. GTOP Cassini-2's Venus [1.05,6] / Earth [1.15,6.5] / Jupiter
        // [1.7,291], sharply tighter than our generic default for the inner
        // planets). Falls back to the historical uniform (1.05, 300.0),
        // which covers giant-planet distant passes (the published
        // Cassini-2 optimum uses a 69.8-R_J Jupiter flyby; GTOP's own
        // upper bound there is 291) but is far wider than official for
        // Venus/Earth — the real surface-safety constraint otherwise is
        // `flyby_min_periapsis_m` in the fitness, not this box.
        let (rp_lo, rp_hi) = mga.rp_norm_bounds.as_ref()
            .and_then(|v| v.get(j))
            .map(|&[lo, hi]| (lo, hi))
            .unwrap_or((1.05, 300.0));
        bounds.push((rp_lo, rp_hi));                       // rp_norm[j]
        bounds.push((-PI, PI));                            // beta_rad[j]
    }
    for k in 0..n {
        // n_rev gene per leg (see `n_rev_gene`): [0, 3) floored to {0,1,2}
        // — equal-width thirds so uniform init gives each branch equal
        // prior probability. A leg whose TOF can't geometrically support
        // the gene's N simply evaluates infeasible (graded penalty), the
        // same as any other infeasible chromosome.
        //
        // Same-body legs (start/end body match — same detection
        // `evaluate_chromosome_graded` uses) widen this SAME slot to the
        // 7-option MODEL-CHOICE gene ([`leg_model_gene`]: Lambert N=0/1/2
        // or one of 4 VILM variants) — the search chooses the sub-arc-2
        // model per candidate, VILM is never imposed (fix; the
        // previous forced-VILM routing with a [0, 16) variant gene
        // regressed the search — see the design notes).
        // Ordinary legs keep the original [0, 3) n_rev bound unchanged.
        let vilm_enabled = std::env::var("MGA_DISABLE_VILM").is_err();
        if vilm_enabled && body_names[k].eq_ignore_ascii_case(body_names[k + 1]) {
            bounds.push((0.0, 6.999_999));                 // leg_model[k]
        } else {
            bounds.push((0.0, 2.999_999));                 // n_rev[k]
        }
    }

    Some(bounds)
}

// ── Body state at a given JD ──────────────────────────────────────────────────

/// Pre-sampled ephemeris cache (search-speed item 2): the MGA
/// fitness makes ~1 live ANISE query PER LEG PER EVALUATION through
/// [`get_body_state`], and at ~10 ms/evaluation those queries dominate the
/// search's wall-clock. This cache samples each body's heliocentric state
/// once over the search's full epoch envelope at [`EPH_CACHE_STEP_DAYS`]
/// intervals, then answers lookups by cubic Hermite interpolation —
/// position interpolated with the sampled VELOCITY as its exact endpoint
/// derivative (velocity from the same polynomial's derivative). For
/// planetary orbits at a 0.5-day step the interpolation error is sub-km in
/// position (h⁴ scaling; even Mercury's 88-day period gives ωh ≈ 0.036),
/// far below the patched-conic model error this search already carries.
/// Same pattern as `AstroProbs/Artemis`'s pre-sampled `MoonTrack`.
///
/// Installed process-wide via [`install_eph_cache`] (an `RwLock` slot
/// consulted by [`get_body_state`], falling back to the live almanac on any
/// miss) rather than threaded as a parameter — `get_body_state` has ~15
/// call sites across three evaluator families (chromosome, Ceriotti prefix,
/// backward suffix), and a cached state is an OBJECTIVE fact of
/// `(body, jd)`, not run-specific: even two concurrent server runs
/// overwriting each other's install can only cause cache misses, never
/// wrong data.
struct EphemerisCache {
    jd0: f64,
    step_days: f64,
    /// Per lowercase body name: `(r [m], v [m/s])` at `jd0 + i·step_days`.
    bodies: std::collections::HashMap<String, Vec<(Vector3<f64>, Vector3<f64>)>>,
}

const EPH_CACHE_STEP_DAYS: f64 = 0.5;

static ACTIVE_EPH_CACHE: std::sync::RwLock<Option<std::sync::Arc<EphemerisCache>>> =
    std::sync::RwLock::new(None);

impl EphemerisCache {
    /// Sample `names` over `[jd_min, jd_max]`. Bodies the almanac cannot
    /// resolve are silently skipped (lookups for them miss and fall back to
    /// the live path, which reports the failure exactly as before).
    fn build(almanac: &Almanac, names: &[&str], jd_min: f64, jd_max: f64) -> Self {
        let n_samples = (((jd_max - jd_min) / EPH_CACHE_STEP_DAYS).ceil() as usize + 2).max(2);
        let mut bodies = std::collections::HashMap::new();
        for name in names {
            let lower = name.to_lowercase();
            if bodies.contains_key(&lower) { continue; }
            let mut samples = Vec::with_capacity(n_samples);
            let mut ok = true;
            for i in 0..n_samples {
                let jd = jd_min + i as f64 * EPH_CACHE_STEP_DAYS;
                match get_body_state_uncached(almanac, &lower, jd) {
                    Some(rv) => samples.push(rv),
                    None => { ok = false; break; }
                }
            }
            if ok { bodies.insert(lower, samples); }
        }
        EphemerisCache { jd0: jd_min, step_days: EPH_CACHE_STEP_DAYS, bodies }
    }

    /// Cubic Hermite lookup; `None` outside coverage or for unknown bodies.
    fn lookup(&self, name_lower: &str, jd: f64) -> Option<(Vector3<f64>, Vector3<f64>)> {
        let samples = self.bodies.get(name_lower)?;
        let t = (jd - self.jd0) / self.step_days;
        if !(0.0..=(samples.len() - 1) as f64).contains(&t) { return None; }
        let i = (t.floor() as usize).min(samples.len() - 2);
        let u = t - i as f64;
        let h_s = self.step_days * 86_400.0;
        let (r0, v0) = samples[i];
        let (r1, v1) = samples[i + 1];

        let (u2, u3) = (u * u, u * u * u);
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;
        let r = h00 * r0 + h10 * h_s * v0 + h01 * r1 + h11 * h_s * v1;

        // Velocity = d/dt of the same polynomial (d/du scaled by 1/h).
        let d00 = 6.0 * u2 - 6.0 * u;
        let d10 = 3.0 * u2 - 4.0 * u + 1.0;
        let d01 = -6.0 * u2 + 6.0 * u;
        let d11 = 3.0 * u2 - 2.0 * u;
        let v = (d00 / h_s) * r0 + d10 * v0 + (d01 / h_s) * r1 + d11 * v1;

        Some((r, v))
    }
}

/// Build and install the process-wide ephemeris cache for a search run.
/// See [`EphemerisCache`] for why this is a global slot, not a parameter.
fn install_eph_cache(almanac: &Almanac, names: &[&str], jd_min: f64, jd_max: f64) {
    let cache = EphemerisCache::build(almanac, names, jd_min, jd_max);
    let n_bodies = cache.bodies.len();
    let n_samples = cache.bodies.values().next().map_or(0, |v| v.len());
    println!(
        "  Ephemeris cache: {n_bodies} bodies × {n_samples} samples ({:.0} d span, {} d step)",
        jd_max - jd_min, EPH_CACHE_STEP_DAYS,
    );
    *ACTIVE_EPH_CACHE.write().unwrap() = Some(std::sync::Arc::new(cache));
}

/// The pre-cache live-almanac path, still the single source of truth —
/// the cache is BUILT from this and falls back to it on any miss.
fn get_body_state_uncached(
    almanac: &Almanac,
    name_lower: &str,
    jd: f64,
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let anise = anise_body(name_lower)?;
    let (r_arr, v_arr) = body_state(almanac, EphemerisSource::Anise, Some(anise), &None, jd)?;
    Some((
        Vector3::new(r_arr[0], r_arr[1], r_arr[2]),
        Vector3::new(v_arr[0], v_arr[1], v_arr[2]),
    ))
}

fn get_body_state(
    almanac: &Almanac,
    name: &str,
    jd: f64,
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let lower = name.to_lowercase();
    if let Some(cache) = ACTIVE_EPH_CACHE.read().unwrap().as_ref() {
        if let Some(rv) = cache.lookup(&lower, jd) {
            return Some(rv);
        }
    }
    get_body_state_uncached(almanac, &lower, jd)
}

// ── Chromosome evaluation ─────────────────────────────────────────────────────

/// Per-leg evaluation result, returned by `evaluate_chromosome_detailed`.
/// `pub` (Stage 1 primer-vector diagnostic): needed by
/// `src/bin/primer_vector_check.rs`, a separate binary target that only
/// sees genuinely `pub` items of this library crate.
pub struct LegEval {
    pub leg: MgaLegResult,
    pub r_sc_start: Vector3<f64>,
    pub v_sc_start: Vector3<f64>,
    /// Elapsed time at leg start [days] since the actual departure epoch
    /// (`ChromosomeEval::dep_jd`); 0.0 for the first leg.
    pub t_start_days: f64,
}

/// Full chromosome evaluation result when feasible. `pub` for the same
/// reason as [`LegEval`].
pub struct ChromosomeEval {
    pub legs: Vec<LegEval>,
    pub v_inf_dep_ms: f64,      // magnitude of departure v∞ [m/s]
    pub v_inf_arr: Vector3<f64>, // arrival v∞ at last body [m/s]
    pub dep_jd: f64,            // actual departure JD
    pub body_mus: Vec<f64>,     // mu of each body in the sequence [m³/s²]
    pub body_radii: Vec<f64>,   // radius of each body [m]
    pub body_rvs: Vec<(Vector3<f64>, Vector3<f64>)>, // body states at each encounter epoch
    /// Graded constraint-violation penalty [m/s-equivalent] accumulated during
    /// evaluation (Phase 9v-viii): solar-perihelion-floor and flyby
    /// minimum-periapsis violations are penalised proportionally to their
    /// depth instead of rejecting the chromosome outright, so the DE
    /// population sees a slope toward feasibility rather than an f64::MAX
    /// cliff. 0.0 for a fully feasible chromosome. Fitness functions add this
    /// to the real ΔV; result/reporting paths must NOT fold it into any
    /// reported ΔV number.
    pub penalty_ms: f64,
}

/// Evaluate all legs of a chromosome.
///
/// Soft constraints (solar-perihelion floor, flyby minimum periapsis) do not
/// reject the chromosome — they accumulate into [`ChromosomeEval::penalty_ms`]
/// (Phase 9v-viii graded penalties). Only genuinely un-evaluable chromosomes
/// return `Err(legs_completed)`: no Lambert solution, Kepler non-convergence,
/// or a missing body state. The number of successfully evaluated legs lets
/// the fitness grade even the deep-infeasible region.
pub fn evaluate_chromosome_graded(
    params: &[f64],
    cfg: &MissionConfig,
    almanac: &Almanac,
    dep_jd_base: f64,
    flyby_bodies: &[String],
) -> Result<ChromosomeEval, usize> {
    let opt = cfg.optimization.as_ref().ok_or(0usize)?;
    let mga = opt.mga.as_ref().ok_or(0usize)?;
    let n   = n_legs(flyby_bodies);

    // Body sequence: departure + flybys + target.
    let mut body_names: Vec<&str> = Vec::with_capacity(n + 1);
    body_names.push(&opt.departure_body);
    for fb in flyby_bodies { body_names.push(fb.as_str()); }
    body_names.push(&opt.target_body);

    // Gather body catalog parameters.
    let mut body_mus    = Vec::with_capacity(n + 1);
    let mut body_radii  = Vec::with_capacity(n + 1);
    for name in &body_names {
        let cat = body_models::TargetBody::by_name(name).ok_or(0usize)?;
        body_mus.push(cat.mu_m3s2);
        body_radii.push(cat.radius_m);
    }

    let dep_offset_d = dep_offset(params);
    let dep_jd       = dep_jd_base + dep_offset_d;

    // Departure body state at the actual departure epoch.
    let (r_dep, v_dep) = get_body_state(almanac, body_names[0], dep_jd).ok_or(0usize)?;

    // Build departure velocity: body velocity + v∞ vector in ecliptic spherical frame.
    let (theta, phi) = decode_theta_phi(theta_dep(params), phi_dep(params));
    let vinf  = dep_vinf(params);
    let v_inf_vec = Vector3::new(
        vinf * phi.cos() * theta.cos(),
        vinf * phi.cos() * theta.sin(),
        vinf * phi.sin(),
    );
    let mut r_sc = r_dep;
    let mut v_sc = v_dep + v_inf_vec;

    let mut legs: Vec<LegEval> = Vec::with_capacity(n);
    let mut body_rvs: Vec<(Vector3<f64>, Vector3<f64>)> = Vec::with_capacity(n + 1);
    body_rvs.push((r_dep, v_dep));

    // Elapsed mission time [days] since the ACTUAL departure epoch `dep_jd`
    // (which already includes the chromosome's departure offset). This must
    // start at 0.0, not `dep_offset_d` — initializing it to the offset
    // double-counted the offset in every `arr_jd` below, so every encounter
    // body was queried at an epoch shifted by `dep_offset_d` from the true
    // arrival time (a "phantom" body position the spacecraft never actually
    // meets). Found via a refined-arc flyby that visibly missed
    // the real Venus.
    let mut t_days = 0.0;
    let mut penalty_ms = 0.0;

    for k in 0..n {
        let tof_k   = tof_days(params, k);
        let eta_k   = eta(params, k);
        let arr_jd  = dep_jd + t_days + tof_k;
        let (r_next, v_next) = get_body_state(almanac, body_names[k + 1], arr_jd).ok_or(k)?;

        // A leg whose start and end body match (a repeat-flyby leg — same
        // detection `resonance_bias_windows` uses) carries a MODEL-CHOICE
        // gene ([`leg_model_gene`]): the SEARCH selects plain
        // (possibly multi-rev) Lambert or a tangent VILT per candidate.
        // VILM is never imposed by body name — the previous forced-VILM
        // routing removed plain Lambert from the model space for every
        // same-body leg (a strict restriction off-resonance) and turned
        // every VILT-infeasible geometry into a whole-chromosome failure,
        // which regressed the search badly).
        // Diagnostic escape hatch: `MGA_DISABLE_VILM=1` forces the
        // historical Lambert-only path everywhere (gene values 3-6 then
        // clamp to Lambert N=2 via `n_rev_gene`) — for controlled A/Bs,
        // never used inside a live search; matches `MGA_MAX_LEG_N_REV`'s
        // existing role.
        let vilm_enabled = std::env::var("MGA_DISABLE_VILM").is_err();
        let leg = if vilm_enabled && body_names[k].eq_ignore_ascii_case(body_names[k + 1]) {
            match leg_model_gene(params, n, k) {
                LegModel::Lambert(n_rev) => evaluate_mga_leg_n(
                    r_sc, v_sc, eta_k, tof_k * 86_400.0, r_next, v_next, MU_SUN_M3S2, n_rev,
                ).ok_or(k)?,
                LegModel::Vilm(domain, solution) => {
                    let vilm = evaluate_vilm_leg(
                        r_sc, v_sc, eta_k, tof_k * 86_400.0, r_next, v_next, MU_SUN_M3S2,
                        domain, true, 0, 0, solution,
                    ).ok_or(k)?;
                    // Combine the departure-side DSM and VILM's internal
                    // leveraging burn into one total `dv_dsm_ms` so every
                    // downstream consumer (fitness sums, CSV writers,
                    // primer-vector diagnostic, arc plots) sees this leg's
                    // real total cost without needing to know VILM has two
                    // internal impulses instead of Lambert's one.
                    MgaLegResult { dv_dsm_ms: vilm.leg.dv_dsm_ms + vilm.dv_leverage_ms, ..vilm.leg }
                }
            }
        } else {
            // Revolution count comes from the chromosome (N-as-gene,
            //) — never greedily searched inside the evaluator;
            // see `evaluate_mga_leg_n`'s doc comment for why.
            evaluate_mga_leg_n(
                r_sc, v_sc, eta_k, tof_k * 86_400.0, r_next, v_next, MU_SUN_M3S2,
                n_rev_gene(params, n, k),
            ).ok_or(k)?
        };

        // Graded near-Sun penalty (9v-viii): a perihelion below the configured
        // floor on either sub-arc is penalised proportionally to violation
        // depth instead of rejecting the chromosome outright — DE needs a
        // slope toward feasibility, not a cliff where most of the population
        // returns the f64::MAX sentinel. (The floor itself exists because the
        // optimizer previously exploited physically implausible close solar
        // passages as a "free" way to reshape a transfer.)
        let floor = mga.min_solar_perihelion_m;
        for rp in [leg.rp_dep_m, leg.rp_lambert_m] {
            if rp < floor {
                penalty_ms += CONSTRAINT_PENALTY_MS * ((floor - rp) / floor).min(1.0);
            }
        }

        let r_sc_start = r_sc;
        let v_sc_start = v_sc;

        if k < n - 1 {
            // Intermediate flyby: apply unpowered gravity turn.
            let j          = k; // flyby index
            let rp_raw     = rp_norm(params, n, j) * body_radii[k + 1];
            // Graded minimum-periapsis constraint (9v-viii): clamp the turn
            // to the legal floor and penalise the violation depth, instead of
            // rejecting — the turn stays physical while DE gets a gradient.
            let rp_j = if rp_raw < mga.flyby_min_periapsis_m {
                penalty_ms += CONSTRAINT_PENALTY_MS
                    * ((mga.flyby_min_periapsis_m - rp_raw) / mga.flyby_min_periapsis_m).min(1.0);
                mga.flyby_min_periapsis_m
            } else {
                rp_raw
            };
            let beta_j     = beta(params, n, j);
            let v_inf_out  = flyby_turn(leg.v_inf_arr_mps, rp_j, beta_j, body_mus[k + 1]);

            // Next leg: start from flyby body with post-turn v∞ + body velocity.
            r_sc = r_next;
            v_sc = v_next + v_inf_out;
        }

        body_rvs.push((r_next, v_next));
        t_days += tof_k;
        legs.push(LegEval { leg, r_sc_start, v_sc_start, t_start_days: t_days - tof_k });
    }

    let v_inf_arr = legs.last().ok_or(0usize)?.leg.v_inf_arr_mps;

    Ok(ChromosomeEval {
        legs,
        v_inf_dep_ms: vinf,
        v_inf_arr,
        dep_jd,
        body_mus,
        body_radii,
        body_rvs,
        penalty_ms,
    })
}

/// Feasible-only view of [`evaluate_chromosome_graded`] for result/reporting
/// paths that have no use for the failure grading. Note the returned
/// `ChromosomeEval` can still carry `penalty_ms > 0` — callers reporting a
/// final solution should check and warn (a converged winner should be 0).
/// `pub` (Stage 1): needed by `src/bin/primer_vector_check.rs`.
pub fn evaluate_chromosome_detailed(
    params: &[f64],
    cfg: &MissionConfig,
    almanac: &Almanac,
    dep_jd_base: f64,
    flyby_bodies: &[String],
) -> Option<ChromosomeEval> {
    evaluate_chromosome_graded(params, cfg, almanac, dep_jd_base, flyby_bodies).ok()
}

/// Per-leg trajectory state for one evaluated chromosome, streamed live over
/// `/api/optimize`'s WebSocket for MGA jobs only (backlog item #18,
///). Every field here is already computed as an ordinary
/// byproduct of fitness evaluation ([`MgaLegResult`], `mga_leg.rs`) — this
/// struct is purely a serializable subset of it, extracted once per streamed
/// generation (not once per fitness evaluation, so the added cost is one
/// extra [`evaluate_chromosome_detailed`] call per generation, not per
/// candidate). Exists so the frontend's live-search animation can draw the
/// REAL solved Lambert arc / DSM point / flyby geometry for the best-so-far
/// chromosome, instead of client-side reconstructing an approximate,
/// Lambert-branch-blind re-solve from the raw chromosome alone (see
/// the design notes for the full
/// motivation).
#[derive(Serialize, Clone)]
pub struct MgaLegStepInfo {
    /// Spacecraft position at the deep-space maneuver point [m], heliocentric.
    pub r_dsm_m: [f64; 3],
    /// Spacecraft velocity just after the DSM [m/s], heliocentric — the
    /// Lambert-arc departure velocity (`v_L1`) that seeds segment B of this leg.
    pub v_dsm_after_mps: [f64; 3],
    /// Spacecraft heliocentric velocity at leg arrival (Lambert arc endpoint
    /// `v_L2`, before subtracting the next body's velocity).
    pub v_arrival_helio_mps: [f64; 3],
    /// Hyperbolic excess velocity on arrival relative to the next body [m/s]
    /// — feeds the flyby turn at an intermediate body, or the arrival v_∞ at
    /// the final target. Useful for a live flyby-turn indicator.
    pub v_inf_arr_mps: [f64; 3],
}

/// Evaluate `params` and extract the per-leg state a live-streaming client
/// needs to draw the real solved trajectory (see [`MgaLegStepInfo`]).
/// Returns an empty `Vec` (never panics) when `params` is not evaluable —
/// same graceful-degradation convention as the rest of this module's graded
/// evaluation path; the stream simply omits leg detail for that generation.
fn mga_leg_step_info(
    params: &[f64],
    cfg: &MissionConfig,
    almanac: &Almanac,
    dep_jd_base: f64,
    flyby_bodies: &[String],
) -> Vec<MgaLegStepInfo> {
    match evaluate_chromosome_detailed(params, cfg, almanac, dep_jd_base, flyby_bodies) {
        Some(eval) => eval.legs.iter().map(|le| MgaLegStepInfo {
            r_dsm_m: [le.leg.r_dsm_m.x, le.leg.r_dsm_m.y, le.leg.r_dsm_m.z],
            v_dsm_after_mps: [le.leg.v_dsm_after_mps.x, le.leg.v_dsm_after_mps.y, le.leg.v_dsm_after_mps.z],
            v_arrival_helio_mps: [le.leg.v_arrival_helio_mps.x, le.leg.v_arrival_helio_mps.y, le.leg.v_arrival_helio_mps.z],
            v_inf_arr_mps: [le.leg.v_inf_arr_mps.x, le.leg.v_inf_arr_mps.y, le.leg.v_inf_arr_mps.z],
        }).collect(),
        None => Vec::new(),
    }
}

// ── Fitness functions ─────────────────────────────────────────────────────────

/// Fitness value for a chromosome whose evaluation fails outright (no
/// Lambert solution / Kepler non-convergence / missing body state). The
/// per-remaining-leg term grades the deep-infeasible region so DE can still
/// rank two failures against each other (Phase 9v-viii).
pub fn infeasible_fitness_ms(n: usize, legs_completed: usize) -> f64 {
    INFEASIBLE_BASE_MS + INFEASIBLE_PER_LEG_MS * (n.saturating_sub(legs_completed)) as f64
}

/// Phase 1 fitness: sum of DSM ΔVs only — drives the search toward
/// geometrically feasible paths without over-penalising the departure cost.
/// Always returns a graded value (9v-viii): soft-constraint violations add
/// `penalty_ms`; outright evaluation failures score by legs completed.
fn phase1_fitness(
    params: &[f64],
    cfg: &MissionConfig,
    almanac: &Almanac,
    dep_jd_base: f64,
    flyby_bodies: &[String],
) -> Option<f64> {
    match evaluate_chromosome_graded(params, cfg, almanac, dep_jd_base, flyby_bodies) {
        Ok(ev) => Some(ev.legs.iter().map(|l| l.leg.dv_dsm_ms).sum::<f64>() + ev.penalty_ms),
        Err(legs_completed) => Some(infeasible_fitness_ms(n_legs(flyby_bodies), legs_completed)),
    }
}

/// Arrival / LOI burn ΔV [m/s]: 0 for Flyby; the full arrival relative-
/// velocity magnitude for Rendezvous (no Oberth reduction — a genuine
/// heliocentric velocity match); otherwise (Orbit/Landing/SampleReturn) the
/// periapsis burn from the arrival hyperbola into the configured capture
/// orbit (Phase 9v-ii).
///
/// `target_orbit_radius_m` is the capture orbit's periapsis radius r_p;
/// `capture_eccentricity` e (default 0 = circular) sets the post-burn orbit:
///   v_hyp  = √(v∞² + 2μ/r_p)
///   v_peri = √(μ·(1+e)/r_p)   — vis-viva at periapsis, a = r_p/(1−e)
///   ΔV     = v_hyp − v_peri
///
/// The `Rendezvous` branch (added) matches the ORIGINAL ESA
/// GTOPtoolbox source exactly (`trajobjfuns.cpp`/`mga_dsm.cpp`,
/// `final_block()`: `DVarr = DVrel` for `problem.type == total_DV_rndv`,
/// the FULL relative-velocity magnitude, no reduction at all).
/// **Correction: the GTOP Cassini-2 benchmark does NOT use
/// r_p=108,950 km/e=0.98 — those are Cassini-1's parameters. Cassini-2 and
/// Messenger both use `total_DV_rndv` (`mga.h`: "Cassini 2 and Messenger"),
/// i.e. `MissionObjective::Rendezvous`, not `Orbit`.** A previous version
/// of this comment (and the configs built against it) incorrectly
/// attributed Cassini-1's capture parameters to Cassini-2 — see
/// `docs/MP/BENCHMARKS.md` for the full correction history.
///
/// The `Flyby` branch (Phase 9y-h, revised replacing a flat
/// `0.0`): a CLOSED-FORM floor check, not a search problem. An earlier
/// version of this fix searched the achieved periapsis as a chromosome
/// gene (`rp_final_norm`) — wrong, because unlike an intermediate flyby's
/// `(rp_norm, β)` pair, the final leg's "chosen" periapsis has no
/// downstream leg to affect, so it has zero effect on ΔV or on any other
/// part of the trajectory in this analytic model. The optimal choice is
/// therefore always `target_orbit_radius_m` itself (whenever
/// `[trajectory.capture].target_orbit_radius_m` is configured), clamped to
/// a safety floor of `1.05 × target_body.radius_m` — mirroring the
/// removed gene's own floor convention. This is exactly the same
/// clamped-floor pattern the `Orbit`/`Landing` branch below already uses
/// for `target_orbit_radius_m`, applied here without a capture burn since
/// Flyby has none. **Fallback, no target configured: `0.0`, exactly the
/// pre-9y-h behaviour** — matches `config.rs::check_config`'s existing
/// `needs_capture_radius` rule, which has never required a capture radius
/// for `Flyby` and still doesn't.
pub fn arrival_dv_ms(cfg: &MissionConfig, ev: &ChromosomeEval) -> f64 {
    match cfg.mission.objective {
        MissionObjective::Flyby => cfg.trajectory.capture.as_ref()
            .and_then(|c| c.target_orbit_radius_m)
            .map(|target_r| {
                let body_radius_m = ev.body_radii.last().copied().unwrap_or(1e6);
                let floor_m = 1.05 * body_radius_m;
                (floor_m - target_r).max(0.0)
            })
            .unwrap_or(0.0),
        MissionObjective::Rendezvous => ev.v_inf_arr.norm(),
        _ => {
            let v_inf_arr_mag = ev.v_inf_arr.norm();
            let mu_target = *ev.body_mus.last().unwrap_or(&1.0);
            let body_radius_m = ev.body_radii.last().copied().unwrap_or(1e6);
            let cap = cfg.trajectory.capture.as_ref();
            let r_cap_configured = cap.and_then(|c| c.target_orbit_radius_m)
                .unwrap_or_else(|| body_radius_m * 3.0);
            // Phase 9y: a misconfigured `target_orbit_radius_m`
            // below the target body's own physical radius would otherwise
            // compute a vis-viva burn for an "orbit" inside the body's
            // surface. `config.rs::check_config` is the primary defense
            // (rejects this at load time); this clamp + flat
            // `CONSTRAINT_PENALTY_MS` is a second, in-evaluator safety net
            // (e.g. for a config assembled programmatically, bypassing
            // `check_config`) — flat, not graded, since this is a fixed
            // config value the search never continuously varies near a
            // boundary, unlike the flyby-periapsis/solar-perihelion floors
            // elsewhere in this evaluator.
            let (r_cap, floor_penalty_ms) = if r_cap_configured < body_radius_m {
                (body_radius_m, CONSTRAINT_PENALTY_MS)
            } else {
                (r_cap_configured, 0.0)
            };
            let e_cap = cap.map(|c| c.capture_eccentricity).unwrap_or(0.0);
            let v_peri = (mu_target * (1.0 + e_cap) / r_cap).sqrt();
            let v_hyp  = (v_inf_arr_mag * v_inf_arr_mag + 2.0 * mu_target / r_cap).sqrt();
            v_hyp - v_peri + floor_penalty_ms
        }
    }
}

/// Phase 2 fitness: total ΔV (departure escape + DSMs + LOI for Orbit missions).
pub fn phase2_fitness(
    params: &[f64],
    cfg: &MissionConfig,
    almanac: &Almanac,
    dep_jd_base: f64,
    flyby_bodies: &[String],
) -> Option<f64> {
    let ev = match evaluate_chromosome_graded(params, cfg, almanac, dep_jd_base, flyby_bodies) {
        Ok(ev) => ev,
        Err(legs_completed) => {
            return Some(infeasible_fitness_ms(n_legs(flyby_bodies), legs_completed));
        }
    };

    // Departure cost the SPACECRAFT pays — single source of truth in
    // design.rs: the full parking-orbit escape burn (v_hyp = sqrt(v∞² +
    // 2μ/r_p), not the sqrt(v_c²+v∞²) underestimate this previously
    // duplicated with a missing factor 2) in `ParkingOrbit` mode, or only the
    // perigee top-up beyond the launcher's C3 at this mass in `Launch` mode
    // (Phase 14b) — the reported `dv_departure_ms` stays the physical burn.
    let dv_dep = crate::design::departure_onboard_cost_ms(
        cfg, &cfg.optimization.as_ref()?.departure_body, ev.v_inf_dep_ms)?;

    let dv_dsm: f64 = ev.legs.iter().map(|l| l.leg.dv_dsm_ms).sum();

    // LOI burn (Orbit/Landing objectives).
    let dv_arr = arrival_dv_ms(cfg, &ev);

    // Graded soft-constraint penalty (9v-viii) — included in the search
    // fitness only, never in reported ΔV.
    Some(dv_dep + dv_dsm + dv_arr + ev.penalty_ms)
}

/// Post-search coordinate-descent local polish over the winning chromosome's
/// `eta` genes only (Phase 9x — the higher-value alternative
/// found while scoping 2-DSM-per-leg: primer-vector `‖λV‖ > 1` violations on
/// converged winners were overwhelmingly explained by DE/MBH leaving `eta`
/// short of its own per-leg optimum — up to 278 m/s per leg in the data that
/// motivated this — not by the one-DSM-per-leg model being structurally
/// insufficient; see the design notes
/// full controlled comparison).
///
/// For each leg in turn, grid-searches (then locally polishes) that leg's
/// `eta` ALONE against the FULL chromosome's total ΔV fitness
/// ([`phase2_fitness`]), never that leg's own DSM in isolation: a leg's
/// arrival v∞ feeds the next leg's departure state through [`flyby_turn`]
/// (see `evaluate_chromosome_graded`'s loop), so re-timing one leg's DSM
/// changes every downstream leg's cost too. An isolated per-leg repolish
/// (as `MissionPlanner/src/bin/two_dsm_refine.rs`'s diagnostic deliberately
/// does, for a different, narrower purpose) would miss that coupling and
/// could make the OVERALL trajectory worse while making one leg individually
/// cheaper — this function never does that: each leg's candidate `eta` is
/// only accepted if it strictly improves the FULL chromosome fitness, so
/// this can only ever help or leave the winner unchanged, never regress it.
///
/// Two coordinate-descent sweeps (each leg polished once, in order, then
/// repeated once) — matches the diminishing-returns-from-more-passes lesson
/// already recorded elsewhere in this project (Phase 9x-iv's Phase-1-restart
/// experiment): the first sweep captures the easy gains, a second catches
/// cross-leg knock-on effects from the first, a third was not found worth
/// its extra cost during development.
fn repolish_leg_etas(
    params: &mut Vec<f64>,
    cfg: &MissionConfig,
    almanac: &Almanac,
    dep_jd_base: f64,
    flyby_bodies: &[String],
    n: usize,
) -> (f64, f64) {
    const GRID_POINTS: usize = 21;
    const SWEEPS: usize = 2;

    let full_fitness = |p: &[f64]| -> f64 {
        phase2_fitness(p, cfg, almanac, dep_jd_base, flyby_bodies).unwrap_or(f64::MAX)
    };

    let before = full_fitness(params);
    let nm = NelderMead { max_iter: 60, ..Default::default() };

    for _ in 0..SWEEPS {
        for k in 0..n {
            // Same index formula as the `eta` accessor above — must stay in
            // lockstep with it.
            let eta_idx = 4 + 2 * k + 1;
            let base_params = params.clone();
            let leg_fitness = |x: &[f64]| -> Option<f64> {
                let mut trial = base_params.clone();
                trial[eta_idx] = x[0];
                phase2_fitness(&trial, cfg, almanac, dep_jd_base, flyby_bodies)
            };

            let current_total = full_fitness(params);
            let mut grid_best_eta = base_params[eta_idx];
            let mut grid_best_val = current_total;
            for i in 1..GRID_POINTS {
                let cand = i as f64 / GRID_POINTS as f64;
                if let Some(val) = leg_fitness(&[cand]) {
                    if val < grid_best_val {
                        grid_best_val = val;
                        grid_best_eta = cand;
                    }
                }
            }

            let nm_result = nm.run(&[(0.001, 0.999)], &[grid_best_eta], leg_fitness);
            if nm_result.best_fitness < current_total - 1.0e-6 {
                params[eta_idx] = nm_result.best_params[0];
            }
        }
    }

    let after = full_fitness(params);
    (before, after)
}

// ── DE variant dispatch ───────────────────────────────────────────────────────

/// Run one DE search with the configured variant: SHADE (self-adaptive F/CR,
/// `de_adaptive = true`, the default) or classic fixed-parameter
/// DE/rand/1/bin. Both accept elite seeds into the initial population and
/// return the final population for downstream elite harvesting.
fn run_de_variant<F, G>(
    mga: &crate::config::MgaParams,
    population_size: usize,
    generations: usize,
    seed: u64,
    bounds: &[(f64, f64)],
    seeds: &[Vec<f64>],
    fitness: F,
    on_generation: G,
) -> (DeResult, Vec<(Vec<f64>, f64)>)
where
    F: FnMut(&[f64]) -> Option<f64>,
    G: FnMut(usize, f64, &[f64]),
{
    if mga.de_adaptive {
        ShadeSolver {
            population_size,
            generations,
            seed,
            f_init: mga.de_f_weight,
            cr_init: mga.de_cr,
        }.run_seeded_with_progress(bounds, seeds, fitness, on_generation)
    } else {
        DeSolver {
            population_size,
            generations,
            f_weight: mga.de_f_weight,
            cr: mga.de_cr,
            seed,
        }.run_seeded_with_progress(bounds, seeds, fitness, on_generation)
    }
}

// ── Arc sampling ─────────────────────────────────────────────────────────────

/// Generate heliocentric position samples for each leg by Kepler-propagating
/// the two conic sub-arcs: leg-start → DSM under the pre-DSM state, and
/// DSM → leg-end under the post-DSM Lambert state. This is the exact two-body
/// dynamics the MGA-1DSM evaluator itself assumes, so the plotted arc is the
/// real analytic trajectory (replaced the original straight-line interpolation
/// between key points — the "arcs" it drew were literal chords).
fn sample_arc(ev: &ChromosomeEval, params: &[f64], _n_legs: usize) -> Vec<MgaArcPoint> {
    let mut pts = Vec::new();
    let half = ARC_SAMPLES_PER_LEG / 2;

    for (k, leg_ev) in ev.legs.iter().enumerate() {
        let tof_k = tof_days(params, k);
        let eta_k = eta(params, k);
        let t0    = leg_ev.t_start_days;

        let dt_a_s = eta_k * tof_k * 86_400.0;
        let dt_b_s = (1.0 - eta_k) * tof_k * 86_400.0;

        // Sub-arc 1: Kepler propagation from leg-start to DSM.
        for i in 0..=half {
            let frac = i as f64 / half as f64;
            let Some((r, v)) = propagate_kepler(
                leg_ev.r_sc_start, leg_ev.v_sc_start, frac * dt_a_s, MU_SUN_M3S2,
            ) else { continue };
            let t = t0 + frac * eta_k * tof_k;
            pts.push(MgaArcPoint { t_days: t, x_m: r.x, y_m: r.y, z_m: r.z, vx_mps: v.x, vy_mps: v.y, vz_mps: v.z, leg_idx: k, central_body: None });
        }

        // Sub-arc 2: Kepler propagation from the post-DSM Lambert state to leg-end.
        for i in 1..=half {
            let frac = i as f64 / half as f64;
            let Some((r, v)) = propagate_kepler(
                leg_ev.leg.r_dsm_m, leg_ev.leg.v_dsm_after_mps, frac * dt_b_s, MU_SUN_M3S2,
            ) else { continue };
            let t = t0 + eta_k * tof_k + frac * (1.0 - eta_k) * tof_k;
            pts.push(MgaArcPoint { t_days: t, x_m: r.x, y_m: r.y, z_m: r.z, vx_mps: v.x, vy_mps: v.y, vz_mps: v.z, leg_idx: k, central_body: None });
        }
    }

    pts
}

// ── CSV writers ───────────────────────────────────────────────────────────────

/// Write the best chromosome to `mga_best_chromosome.csv` so that
/// `run_mga_geometry` can re-evaluate it without re-running the optimizer.
///
/// Format: one header row + one data row containing all chromosome floats,
/// prefixed by a `flyby_bodies` column (semicolon-separated intermediate body names).
fn write_chromosome_csv(params: &[f64], flyby_bodies: &[String], out_dir: &str) {
    let bodies_str = flyby_bodies.join(";");
    let param_strs: Vec<String> = params.iter().map(|v| format!("{v:.15e}")).collect();
    let rows = vec![
        format!("flyby_bodies,{}", (0..params.len()).map(|i| format!("p{i}")).collect::<Vec<_>>().join(",")),
        format!("{bodies_str},{}", param_strs.join(",")),
    ];
    let path = format!("{out_dir}/mga_best_chromosome.csv");
    if let Err(e) = std::fs::write(&path, rows.join("\n") + "\n") {
        eprintln!("Warning: could not write {path}: {e}");
    } else {
        println!("  {path}");
    }
}

fn write_arc_csv(arc: &[MgaArcPoint], out_dir: &str) {
    let mut rows = vec!["t_days,x_m,y_m,z_m,leg_idx".to_string()];
    for p in arc {
        rows.push(format!("{},{},{},{},{}", p.t_days, p.x_m, p.y_m, p.z_m, p.leg_idx));
    }
    let path = format!("{out_dir}/mga_best.csv");
    if let Err(e) = std::fs::write(&path, rows.join("\n") + "\n") {
        eprintln!("Warning: could not write {path}: {e}");
    } else {
        println!("  {path}");
    }
}

/// Writes `mga_convergence.csv`: `generation,best_fitness`, plus one `pN`
/// column per chromosome parameter when `param_history` is non-empty and
/// its length matches `history` (for per-parameter convergence
/// plots). Falls back to fitness-only columns otherwise — callers that
/// don't have parameter data yet (or a length mismatch) degrade gracefully
/// rather than erroring.
fn write_convergence_csv(history: &[f64], param_history: &[Vec<f64>], out_dir: &str) {
    let with_params = param_history.len() == history.len() && !param_history.is_empty();
    let n_params = if with_params { param_history[0].len() } else { 0 };

    let mut header = "generation,best_fitness".to_string();
    if with_params {
        for j in 0..n_params {
            header.push_str(&format!(",p{j}"));
        }
    }
    let mut rows = vec![header];
    for (i, &f) in history.iter().enumerate() {
        let mut row = format!("{i},{f}");
        if with_params {
            for &v in &param_history[i] {
                row.push_str(&format!(",{v}"));
            }
        }
        rows.push(row);
    }
    let path = format!("{out_dir}/mga_convergence.csv");
    if let Err(e) = std::fs::write(&path, rows.join("\n") + "\n") {
        eprintln!("Warning: could not write {path}: {e}");
    } else {
        println!("  {path}");
    }
}

fn write_params_csv(result: &MgaResult, out_dir: &str) {
    let rows = vec![
        "dv_total_ms,dv_departure_ms,dv_arrival_ms,tof_total_days".to_string(),
        format!("{},{},{},{}",
            result.dv_total_ms, result.dv_departure_ms,
            result.dv_arrival_ms, result.tof_total_days),
    ];
    let path = format!("{out_dir}/mga_params.csv");
    if let Err(e) = std::fs::write(&path, rows.join("\n") + "\n") {
        eprintln!("Warning: could not write {path}: {e}");
    } else {
        println!("  {path}");
    }
}

/// Write per-leg details to `mga_legs.csv`.
///
/// Columns: leg_idx, body_dep, body_arr, t_dep_jd, t_arr_jd, tof_days, eta,
/// dv_dsm_ms, vinf_dep_ms, vinf_arr_ms, turn_deg, rp_km, rp_norm,
/// x_dsm_m, y_dsm_m, z_dsm_m.
///
/// `vinf_dep_ms` for leg 0 = departure v∞ from the launch body;
/// for later legs it equals `vinf_arr_ms` of the previous leg
/// (unpowered flyby conserves |v_∞| exactly).
/// Flyby columns (`turn_deg`, `rp_km`, `rp_norm`) are 0.0 for the final
/// leg (arrival at the target — no flyby performed there).
/// Orbital inclination of a heliocentric orbit from position and velocity
/// vectors in the ecliptic J2000 frame (z-axis = ecliptic north).
fn orbital_inclination_deg(r: &Vector3<f64>, v: &Vector3<f64>) -> f64 {
    let h = r.cross(v);
    let h_norm = h.norm();
    if h_norm < 1e-10 { return 0.0; }
    (h.z / h_norm).clamp(-1.0, 1.0).acos().to_degrees()
}

fn write_legs_csv(
    ev:           &ChromosomeEval,
    params:       &[f64],
    flyby_bodies: &[String],
    body_names:   &[String],  // full sequence: dep + flybys + target
    out_dir:      &str,
) {
    let n = n_legs(flyby_bodies);
    let hdr = "leg_idx,body_dep,body_arr,t_dep_jd,t_arr_jd,tof_days,eta,\
               dv_dsm_ms,vinf_dep_ms,vinf_arr_ms,turn_deg,rp_km,rp_norm,\
               x_dsm_m,y_dsm_m,z_dsm_m,\
               rp_dep_m,ra_dep_m,rp_lambert_m,ra_lambert_m,\
               i_dep_deg,i_lambert_deg";

    let mut rows = vec![hdr.to_string()];
    let mut t_dep_jd = ev.dep_jd;

    for k in 0..n {
        let tof_k = tof_days(params, k);
        let eta_k = eta(params, k);
        let t_arr_jd = t_dep_jd + tof_k;
        let leg = &ev.legs[k];

        let vinf_dep_ms = if k == 0 {
            ev.v_inf_dep_ms
        } else {
            // Unpowered flyby conserves |v_∞|.
            ev.legs[k - 1].leg.v_inf_arr_mps.norm()
        };
        let vinf_arr_ms = leg.leg.v_inf_arr_mps.norm();

        // Flyby columns — only meaningful for intermediate bodies.
        let (turn_deg, rp_km, rp_norm_val) = if k < n - 1 {
            let j      = k;
            let rp_n   = rp_norm(params, n, j);
            let rp_m   = rp_n * ev.body_radii[k + 1];
            let mu_b   = ev.body_mus[k + 1];
            let v_inf_sq = vinf_arr_ms * vinf_arr_ms;
            let e      = 1.0 + rp_m * v_inf_sq / mu_b;
            let turn   = (2.0 * (1.0 / e).min(1.0).asin()).to_degrees();
            (turn, rp_m / 1000.0, rp_n)
        } else {
            (0.0, 0.0, 0.0)
        };

        let dsm = &leg.leg;
        let AU = 1.495_978_707e11_f64;

        // Inclination of the Keplerian departure sub-arc (h = r_start × v_start).
        let i_dep_deg     = orbital_inclination_deg(&leg.r_sc_start, &leg.v_sc_start);
        // Inclination of the Lambert sub-arc (h = r_dsm × v_after_dsm).
        let i_lambert_deg = orbital_inclination_deg(&dsm.r_dsm_m, &dsm.v_dsm_after_mps);

        rows.push(format!(
            "{k},{},{},{:.6},{:.6},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4}",
            body_names[k], body_names[k + 1],
            t_dep_jd, t_arr_jd,
            tof_k, eta_k,
            dsm.dv_dsm_ms,
            vinf_dep_ms, vinf_arr_ms,
            turn_deg, rp_km, rp_norm_val,
            dsm.r_dsm_m.x, dsm.r_dsm_m.y, dsm.r_dsm_m.z,
            dsm.rp_dep_m / AU, dsm.ra_dep_m / AU,
            dsm.rp_lambert_m / AU, dsm.ra_lambert_m / AU,
            i_dep_deg, i_lambert_deg,
        ));

        t_dep_jd = t_arr_jd;
    }

    let path = format!("{out_dir}/mga_legs.csv");
    if let Err(e) = std::fs::write(&path, rows.join("\n") + "\n") {
        eprintln!("Warning: could not write {path}: {e}");
    } else {
        println!("  {path}");
    }
}

// ── Dopri5 re-propagation ─────────────────────────────────────────────────────

/// Tolerances for the MGA re-propagation step — same values as the rest of the
/// Layer 1 propagator (see `design.rs` `PROPAGATOR_RTOL`/`PROPAGATOR_ATOL`).
const REPROP_RTOL: f64 = 1e-8;
const REPROP_ATOL: f64 = 1e-3;

/// Number of trajectory sample points per sub-arc segment (DSM to body).
/// Two segments per leg × this count gives roughly `ARC_SAMPLES_PER_LEG`
/// total points per leg, matching the Keplerian arc density.
const REPROP_SAMPLES_PER_SEGMENT: usize = ARC_SAMPLES_PER_LEG / 2;

/// Re-propagate the MGA best solution using the Dopri5 integrator instead of
/// the Keplerian/Lambert arc used during optimisation.
///
/// For each leg `k`:
/// 1. Segment A — propagate from leg-start state for `η·T` seconds.
/// 2. Apply DSM ΔV instantaneously: `v_new = v_at_dsm_time + (v_L1 - v_dsm_before)`.
///    Under real dynamics the spacecraft won't be exactly at the Keplerian DSM
///    position, but it will be close — differential correction to close the
///    gap is a future task (Phase 9 follow-on), not part of this step.
/// 3. Segment B — propagate from post-DSM state for `(1-η)·T` seconds.
/// 4. At an intermediate flyby body: apply the pre-computed flyby ΔV
///    (`v_inf_out - v_inf_arr` plus the body's heliocentric velocity) to
///    obtain the post-flyby state for the next leg.
///
/// Returns an empty `Vec` on any failure (non-fatal — caller still has the
/// Keplerian arc).
fn reprop_mga_arc(
    ev: &ChromosomeEval,
    params: &[f64],
    flyby_bodies: &[String],
    cfg: &MissionConfig,
    almanac: &Almanac,
) -> Vec<MgaArcPoint> {
    let opt = match cfg.optimization.as_ref() { Some(o) => o, None => return vec![] };
    let n   = n_legs(flyby_bodies);

    // Absolute departure time in seconds from J2000 for ephemeris lookups.
    // `propagate`'s `t0_abs_s` is passed to body `state_at` callbacks; since
    // we're using an empty bodies list (point-mass Sun only — no SOI bodies
    // needed for a heliocentric-only multi-leg tour), the value only matters
    // internally as an offset. Use 0.0 consistently and advance it leg by leg.
    let mut r_sc = ev.legs[0].r_sc_start;
    let mut v_sc = ev.legs[0].v_sc_start;
    let mut t_acc_days = ev.legs[0].t_start_days; // cumulative time from departure

    let mut pts: Vec<MgaArcPoint> = Vec::new();

    for k in 0..n {
        let tof_k = tof_days(params, k);
        let eta_k = eta(params, k);
        let dt_a  = eta_k * tof_k * 86_400.0;          // segment A duration [s]
        let dt_b  = (1.0 - eta_k) * tof_k * 86_400.0;  // segment B duration [s]

        let leg = &ev.legs[k];

        // ── Segment A: leg-start to DSM ─────────────────────────────────────
        let sample_dt_a = (dt_a / REPROP_SAMPLES_PER_SEGMENT as f64).max(1.0);
        let seg_a = propagate(
            r_sc, v_sc,
            0.0, dt_a,
            MU_SUN_M3S2,
            &[],           // point-mass Sun only for heliocentric MGA legs
            sample_dt_a, REPROP_RTOL, REPROP_ATOL,
        );
        if seg_a.is_empty() {
            eprintln!("Warning: MGA re-propagation segment A empty on leg {k} — skipping reprop");
            return vec![];
        }
        for p in &seg_a {
            let t_days = t_acc_days + p.t_s / 86_400.0;
            pts.push(MgaArcPoint { t_days, x_m: p.r_m.x, y_m: p.r_m.y, z_m: p.r_m.z, vx_mps: p.v_mps.x, vy_mps: p.v_mps.y, vz_mps: p.v_mps.z, leg_idx: k, central_body: None });
        }

        // State at end of segment A (= real DSM position under Dopri5 dynamics).
        let last_a = seg_a.last().unwrap();
        let r_dsm_real = last_a.r_m;
        let v_dsm_real_before = last_a.v_mps;

        // ── Apply DSM ΔV ────────────────────────────────────────────────────
        // The DSM ΔV vector comes from the Keplerian evaluation: it is the
        // difference between v_L1 (Lambert departure) and v_dsm_before
        // (Keplerian propagation endpoint). Under real dynamics the spacecraft
        // is not exactly at r_dsm_keplerian, but we apply the same ΔV *vector*
        // — this is the first-cut approximation (no differential correction yet).
        let dv_vec = leg.leg.v_dsm_after_mps - leg.leg.v_dsm_before_mps;
        let v_dsm_real_after = v_dsm_real_before + dv_vec;

        // ── Segment B: DSM to next body ─────────────────────────────────────
        let sample_dt_b = (dt_b / REPROP_SAMPLES_PER_SEGMENT as f64).max(1.0);
        let seg_b = propagate(
            r_dsm_real, v_dsm_real_after,
            0.0, dt_b,
            MU_SUN_M3S2,
            &[],
            sample_dt_b, REPROP_RTOL, REPROP_ATOL,
        );
        if seg_b.is_empty() {
            eprintln!("Warning: MGA re-propagation segment B empty on leg {k} — skipping reprop");
            return vec![];
        }
        // Skip the first point of segment B — it duplicates the last of segment A.
        let t_base_b = t_acc_days + dt_a / 86_400.0;
        for p in seg_b.iter().skip(1) {
            let t_days = t_base_b + p.t_s / 86_400.0;
            pts.push(MgaArcPoint { t_days, x_m: p.r_m.x, y_m: p.r_m.y, z_m: p.r_m.z, vx_mps: p.v_mps.x, vy_mps: p.v_mps.y, vz_mps: p.v_mps.z, leg_idx: k, central_body: None });
        }

        // Advance cumulative time.
        t_acc_days += tof_k;

        // ── Intermediate flyby ───────────────────────────────────────────────
        // At a flyby body, take the real end-of-leg-B state, compute the
        // spacecraft-to-body relative velocity (v_∞_in under real dynamics),
        // apply the Rodrigues flyby turn using the same (rp, β) parameters
        // as the optimizer used, then set up the next leg start state.
        if k < n - 1 {
            let last_b = seg_b.last().unwrap();
            let r_fb = last_b.r_m;
            let v_sc_helic = last_b.v_mps;

            // Query flyby body's heliocentric velocity at this epoch.
            let body_name = &flyby_bodies[k];
            let arr_jd = ev.dep_jd + t_acc_days;
            let v_body = match get_body_state(almanac, body_name, arr_jd) {
                Some((_, vb)) => vb,
                None => {
                    eprintln!("Warning: could not get body state for {body_name} — skipping reprop");
                    return vec![];
                }
            };

            let v_inf_in = v_sc_helic - v_body;

            // Apply flyby turn with the chromosome's (rp_norm, beta) for this flyby.
            let j          = k;
            let rp_norm_j  = rp_norm(params, n, j);
            let beta_j     = beta(params, n, j);
            let r_body_cat = match body_models::TargetBody::by_name(body_name) {
                Some(c) => c,
                None => {
                    eprintln!("Warning: flyby body {body_name} not in catalog — skipping reprop");
                    return vec![];
                }
            };
            let rp_m       = rp_norm_j * r_body_cat.radius_m;
            let mu_body    = r_body_cat.mu_m3s2;
            let v_inf_out  = flyby_turn(v_inf_in, rp_m, beta_j, mu_body);

            r_sc = r_fb;
            v_sc = v_body + v_inf_out;
        }
    }

    pts
}

fn write_repropagated_csv(arc: &[MgaArcPoint], out_dir: &str) {
    if arc.is_empty() {
        return;
    }
    let mut rows = vec!["t_days,x_m,y_m,z_m,leg_idx".to_string()];
    for p in arc {
        rows.push(format!("{},{},{},{},{}", p.t_days, p.x_m, p.y_m, p.z_m, p.leg_idx));
    }
    let path = format!("{out_dir}/mga_repropagated.csv");
    if let Err(e) = std::fs::write(&path, rows.join("\n") + "\n") {
        eprintln!("Warning: could not write {path}: {e}");
    } else {
        println!("  {path}");
    }
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Run the MGA-1DSM DE optimizer for the configured flyby sequence and return
/// a result compatible with the existing `OptimizeComputeResult` shape so that
/// the server routes and CLI output layer can handle it uniformly.
///
/// Progress callback: `on_step(step_index, phase, best_fitness, best_params_so_far,
/// best_legs_so_far)`. `best_legs_so_far` is the per-leg trajectory state
/// ([`MgaLegStepInfo`]) for `best_params_so_far`, re-derived once per
/// generation (backlog item #18) — empty when that chromosome
/// isn't evaluable (should not normally happen for a generation's own best).
///
/// `on_sequence(seq_idx, seq_count, flyby_bodies, is_direct_baseline)` fires
/// once, BEFORE that sequence's own generations start streaming through
/// `on_step` — for the `sequence_search` (Tisserand auto-discovery) path,
/// once per candidate sequence being optimized (Phase 9k "step-stream
/// context" ask: the frontend's live-replay view needs to know
/// which real flyby-body sequence a stream of steps belongs to, since under
/// auto-discovery this isn't knowable from the request alone — several
/// candidate sequences are evaluated in turn). Also fires exactly once for
/// the fixed-sequence (non-auto) path, with `seq_idx = 0, seq_count = 1`,
/// so callers can rely on it firing unconditionally rather than branching
/// on which path was taken.
pub fn run_mga<F, S>(
    cfg: &MissionConfig,
    almanac: &Almanac,
    mut on_step: F,
    mut on_sequence: S,
) -> Result<MgaResult, String>
where
    F: FnMut(usize, u8, f64, &[f64], &[MgaLegStepInfo]),
    S: FnMut(usize, usize, &[String], bool),
{
    let opt = cfg.optimization.as_ref()
        .ok_or("optimization config is required for MGA")?;
    let mga = opt.mga.as_ref()
        .ok_or("optimization.mga is required when method = \"MGA\"")?;

    // Determine the flyby-body sequence — either from the Tisserand beam
    // search (when `sequence_search` is configured) or from the static
    // `flyby_bodies` list the user specified in the config.
    if let Some(ss_cfg) = &mga.sequence_search {
        // ── Sequence-search path ────────────────────────────────────────────
        println!(
            "MGA sequence search: departure={}, target={}, {} candidate bodies, beam_width={}, max_legs={}",
            opt.departure_body, opt.target_body,
            ss_cfg.candidate_bodies.len(), ss_cfg.beam_width, ss_cfg.max_legs
        );
        let sequences = sequence_search::run_sequence_search(ss_cfg, &opt.departure_body, &opt.target_body);
        sequence_search::print_sequence_table(&sequences, &opt.departure_body, &opt.target_body);

        if sequences.is_empty() {
            return Err("Tisserand beam search found no feasible sequences — try relaxing max_legs or beam_width".into());
        }

        // Run the fixed-sequence inner optimizer for each top candidate,
        // plus the direct (zero-flyby) baseline when the beam search didn't
        // already emit one — the ranking should always show what the gravity
        // assists are actually beating (added).
        let n_opt = ss_cfg.max_sequences_to_optimize.min(sequences.len());
        let mut to_run: Vec<sequence_search::RankedSequence> =
            sequences.iter().take(n_opt).cloned().collect();
        if !to_run.iter().any(|s| s.flyby_bodies.is_empty()) {
            to_run.push(sequence_search::direct_baseline(
                ss_cfg, &opt.departure_body, &opt.target_body,
            ));
        }
        let mut best_result: Option<MgaResult> = None;
        let mut best_flyby_bodies: Vec<String> = vec![];
        let mut best_dv = f64::MAX;

        // Collect per-sequence results for mga_sequence_ranking.csv.
        struct SeqRow {
            rank: usize,
            sequence: String,
            n_legs: usize,
            total_dv_ms: f64,
            tisserand_score: f64,
            estimated_vinf_arr_ms: f64,
        }
        let mut ranking_rows: Vec<SeqRow> = Vec::new();
        let mut rank_counter = 1usize;

        for (seq_idx, seq) in to_run.iter().enumerate() {
            let body_chain: Vec<String> = {
                let mut chain = vec![opt.departure_body.clone()];
                chain.extend_from_slice(&seq.flyby_bodies);
                chain.push(opt.target_body.clone());
                chain
            };
            let sequence_str = body_chain.join(" → ");
            let n_legs = body_chain.len() - 1;
            let label = if seq.flyby_bodies.is_empty() { " [direct baseline]" } else { "" };
            println!(
                "\n── Sequence {}/{}: {}{} (Tisserand score: {:.2}) ──",
                seq_idx + 1, to_run.len(),
                sequence_str, label,
                seq.tisserand_score
            );
            on_sequence(seq_idx, to_run.len(), &seq.flyby_bodies, seq.flyby_bodies.is_empty());
            match run_mga_fixed_sequence(cfg, almanac, &seq.flyby_bodies, &mut on_step) {
                Ok(result) => {
                    ranking_rows.push(SeqRow {
                        rank: rank_counter,
                        sequence: sequence_str.clone(),
                        n_legs,
                        total_dv_ms: result.dv_total_ms,
                        tisserand_score: seq.tisserand_score,
                        estimated_vinf_arr_ms: seq.estimated_vinf_arr_ms,
                    });
                    rank_counter += 1;
                    if result.dv_total_ms < best_dv {
                        best_dv = result.dv_total_ms;
                        best_flyby_bodies = seq.flyby_bodies.clone();
                        best_result = Some(result);
                    }
                }
                Err(e) => eprintln!("  [skip] sequence {}: {e}", sequence_str),
            }
        }

        let out_dir = cfg.simulation.output_dir.trim_end_matches('/');
        let _ = std::fs::create_dir_all(out_dir);

        // Write the best chromosome so `mga-geometry` can re-evaluate it.
        if let Some(ref r) = best_result {
            write_chromosome_csv(&r.best_params, &best_flyby_bodies, out_dir);

            // Re-write the WINNER's output CSVs. Each candidate's
            // `run_mga_fixed_sequence` call above overwrote these same files
            // in turn, so without this the on-disk CSVs (and every plot that
            // reads them) hold the LAST-RUN sequence, not the best one.
            // Found the per-sequence table said sequence 1 won at
            // 7239.8 m/s, but mga_params.csv held sequence 3's 11213.1 m/s.
            let dep_epoch_str = opt.departure_epoch.as_deref()
                .ok_or("optimization.departure_epoch is required for MGA")?;
            let dep_epoch = parse_epoch(dep_epoch_str)
                .map_err(|e| format!("departure_epoch parse: {e}"))?;
            let dep_jd_base = epoch_to_jd(dep_epoch);
            if let Some(ev) = evaluate_chromosome_detailed(
                &r.best_params, cfg, almanac, dep_jd_base, &best_flyby_bodies,
            ) {
                let mut full_history = r.phase1_history.clone();
                full_history.extend_from_slice(&r.convergence);
                let mut full_param_history = r.phase1_param_history.clone();
                full_param_history.extend_from_slice(&r.param_history);
                write_arc_csv(&r.arc, out_dir);
                write_repropagated_csv(&r.repropagated_arc, out_dir);
                write_convergence_csv(&full_history, &full_param_history, out_dir);
                write_params_csv(r, out_dir);
                write_legs_csv(&ev, &r.best_params, &best_flyby_bodies, &r.body_sequence, out_dir);
                println!("  Winner's outputs re-written: {}", r.body_sequence.join(" → "));
            }
        }

        // Write mga_sequence_ranking.csv so plot_sequence_ranking.py can read it.
        {
            let mut rows = vec![
                "rank,sequence,n_legs,total_dv_ms,tisserand_score,estimated_vinf_arr_ms".to_string()
            ];
            // Re-sort ranking_rows by total_dv_ms ascending before writing.
            ranking_rows.sort_by(|a, b| a.total_dv_ms.partial_cmp(&b.total_dv_ms).unwrap_or(std::cmp::Ordering::Equal));
            for (i, row) in ranking_rows.iter_mut().enumerate() {
                row.rank = i + 1;
                rows.push(format!("{},{},{},{:.2},{:.4},{:.2}",
                    row.rank, row.sequence, row.n_legs,
                    row.total_dv_ms, row.tisserand_score, row.estimated_vinf_arr_ms));
            }
            let path = format!("{out_dir}/mga_sequence_ranking.csv");
            match std::fs::write(&path, rows.join("\n") + "\n") {
                Ok(()) => println!("  {path}"),
                Err(e) => eprintln!("Warning: could not write {path}: {e}"),
            }
        }

        best_result.ok_or_else(|| "all candidate sequences failed — try more restarts or a wider beam".to_string())
    } else {
        // ── Fixed-sequence path ─────────────────────────────────────────────
        on_sequence(0, 1, &mga.flyby_bodies, false);
        let result = run_mga_fixed_sequence(cfg, almanac, &mga.flyby_bodies, &mut on_step)?;
        let out_dir = cfg.simulation.output_dir.trim_end_matches('/');
        let _ = std::fs::create_dir_all(out_dir);
        write_chromosome_csv(&result.best_params, &mga.flyby_bodies, out_dir);
        Ok(result)
    }
}

// ── Phase 9x: incremental pruning pre-search (Ceriotti 2010, Ch. 3) ──────────
//
// Decomposes the chromosome into per-leg "levels" (departure + leg 0 first,
// then one [flyby turn, leg k] block per subsequent leg — Ceriotti's Table
// 3.1) and prunes each level's search space down to a generous-threshold
// survivor set before adding the next leg's variables. The survivors are
// full-chromosome candidates (once pruning has processed every level) that
// are fed as elite seeds into the EXISTING Phase 2 DE — this augments, not
// replaces, the DE search (the design notes Phase 9x-ii is explicit: "don't throw
// away the DE").
//
// This module's chromosome layout ([`tof_days`]/[`eta`]/[`rp_norm`]/[`beta`])
// groups all TOFs together and all flyby turns together, which is NOT
// Ceriotti's per-leg level order — see `ceriotti_level_bounds`/
// `scatter_ceriotti_to_params` for the two-way mapping between the two.

/// Build Ceriotti-order per-level bounds from the flat chromosome bounds
/// already computed by [`build_bounds`] (same values, different grouping —
/// no bound is re-derived).
///
/// Level 0 = `[dep_offset, dep_vinf, theta_dep, phi_dep, tof_0, eta_0]`
/// (6 vars). Level `m` (`m = 1..n-1`) = `[rp_norm_{m-1}, beta_{m-1}, tof_m,
/// eta_m]` (4 vars) — the flyby turn that PRECEDES leg `m`, together with
/// leg `m`'s own variables, matching Ceriotti's Table 3.1 grouping (§3.2.4).
fn ceriotti_level_bounds(flat: &[(f64, f64)], n: usize) -> Vec<Vec<(f64, f64)>> {
    // Level blocks (N-as-gene): each leg's n_rev gene travels
    // with its own level, appended at the block's END so the TOF variable
    // keeps its block-local index (4 for level 0, 2 for levels 1+ — which
    // `resonance_bias_windows` relies on). Level 0 = 7 vars, levels 1+ = 5.
    let mut levels = Vec::with_capacity(n);
    let nrev_idx = |k: usize| 2 + 4 * n + k; // flat-layout index of leg k's gene
    levels.push(vec![flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], flat[nrev_idx(0)]]);
    for m in 1..n {
        let rp_beta_idx = 4 + 2 * n + 2 * (m - 1);
        let tof_eta_idx = 4 + 2 * m;
        levels.push(vec![
            flat[rp_beta_idx], flat[rp_beta_idx + 1],
            flat[tof_eta_idx], flat[tof_eta_idx + 1],
            flat[nrev_idx(m)],
        ]);
    }
    levels
}

/// Inverse of `ceriotti_level_bounds`'s grouping: scatter a full Ceriotti-
/// order variable vector (length `5n+2`, all `n` levels concatenated: a
/// 7-var level 0 + 5-var levels 1+) back into this module's `params` layout
/// (`chromosome_len(n)`, same length — the two orderings are a permutation
/// of the same variables).
fn scatter_ceriotti_to_params(n: usize, ceriotti: &[f64]) -> Vec<f64> {
    let mut params = vec![0.0; chromosome_len(n)];
    params[..6].copy_from_slice(&ceriotti[..6]); // dep_offset, dep_vinf, theta, phi, tof_0, eta_0
    params[2 + 4 * n] = ceriotti[6];             // n_rev_0 (level 0's last var)
    for m in 1..n {
        let base_m = 7 + 5 * (m - 1); // this level's block: [rp_{m-1}, beta_{m-1}, tof_m, eta_m, n_rev_m]
        let (rp, beta, tof_m, eta_m, nrev_m) = (
            ceriotti[base_m], ceriotti[base_m + 1], ceriotti[base_m + 2],
            ceriotti[base_m + 3], ceriotti[base_m + 4],
        );
        params[4 + 2 * m]     = tof_m;
        params[4 + 2 * m + 1] = eta_m;
        params[4 + 2 * n + 2 * (m - 1)]     = rp;
        params[4 + 2 * n + 2 * (m - 1) + 1] = beta;
        params[2 + 4 * n + m] = nrev_m;
    }
    params
}

/// Evaluate a Ceriotti-order prefix through `level_idx` (inclusive),
/// returning the cumulative partial cost (departure v∞ + DSM sum through
/// this level) or `None` if any leg up to this level is infeasible.
///
/// Mirrors the relevant subset of [`evaluate_chromosome_graded`]'s loop
/// body, restricted to legs `0..=level_idx` and WITHOUT the soft-constraint
/// graded-penalty tracking (9v-viii) — pruning is a fast coarse proxy
/// search, not the final fitness; soft constraints are still enforced when
/// the survivor seeds are later scored by the real `phase1_fitness`/
/// `phase2_fitness` functions inside the DE.
///
/// Including `dep_vinf` itself in the partial cost (rather than starting
/// from 0 at leg 0) mirrors Ceriotti Eq. (3.11), where the launch excess
/// velocity is one of the summed engine-provided ΔVs — here it is a
/// monotonic proxy for `departure_escape_dv_ms` (not identical to it, since
/// that also depends on the parking-orbit radius), but this is exactly
/// Ceriotti's own justification for using a different pruning criterion
/// than the true objective (§3.2.3): only relative ranking matters for
/// pruning to be safe, not an exact ΔV match.
fn evaluate_ceriotti_prefix(
    ceriotti_prefix: &[f64],
    level_idx: usize,
    cfg: &MissionConfig,
    almanac: &Almanac,
    dep_jd_base: f64,
    flyby_bodies: &[String],
) -> Option<f64> {
    evaluate_ceriotti_prefix_full(ceriotti_prefix, level_idx, cfg, almanac, dep_jd_base, flyby_bodies)
        .map(|ev| ev.cost)
}

/// Richer result of a Ceriotti-prefix evaluation, exposing the state at the
/// prefix's final body — the "interface" quantities the bidirectional
/// matching stage (Phase 9x) pairs against backward suffixes:
/// a forward prefix through level `m-1` arrives at body `m` at `arr_jd`
/// with incoming hyperbolic excess `v_inf_in` (the flyby turn at body `m`
/// belongs to level `m` and is deliberately NOT applied here).
struct CeriottiPrefixEval {
    /// Cumulative partial cost (departure v∞ + DSM sum through this level) [m/s].
    cost: f64,
    /// JD of arrival at the body that ends leg `level_idx`.
    arr_jd: f64,
    /// Incoming v∞ vector at that body [m/s].
    v_inf_in: Vector3<f64>,
    /// Graded solar-perihelion-floor penalty accumulated across these legs
    /// [m/s-equivalent] — mirrors `evaluate_chromosome_graded`'s check
    /// (Phase 9v-viii), NOT folded into `cost`. Added after a
    /// real bug: `backfit_prefix_to_suffix`'s DE had no awareness of this
    /// floor at all (this function previously didn't track it), so it was
    /// free to swing a sub-arc arbitrarily close to the Sun to hit an exact
    /// target velocity — the same "free" exploit 9v-viii already found and
    /// fixed for the main chromosome DE, just missing here because this
    /// evaluator was written fresh. Confirmed: a live Cassini-2 run showed
    /// a near-zero-mismatch (0.5 m/s) backfit stitch score 127,853.7 m/s in
    /// the real evaluator despite the backfit DE itself reporting only
    /// 6,561.3 — a ~121,000 m/s gap consistent with ~1 full violation at
    /// `CONSTRAINT_PENALTY_MS = 1.0e5` plus real suffix/arrival cost.
    /// Ordinary forward-pruning callers (`evaluate_ceriotti_prefix`,
    /// interface-matching) still ignore this field — pruning is meant to
    /// stay a fast, penalty-free coarse proxy; only `backfit_prefix_to_suffix`
    /// (the one caller whose DE result gets used AS a final chromosome
    /// candidate) needs to see it.
    penalty_ms: f64,
}

fn evaluate_ceriotti_prefix_full(
    ceriotti_prefix: &[f64],
    level_idx: usize,
    cfg: &MissionConfig,
    almanac: &Almanac,
    dep_jd_base: f64,
    flyby_bodies: &[String],
) -> Option<CeriottiPrefixEval> {
    let opt = cfg.optimization.as_ref()?;
    let mga = opt.mga.as_ref()?;
    let n = n_legs(flyby_bodies);

    let mut body_names: Vec<&str> = Vec::with_capacity(n + 1);
    body_names.push(&opt.departure_body);
    for fb in flyby_bodies { body_names.push(fb.as_str()); }
    body_names.push(&opt.target_body);

    let dep_offset_d = ceriotti_prefix[0];
    let dep_vinf     = ceriotti_prefix[1];
    let (theta, phi) = decode_theta_phi(ceriotti_prefix[2], ceriotti_prefix[3]);
    let dep_jd = dep_jd_base + dep_offset_d;

    let (r_dep, v_dep) = get_body_state(almanac, body_names[0], dep_jd)?;
    let v_inf_vec = Vector3::new(
        dep_vinf * phi.cos() * theta.cos(),
        dep_vinf * phi.cos() * theta.sin(),
        dep_vinf * phi.sin(),
    );
    let mut r_sc = r_dep;
    let mut v_sc = v_dep + v_inf_vec;
    let mut t_days = 0.0;
    let mut cumulative_dsm = 0.0;
    let mut last_v_inf_arr = Vector3::zeros();
    let mut penalty_ms = 0.0;

    for k in 0..=level_idx {
        // Level block layout (N-as-gene): level 0 = 7 vars
        // [dep×4, tof_0, eta_0, n_rev_0]; level m = 5 vars
        // [rp_{m-1}, beta_{m-1}, tof_m, eta_m, n_rev_m].
        let (tof_k, eta_k, nrev_k) = if k == 0 {
            (ceriotti_prefix[4], ceriotti_prefix[5], ceriotti_prefix[6])
        } else {
            let base_k = 7 + 5 * (k - 1);
            (ceriotti_prefix[base_k + 2], ceriotti_prefix[base_k + 3], ceriotti_prefix[base_k + 4])
        };
        let nrev_k = (nrev_k.floor() as i64).clamp(0, 2) as u32;

        let arr_jd = dep_jd + t_days + tof_k;
        let (r_next, v_next) = get_body_state(almanac, body_names[k + 1], arr_jd)?;

        let leg = evaluate_mga_leg_n(r_sc, v_sc, eta_k, tof_k * 86_400.0, r_next, v_next, MU_SUN_M3S2, nrev_k)?;
        cumulative_dsm += leg.dv_dsm_ms;
        last_v_inf_arr = leg.v_inf_arr_mps;
        t_days += tof_k;

        // Same solar-perihelion-floor check as evaluate_chromosome_graded
        // (Phase 9v-viii) — see `CeriottiPrefixEval::penalty_ms` doc comment
        // for why this evaluator needs it too.
        let floor = mga.min_solar_perihelion_m;
        for rp in [leg.rp_dep_m, leg.rp_lambert_m] {
            if rp < floor {
                penalty_ms += CONSTRAINT_PENALTY_MS * ((floor - rp) / floor).min(1.0);
            }
        }

        // Advance to the next leg's start state only if this call continues
        // past leg k (i.e. k < level_idx) AND leg k ends at an intermediate
        // flyby body (k < n-1) — the turn variables for that flyby belong to
        // level k+1's own block.
        if k < level_idx && k < n - 1 {
            let base_next = 7 + 5 * k; // level (k+1)'s block: [rp_k, beta_k, ...]
            let rp_norm_k = ceriotti_prefix[base_next];
            let beta_k    = ceriotti_prefix[base_next + 1];
            let cat = body_models::TargetBody::by_name(body_names[k + 1])?;
            let rp_m = rp_norm_k * cat.radius_m;
            let v_inf_out = flyby_turn(leg.v_inf_arr_mps, rp_m, beta_k, cat.mu_m3s2);
            r_sc = r_next;
            v_sc = v_next + v_inf_out;
        }
    }

    Some(CeriottiPrefixEval {
        cost: dep_vinf + cumulative_dsm,
        arr_jd: dep_jd + t_days,
        v_inf_in: last_v_inf_arr,
        penalty_ms,
    })
}

/// Fraction of a return-leg level's samples drawn from a resonance window
/// instead of the ordinary uniform box, when [`resonance_bias_windows`]
/// finds one. Deliberately well short of 1.0 — this is a bias, not a
/// constraint: the search must still be able to find a cheap non-resonant
/// TOF if one exists (a return leg CAN occasionally be cheap off-resonance
/// for an unusual geometry), and the ordinary uniform draws keep covering
/// the rest of the box for every other level regardless.
const RESONANCE_BIAS_FRACTION: f64 = 0.35;

/// Returns [`RESONANCE_BIAS_FRACTION`], overridable via `MGA_RESONANCE_BIAS_FRACTION`
/// — lets an isolation test set this to 0.0 to run the pruning
/// engine with plain uniform sampling (no resonance bias at all), separating
/// "does pruning itself regress this benchmark" from "does the resonance
/// bias specifically regress it," without a code change per experiment.
fn resonance_bias_fraction() -> f64 {
    std::env::var("MGA_RESONANCE_BIAS_FRACTION")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(RESONANCE_BIAS_FRACTION)
}

/// Highest resonance order considered (return TOF ≈ N × body period, up to
/// this N). VEEGA-class real missions use N=1-2; this is a generous margin,
/// not a hardcoded fact about any specific mission.
const RESONANCE_MAX_N: u32 = 8;

/// Half-width of each resonance window, as a fraction of the body's own
/// period — e.g. 0.05 means ±5% of one period around each `N × period`
/// candidate. Wide enough that the pruning engine's own sampling/threshold
/// logic still has real work to do inside the window (this is a prior on
/// WHERE to look, not a precomputed answer).
const RESONANCE_WINDOW_FRAC: f64 = 0.05;

/// For any level whose newly-introduced leg is a same-body return (its
/// start body equals its end body, e.g. VEEGA's Earth->Earth resonant
/// leg), compute TOF windows near integer multiples of that body's real
/// heliocentric period — the generic form of the domain knowledge that
/// hand-solves Galileo's VEEGA leg 2. Not
/// specific to Earth or to VEEGA: fires for ANY same-body-to-same-body leg
/// in ANY sequence, derived at runtime from the body's own catalog
/// `sma_m`. A return leg is only physically feasible (cheap) near such a
/// TOF regardless of which body or mission it is — this is celestial
/// mechanics, not mission-specific tuning.
///
/// Returns a map from `level_idx` to `(tof_local_idx, nrev_local_idx,
/// windows)`, where the two indices are that level's own block-local
/// positions of the TOF and n_rev-gene variables (see
/// [`ceriotti_level_bounds`]'s layout: TOF at 4 / n_rev at 6 for level 0's
/// 7-var block, TOF at 2 / n_rev at 4 for every later level's 5-var block)
/// and `windows` is a list of `(lo, hi, n_rev)` TOF-day sub-ranges, each
/// clipped to that level's actual configured TOF bounds and tagged with the
/// resonance order N it corresponds to — a return leg near `N × period`
/// has the spacecraft itself sweeping ~N revolutions (its transfer orbit's
/// period is necessarily comparable to the body's own for a same-body
/// return), so the sampler sets the leg's n_rev gene alongside the TOF.
fn resonance_bias_windows(
    n: usize,
    flyby_bodies: &[String],
    departure_body: &str,
    target_body: &str,
    level_bounds: &[Vec<(f64, f64)>],
) -> std::collections::HashMap<usize, (usize, usize, Vec<(f64, f64, u32)>)> {
    let mut body_names: Vec<&str> = Vec::with_capacity(n + 1);
    body_names.push(departure_body);
    for fb in flyby_bodies { body_names.push(fb.as_str()); }
    body_names.push(target_body);

    let mut out = std::collections::HashMap::new();
    for level_idx in 0..n {
        // Level `level_idx` introduces leg `level_idx`'s own TOF (see
        // ceriotti_level_bounds: level 0 = leg 0's vars, level m = leg m's
        // vars alongside flyby m-1's turn).
        let leg_k = level_idx;
        let start_body = body_names[leg_k];
        let end_body = body_names[leg_k + 1];
        if !start_body.eq_ignore_ascii_case(end_body) { continue; }

        let Some(cat) = body_models::TargetBody::by_name(start_body) else { continue };
        let Some(sma_m) = cat.sma_m else { continue };
        let period_days = orbital_period_s(sma_m, MU_SUN_M3S2) / 86_400.0;

        let (tof_local_idx, nrev_local_idx) = if level_idx == 0 { (4, 6) } else { (2, 4) };
        let (lo, hi) = level_bounds[level_idx][tof_local_idx];
        let half_width = RESONANCE_WINDOW_FRAC * period_days;

        let mut windows = Vec::new();
        for nrev in 1..=RESONANCE_MAX_N {
            let center = nrev as f64 * period_days;
            if center + half_width < lo || center - half_width > hi { continue; }
            windows.push(((center - half_width).max(lo), (center + half_width).min(hi), nrev));
        }
        if !windows.is_empty() {
            println!(
                "  Phase 9x resonance bias: level {level_idx} ({start_body}->{end_body} return leg, period {period_days:.1} d) — {} window(s) in TOF bounds [{lo:.1}, {hi:.1}] d",
                windows.len(),
            );
            out.insert(level_idx, (tof_local_idx, nrev_local_idx, windows));
        }
    }
    out
}

/// Highest resonance order enumerated by [`resonance_family_branches`] —
/// user-set ("up to N=3 is enough, for now"). Deliberately lower
/// than the bias sampler's `RESONANCE_MAX_N` (8): branching multiplies whole
/// search runs, so it only enumerates the families that plausibly matter;
/// the probabilistic bias still covers the higher orders inside the
/// unrestricted branch.
const FAMILY_BRANCH_MAX_N: u32 = 3;

/// Enumerate resonance-family search branches.
///
/// The resonance ORDER of a same-body return leg (its TOF ≈ N × the body's
/// heliocentric period) is a discrete structural choice separating fitness
/// basins by hundreds of days of TOF — far beyond the reach of any downhill
/// method (eta grid, L-BFGS, Nelder-Mead descent) and of MBH's kicks
/// (±`kick_scale`×bound-width ≈ ±97 d on a typical leg bound, vs. the ~365 d
/// jump between adjacent families). Whether the search found the right
/// family was therefore seed-luck (live example: VEEGA seed 42 landing the
/// 1-year Earth-Earth family at 6,237 m/s where the 2-year family holds
/// ~4,100). This function makes the choice EXHAUSTIVE instead of
/// probabilistic: one branch per (same-body leg, N ∈ 1..=[`FAMILY_BRANCH_MAX_N`])
/// window that fits inside the leg's configured TOF bounds — each branch is
/// the full chromosome bounds with that leg's TOF RESTRICTED to the window
/// (±[`RESONANCE_WINDOW_FRAC`]·period, same width the bias sampler uses) —
/// plus always the unrestricted "full bounds" branch first, so best-of-
/// branches can only match or beat the previous single-run behaviour.
/// Multiple same-body legs expand as a cartesian product (each leg:
/// unrestricted ∪ its windows). A sequence with no same-body legs returns
/// exactly one branch, preserving today's behaviour and runtime.
fn resonance_family_branches(
    bounds: &[(f64, f64)],
    body_names: &[&str],
    n: usize,
) -> Vec<(String, Vec<(f64, f64)>)> {
    let mut branches: Vec<(String, Vec<(f64, f64)>)> =
        vec![("full bounds".to_string(), bounds.to_vec())];

    for k in 0..n {
        if !body_names[k].eq_ignore_ascii_case(body_names[k + 1]) { continue; }
        let Some(cat) = body_models::TargetBody::by_name(body_names[k]) else { continue };
        let Some(sma_m) = cat.sma_m else { continue };
        let period_days = orbital_period_s(sma_m, MU_SUN_M3S2) / 86_400.0;
        let (lo, hi) = bounds[4 + 2 * k];
        let half = RESONANCE_WINDOW_FRAC * period_days;

        let mut windows: Vec<(String, f64, f64)> = Vec::new();
        for nrev in 1..=FAMILY_BRANCH_MAX_N {
            let c = nrev as f64 * period_days;
            if c + half < lo || c - half > hi { continue; }
            windows.push((
                format!("leg{k} {}={nrev} ({:.0}-{:.0} d)", body_names[k], (c - half).max(lo), (c + half).min(hi)),
                (c - half).max(lo),
                (c + half).min(hi),
            ));
        }
        if windows.is_empty() { continue; }

        // Cartesian expansion: every existing branch × (leg k unrestricted ∪
        // each of leg k's windows).
        let mut expanded = Vec::with_capacity(branches.len() * (windows.len() + 1));
        for (label, b) in &branches {
            expanded.push((label.clone(), b.clone()));
            for (wl, w_lo, w_hi) in &windows {
                let mut nb = b.clone();
                nb[4 + 2 * k] = (*w_lo, *w_hi);
                let nl = if label == "full bounds" { wl.clone() } else { format!("{label} + {wl}") };
                expanded.push((nl, nb));
            }
        }
        branches = expanded;
    }

    branches
}

// ── Phase 9x bidirectional pruning ("meet in the middle") ────────
//
// Generic promotion of the backward-fit decomposition that hand-solved
// Galileo's VEEGA (galileo_leg3_search / galileo_backfit_legs012,
//). A forward-sequential search has no mechanism to carry a
// rigid downstream requirement (a resonant leg freezes the post-resonance
// departure epoch, position, AND |v∞| at once) back upstream — it only
// discovers the mismatch as a huge DSM after the fact. The backward pass
// makes that requirement explicit: suffixes are searched from the target
// with a FREE handoff v∞ at their first body (standing in for "whatever
// the prefix delivers", exactly the leg3_search trick), and survivors
// carry (interface epoch, required outgoing v∞ vector, suffix cost). The
// matching stage then scores prefix×suffix pairs physically — the same
// fitness galileo_backfit_legs012 validated — and stitches the best pairs
// into full chromosomes injected as Phase 2 seeds. No resonance or body
// knowledge anywhere: it automatically covers ANY rigid-downstream-
// constraint case (resonances, tight arrival-v∞ caps, fixed arrival dates).

/// Upper bound on the sampled handoff v∞ magnitude at a suffix's first body
/// [m/s]. Intermediate-body hyperbolic excesses on real MGA missions run to
/// ~9 km/s (Galileo's EGA2: 8,919 m/s); 20 km/s is the same generous margin
/// `galileo_leg3_search` used — a handoff this fast is search headroom, not
/// an expected solution.
const BWD_VINF_MAX_MS: f64 = 20_000.0;

/// Constant added to every backward-suffix partial cost [m/s]. The pruning
/// threshold is multiplicative (`best × threshold_factor`), and a suffix's
/// cost is its DSM sum alone (the handoff v∞ is free — it stands in for the
/// prefix's delivery, not a burn), so a fully ballistic suffix (DSM ≈ 0,
/// exactly the case that matters most — see leg 3 of Galileo's VEEGA)
/// would otherwise collapse the threshold to ~0 and prune everything but
/// bit-identical zeros. The floor keeps the admitted band physically
/// meaningful: with factor 3, survivors stay within ~2×floor + 3×best_dsm
/// of the best. Constant across all candidates, so ranking is unaffected.
const BWD_COST_FLOOR_MS: f64 = 100.0;

/// Hard gate on the epoch gap between a forward prefix's arrival at the
/// interface body and a backward suffix's sampled epoch there [days]. Wider
/// gaps mean the suffix legs would re-evaluate at materially different body
/// geometry once stitched; the joint DE can absorb a small slide, not a
/// large one.
const MATCH_EPOCH_TOL_DAYS: f64 = 15.0;

/// Linear penalty per day of interface-epoch gap inside the gate
/// [m/s-equivalent per day]. A coarse proxy for the correction the suffix's
/// first leg must absorb when its epochs slide (body positions move at
/// ~30 km/s but the Lambert re-solve absorbs most of it via TOF); only the
/// relative ranking of candidate matches needs this, not an exact price.
const MATCH_EPOCH_PENALTY_MS_PER_DAY: f64 = 100.0;

/// Suffix-refinement DE budget (population, generations): after the
/// backward pruning pass, each split's suffix is polished by a short DE
/// over the full suffix variable set, minimizing suffix DSM — the direct
/// generalization of `galileo_leg3_search` (which needed a DE, not random
/// sampling, to find leg 3's genuinely BALLISTIC solution; random samples
/// bottom out at multi-km/s DSMs in a 6+-dimensional suffix space). A
/// near-ballistic suffix is what makes the required handoff v∞ a real
/// downstream requirement instead of sampling noise. Matching considers
/// BOTH pools — epoch-anchored pruning survivors and DE-refined
/// individuals — since the DE may migrate to a ballistic suffix at an
/// epoch no forward prefix can reach (the epoch gate sorts that out).
const BWD_REFINE_POP: usize = 64;
const BWD_REFINE_GEN: usize = 200;

/// Fraction of backward level-0 samples anchored to a (randomly chosen)
/// forward survivor's actual interface state — BOTH its arrival epoch (a
/// ±[`MATCH_EPOCH_TOL_DAYS`] window around it) AND its delivered v∞ vector
/// (magnitude jittered by ±[`BWD_VINF_ANCHOR_JITTER_FRAC`], direction by
/// [`BWD_VINF_ANCHOR_ANGLE_JITTER_RAD`]) — instead of drawing both
/// independently and uniformly. Epoch-only anchoring (the first version of
/// this bias) found real matches but with large magnitude
/// mismatch: the backward suffix search freely minimizes its own DSM with
/// no notion of what any real prefix delivers, so its cheapest solution's
/// v∞ can land far from any prefix's actual v∞ at a compatible epoch — the
/// VEEGA full-budget validation that day showed exactly this (a 53 m/s
/// near-ballistic suffix found, but 13,366 m/s total match score, dominated
/// by magnitude mismatch). Anchoring the vector directly is the generic
/// form of `galileo_backfit_legs012`'s actual method — it fixed the leg-3
/// TARGET v∞ and searched legs 0-2 for a delivery matching it; here the
/// (epoch, v∞) pair is read off an already-computed forward survivor
/// instead of a hardcoded historical value, so the same physics applies to
/// any sequence. Both anchors are drawn from the SAME chosen prefix (not
/// independently) — they must be physically consistent with one another.
const BWD_EPOCH_ANCHOR_FRACTION: f64 = 0.5;

/// Relative jitter applied to an anchored suffix's handoff v∞ MAGNITUDE
/// (fraction of the anchor prefix's own |v∞|). Nonzero so the pruning tree
/// still explores near the anchor rather than replaying one exact value —
/// the anchor is a strong prior on where the answer is, not a fixed
/// constraint (a prefix's own v∞ is itself only one sample from a
/// continuous family the pruning engine explores elsewhere).
const BWD_VINF_ANCHOR_JITTER_FRAC: f64 = 0.05;

/// Absolute jitter [rad] applied to an anchored suffix's handoff v∞
/// DIRECTION (both spherical angles), same rationale as
/// [`BWD_VINF_ANCHOR_JITTER_FRAC`].
const BWD_VINF_ANCHOR_ANGLE_JITTER_RAD: f64 = 0.15;

/// Backfit DE budget (population, generations) — the load-bearing stage of
/// the bidirectional pipeline (second iteration): for each
/// split's best backward suffix, a dedicated DE optimizes the PREFIX
/// variables against that suffix's handoff requirement, with the mismatch
/// INSIDE the fitness (not a post-hoc filter over random pruning samples,
/// which is measure-zero matching in a continuous 15+-dim space — the
/// first iteration's mistake). Direct generalization of
/// `galileo_backfit_legs012` (pop 150 × gen 300 was the budget that closed
/// the real VEEGA opening to 270-877 m/s across 3 seeds in ~2 min/seed
/// under history-pinned bounds); the interface epoch is pinned the same
/// way — the last prefix leg's TOF is the REMAINDER to the suffix's
/// sampled epoch, not a free variable — which removes the epoch-matching
/// problem entirely for these stitches.
const BACKFIT_POP: usize = 150;
const BACKFIT_GEN: usize = 300;

/// How many suffix targets get a backfit DE per split. Taken best-first
/// from the suffix pool, requiring at least
/// [`BACKFIT_TARGET_MIN_EPOCH_SEP_DAYS`] between chosen targets' interface
/// epochs — the pool's top entries are usually near-duplicates of one
/// solution, and a second target only buys anything if it represents a
/// genuinely different phasing family.
const BACKFIT_TARGETS_PER_SPLIT: usize = 2;
const BACKFIT_TARGET_MIN_EPOCH_SEP_DAYS: f64 = 30.0;

/// Level bounds for the backward suffix search covering legs `m..n-1`
/// (interface at body `m`, which is flyby index `m-1`). Same block shapes
/// as the forward Ceriotti levels — level 0 has 4 leading non-leg vars then
/// [tof, eta, n_rev], later levels are [rp, beta, tof, eta, n_rev] — so
/// [`resonance_bias_windows`]'s hardcoded block-local indices apply
/// unchanged. Level 0's leading vars: interface-epoch offset from
/// `dep_jd_base` [days] (bounds = departure window + the sum of the prefix
/// legs' TOF bounds — every epoch a forward prefix can arrive at), then the
/// free handoff v∞ (magnitude, theta, phi).
fn suffix_level_bounds(flat: &[(f64, f64)], n: usize, m: usize) -> Vec<Vec<(f64, f64)>> {
    let mut t_lo = flat[0].0;
    let mut t_hi = flat[0].1;
    for k in 0..m {
        t_lo += flat[4 + 2 * k].0;
        t_hi += flat[4 + 2 * k].1;
    }
    let nrev_idx = |k: usize| 2 + 4 * n + k;
    let mut levels = Vec::with_capacity(n - m);
    levels.push(vec![
        (t_lo, t_hi),
        (0.0, BWD_VINF_MAX_MS),
        (0.0, 2.0 * PI),
        (-PI / 2.0, PI / 2.0),
        flat[4 + 2 * m],
        flat[4 + 2 * m + 1],
        flat[nrev_idx(m)],
    ]);
    for j in 1..(n - m) {
        let leg = m + j;
        let rp_beta_idx = 4 + 2 * n + 2 * (leg - 1);
        levels.push(vec![
            flat[rp_beta_idx],
            flat[rp_beta_idx + 1],
            flat[4 + 2 * leg],
            flat[4 + 2 * leg + 1],
            flat[nrev_idx(leg)],
        ]);
    }
    levels
}

/// Evaluate a backward-suffix prefix through suffix-local level
/// `level_local` (inclusive): legs `m..=m+level_local`, started at body `m`
/// at the sampled epoch with the sampled handoff v∞. Mirror image of
/// [`evaluate_ceriotti_prefix_full`] — evaluated forward *within* the
/// suffix (each level extension appends the NEXT leg toward the target, so
/// earlier levels' variables and costs stay valid, unlike a literal
/// backward-in-time extension where the suffix-start v∞ would move with
/// every added leg). Cost = [`BWD_COST_FLOOR_MS`] + DSM sum (no v∞ term:
/// the handoff stands in for the prefix's delivery and is priced by the
/// matching stage, not here).
fn evaluate_suffix_prefix(
    suffix_vars: &[f64],
    level_local: usize,
    m: usize,
    cfg: &MissionConfig,
    almanac: &Almanac,
    dep_jd_base: f64,
    flyby_bodies: &[String],
) -> Option<f64> {
    let opt = cfg.optimization.as_ref()?;
    let mga = opt.mga.as_ref()?;
    let n = n_legs(flyby_bodies);

    let mut body_names: Vec<&str> = Vec::with_capacity(n + 1);
    body_names.push(&opt.departure_body);
    for fb in flyby_bodies { body_names.push(fb.as_str()); }
    body_names.push(&opt.target_body);

    let start_jd = dep_jd_base + suffix_vars[0];
    let vinf     = suffix_vars[1];
    let theta    = suffix_vars[2];
    let phi      = suffix_vars[3];

    let (r_start, v_body_start) = get_body_state(almanac, body_names[m], start_jd)?;
    let v_inf_vec = Vector3::new(
        vinf * phi.cos() * theta.cos(),
        vinf * phi.cos() * theta.sin(),
        vinf * phi.sin(),
    );
    let mut r_sc = r_start;
    let mut v_sc = v_body_start + v_inf_vec;
    let mut t_days = 0.0;
    let mut cumulative_dsm = 0.0;
    let mut penalty_ms = 0.0;

    for j in 0..=level_local {
        let leg_k = m + j;
        // Suffix block layout mirrors the forward Ceriotti blocks: level 0 =
        // 7 vars [t_off, vinf, theta, phi, tof, eta, n_rev]; level j ≥ 1 =
        // 5 vars [rp, beta, tof, eta, n_rev].
        let (tof_k, eta_k, nrev_k) = if j == 0 {
            (suffix_vars[4], suffix_vars[5], suffix_vars[6])
        } else {
            let base_j = 7 + 5 * (j - 1);
            (suffix_vars[base_j + 2], suffix_vars[base_j + 3], suffix_vars[base_j + 4])
        };
        let nrev_k = (nrev_k.floor() as i64).clamp(0, 2) as u32;

        let arr_jd = start_jd + t_days + tof_k;
        let (r_next, v_next) = get_body_state(almanac, body_names[leg_k + 1], arr_jd)?;
        let leg = evaluate_mga_leg_n(
            r_sc, v_sc, eta_k, tof_k * 86_400.0, r_next, v_next, MU_SUN_M3S2, nrev_k,
        )?;
        cumulative_dsm += leg.dv_dsm_ms;
        t_days += tof_k;

        // Same solar-perihelion-floor check as evaluate_chromosome_graded /
        // evaluate_ceriotti_prefix_full (Phase 9v-viii / the 
        // backfit bug fix) — without this, a suffix can look artificially
        // cheap during the backward search by secretly swinging close to
        // the Sun, get PICKED as a backfit target because it sorts as
        // cheap, and only reveal its true cost once fully stitched and
        // honestly re-evaluated (confirmed live: a Cassini-2 smoke run's
        // backfit target scored 21,753.6 m/s here but 232,657.3 m/s once
        // stitched — the suffix side of the same bug the prefix side had).
        let floor = mga.min_solar_perihelion_m;
        for rp in [leg.rp_dep_m, leg.rp_lambert_m] {
            if rp < floor {
                penalty_ms += CONSTRAINT_PENALTY_MS * ((floor - rp) / floor).min(1.0);
            }
        }

        // Advance through the flyby at body `leg_k + 1` only when this call
        // continues past leg `leg_k` — that turn's (rp, β) belong to the
        // next suffix level's own block.
        if j < level_local && leg_k < n - 1 {
            let base_next = 7 + 5 * j;
            let rp_norm_j = suffix_vars[base_next];
            let beta_j    = suffix_vars[base_next + 1];
            let cat = body_models::TargetBody::by_name(body_names[leg_k + 1])?;
            // Same periapsis-floor clamp as evaluate_chromosome_graded's
            // intermediate flybys — an unclamped rp here let the DE/pruning
            // request a physically-illegal turn "for free."
            let rp_raw = rp_norm_j * cat.radius_m;
            let rp_used = if rp_raw < mga.flyby_min_periapsis_m {
                penalty_ms += CONSTRAINT_PENALTY_MS
                    * ((mga.flyby_min_periapsis_m - rp_raw) / mga.flyby_min_periapsis_m).min(1.0);
                mga.flyby_min_periapsis_m
            } else {
                rp_raw
            };
            let v_inf_out = flyby_turn(leg.v_inf_arr_mps, rp_used, beta_j, cat.mu_m3s2);
            r_sc = r_next;
            v_sc = v_next + v_inf_out;
        }
    }

    Some(BWD_COST_FLOOR_MS + cumulative_dsm + penalty_ms)
}

/// Invert [`flyby_turn`]: given the incoming v∞ and the DESIRED outgoing v∞
/// direction, recover the `(r_p [m], β [rad])` pair that turns one into the
/// other. Well-posed because `flyby_turn`'s B-plane basis is deterministic
/// from `v_inf_in` alone: the Rodrigues rotation gives
/// `v_in × v_out = n̂ |v_in|² sin δ`, so the rotation axis is
/// `normalize(v_in × v_out)` and β follows from projecting it onto the same
/// (T̂, R̂) basis `flyby_turn` constructs; r_p comes from the turn-angle
/// equation `sin(δ/2) = μ/(μ + r_p v∞²)` (Battin 1999 §6.3). Degenerate
/// cases (near-zero turn or near-zero v∞) return `(f64::INFINITY, 0.0)` —
/// "no meaningful turn", caller clamps into its bounds.
fn invert_flyby_turn(
    v_inf_in: Vector3<f64>,
    v_inf_out: Vector3<f64>,
    mu_body: f64,
) -> (f64, f64) {
    let v = v_inf_in.norm();
    let vo = v_inf_out.norm();
    if v < 1.0 || vo < 1.0 {
        return (f64::INFINITY, 0.0);
    }
    let cos_d = (v_inf_in.dot(&v_inf_out) / (v * vo)).clamp(-1.0, 1.0);
    let delta = cos_d.acos();
    let s = (delta / 2.0).sin();
    let rp_m = if s < 1e-9 {
        f64::INFINITY
    } else {
        // sin(δ/2) = μ/(μ + r_p v²)  →  r_p = μ (1 − sin) / (sin · v²)
        mu_body * (1.0 - s) / (s * v * v)
    };
    let cross = v_inf_in.cross(&v_inf_out);
    let beta = if cross.norm() < 1e-9 * v * vo {
        0.0
    } else {
        let n_hat = cross / cross.norm();
        // Same basis construction as flyby_turn — must stay in lockstep.
        let s_hat = v_inf_in / v;
        let ref_vec = if s_hat.x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let t_hat = s_hat.cross(&ref_vec).normalize();
        let r_hat = s_hat.cross(&t_hat);
        n_hat.dot(&r_hat).atan2(n_hat.dot(&t_hat))
    };
    (rp_m, beta)
}

/// Assemble a full chromosome (this module's `params` layout) from a matched
/// (forward prefix, backward suffix) pair at split body `m`: departure block
/// + prefix legs/flybys from the forward vars, the interface flyby's
/// geometry-derived `(rp_norm, β)`, and the suffix legs/flybys from the
/// backward vars (whose first 4 vars — sampled interface epoch and handoff
/// v∞ — are discarded: the stitched chromosome's epoch and handoff come
/// from the prefix side). Every value is clamped into the DE box so the
/// diagnostic evaluation matches what the solver actually receives.
fn stitch_chromosome(
    n: usize,
    m: usize,
    fwd_vars: &[f64],
    bwd_vars: &[f64],
    rp_norm_iface: f64,
    beta_iface: f64,
    bounds: &[(f64, f64)],
) -> Vec<f64> {
    let mut p = vec![0.0; chromosome_len(n)];
    // Prefix (forward Ceriotti levels 0..m-1).
    p[..6].copy_from_slice(&fwd_vars[..6]);
    p[2 + 4 * n] = fwd_vars[6];
    for lvl in 1..m {
        let base = 7 + 5 * (lvl - 1);
        p[4 + 2 * n + 2 * (lvl - 1)]     = fwd_vars[base];
        p[4 + 2 * n + 2 * (lvl - 1) + 1] = fwd_vars[base + 1];
        p[4 + 2 * lvl]     = fwd_vars[base + 2];
        p[4 + 2 * lvl + 1] = fwd_vars[base + 3];
        p[2 + 4 * n + lvl] = fwd_vars[base + 4];
    }
    // Interface flyby (index m-1): geometry-derived turn parameters.
    p[4 + 2 * n + 2 * (m - 1)]     = rp_norm_iface;
    p[4 + 2 * n + 2 * (m - 1) + 1] = beta_iface;
    // Suffix leg m from the backward level-0 block.
    p[4 + 2 * m]     = bwd_vars[4];
    p[4 + 2 * m + 1] = bwd_vars[5];
    p[2 + 4 * n + m] = bwd_vars[6];
    // Remaining suffix legs and their flybys.
    for j in 1..(n - m) {
        let leg = m + j;
        let base = 7 + 5 * (j - 1);
        p[4 + 2 * n + 2 * (leg - 1)]     = bwd_vars[base];
        p[4 + 2 * n + 2 * (leg - 1) + 1] = bwd_vars[base + 1];
        p[4 + 2 * leg]     = bwd_vars[base + 2];
        p[4 + 2 * leg + 1] = bwd_vars[base + 3];
        p[2 + 4 * n + leg] = bwd_vars[base + 4];
    }
    for (v, (lo, hi)) in p.iter_mut().zip(bounds) {
        *v = v.clamp(*lo, *hi);
    }
    p
}

/// Ceriotti-order flat index of level `lvl`'s TOF variable (level 0's
/// block is 7 vars with TOF at local index 4; level `j ≥ 1` blocks are 5
/// vars starting at `7 + 5(j-1)` with TOF at local index 2).
fn ceriotti_tof_slot(lvl: usize) -> usize {
    if lvl == 0 { 4 } else { 7 + 5 * (lvl - 1) + 2 }
}

/// The generic backfit stage: optimize the PREFIX (legs `0..m-1` in
/// Ceriotti order, plus the interface flyby's own (rp, β)) against a FIXED
/// backward-suffix handoff requirement — the suffix's sampled interface
/// epoch and required outgoing v∞ vector at split body `m`.
///
/// This is `galileo_backfit_legs012` generalized: the interface epoch is
/// pinned by construction (the LAST prefix leg's TOF is computed as the
/// remainder `t_suffix − dep_offset − Σ earlier TOFs`, infeasible outside
/// that leg's configured bounds — its chromosome slot is carried with
/// zero-width bounds and overwritten every evaluation), and the fitness is
/// the real physical objective of the stitched trajectory under a
/// ballistic suffix:
///
/// `fitness = departure_escape_dv(v∞_dep) + Σ prefix DSMs
///            + |flyby_turn(v∞_in, rp, β) − v∞_required|`
///
/// The mismatch term is 1:1 with real ΔV — it IS the residual impulse the
/// suffix's first leg would have to absorb at the handoff (validated by
/// the prototype: a 29 m/s seam collapsed into a 13.9 m/s DSM under the
/// joint evaluator). Seeded with the forward pruning survivors' prefixes
/// (missing rp/β dims midpoint-filled by the solver).
///
/// Returns the stitched full chromosome and its backfit fitness, or `None`
/// if the DE never found a feasible prefix (e.g. no remainder TOF inside
/// the leg's bounds for this suffix epoch).
#[allow(clippy::too_many_arguments)]
fn backfit_prefix_to_suffix(
    cfg: &MissionConfig,
    almanac: &Almanac,
    flyby_bodies: &[String],
    dep_jd_base: f64,
    m: usize,
    suffix_vars: &[f64],
    flat_bounds: &[(f64, f64)],
    fwd: &[PruningCandidate],
    de_seed: u64,
) -> Option<(Vec<f64>, f64)> {
    let opt = cfg.optimization.as_ref()?;
    let mga = opt.mga.as_ref()?;
    let n = n_legs(flyby_bodies);
    let iface_name = &flyby_bodies[m - 1];
    let iface_cat = body_models::TargetBody::by_name(iface_name)?;

    // Suffix handoff requirement (see suffix_level_bounds' level-0 layout).
    let t_suffix = suffix_vars[0];
    let (vr, tr, pr) = (suffix_vars[1], suffix_vars[2], suffix_vars[3]);
    let v_req = Vector3::new(
        vr * pr.cos() * tr.cos(),
        vr * pr.cos() * tr.sin(),
        vr * pr.sin(),
    );

    // Prefix bounds: Ceriotti levels 0..m-1 concatenated, the last prefix
    // leg's TOF slot collapsed to zero width (it is the epoch-pinning
    // remainder, not a search variable), then the interface flyby's own
    // (rp, β) bounds appended from the full chromosome's layout.
    let level_bounds = ceriotti_level_bounds(flat_bounds, n);
    let mut bounds: Vec<(f64, f64)> = level_bounds[..m].concat();
    let last_tof_slot = ceriotti_tof_slot(m - 1);
    let (leg_tof_lo, leg_tof_hi) = bounds[last_tof_slot];
    bounds[last_tof_slot] = (0.0, 0.0);
    let prefix_len = bounds.len();
    bounds.push(flat_bounds[4 + 2 * n + 2 * (m - 1)]);     // rp_norm_iface
    bounds.push(flat_bounds[4 + 2 * n + 2 * (m - 1) + 1]); // beta_iface

    // Epoch-pin + evaluate: returns the completed prefix vector alongside
    // its interface state, shared by the fitness closure and the final
    // stitching below so both use identical arithmetic.
    let pin_and_eval = |p: &[f64]| -> Option<(Vec<f64>, CeriottiPrefixEval)> {
        let mut pf = p[..prefix_len].to_vec();
        let sum_prior: f64 = (0..m - 1).map(|k| pf[ceriotti_tof_slot(k)]).sum();
        let tof_last = t_suffix - pf[0] - sum_prior;
        if tof_last < leg_tof_lo || tof_last > leg_tof_hi {
            return None;
        }
        pf[last_tof_slot] = tof_last;
        let pe = evaluate_ceriotti_prefix_full(&pf, m - 1, cfg, almanac, dep_jd_base, flyby_bodies)?;
        Some((pf, pe))
    };

    // Clamp the interface flyby's own periapsis to the configured floor and
    // add a graded penalty for the violation depth — the SAME clamp
    // `evaluate_chromosome_graded` applies to every ordinary flyby (Phase
    // 9v-viii). Without this, the DE could hit an exact target v_out by
    // requesting a physically-illegal (below-floor) turn — legal in this
    // function's own accounting but rejected/penalized by the real
    // evaluator later, producing a mismatch value that looks great here but
    // is not actually achievable. Returns (turned v∞, clamped rp_norm,
    // periapsis-floor penalty).
    let clamped_turn = |v_in: Vector3<f64>, rp_norm_raw: f64, beta: f64| -> (Vector3<f64>, f64, f64) {
        let rp_raw = rp_norm_raw * iface_cat.radius_m;
        let (rp_used, floor_penalty) = if rp_raw < mga.flyby_min_periapsis_m {
            let pen = CONSTRAINT_PENALTY_MS
                * ((mga.flyby_min_periapsis_m - rp_raw) / mga.flyby_min_periapsis_m).min(1.0);
            (mga.flyby_min_periapsis_m, pen)
        } else {
            (rp_raw, 0.0)
        };
        (flyby_turn(v_in, rp_used, beta, iface_cat.mu_m3s2), rp_used / iface_cat.radius_m, floor_penalty)
    };

    let fitness = |p: &[f64]| -> Option<f64> {
        let (pf, pe) = pin_and_eval(p)?;
        let (v_out, _, floor_penalty) = clamped_turn(pe.v_inf_in, p[prefix_len], p[prefix_len + 1]);
        let mismatch = (v_out - v_req).norm();
        // Pool-aware departure cost (Phase 14b), same as `phase2_fitness`.
        let dv_dep = crate::design::departure_onboard_cost_ms(cfg, &opt.departure_body, pf[1])?;
        let dsms = pe.cost - pf[1]; // prefix cost = dep_vinf + DSM sum
        Some(dv_dep + dsms + mismatch + pe.penalty_ms + floor_penalty)
    };

    // Seeds: forward pruning survivors' prefixes as-is — the solver
    // midpoint-fills the two missing (rp, β) dims and clamps the pinned
    // TOF slot to its zero-width bound (overwritten each eval anyway).
    let seeds: Vec<Vec<f64>> = fwd.iter().map(|f| f.vars.clone()).collect();

    let (result, _) = run_de_variant(
        mga, BACKFIT_POP, BACKFIT_GEN, de_seed,
        &bounds, &seeds, fitness, |_, _, _| {},
    );
    if !result.best_fitness.is_finite() {
        return None;
    }

    let (pf, pe) = pin_and_eval(&result.best_params)?;
    let beta_iface = result.best_params[prefix_len + 1];
    let (v_out, rp_norm_iface, floor_penalty) =
        clamped_turn(pe.v_inf_in, result.best_params[prefix_len], beta_iface);
    println!(
        "  Phase 9x backfit: split at {iface_name} — target |v∞_req|={:.1} m/s @ t={:.1} d: \
         fit={:.1} m/s (prefix DSMs {:.1}, handoff mismatch {:.1}{})",
        vr, t_suffix, result.best_fitness,
        pe.cost - pf[1], (v_out - v_req).norm(),
        if pe.penalty_ms > 0.0 || floor_penalty > 0.0 {
            format!(", constraint penalty {:.1}", pe.penalty_ms + floor_penalty)
        } else {
            String::new()
        },
    );

    let stitched = stitch_chromosome(n, m, &pf, suffix_vars, rp_norm_iface, beta_iface, flat_bounds);
    Some((stitched, result.best_fitness))
}

/// Run the backward pass at every split body and match against the forward
/// per-level survivor snapshots, returning up to `pruning.n_seeds` stitched
/// full chromosomes ranked by match score (prefix cost + suffix cost +
/// |v∞| magnitude mismatch + turn-angle deficit above the periapsis floor +
/// epoch-gap penalty — the physically-meaningful m/s scale
/// `galileo_backfit_legs012` validated), PLUS the backfit-stage stitches
/// (see [`backfit_prefix_to_suffix`] — always included, never truncated
/// away: they are the pipeline's load-bearing output; the sample-matched
/// stitches are the cheap complement).
///
/// Second return value: the single best stitched chromosome by its OWN
/// real `phase2_fitness` (departure + DSMs + arrival, no penalty), if any
/// stitch was feasible under the real objective. Found necessary
/// seeding alone is not enough — a genuinely good stitched
/// candidate can still lose inside Phase 2's population dynamics (a live
/// full-budget VEEGA run produced a winner in a completely different,
/// non-resonant basin while the best stitched candidate, with handoff
/// mismatch driven to ~0, scored better but never won the population).
/// The caller compares this against Phase 2's own DE winner directly
/// (`min()`, can only help) instead of hoping the population preserves it.
fn run_mga_bidirectional_stitches(
    cfg: &MissionConfig,
    almanac: &Almanac,
    flyby_bodies: &[String],
    dep_jd_base: f64,
    pruning: &PruningConfigToml,
    flat_bounds: &[(f64, f64)],
    fwd_levels: &[Vec<PruningCandidate>],
) -> (Vec<Vec<f64>>, Option<(Vec<f64>, f64)>) {
    let n = n_legs(flyby_bodies);
    let Some(opt) = cfg.optimization.as_ref() else { return (Vec::new(), None) };
    let Some(mga) = opt.mga.as_ref() else { return (Vec::new(), None) };

    let mut matches: Vec<(f64, Vec<f64>)> = Vec::new();
    let mut backfit_stitched: Vec<Vec<f64>> = Vec::new();
    let mut best_real: Option<(Vec<f64>, f64)> = None;

    for m in 1..n {
        // Forward survivors at level m-1: prefixes ending with arrival at body m.
        let Some(fwd) = fwd_levels.get(m - 1) else { continue };
        if fwd.is_empty() { continue; }

        let iface_name = &flyby_bodies[m - 1];
        let Some(cat) = body_models::TargetBody::by_name(iface_name) else { continue };

        // Prefix interface evals, computed once per split: they anchor the
        // backward pass's epoch sampling AND drive the matching loop below.
        let prefix_evals: Vec<(&PruningCandidate, CeriottiPrefixEval)> = fwd.iter()
            .filter_map(|f| {
                evaluate_ceriotti_prefix_full(&f.vars, m - 1, cfg, almanac, dep_jd_base, flyby_bodies)
                    .map(|pe| (f, pe))
            })
            .collect();
        if prefix_evals.is_empty() { continue; }
        // Anchors: (interface epoch offset, delivered v∞ vector) pairs from
        // the SAME prefix, kept together — see BWD_EPOCH_ANCHOR_FRACTION's
        // doc comment for why the two must be drawn jointly, not independently.
        let anchors: Vec<(f64, Vector3<f64>)> = prefix_evals.iter()
            .map(|(_, pe)| (pe.arr_jd - dep_jd_base, pe.v_inf_in))
            .collect();

        // Backward suffix pruning for legs m..n-1 (distinctly seeded per
        // split so results don't silently correlate across splits).
        let suffix_bounds = suffix_level_bounds(flat_bounds, n, m);
        let bwd_cfg = PruningConfig {
            samples_level0: pruning.samples_level0,
            children_per_survivor: pruning.children_per_survivor,
            max_survivors_per_level: pruning.max_survivors_per_level,
            threshold_factor: pruning.threshold_factor,
            seed: mga_pruning_seed(cfg).wrapping_add(7_777u64.wrapping_mul(m as u64)),
        };
        // Resonance bias applies inside the suffix too (a same-body return
        // leg there is just as geometrically special): the suffix's
        // "departure" body is the interface body, its flyby list is the
        // tail of the full list, and its level blocks share the forward
        // blocks' local TOF/n_rev indices by construction.
        let bias_map = resonance_bias_windows(
            n - m, &flyby_bodies[m..], iface_name, &opt.target_body, &suffix_bounds,
        );
        let bwd_snapshots = run_incremental_pruning_all_levels(
            &suffix_bounds,
            |lvl, prefix| evaluate_suffix_prefix(prefix, lvl, m, cfg, almanac, dep_jd_base, flyby_bodies),
            &bwd_cfg,
            |lvl, b, next| {
                let mut vars = sample_box_uniform(b, next);
                if let Some((tof_idx, nrev_idx, windows)) = bias_map.get(&lvl) {
                    if next() < resonance_bias_fraction() {
                        let w = &windows[((next() * windows.len() as f64) as usize).min(windows.len() - 1)];
                        vars[*tof_idx] = w.0 + next() * (w.1 - w.0);
                        vars[*nrev_idx] = (w.2.min(2) as f64 + 0.5).min(2.999);
                    }
                }
                // Joint epoch + v∞-vector anchoring (see
                // BWD_EPOCH_ANCHOR_FRACTION): draw a fraction of interface
                // states from around a real forward prefix's actual
                // (epoch, v∞) delivery, so matching does not depend on
                // chance coincidence over a years-wide, ~20 km/s-wide box.
                if lvl == 0 && next() < BWD_EPOCH_ANCHOR_FRACTION {
                    let (a_epoch, a_vinf) = anchors[
                        ((next() * anchors.len() as f64) as usize)
                            .min(anchors.len() - 1)];
                    vars[0] = (a_epoch - MATCH_EPOCH_TOL_DAYS
                        + next() * 2.0 * MATCH_EPOCH_TOL_DAYS)
                        .clamp(b[0].0, b[0].1);
                    let mag = a_vinf.norm();
                    if mag > 1.0 {
                        let jit = 1.0 + (2.0 * next() - 1.0) * BWD_VINF_ANCHOR_JITTER_FRAC;
                        let theta0 = a_vinf.y.atan2(a_vinf.x);
                        let phi0   = (a_vinf.z / mag).clamp(-1.0, 1.0).asin();
                        let dtheta = (2.0 * next() - 1.0) * BWD_VINF_ANCHOR_ANGLE_JITTER_RAD;
                        let dphi   = (2.0 * next() - 1.0) * BWD_VINF_ANCHOR_ANGLE_JITTER_RAD;
                        vars[1] = (mag * jit).clamp(b[1].0, b[1].1);
                        vars[2] = (theta0 + dtheta).rem_euclid(2.0 * PI);
                        vars[3] = (phi0 + dphi).clamp(b[3].0, b[3].1);
                    }
                }
                vars
            },
        );
        // Backward candidate pool: epoch-anchored pruning survivors ...
        let mut bwd_pool: Vec<(Vec<f64>, f64)> = bwd_snapshots.last()
            .map(|s| s.iter().map(|c| (c.vars.clone(), c.partial_cost)).collect())
            .unwrap_or_default();

        // ... plus a short suffix-refinement DE seeded with them (see
        // BWD_REFINE_POP's doc comment for why sampling alone is not
        // enough). Runs even when pruning found nothing — the prototype's
        // leg-3 DE was unseeded and still found the ballistic solution.
        //
        // Critical: the refinement DE minimizes suffix DSM ALONE, with no
        // notion of matching — left free over the full [0, BWD_VINF_MAX_MS]
        // handoff box, it drifts away from any anchored (epoch, v∞) over
        // its 200 generations even when SEEDED there, defeating the
        // anchoring above (found: a smoke run's best-matching
        // suffix still scored a ~19,000 m/s mismatch after refinement).
        // Fix: narrow levels 0's handoff-v∞ bounds (indices 1..4) to the
        // union of the anchor set (± the same jitter margins used to draw
        // them), so refinement can only explore handoff states a real
        // forward prefix actually delivers. Epoch (index 0) and every
        // other variable keep their full physical bounds unchanged.
        let mut flat_suffix_bounds: Vec<(f64, f64)> = suffix_bounds.concat();
        {
            let (mut vlo, mut vhi) = (f64::MAX, f64::MIN);
            let (mut tlo, mut thi) = (f64::MAX, f64::MIN);
            let (mut plo, mut phi_hi) = (f64::MAX, f64::MIN);
            for (_, v) in &anchors {
                let mag = v.norm();
                if mag < 1.0 { continue; }
                let theta0 = v.y.atan2(v.x);
                let phi0   = (v.z / mag).clamp(-1.0, 1.0).asin();
                vlo = vlo.min(mag); vhi = vhi.max(mag);
                tlo = tlo.min(theta0); thi = thi.max(theta0);
                plo = plo.min(phi0); phi_hi = phi_hi.max(phi0);
            }
            if vlo.is_finite() {
                let pad_v = BWD_VINF_ANCHOR_JITTER_FRAC * vhi.max(1.0);
                let pad_a = BWD_VINF_ANCHOR_ANGLE_JITTER_RAD;
                flat_suffix_bounds[1] = ((vlo - pad_v).max(flat_suffix_bounds[1].0), (vhi + pad_v).min(flat_suffix_bounds[1].1));
                flat_suffix_bounds[2] = ((tlo - pad_a).max(0.0), (thi + pad_a).min(2.0 * PI));
                flat_suffix_bounds[3] = ((plo - pad_a).max(flat_suffix_bounds[3].0), (phi_hi + pad_a).min(flat_suffix_bounds[3].1));
            }
        }
        let last_lvl = suffix_bounds.len() - 1;
        let de_seeds: Vec<Vec<f64>> = bwd_pool.iter().map(|(v, _)| v.clone()).collect();
        let (_, mut de_pop) = run_de_variant(
            mga, BWD_REFINE_POP, BWD_REFINE_GEN,
            bwd_cfg.seed.wrapping_add(500_009),
            &flat_suffix_bounds, &de_seeds,
            |p| evaluate_suffix_prefix(p, last_lvl, m, cfg, almanac, dep_jd_base, flyby_bodies),
            |_, _, _| {},
        );
        de_pop.retain(|(_, fit)| fit.is_finite());
        de_pop.sort_by(|a, b| a.1.total_cmp(&b.1));
        de_pop.truncate(pruning.max_survivors_per_level);
        bwd_pool.extend(de_pop);

        if bwd_pool.is_empty() {
            println!("  Phase 9x bidir: split at {iface_name} — backward pass found no feasible suffixes");
            continue;
        }
        bwd_pool.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Backfit stage (the load-bearing step): pick up to
        // BACKFIT_TARGETS_PER_SPLIT epoch-distinct best suffixes and
        // optimize a prefix DIRECTLY against each one's handoff
        // requirement, mismatch inside the fitness.
        let mut target_epochs: Vec<f64> = Vec::new();
        for (idx, (sv, _)) in bwd_pool.iter().enumerate() {
            if target_epochs.len() >= BACKFIT_TARGETS_PER_SPLIT { break; }
            if target_epochs.iter().any(|e| (e - sv[0]).abs() < BACKFIT_TARGET_MIN_EPOCH_SEP_DAYS) {
                continue;
            }
            target_epochs.push(sv[0]);
            if let Some((stitched, _fit)) = backfit_prefix_to_suffix(
                cfg, almanac, flyby_bodies, dep_jd_base, m, sv, flat_bounds, fwd,
                mga_pruning_seed(cfg)
                    .wrapping_add(31_337u64.wrapping_mul(m as u64))
                    .wrapping_add(idx as u64),
            ) {
                if let Some(p2fit) = phase2_fitness(&stitched, cfg, almanac, dep_jd_base, flyby_bodies) {
                    println!("    → stitched full-chromosome Phase-2 fitness = {p2fit:.1} m/s");
                    if best_real.as_ref().is_none_or(|(_, f)| p2fit < *f) {
                        best_real = Some((stitched.clone(), p2fit));
                    }
                }
                backfit_stitched.push(stitched);
            }
        }

        // Interface matching: prefix arrival (epoch, v∞_in) × suffix
        // requirement (epoch, v∞_out).
        let mut split_best: Option<f64> = None;
        let mut split_count = 0usize;
        for (f, pe) in &prefix_evals {
            let v_in = pe.v_inf_in;
            let v_in_mag = v_in.norm();
            for (bv, bcost) in &bwd_pool {
                let bwd_jd = dep_jd_base + bv[0];
                let dt_days = (pe.arr_jd - bwd_jd).abs();
                if dt_days > MATCH_EPOCH_TOL_DAYS { continue; }

                let vb = bv[1];
                let (st, sp) = (bv[2], bv[3]);
                let v_out = Vector3::new(
                    vb * sp.cos() * st.cos(),
                    vb * sp.cos() * st.sin(),
                    vb * sp.sin(),
                );
                // Magnitude mismatch: the one thing an unpowered flyby can
                // never fix (energy is conserved across the turn).
                let mag_mismatch = (v_in_mag - vb).abs();

                // Required turn vs. maximum achievable above the periapsis
                // floor; excess turn priced as the residual chord an
                // at-the-floor flyby leaves un-turned.
                let turn_deficit_ms = if v_in_mag * vb < 1.0 {
                    0.0
                } else {
                    let d_req = (v_in.dot(&v_out) / (v_in_mag * vb)).clamp(-1.0, 1.0).acos();
                    let e_min = 1.0 + mga.flyby_min_periapsis_m * v_in_mag * v_in_mag / cat.mu_m3s2;
                    let d_max = 2.0 * (1.0 / e_min).asin();
                    if d_req > d_max {
                        2.0 * v_in_mag * ((d_req - d_max) / 2.0).sin()
                    } else {
                        0.0
                    }
                };

                let score = f.partial_cost + bcost
                    + mag_mismatch + turn_deficit_ms
                    + dt_days * MATCH_EPOCH_PENALTY_MS_PER_DAY;

                let (rp_m_iface, beta_iface) = invert_flyby_turn(v_in, v_out, cat.mu_m3s2);
                let rp_norm_iface = if rp_m_iface.is_finite() {
                    rp_m_iface / cat.radius_m
                } else {
                    // No meaningful turn required — park at the box's far
                    // edge (a distant, near-straight pass); the clamp in
                    // stitch_chromosome bounds it anyway.
                    f64::MAX
                };
                let stitched = stitch_chromosome(
                    n, m, &f.vars, bv, rp_norm_iface, beta_iface, flat_bounds,
                );
                matches.push((score, stitched));
                split_count += 1;
                split_best = Some(split_best.map_or(score, |s: f64| s.min(score)));
            }
        }
        match split_best {
            Some(s) => println!(
                "  Phase 9x bidir: split at {iface_name} — {} suffix candidate(s) (best suffix DSM {:.1} m/s), {} epoch-compatible match(es), best match score {:.1} m/s",
                bwd_pool.len(), (bwd_pool[0].1 - BWD_COST_FLOOR_MS).max(0.0), split_count, s,
            ),
            None => println!(
                "  Phase 9x bidir: split at {iface_name} — {} suffix candidate(s) (best suffix DSM {:.1} m/s), but no forward prefix within ±{MATCH_EPOCH_TOL_DAYS:.0} d of any suffix epoch",
                bwd_pool.len(), (bwd_pool[0].1 - BWD_COST_FLOOR_MS).max(0.0),
            ),
        }
    }

    matches.sort_by(|a, b| a.0.total_cmp(&b.0));
    matches.truncate(pruning.n_seeds);
    // Backfit stitches first (never truncated — they are the pipeline's
    // real output), then the best sample-matched stitches as complement.
    backfit_stitched.extend(matches.into_iter().map(|(_, p)| p));
    (backfit_stitched, best_real)
}

/// Run Phase 9x incremental pruning and return up to `pruning.n_seeds`
/// full-chromosome candidates (this module's `params` layout, ready for
/// `run_de_variant`'s `seeds` argument), sorted by ascending partial cost,
/// alongside the single best bidirectional-stitched chromosome by its own
/// real fitness (see [`run_mga_bidirectional_stitches`] — `None` when
/// bidirectional stitching is off or found nothing feasible).
///
/// Returns `None` (with a printed warning, not an error — pruning is an
/// optional pre-search) if every level-0 sample was infeasible or bounds
/// could not be built; the caller falls back to running the DE without
/// pruning-derived seeds, exactly as before this phase existed.
/// Per-split remaining-cost table for the branch-and-bound pruning
/// extension: the cheapest sampled suffix cost (DSM sum +
/// penalties, floor constant removed), binned over (interface epoch offset,
/// interface |v∞|). A sampled minimum OVERESTIMATES the true minimum, so
/// lookups discount by `1 − safety_frac` before use — see
/// `PruningBoundToml::safety_frac`. This is GASP's backward-propagated
/// pruning criterion (Myatt et al. 2004; Izzo, Becerra, Myatt, Nasuto &
/// Bishop 2007, J. Global Optimization 38(2):283–296) adapted to this
/// module's sampled Ceriotti-style pruning: forward candidates are pruned
/// on A*'s `f = g + h` instead of `g` alone.
struct SuffixBoundTable {
    epoch_lo: f64,
    epoch_hi: f64,
    vinf_lo: f64,
    vinf_hi: f64,
    epoch_bins: usize,
    vinf_bins: usize,
    /// Min sampled suffix cost per bin [m/s]; `INFINITY` = no sample landed.
    bins: Vec<f64>,
    /// Min over every bin — the weakest admissible nonzero fallback for a
    /// lookup landing in an empty bin (never discard on absence of data).
    global_min: f64,
}

impl SuffixBoundTable {
    fn bin_idx(&self, epoch_off_days: f64, vinf_ms: f64) -> Option<usize> {
        if !(self.epoch_lo..=self.epoch_hi).contains(&epoch_off_days)
            || !(self.vinf_lo..=self.vinf_hi).contains(&vinf_ms)
        {
            return None;
        }
        let e = (((epoch_off_days - self.epoch_lo) / (self.epoch_hi - self.epoch_lo))
            * self.epoch_bins as f64) as usize;
        let v = (((vinf_ms - self.vinf_lo) / (self.vinf_hi - self.vinf_lo))
            * self.vinf_bins as f64) as usize;
        Some(e.min(self.epoch_bins - 1) * self.vinf_bins + v.min(self.vinf_bins - 1))
    }

    /// Discounted remaining-cost bound for a forward prefix arriving at
    /// this table's split body. Empty bin (or out-of-table state) falls
    /// back to the table's global minimum — weakest nonzero bound, never a
    /// discard-by-default.
    fn lookup(&self, epoch_off_days: f64, vinf_ms: f64, safety_frac: f64) -> f64 {
        let raw = self
            .bin_idx(epoch_off_days, vinf_ms)
            .map(|i| self.bins[i])
            .filter(|c| c.is_finite())
            .unwrap_or(self.global_min);
        if raw.is_finite() {
            raw * (1.0 - safety_frac)
        } else {
            0.0
        }
    }
}

/// Build one [`SuffixBoundTable`] per intermediate split (`tables[m]` for
/// `m` in `1..n`; indices 0 and `n` stay `None`). Samples full-length
/// suffixes uniformly over each split's own bounds (with the same
/// resonance-window bias the forward sampler uses — tighter minima exactly
/// on the legs where it matters), evaluated by the same
/// `evaluate_suffix_prefix` the bidirectional stage trusts.
fn build_suffix_bound_tables(
    cfg: &MissionConfig,
    almanac: &Almanac,
    flyby_bodies: &[String],
    dep_jd_base: f64,
    bound: &PruningBoundToml,
    flat_bounds: &[(f64, f64)],
) -> Vec<Option<SuffixBoundTable>> {
    let n = n_legs(flyby_bodies);
    let mut tables: Vec<Option<SuffixBoundTable>> = (0..=n).map(|_| None).collect();
    let Some(opt) = cfg.optimization.as_ref() else { return tables };

    for m in 1..n {
        let suffix_bounds = suffix_level_bounds(flat_bounds, n, m);
        let flat_suffix: Vec<(f64, f64)> = suffix_bounds.concat();
        let last_lvl = suffix_bounds.len() - 1;
        let iface_name = &flyby_bodies[m - 1];
        let bias_map = resonance_bias_windows(
            n - m, &flyby_bodies[m..], iface_name, &opt.target_body, &suffix_bounds,
        );
        // Flat-vector offset of each suffix level's block, so the per-level
        // bias indices map onto the flat sample vector.
        let block_offset = |lvl: usize| if lvl == 0 { 0 } else { 7 + 5 * (lvl - 1) };

        let mut rng = SplitMix64::new(mga_pruning_seed(cfg).wrapping_add(24_601u64.wrapping_mul(m as u64)));
        let (epoch_lo, epoch_hi) = flat_suffix[0];
        let (vinf_lo, vinf_hi) = flat_suffix[1];
        let mut table = SuffixBoundTable {
            epoch_lo,
            epoch_hi,
            vinf_lo,
            vinf_hi,
            epoch_bins: bound.epoch_bins,
            vinf_bins: bound.vinf_bins,
            bins: vec![f64::INFINITY; bound.epoch_bins * bound.vinf_bins],
            global_min: f64::INFINITY,
        };
        let mut feasible = 0usize;
        for _ in 0..bound.samples_per_split {
            let mut vars: Vec<f64> = flat_suffix
                .iter()
                .map(|&(lo, hi)| lo + rng.next_f64() * (hi - lo))
                .collect();
            for (lvl, (tof_idx, nrev_idx, windows)) in &bias_map {
                if rng.next_f64() < resonance_bias_fraction() {
                    let w = &windows[((rng.next_f64() * windows.len() as f64) as usize).min(windows.len() - 1)];
                    let off = block_offset(*lvl);
                    vars[off + tof_idx] = w.0 + rng.next_f64() * (w.1 - w.0);
                    vars[off + nrev_idx] = (w.2.min(2) as f64 + 0.5).min(2.999);
                }
            }
            if let Some(cost) = evaluate_suffix_prefix(&vars, last_lvl, m, cfg, almanac, dep_jd_base, flyby_bodies) {
                let cost = (cost - BWD_COST_FLOOR_MS).max(0.0);
                feasible += 1;
                if let Some(i) = table.bin_idx(vars[0], vars[1]) {
                    if cost < table.bins[i] {
                        table.bins[i] = cost;
                    }
                }
                if cost < table.global_min {
                    table.global_min = cost;
                }
            }
        }
        println!(
            "  Phase 9x bound: split at {iface_name} — {feasible}/{} feasible suffix samples, global min remaining cost {:.1} m/s",
            bound.samples_per_split,
            if table.global_min.is_finite() { table.global_min } else { f64::NAN },
        );
        if table.global_min.is_finite() {
            tables[m] = Some(table);
        }
    }
    tables
}

/// Shared state for surrogate-assisted pruning sampling:
/// per-level training records `(child block vars → added cost vs. parent)`
/// harvested by the evaluate closure, per-level fitted RBF models refit
/// lazily as data accumulates, and a `(level, prefix-bits) → cumulative
/// cost` map so a child's ADDED cost can be recovered without re-evaluating
/// its parent. Lives behind a `RefCell` because the pruning engine's
/// sampler and evaluator are two separate closures called in strict
/// alternation (never nested).
struct SurrogatePruningState {
    cost_map: HashMap<(usize, Vec<u64>), f64>,
    train: Vec<Vec<(Vec<f64>, f64)>>,
    models: Vec<Option<RbfSurrogate>>,
    fitted_at: Vec<usize>,
}

fn var_bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Ceriotti level-block length: level 0 carries the 4 departure vars +
/// (tof, eta, n_rev); levels 1+ carry (rp, beta, tof, eta, n_rev).
fn ceriotti_block_len(level_idx: usize) -> usize {
    if level_idx == 0 { 7 } else { 5 }
}

fn run_mga_pruning_seeds(
    cfg: &MissionConfig,
    almanac: &Almanac,
    flyby_bodies: &[String],
    dep_jd_base: f64,
    pruning: &PruningConfigToml,
    flat_bounds: &[(f64, f64)],
) -> Option<(Vec<Vec<f64>>, Option<(Vec<f64>, f64)>)> {
    // `flat_bounds` is caller-supplied (resonance-family
    // branching) rather than rebuilt here, so a family branch's restricted
    // leg-TOF bounds propagate into the pruning level bounds — and thereby
    // into the resonance-bias windows, which are clipped to them.
    let n = n_legs(flyby_bodies);
    let opt = cfg.optimization.as_ref()?;
    let level_bounds = ceriotti_level_bounds(flat_bounds, n);

    let prune_cfg = PruningConfig {
        samples_level0: pruning.samples_level0,
        children_per_survivor: pruning.children_per_survivor,
        max_survivors_per_level: pruning.max_survivors_per_level,
        threshold_factor: pruning.threshold_factor,
        seed: mga_pruning_seed(cfg),
    };

    println!(
        "  Phase 9x pruning: {} levels, samples_level0={}, children/survivor={}, max_survivors={}, threshold×{:.1}",
        level_bounds.len(), prune_cfg.samples_level0, prune_cfg.children_per_survivor,
        prune_cfg.max_survivors_per_level, prune_cfg.threshold_factor,
    );

    // Resonance-aware sampling (Phase 9x-iv): bias any
    // same-body return leg's TOF sampling toward integer multiples of that
    // body's real period. Empty map (no return legs in this sequence) makes
    // this byte-identical to plain uniform sampling — see resonance_bias_windows.
    let bias_map = resonance_bias_windows(
        n, flyby_bodies, &opt.departure_body, &opt.target_body, &level_bounds,
    );

    // Branch-and-bound extension: with `[...pruning.bound]`
    // configured, backward suffix tables are built up front and the forward
    // pass prunes on `f = g + h` (partial cost + discounted remaining-cost
    // bound at the prefix's interface state) instead of `g` alone. Absent
    // config skips the tables entirely — byte-identical prior behaviour.
    let bound_tables: Option<(&PruningBoundToml, Vec<Option<SuffixBoundTable>>)> =
        pruning.bound.as_ref().map(|b| {
            (b, build_suffix_bound_tables(cfg, almanac, flyby_bodies, dep_jd_base, b, flat_bounds))
        });

    // Surrogate-assisted sampling: evaluate records per-level
    // (child vars → added cost) training pairs; the sampler oversamples
    // candidate children and keeps the best-predicted one once a level's
    // model has enough data. Entirely inside these two closures — the
    // engine is unchanged. NOTE: enabling this changes the sampler's RNG
    // draw pattern (extra draws per child slot), so a surrogate run is not
    // draw-for-draw comparable with a non-surrogate run by construction.
    let surro_cfg: Option<PruningSurrogateToml> = if pruning.surrogate.enabled { Some(pruning.surrogate) } else { None };
    let surro: Option<RefCell<SurrogatePruningState>> = surro_cfg.map(|_| {
        RefCell::new(SurrogatePruningState {
            cost_map: HashMap::new(),
            train: vec![Vec::new(); level_bounds.len()],
            models: (0..level_bounds.len()).map(|_| None).collect(),
            fitted_at: vec![0; level_bounds.len()],
        })
    });
    if let Some(sc) = &surro_cfg {
        println!(
            "  Phase 9x surrogate: enabled (oversample={}, min_fit_samples={}, length_scale={:.2})",
            sc.oversample_factor, sc.min_fit_samples, sc.length_scale,
        );
    }

    let fwd_snapshots = run_incremental_pruning_all_levels(
        &level_bounds,
        |level_idx, prefix| {
            // g: the real cumulative partial cost, exactly as before.
            let (g, iface) = match &bound_tables {
                None => (evaluate_ceriotti_prefix(prefix, level_idx, cfg, almanac, dep_jd_base, flyby_bodies)?, None),
                Some(_) => {
                    let ev = evaluate_ceriotti_prefix_full(prefix, level_idx, cfg, almanac, dep_jd_base, flyby_bodies)?;
                    (ev.cost, Some((ev.arr_jd, ev.v_inf_in)))
                }
            };
            // Surrogate training record: the child block's ADDED cost
            // (g − parent's g, recovered from the cost map) — always the
            // real g, never the bound-augmented f.
            if let Some(s) = &surro {
                let mut st = s.borrow_mut();
                let block = ceriotti_block_len(level_idx);
                let child_local = prefix[prefix.len() - block..].to_vec();
                let delta = if level_idx == 0 {
                    g
                } else {
                    let parent_bits = var_bits(&prefix[..prefix.len() - block]);
                    st.cost_map
                        .get(&(level_idx - 1, parent_bits))
                        .map(|pc| (g - pc).max(0.0))
                        .unwrap_or(g)
                };
                st.cost_map.insert((level_idx, var_bits(prefix)), g);
                st.train[level_idx].push((child_local, delta));
            }
            // h: discounted remaining-cost bound at the interface this
            // prefix arrives at (split m = level_idx + 1); zero at the last
            // level (no table — only the arrival term remains).
            let h = match (&bound_tables, iface) {
                (Some((bcfg, tables)), Some((arr_jd, v_inf_in))) => tables
                    .get(level_idx + 1)
                    .and_then(|t| t.as_ref())
                    .map(|t| t.lookup(arr_jd - dep_jd_base, v_inf_in.norm(), bcfg.safety_frac))
                    .unwrap_or(0.0),
                _ => 0.0,
            };
            Some(g + h)
        },
        &prune_cfg,
        |level_idx, bounds, next| {
            let mut base_draw = |next: &mut dyn FnMut() -> f64| {
                let mut vars = sample_box_uniform(bounds, next);
                if let Some((tof_idx, nrev_idx, windows)) = bias_map.get(&level_idx) {
                    if next() < resonance_bias_fraction() {
                        let w = &windows[((next() * windows.len() as f64) as usize).min(windows.len() - 1)];
                        vars[*tof_idx] = w.0 + next() * (w.1 - w.0);
                        // Set the leg's n_rev gene to match the window's
                        // resonance order (spacecraft sweeps ~N revolutions on
                        // an N-period return) — +0.5 so the floor decode lands
                        // exactly on N; clamp keeps it inside the gene's box.
                        vars[*nrev_idx] = (w.2.min(2) as f64 + 0.5).min(2.999);
                    }
                }
                vars
            };
            match (&surro, &surro_cfg) {
                (Some(s), Some(sc)) => {
                    // Lazy refit: once a level has enough records AND has
                    // grown ≥50% since the last fit, refit its model.
                    {
                        let mut st = s.borrow_mut();
                        let n_rec = st.train[level_idx].len();
                        if n_rec >= sc.min_fit_samples && n_rec * 2 >= st.fitted_at[level_idx] * 3 {
                            let (xs, fs): (Vec<Vec<f64>>, Vec<f64>) =
                                st.train[level_idx].iter().cloned().unzip();
                            st.models[level_idx] =
                                RbfSurrogate::fit(bounds, &xs, &fs, sc.length_scale, 1e-6, 400);
                            st.fitted_at[level_idx] = n_rec;
                        }
                    }
                    let st = s.borrow();
                    match &st.models[level_idx] {
                        Some(model) => (0..sc.oversample_factor.max(2))
                            .map(|_| base_draw(next))
                            .min_by(|a, b| {
                                model.predict(bounds, a).total_cmp(&model.predict(bounds, b))
                            })
                            .expect("oversample_factor >= 2"),
                        None => base_draw(next),
                    }
                }
                _ => base_draw(next),
            }
        },
    );

    // Forward seeds: the LAST level's survivors, exactly as before the
    // bidirectional extension existed.
    let mut seeds: Vec<Vec<f64>> = Vec::new();
    match fwd_snapshots.last() {
        Some(survivors) if !survivors.is_empty() => {
            println!(
                "  Phase 9x pruning: {} survivors, best partial cost (v∞ + DSM sum) = {:.1} m/s",
                survivors.len(), survivors[0].partial_cost,
            );
            seeds.extend(survivors.iter()
                .take(pruning.n_seeds)
                .map(|c| scatter_ceriotti_to_params(n, &c.vars)));
        }
        _ => {
            // The forward pass dead-ending at some level is NOT fatal for
            // the bidirectional extension: its earlier-level snapshots can
            // still match against backward suffixes (that is exactly the
            // rigid-downstream-constraint case meet-in-the-middle exists
            // for), so only warn here and let the stitcher try.
            eprintln!("  Warning: Phase 9x forward pruning found zero full-length survivors.");
        }
    }

    // Bidirectional extension: backward suffixes + interface
    // matching + the backfit stage, stitched into additional full-chromosome
    // seeds. `best_stitched` is the single best by its OWN real fitness —
    // propagated up so the caller can compare it directly against Phase 2's
    // DE winner instead of only hoping the population preserves it (see
    // run_mga_bidirectional_stitches's doc comment for why that hope alone
    // isn't reliable).
    let mut best_stitched: Option<(Vec<f64>, f64)> = None;
    if pruning.bidirectional && n >= 2 {
        let (stitched, best) = run_mga_bidirectional_stitches(
            cfg, almanac, flyby_bodies, dep_jd_base, pruning, flat_bounds, &fwd_snapshots,
        );
        best_stitched = best;
        if !stitched.is_empty() {
            // Diagnostic: the best stitched chromosome's REAL Phase-2
            // fitness (total ΔV + graded penalties), so runs show at a
            // glance whether stitching produced a genuinely competitive
            // seed or just a well-matched-but-expensive one.
            if let Some(fit) = phase2_fitness(&stitched[0], cfg, almanac, dep_jd_base, flyby_bodies) {
                println!(
                    "  Phase 9x bidir: {} stitched seed(s) injected; best stitched Phase-2 fitness = {:.1} m/s",
                    stitched.len(), fit,
                );
            }
            seeds.extend(stitched);
        }
    }

    if seeds.is_empty() {
        eprintln!("  Warning: Phase 9x pruning found zero survivors — falling back to unseeded DE restarts.");
        return None;
    }
    Some((seeds, best_stitched))
}

/// Base RNG seed for the pruning pre-search, derived from `de_seed` (not
/// hardcoded) so a full seed-sweep of one config sweeps pruning too.
fn mga_pruning_seed(cfg: &MissionConfig) -> u64 {
    cfg.optimization.as_ref()
        .and_then(|o| o.mga.as_ref())
        .map(|m| m.de_seed.wrapping_add(900_001))
        .unwrap_or(900_001)
}

/// Inner MGA optimizer for a single fixed body sequence.
///
/// One complete search pass over the given chromosome bounds: Phase 1
/// (DE-only multi-restart exploration), Phase 9x pruning/bidirectional
/// seeding, Phase 2 (DE or MBH), the stitched-candidate min() guard, and
/// the eta+L-BFGS polish loop. Extracted from `run_mga_fixed_sequence`
/// so the resonance-family branching wrapper can run it once
/// per family branch with that branch's restricted bounds — see
/// [`resonance_family_branches`]. Returns `(best_params, best_fitness,
/// phase1_history, phase2_history, phase1_param_history, phase2_param_history)`;
/// the winner is already polished (the polish loop's own incremental gains
/// are not appended to the returned histories, matching the pre-existing
/// fitness-history behaviour — the final `best_params`/`best_fitness` can be
/// slightly better than the last history entry).
fn run_search_over_bounds<F>(
    cfg: &MissionConfig,
    almanac: &Almanac,
    flyby_bodies: &[String],
    dep_jd_base: f64,
    bounds: &[(f64, f64)],
    external_seed: Option<&[f64]>,
    on_step: &mut F,
) -> Result<(Vec<f64>, f64, Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>), String>
where
    F: FnMut(usize, u8, f64, &[f64], &[MgaLegStepInfo]),
{
    let opt = cfg.optimization.as_ref()
        .ok_or("optimization config is required for MGA")?;
    let mga = opt.mga.as_ref()
        .ok_or("optimization.mga is required when method = \"MGA\"")?;
    let n = n_legs(flyby_bodies);

    let pop = mga.de_population_size;
    let gen = mga.de_generations;

    // ── Phase 1: minimise sum(DSM ΔVs) with multi-start ─────────────────────
    // Each restart's final population contributes its top ELITES_PER_RESTART
    // individuals to Phase 2's seed pool, so Phase 2 can exploit EVERY basin
    // Phase 1 found — not just the single best restart's (the old ±20%
    // bound-narrowing permanently committed Phase 2 to one basin).

    const ELITES_PER_RESTART: usize = 8;

    let phase1_gen = (gen as f64 * 0.6).ceil() as usize;
    let mut phase1_history: Vec<f64> = Vec::new();
    let mut phase1_param_history: Vec<Vec<f64>> = Vec::new();
    let mut elite_seeds: Vec<Vec<f64>> = Vec::new();

    // External seed injection (`MGA_SEED_CHROMOSOME`): a
    // previous run's winner (or any hand-built chromosome) as an ADDITIONAL
    // elite seed — never a replacement for the search's own exploration.
    // Added once per branch, ahead of Phase 1/pruning, so it participates in
    // every downstream stage (Phase 1 restarts under DE still explore
    // independently; Phase 2 gets it directly as one more chain/population
    // member). DE/SHADE/MBH all clamp seeds into `bounds` themselves, so an
    // out-of-window value (e.g. seeding a resonance-family branch with a
    // chromosome from a different family) degrades gracefully to that
    // branch's boundary rather than erroring — exactly the behaviour wanted
    // for "does biasing toward a known-good basin help this branch converge
    // further" experiments.
    if let Some(seed) = external_seed {
        elite_seeds.push(seed.to_vec());
    }

    // Under MBH, Phase 1's multi-restart DSM-only exploration is skipped
    // entirely (Phase 9x-v Stage 3): MBH's own hop loop, seeded from the
    // pruning/backfit/Tisserand pool, natively provides both the exploration
    // and refinement roles DE's two-phase split needed separately.
    if mga.search_method == SearchMethod::De {
        for restart in 0..mga.de_restarts {
            let seed = mga.de_seed + restart as u64 * 137;
            let gen_offset = restart * phase1_gen;
            let (result, mut population) = run_de_variant(
                mga, pop, phase1_gen, seed,
                bounds, &[],
                |p| phase1_fitness(p, cfg, almanac, dep_jd_base, flyby_bodies),
                |g, best, params| {
                    let legs = mga_leg_step_info(params, cfg, almanac, dep_jd_base, flyby_bodies);
                    on_step(gen_offset + g, 1, best, params, &legs);
                },
            );
            phase1_history.extend_from_slice(&result.history);
            phase1_param_history.extend_from_slice(&result.param_history);
            population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            elite_seeds.extend(population.into_iter()
                .take(ELITES_PER_RESTART)
                .map(|(params, _)| params));
            println!(
                "  Phase 1 restart {}/{}: best DSM ΔV sum = {:.1} m/s",
                restart + 1, mga.de_restarts, result.best_fitness
            );
        }
    } else {
        println!("  Phase 1 skipped (search_method = MBH — hop chains provide exploration natively)");
    }

    // ── Phase 9x: incremental pruning pre-search, ADDS to (never replaces) ──
    // Phase 1's elites — an optional leg-by-leg decomposition that locates
    // candidate basins the all-at-once DE restarts above may not reach
    // (Ceriotti 2010 Ch. 3). No-op when `[optimization
    // .mga.pruning]` is absent from the config — identical behaviour to
    // before this phase existed.
    let mut best_stitched_candidate: Option<(Vec<f64>, f64)> = None;
    let pruning_t0 = std::time::Instant::now();
    if let Some(pruning) = mga.pruning.as_ref() {
        if let Some((pruning_seeds, best_stitched)) =
            run_mga_pruning_seeds(cfg, almanac, flyby_bodies, dep_jd_base, pruning, bounds)
        {
            elite_seeds.extend(pruning_seeds);
            best_stitched_candidate = best_stitched;
        }
    }
    if std::env::var("MGA_TIME_PHASES").is_ok() {
        println!("  [MGA_TIME_PHASES] pruning+backfit: {:.1} s", pruning_t0.elapsed().as_secs_f64());
    }

    // Diagnostic (MGA_DUMP_ELITE_SEEDS=1): dump the single BEST
    // elite seed -- by the SAME phase2_fitness the real MBH/DE search uses,
    // not the partial/intermediate costs the console log prints -- to a
    // file, BEFORE Phase 2 (MBH/DE) touches it at all. Exists to get a
    // genuine apples-to-apples "how much does search improve THIS seed"
    // comparison (ours vs. a real pagmo run) -- feeding a POST-search
    // winner into either search only tests whether it's already converged,
    // which is a different (and less interesting) question.
    if std::env::var("MGA_DUMP_ELITE_SEEDS").is_ok() && !elite_seeds.is_empty() {
        let mut best: Option<(f64, &Vec<f64>)> = None;
        for s in &elite_seeds {
            if let Some(f) = phase2_fitness(s, cfg, almanac, dep_jd_base, flyby_bodies) {
                if best.map_or(true, |(bf, _)| f < bf) {
                    best = Some((f, s));
                }
            }
        }
        if let Some((f, s)) = best {
            let out_dir = cfg.simulation.output_dir.trim_end_matches('/');
            let _ = std::fs::create_dir_all(out_dir);
            let path = format!("{out_dir}/elite_seed_pre_mbh.csv");
            let row = format!("fitness,{}\n{f},{}", (0..s.len()).map(|i| format!("p{i}")).collect::<Vec<_>>().join(","),
                s.iter().map(|v| format!("{v:.15e}")).collect::<Vec<_>>().join(","));
            let _ = std::fs::write(&path, row);
            println!("  [MGA_DUMP_ELITE_SEEDS] best elite seed (pre-MBH) fitness = {f:.1} m/s -> {path}");
        }
    }

    // ── Phase 2: minimise total ΔV at the branch's full bounds, seeded with ──
    // Phase 1's elites from all restarts (exploitation without imprisonment),
    // plus any Phase 9x pruning-derived seeds added above.

    let phase2_gen = gen.saturating_sub(phase1_gen * mga.de_restarts).max(gen / 2);
    let seed_p2    = mga.de_seed + mga.de_restarts as u64 * 137;
    let gen_offset_p2 = mga.de_restarts * phase1_gen;

    // Pre-search-only mode (MGA_PRE_SEARCH_ONLY=1): skip Phase 2
    // entirely and take the best elite seed (by the same phase2_fitness
    // Phase 2 would minimize) as the result. The pre-search (Phase 1 DE /
    // pruning / backfit / bidirectional) is where the real leverage lives —
    // measured the full pipeline's MBH phase improved the raw
    // pre-search seed by only ~6% while dominating wall-clock — so for
    // quick A/B iteration on pre-search changes this gives the signal at a
    // fraction of the cost. Everything downstream (stitched-candidate
    // min-compare, eta/L-BFGS polish, CSV writing) still runs. Falls
    // through to a normal Phase 2 when no elite seed evaluates finitely.
    let pre_search_best: Option<(Vec<f64>, f64)> = if std::env::var("MGA_PRE_SEARCH_ONLY").is_ok() {
        let mut best: Option<(f64, &Vec<f64>)> = None;
        for s in &elite_seeds {
            if let Some(f) = phase2_fitness(s, cfg, almanac, dep_jd_base, flyby_bodies) {
                if best.map_or(true, |(bf, _)| f < bf) {
                    best = Some((f, s));
                }
            }
        }
        best.map(|(f, s)| {
            println!("  [MGA_PRE_SEARCH_ONLY] skipping Phase 2 — best pre-search seed fitness = {f:.1} m/s");
            (s.clone(), f)
        })
    } else {
        None
    };

    let (p2_best_params, p2_best_fitness, p2_history, p2_param_history) = if let Some((p, f)) = pre_search_best {
        (p.clone(), f, vec![f], vec![p])
    } else {
        match mga.search_method {
        SearchMethod::De => {
            let (p2_result, _) = run_de_variant(
                mga, pop, phase2_gen, seed_p2,
                bounds, &elite_seeds,
                |p| phase2_fitness(p, cfg, almanac, dep_jd_base, flyby_bodies),
                |g, best, params| {
                    let legs = mga_leg_step_info(params, cfg, almanac, dep_jd_base, flyby_bodies);
                    on_step(gen_offset_p2 + g, 2, best, params, &legs);
                },
            );
            (p2_result.best_params, p2_result.best_fitness, p2_result.history, p2_result.param_history)
        }
        SearchMethod::Mbh => {
            let local: LocalOptimizer = match mga.mbh.local_optimizer {
                MbhLocalOptimizerToml::NelderMead => {
                    NelderMead { max_iter: mga.mbh.local_max_iter, ..Default::default() }.into()
                }
                MbhLocalOptimizerToml::CompassSearch => {
                    CompassSearch { max_iter: mga.mbh.local_max_iter, ..Default::default() }.into()
                }
                MbhLocalOptimizerToml::HookeJeeves => {
                    let hj = &mga.mbh.hooke_jeeves;
                    HookeJeevesSearch {
                        max_fevals: hj.max_fevals,
                        start_range: hj.start_range,
                        stop_range: hj.stop_range,
                        reduction_coeff: hj.reduction_coeff,
                    }
                    .into()
                }
            };
            let mbh = MbhSolver {
                hops: mga.mbh.hops,
                perturb_fraction: mga.mbh.perturb_fraction,
                kick_scale: mga.mbh.kick_scale,
                local,
                seed: mga.mbh.seed,
                stop_after: mga.mbh.stop_after,
                global_stall: mga.mbh.global_stall.map(|gs| GlobalStallConfig {
                    patience: gs.patience,
                    margin_frac: gs.margin_frac,
                }),
                extra_random_chains: mga.mbh.extra_random_chains,
                migration: if mga.mbh.migration.enabled {
                    Some(MigrationConfig {
                        interval: mga.mbh.migration.interval,
                        margin_frac: mga.mbh.migration.margin_frac,
                    })
                } else {
                    None
                },
            };
            println!(
                "  Phase 2 (MBH): {} seed chain(s){}, {} hops/chain{}{}{}",
                elite_seeds.len().max(1),
                if mga.mbh.extra_random_chains > 0 { format!(" + {} random", mga.mbh.extra_random_chains) } else { String::new() },
                mga.mbh.hops,
                mga.mbh.stop_after.map(|s| format!(", early-stop after {s} stagnant hops")).unwrap_or_default(),
                mga.mbh.global_stall.map(|gs| format!(", global-stall after {} hops beyond {:.0}% of best", gs.patience, gs.margin_frac * 100.0)).unwrap_or_default(),
                if mga.mbh.migration.enabled {
                    format!(", migration every {} hops beyond {:.0}% of best", mga.mbh.migration.interval, mga.mbh.migration.margin_frac * 100.0)
                } else {
                    String::new()
                },
            );
            let mbh_t0 = std::time::Instant::now();
            let result = mbh.run_seeded_with_progress(
                bounds, &elite_seeds,
                |p| phase2_fitness(p, cfg, almanac, dep_jd_base, flyby_bodies),
                |h, best, params| {
                    let legs = mga_leg_step_info(params, cfg, almanac, dep_jd_base, flyby_bodies);
                    on_step(gen_offset_p2 + h, 2, best, params, &legs);
                },
            );
            if std::env::var("MGA_TIME_PHASES").is_ok() {
                println!("  [MGA_TIME_PHASES] mbh: {:.1} s", mbh_t0.elapsed().as_secs_f64());
            }
            // Chain-contribution summary: the "how many chains
            // actually mattered" number previously reconstructed by hand
            // from history row counts when diagnosing wasted compute.
            if result.best_fitness.is_finite() {
                let n_close = result.chain_bests.iter().filter(|f| **f <= result.best_fitness * 1.01).count();
                println!(
                    "  Phase 2 (MBH): {} of {} chain(s) finished within 1% of the best result",
                    n_close, result.chain_bests.len(),
                );
            }
            (result.best_params, result.best_fitness, result.history, result.param_history)
        }
        }
    };

    let mut best_params  = p2_best_params;
    let mut best_fitness = p2_best_fitness;

    // Take the best bidirectional-stitched candidate directly if it beats
    // Phase 2's own DE winner: seeding alone doesn't
    // guarantee a good stitched chromosome survives the population's
    // crossover/selection dynamics — this is a strict min() over two
    // already-evaluated real candidates, so it can only ever help, never
    // regress a run relative to not having this comparison at all.
    if let Some((cand_params, cand_fitness)) = best_stitched_candidate {
        if cand_fitness < best_fitness {
            println!(
                "  Phase 9x: bidirectional-stitched candidate beats Phase 2 DE winner \
                 ({cand_fitness:.1} < {best_fitness:.1} m/s) — using it as the final result",
            );
            best_params  = cand_params;
            best_fitness = cand_fitness;
        }
    }

    println!("  Phase 2: best total ΔV = {:.1} m/s", best_fitness);

    // ── Eta re-polish + joint NLP polish, alternated to a fixed point ────────
    // (Phase 9x). See `repolish_leg_etas`'s and `LbfgsSolver`'s
    // doc comments: eta's per-leg GRID can basin-hop a single coordinate
    // (which purely local L-BFGS can't), and L-BFGS moves ALL variables
    // jointly along a curvature-informed direction (which coordinate descent
    // can't) — each re-opens work for the other, so the pair alternates
    // until neither finds meaningful work. Both accept strict improvement
    // only, so the loop is monotone: it can only help or leave the winner
    // unchanged. The discrete tail genes (n_rev / leg-model, floor-decoded)
    // have zero finite-difference gradient within a unit interval, so
    // L-BFGS correctly leaves the already-chosen branch alone.
    // Rounds cap is a runaway guard only — converged winners exit via the
    // fixed-point break well before it. 4 was observed too low for rough
    // (tiny-budget / freshly-RNG-rerouted) winners still finding >1 m/s in
    // round 4 (caught by the idempotency regression test), which
    // truncated convergence and broke the fixed-point contract; 10 gives
    // ample headroom at negligible cost since each converged round is cheap.
    const POLISH_ROUNDS_MAX: usize = 10;
    const POLISH_FIXED_POINT_MS: f64 = 1.0;
    let nlp = LbfgsSolver { max_iter: 200, ..Default::default() };
    for round in 1..=POLISH_ROUNDS_MAX {
        let (eta_before, eta_after) =
            repolish_leg_etas(&mut best_params, cfg, almanac, dep_jd_base, flyby_bodies, n);
        if eta_after < eta_before - 1.0e-6 {
            println!(
                "  Eta repolish (round {round}): {:.1} -> {:.1} m/s ({:+.1} m/s)",
                eta_before, eta_after, eta_after - eta_before,
            );
            best_fitness = eta_after;
        }

        let nlp_result = nlp.run(bounds, &best_params, |p| {
            phase2_fitness(p, cfg, almanac, dep_jd_base, flyby_bodies)
        });
        let mut nlp_gain = 0.0;
        if nlp_result.best_fitness < best_fitness - 1.0e-6 {
            nlp_gain = best_fitness - nlp_result.best_fitness;
            println!(
                "  NLP polish (L-BFGS, round {round}, {} iters): {:.1} -> {:.1} m/s ({:+.1} m/s)",
                nlp_result.iterations, best_fitness, nlp_result.best_fitness, -nlp_gain,
            );
            best_params  = nlp_result.best_params;
            best_fitness = nlp_result.best_fitness;
        }

        // Fixed point: neither polish found meaningful work this round.
        if (eta_before - eta_after) < POLISH_FIXED_POINT_MS && nlp_gain < POLISH_FIXED_POINT_MS {
            break;
        }
    }

    Ok((best_params, best_fitness, phase1_history, p2_history, phase1_param_history, p2_param_history))
}

/// Takes `flyby_bodies` explicitly so that both the static-config path and
/// the sequence-search path can share the same optimization logic.
/// Phase 9w-vi: run the Phase 9w ballistic scan for this exact flyby
/// sequence and derive `(dep_jd_center, window_days, leg_tof_days_bounds)`
/// — the same three quantities `build_bounds_with_overrides` would
/// otherwise read straight from `departure_epoch`/`departure_window_days`/
/// `leg_tof_days` — from the scan's best feasible branch(es), instead of
/// trusting the config's own hand-written values.
///
/// Neighborhood selection: every scan record within `NEIGHBORHOOD_COST_FACTOR`
/// (2x) of the best record's Flyby-style cost (`vinf_dep_ms + sum_flyby_dv_ms`
/// — the only cost every branch shares regardless of mission objective; the
/// scan has no capture-burn chromosome gene at all, see
/// `mga_scan_run.rs::capture_dv_ms`'s doc comment). "Generous" is
/// intentional, Ceriotti's own term for the same idea applied to pruning
/// (`PruningConfigToml::threshold_factor`) — this should capture the local
/// minimum's real basin, not just the single best grid sample.
///
/// Departure window: ± the neighborhood's own departure-date spread around
/// the best date, padded by `10 * departure_step_days` so the DE search
/// isn't pinned to exactly the scan's own (coarse) grid resolution.
///
/// Per-leg TOF bounds: min/max TOF actually seen in the neighborhood, per
/// leg, padded `TOF_PADDING_FRACTION` (15%) on each side for the same
/// reason.
///
/// Errors if the scan finds no feasible branches at all for this sequence —
/// a real infeasibility must never be silently hidden behind a fallback to
/// the config's own (possibly wrong, per the leg_tof_days position-indexing
/// bug class this feature exists to route around) bounds.
fn derive_scan_informed_bounds(
    cfg: &MissionConfig,
    almanac: &Almanac,
    flyby_bodies: &[String],
    scan_cfg: &crate::config::MgaScanConfigToml,
) -> Result<(f64, f64, Vec<[f64; 2]>), String> {
    let (records, _capture_dvs, _names, n_legs_evaluated, n_records_dropped, elapsed_s) =
        crate::mga_scan_run::compute_mga_scan_for_sequence(cfg, almanac, flyby_bodies)?;

    if records.is_empty() {
        let seq_str = if flyby_bodies.is_empty() {
            "(direct)".to_string()
        } else {
            flyby_bodies.join(" → ")
        };
        return Err(format!(
            "scan_informed_window: ballistic scan found no feasible branches for sequence {seq_str} \
             ({n_legs_evaluated} legs evaluated, {n_records_dropped} dropped by max_records, \
             {elapsed_s:.1}s) — widen optimization.mga.scan.flyby_dv_max_ms / \
             optimization.mga.leg_tof_days / optimization.mga.scan.horizon_years before \
             enabling scan_informed_window"
        ));
    }

    // Flyby-style cost — see doc comment above for why this, not a
    // mission-objective-specific cost, is used to rank branches here.
    let cost = |r: &ScanRecord| r.vinf_dep_ms + r.sum_flyby_dv_ms;

    let (best_idx, _) = records.iter().enumerate()
        .min_by(|(_, a), (_, b)| cost(a).partial_cmp(&cost(b)).unwrap())
        .expect("records is non-empty, checked above");
    let best = &records[best_idx];
    let best_cost = cost(best);

    const NEIGHBORHOOD_COST_FACTOR: f64 = 2.0;
    let neighborhood: Vec<&ScanRecord> = records.iter()
        .filter(|r| cost(r) <= best_cost * NEIGHBORHOOD_COST_FACTOR)
        .collect();

    let best_dep_day = best.dep_epoch_s / 86_400.0;
    let dep_days: Vec<f64> = neighborhood.iter().map(|r| r.dep_epoch_s / 86_400.0).collect();
    let dep_lo = dep_days.iter().cloned().fold(f64::INFINITY, f64::min);
    let dep_hi = dep_days.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let margin_days = 10.0 * scan_cfg.departure_step_days;
    let half_window_days = (best_dep_day - dep_lo).max(dep_hi - best_dep_day) + margin_days;
    let window_days = 2.0 * half_window_days;
    // JD 2_451_544.5 = MJD2000 epoch — same convention `ScanRecord::dep_epoch_s`
    // uses (`mga_scan_run.rs`'s `state_fns`/`anise_state_at`), matching
    // `epoch_to_jd`'s real-JD scale so `dep_jd_center` drops straight into
    // this function's `dep_jd_base` local exactly like the non-scan path's
    // `epoch_to_jd(dep_epoch)` does.
    let dep_jd_center = 2_451_544.5 + best_dep_day;

    const TOF_PADDING_FRACTION: f64 = 0.15;
    let n = flyby_bodies.len() + 1;
    let leg_tof_bounds: Vec<[f64; 2]> = (0..n).map(|k| {
        let tofs: Vec<f64> = neighborhood.iter().map(|r| r.leg_tofs_s[k] / 86_400.0).collect();
        let lo = tofs.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = tofs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let pad = (hi - lo).max(1.0) * TOF_PADDING_FRACTION;
        [(lo - pad).max(1.0), hi + pad]
    }).collect();

    println!(
        "  Scan-informed window: {} feasible branches total ({} in local-minimum neighborhood; \
         {n_legs_evaluated} legs evaluated in {elapsed_s:.1}s, {n_records_dropped} dropped) — \
         best cost {best_cost:.1} m/s",
        records.len(), neighborhood.len()
    );
    println!(
        "    departure_epoch (auto) = JD {dep_jd_center:.2}, departure_window_days (auto) = {window_days:.1}"
    );
    for (k, [lo, hi]) in leg_tof_bounds.iter().enumerate() {
        println!("    leg {k} leg_tof_days (auto) = [{lo:.1}, {hi:.1}]");
    }

    Ok((dep_jd_center, window_days, leg_tof_bounds))
}

fn run_mga_fixed_sequence<F>(
    cfg: &MissionConfig,
    almanac: &Almanac,
    flyby_bodies: &[String],
    on_step: &mut F,
) -> Result<MgaResult, String>
where
    F: FnMut(usize, u8, f64, &[f64], &[MgaLegStepInfo]),
{
    let opt = cfg.optimization.as_ref()
        .ok_or("optimization config is required for MGA")?;
    let mga = opt.mga.as_ref()
        .ok_or("optimization.mga is required when method = \"MGA\"")?;

    let n = n_legs(flyby_bodies);

    let (dep_jd_base, window_days_override, leg_tof_days_override):
        (f64, Option<f64>, Option<Vec<[f64; 2]>>) = if mga.scan_informed_window {
        let scan_cfg = mga.scan.as_ref().ok_or(
            "optimization.mga.scan_informed_window is true but optimization.mga.scan is not \
             configured — this should have been caught by check_config"
        )?;
        let seq_str = if flyby_bodies.is_empty() { "(direct)".to_string() } else { flyby_bodies.join(" → ") };
        println!("  Phase 9w-vi: running ballistic scan to auto-derive DE search bounds for {}", seq_str);
        let (dep_jd_center, window_days, leg_bounds) =
            derive_scan_informed_bounds(cfg, almanac, flyby_bodies, scan_cfg)?;
        (dep_jd_center, Some(window_days), Some(leg_bounds))
    } else {
        let dep_epoch_str = opt.departure_epoch.as_deref()
            .ok_or("optimization.departure_epoch is required for MGA")?;
        let dep_epoch = parse_epoch(dep_epoch_str).map_err(|e| format!("departure_epoch parse: {e}"))?;
        (epoch_to_jd(dep_epoch), None, None)
    };

    let bounds = build_bounds_with_overrides(
        cfg, flyby_bodies, window_days_override, leg_tof_days_override.as_deref(),
    )
        .ok_or("could not build chromosome bounds — check mga config")?;

    println!(
        "MGA optimizer: {} body sequence ({} legs), {} chromosome params, pop={}, gen={}×restarts={}",
        n + 1, n, bounds.len(), mga.de_population_size, mga.de_generations, mga.de_restarts
    );

    // ── Resonance-family branching ──────────────────────────────
    // One full search per resonance family of each same-body leg, plus the
    // unrestricted branch — see `resonance_family_branches` for why this
    // discrete choice must be enumerated rather than left to sampling bias
    // and MBH kicks. Best branch wins; a sequence with no same-body legs has
    // exactly one branch (today's behaviour, no overhead).
    let body_names_owned: Vec<&str> = {
        let mut v: Vec<&str> = Vec::with_capacity(n + 1);
        v.push(&opt.departure_body);
        for fb in flyby_bodies { v.push(fb.as_str()); }
        v.push(&opt.target_body);
        v
    };
    // Pre-sample every body over the search's full epoch envelope
    // (departure window through max total TOF, with margin) so fitness
    // evaluations interpolate instead of querying ANISE live — see
    // `EphemerisCache`. Installed before the branch loop: one build serves
    // every branch, every phase (pruning, backfit, Phase 2, polish).
    {
        let jd_min = dep_jd_base + bounds[0].0 - 5.0;
        let max_total_tof: f64 = (0..n).map(|k| bounds[4 + 2 * k].1).sum();
        let jd_max = dep_jd_base + bounds[0].1 + max_total_tof + 30.0;
        install_eph_cache(almanac, &body_names_owned, jd_min, jd_max);
    }

    // External seed injection: `MGA_SEED_CHROMOSOME=<path>`
    // loads a previous run's `mga_best_chromosome.csv` (or any
    // hand-built one in that format) as an additional Phase-2 seed for
    // EVERY branch — for "does biasing the search toward a known-good
    // basin help it converge further" experiments. Loaded once here (not
    // per branch) since it's the same file for the whole run; flyby-body
    // sequence is verified to match (case-insensitive) so a mismatched
    // file fails loudly instead of silently seeding garbage. See
    // `run_search_over_bounds`'s doc comment for how it's used downstream.
    let external_seed: Option<Vec<f64>> = match std::env::var("MGA_SEED_CHROMOSOME") {
        Ok(path) => {
            let (seed_bodies, seed_params) = load_best_chromosome_csv(&path)?;
            let seed_bodies_lower: Vec<String> = seed_bodies.iter().map(|s| s.to_lowercase()).collect();
            let flyby_bodies_lower: Vec<String> = flyby_bodies.iter().map(|s| s.to_lowercase()).collect();
            if seed_bodies_lower != flyby_bodies_lower {
                return Err(format!(
                    "MGA_SEED_CHROMOSOME='{path}': flyby sequence {seed_bodies:?} does not match \
                     this run's {flyby_bodies:?}"
                ));
            }
            println!("  External seed loaded from '{path}' ({} params)", seed_params.len());
            Some(seed_params)
        }
        Err(_) => None,
    };

    let mut branches = resonance_family_branches(&bounds, &body_names_owned, n);
    if branches.len() > 1 {
        println!("  Resonance-family branching: {} branches (full + {} family window(s))",
            branches.len(), branches.len() - 1);
    }
    // Diagnostic branch filter: `MGA_BRANCH_ONLY=<substring>`
    // runs only branches whose label contains the substring
    // (case-insensitive) — e.g. `MGA_BRANCH_ONLY=Earth=2` runs just the
    // 2-year Earth-Earth family. For targeted validation/debugging of one
    // family without paying for the full enumeration; never needed in
    // normal use. Same env-knob pattern as `MGA_DISABLE_VILM`.
    if let Ok(filter) = std::env::var("MGA_BRANCH_ONLY") {
        let f = filter.to_lowercase();
        let before = branches.len();
        branches.retain(|(label, _)| label.to_lowercase().contains(&f));
        println!(
            "  MGA_BRANCH_ONLY='{filter}': {} of {before} branch(es) retained",
            branches.len()
        );
        if branches.is_empty() {
            return Err(format!("MGA_BRANCH_ONLY='{filter}' matched no branch label"));
        }
    }

    let mut best: Option<(Vec<f64>, f64, Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>, String)> = None;
    for (branch_idx, (label, branch_bounds)) in branches.iter().enumerate() {
        if branches.len() > 1 {
            println!("── Branch {}/{}: {label} ──", branch_idx + 1, branches.len());
        }
        let (params, fitness, h1, h2, hp1, hp2) = run_search_over_bounds(
            cfg, almanac, flyby_bodies, dep_jd_base, branch_bounds, external_seed.as_deref(), on_step,
        )?;
        if branches.len() > 1 {
            println!("  Branch '{label}': best total ΔV = {fitness:.1} m/s");
        }
        if best.as_ref().map_or(true, |(_, bf, ..)| fitness < *bf) {
            best = Some((params, fitness, h1, h2, hp1, hp2, label.clone()));
        }
    }
    let (best_params, best_fitness, phase1_history_saved, p2_history, phase1_param_history_saved, p2_param_history, win_label) =
        best.ok_or("no search branch produced a result")?;
    if branches.len() > 1 {
        println!("  Resonance-family branching: winner = '{win_label}' at {best_fitness:.1} m/s");
    }
    let mut full_history = phase1_history_saved.clone();
    full_history.extend_from_slice(&p2_history);
    let mut full_param_history = phase1_param_history_saved.clone();
    full_param_history.extend_from_slice(&p2_param_history);
    let _ = best_fitness; // reported via the branch prints above; ΔV re-derived below

    // ── Re-evaluate best chromosome for detailed output ───────────────────────

    let ev = evaluate_chromosome_detailed(&best_params, cfg, almanac, dep_jd_base, flyby_bodies)
        .ok_or("best chromosome is infeasible after optimisation — try more generations or restarts")?;

    if ev.penalty_ms > 0.0 {
        println!(
            "  WARNING: winner violates soft constraints (graded penalty {:.1} m/s-equiv — \
             perihelion floor and/or flyby minimum periapsis); ΔV numbers below exclude the penalty",
            ev.penalty_ms
        );
    }

    let (dv_dep, _) = crate::design::departure_escape_dv_ms(cfg, &opt.departure_body, ev.v_inf_dep_ms)
        .ok_or("departure body not found in catalog")?;
    let dv_dsms: Vec<f64> = ev.legs.iter().map(|l| l.leg.dv_dsm_ms).collect();
    let dv_dsm_total: f64 = dv_dsms.iter().sum();
    let dv_arr = arrival_dv_ms(cfg, &ev);
    let dv_total = dv_dep + dv_dsm_total + dv_arr;
    let tof_total: f64 = (0..n).map(|k| tof_days(&best_params, k)).sum();

    println!("  ΔV breakdown: dep={:.1}  DSMs={:.1}  arr={:.1}  total={:.1} m/s",
        dv_dep, dv_dsm_total, dv_arr, dv_total);
    println!("  TOF total: {:.1} days", tof_total);

    // ── Build arc for plotting ────────────────────────────────────────────────

    let arc = sample_arc(&ev, &best_params, n);

    // ── Re-propagate using Dopri5 ─────────────────────────────────────────────

    println!("  Re-propagating best chromosome with Dopri5 integrator …");
    let repropagated_arc = reprop_mga_arc(&ev, &best_params, flyby_bodies, cfg, almanac);
    if repropagated_arc.is_empty() {
        eprintln!("  Warning: Dopri5 re-propagation produced no points — only Keplerian arc available.");
    } else {
        println!("  Re-propagation: {} points across {} legs", repropagated_arc.len(), n);
    }

    // ── Write CSVs ────────────────────────────────────────────────────────────

    let out_dir = cfg.simulation.output_dir.trim_end_matches('/');
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("Warning: could not create output directory '{out_dir}': {e}");
    }

    // Build the full body-name sequence for per-leg labelling.
    let body_names: Vec<String> = {
        let mut v = vec![opt.departure_body.clone()];
        v.extend_from_slice(flyby_bodies);
        v.push(opt.target_body.clone());
        v
    };

    let leg_tofs_days: Vec<f64> = (0..n).map(|k| tof_days(&best_params, k)).collect();
    let dsm_positions_m: Vec<[f64; 3]> = ev.legs.iter()
        .map(|le| [le.leg.r_dsm_m.x, le.leg.r_dsm_m.y, le.leg.r_dsm_m.z])
        .collect();
    let dsm_epochs_s: Vec<f64> = ev.legs.iter().enumerate()
        .map(|(k, le)| (le.t_start_days + eta(&best_params, k) * tof_days(&best_params, k)) * 86_400.0)
        .collect();
    let dep_jd = dep_jd_base + dep_offset(&best_params);

    let mga_result = MgaResult {
        dv_total_ms:      dv_total,
        dv_departure_ms:  dv_dep,
        dv_dsms_ms:       dv_dsms,
        dv_arrival_ms:    dv_arr,
        tof_total_days:   tof_total,
        leg_tofs_days,
        dsm_positions_m,
        dsm_epochs_s,
        body_sequence:    body_names.clone(),
        dep_jd,
        dep_jd_base,
        best_params:      best_params.clone(),
        convergence:      p2_history.clone(),
        phase1_history:   phase1_history_saved,
        param_history:    p2_param_history,
        phase1_param_history: phase1_param_history_saved,
        arc:              arc.clone(),
        repropagated_arc: repropagated_arc.clone(),
    };

    write_arc_csv(&arc, out_dir);
    write_repropagated_csv(&repropagated_arc, out_dir);
    write_convergence_csv(&full_history, &full_param_history, out_dir);
    write_params_csv(&mga_result, out_dir);
    write_legs_csv(&ev, &best_params, flyby_bodies, &body_names, out_dir);

    Ok(mga_result)
}

// ── mga-geometry: re-evaluate from saved chromosome ──────────────────────────

/// Read and parse a saved `mga_best_chromosome.csv` (written by
/// `write_chromosome_csv`) into `(flyby_bodies, params)`. Shared by
/// `run_mga_geometry` and `src/bin/primer_vector_check.rs` (
/// Stage 1) so the parsing logic lives in exactly one place.
pub fn load_best_chromosome_csv(csv_path: &str) -> Result<(Vec<String>, Vec<f64>), String> {
    let content = std::fs::read_to_string(csv_path)
        .map_err(|e| format!("cannot read {csv_path}: {e}\nRun the MGA optimizer first."))?;

    let mut lines = content.lines();
    let _header  = lines.next().ok_or("mga_best_chromosome.csv is empty")?;
    let data_row = lines.next().ok_or("mga_best_chromosome.csv has no data row")?;

    // Split on the FIRST comma only — flyby_bodies is semicolon-separated.
    let comma = data_row.find(',')
        .ok_or("mga_best_chromosome.csv: no comma separating flyby_bodies from params")?;
    let bodies_str = &data_row[..comma];
    let params_str = &data_row[comma + 1..];

    let flyby_bodies: Vec<String> = if bodies_str.is_empty() {
        vec![]
    } else {
        bodies_str.split(';').map(str::to_string).collect()
    };
    let params: Vec<f64> = params_str
        .split(',')
        .map(|s| s.trim().parse::<f64>().map_err(|e| format!("parse param '{}': {e}", s.trim())))
        .collect::<Result<Vec<f64>, _>>()?;

    let n = n_legs(&flyby_bodies);
    let expected_len = chromosome_len(n);
    if params.len() != expected_len {
        return Err(format!(
            "chromosome has {} params but expected {} for {n} legs",
            params.len(), expected_len
        ));
    }
    Ok((flyby_bodies, params))
}

/// Re-evaluate the best MGA chromosome from `<out_dir>/mga_best_chromosome.csv`
/// and print a detailed per-leg table. No optimizer rerun — fast diagnostic.
///
/// Run via: `cargo run -p mission_planner --release mga-geometry <config.toml>`
pub fn run_mga_geometry(cfg: &MissionConfig, almanac: &Almanac) -> Result<(), String> {
    let opt = cfg.optimization.as_ref()
        .ok_or("mga-geometry requires an [optimization] section")?;

    let out_dir = cfg.simulation.output_dir.trim_end_matches('/');
    let csv_path = format!("{out_dir}/mga_best_chromosome.csv");
    let (flyby_bodies, params) = load_best_chromosome_csv(&csv_path)?;
    let n = n_legs(&flyby_bodies);

    // ── Build departure JD and evaluate ──────────────────────────────────────
    let dep_epoch_str = opt.departure_epoch.as_deref()
        .ok_or("optimization.departure_epoch is required")?;
    let dep_epoch   = parse_epoch(dep_epoch_str)
        .map_err(|e| format!("departure_epoch parse: {e}"))?;
    let dep_jd_base = epoch_to_jd(dep_epoch);

    let ev = evaluate_chromosome_detailed(&params, cfg, almanac, dep_jd_base, &flyby_bodies)
        .ok_or("chromosome is infeasible — try re-running the optimizer")?;

    if ev.penalty_ms > 0.0 {
        println!(
            "  WARNING: chromosome violates soft constraints (graded penalty {:.1} m/s-equiv — \
             perihelion floor and/or flyby minimum periapsis)",
            ev.penalty_ms
        );
    }

    // ── Body name sequence ────────────────────────────────────────────────────
    let mut body_names: Vec<String> = Vec::with_capacity(n + 1);
    body_names.push(opt.departure_body.clone());
    body_names.extend_from_slice(&flyby_bodies);
    body_names.push(opt.target_body.clone());

    // ── Departure ─────────────────────────────────────────────────────────────
    let (dv_dep, _) = crate::design::departure_escape_dv_ms(cfg, &opt.departure_body, ev.v_inf_dep_ms)
        .ok_or_else(|| format!("departure body '{}' not in catalog", opt.departure_body))?;

    let dep_offset_d = dep_offset(&params);
    let (theta_raw, phi_raw) = decode_theta_phi(theta_dep(&params), phi_dep(&params));
    let theta_d = theta_raw.to_degrees();
    let phi_d   = phi_raw.to_degrees();

    println!();
    println!("MGA best solution — detailed re-evaluation");
    println!("  Source: {csv_path}");
    println!("  Flyby sequence: {}", body_names.join(" → "));
    println!(
        "  Departure offset: {:+.2} days  dep v∞ = {:.3} km/s  (θ={:.1}°, φ={:.1}°)",
        dep_offset_d, ev.v_inf_dep_ms / 1000.0, theta_d, phi_d
    );
    println!();

    // ── Per-leg table ─────────────────────────────────────────────────────────
    let dv_dsms: Vec<f64> = ev.legs.iter().map(|l| l.leg.dv_dsm_ms).collect();
    for k in 0..n {
        let tof_k = tof_days(&params, k);
        let eta_k = eta(&params, k);
        let v_inf_arr_ms = ev.legs[k].leg.v_inf_arr_mps.norm();
        println!(
            "  Leg {k}: {:10} → {:10}   TOF = {:7.1} d   η = {:.2}   DSM ΔV = {:7.1} m/s   arr v∞ = {:.3} km/s",
            body_names[k], body_names[k + 1],
            tof_k, eta_k, dv_dsms[k], v_inf_arr_ms / 1000.0
        );

        // Print flyby details for intermediate bodies.
        if k < n - 1 {
            let j        = k;
            let rp_norm_j = rp_norm(&params, n, j);
            let beta_j   = beta(&params, n, j);
            let r_body   = ev.body_radii[k + 1];
            let rp_m     = rp_norm_j * r_body;
            let rp_km    = rp_m / 1000.0;
            let mu_body  = ev.body_mus[k + 1];
            let v_inf_in = ev.legs[k].leg.v_inf_arr_mps;
            let v_inf_in_sq = v_inf_in.norm_squared();
            let sin_half = (mu_body / (mu_body + rp_m * v_inf_in_sq)).min(1.0);
            let turn_deg = 2.0 * sin_half.asin().to_degrees();
            let v_inf_out = flyby_turn(v_inf_in, rp_m, beta_j, mu_body);
            println!(
                "    ↪  Flyby {:10}: r_p = {:8.0} km ({:.3}× R)   β = {:+.1}°   turn = {:.1}°   v∞_out = {:.3} km/s",
                body_names[k + 1], rp_km, rp_norm_j,
                beta_j.to_degrees(), turn_deg,
                v_inf_out.norm() / 1000.0
            );
        }
    }

    // ── Arrival / LOI ─────────────────────────────────────────────────────────
    let v_inf_arr_mag = ev.v_inf_arr.norm();
    let dv_arr = arrival_dv_ms(cfg, &ev);

    let dv_dsm_total: f64 = dv_dsms.iter().sum();
    let tof_total: f64    = (0..n).map(|k| tof_days(&params, k)).sum();
    let dv_total          = dv_dep + dv_dsm_total + dv_arr;

    println!();
    println!("  ΔV summary:");
    println!("    Departure burn:  {:8.1} m/s", dv_dep);
    for (k, dv) in dv_dsms.iter().enumerate() {
        println!("    DSM {k}:           {:8.1} m/s", dv);
    }
    let arr_label = if dv_arr > 0.1 { "LOI / arrival: " } else { "Arrival (flyby):" };
    println!("    {arr_label}  {:8.1} m/s  arr v∞ = {:.3} km/s",
        dv_arr, v_inf_arr_mag / 1000.0);
    println!("    {}", "─".repeat(36));
    println!("    Total ΔV:        {:8.1} m/s", dv_total);
    println!("    Total TOF:       {:8.1} days", tof_total);
    println!();

    // Re-write legs CSV so plot scripts always have fresh data after a geometry run.
    write_legs_csv(&ev, &params, &flyby_bodies, &body_names, out_dir);

    // Re-sample and re-write the analytic arc too — lets a geometry run
    // refresh mga_best.csv after an arc-sampling fix without a DE rerun
    // (added, prompted by the propagate_kepler non-convergence
    // garbage point this exact file carried for the Cassini-2 run).
    let arc = sample_arc(&ev, &params, n);
    write_arc_csv(&arc, out_dir);

    // Re-propagate with Dopri5 and write `mga_repropagated.csv`.
    println!("  Re-propagating with Dopri5 integrator …");
    let reprop = reprop_mga_arc(&ev, &params, &flyby_bodies, cfg, almanac);
    if reprop.is_empty() {
        eprintln!("  Warning: Dopri5 re-propagation produced no points.");
    } else {
        println!("  Re-propagation: {} points", reprop.len());
        write_repropagated_csv(&reprop, out_dir);
    }

    Ok(())
}

// ── Multiple-shooting refinement ─────────────────────────────────────────────

/// Maximum Newton–Raphson iterations for multiple shooting. Raised from the
/// original single-shooting formulation's 50 (Phase 9v-ix): trust-region step
/// limiting means more, smaller steps are sometimes needed to reach
/// convergence on a multi-flyby chain.
const MAX_MS_ITER: usize = 150;

/// Trust-region cap on the largest position-type component [m] of a single
/// Newton step (node r corrections). ~1.3 AU — a backstop against a
/// genuinely pathological pseudo-inverse column, not a normal-operation
/// limiter: once the Jacobian/residual are properly non-dimensionalized
/// (`MS_LEN_SCALE_M`/`MS_VEL_SCALE_MS`), a converging Newton step should
/// stay far under this on its own. An earlier, much tighter cap (5e9 m)
/// was found to be the actual bottleneck on a real 2-flyby case — it was
/// throttling normal Newton convergence to ~2%/iteration instead of the
/// scaled solve's much faster natural rate (Phase 9v-ix, found).
const MS_MAX_STEP_POS_M: f64 = 2.0e11;

/// Trust-region cap on the largest velocity-type component [m/s] of a single
/// Newton step (DSM ΔV corrections + node v corrections) — same backstop
/// rationale as `MS_MAX_STEP_POS_M`.
const MS_MAX_STEP_VEL_MS: f64 = 2.0e5;

/// Characteristic length scale [m] for non-dimensionalizing position-type
/// free variables/residuals before the SVD pseudo-inverse (~1 AU).
const MS_LEN_SCALE_M: f64 = 1.495_978_707e11;

/// Characteristic velocity scale [m/s] for non-dimensionalizing
/// velocity-type free variables/residuals (DSM ΔVs + node velocities)
/// before the SVD pseudo-inverse (order of Earth's heliocentric orbital
/// speed — a reasonable interplanetary velocity scale).
const MS_VEL_SCALE_MS: f64 = 3.0e4;

/// Per-free-variable scale factors matching [`ms_dv`]/[`ms_node`]'s layout:
/// `MS_VEL_SCALE_MS` for DSM components and node velocity components,
/// `MS_LEN_SCALE_M` for node position components.
fn ms_scale_x(n: usize) -> Vec<f64> {
    let mut s = vec![MS_VEL_SCALE_MS; 3 * n];
    for _ in 0..n.saturating_sub(1) {
        s.extend([MS_LEN_SCALE_M; 3]);
        s.extend([MS_VEL_SCALE_MS; 3]);
    }
    s
}

/// Per-constraint scale factors matching [`ms_constraint`]'s residual block
/// layout: `MS_LEN_SCALE_M` for position residuals, `MS_VEL_SCALE_MS` for
/// velocity residuals (intermediate nodes only).
fn ms_scale_f(n: usize) -> Vec<f64> {
    let mut s = Vec::with_capacity(ms_constraint_dim(n));
    for k in 0..n {
        s.extend([MS_LEN_SCALE_M; 3]);
        if k < n - 1 {
            s.extend([MS_VEL_SCALE_MS; 3]);
        }
    }
    s
}

/// Position continuity tolerance [m] at each patch point. 1 km is achievable
/// interplanetarily.
const MS_TOL_M: f64 = 1_000.0;

/// Velocity continuity tolerance [m/s] at each intermediate flyby node. This
/// is what makes the node state a genuine "no impulse" patch point rather
/// than an independent free position (Phase 9v-ix).
const MS_VEL_TOL_MS: f64 = 1.0;

/// Finite-difference step sizes for the Jacobian, ABSOLUTE and per variable
/// type. The previous relative rule (`max(1,|x_j|)·1e-4`) gave
/// ~15,000 km position perturbations for node components (heliocentric
/// magnitudes ~1.5e11 m) — applied to a node sitting ~6,500 km from a flyby
/// body's center, the perturbed node lands inside the planet and the column
/// is garbage, which is why the joint LM stalled instantly (λ→1e8, no
/// improving step) from a km-scale presolved start. Integrator noise floor
/// is ~rtol·AU ≈ 15 m, so these steps keep plenty of signal above it.
const MS_FD_EPS_DV_MS: f64 = 0.05;   // DSM ΔV components [m/s]
const MS_FD_EPS_POS_M: f64 = 1.0e4;  // node position components [m] (10 km)
const MS_FD_EPS_VEL_MS: f64 = 0.05;  // node velocity components [m/s]

/// Initial Levenberg-Marquardt damping factor λ (in the scaled/
/// non-dimensionalized problem, so this is dimensionless-ish — order 1
/// relative to typical scaled singular values). Adapted across outer
/// iterations, not reset each iteration.
const MS_LM_LAMBDA_INIT: f64 = 1.0;

/// λ floor/ceiling — keeps damping from vanishing to a numerically singular
/// pure-Newton step, or growing so large the step underflows to nothing.
const MS_LM_LAMBDA_MIN: f64 = 1e-8;
const MS_LM_LAMBDA_MAX: f64 = 1e8;

/// Multiplicative λ decrease on an accepted step / increase on a rejected
/// one — standard Levenberg-Marquardt adaptation factors.
const MS_LM_DECREASE_FACTOR: f64 = 3.0;
const MS_LM_INCREASE_FACTOR: f64 = 3.0;

/// Maximum λ-escalation tries per outer iteration before declaring this
/// iteration stalled (no improving step found even under heavy damping).
const MS_LM_MAX_TRIES: usize = 15;

/// Write the corrected arc to `mga_refined.csv` (same format as `mga_repropagated.csv`).
fn write_refined_csv(arc: &[MgaArcPoint], out_dir: &str) {
    if arc.is_empty() {
        return;
    }
    let mut rows = vec!["t_days,x_m,y_m,z_m,leg_idx,central_body".to_string()];
    for p in arc {
        let cb = p.central_body.as_deref().unwrap_or("");
        rows.push(format!("{},{},{},{},{},{cb}", p.t_days, p.x_m, p.y_m, p.z_m, p.leg_idx));
    }
    let path = format!("{out_dir}/mga_refined.csv");
    match std::fs::write(&path, rows.join("\n") + "\n") {
        Ok(()) => println!("  {path}"),
        Err(e) => eprintln!("Warning: could not write {path}: {e}"),
    }
}

/// Write corrected per-leg DSM ΔV vectors to `mga_refined_legs.csv`.
fn write_refined_legs_csv(dv_vecs: &[[f64; 3]], leg_tofs: &[f64], out_dir: &str) {
    let mut rows = vec!["leg_idx,dv_x_ms,dv_y_ms,dv_z_ms,dv_mag_ms,tof_days".to_string()];
    for (k, (dv, tof)) in dv_vecs.iter().zip(leg_tofs.iter()).enumerate() {
        let mag = (dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2]).sqrt();
        rows.push(format!("{k},{},{},{},{},{tof}", dv[0], dv[1], dv[2], mag));
    }
    let path = format!("{out_dir}/mga_refined_legs.csv");
    match std::fs::write(&path, rows.join("\n") + "\n") {
        Ok(()) => println!("  {path}"),
        Err(e) => eprintln!("Warning: could not write {path}: {e}"),
    }
}

/// Build ANISE-backed gravitational perturber entries for every MGA flyby body.
///
/// Each returned entry provides a time-varying heliocentric state callback; the
/// closure maps absolute mission seconds `t_abs_s` → JD via
/// `dep_jd + t_abs_s / 86400.0`. Bodies with a known SMA in the catalog get a
/// real Laplace SOI radius (so the propagator's central-body-switching machinery
/// can model the gravity assist as the spacecraft passes through the SOI);
/// bodies without an SMA are registered as point-mass perturbers only.
fn build_mga_perturber_entries<'a>(
    flyby_bodies: &[String],
    almanac: &'a Almanac,
    dep_jd: f64,
) -> Vec<PropagatorBodyEntry<'a>> {
    let mut entries = Vec::new();
    // Deduplicate by name: a VEEGA-style sequence lists Earth three times
    // (departure + two Earth flybys), and registering a body more than once
    // multiplies its real gravity by the duplicate count — near Earth the
    // spacecraft felt up to 3x mu_Earth (one central + two co-located
    // "third-body" copies), corrupting every hyperbolic passage. Found
    // via 3.4 AU iteration-0 multiple-shooting residuals from a
    // near-ballistic seed.
    let mut seen: Vec<String> = Vec::new();
    for name in flyby_bodies {
        let key = name.to_lowercase();
        if seen.contains(&key) { continue; }
        seen.push(key);
        let Some(anise) = anise_body(&name.to_lowercase()) else {
            eprintln!("Warning: flyby body '{name}' has no ANISE coverage — skipping N-body perturber");
            continue;
        };
        let Some(cat) = body_models::TargetBody::by_name(name) else {
            eprintln!("Warning: flyby body '{name}' not in body catalog — skipping N-body perturber");
            continue;
        };
        let mu_m3s2     = cat.mu_m3s2;
        let radius_m    = cat.radius_m;
        let soi_radius_m = cat.sma_m.map(|sma| laplace_soi_radius_m(sma, mu_m3s2 / MU_SUN_M3S2));
        let name_for_err = name.clone();
        entries.push(PropagatorBodyEntry {
            name: name.clone(),
            mu_m3s2,
            soi_radius_m,
            central_fidelity: None,
            state_at: Box::new(move |t_abs_s: f64| {
                let jd = dep_jd + t_abs_s / 86_400.0;
                body_state(almanac, EphemerisSource::Anise, Some(anise), &None, jd)
                    .map(|(r, v)| (Vector3::new(r[0], r[1], r[2]), Vector3::new(v[0], v[1], v[2])))
                    .unwrap_or_else(|| {
                        eprintln!("Warning: ephemeris unavailable for '{name_for_err}' at jd={jd}");
                        (Vector3::zeros(), Vector3::zeros())
                    })
            }),
            radius_m: Some(radius_m),
        });
    }
    entries
}

/// Closed-form time-of-flight [s] from periapsis to a given crossing radius
/// on an OUTBOUND hyperbolic departure trajectory. Same hyperbolic-Kepler
/// geometry [`flyby_soi_entry_state`] uses for an INBOUND flyby approach —
/// the magnitude is identical (a hyperbola's time-of-flight is symmetric
/// about periapsis); only the sign of true anomaly flips (positive =
/// outbound here, vs. negative = inbound there).
fn departure_transit_time_s(v_inf_mag: f64, r_park_m: f64, mu_body: f64, soi_radius_m: f64) -> f64 {
    let a = -mu_body / v_inf_mag.max(1.0).powi(2);
    let e = (1.0 - r_park_m / a).max(1.0 + 1e-9);
    let soi_r = soi_radius_m.max(r_park_m * 1.01);
    let p = r_park_m * (1.0 + e);
    let cos_nu = ((p / soi_r) - 1.0) / e;
    let nu = cos_nu.clamp(-1.0, 1.0).acos(); // outbound: positive true anomaly
    let tan_half_f = ((e - 1.0) / (e + 1.0)).sqrt() * (nu / 2.0).tan();
    let f_h = 2.0 * tan_half_f.atanh();
    let m_h = e * f_h.sinh() - f_h;
    let n_h = (mu_body / (-a).powi(3)).sqrt();
    (m_h / n_h).max(0.0)
}

/// Same as [`departure_transit_time_s`], but bisects `soi_radius_m` DOWN
/// when the full Laplace SOI's transit time would exceed `max_transit_s` —
/// same rationale and bisection pattern as
/// [`flyby_soi_entry_state_time_limited`] (a giant-planet-scale SOI can
/// imply a transit time longer than leg 0's own segment-A time budget).
fn departure_transit_time_s_limited(v_inf_mag: f64, r_park_m: f64, mu_body: f64, soi_radius_m: f64, max_transit_s: f64) -> f64 {
    let full = departure_transit_time_s(v_inf_mag, r_park_m, mu_body, soi_radius_m);
    if full <= max_transit_s || max_transit_s <= 0.0 {
        return full;
    }
    let mut lo = r_park_m * 1.01;
    let mut hi = soi_radius_m;
    let mut best = departure_transit_time_s(v_inf_mag, r_park_m, mu_body, lo);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let t = departure_transit_time_s(v_inf_mag, r_park_m, mu_body, mid);
        if t <= max_transit_s {
            best = t;
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) / hi.max(1.0) < 1e-6 { break; }
    }
    best
}

/// Fixed departure state for leg 0 in the multiple-shooting problem.
///
/// Real periapsis-burn state, replacing the earlier idealized
/// SOI-boundary/asymptotic-velocity placeholder — mirrors exactly how
/// intermediate flybys moved from a periapsis-turn construction to a
/// physically real state (9x-iv), just for the departure end of the chain
/// instead of an intermediate one. Uses
/// [`trajectory_solver::hyperbolic_departure_state`] (same closed-form
/// construction `mga_departure_injection_state`/single-leg's
/// `circular_orbit_burn_state` use) to get the real periapsis injection
/// state for the chromosome's own already-optimized departure v∞ vector,
/// from a real, finite parking-orbit radius — non-singular, unlike the
/// departure body's literal center (see the removed doc comment below for
/// the crash that construction caused).
///
/// Unlike an intermediate flyby's SOI-ENTRY node, this state is NOT a free
/// Newton-solve variable — it's fixed directly from the converged
/// chromosome, so 9v-ix's stiffness concern (a *perturbed* node deep in a
/// gravity well causing pathological sensitivity) does not apply here; a
/// fixed, non-singular periapsis point is safe. Segment A's own real
/// N-body propagation carries the spacecraft from this real burn state,
/// out through the departure body's SOI, and onward — exactly mirroring how
/// a departing leg's segment A already carries a flyby through SOI-entry →
/// periapsis → SOI-exit via real dynamics, not a separate closed-form leg.
///
/// Returns `(r_helio_m, v_helio_mps, transit_s)` — heliocentric state at
/// the real burn epoch, and the time-limited closed-form transit time
/// [`departure_transit_time_s_limited`] from periapsis to SOI crossing
/// (used by [`ms_leg_timing`] to shift leg 0's own start epoch/segment-A
/// duration the same way an intermediate flyby's `transit_time_s[k]` does).
///
/// Falls back to the old idealized SOI-boundary/asymptotic-velocity point
/// (zero transit time) if the departure body is missing from the catalog,
/// has no `sma_m` (no way to size an SOI), or `hyperbolic_departure_state`
/// returns `None` (a numerically zero v∞ — degenerate, no defined
/// asymptote direction).
fn ms_leg0_start(
    ev: &ChromosomeEval,
    params: &[f64],
    departure_body: &str,
    cfg: &MissionConfig,
    almanac: &Almanac,
) -> (Vector3<f64>, Vector3<f64>, f64) {
    let (r_body, v_body) = ev.body_rvs[0];
    let v_start   = ev.legs[0].v_sc_start; // asymptotic heliocentric velocity
    let v_inf     = v_start - v_body;
    let v_inf_mag = v_inf.norm();
    let Some(cat) = body_models::TargetBody::by_name(departure_body) else {
        return (r_body, v_start, 0.0);
    };
    let soi = cat.sma_m
        .map(|sma| laplace_soi_radius_m(sma, cat.mu_m3s2 / MU_SUN_M3S2))
        .unwrap_or(1.0e9);
    if v_inf_mag < 1.0 {
        // Degenerate (no meaningful asymptote direction) — offset along +x.
        return (r_body + Vector3::new(soi, 0.0, 0.0), v_start, 0.0);
    }
    let r_park_m = crate::design::resolve_parking_orbit_radius_m(cfg, &cat);
    let Some(dep) = hyperbolic_departure_state(cat.mu_m3s2, r_park_m, v_inf) else {
        return (r_body + v_inf * (soi / v_inf_mag), v_start, 0.0);
    };
    let naive_seg_a_s = eta(params, 0) * tof_days(params, 0) * 86_400.0;
    let max_transit_s = 0.8 * naive_seg_a_s.max(0.0);
    let transit_s = departure_transit_time_s_limited(v_inf_mag, r_park_m, cat.mu_m3s2, soi, max_transit_s);

    // Re-query the departure body's own heliocentric state at the SHIFTED
    // (real burn) epoch — same reasoning as the flyby SOI-entry
    // fix: the body moves during the transit, so reusing the nominal-epoch
    // `r_body`/`v_body` here would reintroduce a real position error at
    // exactly the scale that fix removed.
    let burn_jd = ev.dep_jd - transit_s / 86_400.0;
    let (r_body_burn, v_body_burn) = anise_body(&departure_body.to_lowercase())
        .and_then(|anise| body_state(almanac, EphemerisSource::Anise, Some(anise), &None, burn_jd))
        .map(|(r, v)| (Vector3::new(r[0], r[1], r[2]), Vector3::new(v[0], v[1], v[2])))
        .unwrap_or((r_body, v_body));

    (r_body_burn + dep.r0_m, v_body_burn + dep.v0_mps, transit_s)
}

/// Real parking-orbit departure injection state (Phase 12j,
/// display-only — see below for why it deliberately does NOT feed
/// [`ms_leg0_start`]/multiple shooting). Given the chromosome's already-
/// optimized departure v∞ vector, computes the REAL periapsis injection
/// point on a real circular parking orbit via
/// [`trajectory_solver::hyperbolic_departure_state`] — an exact, closed-form
/// construction (no iteration, no new chromosome dimension): any target v∞
/// vector has exactly one real burn, from any given parking-orbit radius,
/// that achieves it. Mirrors `optimize.rs::evaluate_candidate`'s single-leg
/// construction exactly, just reusing MGA's own already-computed departure
/// vector instead of a separately-searched `theta_burn_rad`.
///
/// Returns body-relative `(r0_m, v0_mps)` at the real injection point —
/// caller adds the departure body's own heliocentric state for the
/// heliocentric frame, or uses body-relative directly for a body-centered
/// display (see `optimize.rs::pre_departure_orbit_arc`'s MGA branch).
///
/// **Why this is NOT wired into `ms_leg0_start`**: that function seeds
/// multiple shooting's actual Newton solve, and it deliberately sits at the
/// SOI boundary (mild gravity, well-conditioned) rather than periapsis —
/// exactly the same reasoning 9v-ix moved INTERMEDIATE flyby nodes to
/// SOI-entry for (deep-well periapsis nodes make the joint LM problem
/// pathologically stiff, see that section's "why" above). Feeding this
/// function's periapsis-based state into `ms_leg0_start` instead would very
/// likely reintroduce that exact stiffness for the departure leg. Real
/// fixing of multiple shooting's departure seeding — if wanted later — needs
/// the SAME SOI-entry treatment (a departure-side `flyby_soi_entry_state`
/// analogue), not this function directly.
pub fn mga_departure_injection_state(
    params: &[f64],
    cfg: &MissionConfig,
    almanac: &Almanac,
    dep_jd_base: f64,
    flyby_bodies: &[String],
) -> Option<(Vector3<f64>, Vector3<f64>, f64)> {
    let opt = cfg.optimization.as_ref()?;
    let departure_body = opt.departure_body.as_str();
    let ev = evaluate_chromosome_detailed(params, cfg, almanac, dep_jd_base, flyby_bodies)?;
    let (r_dep, v_dep) = ev.body_rvs[0];
    let v_inf_vec = ev.legs.first()?.v_sc_start - v_dep;

    let cat = body_models::TargetBody::by_name(departure_body)?;
    let r_park_m = crate::design::resolve_parking_orbit_radius_m(cfg, &cat);
    let dep = hyperbolic_departure_state(cat.mu_m3s2, r_park_m, v_inf_vec)?;

    let _ = r_dep; // body-relative result; caller adds heliocentric state if needed
    Some((dep.r0_m, dep.v0_mps, dep.dv_escape_ms))
}

/// Heliocentric (r, v) of the flyby periapsis point implied by an incoming/
/// outgoing v∞ pair and a periapsis radius — used to initialize the free
/// multiple-shooting nodes at a physically valid point instead of the flyby
/// body's center (which is singular, see [`ms_leg0_start`]).
///
/// Geometry (hyperbolic flyby), from the true-anomaly picture with the
/// periapsis (eccentricity vector) direction `ê` at ν = 0 and the position
/// asymptotes at ν = ∓ν∞ (cos ν∞ = −1/e): the incoming/outgoing VELOCITY
/// asymptote directions are `û_in = −r̂(−ν∞)`, `û_out = +r̂(+ν∞)`, which
/// gives `û_in − û_out = −2 cos ν∞ · ê` with `−cos ν∞ > 0` — so the
/// periapsis direction is `normalize(û_in − û_out)`. (The first version of
/// this function used `û_out − û_in`, i.e. `−ê`: every node initialized on
/// the OPPOSITE side of the body, a mirror hyperbola with reversed angular
/// momentum whose escape asymptote is multi-km/s wrong — caught 
/// by the `MGA_MS_DEBUG` segment-A diagnostic and pinned down by the
/// Kepler-propagation unit test below. Sanity anchors: δ→π (head-on
/// reflection) must give `r̂_p = +û_in` — periapsis directly in front of
/// the incoming spacecraft — and the formula does.)
/// The periapsis velocity is perpendicular to `ê` in the flyby plane, along
/// `normalize(û_in + û_out)` (exactly perpendicular since both are unit
/// vectors), with magnitude `√(v∞² + 2μ/r_p)` (vis-viva on the hyperbola).
/// Reference: Battin (1999) §6.3 (hyperbolic flyby geometry).
fn flyby_periapsis_state(
    r_body:    Vector3<f64>,
    v_body:    Vector3<f64>,
    v_inf_in:  Vector3<f64>,
    v_inf_out: Vector3<f64>,
    rp_m:      f64,
    mu_body:   f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let u_in  = v_inf_in.normalize();
    let u_out = v_inf_out.normalize();
    // The analytic turn conserves |v∞| exactly; average defensively anyway.
    let v_inf_mag = 0.5 * (v_inf_in.norm() + v_inf_out.norm());

    let apse = u_in - u_out;
    let r_hat = if apse.norm() > 1e-8 {
        apse.normalize()
    } else {
        // Degenerate: essentially no turn — any perpendicular to û_in works.
        let ref_vec = if u_in.x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        u_in.cross(&ref_vec).normalize()
    };
    let tangent = u_in + u_out;
    let v_hat = if tangent.norm() > 1e-8 {
        tangent.normalize()
    } else {
        // Degenerate: 180° turn (r_hat = û_in there, so crossing with û_in
        // is useless) — any direction perpendicular to r_hat works.
        let ref_vec = if r_hat.x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        r_hat.cross(&ref_vec).normalize()
    };

    let v_p = (v_inf_mag * v_inf_mag + 2.0 * mu_body / rp_m).sqrt();
    (r_body + r_hat * rp_m, v_body + v_hat * v_p)
}

/// Body-relative (r, v) of a hyperbolic flyby's SOI-ENTRY point along the
/// INCOMING asymptote branch, plus the analytic entry-to-periapsis transit
/// time \[s\], used to initialize (and time-align) the multiple-shooting free
/// nodes (Phase 9x-iv rework) — SUPERSEDES placing the node at
/// periapsis (`flyby_periapsis_state`, still used elsewhere for the initial
/// DE-search chromosome evaluation and left untouched).
///
/// ## Why periapsis nodes are wrong (diagnosed `docs/ROADMAP_ARCHIVE.md`
/// "CRITICAL correction to 9v-ix")
///
/// A leg TERMINATING at periapsis sits deep in the gravity well (order
/// 10 km/s local speed for a real planetary flyby) where the end-velocity is
/// hypersensitive to mid-course perturbations: a fraction-of-a-m/s change
/// over a multi-hundred-day leg can move the B-plane by thousands of km at a
/// periapsis of only a few thousand km, i.e. a multi-km/s end-velocity
/// change. No practical Levenberg-Marquardt damping level captures this —
/// the joint Newton solve's position residuals converge to km-scale while
/// velocity-continuity residuals stall at hundreds to thousands of m/s.
/// Classical MGA multiple shooting avoids this by placing patch points at
/// SOI boundaries (mild-field region, v ~ v∞ scale — well-conditioned), not
/// at periapsis — this function is that fix.
///
/// ## Formulation
///
/// The full hyperbolic passage (SOI entry → periapsis → SOI exit) is left
/// entirely to the DEPARTING leg's own segment-A real-dynamics propagation
/// (the flyby body is already a registered N-body perturber with SOI
/// switching there) — so exactly one SOI passage sits between a node and
/// its constraint, instead of the node itself sitting at the stiffest point
/// of the well. The node's (r, v) IS the arriving leg's own end state (no
/// separate impulsive turn at the node — the turn emerges from the
/// departing leg's dynamics), so both adjacent legs share one physical
/// state at one epoch.
///
/// Geometry: build the periapsis state via [`flyby_periapsis_state`] (body-
/// relative, i.e. called with `r_body = v_body = 0`), then Kepler-propagate
/// it BACKWARD by the analytic entry-to-periapsis transit time to reach
/// `|r| = soi_radius_m` on the incoming branch. The transit time comes from
/// the standard hyperbolic Kepler equation (Curtis (2013), *Orbital
/// Mechanics for Engineering Students*, 3rd ed., §3.5, eqs. 3.35/3.41b/3.45,
/// hyperbolic trajectories):
///
/// ```text
/// a  = -μ / v∞²                                  (< 0, hyperbolic)
/// e  = 1 - rp/a                                  (= 1 + rp·v∞²/μ)
/// p  = rp·(1+e)                                  (semi-latus rectum)
/// cos(ν_soi) = (p/r_soi - 1) / e                 (true anomaly at r = soi)
/// tanh(F/2)  = tan(ν_soi/2) · sqrt[(e-1)/(e+1)]   (hyperbolic eccentric anomaly)
/// Mh = e·sinh(F) - F                              (hyperbolic mean anomaly)
/// t - t_peri = Mh / n_h,  n_h = sqrt(μ / (-a)³)   (mean motion)
/// ```
///
/// The incoming branch has `ν < 0` (before periapsis), so `t - t_peri < 0`;
/// `transit_time_s = t_peri - t = -(t - t_peri)` is returned positive
/// (duration from SOI entry to periapsis).
///
/// Returns `(r_entry_rel, v_entry_rel, transit_time_s)`, all body-relative —
/// the caller adds the flyby body's own heliocentric state AT THE SHIFTED
/// (entry) EPOCH, not at the periapsis epoch (the body moves during the
/// transit, so using the periapsis-epoch body state here would reintroduce
/// a real position error at the km-to-1000s-of-km scale this rework exists
/// to avoid).
fn flyby_soi_entry_state(
    v_inf_in:  Vector3<f64>,
    v_inf_out: Vector3<f64>,
    rp_m:      f64,
    mu_body:   f64,
    soi_radius_m: f64,
) -> (Vector3<f64>, Vector3<f64>, f64) {
    let (r_p, v_p) = flyby_periapsis_state(
        Vector3::zeros(), Vector3::zeros(), v_inf_in, v_inf_out, rp_m, mu_body);

    let v_inf_mag = 0.5 * (v_inf_in.norm() + v_inf_out.norm());
    // a < 0 for a hyperbola; guard a degenerate (near-zero) v_inf defensively
    // — should not occur for a real flyby (v_inf_mag is bounded away from 0
    // by the chromosome's departure_vinf_max_ms/arrival_vinf_max_ms bounds).
    let a = -mu_body / v_inf_mag.max(1.0).powi(2);
    let e = (1.0 - rp_m / a).max(1.0 + 1e-9); // guard e > 1 (must be hyperbolic)

    let soi_r = soi_radius_m.max(rp_m * 1.01); // guard soi strictly outside periapsis
    let p = rp_m * (1.0 + e);
    let cos_nu = ((p / soi_r) - 1.0) / e;
    let nu = -cos_nu.clamp(-1.0, 1.0).acos(); // negative: before periapsis (inbound)

    let tan_half_f = ((e - 1.0) / (e + 1.0)).sqrt() * (nu / 2.0).tan();
    let f_h = 2.0 * tan_half_f.atanh();
    let m_h = e * f_h.sinh() - f_h;
    let n_h = (mu_body / (-a).powi(3)).sqrt();
    let transit_time_s = -(m_h / n_h); // positive: entry precedes periapsis

    match propagate_kepler(r_p, v_p, -transit_time_s, mu_body) {
        Some((r_entry, v_entry)) => (r_entry, v_entry, transit_time_s),
        None => {
            // Should not happen for a well-posed hyperbola — fall back to
            // the periapsis state itself with zero transit time rather than
            // panicking; the joint solve will simply see a slightly-stiffer
            // node for this one flyby.
            eprintln!("Warning: flyby_soi_entry_state: backward Kepler propagation failed — using periapsis state directly");
            (r_p, v_p, 0.0)
        }
    }
}

/// Same as [`flyby_soi_entry_state`], but bisects `soi_radius_m` DOWN when
/// the full Laplace SOI would produce a transit time exceeding
/// `max_transit_s` (found: a real, reproducible crash — Jupiter's
/// SOI radius is so large (~48 million km, vs. Earth's ~924,000 km) that its
/// entry-to-periapsis transit time can exceed the ENTIRE time budget
/// [`ms_leg_timing`] has available for the arriving leg's segment B, driving
/// that segment's duration negative and making `propagate()` fail outright —
/// not a convergence problem, a hard crash at iteration 0. Post-hoc clamping
/// the returned `transit_time_s` alone would desync the node's POSITION
/// (still at the true, distant SOI boundary) from its EPOCH (shifted by only
/// the clamped time) — a real physical inconsistency. Bisecting the INPUT
/// radius instead keeps position/velocity/transit_time_s mutually consistent
/// at every step, at the cost of a smaller, non-canonical "entry" radius for
/// this one flyby when its own leg genuinely can't afford the real SOI's
/// transit time. `transit_time_s` is monotonically increasing in
/// `soi_radius_m` (a larger entry radius is strictly farther from periapsis
/// on the incoming asymptote), and shrinks to ~0 as the radius shrinks to
/// `rp_m` — so a solution respecting `max_transit_s` always exists for any
/// `max_transit_s > 0`.
fn flyby_soi_entry_state_time_limited(
    v_inf_in:  Vector3<f64>,
    v_inf_out: Vector3<f64>,
    rp_m:      f64,
    mu_body:   f64,
    soi_radius_m: f64,
    max_transit_s: f64,
) -> (Vector3<f64>, Vector3<f64>, f64) {
    let full = flyby_soi_entry_state(v_inf_in, v_inf_out, rp_m, mu_body, soi_radius_m);
    if full.2 <= max_transit_s || max_transit_s <= 0.0 {
        return full;
    }
    let mut lo = rp_m * 1.01; // transit_time_s -> ~0 here (entry ~= periapsis)
    let mut hi = soi_radius_m;
    let mut best = flyby_soi_entry_state(v_inf_in, v_inf_out, rp_m, mu_body, lo);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let candidate = flyby_soi_entry_state(v_inf_in, v_inf_out, rp_m, mu_body, mid);
        if candidate.2 <= max_transit_s {
            best = candidate;
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) / hi.max(1.0) < 1e-6 { break; }
    }
    best
}

/// Per-leg absolute start epoch [s] and segment-A/segment-B durations [s],
/// accounting for the SOI-entry node shift (Phase 9x-iv rework).
///
/// On the original (periapsis-node / chromosome) schedule, leg `k` starts at
/// `ev.legs[k].t_start_days` and splits into segment A (`eta_k·tof_k`, start
/// → DSM) and segment B (`(1-eta_k)·tof_k`, DSM → end) — the DSM epoch is
/// what every downstream leg's own schedule is built relative to, and must
/// stay fixed. Moving an intermediate node from periapsis to SOI entry
/// shifts that node's epoch EARLIER by `transit_time_s[node]` (time from
/// entry to periapsis on the incoming hyperbola) — applied purely locally,
/// with no cascading to any other leg's timing:
///
/// - The ARRIVING leg (this leg ends at node `k`, `k < n-1`): segment B is
///   shortened by `transit_time_s[k]` — it now stops at the earlier SOI-entry
///   epoch instead of continuing on to periapsis. Segment A (DSM epoch) is
///   untouched.
/// - The DEPARTING leg (this leg starts at node `k-1`, `k >= 1`): starts
///   `transit_time_s[k-1]` earlier, but segment A is lengthened by the same
///   amount — so its own DSM epoch (start + segment A) lands at EXACTLY the
///   original, unshifted epoch. The full hyperbolic passage (entry →
///   periapsis → exit) is thus absorbed inside this leg's segment-A
///   propagation, per `flyby_soi_entry_state`'s doc comment.
///
/// Leg 0 (`k == 0`) gets the exact same treatment via `dep_transit_s`
/// ([`ms_leg0_start`]'s real periapsis-burn rework) — the
/// mission's own departure leg, absorbing the real burn → SOI-exit passage
/// into its own segment A, same as any other departing leg above.
///
/// These two adjustments are independent per node and do not compound
/// across legs — each leg's OWN end epoch (DSM and beyond) is unaffected by
/// its own start-side adjustment, so the final target's fixed arrival epoch
/// (queried once, in `evaluate_chromosome_detailed`) is never disturbed.
#[inline]
fn ms_leg_timing(params: &[f64], ev: &ChromosomeEval, k: usize, n: usize, transit_time_s: &[f64], dep_transit_s: f64) -> (f64, f64, f64) {
    let tof_k = tof_days(params, k);
    let eta_k = eta(params, k);
    let dt_a_extra    = if k > 0     { transit_time_s[k - 1] } else { dep_transit_s };
    let dt_b_reduction = if k < n - 1 { transit_time_s[k] } else { 0.0 };
    let t0_abs_a = ev.legs[k].t_start_days * 86_400.0 - dt_a_extra;
    let dt_a = eta_k * tof_k * 86_400.0 + dt_a_extra;
    let dt_b = (1.0 - eta_k) * tof_k * 86_400.0 - dt_b_reduction;
    if dt_b <= 0.0 && std::env::var("MGA_MS_DEBUG").is_ok() {
        eprintln!("[MS_DEBUG] leg {k}: segment B duration went non-positive ({dt_b:.1} s) — transit_time_s[{k}] ({dt_b_reduction:.1} s) exceeds the leg's own (1-eta)*tof span");
    }
    (t0_abs_a, dt_a, dt_b)
}

/// Pre-solve each leg's DSM ΔV vector so the leg's REAL-dynamics propagation
/// hits its target node position, before the joint LM solve starts
///. The analytic DSM vectors are computed against Keplerian
/// sub-arcs; applying them verbatim to the real-dynamics segment-A endpoint
/// leaves position errors that the long legs amplify to AU scale (the
/// 729-day resonant Earth-Earth leg sweeps 2 extra revolutions — a small
/// energy/phase offset at the DSM becomes a ~0.1-AU-per-revolution drift),
/// and the joint LM then starts so deep in the nonlinear regime it stalls.
/// This is a per-leg 3-unknown/3-equation Newton (df/d(dv) by finite
/// differences, LM-damped on singular geometry): segment A is dv-independent
/// and propagated once per leg; each iteration costs only seg-B
/// re-propagations. Position-only — the joint solve still owns the velocity
/// continuity (the flyby physics) and the final polish.
fn ms_presolve_dsms(
    x: &mut [f64],
    ev: &ChromosomeEval,
    params: &[f64],
    flyby_bodies: &[String],
    departure_body: &str,
    cfg: &MissionConfig,
    almanac: &Almanac,
    transit_time_s: &[f64],
) {
    let n = n_legs(flyby_bodies);
    let mut all_perturber_names: Vec<String> = vec![departure_body.to_string()];
    all_perturber_names.extend_from_slice(flyby_bodies);
    let perturber_entries = build_mga_perturber_entries(&all_perturber_names, almanac, ev.dep_jd);
    let perturbers = as_propagator_bodies(&perturber_entries);
    let leg0_start = ms_leg0_start(ev, params, departure_body, cfg, almanac);

    const PRESOLVE_MAX_ITER: usize = 15;
    const PRESOLVE_TOL_M: f64 = 1.0e6; // 1000 km — plenty for an initial guess
    const PRESOLVE_FD_EPS_MS: f64 = 0.1;
    const PRESOLVE_MAX_STEP_MS: f64 = 200.0;
    /// Hard cap on total deviation from the analytic seed dv. Without this,
    /// a position-only 3x3 Newton happily converges onto a DIFFERENT Lambert
    /// branch that also hits the target position — observed on the first
    /// presolve attempt: leg 3 reached 0.0 km position error
    /// via a 17.9 km/s DSM (seed: 13.9 m/s), wrecking the joint solve's
    /// velocity-continuity start. The presolve's job is to correct the
    /// Kepler-vs-real-dynamics gap NEAR the seed, not to re-solve the leg.
    const PRESOLVE_TRUST_RADIUS_MS: f64 = 1_500.0;
    const PRESOLVE_BACKTRACK_TRIES: usize = 4;

    for k in 0..n {
        let (t0_abs_a, dt_a, dt_b) = ms_leg_timing(params, ev, k, n, transit_time_s, leg0_start.2);

        let (r_sc, v_sc) = if k == 0 {
            (leg0_start.0, leg0_start.1)
        } else {
            ms_node(x, n, k - 1)
        };

        // Segment A once — independent of the DSM.
        let sample_dt_a = (dt_a / REPROP_SAMPLES_PER_SEGMENT as f64).max(1.0);
        let seg_a = propagate(
            r_sc, v_sc, t0_abs_a, dt_a,
            MU_SUN_M3S2, &perturbers, sample_dt_a, REPROP_RTOL, REPROP_ATOL,
        );
        let Some(last_a) = seg_a.last() else { continue };
        let r_dsm = last_a.r_m;
        let v_dsm_before = last_a.v_mps;
        let t0_abs_b = t0_abs_a + dt_a;

        if std::env::var("MGA_MS_DEBUG").is_ok() {
            // Where did the REAL segment A end vs. the analytic DSM point?
            // Separates "the escape from the node went wrong" (seg A far
            // off) from "the post-DSM coast diverges" (seg A close).
            let r_dsm_analytic = ev.legs[k].leg.r_dsm_m;
            let dt_covered = last_a.t_s - t0_abs_a;
            eprintln!(
                "[MS_DEBUG] leg {k} segA: covered {:.2}/{:.2} d, end vs analytic DSM = {:.1} km, v_end vs analytic v_dsm = {:.1} m/s",
                dt_covered / 86_400.0, dt_a / 86_400.0,
                (r_dsm - r_dsm_analytic).norm() / 1000.0,
                (v_dsm_before - ev.legs[k].leg.v_dsm_before_mps).norm(),
            );
        }

        let r_target = if k < n - 1 {
            ms_node(x, n, k).0
        } else {
            ev.body_rvs[n].0
        };

        let sample_dt_b = (dt_b / REPROP_SAMPLES_PER_SEGMENT as f64).max(1.0);
        let end_pos = |dv: Vector3<f64>| -> Option<Vector3<f64>> {
            let seg_b = propagate(
                r_dsm, v_dsm_before + dv, t0_abs_b, dt_b,
                MU_SUN_M3S2, &perturbers, sample_dt_b, REPROP_RTOL, REPROP_ATOL,
            );
            seg_b.last().map(|p| p.r_m)
        };

        let dv_seed = ms_dv(x, k);
        let mut dv = dv_seed;
        let seed_err = match end_pos(dv) {
            Some(r_end) => (r_end - r_target).norm(),
            None => { println!("  Presolve leg {k}: seed propagation failed — keeping analytic dv"); continue }
        };
        let mut cur_err = seed_err;
        let mut best = (dv, cur_err);

        'newton: for _ in 0..PRESOLVE_MAX_ITER {
            if cur_err < PRESOLVE_TOL_M { break; }
            let Some(r_end) = end_pos(dv) else { break };
            let f = r_end - r_target;

            // FD Jacobian df/d(dv), 3x3.
            let mut jac = nalgebra::Matrix3::<f64>::zeros();
            for c in 0..3 {
                let mut dvp = dv;
                dvp[c] += PRESOLVE_FD_EPS_MS;
                let Some(r_p) = end_pos(dvp) else { break 'newton };
                jac.set_column(c, &((r_p - r_end) / PRESOLVE_FD_EPS_MS));
            }

            // Lightly-damped Newton step, capped per-iteration.
            let jt = jac.transpose();
            let mut a = jt * jac;
            for i in 0..3 { a[(i, i)] *= 1.0 + 1e-6; }
            let Some(step0) = a.lu().solve(&(-jt * f)) else { break };
            let step_norm = step0.norm();
            let mut step = if step_norm > PRESOLVE_MAX_STEP_MS {
                step0 * (PRESOLVE_MAX_STEP_MS / step_norm)
            } else { step0 };

            // Improving-only with backtracking: reject any step that raises
            // the position error (halve up to PRESOLVE_BACKTRACK_TRIES);
            // stop if nothing improves. This is what keeps a divergent leg
            // (first attempt's leg 1: 1.5e8 km) pinned at its seed instead
            // of running away.
            let mut improved = false;
            for _ in 0..PRESOLVE_BACKTRACK_TRIES {
                let mut dv_trial = dv + step;
                // Trust region: project back onto the ball around the seed.
                let dev = dv_trial - dv_seed;
                if dev.norm() > PRESOLVE_TRUST_RADIUS_MS {
                    dv_trial = dv_seed + dev * (PRESOLVE_TRUST_RADIUS_MS / dev.norm());
                }
                if let Some(r_trial) = end_pos(dv_trial) {
                    let err_trial = (r_trial - r_target).norm();
                    if err_trial < cur_err {
                        dv = dv_trial;
                        cur_err = err_trial;
                        if err_trial < best.1 { best = (dv_trial, err_trial); }
                        improved = true;
                        break;
                    }
                }
                step *= 0.5;
            }
            if !improved { break; }
        }

        x[3*k] = best.0.x; x[3*k+1] = best.0.y; x[3*k+2] = best.0.z;
        println!("  Presolve leg {k}: position residual {:.1} km -> {:.1} km  (dv {:.1} -> {:.1} m/s)",
            seed_err / 1000.0, best.1 / 1000.0, dv_seed.norm(), best.0.norm());
    }
}

/// Free-variable layout for true multiple shooting (Phase 9v-ix):
///
/// `x = [ dv_0, dv_1, ..., dv_{N-1},  node_1, node_2, ..., node_{N-1} ]`
///
/// where each `dv_k` is a 3-vector (DSM ΔV for leg k, [m/s]) and each
/// `node_m` (m = 1..N-1, one per intermediate flyby body) is a 6-vector
/// `(r [m], v [m/s])` — the spacecraft's heliocentric state at that flyby's
/// fixed encounter epoch. `node_0` (departure) and `node_N` (target arrival)
/// are NOT free: they are the fixed states already computed by the analytic
/// chromosome evaluation (`ev.legs[0]` for departure, `ev.body_rvs[n]` for
/// the target). Total unknowns: `3N + 6(N-1) = 9N - 6`.
#[inline]
fn ms_dv(x: &[f64], k: usize) -> Vector3<f64> {
    Vector3::new(x[3 * k], x[3 * k + 1], x[3 * k + 2])
}

/// Node state for intermediate flyby `m` (0-indexed among the N-1 flyby
/// bodies; this is node `m+1` in the 0..N node numbering). Only valid for
/// `m in 0..n-1`.
#[inline]
fn ms_node(x: &[f64], n: usize, m: usize) -> (Vector3<f64>, Vector3<f64>) {
    let base = 3 * n + 6 * m;
    (
        Vector3::new(x[base], x[base + 1], x[base + 2]),
        Vector3::new(x[base + 3], x[base + 4], x[base + 5]),
    )
}

/// Number of free variables: `9N - 6` (3 DSM components per leg + 6 node
/// state components per intermediate flyby).
fn ms_free_dim(n: usize) -> usize { 9 * n - 6 }

/// Number of constraint equations: `6(N-1) + 3` — position+velocity
/// continuity at each of the N-1 intermediate nodes, plus a position-only
/// match to the fixed target at the final leg. Underdetermined by `3(N-1)`
/// relative to `ms_free_dim` — solved via minimum-norm Newton step.
fn ms_constraint_dim(n: usize) -> usize { 6 * (n - 1) + 3 }

/// Real, non-degenerate aim POINT for the final leg's position constraint
/// (Phase 9y/12) — replaces the target body's literal center.
///
/// Before this fix, `ms_constraint`'s final-leg block targeted
/// `ev.body_rvs[n].0` (the body's own center) directly for every mission
/// objective, with no floor at all. A real propagated trajectory converging
/// onto that is a genuine collision-course targeting bug, not just a
/// missing "targeting feature" — the exact same "raw Lambert-to-body-center
/// solve can produce an arbitrarily deep, numerically extreme close pass"
/// failure mode already flagged (and left as a known, accepted limitation
/// for the direct/single-leg path) in Phase 8h's own notes.
///
/// Offsets the aim point away from the body's center by the effective
/// target radius each objective already uses for its ΔV pricing
/// ([`arrival_dv_ms`] — this function intentionally mirrors that function's
/// exact floor/default logic so the two can never diverge), along a
/// direction perpendicular to the incoming arrival v∞. Per Phase 12b's own
/// finding, WHICH perpendicular direction is chosen costs no extra ΔV in
/// this patched-conic model (unlike an intermediate flyby, the final
/// encounter has no downstream leg for the aim direction to redirect into)
/// — so an arbitrary but fixed, well-defined convention (same
/// least-aligned-axis trick [`hyperbolic_departure_state`] already uses) is
/// sufficient; no new chromosome dimension is spent searching for it.
/// `v_inf_arr` comes from the already-converged, `x`-independent analytic
/// evaluation, so this point is fixed for the whole multiple-shooting
/// solve, exactly like `ev.body_rvs[n].0` was fixed before — the LM
/// solver's finite-difference Jacobian is unaffected by this change in kind.
///
/// `Rendezvous` is deliberately excluded (zero offset, still targets the
/// body's exact center) — a real rendezvous wants position co-location, not
/// a stand-off distance.
fn ms_final_target_point(ev: &ChromosomeEval, cfg: &MissionConfig) -> Vector3<f64> {
    let r_body = ev.body_rvs[ev.body_rvs.len() - 1].0;
    let body_radius_m = ev.body_radii.last().copied().unwrap_or(1.0e6);
    let offset_m = match cfg.mission.objective {
        MissionObjective::Flyby => {
            let target_r = cfg.trajectory.capture.as_ref()
                .and_then(|c| c.target_orbit_radius_m)
                .unwrap_or(1.05 * body_radius_m);
            target_r.max(1.05 * body_radius_m)
        }
        MissionObjective::Rendezvous => 0.0,
        _ => {
            let r_cap_configured = cfg.trajectory.capture.as_ref()
                .and_then(|c| c.target_orbit_radius_m)
                .unwrap_or_else(|| body_radius_m * 3.0);
            r_cap_configured.max(body_radius_m)
        }
    };
    if offset_m <= 0.0 {
        return r_body;
    }
    let v_inf_hat = if ev.v_inf_arr.norm() > 1.0 {
        ev.v_inf_arr.normalize()
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let reference = if v_inf_hat.z.abs() < 0.9 { Vector3::z() } else { Vector3::x() };
    let offset_hat = v_inf_hat.cross(&reference).normalize();
    r_body + offset_hat * offset_m
}

/// Real heliocentric state at the multiple-shooting-converged final leg's
/// end, obtained by re-propagating ONLY that leg (segment A + B) from its
/// converged start state — mirrors [`ms_reprop_arc`]'s own `k = N-1`
/// iteration exactly, but returns the raw state instead of plot points.
///
/// Used to seed a REAL captured-orbit visualization for Orbit/Landing MGA
/// results — previously approximated by the target body's own
/// heliocentric orbital plane, because no real propagated crossing state
/// existed before [`ms_final_target_point`] gave the final leg a real,
/// non-degenerate point to converge onto in the first place. Mirrors the
/// single-leg path's own `eval.arrival.r_rel_m`/`v_rel_mps`, which already
/// comes from a genuine propagated crossing.
///
/// Returns `(r_sc_end_helio_m, v_sc_end_helio_mps, t_abs_end_s)`, or `None`
/// if the final leg's re-propagation produces an empty result.
fn ms_final_arrival_state(
    x: &[f64],
    ev: &ChromosomeEval,
    params: &[f64],
    flyby_bodies: &[String],
    departure_body: &str,
    cfg: &MissionConfig,
    almanac: &Almanac,
    transit_time_s: &[f64],
) -> Option<(Vector3<f64>, Vector3<f64>, f64)> {
    let n = n_legs(flyby_bodies);
    let mut all_perturber_names: Vec<String> = vec![departure_body.to_string()];
    all_perturber_names.extend_from_slice(flyby_bodies);
    let perturber_entries = build_mga_perturber_entries(&all_perturber_names, almanac, ev.dep_jd);
    let perturbers = as_propagator_bodies(&perturber_entries);
    let leg0_start = ms_leg0_start(ev, params, departure_body, cfg, almanac);

    let k = n - 1;
    let (r_sc, v_sc) = if k == 0 {
        (leg0_start.0, leg0_start.1)
    } else {
        ms_node(x, n, k - 1)
    };
    let (t0_abs_a, dt_a, dt_b) = ms_leg_timing(params, ev, k, n, transit_time_s, leg0_start.2);

    let sample_dt_a = (dt_a / REPROP_SAMPLES_PER_SEGMENT as f64).max(1.0);
    let seg_a = propagate(
        r_sc, v_sc, t0_abs_a, dt_a,
        MU_SUN_M3S2, &perturbers, sample_dt_a, REPROP_RTOL, REPROP_ATOL,
    );
    let last_a = seg_a.last()?;
    let v_dsm_after = last_a.v_mps + ms_dv(x, k);
    let t0_abs_b = t0_abs_a + dt_a;

    let sample_dt_b = (dt_b / REPROP_SAMPLES_PER_SEGMENT as f64).max(1.0);
    let seg_b = propagate(
        last_a.r_m, v_dsm_after, t0_abs_b, dt_b,
        MU_SUN_M3S2, &perturbers, sample_dt_b, REPROP_RTOL, REPROP_ATOL,
    );
    let last_b = seg_b.last()?;
    Some((last_b.r_m, last_b.v_mps, last_b.t_s))
}

/// Real body-relative arrival state at the multiple-shooting-converged final
/// leg's end — [`ms_final_arrival_state`]'s heliocentric result minus the
/// target body's own real ephemeris state at that exact epoch. `None` if
/// the re-propagation fails, the mission has no `[optimization]` section,
/// or the target body isn't ANISE-covered.
fn ms_arrival_relative_state(
    x: &[f64],
    ev: &ChromosomeEval,
    params: &[f64],
    flyby_bodies: &[String],
    departure_body: &str,
    cfg: &MissionConfig,
    almanac: &Almanac,
    transit_time_s: &[f64],
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let (r_end, v_end, t_end_s) = ms_final_arrival_state(
        x, ev, params, flyby_bodies, departure_body, cfg, almanac, transit_time_s)?;
    let opt = cfg.optimization.as_ref()?;
    let target_anise = anise_body(&opt.target_body.to_lowercase())?;
    let jd = ev.dep_jd + t_end_s / 86_400.0;
    let (r_body, v_body) = body_state(almanac, EphemerisSource::Anise, Some(target_anise), &None, jd)?;
    let r_body_v = Vector3::new(r_body[0], r_body[1], r_body[2]);
    let v_body_v = Vector3::new(v_body[0], v_body[1], v_body[2]);
    Some((r_end - r_body_v, v_end - v_body_v))
}

/// Evaluate the true multiple-shooting constraint (Phase 9v-ix).
///
/// Each leg `k` is propagated INDEPENDENTLY from its own node state — the
/// fixed departure state for `k = 0`, otherwise the free node `k` variable —
/// never chained through a prior leg's propagated end state. This is what
/// makes it *true* multiple shooting rather than the earlier single-shooting-
/// in-disguise formulation, whose finite-difference Jacobian became
/// astronomically ill-conditioned beyond one flyby (an upstream DSM
/// perturbation could only reach a downstream constraint by surviving
/// amplification through every intervening hyperbolic SOI passage).
///
/// The departure body and all flyby bodies remain registered as real N-body
/// gravitational perturbers with SOI-switching for every leg's propagation
/// (unchanged from the prior formulation), so the gravity-assist turn at
/// each flyby still emerges from the dynamics — it now happens within the
/// one leg whose node sits at that body's vicinity (per the analytic
/// chromosome's initial guess), rather than by chaining sensitivity through
/// an explicit periapsis-targeting constraint. Flyby geometry (periapsis
/// radius, B-plane angle) is therefore implicit in the converged node
/// states, not a frozen input — the `rp_norm`/`beta` chromosome parameters
/// are no longer read by this function at all.
///
/// Constraint block for leg `k < N-1`: propagated end state vs. free node
/// `k+1`, position [m] (3) + velocity [m/s] (3) — velocity continuity is
/// what makes node `k+1` a genuine unpowered flyby patch point rather than
/// an arbitrary free position. Constraint block for the final leg
/// (`k = N-1`): position-only [m] (3) match to a real, non-degenerate aim
/// point offset from the target body's center — see
/// [`ms_final_target_point`]'s doc comment for why this is not the body's
/// exact center.
///
/// Returns `None` if any propagation segment produces an empty result.
fn ms_constraint(
    x: &[f64],
    ev: &ChromosomeEval,
    params: &[f64],
    flyby_bodies: &[String],
    departure_body: &str,
    cfg: &MissionConfig,
    almanac: &Almanac,
    transit_time_s: &[f64],
) -> Option<Vec<f64>> {
    let n = n_legs(flyby_bodies);
    let mut f_vec = vec![0.0_f64; ms_constraint_dim(n)];

    // Build N-body perturbers from the departure body + all flyby bodies: real
    // gravitational bodies with SOI-switching so the gravity assist trajectory
    // emerges from the dynamics.
    let mut all_perturber_names: Vec<String> = vec![departure_body.to_string()];
    all_perturber_names.extend_from_slice(flyby_bodies);
    let perturber_entries = build_mga_perturber_entries(&all_perturber_names, almanac, ev.dep_jd);
    let perturbers = as_propagator_bodies(&perturber_entries);
    let leg0_start = ms_leg0_start(ev, params, departure_body, cfg, almanac);

    let mut f_offset = 0usize;

    for k in 0..n {
        // ── Leg-start state: fixed departure for k=0, else free node k ─────
        // Leg 0 starts at the departure body's real periapsis-burn state
        // (rework), NOT at the body's center (singular — see
        // ms_leg0_start's doc comment). Intermediate nodes are the SOI-
        // ENTRY state (Phase 9x-iv rework) — see [`ms_leg_timing`] for how
        // the per-leg segment durations/start epoch absorb both shifts.
        let (r_sc, v_sc) = if k == 0 {
            (leg0_start.0, leg0_start.1)
        } else {
            ms_node(x, n, k - 1)
        };
        let (t0_abs_a, dt_a, dt_b) = ms_leg_timing(params, ev, k, n, transit_time_s, leg0_start.2);

        // ── Segment A: leg-start → DSM ─────────────────────────────────────
        let sample_dt_a = (dt_a / REPROP_SAMPLES_PER_SEGMENT as f64).max(1.0);
        let seg_a = propagate(
            r_sc, v_sc, t0_abs_a, dt_a,
            MU_SUN_M3S2, &perturbers, sample_dt_a, REPROP_RTOL, REPROP_ATOL,
        );
        if seg_a.is_empty() { return None; }

        let last_a = seg_a.last().unwrap();
        let r_dsm_real = last_a.r_m;
        let v_dsm_before = last_a.v_mps;

        let v_dsm_after = v_dsm_before + ms_dv(x, k);
        let t0_abs_b = t0_abs_a + dt_a;

        // ── Segment B: DSM → leg end ────────────────────────────────────────
        let sample_dt_b = (dt_b / REPROP_SAMPLES_PER_SEGMENT as f64).max(1.0);
        let seg_b = propagate(
            r_dsm_real, v_dsm_after, t0_abs_b, dt_b,
            MU_SUN_M3S2, &perturbers, sample_dt_b, REPROP_RTOL, REPROP_ATOL,
        );
        if seg_b.is_empty() { return None; }

        let last_b = seg_b.last().unwrap();
        let r_sc_end = last_b.r_m;
        let v_sc_end = last_b.v_mps;

        if std::env::var("MGA_MS_DEBUG").is_ok() {
            // Anomaly-only: report a segment that stopped short of its
            // requested duration (early SOI/collision stop) — the discrete
            // event that makes the constraint function discontinuous.
            let covered_b = last_b.t_s - t0_abs_b;
            if covered_b < dt_b - 1.0 {
                eprintln!("[MS_DEBUG] leg {k} segB EARLY STOP: covered {:.3}/{:.3} d",
                    covered_b / 86_400.0, dt_b / 86_400.0);
            }
            let covered_a = seg_a.last().map(|p| p.t_s - t0_abs_a).unwrap_or(0.0);
            if covered_a < dt_a - 1.0 {
                eprintln!("[MS_DEBUG] leg {k} segA EARLY STOP: covered {:.3}/{:.3} d",
                    covered_a / 86_400.0, dt_a / 86_400.0);
            }
        }

        // ── Patch-point constraint ──────────────────────────────────────────
        if k < n - 1 {
            let (r_target, v_target) = ms_node(x, n, k);
            let dr = r_sc_end - r_target;
            let dv = v_sc_end - v_target;
            f_vec[f_offset]     = dr.x;
            f_vec[f_offset + 1] = dr.y;
            f_vec[f_offset + 2] = dr.z;
            f_vec[f_offset + 3] = dv.x;
            f_vec[f_offset + 4] = dv.y;
            f_vec[f_offset + 5] = dv.z;
            f_offset += 6;
        } else {
            let r_target = ms_final_target_point(ev, cfg);
            let dr = r_sc_end - r_target;
            f_vec[f_offset]     = dr.x;
            f_vec[f_offset + 1] = dr.y;
            f_vec[f_offset + 2] = dr.z;
            f_offset += 3;
        }
    }

    Some(f_vec)
}

/// Split a constraint vector produced by [`ms_constraint`] into per-leg
/// position residual magnitudes [m] (length N) and per-intermediate-node
/// velocity residual magnitudes [m/s] (length N-1).
fn ms_split_residuals(f_vec: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut pos = Vec::with_capacity(n);
    let mut vel = Vec::with_capacity(n.saturating_sub(1));
    let mut off = 0usize;
    for k in 0..n {
        let dr = Vector3::new(f_vec[off], f_vec[off + 1], f_vec[off + 2]);
        pos.push(dr.norm());
        if k < n - 1 {
            let dv = Vector3::new(f_vec[off + 3], f_vec[off + 4], f_vec[off + 5]);
            vel.push(dv.norm());
            off += 6;
        } else {
            off += 3;
        }
    }
    (pos, vel)
}

/// Re-propagate the full trajectory using the corrected free variables
/// (per-leg DSM ΔVs + converged node states) and collect arc sample points
/// for plotting. Each leg is propagated independently from its own node,
/// exactly as in [`ms_constraint`] — after convergence the position/velocity
/// gaps between adjacent legs' propagated arcs are within `MS_TOL_M`/
/// `MS_VEL_TOL_MS`, so the plotted arc reads as continuous.
fn ms_reprop_arc(
    x: &[f64],
    ev: &ChromosomeEval,
    params: &[f64],
    flyby_bodies: &[String],
    departure_body: &str,
    cfg: &MissionConfig,
    almanac: &Almanac,
    transit_time_s: &[f64],
) -> Vec<MgaArcPoint> {
    let n = n_legs(flyby_bodies);
    let mut pts: Vec<MgaArcPoint> = Vec::new();

    let mut all_perturber_names: Vec<String> = vec![departure_body.to_string()];
    all_perturber_names.extend_from_slice(flyby_bodies);
    let perturber_entries = build_mga_perturber_entries(&all_perturber_names, almanac, ev.dep_jd);
    let perturbers = as_propagator_bodies(&perturber_entries);
    let leg0_start = ms_leg0_start(ev, params, departure_body, cfg, almanac);

    for k in 0..n {
        // Same real periapsis-burn departure state as ms_constraint — the
        // two must stay consistent or the plotted arc won't be the
        // converged one.
        let (r_sc, v_sc) = if k == 0 {
            (leg0_start.0, leg0_start.1)
        } else {
            ms_node(x, n, k - 1)
        };
        let (t0_abs_a, dt_a, dt_b) = ms_leg_timing(params, ev, k, n, transit_time_s, leg0_start.2);

        let sample_dt_a = (dt_a / REPROP_SAMPLES_PER_SEGMENT as f64).max(1.0);
        let seg_a = propagate(
            r_sc, v_sc, t0_abs_a, dt_a,
            MU_SUN_M3S2, &perturbers, sample_dt_a, REPROP_RTOL, REPROP_ATOL,
        );
        if seg_a.is_empty() { return pts; }

        // `propagate` returns t_s on the ABSOLUTE mission clock (it already
        // includes the t0_abs_s offset).
        for p in &seg_a {
            let t_days = p.t_s / 86_400.0;
            let central_body = p.central_body_index.map(|i| perturber_entries[i].name.clone());
            pts.push(MgaArcPoint { t_days, x_m: p.r_m.x, y_m: p.r_m.y, z_m: p.r_m.z, vx_mps: p.v_mps.x, vy_mps: p.v_mps.y, vz_mps: p.v_mps.z, leg_idx: k, central_body });
        }

        let last_a = seg_a.last().unwrap();
        let r_dsm_real = last_a.r_m;
        let v_dsm_before = last_a.v_mps;
        let v_dsm_after = v_dsm_before + ms_dv(x, k);
        let t0_abs_b = t0_abs_a + dt_a;

        let sample_dt_b = (dt_b / REPROP_SAMPLES_PER_SEGMENT as f64).max(1.0);
        let seg_b = propagate(
            r_dsm_real, v_dsm_after, t0_abs_b, dt_b,
            MU_SUN_M3S2, &perturbers, sample_dt_b, REPROP_RTOL, REPROP_ATOL,
        );
        if seg_b.is_empty() { return pts; }

        for p in seg_b.iter().skip(1) {
            let t_days = p.t_s / 86_400.0;
            let central_body = p.central_body_index.map(|i| perturber_entries[i].name.clone());
            pts.push(MgaArcPoint { t_days, x_m: p.r_m.x, y_m: p.r_m.y, z_m: p.r_m.z, vx_mps: p.v_mps.x, vy_mps: p.v_mps.y, vz_mps: p.v_mps.z, leg_idx: k, central_body });
        }
    }

    pts
}

/// True multiple-shooting refinement of an MGA-1DSM solution (Phase 9v-ix).
///
/// Takes the open-loop `MgaResult` (analytic Keplerian + Rodrigues-turn
/// solution) and refines it into a real N-body trajectory. Free variables
/// are each leg's DSM ΔV vector PLUS the spacecraft's heliocentric state at
/// every intermediate flyby node — see [`ms_constraint`]'s doc comment for
/// the exact layout. Each leg propagates independently from its own node;
/// continuity (position AND velocity at intermediate nodes, position-only
/// at the final target) is enforced as a Newton constraint, not by
/// construction. This is what makes it *true* multiple shooting: an
/// upstream leg's Jacobian sensitivity never has to survive amplification
/// through a downstream leg's hyperbolic SOI passage, because downstream
/// legs are never propagated as a continuation of upstream ones — only
/// tied together by the (initially loose, then Newton-corrected) node
/// variables.
///
/// The flyby (rp, beta) parameters are only read to seed the INITIAL guess
/// (via [`flyby_soi_entry_state`]) — flyby geometry is implicit in the
/// converged node states, not a frozen constraint. All TOFs/eta fractions
/// remain fixed (first-cut simplification); node epochs are also fixed, but
/// no longer equal the chromosome's periapsis-passage schedule directly —
/// see [`ms_leg_timing`] for the (also fixed, precomputed once here) SOI-
/// entry epoch shift.
///
/// The departure body and all flyby bodies are registered as real
/// gravitational perturbers with Laplace SOI switching for every leg's
/// propagation, so the gravity-assist turn still emerges from the dynamics.
///
/// ## Algorithm
///
/// Free variables: `x ∈ ℝ^{9N-6}` — 3N DSM components + 6(N-1) node states.
/// Initial guess: DSM ΔVs from the analytic evaluation; node states at each
/// flyby's SOI-ENTRY point along the incoming asymptote (built from the
/// chromosome's (rp, β) via [`flyby_soi_entry_state`]) — NOT at periapsis
/// (superseded, see that function's doc comment for the
/// hypersensitive-end-velocity diagnosis this fixes) and NOT at the body's
/// center, which is singular since the flyby bodies are registered
/// gravitating perturbers (rework; see [`ms_leg0_start`] for the
/// failure this fixed). Leg 0's fixed departure state likewise sits at the
/// departure body's SOI boundary along the outgoing asymptote, not at the
/// body's center.
/// Constraint: `f(x) ∈ ℝ^{6(N-1)+3}` — position+velocity match at each
/// intermediate node, position-only match at the final target.
/// Jacobian: finite-difference (column-wise perturbation), shape
/// `(6(N-1)+3) × (9N-6)` — rectangular (underdetermined by `3(N-1)`).
/// Position-type and velocity-type entries are non-dimensionalized
/// (`ms_scale_x`/`ms_scale_f`) before every linear-algebra step below —
/// required for numerical sanity given the entries span 7+ orders of
/// magnitude (1 AU vs. km/s), found the hard way (Phase 9v-ix).
/// Step: Levenberg-Marquardt-damped minimum-norm step, `Δx̂ =
/// V·diag(σᵢ/(σᵢ²+λ))·Uᵀ·(-f̂)` from the Jacobian's SVD — a plain
/// (undamped) minimum-norm pseudo-inverse step was found to stall well
/// short of tolerance on a real 2-flyby case (classic Gauss-Newton
/// flat-region failure); λ is adapted globally across outer iterations
/// (shrinks on an accepted step, grows on a rejected one), recovering pure
/// Newton behavior near a well-conditioned solution and gradient-descent-
/// like small steps when the local Jacobian is uninformative.
/// Convergence: max per-leg position residual `< MS_TOL_M` (1 km) AND max
/// per-node velocity residual `< MS_VEL_TOL_MS` (1 m/s).
pub fn run_multiple_shooting(
    result: &MgaResult,
    flyby_bodies: &[String],
    cfg: &MissionConfig,
    almanac: &Almanac,
) -> Result<MultipleShotResult, String> {
    let opt = cfg.optimization.as_ref()
        .ok_or("multiple shooting requires an [optimization] section")?;
    let n = n_legs(flyby_bodies);
    let dim_x = ms_free_dim(n);

    // ── Recover the chromosome evaluation for fixed geometry ─────────────────
    let dep_epoch_str = opt.departure_epoch.as_deref()
        .ok_or("optimization.departure_epoch is required")?;
    let dep_epoch   = crate::design::parse_epoch(dep_epoch_str)
        .map_err(|e| format!("departure_epoch parse: {e}"))?;
    let dep_jd_base = crate::design::epoch_to_jd(dep_epoch);

    let ev = evaluate_chromosome_detailed(&result.best_params, cfg, almanac, dep_jd_base, flyby_bodies)
        .ok_or("best chromosome is infeasible — cannot initialise multiple shooting")?;

    let departure_body = opt.departure_body.as_str();

    // ── Initial guess ─────────────────────────────────────────────────────────
    // DSM ΔV vectors from the Keplerian evaluation (v_dsm_after - v_dsm_before,
    // already computed by `evaluate_mga_leg` and stored per leg), followed by
    // the intermediate node states at each flyby's SOI-ENTRY point (Phase
    // 9x-iv rework — supersedes the periapsis-node formulation,
    // see [`flyby_soi_entry_state`]'s doc comment for why), built from the
    // chromosome's (rp, beta) geometry via the analytic v∞_in/v∞_out pair.
    // The nodes must NOT initialize at the flyby body's center (the analytic
    // evaluation's leg-start states): every flyby body is a registered N-body
    // perturber, so a propagation from its exact center is singular — see
    // ms_leg0_start's doc comment for the failure this produced.
    let mga_params = opt.mga.as_ref();
    let min_rp_m = mga_params.map(|m| m.flyby_min_periapsis_m).unwrap_or(0.0);
    let mut x: Vec<f64> = vec![0.0; dim_x];
    for k in 0..n {
        let dv = ev.legs[k].leg.v_dsm_after_mps - ev.legs[k].leg.v_dsm_before_mps;
        x[3*k] = dv.x; x[3*k+1] = dv.y; x[3*k+2] = dv.z;
    }
    // One entry per intermediate flyby (node m, m = 0..n-2): the analytic
    // entry-to-periapsis transit time [s], used both to build the SOI-entry
    // node state below and — via [`ms_leg_timing`] — to locally shift the
    // adjacent legs' segment durations/start epoch (see that function's doc
    // comment). Fixed once here, from the chromosome's own geometry; never
    // updated during the Newton loop (same "first-cut simplification" as the
    // rest of the node-epoch bookkeeping).
    let mut transit_time_s: Vec<f64> = Vec::with_capacity(n.saturating_sub(1));
    for m in 0..n.saturating_sub(1) {
        let base = 3*n + 6*m;
        let (r_body_peri, v_body_peri) = ev.body_rvs[m + 1];
        // Same minimum-periapsis clamp the graded evaluation applies, so the
        // initial node matches the geometry the chromosome actually flew.
        let rp_m_init = (rp_norm(&result.best_params, n, m) * ev.body_radii[m + 1]).max(min_rp_m);
        let v_inf_in  = ev.legs[m].leg.v_inf_arr_mps;
        let v_inf_out = ev.legs[m + 1].v_sc_start - v_body_peri;
        let mu_body   = ev.body_mus[m + 1];
        let body_name = &flyby_bodies[m];
        let soi_radius_m = body_models::TargetBody::by_name(body_name)
            .and_then(|cat| cat.sma_m.map(|sma| laplace_soi_radius_m(sma, cat.mu_m3s2 / MU_SUN_M3S2)))
            .unwrap_or(1.0e9);
        // Time-budget-limited (see `flyby_soi_entry_state_time_limited`'s
        // doc comment): the full Laplace SOI's transit time can exceed the
        // arriving leg's (leg m's) own segment-B time budget for a
        // giant-planet flyby (real, reproduced crash: Jupiter's ~48M km SOI
        // gave a 58-day transit against a leg with less than that available),
        // driving that segment's duration negative and crashing propagation
        // outright. Capped at 80% of the naive (uncorrected) segment-B span,
        // leaving real margin for `ms_leg_timing`'s subtraction to stay
        // positive.
        let naive_seg_b_s = (1.0 - eta(&result.best_params, m)) * tof_days(&result.best_params, m) * 86_400.0;
        let max_transit_s = 0.8 * naive_seg_b_s.max(0.0);
        let (r_rel, v_rel, transit_s) = flyby_soi_entry_state_time_limited(
            v_inf_in, v_inf_out, rp_m_init, mu_body, soi_radius_m, max_transit_s);
        transit_time_s.push(transit_s);

        // The node's heliocentric state needs the flyby body's own state AT
        // THE SHIFTED (entry) EPOCH, not at the periapsis epoch used above
        // only to derive rp_m_init/v_inf_out — the body moves during the
        // transit, so reusing `r_body_peri`/`v_body_peri` here would
        // reintroduce a real position error at exactly the scale this
        // rework exists to remove.
        // NOTE (bug found): must use `ev.dep_jd` (the ACTUAL
        // departure epoch, with the chromosome's own `dep_offset_days` gene
        // already applied), not the local `dep_jd_base` (the raw window-
        // reference epoch from config, offset NOT applied). `t_start_days`
        // is accumulated from 0.0 at `ev.dep_jd` (see `evaluate_chromosome_graded`:
        // `t_days` starts at 0.0, `dep_jd = dep_jd_base + dep_offset_d`) — using
        // `dep_jd_base` here silently mis-dated every entry-epoch ephemeris
        // query by `dep_offset_days` (up to half the departure window), which
        // for Cassini-2's real chromosome produced hundreds-of-millions-of-km
        // body-position errors and made the presolve/joint-LM start from a
        // wildly wrong point. Caught by an A/B run against the periapsis-node
        // baseline showing catastrophically worse presolve residuals.
        let entry_jd = ev.dep_jd + (ev.legs[m + 1].t_start_days - transit_s / 86_400.0);
        let (r_body_entry, v_body_entry) = anise_body(&body_name.to_lowercase())
            .and_then(|anise| body_state(almanac, EphemerisSource::Anise, Some(anise), &None, entry_jd))
            .map(|(r, v)| (Vector3::new(r[0], r[1], r[2]), Vector3::new(v[0], v[1], v[2])))
            .unwrap_or((r_body_peri, v_body_peri)); // fallback: periapsis-epoch state (small error)

        let r = r_body_entry + r_rel;
        let v = v_body_entry + v_rel;
        x[base] = r.x; x[base+1] = r.y; x[base+2] = r.z;
        x[base+3] = v.x; x[base+4] = v.y; x[base+5] = v.z;
    }

    // Per-leg DSM pre-solve under real dynamics — see
    // ms_presolve_dsms' doc comment. Gives the joint LM a start with
    // ~km-scale position residuals instead of AU-scale ones.
    ms_presolve_dsms(&mut x, &ev, &result.best_params, flyby_bodies, departure_body, cfg, almanac, &transit_time_s);

    println!("  Multiple shooting: {} legs, {} free variables, {} constraints",
        n, dim_x, ms_constraint_dim(n));
    println!("  {:>6}  {:>18}  {:>16}", "Iter", "max ‖pos‖ [m]", "max ‖vel‖ [m/s]");

    let scale_x = ms_scale_x(n);
    let scale_f = ms_scale_f(n);

    let mut iterations = 0usize;
    let mut lambda_lm = MS_LM_LAMBDA_INIT;

    for iter in 0..MAX_MS_ITER {
        // ── Evaluate constraint function ──────────────────────────────────────
        let f_vec = ms_constraint(&x, &ev, &result.best_params, flyby_bodies, departure_body, cfg, almanac, &transit_time_s)
            .ok_or_else(|| format!("propagation failed inside multiple shooting at iteration {iter}"))?;

        let (pos_res, vel_res) = ms_split_residuals(&f_vec, n);
        let pos_max = pos_res.iter().cloned().fold(0.0_f64, f64::max);
        let vel_max = vel_res.iter().cloned().fold(0.0_f64, f64::max);
        println!("  {:>6}  {:>18.3}  {:>16.6}", iter, pos_max, vel_max);

        iterations = iter;

        if pos_max < MS_TOL_M && vel_max < MS_VEL_TOL_MS {
            // Converged.
            println!("  Converged at iteration {iter} — max ‖pos‖ = {pos_max:.1} m, max ‖vel‖ = {vel_max:.4} m/s");

            // Post-check (not a constraint): under the SOI-entry node
            // formulation (Phase 9x-iv), the converged node's own distance
            // to its flyby body is ~`soi_radius_m` by construction — no
            // longer a meaningful periapsis diagnostic. The REAL periapsis
            // distance is now an emergent property of the DEPARTING leg's
            // segment-A propagation (entry → periapsis → exit all happen
            // inside it, per `flyby_soi_entry_state`'s doc comment) — find
            // the minimum body-relative distance actually achieved there
            // and report/warn against the configured floor, same as before.
            //
            // Same loop also measures the REAL ΔV each flyby actually
            // delivered: `propagate`'s
            // returned points are always in ONE consistent (heliocentric)
            // frame regardless of which body was internally central during
            // integration (see its own doc comment — `central_body_index`
            // is a label, not a different coordinate frame) — so this is a
            // direct measurement, not a re-derivation from the analytic
            // v∞-in/v∞-out model. `v_sc` (the node's own converged state) IS
            // the real heliocentric velocity at SOI entry; the real
            // heliocentric velocity at SOI EXIT is found by walking `seg_a`
            // for the point where `central_body_index` stops being this
            // flyby body's index (falls back to the segment's last point if
            // the leg's own sampled span ends before the real exit —
            // physically rare, since SOI transit is hours-days against a
            // leg spanning weeks-months to its DSM). The real ΔV a flyby
            // "achieved" is exactly `|v_helio_exit − v_helio_entry|` — real
            // physics (conservation of momentum with the flyby body), not
            // an equivalent-burn analogy; `speed_before_ms`/`speed_after_ms`
            // additionally let a caller see whether a given flyby ADDED or
            // REMOVED heliocentric energy (the latter is exactly what an
            // inner-planet deceleration sequence, e.g. MESSENGER's resonant
            // Mercury flybys, needs to show up as).
            let mut flyby_dv_gained_ms = Vec::with_capacity(n.saturating_sub(1));
            let mut flyby_speed_before_ms = Vec::with_capacity(n.saturating_sub(1));
            let mut flyby_speed_after_ms = Vec::with_capacity(n.saturating_sub(1));
            {
                let mut all_perturber_names: Vec<String> = vec![departure_body.to_string()];
                all_perturber_names.extend_from_slice(flyby_bodies);
                let perturber_entries = build_mga_perturber_entries(&all_perturber_names, almanac, ev.dep_jd);
                let perturbers = as_propagator_bodies(&perturber_entries);
                for m in 0..n.saturating_sub(1) {
                    let k = m + 1; // departing leg — carries this flyby's real passage
                    let (r_sc, v_sc) = ms_node(&x, n, m);
                    // k = m+1 >= 1 here always, so dep_transit_s is unused (only k==0 reads it).
                    let (t0_abs_a, dt_a, _dt_b) = ms_leg_timing(&result.best_params, &ev, k, n, &transit_time_s, 0.0);
                    let sample_dt_a = (dt_a / REPROP_SAMPLES_PER_SEGMENT as f64).max(1.0);
                    let seg_a = propagate(
                        r_sc, v_sc, t0_abs_a, dt_a,
                        MU_SUN_M3S2, &perturbers, sample_dt_a, REPROP_RTOL, REPROP_ATOL,
                    );
                    let anise = anise_body(&flyby_bodies[m].to_lowercase());
                    let mut min_dist = f64::MAX;
                    for p in &seg_a {
                        let jd = ev.dep_jd + p.t_s / 86_400.0;
                        if let Some((r_b, _)) = anise.and_then(|a| body_state(almanac, EphemerisSource::Anise, Some(a), &None, jd)) {
                            let dist = (p.r_m - Vector3::new(r_b[0], r_b[1], r_b[2])).norm();
                            min_dist = min_dist.min(dist);
                        }
                    }
                    if min_dist.is_finite() {
                        let alt_km = (min_dist - ev.body_radii[m + 1]) / 1000.0;
                        let warn = if min_dist < min_rp_m.max(ev.body_radii[m + 1]) { "  ⚠ BELOW minimum periapsis!" } else { "" };
                        println!("  Node {m}: minimum real-dynamics flyby distance = {:.0} km (altitude {alt_km:.0} km){warn}",
                            min_dist / 1000.0);
                    } else {
                        println!("  Node {m}: could not evaluate real-dynamics flyby distance (ephemeris unavailable)");
                    }

                    // Real ΔV measurement for this flyby.
                    let flyby_idx = perturber_entries.iter().position(|e| e.name == flyby_bodies[m]);
                    let v_exit = flyby_idx.and_then(|idx| {
                        let last_inside = seg_a.iter().rposition(|p| p.central_body_index == Some(idx))?;
                        seg_a.get(last_inside + 1).or_else(|| seg_a.last()).map(|p| p.v_mps)
                    }).unwrap_or_else(|| seg_a.last().map(|p| p.v_mps).unwrap_or(v_sc));
                    let speed_before = v_sc.norm();
                    let speed_after = v_exit.norm();
                    let dv_gained = (v_exit - v_sc).norm();
                    println!("  Node {m}: real ΔV delivered = {dv_gained:.1} m/s (speed {:.1} -> {:.1} m/s, {})",
                        speed_before, speed_after,
                        if speed_after >= speed_before { "energy gained" } else { "energy shed" });
                    flyby_dv_gained_ms.push(dv_gained);
                    flyby_speed_before_ms.push(speed_before);
                    flyby_speed_after_ms.push(speed_after);
                }
            }

            let dv_dsm_corrected: Vec<[f64; 3]> = (0..n).map(|k| {
                let dv = ms_dv(&x, k);
                [dv.x, dv.y, dv.z]
            }).collect();

            let arc = ms_reprop_arc(&x, &ev, &result.best_params, flyby_bodies, departure_body, cfg, almanac, &transit_time_s);

            // Recompute total ΔV with corrected DSM magnitudes.
            let dv_dsm_total: f64 = dv_dsm_corrected.iter()
                .map(|dv| (dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2]).sqrt())
                .sum();
            let dv_total_ms = result.dv_departure_ms + dv_dsm_total + result.dv_arrival_ms;

            let arrival_rel = ms_arrival_relative_state(
                &x, &ev, &result.best_params, flyby_bodies, departure_body, cfg, almanac, &transit_time_s);

            return Ok(MultipleShotResult {
                converged: true,
                iterations,
                residuals_m: pos_res,
                vel_residuals_ms: vel_res,
                dv_dsm_corrected,
                arc,
                dv_total_ms,
                arrival_r_rel_m: arrival_rel.map(|(r, _)| [r.x, r.y, r.z]),
                arrival_v_rel_mps: arrival_rel.map(|(_, v)| [v.x, v.y, v.z]),
                flyby_dv_gained_ms,
                flyby_speed_before_ms,
                flyby_speed_after_ms,
            });
        }

        // ── Build Jacobian via finite differences, NON-DIMENSIONALIZED ────────
        // J is (6(N-1)+3) × (9N-6); raw column j = (f(x + h*e_j) - f_base) / h.
        // Position entries (~1e11 m) and velocity/DSM entries (~1e3-1e4 m/s)
        // span 7+ orders of magnitude — an SVD-based pseudo-inverse on the
        // RAW mixed-unit matrix is numerically dominated by the huge-magnitude
        // columns, so its "minimum norm" solution is minimum in a physically
        // meaningless norm (verified empirically: without scaling, Newton
        // oscillated chaotically instead of converging on a 2-flyby chain).
        // Scale every column/row by `scale_x`/`scale_f` (`MS_LEN_SCALE_M` for
        // position-type, `MS_VEL_SCALE_MS` for velocity-type) before the
        // pseudo-inverse, then rescale the solved step back to physical units.
        let dim_f = f_vec.len();
        let mut j_mat = DMatrix::<f64>::zeros(dim_f, dim_x);
        for j in 0..dim_x {
            // Per-type absolute FD step — see the MS_FD_EPS_* doc comment.
            let h = if j < 3 * n {
                MS_FD_EPS_DV_MS
            } else if (j - 3 * n) % 6 < 3 {
                MS_FD_EPS_POS_M
            } else {
                MS_FD_EPS_VEL_MS
            };
            let mut x_pert = x.clone();
            x_pert[j] += h;
            let f_pert = ms_constraint(&x_pert, &ev, &result.best_params, flyby_bodies, departure_body, cfg, almanac, &transit_time_s)
                .ok_or_else(|| format!("propagation failed in Jacobian column {j} at iteration {iter}"))?;
            for i in 0..dim_f {
                let raw = (f_pert[i] - f_vec[i]) / h;
                j_mat[(i, j)] = raw * scale_x[j] / scale_f[i];
            }
        }

        // ── Levenberg-Marquardt damped step (replaces pure Gauss-Newton) ──────
        // A plain minimum-norm pseudo-inverse step (even scaled + line-
        // searched) was found empirically to stall well short of tolerance on
        // a real 2-flyby case (Phase 9v-ix): the residual dropped
        // 2 orders of magnitude then got stuck — the classic Gauss-Newton
        // flat-region failure mode. LM damping fixes this by regularizing the
        // step: `Δx̂ = V · diag(σ_i/(σ_i²+λ)) · Uᵀ · (-f̂)` (Tikhonov-damped
        // pseudo-inverse via the SVD `J = U·Σ·Vᵀ`, computed once per outer
        // iteration). Large λ shrinks the step toward gradient descent (small,
        // always-improving, escapes flat/degenerate directions); λ → 0
        // recovers the original minimum-norm Newton step. λ is adapted
        // globally across outer iterations (not reset each iteration) —
        // shrink on an accepted step, grow on a rejected one — standard LM.
        let f_hat: Vec<f64> = f_vec.iter().zip(scale_f.iter()).map(|(f, s)| f / s).collect();
        let f_dvec = DVector::from_vec(f_hat);
        let neg_f  = -f_dvec;
        let base_hat_norm = neg_f.norm();

        let svd = SVD::new(j_mat, true, true);
        let u   = svd.u.ok_or_else(|| format!("SVD U missing at iteration {iter}"))?;
        let v_t = svd.v_t.ok_or_else(|| format!("SVD Vᵀ missing at iteration {iter}"))?;
        let sv  = svd.singular_values;
        let k_sv = sv.len();
        let ut_negf = u.transpose() * &neg_f; // length k_sv

        let mut accepted = false;
        for _try in 0..MS_LM_MAX_TRIES {
            let d: Vec<f64> = (0..k_sv).map(|i| sv[i] / (sv[i]*sv[i] + lambda_lm)).collect();
            let db = DVector::from_iterator(k_sv, (0..k_sv).map(|i| d[i] * ut_negf[i]));
            let dx_hat = v_t.transpose() * db;
            let dx_dvec = DVector::from_iterator(
                dim_x,
                dx_hat.iter().zip(scale_x.iter()).map(|(d, s)| d * s),
            );

            // Trust-region backstop against a genuinely pathological step —
            // should rarely bind once LM damping is doing its job.
            let mut max_pos_delta = 0.0_f64;
            let mut max_vel_delta = 0.0_f64;
            for kk in 0..n {
                for c in 0..3 { max_vel_delta = max_vel_delta.max(dx_dvec[3*kk + c].abs()); }
            }
            for m in 0..n.saturating_sub(1) {
                let base = 3*n + 6*m;
                for c in 0..3 { max_pos_delta = max_pos_delta.max(dx_dvec[base + c].abs()); }
                for c in 3..6 { max_vel_delta = max_vel_delta.max(dx_dvec[base + c].abs()); }
            }
            let trust_scale = (MS_MAX_STEP_POS_M / max_pos_delta.max(1.0))
                .min(MS_MAX_STEP_VEL_MS / max_vel_delta.max(1e-6))
                .min(1.0);

            let x_trial: Vec<f64> = (0..dim_x).map(|j| x[j] + trust_scale * dx_dvec[j]).collect();
            let trial = ms_constraint(&x_trial, &ev, &result.best_params, flyby_bodies, departure_body, cfg, almanac, &transit_time_s);
            if std::env::var("MGA_MS_DEBUG").is_ok() {
                let trial_norm_str = match &trial {
                    Some(f_trial) => format!("{:.6e}", f_trial.iter().zip(scale_f.iter())
                        .map(|(f, s)| (f / s).powi(2)).sum::<f64>().sqrt()),
                    None => "PROPAGATION FAILED".to_string(),
                };
                eprintln!(
                    "[MS_DEBUG] iter {iter} try {_try}: λ={lambda_lm:.2e} |dx̂|={:.3e} trust={trust_scale:.3} base={base_hat_norm:.6e} trial={trial_norm_str}",
                    dx_hat.norm(),
                );
            }
            if let Some(f_trial) = trial {
                let trial_hat_norm: f64 = f_trial.iter().zip(scale_f.iter())
                    .map(|(f, s)| (f / s).powi(2)).sum::<f64>().sqrt();
                if trial_hat_norm < base_hat_norm {
                    x = x_trial;
                    lambda_lm = (lambda_lm / MS_LM_DECREASE_FACTOR).max(MS_LM_LAMBDA_MIN);
                    accepted = true;
                    break;
                }
            }
            lambda_lm = (lambda_lm * MS_LM_INCREASE_FACTOR).min(MS_LM_LAMBDA_MAX);
        }

        if !accepted {
            // Even maximal damping (near-pure gradient descent) couldn't find
            // an improving step this iteration — stop rather than spin.
            eprintln!("  LM stalled at iteration {iter} (λ={lambda_lm:.3e}) — no improving step found.");
            break;
        }
    }

    // ── Did not converge within MAX_MS_ITER — return best-so-far ─────────────
    let f_final = ms_constraint(&x, &ev, &result.best_params, flyby_bodies, departure_body, cfg, almanac, &transit_time_s)
        .unwrap_or_else(|| vec![f64::MAX; ms_constraint_dim(n)]);

    let (residuals_m, vel_residuals_ms) = ms_split_residuals(&f_final, n);

    let dv_dsm_corrected: Vec<[f64; 3]> = (0..n).map(|k| {
        let dv = ms_dv(&x, k);
        [dv.x, dv.y, dv.z]
    }).collect();

    let arc = ms_reprop_arc(&x, &ev, &result.best_params, flyby_bodies, departure_body, cfg, almanac, &transit_time_s);
    let dv_dsm_total: f64 = dv_dsm_corrected.iter()
        .map(|dv| (dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2]).sqrt())
        .sum();
    let dv_total_ms = result.dv_departure_ms + dv_dsm_total + result.dv_arrival_ms;

    eprintln!("  Warning: multiple shooting did not converge within {MAX_MS_ITER} iterations.");
    eprintln!("  Final position residuals [km]: {}",
        residuals_m.iter().map(|r| format!("{:.1}", r/1000.0)).collect::<Vec<_>>().join(", "));
    eprintln!("  Final velocity residuals [m/s]: {}",
        vel_residuals_ms.iter().map(|r| format!("{:.3}", r)).collect::<Vec<_>>().join(", "));

    Ok(MultipleShotResult {
        converged: false,
        iterations,
        residuals_m,
        vel_residuals_ms,
        dv_dsm_corrected,
        arc,
        dv_total_ms,
        // No genuinely converged crossing to report — same gating single-leg's
        // `post_capture_orbit_arc` already applies (never built on top of a
        // non-converged/errored result).
        arrival_r_rel_m: None,
        arrival_v_rel_mps: None,
        flyby_dv_gained_ms: Vec::new(),
        flyby_speed_before_ms: Vec::new(),
        flyby_speed_after_ms: Vec::new(),
    })
}

/// CLI entry point for the `mga-refine` subcommand.
///
/// Loads the saved `mga_best_chromosome.csv`, runs multiple-shooting
/// refinement, prints the convergence table, and writes the corrected arc.
pub fn run_mga_refine(cfg: &MissionConfig, almanac: &Almanac) -> Result<(), String> {
    let opt = cfg.optimization.as_ref()
        .ok_or("mga-refine requires an [optimization] section")?;

    let out_dir = cfg.simulation.output_dir.trim_end_matches('/');
    let csv_path = format!("{out_dir}/mga_best_chromosome.csv");

    // ── Read chromosome CSV (same logic as run_mga_geometry) ─────────────────
    let content = std::fs::read_to_string(&csv_path)
        .map_err(|e| format!("cannot read {csv_path}: {e}\nRun the MGA optimizer first."))?;

    let mut lines = content.lines();
    let _header  = lines.next().ok_or("mga_best_chromosome.csv is empty")?;
    let data_row = lines.next().ok_or("mga_best_chromosome.csv has no data row")?;

    let comma = data_row.find(',')
        .ok_or("mga_best_chromosome.csv: no comma separating flyby_bodies from params")?;
    let bodies_str = &data_row[..comma];
    let params_str = &data_row[comma + 1..];

    let flyby_bodies: Vec<String> = if bodies_str.is_empty() {
        vec![]
    } else {
        bodies_str.split(';').map(str::to_string).collect()
    };
    let params: Vec<f64> = params_str
        .split(',')
        .map(|s| s.trim().parse::<f64>().map_err(|e| format!("parse param '{}': {e}", s.trim())))
        .collect::<Result<Vec<f64>, _>>()?;

    let n = n_legs(&flyby_bodies);
    let expected_len = chromosome_len(n);
    if params.len() != expected_len {
        return Err(format!(
            "chromosome has {} params but expected {} for {n} legs",
            params.len(), expected_len
        ));
    }

    // ── Reconstruct enough of an MgaResult to call run_multiple_shooting ────
    // We need: dep_jd, leg_tofs_days, dv_dsms_ms, dv_departure_ms, dv_arrival_ms.
    // Re-evaluate the chromosome to get these.
    let dep_epoch_str = opt.departure_epoch.as_deref()
        .ok_or("optimization.departure_epoch is required")?;
    let dep_epoch   = crate::design::parse_epoch(dep_epoch_str)
        .map_err(|e| format!("departure_epoch parse: {e}"))?;
    let dep_jd_base = crate::design::epoch_to_jd(dep_epoch);

    let ev = evaluate_chromosome_detailed(&params, cfg, almanac, dep_jd_base, &flyby_bodies)
        .ok_or("chromosome is infeasible — try re-running the optimizer")?;

    let (dv_dep, _) = crate::design::departure_escape_dv_ms(cfg, &opt.departure_body, ev.v_inf_dep_ms)
        .ok_or_else(|| format!("departure body '{}' not in catalog", opt.departure_body))?;

    let dv_dsms: Vec<f64> = ev.legs.iter().map(|l| l.leg.dv_dsm_ms).collect();
    let dv_arr = arrival_dv_ms(cfg, &ev);

    let dep_jd = dep_jd_base + dep_offset(&params);
    let leg_tofs_days: Vec<f64> = (0..n).map(|k| tof_days(&params, k)).collect();
    let dsm_positions_m: Vec<[f64; 3]> = ev.legs.iter()
        .map(|le| [le.leg.r_dsm_m.x, le.leg.r_dsm_m.y, le.leg.r_dsm_m.z])
        .collect();
    let dsm_epochs_s: Vec<f64> = ev.legs.iter().enumerate()
        .map(|(k, le)| (le.t_start_days + eta(&params, k) * tof_days(&params, k)) * 86_400.0)
        .collect();

    let mut body_names: Vec<String> = vec![opt.departure_body.clone()];
    body_names.extend_from_slice(&flyby_bodies);
    body_names.push(opt.target_body.clone());

    let result = MgaResult {
        dv_total_ms:     dv_dep + dv_dsms.iter().sum::<f64>() + dv_arr,
        dv_departure_ms: dv_dep,
        dv_dsms_ms:      dv_dsms,
        dv_arrival_ms:   dv_arr,
        tof_total_days:  leg_tofs_days.iter().sum(),
        leg_tofs_days:   leg_tofs_days.clone(),
        dsm_positions_m,
        dsm_epochs_s,
        body_sequence:   body_names,
        dep_jd,
        dep_jd_base,
        best_params:     params.clone(),
        convergence:     vec![],
        phase1_history:  vec![],
        param_history:   vec![],
        phase1_param_history: vec![],
        arc:             vec![],
        repropagated_arc: vec![],
    };

    println!("\nMGA multiple-shooting refinement");
    println!("  Source: {csv_path}");
    println!("  Sequence: {}", result.body_sequence.join(" → "));
    println!("  Analytic ΔV: dep={:.1}  DSMs={:.1}  arr={:.1}  total={:.1} m/s",
        result.dv_departure_ms,
        result.dv_dsms_ms.iter().sum::<f64>(),
        result.dv_arrival_ms,
        result.dv_total_ms,
    );
    println!();

    // ── Run multiple shooting ─────────────────────────────────────────────────
    let ms = run_multiple_shooting(&result, &flyby_bodies, cfg, almanac)?;

    // ── Print summary ─────────────────────────────────────────────────────────
    println!();
    if ms.converged {
        println!("  ✓ Converged in {} iterations", ms.iterations + 1);
    } else {
        println!("  ✗ Did NOT converge within {MAX_MS_ITER} iterations");
    }
    println!("  Per-leg position residuals after refinement:");
    for (k, res) in ms.residuals_m.iter().enumerate() {
        println!("    Leg {k}: {:.3} km", res / 1000.0);
    }
    if !ms.vel_residuals_ms.is_empty() {
        println!("  Per-node velocity residuals after refinement (flyby continuity check):");
        for (m, res) in ms.vel_residuals_ms.iter().enumerate() {
            println!("    Node {}: {:.4} m/s", m + 1, res);
        }
    }

    let dv_dsm_corrected_total: f64 = ms.dv_dsm_corrected.iter()
        .map(|dv| (dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2]).sqrt())
        .sum();
    println!();
    println!("  Corrected ΔV: dep={:.1}  DSMs={:.1}  arr={:.1}  total={:.1} m/s",
        result.dv_departure_ms, dv_dsm_corrected_total,
        result.dv_arrival_ms, ms.dv_total_ms);
    println!("  ΔΔV vs analytic: {:+.1} m/s", ms.dv_total_ms - result.dv_total_ms);

    // ── Write output files ────────────────────────────────────────────────────
    let _ = std::fs::create_dir_all(out_dir);
    println!("\n  Output:");
    write_refined_csv(&ms.arc, out_dir);
    write_refined_legs_csv(&ms.dv_dsm_corrected, &leg_tofs_days, out_dir);

    Ok(())
}

// ── Integration tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MissionConfig;

    /// `invert_flyby_turn` must recover a `(r_p, β)` pair that reproduces
    /// the exact outgoing v∞ `flyby_turn` produced — round-trip through the
    /// forward turn, across several turn geometries including a retrograde
    /// β and a near-polar incoming direction (the `ref_vec` fallback
    /// branch). Pure math, no ephemeris. (Phase 9x bidirectional stitching
    /// depends on this inversion to derive the interface flyby's chromosome
    /// parameters from a matched prefix/suffix v∞ pair.)
    #[test]
    fn invert_flyby_turn_roundtrip() {
        let mu_earth = 3.986_004_418e14_f64;
        let cases = [
            (Vector3::new(5_000.0, 3_000.0, -1_000.0), 7.0e6, 0.7),
            (Vector3::new(5_000.0, 3_000.0, -1_000.0), 1.2e7, -2.1),
            (Vector3::new(-2_000.0, 1_500.0, 8_500.0), 6.6e6, 3.0),
            // x-dominant incoming direction exercises the ref_vec fallback.
            (Vector3::new(9_000.0, 500.0, -300.0), 8.0e6, -0.4),
        ];
        for (v_in, rp, beta) in cases {
            let v_out = flyby_turn(v_in, rp, beta, mu_earth);
            let (rp_inv, beta_inv) = invert_flyby_turn(v_in, v_out, mu_earth);
            assert!(
                (rp_inv - rp).abs() / rp < 1e-6,
                "rp not recovered: {rp_inv} vs {rp} (beta={beta})"
            );
            let v_out2 = flyby_turn(v_in, rp_inv, beta_inv, mu_earth);
            assert!(
                (v_out2 - v_out).norm() < 1e-6 * v_out.norm(),
                "outgoing v∞ not reproduced: err {} m/s (beta={beta})",
                (v_out2 - v_out).norm()
            );
        }
    }

    /// `stitch_chromosome` must place every forward-prefix, interface, and
    /// backward-suffix variable at the exact `params`-layout position the
    /// module accessors read from (n = 4 legs, split at body m = 2 —
    /// VEEGA-shaped). Sentinel values make any transposition visible.
    #[test]
    fn stitch_chromosome_layout_roundtrip() {
        let n = 4usize;
        let m = 2usize;
        // Forward Ceriotti vars through level m-1 = 1: 7 + 5 = 12.
        let fwd: Vec<f64> = (0..12).map(|i| 100.0 + i as f64).collect();
        // Backward suffix vars for legs 2..3: 7 + 5 = 12.
        let bwd: Vec<f64> = (0..12).map(|i| 200.0 + i as f64).collect();
        let bounds = vec![(-1.0e9, 1.0e9); chromosome_len(n)];
        let p = stitch_chromosome(n, m, &fwd, &bwd, 55.5, -0.25, &bounds);
        assert_eq!(p.len(), chromosome_len(n));
        // Departure block + leg 0 straight from the forward vars.
        assert_eq!(&p[..6], &fwd[..6]);
        assert_eq!(tof_days(&p, 0), fwd[4]);
        assert_eq!(eta(&p, 0), fwd[5]);
        // Forward level 1 block: [rp_0, beta_0, tof_1, eta_1, n_rev_1].
        assert_eq!(rp_norm(&p, n, 0), fwd[7]);
        assert_eq!(beta(&p, n, 0), fwd[8]);
        assert_eq!(tof_days(&p, 1), fwd[9]);
        assert_eq!(eta(&p, 1), fwd[10]);
        // Interface flyby (index m-1 = 1): geometry-derived values.
        assert_eq!(rp_norm(&p, n, 1), 55.5);
        assert_eq!(beta(&p, n, 1), -0.25);
        // Suffix leg 2 from the backward level-0 block (first 4 discarded).
        assert_eq!(tof_days(&p, 2), bwd[4]);
        assert_eq!(eta(&p, 2), bwd[5]);
        // Suffix level 1 block: [rp_2, beta_2, tof_3, eta_3, n_rev_3].
        assert_eq!(rp_norm(&p, n, 2), bwd[7]);
        assert_eq!(beta(&p, n, 2), bwd[8]);
        assert_eq!(tof_days(&p, 3), bwd[9]);
        assert_eq!(eta(&p, 3), bwd[10]);
        // n_rev genes (chromosome tail, one per leg).
        assert_eq!(p[2 + 4 * n], fwd[6]);
        assert_eq!(p[2 + 4 * n + 1], fwd[11]);
        assert_eq!(p[2 + 4 * n + 2], bwd[6]);
        assert_eq!(p[2 + 4 * n + 3], bwd[11]);
    }

    /// Suffix level bounds must mirror the forward block shapes (7-var
    /// level 0, 5-var levels after) with the interface-epoch window equal
    /// to the departure window plus the prefix legs' summed TOF bounds.
    #[test]
    fn suffix_level_bounds_shapes_and_epoch_window() {
        let n = 4usize;
        // Build a synthetic flat bounds vector in the chromosome layout.
        let mut flat = vec![(0.0, 0.0); chromosome_len(n)];
        flat[0] = (-30.0, 30.0); // departure window
        for k in 0..n {
            flat[4 + 2 * k] = (100.0 * (k + 1) as f64, 200.0 * (k + 1) as f64); // tof_k
            flat[4 + 2 * k + 1] = (0.01, 0.99); // eta_k
            flat[2 + 4 * n + k] = (0.0, 2.999); // n_rev_k
        }
        for j in 0..(n - 1) {
            flat[4 + 2 * n + 2 * j] = (1.05, 300.0); // rp_norm_j
            flat[4 + 2 * n + 2 * j + 1] = (-PI, PI); // beta_j
        }
        let m = 2usize;
        let levels = suffix_level_bounds(&flat, n, m);
        assert_eq!(levels.len(), n - m);
        assert_eq!(levels[0].len(), 7);
        assert_eq!(levels[1].len(), 5);
        // Epoch window: [-30 + 100 + 200, 30 + 200 + 400].
        assert_eq!(levels[0][0], (270.0, 630.0));
        // Level 0 carries leg m's TOF bounds; level 1 carries leg m+1's,
        // preceded by the interface-after-next flyby's (rp, beta) bounds.
        assert_eq!(levels[0][4], flat[4 + 2 * m]);
        assert_eq!(levels[1][2], flat[4 + 2 * (m + 1)]);
        assert_eq!(levels[1][0], flat[4 + 2 * n + 2 * (m + 1 - 1)]);
    }

    /// `flyby_periapsis_state` must produce a periapsis state whose REAL
    /// Kepler propagation escapes along the outgoing asymptote û_out (and
    /// arrives from û_in going backward). This is the test that would have
    /// caught the original sign error (r̂_p = û_out − û_in instead of
    /// û_in − û_out): a mirror-side node propagates to a multi-km/s-wrong
    /// asymptote, which broke every multiple-shooting leg that starts from
    /// a node.
    #[test]
    fn flyby_periapsis_state_matches_kepler_asymptotes() {
        let mu_earth = 3.986_004_418e14_f64;
        let v_inf = 5_000.0_f64; // m/s

        // Turn angle from a real periapsis: e = 1 + rp·v∞²/μ, δ = 2·asin(1/e).
        let rp = 2.0e7_f64; // 20,000 km — comfortably hyperbolic geometry
        let e = 1.0 + rp * v_inf * v_inf / mu_earth;
        let delta = 2.0 * (1.0 / e).asin();

        // Incoming along +x, turn by δ about +z (counterclockwise in xy).
        let u_in = Vector3::new(1.0, 0.0, 0.0);
        let u_out = Vector3::new(delta.cos(), delta.sin(), 0.0);

        let (r_p_vec, v_p_vec) = flyby_periapsis_state(
            Vector3::zeros(), Vector3::zeros(), // body at rest at origin
            u_in * v_inf, u_out * v_inf,
            rp, mu_earth,
        );

        assert!((r_p_vec.norm() - rp).abs() / rp < 1e-12, "periapsis radius wrong");
        assert!(r_p_vec.dot(&v_p_vec).abs() < 1e-3 * rp * v_p_vec.norm(),
            "periapsis velocity not perpendicular to radius");

        // Propagate FORWARD 30 days around the body: velocity direction must
        // approach û_out (the departing leg's escape asymptote).
        let t_fwd = 30.0 * 86_400.0;
        let (_, v_fwd) = propagate_kepler(r_p_vec, v_p_vec, t_fwd, mu_earth)
            .expect("forward Kepler propagation failed");
        let cos_out = v_fwd.normalize().dot(&u_out);
        assert!(cos_out > (2.0_f64).to_radians().cos(),
            "escape asymptote off by {:.2} deg from u_out (sign error?)",
            cos_out.clamp(-1.0, 1.0).acos().to_degrees());

        // Propagate BACKWARD 30 days: velocity direction must match û_in
        // (the arriving leg's incoming asymptote).
        let (_, v_back) = propagate_kepler(r_p_vec, v_p_vec, -t_fwd, mu_earth)
            .expect("backward Kepler propagation failed");
        let cos_in = v_back.normalize().dot(&u_in);
        assert!(cos_in > (2.0_f64).to_radians().cos(),
            "incoming asymptote off by {:.2} deg from u_in (sign error?)",
            cos_in.clamp(-1.0, 1.0).acos().to_degrees());
    }

    /// `flyby_soi_entry_state` (Phase 9x-iv rework) must produce
    /// a body-relative state at exactly `|r| = soi_radius_m`, on the
    /// INCOMING branch (before periapsis), whose own REAL Kepler propagation
    /// — forward by the function's own reported `transit_time_s` — reaches
    /// periapsis (`|r| = rp`) and, continuing further forward, escapes along
    /// the outgoing asymptote û_out exactly as
    /// `flyby_periapsis_state_matches_kepler_asymptotes` above already
    /// confirms for the periapsis state itself. This is the rigor level
    /// that test already established, applied to the new SOI-entry function:
    /// a wrong sign/formula here would either put the entry point on the
    /// wrong (outgoing) branch or fail to reach periapsis at the predicted
    /// transit time, both of which this test would catch.
    #[test]
    fn flyby_soi_entry_state_reaches_periapsis_and_correct_asymptotes() {
        let mu_earth = 3.986_004_418e14_f64;
        let v_inf = 5_000.0_f64; // m/s
        let rp = 2.0e7_f64;      // 20,000 km — same geometry as the periapsis-state test
        let soi_radius_m = 1.0e9_f64; // 1,000,000 km — comfortably outside rp

        let e = 1.0 + rp * v_inf * v_inf / mu_earth;
        let delta = 2.0 * (1.0 / e).asin();
        let u_in = Vector3::new(1.0, 0.0, 0.0);
        let u_out = Vector3::new(delta.cos(), delta.sin(), 0.0);

        let (r_entry, v_entry, transit_time_s) = flyby_soi_entry_state(
            u_in * v_inf, u_out * v_inf, rp, mu_earth, soi_radius_m);

        // Entry point is really at the SOI boundary.
        assert!((r_entry.norm() - soi_radius_m).abs() / soi_radius_m < 1e-9,
            "SOI-entry radius wrong: {} vs requested {}", r_entry.norm(), soi_radius_m);
        // Transit time (entry -> periapsis) must be a real, positive duration —
        // for a v_inf = 5 km/s hyperbola reaching out to 1e6 km, this should be
        // on the order of hours to a couple of days, comfortably inside (0, 30 days].
        assert!(transit_time_s.is_finite() && transit_time_s > 0.0
            && transit_time_s < 30.0 * 86_400.0,
            "transit_time_s not sane: {transit_time_s}");

        // Propagating forward by exactly transit_time_s must reach periapsis.
        let (r_peri, v_peri) = propagate_kepler(r_entry, v_entry, transit_time_s, mu_earth)
            .expect("forward Kepler propagation to periapsis failed");
        assert!((r_peri.norm() - rp).abs() / rp < 1e-6,
            "propagating by transit_time_s did not reach periapsis: |r| = {} vs rp = {}",
            r_peri.norm(), rp);
        assert!(r_peri.dot(&v_peri).abs() < 1e-3 * rp * v_peri.norm(),
            "state at reported periapsis time is not actually at periapsis (r.v != 0)");

        // Continuing forward past periapsis must escape along û_out (same
        // asymptote check as flyby_periapsis_state_matches_kepler_asymptotes).
        let t_fwd = 30.0 * 86_400.0;
        let (_, v_fwd) = propagate_kepler(r_peri, v_peri, t_fwd, mu_earth)
            .expect("forward Kepler propagation past periapsis failed");
        let cos_out = v_fwd.normalize().dot(&u_out);
        assert!(cos_out > (2.0_f64).to_radians().cos(),
            "escape asymptote off by {:.2} deg from u_out",
            cos_out.clamp(-1.0, 1.0).acos().to_degrees());

        // The entry state's OWN velocity direction should already be close to
        // û_in (it is on the incoming branch, far out at the SOI boundary).
        let cos_in = v_entry.normalize().dot(&u_in);
        assert!(cos_in > (5.0_f64).to_radians().cos(),
            "SOI-entry velocity direction off by {:.2} deg from u_in",
            cos_in.clamp(-1.0, 1.0).acos().to_degrees());
    }

    /// Smoke test: load evj_flyby.toml, run with a tiny DE budget (pop=20,
    /// gen=50, 1 restart), assert the optimizer runs without panicking and
    /// returns a physically plausible total ΔV < 8 km/s.
    ///
    /// This is *not* a convergence proof — the tiny budget may produce a
    /// mediocre result. The bound (8 000 m/s) is deliberately generous: it
    /// catches returning `f64::MAX` (infeasible chromosome survived to the
    /// end) while accepting any physically real EVJ trajectory.
    ///
    /// Requires `kernels/de440s.bsp` to be present at `MissionPlanner/` root.
    /// Silently skips when the kernel is absent (not in git).
    #[test]
    fn mga_evj_smoke() {
        // Load kernel — skip gracefully if not present.
        let almanac = match ephemeris::Almanac::new("kernels/de440s.bsp") {
            Ok(a) => a,
            Err(_) => {
                eprintln!("[skip] mga_evj_smoke: de440s.bsp not found");
                return;
            }
        };

        // Load config.
        let toml_str = match std::fs::read_to_string("config/evj_flyby.toml") {
            Ok(s) => s,
            Err(e) => panic!("could not read config/evj_flyby.toml: {e}"),
        };
        let mut cfg: MissionConfig = match toml::from_str(&toml_str) {
            Ok(c) => c,
            Err(e) => panic!("could not parse evj_flyby.toml: {e}"),
        };

        // Override to a tiny DE budget so the test runs in < 10 s. Pin
        // search_method = De explicitly (Stage 4, flipped the
        // config default to Mbh) — this test's budget knobs and <8,000 m/s
        // threshold were written for DE; mga_evj_smoke_mbh below is the
        // dedicated MBH check.
        if let Some(opt) = cfg.optimization.as_mut() {
            if let Some(mga) = opt.mga.as_mut() {
                mga.search_method      = crate::config::SearchMethod::De;
                mga.de_population_size = 20;
                mga.de_generations     = 50;
                mga.de_restarts        = 1;
            }
            // Redirect output to a temp directory so the test doesn't leave
            // files under the repo root.
            cfg.simulation.output_dir = std::env::temp_dir()
                .join("mga_evj_smoke_test")
                .to_string_lossy()
                .into_owned();
        }

        let result = run_mga(&cfg, &almanac, |_, _, _, _, _| {}, |_, _, _, _| {});

        match result {
            Ok(r) => {
                assert!(
                    r.dv_total_ms < 8_000.0,
                    "EVJ MGA best ΔV {:.1} m/s exceeds 8 000 m/s — something is wrong",
                    r.dv_total_ms
                );
                println!("[mga_evj_smoke] best ΔV = {:.1} m/s", r.dv_total_ms);
            }
            Err(e) => panic!("run_mga returned Err: {e}"),
        }
    }

    /// Phase 9x-v Stage 3 smoke test: `search_method = "Mbh"` on the same
    /// EVJ config as `mga_evj_smoke`, tiny budget, confirms MBH mode produces
    /// a feasible, finite result end-to-end on a real MGA config (Phase 1
    /// skipped entirely under MBH per `run_mga_fixed_sequence`'s dispatch —
    /// this test is also the check that the skip path itself doesn't panic
    /// or leave `elite_seeds` in a state MBH's `run_seeded_with_progress`
    /// chokes on when empty).
    #[test]
    fn mga_evj_smoke_mbh() {
        let almanac = match ephemeris::Almanac::new("kernels/de440s.bsp")
            .or_else(|_| ephemeris::Almanac::new("../kernels/de440s.bsp"))
        {
            Ok(a) => a,
            Err(_) => {
                eprintln!("[skip] mga_evj_smoke_mbh: de440s.bsp not found");
                return;
            }
        };

        let toml_str = match std::fs::read_to_string("config/evj_flyby.toml")
            .or_else(|_| std::fs::read_to_string("MissionPlanner/config/evj_flyby.toml"))
        {
            Ok(s) => s,
            Err(e) => panic!("could not read config/evj_flyby.toml: {e}"),
        };
        let mut cfg: MissionConfig = match toml::from_str(&toml_str) {
            Ok(c) => c,
            Err(e) => panic!("could not parse evj_flyby.toml: {e}"),
        };

        if let Some(opt) = cfg.optimization.as_mut() {
            if let Some(mga) = opt.mga.as_mut() {
                mga.search_method = crate::config::SearchMethod::Mbh;
                mga.mbh.hops = 15;
                mga.mbh.local_max_iter = 60;
            }
            cfg.simulation.output_dir = std::env::temp_dir()
                .join("mga_evj_smoke_mbh_test")
                .to_string_lossy()
                .into_owned();
        }

        let result = run_mga(&cfg, &almanac, |_, _, _, _, _| {}, |_, _, _, _| {});

        match result {
            Ok(r) => {
                assert!(
                    r.dv_total_ms.is_finite() && r.dv_total_ms > 0.0,
                    "MBH EVJ result is not a finite positive ΔV: {}",
                    r.dv_total_ms
                );
                println!("[mga_evj_smoke_mbh] best ΔV = {:.1} m/s", r.dv_total_ms);
            }
            Err(e) => panic!("run_mga (MBH) returned Err: {e}"),
        }
    }

    /// [`EphemerisCache::lookup`]'s cubic Hermite must reproduce a real
    /// orbit far more accurately than the search cares about: for a
    /// synthetic circular 1-AU heliocentric orbit sampled at the cache's
    /// own step, mid-interval interpolation error must be sub-km in
    /// position and sub-mm/s in velocity (analytic bound ~ h⁴; this test
    /// pins it empirically so a future step-size or formula change that
    /// degrades accuracy fails loudly).
    #[test]
    fn eph_cache_hermite_is_subkm_on_circular_orbit() {
        let au = 1.495_978_707e11_f64;
        let mu = MU_SUN_M3S2;
        let omega = (mu / au.powi(3)).sqrt(); // rad/s
        let state_at = |jd: f64| {
            let t_s = jd * 86_400.0;
            let th = omega * t_s;
            (
                Vector3::new(au * th.cos(), au * th.sin(), 0.0),
                Vector3::new(-au * omega * th.sin(), au * omega * th.cos(), 0.0),
            )
        };

        let n_samples = 40;
        let samples: Vec<_> = (0..n_samples)
            .map(|i| state_at(i as f64 * EPH_CACHE_STEP_DAYS))
            .collect();
        let mut bodies = std::collections::HashMap::new();
        bodies.insert("testbody".to_string(), samples);
        let cache = EphemerisCache { jd0: 0.0, step_days: EPH_CACHE_STEP_DAYS, bodies };

        // Worst case is mid-interval; check several across the span.
        for i in [1usize, 10, 25, 37] {
            let jd = (i as f64 + 0.5) * EPH_CACHE_STEP_DAYS;
            let (r_i, v_i) = cache.lookup("testbody", jd).expect("in coverage");
            let (r_t, v_t) = state_at(jd);
            assert!((r_i - r_t).norm() < 1.0e3,
                "position error {:.1} m at jd={jd}", (r_i - r_t).norm());
            assert!((v_i - v_t).norm() < 1.0e-3,
                "velocity error {:.6} m/s at jd={jd}", (v_i - v_t).norm());
        }

        // Outside coverage / unknown body → miss, never wrong data.
        assert!(cache.lookup("testbody", -1.0).is_none());
        assert!(cache.lookup("testbody", 1.0e9).is_none());
        assert!(cache.lookup("nosuchbody", 1.0).is_none());
    }

    /// [`resonance_family_branches`]: for VEEGA's sequence
    /// (Earth-Venus-Earth-Earth-Jupiter, one same-body leg: leg 2
    /// Earth→Earth) with TOF bounds [250, 900] d, exactly the 1-year and
    /// 2-year Earth windows fit (N=3 ≈ 1096 d is out of bounds) → 3
    /// branches: full + N=1 + N=2, each restricting ONLY leg 2's TOF slot.
    /// A sequence with no same-body legs must return exactly one branch
    /// (today's behaviour, zero overhead).
    #[test]
    fn resonance_family_branches_enumerates_earth_windows() {
        let n = 4;
        let body_names = ["Earth", "Venus", "Earth", "Earth", "Jupiter"];
        let mut bounds = vec![(0.0_f64, 1.0); chromosome_len(n)];
        for k in 0..n { bounds[4 + 2 * k] = (30.0, 1500.0); }
        bounds[4 + 2 * 2] = (250.0, 900.0); // leg 2 (Earth->Earth) TOF

        let branches = resonance_family_branches(&bounds, &body_names, n);
        assert_eq!(branches.len(), 3, "expected full + N=1 + N=2: {:?}",
            branches.iter().map(|(l, _)| l.clone()).collect::<Vec<_>>());
        assert_eq!(branches[0].0, "full bounds");
        assert_eq!(branches[0].1, bounds);

        let year = 365.25;
        for (i, nrev) in [(1usize, 1.0_f64), (2, 2.0)] {
            let (lo, hi) = branches[i].1[4 + 2 * 2];
            assert!(lo < nrev * year && nrev * year < hi,
                "branch {i} window [{lo:.0}, {hi:.0}] should contain {:.0} d", nrev * year);
            // Every other slot unchanged.
            for (j, (a, b)) in branches[i].1.iter().enumerate() {
                if j != 4 + 2 * 2 {
                    assert_eq!((*a, *b), bounds[j], "slot {j} of branch {i} must be unrestricted");
                }
            }
        }

        // No same-body legs -> single branch.
        let body_names2 = ["Earth", "Venus", "Jupiter", "Saturn", "Neptune"];
        let branches2 = resonance_family_branches(&bounds, &body_names2, n);
        assert_eq!(branches2.len(), 1);
    }

    /// [`leg_model_gene`] decode (model-choice-as-gene fix):
    /// values 0-2 must reproduce the historical Lambert n_rev behaviour
    /// exactly (including the resonance-bias sampler's `N + 0.5` seeding),
    /// 3-6 must map to the four VILM variants, and out-of-range values
    /// must clamp instead of panicking.
    #[test]
    fn leg_model_gene_decodes_all_seven_options_and_clamps() {
        let n = 2;
        let mut p = vec![0.0; chromosome_len(n)];
        let gene_idx = 2 + 4 * n; // leg 0's tail gene

        for (raw, want_lambert_n) in [(0.0, 0u32), (0.5, 0), (1.5, 1), (2.5, 2), (2.999, 2), (-1.0, 0)] {
            p[gene_idx] = raw;
            assert!(
                matches!(leg_model_gene(&p, n, 0), LegModel::Lambert(nr) if nr == want_lambert_n),
                "gene={raw} should decode Lambert({want_lambert_n}), got {:?}", leg_model_gene(&p, n, 0),
            );
        }
        for (raw, want_domain_interior, want_lower) in [
            (3.5, true, true), (4.5, true, false), (5.5, false, true), (6.5, false, false),
            (99.0, false, false), // clamps to the last option
        ] {
            p[gene_idx] = raw;
            let LegModel::Vilm(domain, solution) = leg_model_gene(&p, n, 0) else {
                panic!("gene={raw} should decode a VILM variant");
            };
            assert_eq!(matches!(domain, VilmDomain::Interior), want_domain_interior, "gene={raw} domain");
            assert_eq!(matches!(solution, VilmSolution::Lower), want_lower, "gene={raw} solution");
        }
    }

    /// The post-search polish loop (Phase 9x — alternating
    /// [`repolish_leg_etas`] and joint L-BFGS to a fixed point) runs
    /// automatically at the end of every `run_mga_fixed_sequence` call, so
    /// its winner is already polished — this test checks the property that
    /// actually matters for correctness: repolishing the pipeline's OWN
    /// output again must find (near) zero further improvement (a fixed
    /// point), not keep finding "gains" forever. Tolerance is a small
    /// multiple of the pipeline's own `POLISH_FIXED_POINT_MS` termination
    /// threshold, since an external re-check's grid pass can legitimately
    /// find slightly more than the loop's break threshold. Uses a real
    /// 3-flyby config (`veega_flyby.toml`, committed) at a tiny budget so
    /// the winner is very likely still eta-suboptimal going in — giving the
    /// automatic polish real room to act — while keeping the test fast and
    /// independent of any scratch/uncommitted output files.
    #[test]
    fn eta_repolish_is_idempotent_on_its_own_output() {
        let almanac = match ephemeris::Almanac::new("kernels/de440s.bsp")
            .or_else(|_| ephemeris::Almanac::new("../kernels/de440s.bsp"))
        {
            Ok(a) => a,
            Err(_) => {
                eprintln!("[skip] eta_repolish_is_idempotent_on_its_own_output: de440s.bsp not found");
                return;
            }
        };

        let toml_str = match std::fs::read_to_string("config/veega_flyby.toml")
            .or_else(|_| std::fs::read_to_string("MissionPlanner/config/veega_flyby.toml"))
        {
            Ok(s) => s,
            Err(e) => panic!("could not read config/veega_flyby.toml: {e}"),
        };
        let mut cfg: MissionConfig = match toml::from_str(&toml_str) {
            Ok(c) => c,
            Err(e) => panic!("could not parse veega_flyby.toml: {e}"),
        };

        let flyby_bodies = cfg.optimization.as_ref()
            .and_then(|o| o.mga.as_ref())
            .map(|m| m.flyby_bodies.clone())
            .unwrap_or_default();
        assert!(!flyby_bodies.is_empty(), "fixture config must have a fixed flyby sequence");

        if let Some(opt) = cfg.optimization.as_mut() {
            if let Some(mga) = opt.mga.as_mut() {
                mga.search_method      = crate::config::SearchMethod::De;
                mga.de_population_size = 20;
                mga.de_generations     = 40;
                mga.de_restarts        = 1;
            }
            cfg.simulation.output_dir = std::env::temp_dir()
                .join("eta_repolish_idempotent_test")
                .to_string_lossy()
                .into_owned();
        }

        let result = run_mga_fixed_sequence(&cfg, &almanac, &flyby_bodies, &mut |_, _, _, _, _| {})
            .expect("run_mga_fixed_sequence should succeed");
        assert!(result.dv_total_ms.is_finite() && result.dv_total_ms > 0.0);

        let opt = cfg.optimization.as_ref().unwrap();
        let dep_epoch = crate::design::parse_epoch(opt.departure_epoch.as_deref().unwrap())
            .expect("departure_epoch should parse");
        let dep_jd_base = crate::design::epoch_to_jd(dep_epoch);
        let n = n_legs(&flyby_bodies);

        let mut params_again = result.best_params.clone();
        let (before2, after2) =
            repolish_leg_etas(&mut params_again, &cfg, &almanac, dep_jd_base, &flyby_bodies, n);
        assert!(
            after2 <= before2 + 1.0e-6,
            "second repolish pass must never increase total ΔV: before={before2:.3} after={after2:.3}",
        );
        assert!(
            (before2 - after2).abs() < 5.0,
            "repolishing the pipeline's already-polished winner should find (near) zero \
             further improvement, not keep finding gains: before={before2:.3} after={after2:.3} m/s",
        );
    }

    /// Kernel loader for the new 9v-vi regression tests: tries the
    /// MissionPlanner-relative path first, then the repo-root fallback the
    /// debug binaries (mga_demo, debug_neptune_leg) already use — in this
    /// repo the kernels actually live at the workspace root, so without the
    /// fallback these tests would silently skip (as mga_evj_smoke above
    /// currently does — flagged, deliberately not modified here).
    fn load_test_almanac() -> Option<ephemeris::Almanac> {
        ephemeris::Almanac::new("kernels/de440s.bsp")
            .or_else(|_| ephemeris::Almanac::new("../kernels/de440s.bsp"))
            .ok()
    }

    /// Regression (9v-vi, bug found): the MGA departure burn must
    /// use the full escape energy, `dv = √(v∞² + 2μ/r_p) − v_c`, not the
    /// `√(v_c² + v∞²) − v_c` underestimate that was previously duplicated at
    /// four sites in this module (missing factor 2 on the escape term).
    ///
    /// Two anchors, both independent of the formula's own algebra:
    /// - At v∞ = 0 the burn is exactly escape-from-circular, `(√2 − 1)·v_c`
    ///   — the buggy formula returns 0 here.
    /// - At v∞ = 10 km/s from a ~200 km Earth parking orbit the burn is
    ///   ~7 089 m/s (hand-computed; Voyager-2-class C3 ≈ 100 km²/s²) — the
    ///   buggy formula returned ~5 098 m/s.
    #[test]
    fn departure_burn_uses_full_escape_energy() {
        let toml_str = std::fs::read_to_string("config/evj_flyby.toml")
            .expect("could not read config/evj_flyby.toml");
        let mut cfg: MissionConfig = toml::from_str(&toml_str).expect("parse evj_flyby.toml");
        // Force the default parking-orbit heuristic (Earth: radius + 200 km).
        cfg.trajectory.departure = None;

        let earth = body_models::TargetBody::by_name("Earth").unwrap();
        let r_park = earth.radius_m + 200_000.0;
        let v_c = (earth.mu_m3s2 / r_park).sqrt();

        let (dv0, r_park_used) =
            crate::design::departure_escape_dv_ms(&cfg, "Earth", 0.0).unwrap();
        assert!((r_park_used - r_park).abs() < 1.0, "unexpected parking orbit radius");
        let escape_dv = (2.0_f64.sqrt() - 1.0) * v_c;
        assert!(
            (dv0 - escape_dv).abs() < 1e-6 * escape_dv,
            "v∞=0 departure burn should equal escape-from-circular {escape_dv:.1} m/s, got {dv0:.1}"
        );

        let (dv10, _) = crate::design::departure_escape_dv_ms(&cfg, "Earth", 10_000.0).unwrap();
        assert!(
            (7_000.0..7_200.0).contains(&dv10),
            "v∞=10 km/s departure burn should be ~7 089 m/s, got {dv10:.1} \
             (~5 098 would mean the missing-factor-2 bug is back)"
        );
    }

    /// Regression (9v-vi, phantom-epoch bug fixed): with a
    /// nonzero departure offset, every encounter body must be queried at
    /// `dep_jd_base + offset + elapsed`, NOT with the offset double-counted.
    /// The old bug placed each encounter body at an epoch shifted by
    /// `dep_offset_days` from the true arrival — for a 40-day offset Venus
    /// moves ~1.2e8 km, so this discriminates unambiguously.
    ///
    /// Requires `kernels/de440s.bsp`; skips gracefully when absent.
    #[test]
    fn phantom_epoch_regression() {
        let Some(almanac) = load_test_almanac() else {
            eprintln!("[skip] phantom_epoch_regression: de440s.bsp not found");
            return;
        };
        let toml_str = std::fs::read_to_string("config/evj_flyby.toml")
            .expect("could not read config/evj_flyby.toml");
        let cfg: MissionConfig = toml::from_str(&toml_str).expect("parse evj_flyby.toml");
        let opt = cfg.optimization.as_ref().unwrap();
        let dep_epoch = crate::design::parse_epoch(opt.departure_epoch.as_deref().unwrap()).unwrap();
        let dep_jd_base = crate::design::epoch_to_jd(dep_epoch);
        let flybys = vec!["Venus".to_string()];

        // Scan a small chromosome grid until one evaluates feasibly — the
        // assertion only needs a single Some(ev).
        let offset_days = 40.0;
        let mut checked = false;
        'outer: for tof0 in [120.0, 160.0, 200.0, 260.0] {
            for theta in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] {
                let params = vec![
                    offset_days, 4_000.0, theta, 0.05, // dep offset, v∞, θ, φ
                    tof0, 0.4,                         // leg 0 TOF, η
                    800.0, 0.5,                        // leg 1 TOF, η
                    3.0, 0.3,                          // flyby r_p [radii], β
                    0.0, 0.0,                          // n_rev genes (N-as-gene): both legs single-rev
                ];
                let Some(ev) = evaluate_chromosome_detailed(
                    &params, &cfg, &almanac, dep_jd_base, &flybys,
                ) else { continue };

                // Venus (encounter body 1) must sit at dep_jd + tof0 …
                let correct_jd = dep_jd_base + offset_days + tof0;
                let (r_correct, _) = get_body_state(&almanac, "Venus", correct_jd).unwrap();
                let err_correct = (ev.body_rvs[1].0 - r_correct).norm();
                assert!(
                    err_correct < 1.0e3,
                    "Venus encounter state off by {:.3e} m from the correct epoch",
                    err_correct
                );
                // … and must NOT sit at the old buggy epoch (offset double-counted).
                let buggy_jd = correct_jd + offset_days;
                let (r_buggy, _) = get_body_state(&almanac, "Venus", buggy_jd).unwrap();
                let err_buggy = (ev.body_rvs[1].0 - r_buggy).norm();
                assert!(
                    err_buggy > 1.0e9,
                    "Venus encounter state suspiciously close ({err_buggy:.3e} m) to the \
                     double-counted epoch — phantom-epoch bug is back"
                );
                checked = true;
                break 'outer;
            }
        }
        assert!(checked, "no chromosome in the scan grid evaluated feasibly — widen the grid");
    }

    /// Regression (9v-vi): after a sequence-search run, (a) the on-disk
    /// `mga_params.csv` must hold the WINNER's totals, not the last-run
    /// sequence's (winner-overwrite bug fixed), and (b) no leg of
    /// the winner may dive below the solar-perihelion floor (Sun-dive
    /// exploit fixed).
    ///
    /// Requires `kernels/de440s.bsp`; skips gracefully when absent.
    #[test]
    fn sequence_search_winner_csv_and_perihelion_floor() {
        let Some(almanac) = load_test_almanac() else {
            eprintln!("[skip] sequence_search_winner_csv_and_perihelion_floor: de440s.bsp not found");
            return;
        };
        let toml_str = std::fs::read_to_string("config/earth_neptune_auto.toml")
            .expect("could not read config/earth_neptune_auto.toml");
        let mut cfg: MissionConfig = toml::from_str(&toml_str).expect("parse earth_neptune_auto.toml");

        let out_dir = std::env::temp_dir().join("mga_seq_search_regression_test");
        cfg.simulation.output_dir = out_dir.to_string_lossy().into_owned();
        let min_perihelion_m;
        {
            let opt = cfg.optimization.as_mut().unwrap();
            let mga = opt.mga.as_mut().unwrap();
            mga.de_population_size = 80;
            mga.de_generations     = 250;
            mga.de_restarts        = 1;
            min_perihelion_m = mga.min_solar_perihelion_m;
            mga.sequence_search.as_mut().unwrap().max_sequences_to_optimize = 2;
        }

        let r = run_mga(&cfg, &almanac, |_, _, _, _, _| {}, |_, _, _, _| {}).expect("run_mga failed");

        // (a) mga_params.csv holds the winner's total.
        let params_csv = std::fs::read_to_string(out_dir.join("mga_params.csv"))
            .expect("mga_params.csv not written");
        let data_line = params_csv.lines().nth(1).expect("mga_params.csv has no data row");
        let csv_total: f64 = data_line.split(',').next().unwrap().parse().unwrap();
        assert!(
            (csv_total - r.dv_total_ms).abs() < 0.5,
            "mga_params.csv dv_total {:.1} != returned winner {:.1} — winner-overwrite bug is back",
            csv_total, r.dv_total_ms
        );

        // (b) every leg of the winner respects the solar-perihelion floor.
        let opt = cfg.optimization.as_ref().unwrap();
        let dep_epoch = crate::design::parse_epoch(opt.departure_epoch.as_deref().unwrap()).unwrap();
        let dep_jd_base = crate::design::epoch_to_jd(dep_epoch);
        let winner_flybys: Vec<String> =
            r.body_sequence[1..r.body_sequence.len() - 1].to_vec();
        let ev = evaluate_chromosome_detailed(
            &r.best_params, &cfg, &almanac, dep_jd_base, &winner_flybys,
        ).expect("winner chromosome must re-evaluate feasibly");
        for (k, le) in ev.legs.iter().enumerate() {
            let rp_min = le.leg.rp_dep_m.min(le.leg.rp_lambert_m);
            assert!(
                rp_min >= min_perihelion_m * 0.999,
                "leg {k} perihelion {:.3e} m below the {:.3e} m floor — Sun-dive exploit is back",
                rp_min, min_perihelion_m
            );
        }
    }

    /// Phase 9w-vi: `check_config` must reject `scan_informed_window = true`
    /// without a `[optimization.mga.scan]` section — silently falling back
    /// to unscanned config bounds would defeat the whole point of the flag
    /// (routing around the leg_tof_days position-indexing bug class,
    /// the design notes Phase 9v).
    #[test]
    fn scan_informed_window_requires_scan_config() {
        let toml_str = std::fs::read_to_string("config/evj_flyby.toml")
            .expect("could not read config/evj_flyby.toml");
        let mut cfg: MissionConfig = toml::from_str(&toml_str).expect("parse evj_flyby.toml");
        {
            let opt = cfg.optimization.as_mut().unwrap();
            let mga = opt.mga.as_mut().unwrap();
            mga.scan_informed_window = true;
            mga.scan = None;
        }
        let errors = crate::config::check_config(&cfg);
        assert!(
            errors.iter().any(|e| e.contains("scan_informed_window") && e.contains("scan")),
            "expected a scan_informed_window/scan validation error, got: {errors:?}"
        );

        // Sanity: the same config WITH a scan section attached introduces no
        // new scan_informed_window error.
        {
            let opt = cfg.optimization.as_mut().unwrap();
            let mga = opt.mga.as_mut().unwrap();
            mga.scan = Some(crate::config::MgaScanConfigToml {
                horizon_years: 2.0,
                departure_step_days: 10.0,
                tof_grid_points_per_leg: 6,
                flyby_dv_max_ms: 3_000.0,
                max_records: 10_000,
            });
        }
        let errors2 = crate::config::check_config(&cfg);
        assert!(
            !errors2.iter().any(|e| e.contains("scan_informed_window")),
            "scan_informed_window with a real scan section should not error: {errors2:?}"
        );
    }

    /// Phase 9w-vi end-to-end: with `scan_informed_window = true`,
    /// `run_mga_fixed_sequence` must (a) derive departure-window/TOF bounds
    /// that genuinely contain the ballistic scan's own best feasible record
    /// for this exact sequence — proven independently below by re-running
    /// the raw scan and checking its best record falls inside the derived
    /// bounds, not by trusting the derivation blindly — and (b) complete
    /// the subsequent DE search without error using those derived bounds.
    ///
    /// Requires `kernels/de440s.bsp`; skips gracefully when absent.
    #[test]
    fn scan_informed_window_derives_sane_bounds_and_runs() {
        let Some(almanac) = load_test_almanac() else {
            eprintln!("[skip] scan_informed_window_derives_sane_bounds_and_runs: de440s.bsp not found");
            return;
        };
        let toml_str = std::fs::read_to_string("config/evj_flyby.toml")
            .expect("could not read config/evj_flyby.toml");
        let mut cfg: MissionConfig = toml::from_str(&toml_str).expect("parse evj_flyby.toml");

        let out_dir = std::env::temp_dir().join("mga_scan_informed_window_test");
        cfg.simulation.output_dir = out_dir.to_string_lossy().into_owned();

        // Fix the sequence to Earth -> Venus -> Jupiter (drop
        // sequence_search) so both the scan and the DE search analyze the
        // exact same, known 2-leg problem — this test is about the
        // bounds-derivation wiring, not sequence discovery.
        let flyby_bodies = vec!["Venus".to_string()];
        {
            let opt = cfg.optimization.as_mut().unwrap();
            let mga = opt.mga.as_mut().unwrap();
            mga.sequence_search = None;
            mga.flyby_bodies = flyby_bodies.clone();
            // Realistic per-leg TOF ranges for Earth->Venus (~100-220 days)
            // and Venus->Jupiter (~400-1000 days) — wide enough that the
            // scan (which spans these SAME bounds for its own TOF grid, see
            // compute_mga_scan_for_sequence) can actually find feasible
            // branches, narrower than the committed config's generic
            // [80,1200]x4 so the scan's grid resolution concentrates where
            // real solutions live.
            opt.departure_window_days = Some(5.0); // DE-search-only knob, unrelated to the scan's own horizon
            mga.leg_tof_days = vec![[100.0, 220.0], [400.0, 1000.0]];
            mga.de_population_size = 40;
            mga.de_generations     = 30;
            mga.de_restarts        = 1;
            mga.mbh.extra_random_chains = 4;
            mga.scan = Some(crate::config::MgaScanConfigToml {
                horizon_years: 2.0,
                departure_step_days: 10.0,
                tof_grid_points_per_leg: 20,
                flyby_dv_max_ms: 5_000.0,
                max_records: 50_000,
            });
            mga.scan_informed_window = true;
        }

        let scan_cfg = cfg.optimization.as_ref().unwrap().mga.as_ref().unwrap()
            .scan.as_ref().unwrap().clone();
        let (dep_jd_center, window_days, leg_tof_bounds) =
            derive_scan_informed_bounds(&cfg, &almanac, &flyby_bodies, &scan_cfg)
                .expect("scan-informed bounds derivation failed");

        // Sanity: the derived window/bounds are real, non-degenerate numbers.
        assert!(
            window_days > 0.0 && window_days.is_finite(),
            "derived window_days is not sane: {window_days}"
        );
        assert!(
            dep_jd_center.is_finite() && dep_jd_center > 2_400_000.0,
            "derived dep_jd_center is not a real JD: {dep_jd_center}"
        );
        assert_eq!(leg_tof_bounds.len(), 2, "expected 2 legs (Earth→Venus, Venus→Jupiter)");
        for (k, [lo, hi]) in leg_tof_bounds.iter().enumerate() {
            assert!(
                lo.is_finite() && hi.is_finite() && *lo < *hi,
                "leg {k} derived TOF bounds not sane: [{lo}, {hi}]"
            );
        }

        // Re-run the raw scan directly and confirm the derived window/bounds
        // really do contain its own best record.
        let (records, _capture_dvs, _names, _n_legs_evaluated, _n_dropped, _elapsed_s) =
            crate::mga_scan_run::compute_mga_scan_for_sequence(&cfg, &almanac, &flyby_bodies)
                .expect("raw scan for comparison failed");
        assert!(!records.is_empty(), "scan found no feasible records to compare against");
        let cost = |r: &ScanRecord| r.vinf_dep_ms + r.sum_flyby_dv_ms;
        let best = records.iter().min_by(|a, b| cost(a).partial_cmp(&cost(b)).unwrap()).unwrap();
        let best_dep_jd = 2_451_544.5 + best.dep_epoch_s / 86_400.0;
        assert!(
            (best_dep_jd - dep_jd_center).abs() <= window_days / 2.0 + 1e-6,
            "derived departure window (center JD {dep_jd_center:.2}, ±{:.2} days) does not \
             contain the scan's own best record's departure date (JD {best_dep_jd:.2})",
            window_days / 2.0
        );
        for (k, [lo, hi]) in leg_tof_bounds.iter().enumerate() {
            let tof_days_k = best.leg_tofs_s[k] / 86_400.0;
            assert!(
                tof_days_k >= *lo - 1e-6 && tof_days_k <= *hi + 1e-6,
                "leg {k} derived TOF bounds [{lo:.1}, {hi:.1}] do not contain the scan's own \
                 best record's TOF {tof_days_k:.1} days"
            );
        }

        // Real end-to-end path: run_mga_fixed_sequence must pick up
        // scan_informed_window and complete without error, using the
        // derived bounds instead of the deliberately-too-narrow config ones
        // set above.
        let result = run_mga_fixed_sequence(&cfg, &almanac, &flyby_bodies, &mut |_, _, _, _, _| {})
            .expect("run_mga_fixed_sequence with scan_informed_window=true failed");
        assert!(result.dv_total_ms.is_finite() && result.dv_total_ms > 0.0);
    }


    /// Phase 9y-h (revised): `arrival_dv_ms`'s `Flyby` branch is
    /// a closed-form floor check against `target_orbit_radius_m` -- not a
    /// function of any chromosome gene (the earlier gene-based design was
    /// replaced; see this function's doc comment for why). Verifies: no
    /// target configured -> pre-9y-h flat `0.0`; target at/above the body's
    /// safety floor -> `0.0` (always achievable, no ΔV cost); target below
    /// the floor -> a real, positive, `(floor - target)` penalty.
    /// `evj_flyby.toml` (a real, committed benchmark config) has no
    /// `[trajectory.capture]` section today, so the "not configured" case
    /// reproduces its actual real-world behaviour exactly.
    #[test]
    fn arrival_dv_flyby_is_closed_form_floor_check_not_gene_dependent() {
        let toml_str = std::fs::read_to_string("config/evj_flyby.toml")
            .expect("could not read config/evj_flyby.toml");
        let mut cfg: MissionConfig = toml::from_str(&toml_str).expect("parse evj_flyby.toml");
        assert!(matches!(cfg.mission.objective, MissionObjective::Flyby));
        assert!(
            cfg.trajectory.capture.as_ref().and_then(|c| c.target_orbit_radius_m).is_none(),
            "evj_flyby.toml is expected to have no configured capture radius (real-config baseline)"
        );

        let body_radius_m = 1.0e6;
        let dummy_ev = ChromosomeEval {
            legs: Vec::new(),
            v_inf_dep_ms: 0.0,
            v_inf_arr: Vector3::zeros(),
            dep_jd: 0.0,
            body_mus: vec![0.0],
            body_radii: vec![body_radius_m],
            body_rvs: Vec::new(),
            penalty_ms: 0.0,
        };

        // No target configured: the pre-9y-h flat 0.0.
        assert_eq!(arrival_dv_ms(&cfg, &dummy_ev), 0.0);

        // Target well above the floor: always zero cost, regardless of value.
        cfg.trajectory.capture = Some(crate::config::CaptureConfig {
            target_orbit_radius_m: Some(1.0e8),
            approach_v_inf_mps: None,
            terminator_orbit: false,
            capture_eccentricity: 0.0,
        });
        assert_eq!(arrival_dv_ms(&cfg, &dummy_ev), 0.0);

        // Target below the body's own safety floor (1.05 x body_radius_m):
        // a real, positive penalty equal to the shortfall.
        let floor_m = 1.05 * body_radius_m;
        let below_floor = 0.5 * body_radius_m;
        cfg.trajectory.capture.as_mut().unwrap().target_orbit_radius_m = Some(below_floor);
        let penalty = arrival_dv_ms(&cfg, &dummy_ev);
        assert!(
            (penalty - (floor_m - below_floor)).abs() < 1.0,
            "expected a {:.1} m floor penalty, got {penalty}", floor_m - below_floor
        );

        // Exact match against the floor: zero.
        cfg.trajectory.capture.as_mut().unwrap().target_orbit_radius_m = Some(floor_m);
        assert_eq!(arrival_dv_ms(&cfg, &dummy_ev), 0.0);
    }
}

// Compile-time proof that `Almanac` can be shared across MBH worker threads
// (parallel chains): ANISE's loaded context is immutable after
// construction. If a future `ephemeris`/`anise` upgrade breaks this, the
// build fails here instead of introducing silent unsoundness.
const _: () = {
    const fn assert_sync<T: Sync>() {}
    assert_sync::<Almanac>()
};
