//! V∞ Leveraging Transfer (VILT) boundary-value solver — tangent case only
//! (Phase 9x-v Stage 5).
//!
//! A VILT patches two Keplerian arcs through a shared "leveraging apse" at
//! radius `r_c`, with an instantaneous velocity-aligned maneuver at that
//! apse (the VILM). It is the field's purpose-built technique for a leg
//! immediately following a resonant/repeat flyby — exactly the leg type
//! that has absorbed 68-86% of total DSM budget in every generic
//! Lambert+DSM search this project has run (VEEGA, Cassini-2). Unlike the
//! generic MGA-1DSM leg model (`crate::mga_leg`), which places one free-
//! position DSM anywhere along a Lambert arc, VILT constrains the maneuver
//! to an apse and solves for the apse radius from the boundary conditions —
//! same inputs as Lambert (two position vectors, a TOF), so it slots into
//! the same outer-loop search architecture.
//!
//! # Scope reduction (deliberate, per the implementation plan)
//!
//! Only the **tangent** VILT is implemented: the low encounter occurs at a
//! spacecraft apse of the low arc (the free parameter `ξ_L` in the source
//! paper is fixed at zero). This removes one continuous degree of freedom
//! and yields a fully closed-form arc shape for the low arc — no root-solve
//! needed there, only for `r_c`. Non-tangent VILTs are out of scope for
//! this pass; revisit only if the tangent case doesn't close the gap it
//! targets (see the Phase 9x-v primer-vector re-check this module's
//! results are meant to feed).
//!
//! The exact per-quantity algebra below was re-derived from first
//! principles (general two-body conic geometry, Kepler's equation) rather
//! than trusted verbatim from the source paper — PDF text extraction
//! mangled the subscripted/rooted symbols in its Eqs. (1)-(15). The
//! paper's *structure* (apse-sharing two-arc decomposition, `r_c` as the
//! root-solving unknown, TOF as the target function) is what's implemented;
//! the derivation was independently verified by two different routes for
//! the exterior-domain case (an apoapsis-referenced conic vs. a unified
//! apse-phase-offset formula), which disagreed on a first pass and were
//! reconciled before trusting either.
//!
//! # Derivation
//!
//! Both encounter position vectors `r_low`, `r_high` and the total transfer
//! angle `Θ` between them (same convention as [`crate::lambert::transfer_angle_rad`])
//! are given, along with the target TOF. The **low arc** connects `r_low`
//! to `r_c`; since the low encounter is itself an apse (tangent condition)
//! and `r_c` is the arc's other apse, the low arc is an ellipse with apses
//! at exactly `r_low` and `r_c` — fully determined by `r_c` alone:
//!
//! ```text
//! a_L = (r_low + r_c) / 2
//! e_L = |r_c - r_low| / (r_c + r_low)
//! ```
//!
//! Apses of any ellipse are exactly π radians apart, so the low arc always
//! sweeps exactly half a revolution (plus `k_low` full extra revolutions):
//! `TOF_L = (k_low + 1/2) · T_L`, `T_L` the low arc's orbital period. This
//! also fixes `r_c`'s *direction*: diametrically opposite `r_low`.
//!
//! The **high arc** shares the same `r_c` as one of its apses (a velocity-
//! aligned burn at an apse keeps it an apse of the post-burn orbit too) but
//! `r_high` is generally not an apse of the high arc. Let `φ_apse = 0` when
//! `r_c` is the high arc's periapsis (`VilmDomain::Interior`) or `π` when
//! it's the apoapsis (`VilmDomain::Exterior`), and `ν_H = (Θ - π + φ_apse)`
//! reduced to `[0, 2π)` — the high arc's true anomaly at `r_high`, measured
//! conventionally from periapsis. Then, from the conic equation evaluated
//! at both `r_c` (ν = `φ_apse`) and `r_high` (ν = `ν_H`):
//!
//! ```text
//! e_H = (r_c - r_high) / (r_high·cos(ν_H) - r_c·cos(φ_apse))
//! a_H = r_c / (1 - e_H·cos(φ_apse))
//! ```
//!
//! `TOF_H` follows from Kepler's equation at `ν_H` (via
//! [`orbital_math::kepler::true_to_mean_anomaly`]), offset so that time is
//! measured from `r_c` (mean anomaly `φ_apse`) rather than from periapsis.
//! Total `TOF(r_c) = TOF_L(r_c) + TOF_H(r_c)` is a scalar function of `r_c`
//! alone; root-solving it against the target TOF is the entire boundary
//! value problem.
//!
//! # References
//! - Lantukh, D.V., Russell, R.P. & Campagnola, S., "The V-Infinity
//!   Leveraging Boundary Value Problem and Application in Spacecraft
//!   Trajectory Design", submitted to J. Spacecraft and Rockets (earlier
//!   version: AAS 12-162, AAS/AIAA Space Flight Mechanics Meeting, 2012).

use nalgebra::Vector3;
use orbital_math::kepler::{orbital_period_s, true_to_mean_anomaly};
use orbital_math::lambert::transfer_angle_rad;

/// Which apse of the shared leveraging orbit hosts the VILM burn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VilmDomain {
    /// Maneuver at spacecraft apoapsis (`D = +1` in the source paper).
    Exterior,
    /// Maneuver at spacecraft periapsis (`D = -1`).
    Interior,
}

impl VilmDomain {
    /// True anomaly of the leveraging apse itself, conventionally measured
    /// from periapsis: `0` if the apse IS periapsis, `π` if it's apoapsis.
    fn apse_phase_rad(self) -> f64 {
        match self {
            VilmDomain::Interior => 0.0,
            VilmDomain::Exterior => std::f64::consts::PI,
        }
    }
}

/// Result of a converged tangent VILT boundary-value solve.
#[derive(Clone, Debug)]
pub struct VilmResult {
    /// Leveraging apse radius [m] — the solved-for unknown.
    pub r_c_m: f64,
    /// Spacecraft velocity at `r_low` [m/s], on the low (pre-maneuver) arc.
    pub v_low: Vector3<f64>,
    /// Spacecraft velocity at `r_high` [m/s], on the high (post-maneuver) arc.
    pub v_high: Vector3<f64>,
    /// Signed leveraging maneuver at `r_c` [m/s]: high-arc apse speed minus
    /// low-arc apse speed (both purely tangential at the shared apse, same
    /// direction — a velocity-aligned burn only changes speed there).
    pub dv_leverage_ms: f64,
    pub tof_low_s: f64,
    pub tof_high_s: f64,
}

/// Per-`r_c` intermediate quantities, shared between the TOF root-solve and
/// the final velocity/output computation.
struct ArcGeometry {
    a_low: f64,
    #[allow(dead_code)] // not needed by solve_tangent_vilt yet; kept for future leg-output/plotting use
    e_low: f64,
    a_high: f64,
    e_high: f64,
    nu_high: f64,
    tof_low_s: f64,
    tof_high_s: f64,
}

fn arc_geometry(
    r_low_mag: f64,
    r_high_mag: f64,
    theta_rad: f64,
    r_c: f64,
    mu_m3s2: f64,
    domain: VilmDomain,
    k_low: u32,
    k_high: u32,
) -> Option<ArcGeometry> {
    if r_c <= 0.0 || !r_c.is_finite() {
        return None;
    }

    // Low arc: both ends are apses of the same ellipse — always valid for
    // any r_c > 0, r_low > 0.
    let a_low = 0.5 * (r_low_mag + r_c);
    let e_low = (r_c - r_low_mag).abs() / (r_c + r_low_mag);
    if a_low <= 0.0 {
        return None;
    }
    let t_low = orbital_period_s(a_low, mu_m3s2);
    let tof_low_s = (k_low as f64 + 0.5) * t_low;

    // High arc: r_c is one apse (role fixed by `domain`); r_high is a
    // general point at true anomaly ν_H (conventional, from periapsis).
    let phi_apse = domain.apse_phase_rad();
    let two_pi = 2.0 * std::f64::consts::PI;
    // Angle swept along the LOW arc from r_low to r_c is always exactly π
    // (mod 2π); the remainder of the total transfer angle sweeps the high
    // arc, starting from r_c's own conventional true anomaly (φ_apse).
    let nu_high = (theta_rad - std::f64::consts::PI + phi_apse).rem_euclid(two_pi);

    let denom = r_high_mag * nu_high.cos() - r_c * phi_apse.cos();
    if denom.abs() < 1e-9 {
        return None; // degenerate — r_c on the singular boundary
    }
    let e_high = (r_c - r_high_mag) / denom;
    if !(0.0..1.0).contains(&e_high) {
        return None; // not a physical ellipse for this r_c (exclusion zone)
    }
    let denom_a = 1.0 - e_high * phi_apse.cos();
    if denom_a.abs() < 1e-12 {
        return None;
    }
    let a_high = r_c / denom_a;
    if a_high <= 0.0 {
        return None;
    }

    let t_high = orbital_period_s(a_high, mu_m3s2);
    let n_high = two_pi / t_high;
    let m_high = true_to_mean_anomaly(nu_high, e_high).rem_euclid(two_pi);
    let m_apse = true_to_mean_anomaly(phi_apse, e_high).rem_euclid(two_pi);
    // Time elapsed since r_c (whose own mean anomaly is m_apse — exactly 0
    // for periapsis, π for apoapsis, by construction), reduced positive.
    let t_since_apse = (m_high - m_apse).rem_euclid(two_pi);
    let tof_high_s = (t_since_apse + two_pi * k_high as f64) / n_high;

    Some(ArcGeometry {
        a_low,
        e_low,
        a_high,
        e_high,
        nu_high,
        tof_low_s,
        tof_high_s,
    })
}

/// Scan `[lo, hi]` (log-spaced) for `residual(r_c)`, then adaptively
/// refine near every `None <-> Some` validity transition.
///
/// The feasible-ellipse sub-domain (`0 <= e_high < 1`) is generically an
/// interval bounded by the singularity where the high-arc denominator
/// vanishes on one side, and either the domain edge or an `e_high = 1`
/// crossing on the other — and this interval can be a tiny fraction of the
/// full `[lo, hi]` range (found empirically on a real VEEGA leg: a valid
/// window spanning ~6% of `r_high`, invisible to a uniform coarse scan). A
/// plain uniform scan over `[lo, hi]` has no way to know where to
/// concentrate resolution; bisecting on VALIDITY at every detected
/// transition finds the boundary exactly, then a fine sub-scan just inside
/// it reliably resolves whatever residual behavior lives in that interval.
///
/// Returns `(samples, refined_count)`, `samples` sorted by `r_c`.
fn scan_with_refinement(
    lo: f64,
    hi: f64,
    residual: &dyn Fn(f64) -> Option<f64>,
) -> (Vec<(f64, Option<f64>)>, usize) {
    const N_COARSE: usize = 200;
    let mut samples: Vec<(f64, Option<f64>)> = (0..N_COARSE)
        .map(|i| {
            let frac = i as f64 / (N_COARSE - 1) as f64;
            let r_c = lo * (hi / lo).powf(frac);
            (r_c, residual(r_c))
        })
        .collect();

    let mut refined = Vec::new();
    for w in samples.windows(2) {
        let (ra, resa) = w[0];
        let (rb, resb) = w[1];
        if resa.is_some() == resb.is_some() {
            continue;
        }
        // Bisect on validity to find the boundary precisely.
        let (mut valid_side, mut invalid_side) = if resa.is_some() { (ra, rb) } else { (rb, ra) };
        for _ in 0..40 {
            let mid = 0.5 * (valid_side + invalid_side);
            if residual(mid).is_some() { valid_side = mid; } else { invalid_side = mid; }
        }
        // Fine sub-scan from the boundary toward the known-valid neighbor —
        // the direction the (possibly narrow) valid window extends.
        let far_valid = if resa.is_some() { ra } else { rb };
        const N_FINE: usize = 100;
        for j in 0..N_FINE {
            let f = j as f64 / (N_FINE - 1) as f64;
            let r_c = valid_side + f * (far_valid - valid_side);
            refined.push((r_c, residual(r_c)));
        }
    }
    let n_refined = refined.len();
    samples.extend(refined);
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    (samples, n_refined)
}

/// Every sign change in a sorted `(r_c, Option<residual>)` sample list, in
/// ascending `r_c` order. The source paper documents up to two roots when
/// `TOF(r_c)` has an interior minimum (solutions `S = 1` — lower `r_c` — and
/// `S = 2` — higher `r_c`); this returns all brackets found so the caller
/// can pick.
fn find_all_sign_changes(samples: &[(f64, Option<f64>)]) -> Vec<(f64, f64)> {
    let mut brackets = Vec::new();
    let mut prev: Option<(f64, f64)> = None;
    for &(r_c, res) in samples {
        if let Some(res) = res {
            if let Some((r_prev, res_prev)) = prev {
                if res_prev * res < 0.0 {
                    brackets.push((r_prev, r_c));
                }
            }
            prev = Some((r_c, res));
        } else {
            prev = None;
        }
    }
    brackets
}

/// Which of the (up to two) `TOF(r_c)` roots to return, when `TOF(r_c)` has
/// an interior minimum (source paper's `S` descriptor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VilmSolution {
    /// Lower `r_c` (`S = 1`).
    Lower,
    /// Higher `r_c` (`S = 2`) — only exists when `TOF(r_c)` is non-monotonic;
    /// falls back to the only root found when just one exists.
    Upper,
}

/// Solve the tangent VILT boundary value problem: given a departure
/// position `r_low`, an arrival position `r_high`, a target time-of-flight,
/// and the leveraging domain, find the common apse radius `r_c` that
/// matches the target TOF, then report the maneuver and both encounter
/// velocities.
///
/// `k_low`/`k_high` — extra full spacecraft revolutions on the low/high
/// sub-arc beyond the minimum (`0` for the simplest direct case).
/// `prograde` — same convention as [`crate::lambert::transfer_angle_rad`].
/// `solution` — which root to return when more than one exists (see
/// [`VilmSolution`]).
///
/// Returns `None` if no `r_c` in the domain matches the target TOF.
pub fn solve_tangent_vilt(
    r_low: Vector3<f64>,
    r_high: Vector3<f64>,
    tof_target_s: f64,
    mu_m3s2: f64,
    domain: VilmDomain,
    prograde: bool,
    k_low: u32,
    k_high: u32,
    solution: VilmSolution,
) -> Option<VilmResult> {
    let r_low_mag = r_low.norm();
    let r_high_mag = r_high.norm();
    if r_low_mag <= 0.0 || r_high_mag <= 0.0 || tof_target_s <= 0.0 || mu_m3s2 <= 0.0 {
        return None;
    }

    let theta = transfer_angle_rad(
        [r_low.x, r_low.y, r_low.z],
        [r_high.x, r_high.y, r_high.z],
        prograde,
    );

    let (lo, hi) = match domain {
        VilmDomain::Exterior => (r_low_mag * (1.0 + 1e-6), r_low_mag * 1e4),
        VilmDomain::Interior => (r_low_mag * 1e-6, r_low_mag * (1.0 - 1e-6)),
    };

    let residual = |r_c: f64| -> Option<f64> {
        let g = arc_geometry(r_low_mag, r_high_mag, theta, r_c, mu_m3s2, domain, k_low, k_high)?;
        Some(g.tof_low_s + g.tof_high_s - tof_target_s)
    };

    let debug = std::env::var("VILM_DEBUG").is_ok();
    let (samples, n_refined) = scan_with_refinement(lo, hi, &residual);
    let brackets = find_all_sign_changes(&samples);
    let bracket = match solution {
        VilmSolution::Lower => brackets.first().copied(),
        VilmSolution::Upper => brackets.last().copied(),
    };
    if debug {
        let n_valid = samples.iter().filter(|(_, r)| r.is_some()).count();
        let res_min = samples.iter().filter_map(|(_, r)| *r).fold(f64::INFINITY, f64::min);
        let res_max = samples.iter().filter_map(|(_, r)| *r).fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "[VILM_DEBUG] domain={domain:?} k_low={k_low} k_high={k_high} solution={solution:?} \
             lo={lo:.3e} hi={hi:.3e} r_low={r_low_mag:.3e} r_high={r_high_mag:.3e} theta_deg={:.2} \
             tof_target_s={tof_target_s:.3e} n_valid={n_valid}/{} (refined +{n_refined}) \
             res_range=[{res_min:.3e}, {res_max:.3e}] n_brackets={} bracket_found={}",
            theta.to_degrees(), samples.len(), brackets.len(), bracket.is_some(),
        );
    }
    let (mut a, mut b) = bracket?;
    let (mut fa, _) = (residual(a)?, ());
    for _ in 0..80 {
        let mid = 0.5 * (a + b);
        let Some(fm) = residual(mid) else { break };
        if fa * fm <= 0.0 {
            b = mid;
        } else {
            a = mid;
            fa = fm;
        }
        if (b - a).abs() < 1e-6 * a.max(1.0) {
            break;
        }
    }
    let r_c = 0.5 * (a + b);
    let geom = arc_geometry(r_low_mag, r_high_mag, theta, r_c, mu_m3s2, domain, k_low, k_high)?;

    // Orbit-normal unit vector, consistent with transfer_angle_rad's own
    // flip convention (verified equivalent: flip happens exactly when
    // `(cross_z < 0) == prograde`).
    let h_raw = r_low.cross(&r_high);
    if h_raw.norm() < 1e-6 {
        return None; // degenerate (near-collinear) geometry
    }
    let cross_z = r_low.x * r_high.y - r_low.y * r_high.x;
    let flip = (cross_z < 0.0) == prograde;
    let h_hat = h_raw.normalize() * if flip { -1.0 } else { 1.0 };

    let r_low_hat = r_low / r_low_mag;
    let r_high_hat = r_high / r_high_mag;
    let t_low_hat = h_hat.cross(&r_low_hat);

    let v_low_speed = (mu_m3s2 * (2.0 / r_low_mag - 1.0 / geom.a_low)).sqrt();
    let v_low = v_low_speed * t_low_hat;

    let p_high = geom.a_high * (1.0 - geom.e_high * geom.e_high);
    let v_r = (mu_m3s2 / p_high).sqrt() * geom.e_high * geom.nu_high.sin();
    let v_t = (mu_m3s2 / p_high).sqrt() * (1.0 + geom.e_high * geom.nu_high.cos());
    let v_high = v_r * r_high_hat + v_t * h_hat.cross(&r_high_hat);

    let v_apse_low_side = (mu_m3s2 * (2.0 / r_c - 1.0 / geom.a_low)).sqrt();
    let v_apse_high_side = (mu_m3s2 * (2.0 / r_c - 1.0 / geom.a_high)).sqrt();

    Some(VilmResult {
        r_c_m: r_c,
        v_low,
        v_high,
        dv_leverage_ms: v_apse_high_side - v_apse_low_side,
        tof_low_s: geom.tof_low_s,
        tof_high_s: geom.tof_high_s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU_SUN: f64 = 1.327_124_400_18e20;
    const AU: f64 = 1.495_978_707e11;

    /// Sanity check on the low arc alone: a periapsis-to-apoapsis (K=0)
    /// transfer must take exactly half the orbital period, for ANY
    /// consistent (r_low, r_c) pair — this is a geometric identity, not
    /// something the root-solve should need to touch.
    #[test]
    fn low_arc_tof_is_exactly_half_period() {
        let r_low = 0.8 * AU;
        let r_c = 1.4 * AU;
        let geom = arc_geometry(r_low, 1.2 * AU, 3.0, r_c, MU_SUN, VilmDomain::Exterior, 0, 0)
            .expect("valid geometry");
        let a_low = 0.5 * (r_low + r_c);
        let expected = 0.5 * orbital_period_s(a_low, MU_SUN);
        assert!(
            (geom.tof_low_s - expected).abs() < 1e-6,
            "low arc TOF {} != T/2 {}",
            geom.tof_low_s,
            expected
        );
    }

    /// Round-trip: propagate the low arc from r_low by tof_low_s via the
    /// existing universal-variable propagator; the result must land at
    /// radius r_c (within numerical tolerance) — cross-checks the tangent
    /// low-arc formula against an independent, already-trusted propagator.
    #[test]
    fn low_arc_reaches_r_c_when_propagated() {
        use orbital_math::kepler::propagate_kepler;
        use nalgebra::Vector3;

        let r_low_mag = 0.9 * AU;
        let r_c = 1.3 * AU;
        let a_low = 0.5 * (r_low_mag + r_c);
        let v_peri = (MU_SUN * (2.0 / r_low_mag - 1.0 / a_low)).sqrt();
        let r0 = Vector3::new(r_low_mag, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_peri, 0.0);
        let t_half = 0.5 * orbital_period_s(a_low, MU_SUN);

        let (r1, _) = propagate_kepler(r0, v0, t_half, MU_SUN).expect("propagation ok");
        assert!(
            (r1.norm() - r_c).abs() / r_c < 1e-6,
            "expected r_c={:.3e}, got {:.3e}",
            r_c,
            r1.norm()
        );
        // Half a revolution lands diametrically opposite the start.
        assert!(
            (r1.normalize() + r0.normalize()).norm() < 1e-6,
            "expected r1 opposite r0, got r1_hat={:?} r0_hat={:?}",
            r1.normalize(),
            r0.normalize()
        );
    }

    /// Interior-domain e_high/a_high formula round-trip: construct a known
    /// (a_H, e_H) high arc with periapsis at a chosen r_c, sample r_high at
    /// a chosen true anomaly, then verify `arc_geometry` recovers the same
    /// (a_H, e_H) from (r_c, r_high, that true anomaly encoded via Θ).
    #[test]
    fn interior_high_arc_recovers_known_ellipse() {
        let a_h_true = 1.6 * AU;
        let e_h_true = 0.35;
        let r_c = a_h_true * (1.0 - e_h_true); // periapsis
        let nu_h_true = 2.1_f64; // rad, arbitrary interior point
        let p = a_h_true * (1.0 - e_h_true * e_h_true);
        let r_high_mag = p / (1.0 + e_h_true * nu_h_true.cos());

        // Θ chosen so that theta - pi + phi_apse(=0) == nu_h_true (mod 2pi).
        let theta = (nu_h_true + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI);
        let r_low_mag = 0.85 * AU; // arbitrary, only affects the low arc

        let geom = arc_geometry(
            r_low_mag, r_high_mag, theta, r_c, MU_SUN, VilmDomain::Interior, 0, 0,
        )
        .expect("valid geometry");

        assert!((geom.a_high - a_h_true).abs() / a_h_true < 1e-9);
        assert!((geom.e_high - e_h_true).abs() < 1e-9);
    }

    /// Exterior-domain analogue of the above — this is the case whose
    /// derivation disagreed on a first pass (apoapsis-referenced true
    /// anomaly) and was corrected; this test pins the fix.
    #[test]
    fn exterior_high_arc_recovers_known_ellipse() {
        let a_h_true = 1.6 * AU;
        let e_h_true = 0.35;
        let r_c = a_h_true * (1.0 + e_h_true); // apoapsis
        let nu_h_true = 4.4_f64; // rad, arbitrary point (conventional true anomaly)
        let p = a_h_true * (1.0 - e_h_true * e_h_true);
        let r_high_mag = p / (1.0 + e_h_true * nu_h_true.cos());

        // Θ chosen so that theta - pi + phi_apse(=pi) == nu_h_true (mod 2pi)
        // => theta == nu_h_true (mod 2pi).
        let theta = nu_h_true.rem_euclid(2.0 * std::f64::consts::PI);
        let r_low_mag = 0.7 * AU;

        let geom = arc_geometry(
            r_low_mag, r_high_mag, theta, r_c, MU_SUN, VilmDomain::Exterior, 0, 0,
        )
        .expect("valid geometry");

        assert!(
            (geom.a_high - a_h_true).abs() / a_h_true < 1e-9,
            "a_high: expected {a_h_true:.6e}, got {:.6e}",
            geom.a_high
        );
        assert!(
            (geom.e_high - e_h_true).abs() < 1e-9,
            "e_high: expected {e_h_true:.6}, got {:.6}",
            geom.e_high
        );
    }

    /// End-to-end: build a TRUE tangent VILT by construction (a real low
    /// arc + a real high arc sharing a common apse, propagated to get
    /// r_low/r_high/TOF), then confirm `solve_tangent_vilt` recovers a
    /// consistent r_c and matches the constructed TOF and velocities.
    #[test]
    fn solves_constructed_tangent_vilt_end_to_end() {
        use orbital_math::kepler::propagate_kepler;

        let r_c_true = 1.5 * AU;
        let r_low_mag = 1.0 * AU;
        let a_low = 0.5 * (r_low_mag + r_c_true);
        let v_low_peri = (MU_SUN * (2.0 / r_low_mag - 1.0 / a_low)).sqrt();
        let r_low = Vector3::new(r_low_mag, 0.0, 0.0);
        let v_low_true = Vector3::new(0.0, v_low_peri, 0.0);
        let t_low = 0.5 * orbital_period_s(a_low, MU_SUN);

        // Propagate the low arc to r_c (lands at (-r_c_true, 0, 0)).
        let (r_c_vec, _) = propagate_kepler(r_low, v_low_true, t_low, MU_SUN).unwrap();

        // High arc: apoapsis at r_c_true (exterior domain), some e_H, a_H.
        let e_h = 0.4;
        let a_h = r_c_true / (1.0 + e_h);
        let v_apo_high = (MU_SUN * (2.0 / r_c_true - 1.0 / a_h)).sqrt();
        // Velocity at apoapsis is tangential, same rotational sense as the
        // low arc (prograde, +z angular momentum): direction ĥ×r̂ with ĥ=+z.
        let r_c_hat = r_c_vec.normalize();
        let h_hat = Vector3::new(0.0, 0.0, 1.0);
        let v_c_high = v_apo_high * h_hat.cross(&r_c_hat);

        let t_high_partial = 0.35 * orbital_period_s(a_h, MU_SUN);
        let (r_high, v_high_true) =
            propagate_kepler(r_c_vec, v_c_high, t_high_partial, MU_SUN).unwrap();

        let tof_target = t_low + t_high_partial;

        let result = solve_tangent_vilt(
            r_low, r_high, tof_target, MU_SUN, VilmDomain::Exterior, true, 0, 0, VilmSolution::Lower,
        )
        .expect("should solve the constructed VILT");

        assert!(
            (result.r_c_m - r_c_true).abs() / r_c_true < 1e-4,
            "r_c: expected {r_c_true:.6e}, got {:.6e}",
            result.r_c_m
        );
        assert!(
            (result.v_low - v_low_true).norm() < 1.0,
            "v_low mismatch: {:?} vs {:?}",
            result.v_low,
            v_low_true
        );
        assert!(
            (result.v_high - v_high_true).norm() < 1.0,
            "v_high mismatch: {:?} vs {:?}",
            result.v_high,
            v_high_true
        );
    }

    /// Infeasible target TOF (far outside anything reachable) returns None
    /// rather than panicking or returning a garbage root.
    #[test]
    fn infeasible_tof_returns_none() {
        let r_low = Vector3::new(AU, 0.0, 0.0);
        let r_high = Vector3::new(0.0, 1.2 * AU, 0.0);
        let result = solve_tangent_vilt(
            r_low, r_high, 1e-3, MU_SUN, VilmDomain::Interior, true, 0, 0, VilmSolution::Lower,
        );
        assert!(result.is_none());
    }
}
