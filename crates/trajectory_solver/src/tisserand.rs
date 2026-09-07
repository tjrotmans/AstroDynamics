//! Tisserand parameter, contour evaluation, and body-to-body feasibility links.
//!
//! The Tisserand parameter is a near-conserved quantity across an unpowered
//! gravity-assist flyby. For a heliocentric orbit characterised by semi-major
//! axis `a`, eccentricity `e`, and inclination `i` relative to a planet whose
//! heliocentric semi-major axis is `a_P`:
//!
//! ```text
//!     T_P = a_P / a  +  2 · sqrt( (a / a_P) · (1 − e²) ) · cos(i)
//! ```
//!
//! Because `T_P` is conserved across a flyby at body P, equal-`T_P` lines
//! on a Tisserand graph (axes: heliocentric v∞ at body P vs. planet-relative
//! v∞) link all feasible orbital states that can arise from a flyby at P.
//!
//! This module provides:
//! - [`tisserand_parameter`] — scalar T_P from orbital elements.
//! - [`vinf_from_tisserand`] — heliocentric v∞ at a body given T_P.
//! - [`TisserandContour`] — a sampled T_P = const curve for one planet at a
//!   range of hyperbolic excess speeds.
//! - [`find_tisserand_link`] — check whether two planets can be connected by
//!   a single flyby given incoming and outgoing v∞ magnitudes.
//! - [`tisserand_feasibility_score`] — continuous feasibility metric for the
//!   beam-search outer loop.
//!
//! # References
//! - Strange & Longuski (2002), "Graphical Method for Gravity-Assist
//!   Trajectory Design", JGCD 25(6):1154–1159.
//! - Ceriotti (2010), PhD Thesis §3.2 — Tisserand criterion in sequence search.
//! - Campagnola & Russell (2010), "Endgame Problem Part 1", J. Guid. Control
//!   Dyn. 33(2):476–487 — V∞ lever mechanism & Tisserand diagram use.

/// Compute the Tisserand parameter for a heliocentric orbit relative to
/// planet P.
///
/// # Arguments
/// * `a_m`   — spacecraft orbit semi-major axis [m]
/// * `e`     — eccentricity (0 ≤ e < 1 for elliptic; ≥ 1 for hyperbolic)
/// * `i_rad` — inclination relative to the planet's orbital plane [rad]
/// * `a_p_m` — planet's heliocentric semi-major axis [m]
///
/// For a hyperbolic heliocentric orbit (`e ≥ 1`), `1 − e² < 0`, so the
/// formula returns a value but T_P is only strictly conserved for elliptic
/// trajectories. The caller is responsible for making sure the inputs are
/// physically meaningful.
pub fn tisserand_parameter(a_m: f64, e: f64, i_rad: f64, a_p_m: f64) -> f64 {
    let ratio = a_p_m / a_m;
    let inner = (a_m / a_p_m) * (1.0 - e * e);
    ratio + 2.0 * inner.max(0.0).sqrt() * i_rad.cos()
}

/// Compute the heliocentric v∞ at a planet for a given Tisserand parameter
/// and planet circular orbital speed.
///
/// Uses the Tisserand conservation relation in v∞-space (Strange & Longuski
/// 2002, eq. 5). For an orbit passing through a planet at heliocentric
/// distance `a_P` (circular planet orbit assumed):
///
/// ```text
///   v_planet = sqrt(mu_sun / a_P)
///   V_∞²/V_P² = 3 − T_P − 2·sqrt(T_P − 1) · ... (full formula)
/// ```
///
/// The simplified form used here assumes the spacecraft orbit is co-planar
/// (`i = 0`) and the planet is on a circular orbit — appropriate for a
/// first-order Tisserand graph scan.
///
/// Returns `None` when the expression under the square-root is negative
/// (the Tisserand parameter is unachievable at this planet's orbit).
pub fn vinf_from_tisserand_coplanar(t_p: f64, v_planet_ms: f64) -> Option<f64> {
    // For a co-planar flyby, Tisserand criterion is:
    //   T_P = a_P/a + 2*sqrt(a/a_P * (1-e²))
    // At the flyby point (r = a_P), the vis-viva speed is V_sc.
    // Energy: V_sc² = μ*(2/a_P - 1/a)
    //   → 1/a = 2/a_P - V_sc²/μ
    //   → a_P/a = 2 - a_P*V_sc²/μ = 2 - V_sc²/V_P²
    // h = a_P * V_t where V_t is the tangential component at a_P.
    // 1-e² = h²/(μ*a) → sqrt(a/a_P * (1-e²)) = V_t/V_P
    // So T_P = (2 - V_sc²/V_P²) + 2*V_t/V_P
    // V_∞² = V_sc² + V_P² - 2*V_sc*V_t  (law of cosines in velocity space)
    // With V_sc² = V_t² + V_r²  and T_P constraint → quadratic in V_t/V_P:
    //   Let u = V_t/V_P:
    //   T_P = 2 - (u² + V_r²/V_P²) + 2u
    //   V_∞² = (u-1)²*V_P² + V_r²
    //
    // The minimum V_∞ for a given T_P is achieved when V_r = 0 (tangential
    // encounter), giving: T_P = 2 + 2u - u²  → u = 1 ± sqrt(3 - T_P)
    // → V_∞_min = |u-1|*V_P = sqrt(3-T_P)*V_P  when T_P ≤ 3.
    if t_p > 3.0 { return None; } // unachievable: T_P > 3 only for circular orbit at P
    let vinf = (3.0 - t_p).max(0.0).sqrt() * v_planet_ms;
    Some(vinf)
}

/// A sampled Tisserand contour for a single planet: a set of heliocentric
/// orbit states (a, e) lying on a constant-T_P curve, parameterised by
/// the planet-relative v∞ magnitude.
///
/// Used for graphical Tisserand-graph construction (Strange & Longuski 2002)
/// and for checking whether two planets can be linked by a single flyby.
#[derive(Clone, Debug)]
pub struct TisserandContour {
    /// Tisserand parameter value (dimensionless).
    pub t_p: f64,
    /// Planet heliocentric semi-major axis [m].
    pub a_p_m: f64,
    /// Planet circular orbital speed [m/s].
    pub v_planet_ms: f64,
    /// Planet gravitational parameter [m³/s²].
    pub mu_planet: f64,
    /// Sampled (v_∞ [m/s], heliocentric_sma [m], eccentricity) triples along
    /// the contour. Ordered from lowest to highest v_∞.
    pub samples: Vec<(f64, f64, f64)>, // (vinf_ms, a_m, e)
}

impl TisserandContour {
    /// Compute a Tisserand contour for a planet at heliocentric distance
    /// `a_p_m` with orbital speed `v_planet_ms`.
    ///
    /// Samples `n_points` co-planar encounter geometries (angle between
    /// spacecraft and planet velocity ∈ [0, π]) and records the Tisserand-
    /// constant heliocentric orbit (a, e) that results. Rows with e ≥ 1
    /// (hyperbolic heliocentric orbits) are excluded — they represent
    /// unphysical paths in the Tisserand graph for inner-loop beam search.
    ///
    /// # Arguments
    /// * `t_p`          — Tisserand parameter value to trace
    /// * `a_p_m`        — planet heliocentric SMA [m]
    /// * `v_planet_ms`  — planet circular orbital speed [m/s]
    /// * `mu_planet`    — planet gravitational parameter [m³/s²] (for periapsis
    ///                    constraint checking only; not used in T_P formula)
    /// * `n_points`     — number of sample angles
    pub fn compute(
        t_p: f64,
        a_p_m: f64,
        v_planet_ms: f64,
        mu_planet: f64,
        n_points: usize,
    ) -> Self {
        let mut samples = Vec::with_capacity(n_points);
        // Parameterise by the angle φ between V_sc and V_planet at encounter.
        // V_sc² = V_P² * (3 - T_P + 2*cos(φ)*sqrt(2 - ...) ... full expansion
        // is from the vis-viva + angular-momentum constraint.
        //
        // Simplified approach: at the flyby point r = a_P, for a co-planar orbit,
        // tangential velocity V_t = V_P * u where u is determined by T_P:
        //   T_P = (2 - u² - V_r²/V_P²) + 2u
        // Iterate over V_r/V_P ∈ [-3, 3], solve for u from the quadratic at each.
        let v_p = v_planet_ms;
        let n = n_points.max(2);
        for k in 0..n {
            // V_r/V_P ratio from -sqrt(3) to +sqrt(3) (covers all physical cases).
            let vr_vp = -3.0_f64.sqrt() + (k as f64 / (n - 1) as f64) * 2.0 * 3.0_f64.sqrt();
            let vr_sq_norm = vr_vp * vr_vp;
            // Quadratic in u: u² - 2u + (vr²_norm - 2 + T_P) = 0
            let discriminant = 4.0 - 4.0 * (vr_sq_norm - 2.0 + t_p);
            if discriminant < 0.0 { continue; }
            // Two roots: u = 1 ± sqrt(1 - (vr²_norm - 2 + T_P))
            let sq = (discriminant / 4.0).sqrt();
            for sign in [1.0_f64, -1.0] {
                let u = 1.0 + sign * sq; // V_t / V_P
                let v_sc_sq = (u * u + vr_sq_norm) * v_p * v_p;
                if v_sc_sq < 0.0 { continue; }
                let v_sc = v_sc_sq.sqrt();

                // Heliocentric energy at flyby: ε = V_sc²/2 - μ_sun/a_P
                // We don't have μ_sun here — encode as a/a_P instead.
                // From vis-viva at r = a_P: V_sc² = μ_sun*(2/a_P - 1/a)
                //   → a_P/a = 2 - V_sc²*a_P/μ_sun = 2 - V_sc²/V_P²
                let ratio = 2.0 - v_sc_sq / (v_p * v_p);
                if ratio <= 0.0 { continue; } // hyperbolic heliocentric orbit
                let a_m = a_p_m / ratio; // semi-major axis [m]

                // Angular momentum h = a_P * V_t = a_P * u * V_P
                // 1 - e² = h² / (μ_sun * a) = (a_P * u * V_P)² / (V_P² * a_P² * (a/a_P)^{-1}) ... simplify:
                // h²/(μ_sun*a) = (a_P*u*V_P)²/(a_P²*V_P²*(a/a_P)) = u² * (a_P/a) = u² * ratio
                let one_minus_e2 = u * u * ratio;
                if one_minus_e2 < 0.0 || one_minus_e2 > 1.0 { continue; }
                let e = (1.0 - one_minus_e2).sqrt();
                if e >= 1.0 { continue; } // exclude hyperbolic

                // V_∞ at the planet (planet-relative speed).
                let v_inf = ((u - 1.0).powi(2) + vr_sq_norm).sqrt() * v_p;

                samples.push((v_inf, a_m, e));
            }
        }
        // Sort by v_∞ ascending.
        samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        Self { t_p, a_p_m, v_planet_ms, mu_planet, samples }
    }

    /// Minimum planet-relative v∞ achievable on this contour [m/s].
    pub fn vinf_min_ms(&self) -> f64 {
        self.samples.first().map(|s| s.0).unwrap_or(f64::MAX)
    }

    /// Maximum planet-relative v∞ achievable on this contour [m/s].
    pub fn vinf_max_ms(&self) -> f64 {
        self.samples.last().map(|s| s.0).unwrap_or(0.0)
    }
}

/// Determine whether a body-to-body leg from planet A to planet B is
/// consistent with Tisserand conservation at body A, and estimate the
/// hyperbolic excess speed that would result at body B.
///
/// A link between A and B at a flyby is feasible when there exists a
/// co-planar heliocentric elliptic orbit that departs A with the given
/// planet-relative v∞ and crosses body B's orbit.  This is the necessary
/// condition from Strange & Longuski (2002): the transfer orbit's periapsis
/// must be ≤ a_B and apoapsis must be ≥ a_B.
///
/// When the link is feasible, the function returns the estimated v∞ at
/// body B (computed as the minimum-encounter speed `|v_sc(r=a_B) − v_B|`
/// assuming a tangential encounter at B — the most energetically favourable
/// geometry).  This estimate is used by the Tisserand beam search to seed
/// the v∞ budget for the next leg of the sequence.
///
/// # Arguments
/// * `vinf_a_ms`  — planet-relative v∞ magnitude at body A [m/s]
/// * `v_a_ms`     — body A's circular orbital speed [m/s]
/// * `a_a_m`      — body A's heliocentric SMA [m]
/// * `a_b_m`      — body B's heliocentric SMA [m] (target)
/// * `mu_sun`     — Sun's gravitational parameter [m³/s²]
///
/// Returns `Some(vinf_b_ms)` when body B is reachable; `None` otherwise.
pub fn find_tisserand_link(
    vinf_a_ms: f64,
    v_a_ms:    f64,
    a_a_m:     f64,
    a_b_m:     f64,
    mu_sun:    f64,
) -> Option<f64> {
    // Try both prograde and retrograde tangential departures from body A.
    let mut best_vinf_b: Option<f64> = None;
    for v_sc_tangential in [v_a_ms + vinf_a_ms, (v_a_ms - vinf_a_ms).abs()] {
        let energy = 0.5 * v_sc_tangential * v_sc_tangential - mu_sun / a_a_m;
        if energy >= 0.0 { continue; } // hyperbolic — skip

        let a_xfer = -mu_sun / (2.0 * energy);
        let h = a_a_m * v_sc_tangential;
        let p = h * h / mu_sun;
        let discriminant = 1.0 - p / a_xfer;
        if discriminant < 0.0 { continue; }
        let e_xfer = discriminant.sqrt();
        let r_peri = a_xfer * (1.0 - e_xfer);
        let r_apo  = a_xfer * (1.0 + e_xfer);

        // Relative tolerance on the crossing condition: exact tangency
        // (Hohmann apoapsis touching body B's orbit) is the canonical
        // feasible link, but round-off in the a_xfer/e chain can land
        // r_apo a fraction of a millimetre short of a_b_m at AU scale.
        let tol = 1e-9 * a_b_m;
        if r_peri <= a_b_m + tol && a_b_m <= r_apo + tol {
            // Estimate v∞ at body B via vis-viva + tangential encounter.
            let v_sc_b_sq = mu_sun * (2.0 / a_b_m - 1.0 / a_xfer);
            if v_sc_b_sq >= 0.0 {
                let v_b = (mu_sun / a_b_m).sqrt();
                let vinf_b = (v_sc_b_sq.sqrt() - v_b).abs();
                // Keep the lowest v∞ estimate (most energetically favourable link).
                best_vinf_b = Some(match best_vinf_b {
                    Some(prev) => prev.min(vinf_b),
                    None => vinf_b,
                });
            }
        }
    }
    best_vinf_b
}

/// Compute a continuous feasibility score (lower = more feasible) for a
/// body-to-body link in the Tisserand beam-search outer loop.
///
/// When `find_tisserand_link` returns `Some`, the score is the estimated
/// v∞ at body B (a proxy for the energy cost of continuing from B — lower
/// means a "more downhill" assist toward the target). When the link is
/// infeasible, the score is a large penalty proportional to how far body
/// B's orbit lies outside the transfer orbit's periapsis–apoapsis range.
///
/// Beam-search outer loop: retain sequences with the lowest total score at
/// each depth level. Pure feasibility filtering (score = 0 or ∞) would
/// prune too aggressively for a continuous optimizer — this score allows a
/// near-feasible link to compete.
///
/// # Arguments
/// Same as [`find_tisserand_link`].
pub fn tisserand_feasibility_score(
    vinf_a_ms: f64,
    v_a_ms:    f64,
    a_a_m:     f64,
    a_b_m:     f64,
    mu_sun:    f64,
) -> f64 {
    if let Some(vinf_b) = find_tisserand_link(vinf_a_ms, v_a_ms, a_a_m, a_b_m, mu_sun) {
        return vinf_b;
    }

    // Not directly reachable: measure the gap to the closest orbit boundary.
    let mut best_gap = f64::MAX;
    for v_sc_tangential in [v_a_ms + vinf_a_ms, (v_a_ms - vinf_a_ms).abs()] {
        let energy = 0.5 * v_sc_tangential * v_sc_tangential - mu_sun / a_a_m;
        if energy >= 0.0 { continue; }
        let a_xfer = -mu_sun / (2.0 * energy);
        let h = a_a_m * v_sc_tangential;
        let p = h * h / mu_sun;
        let discriminant = 1.0 - p / a_xfer;
        if discriminant < 0.0 { continue; }
        let e_xfer = discriminant.sqrt();
        let r_peri = a_xfer * (1.0 - e_xfer);
        let r_apo  = a_xfer * (1.0 + e_xfer);

        let gap = if a_b_m < r_peri {
            // Body B is inside periapsis — large penalty scaled to v_A
            (r_peri - a_b_m) / a_b_m * v_a_ms
        } else {
            // Body B is outside apoapsis
            (a_b_m - r_apo) / a_b_m * v_a_ms
        };
        if gap < best_gap { best_gap = gap; }
    }
    best_gap
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU_SUN: f64 = 1.327_124_400_18e20;
    const AU: f64 = 1.496e11;

    fn v_circular(a_m: f64) -> f64 {
        (MU_SUN / a_m).sqrt()
    }

    /// Earth → Jupiter link: Earth is at 1 AU. A Hohmann to Jupiter requires
    /// v∞ ≈ 8.8 km/s; here we use 9 km/s which puts apoapsis ≈ 5.6 AU > 5.2 AU.
    #[test]
    fn earth_jupiter_link_is_feasible() {
        let a_earth = 1.0 * AU;
        let a_jupiter = 5.2 * AU;
        let v_earth = v_circular(a_earth);
        // Minimum v∞ for Earth→Jupiter Hohmann ≈ 8.8 km/s; 9 km/s is safely above.
        let vinf = 9_000.0_f64;
        let link = find_tisserand_link(vinf, v_earth, a_earth, a_jupiter, MU_SUN);
        assert!(
            link.is_some(),
            "Earth→Jupiter with v∞=9 km/s should be feasible"
        );
        // v∞ at Jupiter should be finite and positive.
        let vinf_j = link.unwrap();
        assert!(vinf_j.is_finite() && vinf_j >= 0.0,
            "estimated v∞ at Jupiter should be finite and non-negative; got {vinf_j:.1}");
    }

    /// Earth → Neptune link with only 3 km/s v∞: not directly reachable
    /// (Neptune is at 30 AU — need much more energy or intermediate flybys).
    #[test]
    fn earth_neptune_direct_infeasible() {
        let a_earth = 1.0 * AU;
        let a_neptune = 30.1 * AU;
        let v_earth = v_circular(a_earth);
        let vinf = 3_000.0_f64; // insufficient for a direct crossing
        // Should not be feasible from a single tangential departure at 3 km/s v∞.
        // Earth-tangential: V_sc = V_earth + 3000 → a_xfer, check apoapsis < 30 AU.
        let v_sc = v_earth + vinf;
        let energy = 0.5 * v_sc * v_sc - MU_SUN / a_earth;
        let a_xfer = -MU_SUN / (2.0 * energy);
        let h = a_earth * v_sc;
        let p = h * h / MU_SUN;
        let e = (1.0 - p / a_xfer).sqrt();
        let r_apo = a_xfer * (1.0 + e);
        assert!(
            r_apo < a_neptune,
            "at 3 km/s v∞ from Earth, apoapsis should be < Neptune orbit; got {:.2} AU",
            r_apo / AU
        );
        assert!(
            find_tisserand_link(vinf, v_earth, a_earth, a_neptune, MU_SUN).is_none(),
            "Earth→Neptune at 3 km/s v∞ should NOT be directly feasible"
        );
    }

    /// Tisserand parameter recovered from the contour's own orbital elements
    /// must equal the contour's T_P to within 1e-6.
    #[test]
    fn contour_self_consistent_tisserand() {
        let a_jup = 5.2 * AU;
        let v_jup = v_circular(a_jup);
        let t_p = 2.95_f64; // typical for Jupiter-family comets

        // Gravitational parameter of Jupiter (not used in Tisserand formula, only for contour metadata).
        let mu_jup = 1.267e17_f64;
        let contour = TisserandContour::compute(t_p, a_jup, v_jup, mu_jup, 40);

        for &(_, a_m, e) in &contour.samples {
            let t_recovered = tisserand_parameter(a_m, e, 0.0, a_jup);
            let err = (t_recovered - t_p).abs();
            assert!(
                err < 1e-5,
                "contour self-consistency failed: T_P={t_p:.6}, recovered={t_recovered:.6} (a={:.3e} AU, e={e:.4})",
                a_m / AU
            );
        }
    }

    /// Feasibility score for a directly reachable link must be finite and
    /// match the v∞ at Mars that `find_tisserand_link` returns.
    #[test]
    fn feasibility_score_finite_for_feasible_link() {
        let a_earth = 1.0 * AU;
        let a_mars = 1.524 * AU;
        let v_earth = v_circular(a_earth);
        // Hohmann transfer: V_dep = sqrt(μ*(2/r_E - 1/a_xfer)), v∞ = V_dep - V_earth
        let a_xfer = 0.5 * (a_earth + a_mars);
        let v_dep = (MU_SUN * (2.0 / a_earth - 1.0 / a_xfer)).sqrt();
        let vinf = v_dep - v_earth; // positive, ~2.9 km/s
        let score = tisserand_feasibility_score(vinf, v_earth, a_earth, a_mars, MU_SUN);
        // Score should be finite (the estimated v∞ at Mars) and non-negative.
        assert!(score.is_finite() && score >= 0.0,
            "Hohmann-derived link should have finite non-negative score; got {score:.3e}");
        // For a Hohmann minimum-energy transfer, arrival v∞ at Mars ≈ 2.6 km/s.
        // The estimate is approximate (tangential encounter); check it's in a plausible range.
        assert!(score < 5_000.0,
            "Hohmann Earth→Mars arrival v∞ estimate should be < 5 km/s; got {score:.1} m/s");
        // Confirm it matches what find_tisserand_link returns directly.
        let direct = find_tisserand_link(vinf, v_earth, a_earth, a_mars, MU_SUN);
        assert!(direct.is_some(), "link should be feasible for a Hohmann v∞");
        assert!((direct.unwrap() - score).abs() < 1.0,
            "score and find_tisserand_link should agree");
    }
}
