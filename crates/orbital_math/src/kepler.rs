//! Keplerian orbit mechanics primitives.
//!
//! Generic routines used by any two-body propagation: Kepler equation solver,
//! orbital-frame rotation, and universal-variable state propagation.
//! These carry no body-specific constants.

use nalgebra::Vector3;

use crate::lambert::{stumpff_c, stumpff_s};

/// Heliocentric (or any two-body) orbital period from Kepler's third law:
/// `T = 2π·√(a³/μ)`.
///
/// `sma_m` — semi-major axis [m]; `mu_m3s2` — central body's gravitational
/// parameter [m³/s²]. Undefined for a non-elliptic orbit (`sma_m <= 0`);
/// callers should only call this for a real bound orbit.
pub fn orbital_period_s(sma_m: f64, mu_m3s2: f64) -> f64 {
    2.0 * std::f64::consts::PI * (sma_m.powi(3) / mu_m3s2).sqrt()
}

/// Semi-major axis [m] from a Cartesian state, via the vis-viva/specific-
/// orbital-energy relation `a = 1 / (2/r − v²/μ)` (generic, mu-parameterized
/// — carries no body-specific constants, unlike `orbital_models::orbital::
/// OrbitalElements`, which is hardcoded to `MU_EARTH`). Negative for a
/// hyperbolic state; callers wanting a period should check `> 0` first (see
/// [`orbital_period_s`]'s own contract).
pub fn semi_major_axis_m(r_m: &Vector3<f64>, v_mps: &Vector3<f64>, mu_m3s2: f64) -> f64 {
    1.0 / (2.0 / r_m.norm() - v_mps.norm_squared() / mu_m3s2)
}

/// Orbital eccentricity from a Cartesian state, via the eccentricity-vector
/// relation `e_vec = (v × h)/μ − r̂` (generic, mu-parameterized — same
/// "no body-specific constants" convention as [`semi_major_axis_m`], unlike
/// `orbital_models::orbital::OrbitalElements::from_cartesian`, which
/// hardcodes `MU_EARTH` and is therefore unusable for an arbitrary central
/// body). Well-defined for any conic (elliptic/parabolic/hyperbolic) —
/// callers wanting "is this bound" should check the corresponding
/// [`semi_major_axis_m`] sign (`> 0`) rather than `eccentricity < 1` alone,
/// since floating-point error near `e == 1` is a real edge case for either
/// quantity taken alone.
pub fn eccentricity(r_m: &Vector3<f64>, v_mps: &Vector3<f64>, mu_m3s2: f64) -> f64 {
    let h = r_m.cross(v_mps);
    let e_vec = v_mps.cross(&h) / mu_m3s2 - r_m / r_m.norm();
    e_vec.norm()
}

/// Solve Kepler's equation  M = E − e·sin(E)  for eccentric anomaly E.
///
/// Newton–Raphson iteration from M as initial guess.
/// Converges in ≤ 6 iterations for e < 0.9 and in ≤ 50 for near-parabolic.
///
/// `m` — mean anomaly [rad] (arbitrary range; reduced mod 2π internally)
/// `e` — eccentricity (0 ≤ e < 1)
/// `tol` — convergence tolerance in radians (e.g. `1e-12`)
pub fn solve_kepler(m: f64, e: f64, tol: f64) -> f64 {
    let m = m.rem_euclid(2.0 * std::f64::consts::PI);
    let mut ea = m;
    for _ in 0..50 {
        let f  = ea - e * ea.sin() - m;
        let fp = 1.0 - e * ea.cos();
        let de = f / fp;
        ea -= de;
        if de.abs() < tol { break; }
    }
    ea
}

/// Rotate a perifocal-frame position to the ecliptic J2000 frame.
///
/// Applies the standard 3-1-3 Euler rotation sequence:
///   R = R_z(−RAAN) · R_x(−i) · R_z(−AOP)
///
/// `x`, `y`  — perifocal coordinates [m or nd]
/// `inc`     — inclination [rad]
/// `raan`    — right ascension of ascending node [rad]
/// `aop`     — argument of periapsis [rad]
pub fn perifocal_to_ecliptic(x: f64, y: f64, inc: f64, raan: f64, aop: f64) -> Vector3<f64> {
    let (si, ci) = inc.sin_cos();
    let (sr, cr) = raan.sin_cos();
    let (sw, cw) = aop.sin_cos();

    let r11 =  cr * cw - sr * sw * ci;
    let r12 = -cr * sw - sr * cw * ci;
    let r21 =  sr * cw + cr * sw * ci;
    let r22 = -sr * sw + cr * cw * ci;
    let r31 =  sw * si;
    let r32 =  cw * si;

    Vector3::new(
        r11 * x + r12 * y,
        r21 * x + r22 * y,
        r31 * x + r32 * y,
    )
}

/// Convert true anomaly to mean anomaly for an elliptic orbit.
///
/// Useful when constructing initial conditions from orbital elements at an
/// arbitrary point in the orbit (not periapsis).
pub fn true_to_mean_anomaly(nu: f64, e: f64) -> f64 {
    let ea = 2.0 * f64::atan2(
        ((1.0 - e) / (1.0 + e)).sqrt() * (nu / 2.0).sin(),
        (nu / 2.0).cos(),
    );
    ea - e * ea.sin()
}

/// Propagate a two-body (Keplerian) orbit from state `(r0, v0)` forward by
/// `dt_s` seconds under a point-mass gravity field with parameter `mu_m3s2`.
///
/// Uses the universal-variable (χ) formulation with Lagrange f/g coefficients.
/// Handles elliptic, hyperbolic, and parabolic orbits without branching on
/// orbit type — the Stumpff C/S functions absorb the conic-section cases.
///
/// Returns `None` only when the Newton–Raphson iteration fails to converge
/// within 50 steps (degenerate near-parabolic cases, or `dt_s = 0`).
///
/// # References
/// - Battin (1999), *An Introduction to the Mathematics and Methods of
///   Astrodynamics*, §4.5 — universal variable formulation.
/// - Vallado (2013), *Fundamentals of Astrodynamics and Applications*,
///   4th ed., Algorithm 8 (Kepler universal variable).
pub fn propagate_kepler(
    r0: Vector3<f64>,
    v0: Vector3<f64>,
    dt_s: f64,
    mu_m3s2: f64,
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    if dt_s == 0.0 { return Some((r0, v0)); }

    let r0_mag = r0.norm();
    let v0_sq  = v0.norm_squared();
    let mu_rt  = mu_m3s2.sqrt();

    // Reciprocal semi-major axis: α = 2/r₀ − v₀²/μ
    // α > 0 → elliptic, α < 0 → hyperbolic, α ≈ 0 → parabolic.
    let alpha = 2.0 / r0_mag - v0_sq / mu_m3s2;

    // σ₀ = r⃗₀·v⃗₀ / √μ  (Battin eq. 4.5-4)
    let sigma0 = r0.dot(&v0) / mu_rt;

    // Initial guess for the universal variable χ (Battin §4.5; Vallado Alg. 8).
    let chi0 = if alpha > 1e-6 {
        // Elliptic: χ ≈ √μ · Δt · α  (moves roughly one radian per orbital period)
        mu_rt * dt_s * alpha
    } else if alpha < -1e-6 {
        // Hyperbolic: χ₀ from the hyperbolic Kepler equation approximation
        // (Battin eq. 4.5-11; Goodyear 1965)
        let a = 1.0 / alpha; // negative for hyperbolic
        let t1 = 2.0 * mu_m3s2 * dt_s
            / (r0.dot(&v0) + dt_s.signum() * (-mu_m3s2 * a).sqrt() * (1.0 - r0_mag * alpha));
        dt_s.signum() * (-a).sqrt() * t1.abs().ln()
    } else {
        // Parabolic (α ≈ 0): quadratic approximation (Battin eq. 4.5-12)
        let h_sq = r0.cross(&v0).norm_squared();
        let p    = h_sq / mu_m3s2;
        let s    = 0.5 * (std::f64::consts::FRAC_PI_2 - (3.0 * mu_rt * dt_s / p.powf(1.5)).atan());
        let w    = s.tan().cbrt();
        p.sqrt() * 2.0 / w
    };

    // Newton–Raphson iteration on χ to satisfy the time equation:
    //   √μ · Δt = χ³·S(ψ) + σ₀·χ²·C(ψ) + r₀·χ
    // where ψ = α·χ².  Denominator equals the current r (Battin §4.5).
    let mut chi = chi0;
    let mut converged = false;
    for _ in 0..50 {
        let psi = alpha * chi * chi;
        let c2  = stumpff_c(psi);
        let c3  = stumpff_s(psi);

        // Current radius estimate (Battin eq. 4.4-16)
        let r = chi * chi * c2 + sigma0 * chi * (1.0 - psi * c3) + r0_mag * (1.0 - psi * c2);
        if r.abs() < 1.0 { return None; } // degenerate (should not happen for realistic inputs)

        // Time equation residual and its derivative (= r / √μ).
        // Standard universal-variable time equation (Vallado Algorithm 8):
        //   √μ·Δt = σ₀·χ²·C(ψ) + (1 − r₀·α)·χ³·S(ψ) + r₀·χ
        // The (1 − r₀·α) coefficient is 0 for circular orbits (α = 1/r₀)
        // and non-zero for all eccentric/hyperbolic orbits — without it, the
        // Newton step is correct only for circular trajectories.
        let dt_computed = (sigma0 * chi * chi * c2
            + (1.0 - r0_mag * alpha) * chi * chi * chi * c3
            + r0_mag * chi) / mu_rt;
        let dchi = (dt_s - dt_computed) / (r / mu_rt);

        chi += dchi;
        if dchi.abs() < 1e-12 * chi.abs().max(1.0) { converged = true; break; }
    }
    // Newton can stall on high-eccentricity orbits (e ≳ 0.95 — found
    // on a real Cassini-2 Lambert sub-arc, where a non-converged
    // sample previously returned a garbage position at 19,312 AU that went
    // straight into a plotted MGA arc; 45/800 dt samples stalled on that
    // orbit). Fallback: the universal time equation is strictly monotonic
    // in χ (dt/dχ = r/√μ > 0), so bisection on a bracket is guaranteed to
    // converge where Newton oscillates.
    if !converged {
        let t_of = |chi: f64| -> f64 {
            let psi = alpha * chi * chi;
            let c2 = stumpff_c(psi);
            let c3 = stumpff_s(psi);
            (sigma0 * chi * chi * c2
                + (1.0 - r0_mag * alpha) * chi * chi * chi * c3
                + r0_mag * chi) / mu_rt
        };
        let dir = dt_s.signum();
        let mut hi = chi0.abs().max(1.0) * dir;
        let mut expansions = 0;
        while t_of(hi).is_finite() && (t_of(hi) - dt_s) * dir < 0.0 {
            hi *= 2.0;
            expansions += 1;
            if expansions > 200 { return None; }
        }
        if !t_of(hi).is_finite() { return None; } // cosh overflow (extreme hyperbolic)
        let mut lo = 0.0_f64;
        for _ in 0..128 {
            let mid = 0.5 * (lo + hi);
            if (t_of(mid) - dt_s) * dir < 0.0 { lo = mid; } else { hi = mid; }
        }
        chi = 0.5 * (lo + hi);
    }

    // Lagrange f, g coefficients (Battin eq. 4.4-18 / Vallado eq. 2-38)
    let psi = alpha * chi * chi;
    let c2  = stumpff_c(psi);
    let c3  = stumpff_s(psi);

    let f     = 1.0 - chi * chi / r0_mag * c2;
    let g     = dt_s - chi * chi * chi / mu_rt * c3;
    let r_vec = f * r0 + g * v0;
    let r_mag = r_vec.norm();
    if r_mag < 1.0 { return None; }

    let f_dot = mu_rt / (r_mag * r0_mag) * chi * (psi * c3 - 1.0);
    let g_dot = 1.0 - chi * chi / r_mag * c2;
    let v_vec = f_dot * r0 + g_dot * v0;

    Some((r_vec, v_vec))
}

#[cfg(test)]
mod propagate_tests {
    use super::*;
    // Sun's gravitational parameter [m³/s²] — JPL DE430
    const MU_SUN: f64 = 1.327_124_400_18e20;

    /// Earth's orbital period from 1 AU / MU_SUN must match the real
    /// sidereal year (365.256 days) to within 0.1%.
    #[test]
    fn orbital_period_matches_earth_sidereal_year() {
        let a = 1.496e11_f64; // 1 AU [m]
        let period_days = orbital_period_s(a, MU_SUN) / 86_400.0;
        let real_sidereal_year_days = 365.256;
        let rel_err = (period_days - real_sidereal_year_days).abs() / real_sidereal_year_days;
        assert!(rel_err < 1e-3, "expected ~365.256 d, got {period_days:.3} d");
    }

    /// A circular orbit's semi-major axis must recover the orbit radius
    /// exactly (vis-viva: `v_circ = sqrt(mu/r)`, so `2/r - v^2/mu = 1/r`).
    #[test]
    fn semi_major_axis_of_a_circular_orbit_equals_its_radius() {
        let mu = MU_SUN;
        let r = 1.0e9_f64;
        let v_circ = (mu / r).sqrt();
        let a = semi_major_axis_m(&Vector3::new(r, 0.0, 0.0), &Vector3::new(0.0, v_circ, 0.0), mu);
        assert!((a - r).abs() / r < 1e-9, "expected a == r for a circular orbit, got a={a:.6e}");
    }

    /// A circular orbit (velocity purely tangential) must have exactly
    /// zero eccentricity.
    #[test]
    fn eccentricity_of_a_circular_orbit_is_zero() {
        let mu = MU_SUN;
        let r = 1.0e9_f64;
        let v_circ = (mu / r).sqrt();
        let e = eccentricity(&Vector3::new(r, 0.0, 0.0), &Vector3::new(0.0, v_circ, 0.0), mu);
        assert!(e < 1e-9, "expected e ~= 0 for a circular orbit, got e={e:.6e}");
    }

    /// A known elliptical orbit (Mars-like, e=0.093) constructed at
    /// periapsis must recover its own configured eccentricity.
    #[test]
    fn eccentricity_of_a_known_ellipse_matches_its_own_periapsis_state() {
        let mu = MU_SUN;
        let a = 1.524 * 1.496e11_f64;
        let e_expected = 0.093;
        let rp = a * (1.0 - e_expected);
        let vp = (mu * (2.0 / rp - 1.0 / a)).sqrt();
        let e = eccentricity(&Vector3::new(rp, 0.0, 0.0), &Vector3::new(0.0, vp, 0.0), mu);
        assert!((e - e_expected).abs() < 1e-6, "expected e={e_expected}, got e={e:.6e}");
    }

    /// The physical guarantee the new arrival-capture burn construction
    /// relies on (`optimize.rs`): scaling a velocity's
    /// MAGNITUDE down to the local circular speed at the current radius,
    /// while preserving its real (possibly non-tangential) direction,
    /// always yields a BOUND orbit (e < 1) — because specific energy
    /// `v^2/2 - mu/r` depends only on `|v|` and `r`, never on direction.
    /// This is what makes "apply the real burn along the real arrival
    /// velocity direction, not an idealized tangential one" safe to do
    /// unconditionally, for ANY real crossing geometry.
    #[test]
    fn scaling_speed_to_local_circular_speed_is_bound_regardless_of_direction() {
        let mu = MU_SUN;
        let r = 1.0e9_f64;
        let v_circ = (mu / r).sqrt();
        // A deliberately non-tangential real arrival direction -- 40 deg off
        // the local tangential, i.e. a real radial velocity component too
        // (not the periapsis-only assumption the old idealized reseed made).
        let real_dir = Vector3::new(40.0_f64.to_radians().sin(), 40.0_f64.to_radians().cos(), 0.0);
        let r_vec = Vector3::new(r, 0.0, 0.0);
        let v_post = real_dir * v_circ;
        let a = semi_major_axis_m(&r_vec, &v_post, mu);
        let e = eccentricity(&r_vec, &v_post, mu);
        assert!(a > 0.0, "expected a bound (a > 0) orbit, got a={a:.6e}");
        assert!(e < 1.0, "expected e < 1, got e={e:.6}");
        // A genuinely non-tangential arrival must produce a genuinely
        // elliptical (not circular) result -- confirms the real direction
        // is actually being honored, not silently collapsing to e=0.
        assert!(e > 0.1, "expected a real non-circular ellipse from this off-tangential direction, got e={e:.6}");
    }

    /// Chained with `orbital_period_s`, a circular orbit's semi-major axis
    /// must reproduce the exact period implied by its own speed
    /// (`T = 2*pi*r / v_circ`), cross-checking both functions together the
    /// way `cruise.rs`'s TCM horizon-bounding fix (Phase 13p) actually
    /// chains them.
    #[test]
    fn semi_major_axis_and_orbital_period_chain_to_the_circular_period() {
        let mu = MU_SUN;
        let r = 1.0e9_f64;
        let v_circ = (mu / r).sqrt();
        let a = semi_major_axis_m(&Vector3::new(r, 0.0, 0.0), &Vector3::new(0.0, v_circ, 0.0), mu);
        let period_s = orbital_period_s(a, mu);
        let expected_period_s = 2.0 * std::f64::consts::PI * r / v_circ;
        let rel_err = (period_s - expected_period_s).abs() / expected_period_s;
        assert!(rel_err < 1e-9, "expected {expected_period_s:.6e} s, got {period_s:.6e} s");
    }

    /// A hyperbolic (escape) state must give a negative semi-major axis --
    /// callers (e.g. `cruise::solve_tcm_correction`) rely on this sign to
    /// detect "no periodic bound applies" rather than calling
    /// `orbital_period_s` on a nonsensical negative-cubed value.
    #[test]
    fn semi_major_axis_of_a_hyperbolic_state_is_negative() {
        let mu = MU_SUN;
        let r = 1.0e9_f64;
        let v_circ = (mu / r).sqrt();
        // Comfortably above escape speed (v_esc = v_circ * sqrt(2)).
        let v_hyperbolic = v_circ * 2.0;
        let a = semi_major_axis_m(&Vector3::new(r, 0.0, 0.0), &Vector3::new(0.0, v_hyperbolic, 0.0), mu);
        assert!(a < 0.0, "expected a < 0 for a hyperbolic state, got a={a:.6e}");
    }

    /// Propagate Earth one full orbital period — should return to within 1 m
    /// of the starting position and 1 mm/s of the starting velocity.
    #[test]
    fn earth_one_full_period() {
        // Earth circular orbit at 1 AU (approximately).
        // Period T = 2π·√(a³/μ)
        let mu = MU_SUN;
        let a  = 1.496e11_f64; // 1 AU [m]
        let vc = (mu / a).sqrt();
        let r0 = Vector3::new(a, 0.0, 0.0);
        let v0 = Vector3::new(0.0, vc, 0.0);
        let period_s = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();

        let (r1, v1) = propagate_kepler(r0, v0, period_s, mu).expect("propagation failed");
        assert!((r1 - r0).norm() < 1.0,
            "position drift after one period: {:.3e} m", (r1 - r0).norm());
        assert!((v1 - v0).norm() < 1e-3,
            "velocity drift after one period: {:.3e} m/s", (v1 - v0).norm());
    }

    /// Forward then backward propagation must recover the initial state.
    #[test]
    fn forward_backward_inverse() {
        let mu = MU_SUN;
        let a  = 1.524 * 1.496e11_f64; // Mars-like orbit
        let e  = 0.093;
        let rp = a * (1.0 - e);
        let vp = (mu * (2.0 / rp - 1.0 / a)).sqrt();
        let r0 = Vector3::new(rp, 0.0, 0.0);
        let v0 = Vector3::new(0.0, vp, 0.0);
        let dt = 90.0 * 86_400.0; // 90 days

        let (r1, v1) = propagate_kepler(r0, v0, dt, mu).unwrap();
        let (r2, v2) = propagate_kepler(r1, v1, -dt, mu).unwrap();
        assert!((r2 - r0).norm() < 1.0,    "position roundtrip error: {:.3e} m",   (r2 - r0).norm());
        assert!((v2 - v0).norm() < 1e-3,   "velocity roundtrip error: {:.3e} m/s", (v2 - v0).norm());
    }

    /// Regression (found): on a high-eccentricity heliocentric
    /// ellipse (e ≈ 0.95 — the post-DSM Lambert sub-arc of a real Cassini-2
    /// benchmark solution, rp = 0.24 AU, ra = 9.98 AU), certain Δt values
    /// failed to converge within 50 Newton iterations and the function
    /// returned the GARBAGE position anyway (one sample at 19,312 AU went
    /// straight into a plotted MGA arc). The docstring always promised
    /// `None` on non-convergence; now the code honors it. Every position
    /// this function returns must be physically on the orbit (≤ apoapsis).
    #[test]
    fn high_eccentricity_never_returns_garbage() {
        let mu = MU_SUN;
        let au = 1.496e11_f64;
        let rp = 0.2435 * au;
        let ra = 9.977 * au;
        let a = 0.5 * (rp + ra);
        let vp = (mu * (2.0 / rp - 1.0 / a)).sqrt();
        let r0 = Vector3::new(rp, 0.0, 0.0);
        let v0 = Vector3::new(0.0, vp, 0.0);
        let period_s = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();

        let n = 800;
        let mut converged = 0usize;
        for i in 1..=n {
            let dt = period_s * i as f64 / n as f64;
            if let Some((r, _)) = propagate_kepler(r0, v0, dt, mu) {
                converged += 1;
                let r_mag = r.norm();
                assert!(
                    r_mag <= ra * 1.01,
                    "returned position {:.1} AU beyond apoapsis {:.2} AU at dt = {:.2} d — \
                     non-convergence garbage leaked through",
                    r_mag / au, ra / au, dt / 86_400.0
                );
            }
        }
        // The None path must stay the exception, not the rule.
        assert!(
            converged >= (n * 95) / 100,
            "only {converged}/{n} samples converged — propagator is failing too often"
        );
    }

    /// Specific orbital energy must be conserved to one part in 1e12.
    #[test]
    fn energy_conservation() {
        let mu = MU_SUN;
        let a  = 2.0 * 1.496e11_f64;
        let e  = 0.3;
        let rp = a * (1.0 - e);
        let vp = (mu * (2.0 / rp - 1.0 / a)).sqrt();
        let r0 = Vector3::new(rp, 0.0, 0.0);
        let v0 = Vector3::new(0.0, vp, 0.0);
        let eps0 = 0.5 * v0.norm_squared() - mu / r0.norm();

        let dt = 200.0 * 86_400.0;
        let (r1, v1) = propagate_kepler(r0, v0, dt, mu).unwrap();
        let eps1 = 0.5 * v1.norm_squared() - mu / r1.norm();
        let rel_err = ((eps1 - eps0) / eps0).abs();
        assert!(rel_err < 1e-12, "energy relative error: {:.3e}", rel_err);
    }
}
