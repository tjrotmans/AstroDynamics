//! Departure/injection burn construction at a circular parking orbit.
//!
//! Two constructions, for two different use cases:
//!
//! - [`hyperbolic_departure_state`]: given a *required* hyperbolic excess
//!   velocity vector (e.g. from a Lambert solve), constructs the injection
//!   state that achieves exactly that v-infinity. Useful when something
//!   downstream needs to hit a specific target, but see its own doc comment
//!   for a real caveat: feeding the result into a Lambert-sensitive
//!   targeting scheme can amplify the (small but nonzero) gap between the
//!   real SOI-exit state and the idealized asymptote into a large error.
//! - [`circular_orbit_burn_state`]: given a burn location (true anomaly), a
//!   burn magnitude, and an out-of-plane angle — three independent,
//!   freely-searchable numbers — constructs the resulting injection state
//!   directly, with no target v-infinity involved at all. This is what the
//!   Phase 9 GA/PSO optimizer actually uses: it doesn't presuppose a
//!   solution (unlike a Lambert-anchored search, which always converges
//!   toward the analytic Hohmann-like minimum and isn't a meaningful test of
//!   global search), so reaching the target at all is a genuine, nontrivial
//!   outcome of the search, not a foregone conclusion baked into the
//!   construction.
//!
//! Both start the spacecraft from a real, finite-radius parking orbit
//! instead of the departure body's literal center (degenerate, `r -> 0`, a
//! singularity in the point-mass gravity formula).
//!
//! Standard patched-conic departure geometry (Vallado, *Fundamentals of
//! Astrodynamics and Applications*, 4th ed., Ch. 12; Curtis, *Orbital
//! Mechanics for Engineering Students*, 3rd ed., Ch. 8.8).

use nalgebra::Vector3;

/// Result of constructing a hyperbolic departure injection state.
pub struct HyperbolicDeparture {
    /// Position at injection (periapsis of the departure hyperbola),
    /// relative to the departure body's center [m]. `|r0_m| == r_park_m`.
    pub r0_m: Vector3<f64>,
    /// Velocity at injection, relative to the departure body [m/s].
    /// Purely tangential (perpendicular to `r0_m`), magnitude = hyperbolic
    /// periapsis speed.
    pub v0_mps: Vector3<f64>,
    /// Escape burn magnitude: hyperbolic periapsis speed minus the parking
    /// orbit's circular speed at the same radius. The real onboard ΔV cost
    /// of leaving the parking orbit (before any launch-vehicle/free-departure
    /// accounting the caller applies).
    pub dv_escape_ms: f64,
}

/// Construct the departure hyperbola's injection state at periapsis, given
/// the body's `mu_m3s2`, the parking orbit radius `r_park_m`, and the
/// required hyperbolic excess velocity vector `v_inf_vec_ms` (in whatever
/// inertial frame the caller works in — the returned `r0_m`/`v0_mps` are in
/// that same frame, relative to the departure body).
///
/// Returns `None` if `v_inf_vec_ms` is (numerically) zero — a hyperbola
/// requires nonzero excess energy; a zero-C3 "departure" has no well-defined
/// asymptote direction to target.
pub fn hyperbolic_departure_state(
    mu_m3s2: f64,
    r_park_m: f64,
    v_inf_vec_ms: Vector3<f64>,
) -> Option<HyperbolicDeparture> {
    let v_inf = v_inf_vec_ms.norm();
    if v_inf < 1e-6 {
        return None;
    }
    let v_inf_hat = v_inf_vec_ms / v_inf;

    // Hyperbola shape from energy (a < 0) and periapsis radius.
    let a_m = -mu_m3s2 / (v_inf * v_inf);
    let e = 1.0 - r_park_m / a_m; // = 1 + r_park*v_inf^2/mu, since a < 0
    let true_anomaly_inf = (-1.0 / e).acos(); // nu_infinity, in (90 deg, 180 deg) for e > 1

    let v_p = (v_inf * v_inf + 2.0 * mu_m3s2 / r_park_m).sqrt(); // vis-viva at periapsis
    let v_circ = (mu_m3s2 / r_park_m).sqrt();

    // Orbital plane normal: any unit vector perpendicular to v_inf_hat. Pick
    // via an arbitrary reference axis (least-aligned of the global X/Z axes,
    // to avoid a degenerate near-zero cross product when v_inf_hat happens
    // to be close to one of them).
    let reference = if v_inf_hat.z.abs() < 0.9 { Vector3::z() } else { Vector3::x() };
    let h_hat = v_inf_hat.cross(&reference).normalize();

    // Periapsis direction: rotate the asymptote direction *backward* by
    // true_anomaly_inf around h_hat (periapsis occurs that many degrees of
    // true anomaly before the outgoing asymptote, in the direction of
    // motion). Rodrigues' rotation formula, simplified since h_hat is
    // perpendicular to v_inf_hat (so the "axis . v" term vanishes):
    //   rotate(v, axis, theta) = v*cos(theta) + (axis x v)*sin(theta)
    let theta = -true_anomaly_inf;
    let r_p_hat = v_inf_hat * theta.cos() + h_hat.cross(&v_inf_hat) * theta.sin();

    // Tangential velocity direction at periapsis: perpendicular to r_p_hat,
    // in the orbital plane, consistent with h = r x v (so velocity, not
    // position, advances in the direction of motion toward the asymptote).
    let v_p_hat = h_hat.cross(&r_p_hat);

    Some(HyperbolicDeparture {
        r0_m: r_p_hat * r_park_m,
        v0_mps: v_p_hat * v_p,
        dv_escape_ms: v_p - v_circ,
    })
}

/// Result of [`circular_orbit_burn_state`].
pub struct CircularOrbitBurn {
    /// Position at the burn, relative to the departure body's center [m].
    /// `|r0_m| == r_park_m`.
    pub r0_m: Vector3<f64>,
    /// Velocity immediately after the burn, relative to the departure body
    /// [m/s].
    pub v0_mps: Vector3<f64>,
}

/// Construct the post-burn state at a circular parking orbit, given the
/// burn location and the burn itself — three independent numbers, with no
/// target v-infinity involved (contrast [`hyperbolic_departure_state`]).
///
/// `plane_normal_hat` and `reference_dir_hat` define the parking orbit's
/// plane and a zero-point within it (`reference_dir_hat` must be
/// perpendicular to `plane_normal_hat`) — the caller picks these (e.g. the
/// departure body's own instantaneous heliocentric orbital plane, as a
/// stand-in for "the ecliptic," with the body's own position direction as
/// the zero-point, which is perpendicular to that plane's normal by
/// construction).
///
/// `theta_rad` is the true anomaly (orbital phase) of the burn location,
/// measured from `reference_dir_hat` around `plane_normal_hat`.
///
/// The burn vector is confined to the local tangential/normal plane (i.e.
/// never purely radial — wasteful for an impulsive burn, so deliberately
/// not modeled), parameterized by `dv_mps` (magnitude) and
/// `out_of_plane_rad` (direction within that 2D plane, full range — `0`
/// is purely tangential/prograde, `pi` purely tangential/retrograde, `+-
/// pi/2` purely out-of-plane).
pub fn circular_orbit_burn_state(
    mu_m3s2: f64,
    r_park_m: f64,
    plane_normal_hat: Vector3<f64>,
    reference_dir_hat: Vector3<f64>,
    theta_rad: f64,
    dv_mps: f64,
    out_of_plane_rad: f64,
) -> CircularOrbitBurn {
    // Rotate the reference direction by theta around the plane normal --
    // same simplified Rodrigues formula as `hyperbolic_departure_state`
    // (valid since reference_dir_hat is perpendicular to plane_normal_hat).
    let p_hat = reference_dir_hat * theta_rad.cos() + plane_normal_hat.cross(&reference_dir_hat) * theta_rad.sin();
    // Tangential (prograde) direction at the burn point -- same h x r_hat
    // convention validated in hyperbolic_departure_state's tests.
    let t_hat = plane_normal_hat.cross(&p_hat);

    let v_circ = (mu_m3s2 / r_park_m).sqrt();
    let v0_mps = t_hat * (v_circ + dv_mps * out_of_plane_rad.cos()) + plane_normal_hat * (dv_mps * out_of_plane_rad.sin());

    CircularOrbitBurn { r0_m: p_hat * r_park_m, v0_mps }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagator::propagate;

    /// Earth-like body, departure C3 in a realistic range (Earth-Mars-class,
    /// ~9 km^2/s^2). Confirms the basic invariants directly from the
    /// construction: |r0| matches the requested parking radius exactly,
    /// r0 is exactly perpendicular to v0 (true at periapsis for any conic --
    /// checked via the *normalized* dot product, since the raw r0.v0 mixes
    /// ~1e7 m against ~1e4 m/s, where floating-point noise on the dot
    /// product itself scales with the operands' magnitude, not an absolute
    /// 1e-6), specific energy matches the requested v_inf exactly (the
    /// vis-viva invariant, independent of direction), and the escape ΔV
    /// matches the existing analytic formula already used elsewhere in this
    /// codebase (`design.rs::departure_escape_dv_ms`) — confirming this
    /// isn't a second, divergent implementation of the same physics.
    #[test]
    fn injection_state_matches_known_invariants() {
        const MU_EARTH: f64 = 3.986_004_418e14;
        let r_park = 6_371_000.0 * 1.5; // matches design.rs's "airless" 1.5x-radius heuristic shape
        let v_inf = 3_000.0; // m/s, ~Earth-Mars class
        let v_inf_vec = Vector3::new(v_inf * 0.6, v_inf * 0.8, 0.0); // arbitrary direction, |.|=v_inf

        let dep = hyperbolic_departure_state(MU_EARTH, r_park, v_inf_vec).unwrap();

        assert!((dep.r0_m.norm() - r_park).abs() < 1e-3, "r0 should have magnitude exactly r_park");
        let cos_angle = dep.r0_m.normalize().dot(&dep.v0_mps.normalize());
        assert!(cos_angle.abs() < 1e-9, "r0 and v0 must be perpendicular at periapsis, got cos(angle)={cos_angle:e}");

        let energy = 0.5 * dep.v0_mps.norm_squared() - MU_EARTH / dep.r0_m.norm();
        let energy_expected = 0.5 * v_inf * v_inf;
        assert!(
            (energy - energy_expected).abs() < 1e-3,
            "specific energy should match 0.5*v_inf^2 exactly (vis-viva): got {energy}, expected {energy_expected}"
        );

        let v_circ = (MU_EARTH / r_park).sqrt();
        let v_p_expected = (v_inf * v_inf + 2.0 * MU_EARTH / r_park).sqrt();
        let dv_expected = v_p_expected - v_circ;
        assert!(
            (dep.dv_escape_ms - dv_expected).abs() < 1e-6,
            "escape dV should match the existing analytic v_hyp - v_circ formula: got {}, expected {}",
            dep.dv_escape_ms, dv_expected
        );
    }

    /// The real test of the construction: propagate the injection state
    /// forward under pure point-mass gravity (no perturbers) and confirm the
    /// velocity *direction* converges to the originally requested v_inf
    /// direction -- not just that the speed matches at periapsis, which an
    /// inverted or mirrored geometry could also satisfy. Direction converges
    /// much faster than speed for a hyperbola (confirmed empirically: ~30
    /// days gets the angle within 0.0004 deg here, while speed is still ~15
    /// m/s off v_inf at that point -- an inherent property of the 1/r
    /// approach to the asymptote, not a bug), so this checks *direction*
    /// via propagation and *speed* via the exact energy invariant in the
    /// test above instead of waiting for speed to also converge.
    #[test]
    fn propagated_state_converges_toward_requested_v_infinity_direction() {
        const MU_EARTH: f64 = 3.986_004_418e14;
        let r_park = 6_871_000.0;
        let v_inf = 3_200.0;
        let v_inf_vec = Vector3::new(v_inf * 0.36, -v_inf * 0.48, v_inf * 0.8); // arbitrary 3D direction
        let v_inf_hat = v_inf_vec.normalize();

        let dep = hyperbolic_departure_state(MU_EARTH, r_park, v_inf_vec).unwrap();

        // No perturbers (`&[]`); `reference_mu_m3s2 = MU_EARTH` makes this
        // pure two-body point-mass. 30 days is comfortably past where the
        // direction has settled (confirmed empirically: 0.0004 deg error)
        // without needing speed to have also converged.
        let points = propagate(dep.r0_m, dep.v0_mps, 0.0, 30.0 * 86_400.0, MU_EARTH, &[], 3600.0, 1e-12, 1e-3);
        let end = points.last().expect("expected a propagated trajectory");

        // Speed must still be monotonically decreasing toward v_inf from
        // above (never below it -- that would mean a sign/direction error
        // in the construction), not yet required to have fully converged.
        assert!(
            end.v_mps.norm() > v_inf && end.v_mps.norm() < v_inf + 50.0,
            "speed should be decreasing toward v_inf ({v_inf} m/s) from above, got {}",
            end.v_mps.norm()
        );

        let angle_err_deg = (end.v_mps.normalize().dot(&v_inf_hat)).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(
            angle_err_deg < 0.5,
            "asymptotic velocity direction should converge to the requested v_inf direction, got {angle_err_deg:.4} deg off"
        );
    }

    /// At `out_of_plane_rad = 0` (pure tangential/prograde), the burn must
    /// exactly reproduce the standard analytic departure result: speed
    /// becomes `v_circ + dv`, still tangential, still at `r_park`. This is
    /// the textbook Hohmann-like case the GA should be *able* to find (not
    /// forced into), so it must be reachable exactly at the right
    /// parameter values.
    #[test]
    fn pure_tangential_burn_matches_analytic_circular_orbit() {
        const MU_EARTH: f64 = 3.986_004_418e14;
        let r_park = 6_871_000.0;
        let dv = 3_200.0;
        let plane_normal = Vector3::new(0.0, 0.0, 1.0);
        let reference_dir = Vector3::new(1.0, 0.0, 0.0);
        let theta = 1.234_f64; // arbitrary burn location

        let burn = circular_orbit_burn_state(MU_EARTH, r_park, plane_normal, reference_dir, theta, dv, 0.0);

        assert!((burn.r0_m.norm() - r_park).abs() < 1e-6);
        let v_circ = (MU_EARTH / r_park).sqrt();
        assert!(
            (burn.v0_mps.norm() - (v_circ + dv)).abs() < 1e-6,
            "pure tangential burn should give exactly v_circ + dv, got {} vs expected {}",
            burn.v0_mps.norm(), v_circ + dv,
        );
        assert!(
            burn.r0_m.normalize().dot(&burn.v0_mps.normalize()).abs() < 1e-9,
            "velocity should stay exactly tangential (perpendicular to position) at out_of_plane_rad = 0"
        );
    }

    /// At `out_of_plane_rad = pi/2` (purely out-of-plane), the resulting
    /// velocity should have zero net change in the tangential direction's
    /// in-plane speed contribution beyond v_circ, and the full `dv`
    /// magnitude directed along the plane normal -- confirms the
    /// (tangential, normal) decomposition is wired correctly, not just
    /// validated at the trivial phi=0 case.
    #[test]
    fn purely_out_of_plane_burn_adds_normal_component_only() {
        const MU_EARTH: f64 = 3.986_004_418e14;
        let r_park = 7_000_000.0;
        let dv = 500.0;
        let plane_normal = Vector3::new(0.0, 0.0, 1.0);
        let reference_dir = Vector3::new(1.0, 0.0, 0.0);

        let burn = circular_orbit_burn_state(MU_EARTH, r_park, plane_normal, reference_dir, 0.0, dv, std::f64::consts::FRAC_PI_2);

        let v_circ = (MU_EARTH / r_park).sqrt();
        // p_hat = reference_dir (theta=0); t_hat = plane_normal x p_hat = (0,1,0).
        let expected = Vector3::new(0.0, v_circ, dv);
        assert!(
            (burn.v0_mps - expected).norm() < 1e-6,
            "expected v0 = {expected:?}, got {:?}", burn.v0_mps
        );
    }
}
