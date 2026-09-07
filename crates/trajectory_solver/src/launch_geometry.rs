//! Closed-form launch geometry from a departure asymptote (Phase 14c).
//!
//! Preliminary interplanetary design never propagates the ascent. The
//! heliocentric solve comes first and hands back the departure asymptote —
//! the hyperbolic excess velocity vector `v∞`, quoted as launch energy
//! `C3 = v∞²` and pointing direction RLA/DLA (right ascension / declination
//! of the launch asymptote in the departure body's equatorial frame). For a
//! fixed `v∞` and periapsis radius `r_p`, the escape hyperbola is a ONE-
//! PARAMETER family (rotation about the asymptote): every member reaches the
//! same heliocentric leg to patched-conic accuracy, so the orbital plane is
//! free to be chosen AFTER the interplanetary solve to satisfy the launch
//! site — the plane must contain the asymptote (`i ≥ |DLA|`) and be reachable
//! from the site without a dogleg (`i ≥ |φ_site|`). Everything below is
//! algebra on that family; there is no ascent model, no iteration and no
//! ephemeris. Daily launch-window timing (which needs the body's prime-
//! meridian angle at an epoch) is deliberately out of scope here; RLA/DLA
//! and the parking-orbit node give it almost for free later.
//!
//! References: Sergeyevsky, Snyder & Cunniff (1983), *Interplanetary Mission
//! Design Handbook*, JPL 82-43 (the RLA/DLA characterisation); Vallado,
//! *Fundamentals of Astrodynamics and Applications*, 4th ed., §6.4 (launch
//! azimuth / inclination reachable from a site latitude) and Ch. 12
//! (hyperbolic departure); Curtis, *Orbital Mechanics for Engineering
//! Students*, §8.8. See `docs/MP/MANUAL.md` §13.6.

use nalgebra::Vector3;

/// Body-fixed-equatorial reference frame axes expressed in the caller's
/// inertial frame (ICRF/J2000): `z` = north pole, `x` = the node of the
/// body's equator on the ICRF equator (IAU convention, RA = α₀ + 90°), `y`
/// completes the right-handed set. For a body whose pole IS the ICRF pole
/// (Earth at J2000) the node is undefined and `x` falls back to the ICRF
/// x-axis (the vernal equinox), which is exactly the frame RLA/DLA are
/// conventionally quoted in for Earth departures.
#[derive(Clone, Copy, Debug)]
pub struct EquatorialFrame {
    pub x: Vector3<f64>,
    pub y: Vector3<f64>,
    pub z: Vector3<f64>,
}

impl EquatorialFrame {
    /// From the body's north-pole right ascension / declination [rad]
    /// (`body_models::TargetBody::pole_ra_deg`/`pole_dec_deg`, converted).
    pub fn from_pole(pole_ra_rad: f64, pole_dec_rad: f64) -> Self {
        let z = Vector3::new(
            pole_dec_rad.cos() * pole_ra_rad.cos(),
            pole_dec_rad.cos() * pole_ra_rad.sin(),
            pole_dec_rad.sin(),
        );
        Self::from_pole_vector(z)
    }

    /// From a unit pole vector in the inertial frame.
    pub fn from_pole_vector(pole_hat: Vector3<f64>) -> Self {
        let z = pole_hat.normalize();
        let node = Vector3::z().cross(&z);
        let x = if node.norm() < 1e-9 { Vector3::x() } else { node.normalize() };
        let y = z.cross(&x);
        Self { x, y, z }
    }
}

/// Result of [`launch_geometry`] — the departure asymptote characterised
/// (RLA/DLA), the parking/escape plane chosen for the site, the launch
/// azimuth, and the exact injection state on that plane.
#[derive(Clone, Copy, Debug)]
pub struct LaunchGeometry {
    /// Launch energy `C3 = v∞²` [km²/s²].
    pub c3_km2s2: f64,
    /// Right ascension of the launch asymptote in the body equatorial frame
    /// [rad], in `[0, 2π)`.
    pub rla_rad: f64,
    /// Declination of the launch asymptote [rad], in `[−π/2, π/2]`.
    pub dla_rad: f64,
    /// Inclination of the parking/escape plane to the body equator [rad].
    pub inclination_rad: f64,
    /// Minimum inclination that contains the asymptote AND is reachable from
    /// the site without a dogleg: `max(|DLA|, |φ_site|)` [rad].
    pub min_inclination_rad: f64,
    /// `inclination_rad ≥ min_inclination_rad` (within tolerance). When
    /// false the chosen inclination cannot host this asymptote from this
    /// site: the plane still contains the asymptote if `i ≥ |DLA|` but the
    /// ascent would need a dogleg (priced as infeasible, not as ΔV).
    pub feasible_no_dogleg: bool,
    /// Launch azimuth [rad], measured clockwise from north, for the
    /// north-east-going (ascending) pass: `sin β = cos i / cos φ`. The
    /// south-east-going alternative is `π − β`. Only meaningful when
    /// `feasible_no_dogleg`.
    pub launch_azimuth_rad: f64,
    /// Right ascension of the parking orbit's ascending node in the body
    /// equatorial frame [rad], in `[0, 2π)`.
    pub raan_rad: f64,
    /// Unit normal of the parking/escape plane in the inertial frame.
    pub plane_normal_hat: Vector3<f64>,
    /// In-plane angle [rad] coasted in the parking orbit from the ascent
    /// injection point (the site's latitude crossed on the ascending pass)
    /// to the escape-burn periapsis, in `[0, 2π)`. Schematic: a canonical
    /// ascent's own downrange angle is not modelled.
    pub coast_angle_rad: f64,
    /// Escape-burn point (periapsis of the departure hyperbola), body-
    /// relative inertial [m]; `|r| == r_p`.
    pub injection_r_m: Vector3<f64>,
    /// Velocity immediately after the escape burn, body-relative inertial
    /// [m/s]; tangential, `|v|² − 2μ/r_p == v∞²`.
    pub injection_v_mps: Vector3<f64>,
    /// Parking-orbit circular velocity at the injection point, before the
    /// burn [m/s] (same direction as `injection_v_mps`).
    pub parking_v_mps: Vector3<f64>,
    /// Escape-burn magnitude `√(v∞² + 2μ/r_p) − √(μ/r_p)` [m/s].
    pub dv_injection_ms: f64,
}

/// Solve the launch geometry for a departure asymptote.
///
/// - `mu_m3s2`, `r_p_m`: departure body and parking/injection radius.
/// - `v_inf_vec_ms`: required hyperbolic excess velocity, inertial frame.
/// - `frame`: the body's equatorial frame ([`EquatorialFrame::from_pole`]).
/// - `site_lat_rad`: launch-site geodetic latitude (sign irrelevant to the
///   feasibility rule, which uses `|φ|`).
/// - `inclination_rad`: `None` picks the minimum feasible inclination
///   `max(|DLA|, |φ|)` (the cheapest plane — no plane change, no dogleg);
///   `Some(i)` uses that inclination (clamped to `[|DLA|, π − |DLA|]` so
///   the plane can always contain the asymptote; `feasible_no_dogleg`
///   reports whether the site can reach it).
/// - `descending_node_branch`: the two planes of inclination `i` that
///   contain the asymptote are mirror images about the asymptote–pole
///   plane; `false` takes the branch whose node vector lies on the +y side
///   of the asymptote, `true` the other.
///
/// Returns `None` for a numerically zero `v∞` (no asymptote direction).
pub fn launch_geometry(
    mu_m3s2: f64,
    r_p_m: f64,
    v_inf_vec_ms: Vector3<f64>,
    frame: &EquatorialFrame,
    site_lat_rad: f64,
    inclination_rad: Option<f64>,
    descending_node_branch: bool,
) -> Option<LaunchGeometry> {
    let v_inf = v_inf_vec_ms.norm();
    if v_inf < 1e-6 {
        return None;
    }
    let s_hat = v_inf_vec_ms / v_inf;

    // Asymptote in the body equatorial frame.
    let sx = s_hat.dot(&frame.x);
    let sy = s_hat.dot(&frame.y);
    let sz = s_hat.dot(&frame.z).clamp(-1.0, 1.0);
    let dla = sz.asin();
    let rla = sy.atan2(sx).rem_euclid(std::f64::consts::TAU);

    // Inclination: the plane must contain the asymptote (i >= |DLA|) and be
    // reachable from the site (i >= |phi|).
    let min_incl = dla.abs().max(site_lat_rad.abs());
    let incl = match inclination_rad {
        None => min_incl,
        Some(i) => i.clamp(dla.abs(), std::f64::consts::PI - dla.abs()),
    };
    let feasible_no_dogleg = incl + 1e-12 >= min_incl;

    // Plane normal h with h . s = 0 and h . pole = cos i. Decompose h in the
    // basis (u, w) perpendicular to s: u = pole component perpendicular to
    // s (|u_raw| = cos DLA), w = s x u. Then h . pole = a cos DLA.
    let pole = frame.z;
    let u_raw = pole - s_hat * sz;
    let cos_dla = u_raw.norm();
    let (u_hat, w_hat) = if cos_dla < 1e-12 {
        // Asymptote along the pole: every plane through it is polar; any
        // perpendicular basis works.
        let u = s_hat.cross(&frame.x).normalize();
        (u, s_hat.cross(&u))
    } else {
        let u = u_raw / cos_dla;
        (u, s_hat.cross(&u))
    };
    let a = if cos_dla < 1e-12 { 0.0 } else { (incl.cos() / cos_dla).clamp(-1.0, 1.0) };
    let b = (1.0 - a * a).max(0.0).sqrt() * if descending_node_branch { -1.0 } else { 1.0 };
    let h_hat = (u_hat * a + w_hat * b).normalize();

    // Ascending node of the plane on the body equator.
    let node_raw = pole.cross(&h_hat);
    let node_hat = if node_raw.norm() < 1e-12 { frame.x } else { node_raw.normalize() };
    let raan = node_hat.dot(&frame.y).atan2(node_hat.dot(&frame.x)).rem_euclid(std::f64::consts::TAU);

    // Periapsis of the hyperbola on this plane: the asymptote rotated
    // backward by nu_inf about h (Rodrigues, h perpendicular to s).
    let e = 1.0 + r_p_m * v_inf * v_inf / mu_m3s2;
    let nu_inf = (-1.0 / e).acos();
    let theta = -nu_inf;
    let r_p_hat = s_hat * theta.cos() + h_hat.cross(&s_hat) * theta.sin();
    let v_p_hat = h_hat.cross(&r_p_hat);
    let v_p = (v_inf * v_inf + 2.0 * mu_m3s2 / r_p_m).sqrt();
    let v_circ = (mu_m3s2 / r_p_m).sqrt();

    // Launch azimuth (Vallado §6.4): sin(beta) = cos i / cos phi.
    let sin_beta = (incl.cos() / site_lat_rad.cos()).clamp(-1.0, 1.0);
    let launch_azimuth = sin_beta.asin();

    // Coast angle from the site's ascending-pass argument of latitude to the
    // periapsis argument of latitude, both measured from the ascending node.
    let sin_i = incl.sin();
    let u_site = if sin_i < 1e-12 { 0.0 } else { (site_lat_rad.sin() / sin_i).clamp(-1.0, 1.0).asin() };
    let u_peri = r_p_hat.dot(&h_hat.cross(&node_hat)).atan2(r_p_hat.dot(&node_hat));
    let coast_angle = (u_peri - u_site).rem_euclid(std::f64::consts::TAU);

    Some(LaunchGeometry {
        c3_km2s2: v_inf * v_inf / 1.0e6,
        rla_rad: rla,
        dla_rad: dla,
        inclination_rad: incl,
        min_inclination_rad: min_incl,
        feasible_no_dogleg,
        launch_azimuth_rad: launch_azimuth,
        raan_rad: raan,
        plane_normal_hat: h_hat,
        coast_angle_rad: coast_angle,
        injection_r_m: r_p_hat * r_p_m,
        injection_v_mps: v_p_hat * v_p,
        parking_v_mps: v_p_hat * v_circ,
        dv_injection_ms: v_p - v_circ,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagator::propagate;

    const MU_EARTH: f64 = 3.986_004_418e14;
    const KSC_LAT_RAD: f64 = 28.5_f64 * std::f64::consts::PI / 180.0;

    fn earth_frame() -> EquatorialFrame {
        EquatorialFrame::from_pole(0.0, std::f64::consts::FRAC_PI_2)
    }

    #[test]
    fn earth_pole_frame_falls_back_to_the_vernal_equinox_axis() {
        let f = earth_frame();
        assert!((f.x - Vector3::x()).norm() < 1e-12);
        assert!((f.y - Vector3::y()).norm() < 1e-12);
        assert!((f.z - Vector3::z()).norm() < 1e-12);
    }

    #[test]
    fn rla_dla_recover_the_asymptote_direction() {
        let f = earth_frame();
        let v = Vector3::new(0.0, 30_f64.to_radians().cos(), 30_f64.to_radians().sin()) * 3_000.0;
        let g = launch_geometry(MU_EARTH, 6_578_000.0, v, &f, KSC_LAT_RAD, None, false).unwrap();
        assert!((g.rla_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((g.dla_rad - 30_f64.to_radians()).abs() < 1e-12);
        assert!((g.c3_km2s2 - 9.0).abs() < 1e-12);
    }

    /// The identity that makes the injection state exact: |v|² − 2μ/r_p = v∞²,
    /// tangential at r_p, and the ΔV matches the analytic escape burn.
    #[test]
    fn injection_state_satisfies_the_energy_identity() {
        let f = earth_frame();
        let v_inf = 3_200.0;
        let v = Vector3::new(0.36, -0.48, 0.8) * v_inf;
        let r_p = 6_578_000.0;
        let g = launch_geometry(MU_EARTH, r_p, v, &f, KSC_LAT_RAD, None, false).unwrap();
        assert!((g.injection_r_m.norm() - r_p).abs() < 1e-3);
        let lhs = g.injection_v_mps.norm_squared() - 2.0 * MU_EARTH / r_p;
        assert!((lhs - v_inf * v_inf).abs() < 1e-3, "got {lhs}, expected {}", v_inf * v_inf);
        assert!(g.injection_r_m.normalize().dot(&g.injection_v_mps.normalize()).abs() < 1e-9);
        let dv_expected = (v_inf * v_inf + 2.0 * MU_EARTH / r_p).sqrt() - (MU_EARTH / r_p).sqrt();
        assert!((g.dv_injection_ms - dv_expected).abs() < 1e-6);
        assert!((g.parking_v_mps.norm() - (MU_EARTH / r_p).sqrt()).abs() < 1e-6);
        assert!((g.injection_v_mps - g.parking_v_mps).dot(&g.parking_v_mps) > 0.0, "burn is prograde");
    }

    /// The chosen plane contains the asymptote and has the requested
    /// inclination to the equator; the minimum-inclination default is
    /// max(|DLA|, |φ|).
    #[test]
    fn plane_contains_the_asymptote_at_the_requested_inclination() {
        let f = earth_frame();
        let v = Vector3::new(0.36, -0.48, 0.8) * 3_000.0; // DLA = asin(0.8) = 53.13°
        let g = launch_geometry(MU_EARTH, 6_578_000.0, v, &f, KSC_LAT_RAD, None, false).unwrap();
        let dla = 0.8_f64.asin();
        assert!((g.min_inclination_rad - dla).abs() < 1e-12, "DLA exceeds the site latitude here");
        assert!((g.inclination_rad - dla).abs() < 1e-12);
        assert!(g.feasible_no_dogleg);
        assert!(g.plane_normal_hat.dot(&v.normalize()).abs() < 1e-9);
        assert!((g.plane_normal_hat.dot(&f.z) - g.inclination_rad.cos()).abs() < 1e-9);

        // Explicit higher inclination, both branches.
        for branch in [false, true] {
            let g2 = launch_geometry(MU_EARTH, 6_578_000.0, v, &f, KSC_LAT_RAD, Some(70_f64.to_radians()), branch).unwrap();
            assert!(g2.plane_normal_hat.dot(&v.normalize()).abs() < 1e-9);
            assert!((g2.plane_normal_hat.dot(&f.z) - 70_f64.to_radians().cos()).abs() < 1e-9);
            assert!(g2.feasible_no_dogleg);
            // Injection still exact on this plane.
            assert!((g2.injection_v_mps.norm_squared() - 2.0 * MU_EARTH / 6_578_000.0 - 9.0e6).abs() < 1e-3);
        }
        // Low-DLA asymptote from KSC: the site latitude sets the minimum.
        let v_low = Vector3::new(1.0, 0.2, 0.05).normalize() * 3_000.0;
        let g3 = launch_geometry(MU_EARTH, 6_578_000.0, v_low, &f, KSC_LAT_RAD, None, false).unwrap();
        assert!((g3.inclination_rad - KSC_LAT_RAD).abs() < 1e-12);
    }

    /// Vallado §6.4: ISS (i = 51.6°) from KSC (φ = 28.5°) launches at
    /// β ≈ 45° from north — the familiar Space Shuttle ISS azimuth.
    #[test]
    fn launch_azimuth_matches_the_vallado_iss_from_ksc_example() {
        let f = earth_frame();
        let v = Vector3::new(1.0, 0.0, 0.0) * 3_000.0;
        let g = launch_geometry(MU_EARTH, 6_578_000.0, v, &f, KSC_LAT_RAD, Some(51.6_f64.to_radians()), false).unwrap();
        let expected = (51.6_f64.to_radians().cos() / KSC_LAT_RAD.cos()).asin();
        assert!((g.launch_azimuth_rad - expected).abs() < 1e-12);
        assert!((g.launch_azimuth_rad.to_degrees() - 45.0).abs() < 0.2, "got {}°", g.launch_azimuth_rad.to_degrees());
        // Due east from the site at i = φ.
        let g_east = launch_geometry(MU_EARTH, 6_578_000.0, v, &f, KSC_LAT_RAD, Some(KSC_LAT_RAD), false).unwrap();
        assert!((g_east.launch_azimuth_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }

    /// An inclination below the site latitude is a dogleg — reported
    /// infeasible, never priced; the plane still contains the asymptote
    /// whenever i ≥ |DLA|.
    #[test]
    fn inclination_below_the_site_latitude_is_flagged_as_a_dogleg() {
        let f = earth_frame();
        let v = Vector3::new(1.0, 0.1, 0.05).normalize() * 3_000.0; // |DLA| ≈ 2.9°
        let g = launch_geometry(MU_EARTH, 6_578_000.0, v, &f, KSC_LAT_RAD, Some(10_f64.to_radians()), false).unwrap();
        assert!(!g.feasible_no_dogleg);
        assert!(g.plane_normal_hat.dot(&v.normalize()).abs() < 1e-9);
        // Requesting less than |DLA| is clamped up to |DLA| (the plane
        // must contain the asymptote to reach the target at all).
        let v_hi = Vector3::new(0.36, -0.48, 0.8) * 3_000.0;
        let g2 = launch_geometry(MU_EARTH, 6_578_000.0, v_hi, &f, 0.0, Some(10_f64.to_radians()), false).unwrap();
        assert!((g2.inclination_rad - 0.8_f64.asin()).abs() < 1e-12);
    }

    /// The real check of the plane construction: propagate the injection
    /// state under point-mass gravity and confirm the velocity direction
    /// converges to the requested asymptote (same criterion
    /// `departure::tests` uses for the free-plane construction).
    #[test]
    fn propagated_injection_converges_to_the_asymptote_direction() {
        let f = earth_frame();
        let v_inf = 3_200.0;
        let v = Vector3::new(0.36, -0.48, 0.8) * v_inf;
        for branch in [false, true] {
            let g = launch_geometry(MU_EARTH, 6_871_000.0, v, &f, KSC_LAT_RAD, Some(60_f64.to_radians()), branch).unwrap();
            let pts = propagate(g.injection_r_m, g.injection_v_mps, 0.0, 30.0 * 86_400.0, MU_EARTH, &[], 3600.0, 1e-12, 1e-3);
            let end = pts.last().unwrap();
            let angle_deg = end.v_mps.normalize().dot(&v.normalize()).clamp(-1.0, 1.0).acos().to_degrees();
            assert!(angle_deg < 0.5, "branch {branch}: {angle_deg:.4}° off the asymptote");
            assert!(end.v_mps.norm() > v_inf && end.v_mps.norm() < v_inf + 50.0);
        }
    }

    /// Node and coast-angle bookkeeping: the node lies on the equator and in
    /// the plane; the periapsis sits `coast_angle` past the site's ascending
    /// crossing.
    #[test]
    fn node_and_coast_angle_are_consistent_with_the_plane() {
        let f = earth_frame();
        let v = Vector3::new(0.36, -0.48, 0.8) * 3_000.0;
        let g = launch_geometry(MU_EARTH, 6_578_000.0, v, &f, KSC_LAT_RAD, Some(60_f64.to_radians()), false).unwrap();
        let node = Vector3::new(g.raan_rad.cos(), g.raan_rad.sin(), 0.0);
        assert!(node.dot(&g.plane_normal_hat).abs() < 1e-9, "node must lie in the plane");
        let t_hat = g.plane_normal_hat.cross(&node);
        let u_peri = g.injection_r_m.dot(&t_hat).atan2(g.injection_r_m.dot(&node));
        let u_site = (KSC_LAT_RAD.sin() / 60_f64.to_radians().sin()).asin();
        let expected = (u_peri - u_site).rem_euclid(std::f64::consts::TAU);
        assert!((g.coast_angle_rad - expected).abs() < 1e-12);
        // The site's ascending-pass point really is at the site latitude.
        let site_point = node * u_site.cos() + t_hat * u_site.sin();
        assert!((site_point.z.asin() - KSC_LAT_RAD).abs() < 1e-12);
    }

    #[test]
    fn zero_v_infinity_has_no_geometry() {
        let f = earth_frame();
        assert!(launch_geometry(MU_EARTH, 6_578_000.0, Vector3::zeros(), &f, 0.0, None, false).is_none());
    }
}
