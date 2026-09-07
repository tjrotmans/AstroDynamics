//! Gravitational acceleration model (two-body point mass and third-body perturbations)

use nalgebra::SVector;
use crate::constants;

/// Gravitational acceleration model (two-body point mass)
pub struct GravityModel;

impl GravityModel {
    /// Compute Earth gravitational acceleration vector (two-body, point mass).
    ///
    /// # Arguments
    /// * `position` - Current position vector [x, y, z] in meters (ECI frame)
    /// * `_mass` - Spacecraft mass (unused - gravitational acceleration is mass-independent)
    ///
    /// # Returns
    /// Acceleration vector [ax, ay, az] in m/s² (ECI frame)
    pub fn compute(position: &SVector<f64, 3>, _mass: f64) -> SVector<f64, 3> {
        let r_norm = position.norm();
        let r_unit = position / r_norm;

        (-constants::G * constants::M_EARTH / (r_norm * r_norm)) * r_unit
    }

    /// Zonal harmonic gravity perturbation [m/s²] in ECI frame, including J2, J3, J4.
    ///
    /// Call this in addition to [`GravityModel::compute`] (point-mass) to add the
    /// non-spherical corrections. J2 dominates (~99.7% of the total); J3 and J4 are
    /// included for completeness but contribute <0.01 m/s over a typical TLI burn.
    ///
    /// Reference: Montenbruck & Gill, "Satellite Orbits", §3.2.
    pub fn zonal_harmonics(pos: &SVector<f64, 3>) -> SVector<f64, 3> {
        Self::j2(pos) + Self::j3(pos) + Self::j4(pos)
    }

    /// J2 perturbation only (sufficient for most uses).
    pub fn j2(pos: &SVector<f64, 3>) -> SVector<f64, 3> {
        let r    = pos.norm();
        let r2   = r * r;
        let r5   = r2 * r2 * r;
        let z    = pos[2];
        let re   = constants::EARTH_EQUATORIAL_RADIUS;
        let z_r2 = (z / r) * (z / r);
        let fac  = 1.5 * constants::J2 * constants::MU_EARTH * re * re / r5;

        SVector::<f64, 3>::new(
            fac * pos[0] * (5.0 * z_r2 - 1.0),
            fac * pos[1] * (5.0 * z_r2 - 1.0),
            fac * pos[2] * (5.0 * z_r2 - 3.0),
        )
    }

    /// J3 perturbation (~1/430 of J2; north-south asymmetry).
    fn j3(pos: &SVector<f64, 3>) -> SVector<f64, 3> {
        let r    = pos.norm();
        let r2   = r * r;
        let r4   = r2 * r2;
        let r7   = r4 * r2 * r;
        let r9   = r4 * r4 * r;
        let z    = pos[2];
        let re   = constants::EARTH_EQUATORIAL_RADIUS;
        let re3  = re * re * re;
        let z_r2 = (z / r) * (z / r);

        // a_J3 = −(μ J3 Re³ / 2) ∇[(5z³ − 3z r²) / r⁷]
        // x,y: −(5/2) μ J3 Re³ · xz/r⁷ · (3 − 7(z/r)²)
        // z:    (μ J3 Re³) / (2r⁹) · (3r⁴ − 30z²r² + 35z⁴)
        let fac_xy = -(5.0 / 2.0) * constants::J3 * constants::MU_EARTH * re3 * z / r7;
        let fac_z  = constants::J3 * constants::MU_EARTH * re3 / (2.0 * r9);
        let z2     = z * z;

        SVector::<f64, 3>::new(
            fac_xy * pos[0] * (3.0 - 7.0 * z_r2),
            fac_xy * pos[1] * (3.0 - 7.0 * z_r2),
            fac_z  * (3.0 * r4 - 30.0 * z2 * r2 + 35.0 * z2 * z2),
        )
    }

    /// J4 perturbation (~1/670 of J2; symmetric about equator).
    fn j4(pos: &SVector<f64, 3>) -> SVector<f64, 3> {
        let r    = pos.norm();
        let r2   = r * r;
        let r7   = r2 * r2 * r2 * r;
        let z    = pos[2];
        let re   = constants::EARTH_EQUATORIAL_RADIUS;
        let re4  = re * re * re * re;
        let z_r2 = (z / r) * (z / r);

        // a_J4 = −(μ J4 Re⁴ / 8) ∇[(35z⁴ − 30z²r² + 3r⁴) / r⁹]
        // x,y: (15/8) μ J4 Re⁴ · x/r⁷ · (1 − 14(z/r)² + 21(z/r)⁴)
        // z:   ( 5/8) μ J4 Re⁴ · z/r⁷ · (15 − 70(z/r)² + 63(z/r)⁴)
        let fac = constants::J4 * constants::MU_EARTH * re4 / r7;

        SVector::<f64, 3>::new(
            (15.0 / 8.0) * fac * pos[0] * (1.0 - 14.0 * z_r2 + 21.0 * z_r2 * z_r2),
            (15.0 / 8.0) * fac * pos[1] * (1.0 - 14.0 * z_r2 + 21.0 * z_r2 * z_r2),
            ( 5.0 / 8.0) * fac * pos[2] * (15.0 - 70.0 * z_r2 + 63.0 * z_r2 * z_r2),
        )
    }

    /// Sun third-body perturbation acceleration in the ECI frame [m/s²].
    ///
    /// Same indirect-term formulation as the Moon. At lunar distance the Sun
    /// contributes ~1.6 × 10⁻⁵ m/s², accumulating ~14 m/s over 10 days — the
    /// dominant missing perturbation for inclination on the return leg.
    ///
    /// # Arguments
    /// * `sc_pos`  - Spacecraft position in ECI [m]
    /// * `sun_pos` - Sun position in ECI [m] (from ephemeris, updated each step)
    pub fn sun_third_body(sc_pos: &SVector<f64, 3>, sun_pos: &SVector<f64, 3>) -> SVector<f64, 3> {
        let r_rel      = sun_pos - sc_pos;
        let r_rel_norm = r_rel.norm();
        let r_sun_norm = sun_pos.norm();

        constants::MU_SUN * (
            r_rel / (r_rel_norm * r_rel_norm * r_rel_norm)
            - sun_pos / (r_sun_norm * r_sun_norm * r_sun_norm)
        )
    }

    /// Generic two-body point-mass acceleration [m/s²] for any central body.
    ///
    /// Unlike [`GravityModel::compute`] (hardcoded to Earth), this takes `mu`
    /// directly — use for any central body (Sun, Mars, Bennu, ...).
    pub fn point_mass(position: &SVector<f64, 3>, mu: f64) -> SVector<f64, 3> {
        let r_norm = position.norm();
        -(mu / (r_norm * r_norm * r_norm)) * position
    }

    /// Generic third-body perturbation acceleration in the ECI frame [m/s²].
    ///
    /// Identical in structure to `moon_third_body` and `sun_third_body` but accepts
    /// an arbitrary gravitational parameter `mu_body` [m³/s²].  Use this for Venus,
    /// Mars, Jupiter, or any other perturber with a known GM.
    ///
    /// Formula: `a = mu * ((r_body - r_sc) / |r_body - r_sc|³  -  r_body / |r_body|³)`
    pub fn third_body(
        sc_pos:   &SVector<f64, 3>,
        body_pos: &SVector<f64, 3>,
        mu_body:  f64,
    ) -> SVector<f64, 3> {
        let r_rel      = body_pos - sc_pos;
        let r_rel_norm = r_rel.norm();
        let r_body_norm = body_pos.norm();
        mu_body * (
            r_rel / (r_rel_norm * r_rel_norm * r_rel_norm)
            - body_pos / (r_body_norm * r_body_norm * r_body_norm)
        )
    }

    /// Moon third-body perturbation acceleration in the ECI frame [m/s²].
    ///
    /// Includes the indirect term to account for Earth's acceleration due to the Moon.
    /// This is necessary because ECI is referenced to Earth, which is itself accelerated
    /// by the Moon in the three-body problem.
    ///
    /// Formula: `a = GM_moon * ((r_moon - r_sc) / |r_moon - r_sc|³ - r_moon / |r_moon|³)`
    ///
    /// # Arguments
    /// * `sc_pos`   - Spacecraft position in ECI [m]
    /// * `moon_pos` - Moon position in ECI [m] (from ephemeris)
    ///
    /// # Returns
    /// Perturbation acceleration [m/s²] in ECI frame
    pub fn moon_third_body(sc_pos: &SVector<f64, 3>, moon_pos: &SVector<f64, 3>) -> SVector<f64, 3> {
        let r_rel = moon_pos - sc_pos;
        let r_rel_norm = r_rel.norm();
        let r_moon_norm = moon_pos.norm();

        constants::MU_MOON * (
            r_rel / (r_rel_norm * r_rel_norm * r_rel_norm)
            - moon_pos / (r_moon_norm * r_moon_norm * r_moon_norm)
        )
    }

    /// Parametric zonal harmonic gravity perturbation [m/s²] for any central body.
    ///
    /// Identical in structure to [`GravityModel::zonal_harmonics`] (Earth-specific)
    /// but accepts arbitrary body parameters — use for Bennu, Mars, the Moon, etc.
    ///
    /// # Arguments
    /// * `pos` – Spacecraft position in the body-centred frame [m]
    /// * `mu`  – Gravitational parameter of the central body [m³/s²]
    /// * `r0`  – Body reference radius (normalisation radius for Jn coefficients) [m]
    /// * `j2`, `j3`, `j4` – Zonal harmonic coefficients (dimensionless)
    pub fn zonal_harmonics_body(
        pos: &SVector<f64, 3>,
        mu:  f64,
        r0:  f64,
        j2:  f64,
        j3:  f64,
        j4:  f64,
    ) -> SVector<f64, 3> {
        zonal_j2_body(pos, mu, r0, j2)
            + zonal_j3_body(pos, mu, r0, j3)
            + zonal_j4_body(pos, mu, r0, j4)
    }

    /// Linearised tidal (Hill-frame) acceleration from a distant perturber [m/s²].
    ///
    /// In the Hill frame centred on a primary body (e.g. an asteroid), the Sun's
    /// gravity gradient acts on the spacecraft as:
    ///
    ///   a_tidal = −(μ_p / r_B³) r_sc + 3(μ_p / r_B⁵)(r_B · r_sc) r_B
    ///
    /// Valid when |r_sc| << |r_primary| (e.g. spacecraft within a few km of an asteroid
    /// at heliocentric distance 1 AU).
    ///
    /// # Arguments
    /// * `r_sc_from_primary` – Spacecraft position relative to primary body in the Hill frame [m]
    /// * `primary_pos`       – Primary body position relative to the perturber [m]
    ///                         (e.g. Bennu heliocentric position)
    /// * `mu_perturber`      – Gravitational parameter of the perturber (e.g. Sun) [m³/s²]
    pub fn tidal(
        r_sc_from_primary: &SVector<f64, 3>,
        primary_pos:       &SVector<f64, 3>,
        mu_perturber:       f64,
    ) -> SVector<f64, 3> {
        let rb2 = primary_pos.norm_squared();
        let rb  = rb2.sqrt();
        let rb3 = rb2 * rb;
        let rb5 = rb3 * rb2;
        let dot = primary_pos.dot(r_sc_from_primary);
        -mu_perturber / rb3 * r_sc_from_primary
            + 3.0 * mu_perturber / rb5 * dot * primary_pos
    }
}

impl GravityModel {
    /// Parametric zonal harmonic gravity for a central body whose rotational
    /// pole is NOT aligned with the working inertial frame's z-axis — true
    /// for every body except Earth, where ICRF z is the Earth pole by
    /// construction. [`GravityModel::zonal_harmonics_body`] is only correct
    /// when `pos` is already expressed in a frame whose z-axis is the
    /// body's true pole; calling it directly with an ICRF-frame `pos` for
    /// e.g. Mars (pole tilted ~37° from ICRF z) would apply the right
    /// magnitude in the wrong direction.
    ///
    /// This rotates `pos` into a frame whose z-axis is the body's pole,
    /// applies the standard axisymmetric zonal-harmonic formula, then
    /// rotates the resulting acceleration back into the input frame. The
    /// choice of in-plane (x, y) axes within that rotation is arbitrary —
    /// the zonal-harmonic potential is axisymmetric about the pole, so any
    /// right-handed orthonormal basis with `pole` as z gives the same
    /// round-tripped result (verified for the Earth/ICRF-aligned case
    /// against [`GravityModel::zonal_harmonics_body`] directly in the unit
    /// tests below).
    ///
    /// `pole_ra_rad`/`pole_dec_rad` — body pole right ascension/declination
    /// in the working inertial frame (ICRF), constant (J2000) term only.
    /// Secular/periodic pole drift is dropped — negligible over a single
    /// mission's duration, consistent with this propagator's "fast over
    /// precise" design (see the design notes, Phase 7's "Layer 1 Propagator
    /// Design"). Source citations for specific bodies' pole constants live
    /// with those constants in `body_models::TargetBody`, not here — this
    /// function is generic rotation math, not a physical constant.
    pub fn zonal_harmonics_body_oriented(
        pos: &SVector<f64, 3>,
        mu: f64,
        r0: f64,
        j2: f64,
        j3: f64,
        j4: f64,
        pole_ra_rad: f64,
        pole_dec_rad: f64,
    ) -> SVector<f64, 3> {
        let pole = SVector::<f64, 3>::new(
            pole_dec_rad.cos() * pole_ra_rad.cos(),
            pole_dec_rad.cos() * pole_ra_rad.sin(),
            pole_dec_rad.sin(),
        );
        // Any vector not parallel to `pole` works as the seed for building
        // an orthonormal basis; switch seeds near the pole to avoid a
        // degenerate (near-zero) cross product.
        let seed = if pole.z.abs() < 0.9 {
            SVector::<f64, 3>::new(0.0, 0.0, 1.0)
        } else {
            SVector::<f64, 3>::new(1.0, 0.0, 0.0)
        };
        let u = seed.cross(&pole).normalize();
        let v = pole.cross(&u);

        let pos_body = SVector::<f64, 3>::new(pos.dot(&u), pos.dot(&v), pos.dot(&pole));
        let a_body = zonal_j2_body(&pos_body, mu, r0, j2)
            + zonal_j3_body(&pos_body, mu, r0, j3)
            + zonal_j4_body(&pos_body, mu, r0, j4);
        u * a_body.x + v * a_body.y + pole * a_body.z
    }
}

// ── Parametric zonal harmonic helpers ────────────────────────────────────────
// Same formula as the Earth-specific j2/j3/j4 above, parameterised over mu/r0/Jn.

fn zonal_j2_body(pos: &SVector<f64, 3>, mu: f64, r0: f64, j2: f64) -> SVector<f64, 3> {
    let r    = pos.norm();
    let r2   = r * r;
    let r5   = r2 * r2 * r;
    let z    = pos[2];
    let z_r2 = (z / r) * (z / r);
    let fac  = 1.5 * j2 * mu * r0 * r0 / r5;
    SVector::<f64, 3>::new(
        fac * pos[0] * (5.0 * z_r2 - 1.0),
        fac * pos[1] * (5.0 * z_r2 - 1.0),
        fac * pos[2] * (5.0 * z_r2 - 3.0),
    )
}

fn zonal_j3_body(pos: &SVector<f64, 3>, mu: f64, r0: f64, j3: f64) -> SVector<f64, 3> {
    let r    = pos.norm();
    let r2   = r * r;
    let r4   = r2 * r2;
    let r7   = r4 * r2 * r;
    let r9   = r4 * r4 * r;
    let z    = pos[2];
    let z_r2 = (z / r) * (z / r);
    let r03  = r0 * r0 * r0;
    let z2   = z * z;
    let fac_xy = -(5.0 / 2.0) * j3 * mu * r03 * z / r7;
    let fac_z  = j3 * mu * r03 / (2.0 * r9);
    SVector::<f64, 3>::new(
        fac_xy * pos[0] * (3.0 - 7.0 * z_r2),
        fac_xy * pos[1] * (3.0 - 7.0 * z_r2),
        fac_z  * (3.0 * r4 - 30.0 * z2 * r2 + 35.0 * z2 * z2),
    )
}

fn zonal_j4_body(pos: &SVector<f64, 3>, mu: f64, r0: f64, j4: f64) -> SVector<f64, 3> {
    let r    = pos.norm();
    let r2   = r * r;
    let r7   = r2 * r2 * r2 * r;
    let z    = pos[2];
    let z_r2 = (z / r) * (z / r);
    let r04  = r0 * r0 * r0 * r0;
    let fac  = j4 * mu * r04 / r7;
    SVector::<f64, 3>::new(
        (15.0 / 8.0) * fac * pos[0] * (1.0 - 14.0 * z_r2 + 21.0 * z_r2 * z_r2),
        (15.0 / 8.0) * fac * pos[1] * (1.0 - 14.0 * z_r2 + 21.0 * z_r2 * z_r2),
        ( 5.0 / 8.0) * fac * pos[2] * (15.0 - 70.0 * z_r2 + 63.0 * z_r2 * z_r2),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OrbitalElements;

    #[test]
    fn gravity_points_towards_earth() {
        let state = OrbitalElements {
            a: 500e3 + 6371e3,
            e: 0.0,
            i: 0.0,
            w: 0.0,
            o: 0.0,
            nu: 0.0,
        };
        let mass = 100.0;
        let cartesian = state.as_cartesian();
        let pos: SVector<f64, 3> = cartesian.position().into();
        let accel = GravityModel::compute(&pos, mass);

        // Gravity should point inward (negative radial direction)
        let dot = accel.dot(&pos);
        assert!(dot < 0.0, "Gravity should point toward Earth center");
    }

    #[test]
    fn gravity_at_2000km_circular() {
        let state = OrbitalElements {
            a: 8371000.0,
            e: 0.0,
            i: 0.0,
            w: 0.0,
            o: 0.0,
            nu: 0.0,
        };
        let mass = 10.0;

        let cartesian = state.as_cartesian();
        let position: SVector<f64, 3> = cartesian.position().into();
        let accel = GravityModel::compute(&position, mass);

        assert!(!accel[0].is_nan(), "Gravity X component is NaN!");
        assert!(!accel[1].is_nan(), "Gravity Y component is NaN!");
        assert!(!accel[2].is_nan(), "Gravity Z component is NaN!");

        let dot = accel.dot(&position);
        assert!(dot < 0.0, "Gravity should point toward Earth center, got dot={}", dot);

        // Expected magnitude: GM/r² ≈ 5.7 m/s²
        let accel_mag = accel.norm();
        assert!(accel_mag > 5.6 && accel_mag < 5.8, "Unexpected gravity magnitude: {}", accel_mag);
    }

    #[test]
    fn gravity_inverse_square_law() {
        // Gravity at distance r vs 2r should follow inverse-square: ratio ≈ 4
        let pos_r = SVector::<f64, 3>::new(8371000.0, 0.0, 0.0);
        let pos_2r = SVector::<f64, 3>::new(8371000.0 * 2.0, 0.0, 0.0);
        let accel_r = GravityModel::compute(&pos_r, 10.0).norm();
        let accel_2r = GravityModel::compute(&pos_2r, 10.0).norm();
        let ratio = accel_r / accel_2r;
        assert!((ratio - 4.0).abs() < 0.01, "Inverse-square ratio should be ~4, got {}", ratio);
    }

    #[test]
    fn gravity_mass_independent() {
        let pos = SVector::<f64, 3>::new(8371000.0, 0.0, 0.0);
        let accel_10kg = GravityModel::compute(&pos, 10.0);
        let accel_100kg = GravityModel::compute(&pos, 100.0);
        assert!((accel_10kg - accel_100kg).norm() < 1e-15,
            "Gravity should be mass-independent");
    }

    /// Pole at RA=0/Dec=90° is exactly ICRF z — `zonal_harmonics_body_oriented`
    /// must reduce to plain `zonal_harmonics_body` for every test position,
    /// not just ones lying on a coordinate axis.
    #[test]
    fn oriented_zonal_matches_unrotated_when_pole_is_icrf_z() {
        const MU: f64 = 4.282_837_362_069_909e13; // Mars
        const R0: f64 = 3_396_200.0;
        const J2: f64 = 1.960_45e-3;
        const J3: f64 = 3.142_5e-5;
        const J4: f64 = -1.538_5e-5;

        for pos in [
            SVector::<f64, 3>::new(5_000_000.0, 1_200_000.0, 800_000.0),
            SVector::<f64, 3>::new(-3_000_000.0, 4_000_000.0, -2_500_000.0),
            SVector::<f64, 3>::new(0.0, 0.0, 6_000_000.0),
        ] {
            let direct = zonal_j2_body(&pos, MU, R0, J2)
                + zonal_j3_body(&pos, MU, R0, J3)
                + zonal_j4_body(&pos, MU, R0, J4);
            let oriented = GravityModel::zonal_harmonics_body_oriented(
                &pos, MU, R0, J2, J3, J4, 0.0, std::f64::consts::FRAC_PI_2,
            );
            assert!(
                (direct - oriented).norm() / direct.norm() < 1e-10,
                "oriented (pole=ICRF z) should match unrotated formula exactly: {direct:?} vs {oriented:?}"
            );
        }
    }

    /// A 90°-tilted pole should produce a materially different acceleration
    /// than treating the same position as if the pole were ICRF z — confirms
    /// the rotation is actually doing something, not silently degenerating
    /// to the unrotated case for an arbitrary pole.
    #[test]
    fn oriented_zonal_differs_from_unrotated_for_tilted_pole() {
        const MU: f64 = 4.282_837_362_069_909e13;
        const R0: f64 = 3_396_200.0;
        let pos = SVector::<f64, 3>::new(5_000_000.0, 0.0, 1_000_000.0);

        let unrotated = zonal_j2_body(&pos, MU, R0, 1.960_45e-3);
        // Pole tilted 90° into the x-axis (RA=0, Dec=0) — a deliberately
        // extreme case to make the difference unambiguous.
        let oriented = GravityModel::zonal_harmonics_body_oriented(
            &pos, MU, R0, 1.960_45e-3, 0.0, 0.0, 0.0, 0.0,
        );
        assert!(
            (unrotated - oriented).norm() / unrotated.norm() > 0.1,
            "tilted-pole result should differ materially from the unrotated formula"
        );
    }

    /// A position lying exactly on the pole axis should produce zero
    /// horizontal (in-equatorial-plane) acceleration component, regardless
    /// of which way the pole itself points in ICRF — a basic sanity check
    /// that the rotation correctly identifies "on-axis" for a tilted pole.
    #[test]
    fn oriented_zonal_on_axis_position_has_no_off_axis_component() {
        const MU: f64 = 4.282_837_362_069_909e13;
        const R0: f64 = 3_396_200.0;
        let ra: f64 = 0.7; // arbitrary tilted pole
        let dec: f64 = 0.9;
        let pole = SVector::<f64, 3>::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
        let pos_on_axis = pole * 6_000_000.0;

        let accel = GravityModel::zonal_harmonics_body_oriented(
            &pos_on_axis, MU, R0, 1.960_45e-3, 3.142_5e-5, -1.538_5e-5, ra, dec,
        );
        // On-axis acceleration must be parallel to the pole (no component
        // perpendicular to it) — check via the cross product magnitude.
        let cross_mag = accel.cross(&pole).norm();
        assert!(
            cross_mag / accel.norm() < 1e-9,
            "on-axis acceleration should be parallel to the pole, cross magnitude ratio = {}",
            cross_mag / accel.norm()
        );
    }
}
