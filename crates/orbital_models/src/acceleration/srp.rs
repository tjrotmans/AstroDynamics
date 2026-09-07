//! Solar radiation pressure models — three levels of fidelity.
//!
//! | Model          | Type                | Use case                            |
//! |----------------|---------------------|-------------------------------------|
//! | `SolarSailModel<F>` | Sail thrust    | Solar sail trajectory optimisation  |
//! | `cannonball`   | Free function       | EKF filter model, cruise SRP        |
//! | `flat_plate_*` | Free functions      | Truth dynamics with attitude-dep SRP|
//!
//! Analogous to `GravityModel` having point-mass and zonal-harmonic variants:
//! `cannonball` ≈ point-mass; `flat_plate` ≈ full harmonic field.
//!
//! ## Cannonball model
//! Force is always anti-sun, constant effective area, scalar reflectivity.
//! No attitude dependence — suitable for orbit determination filters where
//! attitude is not a state variable.
//!
//! ## Flat-plate model
//! Each plate contributes `F_i = −P·A_i·cosθ_i·[(1−ρs)·ŝ + 2(ρs·cosθ_i+ρd/3)·n̂_i]`.
//! The in-plane term `(1−ρs)·ŝ` gives a force component tangent to the plate
//! surface — this is the dominant source of attitude-dependent SRP torque when
//! the lever arm from each plate centre to CoM is accounted for.

use nalgebra::{SVector, Vector3, Vector4};
use std::marker::PhantomData;
use crate::constants::{P_SRP, AU};
use crate::frames::AttitudeFrame;
use crate::attitude::{inertial_to_body, body_to_inertial};

// ─────────────────────────────────────────────────────────────────────────────
// Solar sail model (renamed from SolarPressureModel — sail-specific thrust law)
// ─────────────────────────────────────────────────────────────────────────────

/// Solar sail SRP model parameterized by attitude reference frame.
///
/// Uses the sail thrust law F ∝ (1+r)·cos²α·n̂, where α is the cone angle
/// between the sail normal and the Sun direction.  This is **not** the same as
/// the spacecraft cannonball or flat-plate models — it is specific to actively
/// oriented reflective sails used for propulsion.
///
/// Previously named `SolarPressureModel`; renamed to avoid confusion with the
/// spacecraft SRP models below.
pub struct SolarSailModel<F: AttitudeFrame>(PhantomData<F>);

/// Backward-compatibility alias.  Prefer `SolarSailModel` in new code.
pub type SolarPressureModel<F> = SolarSailModel<F>;

impl<F: AttitudeFrame> SolarSailModel<F> {
    /// Compute solar sail acceleration vector.
    ///
    /// # Arguments
    /// * `position`     – Position in inertial frame [m]
    /// * `velocity`     – Velocity in inertial frame [m/s]
    /// * `sail_area`    – Sail area [m²]
    /// * `mass`         – Spacecraft mass [kg]
    /// * `reflectivity` – Sail reflectivity coefficient (0–1)
    /// * `angle1`       – First attitude angle (frame-specific)
    /// * `angle2`       – Second attitude angle (frame-specific)
    /// * `sun_position` – Sun position in same frame [m]; `zeros()` for heliocentric
    pub fn compute(
        position:     &SVector<f64, 3>,
        velocity:     &SVector<f64, 3>,
        sail_area:    f64,
        mass:         f64,
        reflectivity: f64,
        angle1:       f64,
        angle2:       f64,
        sun_position: &SVector<f64, 3>,
    ) -> SVector<f64, 3> {
        let n_inertial = F::sail_normal_inertial(angle1, angle2, position, velocity, sun_position);
        let s_vec  = position - sun_position;
        let s_hat  = s_vec / s_vec.norm();
        let cos_a  = n_inertial.dot(&s_hat);
        let mag    = P_SRP * sail_area * (1.0 + reflectivity) * cos_a * cos_a;
        mag * n_inertial / mass
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helper
// ─────────────────────────────────────────────────────────────────────────────

/// Radiation pressure at a given heliocentric distance [N/m²].
///
/// Scales `P_SRP` (at 1 AU) by the inverse-square law.
/// Used by both cannonball and flat-plate models.
#[inline]
pub fn pressure_at(helio_dist_m: f64) -> f64 {
    let r = AU / helio_dist_m;
    P_SRP * r * r
}

// ─────────────────────────────────────────────────────────────────────────────
// Cannonball model
// ─────────────────────────────────────────────────────────────────────────────

/// Spacecraft cannonball SRP acceleration [m/s²].
///
/// Force is always along the anti-sun direction with constant effective area.
/// No attitude dependence.  Used in EKF filter models where attitude is not
/// a state variable.
///
/// `sun_hat_inertial` = unit spacecraft→Sun direction in the inertial frame.
/// `p_srp`            = radiation pressure at the current heliocentric distance [N/m²]
///                      (use [`pressure_at`] to compute from distance).
#[inline]
pub fn cannonball(
    sun_hat_inertial: &Vector3<f64>,
    p_srp:  f64,
    c_r:    f64,
    area:   f64,
    mass:   f64,
) -> Vector3<f64> {
    // Force pushes spacecraft away from the Sun (along sun_hat, which points S/C→Sun,
    // so the force is in the +sun_hat direction — away from sun).
    p_srp * c_r * area / mass * sun_hat_inertial
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat-plate model
// ─────────────────────────────────────────────────────────────────────────────

/// One flat plate of the spacecraft, in body-fixed coordinates.
///
/// Used by the flat-plate SRP force and torque functions.
/// The `center_body` field is the plate's geometric centre relative to the
/// spacecraft centre of mass, expressed in the body frame.  It is needed to
/// compute the SRP torque: τ_i = center_i × F_i.
#[derive(Clone, Copy, Debug)]
pub struct Plate {
    /// Outward unit normal in the body frame.
    pub normal:       Vector3<f64>,
    /// Plate area [m²].
    pub area:         f64,
    /// Specular reflectivity ρ_s.
    pub rho_s:        f64,
    /// Diffuse reflectivity ρ_d.
    pub rho_d:        f64,
    /// Panels are thin and illuminated from whichever side faces the Sun.
    /// Bus faces are single-sided (only the outward face is exposed to space).
    pub double_sided: bool,
    /// Plate centre relative to spacecraft CoM in the body frame [m].
    /// Required for torque calculation: τ_i = center_body × F_i.
    pub center_body:  Vector3<f64>,
}

/// Net SRP force on the spacecraft in the **body frame** [N].
///
/// Sums the flat-plate force law over all illuminated plates:
///
///   F_i = −P·A_i·cosθ_i·[ (1−ρs)·ŝ + 2·(ρs·cosθ_i + ρd/3)·n̂_i ]
///
/// where ŝ is the spacecraft→Sun unit vector in the body frame.
/// The in-plane term `(1−ρs)·ŝ` contributes a force tangent to the plate
/// surface when ŝ is not perpendicular to n̂_i.
///
/// `sun_hat_body` = unit spacecraft→Sun direction in the body frame.
/// `p_srp`        = radiation pressure at current heliocentric distance [N/m²].
pub fn flat_plate_force_body(
    plates:       &[Plate],
    sun_hat_body: &Vector3<f64>,
    p_srp:        f64,
) -> Vector3<f64> {
    plates.iter().map(|pl| single_plate_force_body(pl, sun_hat_body, p_srp)).sum()
}

/// Per-plate SRP force breakdown, one entry per input plate, in the SAME
/// ORDER — spacecraft-builder derived-mass-properties extension
/// `flat_plate_force_body`'s total is exactly
/// `.iter().sum()` of this function's own output (both call the same
/// per-plate law, [`single_plate_force_body`], never a re-derived copy) —
/// this exists so a caller (the new per-plate-SRP-vector endpoint) can show
/// which plate contributes how much, and in which direction, rather than
/// only the summed net force. An unilluminated plate's entry is the zero
/// vector, not omitted, so the output always has one entry per input plate.
pub fn flat_plate_force_per_plate(
    plates:       &[Plate],
    sun_hat_body: &Vector3<f64>,
    p_srp:        f64,
) -> Vec<Vector3<f64>> {
    plates.iter().map(|pl| single_plate_force_body(pl, sun_hat_body, p_srp)).collect()
}

/// The flat-plate SRP force law (see [`flat_plate_force_body`]'s doc
/// comment for the formula) applied to ONE plate — the shared building
/// block [`flat_plate_force_body`], [`flat_plate_torque_body`], and
/// [`flat_plate_force_per_plate`] all call, so the physics lives in exactly
/// one place. Returns the zero vector for an unilluminated plate (`cos <=
/// 0`), matching those callers' own skip behaviour.
#[inline]
fn single_plate_force_body(pl: &Plate, sun_hat_body: &Vector3<f64>, p_srp: f64) -> Vector3<f64> {
    let (n, cos) = plate_illumination(pl, sun_hat_body);
    if cos <= 0.0 {
        return Vector3::zeros();
    }
    let recoil = 2.0 * (pl.rho_s * cos + pl.rho_d / 3.0);
    -p_srp * pl.area * cos * ((1.0 - pl.rho_s) * sun_hat_body + recoil * n)
}

/// Net SRP torque on the spacecraft in the **body frame** [N·m].
///
/// Computed as τ = Σ_i (center_i × F_i), where F_i is the flat-plate force
/// on plate i acting at that plate's centre relative to CoM.
///
/// This correctly captures:
/// - The in-plane force component from asymmetric illumination.
/// - Attitude-dependent variation in both torque magnitude and direction.
/// - Asymmetric illumination between the two solar panels (when one is partially shadowed).
///
/// Note: the cannonball model has no plate geometry and therefore no valid
/// physical SRP torque — use this function with the detailed plate model instead.
pub fn flat_plate_torque_body(
    plates:       &[Plate],
    sun_hat_body: &Vector3<f64>,
    p_srp:        f64,
) -> Vector3<f64> {
    let mut tau = Vector3::zeros();
    for pl in plates {
        let f_plate = single_plate_force_body(pl, sun_hat_body, p_srp);
        tau += pl.center_body.cross(&f_plate);
    }
    tau
}

/// Convenience: flat-plate SRP acceleration in the **inertial frame** [m/s²].
///
/// Rotates the Sun direction into the body frame (using quaternion `q`),
/// computes the body-frame force via [`flat_plate_force_body`], then rotates
/// back to the inertial frame and divides by mass.
///
/// `q`               = attitude quaternion [w,x,y,z] (body → inertial).
/// `sun_hat_inertial`= unit spacecraft→Sun direction in the inertial frame.
pub fn flat_plate_accel(
    plates:           &[Plate],
    q:                &Vector4<f64>,
    sun_hat_inertial: &Vector3<f64>,
    p_srp:            f64,
    mass:             f64,
) -> Vector3<f64> {
    let sun_hat_body = inertial_to_body(q, sun_hat_inertial);
    let f_body       = flat_plate_force_body(plates, &sun_hat_body, p_srp);
    body_to_inertial(q, &f_body) / mass
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `(outward_normal, cos_theta)` for a plate given the sun direction.
/// Flips the normal for double-sided plates facing away from the Sun.
/// Returns cos ≤ 0 when the plate is unilluminated.
#[inline]
fn plate_illumination(pl: &Plate, sun_hat_body: &Vector3<f64>) -> (Vector3<f64>, f64) {
    let mut n   = pl.normal;
    let mut cos = n.dot(sun_hat_body);
    if pl.double_sided && cos < 0.0 {
        n   = -n;
        cos = -cos;
    }
    (n, cos)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OrbitalElements;
    use crate::frames::SunPointingFrame;
    use crate::constants::SUN_POSITION;

    fn eci_sun() -> SVector<f64, 3> {
        SVector::<f64, 3>::from_column_slice(&SUN_POSITION)
    }

    // ── SolarSailModel tests (unchanged from before) ──────────────────────

    #[test]
    fn sail_zero_when_perpendicular() {
        let state = OrbitalElements { a: 7371000.0, e: 0.0, i: 0.0, w: 0.0, o: 0.0, nu: 0.0 };
        let cartesian = state.as_cartesian();
        let pos: SVector<f64, 3> = cartesian.position().into();
        let vel: SVector<f64, 3> = cartesian.velocity().into();
        let accel = SolarSailModel::<SunPointingFrame>::compute(
            &pos, &vel, 100.0, 10.0, 1.6, std::f64::consts::PI / 2.0, 0.0, &eci_sun(),
        );
        assert!(accel.norm() < 1e-10, "sail SRP should be zero when perpendicular: {}", accel.norm());
    }

    #[test]
    fn sail_scales_with_area() {
        let state = OrbitalElements { a: 8371000.0, e: 0.0, i: 0.0, w: 0.0, o: 0.0, nu: 0.0 };
        let c = state.as_cartesian();
        let pos: SVector<f64, 3> = c.position().into();
        let vel: SVector<f64, 3> = c.velocity().into();
        let sun = eci_sun();
        let a100 = SolarSailModel::<SunPointingFrame>::compute(&pos,&vel,100.0,10.0,1.6,0.1,0.0,&sun);
        let a200 = SolarSailModel::<SunPointingFrame>::compute(&pos,&vel,200.0,10.0,1.6,0.1,0.0,&sun);
        let ratio = a200.norm() / a100.norm();
        assert!((ratio - 2.0).abs() < 0.01, "sail SRP scales with area, ratio={}", ratio);
    }

    // ── Cannonball tests ──────────────────────────────────────────────────

    #[test]
    fn cannonball_points_away_from_sun() {
        let sun_hat = Vector3::new(-1.0_f64, 0.0, 0.0); // sun is in -x, so away is +x
        let a = cannonball(&(-sun_hat), 4.56e-6, 1.4, 4.0, 1000.0);
        assert!(a[0] > 0.0, "cannonball should push away from sun");
    }

    #[test]
    fn cannonball_scales_with_area_and_cr() {
        let sun_hat = Vector3::new(0.0, 0.0, 1.0);
        let a1 = cannonball(&sun_hat, 4.56e-6, 1.0, 1.0, 1.0);
        let a2 = cannonball(&sun_hat, 4.56e-6, 2.0, 2.0, 1.0);
        assert!((a2.norm() / a1.norm() - 4.0).abs() < 1e-10);
    }

    // ── Flat-plate tests ──────────────────────────────────────────────────

    fn simple_plate(normal: Vector3<f64>, center: Vector3<f64>) -> Plate {
        Plate { normal, area: 1.0, rho_s: 0.0, rho_d: 0.0, double_sided: false,
                center_body: center }
    }

    #[test]
    fn flat_plate_force_perpendicular_face() {
        // Sun normal-on to +z face → force along -z (anti-sun for fully absorbing plate)
        let plate = simple_plate(Vector3::z(), Vector3::zeros());
        let sun_hat = Vector3::z(); // sun in +z direction
        let f = flat_plate_force_body(&[plate], &sun_hat, 1.0);
        // (1-rho_s)=1, recoil=0 → F = -P*A*cos * sun_hat = -1 * z
        assert!((f - Vector3::new(0.0, 0.0, -1.0)).norm() < 1e-10,
                "force should be -z for sun-facing +z plate: {:?}", f);
    }

    #[test]
    fn flat_plate_torque_from_offset_center() {
        // Plate at y = +1 m, normal +z, sun hits from +z
        // F = -z (absorption pushes away from sun along -sun_hat = -z)
        // τ = r × F = (+y) × (-z) = -(y × z) = -x
        let plate = Plate { normal: Vector3::z(), area: 1.0, rho_s: 0.0, rho_d: 0.0,
                            double_sided: false, center_body: Vector3::new(0.0, 1.0, 0.0) };
        let sun_hat = Vector3::z();
        let tau = flat_plate_torque_body(&[plate], &sun_hat, 1.0);
        assert!(tau[0] < 0.0, "torque should be -x: {:?}", tau);
        assert!(tau[1].abs() < 1e-10 && tau[2].abs() < 1e-10);
    }

    #[test]
    fn flat_plate_symmetric_panels_cancel_torque() {
        // Two symmetric panels at ±y produce zero net torque when illuminated equally
        let p1 = Plate { normal: Vector3::z(), area: 1.0, rho_s: 0.1, rho_d: 0.1,
                         double_sided: true, center_body: Vector3::new(0.0,  2.0, 0.0) };
        let p2 = Plate { normal: Vector3::z(), area: 1.0, rho_s: 0.1, rho_d: 0.1,
                         double_sided: true, center_body: Vector3::new(0.0, -2.0, 0.0) };
        let sun_hat = Vector3::z(); // illuminates both equally
        let tau = flat_plate_torque_body(&[p1, p2], &sun_hat, 1.0);
        assert!(tau.norm() < 1e-10, "symmetric panels cancel: {:?}", tau);
    }

    /// `flat_plate_force_per_plate`'s per-plate breakdown must sum to
    /// exactly the same total `flat_plate_force_body` returns (both call
    /// the same `single_plate_force_body` helper, per the 
    /// refactor for the spacecraft-builder derived-mass-properties work) —
    /// and one plate facing away from the Sun should show up as an exact
    /// zero entry, not merely be excluded from the list.
    #[test]
    fn per_plate_force_sums_to_the_same_total_and_zeros_unilluminated_plates() {
        let lit = Plate { normal: Vector3::z(), area: 1.0, rho_s: 0.1, rho_d: 0.2,
                           double_sided: false, center_body: Vector3::new(0.0, 1.0, 0.0) };
        let dark = Plate { normal: -Vector3::z(), area: 1.0, rho_s: 0.1, rho_d: 0.2,
                            double_sided: false, center_body: Vector3::new(0.0, -1.0, 0.0) };
        let sun_hat = Vector3::z();
        let plates = [lit, dark];

        let total = flat_plate_force_body(&plates, &sun_hat, 1.0);
        let per_plate = flat_plate_force_per_plate(&plates, &sun_hat, 1.0);
        assert_eq!(per_plate.len(), 2, "one entry per input plate, including unilluminated ones");
        assert!((per_plate[0] + per_plate[1] - total).norm() < 1e-12, "per-plate sum should equal the total");
        assert!(per_plate[1].norm() < 1e-15, "the plate facing away from the Sun should be exactly zero: {:?}", per_plate[1]);
        assert!(per_plate[0].norm() > 1e-6, "the illuminated plate should have a real nonzero force");
    }

    #[test]
    fn flat_plate_unshaded_panel_creates_net_torque() {
        // Only one panel illuminated (other is in shadow) → net torque
        let p1 = Plate { normal: Vector3::z(), area: 2.0, rho_s: 0.08, rho_d: 0.10,
                         double_sided: true, center_body: Vector3::new(0.0,  2.25, 0.0) };
        let sun_hat = Vector3::z();
        let tau = flat_plate_torque_body(&[p1], &sun_hat, 4.56e-6);
        assert!(tau.norm() > 0.0, "single panel should produce torque");
    }

    #[test]
    fn unilluminated_face_contributes_zero() {
        let plate = simple_plate(Vector3::z(), Vector3::zeros()); // normal +z
        let sun_hat = Vector3::new(0.0, 0.0, -1.0); // sun from -z, face unlit
        let f   = flat_plate_force_body(&[plate], &sun_hat, 1.0);
        let tau = flat_plate_torque_body(&[plate], &sun_hat, 1.0);
        assert!(f.norm() < 1e-15 && tau.norm() < 1e-15);
    }
}
