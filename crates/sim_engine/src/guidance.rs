//! Generic pointing-mode dispatch and station-keeping guidance.
//!
//! Wraps `orbital_models::guidance` (already body-agnostic: r, v in, quaternion
//! out) with a mission-facing `PointingMode` enum, and adds a simple
//! proportional station-keeping law used by `OrbitPhase`.

use nalgebra::{Vector3, Vector4};
use orbital_models::guidance::{nadir_orbit_normal_quat, velocity_aligned_quat};

#[derive(Clone, Copy, Debug)]
pub enum PointingMode {
    Nadir,
    VelocityAligned,
}

pub fn desired_quaternion(mode: PointingMode, r: &Vector3<f64>, v: &Vector3<f64>) -> Vector4<f64> {
    match mode {
        PointingMode::Nadir => nadir_orbit_normal_quat(r, v),
        PointingMode::VelocityAligned => velocity_aligned_quat(v),
    }
}

/// Vis-viva station-keeping impulse. Computes the tangential ΔV needed to
/// drive the spacecraft onto the circular orbit at `target_radius_m`, applied
/// at measurement-epoch cadence (caller gates on `measurement_taken` — see
/// `engine::step`).
///
/// Law: the target tangential speed at the *current* radius `r` is the
/// vis-viva speed for a circular orbit of radius `r_target`:
///
///   v_vis = sqrt(mu × (2/r − 1/r_target))
///
/// At r = r_target this equals sqrt(mu/r_target) (the familiar circular
/// speed). Below r_target (v_vis > v_circular_current) the burn is prograde;
/// above r_target it is retrograde.  This ensures the SK ALWAYS corrects
/// toward the target rather than worsening the error.
///
/// Background: the previous law used a fixed v_circ = sqrt(mu/r_target).
/// When the spacecraft drifted inside r_target (inevitable with SRP
/// perturbations), v_tang > v_circ → retrograde burn → orbit lowered further
/// → positive-feedback spiral.  The vis-viva law eliminates this instability.
///
/// Applies 5% of the required speed correction per burn so that repeated
/// burns converge without overshoot (~20 measurement updates to close a large
/// error).
///
/// Returns `None` when both:
/// - orbital radius is within `tolerance_m` of `target_radius_m`, AND
/// - tangential speed is within the equivalent velocity tolerance of `v_vis`.
pub fn station_keeping_dv(
    r: &Vector3<f64>,
    v: &Vector3<f64>,
    target_radius_m: f64,
    tolerance_m: f64,
    mu_central_m3s2: f64,
) -> Option<Vector3<f64>> {
    let r_norm = r.norm();
    let r_hat = r / r_norm;

    // Decompose velocity into radial and tangential components
    let v_radial = r_hat * r_hat.dot(v);
    let v_tang = v - v_radial;
    let v_tang_mag = v_tang.norm();

    // Vis-viva target speed at the current radius for a circular orbit at r_target.
    // Guard against r > 2×r_target (hyperbolic escape regime, negative argument).
    let vis_viva_arg = mu_central_m3s2 * (2.0 / r_norm - 1.0 / target_radius_m);
    if vis_viva_arg <= 0.0 {
        // Spacecraft has escaped the target orbit's energy domain — no burn.
        return None;
    }
    let v_target = vis_viva_arg.sqrt();

    let radius_err_m = (r_norm - target_radius_m).abs();
    let v_circ_ref = (mu_central_m3s2 / target_radius_m).sqrt();
    let v_tol = v_circ_ref * (tolerance_m / target_radius_m);
    // Signed radial velocity (positive = moving away from body)
    let v_radial_signed = r_hat.dot(v);

    // Fire if radius is off, tangential speed is off, OR radial velocity is non-zero.
    // The radial-velocity check is the key addition over the earlier tangential-only law:
    // it kills eccentricity that would otherwise grow secularly from SRP perturbations.
    if radius_err_m <= tolerance_m
        && (v_tang_mag - v_target).abs() <= v_tol
        && v_radial_signed.abs() <= v_tol
    {
        return None;
    }

    let v_hat = if v_tang_mag > 1e-9 * v_circ_ref {
        v_tang / v_tang_mag
    } else {
        let perp = Vector3::new(0.0, 0.0, 1.0).cross(&r_hat);
        if perp.norm() > 1e-9 { perp.normalize() } else { Vector3::new(0.0, 1.0, 0.0) }
    };

    // Target the FULL velocity vector: vis-viva speed in the tangential direction,
    // zero radial component. This drives both speed error AND eccentricity to zero.
    // The full dv vector = FRACTION × (v_target_vec − v_current) automatically
    // includes a radial component −FRACTION × v_radial that damps eccentricity.
    const FRACTION: f64 = 0.05;
    let v_target_vec = v_target * v_hat;
    let dv = FRACTION * (v_target_vec - v);
    if dv.norm() < 1e-15 {
        return None;
    }
    Some(dv)
}
