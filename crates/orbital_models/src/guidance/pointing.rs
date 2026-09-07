//! Generic spacecraft pointing functions — compute desired attitude quaternion.
//!
//! These functions carry no mission-specific configuration.  They produce attitude
//! quaternions from orbital state vectors and are reusable across any mission.
//!
//! Mission-specific pointing modes (enum, mode dispatch, fallback behaviour) live
//! in the application crate (e.g. `GNC/AutonomousNavigation/src/guidance/pointing.rs`).

use nalgebra::{Matrix3, Vector3, Vector4};
use crate::attitude::{align_x_with, rot_to_quat};

/// Nadir-pointing quaternion with orbit-normal secondary constraint [w,x,y,z].
///
/// - Body +x → −r̂  (primary: toward the central body, i.e. camera boresight)
/// - Body +z → ĥ = r×v/|r×v|  (secondary: orbit normal)
/// - Body +y → body_z × body_x  (derived, along-track)
///
/// This fully constrains the attitude, eliminating the 360°/orbit roll artifact
/// that pure `align_x_with(-r̂)` produces from its unconstrained secondary axis.
///
/// Falls back gracefully:
/// - If orbit normal is degenerate (|r×v| < ε), falls back to `align_x_with`.
/// - If `|r| < ε`, returns identity.
pub fn nadir_orbit_normal_quat(r: &Vector3<f64>, v: &Vector3<f64>) -> Vector4<f64> {
    let r_norm = r.norm();
    if r_norm < 1e-12 { return Vector4::new(1.0, 0.0, 0.0, 0.0); }
    let x_b = -r / r_norm;

    let h = r.cross(v);
    let h_norm = h.norm();
    if h_norm < 1e-12 { return align_x_with(&x_b); }
    let h_hat = h / h_norm;

    // Orthogonalise orbit normal against nadir (near-zero correction for circular orbits)
    let z_raw  = h_hat - h_hat.dot(&x_b) * x_b;
    let z_norm = z_raw.norm();
    if z_norm < 1e-10 { return align_x_with(&x_b); }
    let z_b = z_raw / z_norm;
    let y_b = z_b.cross(&x_b);

    rot_to_quat(x_b, y_b, z_b)
}

/// Velocity-aligned quaternion [w,x,y,z].
///
/// Aligns body +x with the velocity direction.  The secondary axes are unconstrained
/// (minimum-rotation solution from `align_x_with`).
///
/// Falls back to identity when |v| < ε.
pub fn velocity_aligned_quat(v: &Vector3<f64>) -> Vector4<f64> {
    let v_norm = v.norm();
    if v_norm < 1e-12 { return Vector4::new(1.0, 0.0, 0.0, 0.0); }
    align_x_with(&(v / v_norm))
}

/// Pick an arbitrary unit vector perpendicular to `v` (`v` assumed unit).
/// Used to complete a triad when a real secondary constraint is unavailable
/// or degenerate — the "roll about the primary axis is free" case (see
/// `docs/MP/MANUAL.md` §9.4). Any consistent choice is as good as any
/// other since nothing constrains that remaining DOF.
fn arbitrary_perp(v: &Vector3<f64>) -> Vector3<f64> {
    let alt = if v.x.abs() < 0.9 { Vector3::new(1.0, 0.0, 0.0) } else { Vector3::new(0.0, 1.0, 0.0) };
    let raw = alt - alt.dot(v) * v;
    let n = raw.norm();
    if n < 1e-9 {
        // v was already almost exactly `alt` (only possible if the 0.9
        // threshold above still picked a near-parallel alt) — fall back to
        // the other candidate axis, guaranteed non-parallel.
        let alt2 = Vector3::new(0.0, 0.0, 1.0);
        (alt2 - alt2.dot(v) * v).normalize()
    } else {
        raw / n
    }
}

/// General two-vector (TRIAD) attitude solve [w,x,y,z] — the generalization
/// of [`nadir_orbit_normal_quat`] from one hardcoded `(body +x, body +z)`
/// pair to arbitrary body vectors, per `docs/MP/MANUAL.md` §9.4 (the
/// physics groundwork for the Phase 03 attitude
/// commander). Returns the quaternion `q` (body → inertial) such that:
///
/// - `R(q) · b1̂ = t1̂` EXACTLY (primary constraint — consumes 2 of the 3
///   rotational DOF)
/// - `R(q) · b2̂` is as close as possible to `t2̂`, using the one remaining
///   DOF (secondary constraint — exact only when `∠(b1̂,b2̂) == ∠(t1̂,t2̂)`,
///   which is generically not the case; see the module-level physics note)
///
/// Degenerate inputs (`b2` parallel to `b1`, or `t2` parallel to `t1`) fall
/// back to leaving the secondary/roll DOF free via [`arbitrary_perp`] —
/// same philosophy as [`nadir_orbit_normal_quat`]'s own degenerate fallback
/// to `align_x_with`.
pub fn triad_quat(b1: &Vector3<f64>, t1: &Vector3<f64>, b2: &Vector3<f64>, t2: &Vector3<f64>) -> Vector4<f64> {
    let (Some(eb1), Some(ei1)) = (b1.try_normalize(1e-12), t1.try_normalize(1e-12)) else {
        return Vector4::new(1.0, 0.0, 0.0, 0.0);
    };

    let eb3_raw = b2 - b2.dot(&eb1) * eb1;
    let eb3 = eb3_raw.try_normalize(1e-10).unwrap_or_else(|| arbitrary_perp(&eb1));
    let eb2 = eb3.cross(&eb1);

    let ei3_raw = t2 - t2.dot(&ei1) * ei1;
    let ei3 = ei3_raw.try_normalize(1e-10).unwrap_or_else(|| arbitrary_perp(&ei1));
    let ei2 = ei3.cross(&ei1);

    // R (body -> inertial) maps each body-triad column to the corresponding
    // inertial-triad column: R * B = T with B, T orthonormal, so
    // R = T * B^{-1} = T * B^T.
    let b_mat = Matrix3::from_columns(&[eb1, eb2, eb3]);
    let t_mat = Matrix3::from_columns(&[ei1, ei2, ei3]);
    let r = t_mat * b_mat.transpose();

    rot_to_quat(r.column(0).into(), r.column(1).into(), r.column(2).into())
}

/// Single-vector alignment for an ARBITRARY body axis [w,x,y,z] — the
/// generalization of [`align_x_with`] (which is hardcoded to body +x) to
/// any body vector. Roll about the aligned axis is left free (via
/// [`arbitrary_perp`]), same semantics as `align_x_with`.
pub fn align_vector_with(body_vec: &Vector3<f64>, target_dir: &Vector3<f64>) -> Vector4<f64> {
    let (Some(eb1), Some(ei1)) = (body_vec.try_normalize(1e-12), target_dir.try_normalize(1e-12)) else {
        return Vector4::new(1.0, 0.0, 0.0, 0.0);
    };
    let eb3 = arbitrary_perp(&eb1);
    let eb2 = eb3.cross(&eb1);
    let ei3 = arbitrary_perp(&ei1);
    let ei2 = ei3.cross(&ei1);

    let b_mat = Matrix3::from_columns(&[eb1, eb2, eb3]);
    let t_mat = Matrix3::from_columns(&[ei1, ei2, ei3]);
    let r = t_mat * b_mat.transpose();

    rot_to_quat(r.column(0).into(), r.column(1).into(), r.column(2).into())
}

#[cfg(test)]
mod triad_tests {
    use super::*;

    fn quat_rotate(q: &Vector4<f64>, v: &Vector3<f64>) -> Vector3<f64> {
        crate::attitude::body_to_inertial(q, v)
    }

    #[test]
    fn triad_satisfies_primary_exactly() {
        let b1 = Vector3::new(1.0, 0.0, 0.0);
        let t1 = Vector3::new(0.0, 1.0, 0.0).normalize();
        let b2 = Vector3::new(0.0, 0.0, 1.0);
        let t2 = Vector3::new(1.0, 0.0, 0.0).normalize();
        let q = triad_quat(&b1, &t1, &b2, &t2);
        let achieved = quat_rotate(&q, &b1);
        assert!((achieved - t1).norm() < 1e-9, "primary should be exact: {achieved:?} vs {t1:?}");
    }

    /// Matches `nadir_orbit_normal_quat`'s own case exactly (body +x
    /// primary, body +z secondary) — confirms the general solver reproduces
    /// the existing hardcoded function's result for the same inputs.
    #[test]
    fn triad_matches_nadir_orbit_normal_quat_for_the_same_pair() {
        let r = Vector3::new(7.0e6, 0.0, 0.0);
        let v = Vector3::new(0.0, 7.5e3, 1.0e3);
        let expected = nadir_orbit_normal_quat(&r, &v);

        let x_b = -r.normalize();
        let h_hat = r.cross(&v).normalize();
        let got = triad_quat(&Vector3::new(1.0, 0.0, 0.0), &x_b, &Vector3::new(0.0, 0.0, 1.0), &h_hat);

        // Quaternions can differ by an overall sign and still represent the
        // same rotation.
        let same = (expected - got).norm() < 1e-6 || (expected + got).norm() < 1e-6;
        assert!(same, "expected {expected:?}, got {got:?}");
    }

    #[test]
    fn triad_secondary_is_best_effort_when_incompatible() {
        // Body vectors are exactly 90 deg apart; targets are exactly 45 deg
        // apart -- incompatible, so the secondary cannot be satisfied
        // exactly. Confirm the primary still IS exact and the secondary
        // error is small but nonzero (not silently "successful").
        let b1 = Vector3::new(1.0, 0.0, 0.0);
        let b2 = Vector3::new(0.0, 1.0, 0.0);
        let t1 = Vector3::new(1.0, 0.0, 0.0);
        let angle = 45.0_f64.to_radians();
        let t2 = Vector3::new(angle.cos(), angle.sin(), 0.0);

        let q = triad_quat(&b1, &t1, &b2, &t2);
        let achieved_primary = quat_rotate(&q, &b1);
        assert!((achieved_primary - t1).norm() < 1e-9);

        let achieved_secondary = quat_rotate(&q, &b2);
        let err_deg = achieved_secondary.dot(&t2).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(err_deg > 1.0, "expected a real, nonzero secondary residual, got {err_deg} deg");
    }

    #[test]
    fn triad_degenerate_secondary_falls_back_to_free_roll() {
        // b2 parallel to b1 -- no real secondary constraint; should still
        // produce a valid, primary-exact quaternion rather than degenerating.
        let b1 = Vector3::new(1.0, 0.0, 0.0);
        let b2 = Vector3::new(2.0, 0.0, 0.0);
        let t1 = Vector3::new(0.0, 0.0, 1.0);
        let t2 = Vector3::new(0.0, 1.0, 0.0);
        let q = triad_quat(&b1, &t1, &b2, &t2);
        let achieved = quat_rotate(&q, &b1);
        assert!((achieved - t1).norm() < 1e-9);
    }

    /// `align_vector_with` and `align_x_with` are two DIFFERENT valid
    /// solutions to the same underconstrained problem (both correctly align
    /// body +x with the target; each picks its own convention for the free
    /// roll DOF via a different construction) -- they are not expected to
    /// produce the identical quaternion. What must hold for any solver of
    /// this problem is the one real invariant: the primary alignment is
    /// achieved exactly.
    #[test]
    fn align_vector_with_achieves_exact_alignment_for_body_x() {
        let target = Vector3::new(0.3, 0.7, 0.2).normalize();
        let got = align_vector_with(&Vector3::new(1.0, 0.0, 0.0), &target);
        let achieved = quat_rotate(&got, &Vector3::new(1.0, 0.0, 0.0));
        assert!((achieved - target).norm() < 1e-9, "expected {achieved:?} to match target {target:?}");
    }
}
