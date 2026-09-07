//! Generic quaternion kinematics for rigid-body attitude dynamics.
//!
//! Quaternion convention: q = [w, x, y, z] (scalar first).
//! Angular velocity ω is expressed in the spacecraft body frame.
//!
//! These functions carry no spacecraft-specific parameters (no inertia tensor,
//! no configuration constants).  Mission-specific wrappers live in the
//! application crate.

use nalgebra::{Vector3, Vector4};

/// Quaternion derivative  q̇ = ½ Ξ(q) ω.
///
/// Returns the time derivative of q = [w, x, y, z] given angular rate ω [rad/s]
/// in the body frame.
#[inline]
pub fn qdot(q: &Vector4<f64>, omega: &Vector3<f64>) -> Vector4<f64> {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let (p, qr, r)   = (omega[0], omega[1], omega[2]);

    0.5 * Vector4::new(
        -x * p  - y * qr - z * r,
         w * p  + y * r  - z * qr,
         w * qr - x * r  + z * p,
         w * r  + x * qr - y * p,
    )
}

/// Normalise quaternion to unit length.
///
/// Call after each integration step to prevent numerical drift from violating
/// the unit-norm constraint.
#[inline]
pub fn qnorm(q: &Vector4<f64>) -> Vector4<f64> {
    q / q.norm()
}

/// Quaternion conjugate [w, -x, -y, -z] — for a unit quaternion this is also
/// the inverse (the rotation undoing q).
#[inline]
pub fn quat_conjugate(q: &Vector4<f64>) -> Vector4<f64> {
    Vector4::new(q[0], -q[1], -q[2], -q[3])
}

/// Quaternion multiplication  q1 ⊗ q2, convention [w, x, y, z].
#[inline]
pub fn quat_multiply(q1: &Vector4<f64>, q2: &Vector4<f64>) -> Vector4<f64> {
    let (w1, x1, y1, z1) = (q1[0], q1[1], q1[2], q1[3]);
    let (w2, x2, y2, z2) = (q2[0], q2[1], q2[2], q2[3]);
    Vector4::new(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
}

/// Rotate vector `v` from body frame to inertial frame using unit quaternion
/// q = [w, x, y, z].
pub fn body_to_inertial(q: &Vector4<f64>, v: &Vector3<f64>) -> Vector3<f64> {
    let (w, qx, qy, qz) = (q[0], q[1], q[2], q[3]);
    let r11 = 1.0 - 2.0 * (qy * qy + qz * qz);
    let r12 = 2.0 * (qx * qy - w * qz);
    let r13 = 2.0 * (qx * qz + w * qy);
    let r21 = 2.0 * (qx * qy + w * qz);
    let r22 = 1.0 - 2.0 * (qx * qx + qz * qz);
    let r23 = 2.0 * (qy * qz - w * qx);
    let r31 = 2.0 * (qx * qz - w * qy);
    let r32 = 2.0 * (qy * qz + w * qx);
    let r33 = 1.0 - 2.0 * (qx * qx + qy * qy);
    Vector3::new(
        r11 * v[0] + r12 * v[1] + r13 * v[2],
        r21 * v[0] + r22 * v[1] + r23 * v[2],
        r31 * v[0] + r32 * v[1] + r33 * v[2],
    )
}

/// Rotate vector `v` from inertial frame to body frame (transpose of
/// [`body_to_inertial`]).
pub fn inertial_to_body(q: &Vector4<f64>, v: &Vector3<f64>) -> Vector3<f64> {
    let (w, qx, qy, qz) = (q[0], q[1], q[2], q[3]);
    let r11 = 1.0 - 2.0 * (qy * qy + qz * qz);
    let r12 = 2.0 * (qx * qy - w * qz);
    let r13 = 2.0 * (qx * qz + w * qy);
    let r21 = 2.0 * (qx * qy + w * qz);
    let r22 = 1.0 - 2.0 * (qx * qx + qz * qz);
    let r23 = 2.0 * (qy * qz - w * qx);
    let r31 = 2.0 * (qx * qz - w * qy);
    let r32 = 2.0 * (qy * qz + w * qx);
    let r33 = 1.0 - 2.0 * (qx * qx + qy * qy);
    // Transpose: column → row swap
    Vector3::new(
        r11 * v[0] + r21 * v[1] + r31 * v[2],
        r12 * v[0] + r22 * v[1] + r32 * v[2],
        r13 * v[0] + r23 * v[1] + r33 * v[2],
    )
}

/// Angular acceleration ω̇ for a rigid body with optional wheel momentum [rad/s²].
///
/// Implements Euler's equations with reaction wheel coupling:
///   I ω̇ = τ − ω × (I ω + H_w)
///
/// This is the generic form that takes the inertia tensor as a parameter — it does
/// not depend on any spacecraft configuration constants.  Mission-specific wrappers
/// that hard-wire the inertia tensor should call this function.
///
/// # Arguments
/// * `omega`   – Body angular rate [rad/s]
/// * `torque`  – Net external + control torque in body frame [N·m]
/// * `h_wheel` – Total reaction wheel angular momentum in body frame [N·m·s]
/// * `inertia` – Diagonal inertia tensor `[Ixx, Iyy, Izz]` [kg·m²]
pub fn omega_dot(
    omega:   &Vector3<f64>,
    torque:  &Vector3<f64>,
    h_wheel: &Vector3<f64>,
    inertia: &Vector3<f64>,
) -> Vector3<f64> {
    let i  = inertia;
    let iw = Vector3::new(i[0] * omega[0], i[1] * omega[1], i[2] * omega[2]);
    let cross = omega.cross(&(iw + h_wheel));
    Vector3::new(
        (torque[0] - cross[0]) / i[0],
        (torque[1] - cross[1]) / i[1],
        (torque[2] - cross[2]) / i[2],
    )
}

/// Rotation matrix (body axes as inertial-frame column vectors) → unit quaternion [w,x,y,z].
///
/// Uses the Shepperd method with 4-case branching for numerical stability — picks
/// the largest diagonal element to avoid dividing by a near-zero value.
///
/// # Convention
/// `col0`, `col1`, `col2` are the body x, y, z axes expressed in the inertial frame.
/// Together they form the columns of the body→inertial rotation matrix R where
/// `v_inertial = R * v_body`.
///
/// This convention is consistent with [`body_to_inertial`] in this module.
pub fn rot_to_quat(col0: Vector3<f64>, col1: Vector3<f64>, col2: Vector3<f64>) -> Vector4<f64> {
    let trace = col0[0] + col1[1] + col2[2];
    let q = if trace > 0.0 {
        let s = 0.5 / (trace + 1.0_f64).sqrt();
        Vector4::new(0.25/s, (col1[2]-col2[1])*s, (col2[0]-col0[2])*s, (col0[1]-col1[0])*s)
    } else if col0[0] > col1[1] && col0[0] > col2[2] {
        let s = 0.5 / (1.0 + col0[0] - col1[1] - col2[2]).sqrt();
        Vector4::new((col1[2]-col2[1])*s, 0.25/s, (col1[0]+col0[1])*s, (col2[0]+col0[2])*s)
    } else if col1[1] > col2[2] {
        let s = 0.5 / (1.0 - col0[0] + col1[1] - col2[2]).sqrt();
        Vector4::new((col2[0]-col0[2])*s, (col1[0]+col0[1])*s, 0.25/s, (col2[1]+col1[2])*s)
    } else {
        let s = 0.5 / (1.0 - col0[0] - col1[1] + col2[2]).sqrt();
        Vector4::new((col0[1]-col1[0])*s, (col2[0]+col0[2])*s, (col2[1]+col1[2])*s, 0.25/s)
    };
    q / q.norm()
}

/// Quaternion [w,x,y,z] that rotates body +x axis to point along `target_dir`.
///
/// Handles the degenerate cases (target exactly aligned or opposite to +x).
/// Returns identity when `target_dir` has near-zero magnitude.
pub fn align_x_with(target_dir: &Vector3<f64>) -> Vector4<f64> {
    let n = target_dir.norm();
    if n < 1e-12 { return Vector4::new(1.0, 0.0, 0.0, 0.0); }
    let t    = target_dir / n;
    let from = Vector3::new(1.0, 0.0, 0.0);
    let dot  = from.dot(&t).clamp(-1.0, 1.0);

    if dot > 0.9999 { return Vector4::new(1.0, 0.0, 0.0, 0.0); }
    if dot < -0.9999 {
        // 180° rotation about body +z
        return Vector4::new(0.0, 0.0, 0.0, 1.0);
    }
    // Half-angle shortcut: w = 1 + cos θ, xyz = sin θ · axis (unnormalised → normalise)
    let cross = from.cross(&t);
    let q = Vector4::new(1.0 + dot, cross[0], cross[1], cross[2]);
    q / q.norm()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `quat_conjugate` (added for Phase 13h's attitude MEKF) should invert
    /// any unit quaternion: `q ⊗ conj(q) = identity`.
    #[test]
    fn quat_conjugate_is_the_inverse_of_a_unit_quaternion() {
        let q = Vector4::new(0.5, 0.5, 0.5, 0.5); // already unit norm
        let identity = quat_multiply(&q, &quat_conjugate(&q));
        assert!((identity - Vector4::new(1.0, 0.0, 0.0, 0.0)).norm() < 1e-12, "got {identity:?}");
    }
}
