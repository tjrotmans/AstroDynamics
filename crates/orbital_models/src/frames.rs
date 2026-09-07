//! Reference frame types for sail attitude parameterization
//!
//! Each frame converts two attitude angles + spacecraft state into a sail normal
//! vector in the inertial frame. Different problems pick the frame that best
//! matches their physics (e.g. drag sails align with velocity, solar sails with sun).
//!
//! The `sun_position` parameter passed to every frame is the position of the Sun
//! in the same inertial frame as `position`.  Pass the actual ephemeris value for
//! ECI problems; pass `SVector::zeros()` for heliocentric problems where the Sun
//! is at the coordinate origin.

use nalgebra::{Matrix3, SVector, Vector3};

/// Trait for attitude reference frames
///
/// Implementors define how two control angles map to a sail normal direction
/// in the inertial frame, given the spacecraft state and the Sun position.
///
/// # Sun position convention
/// `sun_position` must be expressed in the same inertial frame as `position`.
/// - For ECI frames: pass the Sun's ECI position (e.g. from an ephemeris query).
/// - For heliocentric frames: the Sun is at the origin — pass `SVector::zeros()`.
pub trait AttitudeFrame {
    /// Compute sail normal unit vector in the inertial frame.
    ///
    /// # Arguments
    /// * `angle1`        – First attitude angle (frame-specific meaning)
    /// * `angle2`        – Second attitude angle (frame-specific meaning)
    /// * `position`      – Spacecraft position in inertial frame [m]
    /// * `velocity`      – Spacecraft velocity in inertial frame [m/s]
    /// * `sun_position`  – Sun position in the same inertial frame [m]
    ///
    /// # Returns
    /// Unit normal vector of the sail in the inertial frame.
    fn sail_normal_inertial(
        angle1: f64,
        angle2: f64,
        position: &SVector<f64, 3>,
        velocity: &SVector<f64, 3>,
        sun_position: &SVector<f64, 3>,
    ) -> SVector<f64, 3>;
}

/// Velocity-aligned reference frame (LVLH / RSW)
///
/// Used for drag sail problems where the sail orientation is defined
/// relative to the velocity direction.  The Sun position is not used.
///
/// - `angle1` = elevation (chi): Nadir = 0, Zenith = π
/// - `angle2` = direction (ksi): Along velocity = 0, left = π/2, right = -π/2
///
/// Reference: <https://www.sciencedirect.com/science/article/pii/S1270963822003406>
pub struct VelocityFrame;

impl AttitudeFrame for VelocityFrame {
    fn sail_normal_inertial(
        angle1: f64,
        angle2: f64,
        position: &SVector<f64, 3>,
        velocity: &SVector<f64, 3>,
        _sun_position: &SVector<f64, 3>,
    ) -> SVector<f64, 3> {
        let chi = angle1;  // elevation
        let ksi = angle2;  // direction

        let r_hat = position / position.norm();
        let v_hat = velocity / velocity.norm();
        let h_hat = r_hat.cross(&v_hat);

        // Velocity frame axes
        let x_hat = v_hat;
        let y_hat = h_hat.cross(&v_hat);
        let z_hat = h_hat;

        // Intermediate angle
        let zeta = (chi.cos() * ksi.cos()).acos();

        // Normal vector in velocity frame
        let n_v = Vector3::new(zeta.cos(), chi.cos() * ksi.sin(), chi.sin());
        let n_v_hat = n_v / n_v.norm();

        // Transform to inertial frame
        let transform = Matrix3::new(
            x_hat[0], y_hat[0], z_hat[0],
            x_hat[1], y_hat[1], z_hat[1],
            x_hat[2], y_hat[2], z_hat[2],
        );

        transform * n_v_hat
    }
}

/// Sun-pointing reference frame (cone-clock parameterization)
///
/// Usable for both Earth-centred (ECI) and heliocentric frames — pass the
/// appropriate `sun_position` at the call site:
///
/// | Frame        | `sun_position`                          |
/// |-------------|------------------------------------------|
/// | ECI          | Sun's ECI position from the ephemeris   |
/// | Heliocentric | `SVector::zeros()` (Sun at origin)      |
///
/// - `angle1` = cone (α): angle between sail normal and Sun→SC direction [0, π/2]
///   - α = 0: sail faces sun directly → maximum SRP, force away from sun
///   - α = π/2: edge-on → zero SRP
/// - `angle2` = clock (δ): rotation around the Sun→SC axis [-π, π]
pub struct SunPointingFrame;

impl AttitudeFrame for SunPointingFrame {
    #[allow(non_snake_case)]
    fn sail_normal_inertial(
        angle1: f64,
        angle2: f64,
        position: &SVector<f64, 3>,
        _velocity: &SVector<f64, 3>,
        sun_position: &SVector<f64, 3>,
    ) -> SVector<f64, 3> {
        let alpha = angle1; // cone
        let delta = angle2; // clock

        // Vector from Sun to spacecraft (s_hat points away from Sun)
        let s_vec = position - sun_position;
        let s_hat = s_vec / s_vec.norm();

        // Build an orthonormal basis around s_hat.
        // Use the inertial Z axis as the reference; fall back to X when s_hat ∥ Z.
        let z_inertial = SVector::<f64, 3>::new(0.0, 0.0, 1.0);
        let perp = z_inertial.cross(&s_hat);
        let theta_hat = if perp.norm() > 1e-10 {
            perp / perp.norm()
        } else {
            let x_inertial = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
            let p = x_inertial.cross(&s_hat);
            p / p.norm()
        };
        let phi_hat = s_hat.cross(&theta_hat);

        // Sail normal in the sun-pointing basis: [cos(α), sin(α)sin(δ), sin(α)cos(δ)]
        let n_sun_frame = Vector3::new(
            alpha.cos(),
            alpha.sin() * delta.sin(),
            alpha.sin() * delta.cos(),
        );

        // Rotation matrix: columns are s_hat, theta_hat, phi_hat
        let R_sun_to_inertial = Matrix3::new(
            s_hat[0], theta_hat[0], phi_hat[0],
            s_hat[1], theta_hat[1], phi_hat[1],
            s_hat[2], theta_hat[2], phi_hat[2],
        );

        R_sun_to_inertial * n_sun_frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OrbitalElements;
    use crate::constants::SUN_POSITION;

    fn test_state() -> (SVector<f64, 3>, SVector<f64, 3>) {
        let state = OrbitalElements {
            a: 8371000.0,
            e: 0.0,
            i: 0.0,
            w: 0.0,
            o: 0.0,
            nu: 0.0,
        };
        let cart = state.as_cartesian();
        let pos: SVector<f64, 3> = cart.position().into();
        let vel: SVector<f64, 3> = cart.velocity().into();
        (pos, vel)
    }

    fn eci_sun() -> SVector<f64, 3> {
        SVector::<f64, 3>::from_column_slice(&SUN_POSITION)
    }

    #[test]
    fn velocity_frame_normal_is_unit() {
        let (pos, vel) = test_state();
        let n = VelocityFrame::sail_normal_inertial(0.5, 0.3, &pos, &vel, &SVector::zeros());
        assert!((n.norm() - 1.0).abs() < 1e-10, "Normal should be unit vector, got norm={}", n.norm());
    }

    #[test]
    fn sun_frame_normal_is_unit() {
        let (pos, vel) = test_state();
        let n = SunPointingFrame::sail_normal_inertial(0.5, 0.3, &pos, &vel, &eci_sun());
        assert!((n.norm() - 1.0).abs() < 1e-10, "Normal should be unit vector, got norm={}", n.norm());
    }

    #[test]
    fn sun_frame_zero_cone_points_at_sun() {
        let (pos, vel) = test_state();
        // With cone=0, normal should align with s_hat (Sun→SC direction)
        let sun = eci_sun();
        let n = SunPointingFrame::sail_normal_inertial(0.0, 0.0, &pos, &vel, &sun);
        let s_hat = (pos - sun).normalize();
        let dot = n.dot(&s_hat);
        assert!((dot - 1.0).abs() < 1e-10, "At cone=0, normal should point along s_hat, dot={}", dot);
    }

    #[test]
    fn sun_frame_heliocentric_normal_is_unit() {
        // Sun at origin (heliocentric use case)
        let pos = SVector::<f64, 3>::new(1.496e11, 0.0, 0.0); // 1 AU along X
        let vel = SVector::<f64, 3>::new(0.0, 29_780.0, 0.0);
        let n = SunPointingFrame::sail_normal_inertial(0.5, 0.3, &pos, &vel, &SVector::zeros());
        assert!((n.norm() - 1.0).abs() < 1e-10, "Normal should be unit vector, got norm={}", n.norm());
    }

    #[test]
    fn sun_frame_heliocentric_zero_cone_points_away_from_sun() {
        // At cone=0 in heliocentric frame: normal should point along s_hat = r_hat (away from Sun)
        let pos = SVector::<f64, 3>::new(1.496e11, 0.0, 0.0);
        let vel = SVector::<f64, 3>::new(0.0, 29_780.0, 0.0);
        let n = SunPointingFrame::sail_normal_inertial(0.0, 0.0, &pos, &vel, &SVector::zeros());
        let s_hat = pos / pos.norm();
        let dot = n.dot(&s_hat);
        assert!((dot - 1.0).abs() < 1e-10,
            "At cone=0 (heliocentric), normal should point away from Sun, dot={}", dot);
    }

    #[test]
    fn velocity_frame_zero_angles_aligns_with_velocity() {
        let (pos, vel) = test_state();
        // With elevation=0 and direction=0, normal should align with velocity
        let n = VelocityFrame::sail_normal_inertial(0.0, 0.0, &pos, &vel, &SVector::zeros());
        let v_hat = vel / vel.norm();
        let dot = n.dot(&v_hat);
        assert!((dot - 1.0).abs() < 1e-10,
            "At elevation=0, direction=0, normal should point along velocity, dot={}", dot);
    }
}
