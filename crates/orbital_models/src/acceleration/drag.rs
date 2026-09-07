//! Drag acceleration model for solar/drag sails
//!
//! Generic over `AttitudeFrame` — the same model works with any reference frame
//! (velocity-aligned, sun-pointing, etc.) by delegating the sail normal calculation
//! to the frame implementation.

use nalgebra::SVector;
use std::marker::PhantomData;
use crate::environment::AtmosphereModel;
use crate::constants::EARTH_RADIUS;
use crate::frames::AttitudeFrame;

/// Drag acceleration model, parameterized by attitude reference frame
pub struct DragModel<F: AttitudeFrame>(PhantomData<F>);

impl<F: AttitudeFrame> DragModel<F> {
    /// Compute aerodynamic drag acceleration vector
    ///
    /// # Arguments
    /// * `position` - Current position vector [x, y, z] in meters (inertial frame)
    /// * `velocity` - Current velocity vector [vx, vy, vz] in m/s (inertial frame)
    /// * `sail_area` - Solar sail area in m²
    /// * `mass` - Spacecraft mass in kg
    /// * `drag_coefficient` - Dimensionless drag coefficient
    /// * `angle1` - First attitude angle (frame-specific meaning)
    /// * `angle2` - Second attitude angle (frame-specific meaning)
    ///
    /// # Returns
    /// Acceleration vector [ax, ay, az] in m/s² (inertial frame)
    pub fn compute(
        position: &SVector<f64, 3>,
        velocity: &SVector<f64, 3>,
        sail_area: f64,
        mass: f64,
        drag_coefficient: f64,
        angle1: f64,
        angle2: f64,
        sun_position: &SVector<f64, 3>,
    ) -> SVector<f64, 3> {
        let v = *velocity;
        let v_hat = v / v.norm();

        // Get sail normal in inertial frame from the chosen reference frame
        let n_inertial = F::sail_normal_inertial(angle1, angle2, position, velocity, sun_position);

        let dot_product = n_inertial.dot(&v_hat).abs();

        let altitude = position.norm() - EARTH_RADIUS;
        let density = AtmosphereModel::density(altitude);
        let drag_force = v.norm() * (0.5 * density * sail_area * drag_coefficient) * dot_product;

        -v * (drag_force / mass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OrbitalElements;
    use crate::frames::{VelocityFrame, SunPointingFrame};

    #[test]
    fn drag_velocity_frame_at_2000km() {
        let state = OrbitalElements {
            a: 8371000.0,
            e: 0.0,
            i: 0.0,
            w: 0.0,
            o: 0.0,
            nu: 0.0,
        };
        let sail_area = 300.0;
        let mass = 10.0;
        let drag_coefficient = 2.0;

        let cartesian = state.as_cartesian();
        let pos: SVector<f64, 3> = cartesian.position().into();
        let vel: SVector<f64, 3> = cartesian.velocity().into();

        let accel = DragModel::<VelocityFrame>::compute(
            &pos, &vel, sail_area, mass, drag_coefficient, 0.0, 0.0, &SVector::zeros(),
        );

        assert!(!accel[0].is_nan(), "Drag X is NaN");
        assert!(!accel[1].is_nan(), "Drag Y is NaN");
        assert!(!accel[2].is_nan(), "Drag Z is NaN");

        // At 2000 km, density is essentially zero
        assert!(accel.norm() < 1e-6, "Drag too large at 2000 km: {}", accel.norm());
    }

    #[test]
    fn drag_sun_frame_at_2000km() {
        let state = OrbitalElements {
            a: 8371000.0,
            e: 0.0,
            i: 0.0,
            w: 0.0,
            o: 0.0,
            nu: 0.0,
        };
        let sail_area = 300.0;
        let mass = 10.0;
        let drag_coefficient = 2.0;

        let cartesian = state.as_cartesian();
        let pos: SVector<f64, 3> = cartesian.position().into();
        let vel: SVector<f64, 3> = cartesian.velocity().into();

        use crate::constants::SUN_POSITION;
        let sun = SVector::<f64, 3>::from_column_slice(&SUN_POSITION);
        let accel = DragModel::<SunPointingFrame>::compute(
            &pos, &vel, sail_area, mass, drag_coefficient, 0.0, 0.0, &sun,
        );

        assert!(!accel[0].is_nan(), "Drag X is NaN");
        assert!(!accel[1].is_nan(), "Drag Y is NaN");
        assert!(!accel[2].is_nan(), "Drag Z is NaN");

        assert!(accel.norm() < 1e-6, "Drag too large at 2000 km: {}", accel.norm());
    }
}
