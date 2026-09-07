//\! Shared orbital mechanics models
//\!
//\! Provides reusable physics models for orbital mechanics simulations:
//\! - Physical constants (Earth, Sun, SRP)
//\! - Orbital elements and state vector conversions
//\! - Sun position utilities
//\! - Environment models (atmosphere)
//\! - Acceleration models (gravity, drag, SRP)
//\! - Reference frames (velocity, sun-pointing)

pub mod angles;
pub mod propulsion;
pub mod constants;
pub mod orbital;
pub mod environment;
pub mod acceleration;
pub mod frames;
pub mod attitude;
pub mod torque;
pub mod guidance;

pub use orbital::{OrbitalElements, StateVector};
pub use orbital::{sun_vector_from_position, sun_direction_from_position};
pub use environment::AtmosphereModel;
pub use acceleration::{GravityModel, DragModel, SolarPressureModel, SolarSailModel,
                       Plate, pressure_at, cannonball,
                       flat_plate_force_body, flat_plate_force_per_plate, flat_plate_torque_body, flat_plate_accel};
pub use frames::{AttitudeFrame, VelocityFrame, SunPointingFrame};
pub use angles::{Angles, normalize_cone, normalize_clock};
pub use propulsion::FiniteBurn;
pub use torque::{gravity_gradient, gravity_gradient_max};
pub use guidance::{nadir_orbit_normal_quat, velocity_aligned_quat};
