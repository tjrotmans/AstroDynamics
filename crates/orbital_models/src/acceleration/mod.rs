//! Acceleration models (forces and perturbations)
//!
//! This module provides reusable acceleration calculation functions for spacecraft dynamics.
//! Accelerations are returned in m/s² in the inertial frame.

pub mod drag;
pub mod gravity;
pub mod srp;

pub use drag::DragModel;
pub use gravity::GravityModel;
pub use srp::{SolarPressureModel, SolarSailModel, Plate,
              pressure_at, cannonball,
              flat_plate_force_body, flat_plate_force_per_plate, flat_plate_torque_body, flat_plate_accel};
