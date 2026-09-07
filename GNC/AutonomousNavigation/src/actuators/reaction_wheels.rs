//! Reaction wheel actuator — 4-wheel tetrahedral pyramid.
//!
//! All generic wheel cluster logic lives in `attitude_control::ReactionWheelCluster`.
//! This module builds the mission-specific cluster from config constants and
//! re-exports the free functions for backward compatibility with callers.

use nalgebra::Vector3;
use attitude_control::ReactionWheelCluster;
use crate::config::{WHEEL_INERTIA_KGM2, WHEEL_MAX_SPEED_RADS, WHEEL_MAX_TORQUE_NM,
                    WHEEL_DESAT_FRACTION};

/// Build the spacecraft's reaction wheel cluster from mission config constants.
pub fn build_cluster() -> ReactionWheelCluster {
    ReactionWheelCluster::four_wheel_pyramid(
        WHEEL_INERTIA_KGM2,
        WHEEL_MAX_SPEED_RADS,
        WHEEL_MAX_TORQUE_NM,
        WHEEL_DESAT_FRACTION,
    )
}

// Re-export cluster method signatures as module-level free functions so existing
// callers in mod.rs / bin/*.rs don't need updating.

/// Total wheel angular momentum in body frame [N·m·s].
pub fn total_momentum(speeds: &[f64; 4]) -> Vector3<f64> {
    build_cluster().total_momentum(speeds)
}

/// Allocate commanded torque to wheel motor torques [N·m].
pub fn allocate(tau_cmd: &Vector3<f64>) -> [f64; 4] {
    build_cluster().allocate(tau_cmd)
}

/// Wheel speed derivatives [rad/s²].
pub fn speed_dots(tau_motor: &[f64; 4]) -> [f64; 4] {
    build_cluster().speed_dots(tau_motor)
}

/// Actual torque on spacecraft body from wheels [N·m].
pub fn body_torque(tau_motor: &[f64; 4]) -> Vector3<f64> {
    build_cluster().body_torque(tau_motor)
}

/// Check whether desaturation is needed.
pub fn needs_desat(speeds: &[f64; 4]) -> bool {
    build_cluster().needs_desat(speeds)
}

/// Desaturation torque direction for RCS [body frame, N·m].
pub fn desat_torque(speeds: &[f64; 4]) -> Option<Vector3<f64>> {
    build_cluster().desat_torque(speeds)
}
