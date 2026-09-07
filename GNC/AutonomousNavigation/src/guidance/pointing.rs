//! Spacecraft pointing modes — compute the desired attitude quaternion.
//!
//! All generic pointing math (nadir+orbit-normal, velocity-aligned) lives in
//! `orbital_models::guidance::pointing` and is re-used here unchanged.
//! This file adds the mission-specific `PointingMode` enum and dispatches.

use nalgebra::{Vector3, Vector4};
use orbital_models::guidance::{nadir_orbit_normal_quat, velocity_aligned_quat};

/// Pointing mode for the spacecraft attitude controller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointingMode {
    /// Camera (+x body) points at Bennu centre: `−r̂_Hill`.
    Nadir,
    /// Camera (+x body) aligned with orbital velocity vector: `v̂_Hill`.
    VelocityAligned,
}

impl Default for PointingMode {
    fn default() -> Self { PointingMode::Nadir }
}

/// Compute the desired quaternion for the current pointing mode.
///
/// `r` = S/C position in Hill frame [m], `v` = Hill velocity [m/s].
pub fn desired_quaternion(mode: PointingMode, r: &Vector3<f64>, v: &Vector3<f64>)
    -> Vector4<f64>
{
    match mode {
        PointingMode::Nadir          => nadir_orbit_normal_quat(r, v),
        PointingMode::VelocityAligned => velocity_aligned_quat(v),
    }
}
