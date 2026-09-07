//! Hardware catalog — pure data specs for actuators and sensors, with named
//! constructors for representative performance classes.
//!
//! Mirrors `body_models::TargetBody`'s pattern: plain structs built via named
//! constructors (`ReactionWheelSpec::small()`, …), no trait objects. The GNC
//! design stage (`MissionPlanner/src/gnc_design.rs`) selects the smallest/
//! coarsest catalog entry that meets a sizing requirement; the sim engine
//! (Phase 4) will consume the same specs for truth-model actuator/sensor
//! limits.
//!
//! Values are representative performance-class figures (cited to general
//! engineering practice, e.g. Wertz & Larson, *Space Mission Analysis and
//! Design*), not a specific flight unit's datasheet — sized to be "reasonable
//! for the class", not asserted as a measured spec.
//!
//! `LaunchVehicleSpec` is the one exception to that pattern: launch vehicle
//! C3-vs-mass performance is genuinely vehicle-specific, not approximable
//! from general practice, so every data point is a real, individually
//! citable flown mission rather than a representative class — see
//! `launch_vehicle.rs` for why and its citations.

mod reaction_wheel;
mod thruster;
mod sensor;
mod launch_vehicle;
mod panel;

pub use reaction_wheel::ReactionWheelSpec;
pub use thruster::ThrusterSpec;
pub use sensor::{
    DsnLinkSpec, GyroSpec, ImuSpec, LandmarkSensorSpec, LidarSpec, OpNavCameraSpec,
    StarTrackerSpec,
};
pub use launch_vehicle::LaunchVehicleSpec;
pub use panel::PanelSpec;
