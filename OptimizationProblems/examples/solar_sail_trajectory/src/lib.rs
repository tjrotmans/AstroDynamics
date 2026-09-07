//! Solar sail trajectory optimization library

#![expect(incomplete_features)]
#![feature(generic_const_exprs)]

pub mod config;
pub mod control;
pub mod problem;

pub use config::MissionConfig;
pub use optimization::{CoolingSchedule, InterpolationStrategy};
pub use problem::ControlMode;
pub use control::LocallyOptimalSMA;

/// Frame-specific type aliases for solar sail acceleration models
pub type SolarPressureModel = orbital_models::SolarPressureModel<orbital_models::SunPointingFrame>;
pub type DragModel = orbital_models::DragModel<orbital_models::SunPointingFrame>;
