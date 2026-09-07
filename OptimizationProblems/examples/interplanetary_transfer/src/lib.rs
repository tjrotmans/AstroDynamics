//! Solar sail Earth-Mars interplanetary transfer optimizer
//!
//! Uses real ephemeris (ANISE) for Earth and Mars positions, and SRP for propulsion.
//! Optimizes sail pointing angles (cone/clock) and time-of-flight simultaneously.

pub mod config;
pub mod control;
pub mod problem;

pub use config::MissionConfig;
pub use control::Angles;
pub use optimization::LoggingStrategy;
pub use problem::{TransferInput, TransferProblem};
