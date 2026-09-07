//! Runtime-configurable target body definitions for generic mission planning.
//!
//! Bodies are built either from a TOML mission config (via `MissionPlanner`) or
//! via named constructors for well-known bodies (Bennu, Earth, Moon, Mars, …).
//! This crate holds only type definitions; force computation stays in
//! `orbital_models::acceleration` and will call into these types from the sim
//! engine (Phase 4).

mod atmosphere;
mod body;
mod gravity;

pub use atmosphere::AtmosphereModel;
pub use body::TargetBody;
pub use gravity::GravityModel;
