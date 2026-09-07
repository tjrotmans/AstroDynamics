//! Shared mathematical utilities for orbital mechanics
//!
//! Provides the `Vector` wrapper for numerical integration compatibility
//! and cyclic angle normalization.

#![expect(incomplete_features)]
#![feature(generic_const_exprs)]

pub mod vector;
pub mod normalize;
pub mod lambert;
pub mod kepler;
pub mod hermite_track;

pub use vector::Vector;
pub use normalize::normalize;
pub use kepler::{eccentricity, orbital_period_s, propagate_kepler, semi_major_axis_m};
