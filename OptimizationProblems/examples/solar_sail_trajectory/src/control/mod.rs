//! Control inputs and steering laws.

pub mod steering;

pub use orbital_models::{Angles, normalize_cone, normalize_clock};
pub use orbital_math::normalize;
pub use steering::LocallyOptimalSMA;
