//! LunarTrajectories — Earth-Moon trajectory design library.
//!
//! All CRTBP/BCR4BP physics now live in the shared `crtbp` crate.
//! These re-exports keep every existing import path (`lunar_trajectories::crtbp::...`) working unchanged.

pub use ::crtbp::crtbp;
pub use ::crtbp::linalg;
pub use ::crtbp::propagator;
pub use ::crtbp::periodic_orbits;
pub use ::crtbp::manifolds;
pub use ::crtbp::transfers;
