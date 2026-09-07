//! Attitude control actuator models — reaction wheels and RCS.
//!
//! All models are parameterized (no configuration constants imported); mission-specific
//! parameter values live in the application crate's `config.rs`.
//!
//! | Module             | Key exports                                   |
//! |--------------------|-----------------------------------------------|
//! | `reaction_wheels`  | `ReactionWheelCluster`, `four_wheel_pyramid`  |
//! | `rcs`              | `Thruster`, `PdGains`, `pd_torque`,           |
//! |                    | `rcs_thrusters`, `thruster_selection`, etc.   |

pub mod reaction_wheels;
pub mod rcs;

pub use reaction_wheels::ReactionWheelCluster;
pub use rcs::{Thruster, PdGains, pd_torque,
              rcs_thrusters, translation_thrusters,
              thruster_selection, translation_thrust_select,
              select_firing_thrusters};
