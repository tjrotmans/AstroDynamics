//! Autonomous navigation simulation for a near-Earth asteroid (Bennu) mission.
//!
//! # Module map
//!
//! | Module        | Responsibility                                                   |
//! |---------------|------------------------------------------------------------------|
//! | `config`      | Simulation parameters (spacecraft, Bennu, sensors, EKF tuning) |
//! | `dynamics`    | Truth-model equations of motion (6DOF: translation + rotation) |
//! | `sensors`     | Measurement simulation — star tracker, OpNav camera, IMU       |
//! | `navigation`  | Extended Kalman Filter — predict / update / STM                 |

pub mod config;
pub mod dynamics;
pub mod sensors;
pub mod navigation;
pub mod actuators;
pub mod guidance;
pub mod bennu_ephem;
pub mod proximity_init;
