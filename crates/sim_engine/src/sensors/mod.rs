//! Generic sensor measurement models. Each function takes its noise parameters
//! explicitly (sourced from `hardware_catalog` specs by the caller) — no sensor
//! here knows about a specific target body or mission.

pub mod dsn;
pub mod gyro;
pub mod imu;
pub mod lidar;
pub mod opnav;
pub mod star_tracker;

pub use dsn::{measure_range, measure_range_rate, RangeMeas, RangeRateMeas};
pub use gyro::{measure as gyro_measure, GyroBiasState};
pub use imu::measure_dv;
pub use lidar::measure as lidar_measure;
pub use opnav::{measure as opnav_measure, OpNavMeas};
pub use star_tracker::measure as star_tracker_measure;
