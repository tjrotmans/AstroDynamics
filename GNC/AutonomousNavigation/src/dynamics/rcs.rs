//! Reaction Control System — mission-specific wiring over attitude_control::rcs.
//!
//! Provides constructor functions that bind config constants into the generic
//! `attitude_control` types, and re-exports the public API unchanged.

use attitude_control::{rcs_thrusters as ac_rcs_thrusters,
                       translation_thrusters as ac_trans_thrusters};
use crate::config::{RCS_THRUST_N, RCS_MOMENT_ARM,
                    RCS_POINTING_DB_RAD, RCS_RATE_DB_RADS,
                    ATTITUDE_KP, ATTITUDE_KD};

// Re-export generic types so callers keep the same import path.
pub use attitude_control::{Thruster, PdGains,
                           pd_torque, thruster_selection, translation_thrust_select};

/// Build the 12-thruster attitude-control configuration from mission config.
pub fn rcs_thrusters() -> Vec<Thruster> {
    ac_rcs_thrusters(RCS_THRUST_N, RCS_MOMENT_ARM)
}

/// Build 6 dedicated translation thrusters from mission config.
pub fn translation_thrusters() -> Vec<Thruster> {
    ac_trans_thrusters(RCS_THRUST_N)
}

/// RCS PD gains from mission config (bang-bang attitude control).
pub fn mission_pd_gains() -> PdGains {
    PdGains {
        kp: ATTITUDE_KP,
        kd: ATTITUDE_KD,
        pointing_db_rad: RCS_POINTING_DB_RAD,
        rate_db_rads:    RCS_RATE_DB_RADS,
    }
}
