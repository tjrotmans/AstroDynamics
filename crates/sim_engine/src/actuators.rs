//! Thin actuator wrappers around `attitude_control` — reaction-wheel cluster
//! and RCS thruster geometry — parameterized by `hardware_catalog` specs
//! instead of mission-specific config constants.

use attitude_control::{pd_torque, rcs_thrusters, thruster_selection, PdGains, ReactionWheelCluster, Thruster};
use hardware_catalog::{ReactionWheelSpec, ThrusterSpec};
use nalgebra::{Vector3, Vector4};
use orbital_models::constants::G0;

pub fn build_wheel_cluster(spec: &ReactionWheelSpec, desat_fraction: f64) -> ReactionWheelCluster {
    ReactionWheelCluster::four_wheel_pyramid(spec.inertia_kgm2, spec.max_speed_rads, spec.max_torque_nm, desat_fraction)
}

pub fn build_rcs(spec: &ThrusterSpec, moment_arm_m: f64) -> Vec<Thruster> {
    rcs_thrusters(spec.thrust_n, moment_arm_m)
}

/// One reaction-wheel control step: allocate the commanded torque to the four
/// wheels and integrate wheel speeds over `dt`. Exact for a zero-order-hold
/// motor torque, since the wheel-speed ODE is linear in torque. Returns the
/// actual torque delivered to the spacecraft body plus the updated speeds.
pub fn wheel_step(
    cluster: &ReactionWheelCluster,
    speeds: &[f64; 4],
    tau_cmd: &Vector3<f64>,
    dt: f64,
) -> (Vector3<f64>, [f64; 4]) {
    let tau_motor = cluster.allocate(tau_cmd);
    let speed_dots = cluster.speed_dots(&tau_motor);
    let mut new_speeds = *speeds;
    for i in 0..4 {
        new_speeds[i] += speed_dots[i] * dt;
    }
    let body_torque = cluster.body_torque(&tau_motor);
    (body_torque, new_speeds)
}

/// RCS desaturation: if the wheel cluster needs desaturation, fire the
/// thrusters opposing the wheel momentum and return the net body torque plus
/// propellant mass burned this step.
pub fn rcs_desaturation_step(
    cluster: &ReactionWheelCluster,
    speeds: &[f64; 4],
    thrusters: &[Thruster],
    isp_s: f64,
    dt: f64,
) -> Option<(Vector3<f64>, f64)> {
    let tau_cmd = cluster.desat_torque(speeds)?;
    let (_net_force, net_torque, total_thrust_n) = thruster_selection(&tau_cmd, thrusters);
    let mass_flow_kgps = total_thrust_n / (isp_s * G0);
    Some((net_torque, mass_flow_kgps * dt))
}

/// RCS attitude-hold PD control step — used when `AttitudeController::RcsBangBang`
/// is configured (bang-bang thruster firing in place of continuous wheel torque).
pub fn rcs_pd_step(
    thrusters: &[Thruster],
    q_cur: &Vector4<f64>,
    q_cmd: &Vector4<f64>,
    omega: &Vector3<f64>,
    gains: &PdGains,
) -> (Vector3<f64>, f64) {
    let tau_cmd = pd_torque(q_cur, q_cmd, omega, gains);
    let (_net_force, net_torque, total_thrust_n) = thruster_selection(&tau_cmd, thrusters);
    (net_torque, total_thrust_n)
}
