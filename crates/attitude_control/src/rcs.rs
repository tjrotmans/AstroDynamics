//! Reaction Control System — thruster geometry, allocation, and PD attitude controller.
//!
//! All hardware parameters (thrust, moment arm, gains, dead-bands) are passed as
//! arguments or stored in structs — no mission-specific constants are imported.

use nalgebra::{Vector3, Vector4};

// ── Thruster ──────────────────────────────────────────────────────────────────

/// One RCS thruster: direction, moment-arm position, thrust magnitude, and
/// specific impulse.
#[derive(Clone, Copy, Debug)]
pub struct Thruster {
    /// Unit vector of thrust direction (body frame).
    pub dir: Vector3<f64>,
    /// Position of thruster relative to CoM (body frame) [m].
    pub pos: Vector3<f64>,
    /// Thrust magnitude [N].
    pub thrust_n: f64,
    /// Specific impulse [s] — per-thruster
    /// so a real,
    /// per-unit-placed layout mixing thruster classes (e.g. fine ColdGas
    /// thrusters for pointing alongside a higher-thrust Monoprop set for
    /// coarse slews) prices propellant against EACH thruster's own real
    /// Isp, not one aggregate value assumed for the whole set. See
    /// `sim_engine::control::allocate`'s propellant formula, which now
    /// sums per-thruster instead of using a single scalar Isp.
    pub isp_s: f64,
}

impl Thruster {
    /// Force produced by this thruster in body frame [N].
    pub fn force(&self) -> Vector3<f64> { self.dir * self.thrust_n }
    /// Torque produced by this thruster about CoM in body frame [N·m].
    pub fn torque(&self) -> Vector3<f64> { self.pos.cross(&self.force()) }
}

/// Representative monopropellant-class specific impulse [s] — used as the
/// `isp_s` for `rcs_thrusters`/`translation_thrusters`'s generic/aggregate
/// builders below, which have no notion of a real per-unit spec (matches
/// `hardware_catalog::ThrusterSpec::monoprop().isp_s`, this crate's own
/// generic-path default before `Thruster` carried Isp at all — this
/// constant preserves that exact prior behavior, not a new assumption).
/// `attitude_control` does not depend on `hardware_catalog`, so this is a
/// local, cited duplicate rather than a cross-crate reference.
const GENERIC_RCS_ISP_S: f64 = 220.0;

/// Build a 12-thruster attitude-control configuration with full 3-axis authority.
///
/// Each axis (x, y, z) gets a dedicated couple — two thrusters whose combined
/// torque is purely about that axis with no cross-coupling.
///
/// Layout (body frame):
///   ±x torque : thrust ±z, arm at ±y
///   ±y torque : thrust ±x, arm at ±z
///   ±z torque : thrust ±y, arm at ±x
///
/// # Arguments
/// * `thrust_n`    – Thrust per thruster [N]
/// * `moment_arm`  – Distance from CoM to thruster position [m]
pub fn rcs_thrusters(thrust_n: f64, moment_arm: f64) -> Vec<Thruster> {
    let a = moment_arm;
    let config: [(Vector3<f64>, Vector3<f64>); 12] = [
        // +x torque: τ = (0,+a,0)×(0,0,+1)·F = (+a·F, 0, 0)
        (Vector3::new(0., 0., 1.), Vector3::new(0., a, 0.)),
        (Vector3::new(0., 0.,-1.), Vector3::new(0.,-a, 0.)),
        // -x torque
        (Vector3::new(0., 0., 1.), Vector3::new(0.,-a, 0.)),
        (Vector3::new(0., 0.,-1.), Vector3::new(0., a, 0.)),
        // +y torque: τ = (0,0,+a)×(+1,0,0)·F = (0,+a·F,0)
        (Vector3::new( 1., 0., 0.), Vector3::new(0., 0., a)),
        (Vector3::new(-1., 0., 0.), Vector3::new(0., 0.,-a)),
        // -y torque
        (Vector3::new( 1., 0., 0.), Vector3::new(0., 0.,-a)),
        (Vector3::new(-1., 0., 0.), Vector3::new(0., 0., a)),
        // +z torque: τ = (+a,0,0)×(0,+1,0)·F = (0,0,+a·F)
        (Vector3::new(0.,  1., 0.), Vector3::new( a, 0., 0.)),
        (Vector3::new(0., -1., 0.), Vector3::new(-a, 0., 0.)),
        // -z torque
        (Vector3::new(0.,  1., 0.), Vector3::new(-a, 0., 0.)),
        (Vector3::new(0., -1., 0.), Vector3::new( a, 0., 0.)),
    ];
    config.iter().map(|(dir, pos)| Thruster { dir: *dir, pos: *pos, thrust_n, isp_s: GENERIC_RCS_ISP_S }).collect()
}

/// Build 6 dedicated translation thrusters (±x, ±y, ±z), each at the CoM.
///
/// Zero moment arm → pure translational force, no attitude torque.
pub fn translation_thrusters(thrust_n: f64) -> Vec<Thruster> {
    [
        Vector3::new( 1., 0., 0.),
        Vector3::new(-1., 0., 0.),
        Vector3::new( 0., 1., 0.),
        Vector3::new( 0.,-1., 0.),
        Vector3::new( 0., 0., 1.),
        Vector3::new( 0., 0.,-1.),
    ]
    .iter()
    .map(|&dir| Thruster { dir, pos: Vector3::zeros(), thrust_n, isp_s: GENERIC_RCS_ISP_S })
    .collect()
}

// ── PD attitude controller ────────────────────────────────────────────────────

/// PD controller gains and dead-band thresholds.
#[derive(Clone, Copy, Debug)]
pub struct PdGains {
    pub kp: f64,
    pub kd: f64,
    /// Pointing angle error dead-band [rad].
    pub pointing_db_rad: f64,
    /// Angular rate dead-band [rad/s].
    pub rate_db_rads: f64,
}

/// Commanded torque from a quaternion PD controller with dual dead-band [N·m].
///
/// Returns zero when BOTH angle error < `pointing_db_rad` AND |ω| < `rate_db_rads`,
/// preventing limit-cycle chatter once attitude is acquired and residual rate is braked.
///
/// # Arguments
/// * `q_cur`  – Current attitude quaternion [w,x,y,z]
/// * `q_cmd`  – Commanded attitude quaternion [w,x,y,z]
/// * `omega`  – Current angular rate in body frame [rad/s]
/// * `gains`  – PD gains and dead-band thresholds
pub fn pd_torque(
    q_cur: &Vector4<f64>,
    q_cmd: &Vector4<f64>,
    omega:  &Vector3<f64>,
    gains:  &PdGains,
) -> Vector3<f64> {
    let q_err = quat_error(q_cur, q_cmd);
    let angle_err = 2.0 * q_err[0].abs().min(1.0).acos();

    if angle_err < gains.pointing_db_rad && omega.norm() < gains.rate_db_rads {
        return Vector3::zeros();
    }

    let err_vec = Vector3::new(q_err[1], q_err[2], q_err[3]);
    let sign = if q_err[0] >= 0.0 { 1.0 } else { -1.0 };
    -gains.kp * sign * err_vec - gains.kd * omega
}

// ── Thruster selection ────────────────────────────────────────────────────────

/// Map a continuous torque demand to individual thruster on/off commands.
///
/// Returns `(net_force [N], net_torque [N·m], total_thrust [N])`.
/// `total_thrust` counts every firing thruster (useful for propellant accounting);
/// it is not the same as the net force, because couples cancel in translation.
pub fn thruster_selection(
    tau_cmd:   &Vector3<f64>,
    thrusters: &[Thruster],
) -> (Vector3<f64>, Vector3<f64>, f64) {
    let mut total_force  = Vector3::zeros();
    let mut total_torque = Vector3::zeros();
    let mut total_thrust = 0.0_f64;
    for t in thrusters {
        let tau = t.torque();
        if tau.dot(tau_cmd) > 0.0 {
            total_force  += t.force();
            total_torque += tau;
            total_thrust += t.thrust_n;
        }
    }
    (total_force, total_torque, total_thrust)
}

/// Which thrusters fire for a given torque demand — the same per-thruster
/// selection rule `thruster_selection` already aggregates
/// (`thruster.torque().dot(tau_cmd) > 0.0`), exposed individually so a
/// caller can report exactly which physical thruster is doing the work,
/// not just the aggregate net force/torque/total_thrust. Lets
///
/// `sim_engine::control::allocate` report a per-thruster PWM duty cycle
/// instead of only the aggregate `rcs_duty_cycle`.
pub fn select_firing_thrusters(tau_cmd: &Vector3<f64>, thrusters: &[Thruster]) -> Vec<bool> {
    thrusters.iter().map(|t| t.torque().dot(tau_cmd) > 0.0).collect()
}

/// Select translation thrusters aligned with a commanded body-frame force.
///
/// Returns `(net_force [N, body frame], total_thrust [N])`.
pub fn translation_thrust_select(
    force_cmd:       &Vector3<f64>,
    trans_thrusters: &[Thruster],
) -> (Vector3<f64>, f64) {
    let mut total_force  = Vector3::zeros();
    let mut total_thrust = 0.0_f64;
    for t in trans_thrusters {
        if t.dir.dot(force_cmd) > 0.0 {
            total_force  += t.force();
            total_thrust += t.thrust_n;
        }
    }
    (total_force, total_thrust)
}

// ── Internal helper ───────────────────────────────────────────────────────────

/// q_err = conj(q_cmd) ⊗ q_cur  (quaternion attitude error, [w,x,y,z]).
fn quat_error(q_cur: &Vector4<f64>, q_cmd: &Vector4<f64>) -> Vector4<f64> {
    let (wc, xc, yc, zc) = (q_cur[0], q_cur[1], q_cur[2], q_cur[3]);
    let (wd, xd, yd, zd) = (q_cmd[0], q_cmd[1], q_cmd[2], q_cmd[3]);
    Vector4::new(
        wd * wc + xd * xc + yd * yc + zd * zc,
        wd * xc - xd * wc - yd * zc + zd * yc,
        wd * yc + xd * zc - yd * wc - zd * xc,
        wd * zc - xd * yc + yd * xc - zd * wc,
    )
}
