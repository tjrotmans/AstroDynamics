//! `SimEngine` — drives truth propagation, attitude control, station-keeping
//! guidance, and the GNC measurement/EKF loop for one truth step at a time.
//! The caller (`MissionPlanner/src/simulate.rs`) owns the phase sequence and
//! steps the engine through it, writing telemetry CSVs from the returned rows.

use attitude_control::{pd_torque, PdGains, ReactionWheelCluster, Thruster};
use nalgebra::Vector3;
use rand::{rngs::StdRng, SeedableRng};

use crate::actuators;
use crate::ekf::{self, EkfConfig, EkfState};
use crate::guidance;
use crate::phase::Phase;
use crate::sensors;
use crate::truth::{self, Environment, SpacecraftProperties, TruthState};

/// Per-step telemetry — enough to reconstruct the nav/attitude/maneuver CSVs
/// the existing AutonomousNavigation plot scripts already expect the shape of.
#[derive(Clone, Debug)]
pub struct StepTelemetry {
    pub t_s: f64,
    pub phase_name: &'static str,
    pub r_truth_m: Vector3<f64>,
    pub v_truth_mps: Vector3<f64>,
    pub r_ekf_m: Vector3<f64>,
    pub v_ekf_mps: Vector3<f64>,
    pub sigma_pos_m: f64,
    pub sigma_vel_mps: f64,
    pub omega_norm_radps: f64,
    pub pointing_err_rad: f64,
    pub wheel_speeds_radps: [f64; 4],
    pub wheel_momentum_nms: f64,
    pub wheel_sat_frac: f64,
    pub desat_fired: bool,
    pub dv_applied_mps: Option<Vector3<f64>>,
    pub measurement_taken: bool,
    /// Measurement residual norms, `None` when that sensor isn't configured
    /// or didn't return a measurement this step — for GNC dashboard residual
    /// histograms (Phase 6f). Magnitude only (not per-component); see
    /// `ekf::update_bearing` etc.
    pub bearing_residual_rad: Option<f64>,
    pub angular_size_residual_rad: Option<f64>,
    pub lidar_residual_m: Option<f64>,
}

/// Optional OpNav sensor config: (bearing noise, angular-size noise) [rad].
pub type OpNavCfg = (f64, f64);
/// Optional LIDAR sensor config: (range noise [m], max range [m]).
pub type LidarCfg = (f64, f64);

pub struct SimEngine {
    pub truth: TruthState,
    pub ekf: EkfState,
    pub sc: SpacecraftProperties,
    pub env: Environment,
    pub ekf_cfg: EkfConfig,
    pub wheel_cluster: ReactionWheelCluster,
    pub rcs_thrusters: Vec<Thruster>,
    pub rcs_isp_s: f64,
    pub pd_gains: PdGains,
    pub opnav_cfg: Option<OpNavCfg>,
    pub lidar_cfg: Option<LidarCfg>,
    pub dt_truth_s: f64,
    pub dt_meas_s: f64,
    /// Radius dead-band [m] for station-keeping — see `guidance::station_keeping_dv`.
    pub sk_tolerance_m: f64,
    pub propellant_kg: f64,
    time_since_meas_s: f64,
    rng: StdRng,
}

impl SimEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        truth: TruthState,
        ekf: EkfState,
        sc: SpacecraftProperties,
        env: Environment,
        ekf_cfg: EkfConfig,
        wheel_cluster: ReactionWheelCluster,
        rcs_thrusters: Vec<Thruster>,
        rcs_isp_s: f64,
        pd_gains: PdGains,
        opnav_cfg: Option<OpNavCfg>,
        lidar_cfg: Option<LidarCfg>,
        dt_truth_s: f64,
        dt_meas_s: f64,
        sk_tolerance_m: f64,
        propellant_kg: f64,
        rng_seed: u64,
    ) -> Self {
        Self {
            truth, ekf, sc, env, ekf_cfg, wheel_cluster, rcs_thrusters, rcs_isp_s, pd_gains,
            opnav_cfg, lidar_cfg, dt_truth_s, dt_meas_s, sk_tolerance_m,
            propellant_kg, time_since_meas_s: 0.0, rng: StdRng::seed_from_u64(rng_seed),
        }
    }

    /// Advance one truth step (`dt_truth_s`), running attitude control,
    /// and — at the configured `dt_meas_s` cadence — sensor measurements,
    /// an EKF predict+update, and a station-keeping impulse. Tying SK to the
    /// measurement epoch (not every truth step) prevents the rapid ΔV
    /// accumulation that destabilizes orbits in micro-gravity environments
    /// (e.g. Bennu, where v_circ ≈ 0.04 m/s but n² ≈ 1.8e-10 s⁻²).
    ///
    /// Takes `&mut dyn Phase` so phases with one-shot burns (e.g.
    /// `OrbitInsertionPhase`) can update their internal state on the call.
    pub fn step(&mut self, phase: &mut dyn Phase) -> StepTelemetry {
        let q_cmd = guidance::desired_quaternion(phase.pointing_mode(), &self.truth.r_m, &self.truth.v_mps);

        let tau_cmd = pd_torque(&self.truth.q, &q_cmd, &self.truth.omega_radps, &self.pd_gains);
        let (wheel_torque, new_speeds) =
            actuators::wheel_step(&self.wheel_cluster, &self.truth.wheel_speeds_radps, &tau_cmd, self.dt_truth_s);

        let mut control_torque = wheel_torque;
        let mut desat_fired = false;
        if let Some((rcs_torque, propellant_used)) = actuators::rcs_desaturation_step(
            &self.wheel_cluster, &new_speeds, &self.rcs_thrusters, self.rcs_isp_s, self.dt_truth_s,
        ) {
            desat_fired = true;
            control_torque += rcs_torque;
            self.propellant_kg -= propellant_used;
        }
        let h_wheel = self.wheel_cluster.total_momentum(&new_speeds);

        let disturbance = truth::disturbance_torque_body(&self.truth, &self.sc, &self.env);
        let net_torque = disturbance + control_torque;

        let mut new_truth = truth::propagate_step(&self.truth, self.dt_truth_s, &self.sc, &self.env, &net_torque, &h_wheel);
        new_truth.wheel_speeds_radps = new_speeds;

        let mut new_ekf = ekf::predict(&self.ekf, &self.ekf_cfg, self.dt_truth_s);

        // One-shot impulsive burn (e.g. LOI at periapsis) — fires at most once per
        // phase instance; applied every truth step so periapsis timing is not missed.
        if let Some(loi_dv) = phase.one_time_burn(&new_truth, self.env.body.mu_m3s2) {
            new_truth.v_mps += loi_dv;
            new_ekf.x[3] += loi_dv.x;
            new_ekf.x[4] += loi_dv.y;
            new_ekf.x[5] += loi_dv.z;
            // Record alongside SK burns in the same field; caller can distinguish
            // via phase name (insertion phases vs orbit phases).
        }

        self.time_since_meas_s += self.dt_truth_s;
        let mut measurement_taken = false;
        let mut dv_applied: Option<nalgebra::Vector3<f64>> = None;
        let mut bearing_residual_rad: Option<f64> = None;
        let mut angular_size_residual_rad: Option<f64> = None;
        let mut lidar_residual_m: Option<f64> = None;
        if self.time_since_meas_s + 1e-9 >= self.dt_meas_s {
            self.time_since_meas_s = 0.0;
            measurement_taken = true;
            if let Some((sigma_bearing, sigma_size)) = self.opnav_cfg {
                let meas = sensors::opnav_measure(
                    &new_truth.r_m, self.env.body.radius_m, sigma_bearing, sigma_size, &mut self.rng,
                );
                let (ekf1, bearing_res) = ekf::update_bearing(&new_ekf, &meas.los, sigma_bearing);
                new_ekf = ekf1;
                bearing_residual_rad = Some(bearing_res);
                let (ekf2, size_res) =
                    ekf::update_angular_size(&new_ekf, meas.angular_size_rad, self.env.body.radius_m, sigma_size);
                new_ekf = ekf2;
                angular_size_residual_rad = Some(size_res);
            }
            if let Some((sigma_range, max_range)) = self.lidar_cfg {
                if let Some(range_meas) =
                    sensors::lidar_measure(&new_truth.r_m, self.env.body.radius_m, sigma_range, max_range, &mut self.rng)
                {
                    let (ekf3, lidar_res) = ekf::update_lidar(&new_ekf, range_meas, self.env.body.radius_m, sigma_range);
                    new_ekf = ekf3;
                    lidar_residual_m = Some(lidar_res);
                }
            }

            // Station-keeping burn after measurement update: uses the best
            // available state estimate and fires at measurement cadence, not
            // every truth step. The circular-speed law is body-agnostic and
            // sized directly from mu — no hand-tuned gain required.
            if let Some(target_r) = phase.target_radius_m() {
                if let Some(dv) = guidance::station_keeping_dv(
                    &new_truth.r_m, &new_truth.v_mps, target_r,
                    self.sk_tolerance_m, self.env.body.mu_m3s2,
                ) {
                    new_truth.v_mps += dv;
                    new_ekf.x[3] += dv.x;
                    new_ekf.x[4] += dv.y;
                    new_ekf.x[5] += dv.z;
                    dv_applied = Some(dv);
                }
            }
        }

        let pointing_err_rad = quat_angle_error(&new_truth.q, &q_cmd);
        let wheel_sat_frac = new_speeds.iter().fold(0.0_f64, |m, s| m.max(s.abs())) / self.wheel_cluster.max_speed;

        let telemetry = StepTelemetry {
            t_s: new_truth.t_s,
            phase_name: phase.name(),
            r_truth_m: new_truth.r_m,
            v_truth_mps: new_truth.v_mps,
            r_ekf_m: new_ekf.r(),
            v_ekf_mps: new_ekf.v(),
            sigma_pos_m: new_ekf.sigma_pos_m(),
            sigma_vel_mps: new_ekf.sigma_vel_mps(),
            omega_norm_radps: new_truth.omega_radps.norm(),
            pointing_err_rad,
            wheel_speeds_radps: new_speeds,
            wheel_momentum_nms: h_wheel.norm(),
            wheel_sat_frac,
            desat_fired,
            dv_applied_mps: dv_applied,
            measurement_taken,
            bearing_residual_rad,
            angular_size_residual_rad,
            lidar_residual_m,
        };

        self.truth = new_truth;
        self.ekf = new_ekf;
        telemetry
    }
}

/// Angular difference [rad] between two unit quaternions, accounting for the
/// double-cover (q and -q represent the same rotation): `2*acos(|q1 . q2|)`.
fn quat_angle_error(q1: &nalgebra::Vector4<f64>, q2: &nalgebra::Vector4<f64>) -> f64 {
    let dot = q1.dot(q2).clamp(-1.0, 1.0).abs();
    2.0 * dot.acos()
}
