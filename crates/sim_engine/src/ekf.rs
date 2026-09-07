//! Generic Extended Kalman Filter: 6-state (r, v) baseline, extended to
//! 10-state (r, v, C_R, 3x Gauss-Markov stochastic acceleration) when the
//! mission's SRP model is Cannonball and an OpNav camera is configured — the
//! same rule as `MissionPlanner/src/gnc_design.rs::ekf_state_dim()`. State
//! dimension is a runtime choice (not a const generic), so state/covariance
//! use `nalgebra`'s dynamic `DVector`/`DMatrix` — fine at EKF update rates
//! (per `dt_meas_s`, not per truth step).
//!
//! Structurally generalizes `GNC/AutonomousNavigation/src/navigation/{stm,ekf}.rs`'s
//! Jacobian (gravity-gradient + tidal + SRP-partial-wrt-C_R + Gauss-Markov decay),
//! but takes mu, the Sun's position, and spacecraft area/mass as parameters
//! instead of importing named mission constants.
//!
//! Two deliberate deviations from the Bennu reference, both toward more
//! correctness (not blind copies of the simplification being generalized):
//! - The tidal-gradient block uses the FULL form (including the rank-1
//!   `3 mu rb rb^T` correction), not just the reference's `-mu/|r_B|^3 I`.
//! - The angular-size measurement model uses `asin(R/r)` (matching the
//!   sensor truth model in `sensors/opnav.rs`), not the small-angle `R/r`
//!   linearization the reference uses.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use orbital_models::GravityModel as AccelGravityModel;

/// Gauss-Markov correlation time and stationary 1-sigma acceleration — defaults
/// match the validated Bennu mission values
/// (`GNC/AutonomousNavigation/src/navigation/stm.rs::{MARKOV_TAU_S, MARKOV_SIGMA_A}`).
/// These are filter-tuning constants, not cited physical constants — override
/// via `EkfConfig` for a different mission's filter tuning.
pub const DEFAULT_MARKOV_TAU_S: f64 = 86_400.0;
pub const DEFAULT_MARKOV_SIGMA_A: f64 = 3.0e-9;

#[derive(Clone, Debug)]
pub struct EkfConfig {
    /// 6 (r, v) or 10 (r, v, C_R, 3x stochastic accel).
    pub state_dim: usize,
    pub mu_central_m3s2: f64,
    pub mu_sun_m3s2: f64,
    /// Central body's heliocentric position (Sun -> body) [m].
    pub sun_pos_from_body_m: Vector3<f64>,
    /// Spacecraft area/mass [m^2/kg], for the SRP-C_R partial (10-state only).
    pub srp_area_over_mass: f64,
    /// Local solar radiation pressure [N/m^2] (10-state only).
    pub p_srp: f64,
    pub markov_tau_s: f64,
    pub markov_sigma_a: f64,
    /// Position process-noise density [m^2/s].
    pub q_r: f64,
    /// Velocity process-noise density [m^2/s^3].
    pub q_v: f64,
    /// C_R process-noise density (10-state only).
    pub q_cr: f64,
}

#[derive(Clone, Debug)]
pub struct EkfState {
    pub x: DVector<f64>,
    pub p: DMatrix<f64>,
    pub t_s: f64,
}

impl EkfState {
    pub fn r(&self) -> Vector3<f64> {
        Vector3::new(self.x[0], self.x[1], self.x[2])
    }
    pub fn v(&self) -> Vector3<f64> {
        Vector3::new(self.x[3], self.x[4], self.x[5])
    }
    pub fn c_r(&self) -> Option<f64> {
        (self.x.len() > 6).then(|| self.x[6])
    }
    pub fn sigma_pos_m(&self) -> f64 {
        (self.p[(0, 0)] + self.p[(1, 1)] + self.p[(2, 2)]).sqrt()
    }
    pub fn sigma_vel_mps(&self) -> f64 {
        (self.p[(3, 3)] + self.p[(4, 4)] + self.p[(5, 5)]).sqrt()
    }
}

/// Build the initial filter state from an a-priori estimate and 1-sigma
/// uncertainties (diagonal covariance).
#[allow(clippy::too_many_arguments)]
pub fn init(
    cfg: &EkfConfig,
    r0: Vector3<f64>,
    v0: Vector3<f64>,
    c_r0: f64,
    sigma_r0_m: f64,
    sigma_v0_mps: f64,
    sigma_cr0: f64,
    sigma_a0_mps2: f64,
    t0_s: f64,
) -> EkfState {
    let n = cfg.state_dim;
    let mut x = DVector::zeros(n);
    x[0] = r0.x; x[1] = r0.y; x[2] = r0.z;
    x[3] = v0.x; x[4] = v0.y; x[5] = v0.z;
    if n == 10 {
        x[6] = c_r0;
    }
    let mut p = DMatrix::zeros(n, n);
    for i in 0..3 {
        p[(i, i)] = sigma_r0_m * sigma_r0_m;
    }
    for i in 3..6 {
        p[(i, i)] = sigma_v0_mps * sigma_v0_mps;
    }
    if n == 10 {
        p[(6, 6)] = sigma_cr0 * sigma_cr0;
        for i in 7..10 {
            p[(i, i)] = sigma_a0_mps2 * sigma_a0_mps2;
        }
    }
    EkfState { x, p, t_s: t0_s }
}

// ── Dynamics model (filter's simplified model — point-mass + tidal + SRP,
//    no zonal harmonics; the omission is standard EKF practice, absorbed by
//    process noise, same as the Bennu reference) ─────────────────────────────

fn dynamics(x: &DVector<f64>, cfg: &EkfConfig) -> DVector<f64> {
    let n = cfg.state_dim;
    let r = Vector3::new(x[0], x[1], x[2]);
    let v = Vector3::new(x[3], x[4], x[5]);
    let r_norm = r.norm();
    let mut a = -(cfg.mu_central_m3s2 / (r_norm * r_norm * r_norm)) * r;
    a += AccelGravityModel::tidal(&r, &cfg.sun_pos_from_body_m, cfg.mu_sun_m3s2);

    let mut xdot = DVector::zeros(n);
    xdot[0] = v.x; xdot[1] = v.y; xdot[2] = v.z;

    if n == 10 {
        let c_r = x[6];
        // sun_pos_from_body_m stores the body's heliocentric position (Sun→body), which
        // is the anti-sun (away-from-sun) direction.  The cannonball SRP force acts along
        // this direction (pushes the spacecraft away from the Sun).
        let anti_sun = cfg.sun_pos_from_body_m.normalize();
        a += cfg.p_srp * c_r * cfg.srp_area_over_mass * anti_sun;
        let a_stoch = Vector3::new(x[7], x[8], x[9]);
        a += a_stoch;
    }
    xdot[3] = a.x; xdot[4] = a.y; xdot[5] = a.z;

    if n == 10 {
        xdot[6] = 0.0;
        for i in 0..3 {
            xdot[7 + i] = -x[7 + i] / cfg.markov_tau_s;
        }
    }
    xdot
}

fn gravity_gradient_matrix(mu: f64, r_norm: f64, r_hat: &Vector3<f64>) -> Matrix3<f64> {
    let i3 = Matrix3::identity();
    let outer = r_hat * r_hat.transpose();
    -(mu / r_norm.powi(3)) * (i3 - 3.0 * outer)
}

fn jacobian_f(x: &DVector<f64>, cfg: &EkfConfig) -> DMatrix<f64> {
    let n = cfg.state_dim;
    let mut f = DMatrix::zeros(n, n);
    for i in 0..3 {
        f[(i, i + 3)] = 1.0;
    }

    let r = Vector3::new(x[0], x[1], x[2]);
    let r_norm = r.norm();
    let r_hat = r / r_norm;
    let gg = gravity_gradient_matrix(cfg.mu_central_m3s2, r_norm, &r_hat);

    let rb = cfg.sun_pos_from_body_m;
    let rb_norm = rb.norm();
    let rb3 = rb_norm.powi(3);
    let rb5 = rb_norm.powi(5);
    let tidal_grad = -(cfg.mu_sun_m3s2 / rb3) * Matrix3::identity()
        + (3.0 * cfg.mu_sun_m3s2 / rb5) * (rb * rb.transpose());

    let dv_dr = gg + tidal_grad;
    for i in 0..3 {
        for j in 0..3 {
            f[(i + 3, j)] = dv_dr[(i, j)];
        }
    }

    if n == 10 {
        let anti_sun = rb.normalize();  // rb = body heliocentric (Sun→body) = anti-sun direction
        let d_a_d_cr = cfg.p_srp * cfg.srp_area_over_mass * anti_sun;
        for i in 0..3 {
            f[(i + 3, 6)] = d_a_d_cr[i];
        }
        for i in 0..3 {
            f[(i + 3, 7 + i)] = 1.0;
        }
        for i in 0..3 {
            f[(7 + i, 7 + i)] = -1.0 / cfg.markov_tau_s;
        }
    }
    f
}

fn process_noise(cfg: &EkfConfig, dt: f64) -> DMatrix<f64> {
    let n = cfg.state_dim;
    let mut q = DMatrix::zeros(n, n);
    for i in 0..3 {
        q[(i, i)] = cfg.q_r * dt;
    }
    for i in 3..6 {
        q[(i, i)] = cfg.q_v * dt;
    }
    if n == 10 {
        q[(6, 6)] = cfg.q_cr * dt;
        let q_a = cfg.markov_sigma_a.powi(2) * (1.0 - (-2.0 * dt / cfg.markov_tau_s).exp());
        for i in 7..10 {
            q[(i, i)] = q_a;
        }
    }
    q
}

/// Predict the filter state forward by `dt` seconds: RK4 on the mean state,
/// first-order STM (`Phi = I + F*dt`) for the covariance — same pattern as
/// the Bennu reference's `predict()`/`jacobian_f()`.
pub fn predict(state: &EkfState, cfg: &EkfConfig, dt: f64) -> EkfState {
    let x0 = &state.x;
    let k1 = dynamics(x0, cfg);
    let k2 = dynamics(&(x0 + &k1 * (dt / 2.0)), cfg);
    let k3 = dynamics(&(x0 + &k2 * (dt / 2.0)), cfg);
    let k4 = dynamics(&(x0 + &k3 * dt), cfg);
    let x_new = x0 + (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4) * (dt / 6.0);

    let f = jacobian_f(x0, cfg);
    let n = cfg.state_dim;
    let phi = DMatrix::<f64>::identity(n, n) + &f * dt;
    let q = process_noise(cfg, dt);
    let p_new = &phi * &state.p * phi.transpose() + q;

    EkfState { x: x_new, p: p_new, t_s: state.t_s + dt }
}

// ── Measurement updates ──────────────────────────────────────────────────────

/// Joseph-form EKF update kernel — standard textbook form, numerically
/// stable covariance update.
fn ekf_update(state: &EkfState, h: &DMatrix<f64>, innov: &DVector<f64>, r_noise: &DMatrix<f64>) -> EkfState {
    let p_ht = &state.p * h.transpose();
    let s = h * &p_ht + r_noise;
    let s_inv = s.try_inverse().expect("innovation covariance not invertible");
    let k = &p_ht * &s_inv;
    let x_new = &state.x + &k * innov;
    let n = state.x.len();
    let ikh = DMatrix::<f64>::identity(n, n) - &k * h;
    let p_new = &ikh * &state.p * ikh.transpose() + &k * r_noise * k.transpose();
    EkfState { x: x_new, p: p_new, t_s: state.t_s }
}

/// Bearing (line-of-sight) update — 3-component LOS residual, same pattern as
/// the Bennu reference's `update_bearing` (minus the phase-angle centroid bias
/// correction, which belongs to the deferred pixel-rendering OpNav fidelity).
///
/// Returns the residual norm alongside the updated state — for GNC dashboard
/// residual histograms (Phase 6f). The residual itself is still computed and
/// discarded internally by every update function below; only the norm is
/// surfaced, since that's what a residual-magnitude histogram actually plots.
pub fn update_bearing(state: &EkfState, los_meas: &Vector3<f64>, sigma_bearing_rad: f64) -> (EkfState, f64) {
    let r = state.r();
    let r_norm = r.norm();
    let r_hat = r / r_norm;
    let los_pred = -r_hat;
    let innov_v = los_meas - los_pred;

    let dlos_dr = -(Matrix3::identity() - r_hat * r_hat.transpose()) / r_norm;
    let n = state.x.len();
    let mut h = DMatrix::zeros(3, n);
    for i in 0..3 {
        for j in 0..3 {
            h[(i, j)] = dlos_dr[(i, j)];
        }
    }
    let r_noise = DMatrix::<f64>::identity(3, 3) * (sigma_bearing_rad * sigma_bearing_rad);
    let innov = DVector::from_iterator(3, innov_v.iter().cloned());
    (ekf_update(state, &h, &innov, &r_noise), innov_v.norm())
}

/// Apparent angular-size update — `y = asin(R_body / |r|)`.
pub fn update_angular_size(state: &EkfState, size_meas_rad: f64, body_radius_m: f64, sigma_size_rad: f64) -> (EkfState, f64) {
    let r = state.r();
    let r_norm = r.norm();
    let u = (body_radius_m / r_norm).clamp(-1.0, 1.0);
    let y_pred = u.asin();
    let innov_scalar = size_meas_rad - y_pred;

    let dy_dr_norm = -body_radius_m / (r_norm * r_norm * (1.0 - u * u).sqrt().max(1e-9));
    let r_hat = r / r_norm;
    let n = state.x.len();
    let mut h = DMatrix::zeros(1, n);
    for j in 0..3 {
        h[(0, j)] = dy_dr_norm * r_hat[j];
    }
    let r_noise = DMatrix::from_element(1, 1, sigma_size_rad * sigma_size_rad);
    let innov = DVector::from_element(1, innov_scalar);
    (ekf_update(state, &h, &innov, &r_noise), innov_scalar.abs())
}

/// LIDAR slant-range update — `y = |r| - R_body`.
pub fn update_lidar(state: &EkfState, range_meas_m: f64, body_radius_m: f64, sigma_range_m: f64) -> (EkfState, f64) {
    let r = state.r();
    let r_norm = r.norm();
    let y_pred = r_norm - body_radius_m;
    let innov_scalar = range_meas_m - y_pred;

    let r_hat = r / r_norm;
    let n = state.x.len();
    let mut h = DMatrix::zeros(1, n);
    for j in 0..3 {
        h[(0, j)] = r_hat[j];
    }
    let r_noise = DMatrix::from_element(1, 1, sigma_range_m * sigma_range_m);
    let innov = DVector::from_element(1, innov_scalar);
    (ekf_update(state, &h, &innov, &r_noise), innov_scalar.abs())
}

/// IMU delta-V update: corrects the velocity estimate by the discrepancy
/// between the IMU-measured delta-V and the commanded delta-V, both expressed
/// in the same inertial/Hill frame as the velocity state (rotate the IMU's
/// body-frame measurement to inertial using the best attitude estimate before
/// calling this).
///
/// Not currently called from `SimEngine::step()` (no IMU measurement wired
/// into the measurement loop yet) — signature updated to match the other
/// update functions for consistency, but there's no telemetry field for its
/// residual since nothing produces one yet.
pub fn update_dv(state: &EkfState, dv_inertial_meas: &Vector3<f64>, dv_inertial_cmd: &Vector3<f64>, sigma_mps: f64) -> (EkfState, f64) {
    let innov_v = dv_inertial_meas - dv_inertial_cmd;
    let n = state.x.len();
    let mut h = DMatrix::zeros(3, n);
    for i in 0..3 {
        h[(i, 3 + i)] = 1.0;
    }
    let r_noise = DMatrix::<f64>::identity(3, 3) * (sigma_mps * sigma_mps);
    let innov = DVector::from_iterator(3, innov_v.iter().cloned());
    (ekf_update(state, &h, &innov, &r_noise), innov_v.norm())
}
