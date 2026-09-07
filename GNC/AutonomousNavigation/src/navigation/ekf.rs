//! Extended Kalman Filter — predict and update steps.
//!
//! State: X = [r(3), v(3), C_SRP(1)]
//! Covariance: P (7×7)
//!
//! Measurement types handled:
//!   - OpNav bearing (3-element unit vector) — phase-angle centroid bias corrected
//!   - OpNav angular size (scalar)
//!   - IMU ΔV (3-element vector, applied at manoeuvre epochs)
//!   - DSN heliocentric state (position + velocity, 6-element)

use nalgebra::{SMatrix, SVector, Vector3, Matrix3};
use crate::config::{
    R_BENNU, SNC_QR, SNC_QV,
    OPNAV_SIZE_SIGMA_RAD, IMU_DV_SIGMA_MPS, LIDAR_SIGMA_M,
    DSN_RANGE_SIGMA_M, DSN_VELOCITY_SIGMA_MPS, BENNU_EPHEM_SIGMA_M, DSN_BENNU_V_SIGMA_MPS,
};
use crate::dynamics::hill::accel_filter;
use crate::dynamics::bennu::bennu_heliocentric_pos;
use super::stm::{NX, StateVec, StateMat, jacobian_f, stm, process_noise,
                 MARKOV_SIGMA_A, MARKOV_TAU_S};

// ── EKF state ─────────────────────────────────────────────────────────────────

/// EKF estimate: mean state + covariance.
#[derive(Clone, Debug)]
pub struct EkfState {
    /// State estimate X = [r(3), v(3), C_SRP]
    pub x: StateVec,
    /// Covariance P (7×7)
    pub p: StateMat,
    /// Current time [s]
    pub t: f64,
}

impl EkfState {
    /// Initialise from config uncertainties and a truth-state starting point.
    pub fn from_config(r0: Vector3<f64>, v0: Vector3<f64>) -> Self {
        use crate::config::{C_R_NOMINAL, EKF_SIGMA_R0_M, EKF_SIGMA_V0_MPS, EKF_SIGMA_CR0};
        let mut x = StateVec::zeros();
        x[0] = r0[0]; x[1] = r0[1]; x[2] = r0[2];
        x[3] = v0[0]; x[4] = v0[1]; x[5] = v0[2];
        x[6] = C_R_NOMINAL;

        let mut p = StateMat::zeros();
        let sr2 = EKF_SIGMA_R0_M   * EKF_SIGMA_R0_M;
        let sv2 = EKF_SIGMA_V0_MPS * EKF_SIGMA_V0_MPS;
        let sc2 = EKF_SIGMA_CR0    * EKF_SIGMA_CR0;
        p[(0,0)] = sr2; p[(1,1)] = sr2; p[(2,2)] = sr2;
        p[(3,3)] = sv2; p[(4,4)] = sv2; p[(5,5)] = sv2;
        p[(6,6)] = sc2;
        // Stochastic acceleration starts at zero with its stationary variance.
        let sa2 = MARKOV_SIGMA_A * MARKOV_SIGMA_A;
        p[(7,7)] = sa2; p[(8,8)] = sa2; p[(9,9)] = sa2;

        Self { x, p, t: 0.0 }
    }

    /// Initialise from a cruise-to-proximity handoff with externally supplied
    /// position/velocity and uncertainty values.
    pub fn with_initial_state(
        r0:       Vector3<f64>,
        v0:       Vector3<f64>,
        sigma_r:  f64,
        sigma_v:  f64,
        sigma_cr: f64,
        t_arr:    f64,
    ) -> Self {
        use crate::config::C_R_NOMINAL;
        let mut x = StateVec::zeros();
        x[0] = r0[0]; x[1] = r0[1]; x[2] = r0[2];
        x[3] = v0[0]; x[4] = v0[1]; x[5] = v0[2];
        x[6] = C_R_NOMINAL;

        let mut p = StateMat::zeros();
        let sr2 = sigma_r  * sigma_r;
        let sv2 = sigma_v  * sigma_v;
        let sc2 = sigma_cr * sigma_cr;
        p[(0,0)] = sr2; p[(1,1)] = sr2; p[(2,2)] = sr2;
        p[(3,3)] = sv2; p[(4,4)] = sv2; p[(5,5)] = sv2;
        p[(6,6)] = sc2;
        let sa2 = MARKOV_SIGMA_A * MARKOV_SIGMA_A;
        p[(7,7)] = sa2; p[(8,8)] = sa2; p[(9,9)] = sa2;

        Self { x, p, t: t_arr }
    }

    /// Position estimate [m]
    pub fn r(&self) -> Vector3<f64> { Vector3::new(self.x[0], self.x[1], self.x[2]) }
    /// Velocity estimate [m/s]
    pub fn v(&self) -> Vector3<f64> { Vector3::new(self.x[3], self.x[4], self.x[5]) }
    /// C_SRP estimate
    pub fn c_r(&self) -> f64 { self.x[6] }
    /// Estimated stochastic (unmodelled) acceleration [m/s²].
    pub fn a_stoch(&self) -> Vector3<f64> { Vector3::new(self.x[7], self.x[8], self.x[9]) }
}

// ── Predict step ──────────────────────────────────────────────────────────────

/// Propagate the EKF state and covariance forward by `dt` seconds.
///
/// Uses RK4 for the mean state (matching the truth propagator) and a
/// first-order STM for covariance propagation.
pub fn predict(ekf: &EkfState, dt: f64) -> EkfState {
    // Normal operation: C_R nearly constant (tiny random walk).
    predict_inner(ekf, dt, SNC_QR, SNC_QV, 1e-10)
}

/// Predict step tuned for the quiescent radio-science calibration arc.
///
/// The filter **trusts its dynamics** (much smaller position/velocity process
/// noise) so a systematic SRP residual can no longer be absorbed as random noise,
/// and **opens up C_R** (larger random-walk noise) so it can drift toward the
/// value that reproduces the true (panel) SRP.  This is what lets a maneuver-free
/// arc of natural, SRP-perturbed orbit evolution calibrate the reflectivity.
pub fn predict_calibration(ekf: &EkfState, dt: f64) -> EkfState {
    predict_inner(ekf, dt, CAL_QR, CAL_QV, CAL_QCR)
}

/// Shared predict implementation parametrised by the process-noise densities.
fn predict_inner(ekf: &EkfState, dt: f64, q_r: f64, q_v: f64, q_cr: f64) -> EkfState {
    let r   = ekf.r();
    let v   = ekf.v();
    let c_r = ekf.c_r();
    let a_s = ekf.a_stoch();
    let t   = ekf.t;

    // RK4 for mean state — consistent with truth-model propagation.
    // The stochastic acceleration is added as a (locally constant) extra force.
    let (r_new, v_new) = rk4_filter_state(r, v, c_r, &a_s, t, dt);

    // Gauss-Markov decay of the stochastic acceleration over the step.
    let decay = (-dt / MARKOV_TAU_S).exp();

    let mut x_new = StateVec::zeros();
    x_new[0] = r_new[0]; x_new[1] = r_new[1]; x_new[2] = r_new[2];
    x_new[3] = v_new[0]; x_new[4] = v_new[1]; x_new[5] = v_new[2];
    x_new[6] = c_r;
    x_new[7] = a_s[0] * decay; x_new[8] = a_s[1] * decay; x_new[9] = a_s[2] * decay;

    // Covariance propagation via first-order STM (accurate for small dt)
    let bennu_pos = bennu_heliocentric_pos(t);
    let f   = jacobian_f(&r, &bennu_pos, c_r);
    let phi = stm(&f, dt);
    let q   = process_noise(q_r, q_v, q_cr, dt);
    let p_new = phi * ekf.p * phi.transpose() + q;

    EkfState { x: x_new, p: p_new, t: t + dt }
}

// Calibration-arc process-noise densities: trust dynamics, free up C_R.
const CAL_QR:  f64 = 1e-6;   // position [m²/s³]  — ≪ SNC_QR (0.01)
const CAL_QV:  f64 = 1e-12;  // velocity [m²/s]   — ≪ SNC_QV (1e-8)
const CAL_QCR: f64 = 2e-6;   // C_R random walk [1/s] — lets C_R move ~1/day

// ── Update: OpNav bearing (3-element) with phase-angle bias correction ────────

/// EKF update step for the OpNav line-of-sight bearing measurement.
///
/// `sigma_bearing` is the actual 1-σ noise returned by the OpNav renderer
/// (shot noise + pixel floor), so R is adaptive rather than fixed.
///
/// The predicted measurement includes the intensity-weighted centroid offset
/// caused by partial illumination (phase-angle effect).  The centroid shifts
/// toward the sub-solar limb by approximately (3/8)·sin(φ)·α, where φ is the
/// Sun–Bennu–spacecraft phase angle and α = R_Bennu/range is the apparent
/// angular radius.  Correcting the predicted measurement eliminates the
/// systematic bearing bias that would otherwise appear as a false position error.
pub fn update_bearing(ekf: &EkfState, meas: &Vector3<f64>, sigma_bearing: f64) -> EkfState {
    let r     = ekf.r();
    let range = r.norm();
    let los_geom = -r / range; // geometric disk-centre LOS

    // Phase-angle centroid bias correction
    let bennu_pos = bennu_heliocentric_pos(ekf.t);
    let sun_dir   = (-bennu_pos).normalize(); // Sun direction in inertial frame
    let sc_dir    = r / range;
    let cos_phi   = sun_dir.dot(&sc_dir).clamp(-1.0, 1.0);
    let phi       = cos_phi.acos();
    let alpha     = R_BENNU / range; // apparent angular radius [rad]

    // Component of Sun direction perpendicular to the geometric LOS
    // (this is the direction the centroid shifts toward).
    let sun_perp      = sun_dir - sun_dir.dot(&los_geom) * los_geom;
    let sun_perp_norm = sun_perp.norm();

    let los_pred = if sun_perp_norm > 1e-10 {
        let bias = (3.0 / 8.0) * phi.sin() * alpha * (sun_perp / sun_perp_norm);
        (los_geom + bias).normalize()
    } else {
        los_geom // Sun on LOS axis (opposition/conjunction): no in-plane bias
    };

    let innov = meas - los_pred;

    // Measurement Jacobian H (3×7): ∂los_geom/∂r (bias gradient is second-order,
    // neglected here — same H as without correction).
    let mut h = SMatrix::<f64, 3, NX>::zeros();
    for i in 0..3 {
        for j in 0..3 {
            let delta = if i == j { 1.0 } else { 0.0 };
            h[(i, j)] = -(delta - r[i] * r[j] / (range * range)) / range;
        }
    }

    let r_noise = Matrix3::identity() * (sigma_bearing * sigma_bearing);
    ekf_update(ekf, &h, &innov, &r_noise)
}

// ── Update: landmark bearing (3-element, known body-fixed feature) ────────────

/// EKF update for a single landmark line-of-sight measurement.
///
/// `lm_pos` is the landmark's inertial/Hill position at the measurement epoch
/// (already rotated from body-fixed coordinates by Bennu's spin — the filter
/// knows the catalogue and the rotation, so this is a known quantity).
/// `meas_los` is the measured unit LOS from the spacecraft to the landmark.
///
/// Unlike disk-centre bearing (LOS to Bennu's centre, `−r̂`), the predicted LOS
/// here is toward an offset point `lm_pos`, so the Jacobian sees the relative
/// vector `d = lm_pos − r`.  Tracking several landmarks at different geometry
/// triangulates the full 3-D position **including range** — no LIDAR required.
pub fn update_landmark(ekf: &EkfState, lm_pos: &Vector3<f64>, meas_los: &Vector3<f64>, sigma: f64) -> EkfState {
    let r    = ekf.r();
    let d    = lm_pos - r;          // spacecraft → landmark
    let dist = d.norm();
    if dist < 1.0 { return ekf.clone(); }
    let los_pred = d / dist;

    let innov = meas_los - los_pred;

    // H (3×7): ∂los_pred/∂r.  los_pred = d/|d|, d = lm_pos − r ⇒ ∂d/∂r = −I.
    // ∂los/∂r = −(1/|d|)(I − los·losᵀ).
    let mut h = SMatrix::<f64, 3, NX>::zeros();
    for i in 0..3 {
        for j in 0..3 {
            let delta = if i == j { 1.0 } else { 0.0 };
            h[(i, j)] = -(delta - los_pred[i] * los_pred[j]) / dist;
        }
    }

    let r_noise = Matrix3::identity() * (sigma * sigma);
    ekf_update(ekf, &h, &innov, &r_noise)
}

// ── Update: OpNav angular size (scalar) ──────────────────────────────────────

/// EKF update step for the OpNav angular-size measurement.
///
/// Measurement model: y = R_Bennu / |r|
pub fn update_angular_size(ekf: &EkfState, meas: f64) -> EkfState {
    let r     = ekf.r();
    let range = r.norm();
    let y_pred = R_BENNU / range;
    let innov = nalgebra::SVector::<f64, 1>::new(meas - y_pred);

    let mut h = SMatrix::<f64, 1, NX>::zeros();
    for j in 0..3 {
        h[(0, j)] = -R_BENNU * r[j] / (range * range * range);
    }

    let r_noise = SMatrix::<f64, 1, 1>::new(OPNAV_SIZE_SIGMA_RAD * OPNAV_SIZE_SIGMA_RAD);
    ekf_update(ekf, &h, &innov, &r_noise)
}

// ── Update: LIDAR altimetry (scalar) ─────────────────────────────────────────

/// EKF update step for a LIDAR altimeter measurement.
///
/// The LIDAR fires along the spacecraft boresight (body-x = −r̂ for Nadir pointing)
/// and returns the slant range to Bennu's nearest surface point.
/// For a spherical Bennu this is: y = |r| − R_Bennu.
///
/// This is a direct radial range measurement and dramatically reduces the range
/// uncertainty that bearing-only OpNav cannot constrain.
pub fn update_lidar(ekf: &EkfState, alt_meas: f64) -> EkfState {
    let r     = ekf.r();
    let range = r.norm();
    let r_hat = r / range;

    let y_pred = range - R_BENNU;
    let innov  = nalgebra::SVector::<f64, 1>::new(alt_meas - y_pred);

    // H (1×7): ∂(|r| - R_BENNU)/∂r = r̂,  other states = 0
    let mut h = SMatrix::<f64, 1, NX>::zeros();
    h[(0, 0)] = r_hat[0];
    h[(0, 1)] = r_hat[1];
    h[(0, 2)] = r_hat[2];

    let r_noise = SMatrix::<f64, 1, 1>::new(LIDAR_SIGMA_M * LIDAR_SIGMA_M);
    ekf_update(ekf, &h, &innov, &r_noise)
}

// ── Update: IMU ΔV (3-element) ────────────────────────────────────────────────

/// EKF update step for an IMU ΔV measurement (3-element).
///
/// Measurement model: y = Δv_inertial = Δv applied to velocity state.
pub fn update_dv(ekf: &EkfState, dv_inertial_meas: &Vector3<f64>, dv_commanded: &Vector3<f64>) -> EkfState {
    let innov = dv_inertial_meas - dv_commanded;

    let mut h = SMatrix::<f64, 3, NX>::zeros();
    h[(0, 3)] = 1.0;
    h[(1, 4)] = 1.0;
    h[(2, 5)] = 1.0;

    let r_noise = Matrix3::identity() * (IMU_DV_SIGMA_MPS * IMU_DV_SIGMA_MPS);
    ekf_update(ekf, &h, &innov, &r_noise)
}

// ── Update: DSN heliocentric state ────────────────────────────────────────────

/// EKF update step for a DSN ground-tracking pass.
///
/// The DSN delivers the spacecraft's heliocentric position and velocity after
/// orbit determination.  The Hill-frame EKF state is related to the heliocentric
/// state by:
///   r_sc_helio = r_bennu_helio + r_ekf
///   v_sc_helio = v_bennu_helio + v_ekf
///
/// The measurement noise includes both the DSN intrinsic noise AND the residual
/// Bennu ephemeris uncertainty (~5 km), which limits how tightly the DSN can
/// constrain the Hill-frame position on a single pass.  Over multiple passes the
/// combined filter converges as OpNav progressively refines the Bennu-relative
/// state while DSN constrains the absolute heliocentric solution.
pub fn update_dsn(
    ekf:           &EkfState,
    r_sc_helio:    &Vector3<f64>,
    v_sc_helio:    &Vector3<f64>,
) -> EkfState {
    // Bennu heliocentric position and velocity (Keplerian) at current epoch
    let bp0 = bennu_heliocentric_pos(ekf.t);
    let bp1 = bennu_heliocentric_pos(ekf.t + 10.0);
    let r_bennu = Vector3::new(bp0[0], bp0[1], bp0[2]);
    let v_bennu = Vector3::new(
        (bp1[0] - bp0[0]) / 10.0,
        (bp1[1] - bp0[1]) / 10.0,
        (bp1[2] - bp0[2]) / 10.0,
    );

    // ── Position update ───────────────────────────────────────────────────────
    // y = r_bennu + r_ekf   →   H_r[:3, :3] = I₃
    let innov_r = r_sc_helio - (r_bennu + ekf.r());
    let mut h_r = SMatrix::<f64, 3, NX>::zeros();
    h_r[(0,0)] = 1.0; h_r[(1,1)] = 1.0; h_r[(2,2)] = 1.0;

    // DSN measurement is limited by Bennu ephemeris uncertainty, not hardware noise.
    let sigma_r_total = (DSN_RANGE_SIGMA_M.powi(2) + BENNU_EPHEM_SIGMA_M.powi(2)).sqrt();
    let r_noise_r = Matrix3::identity() * sigma_r_total.powi(2);

    let ekf_r = ekf_update(ekf, &h_r, &innov_r, &r_noise_r);

    // ── Velocity update ───────────────────────────────────────────────────────
    // y = v_bennu + v_ekf   →   H_v[:3, 3:6] = I₃
    let innov_v = v_sc_helio - (v_bennu + ekf_r.v());
    let mut h_v = SMatrix::<f64, 3, NX>::zeros();
    h_v[(0,3)] = 1.0; h_v[(1,4)] = 1.0; h_v[(2,5)] = 1.0;

    let sigma_v_total = (DSN_VELOCITY_SIGMA_MPS.powi(2) + DSN_BENNU_V_SIGMA_MPS.powi(2)).sqrt();
    let r_noise_v = Matrix3::identity() * sigma_v_total.powi(2);

    ekf_update(&ekf_r, &h_v, &innov_v, &r_noise_v)
}

// ── Generic EKF update kernel ─────────────────────────────────────────────────

fn ekf_update<const M: usize>(
    ekf:     &EkfState,
    h:       &SMatrix<f64, M, NX>,
    innov:   &SVector<f64, M>,
    r_noise: &SMatrix<f64, M, M>,
) -> EkfState {
    let ph_t = ekf.p * h.transpose();
    let s    = h * ph_t + r_noise;

    let s_inv = s.try_inverse().unwrap_or_else(|| SMatrix::<f64, M, M>::identity());
    let k = ph_t * s_inv;

    let x_new = ekf.x + k * innov;

    // Joseph form for numerical stability
    let i_kh  = StateMat::identity() - k * h;
    let p_new = i_kh * ekf.p * i_kh.transpose() + k * r_noise * k.transpose();

    EkfState { x: x_new, p: p_new, t: ekf.t }
}

// ── Private: RK4 integration of filter-model EOM ──────────────────────────────

fn rk4_filter_state(
    r0:    Vector3<f64>,
    v0:    Vector3<f64>,
    c_r:   f64,
    a_stoch: &Vector3<f64>,
    t0:    f64,
    dt:    f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let rhs = |r: Vector3<f64>, v: Vector3<f64>, t: f64| -> (Vector3<f64>, Vector3<f64>) {
        let bp = bennu_heliocentric_pos(t);
        // Known forces (gravity + tidal + cannonball SRP) plus the estimated
        // stochastic acceleration held constant over the step.
        let a  = accel_filter(&r, &bp, c_r) + a_stoch;
        (v, a)
    };

    let (k1r, k1v) = rhs(r0, v0, t0);
    let r2 = r0 + k1r * (dt * 0.5); let v2 = v0 + k1v * (dt * 0.5);
    let (k2r, k2v) = rhs(r2, v2, t0 + dt * 0.5);
    let r3 = r0 + k2r * (dt * 0.5); let v3 = v0 + k2v * (dt * 0.5);
    let (k3r, k3v) = rhs(r3, v3, t0 + dt * 0.5);
    let r4 = r0 + k3r * dt;          let v4 = v0 + k3v * dt;
    let (k4r, k4v) = rhs(r4, v4, t0 + dt);

    (
        r0 + (dt / 6.0) * (k1r + 2.0*k2r + 2.0*k3r + k4r),
        v0 + (dt / 6.0) * (k1v + 2.0*k2v + 2.0*k3v + k4v),
    )
}
