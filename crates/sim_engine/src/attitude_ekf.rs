//! Attitude estimation — Multiplicative Extended Kalman Filter (MEKF),
//! Phase 13h / `docs/MP/MANUAL.md` §12.3.
//!
//! No attitude estimator existed anywhere in this codebase before this
//! module — both the Bennu reference (`GNC/AutonomousNavigation`) and
//! `sim_engine` up to this point used the raw noisy star-tracker quaternion
//! directly as "the" attitude. This is the real gap that closes: gyro
//! measurements (`sensors::gyro`) are integrated between star-tracker
//! updates (`sensors::star_tracker`) to propagate a filtered attitude
//! estimate `q_hat` and a gyro bias estimate `bias_hat`, with each
//! star-tracker measurement correcting both.
//!
//! ## Model (Farrenkopf 1978; Lefferts, Markley & Shuster 1982; Markley &
//! Crassidis, *Fundamentals of Spacecraft Attitude Determination and
//! Control*, ch. 6 — the standard gyro-driven MEKF)
//!
//! **Error-state definition.** The true attitude is related to the estimate
//! by a small BODY-FRAME rotation composed on the right,
//! `q_true = q_hat ⊗ δq`, `δq ≈ [1, δα/2]` for small `δα` — chosen to match
//! `sensors::star_tracker::measure`'s own noise model, which composes its
//! noise the same way (`q_meas = q_true ⊗ δq_noise`). The 6-element error
//! state is `[δα (3, small-angle attitude error), δb (3, gyro bias error)]`
//! — never propagated explicitly as a nonzero vector; instead `q_hat`/
//! `bias_hat` absorb every correction immediately (the "reset" step that
//! gives the multiplicative EKF its name), so only its covariance `p`
//! carries state forward between updates.
//!
//! **Propagation (gyro-driven).** Between star-tracker updates:
//! ```text
//! omega_hat = omega_meas - bias_hat                     (bias-corrected rate)
//! q_hat_new = q_hat ⊗ exp([0, omega_hat * dt / 2])       (exact for constant omega_hat over dt)
//! bias_hat_new = bias_hat                                (bias is a pure random walk -- mean unchanged)
//! ```
//! The quaternion update uses the closed-form rotation-vector exponential
//! (exact for a constant body rate over `dt`), not a generic RK4 `qdot`
//! integration — the standard, simpler choice for this specific step
//! (Markley & Crassidis §3.5).
//!
//! **Linearized error dynamics** (standard result, convention-independent
//! to first order — this is the same equation pair already summarized in
//! `sensors::gyro`'s own doc comment and MANUAL.md §11.2/§12.3):
//! ```text
//! delta_alpha_dot = -omega_hat x delta_alpha - delta_b - eta_v
//! delta_b_dot     = eta_u
//! ```
//! Discretized here with a first-order (Euler) state-transition matrix
//! `Phi ~= I6 + F*dt` — a deliberate simplification over the exact
//! closed-form `Phi` some MEKF texts give (Lefferts-Markley-Shuster's own
//! series expansion), chosen per this phase's "keep the math simple"
//! directive; valid for `dt` small relative to `1/|omega_hat|`, which holds
//! for any gyro sampled meaningfully faster than the vehicle's own rotation
//! rate. Process noise `Q` uses the standard closed-form two-state
//! (ARW/RRW) result also cited in `sensors::gyro` (Farrenkopf 1978):
//! ```text
//! Q_aa =  (sigma_v^2 * dt + sigma_u^2 * dt^3 / 3) * I3
//! Q_ab = -(sigma_u^2 * dt^2 / 2) * I3
//! Q_bb =  sigma_u^2 * dt * I3
//! ```
//!
//! **Known, documented limitation, found via `attitude_mekf_demo`
//!: reported covariance is optimistic (real error exceeds the
//! reported 1-sigma bound by up to ~9x) during AGGRESSIVE closed-loop
//! slews (tens of degrees over a handful of ticks), settling to a modest,
//! explainable ~2-6x during quiescent/converged tracking.** Root cause,
//! confirmed by halving the demo's control tick from 5 s to 1 s (which
//! cut the worst-case ratio from ~50x to ~9x): the first-order `Phi`/`Q`
//! discretization above assumes `omega_hat` is ~constant over `dt` (both
//! for the exact rotation-vector `q_hat` update, which truncates real
//! within-tick rate CHANGE under active torque, and for the linearized
//! error covariance). A fast slew violates that assumption more than a
//! quiescent coast does. This is exactly the validity boundary already
//! flagged above ("valid for `dt` small relative to `1/|omega_hat|`") --
//! confirmed real, not hypothetical, by this demo. The clean unit tests in
//! this module (constant-rate, no aggressive maneuvering) are NOT affected
//! by this and remain the correct check that the underlying filter
//! equations themselves are right. Not fixed here -- the standard fix
//! (Lefferts-Markley-Shuster's closed-form `Phi`, or a finer gyro-only
//! sub-step within each control tick) is a real future improvement, kept
//! simple for this first version per this phase's own directive; flagged
//! honestly rather than silently accepted.
//!
//! **Measurement update (star tracker).** `delta_alpha_meas = 2 * vec(delta_q_meas)`,
//! `delta_q_meas = conj(q_hat) ⊗ q_meas` (small-angle, same right-composition
//! convention as propagation) with measurement matrix `H = [I3, 0]` and
//! noise `R = sigma_star_tracker_rad^2 * I3` — an exact match to
//! `sensors::star_tracker::measure`'s own per-axis noise model, so this
//! filter's `R` is not a separate approximation of the truth sensor, it's
//! the same model. Joseph-form covariance update, same numerically-robust
//! form `sim_engine::ekf` already uses.

use nalgebra::{Matrix3, Matrix3x6, Matrix6, Matrix6x3, Vector3, Vector4, Vector6};
use orbital_models::attitude::{quat_conjugate, quat_multiply, qnorm};

/// MEKF filter state — the estimate itself (`q_hat`, `bias_hat`) plus the
/// 6x6 error-state covariance. Order: `[delta_alpha (0..3), delta_b (3..6)]`.
#[derive(Clone, Debug)]
pub struct AttitudeEkfState {
    /// Estimated attitude quaternion [w, x, y, z], body -> inertial.
    pub q_hat: Vector4<f64>,
    /// Estimated gyro bias, body frame [rad/s].
    pub bias_hat: Vector3<f64>,
    pub p: Matrix6<f64>,
    pub t_s: f64,
}

impl AttitudeEkfState {
    /// 1-sigma attitude uncertainty (RSS of the diagonal), for telemetry —
    /// same "sigma-from-trace" convention `sim_engine::ekf::EkfState`
    /// already uses for position/velocity.
    pub fn sigma_alpha_rad(&self) -> f64 {
        (self.p[(0, 0)] + self.p[(1, 1)] + self.p[(2, 2)]).sqrt()
    }
    pub fn sigma_bias_rad_s(&self) -> f64 {
        (self.p[(3, 3)] + self.p[(4, 4)] + self.p[(5, 5)]).sqrt()
    }
}

/// Build the initial filter state from an a-priori attitude/bias estimate
/// and 1-sigma uncertainties (diagonal initial covariance).
pub fn init(
    q0_hat: Vector4<f64>,
    bias0_hat: Vector3<f64>,
    sigma_alpha0_rad: f64,
    sigma_bias0_rad_s: f64,
    t0_s: f64,
) -> AttitudeEkfState {
    let mut p = Matrix6::zeros();
    for i in 0..3 {
        p[(i, i)] = sigma_alpha0_rad * sigma_alpha0_rad;
    }
    for i in 3..6 {
        p[(i, i)] = sigma_bias0_rad_s * sigma_bias0_rad_s;
    }
    AttitudeEkfState { q_hat: qnorm(&q0_hat), bias_hat: bias0_hat, p, t_s: t0_s }
}

/// Skew-symmetric cross-product matrix `[v x]`.
fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0,
    )
}

/// Small-angle rotation-vector quaternion `exp([0, phi/2])` — exact for a
/// constant body rate `phi/dt` held over `dt` (`phi = omega * dt`).
fn rotation_vector_quat(phi: &Vector3<f64>) -> Vector4<f64> {
    let angle = phi.norm();
    if angle < 1e-12 {
        // Small-angle limit: sin(angle/2)/angle -> 1/2.
        Vector4::new(1.0, phi.x * 0.5, phi.y * 0.5, phi.z * 0.5)
    } else {
        let half = angle * 0.5;
        let s = half.sin() / angle;
        Vector4::new(half.cos(), phi.x * s, phi.y * s, phi.z * s)
    }
}

/// Propagate the filter by `dt_s` using a gyro rate measurement — see this
/// module's doc comment for the governing equations. `sigma_v_arw`/
/// `sigma_u_rrw` are the same ARW/RRW coefficients `sensors::gyro::measure`
/// takes (`hardware_catalog::GyroSpec`).
pub fn propagate(
    state: &AttitudeEkfState,
    omega_meas: &Vector3<f64>,
    dt_s: f64,
    sigma_v_arw: f64,
    sigma_u_rrw: f64,
) -> AttitudeEkfState {
    let omega_hat = omega_meas - state.bias_hat;

    let dq = rotation_vector_quat(&(omega_hat * dt_s));
    let q_hat = qnorm(&quat_multiply(&state.q_hat, &dq));
    let bias_hat = state.bias_hat; // pure random walk: predicted mean unchanged

    // First-order discretization of the linearized error dynamics -- see
    // module doc comment for the deliberate simplification this is.
    let neg_skew_dt = -skew(&omega_hat) * dt_s;
    let mut phi = Matrix6::<f64>::identity();
    for i in 0..3 {
        for j in 0..3 {
            phi[(i, j)] += neg_skew_dt[(i, j)];
        }
        phi[(i, i + 3)] += -dt_s; // -delta_b term
    }

    let dt2 = dt_s * dt_s;
    let dt3 = dt2 * dt_s;
    let q_aa = sigma_v_arw * sigma_v_arw * dt_s + sigma_u_rrw * sigma_u_rrw * dt3 / 3.0;
    let q_ab = -sigma_u_rrw * sigma_u_rrw * dt2 / 2.0;
    let q_bb = sigma_u_rrw * sigma_u_rrw * dt_s;
    let mut q = Matrix6::zeros();
    for i in 0..3 {
        q[(i, i)] = q_aa;
        q[(i, i + 3)] = q_ab;
        q[(i + 3, i)] = q_ab;
        q[(i + 3, i + 3)] = q_bb;
    }

    let p = phi * state.p * phi.transpose() + q;

    AttitudeEkfState { q_hat, bias_hat, p, t_s: state.t_s + dt_s }
}

/// Star-tracker measurement update. `q_meas` is a noisy attitude
/// measurement in the same convention `sensors::star_tracker::measure`
/// produces. Returns the corrected state and the residual (attitude error)
/// norm [rad], for telemetry (same `(State, residual_norm)` convention
/// `sim_engine::ekf`'s `update_*` functions already use).
pub fn update_star_tracker(
    state: &AttitudeEkfState,
    q_meas: &Vector4<f64>,
    sigma_star_tracker_rad: f64,
) -> (AttitudeEkfState, f64) {
    let dq_meas = quat_multiply(&quat_conjugate(&state.q_hat), q_meas);
    // Small-angle extraction: delta_alpha = 2 * vec(dq), sign-corrected so
    // the residual is always the shortest-path rotation (dq.w >= 0).
    let sign = if dq_meas[0] >= 0.0 { 1.0 } else { -1.0 };
    let delta_alpha_meas = 2.0 * sign * Vector3::new(dq_meas[1], dq_meas[2], dq_meas[3]);

    let r = Matrix3::identity() * (sigma_star_tracker_rad * sigma_star_tracker_rad);
    // H = [I3, 0] -- P*H^T is just P's left 3 columns (6x3); H*P*H^T is
    // P's top-left 3x3 (attitude-attitude) block. Built via explicit loops
    // rather than fixed-size slicing, to avoid depending on a specific
    // nalgebra minor version's slicing API.
    let mut p_ht = Matrix6x3::<f64>::zeros();
    let mut p_aa = Matrix3::<f64>::zeros();
    for i in 0..6 {
        for j in 0..3 {
            p_ht[(i, j)] = state.p[(i, j)];
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            p_aa[(i, j)] = state.p[(i, j)];
        }
    }
    let s = p_aa + r;
    let s_inv = s.try_inverse().expect("innovation covariance not invertible");
    let k = p_ht * s_inv; // 6x3

    let dx: Vector6<f64> = k * delta_alpha_meas;
    let delta_alpha_hat = Vector3::new(dx[0], dx[1], dx[2]);
    let delta_bias_hat = Vector3::new(dx[3], dx[4], dx[5]);

    let q_hat = qnorm(&quat_multiply(&state.q_hat, &rotation_vector_quat(&delta_alpha_hat)));
    let bias_hat = state.bias_hat + delta_bias_hat;

    // Joseph-form covariance update -- same numerically-robust form
    // sim_engine::ekf::ekf_update uses.
    let mut h = Matrix3x6::<f64>::zeros();
    for i in 0..3 {
        h[(i, i)] = 1.0;
    }
    let ikh = Matrix6::identity() - k * h;
    let p = ikh * state.p * ikh.transpose() + k * r * k.transpose();

    (AttitudeEkfState { q_hat, bias_hat, p, t_s: state.t_s }, delta_alpha_meas.norm())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    fn identity_q() -> Vector4<f64> {
        Vector4::new(1.0, 0.0, 0.0, 0.0)
    }

    /// With zero gyro noise/bias and zero rate, propagation should leave
    /// q_hat exactly at identity and grow covariance only via process noise
    /// (never shrink -- pure propagation never reduces uncertainty).
    #[test]
    fn propagate_at_rest_holds_attitude_and_grows_covariance() {
        let s0 = init(identity_q(), Vector3::zeros(), 1e-4, 1e-8, 0.0);
        let s1 = propagate(&s0, &Vector3::zeros(), 1.0, 1e-6, 1e-9);
        assert!((s1.q_hat - identity_q()).norm() < 1e-12);
        assert!(s1.sigma_alpha_rad() >= s0.sigma_alpha_rad());
        assert!(s1.sigma_bias_rad_s() >= s0.sigma_bias_rad_s());
    }

    /// A constant true rate, gyro reading it exactly (no noise), should
    /// rotate q_hat by the correct angle after propagation -- checked
    /// against the closed-form rotation-vector quaternion directly.
    #[test]
    fn propagate_with_constant_rate_rotates_by_the_correct_angle() {
        let s0 = init(identity_q(), Vector3::zeros(), 1e-4, 1e-8, 0.0);
        let omega = Vector3::new(0.1, 0.0, 0.0); // rad/s about body x
        let dt = 2.0;
        let s1 = propagate(&s0, &omega, dt, 0.0, 0.0);
        let expected = rotation_vector_quat(&(omega * dt));
        assert!((s1.q_hat - expected).norm() < 1e-9, "got {:?} expected {:?}", s1.q_hat, expected);
    }

    /// A nonzero bias estimate must be subtracted from the gyro reading
    /// before propagation -- a gyro reading that exactly equals the
    /// (wrongly) assumed bias should propagate q_hat as if at rest.
    #[test]
    fn propagate_subtracts_the_bias_estimate() {
        let bias = Vector3::new(0.05, 0.0, 0.0);
        let s0 = init(identity_q(), bias, 1e-4, 1e-8, 0.0);
        let s1 = propagate(&s0, &bias, 1.0, 0.0, 0.0); // omega_meas == bias_hat
        assert!((s1.q_hat - identity_q()).norm() < 1e-9, "expected no rotation, got {:?}", s1.q_hat);
    }

    /// A star-tracker measurement should correct q_hat toward the
    /// measurement and shrink the attitude covariance (a real update, not a
    /// no-op).
    #[test]
    fn update_star_tracker_corrects_toward_measurement_and_shrinks_covariance() {
        let s0 = init(identity_q(), Vector3::zeros(), 1e-2, 1e-6, 0.0);
        // A small, exactly-known rotation as the "measurement" -- q_hat
        // should move toward it, not stay at identity.
        let phi_true = Vector3::new(0.01, 0.0, 0.0);
        let q_meas = rotation_vector_quat(&phi_true);
        let (s1, resid) = update_star_tracker(&s0, &q_meas, 1e-3);

        assert!(resid > 0.0, "residual should be nonzero for a real measurement offset");
        // q_hat should have moved toward q_meas, i.e. closer to it than the
        // prior (identity) was.
        let dist_before = (identity_q() - q_meas).norm();
        let dist_after = (s1.q_hat - q_meas).norm();
        assert!(dist_after < dist_before, "update should move q_hat toward the measurement");
        assert!(s1.sigma_alpha_rad() < s0.sigma_alpha_rad(), "a real update should shrink attitude uncertainty");
    }

    /// End-to-end sanity: propagate a constant true rate with a REAL noisy
    /// gyro + star tracker (via the actual sensor models, not idealized
    /// inputs) over many steps; the filter's final attitude error should be
    /// small and, critically, smaller than simply trusting the raw noisy
    /// star-tracker reading directly (the entire point of running a filter
    /// instead of using the raw measurement) -- checked via RMS over many
    /// independent seeds since this is a stochastic comparison.
    #[test]
    fn filtered_attitude_error_beats_the_raw_star_tracker_reading_on_average() {
        use crate::sensors::{gyro, star_tracker};

        let omega_true = Vector3::new(0.02, -0.01, 0.005);
        let dt = 0.1;
        let n_steps = 200; // 20 s
        let star_tracker_sigma = 5e-4; // rad, fine-grade-ish
        let arw = 5.82e-7_f64;
        let rrw = 8.08e-10_f64;

        let mut filtered_sq_err = 0.0_f64;
        let mut raw_sq_err = 0.0_f64;
        let n_trials = 8;
        for trial in 0..n_trials {
            let mut rng = StdRng::seed_from_u64(1000 + trial);
            let mut bias = gyro::GyroBiasState::zero();
            let mut q_true = identity_q();
            let mut ekf = init(identity_q(), Vector3::zeros(), 1e-3, 1e-8, 0.0);
            let mut last_raw_q = identity_q();

            for i in 0..n_steps {
                q_true = qnorm(&quat_multiply(&q_true, &rotation_vector_quat(&(omega_true * dt))));
                let omega_meas = gyro::measure(&omega_true, &mut bias, rrw, arw, dt, &mut rng);
                ekf = propagate(&ekf, &omega_meas, dt, arw, rrw);

                // Star-tracker update every 10 steps (1 Hz-ish cadence).
                if i % 10 == 9 {
                    let q_meas = star_tracker::measure(&q_true, star_tracker_sigma, &mut rng);
                    let (new_ekf, _) = update_star_tracker(&ekf, &q_meas, star_tracker_sigma);
                    ekf = new_ekf;
                    last_raw_q = q_meas;
                }
            }

            let filt_err = quat_multiply(&quat_conjugate(&q_true), &ekf.q_hat);
            let raw_err = quat_multiply(&quat_conjugate(&q_true), &last_raw_q);
            let filt_angle = 2.0 * filt_err[0].clamp(-1.0, 1.0).acos().min(std::f64::consts::PI - filt_err[0].clamp(-1.0, 1.0).acos());
            let raw_angle = 2.0 * raw_err[0].clamp(-1.0, 1.0).acos().min(std::f64::consts::PI - raw_err[0].clamp(-1.0, 1.0).acos());
            filtered_sq_err += filt_angle * filt_angle;
            raw_sq_err += raw_angle * raw_angle;
        }

        let filtered_rms = (filtered_sq_err / n_trials as f64).sqrt();
        let raw_rms = (raw_sq_err / n_trials as f64).sqrt();
        assert!(
            filtered_rms < raw_rms,
            "filtered RMS attitude error ({filtered_rms}) should beat the raw last star-tracker reading's own error ({raw_rms}) -- gyro propagation between updates should add real information"
        );
    }
}
