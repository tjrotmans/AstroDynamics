//! State Transition Matrix (STM) for the EKF predict step.
//!
//! EKF state: X = [r(3), v(3), C_SRP(1), a_stoch(3)]  — 10 elements.
//!
//! `a_stoch` is a 3-axis **stochastic acceleration** modelled as a first-order
//! Gauss-Markov (Ornstein-Uhlenbeck) process: `ȧ = −a/τ + w`.  It captures the
//! time-correlated nongravitational forces the cannonball SRP + scalar `C_R`
//! cannot explain — chiefly the attitude-dependent panel-SRP residual, plus
//! thermal re-radiation and outgassing — exactly as the real OSIRIS-REx OD
//! filter does (3-axis Markov accel, ~1-day correlation; Williams et al. 2018).
//!
//! The STM Φ (10×10) is computed analytically to first order:
//!
//!   Φ = I + F·dt   (first-order truncation, valid for small dt)
//!
//! where F = ∂f/∂X is the Jacobian of the continuous EOM w.r.t. the state.
//!
//! Covariance propagation: P⁻ = Φ P Φᵀ + Q  (process noise, incl. the Markov term).

use nalgebra::{SMatrix, SVector};
use orbital_models::constants::{AU, P_SRP};
use crate::config::{MU_BENNU, SC_AREA_CANNONBALL, SC_MASS};

/// State dimension (r + v + C_SRP + a_stoch)
pub const NX: usize = 10;

/// Gauss-Markov correlation time for the stochastic acceleration [s] (~1 day,
/// matching the OSIRIS-REx OD "1-day batch" stochastic-accel model).
pub const MARKOV_TAU_S: f64 = 86_400.0;

/// Stationary 1-σ of the stochastic acceleration per axis [m/s²].
/// OSIRIS-REx Orbit-B used ~3 nm/s² process noise for the nongravitational
/// stochastic accelerations; we adopt the same scale.
pub const MARKOV_SIGMA_A: f64 = 3.0e-9;

pub type StateVec = SVector<f64, NX>;
pub type StateMat = SMatrix<f64, NX, NX>;

/// Compute the linearised dynamics Jacobian F = ∂f/∂X at the current state.
///
/// `r`         = position w.r.t. Bennu [m]
/// `bennu_pos` = Bennu heliocentric position [m]  (for SRP partial)
/// `c_r`       = current C_SRP estimate
pub fn jacobian_f(
    r:         &nalgebra::Vector3<f64>,
    bennu_pos: &nalgebra::Vector3<f64>,
    _c_r:       f64,
) -> StateMat {
    let mut f = StateMat::zeros();

    // ∂ṙ/∂v = I₃
    f[(0, 3)] = 1.0;
    f[(1, 4)] = 1.0;
    f[(2, 5)] = 1.0;

    // ∂v̇/∂r: gravity gradient (point mass) + tidal gradient
    let r2 = r.norm_squared();
    let r3 = r2.powf(1.5);

    // Gravity gradient ∂a_grav/∂r = -μ/r³ (I - 3 r̂r̂ᵀ)
    for i in 0..3 {
        for j in 0..3 {
            let delta = if i == j { 1.0 } else { 0.0 };
            f[(3 + i, j)] += -MU_BENNU / r3 * (delta - 3.0 * r[i] * r[j] / r2);
        }
    }

    // Tidal gradient ∂a_tidal/∂r ≈ -μ_S/|r_B|³ I  (dominant term)
    let rb3 = bennu_pos.norm().powi(3);
    use orbital_models::constants::MU_SUN;
    for i in 0..3 {
        f[(3 + i, i)] += -MU_SUN / rb3;
    }

    // ∂v̇/∂C_SRP: partial of SRP acceleration w.r.t. C_R
    let r_sun  = bennu_pos;
    let dist   = r_sun.norm();
    let au_r   = AU / dist;
    let p_srp  = P_SRP * au_r * au_r;
    let srp_a  = p_srp * SC_AREA_CANNONBALL / SC_MASS;
    let sun_hat = r_sun / dist;
    f[(3, 6)] = srp_a * sun_hat[0];
    f[(4, 6)] = srp_a * sun_hat[1];
    f[(5, 6)] = srp_a * sun_hat[2];

    // C_SRP is modelled as a constant → ∂Ċ_SRP/∂X = 0

    // ∂v̇/∂a_stoch = I₃  (the stochastic acceleration adds directly to v̇)
    f[(3, 7)] = 1.0;
    f[(4, 8)] = 1.0;
    f[(5, 9)] = 1.0;

    // ∂ȧ_stoch/∂a_stoch = −1/τ·I₃  (Gauss-Markov decay toward zero)
    let inv_tau = -1.0 / MARKOV_TAU_S;
    f[(7, 7)] = inv_tau;
    f[(8, 8)] = inv_tau;
    f[(9, 9)] = inv_tau;

    f
}

/// First-order STM: Φ ≈ I + F·dt
pub fn stm(f: &StateMat, dt: f64) -> StateMat {
    StateMat::identity() + f * dt
}

/// Process noise matrix Γ Q Γᵀ (SNC — state noise compensation).
///
/// Models unmodelled accelerations as white noise on velocity.
/// `q_r`  = position spectral density [m²/s³]
/// `q_v`  = velocity spectral density [m²/s]
/// `q_cr` = C_SRP random-walk spectral density [1/s].  Tiny in normal operation
///          (C_R nearly constant); raised during a calibration arc so the filter
///          can drive C_R toward the value that matches the true SRP.
pub fn process_noise(q_r: f64, q_v: f64, q_cr: f64, dt: f64) -> StateMat {
    let mut q = StateMat::zeros();
    // Position noise (integrated velocity noise over dt)
    let qr = q_r * dt;
    let qv = q_v * dt;
    q[(0, 0)] = qr;  q[(1, 1)] = qr;  q[(2, 2)] = qr;
    q[(3, 3)] = qv;  q[(4, 4)] = qv;  q[(5, 5)] = qv;
    // C_SRP random walk (tiny by default; opened up during calibration)
    q[(6, 6)] = q_cr * dt;

    // Stochastic-acceleration Gauss-Markov driving noise.  The discrete process
    // noise that keeps the state at its stationary variance σ_a² is
    //   q_a = σ_a²·(1 − e^(−2dt/τ)) ≈ σ_a²·2dt/τ   for dt ≪ τ.
    let q_a = MARKOV_SIGMA_A * MARKOV_SIGMA_A * (1.0 - (-2.0 * dt / MARKOV_TAU_S).exp());
    q[(7, 7)] = q_a;  q[(8, 8)] = q_a;  q[(9, 9)] = q_a;
    q
}
