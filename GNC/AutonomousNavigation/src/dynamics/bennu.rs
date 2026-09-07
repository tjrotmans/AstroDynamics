//! Bennu gravity model and Keplerian ephemeris.
//!
//! Gravity is a zonal spherical-harmonic field through degree 4 (J2, J3, J4),
//! with coefficients from Scheeres et al. 2020 (Sci. Adv. 6, abc3350 — particle
//! solution).  Both the truth propagator and the EKF predict step share this
//! field; the residual signal from the unmodelled tesseral/sectoral harmonics
//! (not publicly tabulated) is absorbed by the filter's process noise.

use nalgebra::Vector3;
use orbital_models::GravityModel;
use orbital_models::constants::MU_SUN;
use orbital_math::kepler::{solve_kepler, perifocal_to_ecliptic};
use crate::config::{MU_BENNU, BENNU_A, BENNU_E, BENNU_I,
                    BENNU_RAAN, BENNU_AOP, BENNU_M0, BENNU_EPOCH_S,
                    BENNU_GRAVITY_R0_M, BENNU_J2, BENNU_J3, BENNU_J4};

// ── Point-mass gravity ────────────────────────────────────────────────────────

/// Acceleration from Bennu point-mass gravity [m/s²], given position [m].
#[inline]
pub fn pm_gravity(r_vec: &Vector3<f64>) -> Vector3<f64> {
    let r2 = r_vec.norm_squared();
    let r  = r2.sqrt();
    -(MU_BENNU / (r2 * r)) * r_vec
}

// ── Zonal gravity through degree 4 (J2, J3, J4) ──────────────────────────────

/// Acceleration from Bennu gravity through degree 4 [m/s²].
///
/// Point-mass term plus J₂, J₃, J₄ zonal perturbations, using
/// `GravityModel::zonal_harmonics_body` from `orbital_models` with Bennu's
/// coefficients (Scheeres et al. 2020).  The body spin pole is aligned with +z.
pub fn zonal_gravity(r_vec: &Vector3<f64>) -> Vector3<f64> {
    pm_gravity(r_vec)
        + GravityModel::zonal_harmonics_body(
            r_vec, MU_BENNU, BENNU_GRAVITY_R0_M, BENNU_J2, BENNU_J3, BENNU_J4,
        )
}

// ── Keplerian propagation of Bennu around the Sun ────────────────────────────

/// Heliocentric position of Bennu [m] at time `t` seconds after J2000.
///
/// Uses a simple Kepler-equation solver; accurate to < 100 km over ~1 day.
pub fn bennu_heliocentric_pos(t: f64) -> Vector3<f64> {
    let dt = t - BENNU_EPOCH_S;
    let n  = (MU_SUN / (BENNU_A * BENNU_A * BENNU_A)).sqrt(); // mean motion [rad/s]
    let m  = BENNU_M0 + n * dt;                               // mean anomaly [rad]

    let e_anom = solve_kepler(m, BENNU_E, 1e-12);

    // True anomaly
    let nu = 2.0 * f64::atan2(
        ((1.0 + BENNU_E) / (1.0 - BENNU_E)).sqrt() * (e_anom / 2.0).sin(),
        (e_anom / 2.0).cos(),
    );

    let r_orb = BENNU_A * (1.0 - BENNU_E * e_anom.cos());

    // Perifocal coordinates
    let x_peri = r_orb * nu.cos();
    let y_peri = r_orb * nu.sin();

    // Rotate to ecliptic J2000
    perifocal_to_ecliptic(x_peri, y_peri, BENNU_I, BENNU_RAAN, BENNU_AOP)
}
