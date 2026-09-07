//! Multi-panel SRP plate model for truth dynamics.
//!
//! Provides `spacecraft_plates()` — the mission-specific geometry builder.
//! Generic `Plate` struct, force, torque, and acceleration functions come from
//! `orbital_models::acceleration` so they are not duplicated here.
//!
//! The filter models SRP as a cannonball (constant area, always anti-Sun,
//! attitude-independent) plus a scalar reflectivity `C_R`.  The truth sums the
//! force and torque over the spacecraft's actual flat surfaces — the 6 bus faces
//! and the 2 solar panels — each contributing only when sunlit.  The attitude-
//! dependent residual is what the EKF's Gauss-Markov stochastic-acceleration
//! state is designed to absorb.

use nalgebra::{Vector3, Vector4};
use orbital_models::Plate;
use orbital_models::acceleration::{pressure_at, flat_plate_accel, flat_plate_torque_body};
use orbital_models::attitude::inertial_to_body;
use crate::config::{SC_BUS_DIMS_M, SC_PANEL_AREA_M2, SC_PANEL_SPAN_M, COM_OFFSET_BODY};

// Re-export so callers don't need to import orbital_models directly.
pub use orbital_models::Plate as PlateModel;

/// Build the spacecraft plate model: 6 bus faces + 2 solar panels.
///
/// Each plate includes its centre relative to the spacecraft CoM in the body
/// frame (`center_body`), required for computing the SRP attitude torque via
/// τ_i = center_i × F_i.
///
/// Body-frame axes: +x = camera boresight (nadir), +y = solar panel span axis,
/// +z = orbit-normal / along-track.
///
/// Plate centres are computed as:
///   center_from_com = center_from_geom − COM_OFFSET_BODY
pub fn spacecraft_plates() -> Vec<Plate> {
    let [lx, ly, lz] = SC_BUS_DIMS_M;
    let [cx, cy, cz] = COM_OFFSET_BODY;
    let span = SC_PANEL_SPAN_M;

    // Bus optical properties (gold MLI blanket): moderately specular.
    let (bs, bd) = (0.30, 0.20);
    // Solar-panel optical properties (dark solar cells): mostly absorbing.
    let (ps, pd) = (0.08, 0.10);

    let panel_area = SC_PANEL_AREA_M2 / 2.0; // per panel; total is for both

    // Bus face centres from geometric origin, then subtract CoM offset.
    let c = |gx: f64, gy: f64, gz: f64| Vector3::new(gx - cx, gy - cy, gz - cz);

    vec![
        // ── Bus faces (single-sided) ──────────────────────────────────────────
        Plate { normal: Vector3::x(),  area: ly*lz, rho_s: bs, rho_d: bd,
                double_sided: false, center_body: c( lx/2.,   0.,    0.  ) },
        Plate { normal: -Vector3::x(), area: ly*lz, rho_s: bs, rho_d: bd,
                double_sided: false, center_body: c(-lx/2.,   0.,    0.  ) },
        Plate { normal: Vector3::y(),  area: lx*lz, rho_s: bs, rho_d: bd,
                double_sided: false, center_body: c(  0.,   ly/2.,   0.  ) },
        Plate { normal: -Vector3::y(), area: lx*lz, rho_s: bs, rho_d: bd,
                double_sided: false, center_body: c(  0.,  -ly/2.,   0.  ) },
        Plate { normal: Vector3::z(),  area: lx*ly, rho_s: bs, rho_d: bd,
                double_sided: false, center_body: c(  0.,    0.,   lz/2. ) },
        Plate { normal: -Vector3::z(), area: lx*ly, rho_s: bs, rho_d: bd,
                double_sided: false, center_body: c(  0.,    0.,  -lz/2. ) },
        // ── Solar panels (double-sided, normal ±z, centred at ±y panel midpoint)
        Plate { normal: Vector3::z(), area: panel_area, rho_s: ps, rho_d: pd,
                double_sided: true,
                center_body: c(0., ly/2. + span/2., 0.) },
        Plate { normal: Vector3::z(), area: panel_area, rho_s: ps, rho_d: pd,
                double_sided: true,
                center_body: c(0., -(ly/2. + span/2.), 0.) },
    ]
}

/// Truth-model SRP **acceleration** in the Hill/inertial frame [m/s²].
///
/// `q`         = attitude quaternion (body → Hill),
/// `bennu_pos` = Bennu heliocentric position [m] (gives Sun direction + distance),
/// `mass`      = spacecraft mass [kg].
pub fn srp_accel_panels(
    plates:    &[Plate],
    q:         &Vector4<f64>,
    bennu_pos: &Vector3<f64>,
    mass:      f64,
) -> Vector3<f64> {
    // Sun is opposite Bennu's heliocentric direction; spacecraft→Sun ≈ −r̂_bennu.
    let sun_hat_hill = (-bennu_pos).normalize();
    let p_srp = pressure_at(bennu_pos.norm());
    flat_plate_accel(plates, q, &sun_hat_hill, p_srp, mass)
}

/// Truth-model SRP **torque** in the body frame [N·m].
///
/// Uses the per-plate lever arm (center_body × F_plate) for physically correct
/// attitude-dependent torque.  Only call from the attitude dynamics integrator.
pub fn srp_torque_panels(
    plates:    &[Plate],
    q:         &Vector4<f64>,
    bennu_pos: &Vector3<f64>,
) -> Vector3<f64> {
    let sun_hat_hill = (-bennu_pos).normalize();
    let sun_hat_body = inertial_to_body(q, &sun_hat_hill);
    let p_srp = pressure_at(bennu_pos.norm());
    flat_plate_torque_body(plates, &sun_hat_body, p_srp)
}

/// Cannonball SRP acceleration in the Hill frame [m/s²] — the filter's simplified model.
///
/// Provided for logging the truth-vs-filter residual that the Gauss-Markov
/// stochastic-acceleration state is meant to absorb.
pub fn srp_accel_cannonball(bennu_pos: &Vector3<f64>, c_r: f64, area: f64, mass: f64) -> Vector3<f64> {
    let away_from_sun = bennu_pos / bennu_pos.norm();
    pressure_at(bennu_pos.norm()) * c_r * area / mass * away_from_sun
}
