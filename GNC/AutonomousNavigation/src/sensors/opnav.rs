//! Optical navigation (OpNav) camera measurement model.
//!
//! Two measurements per image:
//!   Y1 — bearing: unit LOS vector to Bennu centroid in **inertial** frame.
//!   Y2 — angular size: apparent angular radius α = R_Bennu / |r_sc| [rad].
//!
//! Measurement pipeline (physically correct):
//!   1. Check if Bennu falls within camera FOV — if not, return None.
//!   2. Sun-avoidance check — if Sun is within the exclusion zone, return None.
//!   3. Render a 64×64 pixel patch of Bennu's disk with Lambert illumination
//!      and shot (Poisson) noise; extract the intensity-weighted centroid.
//!   4. Rotate centroid bearing to inertial frame using the *star-tracker* attitude.
//!
//! Phase-angle effects are physically modelled: the illuminated crescent's
//! centroid shifts toward the sub-solar limb, producing a systematic bearing
//! bias proportional to sin(phase_angle).  The angular-size measurement fits a
//! circle to the full disk boundary (phase-angle-independent).

use nalgebra::{Vector3, Vector4};
use rand_distr::{Distribution, Normal};
use crate::config::{R_BENNU, OPNAV_BEARING_SIGMA_RAD, OPNAV_SIZE_SIGMA_RAD, CAMERA_FOV_HALF_RAD,
                    LIDAR_SIGMA_M, LIDAR_MAX_RANGE_M};
use crate::dynamics::attitude::{inertial_to_body, body_to_inertial};
use crate::dynamics::bennu::bennu_heliocentric_pos;

// ── Rendering constants ───────────────────────────────────────────────────────

/// Number of pixels per axis in the rendered centroid patch.
const RENDER_N: usize = 64;

/// Half-size of the rendered patch in units of Bennu's apparent radius.
/// The patch covers ±PATCH_RADII × α around the geometric disk centre.
const PATCH_RADII: f64 = 5.0;

/// Peak photon count for a surface element at normal incidence (fully lit).
/// Sets the shot-noise floor: σ_shot ≈ α / √N_photons.
const PEAK_PHOTONS: f64 = 2_000.0;

/// Sun exclusion half-angle [rad].
/// If the Sun falls within this angle of the camera boresight, the shutter
/// closes — no measurement is returned.  Matches the camera FOV half-angle.
const SUN_EXCL_RAD: f64 = CAMERA_FOV_HALF_RAD;

// ── Public types ──────────────────────────────────────────────────────────────

/// Packaged OpNav measurement: bearing, angular size, and phase angle.
#[derive(Clone, Debug)]
pub struct OpNavMeas {
    /// Unit line-of-sight vector to Bennu centroid, in inertial/Hill frame.
    pub los_inertial: Vector3<f64>,
    /// Apparent angular radius of Bennu disk [rad].
    pub angular_size: f64,
    /// Sun–Bennu–spacecraft phase angle [rad].  Stored for diagnostics only;
    /// not used by the EKF (the filter does not model phase-angle bias).
    pub phase_angle_rad: f64,
    /// 1-σ bearing noise actually realised by the pixel renderer [rad].
    /// Includes both shot noise (α/√flux) and the pixel-noise floor.
    /// Pass this to update_bearing so the EKF uses the true noise level.
    pub sigma_bearing: f64,
}

// ── Simulated measurement ─────────────────────────────────────────────────────

/// Simulate an OpNav measurement from the true spacecraft state.
///
/// Returns `None` when:
///   - Bennu is outside the camera FOV, or
///   - the Sun is within the camera exclusion zone.
///
/// `r_sc`   = true position of spacecraft w.r.t. Bennu [m] (Hill frame).
/// `q_true` = true spacecraft attitude quaternion [w, x, y, z].
/// `q_st`   = star-tracker attitude (slightly noisy).
/// `t_sim`  = simulation epoch [TDB seconds from J2000], used to compute the
///            Sun direction from Bennu's Keplerian heliocentric position.
pub fn measure<R: rand::Rng>(
    r_sc:   &Vector3<f64>,
    q_true: &Vector4<f64>,
    q_st:   &Vector4<f64>,
    t_sim:  f64,
    rng:    &mut R,
) -> Option<OpNavMeas> {
    let range = r_sc.norm();
    let alpha = R_BENNU / range;            // apparent angular radius [rad]

    // ── Directions ──────────────────────────────────────────────────────────
    // Geometric LOS: from spacecraft toward Bennu (Bennu at origin).
    let los_inertial_true = -r_sc / range;

    // Sun direction at Bennu: r_bennu is Bennu's heliocentric position, so
    // the Sun (at the Solar System barycentre) is in direction -r_bennu̲.
    let r_bennu   = bennu_heliocentric_pos(t_sim);
    let sun_inert = (-r_bennu).normalize();

    // Phase angle φ: angle at Bennu between (Bennu→Sun) and (Bennu→spacecraft).
    let sc_dir    = r_sc / range;
    let phase_angle = sun_inert.dot(&sc_dir).clamp(-1.0, 1.0).acos();

    // ── FOV and Sun-avoidance checks (body frame) ────────────────────────────
    let los_body = inertial_to_body(q_true, &los_inertial_true);
    if los_body[0] < CAMERA_FOV_HALF_RAD.cos() {
        return None; // Bennu outside FOV
    }

    let sun_body = inertial_to_body(q_true, &sun_inert);
    if sun_body[0] > SUN_EXCL_RAD.cos() {
        return None; // Sun in camera exclusion zone
    }

    // ── Pixel rendering → intensity-weighted centroid ────────────────────────
    // Returns (Δy, Δz, σ_bearing) — offsets [rad] in the body/camera frame
    // measured from the geometric disk centre, plus the actual 1-σ noise.
    let (delta_y, delta_z, sigma_centroid) = render_centroid(alpha, &sun_body, rng);

    // Apply centroid offset in the body frame.
    // Camera +x is the boresight; centroid shifts in the y-z image plane.
    let los_body_meas = Vector3::new(
        los_body[0],
        los_body[1] + delta_y,
        los_body[2] + delta_z,
    )
    .normalize();

    // Rotate centroid bearing to inertial using the *noisy* star-tracker attitude.
    let los_inertial_meas = body_to_inertial(q_st, &los_body_meas);

    // ── Angular size (disk boundary fit, phase-angle-independent) ───────────
    let dist_size  = Normal::new(0.0, OPNAV_SIZE_SIGMA_RAD).unwrap();
    let alpha_meas = (alpha + dist_size.sample(rng)).max(0.0);

    Some(OpNavMeas {
        los_inertial:    los_inertial_meas,
        angular_size:    alpha_meas,
        phase_angle_rad: phase_angle,
        sigma_bearing:   sigma_centroid,
    })
}

// ── Predicted measurement (EKF side) ─────────────────────────────────────────

/// Predicted measurement from the EKF position estimate (for innovations).
/// The EKF uses geometric centre — it does not model phase-angle centroid bias.
pub fn predicted(r_est: &Vector3<f64>) -> OpNavMeas {
    let range = r_est.norm();
    OpNavMeas {
        los_inertial:    -r_est / range,
        angular_size:    R_BENNU / range,
        phase_angle_rad: f64::NAN,
        sigma_bearing:   OPNAV_BEARING_SIGMA_RAD,
    }
}

// ── LIDAR altimeter ───────────────────────────────────────────────────────────

/// Simulated LIDAR altimeter measurement: slant range from the spacecraft to
/// Bennu's surface along the boresight (body-x = −r̂ for Nadir pointing).
///
/// Returns `None` when Bennu is farther than LIDAR_MAX_RANGE_M.
/// The measurement is: y = |r_sc| − R_BENNU + noise.
pub fn lidar_measure<R: rand::Rng>(
    r_sc: &Vector3<f64>,
    rng:  &mut R,
) -> Option<f64> {
    let range = r_sc.norm();
    if range > LIDAR_MAX_RANGE_M { return None; }
    let noise = Normal::new(0.0, LIDAR_SIGMA_M).unwrap().sample(rng);
    Some((range - R_BENNU + noise).max(0.0))
}

// ── Pixel-level image renderer ────────────────────────────────────────────────

/// Render a RENDER_N × RENDER_N patch centred on Bennu's disk with Lambert
/// illumination and shot noise, then return the intensity-weighted centroid as
/// angular offsets (Δy, Δz) [rad] in the camera/body frame, plus the actual
/// 1-σ bearing noise realised by the renderer.
///
/// The body frame convention: +x = boresight (toward Bennu); y and z span the
/// image plane.  `sun_body` is the unit Sun direction in this same frame.
///
/// Physics: at phase angle φ the visible crescent centroid shifts toward the
/// sub-solar limb by approximately (3/8) · sin(φ) · α — this emerges
/// naturally from the weighted-centroid calculation.
fn render_centroid<R: rand::Rng>(
    alpha:    f64,              // apparent angular radius of Bennu [rad]
    sun_body: &Vector3<f64>,   // unit Sun direction in camera/body frame
    rng:      &mut R,
) -> (f64, f64, f64) {
    // Fixed FOV render: the camera image always spans ±CAMERA_FOV_HALF_RAD.
    // This is the physical model: pixel scale is constant; Bennu fills more
    // pixels as the spacecraft approaches, improving centroid accuracy until the
    // disk overflows the FOV.  (The previous PATCH_RADII×α zoom was synthetic
    // and gave the wrong σ∝1/range trend.)
    let patch_half = CAMERA_FOV_HALF_RAD;
    let dpix       = 2.0 * patch_half / RENDER_N as f64; // rad per pixel (constant)

    let mut total_flux = 0.0_f64;
    let mut sum_y      = 0.0_f64;
    let mut sum_z      = 0.0_f64;

    for i in 0..RENDER_N {
        for j in 0..RENDER_N {
            // Angular coordinates of this pixel in the image plane.
            // i → body-y direction, j → body-z direction.
            let theta_y = -patch_half + (i as f64 + 0.5) * dpix;
            let theta_z = -patch_half + (j as f64 + 0.5) * dpix;

            let r2 = theta_y * theta_y + theta_z * theta_z;
            if r2 >= alpha * alpha { continue; } // outside disk

            // Outward surface normal at this visible-hemisphere point.
            // cos_rho is the foreshortening factor (= cos of emission angle).
            let cos_rho = (1.0 - r2 / (alpha * alpha)).max(0.0).sqrt();
            let n = Vector3::new(cos_rho, theta_y / alpha, theta_z / alpha);

            // Lambert illumination: I = max(0, n · sun_dir)
            let illum = n.dot(sun_body).max(0.0);

            // Shot noise via Normal approximation to Poisson (valid for mean >> 1).
            let mean = illum * PEAK_PHOTONS;
            let photons = if mean < 1e-9 {
                0.0
            } else {
                let noise = Normal::new(0.0_f64, mean.sqrt()).unwrap().sample(rng);
                (mean + noise).max(0.0)
            };

            total_flux += photons;
            sum_y      += theta_y * photons;
            sum_z      += theta_z * photons;
        }
    }

    // Disk fully in shadow (near solar conjunction) — no centroid observable.
    if total_flux < 1.0 {
        return (0.0, 0.0, OPNAV_BEARING_SIGMA_RAD);
    }

    // Intensity-weighted centroid [rad].
    let cy = sum_y / total_flux;
    let cz = sum_z / total_flux;

    // Centroiding noise: shot-noise floor combined with the config pixel noise.
    // σ_shot ≈ α / √N_photons (Fisher information limit for a disk centroid).
    let sigma_shot      = alpha / total_flux.sqrt();
    let sigma_centroid  = sigma_shot.hypot(OPNAV_BEARING_SIGMA_RAD);
    let dist = Normal::new(0.0, sigma_centroid).unwrap();

    (cy + dist.sample(rng), cz + dist.sample(rng), sigma_centroid)
}
