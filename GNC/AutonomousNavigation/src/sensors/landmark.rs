//! Landmark-based optical navigation (stereophotoclinometry analogue).
//!
//! Unlike center-finding OpNav (which sees Bennu as a featureless disk and yields
//! only a bearing + crude angular-size range), landmark OpNav tracks individual
//! surface features whose coordinates are known in the Bennu **body-fixed** frame.
//! Each recognised landmark is a line-of-sight to a point of known 3-D location,
//! so a handful of landmarks in one image gives the full 3-D spacecraft position
//! relative to Bennu — including range — without any LIDAR.
//!
//! Bennu spin: the body rotates about the Hill +z axis (the same pole the J2/J3/J4
//! gravity model assumes) at Bennu's real sidereal period.  A landmark at body-fixed
//! position `p_bf` appears in the inertial/Hill frame at `p_in(t) = Rz(θ(t))·p_bf`.
//!
//! A landmark produces a measurement only when it is simultaneously:
//!   - on the near side (its outward normal faces the spacecraft),
//!   - sunlit (its outward normal faces the Sun), and
//!   - inside the camera field of view.

use nalgebra::{Vector3, Vector4};
use rand_distr::{Distribution, Normal};
use crate::config::{R_BENNU, OPNAV_BEARING_SIGMA_RAD, CAMERA_FOV_HALF_RAD};
use crate::dynamics::attitude::{inertial_to_body, body_to_inertial};
use crate::dynamics::bennu::bennu_heliocentric_pos;

// ── Bennu rotation ──────────────────────────────────────────────────────────────

/// Bennu sidereal rotation period [s]  (4.296057 h; Nolan et al. 2013 / Lauretta 2019).
pub const BENNU_SPIN_PERIOD_S: f64 = 4.296057 * 3_600.0;

/// Bennu spins retrograde, so the rate about +z is negative.
const BENNU_SPIN_RATE_RADS: f64 = -2.0 * std::f64::consts::PI / BENNU_SPIN_PERIOD_S;

/// Number of catalogued surface landmarks.
///
/// Dense enough that the central FOV cap still holds several features at close
/// range, where Bennu's disk (≈30° radius at 500 m) overflows the 15° camera FOV
/// and only landmarks near the sub-spacecraft point remain trackable.
pub const N_LANDMARKS: usize = 150;

/// Rotate a body-fixed vector into the inertial/Hill frame at time `t`.
/// Rotation is about +z (Bennu's spin pole) by θ(t) = ω·t.
#[inline]
pub fn body_fixed_to_inertial(p_bf: &Vector3<f64>, t: f64) -> Vector3<f64> {
    let th = BENNU_SPIN_RATE_RADS * t;
    let (s, c) = th.sin_cos();
    Vector3::new(
        c * p_bf[0] - s * p_bf[1],
        s * p_bf[0] + c * p_bf[1],
        p_bf[2],
    )
}

// ── Landmark catalogue ──────────────────────────────────────────────────────────

/// One catalogued surface landmark, stored in Bennu body-fixed coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Landmark {
    /// Stable identifier (index into the catalogue).
    pub id:   usize,
    /// Body-fixed position on Bennu's surface [m] (radius ≈ R_BENNU).
    pub p_bf: Vector3<f64>,
}

/// Build the landmark catalogue: `N_LANDMARKS` points spread evenly over Bennu's
/// surface via a Fibonacci sphere, with +z as the polar (spin) axis.
pub fn catalogue() -> Vec<Landmark> {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt()); // golden angle
    (0..N_LANDMARKS)
        .map(|i| {
            // z runs from near +1 to near −1 so the spin axis is the pole.
            let z   = 1.0 - 2.0 * (i as f64 + 0.5) / N_LANDMARKS as f64;
            let rad = (1.0 - z * z).max(0.0).sqrt();
            let phi = i as f64 * golden;
            let p_bf = Vector3::new(rad * phi.cos(), rad * phi.sin(), z) * R_BENNU;
            Landmark { id: i, p_bf }
        })
        .collect()
}

// ── Visibility state (for diagnostics / visualisation) ──────────────────────────

/// Why a landmark is or isn't usable this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmState {
    /// Sunlit, near-side, and inside the FOV — produces a measurement.
    Tracked,
    /// Sunlit and on the near side, but outside the camera FOV.
    VisibleOutOfFov,
    /// On the near side but unlit (night side of the feature).
    Dark,
    /// On the far side of Bennu (occulted by the body).
    FarSide,
}

/// A simulated landmark observation plus everything needed to visualise it.
#[derive(Clone, Debug)]
pub struct LandmarkObs {
    pub id:           usize,
    /// Landmark position in the inertial/Hill frame at this epoch [m].
    pub p_inertial:   Vector3<f64>,
    /// Classification this frame.
    pub state:        LmState,
    /// Measured unit LOS from spacecraft to landmark, inertial frame.
    /// Only meaningful when `state == Tracked`.
    pub los_inertial: Vector3<f64>,
    /// 1-σ bearing noise on this observation [rad].
    pub sigma:        f64,
    /// Camera-plane coordinates (Δy, Δz) of the landmark [rad from boresight],
    /// for the synthetic NavCam inset.  Meaningful when `Tracked`.
    pub cam_yz:       (f64, f64),
}

// ── Measurement simulation ──────────────────────────────────────────────────────

/// Classify and (when tracked) measure every catalogued landmark from the true
/// spacecraft state.  Returns one `LandmarkObs` per landmark so the caller can
/// both feed the EKF (Tracked entries) and render the full visibility picture.
///
/// `r_sc`   = true spacecraft position w.r.t. Bennu [m, Hill frame]
/// `q_true` = true attitude quaternion
/// `q_st`   = star-tracker (noisy) attitude — used to express the measured LOS
/// `t`      = epoch [s]  (drives both Bennu spin and the Sun direction)
pub fn observe<R: rand::Rng>(
    cat:    &[Landmark],
    r_sc:   &Vector3<f64>,
    q_true: &Vector4<f64>,
    q_st:   &Vector4<f64>,
    t:      f64,
    rng:    &mut R,
) -> Vec<LandmarkObs> {
    let sun_inert = (-bennu_heliocentric_pos(t)).normalize();
    let cos_fov   = CAMERA_FOV_HALF_RAD.cos();

    cat.iter().map(|lm| {
        let p_in = body_fixed_to_inertial(&lm.p_bf, t);
        let n_hat = p_in / p_in.norm();              // outward surface normal
        let d     = p_in - r_sc;                      // spacecraft → landmark
        let dist  = d.norm();
        let los   = d / dist;

        // Near side: normal must face the spacecraft (point opposite the LOS).
        let near_side = n_hat.dot(&los) < 0.0;
        // Lit: normal must face the Sun.
        let lit = n_hat.dot(&sun_inert) > 0.0;

        let mut state = if !near_side {
            LmState::FarSide
        } else if !lit {
            LmState::Dark
        } else {
            LmState::VisibleOutOfFov
        };

        let mut los_meas = Vector3::zeros();
        let mut cam_yz   = (0.0, 0.0);
        let mut sigma    = OPNAV_BEARING_SIGMA_RAD;

        if state == LmState::VisibleOutOfFov {
            // FOV check is done in the camera/body frame against the +x boresight.
            let los_body = inertial_to_body(q_true, &los);
            if los_body[0] >= cos_fov {
                state = LmState::Tracked;
                // Centroiding noise: sharper than disk-centre, scale with range.
                sigma = OPNAV_BEARING_SIGMA_RAD;
                let n = Normal::new(0.0, sigma).unwrap();
                // Perturb in the body image plane, then re-express via the noisy
                // star-tracker attitude (pointing knowledge error enters here).
                let los_body_noisy = Vector3::new(
                    los_body[0],
                    los_body[1] + n.sample(rng),
                    los_body[2] + n.sample(rng),
                ).normalize();
                los_meas = body_to_inertial(q_st, &los_body_noisy);
                cam_yz   = (los_body_noisy[1], los_body_noisy[2]);
            }
        }

        LandmarkObs {
            id: lm.id, p_inertial: p_in, state,
            los_inertial: los_meas, sigma, cam_yz,
        }
    }).collect()
}
