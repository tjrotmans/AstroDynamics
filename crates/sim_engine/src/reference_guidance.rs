//! Guidance (Phase 13d) — following a Layer-1 reference trajectory, and the
//! pointing-mode library needed for cruise/burn/comm/target-relative
//! attitude, on top of the single 6DOF propagator (`propagator6dof`).
//!
//! Kept as its own module rather than extending the legacy
//! `guidance::PointingMode`/`engine::SimEngine` pair (the proximity-ops-only
//! path `MissionPlanner::simulate` already uses and has tests against) —
//! `step_tick` doesn't consume a `PointingMode` at all (it takes an
//! already-decided control torque), so this is a clean, additive extension
//! point with no risk to the tested legacy path. See
//! `docs/MP/MANUAL.md` §9 for the governing math.

use nalgebra::{Vector3, Vector4};
use orbital_models::attitude::align_x_with;

// ── §9.1 Reference trajectory + dispersion ──────────────────────────────────

/// One sample of a Layer-1-designed trajectory, used as the guidance
/// reference to fly against. Mirrors the fields of
/// `trajectory_solver::propagator::PropagatedPoint` (t_s, r_m, v_mps) — kept
/// as its own small struct rather than depending on that type directly so
/// this module doesn't need to know about `central_body_index`/SOI
/// bookkeeping, which is irrelevant once a trajectory has been chosen as a
/// fixed guidance reference (it's just a time-parameterized curve in the
/// same heliocentric frame from here on).
#[derive(Clone, Copy, Debug)]
pub struct ReferencePoint {
    pub t_s: f64,
    pub r_m: Vector3<f64>,
    pub v_mps: Vector3<f64>,
}

/// A Layer-1 arc, resampled as the guidance reference. Samples MUST be
/// sorted by `t_s` (ascending) — [`ReferenceTrajectory::state_at`] assumes
/// this for its binary search.
#[derive(Clone, Debug)]
pub struct ReferenceTrajectory {
    points: Vec<ReferencePoint>,
}

impl ReferenceTrajectory {
    /// Build from a Layer-1 arc's sampled points. Panics if `points` is
    /// empty or not sorted by `t_s` — both indicate a caller bug (an empty
    /// or malformed reference trajectory cannot be flown against).
    pub fn new(points: Vec<ReferencePoint>) -> Self {
        assert!(!points.is_empty(), "ReferenceTrajectory needs at least one point");
        assert!(
            points.windows(2).all(|w| w[1].t_s >= w[0].t_s),
            "ReferenceTrajectory points must be sorted by t_s"
        );
        Self { points }
    }

    pub fn t_start_s(&self) -> f64 {
        self.points.first().unwrap().t_s
    }

    pub fn t_end_s(&self) -> f64 {
        self.points.last().unwrap().t_s
    }

    /// The raw samples backing this reference — for inspection/plotting
    /// (e.g. `MissionPlanner`'s `cruise_demo` writes these alongside the
    /// flown trajectory), not for flying against (`state_at` is the
    /// intended query interface for that).
    pub fn points(&self) -> &[ReferencePoint] {
        &self.points
    }

    /// Reference (position, velocity) at time `t_s`, by cubic Hermite
    /// interpolation between the two bracketing samples — upgraded from
    /// linear (design review findings B3/C1). Linear
    /// interpolation of a curved heliocentric arc injects real "phantom"
    /// dispersion into `dispersion()` far larger than realistic TCM
    /// thresholds — the chord-vs-arc sagitta is order `a_grav * dt^2 / 8`,
    /// confirmed ~1,200 km at 1 AU for a Layer-1 arc's typical ~8-11h
    /// sample spacing, against a real 50 km user-configured
    /// `tcm_dr_threshold_m` — which TCM then "corrects" even though
    /// nothing is actually off track. Cubic Hermite uses the real `v_mps`
    /// already stored at both bracketing samples as the spline's tangents
    /// (every `ReferencePoint` carries a real propagator velocity, not a
    /// finite-difference guess — see `ArcApiPoint.vx_mps` upstream),
    /// reducing interpolation error from O(dt^2) to O(dt^4): exact for any
    /// two-body/Keplerian sub-arc between samples, and a large improvement
    /// even under third-body perturbation. Also used by `body_tracks`'
    /// pointing/perturbation queries (same type, `cruise.rs::
    /// resolve_target`), not just the guidance reference — one fix covers
    /// both. Reduces to exactly today's linear formula whenever the two
    /// samples' tangents are already consistent with straight-line motion
    /// between them (`v0 == v1 == (p1-p0)/dt`) — this only changes
    /// behavior for genuinely curved motion. Clamps to the first/last
    /// sample outside the reference's time span rather than extrapolating,
    /// unchanged from before — a dispersion query outside the reference's
    /// covered span is a caller error (asking "how far off track am I"
    /// before the reference even starts, or long after it ends) and
    /// clamping degrades gracefully instead of producing a wild
    /// extrapolated value.
    pub fn state_at(&self, t_s: f64) -> (Vector3<f64>, Vector3<f64>) {
        if t_s <= self.points[0].t_s {
            let p = &self.points[0];
            return (p.r_m, p.v_mps);
        }
        if t_s >= self.points[self.points.len() - 1].t_s {
            let p = &self.points[self.points.len() - 1];
            return (p.r_m, p.v_mps);
        }
        // Linear scan is fine here: reference trajectories are hundreds to
        // low thousands of samples (Layer 1 arcs), not the tick-by-tick
        // scale this guidance module is called at within a mission.
        let idx = self.points.partition_point(|p| p.t_s <= t_s).saturating_sub(1);
        let a = &self.points[idx];
        let b = &self.points[(idx + 1).min(self.points.len() - 1)];
        let span = (b.t_s - a.t_s).max(1e-9);
        let t = ((t_s - a.t_s) / span).clamp(0.0, 1.0);

        // Standard cubic Hermite basis on the normalized parameter `t`.
        // Tangents (`m0`/`m1`) are the real velocities scaled by `span`,
        // since v_mps is a derivative w.r.t. real time (t_s), not w.r.t.
        // the normalized `t`.
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        let m0 = a.v_mps * span;
        let m1 = b.v_mps * span;
        let r = a.r_m * h00 + m0 * h10 + b.r_m * h01 + m1 * h11;

        // Velocity is the position spline's OWN derivative (chain rule:
        // dr/dt_s = (dr/dt) / span), not a separate linear blend of
        // v_mps — keeps the returned position/velocity pair mutually
        // consistent, which the old linear scheme's independent
        // interpolations of r_m and v_mps never guaranteed.
        let dh00 = 6.0 * t2 - 6.0 * t;
        let dh10 = 3.0 * t2 - 4.0 * t + 1.0;
        let dh01 = -6.0 * t2 + 6.0 * t;
        let dh11 = 3.0 * t2 - 2.0 * t;
        let v = (a.r_m * dh00 + m0 * dh10 + b.r_m * dh01 + m1 * dh11) / span;

        (r, v)
    }
}

/// Dispersion (actual − reference) at a common epoch [m, m/s]. This is the
/// quantity guidance acts on — not the absolute state, the ERROR relative
/// to the Layer-1 plan.
#[derive(Clone, Copy, Debug)]
pub struct Dispersion {
    pub dr_m: Vector3<f64>,
    pub dv_mps: Vector3<f64>,
}

pub fn dispersion(actual_r_m: &Vector3<f64>, actual_v_mps: &Vector3<f64>, t_s: f64, reference: &ReferenceTrajectory) -> Dispersion {
    let (r_ref, v_ref) = reference.state_at(t_s);
    Dispersion { dr_m: actual_r_m - r_ref, dv_mps: actual_v_mps - v_ref }
}

// ── §9.2 Trajectory correction maneuver (TCM) targeting ────────────────────

/// TCM correction ΔV [m/s] via Lambert-based fixed-time-of-arrival
/// retargeting: solve a Lambert arc from the DISPERSED current state to the
/// reference trajectory's position at a chosen future epoch (`arrival_t_s`,
/// typically the next reference waypoint or the leg's final arrival time —
/// the caller's choice), and command the departure velocity that arc
/// implies. `ΔV = v_lambert_departure − v_current`.
///
/// This solves the identical two-point boundary value problem an STM-based
/// fixed-time-of-arrival correction would (retarget position at a fixed
/// future time from a perturbed initial state) — Lambert's problem IS that
/// BVP for two-body/patched-conic dynamics, solved in closed form rather
/// than via a linearized STM. Chosen over building a dedicated STM
/// propagator for this module because `orbital_math::lambert` already
/// exists, is tested, and is exactly the right tool for this fixed-TOF
/// re-targeting problem — reusing it is preferred over introducing a
/// second, less-general BVP solver. See `docs/MP/MANUAL.md` §9.2.
///
/// Returns `None` if `arrival_t_s` is not after `current_t_s`, or if no
/// Lambert solution exists for the requested geometry/TOF (see
/// `orbital_math::lambert::lambert`'s own degenerate-geometry guards).
pub fn tcm_lambert_correction(
    current_r_m: &Vector3<f64>,
    current_v_mps: &Vector3<f64>,
    current_t_s: f64,
    arrival_t_s: f64,
    reference: &ReferenceTrajectory,
    mu_central_m3s2: f64,
    prograde: bool,
) -> Option<Vector3<f64>> {
    let tof_s = arrival_t_s - current_t_s;
    if tof_s <= 0.0 {
        return None;
    }
    let (r_target, _v_target) = reference.state_at(arrival_t_s);

    let r1: [f64; 3] = [current_r_m.x, current_r_m.y, current_r_m.z];
    let r2: [f64; 3] = [r_target.x, r_target.y, r_target.z];
    let solutions = orbital_math::lambert::lambert(r1, r2, tof_s, prograde, mu_central_m3s2);
    let (v_dep, _v_arr) = *solutions.first()?;
    let v_dep = Vector3::new(v_dep[0], v_dep[1], v_dep[2]);
    Some(v_dep - current_v_mps)
}

// ── §9.3 Pointing-mode library ──────────────────────────────────────────────

/// Pointing modes needed across a full mission (cruise through arrival) —
/// separate from the legacy `guidance::PointingMode` (Nadir/VelocityAligned
/// only, proximity-ops-specific) per this module's doc comment.
#[derive(Clone, Copy, Debug)]
pub enum CruisePointingMode {
    /// Body +z (solar-panel normal, by this repo's existing body-axis
    /// convention — see `GNC/AutonomousNavigation`'s SRP plate layout) faces
    /// the Sun, for maximum power generation during quiescent cruise.
    SunPointing,
    /// Body +x (the same boresight/thrust-axis convention already used
    /// throughout this repo) aligned with a commanded inertial burn
    /// direction — used to slew to burn attitude before a TCM/finite burn.
    BurnAttitude { thrust_dir_inertial: Vector3<f64> },
    /// Body +x aligned with the spacecraft-to-Earth direction, for
    /// high-gain-antenna communication passes.
    EarthComm { earth_pos_inertial: Vector3<f64> },
    /// Body +x (camera boresight) points at a target body's centre —
    /// the same convention `GNC/AutonomousNavigation`'s Nadir mode uses,
    /// generalized to any target (not just the body currently being
    /// orbited).
    TargetRelative { target_pos_inertial: Vector3<f64> },
}

/// Compute the desired attitude quaternion for a [`CruisePointingMode`].
///
/// `r_m`/`sun_pos_inertial_m` are both needed for [`CruisePointingMode::SunPointing`]
/// (spacecraft → Sun direction); other modes only need their own field(s).
/// All directions reduce to the same primitive — align a body axis with a
/// commanded inertial direction — via `orbital_models::attitude::align_x_with`,
/// which aligns body +x. For [`CruisePointingMode::SunPointing`] (body +z,
/// not +x), the direction is rotated into the +x-aligned convention first
/// (a fixed 90 degree offset applied to the result) rather than adding a
/// second alignment primitive for a different body axis.
pub fn desired_quaternion_cruise(
    mode: CruisePointingMode,
    r_m: &Vector3<f64>,
    sun_pos_inertial_m: &Vector3<f64>,
) -> Vector4<f64> {
    match mode {
        CruisePointingMode::SunPointing => {
            let sun_dir = (sun_pos_inertial_m - r_m).normalize();
            // align_x_with aligns body +x with the target direction; rotate
            // the RESULT by +90 deg about body +y so body +z ends up
            // pointing at the target instead (avoids a second alignment
            // primitive for a different body axis).
            let q_x_aligned = align_x_with(&sun_dir);
            // +90 deg rotation about body +y maps body +z onto body +x:
            // R_y(90deg): e_z -> (sin90, 0, cos90) = (1,0,0) = e_x. Applying
            // this FIRST (as q_local, composed on the right) then aligning
            // the result with the target puts body +z on the target instead
            // of body +x.
            let q_z_to_x = Vector4::new(std::f64::consts::FRAC_1_SQRT_2, 0.0, std::f64::consts::FRAC_1_SQRT_2, 0.0);
            orbital_models::attitude::quat_multiply(&q_x_aligned, &q_z_to_x)
        }
        CruisePointingMode::BurnAttitude { thrust_dir_inertial } => align_x_with(&thrust_dir_inertial.normalize()),
        CruisePointingMode::EarthComm { earth_pos_inertial } => {
            align_x_with(&(earth_pos_inertial - r_m).normalize())
        }
        CruisePointingMode::TargetRelative { target_pos_inertial } => {
            align_x_with(&(target_pos_inertial - r_m).normalize())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(t_s: f64, r: Vector3<f64>, v: Vector3<f64>) -> ReferencePoint {
        ReferencePoint { t_s, r_m: r, v_mps: v }
    }

    // Review finding B3/C1: this fixture used to pair samples
    // 1000 m apart over 100 s (a real 10 m/s average) with v_mps=(1,0,0) —
    // physically inconsistent, but harmless under the old LINEAR scheme
    // (which interpolates r_m and v_mps independently and never checks
    // them against each other). Cubic Hermite differentiates the position
    // spline using the stored v_mps as tangents, so it would use that
    // wrong tangent and produce a nonsense result — fixed to real,
    // self-consistent constant-velocity motion (v_mps = (p1-p0)/dt), which
    // is also the exact degenerate case where cubic Hermite reduces to the
    // old linear formula (verified by hand: same 250.0 expected result).
    #[test]
    fn state_at_reduces_to_linear_for_constant_velocity_motion() {
        let reference = ReferenceTrajectory::new(vec![
            sample(0.0, Vector3::new(0.0, 0.0, 0.0), Vector3::new(10.0, 0.0, 0.0)),
            sample(100.0, Vector3::new(1000.0, 0.0, 0.0), Vector3::new(10.0, 0.0, 0.0)),
        ]);
        let (r, v) = reference.state_at(25.0);
        assert!((r - Vector3::new(250.0, 0.0, 0.0)).norm() < 1e-9);
        assert!((v - Vector3::new(10.0, 0.0, 0.0)).norm() < 1e-9);
    }

    /// The real correctness improvement B3/C1 exists for: real constant-
    /// acceleration motion (p = p0 + v0*t + 0.5*a*t^2), which linear
    /// interpolation gets wrong (it would predict the arithmetic mean of
    /// the endpoint positions, 5000.0, badly missing the true midpoint
    /// value of 2500.0) but cubic Hermite reproduces exactly, since a
    /// Hermite cubic spline is exact for any polynomial up to degree 3 and
    /// quadratic motion is a degree-2 special case.
    #[test]
    fn state_at_is_exact_for_constant_acceleration_motion() {
        let a = 2.0; // m/s^2
        let t_end = 100.0;
        let reference = ReferenceTrajectory::new(vec![
            sample(0.0, Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0)),
            sample(t_end, Vector3::new(0.5 * a * t_end * t_end, 0.0, 0.0), Vector3::new(a * t_end, 0.0, 0.0)),
        ]);
        let t_mid = 50.0;
        let (r, v) = reference.state_at(t_mid);
        let r_true = 0.5 * a * t_mid * t_mid;
        let v_true = a * t_mid;
        assert!((r - Vector3::new(r_true, 0.0, 0.0)).norm() < 1e-6, "expected exact r={r_true}, got {r:?}");
        assert!((v - Vector3::new(v_true, 0.0, 0.0)).norm() < 1e-6, "expected exact v={v_true}, got {v:?}");
    }

    #[test]
    fn state_at_clamps_outside_reference_span() {
        let reference = ReferenceTrajectory::new(vec![
            sample(0.0, Vector3::new(0.0, 0.0, 0.0), Vector3::zeros()),
            sample(100.0, Vector3::new(1000.0, 0.0, 0.0), Vector3::zeros()),
        ]);
        let (r_before, _) = reference.state_at(-50.0);
        let (r_after, _) = reference.state_at(500.0);
        assert!((r_before - Vector3::new(0.0, 0.0, 0.0)).norm() < 1e-9);
        assert!((r_after - Vector3::new(1000.0, 0.0, 0.0)).norm() < 1e-9);
    }

    #[test]
    fn dispersion_is_zero_on_reference() {
        // Same B3/C1 fixture fix as above -- self-consistent constant-
        // velocity motion instead of a (r, v) pair that didn't actually
        // agree with each other.
        let reference = ReferenceTrajectory::new(vec![
            sample(0.0, Vector3::new(0.0, 0.0, 0.0), Vector3::new(10.0, 0.0, 0.0)),
            sample(100.0, Vector3::new(1000.0, 0.0, 0.0), Vector3::new(10.0, 0.0, 0.0)),
        ]);
        let d = dispersion(&Vector3::new(500.0, 0.0, 0.0), &Vector3::new(10.0, 0.0, 0.0), 50.0, &reference);
        assert!(d.dr_m.norm() < 1e-9);
        assert!(d.dv_mps.norm() < 1e-9);
    }

    /// A TCM solved for a spacecraft with ZERO dispersion (sitting exactly
    /// on a circular-orbit reference) should recover very close to the
    /// reference's own velocity at the departure point -- the Lambert arc
    /// degenerates to (approximately) the reference's own path when there is
    /// nothing to correct.
    #[test]
    fn tcm_correction_is_small_when_already_on_reference() {
        let mu: f64 = 3.986_004_418e14;
        let r: f64 = 7.0e6;
        let v_circ = (mu / r).sqrt();
        let period = 2.0 * std::f64::consts::PI * (r * r * r / mu).sqrt();

        // Build a coarse circular-orbit reference (16 samples/orbit).
        let n = 16;
        let mut points = Vec::new();
        for i in 0..=n {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            let t_s = period * i as f64 / n as f64;
            points.push(sample(
                t_s,
                Vector3::new(r * theta.cos(), r * theta.sin(), 0.0),
                Vector3::new(-v_circ * theta.sin(), v_circ * theta.cos(), 0.0),
            ));
        }
        let reference = ReferenceTrajectory::new(points);

        let (r0, v0) = reference.state_at(0.0);
        let arrival_t = period / 4.0; // quarter-orbit ahead
        let dv = tcm_lambert_correction(&r0, &v0, 0.0, arrival_t, &reference, mu, true)
            .expect("Lambert solution should exist for a quarter-orbit transfer");
        assert!(
            dv.norm() < 0.05 * v_circ,
            "TCM from an on-reference state should be small relative to orbital speed: {:.3} vs v_circ {:.3}",
            dv.norm(), v_circ
        );
    }

    #[test]
    fn burn_attitude_aligns_body_x_with_commanded_direction() {
        let q = desired_quaternion_cruise(
            CruisePointingMode::BurnAttitude { thrust_dir_inertial: Vector3::new(0.0, 1.0, 0.0) },
            &Vector3::zeros(), &Vector3::zeros(),
        );
        let body_x_inertial = orbital_models::attitude::body_to_inertial(&q, &Vector3::new(1.0, 0.0, 0.0));
        assert!((body_x_inertial - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-9);
    }

    #[test]
    fn sun_pointing_aligns_body_z_with_sun_direction() {
        let sun_pos = Vector3::new(0.0, 0.0, 1.495_98e11);
        let r = Vector3::zeros();
        let q = desired_quaternion_cruise(CruisePointingMode::SunPointing, &r, &sun_pos);
        let body_z_inertial = orbital_models::attitude::body_to_inertial(&q, &Vector3::new(0.0, 0.0, 1.0));
        let sun_dir = sun_pos.normalize();
        assert!(
            (body_z_inertial - sun_dir).norm() < 1e-6,
            "body +z should align with the sun direction: {:?} vs {:?}", body_z_inertial, sun_dir
        );
    }
}
