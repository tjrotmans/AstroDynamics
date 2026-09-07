//! wsb_continuation_corrected — homotopy with shooting correction at each step.
//!
//! For each lambda level:
//!   1. Propagate with IC inherited from the previous level ("before").
//!   2. Score the result.  If the spacecraft misses the Hill sphere, run a
//!      random search over (Δtheta, Δr_apogee) to find an IC that re-enters
//!      the Hill sphere at this fidelity level ("after").
//!   3. The corrected IC carries forward to the next lambda.
//!
//! Output: out/wsb/continuation_corrected.csv
//!   lambda, phase, run_id, theta_deg, r_apogee_nd,
//!   t_s, x_km, y_km, z_km, moon_x_km, moon_y_km, r_moon_km
//!
//! phase = "before" (inherited IC) | "after" (corrected IC, or same if no correction needed)
//!
//! Usage:
//!   cargo run -p lunar_trajectories --bin wsb_continuation_corrected --release

use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;
use std::fs;

use ephemeris::{Almanac, BodyTrack, Epoch, MoonTrack, SunTrack};
use hifitime::Duration;
use nalgebra::{Matrix3, SVector};
use ode_solvers::dopri5::Dopri5;
use ode_solvers::{SVector as OdeVec, System};
use ode_solvers::dop_shared::OutputType;
use orbital_models::constants::{MOON_RADIUS, MU_MOON, MU_VENUS, MU_MARS, MU_JUPITER, P_SRP};
use orbital_models::GravityModel;
use rand::Rng;
use rand::SeedableRng;

use lunar_trajectories::crtbp::CrtbpParams;
use lunar_trajectories::propagator::{propagate_bcr4bp_raw, Bcr4bpParams};
use lunar_trajectories::transfers::{lunar_hill_radius, tli_dv, tli_injection_ic, min_loi_dv};

// ── Config ────────────────────────────────────────────────────────────────────

const SELECTED_IDX: usize = 0;   // default; override with --idx N on the command line

const R_PARK:    f64 = (6_371.0 + 378.0) / 384_400.0;
const T_PROP_ND: f64 = 20.0 * PI;
const H_MAX_ND:  f64 = 0.02;   // max RK45 step cap [nd]; cruise ~1.7 h, flyby auto-shrinks
const MIN_CAP_ND: f64 = 0.15;

const OMEGA_S:     f64 = 27.321_661 / 365.25 - 1.0;
const A_S_ND:      f64 = 389.172;
const LAMBDAS:     &[f64] = &[0.0, 0.25, 0.5, 0.75, 1.0];
const SAMPLE_DT_S: f64   = 3_600.0;

// Cannonball SRP model (same convention as Artemis propagator):
//   a_srp = P_SRP * area * (1 + reflectivity) / mass  *  ŝ
// These values approximate a 500 kg lunar transfer spacecraft.
const SRP_AREA_M2:    f64 = 10.0;
const SRP_MASS_KG:    f64 = 500.0;
const SRP_REFLECTIVITY: f64 = 1.5;

// Derived: P_SRP * A * (1+r) / m  [m/s² at 1 AU — distance scaling omitted, same as Artemis]
const SRP_COEFF: f64 = P_SRP * SRP_AREA_M2 * (1.0 + SRP_REFLECTIVITY) / SRP_MASS_KG;

const OUT_META: &str = "out/wsb/continuation_corrected_meta.csv";
const OUT_ANIM: &str = "out/wsb/continuation_anim.csv";
// Each corrected IC re-propagated at λ=1 — for the IC-convergence comparison plot.
const OUT_EVAL: &str = "out/wsb/continuation_eval.csv";

// Fixed seed for the shooting search — guarantees identical results across runs.
const RNG_SEED: u64 = 0xDEAD_BEEF_C0DE_0001;

// After the spacecraft climbs above this Earth distance, it must not come back
// below it before reaching the Hill sphere.  Rejects resonant multi-loop
// trajectories that make extra Earth passes before arriving at the Moon.
const EARTH_RETURN_KM: f64 = 200_000.0;

// Moon surface radius [km].  Trajectories whose closest approach falls below
// this are Moon crashes — unusable as a continuation starting point.
const R_MOON_KM: f64 = 1_737.4;

// Shooting parameters.
// Radii kept deliberately tight so corrected ICs stay in the WSB neighbourhood
// of the original BCR4BP solution.  Wide radii (±20-45°) can jump to a completely
// different trajectory family (e.g. a direct 3-day transfer).
const TRIALS_PER_ROUND: usize = 400;
const SEARCH_ROUNDS: &[(f64, f64)] = &[
    (2.0  * PI / 180.0, 0.10),  // round 1: ±2°,  ±0.10 nd  — stay very close
    (5.0  * PI / 180.0, 0.25),  // round 2: ±5°,  ±0.25 nd
    (12.0 * PI / 180.0, 0.50),  // round 3: ±12°, ±0.50 nd  — last resort
];

const OUT_DIR: &str = "out/wsb";
const OUT_CSV: &str = "out/wsb/continuation_corrected.csv";

const SOLUTIONS: &[(u64, u64, f64, f64, f64)] = &[
    (1300001, 13, 205.5490,  27.0190, 3.9000),
    (1800001, 18,  43.9970, 262.2990, 2.8000),
    ( 100001,  1,  76.8740, 115.8740, 3.5000),
    (2100001, 21, 129.3260, 282.8810, 2.8000),
    ( 400001,  4, 297.5540,  35.3950, 3.9000),
    (1400001, 14, 266.5510, 359.2860, 4.6000),
    ( 600001,  6, 109.3690, 224.0100, 4.3000),
    (1200001, 12, 172.6140, 209.5650, 4.1000),
    (1700001, 17, 150.7540, 301.5090, 4.6000),
    (2300001, 23,  24.6570,  67.0480, 3.5000),
];

// ── Frame alignment ───────────────────────────────────────────────────────────

/// Build the 3×3 rotation matrix that maps the BCR4BP inertial frame to ECI.
///
/// Columns: x̂ = Moon direction, ŷ = in-plane perpendicular (Moon velocity direction),
///          ẑ = orbital angular momentum (h = r × v).
///
/// This replaces the old 2-D rotation by psi_m0, which left z=0 in ECI (ecliptic
/// plane) and ignored the Moon's ~5° inclination.  With this matrix the λ=0
/// circular trajectory already lies in the Moon's actual orbital plane.
fn orbital_plane_r3d(moon_pos: SVector<f64, 3>, moon_vel: SVector<f64, 3>) -> Matrix3<f64> {
    let x_hat = moon_pos / moon_pos.norm();
    let h     = moon_pos.cross(&moon_vel);
    let z_hat = h / h.norm();
    let y_hat = z_hat.cross(&x_hat);  // in-plane, toward direction of Moon's motion
    Matrix3::from_columns(&[x_hat, y_hat, z_hat])
}

// ── BodyPositions trait + providers ──────────────────────────────────────────

trait BodyPositions {
    fn moon_pos_m(&self, t_s: f64) -> SVector<f64, 3>;
    fn sun_pos_m(&self,  t_s: f64) -> SVector<f64, 3>;
}

struct CircularOrbits {
    moon_r_m:      f64,
    sun_r_m:       f64,
    n_moon:        f64,
    n_sun_eci:     f64,
    theta_sun_rad: f64,       // Sun angle from Moon direction in BCR4BP inertial at t=0
    r3d:           Matrix3<f64>, // BCR4BP inertial frame → ECI (accounts for inclination)
}
impl BodyPositions for CircularOrbits {
    fn moon_pos_m(&self, t_s: f64) -> SVector<f64, 3> {
        // Moon on circular orbit in BCR4BP plane, rotated into Moon's real orbital plane.
        let theta = self.n_moon * t_s;
        self.r3d * SVector::<f64, 3>::new(
            self.moon_r_m * theta.cos(),
            self.moon_r_m * theta.sin(),
            0.0,
        )
    }
    fn sun_pos_m(&self, t_s: f64) -> SVector<f64, 3> {
        let phi = self.theta_sun_rad + self.n_sun_eci * t_s;
        self.r3d * SVector::<f64, 3>::new(
            self.sun_r_m * phi.cos(),
            self.sun_r_m * phi.sin(),
            0.0,
        )
    }
}

struct AniseTrack {
    moon:    MoonTrack,
    sun:     SunTrack,
    // Third-body perturbers — pre-sampled positions + their GM [m³/s²].
    // Ordered: Venus, Mars, Jupiter (consistent with LAMBDAS blending).
    perturbers: Vec<(BodyTrack, f64)>,
}
impl BodyPositions for AniseTrack {
    fn moon_pos_m(&self, t: f64) -> SVector<f64, 3> { self.moon.position_at(t) }
    fn sun_pos_m(&self,  t: f64) -> SVector<f64, 3> { self.sun .position_at(t) }
}

struct Blended<'a> { circ: &'a dyn BodyPositions, real: &'a dyn BodyPositions, lambda: f64 }
impl<'a> BodyPositions for Blended<'a> {
    fn moon_pos_m(&self, t: f64) -> SVector<f64, 3> {
        self.circ.moon_pos_m(t) * (1.0 - self.lambda) + self.real.moon_pos_m(t) * self.lambda
    }
    fn sun_pos_m(&self, t: f64) -> SVector<f64, 3> {
        self.circ.sun_pos_m(t) * (1.0 - self.lambda) + self.real.sun_pos_m(t) * self.lambda
    }
}

// ── ODE ───────────────────────────────────────────────────────────────────────

type State6 = OdeVec<f64, 6>;

/// N-body ODE in Earth-centred ECI.
///
/// Forces at λ=0 (circular, BCR4BP-equivalent): Earth + Moon + Sun.
/// Forces added as λ increases (blended by `lambda`):
///   - Venus, Mars, Jupiter third-body perturbations  (GravityModel::third_body)
///   - Cannonball SRP  (P_SRP * A*(1+r)/m * ŝ, same pattern as Artemis propagator)
struct NBodyOde<'a> {
    provider:    &'a dyn BodyPositions,
    /// Extra perturbers from ANISE: (pre-sampled position track, GM [m³/s²]).
    /// Populated from AniseTrack.perturbers; empty for the circular provider.
    extra:       &'a [(BodyTrack, f64)],
    lambda:      f64,
    r_moon_m:    f64,
    r_hill_m:    f64,
    entered_soi: bool,
}

impl<'a> System<f64, State6> for NBodyOde<'a> {
    fn system(&self, t: f64, y: &State6, dy: &mut State6) {
        let pos  = SVector::<f64, 3>::new(y[0], y[1], y[2]);
        let moon = self.provider.moon_pos_m(t);
        let sun  = self.provider.sun_pos_m(t);

        // Base forces: always present (λ=0 already includes all three)
        let mut accel = GravityModel::compute(&pos, 1.0)
                      + GravityModel::moon_third_body(&pos, &moon)
                      + GravityModel::sun_third_body(&pos, &sun);

        // Extra perturbations blended in by λ (zero at λ=0, full at λ=1)
        if self.lambda > 0.0 {
            // Planetary third-body (Venus, Mars, Jupiter) — same formula as
            // moon_third_body / sun_third_body, generic over GM.
            for (track, mu) in self.extra {
                let body_pos = track.position_at(t);
                accel += self.lambda * GravityModel::third_body(&pos, &body_pos, *mu);
            }

            // Cannonball SRP — same convention as Artemis propagator:
            //   a = P_SRP * area * (1+r) / mass  * ŝ  where ŝ = (pos - sun) / |pos - sun|
            let s_vec = pos - sun;
            let s_hat = s_vec / s_vec.norm();
            accel += self.lambda * SRP_COEFF * s_hat;
        }

        dy[0] = y[3]; dy[1] = y[4]; dy[2] = y[5];
        dy[3] = accel[0]; dy[4] = accel[1]; dy[5] = accel[2];
    }

    // true = abort integration
    fn solout(&mut self, t: f64, y: &State6, _: &State6) -> bool {
        let pos  = SVector::<f64, 3>::new(y[0], y[1], y[2]);
        let moon = self.provider.moon_pos_m(t);
        let r    = (pos - moon).norm();
        if r < self.r_hill_m { self.entered_soi = true; }
        // crash or clean SOI exit (2× Hill) after capture
        r < self.r_moon_m || (self.entered_soi && r > self.r_hill_m * 2.0)
    }
}

// ── Trajectory step ───────────────────────────────────────────────────────────

struct EciStep { t_s: f64, pos_km: [f64; 3], vel_km_s: [f64; 3], moon_km: [f64; 3], r_moon_km: f64 }

/// Propagate in the N-body ECI model.
///
/// Uses OutputType::Sparse — every accepted RK45 step is logged.  This gives
/// dense output during lunar flybys (where the step shrinks automatically)
/// without wasting storage during quiet cruise phases.
/// h_max_s caps the step during cruise so it stays smooth for plotting.
fn propagate(
    r0: SVector<f64, 3>, v0: SVector<f64, 3>,
    provider:  &dyn BodyPositions,
    extra:     &[(BodyTrack, f64)],
    lambda:    f64,
    r_hill_m:  f64,
    t_end_s:   f64,
    h_max_s:   f64,
) -> Vec<EciStep> {
    const N_MAX: u32 = 5_000_000;
    let y0     = State6::from_column_slice(&[r0[0],r0[1],r0[2],v0[0],v0[1],v0[2]]);
    let h_init = h_max_s.min(t_end_s / 10.0);
    let ode    = NBodyOde { provider, extra, lambda, r_moon_m: MOON_RADIUS, r_hill_m, entered_soi: false };
    let mut s  = Dopri5::from_param(
        ode, 0.0, t_end_s, h_init, y0, 1e-10, 1e-12,
        0.9, 0.04, 0.333, 6.0, h_max_s, h_init,
        N_MAX, 1000, OutputType::Sparse,
    );
    let _ = s.integrate();
    s.x_out().iter().zip(s.y_out().iter()).map(|(&t, y)| {
        let pos  = SVector::<f64, 3>::new(y[0], y[1], y[2]);
        let moon = provider.moon_pos_m(t);
        EciStep {
            t_s:      t,
            pos_km:   [y[0]/1e3, y[1]/1e3, y[2]/1e3],
            vel_km_s: [y[3]/1e3, y[4]/1e3, y[5]/1e3],
            moon_km:  [moon[0]/1e3, moon[1]/1e3, moon[2]/1e3],
            r_moon_km: (pos - moon).norm() / 1e3,
        }
    }).collect()
}

// ── Capture score ─────────────────────────────────────────────────────────────

// Minimum flight time before Hill-sphere entry.  Direct lunar transfers enter in ~3 days;
// WSB transfers take at least 50 days.  Enforce 20 days to reject direct transfers
// without being so strict that it rejects genuinely fast WSB-like solutions.
const MIN_WSB_TRANSFER_S: f64 = 20.0 * 86_400.0;

// Acceptance window around the reference (λ=0) Hill-sphere arrival time.
// Candidates arriving more than this far from the reference are rejected.
const T_HILL_TOL_S: f64 = 15.0 * 86_400.0;   // ±15 days

// LOI ΔV guard: reject candidates whose insertion cost exceeds the BCR4BP reference
// by more than this factor.  Solutions within the budget get a score bonus proportional
// to how much cheaper they are than the reference.
const LOI_DV_TOL_FACTOR: f64 = 1.5;

fn score(traj: &[EciStep], r_hill_m: f64, period_s: f64, min_cap_s: f64,
         t_hill_ref_s: f64, loi_ref_km_s: f64) -> f64 {
    // Returns higher value for better capture quality.
    let t_first_hill = traj.iter()
        .find(|s| s.r_moon_km * 1e3 < r_hill_m)
        .map(|s| s.t_s)
        .unwrap_or(f64::MAX);

    // Reject direct transfers (<20 days to Hill sphere)
    if t_first_hill < MIN_WSB_TRANSFER_S {
        return -1e15;
    }
    // Reject arrival times outside ±T_HILL_TOL_S of the λ=0 reference
    if t_hill_ref_s > 0.0 && t_first_hill < f64::MAX {
        if (t_first_hill - t_hill_ref_s).abs() > T_HILL_TOL_S {
            return -1e14;
        }
    }

    // Reject resonant multi-loop trajectories: once the spacecraft climbs above
    // EARTH_RETURN_KM it must not come back below it before reaching the Hill sphere.
    {
        let mut above = false;
        for s in traj {
            if s.r_moon_km * 1e3 < r_hill_m { break; }
            let r_e = (s.pos_km[0].powi(2) + s.pos_km[1].powi(2) + s.pos_km[2].powi(2)).sqrt();
            if r_e > EARTH_RETURN_KM { above = true; }
            else if above             { return -1e13; }
        }
    }

    // Reject Moon crashes — the IC cannot be carried forward to the next lambda level.
    let min_r_moon = traj.iter().map(|s| s.r_moon_km).fold(f64::MAX, f64::min);
    if min_r_moon < R_MOON_KM {
        return -1e12;
    }

    let mut min_r_m   = f64::MAX;
    let mut in_cap    = false;
    let mut t_start   = 0.0_f64;
    let mut max_iv    = 0.0_f64;
    let mut total     = 0.0_f64;
    let mut n_entries = 0_usize;

    for s in traj {
        let r = s.r_moon_km * 1e3;
        if r < min_r_m { min_r_m = r; }
        let cap = r < r_hill_m;
        match (in_cap, cap) {
            (false, true) => { in_cap = true; t_start = s.t_s; n_entries += 1; }
            (true, false) => {
                in_cap = false;
                let dur = s.t_s - t_start;
                total += dur;
                if dur > max_iv { max_iv = dur; }
            }
            _ => {}
        }
    }
    if in_cap {
        let dur = traj.last().map(|s| s.t_s - t_start).unwrap_or(0.0);
        total += dur;
        if dur > max_iv { max_iv = dur; }
    }

    let n_orbits = max_iv / period_s;
    let captured = n_entries > 0 && total >= min_cap_s;

    // LOI ΔV filter — only applied when the trajectory enters the Hill sphere.
    // Reject if > LOI_DV_TOL_FACTOR × BCR4BP reference; bonus for being cheaper.
    let loi_dv_km_s = if n_entries > 0 && loi_ref_km_s > 0.0 {
        compute_loi_dv(traj).map(|(dv, _, _)| dv).unwrap_or(f64::MAX)
    } else {
        f64::MAX
    };
    if loi_ref_km_s > 0.0 && loi_dv_km_s < f64::MAX
        && loi_dv_km_s > loi_ref_km_s * LOI_DV_TOL_FACTOR
    {
        return -1e11;
    }
    // Among captured solutions prefer lower ΔV: each 0.1 km/s below reference adds ~1e4.
    let dv_bonus = if loi_ref_km_s > 0.0 && loi_dv_km_s < f64::MAX {
        (loi_ref_km_s - loi_dv_km_s) * 1e5
    } else {
        0.0
    };

    if captured       { 1e9 + n_orbits * 1e6 + dv_bonus }
    else if n_entries > 0 { n_orbits * 1e3 - min_r_m / 1e3 }
    else              { -min_r_m }
}

// ── IC conversion ─────────────────────────────────────────────────────────────

fn to_eci(
    theta: f64, r_apo: f64,
    mu: f64, r_park: f64, l_star: f64, v_star: f64,
    r3d: &Matrix3<f64>,
) -> Option<(SVector<f64, 3>, SVector<f64, 3>)> {
    let ic = tli_injection_ic(mu, r_park, r_apo, theta)?;
    let (x, y, z, vx, vy, vz) = (ic[0], ic[1], ic[2], ic[3], ic[4], ic[5]);
    let (rx, ry) = (x + mu, y);
    // Rotating → Earth-centred inertial velocity: v_EC = v_rot + ω × r_EC
    // ω × r_EC = (−ry, rx, 0) → v_EC = (vx−ry, vy+rx, vz)
    let pos_bcr = SVector::<f64, 3>::new(rx, ry, z);
    let vel_bcr = SVector::<f64, 3>::new(vx - ry, vy + rx, vz);
    // Apply full 3-D rotation into Moon's orbital plane instead of 2-D azimuth-only rotation.
    Some((r3d * pos_bcr * l_star, r3d * vel_bcr * v_star))
}

// ── ANISE loader + epoch search ───────────────────────────────────────────────

fn find_kernel_opt(name: &str) -> Option<String> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let p = dir.join("kernels").join(name);
        if p.exists() { return Some(p.to_string_lossy().into_owned()); }
        if !dir.pop() { return None; }
    }
}

/// Wrap angle difference into [0, π].
fn angle_err(a: f64, b: f64) -> f64 {
    let mut d = (a - b).rem_euclid(2.0 * PI);
    if d > PI { d = 2.0 * PI - d; }
    d
}

/// Find the calendar epoch where the real Sun–Moon angle in ECI best matches
/// `theta_sun_deg` (the BCR4BP Sun phase relative to the Moon direction).
///
/// `year_range` restricts the coarse search to [start_year, end_year] (inclusive).
/// Use (2000, 2030) for the full archive.
///
/// Two-stage search:
///   1. Coarse: 1-day steps over the year range.
///   2. Fine:   1-hour steps in a ±2-day window around the coarse best.
fn find_matching_epoch(
    almanac:      &Almanac,
    theta_sun_deg: f64,
    year_range:   (i32, i32),
) -> (Epoch, f64) {
    let target  = theta_sun_deg.to_radians();
    let day_s   = 86_400.0_f64;
    let hour_s  =  3_600.0_f64;

    let (start_year, end_year) = year_range;
    // Coarse pass
    let coarse_start = Epoch::from_gregorian_utc(start_year, 1,  1,  0, 0, 0, 0);
    let coarse_end   = Epoch::from_gregorian_utc(end_year,   12, 31, 23, 59, 0, 0);

    let mut best_ep  = coarse_start;
    let mut best_err = f64::MAX;
    let mut t = coarse_start;

    while t <= coarse_end {
        if let (Ok(mp), Ok(sp)) = (almanac.moon_position(t), almanac.sun_position(t)) {
            let psi_m = mp.inner[1].atan2(mp.inner[0]);
            let psi_s = sp.inner[1].atan2(sp.inner[0]);
            let err   = angle_err(psi_s - psi_m, target);
            if err < best_err { best_err = err; best_ep = t; }
        }
        t = t + Duration::from_seconds(day_s);
    }

    // Fine pass: ±2 days around coarse best, 1-hour steps
    let fine_start = best_ep + Duration::from_seconds(-2.0 * day_s);
    let fine_end   = best_ep + Duration::from_seconds( 2.0 * day_s);
    let mut t = fine_start;

    while t <= fine_end {
        if let (Ok(mp), Ok(sp)) = (almanac.moon_position(t), almanac.sun_position(t)) {
            let psi_m = mp.inner[1].atan2(mp.inner[0]);
            let psi_s = sp.inner[1].atan2(sp.inner[0]);
            let err   = angle_err(psi_s - psi_m, target);
            if err < best_err { best_err = err; best_ep = t; }
        }
        t = t + Duration::from_seconds(hour_s);
    }

    (best_ep, best_err)
}

/// Open the ANISE almanac, search for the matching epoch, then pre-sample tracks.
///
/// Samples Moon + Sun (for the blended provider) and Venus + Mars + Jupiter
/// (for the extra third-body perturbations).  All sampled at 1-hour intervals.
fn open_and_sample(
    theta_sun_deg: f64,
    t_prop_s:      f64,
    year_range:    (i32, i32),
) -> Option<(AniseTrack, SVector<f64, 3>, SVector<f64, 3>, SVector<f64, 3>, Epoch, f64)> {
    let bsp = find_kernel_opt("de440s.bsp")?;
    let almanac = Almanac::new(&bsp).ok()?;

    let (epoch, err_deg) = find_matching_epoch(&almanac, theta_sun_deg, year_range);

    let n = (t_prop_s / SAMPLE_DT_S) as usize + 2;
    let mut moons:   Vec<SVector<f64, 3>> = Vec::with_capacity(n);
    let mut suns:    Vec<SVector<f64, 3>> = Vec::with_capacity(n);
    let mut venuses: Vec<SVector<f64, 3>> = Vec::with_capacity(n);
    let mut marses:  Vec<SVector<f64, 3>> = Vec::with_capacity(n);
    let mut jupiters:Vec<SVector<f64, 3>> = Vec::with_capacity(n);

    for i in 0..n {
        let ep = epoch + Duration::from_seconds(i as f64 * SAMPLE_DT_S);
        moons   .push(almanac.moon_position(ep)   .ok()?.inner);
        suns    .push(almanac.sun_position(ep)    .ok()?.inner);
        venuses .push(almanac.venus_position(ep)  .ok()?.inner);
        marses  .push(almanac.mars_position(ep)   .ok()?.inner);
        jupiters.push(almanac.jupiter_position(ep).ok()?.inner);
    }

    let (m0, s0) = (moons[0], suns[0]);
    // Finite-difference Moon velocity at epoch [m/s] — used to define the orbital plane.
    let moon_vel0 = (moons[1] - moons[0]) / SAMPLE_DT_S;

    let mk = |v: Vec<SVector<f64, 3>>| BodyTrack { positions: v, sample_dt_s: SAMPLE_DT_S };

    Some((
        AniseTrack {
            moon: mk(moons),
            sun:  mk(suns),
            perturbers: vec![
                (mk(venuses),  MU_VENUS),
                (mk(marses),   MU_MARS),
                (mk(jupiters), MU_JUPITER),
            ],
        },
        m0, s0, moon_vel0, epoch, err_deg,
    ))
}

// ── LOI ΔV estimate ───────────────────────────────────────────────────────────

/// Minimum ΔV [km/s] to circularise at periapsis.
///
/// At periapsis the radial component of the Moon-relative velocity is zero, so
/// the total speed equals the tangential (in-track) component.  The Moon
/// velocity is estimated by central finite difference of the stored moon_km
/// positions.  Returns (dv_km_s, r_peri_km, t_peri_days).
fn compute_loi_dv(traj: &[EciStep]) -> Option<(f64, f64, f64)> {
    // Find periapsis index
    let (i_peri, s_peri) = traj.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.r_moon_km.partial_cmp(&b.r_moon_km).unwrap())?;

    let r_peri_m = s_peri.r_moon_km * 1e3;   // [m]

    // Moon velocity at periapsis via central finite difference [km/s]
    let moon_vel_km_s: [f64; 3] = if i_peri > 0 && i_peri + 1 < traj.len() {
        let dt = traj[i_peri + 1].t_s - traj[i_peri - 1].t_s;
        if dt > 0.0 {
            std::array::from_fn(|k|
                (traj[i_peri + 1].moon_km[k] - traj[i_peri - 1].moon_km[k]) / dt
            )
        } else {
            [0.0; 3]
        }
    } else {
        [0.0; 3]
    };

    // Moon-relative speed at periapsis [km/s]
    let v_rel_km_s = (0..3)
        .map(|k| s_peri.vel_km_s[k] - moon_vel_km_s[k])
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();

    // Circular orbit speed at periapsis altitude [km/s]
    // MU_MOON is in m³/s²; r_peri_m in m → result in m/s → /1e3 for km/s
    let v_circ_km_s = (MU_MOON / r_peri_m).sqrt() / 1e3;

    let dv = (v_rel_km_s - v_circ_km_s).abs();
    Some((dv, s_peri.r_moon_km, s_peri.t_s / 86_400.0))
}

// ── CSV writer ────────────────────────────────────────────────────────────────

fn write_traj(
    csv: &mut String, lambda: f64, phase: &str,
    run_id: u64, theta_deg: f64, r_apo: f64,
    traj: &[EciStep],
) {
    for s in traj {
        writeln!(csv,
            "{lambda:.2},{phase},{run_id},{theta_deg:.4},{r_apo:.4},\
             {:.1},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            s.t_s,
            s.pos_km[0], s.pos_km[1], s.pos_km[2],
            s.moon_km[0], s.moon_km[1], s.moon_km[2],
            s.r_moon_km,
        ).unwrap();
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    // Parse --idx N  and  --year YYYY  from command line.
    let selected_idx: usize = std::env::args()
        .skip_while(|a| a != "--idx")
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(SELECTED_IDX);
    if selected_idx >= SOLUTIONS.len() {
        eprintln!("--idx {selected_idx} out of range (0–{})", SOLUTIONS.len() - 1);
        std::process::exit(1);
    }
    let year_filter: Option<i32> = std::env::args()
        .skip_while(|a| a != "--year")
        .nth(1)
        .and_then(|v| v.parse().ok());
    let year_range = year_filter
        .map(|y| (y, y))
        .unwrap_or((2000, 2030));
    if let Some(y) = year_filter {
        eprintln!("Restricting epoch search to year {y}");
    }

    let params    = CrtbpParams::earth_moon();
    let mu        = params.mu;
    let l_star    = params.l_star;
    let v_star    = params.v_star;
    let t_star    = params.t_star;
    let r_hill_m  = lunar_hill_radius(mu) * l_star;
    let t_prop_s  = T_PROP_ND * t_star;
    let h_max_s   = H_MAX_ND  * t_star;   // step cap for OutputType::Sparse
    let min_cap_s = MIN_CAP_ND * t_star;
    let period_s  = 2.0 * PI   * t_star;
    let n_moon    = 1.0 / t_star;
    let n_sun_eci = (1.0 + OMEGA_S) / t_star;
    let a_s_m     = A_S_ND * l_star;

    let (run_id, _seed_id, theta_deg, theta_sun_deg, r_apogee_nd) = SOLUTIONS[selected_idx];
    eprintln!("Solution: run_id={run_id}  θ={theta_deg:.3}°  θ_sun={theta_sun_deg:.3}°  r_apo={r_apogee_nd:.2}");

    let t_sun = theta_sun_deg.to_radians();

    // ── BCR4BP reference LOI ΔV ───────────────────────────────────────────────
    // Propagate the original BCR4BP IC (same as wsb_dense_traj) and compute the
    // minimum LOI ΔV using Moon-centred periapsis speed in the rotating frame.
    let bcr4bp_loi_km_s: f64 = {
        let v_km_s = v_star / 1e3;
        let bcr    = Bcr4bpParams::earth_moon_sun(t_sun);
        let ic     = tli_injection_ic(mu, R_PARK, r_apogee_nd, theta_deg.to_radians())
            .expect("BCR4BP IC failed");
        let traj   = propagate_bcr4bp_raw(mu, bcr, ic, T_PROP_ND, 0.02, 1e-8, 1e-10);
        let dv_tli = tli_dv(mu, R_PARK, r_apogee_nd) * v_km_s;
        let dv_loi = min_loi_dv(&traj, mu).unwrap_or(0.0) * v_km_s;
        eprintln!("BCR4BP reference:  TLI={dv_tli:.4} km/s  LOI={dv_loi:.4} km/s  total={:.4} km/s",
            dv_tli + dv_loi);
        dv_loi
    };

    // Search for the epoch where the real Sun–Moon angle matches theta_sun_deg.
    // At that epoch the circular (λ=0) and real (λ=1) providers have nearly the
    // same Sun position → blending is smooth and physically meaningful.
    let anise_result = open_and_sample(theta_sun_deg, t_prop_s, year_range);
    let mut tli_epoch: Option<Epoch> = None;
    let (anise_opt, r3d) = match anise_result {
        Some((track, m0, s0, moon_vel0, epoch, err)) => {
            tli_epoch = Some(epoch);
            let r3d_val = orbital_plane_r3d(m0, moon_vel0);
            let psi_m = m0[1].atan2(m0[0]);
            let psi_s_real = s0[1].atan2(s0[0]);
            let incl_deg = (m0[2] / m0.norm()).asin().to_degrees();
            eprintln!(
                "Matched epoch: {epoch}  err={:.3}°\n  \
                 Moon ECI: {:.2}°  incl={:.2}°  Sun ECI (real): {:.2}°  Δsun={:.2}°",
                err.to_degrees(),
                psi_m.to_degrees(),
                incl_deg,
                psi_s_real.to_degrees(),
                (psi_s_real - (psi_m + t_sun)).to_degrees(),
            );
            (Some(track), r3d_val)
        }
        None => {
            eprintln!("kernels/de440s.bsp not found — λ=0 only (Moon at +X in equatorial plane).");
            // Fallback: identity (Moon in x-direction, no inclination)
            (None, Matrix3::identity())
        }
    };

    fs::create_dir_all(OUT_DIR).expect("cannot create out/wsb");
    let mut csv  = String::from(
        "lambda,phase,run_id,theta_deg,r_apogee_nd,\
         t_s,x_km,y_km,z_km,moon_x_km,moon_y_km,moon_z_km,r_moon_km\n"
    );
    let mut meta = String::from(
        "lambda,model,theta_deg_before,r_apo_before,score_before,min_r_before_km,\
         theta_deg_after,r_apo_after,score_after,min_r_after_km,\
         delta_theta_deg,delta_r_apo,correction_applied\n"
    );
    // Animation CSV — "after" trajectories at each λ, downsampled to ≤2000 pts.
    let mut anim = String::from(
        "lambda,t_s,x_km,y_km,z_km,moon_x_km,moon_y_km,moon_z_km,r_moon_km\n"
    );

    let mut cur_theta    = theta_deg.to_radians();
    let mut cur_r_apo    = r_apogee_nd;
    let mut rng          = rand::rngs::StdRng::seed_from_u64(RNG_SEED);
    // Set from the λ=0 "before" run; used as the arrival-time anchor for all λ levels.
    let mut t_hill_ref_s = 0.0_f64;
    let mut loi_result: Option<(f64, f64, f64)> = None;
    // Collect (lambda, after_theta [rad], after_r_apo [nd]) for the eval re-propagation.
    let mut after_ics: Vec<(f64, f64, f64)> = Vec::new();

    let lambdas: &[f64] = if anise_opt.is_some() { LAMBDAS } else { &[0.0] };

    for &lambda in lambdas {
        let model_desc = format!(
            "Earth+Moon+Sun{}{}",
            if lambda > 0.0 { "+Venus+Mars+Jupiter+SRP" } else { "" },
            if lambda > 0.0 { format!(" (λ={lambda:.2} real ephemeris)") } else { " (circular)".to_string() },
        );
        eprintln!("\n  λ={lambda:.2}  [{model_desc}]");

        let circ = CircularOrbits {
            moon_r_m: l_star, sun_r_m: a_s_m,
            n_moon, n_sun_eci,
            theta_sun_rad: t_sun,
            r3d,
        };

        // Helper: propagate with correct provider + extra bodies
        let propagate_ic = |theta: f64, r_apo: f64| -> Option<Vec<EciStep>> {
            let (r0, v0) = to_eci(theta, r_apo, mu, R_PARK, l_star, v_star, &r3d)?;
            Some(if let Some(ref anise) = anise_opt {
                let blend = Blended { circ: &circ, real: anise, lambda };
                propagate(r0, v0, &blend, &anise.perturbers, lambda, r_hill_m, t_prop_s, h_max_s)
            } else {
                propagate(r0, v0, &circ, &[], lambda, r_hill_m, t_prop_s, h_max_s)
            })
        };

        // ── "before" trajectory ───────────────────────────────────────────────
        let before_traj  = propagate_ic(cur_theta, cur_r_apo).expect("IC failed");

        // Lock the arrival-time reference from the λ=0 run (before any correction).
        if t_hill_ref_s == 0.0 {
            t_hill_ref_s = before_traj.iter()
                .find(|s| s.r_moon_km * 1e3 < r_hill_m)
                .map(|s| s.t_s)
                .unwrap_or(0.0);
            if t_hill_ref_s > 0.0 {
                eprintln!("    reference Hill arrival: {:.1} days", t_hill_ref_s / 86_400.0);
            }
        }

        let before_score = score(&before_traj, r_hill_m, period_s, min_cap_s, t_hill_ref_s, bcr4bp_loi_km_s);
        let before_min   = before_traj.iter().map(|s| s.r_moon_km).fold(f64::MAX, f64::min);

        eprintln!("    before: θ={:.3}°  r={:.3}  min_r={:.0} km  score={:.1}",
            cur_theta.to_degrees(), cur_r_apo, before_min, before_score);

        write_traj(&mut csv, lambda, "before", run_id,
                   cur_theta.to_degrees(), cur_r_apo, &before_traj);

        // ── Shooting correction ───────────────────────────────────────────────
        let needs_correction = before_score < 1e8;

        let (after_theta, after_r_apo, after_traj, corrected) = if needs_correction {
            eprint!("    shooting");
            let mut best_score  = before_score;
            let mut best_theta  = cur_theta;
            let mut best_r_apo  = cur_r_apo;
            let mut best_traj: Vec<EciStep> = Vec::new();
            let mut used_before = true;

            'outer: for (round, &(r_theta, r_apo)) in SEARCH_ROUNDS.iter().enumerate() {
                eprint!(" [r{}: ±{:.0}°/±{:.1}]", round+1,
                    r_theta.to_degrees(), r_apo);
                for _ in 0..TRIALS_PER_ROUND {
                    let dth = rng.gen_range(-r_theta..r_theta);
                    let dra = rng.gen_range(-r_apo..r_apo);
                    let th  = cur_theta + dth;
                    let ra  = (cur_r_apo + dra).max(1.5);

                    let Some(traj) = propagate_ic(th, ra) else { continue };
                    let s = score(&traj, r_hill_m, period_s, min_cap_s, t_hill_ref_s, bcr4bp_loi_km_s);
                    if s > best_score {
                        best_score = s; best_theta = th;
                        best_r_apo = ra; best_traj = traj;
                        used_before = false;
                    }
                    if best_score > 1e9 { break 'outer; }
                }
                if best_score > 0.0 { break; }
            }

            let after_traj = if used_before { before_traj } else { best_traj };
            let after_min  = after_traj.iter().map(|s| s.r_moon_km).fold(f64::MAX, f64::min);
            eprintln!("\n    after:  θ={:.3}°  r={:.3}  min_r={:.0} km  score={:.1}  \
                       Δθ={:+.3}°  Δr={:+.4}",
                best_theta.to_degrees(), best_r_apo, after_min, best_score,
                (best_theta - cur_theta).to_degrees(), best_r_apo - cur_r_apo);

            (best_theta, best_r_apo, after_traj, true)
        } else {
            eprintln!("    no correction needed — already captures");
            (cur_theta, cur_r_apo, before_traj, false)
        };

        // Metadata row
        let after_min  = after_traj.iter().map(|s| s.r_moon_km).fold(f64::MAX, f64::min);
        let after_score = score(&after_traj, r_hill_m, period_s, min_cap_s, t_hill_ref_s, bcr4bp_loi_km_s);
        writeln!(meta,
            "{lambda:.2},{model_desc},{:.4},{:.4},{:.2},{:.1},\
             {:.4},{:.4},{:.2},{:.1},{:+.4},{:+.5},{corrected}",
            cur_theta.to_degrees(), cur_r_apo, before_score, before_min,
            after_theta.to_degrees(), after_r_apo, after_score, after_min,
            (after_theta - cur_theta).to_degrees(), after_r_apo - cur_r_apo,
        ).unwrap();

        write_traj(&mut csv, lambda, "after", run_id,
                   after_theta.to_degrees(), after_r_apo, &after_traj);

        // Downsample for animation (every Nth step, max 2000 points)
        let step = (after_traj.len() / 2000).max(1);
        for s in after_traj.iter().step_by(step) {
            writeln!(anim,
                "{lambda:.2},{:.1},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
                s.t_s,
                s.pos_km[0], s.pos_km[1], s.pos_km[2],
                s.moon_km[0], s.moon_km[1], s.moon_km[2],
                s.r_moon_km,
            ).unwrap();
        }

        loi_result = compute_loi_dv(&after_traj);
        after_ics.push((lambda, after_theta, after_r_apo));

        cur_theta = after_theta;
        cur_r_apo = after_r_apo;
    }

    eprintln!();
    eprintln!("── Solution summary ─────────────────────────────────────────");
    if let Some(ep) = tli_epoch {
        eprintln!("  TLI departure : {ep}");
        if let Some((_, _, t_peri)) = loi_result {
            // Lunar arrival = TLI epoch + t_peri days
            let arrival = ep + hifitime::Duration::from_seconds(t_peri * 86_400.0);
            eprintln!("  Lunar arrival : {arrival}  ({t_peri:.1} days transfer)");
        }
    }
    eprintln!("── LOI ΔV ───────────────────────────────────────────────────");
    eprintln!("  BCR4BP (λ=0, reference) : {:.4} km/s", bcr4bp_loi_km_s);
    if let Some((dv, r_peri, t_peri)) = loi_result {
        let alt_km = r_peri - 1_737.4;
        eprintln!("  Real ephemeris (λ=1)    : {:.4} km/s", dv);
        eprintln!("  Δ(LOI)                  : {:+.4} km/s", dv - bcr4bp_loi_km_s);
        eprintln!("  Periapsis               : {:.0} km from Moon centre  ({:.0} km altitude)", r_peri, alt_km);
    } else {
        eprintln!("  Real ephemeris (λ=1)    : n/a (no Hill sphere entry)");
    }
    eprintln!("─────────────────────────────────────────────────────────────");

    // ── Eval: re-propagate every corrected IC with full real ephemeris (λ=1) ──────
    // This shows IC convergence: does the λ=0 IC capture in reality? The λ=0.25? etc.
    // All trajectories use the same force model (λ=1), so the SOI view is comparable.
    let mut eval_csv = String::from(
        "lambda_origin,t_s,x_km,y_km,z_km,moon_x_km,moon_y_km,moon_z_km,r_moon_km\n"
    );
    if let Some(ref anise) = anise_opt {
        eprintln!("\nRe-propagating {} ICs at λ=1 for eval CSV ...", after_ics.len());
        for &(lam_orig, theta, r_apo) in &after_ics {
            eprint!("  IC from λ={lam_orig:.2} … ");
            let Some((r0, v0)) = to_eci(theta, r_apo, mu, R_PARK, l_star, v_star, &r3d) else {
                eprintln!("IC failed"); continue
            };
            let traj = propagate(r0, v0, anise, &anise.perturbers, 1.0,
                                 r_hill_m, t_prop_s, h_max_s);
            let min_r = traj.iter().map(|s| s.r_moon_km).fold(f64::MAX, f64::min);
            eprintln!("{} pts  min_r={min_r:.0} km", traj.len());
            let step = (traj.len() / 2000).max(1);
            for s in traj.iter().step_by(step) {
                writeln!(eval_csv,
                    "{lam_orig:.2},{:.1},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
                    s.t_s,
                    s.pos_km[0], s.pos_km[1], s.pos_km[2],
                    s.moon_km[0], s.moon_km[1], s.moon_km[2],
                    s.r_moon_km,
                ).unwrap();
            }
        }
    }

    fs::write(OUT_CSV,  &csv) .expect("traj csv write failed");
    fs::write(OUT_META, &meta).expect("meta csv write failed");
    fs::write(OUT_ANIM, &anim).expect("anim csv write failed");
    fs::write(OUT_EVAL, &eval_csv).expect("eval csv write failed");
    eprintln!("Saved {OUT_EVAL}");
    eprintln!("\nSaved {OUT_CSV},  {OUT_META},  {OUT_ANIM}");
}
