//! wsb_circularize — Lunar Orbit Insertion (LOI) burn after WSB ballistic capture.
//!
//! 1. Reads the max-fidelity BCR4BP trajectory, finds the first periapsis inside
//!    the Moon's Hill sphere, computes a circularization ΔV in rotating-frame coords.
//! 2. Converts the burn state from BCR4BP rotating frame to ECI (J2000, km / km s⁻¹)
//!    using the WSB departure epoch derived from `theta_sun_deg` + DE440S Moon+Sun
//!    ephemeris (Newton refinement, same algorithm as the Python plotting script).
//! 3. Propagates the post-LOI orbit with 2-body Moon-centred dynamics (km / km s⁻¹),
//!    then adds DE440S Moon position back to produce ECI output.
//!
//! # Outputs (saved to out/wsb_circularize/)
//!   capture.csv          — pre-burn BCR4BP trajectory (time_nd, x_nd, …)
//!   loi_orbit.csv        — post-burn ECI orbit (time_s, x_km, y_km, z_km,
//!                          moon_x_km, moon_y_km, moon_z_km)
//!   epoch_info.txt       — WSB departure epoch + R0_wsb for the Python plotter
//!   circularize_info.txt — LOI summary (burn time, altitude, ΔV)
//!
//! # Usage
//! ```
//!   cargo run -p lunar_trajectories --bin wsb_circularize --release
//! ```

use std::f64::consts::PI;
use std::fs;
use std::io::{BufRead, BufWriter, Write as IoWrite};
use std::time::Instant;

use ode_solvers::dopri5::Dopri5;
use ode_solvers::dop_shared::OutputType;
use ode_solvers::{SVector as OdeVec, System};

use lunar_trajectories::crtbp::CrtbpParams;
use lunar_trajectories::propagator::Bcr4bpParams;
use lunar_trajectories::transfers::lunar_hill_radius;

// ── Paths ─────────────────────────────────────────────────────────────────────
const MAXHIFI_CSV:    &str = "out/wsb/maxhifi.csv";
const HIFI_CSV:       &str = "out/wsb/solution_hifi.csv";
const ART_CSV:        &str = "../Artemis/out/artemis2_trajectory.csv";
const MOON_EPHEM_CSV: &str = "../Artemis/out/moon_ephem.csv";
const OUT_DIR:        &str = "out/wsb";

// ── BCR4BP / EM system constants ──────────────────────────────────────────────
const T_STAR: f64 = 375_700.0;         // s (Earth-Moon period / 2π)
const L_STAR: f64 = 384_400.0;         // km (Earth-Moon distance)
const MU:     f64 = 0.012_155_65;

// ── Physical constants ────────────────────────────────────────────────────────
const GM_MOON: f64 = 4_902.8;
const R_MOON:  f64 = 1_737.4;

// ── Synodic period (Moon around Earth) ───────────────────────────────────────
const MOON_SIDEREAL_DAYS: f64 = 27.321_661;
const SYNODIC_DAYS: f64 = 1.0 / (1.0 / MOON_SIDEREAL_DAYS - 1.0 / 365.25);

// ── Integrator settings ───────────────────────────────────────────────────────
const RTOL:   f64 = 1e-10;
const ATOL:   f64 = 1e-12;
const _H_INIT: f64 = 1e-4;            // ND (capture phase)
const H_INIT_LOI: f64 = 1.0;          // seconds (LOI orbit phase)
const N_MAX:  u32 = 1_000_000_000;

const N_ORBITS: f64 = 8.0;
const LUNAR_PERIOD_ND: f64 = 2.0 * PI;
const _SAVE_DT_ND: f64 = 3_600.0 / T_STAR;  // 1 hour in ND
const SAVE_DT_S:  f64 = 3_600.0;

type State6 = OdeVec<f64, 6>;

// ════════════════════════════════════════════════════════════════════════════
// Ephemeris table
// ════════════════════════════════════════════════════════════════════════════

struct EphemTrack {
    time_s: Vec<f64>,
    x_km:   Vec<f64>,
    y_km:   Vec<f64>,
    z_km:   Vec<f64>,
}

impl EphemTrack {
    fn interp(&self, t: f64) -> [f64; 3] {
        let n = self.time_s.len();
        if n == 0 { return [0.0; 3]; }
        let i = self.time_s.partition_point(|&ts| ts <= t).min(n - 1);
        if i == 0 { return [self.x_km[0], self.y_km[0], self.z_km[0]]; }
        let i0 = i - 1;
        let dt = self.time_s[i] - self.time_s[i0];
        if dt.abs() < 1e-15 { return [self.x_km[i0], self.y_km[i0], self.z_km[i0]]; }
        let f = (t - self.time_s[i0]) / dt;
        [
            self.x_km[i0] + f * (self.x_km[i] - self.x_km[i0]),
            self.y_km[i0] + f * (self.y_km[i] - self.y_km[i0]),
            self.z_km[i0] + f * (self.z_km[i] - self.z_km[i0]),
        ]
    }
}

fn load_ephem(path: &str) -> (EphemTrack, EphemTrack) {
    let file = fs::File::open(path)
        .unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let mut moon = EphemTrack { time_s: vec![], x_km: vec![], y_km: vec![], z_km: vec![] };
    let mut sun  = EphemTrack { time_s: vec![], x_km: vec![], y_km: vec![], z_km: vec![] };
    for (i, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.unwrap();
        if i == 0 { continue; }
        let c: Vec<f64> = line.split(',')
            .map(|s| s.trim().parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        if c.len() < 7 { continue; }
        moon.time_s.push(c[0]); moon.x_km.push(c[1]*1e-3); moon.y_km.push(c[2]*1e-3); moon.z_km.push(c[3]*1e-3);
        sun.time_s.push(c[0]);  sun.x_km.push(c[4]*1e-3);  sun.y_km.push(c[5]*1e-3);  sun.z_km.push(c[6]*1e-3);
    }
    (moon, sun)
}

// ════════════════════════════════════════════════════════════════════════════
// 2-body ODE (Moon-centred, km / km s⁻¹)
// ════════════════════════════════════════════════════════════════════════════

struct TwoBodyMoon;

impl System<f64, State6> for TwoBodyMoon {
    fn system(&self, _t: f64, y: &State6, dy: &mut State6) {
        let r  = (y[0]*y[0] + y[1]*y[1] + y[2]*y[2]).sqrt();
        let a  = -GM_MOON / (r * r * r);
        dy[0] = y[3]; dy[1] = y[4]; dy[2] = y[5];
        dy[3] = a * y[0]; dy[4] = a * y[1]; dy[5] = a * y[2];
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BCR4BP capture CSV loader
// ════════════════════════════════════════════════════════════════════════════

struct Row { time: f64, x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, dist_moon: f64, in_hill: bool }

fn load_maxhifi(path: &str) -> Vec<Row> {
    let file = fs::File::open(path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let mut rows = Vec::new();
    for (i, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.unwrap();
        if i == 0 { continue; }
        let c: Vec<f64> = line.split(',').map(|s| s.trim().parse::<f64>().unwrap_or(0.0)).collect();
        if c.len() < 11 { continue; }
        rows.push(Row { time: c[1], x: c[2], y: c[3], z: c[4], vx: c[5], vy: c[6], vz: c[7], dist_moon: c[9], in_hill: c[10] > 0.5 });
    }
    rows
}

fn load_theta_sun(path: &str) -> f64 {
    let file = fs::File::open(path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let mut lines = std::io::BufReader::new(file).lines();
    let header = lines.next().expect("empty hifi csv").unwrap();
    let row    = lines.next().expect("no data row").unwrap();
    let cols: Vec<&str> = header.split(',').collect();
    let vals: Vec<f64>  = row.split(',').map(|s| s.trim().parse::<f64>().unwrap_or(f64::NAN)).collect();
    let idx = cols.iter().position(|c| c.trim() == "theta_sun_deg").expect("theta_sun_deg not found");
    vals[idx]
}

/// Read art_t_s0 (time_s of first row) from artemis2_trajectory.csv.
fn load_art_t_s0(path: &str) -> f64 {
    let file = fs::File::open(path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let mut lines = std::io::BufReader::new(file).lines();
    let header = lines.next().expect("empty art csv").unwrap();
    let row    = lines.next().expect("no data row in art csv").unwrap();
    let cols: Vec<&str> = header.split(',').collect();
    let vals: Vec<f64>  = row.split(',').map(|s| s.trim().parse::<f64>().unwrap_or(f64::NAN)).collect();
    let idx = cols.iter().position(|c| c.trim() == "time_s").expect("time_s not found");
    vals[idx]
}

fn find_periapsis(rows: &[Row]) -> Option<usize> {
    let entry = rows.iter().position(|r| r.in_hill)?;
    for i in (entry + 1)..(rows.len() - 1) {
        if rows[i].in_hill && rows[i].dist_moon < rows[i-1].dist_moon && rows[i].dist_moon < rows[i+1].dist_moon {
            return Some(i);
        }
    }
    None
}

// ════════════════════════════════════════════════════════════════════════════
// BCR4BP rotating frame → ECI conversion
// ════════════════════════════════════════════════════════════════════════════

/// Normalise a 3-vector in place, returning the original magnitude.
fn normalise(v: &mut [f64; 3]) -> f64 {
    let mag = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    v[0] /= mag; v[1] /= mag; v[2] /= mag;
    mag
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1],
     a[2]*b[0] - a[0]*b[2],
     a[0]*b[1] - a[1]*b[0]]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }

/// Multiply 3×3 matrix (column-major) by vector.
fn mat_vec(r: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0]*v[0] + r[1][0]*v[1] + r[2][0]*v[2],
        r[0][1]*v[0] + r[1][1]*v[1] + r[2][1]*v[2],
        r[0][2]*v[0] + r[1][2]*v[1] + r[2][2]*v[2],
    ]
}

/// BCR4BP rotating frame → ECI (km).
/// t_nd: dimensionless time (angle the EM frame has rotated since departure).
#[allow(dead_code)]
fn rot_to_eci(x_nd: f64, y_nd: f64, z_nd: f64, t_nd: f64, r0: &[[f64; 3]; 3]) -> [f64; 3] {
    let ec_x = x_nd + MU;
    let xi = (ec_x * t_nd.cos() - y_nd * t_nd.sin()) * L_STAR;
    let yi = (ec_x * t_nd.sin() + y_nd * t_nd.cos()) * L_STAR;
    let zi = z_nd * L_STAR;
    mat_vec(r0, [xi, yi, zi])
}

/// BCR4BP rotating-frame velocity → ECI velocity (km/s).
/// v_nd: [vx_nd, vy_nd, vz_nd] in rotating frame.
#[allow(dead_code)]
fn vel_rot_to_eci(x_nd: f64, y_nd: f64, _z_nd: f64,
                  vx_nd: f64, vy_nd: f64, vz_nd: f64,
                  t_nd: f64, r0: &[[f64; 3]; 3]) -> [f64; 3] {
    // Inertial velocity in EM-inertial frame (ND units):
    //   v_inert = R(t_nd)^T · (v_rot + ω×r_rot)   where ω = ẑ in ND
    // In ND rotating frame: ω×r = (-y, x, 0)
    let ec_x = x_nd + MU;
    let vix_nd = vx_nd - y_nd;              // v_inert_x (ND) in EM-inertial at t=0
    let viy_nd = vy_nd + ec_x;              // v_inert_y
    let viz_nd = vz_nd;
    // Rotate by angle t_nd (EM rotating frame has rotated by t_nd since t=0)
    let vix_rot = vix_nd * t_nd.cos() - viy_nd * t_nd.sin();
    let viy_rot = vix_nd * t_nd.sin() + viy_nd * t_nd.cos();
    let viz_rot = viz_nd;
    let v_kms = [vix_rot * L_STAR / T_STAR,
                 viy_rot * L_STAR / T_STAR,
                 viz_rot * L_STAR / T_STAR];
    mat_vec(r0, v_kms)
}

// ════════════════════════════════════════════════════════════════════════════
// WSB departure epoch — mirrors plot_wsb_vs_artemis.py's compute_wsb_epoch().
//
// Uses circular-Moon approximation for R0_wsb (same as the reference plot),
// plus synodic N selection to place the WSB LOI near target_loi_days.
// Returns (departure_t_s, R0_wsb).
// ════════════════════════════════════════════════════════════════════════════

/// R0 at Artemis TLI time (same as Python's compute_eci_frame).
fn compute_r0_tli(moon: &EphemTrack, art_t_s0: f64) -> [[f64; 3]; 3] {
    let m0 = moon.interp(art_t_s0);
    let m1 = moon.interp(art_t_s0 + 3600.0);
    let m_vel = [(m1[0]-m0[0])/3600.0, (m1[1]-m0[1])/3600.0, (m1[2]-m0[2])/3600.0];
    let mut x_em = m0;  normalise(&mut x_em);
    let mut z_em = cross(&m0, &m_vel); normalise(&mut z_em);
    let mut y_em = cross(&z_em, &x_em); normalise(&mut y_em);
    [x_em, y_em, z_em]
}

/// Compute u_sun_eci at art_t_s0.
fn compute_u_sun(sun: &EphemTrack, art_t_s0: f64) -> [f64; 3] {
    let mut s = sun.interp(art_t_s0);
    normalise(&mut s);
    s
}

fn compute_wsb_departure(
    theta_sun_deg: f64,
    art_t_s0: f64,
    burn_time_days: f64,   // BCR4BP trajectory duration to LOI
    target_loi_days: f64,  // target WSB LOI time from art_t_s0
    moon: &EphemTrack,
    sun: &EphemTrack,
) -> (f64, [[f64; 3]; 3]) {
    let r0        = compute_r0_tli(moon, art_t_s0);
    let u_sun_eci = compute_u_sun(sun, art_t_s0);

    // Sun angle in EM frame at Artemis TLI
    let sun_in_em_x = dot(&u_sun_eci, &r0[0]);
    let sun_in_em_y = dot(&u_sun_eci, &r0[1]);
    let theta_sun_artemis = sun_in_em_y.atan2(sun_in_em_x).to_degrees();

    // Synodic offset (single step, same as plot_wsb_vs_artemis.py)
    let delta_theta = ((theta_sun_deg - theta_sun_artemis) + 180.0).rem_euclid(360.0) - 180.0;
    let offset_base = delta_theta / (-360.0 / SYNODIC_DAYS);

    // Pick synodic N to place WSB LOI near target_loi_days
    let target_dep = target_loi_days - burn_time_days;
    let best_n     = ((target_dep - offset_base) / SYNODIC_DAYS).round() as i64;
    let offset_days = offset_base + best_n as f64 * SYNODIC_DAYS;

    eprintln!("  Synodic N={best_n:+}: departure={offset_days:.2} d, WSB LOI @ {:.2} d",
        offset_days + burn_time_days);

    // R0_wsb: circular-Moon rotation from art_t_s0 (same as plot_wsb_vs_artemis.py)
    let moon_angle_rad = 2.0 * PI * offset_days / MOON_SIDEREAL_DAYS;
    let ca = moon_angle_rad.cos();
    let sa = moon_angle_rad.sin();
    let new_x = mat_vec(&r0, [ca, sa, 0.0]);
    let z_em  = r0[2];
    let mut new_y = cross(&z_em, &new_x); normalise(&mut new_y);
    let new_z = cross(&new_x, &new_y);
    let r0_wsb = [new_x, new_y, new_z];

    let dep_t_s = art_t_s0 + offset_days * 86_400.0;
    (dep_t_s, r0_wsb)
}

// ════════════════════════════════════════════════════════════════════════════
// ΔV computation (BCR4BP rotating frame)
// ════════════════════════════════════════════════════════════════════════════

struct BurnResult { dv: [f64; 3], dv_mag: f64, v_rel_pre: f64, v_circ: f64 }

fn circularization_dv(mu: f64, r: &Row) -> BurnResult {
    let moon_x = 1.0 - mu;
    let (rx, ry, rz) = (r.x - moon_x, r.y, r.z);
    let r_mag = (rx*rx + ry*ry + rz*rz).sqrt();
    let (vrx, vry, vrz) = (r.vx - r.y, r.vy + r.x - moon_x, r.vz);
    let v_rel_mag = (vrx*vrx + vry*vry + vrz*vrz).sqrt();
    let v_circ = (mu / r_mag).sqrt();
    let (hx, hy, hz) = (ry*vrz - rz*vry, rz*vrx - rx*vrz, rx*vry - ry*vrx);
    let h_mag = (hx*hx + hy*hy + hz*hz).sqrt();
    let (hxn, hyn, hzn) = (hx/h_mag, hy/h_mag, hz/h_mag);
    let (rxn, ryn, rzn) = (rx/r_mag, ry/r_mag, rz/r_mag);
    let tx = hyn*rzn - hzn*ryn;
    let ty = hzn*rxn - hxn*rzn;
    let tz = hxn*ryn - hyn*rxn;
    let (dvx, dvy, dvz) = (v_circ*tx - vrx, v_circ*ty - vry, v_circ*tz - vrz);
    let dv_mag = (dvx*dvx + dvy*dvy + dvz*dvz).sqrt();
    BurnResult { dv: [dvx, dvy, dvz], dv_mag, v_rel_pre: v_rel_mag, v_circ }
}

// ════════════════════════════════════════════════════════════════════════════
// CSV writers
// ════════════════════════════════════════════════════════════════════════════

fn write_capture_csv(path: &str, rows: &[Row]) {
    let f = fs::File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,dist_moon_nd").unwrap();
    for r in rows {
        writeln!(w, "{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10}",
            r.time, r.x, r.y, r.z, r.vx, r.vy, r.vz, r.dist_moon).unwrap();
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════

fn main() {
    let params  = CrtbpParams::earth_moon();
    let mu      = params.mu;
    let l_km    = params.l_star / 1e3;
    let v_km_s  = params.v_star / 1e3;
    let r_hill  = lunar_hill_radius(mu);

    fs::create_dir_all(OUT_DIR).unwrap();

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║     WSB Lunar Orbit Insertion (LOI) Burn                     ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // ── Load DE440S ephemeris ─────────────────────────────────────────────────
    eprintln!("  Loading DE440S ephemeris ({MOON_EPHEM_CSV}) ...");
    let (moon_ephem, sun_ephem) = load_ephem(MOON_EPHEM_CSV);
    eprintln!("  Ephemeris: {} pts", moon_ephem.time_s.len());

    // ── Artemis departure reference epoch ─────────────────────────────────────
    let art_t_s0 = load_art_t_s0(ART_CSV);
    eprintln!("  Artemis CSV t_s0 = {art_t_s0:.0} s from TLI");

    // ── Load BCR4BP trajectory ────────────────────────────────────────────────
    let theta_sun_deg = load_theta_sun(HIFI_CSV);
    let _bcr = Bcr4bpParams::earth_moon_sun(theta_sun_deg.to_radians());
    eprintln!("  θ_sun = {theta_sun_deg:.4}°   Hill r = {:.4} nd ({:.0} km)", r_hill, r_hill*l_km);

    eprintln!("  Loading {MAXHIFI_CSV} ...");
    let traj = load_maxhifi(MAXHIFI_CSV);
    eprintln!("  Loaded {} pts  t_end = {:.2} nd ({:.1} d)",
        traj.len(),
        traj.last().map(|r| r.time).unwrap_or(0.0),
        traj.last().map(|r| r.time).unwrap_or(0.0) * T_STAR / 86_400.0);

    // ── Find LOI burn point ───────────────────────────────────────────────────
    let burn_idx = find_periapsis(&traj)
        .expect("No periapsis inside Hill sphere");
    let burn = &traj[burn_idx];
    let burn_alt_km = burn.dist_moon * l_km - R_MOON;
    let burn_time_days = burn.time * T_STAR / 86_400.0;

    eprintln!();
    eprintln!("  LOI burn: t={:.4} nd ({:.2} d)  alt={:.1} km  r_moon={:.4} nd",
        burn.time, burn_time_days, burn_alt_km, burn.dist_moon);

    // ── Compute ΔV in rotating frame ──────────────────────────────────────────
    let br = circularization_dv(mu, burn);
    eprintln!("  v_rel={:.5} nd ({:.4} km/s)  v_circ={:.5} nd ({:.4} km/s)  ΔV={:.5} nd ({:.4} km/s)",
        br.v_rel_pre, br.v_rel_pre*v_km_s, br.v_circ, br.v_circ*v_km_s, br.dv_mag, br.dv_mag*v_km_s);

    // ── Compute WSB departure epoch ────────────────────────────────────────────
    eprintln!();
    eprintln!("  Computing WSB departure epoch (Newton on theta_sun) ...");
    // Target: WSB LOI roughly 15 d after Artemis LOI.
    // Artemis LOI is typically ~4.7 d from art_t_s0 → target ~20 d from art_t_s0.
    // This just steers synodic N; exact value doesn't affect the trajectory.
    let target_loi_days = 20.0;
    let (dep_t_s, r0_wsb) = compute_wsb_departure(
        theta_sun_deg, art_t_s0, burn_time_days, target_loi_days,
        &moon_ephem, &sun_ephem,
    );
    let burn_t_s = dep_t_s + burn.time * T_STAR;
    eprintln!("  WSB departure t_s = {dep_t_s:.0} s  burn t_s = {burn_t_s:.0} s");

    // ── Moon-centred initial state from BCR4BP rotating frame ────────────────
    // Compute Moon-relative pos/vel directly in rotating frame then rotate to
    // Moon-centred ECI — avoids mixing circular-Moon ECI with DE440S Moon.
    let vx_post = burn.vx + br.dv[0];
    let vy_post = burn.vy + br.dv[1];
    let vz_post = burn.vz + br.dv[2];
    let t_burn  = burn.time;

    // Moon-relative position in rotating frame (ND) → unrotate → apply R0_wsb
    let x_mc_nd = burn.x - (1.0 - MU);
    let r_mc_ix = (x_mc_nd * t_burn.cos() - burn.y * t_burn.sin()) * L_STAR;
    let r_mc_iy = (x_mc_nd * t_burn.sin() + burn.y * t_burn.cos()) * L_STAR;
    let r_mc_iz = burn.z * L_STAR;
    let r0_mc = mat_vec(&r0_wsb, [r_mc_ix, r_mc_iy, r_mc_iz]);

    // Moon-centred inertial velocity (ND): v_sc_inert - v_moon_inert
    //   v_sc_inert  = (vx - y,  vy + x,      vz)   [ω×r, barycenter coords]
    //   v_moon_inert= (0,        1-MU,         0)   [ω×r_moon = (0,1-MU,0)]
    let vrel_x = vx_post - burn.y;
    let vrel_y = vy_post + burn.x - (1.0 - MU);
    let vrel_z = vz_post;
    let v_scale = L_STAR / T_STAR;
    let v_mc_ix = (vrel_x * t_burn.cos() - vrel_y * t_burn.sin()) * v_scale;
    let v_mc_iy = (vrel_x * t_burn.sin() + vrel_y * t_burn.cos()) * v_scale;
    let v_mc_iz = vrel_z * v_scale;
    let v0_mc = mat_vec(&r0_wsb, [v_mc_ix, v_mc_iy, v_mc_iz]);

    let r_mc_mag = (r0_mc[0]*r0_mc[0] + r0_mc[1]*r0_mc[1] + r0_mc[2]*r0_mc[2]).sqrt();
    let t_orbit  = 2.0 * PI * (r_mc_mag.powi(3) / GM_MOON).sqrt();
    let v0_mc_mag = (v0_mc[0]*v0_mc[0]+v0_mc[1]*v0_mc[1]+v0_mc[2]*v0_mc[2]).sqrt();
    eprintln!("  Moon-relative r = {:.1} km  (expected {:.1} km)",
        r_mc_mag, burn.dist_moon * l_km);
    eprintln!("  v_circ expected = {:.4} km/s  |v0_mc| = {:.4} km/s",
        (GM_MOON/r_mc_mag).sqrt(), v0_mc_mag);

    // ── 2-body Moon-centred propagation ───────────────────────────────────────
    let t_prop_s = N_ORBITS * LUNAR_PERIOD_ND * T_STAR;
    eprintln!();
    eprintln!("  T_orb={:.0} s ({:.2} h)  propagating {N_ORBITS:.0} lunar periods ({:.1} d) ...",
        t_orbit, t_orbit/3600.0, t_prop_s / 86_400.0);

    let y0   = State6::from_column_slice(&[r0_mc[0], r0_mc[1], r0_mc[2], v0_mc[0], v0_mc[1], v0_mc[2]]);
    let wall = Instant::now();
    let mut stepper = Dopri5::from_param(
        TwoBodyMoon,
        0.0_f64, t_prop_s, SAVE_DT_S, y0, RTOL, ATOL,
        0.9_f64, 0.04_f64, 0.2_f64, 10.0_f64,
        SAVE_DT_S, H_INIT_LOI, N_MAX, 1000_u32,
        OutputType::Dense,
    );
    let _ = stepper.integrate();
    eprintln!("  Done: {} steps in {:.2} s",
        stepper.x_out().len(), wall.elapsed().as_secs_f64());

    // ── Health check ─────────────────────────────────────────────────────────
    let (r_min, r_max) = stepper.y_out().iter().fold((f64::MAX, 0.0_f64), |(mn, mx), y| {
        let r = (y[0]*y[0] + y[1]*y[1] + y[2]*y[2]).sqrt();
        (mn.min(r), mx.max(r))
    });
    eprintln!("  Post-burn alt: min={:.1} km  max={:.1} km", r_min-R_MOON, r_max-R_MOON);

    // ── Save capture.csv ───────────────────────────────────────────────────────
    let capture_path = format!("{OUT_DIR}/capture.csv");
    write_capture_csv(&capture_path, &traj[..=burn_idx]);
    eprintln!("  Saved capture  → {capture_path} ({} rows)", burn_idx+1);

    // ── Save loi_orbit.csv — S/C ECI = Moon-centred + DE440S Moon ────────────
    let orbit_path = format!("{OUT_DIR}/loi_orbit.csv");
    {
        let f = fs::File::create(&orbit_path).unwrap();
        let mut w = BufWriter::new(f);
        writeln!(w, "time_s,x_km,y_km,z_km,moon_x_km,moon_y_km,moon_z_km").unwrap();
        for (ti, yi) in stepper.x_out().iter().zip(stepper.y_out().iter()) {
            let abs_t = burn_t_s + ti;
            let [mx, my, mz] = moon_ephem.interp(abs_t);
            writeln!(w, "{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                abs_t, yi[0]+mx, yi[1]+my, yi[2]+mz, mx, my, mz).unwrap();
        }
    }
    eprintln!("  Saved loi_orbit → {orbit_path} ({} rows)", stepper.x_out().len());

    // ── Save epoch_info.txt (for Python plotter alignment) ────────────────────
    let epoch_path = format!("{OUT_DIR}/epoch_info.txt");
    {
        let mut f = fs::File::create(&epoch_path).unwrap();
        writeln!(f, "dep_t_s          : {dep_t_s:.3}").unwrap();
        writeln!(f, "burn_t_s         : {burn_t_s:.3}").unwrap();
        let dep_days = (dep_t_s - art_t_s0) / 86_400.0;
        writeln!(f, "dep_offset_days  : {dep_days:.6}").unwrap();
        writeln!(f, "R0_wsb_col0      : {:.10} {:.10} {:.10}", r0_wsb[0][0], r0_wsb[0][1], r0_wsb[0][2]).unwrap();
        writeln!(f, "R0_wsb_col1      : {:.10} {:.10} {:.10}", r0_wsb[1][0], r0_wsb[1][1], r0_wsb[1][2]).unwrap();
        writeln!(f, "R0_wsb_col2      : {:.10} {:.10} {:.10}", r0_wsb[2][0], r0_wsb[2][1], r0_wsb[2][2]).unwrap();
    }
    eprintln!("  Saved epoch_info → {epoch_path}");

    // ── Save circularize_info.txt ─────────────────────────────────────────────
    let info_path = format!("{OUT_DIR}/circularize_info.txt");
    {
        let mut f = fs::File::create(&info_path).unwrap();
        writeln!(f, "WSB Lunar Orbit Insertion (LOI) Summary").unwrap();
        writeln!(f, "Burn time (nd)   : {:.6}", burn.time).unwrap();
        writeln!(f, "Burn time (days) : {:.3}", burn_time_days).unwrap();
        writeln!(f, "Burn altitude km : {:.1}", burn_alt_km).unwrap();
        writeln!(f, "v_rel pre-burn   : {:.6} nd  ({:.5} km/s)", br.v_rel_pre, br.v_rel_pre*v_km_s).unwrap();
        writeln!(f, "v_circ           : {:.6} nd  ({:.5} km/s)", br.v_circ, br.v_circ*v_km_s).unwrap();
        writeln!(f, "LOI DeltaV nd    : {:.6}", br.dv_mag).unwrap();
        writeln!(f, "LOI DeltaV km/s  : {:.5}", br.dv_mag*v_km_s).unwrap();
        writeln!(f, "Post-burn alt min: {:.1} km", r_min-R_MOON).unwrap();
        writeln!(f, "Post-burn alt max: {:.1} km", r_max-R_MOON).unwrap();
    }
    eprintln!("  Saved summary    → {info_path}");
}
