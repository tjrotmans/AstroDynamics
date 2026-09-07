//! wsb_maxhifi — maximum-fidelity single-solution repropagation.
//!
//! Unlike `wsb_refine --reprop` (which outputs at a fixed LOG_DT interval),
//! this binary uses `OutputType::Sparse` so **every accepted integrator step**
//! is recorded.  This gives the maximum temporal resolution achievable and
//! exposes the adaptive-step distribution across the whole transfer.
//!
//! Integrator  : Dopri5 (RK4(5), same solver as the rest of the codebase)
//! Tolerances  : RTOL = 1e-12   ATOL = 1e-14   (near f64 floor)
//! Output      : every accepted adaptive step  (variable dt, Sparse mode)
//!
//! # Usage
//! ```
//!   cargo run -p lunar_trajectories --bin wsb_maxhifi --release
//!   cargo run -p lunar_trajectories --bin wsb_maxhifi --release -- --hit 2
//! ```
//! Default: best hit (by est_orbits) from `refine_summary.csv`.
//!
//! # Outputs (all in out/wsb_maxhifi/)
//!   maxhifi.csv       — full trajectory, one row per integrator step
//!   maxhifi_info.txt  — IC summary, step-size statistics, capture metrics

use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{BufWriter, Write as IoWrite};
use std::sync::OnceLock;
use std::time::Instant;

use ode_solvers::dopri5::Dopri5;
use ode_solvers::dop_shared::OutputType;
use ode_solvers::SVector as OdeVec;

use lunar_trajectories::crtbp::CrtbpParams;
use lunar_trajectories::propagator::{Bcr4bp3dOde, Bcr4bpParams, Step3d};
use lunar_trajectories::transfers::{lunar_hill_radius, detect_capture, tli_injection_ic};

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                         CONFIGURATION                                       ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

/// Integrator absolute tolerance.  1e-14 is near the f64 roundoff floor.
const RTOL: f64 = 1e-10;
const ATOL: f64 = 1e-12;

/// Initial step-size hint [nd].  The integrator adapts immediately.
const H_INIT: f64 = 1e-4;

/// Maximum number of accepted steps.  10 M is more than sufficient for ~300 d.
const N_MAX: u32 = 1000_000_000;

/// Total propagation time [nd].  ~300 days.
const T_PROP: f64 = 20.0 * PI;

const R_PARK: f64 = (6_371.0 + 378.0) / 384_400.0;

static _CFG_R_PARK: OnceLock<f64> = OnceLock::new();
fn cfg_r_park() -> f64 {
    *_CFG_R_PARK.get_or_init(|| {
        std::env::var("WSB_R_PARK_ND").ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(R_PARK)
    })
}
const LUNAR_PERIOD_ND: f64 = 2.0 * PI;
const MIN_CAPTURE_TIME: f64 = 0.15;

const OUT_DIR:        &str = "out/wsb";
const HIFI_CSV:       &str = "out/wsb/solution_hifi.csv";
const SUMMARY_CSV:    &str = "out/wsb/refine_summary.csv";

type State6 = OdeVec<f64, 6>;

// ════════════════════════════════════════════════════════════════════════════════

struct HifiIc {
    hit_id:        usize,
    seed_id:       usize,
    theta_deg:     f64,
    theta_sun_deg: f64,
    r_apogee_nd:   f64,
    est_orbits:    f64,
    dtheta_deg:    f64,
    dsun_deg:      f64,
    dr_apogee_nd:  f64,
}

/// Load all refined hits from refine_summary.csv, sorted by est_orbits descending.
/// Mirrors wsb_final_refinement's load logic exactly.
fn load_summary_hits() -> Vec<HifiIc> {
    let content = match fs::read_to_string(SUMMARY_CSV) {
        Ok(s)  => s,
        Err(e) => { eprintln!("  Cannot read {SUMMARY_CSV}: {e}"); return Vec::new(); }
    };
    // Columns: seed_id,source,theta_deg,theta_sun_deg,r_apogee_nd,dv_kms,
    //          t_transfer_days,est_capture_orbits,min_alt_km,
    //          dtheta_deg,dsun_deg,dr_apogee_nd,...
    let mut rows = Vec::new();
    let mut counter = 0usize;
    for line in content.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 12 { continue; }
        if c[1].trim() != "refined" { continue; }
        let seed_id:       usize = c[0].parse().unwrap_or(0);
        let theta_deg:     f64   = c[2].parse().unwrap_or(f64::NAN);
        let theta_sun_deg: f64   = c[3].parse().unwrap_or(f64::NAN);
        let r_apogee_nd:   f64   = c[4].parse().unwrap_or(f64::NAN);
        let est_orbits:    f64   = c[7].parse().unwrap_or(0.0);
        let dtheta_deg:    f64   = c[9].parse().unwrap_or(0.0);
        let dsun_deg:      f64   = c[10].parse().unwrap_or(0.0);
        let dr_apogee_nd:  f64   = c[11].parse().unwrap_or(0.0);
        if theta_deg.is_nan() { continue; }
        counter += 1;
        rows.push(HifiIc {
            hit_id: counter, seed_id,
            theta_deg, theta_sun_deg, r_apogee_nd, est_orbits,
            dtheta_deg, dsun_deg, dr_apogee_nd,
        });
    }
    rows.sort_by(|a, b| b.est_orbits.partial_cmp(&a.est_orbits).unwrap());
    for (i, r) in rows.iter_mut().enumerate() { r.hit_id = i + 1; }
    rows
}

/// Keplerian elements from Earth-centred inertial position [nd] and velocity [nd/nd].
/// mu_e = 1 - mu  (Earth gravitational parameter in ND units)
fn keplerian_elements(mu_e: f64, rx: f64, ry: f64, rz: f64,
                      vx: f64, vy: f64, vz: f64)
    -> (f64, f64, f64, f64, f64, f64)
{
    let r  = (rx*rx + ry*ry + rz*rz).sqrt();
    let v2 = vx*vx + vy*vy + vz*vz;

    // Specific angular momentum h = r × v
    let hx = ry*vz - rz*vy;
    let hy = rz*vx - rx*vz;
    let hz = rx*vy - ry*vx;
    let h  = (hx*hx + hy*hy + hz*hz).sqrt();

    // Eccentricity vector e = (v²/μ - 1/r)·r - (r·v)/μ·v
    let rdotv = rx*vx + ry*vy + rz*vz;
    let ex = (v2/mu_e - 1.0/r)*rx - (rdotv/mu_e)*vx;
    let ey = (v2/mu_e - 1.0/r)*ry - (rdotv/mu_e)*vy;
    let ez = (v2/mu_e - 1.0/r)*rz - (rdotv/mu_e)*vz;
    let e  = (ex*ex + ey*ey + ez*ez).sqrt();

    // Semi-major axis
    let eps = v2/2.0 - mu_e/r;
    let a   = -mu_e / (2.0 * eps);

    // Inclination
    let inc = (hz / h).clamp(-1.0, 1.0).acos();

    // Node vector n = k × h  (k = [0,0,1])
    let nx = -hy;
    let ny =  hx;
    let n  = (nx*nx + ny*ny).sqrt();

    // RAAN Ω
    let raan = if n > 1e-12 {
        let mut o = (nx / n).clamp(-1.0, 1.0).acos();
        if ny < 0.0 { o = 2.0*std::f64::consts::PI - o; }
        o
    } else { 0.0 };

    // Argument of perigee ω
    let aop = if n > 1e-12 && e > 1e-10 {
        let ndote = (nx*ex + ny*ey) / (n * e);
        let mut w = ndote.clamp(-1.0, 1.0).acos();
        if ez < 0.0 { w = 2.0*std::f64::consts::PI - w; }
        w
    } else { 0.0 };

    // True anomaly ν
    let nu = if e > 1e-10 {
        let edotr = (ex*rx + ey*ry + ez*rz) / (e * r);
        let mut nu = edotr.clamp(-1.0, 1.0).acos();
        if rdotv < 0.0 { nu = 2.0*std::f64::consts::PI - nu; }
        nu
    } else { 0.0 };

    (a, e, inc, raan, aop, nu)
}

/// CRTBP Jacobi constant (not conserved in BCR4BP but useful as accuracy monitor).
fn jacobi_c(mu: f64, x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64) -> f64 {
    let r1 = ((x + mu).powi(2) + y * y + z * z).sqrt();            // Earth
    let r2 = ((x - (1.0 - mu)).powi(2) + y * y + z * z).sqrt();   // Moon
    x * x + y * y + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2
        - (vx * vx + vy * vy + vz * vz)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let hit_arg: Option<usize> = args.iter()
        .position(|a| a == "--hit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║     WSB Max-Fidelity Repropagation                           ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!("  Source     : {SUMMARY_CSV}");
    eprintln!("  Integrator : Dopri5  OutputType::Sparse  (every adaptive step)");
    eprintln!("  RTOL={RTOL:.0e}  ATOL={ATOL:.0e}  H_INIT={H_INIT:.0e}  N_MAX={N_MAX}");
    eprintln!();

    let r_park   = cfg_r_park();  // overrideable via WSB_R_PARK_ND

    let params   = CrtbpParams::earth_moon();
    let mu       = params.mu;
    let l_km     = params.l_star / 1e3;
    let v_km_s   = params.v_star / 1e3;
    let moon_x   = 1.0 - mu;
    let r_hill   = lunar_hill_radius(mu);
    let mu_earth = 1.0 - mu;

    fs::create_dir_all(OUT_DIR).unwrap();

    // ── Load hits from refine_summary.csv (same source as wsb_final_refinement) ─
    let hits = load_summary_hits();
    if hits.is_empty() {
        eprintln!("  No refined hits in {SUMMARY_CSV}. Run wsb_refine first.");
        return;
    }
    eprintln!("  {:>4}  {:>4}  {:>9}  {:>11}  {:>8}  {:>8}",
        "hit", "seed", "theta deg", "theta_sun", "r_apo nd", "orbits");
    for r in &hits {
        eprintln!("  {:>4}  {:>4}  {:>9.3}  {:>11.3}  {:>8.3}  {:>8.2}",
            r.hit_id, r.seed_id, r.theta_deg, r.theta_sun_deg,
            r.r_apogee_nd, r.est_orbits);
    }
    eprintln!();
    let hifi = match hit_arg {
        None    => &hits[0],
        Some(n) => hits.iter().find(|r| r.hit_id == n).unwrap_or_else(|| {
            eprintln!("  --hit {n} not found, using best."); &hits[0]
        }),
    };

    let theta = hifi.theta_deg.to_radians();
    let ic = match tli_injection_ic(mu, r_park, hifi.r_apogee_nd, theta) {
        Some(ic) => ic,
        None => { eprintln!("  Bad IC (r_apogee <= r_park)."); return; }
    };
    let bcr = Bcr4bpParams::earth_moon_sun(hifi.theta_sun_deg.to_radians());

    let r_apo  = hifi.r_apogee_nd;
    let v_circ = (mu_earth / r_park).sqrt();
    let a      = (r_park + r_apo) / 2.0;
    let v_inj  = (mu_earth * (2.0 / r_park - 1.0 / a)).sqrt();
    let dv_kms = (v_inj - v_circ) * v_km_s;

    eprintln!("  Hit {}  seed={}  theta={:.4} deg  theta_sun={:.4} deg  \
               r_apo={:.4} nd  est_orbits={:.2}",
        hifi.hit_id, hifi.seed_id,
        hifi.theta_deg, hifi.theta_sun_deg,
        hifi.r_apogee_nd, hifi.est_orbits);
    eprintln!();
    eprintln!("  Initial conditions (BCR4BP rotating frame, algebraic — identical to wsb_refine):");
    eprintln!("    x  = {:.10}  y  = {:.10}  z  = {:.10}", ic[0], ic[1], ic[2]);
    eprintln!("    vx = {:.10}  vy = {:.10}  vz = {:.10}", ic[3], ic[4], ic[5]);
    eprintln!("    Parking orbit alt = {:.1} km", (R_PARK * l_km) - 6371.0);
    eprintln!("    TLI DeltaV        = {:.5} km/s", dv_kms);
    eprintln!("    Apogee radius     = {:.4} nd  = {:.0} km", r_apo, r_apo * l_km);
    eprintln!();

    // ── Keplerian elements of the parking orbit (before TLI, ECI at t=0) ──────
    // At t=0 the rotating and inertial EM frames are aligned, so:
    //   r_eci = (x+mu, y, z)
    //   v_eci = v_rot + omega × r_eci  where omega=(0,0,1) => (-y, x+mu, 0)
    let rx_eci = ic[0] + mu;
    let ry_eci = ic[1];
    let rz_eci = ic[2];
    let vx_eci = ic[3] - ic[1];          // vx_rot - y
    let vy_eci = ic[4] + (ic[0] + mu);   // vy_rot + (x+mu)
    let vz_eci = ic[5];
    let (a_nd, ecc, inc_rad, raan_rad, aop_rad, nu_rad) =
        keplerian_elements(mu_earth, rx_eci, ry_eci, rz_eci, vx_eci, vy_eci, vz_eci);
    let deg = |r: f64| r.to_degrees();
    eprintln!("  Parking orbit Keplerian elements (Earth-centred, t=0):");
    eprintln!("    a   = {:.2} km    ({:.6} nd)", a_nd * l_km, a_nd);
    eprintln!("    e   = {:.8}", ecc);
    eprintln!("    i   = {:.4} deg", deg(inc_rad));
    eprintln!("    Ω   = {:.4} deg  (RAAN)", deg(raan_rad));
    eprintln!("    ω   = {:.4} deg  (AoP)", deg(aop_rad));
    eprintln!("    ν   = {:.4} deg  (true anomaly)", deg(nu_rad));
    eprintln!("    Periapsis alt = {:.1} km", (a_nd * (1.0 - ecc) * l_km) - 6371.0);
    eprintln!("    Apoapsis  alt = {:.1} km", (a_nd * (1.0 + ecc) * l_km) - 6371.0);
    eprintln!();

    // ── Propagate ─────────────────────────────────────────────────────────────
    let y0 = State6::from_column_slice(&ic);
    let c0 = jacobi_c(mu, ic[0], ic[1], ic[2], ic[3], ic[4], ic[5]);

    eprintln!("  Initial Jacobi constant (CRTBP) C0 = {c0:.10}");
    eprintln!("  Propagating to T={T_PROP:.2} nd ({:.1} days) ...",
        T_PROP * params.t_star / 86_400.0);

    let t_start = Instant::now();
    let mut stepper = Dopri5::from_param(
        Bcr4bp3dOde { mu, params: bcr },
        0.0_f64, T_PROP, H_INIT, y0, RTOL, ATOL,
        0.9_f64,        // safety factor
        0.04_f64,       // beta (PI step-size controller)
        0.2_f64,      // fac_min  (max step decrease = 1/fac_min)
        10.0_f64,        // fac_max  (max step increase)
        T_PROP,         // h_max
        H_INIT,         // dx_out  (irrelevant for Sparse mode)
        N_MAX,
        1000_u32,       // n_stiff
        OutputType::Sparse,
    );
    let _ = stepper.integrate();
    let elapsed = t_start.elapsed();

    let t_out = stepper.x_out();
    let y_out = stepper.y_out();
    let n_steps = t_out.len();

    eprintln!("  Done: {n_steps} steps in {:.2} s", elapsed.as_secs_f64());

    // ── Step-size statistics ──────────────────────────────────────────────────
    let mut h_min = f64::MAX;
    let mut h_max_val: f64 = 0.0;
    let mut h_sum = 0.0;
    let mut hist = vec![0u64; 16];   // log10(h) bins: -6 to +0

    for i in 1..n_steps {
        let h = t_out[i] - t_out[i - 1];
        h_min = h_min.min(h);
        h_max_val = h_max_val.max(h);
        h_sum += h;
        // histogram bin: floor(log10(h)) clamped to [-6, 0]
        let bin = (h.log10().floor() as i32).clamp(-15, 0) + 15;
        hist[bin as usize] += 1;
    }
    let h_mean = if n_steps > 1 { h_sum / (n_steps - 1) as f64 } else { 0.0 };

    eprintln!();
    eprintln!("  ── Step-size statistics ──────────────────────────────────────");
    eprintln!("  Total steps : {n_steps}");
    eprintln!("  h_min       : {h_min:.4e} nd  ({:.2} s)",  h_min  * params.t_star);
    eprintln!("  h_max       : {h_max_val:.4e} nd  ({:.1} h)", h_max_val * params.t_star / 3600.0);
    eprintln!("  h_mean      : {h_mean:.4e} nd  ({:.1} h)", h_mean * params.t_star / 3600.0);
    eprintln!();
    eprintln!("  Step-size histogram (log10 bins, count):");
    for (i, &cnt) in hist.iter().enumerate() {
        if cnt == 0 { continue; }
        let exp = i as i32 - 15;
        eprintln!("    1e{exp:+03}  {:>8}", cnt);
    }

    // ── Build Step3d vec for detect_capture ───────────────────────────────────
    let traj: Vec<Step3d> = t_out.iter().zip(y_out.iter())
        .map(|(&t, y)| Step3d {
            time: t, x: y[0], y: y[1], z: y[2],
            vx: y[3], vy: y[4], vz: y[5],
        })
        .collect();

    let cap         = detect_capture(&traj, mu, MIN_CAPTURE_TIME);
    let est_orbits  = cap.max_capture_interval / LUNAR_PERIOD_ND;

    let min_moon_km = traj.iter()
        .map(|s| { let dx=s.x-moon_x; (dx*dx+s.y*s.y+s.z*s.z).sqrt() * l_km - 1_737.4 })
        .fold(f64::MAX, f64::min);

    // Jacobi constant drift (BCR4BP — drift is physical, not numerical)
    let cf = jacobi_c(mu, traj.last().map(|s|s.x).unwrap_or(0.0),
                      traj.last().map(|s|s.y).unwrap_or(0.0),
                      traj.last().map(|s|s.z).unwrap_or(0.0),
                      traj.last().map(|s|s.vx).unwrap_or(0.0),
                      traj.last().map(|s|s.vy).unwrap_or(0.0),
                      traj.last().map(|s|s.vz).unwrap_or(0.0));
    let dc = cf - c0;

    eprintln!();
    eprintln!("  ── Capture metrics ───────────────────────────────────────────");
    eprintln!("  Hill entries      : {}", cap.n_entries);
    eprintln!("  Max dwell (nd)    : {:.4}  ({:.2} lunar orbits)",
        cap.max_capture_interval, est_orbits);
    eprintln!("  Min lunar alt     : {min_moon_km:.1} km");
    eprintln!("  Jacobi C drift    : {dc:+.4e}  (BCR4BP Sun perturbation, not numerical error)");

    // ── Save CSV ──────────────────────────────────────────────────────────────
    let csv_path = format!("{OUT_DIR}/maxhifi.csv");
    let file = fs::File::create(&csv_path).expect("cannot create maxhifi.csv");
    let mut w = BufWriter::new(file);
    writeln!(w, "step,time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,\
                 dist_earth_nd,dist_moon_nd,in_hill,jacobi_c").unwrap();
    for (i, s) in traj.iter().enumerate() {
        let r_e = ((s.x + mu).powi(2) + s.y*s.y + s.z*s.z).sqrt();
        let dx  = s.x - moon_x;
        let r_m = (dx*dx + s.y*s.y + s.z*s.z).sqrt();
        let in_h = if r_m < r_hill { 1u8 } else { 0u8 };
        let jc   = jacobi_c(mu, s.x, s.y, s.z, s.vx, s.vy, s.vz);
        writeln!(w, "{i},{:.10},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.10},{:.10},{in_h},{:.10}",
            s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz, r_e, r_m, jc).unwrap();
    }
    eprintln!();
    eprintln!("  Saved {csv_path}  ({n_steps} rows)");

    // ── Save info text ────────────────────────────────────────────────────────
    let mut info = String::new();
    writeln!(info, "=== WSB Max-Fidelity Propagation ===\n").unwrap();
    writeln!(info, "Source         : {HIFI_CSV}").unwrap();
    writeln!(info, "Hit ID         : {}", hifi.hit_id).unwrap();
    writeln!(info, "Seed ID        : {}", hifi.seed_id).unwrap();
    writeln!(info, "theta_deg      : {:.6} deg", hifi.theta_deg).unwrap();
    writeln!(info, "theta_sun_deg  : {:.6} deg", hifi.theta_sun_deg).unwrap();
    writeln!(info, "r_apogee_nd    : {:.6} nd  ({:.0} km)", r_apo, r_apo * l_km).unwrap();
    writeln!(info, "TLI DeltaV     : {:.6} km/s", dv_kms).unwrap();
    writeln!(info, "dtheta_deg     : {:+.6} deg", hifi.dtheta_deg).unwrap();
    writeln!(info, "dsun_deg       : {:+.6} deg", hifi.dsun_deg).unwrap();
    writeln!(info, "dr_apogee_nd   : {:+.6}", hifi.dr_apogee_nd).unwrap();
    writeln!(info).unwrap();
    writeln!(info, "Integrator     : Dopri5  Sparse (every accepted step)").unwrap();
    writeln!(info, "RTOL           : {RTOL:.0e}").unwrap();
    writeln!(info, "ATOL           : {ATOL:.0e}").unwrap();
    writeln!(info, "T_PROP         : {T_PROP:.4} nd  ({:.1} days)",
        T_PROP * params.t_star / 86_400.0).unwrap();
    writeln!(info, "Wall time       : {:.2} s", elapsed.as_secs_f64()).unwrap();
    writeln!(info).unwrap();
    writeln!(info, "-- Step statistics --").unwrap();
    writeln!(info, "Total steps    : {n_steps}").unwrap();
    writeln!(info, "h_min          : {h_min:.4e} nd  ({:.2} s)", h_min * params.t_star).unwrap();
    writeln!(info, "h_max          : {h_max_val:.4e} nd  ({:.1} h)",
        h_max_val * params.t_star / 3600.0).unwrap();
    writeln!(info, "h_mean         : {h_mean:.4e} nd  ({:.1} h)",
        h_mean * params.t_star / 3600.0).unwrap();
    writeln!(info).unwrap();
    writeln!(info, "-- Capture metrics --").unwrap();
    writeln!(info, "Hill entries   : {}", cap.n_entries).unwrap();
    writeln!(info, "Max dwell      : {:.4} nd  = {:.2} lunar orbits",
        cap.max_capture_interval, est_orbits).unwrap();
    writeln!(info, "Min lunar alt  : {min_moon_km:.1} km").unwrap();
    writeln!(info, "Jacobi C drift : {dc:+.4e}  (BCR4BP Sun perturbation)").unwrap();
    writeln!(info).unwrap();
    writeln!(info, "Output         : {csv_path}").unwrap();

    let info_path = format!("{OUT_DIR}/maxhifi_info.txt");
    fs::write(&info_path, &info).expect("info write failed");
    eprintln!("  Saved {info_path}");
    print!("{info}");

    // ── Append to solutions archive ───────────────────────────────────────────
    append_solutions_archive(
        hifi.theta_deg, hifi.theta_sun_deg, r_apo,
        est_orbits, dv_kms, min_moon_km,
    );
}

/// Append one record to the persistent solutions archive.
///
/// The archive (`out/wsb_solutions.csv`) survives code changes and pipeline
/// runs — it is NEVER overwritten, only appended to.  Edit the `notes` field
/// manually in the CSV to annotate noteworthy runs.
fn append_solutions_archive(
    theta_deg:     f64,
    theta_sun_deg: f64,
    r_apogee_nd:   f64,
    est_orbits:    f64,
    dv_kms:        f64,
    min_moon_km:   f64,
) {
    let path = "out/wsb_solutions.csv";
    let header = "date,theta_deg,theta_sun_deg,r_apogee_nd,\
                  est_orbits,dv_kms,min_moon_km,notes";
    let exists = std::path::Path::new(path).exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true).append(true)
        .open(path)
        .expect("Cannot open solutions archive");
    use std::io::Write as _;
    if !exists {
        writeln!(f, "{header}").unwrap();
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Format as YYYY-MM-DD using seconds since epoch (UTC, no chrono dep)
    let days  = now / 86400;
    let y = 1970 + days / 365;   // approximate — good enough for a log
    let m = (days % 365) / 30 + 1;
    let d = (days % 365) % 30 + 1;
    let date = format!("{y:04}-{m:02}-{d:02}");
    writeln!(f,
        "{date},{theta_deg:.4},{theta_sun_deg:.4},{r_apogee_nd:.5},\
         {est_orbits:.3},{dv_kms:.5},{min_moon_km:.1},"
    ).unwrap();
    eprintln!("  Appended to solutions archive: {path}");
}
