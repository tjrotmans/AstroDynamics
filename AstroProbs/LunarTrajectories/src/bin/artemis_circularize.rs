//! artemis_circularize — Lunar Orbit Insertion (LOI) burn for the Artemis II trajectory.
//!
//! Reads `../Artemis/out/artemis2_trajectory.csv` (ECI, SI units), locates the
//! first Moon periapsis, computes a circularization ΔV, and propagates the
//! resulting lunar orbit with Moon-centred 2-body dynamics.
//!
//! Moon positions in the output CSV come from DE440S ephemeris
//! (`../Artemis/out/moon_ephem.csv`), replacing the old 2-body Moon propagation.
//!
//! # Outputs (saved to out/artemis_circularize/)
//!   loi_orbit.csv — time_s, x_km, y_km, z_km, moon_x_km, moon_y_km, moon_z_km
//!   info.txt      — LOI summary
//!
//! # Usage
//! ```
//!   cargo run -p lunar_trajectories --bin artemis_circularize --release
//! ```

use std::f64::consts::PI;
use std::fs;
use std::io::{BufRead, BufWriter, Write as IoWrite};
use std::time::Instant;

use ode_solvers::dopri5::Dopri5;
use ode_solvers::dop_shared::OutputType;
use ode_solvers::{SVector as OdeVec, System};

// ── Paths ─────────────────────────────────────────────────────────────────────
const ART_CSV:        &str = "../Artemis/out/artemis2_trajectory.csv";
const MOON_EPHEM_CSV: &str = "../Artemis/out/moon_ephem.csv";
const OUT_DIR:        &str = "out/artemis_circularize";

// ── Physical constants ────────────────────────────────────────────────────────
const GM_MOON:  f64 = 4_902.8;     // km³ s⁻²
const R_MOON:   f64 = 1_737.4;     // km

// ── Integrator settings ───────────────────────────────────────────────────────
const RTOL:   f64 = 1e-10;
const ATOL:   f64 = 1e-12;
const H_INIT: f64 = 1.0;           // seconds
const N_MAX:  u32 = 1_000_000_000;

const PROP_TIME_S: f64 = 8.0 * 2.0 * PI * 375_700.0;  // ≈ 218.6 days
const SAVE_DT_S:   f64 = 3_600.0;

type State6 = OdeVec<f64, 6>;

// ════════════════════════════════════════════════════════════════════════════
// DE440S ephemeris table
// ════════════════════════════════════════════════════════════════════════════

struct EphemTrack { time_s: Vec<f64>, x_km: Vec<f64>, y_km: Vec<f64>, z_km: Vec<f64> }

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
        [self.x_km[i0] + f*(self.x_km[i]-self.x_km[i0]),
         self.y_km[i0] + f*(self.y_km[i]-self.y_km[i0]),
         self.z_km[i0] + f*(self.z_km[i]-self.z_km[i0])]
    }
}

fn load_moon_ephem(path: &str) -> EphemTrack {
    let file = fs::File::open(path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let mut track = EphemTrack { time_s: vec![], x_km: vec![], y_km: vec![], z_km: vec![] };
    for (i, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.unwrap();
        if i == 0 { continue; }
        let c: Vec<f64> = line.split(',').map(|s| s.trim().parse::<f64>().unwrap_or(f64::NAN)).collect();
        if c.len() < 4 { continue; }
        track.time_s.push(c[0]);
        track.x_km.push(c[1]*1e-3); track.y_km.push(c[2]*1e-3); track.z_km.push(c[3]*1e-3);
    }
    track
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
// Artemis CSV loading
// ════════════════════════════════════════════════════════════════════════════

struct ArtRow {
    time_s: f64, x_km: f64, y_km: f64, z_km: f64,
    vx_kms: f64, vy_kms: f64, vz_kms: f64,
    moon_x_km: f64, moon_y_km: f64, moon_z_km: f64,
}

fn load_artemis(path: &str) -> Vec<ArtRow> {
    let file = fs::File::open(path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let mut rows = Vec::new();
    let mut idx: Option<[usize; 10]> = None;
    for (i, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.unwrap();
        let cols: Vec<&str> = line.split(',').collect();
        if i == 0 {
            let find = |name: &str| cols.iter().position(|c| c.trim() == name)
                .unwrap_or_else(|| panic!("column '{name}' not found"));
            idx = Some([find("time_s"),
                find("x_m"), find("y_m"), find("z_m"),
                find("vx_ms"), find("vy_ms"), find("vz_ms"),
                find("moon_x_m"), find("moon_y_m"), find("moon_z_m")]);
            continue;
        }
        if cols.len() < 10 { continue; }
        let c = idx.unwrap();
        let v = |k: usize| cols[c[k]].trim().parse::<f64>().unwrap_or(0.0);
        rows.push(ArtRow {
            time_s: v(0),
            x_km: v(1)*1e-3, y_km: v(2)*1e-3, z_km: v(3)*1e-3,
            vx_kms: v(4)*1e-3, vy_kms: v(5)*1e-3, vz_kms: v(6)*1e-3,
            moon_x_km: v(7)*1e-3, moon_y_km: v(8)*1e-3, moon_z_km: v(9)*1e-3,
        });
    }
    rows
}

fn moon_dist(r: &ArtRow) -> f64 {
    let (dx, dy, dz) = (r.x_km-r.moon_x_km, r.y_km-r.moon_y_km, r.z_km-r.moon_z_km);
    (dx*dx + dy*dy + dz*dz).sqrt()
}

// ════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════

fn main() {
    fs::create_dir_all(OUT_DIR).unwrap();

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║   Artemis II Lunar Orbit Insertion (LOI) Burn                ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    eprintln!("  Loading DE440S Moon ephemeris ...");
    let moon_ephem = load_moon_ephem(MOON_EPHEM_CSV);
    eprintln!("  {} pts,  t=[{:.0}, {:.0}] s", moon_ephem.time_s.len(),
        moon_ephem.time_s.first().copied().unwrap_or(0.0),
        moon_ephem.time_s.last().copied().unwrap_or(0.0));

    eprintln!("  Loading {ART_CSV} ...");
    let traj = load_artemis(ART_CSV);
    eprintln!("  Loaded {} points", traj.len());

    // ── Find Moon periapsis ────────────────────────────────────────────────────
    let dists: Vec<f64> = traj.iter().map(moon_dist).collect();
    let burn_idx = dists.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i).expect("empty trajectory");

    let burn     = &traj[burn_idx];
    let burn_alt = dists[burn_idx] - R_MOON;
    let t0       = traj[0].time_s;
    eprintln!();
    eprintln!("  LOI burn: T+{:.3} d  alt={burn_alt:.1} km", (burn.time_s - t0)/86400.0);

    // ── Moon velocity at burn (central diff on CSV) ────────────────────────────
    let prev = &traj[burn_idx - 1];
    let next = &traj[burn_idx + 1];
    let dt2  = next.time_s - prev.time_s;
    let vm = [(next.moon_x_km-prev.moon_x_km)/dt2,
              (next.moon_y_km-prev.moon_y_km)/dt2,
              (next.moon_z_km-prev.moon_z_km)/dt2];

    // ── Circularization ΔV ────────────────────────────────────────────────────
    let rx = burn.x_km - burn.moon_x_km;
    let ry = burn.y_km - burn.moon_y_km;
    let rz = burn.z_km - burn.moon_z_km;
    let r_mag = (rx*rx + ry*ry + rz*rz).sqrt();
    let vrx = burn.vx_kms - vm[0];
    let vry = burn.vy_kms - vm[1];
    let vrz = burn.vz_kms - vm[2];
    let v_rel_mag = (vrx*vrx + vry*vry + vrz*vrz).sqrt();
    let v_circ = (GM_MOON / r_mag).sqrt();

    let hx = ry*vrz - rz*vry; let hy = rz*vrx - rx*vrz; let hz = rx*vry - ry*vrx;
    let h_mag = (hx*hx + hy*hy + hz*hz).sqrt();
    let (hxn, hyn, hzn) = (hx/h_mag, hy/h_mag, hz/h_mag);
    let (rxn, ryn, rzn) = (rx/r_mag, ry/r_mag, rz/r_mag);
    let tx = hyn*rzn - hzn*ryn;
    let ty = hzn*rxn - hxn*rzn;
    let tz = hxn*ryn - hyn*rxn;
    let (dvx, dvy, dvz) = (v_circ*tx - vrx, v_circ*ty - vry, v_circ*tz - vrz);
    let dv_mag = (dvx*dvx + dvy*dvy + dvz*dvz).sqrt();

    eprintln!("  v_rel={v_rel_mag:.4} km/s  v_circ={v_circ:.4} km/s  ΔV={dv_mag:.4} km/s");

    // ── 2-body Moon-centred propagation ────────────────────────────────────────
    let v0_mc = [burn.vx_kms + dvx - vm[0],
                 burn.vy_kms + dvy - vm[1],
                 burn.vz_kms + dvz - vm[2]];
    let t_orbit = 2.0 * PI * (r_mag.powi(3) / GM_MOON).sqrt();
    let n_orbits = PROP_TIME_S / t_orbit;

    eprintln!("  T_orb={:.0} s ({:.2} h)  propagating {n_orbits:.1} orbits ({:.1} d) ...",
        t_orbit, t_orbit/3600.0, PROP_TIME_S/86400.0);

    let y0   = State6::from_column_slice(&[rx, ry, rz, v0_mc[0], v0_mc[1], v0_mc[2]]);
    let wall = Instant::now();
    let mut stepper = Dopri5::from_param(
        TwoBodyMoon,
        0.0_f64, PROP_TIME_S, SAVE_DT_S, y0, RTOL, ATOL,
        0.9_f64, 0.04_f64, 0.2_f64, 10.0_f64,
        SAVE_DT_S, H_INIT, N_MAX, 1000_u32,
        OutputType::Dense,
    );
    let _ = stepper.integrate();
    eprintln!("  Done: {} steps in {:.2} s", stepper.x_out().len(), wall.elapsed().as_secs_f64());

    // ── Orbital health check ───────────────────────────────────────────────────
    let (r_min, r_max) = stepper.y_out().iter().fold((f64::MAX, 0.0_f64), |(mn,mx), y| {
        let r = (y[0]*y[0]+y[1]*y[1]+y[2]*y[2]).sqrt();
        (mn.min(r), mx.max(r))
    });
    eprintln!("  Post-burn alt: min={:.1} km  max={:.1} km", r_min-R_MOON, r_max-R_MOON);

    // ── Save loi_orbit.csv — S/C ECI = Moon-centred + DE440S Moon ─────────────
    let loi_path = format!("{OUT_DIR}/loi_orbit.csv");
    {
        let f = fs::File::create(&loi_path).unwrap();
        let mut w = BufWriter::new(f);
        writeln!(w, "time_s,x_km,y_km,z_km,moon_x_km,moon_y_km,moon_z_km").unwrap();
        for (ti, yi) in stepper.x_out().iter().zip(stepper.y_out().iter()) {
            let abs_t = burn.time_s + ti;
            let [mx, my, mz] = moon_ephem.interp(abs_t);
            writeln!(w, "{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                abs_t, yi[0]+mx, yi[1]+my, yi[2]+mz, mx, my, mz).unwrap();
        }
    }
    eprintln!("  Saved → {loi_path}  ({} rows)", stepper.x_out().len());

    // ── Save info.txt ──────────────────────────────────────────────────────────
    let info_path = format!("{OUT_DIR}/info.txt");
    {
        let mut f = fs::File::create(&info_path).unwrap();
        writeln!(f, "Artemis II Lunar Orbit Insertion (LOI) Summary").unwrap();
        writeln!(f, "Burn time s                   : {:.3}", burn.time_s).unwrap();
        writeln!(f, "Burn time days from departure : {:.4}", (burn.time_s - t0)/86400.0).unwrap();
        writeln!(f, "Burn altitude km              : {:.1}", burn_alt).unwrap();
        writeln!(f, "v_rel pre-burn km/s           : {:.5}", v_rel_mag).unwrap();
        writeln!(f, "v_circ km/s                   : {:.5}", v_circ).unwrap();
        writeln!(f, "LOI DeltaV km/s               : {:.5}", dv_mag).unwrap();
        writeln!(f, "Circular orbit radius km      : {:.1}", r_mag).unwrap();
        writeln!(f, "Orbital period s              : {:.1}", t_orbit).unwrap();
        writeln!(f, "Post-burn altitude km         : {:.1}", r_mag - R_MOON).unwrap();
    }
    eprintln!("  Saved → {info_path}");
    eprintln!("Done.");
}
