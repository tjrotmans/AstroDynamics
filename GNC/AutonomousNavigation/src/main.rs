//! Autonomous navigation simulation -- proximity phase (Bennu approach).
//!
//! Scenario:
//!   Spacecraft begins at the cruise arrival point relative to Bennu.
//!   The 6DOF truth model propagates the real state (pos, vel, attitude, w).
//!   OpNav + star tracker measurements arrive every 60 s.
//!   An EKF estimates [r, v, C_SRP] and its covariance.
//!
//! Run:  `cargo run -p autonomous_navigation --bin autonav`
//! Plot: `python plot/plot_nav.py`

use autonomous_navigation::config::{SIM_DURATION_S, DT_TRUTH_S, DT_MEAS_S, SC_MASS};
use autonomous_navigation::dynamics::{TruthState, Propagator};
use autonomous_navigation::navigation::ekf::{EkfState, predict, update_bearing, update_angular_size};
use autonomous_navigation::proximity_init::{ProximityHandoff, HANDOFF_SIGMA_CR};
use autonomous_navigation::sensors::{opnav, star_tracker};

fn main() {
    std::fs::create_dir_all("out").unwrap();

    println!("=== Autonomous Navigation -- Bennu Proximity Phase ===");
    println!("  Sim duration : {:.1} h", SIM_DURATION_S / 3600.0);
    println!("  Truth dt     : {:.1} s", DT_TRUTH_S);
    println!("  Meas interval: {:.1} s", DT_MEAS_S);

    // -- Initialise -----------------------------------------------------------
    let mut rng = rand::thread_rng();

    // Try to load initial conditions from cruise phase outputs.
    // Falls back to config defaults if cruise has not been run yet.
    let (mut truth, mut ekf) = match ProximityHandoff::load(&mut rng) {
        Ok(h) => {
            println!("  Init mode      : cruise handoff");
            println!("  Arrival epoch  : {:.1} d from J2000", h.t_arr / 86_400.0);
            println!("  Bennu offset   : {:.2} km (ephem uncertainty)", h.dr_bennu.norm() / 1e3);
            println!("  Handoff range  : {:.2} km from Bennu", h.handoff_range_m / 1e3);
            println!("  Fast coast     : {:.1} h to {:.1} km start",
                     h.coast_duration_s / 3600.0, h.r_truth.norm() / 1e3);
            println!("  EKF init s_r   : {:.2} km", h.sigma_r0_m / 1e3);
            let truth = TruthState::from_handoff(h.r_truth, h.v_truth, h.t_arr);
            let ekf   = EkfState::with_initial_state(
                h.r_est, h.v_est,
                h.sigma_r0_m, h.sigma_v0_mps, HANDOFF_SIGMA_CR,
                h.t_arr,
            );
            (truth, ekf)
        }
        Err(msg) => {
            println!("  Init mode      : standalone config defaults");
            println!("  [WARN] Cruise handoff unavailable: {}", msg);
            let truth = TruthState::from_config();
            let ekf   = EkfState::from_config(truth.r.clone(), truth.v.clone());
            (truth, ekf)
        }
    };

    let prop = Propagator::new(DT_TRUTH_S);

    let n_steps    = (SIM_DURATION_S / DT_TRUTH_S).round() as usize;
    let meas_every = (DT_MEAS_S / DT_TRUTH_S).round() as usize;

    // -- Logging buffers ------------------------------------------------------
    let mut truth_log:    Vec<LogRow>          = Vec::with_capacity(n_steps / meas_every + 1);
    let mut ekf_log:      Vec<LogRow>          = Vec::with_capacity(n_steps / meas_every + 1);
    let mut residual_log: Vec<MeasResidualRow> = Vec::new();
    let mut rcs_log:      Vec<RcsRow>          = Vec::with_capacity(n_steps);
    let mut n_missed: usize = 0;
    let mut dv_total: f64 = 0.0;

    // -- Main loop ------------------------------------------------------------
    for step in 0..n_steps {
        // 1. Advance truth model one DT_TRUTH_S step; capture RCS output for logging
        let (new_truth, rcs_step) = prop.step(&truth);
        truth = new_truth;

        // Accumulate DV: total unsigned thrust (all firing thrusters, pre-cancellation)
        // divided by mass.  Net translational force from pure torque couples is ~0,
        // but each individual thruster still expends propellant.
        let dv_inc = rcs_step.total_thrust_n / SC_MASS * DT_TRUTH_S;
        dv_total  += dv_inc;

        rcs_log.push(RcsRow {
            t:           truth.t,
            tau_x:       rcs_step.torque_body[0],
            tau_y:       rcs_step.torque_body[1],
            tau_z:       rcs_step.torque_body[2],
            omega_x:     truth.omega[0],
            omega_y:     truth.omega[1],
            omega_z:     truth.omega[2],
            dv_inc_ms:   dv_inc,
            dv_total_ms: dv_total,
        });

        // 2. EKF predict (runs at truth rate; predict() integrates by dt internally)
        ekf = predict(&ekf, DT_TRUTH_S);

        // 3. Measurement update every meas_every steps
        if step % meas_every == 0 {
            // Star tracker: noisy attitude measurement
            let q_st = star_tracker::measure(&truth.q, &mut rng);

            // OpNav: returns None if Bennu is outside camera FOV or Sun in exclusion zone
            let opnav_opt = opnav::measure(&truth.r, &truth.q, &q_st, truth.t, &mut rng);
            let meas_available = opnav_opt.is_some();
            if !meas_available { n_missed += 1; }

            // Log truth and estimate at this epoch (regardless of measurement)
            truth_log.push(LogRow::from_truth(&truth, dv_total));
            ekf_log.push(LogRow::from_ekf(&ekf));

            if let Some(opnav_meas) = opnav_opt {
                // Record pre-update residual (measured minus predicted)
                let r_ekf     = ekf.r();
                let range_ekf = r_ekf.norm();
                let los_pred  = -r_ekf / range_ekf;
                // Bearing residual: angle between predicted and measured LOS
                let los_residual  = (opnav_meas.los_inertial - los_pred).norm();
                // Angular-size residual: measured minus predicted size (radians)
                let size_residual = (opnav_meas.angular_size
                    - autonomous_navigation::config::R_BENNU / range_ekf).abs();

                residual_log.push(MeasResidualRow {
                    t: truth.t, los_residual, size_residual,
                    opnav_available: true,
                    phase_angle_rad: opnav_meas.phase_angle_rad,
                });

                // EKF updates
                ekf = update_bearing(&ekf, &opnav_meas.los_inertial, opnav_meas.sigma_bearing);
                ekf = update_angular_size(&ekf, opnav_meas.angular_size);
            } else {
                residual_log.push(MeasResidualRow {
                    t: truth.t,
                    los_residual:    f64::NAN,
                    size_residual:   f64::NAN,
                    opnav_available: false,
                    phase_angle_rad: f64::NAN,
                });
            }
        }
    }

    // -- Report ---------------------------------------------------------------
    let final_err = {
        let dr = truth.r - ekf.r();
        dr.norm()
    };
    let final_verr = {
        let dv = truth.v - ekf.v();
        dv.norm()
    };
    let n_meas_epochs = n_steps / meas_every;
    println!("\n-- Final navigation error --");
    println!("  Position error: {:.2} m", final_err);
    println!("  Velocity error: {:.4} m/s", final_verr);
    println!("  C_SRP estimate: {:.4}  (truth: {:.4})", ekf.c_r(), truth.c_r);
    println!("  Measurements obtained: {}/{} ({:.1}% FOV coverage)",
        n_meas_epochs - n_missed, n_meas_epochs,
        100.0 * (n_meas_epochs - n_missed) as f64 / n_meas_epochs as f64);
    println!("  Total RCS DV spent:  {:.4} m/s", dv_total);

    // -- Save outputs ---------------------------------------------------------
    save_log("out/truth.csv", &truth_log);
    save_log("out/ekf_est.csv", &ekf_log);
    save_residuals("out/opnav_residuals.csv", &residual_log);
    save_rcs("out/rcs.csv", &rcs_log);

    println!("\n=== Output files ===");
    println!("  out/truth.csv");
    println!("  out/ekf_est.csv");
    println!("  out/opnav_residuals.csv");
    println!("  out/rcs.csv");
    println!("\nTo visualise: python plot/plot_nav.py");
}

// -- Log structures -----------------------------------------------------------

struct LogRow {
    t:  f64,
    x:  f64, y:  f64, z:  f64,
    vx: f64, vy: f64, vz: f64,
    c_r: f64,
    /// Camera boresight unit vector in Hill frame (NaN for EKF rows)
    bx: f64, by: f64, bz: f64,
    /// Attitude quaternion [w, x, y, z]  (NaN for EKF rows)
    qw: f64, qx: f64, qy: f64, qz: f64,
    /// Body-frame angular rates [rad/s]  (NaN for EKF rows)
    omega_x: f64, omega_y: f64, omega_z: f64,
    /// Cumulative RCS DV [m/s]  (NaN for EKF rows)
    dv_total_ms: f64,
    // EKF covariance 1-sigma values (NaN for truth rows)
    /// sqrt(P_xx + P_yy + P_zz)  — overall position 1-sigma [m]
    sigma_r_m:   f64,
    /// sqrt(P_xx), sqrt(P_yy), sqrt(P_zz) — per-axis position sigmas [m]
    sigma_x_m:   f64,
    sigma_y_m:   f64,
    sigma_z_m:   f64,
    /// sqrt((P_vx+P_vy+P_vz)/3) — RMS velocity 1-sigma [m/s]
    sigma_v_mps: f64,
    /// sqrt(P_cr) — C_SRP 1-sigma
    sigma_cr:    f64,
}

impl LogRow {
    fn from_truth(s: &TruthState, dv_total: f64) -> Self {
        Self {
            t: s.t, x: s.r[0], y: s.r[1], z: s.r[2],
            vx: s.v[0], vy: s.v[1], vz: s.v[2], c_r: s.c_r,
            bx: s.boresight[0], by: s.boresight[1], bz: s.boresight[2],
            qw: s.q[0], qx: s.q[1], qy: s.q[2], qz: s.q[3],
            omega_x: s.omega[0], omega_y: s.omega[1], omega_z: s.omega[2],
            dv_total_ms: dv_total,
            sigma_r_m: f64::NAN, sigma_x_m: f64::NAN,
            sigma_y_m: f64::NAN, sigma_z_m: f64::NAN,
            sigma_v_mps: f64::NAN, sigma_cr: f64::NAN,
        }
    }
    fn from_ekf(e: &EkfState) -> Self {
        let sigma_x  = e.p[(0,0)].max(0.0).sqrt();
        let sigma_y  = e.p[(1,1)].max(0.0).sqrt();
        let sigma_z  = e.p[(2,2)].max(0.0).sqrt();
        let sigma_r  = (e.p[(0,0)] + e.p[(1,1)] + e.p[(2,2)]).max(0.0).sqrt();
        let sigma_v  = ((e.p[(3,3)] + e.p[(4,4)] + e.p[(5,5)]).max(0.0) / 3.0).sqrt();
        let sigma_cr = e.p[(6,6)].max(0.0).sqrt();
        Self {
            t: e.t, x: e.x[0], y: e.x[1], z: e.x[2],
            vx: e.x[3], vy: e.x[4], vz: e.x[5], c_r: e.x[6],
            bx: f64::NAN, by: f64::NAN, bz: f64::NAN,
            qw: f64::NAN, qx: f64::NAN, qy: f64::NAN, qz: f64::NAN,
            omega_x: f64::NAN, omega_y: f64::NAN, omega_z: f64::NAN,
            dv_total_ms: f64::NAN,
            sigma_r_m: sigma_r, sigma_x_m: sigma_x,
            sigma_y_m: sigma_y, sigma_z_m: sigma_z,
            sigma_v_mps: sigma_v, sigma_cr,
        }
    }
}

/// Pre-update measurement residual (measured minus EKF-predicted).
/// Named "residual" rather than "innovation" to be immediately self-descriptive.
struct MeasResidualRow {
    t:               f64,
    /// |measured_LOS - predicted_LOS| [rad] — captures bearing prediction error
    los_residual:    f64,
    /// |measured_angular_size - predicted_size| [rad] — captures range prediction error
    size_residual:   f64,
    /// True when Bennu was inside camera FOV and a measurement was processed
    opnav_available: bool,
    /// Sun–Bennu–spacecraft phase angle [rad] at this epoch (NaN if no measurement)
    phase_angle_rad: f64,
}

/// Per-step RCS log (1 s cadence).
struct RcsRow {
    t:           f64,
    tau_x:       f64,
    tau_y:       f64,
    tau_z:       f64,
    omega_x:     f64,
    omega_y:     f64,
    omega_z:     f64,
    dv_inc_ms:   f64,
    dv_total_ms: f64,
}

fn save_log(path: &str, rows: &[LogRow]) {
    use std::fmt::Write as W;
    let mut out = String::with_capacity(rows.len() * 200);
    writeln!(out,
        "time_s,x_m,y_m,z_m,vx_ms,vy_ms,vz_ms,c_r,bx,by,bz,\
         qw,qx,qy,qz,omega_x_rads,omega_y_rads,omega_z_rads,dv_total_ms,\
         sigma_r_m,sigma_x_m,sigma_y_m,sigma_z_m,sigma_v_mps,sigma_cr").unwrap();
    for r in rows {
        writeln!(out,
            "{:.2},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},\
             {:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.6},\
             {:.4},{:.4},{:.4},{:.4},{:.6},{:.6}",
            r.t, r.x, r.y, r.z, r.vx, r.vy, r.vz, r.c_r, r.bx, r.by, r.bz,
            r.qw, r.qx, r.qy, r.qz,
            r.omega_x, r.omega_y, r.omega_z, r.dv_total_ms,
            r.sigma_r_m, r.sigma_x_m, r.sigma_y_m, r.sigma_z_m,
            r.sigma_v_mps, r.sigma_cr).unwrap();
    }
    std::fs::write(path, &out).expect("write failed");
    println!("  Saved {path} ({} rows)", rows.len());
}

fn save_rcs(path: &str, rows: &[RcsRow]) {
    use std::fmt::Write as W;
    let mut out = String::with_capacity(rows.len() * 110);
    writeln!(out,
        "time_s,tau_x_nm,tau_y_nm,tau_z_nm,\
         omega_x_rads,omega_y_rads,omega_z_rads,dv_inc_ms,dv_total_ms").unwrap();
    for r in rows {
        writeln!(out, "{:.2},{:.6},{:.6},{:.6},{:.9},{:.9},{:.9},{:.8e},{:.6}",
            r.t, r.tau_x, r.tau_y, r.tau_z,
            r.omega_x, r.omega_y, r.omega_z,
            r.dv_inc_ms, r.dv_total_ms).unwrap();
    }
    std::fs::write(path, &out).expect("write failed");
    println!("  Saved {path} ({} rows)", rows.len());
}

/// Save pre-update OpNav measurement residuals.
/// "Residual" = measured minus EKF-predicted, computed before the EKF update step.
/// A well-tuned filter has residuals that are zero-mean and consistent with
/// the noise sigmas (i.e. |residual| / sigma ~ 1).
fn save_residuals(path: &str, rows: &[MeasResidualRow]) {
    use std::fmt::Write as W;
    let mut out = String::with_capacity(rows.len() * 90);
    writeln!(out,
        "time_s,bearing_residual_rad,size_residual_rad,opnav_available,phase_angle_rad"
    ).unwrap();
    for r in rows {
        let avail = if r.opnav_available { 1 } else { 0 };
        writeln!(out, "{:.2},{:.6e},{:.6e},{},{:.6}",
            r.t, r.los_residual, r.size_residual, avail, r.phase_angle_rad).unwrap();
    }
    std::fs::write(path, &out).expect("write failed");
    println!("  Saved {path} ({} rows)", rows.len());
}
