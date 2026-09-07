//! Cruise-to-proximity handoff: reads the final cruise-phase state and converts
//! it to a Bennu-centred Hill-frame initial condition for the proximity simulator.
//!
//! Injects a sampled Bennu ephemeris offset to represent the residual position
//! uncertainty that ground-based ranging cannot resolve.  The proximity EKF
//! then corrects for this offset through OpNav bearing and angular-size updates.

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Normal, Distribution};
use crate::bennu_ephem::BennuEphem;
use crate::dynamics::hill::accel_filter;
use crate::dynamics::bennu::bennu_heliocentric_pos;
use crate::config::{C_R_NOMINAL, BENNU_EPHEM_SIGMA_M};

/// End-of-cruise ground-OD position 1-σ [m].
pub const OD_SIGMA_R_M: f64 = 300.0;

/// End-of-cruise ground-OD velocity 1-σ [m/s].
pub const OD_SIGMA_V_MPS: f64 = 0.01;

/// C_SRP 1-σ carried into proximity from cruise OD.
pub const HANDOFF_SIGMA_CR: f64 = 0.05;

/// Range at which the fast-approach coast ends and the proximity EKF sim begins [m].
/// At this range OpNav SNR is ~40×, giving tight bearing and range updates.
const PROXIMITY_START_RANGE_M: f64 = 30_000.0;

/// Cruise-to-proximity handoff: initial conditions for the proximity phase.
pub struct ProximityHandoff {
    /// Hill-frame truth initial position [m]  (referenced to perturbed Bennu)
    pub r_truth:          Vector3<f64>,
    /// Hill-frame truth initial velocity [m/s]
    pub v_truth:          Vector3<f64>,
    /// Hill-frame EKF mean initial position [m]  (referenced to nominal Bennu)
    pub r_est:            Vector3<f64>,
    /// Hill-frame EKF mean initial velocity [m/s]
    pub v_est:            Vector3<f64>,
    /// Combined position 1-σ for EKF P0: √(σ_OD² + σ_ephem²) [m]
    pub sigma_r0_m:       f64,
    /// Velocity 1-σ for EKF P0 [m/s]
    pub sigma_v0_mps:     f64,
    /// Epoch at the start of the proximity EKF sim [TDB seconds from J2000]
    /// (= cruise arrival epoch + fast-approach coast duration)
    pub t_arr:            f64,
    /// Injected Bennu ephemeris offset applied to truth only [m] (for diagnostics)
    pub dr_bennu:         Vector3<f64>,
    /// Range at cruise arrival (before fast-approach coast) [m]
    pub handoff_range_m:  f64,
    /// Duration of the fast-approach coast [s]
    pub coast_duration_s: f64,
}

impl ProximityHandoff {
    /// Load handoff from cruise output CSVs.
    ///
    /// Returns `Err` with a diagnostic message if any required file is missing,
    /// so the caller can fall back to the standalone config defaults gracefully.
    pub fn load<R: Rng>(rng: &mut R) -> Result<Self, String> {
        // Arrival epoch from cruise design output
        let (dep_days, tof_days) = load_best_solution()?;
        let t_arr = (dep_days + tof_days) * 86_400.0;

        // Final heliocentric states from cruise simulation
        let (r_sc_truth, v_sc_truth) =
            read_last_helio("out/cruise_ops/truth.csv")?;
        let (r_sc_od, v_sc_od) =
            read_last_helio("out/cruise_ops/ground_od.csv")?;

        // Nominal Bennu state at arrival from ephemeris
        let ephem = BennuEphem::load("horizons_results_bennu.txt");
        let (rb, vb) = ephem.query(t_arr);
        let r_bennu_nom = Vector3::new(rb[0], rb[1], rb[2]);
        let v_bennu_nom = Vector3::new(vb[0], vb[1], vb[2]);

        // Sample a fixed Bennu position offset to simulate ephemeris uncertainty.
        // Drawn once at scenario start; the proximity filter uses the unperturbed
        // nominal Bennu and must correct via OpNav observations.
        let n01 = Normal::new(0.0_f64, 1.0).unwrap();
        let dr_bennu = Vector3::new(
            BENNU_EPHEM_SIGMA_M * n01.sample(rng),
            BENNU_EPHEM_SIGMA_M * n01.sample(rng),
            BENNU_EPHEM_SIGMA_M * n01.sample(rng),
        );

        // Hill-frame conversion
        // Truth: spacecraft relative to the *perturbed* true Bennu position
        // Estimate: spacecraft relative to the *nominal* ephemeris Bennu
        let r_truth = r_sc_truth - (r_bennu_nom + dr_bennu);
        let v_truth = v_sc_truth - v_bennu_nom;
        let r_est   = r_sc_od   - r_bennu_nom;
        let v_est   = v_sc_od   - v_bennu_nom;

        // Fast-approach coast: propagate the spacecraft from the cruise handoff
        // (~465 km) to PROXIMITY_START_RANGE_M (~30 km) using the filter dynamics.
        // The 24-hour long-range coast is near-free-flight and boring; skipping it
        // lets the EKF sim focus on the regime where OpNav is informative.
        // Truth and estimate are propagated independently (they have slightly
        // different initial positions due to navigation errors).
        let handoff_range_m = r_truth.norm();
        let (r_truth, v_truth, t_prox) = fast_approach(r_truth, v_truth, t_arr);
        let coast_s = t_prox - t_arr;
        // Propagate estimate for exactly the same duration so the EKF clock matches.
        let (r_est, v_est, _) = propagate_for(r_est, v_est, t_arr, coast_s);

        // Combined P0 sigma.  The dominant term after a long coast is the velocity
        // error from the arrival-approach burn (AB) execution uncertainty (~0.1% of
        // a ~km/s burn ≈ 1–2 m/s), which propagates to ~100–200 km over a 20–30 h
        // coast.  Without inflating sigma_r0_m to cover this, the EKF receives
        // enormous (~10 σ) innovations on its first OpNav measurements and diverges.
        //   σ_r_coast = σ_v_handoff × coast_s
        // where σ_v_handoff includes OD velocity error and AB execution uncertainty.
        const HANDOFF_SIGMA_V_MPS: f64 = 2.0;   // conservative post-AB velocity uncertainty [m/s]
        let sigma_r_coast = HANDOFF_SIGMA_V_MPS * coast_s;
        let sigma_r0_m = (OD_SIGMA_R_M.powi(2)
                        + BENNU_EPHEM_SIGMA_M.powi(2)
                        + sigma_r_coast.powi(2)).sqrt();

        // Sanity check: if cruise OD diverged (error > 100 km), the proximity EKF
        // would start with enormous innovations that cause filter divergence.
        // Return Err so the caller falls back to standalone proximity defaults.
        let od_error = (r_est - r_truth).norm();
        if od_error > 100_000.0 {
            return Err(format!(
                "Cruise OD position error {:.1} km at proximity start — \
                 falling back to standalone defaults (3 km circular orbit)",
                od_error / 1_000.0
            ));
        }

        // Sanity check: if the spacecraft never reached < 35 km from Bennu during
        // fast_approach, the proximity guidance would need to handle a very long
        // capture phase with tidal-dominated dynamics. Fall back to standalone.
        if r_truth.norm() > 35_000.0 {
            return Err(format!(
                "Proximity start range {:.1} km > 35 km — \
                 falling back to standalone defaults (3 km circular orbit)",
                r_truth.norm() / 1_000.0
            ));
        }

        Ok(Self {
            r_truth,
            v_truth,
            r_est,
            v_est,
            sigma_r0_m,
            sigma_v0_mps: OD_SIGMA_V_MPS,
            t_arr: t_prox,
            dr_bennu,
            handoff_range_m,
            coast_duration_s: coast_s,
        })
    }
}

// ── CSV helpers ────────────────────────────────────────────────────────────────

/// Read the last data row from a cruise heliocentric state CSV.
///
/// CSV columns: `time_s, x_m, y_m, z_m, vx_ms, vy_ms, vz_ms, cr`
fn read_last_helio(path: &str) -> Result<(Vector3<f64>, Vector3<f64>), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|_| format!("Cannot read '{path}' — run cruise_operations first"))?;

    let last = text
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with("time")
        })
        .last()
        .ok_or_else(|| format!("No data rows in '{path}'"))?;

    let cols: Vec<f64> = last
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|_| format!("Parse error in last row of '{path}'"))
        })
        .collect::<Result<_, _>>()?;

    if cols.len() < 7 {
        return Err(format!("Expected ≥7 columns, got {} in '{path}'", cols.len()));
    }

    // cols: [time_s, x_m, y_m, z_m, vx_ms, vy_ms, vz_ms, cr]
    Ok((
        Vector3::new(cols[1], cols[2], cols[3]),
        Vector3::new(cols[4], cols[5], cols[6]),
    ))
}

/// Read `dep_day` and `tof_day` from the cruise design output.
fn load_best_solution() -> Result<(f64, f64), String> {
    let path = "out/cruise/best_solution.csv";
    let text = std::fs::read_to_string(path)
        .map_err(|_| format!("Cannot read '{path}' — run cruise_design first"))?;

    let line = text
        .lines()
        .nth(1)
        .ok_or_else(|| format!("'{path}' has no data row"))?;

    let mut cols = line.split(',');
    let dep: f64 = cols
        .next()
        .unwrap_or("")
        .trim()
        .parse()
        .map_err(|_| format!("Cannot parse dep_day in '{path}'"))?;
    let tof: f64 = cols
        .next()
        .unwrap_or("")
        .trim()
        .parse()
        .map_err(|_| format!("Cannot parse tof_day in '{path}'"))?;

    Ok((dep, tof))
}

// ── Fast-approach coast propagators ────────────────────────────────────────────

/// Propagate a Hill-frame state forward using RK4 + filter dynamics to the
/// closest-approach point (periapsis) of the flyby trajectory.
///
/// At 465 km from Bennu (far outside its ~31 km Hill sphere), solar tidal
/// forces dominate; the 5 m/s approach velocity follows a heliocentric flyby
/// rather than a monotonically decreasing range.  We therefore stop at the
/// natural periapsis (when range begins increasing again) or at
/// PROXIMITY_START_RANGE_M, whichever comes first.
///
/// Uses 60-second steps. A 30-day guard prevents an infinite loop.
fn fast_approach(
    r0: Vector3<f64>,
    v0: Vector3<f64>,
    t0: f64,
) -> (Vector3<f64>, Vector3<f64>, f64) {
    const DT: f64 = 60.0;
    const MAX_STEPS: usize = 43_200; // 30-day guard
    let mut r = r0;
    let mut v = v0;
    let mut t = t0;
    let mut prev_range = r.norm();
    for _ in 0..MAX_STEPS {
        let cur_range = r.norm();
        // Stop at target range or when range starts growing (periapsis passed)
        if cur_range <= PROXIMITY_START_RANGE_M { break; }
        if cur_range > prev_range + 1.0 { break; } // periapsis passed
        prev_range = cur_range;
        let (rn, vn) = hill_rk4(r, v, t, DT);
        r = rn; v = vn; t += DT;
    }
    (r, v, t)
}

/// Propagate a Hill-frame state forward for exactly `duration` seconds using RK4.
fn propagate_for(
    r0: Vector3<f64>,
    v0: Vector3<f64>,
    t0: f64,
    duration: f64,
) -> (Vector3<f64>, Vector3<f64>, f64) {
    const DT: f64 = 60.0;
    let n = ((duration / DT).ceil() as usize).max(1);
    let dt = duration / n as f64;
    let mut r = r0;
    let mut v = v0;
    let mut t = t0;
    for _ in 0..n {
        let (rn, vn) = hill_rk4(r, v, t, dt);
        r = rn; v = vn; t += dt;
    }
    (r, v, t)
}

/// Single RK4 step in the Hill frame using the EKF filter dynamics
/// (point-mass Bennu gravity + solar tidal + cannonball SRP).
fn hill_rk4(r: Vector3<f64>, v: Vector3<f64>, t: f64, dt: f64)
    -> (Vector3<f64>, Vector3<f64>)
{
    let bp0 = bennu_heliocentric_pos(t);
    let bp1 = bennu_heliocentric_pos(t + dt * 0.5);
    let bp2 = bennu_heliocentric_pos(t + dt);

    let k1r = v;
    let k1v = accel_filter(&r, &bp0, C_R_NOMINAL);

    let r2 = r + k1r * (dt * 0.5);
    let v2 = v + k1v * (dt * 0.5);
    let k2r = v2;
    let k2v = accel_filter(&r2, &bp1, C_R_NOMINAL);

    let r3 = r + k2r * (dt * 0.5);
    let v3 = v + k2v * (dt * 0.5);
    let k3r = v3;
    let k3v = accel_filter(&r3, &bp1, C_R_NOMINAL);

    let r4 = r + k3r * dt;
    let v4 = v + k3v * dt;
    let k4r = v4;
    let k4v = accel_filter(&r4, &bp2, C_R_NOMINAL);

    let r_new = r + (dt / 6.0) * (k1r + 2.0*k2r + 2.0*k3r + k4r);
    let v_new = v + (dt / 6.0) * (k1v + 2.0*k2v + 2.0*k3v + k4v);
    (r_new, v_new)
}
