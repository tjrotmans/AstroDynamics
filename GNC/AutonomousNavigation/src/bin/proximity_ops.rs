//! Bennu proximity operations — orbit insertion and station-keeping.
//!
//! Scenario (7 days):
//!   1. Start at a 3 km circular orbit (standalone defaults) or from the
//!      cruise handoff initial state.
//!   2. Execute a two-burn Hohmann transfer to a 1.5 km circular orbit.
//!      Burns are impulsive within one 10-second truth step.
//!   3. Station-keep for the remainder: correct orbital energy drift every hour
//!      due to SRP perturbations.
//!   4. DSN ground tracking pass every 8 h feeds a heliocentric state update
//!      into the EKF, supplementing OpNav bearing and angular-size measurements.
//!
//! Outputs (written to `out/prox_ops/`):
//!   truth.csv          — ground-truth spacecraft state
//!   ekf_est.csv        — EKF mean + 1-σ covariance
//!   maneuvers.csv      — manoeuvre log (time, ΔV vector, phase)
//!   dsn_updates.csv    — DSN update log (time, innovation magnitude)
//!
//! Run:  `cargo run -p autonomous_navigation --bin proximity_ops --release`
//! Plot: `python plot/plot_prox_ops.py`

use autonomous_navigation::{
    config::{
        MU_BENNU, SC_MASS, SC_AREA_CANNONBALL, C_R_NOMINAL,
        TARGET_ORBIT_R_M, SK_ENERGY_TOL, SK_INTERVAL_S,
        DSN_UPDATE_INTERVAL_S,
        PROX_OPS_DURATION_S, PROX_DT_TRUTH_S, PROX_DT_MEAS_S,
    },
    dynamics::{TruthState, Propagator},
    navigation::ekf::{EkfState, predict, update_bearing, update_angular_size, update_dsn},
    proximity_init::{ProximityHandoff, HANDOFF_SIGMA_CR},
    sensors::{opnav, star_tracker, dsn},
    dynamics::bennu::bennu_heliocentric_pos,
};
use nalgebra::{Vector3, Vector4};

// ── Orbit parameters ──────────────────────────────────────────────────────────

/// Starting orbit radius for default (non-handoff) initial conditions [m].
const START_ORBIT_R_M: f64 = 3_000.0;

/// Braking threshold for ΔV2 detection: fire circularisation when range ≤ this [m].
const PERI_DETECT_R_M: f64 = TARGET_ORBIT_R_M * 1.08;

/// Maximum translational thrust force [N] (3 translation thrusters at RCS_THRUST_N each).
const MAX_TRANS_FORCE_N: f64 = 3.0;

// ── Guidance phase ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
enum Phase {
    /// Large continuous retrograde burn to capture from a hyperbolic flyby approach.
    /// Only entered when the cruise handoff delivers a hyperbolic (ε > 0) state.
    OrbInsertion,
    /// Waiting for initial EKF convergence (first 5 min after capture).
    InitCoast,
    /// ΔV1: retrograde burn to enter the Hohmann transfer.
    Burn1,
    /// Coasting on the elliptic transfer orbit toward periapsis.
    TransferCoast,
    /// ΔV2: circularisation burn at periapsis of the transfer orbit.
    Burn2,
    /// Circular orbit at TARGET_ORBIT_R_M + station-keeping corrections.
    OrbitHold,
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    std::fs::create_dir_all("out/prox_ops").unwrap();

    println!("=== Bennu Proximity Operations ===");
    println!("  Target orbit : {:.1} km circular", TARGET_ORBIT_R_M / 1e3);
    println!("  Duration     : {:.1} days", PROX_OPS_DURATION_S / 86_400.0);

    let mut rng = rand::thread_rng();

    // ── Initial conditions ────────────────────────────────────────────────────
    let (mut truth, mut ekf, sigma_r0, _sigma_v0, init_phase) =
        init_from_handoff_or_default(&mut rng);
    let prop = Propagator::new(PROX_DT_TRUTH_S);

    println!("  Init range   : {:.2} km from Bennu", truth.r.norm() / 1e3);
    println!("  Init vel     : {:.4} m/s", truth.v.norm());
    println!("  σ_r0         : {:.1} m", sigma_r0);

    // ── Ground OD EKF (heliocentric, full 7-state STM) ────────────────────────
    // Mirrors what the ground tracking network runs: processes raw DSN range,
    // range-rate, and Delta-DOR observables every 1–2 h, then uplinks the OD
    // solution to the spacecraft every DSN_UPDATE_INTERVAL_S.
    let r_bennu_init = bennu_heliocentric_pos(truth.t);
    let v_bennu_init = (bennu_heliocentric_pos(truth.t + 10.0) - r_bennu_init) / 10.0;
    let mut ground_od = dsn::GroundOdEkf::new(
        r_bennu_init + truth.r,
        v_bennu_init + truth.v,
        C_R_NOMINAL,
        5_000.0, 0.05, 0.1,      // σ_r = 5 km (Bennu ephem), σ_v = 5 cm/s, σ_C_R = 0.1
        SC_AREA_CANNONBALL / SC_MASS,
    );
    let od_every     = (dsn::OD_STEP_S    / PROX_DT_TRUTH_S).round() as usize;
    let uplink_every = (DSN_UPDATE_INTERVAL_S / dsn::OD_STEP_S).round() as usize;
    let mut od_step      = 0usize;
    let mut od_meas_cnt  = 0usize;

    // ── Timing ────────────────────────────────────────────────────────────────
    let n_steps    = (PROX_OPS_DURATION_S / PROX_DT_TRUTH_S).round() as usize;
    let meas_every = (PROX_DT_MEAS_S      / PROX_DT_TRUTH_S).round() as usize;
    let sk_every   = (SK_INTERVAL_S         / PROX_DT_TRUTH_S).round() as usize;
    let init_coast_steps = (300.0 / PROX_DT_TRUTH_S).round() as usize; // 5-min nav settle

    // ── Logging ───────────────────────────────────────────────────────────────
    let mut truth_log:    Vec<ProxLogRow> = Vec::with_capacity(n_steps / meas_every + 1);
    let mut ekf_log:      Vec<ProxLogRow> = Vec::with_capacity(n_steps / meas_every + 1);
    let mut maneuver_log: Vec<ManRow>     = Vec::new();
    let mut dsn_log:      Vec<DsnRow>     = Vec::new();

    let mut dv_total_ms  = 0.0_f64;
    let mut n_missed_opnav = 0usize;
    let mut n_opnav_total  = 0usize;
    let mut n_dsn          = 0usize;

    // ── Phase machine ─────────────────────────────────────────────────────────
    let mut phase           = init_phase;
    let mut burn1_done      = false;
    let mut orb_ins_logged  = false; // log insertion manoeuvre once on capture
    let mut burn2_done      = false;
    let mut last_sk_step    = 0usize;
    let mut prev_range      = truth.r.norm();

    // ── Main simulation loop ──────────────────────────────────────────────────
    for step in 0..n_steps {
        let range   = truth.r.norm();
        let r_hat   = truth.r / range;
        let v_radial = truth.v.dot(&r_hat);  // + = receding

        // ── Guidance: compute Hill-frame translational force ──────────────────
        let thrust_hill = match phase {

            Phase::OrbInsertion => {
                let eps = orbital_energy(&truth.r, &truth.v);
                if eps < 0.0 {
                    // Captured — log the total insertion burn and move on.
                    if !orb_ins_logged {
                        orb_ins_logged = true;
                        let (sma, ecc) = orbital_elements(&truth.r, &truth.v);
                        println!("  Orbit insertion complete @ t={:.2} h", truth.t / 3600.0);
                        println!("    Captured orbit: SMA={:.1} km  e={:.4}", sma/1e3, ecc);
                        println!("    Total insertion ΔV so far: {:.3} m/s", dv_total_ms);
                        maneuver_log.push(ManRow {
                            t_s: truth.t,
                            dv_x: 0.0, dv_y: 0.0, dv_z: 0.0,
                            dv_mag: dv_total_ms,
                            phase: "OrbInsertion_end",
                        });
                    }
                    phase = Phase::InitCoast;
                    None
                } else {
                    // Full retrograde thrust until captured.
                    let v_hat = if truth.v.norm() > 1e-10 {
                        truth.v / truth.v.norm()
                    } else {
                        Vector3::zeros()
                    };
                    let force = -v_hat * MAX_TRANS_FORCE_N;
                    let dv_step = force * (PROX_DT_TRUTH_S / SC_MASS);
                    dv_total_ms += dv_step.norm();
                    Some(force)
                }
            }

            Phase::InitCoast => {
                if step >= init_coast_steps { phase = Phase::Burn1; }
                None
            }

            Phase::Burn1 if !burn1_done => {
                // Hohmann first burn: reduce tangential velocity from v_circ(start)
                // to transfer orbit apoapsis speed.
                let r_apo  = range;
                let r_peri = TARGET_ORBIT_R_M;
                let v_apo_transfer = ((MU_BENNU * 2.0 * r_peri) / (r_apo * (r_apo + r_peri))).sqrt();
                let v_tang = truth.v - v_radial * r_hat;
                let v_tang_mag = v_tang.norm();
                let dv_needed = v_tang_mag - v_apo_transfer; // positive → retrograde

                if dv_needed.abs() < 1e-6 {
                    burn1_done = true;
                    phase = Phase::TransferCoast;
                    None
                } else {
                    burn1_done = true;
                    phase = Phase::TransferCoast;

                    let tang_hat = if v_tang_mag > 1e-10 { v_tang / v_tang_mag } else { Vector3::zeros() };
                    let force_mag = (SC_MASS * dv_needed.abs() / PROX_DT_TRUTH_S).min(MAX_TRANS_FORCE_N);
                    let sign = if dv_needed > 0.0 { -1.0 } else { 1.0 }; // retrograde if positive
                    let force = tang_hat * sign * force_mag;

                    let dv_vec = force / SC_MASS * PROX_DT_TRUTH_S;
                    dv_total_ms += dv_vec.norm();
                    maneuver_log.push(ManRow {
                        t_s: truth.t,
                        dv_x: dv_vec[0], dv_y: dv_vec[1], dv_z: dv_vec[2],
                        dv_mag: dv_vec.norm(),
                        phase: "Burn1",
                    });
                    println!("  Burn 1 @ t={:.2} h  r={:.2} km  ΔV={:.4} m/s  → transfer orbit",
                             truth.t / 3600.0, range / 1e3, dv_vec.norm());
                    Some(force)
                }
            }

            Phase::Burn1 => { phase = Phase::TransferCoast; None }

            Phase::TransferCoast => {
                // Detect periapsis: range was decreasing and now increases, OR range ≤ threshold.
                let at_peri = range <= PERI_DETECT_R_M && v_radial < 0.0;
                let past_peri = range > prev_range + 0.1 && prev_range < PERI_DETECT_R_M * 1.5;
                if at_peri || past_peri {
                    phase = Phase::Burn2;
                }
                None
            }

            Phase::Burn2 if !burn2_done => {
                // Circularisation: match tangential speed to circular orbit velocity at current range.
                let v_circ = (MU_BENNU / range).sqrt();
                let v_tang = truth.v - v_radial * r_hat;
                let v_tang_mag = v_tang.norm();
                let dv_needed = v_tang_mag - v_circ; // positive → retrograde (too fast)

                burn2_done = true;
                phase = Phase::OrbitHold;
                last_sk_step = step;

                if dv_needed.abs() < 1e-6 {
                    None
                } else {
                    let tang_hat = if v_tang_mag > 1e-10 { v_tang / v_tang_mag } else { Vector3::zeros() };
                    let force_mag = (SC_MASS * dv_needed.abs() / PROX_DT_TRUTH_S).min(MAX_TRANS_FORCE_N);
                    let sign = if dv_needed > 0.0 { -1.0 } else { 1.0 };
                    let force = tang_hat * sign * force_mag;

                    let dv_vec = force / SC_MASS * PROX_DT_TRUTH_S;
                    dv_total_ms += dv_vec.norm();
                    maneuver_log.push(ManRow {
                        t_s: truth.t,
                        dv_x: dv_vec[0], dv_y: dv_vec[1], dv_z: dv_vec[2],
                        dv_mag: dv_vec.norm(),
                        phase: "Burn2",
                    });
                    println!("  Burn 2 @ t={:.2} h  r={:.2} km  ΔV={:.4} m/s  → {:.2} km orbit",
                             truth.t / 3600.0, range / 1e3, dv_vec.norm(), range / 1e3);
                    Some(force)
                }
            }

            Phase::Burn2 => { phase = Phase::OrbitHold; None }

            Phase::OrbitHold => {
                // Station-keeping: once per SK_INTERVAL_S correct orbital energy drift.
                if step.saturating_sub(last_sk_step) >= sk_every {
                    last_sk_step = step;
                    let eps_current = orbital_energy(&truth.r, &truth.v);
                    let eps_target  = -MU_BENNU / (2.0 * TARGET_ORBIT_R_M);
                    let de = eps_target - eps_current;
                    if de.abs() > SK_ENERGY_TOL {
                        // ΔV ≈ ΔE / v_tangential  (first-order)
                        let v_tang = truth.v - v_radial * r_hat;
                        let v_tang_mag = v_tang.norm().max(1e-10);
                        let dv_needed = de / v_tang_mag; // small signed correction
                        let tang_hat  = v_tang / v_tang_mag;
                        let force_mag = (SC_MASS * dv_needed.abs() / PROX_DT_TRUTH_S)
                                          .min(MAX_TRANS_FORCE_N);
                        let sign = if dv_needed > 0.0 { 1.0 } else { -1.0 };
                        let force = tang_hat * sign * force_mag;

                        let dv_vec = force / SC_MASS * PROX_DT_TRUTH_S;
                        dv_total_ms += dv_vec.norm();
                        maneuver_log.push(ManRow {
                            t_s: truth.t,
                            dv_x: dv_vec[0], dv_y: dv_vec[1], dv_z: dv_vec[2],
                            dv_mag: dv_vec.norm(),
                            phase: "SK",
                        });
                        Some(force)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        };

        prev_range = range;

        // ── Propagate truth ───────────────────────────────────────────────────
        let (new_truth, _rcs) = prop.step_with_force(
            &truth, thrust_hill, autonomous_navigation::guidance::pointing::PointingMode::Nadir);
        truth = new_truth;

        // ── EKF predict ───────────────────────────────────────────────────────
        ekf = predict(&ekf, PROX_DT_TRUTH_S);

        // Inform EKF of any applied translational burn (commanded ΔV is known exactly).
        // Without this, the EKF velocity diverges by the full ΔV during long burns.
        if let Some(f) = thrust_hill {
            let dv = f * (PROX_DT_TRUTH_S / SC_MASS);
            ekf.x[3] += dv[0];
            ekf.x[4] += dv[1];
            ekf.x[5] += dv[2];
        }

        // ── OpNav measurement update (every PROX_DT_MEAS_S) ──────────────────
        if step % meas_every == 0 {
            n_opnav_total += 1;
            let q_st = star_tracker::measure(&truth.q, &mut rng);
            if let Some(meas) = opnav::measure(&truth.r, &truth.q, &q_st, truth.t, &mut rng) {
                ekf = update_bearing(&ekf, &meas.los_inertial, meas.sigma_bearing);
                ekf = update_angular_size(&ekf, meas.angular_size);
            } else {
                n_missed_opnav += 1;
            }

            truth_log.push(ProxLogRow::from_truth(&truth, dv_total_ms, &phase));
            ekf_log.push(ProxLogRow::from_ekf(&ekf, &phase));
        }

        // ── Ground OD predict + update at 1h cadence ─────────────────────────
        // Range + range-rate every 2 OD steps; Delta-DOR every 24 OD steps.
        // Uplink the OD solution to the onboard EKF every uplink_every OD steps.
        if step > 0 && step % od_every == 0 {
            od_step     += 1;
            od_meas_cnt += 1;
            let r_bennu = bennu_heliocentric_pos(truth.t);
            let v_bennu = (bennu_heliocentric_pos(truth.t + 10.0) - r_bennu) / 10.0;
            let (r_earth, v_earth) = dsn::earth_rv(truth.t);
            let r_sc_true = r_bennu + truth.r;
            let v_sc_true = v_bennu + truth.v;

            ground_od.predict(dsn::OD_STEP_S);
            if od_meas_cnt % dsn::OD_MEAS_HZ == 0 {
                let do_ddor = od_meas_cnt % dsn::OD_DDOR_HZ == 0;
                ground_od.update_from_truth(
                    &r_sc_true, &v_sc_true,
                    &r_earth, &v_earth,
                    do_ddor, &mut rng,
                );
            }

            // Uplink to onboard EKF (every 8 OD steps = 8 h = DSN_UPDATE_INTERVAL_S)
            if od_step % uplink_every == 0 {
                let uplink  = ground_od.uplink();
                let r_pre   = ekf.r();
                ekf = update_dsn(&ekf, &uplink.r_helio_m, &uplink.v_helio_mps);
                let innov_mag = (uplink.r_helio_m - (r_bennu + r_pre)).norm();

                n_dsn += 1;
                dsn_log.push(DsnRow {
                    t_s: truth.t,
                    innov_r_m: innov_mag,
                    sigma_r_m: ekf.p[(0,0)].max(0.0).sqrt(),
                });
                println!("  DSN uplink #{n_dsn:2}  t={:.1} h  innov={:.1} m  σ_r={:.1} m",
                         truth.t / 3600.0, innov_mag, ekf.p[(0,0)].max(0.0).sqrt());
            }
        }
    }

    // ── Final report ─────────────────────────────────────────────────────────
    let final_r_err = (truth.r - ekf.r()).norm();
    let final_v_err = (truth.v - ekf.v()).norm();
    let (sma, ecc)  = orbital_elements(&truth.r, &truth.v);
    println!("\n── Final state (t = {:.1} days) ──", PROX_OPS_DURATION_S / 86_400.0);
    println!("  Position error : {:.2} m", final_r_err);
    println!("  Velocity error : {:.4} m/s", final_v_err);
    println!("  Orbit SMA      : {:.2} km  (target {:.2} km)", sma/1e3, TARGET_ORBIT_R_M/1e3);
    println!("  Eccentricity   : {:.4}", ecc);
    println!("  Total ΔV       : {:.4} m/s  ({} burns)", dv_total_ms, maneuver_log.len());
    println!("  OpNav coverage : {}/{} ({:.1}%)",
             n_opnav_total - n_missed_opnav, n_opnav_total,
             100.0 * (n_opnav_total - n_missed_opnav) as f64 / n_opnav_total as f64);
    println!("  DSN passes     : {n_dsn}");

    // ── Save CSVs ─────────────────────────────────────────────────────────────
    save_prox_log("out/prox_ops/truth.csv",   &truth_log);
    save_prox_log("out/prox_ops/ekf_est.csv", &ekf_log);
    save_maneuvers("out/prox_ops/maneuvers.csv", &maneuver_log);
    save_dsn("out/prox_ops/dsn_updates.csv", &dsn_log);

    println!("\nOutputs in out/prox_ops/");
    println!("To visualise: python plot/plot_prox_ops.py");
}

// ── Initialisation ────────────────────────────────────────────────────────────

/// Try to load initial conditions from cruise handoff.  Falls back to a 3 km
/// circular orbit if cruise outputs are not present.
///
/// Returns (truth, ekf, sigma_r0, sigma_v0, initial_phase).
/// If the handoff delivers a hyperbolic state the initial phase is OrbInsertion;
/// otherwise it is InitCoast (or standalone defaults, also InitCoast).
fn init_from_handoff_or_default(
    rng: &mut impl rand::Rng,
) -> (TruthState, EkfState, f64, f64, Phase) {
    if let Ok(h) = ProximityHandoff::load(rng) {
        let eps = orbital_energy(&h.r_truth, &h.v_truth);
        let truth = TruthState::from_handoff(h.r_truth, h.v_truth, h.t_arr);

        if eps >= 0.0 {
            // Hyperbolic approach — spacecraft not yet captured by Bennu.
            // The OD estimate from cruise is unreliable after the fast-approach
            // coast (different trajectory), so initialise the EKF at the truth
            // position with the handoff uncertainty.  The OrbInsertion phase will
            // burn retrogradely until ε < 0 before the Hohmann descent begins.
            let v_esc   = (2.0 * MU_BENNU / h.r_truth.norm()).sqrt();
            let dv_req  = (h.v_truth.norm() - v_esc).max(0.0);
            println!("  Init mode  : cruise handoff (HYPERBOLIC — orbit insertion required)");
            println!("    r={:.1} km  |v|={:.4} m/s  v_esc={:.4} m/s  ε={:.4} J/kg",
                     h.r_truth.norm()/1e3, h.v_truth.norm(), v_esc, eps);
            println!("    Min ΔV to capture ≈ {:.3} m/s  (will burn at {:.0} mm/s² for ~{:.0} min)",
                     dv_req, MAX_TRANS_FORCE_N * 1e3 / SC_MASS,
                     dv_req / (MAX_TRANS_FORCE_N / SC_MASS) / 60.0);
            let ekf = EkfState::with_initial_state(
                h.r_truth, h.v_truth,
                h.sigma_r0_m, h.sigma_v0_mps, HANDOFF_SIGMA_CR,
                h.t_arr,
            );
            (truth, ekf, h.sigma_r0_m, h.sigma_v0_mps, Phase::OrbInsertion)
        } else {
            // Already captured — use the OD estimate directly.
            println!("  Init mode  : cruise handoff (captured, ε={:.4} J/kg, r={:.1} km)",
                     eps, h.r_truth.norm()/1e3);
            let ekf = EkfState::with_initial_state(
                h.r_est, h.v_est,
                h.sigma_r0_m, h.sigma_v0_mps, HANDOFF_SIGMA_CR,
                h.t_arr,
            );
            (truth, ekf, h.sigma_r0_m, h.sigma_v0_mps, Phase::InitCoast)
        }
    } else {
        // No cruise handoff found — start at a 3 km circular orbit.
        let r0       = Vector3::new(0.0, START_ORBIT_R_M, 0.0);
        let v_circ   = (MU_BENNU / START_ORBIT_R_M).sqrt();
        let v0       = Vector3::new(v_circ, 0.0, 0.0);
        let q0       = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let truth    = TruthState::from_state(r0, v0, q0, Vector3::zeros(), 0.0);
        let sigma_r0 = 100.0_f64;
        let sigma_v0 = 0.005_f64;
        let ekf      = EkfState::with_initial_state(
            r0, v0, sigma_r0, sigma_v0, 0.05, 0.0,
        );
        println!("  Init mode  : standalone defaults (3 km circular orbit, no handoff)");
        (truth, ekf, sigma_r0, sigma_v0, Phase::InitCoast)
    }
}

// ── Orbital mechanics helpers ─────────────────────────────────────────────────

fn orbital_energy(r: &Vector3<f64>, v: &Vector3<f64>) -> f64 {
    v.norm_squared() / 2.0 - MU_BENNU / r.norm()
}

fn orbital_elements(r: &Vector3<f64>, v: &Vector3<f64>) -> (f64, f64) {
    let eps = orbital_energy(r, v);
    let sma = if eps.abs() > 1e-20 { -MU_BENNU / (2.0 * eps) } else { f64::INFINITY };
    let h   = r.cross(v);
    let e_vec = v.cross(&h) / MU_BENNU - r / r.norm();
    (sma, e_vec.norm())
}

// ── Log structures ────────────────────────────────────────────────────────────

struct ProxLogRow {
    t_s:       f64,
    x_m:       f64, y_m: f64, z_m: f64,
    vx_ms:     f64, vy_ms: f64, vz_ms: f64,
    range_km:  f64,
    sigma_r_m: f64,
    sigma_v_mps: f64,
    phase:     &'static str,
}

impl ProxLogRow {
    fn from_truth(s: &TruthState, _dv: f64, phase: &Phase) -> Self {
        Self {
            t_s: s.t,
            x_m: s.r[0], y_m: s.r[1], z_m: s.r[2],
            vx_ms: s.v[0], vy_ms: s.v[1], vz_ms: s.v[2],
            range_km: s.r.norm() / 1e3,
            sigma_r_m: f64::NAN, sigma_v_mps: f64::NAN,
            phase: phase_name(phase),
        }
    }
    fn from_ekf(e: &EkfState, phase: &Phase) -> Self {
        let sr = (e.p[(0,0)] + e.p[(1,1)] + e.p[(2,2)]).max(0.0).sqrt();
        let sv = ((e.p[(3,3)] + e.p[(4,4)] + e.p[(5,5)]).max(0.0) / 3.0).sqrt();
        Self {
            t_s: e.t,
            x_m: e.x[0], y_m: e.x[1], z_m: e.x[2],
            vx_ms: e.x[3], vy_ms: e.x[4], vz_ms: e.x[5],
            range_km: e.r().norm() / 1e3,
            sigma_r_m: sr, sigma_v_mps: sv,
            phase: phase_name(phase),
        }
    }
}

struct ManRow { t_s: f64, dv_x: f64, dv_y: f64, dv_z: f64, dv_mag: f64, phase: &'static str }
struct DsnRow  { t_s: f64, innov_r_m: f64, sigma_r_m: f64 }

fn phase_name(p: &Phase) -> &'static str {
    match p {
        Phase::OrbInsertion  => "OrbInsertion",
        Phase::InitCoast     => "InitCoast",
        Phase::Burn1         => "Burn1",
        Phase::TransferCoast => "TransferCoast",
        Phase::Burn2         => "Burn2",
        Phase::OrbitHold     => "OrbitHold",
    }
}

// ── CSV helpers ───────────────────────────────────────────────────────────────

fn save_prox_log(path: &str, rows: &[ProxLogRow]) {
    use std::fmt::Write as W;
    let mut out = String::with_capacity(rows.len() * 120);
    writeln!(out, "time_s,x_m,y_m,z_m,vx_ms,vy_ms,vz_ms,range_km,sigma_r_m,sigma_v_mps,phase").unwrap();
    for r in rows {
        writeln!(out, "{:.2},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.4},{:.4},{:.6},{}",
            r.t_s, r.x_m, r.y_m, r.z_m,
            r.vx_ms, r.vy_ms, r.vz_ms,
            r.range_km, r.sigma_r_m, r.sigma_v_mps, r.phase).unwrap();
    }
    std::fs::write(path, &out).expect("write failed");
    println!("  Saved {path} ({} rows)", rows.len());
}

fn save_maneuvers(path: &str, rows: &[ManRow]) {
    use std::fmt::Write as W;
    let mut out = String::new();
    writeln!(out, "time_s,dv_x_ms,dv_y_ms,dv_z_ms,dv_mag_ms,phase").unwrap();
    for r in rows {
        writeln!(out, "{:.2},{:.6e},{:.6e},{:.6e},{:.6e},{}",
            r.t_s, r.dv_x, r.dv_y, r.dv_z, r.dv_mag, r.phase).unwrap();
    }
    std::fs::write(path, &out).expect("write failed");
    println!("  Saved {path} ({} maneuvers)", rows.len());
}

fn save_dsn(path: &str, rows: &[DsnRow]) {
    use std::fmt::Write as W;
    let mut out = String::new();
    writeln!(out, "time_s,innov_r_m,sigma_r_m").unwrap();
    for r in rows {
        writeln!(out, "{:.2},{:.4},{:.4}", r.t_s, r.innov_r_m, r.sigma_r_m).unwrap();
    }
    std::fs::write(path, &out).expect("write failed");
    println!("  Saved {path} ({} DSN passes)", rows.len());
}
