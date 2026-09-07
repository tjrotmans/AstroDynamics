//! 5-phase Bennu proximity operations mission simulator.
//!
//! Phases (in order):
//!   1. Capture      — orbit insertion from handoff approach, then transfer to 3 km
//!   2. Survey       — 3 km circular survey orbit, 3 days
//!   3. CloseOrbit   — 1 km science orbit, 3 days
//!   4. Flyover      — 500 m altitude passes, 2 days
//!   5. ScienceHold  — 1 km stable science hold, 2 days
//!
//! Outputs (written to `out/mission/`):
//!   nav.csv          — navigation state (truth + EKF, per phase)
//!   attitude.csv     — attitude + reaction wheel dynamics (per step)
//!   maneuvers.csv    — all delta-V events
//!   dsn_updates.csv  — DSN uplink log
//!
//! Run:  `cargo run -p autonomous_navigation --bin proximity_mission --release`
//! Fast: `cargo run -p autonomous_navigation --bin proximity_mission --release -- --dt 60 --meas-dt 600`
//! Plot: `python plot/plot_mission.py`
//!
//! CLI flags (all optional):
//!   --dt  <s>       truth integration step  [default: PROX_DT_TRUTH_S from config]
//!   --meas-dt <s>   OpNav/logging interval  [default: PROX_DT_MEAS_S  from config]

use autonomous_navigation::{
    config::{
        MU_BENNU, SC_MASS, SC_AREA_CANNONBALL, C_R_NOMINAL,
        SK_ENERGY_TOL, SK_INTERVAL_S,
        DSN_UPDATE_INTERVAL_S,
        PROX_DT_TRUTH_S, PROX_DT_MEAS_S,  // used as CLI defaults
        BENNU_EPHEM_SIGMA_M,               // initial EKF position uncertainty floor
    },
    dynamics::{TruthState, Propagator},
    guidance::pointing::{PointingMode, desired_quaternion},
    navigation::ekf::{EkfState, predict, predict_calibration, update_bearing, update_angular_size, update_lidar, update_dsn, update_landmark},
    proximity_init::{ProximityHandoff, HANDOFF_SIGMA_CR},
    sensors::{opnav, star_tracker, dsn, landmark},
    sensors::landmark::LmState,
    dynamics::bennu::bennu_heliocentric_pos,
    dynamics::srp::{spacecraft_plates, srp_accel_panels, srp_accel_cannonball},
};
use nalgebra::Vector3;

// ── Phase definitions ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
enum Phase {
    Capture,      // orbit insertion + transfer to 3 km survey orbit
    Survey,       // 3 km circular orbit
    CloseOrbit,   // 1 km science orbit
    Flyover,      // 500 m altitude passes
    ScienceHold,  // 1 km science hold
    RadioScience, // quiescent SRP/gravity-calibration arc (Orbit-B analogue)
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Phase::Capture      => "Capture",
            Phase::Survey       => "Survey",
            Phase::CloseOrbit   => "CloseOrbit",
            Phase::Flyover      => "Flyover",
            Phase::ScienceHold  => "ScienceHold",
            Phase::RadioScience => "RadioScience",
        }
    }

    /// Target circular orbit radius [m] for the end of this phase.
    fn target_r_m(self) -> f64 {
        match self {
            Phase::Capture      => 3_000.0,   // survey orbit
            Phase::Survey       => 3_000.0,
            Phase::CloseOrbit   =>   900.0,
            Phase::Flyover      =>   500.0,
            Phase::ScienceHold  =>   900.0,
            Phase::RadioScience => 3_000.0,   // hold the survey orbit, no burns
        }
    }

    /// Nominal phase duration [s] (Capture is variable).
    fn duration_s(self) -> f64 {
        match self {
            Phase::Capture      => 12.0 * 3_600.0,  // max 12 h capture + transfer
            Phase::Survey       =>  3.0 * 86_400.0,
            Phase::CloseOrbit   =>  3.0 * 86_400.0,
            Phase::Flyover      =>  2.0 * 86_400.0,
            Phase::ScienceHold  =>  2.0 * 86_400.0,
            Phase::RadioScience =>  6.0 * 86_400.0,  // long quiescent calibration arc
        }
    }

    /// True during the quiescent radio-science arc: no maneuvers, no desats, and
    /// only sparse OpNav, so the natural SRP-perturbed orbit evolution becomes
    /// observable and the filter can calibrate C_R + the stochastic acceleration.
    fn is_quiescent(self) -> bool {
        matches!(self, Phase::RadioScience)
    }

    /// Desired pointing mode during steady-state science operations.
    fn pointing_mode(self) -> PointingMode {
        // Nadir for ALL phases — the OpNav camera (+x body) must keep Bennu in the
        // FOV to produce bearing measurements. VelocityAligned points the boresight
        // ~90° off Bennu in a circular orbit, blinding OpNav and starving the filter
        // of the bearing observable (LIDAR alone cannot constrain along/cross-track).
        // This matches OSIRIS-REx Orbit-B, whose science orbit is nadir-pointed with
        // the instrument deck toward Bennu while collecting ~12 OpNav images/day.
        let _ = self;
        PointingMode::Nadir
    }

    fn next(self) -> Option<Phase> {
        match self {
            Phase::Capture      => Some(Phase::Survey),
            // Calibrate SRP/C_R in a quiescent arc as soon as the first stable
            // orbit is established, so every later phase uses the corrected force model.
            Phase::Survey       => Some(Phase::RadioScience),
            Phase::RadioScience => Some(Phase::CloseOrbit),
            Phase::CloseOrbit   => Some(Phase::Flyover),
            Phase::Flyover      => Some(Phase::ScienceHold),
            Phase::ScienceHold  => None,
        }
    }
}

const MAX_TRANS_FORCE_N:  f64 = 3.0;
const PERI_DETECT_FRAC:   f64 = 1.08;
const INIT_COAST_S:       f64 = 300.0;

/// Range below which surface landmarks resolve in the NavCam and become the
/// primary navigation observable [m].  Above this, fall back to disk center-finding.
const LANDMARK_MAX_RANGE_M: f64 = 5_000.0;

/// Log one landmark-tracking frame every this many measurement steps (keeps the
/// visualisation CSVs to a few hundred frames regardless of dt).
const LM_LOG_EVERY: usize = 8;

/// OpNav fix cadence during the quiescent radio-science arc [s].  Deliberately
/// sparse: at 6 h the SRP-driven drift between fixes (~tens of m) rises well above
/// the measurement noise, making C_R and the stochastic acceleration observable —
/// the natural-orbit-evolution principle behind the real Orbit-B radio-science arc.
const QUIESCENT_MEAS_S: f64 = 6.0 * 3_600.0;

/// C_R 1-σ a priori re-opened when the calibration arc begins, so the filter is
/// free to move the reflectivity from its nominal toward the value that matches
/// the true (panel) SRP.  Without this the cannonball C_R is pinned too tightly.
const QUIESCENT_SIGMA_CR: f64 = 0.8;

/// Minimum P_rr per axis [m²] enforced during a Hohmann transfer coast.
/// Keeps K_lidar ≈ 0.99 and K_bearing ≈ 1 so every measurement corrects the
/// growing orbital-phase divergence instead of being discarded by a near-zero K.
const HOHMANN_P_MIN: f64 = 2_500.0; // 50 m sigma per axis

/// Minimum P_vv per axis [m²/s²] during coast + ScienceHold.
/// Bearing-only observations correct velocity through cross-covariance P_rv built
/// each dt_meas window: K_v ≈ dt_meas·P_vv·r/P_rr. With HOHMANN_P_MIN=2500 and
/// r=900 m, HOHMANN_V_MIN=9e-4 gives a velocity correction time-constant of ~6 h,
/// reducing a 15 mm/s initial error to <0.2 mm/s within half of ScienceHold.
const HOHMANN_V_MIN: f64 = 9e-4; // 30 mm/s sigma per axis

// ── CLI arg parsing ───────────────────────────────────────────────────────────

fn parse_args() -> (f64, f64) {
    let args: Vec<String> = std::env::args().collect();
    let mut dt_truth = PROX_DT_TRUTH_S;
    let mut dt_meas  = PROX_DT_MEAS_S;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dt"      => { i += 1; dt_truth = args[i].parse().expect("--dt must be a number"); }
            "--meas-dt" => { i += 1; dt_meas  = args[i].parse().expect("--meas-dt must be a number"); }
            "--help" | "-h" => {
                println!("proximity_mission [--dt <s>] [--meas-dt <s>]");
                println!("  --dt      truth integration step  (default {PROX_DT_TRUTH_S} s)");
                println!("  --meas-dt OpNav/log interval      (default {PROX_DT_MEAS_S} s)");
                println!("  Fast run example: --dt 60 --meas-dt 600");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    (dt_truth, dt_meas)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let (dt_truth, dt_meas) = parse_args();
    std::fs::create_dir_all("out/mission").unwrap();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  BENNU PROXIMITY MISSION  —  5-phase scenario           ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("  dt_truth={dt_truth} s  dt_meas={dt_meas} s");

    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let (mut truth, mut ekf, sigma_r0) = init(&mut rng);
    let prop = Propagator::new(dt_truth);

    println!("  Init range   : {:.2} km", truth.r.norm() / 1e3);
    println!("  Init speed   : {:.4} m/s", truth.v.norm());
    println!("  σ_r0         : {:.1} m", sigma_r0);

    // ── Ground OD EKF (persists across all phases) ────────────────────────────
    let r_bennu_init = bennu_heliocentric_pos(truth.t);
    let v_bennu_init = (bennu_heliocentric_pos(truth.t + 10.0) - r_bennu_init) / 10.0;
    let mut ground_od = dsn::GroundOdEkf::new(
        r_bennu_init + truth.r,
        v_bennu_init + truth.v,
        C_R_NOMINAL,
        5_000.0, 0.05, 0.1,
        SC_AREA_CANNONBALL / SC_MASS,
    );
    let od_every     = (dsn::OD_STEP_S    / dt_truth).round() as usize;
    let uplink_every = (DSN_UPDATE_INTERVAL_S / dsn::OD_STEP_S).round() as usize;
    let mut od_step     = 0usize;
    let mut od_meas_cnt = 0usize;
    let mut n_dsn       = 0usize;

    // ── SRP panel model + Markov-residual log ─────────────────────────────────
    let plates = spacecraft_plates();
    let mut srp_log: Vec<SrpRow> = Vec::new();

    // ── Landmark catalogue (body-fixed surface features) ──────────────────────
    let lm_cat = landmark::catalogue();
    let mut lm_frame_log: Vec<LmFrameRow> = Vec::new();
    let mut lm_obs_log:   Vec<LmObsRow>   = Vec::new();
    let mut lm_frame_id    = 0usize;
    let mut lm_meas_count  = 0usize;  // # measurement steps that used landmarks
    let mut lm_inrange_count = 0usize; // # in-range steps (for frame downsampling)
    let mut n_lm_updates   = 0usize;  // total landmark LOS updates applied

    // ── Logging ───────────────────────────────────────────────────────────────
    let mut nav_log: Vec<NavRow>     = Vec::new();
    let mut att_log: Vec<AttRow>     = Vec::new();
    let mut man_log: Vec<ManRow>     = Vec::new();
    let mut dsn_log: Vec<DsnRow>     = Vec::new();

    let mut dv_total   = 0.0_f64;
    let mut n_opnav    = 0usize;
    let mut n_missed   = 0usize;

    let meas_every = (dt_meas          / dt_truth).round().max(1.0) as usize;
    let sk_every   = (SK_INTERVAL_S    / dt_truth).round() as usize;
    let quiescent_every = (QUIESCENT_MEAS_S / dt_truth).round().max(1.0) as usize;

    // ── Phase machine ─────────────────────────────────────────────────────────
    let mut phase           = Phase::Capture;
    let mut phase_step      = 0usize;           // steps within current phase
    let mut phase_t0        = truth.t;
    let mut global_step     = 0usize;

    // Capture-phase internals
    let mut cap_insertion_done    = false;
    let mut cap_burn1_done        = false;
    let mut cap_burn2_done        = false;
    let mut cap_plane_change_done = false;
    let mut cap_prev_range        = truth.r.norm();
    let mut cap_init_coast        = false;
    let mut last_sk_step       = 0usize;

    // Hohmann transit P floor — steps remaining where P_rr is clamped to HOHMANN_P_MIN.
    // Set at burn1; expires after 1.2 × Hohmann half-period so measurements cannot
    // collapse P back to overconfident values before the filter has tracked the orbit.
    let mut hohmann_coast_steps: usize = 0;

    // ── Main loop ─────────────────────────────────────────────────────────────
    loop {
        let range   = truth.r.norm();
        let r_hat   = truth.r / range;
        let v_radial = truth.v.dot(&r_hat);

        // ── Per-phase guidance ────────────────────────────────────────────────
        let (thrust_hill, pointing) = guidance(
            phase, phase_step, global_step, &truth, range, r_hat, v_radial,
            &mut cap_insertion_done, &mut cap_burn1_done, &mut cap_burn2_done,
            &mut cap_plane_change_done,
            &mut cap_prev_range, &mut cap_init_coast,
            &mut last_sk_step, sk_every,
            &mut dv_total, &mut man_log, dt_truth,
        );

        // ── Propagate truth ───────────────────────────────────────────────────
        let (new_truth, rcs) = prop.step_with_force(&truth, thrust_hill, pointing);
        truth = new_truth;

        // ── EKF predict ───────────────────────────────────────────────────────
        // During the quiescent arc, trust dynamics + free C_R so the SRP residual
        // is forced into the reflectivity estimate (calibration).
        ekf = if phase.is_quiescent() {
            predict_calibration(&ekf, dt_truth)
        } else {
            predict(&ekf, dt_truth)
        };
        if let Some(f) = thrust_hill {
            let dv = f * (dt_truth / SC_MASS);
            ekf.x[3] += dv[0]; ekf.x[4] += dv[1]; ekf.x[5] += dv[2];
            // Hohmann burn1 (outbound, cap_burn2_done still false): inflate P once for
            // the first measurement step and start the coast-window P floor timer.
            // Without the floor, LIDAR + bearing at close range collapse P to < 1 m²
            // within 2 minutes, driving K to ~0 for the entire 7 h Hohmann transit
            // while the true along-track error grows to ~100–200 m (orbital phase drift).
            if dv.norm() > 5e-3 && !cap_burn2_done {
                let current_sigma = (ekf.p[(0,0)] + ekf.p[(1,1)] + ekf.p[(2,2)]).max(0.0).sqrt();
                if current_sigma < 100.0 {
                    ekf.p[(0, 0)] += 250_000.0;
                    ekf.p[(1, 1)] += 250_000.0;
                    ekf.p[(2, 2)] += 250_000.0;
                    for i in 0..3 { for j in 3..6 {
                        ekf.p[(i, j)] = 0.0; ekf.p[(j, i)] = 0.0;
                    }}
                    if phase == Phase::ScienceHold {
                        // ScienceHold entry only: inflate P_vv so bearing cross-covariance
                        // can correct the ~15 mm/s velocity error from Flyover's tight filter,
                        // and extend the coast timer to cover the full 48 h hold.
                        // P_vv is NOT inflated for earlier burns to avoid accumulating
                        // velocity errors from systematic bearing bias at close range.
                        ekf.p[(3, 3)] += 0.0025; // 50 mm/s sigma initial kick
                        ekf.p[(4, 4)] += 0.0025;
                        ekf.p[(5, 5)] += 0.0025;
                        let coast = (55.0_f64 * 3_600.0 / dt_truth).ceil() as usize;
                        hohmann_coast_steps = hohmann_coast_steps.max(coast);
                    } else {
                        // Earlier burns: original 1.2 × Hohmann half-period timer only.
                        let r1 = ekf.r().norm();
                        let r_t = phase.target_r_m();
                        let a_t = 0.5 * (r1 + r_t);
                        let t_half = std::f64::consts::PI * (a_t * a_t * a_t / MU_BENNU).sqrt();
                        let coast = ((t_half * 1.2) / dt_truth).ceil() as usize;
                        hohmann_coast_steps = hohmann_coast_steps.max(coast);
                    }
                }
            }
        }

        // ── Independent LIDAR every 30 s ─────────────────────────────────────
        // Fires 4× per OpNav interval to prevent P_rr from collapsing to sub-meter
        // sigma during Flyover, which would cause overconfidence at the final burn.
        // Suppressed during the quiescent arc — range fixes would mask the SRP drift.
        if global_step % 3 == 0 && !phase.is_quiescent() {
            if let Some(alt) = opnav::lidar_measure(&truth.r, &mut rng) {
                ekf = update_lidar(&ekf, alt);
            }
        }

        // ── OpNav update: landmark OpNav (primary) or center-finding (fallback) ─
        if global_step % meas_every == 0 {
            n_opnav += 1;
            let q_st = star_tracker::measure(&truth.q, &mut rng);

            // During the quiescent arc, apply position fixes only sparsely so the
            // SRP-driven drift between them rises above the noise (→ C_R observable).
            let do_update = !phase.is_quiescent() || global_step % quiescent_every == 0;

            // Try landmark OpNav first when Bennu's surface resolves (close range).
            let obs = if range < LANDMARK_MAX_RANGE_M {
                landmark::observe(&lm_cat, &truth.r, &truth.q, &q_st, truth.t, &mut rng)
            } else {
                Vec::new()
            };
            let n_tracked = obs.iter().filter(|o| o.state == LmState::Tracked).count();

            if do_update {
                if n_tracked >= 2 {
                    // Landmark OpNav: each tracked feature is a LOS to a known body-fixed
                    // point → triangulates full 3-D position (incl. range) without LIDAR.
                    for o in obs.iter().filter(|o| o.state == LmState::Tracked) {
                        ekf = update_landmark(&ekf, &o.p_inertial, &o.los_inertial, o.sigma);
                        n_lm_updates += 1;
                    }
                    lm_meas_count += 1;
                } else if let Some(m) = opnav::measure(&truth.r, &truth.q, &q_st, truth.t, &mut rng) {
                    // Fallback: disk center-finding bearing + angular size + LIDAR range.
                    ekf = update_bearing(&ekf, &m.los_inertial, m.sigma_bearing);
                    ekf = update_angular_size(&ekf, m.angular_size);
                    if let Some(alt) = opnav::lidar_measure(&truth.r, &mut rng) {
                        ekf = update_lidar(&ekf, alt);
                    }
                } else {
                    n_missed += 1;
                }
            }

            // ── Landmark-tracking visualisation log (downsampled) ─────────────
            // Count every in-range measurement step; log one frame per LM_LOG_EVERY
            // so the animation captures Dark/FarSide/OutOfFov states too, not just
            // steps where the filter got a fix.
            if !obs.is_empty() {
                if lm_inrange_count % LM_LOG_EVERY == 0 {
                let sun_hat = (-bennu_heliocentric_pos(truth.t)).normalize();
                let nav_err = (truth.r - ekf.r()).norm();
                lm_frame_log.push(LmFrameRow {
                    frame: lm_frame_id, t: truth.t, phase: phase.name(),
                    scx: truth.r[0], scy: truth.r[1], scz: truth.r[2],
                    bx: truth.boresight[0], by: truth.boresight[1], bz: truth.boresight[2],
                    sunx: sun_hat[0], suny: sun_hat[1], sunz: sun_hat[2],
                    n_tracked, nav_err_m: nav_err,
                });
                for o in &obs {
                    let st = match o.state {
                        LmState::FarSide         => 0,
                        LmState::Dark            => 1,
                        LmState::VisibleOutOfFov => 2,
                        LmState::Tracked         => 3,
                    };
                    lm_obs_log.push(LmObsRow {
                        frame: lm_frame_id, id: o.id,
                        x: o.p_inertial[0], y: o.p_inertial[1], z: o.p_inertial[2],
                        state: st, cam_y: o.cam_yz.0, cam_z: o.cam_yz.1,
                    });
                }
                lm_frame_id += 1;
                }
                lm_inrange_count += 1;
            }

            nav_log.push(NavRow {
                t: truth.t, phase: phase.name(),
                tx: truth.r[0], ty: truth.r[1], tz: truth.r[2],
                tvx: truth.v[0], tvy: truth.v[1], tvz: truth.v[2],
                range_km: range / 1e3,
                ex: ekf.x[0], ey: ekf.x[1], ez: ekf.x[2],
                evx: ekf.x[3], evy: ekf.x[4], evz: ekf.x[5],
                sigma_r: ekf_sigma_r(&ekf),
            });

            // ── SRP / Markov stochastic-acceleration log (downsampled) ────────
            // Truth uses the attitude-dependent panel SRP; the filter models a
            // cannonball (scaled by its C_R estimate).  Their difference is the
            // residual the Gauss-Markov state estimates.
            if n_opnav % LM_LOG_EVERY == 0 {
                let bp = bennu_heliocentric_pos(truth.t);
                let a_truth  = srp_accel_panels(&plates, &truth.q, &bp, SC_MASS);
                let a_filter = srp_accel_cannonball(&bp, ekf.c_r(), SC_AREA_CANNONBALL, SC_MASS);
                let a_resid  = a_truth - a_filter;            // what the Markov should track
                let a_est    = ekf.a_stoch();                 // what the filter estimates
                let sun_hat  = (-bp).normalize();             // spacecraft → Sun (Hill)
                srp_log.push(SrpRow {
                    t: truth.t, phase: phase.name(),
                    qw: truth.q[0], qx: truth.q[1], qy: truth.q[2], qz: truth.q[3],
                    sunx: sun_hat[0], suny: sun_hat[1], sunz: sun_hat[2],
                    atx: a_truth[0], aty: a_truth[1], atz: a_truth[2],
                    rx: a_resid[0], ry: a_resid[1], rz: a_resid[2],
                    ex: a_est[0], ey: a_est[1], ez: a_est[2],
                    cr: ekf.c_r(),
                });
            }
        }

        // ── Attitude log (every step) ─────────────────────────────────────────
        let pointing_err = pointing_error_rad(&truth, pointing);
        att_log.push(AttRow {
            t: truth.t, phase: phase.name(),
            omega_norm: truth.omega.norm(),
            pointing_err_mrad: pointing_err * 1e3,
            pointing_mode: match pointing {
                PointingMode::Nadir          => "Nadir",
                PointingMode::VelocityAligned => "VelocityAligned",
            },
            w1: truth.wheel_speeds[0], w2: truth.wheel_speeds[1],
            w3: truth.wheel_speeds[2], w4: truth.wheel_speeds[3],
            h_w_norm: truth.wheel_momentum().norm(),
            wheel_sat_pct: truth.wheel_saturation_fraction() * 100.0,
            rw_tau_x: rcs.wheel_torque[0],
            rw_tau_y: rcs.wheel_torque[1],
            rw_tau_z: rcs.wheel_torque[2],
            tau_pd_x:  rcs.tau_pd[0],  tau_pd_y:  rcs.tau_pd[1],  tau_pd_z:  rcs.tau_pd[2],
            tau_srp_x: rcs.tau_srp[0], tau_srp_y: rcs.tau_srp[1], tau_srp_z: rcs.tau_srp[2],
            tau_gg_x:  rcs.tau_gg[0],  tau_gg_y:  rcs.tau_gg[1],  tau_gg_z:  rcs.tau_gg[2],
            desat: rcs.desat_active,
        });

        // ── Ground OD + DSN uplink ────────────────────────────────────────────
        if global_step > 0 && global_step % od_every == 0 {
            od_step += 1; od_meas_cnt += 1;
            let r_bennu = bennu_heliocentric_pos(truth.t);
            let v_bennu = (bennu_heliocentric_pos(truth.t + 10.0) - r_bennu) / 10.0;
            let (r_earth, v_earth) = dsn::earth_rv(truth.t);
            let r_sc = r_bennu + truth.r;
            let v_sc = v_bennu + truth.v;

            ground_od.predict(dsn::OD_STEP_S);
            if od_meas_cnt % dsn::OD_MEAS_HZ == 0 {
                let do_ddor = od_meas_cnt % dsn::OD_DDOR_HZ == 0;
                ground_od.update_from_truth(&r_sc, &v_sc, &r_earth, &v_earth, do_ddor, &mut rng);
            }
            // During ScienceHold, OpNav/LIDAR at <1 km dominates DSN (σ_r_dsn≈5000m,
            // σ_v_dsn≈50 mm/s >> true errors). Applying the uplink here would corrupt
            // the velocity state whose P_vv is intentionally inflated for bearing-driven
            // velocity correction.  Also skip during the quiescent arc, where the coarse
            // DSN state fix would mask the SRP drift the filter is calibrating against.
            if od_step % uplink_every == 0
                && phase != Phase::ScienceHold && !phase.is_quiescent() {
                let uplink = ground_od.uplink();
                let r_pre  = ekf.r();
                ekf = update_dsn(&ekf, &uplink.r_helio_m, &uplink.v_helio_mps);
                let innov  = (uplink.r_helio_m - (r_bennu + r_pre)).norm();
                n_dsn += 1;
                dsn_log.push(DsnRow { t: truth.t, innov_r_m: innov, sigma_r_m: ekf_sigma_r(&ekf) });
            }
        }

        // ── Hohmann transit + ScienceHold P floor ────────────────────────────
        // Clamp P_rr ≥ HOHMANN_P_MIN throughout any active coast window.
        // For ScienceHold also clamp P_vv ≥ HOHMANN_V_MIN so bearing cross-
        // covariance can correct the accumulated velocity error over the 48 h hold.
        // P_vv is NOT floored during earlier phases to avoid amplifying systematic
        // bearing bias from Bennu's close-range disk overflow into velocity errors.
        if hohmann_coast_steps > 0 {
            for i in 0..3 {
                ekf.p[(i, i)] = ekf.p[(i, i)].max(HOHMANN_P_MIN);
                if phase == Phase::ScienceHold {
                    ekf.p[(i+3, i+3)] = ekf.p[(i+3, i+3)].max(HOHMANN_V_MIN);
                }
            }
            hohmann_coast_steps -= 1;
        }

        phase_step  += 1;
        global_step += 1;
        cap_prev_range = range;

        // ── Phase transition ──────────────────────────────────────────────────
        if phase_elapsed(phase, phase_step, cap_insertion_done, cap_burn2_done, cap_init_coast, cap_plane_change_done, dt_truth) {
            let elapsed = (truth.t - phase_t0) / 3600.0;
            let (sma, ecc) = orbital_elements(&truth.r, &truth.v);
            println!("  ✓ {:12}  t={:.1} h  r={:.2} km  SMA={:.2} km  e={:.4}",
                     phase.name(), elapsed, range/1e3, sma/1e3, ecc);

            match phase.next() {
                Some(next) => {
                    phase      = next;
                    phase_step = 0;
                    phase_t0   = truth.t;
                    cap_insertion_done    = true; // previous phases already done
                    cap_burn1_done        = false;
                    cap_burn2_done        = false;
                    cap_plane_change_done = false;
                    cap_prev_range        = range;
                    last_sk_step       = global_step;
                    // Re-open the C_R prior at the start of the calibration arc so
                    // the filter can move reflectivity toward the true (panel) SRP.
                    if next == Phase::RadioScience {
                        ekf.p[(6, 6)] = ekf.p[(6, 6)].max(QUIESCENT_SIGMA_CR * QUIESCENT_SIGMA_CR);
                        println!("    ↳ radio-science arc: C_R prior re-opened (σ={QUIESCENT_SIGMA_CR}), \
                                  OpNav every {:.0} h, no maneuvers", QUIESCENT_MEAS_S / 3600.0);
                    }
                    println!("  → Starting {:12}  (target {:.1} km for {:.1} days)",
                             next.name(), next.target_r_m()/1e3, next.duration_s()/86400.0);
                }
                None => break,
            }
        }
    }

    // ── Final report ─────────────────────────────────────────────────────────
    let (sma, ecc) = orbital_elements(&truth.r, &truth.v);
    println!("\n── Mission complete ──────────────────────────────────────");
    println!("  Final orbit    : SMA={:.2} km  e={:.4}", sma/1e3, ecc);
    println!("  Nav error r    : {:.2} m", (truth.r - ekf.r()).norm());
    println!("  Nav error v    : {:.4} m/s", (truth.v - ekf.v()).norm());
    println!("  Total ΔV       : {:.3} m/s  ({} burns)", dv_total, man_log.len());
    println!("  OpNav coverage : {}/{} ({:.1}%)",
             n_opnav - n_missed, n_opnav,
             100.0 * (n_opnav - n_missed) as f64 / n_opnav.max(1) as f64);
    println!("  Landmark fixes : {lm_meas_count} steps, {n_lm_updates} LOS updates");
    println!("  DSN passes     : {n_dsn}");
    println!("  Max wheel sat  : {:.1}%", att_log.iter().map(|r| r.wheel_sat_pct).fold(0.0_f64, f64::max));
    println!("  Desat events   : {}", att_log.iter().filter(|r| r.desat).count());

    save_nav("out/mission/nav.csv", &nav_log);
    save_att("out/mission/attitude.csv", &att_log);
    save_man("out/mission/maneuvers.csv", &man_log);
    save_dsn("out/mission/dsn_updates.csv", &dsn_log);
    save_lm_frames("out/mission/landmark_frames.csv", &lm_frame_log);
    save_lm_obs("out/mission/landmark_obs.csv", &lm_obs_log);
    save_srp("out/mission/srp.csv", &srp_log);
    println!("\nOutputs in out/mission/");
    println!("Plot: python plot/plot_mission.py");
    println!("Plot: python plot/plot_landmarks.py   (landmark tracking animation)");
    println!("Plot: python plot/plot_srp.py         (panel SRP + Markov animation)");
}

// ── Initialisation ────────────────────────────────────────────────────────────

fn init(rng: &mut impl rand::Rng) -> (TruthState, EkfState, f64) {
    if let Ok(h) = ProximityHandoff::load(rng) {
        let truth = TruthState::from_handoff(h.r_truth, h.v_truth, h.t_arr);
        // Set initial P_rr to at least BENNU_EPHEM_SIGMA_M so that the first
        // OpNav/LIDAR measurement (K_ang ≈ 1) can correct the large initial
        // position error from Bennu's ephemeris uncertainty.
        let sigma_r_init = h.sigma_r0_m.max(BENNU_EPHEM_SIGMA_M);
        let ekf   = EkfState::with_initial_state(
            h.r_est, h.v_est, sigma_r_init, h.sigma_v0_mps, HANDOFF_SIGMA_CR, h.t_arr,
        );
        println!("  Init: cruise handoff  r={:.1} km  σ_r0={:.1} m  σ_r_init={:.0} m",
                 h.r_truth.norm()/1e3, h.sigma_r0_m, sigma_r_init);
        (truth, ekf, sigma_r_init)
    } else {
        // Terminator orbit: orbit normal h_hat aligned with the Sun-from-Bennu
        // direction.  The spacecraft is always at 90° to the Sun, so SRP forces
        // are symmetric over each revolution and secular node drift is suppressed.
        // This matches the OSIRIS-REx Orbit-B / Orbit-A philosophy (Scheeres 2020 §4).
        const R0_M: f64 = 3_000.0;
        let sun_vec = bennu_heliocentric_pos(0.0);
        let sun_hat = -sun_vec / sun_vec.norm();   // unit: Bennu → Sun
        // Choose initial position perpendicular to sun_hat (avoid z-collinearity)
        let ref_up = if sun_hat.z.abs() < 0.9 {
            Vector3::new(0.0, 0.0, 1.0)
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let r_hat  = sun_hat.cross(&ref_up).normalize();
        let v_hat  = sun_hat.cross(&r_hat);          // h_hat × r_hat, already unit
        let r0     = r_hat * R0_M;
        let v0     = v_hat * (MU_BENNU / R0_M).sqrt();
        // Start at Nadir attitude so OpNav can see Bennu immediately (no settling delay).
        let q_nadir = desired_quaternion(PointingMode::Nadir, &r0, &v0);
        let truth = TruthState::from_state(r0, v0, q_nadir, Vector3::zeros(), 0.0);
        let ekf  = EkfState::with_initial_state(r0, v0, 100.0, 0.005, 0.05, 0.0);
        let incl_deg = sun_hat.z.acos().to_degrees();
        println!("  Init: terminator orbit  h ∥ Sun  ecliptic-incl={:.1}°  r={:.0} m",
                 incl_deg, R0_M);
        (truth, ekf, 100.0)
    }
}

// ── Per-phase guidance (returns Hill-frame thrust + pointing mode) ────────────

#[allow(clippy::too_many_arguments)]
fn guidance(
    phase: Phase,
    phase_step: usize,
    global_step: usize,
    truth: &TruthState,
    range: f64,
    r_hat: Vector3<f64>,
    v_radial: f64,
    cap_insertion_done: &mut bool,
    cap_burn1_done: &mut bool,
    cap_burn2_done: &mut bool,
    cap_plane_change_done: &mut bool,
    cap_prev_range: &mut f64,
    cap_init_coast: &mut bool,
    last_sk_step: &mut usize,
    sk_every: usize,
    dv_total: &mut f64,
    man_log: &mut Vec<ManRow>,
    dt: f64,
) -> (Option<Vector3<f64>>, PointingMode) {
    let mode = phase.pointing_mode();

    let thrust = match phase {
        Phase::Capture => {
            capture_guidance(
                truth, range, r_hat, v_radial,
                cap_insertion_done, cap_burn1_done, cap_burn2_done,
                cap_plane_change_done,
                cap_init_coast, phase_step,
                dv_total, man_log, dt,
            )
        }

        Phase::Survey | Phase::CloseOrbit | Phase::ScienceHold => {
            let r_target = phase.target_r_m();
            if phase_step == 0 {
                let res = hohmann_burn1(truth, range, r_hat, v_radial, r_target, dv_total, man_log, dt);
                // Trivial burn1 (already at orbit) → skip transfer coast, go straight to SK
                if res.is_none() { *cap_burn2_done = true; }
                res
            } else if !*cap_burn2_done && needs_burn2(truth, range, *cap_prev_range, r_target) {
                *cap_burn2_done = true;
                hohmann_burn2(truth, range, r_hat, v_radial, r_target, dv_total, man_log, dt)
            } else if !*cap_burn2_done {
                None  // coasting on Hohmann transfer ellipse; SK would perturb the orbit
            } else {
                station_keep(truth, range, r_hat, v_radial, r_target,
                             global_step, last_sk_step, sk_every, dv_total, man_log, dt)
            }
        }

        Phase::Flyover => {
            let r_target = phase.target_r_m();
            if phase_step == 0 {
                hohmann_burn1(truth, range, r_hat, v_radial, r_target, dv_total, man_log, dt)
            } else if !*cap_burn2_done && needs_burn2(truth, range, *cap_prev_range, r_target) {
                *cap_burn2_done = true;
                hohmann_burn2(truth, range, r_hat, v_radial, r_target, dv_total, man_log, dt)
            } else {
                None  // coast through flyovers
            }
        }

        // Quiescent calibration arc: pure coast, no burns or station-keeping, so
        // the natural SRP-perturbed motion is what the filter calibrates against.
        Phase::RadioScience => None,
    };

    (thrust, mode)
}

// ── Capture guidance ──────────────────────────────────────────────────────────

fn capture_guidance(
    truth: &TruthState,
    range: f64,
    r_hat: Vector3<f64>,
    v_radial: f64,
    insertion_done: &mut bool,
    burn1_done: &mut bool,
    burn2_done: &mut bool,
    plane_change_done: &mut bool,
    init_coast: &mut bool,
    phase_step: usize,
    dv_total: &mut f64,
    man_log: &mut Vec<ManRow>,
    dt: f64,
) -> Option<Vector3<f64>> {
    let eps = orbital_energy(&truth.r, &truth.v);

    if !*insertion_done {
        if eps < 0.0 {
            *insertion_done = true;
            println!("  Orbit captured  @ t={:.1} h  r={:.1} km  ε={:.4} J/kg",
                     truth.t / 3600.0, range / 1e3, eps);
            return None;
        }
        let v_hat = if truth.v.norm() > 1e-10 { truth.v / truth.v.norm() } else { Vector3::zeros() };
        let force = -v_hat * MAX_TRANS_FORCE_N;
        *dv_total += (force * (dt / SC_MASS)).norm();
        return Some(force);
    }

    if !*init_coast {
        if phase_step > (INIT_COAST_S / dt) as usize {
            *init_coast = true;
        }
        return None;
    }

    let r_target = Phase::Capture.target_r_m();
    if !*burn1_done {
        *burn1_done = true;
        let result = hohmann_burn1(truth, range, r_hat, v_radial, r_target, dv_total, man_log, dt);
        // If burn1 was a no-op (already at target orbit), skip burn2 too
        if result.is_none() {
            let v_circ = (MU_BENNU / range).sqrt();
            let v_tang = (truth.v - v_radial * r_hat).norm();
            if (v_tang - v_circ).abs() < 0.05 * v_circ {
                *burn2_done = true;
            }
        }
        return result;
    }
    if !*burn2_done && needs_burn2_simple(range, r_target) && v_radial < 0.0 {
        *burn2_done = true;
        return hohmann_burn2(truth, range, r_hat, v_radial, r_target, dv_total, man_log, dt);
    }

    // Plane change: rotate orbit into terminator plane (h_hat ∥ Bennu→Sun).
    // Fires each step after circularization until incl_err < 1°.
    // ΔV = 2·v_circ·sin(Δi/2) ≈ 29 mm/s for a 42° error at 3 km.
    if *burn2_done && !*plane_change_done {
        let bp      = bennu_heliocentric_pos(truth.t);
        let sun_hat = (-bp).normalize();
        let h       = truth.r.cross(&truth.v);
        let h_cur_n = if h.norm() > 1e-12 { h.normalize() }
                      else { *plane_change_done = true; return None; };
        // Choose the terminator polarity (±sun_hat) closest to current h to minimise ΔV.
        let h_des = if h_cur_n.dot(&sun_hat) >= 0.0 { sun_hat } else { -sun_hat };
        let incl_err_deg = h_cur_n.dot(&h_des).clamp(-1.0, 1.0).acos().to_degrees();

        if incl_err_deg < 1.0 {
            *plane_change_done = true;
            println!("  Plane change  done  @ r={:.2} km  final Δi={:.2}°",
                     range / 1e3, incl_err_deg);
            return None;
        }

        let v_tang     = truth.v - v_radial * r_hat;
        let v_tang_mag = v_tang.norm();
        if v_tang_mag < 1e-6 { return None; }

        let t_hat_old = v_tang / v_tang_mag;
        let h_x_r     = h_des.cross(&r_hat);
        if h_x_r.norm() < 1e-6 { *plane_change_done = true; return None; }
        let t_hat_new = {
            let t = h_x_r.normalize();
            if t.dot(&t_hat_old) >= 0.0 { t } else { -t }
        };

        // ΔV = v_tang_mag · (t̂_new − t̂_old)  — keeps speed, rotates direction.
        let dv_vec = v_tang_mag * (t_hat_new - t_hat_old);
        let dv_mag = dv_vec.norm();
        if dv_mag < 1e-6 { *plane_change_done = true; return None; }

        let force = (dv_vec / dv_mag) * (SC_MASS * dv_mag / dt).min(MAX_TRANS_FORCE_N);
        let dv    = force * (dt / SC_MASS);
        *dv_total += dv.norm();
        man_log.push(ManRow {
            t: truth.t, dv_x: dv[0], dv_y: dv[1], dv_z: dv[2],
            dv_mag: dv.norm(), label: "PlaneChange",
        });
        return Some(force);
    }

    None
}

// ── Hohmann transfer helpers ──────────────────────────────────────────────────

fn hohmann_burn1(
    truth: &TruthState,
    range: f64,
    r_hat: Vector3<f64>,
    v_radial: f64,
    r_target: f64,
    dv_total: &mut f64,
    man_log: &mut Vec<ManRow>,
    dt: f64,
) -> Option<Vector3<f64>> {
    let v_tang = truth.v - v_radial * r_hat;
    let v_tang_mag = v_tang.norm();
    // Velocity at apoapsis of the Hohmann transfer ellipse (= velocity we want at current position)
    let v_transfer = ((MU_BENNU * 2.0 * r_target) / (range * (range + r_target))).sqrt();
    let dv_needed = v_tang_mag - v_transfer;
    if dv_needed.abs() < 1e-6 { return None; }

    // Tangential direction: use velocity if available; otherwise derive from geometry.
    // Near a radial orbit (e.g. just after capture insertion), v_tang ≈ 0 so we need
    // to choose a tangential direction to give the spacecraft its first tangential kick.
    let tang_hat = if v_tang_mag > 1e-8 {
        v_tang / v_tang_mag
    } else {
        // Choose a direction perpendicular to r̂ in the equatorial plane
        let up = Vector3::new(0.0, 0.0, 1.0);
        let t  = up.cross(&r_hat);
        if t.norm() > 1e-10 { t / t.norm() }
        else {
            let alt = Vector3::new(0.0, 1.0, 0.0);
            let t2  = alt.cross(&r_hat);
            if t2.norm() > 1e-10 { t2 / t2.norm() } else { return None; }
        }
    };

    let sign  = if dv_needed > 0.0 { -1.0 } else { 1.0 };
    let force = tang_hat * sign * (SC_MASS * dv_needed.abs() / dt).min(MAX_TRANS_FORCE_N);
    let dv    = force * (dt / SC_MASS);
    *dv_total += dv.norm();
    man_log.push(ManRow { t: truth.t, dv_x: dv[0], dv_y: dv[1], dv_z: dv[2], dv_mag: dv.norm(), label: "Burn1" });
    Some(force)
}

fn hohmann_burn2(
    truth: &TruthState,
    range: f64,
    r_hat: Vector3<f64>,
    v_radial: f64,
    _r_target: f64,
    dv_total: &mut f64,
    man_log: &mut Vec<ManRow>,
    dt: f64,
) -> Option<Vector3<f64>> {
    let v_circ    = (MU_BENNU / range).sqrt();
    let v_tang    = truth.v - v_radial * r_hat;
    let v_tang_mag = v_tang.norm();
    let tang_hat  = if v_tang_mag > 1e-10 { v_tang / v_tang_mag } else { return None; };

    // Combined impulse: circularise tangential speed AND cancel residual radial velocity.
    // This ensures e ≈ 0 even when burn2 fires slightly off the exact apse.
    let dv_tang = v_circ - v_tang_mag;   // positive → prograde
    let dv_rad  = -v_radial;             // cancel outward / inward drift

    let total_dv = (dv_tang * dv_tang + dv_rad * dv_rad).sqrt();
    if total_dv < 1e-6 { return None; }

    let dv_vec = tang_hat * dv_tang - r_hat * v_radial;
    let dv_dir = if dv_vec.norm() > 1e-10 { dv_vec / dv_vec.norm() } else { return None; };
    let force  = dv_dir * (SC_MASS * total_dv / dt).min(MAX_TRANS_FORCE_N);
    let dv     = force * (dt / SC_MASS);
    *dv_total += dv.norm();
    man_log.push(ManRow { t: truth.t, dv_x: dv[0], dv_y: dv[1], dv_z: dv[2], dv_mag: dv.norm(), label: "Burn2" });
    Some(force)
}

fn station_keep(
    truth: &TruthState,
    _range: f64,
    r_hat: Vector3<f64>,
    v_radial: f64,
    r_target: f64,
    global_step: usize,
    last_sk_step: &mut usize,
    sk_every: usize,
    dv_total: &mut f64,
    man_log: &mut Vec<ManRow>,
    dt: f64,
) -> Option<Vector3<f64>> {
    if global_step.saturating_sub(*last_sk_step) < sk_every { return None; }
    *last_sk_step = global_step;

    let eps_cur = orbital_energy(&truth.r, &truth.v);
    let eps_tgt = -MU_BENNU / (2.0 * r_target);
    let de = eps_tgt - eps_cur;
    if de.abs() <= SK_ENERGY_TOL { return None; }

    let v_tang = truth.v - v_radial * r_hat;
    let v_tang_mag = v_tang.norm().max(1e-10);
    let tang_hat = v_tang / v_tang_mag;
    let dv_needed = de / v_tang_mag;
    let sign  = if dv_needed > 0.0 { 1.0 } else { -1.0 };
    let force = tang_hat * sign * (SC_MASS * dv_needed.abs() / dt).min(MAX_TRANS_FORCE_N);
    let dv    = force * (dt / SC_MASS);
    *dv_total += dv.norm();
    man_log.push(ManRow { t: truth.t, dv_x: dv[0], dv_y: dv[1], dv_z: dv[2], dv_mag: dv.norm(), label: "SK" });
    Some(force)
}

fn needs_burn2(truth: &TruthState, range: f64, prev_range: f64, r_target: f64) -> bool {
    let eps      = orbital_energy(&truth.r, &truth.v);
    let v_radial = truth.v.dot(&(truth.r / range));
    // Fire at the TARGET apse (works for both inward Hohmann periapsis AND outward apoapsis):
    //   • range close to r_target (within 20 %)
    //   • v_radial small (near an apse, not flying through r_target mid-transit)
    ((range - r_target).abs() < r_target * 0.20 && v_radial.abs() < 0.002 && eps < 0.0)
        // Fallback: missed the sign-flip; range just reversed near target
        || (range > prev_range + 1.0 && prev_range < r_target * 1.25 && eps < 0.0)
        || (range < prev_range - 1.0 && prev_range > r_target * 0.85
            && prev_range < r_target * 1.20 && eps < 0.0)
}

fn needs_burn2_simple(range: f64, r_target: f64) -> bool {
    range <= r_target * PERI_DETECT_FRAC
}

// ── Phase end condition ───────────────────────────────────────────────────────

fn phase_elapsed(
    phase: Phase, step: usize,
    insertion_done: bool, burn2_done: bool, init_coast: bool,
    plane_change_done: bool,
    dt: f64,
) -> bool {
    let dur_steps = (phase.duration_s() / dt).round() as usize;
    match phase {
        Phase::Capture => {
            let ready   = insertion_done && init_coast && burn2_done && plane_change_done;
            let timeout = insertion_done && init_coast && step >= dur_steps;
            ready || timeout  // timeout ensures we never loop forever
        }
        _ => step >= dur_steps,
    }
}

// ── Orbital mechanics ─────────────────────────────────────────────────────────

fn orbital_energy(r: &Vector3<f64>, v: &Vector3<f64>) -> f64 {
    v.norm_squared() / 2.0 - MU_BENNU / r.norm()
}

fn orbital_elements(r: &Vector3<f64>, v: &Vector3<f64>) -> (f64, f64) {
    let eps = orbital_energy(r, v);
    let sma = if eps.abs() > 1e-20 { -MU_BENNU / (2.0 * eps) } else { f64::INFINITY };
    let h   = r.cross(v);
    let e   = (v.cross(&h) / MU_BENNU - r / r.norm()).norm();
    (sma, e)
}

// ── Attitude helpers ──────────────────────────────────────────────────────────

fn pointing_error_rad(truth: &TruthState, mode: PointingMode) -> f64 {
    use autonomous_navigation::guidance::pointing::desired_quaternion;
    let q_des = desired_quaternion(mode, &truth.r, &truth.v);
    // Error angle = 2 * acos(|q_err_w|)
    let dw = truth.q[0]*q_des[0] + truth.q[1]*q_des[1]
           + truth.q[2]*q_des[2] + truth.q[3]*q_des[3];
    2.0 * dw.abs().min(1.0).acos()
}

fn ekf_sigma_r(ekf: &EkfState) -> f64 {
    (ekf.p[(0,0)] + ekf.p[(1,1)] + ekf.p[(2,2)]).max(0.0).sqrt()
}

// ── Log rows ──────────────────────────────────────────────────────────────────

struct NavRow {
    t: f64, phase: &'static str,
    tx: f64, ty: f64, tz: f64,
    tvx: f64, tvy: f64, tvz: f64,
    range_km: f64,
    ex: f64, ey: f64, ez: f64,
    evx: f64, evy: f64, evz: f64,
    sigma_r: f64,
}

struct AttRow {
    t: f64, phase: &'static str,
    omega_norm: f64,
    pointing_err_mrad: f64,
    pointing_mode: &'static str,
    w1: f64, w2: f64, w3: f64, w4: f64,
    h_w_norm: f64,
    wheel_sat_pct: f64,
    rw_tau_x: f64, rw_tau_y: f64, rw_tau_z: f64,
    tau_pd_x: f64, tau_pd_y: f64, tau_pd_z: f64,
    tau_srp_x: f64, tau_srp_y: f64, tau_srp_z: f64,
    tau_gg_x: f64, tau_gg_y: f64, tau_gg_z: f64,
    desat: bool,
}

struct ManRow { t: f64, dv_x: f64, dv_y: f64, dv_z: f64, dv_mag: f64, label: &'static str }
struct DsnRow  { t: f64, innov_r_m: f64, sigma_r_m: f64 }

/// One landmark-tracking visualisation frame (spacecraft + camera + Sun geometry).
struct LmFrameRow {
    frame: usize, t: f64, phase: &'static str,
    scx: f64, scy: f64, scz: f64,
    bx: f64, by: f64, bz: f64,
    sunx: f64, suny: f64, sunz: f64,
    n_tracked: usize, nav_err_m: f64,
}

/// One SRP / Markov frame: attitude, Sun geometry, truth SRP accel, and the
/// truth-vs-filter residual alongside the filter's stochastic-accel estimate.
struct SrpRow {
    t: f64, phase: &'static str,
    qw: f64, qx: f64, qy: f64, qz: f64,
    sunx: f64, suny: f64, sunz: f64,
    atx: f64, aty: f64, atz: f64,   // truth SRP acceleration [m/s²]
    rx: f64, ry: f64, rz: f64,      // residual = truth − filter cannonball
    ex: f64, ey: f64, ez: f64,      // filter Gauss-Markov estimate
    cr: f64,                        // filter C_R estimate
}

/// One landmark's state within a frame (inertial position + classification).
struct LmObsRow {
    frame: usize, id: usize,
    x: f64, y: f64, z: f64,
    state: u8,        // 0 FarSide, 1 Dark, 2 VisibleOutOfFov, 3 Tracked
    cam_y: f64, cam_z: f64,
}

// ── CSV writers ───────────────────────────────────────────────────────────────

fn save_nav(path: &str, rows: &[NavRow]) {
    use std::fmt::Write as W;
    let mut out = String::new();
    writeln!(out, "time_s,phase,tx_m,ty_m,tz_m,tvx,tvy,tvz,range_km,\
                   ex_m,ey_m,ez_m,evx,evy,evz,sigma_r_m").unwrap();
    for r in rows {
        writeln!(out, "{:.1},{},{:.2},{:.2},{:.2},{:.4},{:.4},{:.4},{:.3},\
                       {:.2},{:.2},{:.2},{:.4},{:.4},{:.4},{:.2}",
            r.t, r.phase, r.tx, r.ty, r.tz, r.tvx, r.tvy, r.tvz, r.range_km,
            r.ex, r.ey, r.ez, r.evx, r.evy, r.evz, r.sigma_r).unwrap();
    }
    std::fs::write(path, &out).expect("write nav");
    println!("  Saved {path} ({} rows)", rows.len());
}

fn save_att(path: &str, rows: &[AttRow]) {
    use std::fmt::Write as W;
    let mut out = String::new();
    writeln!(out, "time_s,phase,omega_norm_rads,pointing_err_mrad,pointing_mode,\
                   w1_rads,w2_rads,w3_rads,w4_rads,\
                   h_w_norm_nms,wheel_sat_pct,\
                   rw_tau_x_nm,rw_tau_y_nm,rw_tau_z_nm,\
                   tau_pd_x_nm,tau_pd_y_nm,tau_pd_z_nm,\
                   tau_srp_x_nm,tau_srp_y_nm,tau_srp_z_nm,\
                   tau_gg_x_nm,tau_gg_y_nm,tau_gg_z_nm,desat").unwrap();
    for r in rows {
        writeln!(out,
            "{:.1},{},{:.6},{:.4},{},{:.2},{:.2},{:.2},{:.2},\
             {:.4},{:.2},{:.6},{:.6},{:.6},\
             {:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            r.t, r.phase, r.omega_norm, r.pointing_err_mrad, r.pointing_mode,
            r.w1, r.w2, r.w3, r.w4,
            r.h_w_norm, r.wheel_sat_pct,
            r.rw_tau_x, r.rw_tau_y, r.rw_tau_z,
            r.tau_pd_x,  r.tau_pd_y,  r.tau_pd_z,
            r.tau_srp_x, r.tau_srp_y, r.tau_srp_z,
            r.tau_gg_x,  r.tau_gg_y,  r.tau_gg_z,
            r.desat as u8).unwrap();
    }
    std::fs::write(path, &out).expect("write att");
    println!("  Saved {path} ({} rows)", rows.len());
}

fn save_man(path: &str, rows: &[ManRow]) {
    use std::fmt::Write as W;
    let mut out = String::new();
    writeln!(out, "time_s,dv_x_ms,dv_y_ms,dv_z_ms,dv_mag_ms,label").unwrap();
    for r in rows {
        writeln!(out, "{:.1},{:.6e},{:.6e},{:.6e},{:.6e},{}",
            r.t, r.dv_x, r.dv_y, r.dv_z, r.dv_mag, r.label).unwrap();
    }
    std::fs::write(path, &out).expect("write man");
    println!("  Saved {path} ({} maneuvers)", rows.len());
}

fn save_dsn(path: &str, rows: &[DsnRow]) {
    use std::fmt::Write as W;
    let mut out = String::new();
    writeln!(out, "time_s,innov_r_m,sigma_r_m").unwrap();
    for r in rows {
        writeln!(out, "{:.1},{:.2},{:.2}", r.t, r.innov_r_m, r.sigma_r_m).unwrap();
    }
    std::fs::write(path, &out).expect("write dsn");
    println!("  Saved {path} ({} DSN passes)", rows.len());
}

fn save_lm_frames(path: &str, rows: &[LmFrameRow]) {
    use std::fmt::Write as W;
    let mut out = String::new();
    writeln!(out, "frame,time_s,phase,scx,scy,scz,bx,by,bz,sunx,suny,sunz,n_tracked,nav_err_m").unwrap();
    for r in rows {
        writeln!(out, "{},{:.1},{},{:.2},{:.2},{:.2},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{},{:.2}",
            r.frame, r.t, r.phase, r.scx, r.scy, r.scz,
            r.bx, r.by, r.bz, r.sunx, r.suny, r.sunz, r.n_tracked, r.nav_err_m).unwrap();
    }
    std::fs::write(path, &out).expect("write lm frames");
    println!("  Saved {path} ({} frames)", rows.len());
}

fn save_lm_obs(path: &str, rows: &[LmObsRow]) {
    use std::fmt::Write as W;
    let mut out = String::new();
    writeln!(out, "frame,id,x,y,z,state,cam_y,cam_z").unwrap();
    for r in rows {
        writeln!(out, "{},{},{:.2},{:.2},{:.2},{},{:.6},{:.6}",
            r.frame, r.id, r.x, r.y, r.z, r.state, r.cam_y, r.cam_z).unwrap();
    }
    std::fs::write(path, &out).expect("write lm obs");
    println!("  Saved {path} ({} rows)", rows.len());
}

fn save_srp(path: &str, rows: &[SrpRow]) {
    use std::fmt::Write as W;
    let mut out = String::new();
    writeln!(out, "time_s,phase,qw,qx,qy,qz,sunx,suny,sunz,\
                   atx,aty,atz,rx,ry,rz,ex,ey,ez,cr").unwrap();
    for r in rows {
        writeln!(out, "{:.1},{},{:.6},{:.6},{:.6},{:.6},{:.5},{:.5},{:.5},\
                       {:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4}",
            r.t, r.phase, r.qw, r.qx, r.qy, r.qz, r.sunx, r.suny, r.sunz,
            r.atx, r.aty, r.atz, r.rx, r.ry, r.rz, r.ex, r.ey, r.ez, r.cr).unwrap();
    }
    std::fs::write(path, &out).expect("write srp");
    println!("  Saved {path} ({} frames)", rows.len());
}
