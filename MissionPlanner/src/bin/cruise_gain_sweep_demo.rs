//! Parametric validation of `cruise.rs`'s hand-picked `MomentumManagementLaw::
//! ThresholdRcs` gains (`DEFAULT_MOMENTUM_DUMP_GAIN_PER_S = 0.02`,
//! `DEFAULT_NULL_MOTION_GAIN_PER_S = 0.05`), requested after
//! `cruise_commander_demo` found 8 of 18 mode transitions never settle under
//! those defaults on an aggressive comm-pass schedule.
//!
//! Reuses the IDENTICAL real ANISE Earth->Mars scenario + comm-pass schedule
//! `cruise_commander_demo.rs` uses (same vehicle, same 8 h/1 h comm cadence,
//! same PD gains -- only the momentum-management gains vary), but calls
//! `run_cruise_leg` directly in a loop instead of going through
//! `run_cruise_streaming` (which hardcodes the two defaults) so each grid
//! point can override them.
//!
//! Two independent 1-D sweeps, not a full 2-D grid: gain swept with
//! null_motion_gain held at its default, then null_motion_gain swept with
//! gain held at its default. This isolates which knob (if either) actually
//! moves the settle rate, which a combined grid would blur together for the
//! same reason the design notes Phase 9x lessons warn against combining un-
//! isolated changes before understanding each one alone.
//!
//! Run from `MissionPlanner/`: `cargo run -p mission_planner --bin cruise_gain_sweep_demo --release`
//! Plot: `python plot/plot_cruise_gain_sweep.py`

use std::fs;

use ephemeris::{Almanac, Body, Epoch};
use hifitime::Duration;
use nalgebra::Vector3;
use trajectory_solver::{keplerian::MU_SUN_M3S2, LambertArc};

use mission_planner::config::MissionConfig;
use mission_planner::cruise::{
    build_gnc_commander, detect_mode_transitions, run_cruise_leg, sample_reference_trajectory_uniform,
};
use mission_planner::simulate::{
    build_spacecraft_properties, pd_gains_from_cfg, rcs_from_hardware, wheel_cluster_from_hardware,
};
use sim_engine::{ControlMode, CruisePointingMode, MomentumManagementLaw, ReferenceTrajectory, SixDofState};

// Identical scenario constants to cruise_commander_demo.rs.
const DEPARTURE_EPOCH: (i32, u8, u8) = (2026, 10, 28);
const DEP_OFFSET_DAYS: f64 = 3.80;
const TOF_DAYS: f64 = 291.14;

const REFERENCE_SAMPLE_DT_S: f64 = 60.0;
const CONTROL_TICK_S: f64 = 10.0;
const DEMO_DURATION_S: f64 = 3.0 * 86_400.0; // 3 days

const COMM_PASS_PERIOD_S: f64 = 8.0 * 3600.0; // one comm pass every 8 h
const COMM_PASS_DURATION_S: f64 = 3600.0; // 1 h per pass

const DEFAULT_GAIN: f64 = 0.02;
const DEFAULT_NULL_MOTION_GAIN: f64 = 0.05;
const SETTLE_THRESHOLD_DEG: f64 = 0.5;

fn find_kernel() -> Option<String> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join("kernels").join("de440s.bsp");
        if candidate.exists() {
            return Some(candidate.to_string_lossy().into_owned());
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// One grid point's summary — the same quantities `ModeTransitionReport`
/// tracks per-transition, rolled up over the whole run.
struct SweepResult {
    sweep_axis: &'static str,
    gain: f64,
    null_motion_gain: f64,
    n_transitions: usize,
    n_settled: usize,
    mean_settling_time_s: Option<f64>,
    max_wheel_momentum_nms: f64,
    max_wheel_sat_frac: f64,
    rcs_propellant_kg_used: f64,
    final_pointing_error_deg: f64,
}

fn run_one(
    sweep_axis: &'static str,
    gain: f64,
    null_motion_gain: f64,
    cfg: &MissionConfig,
    reference: &ReferenceTrajectory,
    r0: Vector3<f64>,
    v0: Vector3<f64>,
) -> SweepResult {
    let sc = build_spacecraft_properties(cfg);
    let wheel_cluster = wheel_cluster_from_hardware(cfg);
    let (rcs_thrusters, _) = rcs_from_hardware(cfg);
    let gains = pd_gains_from_cfg(cfg);
    let commander = build_gnc_commander(cfg);

    let initial = SixDofState {
        t_s: 0.0,
        r_m: r0,
        v_mps: v0,
        q: sim_engine::reference_guidance::desired_quaternion_cruise(CruisePointingMode::SunPointing, &r0, &Vector3::zeros()),
        omega_radps: Vector3::zeros(),
        wheel_speeds_radps: [0.0; 4],
        mass_kg: sc.mass_kg,
    };

    // One fixed PD gain set for every mode (this sweep varies only the
    // momentum-management gains) — `from_pd` disables activity scaling
    // and scheduling so the sweep's independent variable stays isolated.
    let controls = mission_planner::attitude_tuning::AttitudeControlSet::from_pd(
        gains, sc.inertia_diag_kgm2, sc.mass_kg, wheel_cluster.max_torque,
        sim_engine::rcs_worst_axis_authority_nm(&rcs_thrusters), CONTROL_TICK_S,
    );
    let rows = run_cruise_leg(
        initial, reference, &[], MU_SUN_M3S2, &sc, &wheel_cluster, &rcs_thrusters,
        &controls, ControlMode::WheelsPrimary,
        MomentumManagementLaw::ThresholdRcs { gain, null_motion_gain },
        CruisePointingMode::SunPointing, commander.as_ref(), &|_t_s| Vector3::zeros(),
        CONTROL_TICK_S, DEMO_DURATION_S, cfg.spacecraft.propellant_mass_kg, None,
        cfg.simulation.rtol, cfg.simulation.atol, Vector3::zeros(), &[],
        &|_, _| None,
        &mut |_row| true,
    );

    let transitions = detect_mode_transitions(&rows, SETTLE_THRESHOLD_DEG);
    let n_transitions = transitions.len();
    let settled: Vec<f64> = transitions.iter().filter_map(|t| t.settling_time_s).collect();
    let n_settled = settled.len();
    let mean_settling_time_s = if settled.is_empty() { None } else { Some(settled.iter().sum::<f64>() / settled.len() as f64) };
    let max_wheel_momentum_nms = rows.iter().map(|r| r.wheel_momentum_nms).fold(0.0_f64, f64::max);
    let max_wheel_sat_frac = rows.iter().map(|r| r.wheel_sat_frac).fold(0.0_f64, f64::max);
    let rcs_propellant_kg_used = rows.last().map(|r| r.rcs_propellant_kg_cum).unwrap_or(0.0);
    let final_pointing_error_deg = rows.last().map(|r| r.pointing_error_deg).unwrap_or(f64::NAN);

    println!(
        "  [{sweep_axis}] gain={gain:.4} null_motion_gain={null_motion_gain:.4}  settle {n_settled}/{n_transitions}  \
         mean_settle={} max_H={max_wheel_momentum_nms:.4} Nms  max_wheel_sat={max_wheel_sat_frac:.4}  \
         rcs={rcs_propellant_kg_used:.6} kg  final_err={final_pointing_error_deg:.4} deg",
        mean_settling_time_s.map(|s| format!("{s:.0}s")).unwrap_or_else(|| "n/a".to_string()),
    );

    SweepResult {
        sweep_axis, gain, null_motion_gain, n_transitions, n_settled,
        mean_settling_time_s, max_wheel_momentum_nms, max_wheel_sat_frac, rcs_propellant_kg_used, final_pointing_error_deg,
    }
}

fn main() {
    let kernel = find_kernel().expect("kernels/de440s.bsp not found -- see the README for download instructions");
    let almanac = Almanac::new(&kernel).expect("failed to load de440s.bsp");

    let (y, m, d) = DEPARTURE_EPOCH;
    let epoch0 = Epoch::from_gregorian_utc(y, m, d, 0, 0, 0, 0);
    let dep_epoch = epoch0 + Duration::from_days(DEP_OFFSET_DAYS);
    let tof_s = TOF_DAYS * 86_400.0;
    let arr_epoch = dep_epoch + Duration::from_days(TOF_DAYS);

    let earth_dep = almanac.body_state_heliocentric(Body::Earth, dep_epoch).expect("Earth state");
    let mars_arr = almanac.body_state_heliocentric(Body::Mars, arr_epoch).expect("Mars state");

    let r_dep = [earth_dep.position.inner.x, earth_dep.position.inner.y, earth_dep.position.inner.z];
    let v_dep = [earth_dep.velocity.inner.x, earth_dep.velocity.inner.y, earth_dep.velocity.inner.z];
    let r_arr = [mars_arr.position.inner.x, mars_arr.position.inner.y, mars_arr.position.inner.z];
    let v_arr = [mars_arr.velocity.inner.x, mars_arr.velocity.inner.y, mars_arr.velocity.inner.z];

    let lambert = LambertArc { r_dep, v_dep, r_arr, v_arr, tof_s, mu: MU_SUN_M3S2 };
    let sol = lambert.solve().expect("Lambert solve failed");

    let r0 = Vector3::new(r_dep[0], r_dep[1], r_dep[2]);
    let v0 = Vector3::new(sol.v_transfer_dep[0], sol.v_transfer_dep[1], sol.v_transfer_dep[2]);
    println!("Real ANISE Earth->Mars Lambert transfer: dep {dep_epoch}, TOF {TOF_DAYS:.2} d");

    let reference = sample_reference_trajectory_uniform(r0, v0, MU_SUN_M3S2, &[], DEMO_DURATION_S, REFERENCE_SAMPLE_DT_S, 1e-10, 1e-3);

    let mut schedule = vec![serde_json::json!({ "start_s": 0.0, "end_s": DEMO_DURATION_S, "mode": "Cruise" })];
    let mut t = 0.0_f64;
    while t < DEMO_DURATION_S {
        let pass_start = t;
        let pass_end = (t + COMM_PASS_DURATION_S).min(DEMO_DURATION_S);
        schedule.push(serde_json::json!({ "start_s": pass_start, "end_s": pass_end, "mode": "Comm" }));
        t += COMM_PASS_PERIOD_S;
    }

    let n_earth_samples = 200;
    let earth_track: Vec<serde_json::Value> = (0..=n_earth_samples)
        .map(|i| {
            let t_s = DEMO_DURATION_S * i as f64 / n_earth_samples as f64;
            let t_abs = dep_epoch + Duration::from_seconds(t_s);
            let st = almanac.body_state_heliocentric(Body::Earth, t_abs).expect("Earth state");
            serde_json::json!({
                "t_s": t_s,
                "r_m": [st.position.inner.x, st.position.inner.y, st.position.inner.z],
                "v_mps": [st.velocity.inner.x, st.velocity.inner.y, st.velocity.inner.z],
            })
        })
        .collect();

    let reference_json: Vec<serde_json::Value> = reference
        .points()
        .iter()
        .map(|p| serde_json::json!({ "t_s": p.t_s, "r_m": [p.r_m.x, p.r_m.y, p.r_m.z], "v_mps": [p.v_mps.x, p.v_mps.y, p.v_mps.z] }))
        .collect();

    let cfg_json = serde_json::json!({
        "mission": { "name": "cruise_gain_sweep_demo", "objective": "Orbit" },
        "target_body": { "name": "Mars", "ephemeris": "Anise" },
        "spacecraft": {
            "mass_kg": 400.0, "dry_mass_kg": 320.0, "propellant_mass_kg": 80.0,
            "bus_dims_m": [1.0, 1.0, 1.2], "inertia_diag_kgm2": [60.0, 60.0, 40.0],
            "srp_model": "Cannonball",
            "hardware": [
                { "type": "CommAntenna", "boresight": [0.0, 0.0, 1.0], "beamwidth_deg": 20.0 },
                { "type": "SolarPanel", "area_m2": 4.0, "position_m": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0] },
            ],
        },
        "trajectory": { "phases": ["Cruise"], "solver": "Lambert", "departure_body": "Earth" },
        "gnc": { "navigation_filter": "EKF", "pointing_mode": "Nadir", "attitude_controller": "ReactionWheelPD" },
        "simulation": {
            "integrator": "DormandPrince45", "rtol": 1.0e-9, "atol": 1.0e-7,
            "dt_truth_s": 10.0, "dt_meas_s": 120.0, "monte_carlo_runs": 0,
            "output_dir": "out/cruise_gain_sweep_demo/",
        },
        "cruise_seed": {
            "r0_m": [r0.x, r0.y, r0.z],
            "v0_m": [v0.x, v0.y, v0.z],
            "reference": reference_json,
            "duration_s": DEMO_DURATION_S,
            "tick_s": CONTROL_TICK_S,
            "modes": [
                { "name": "Cruise", "rules": [{ "hardware_index": 1, "target": { "type": "Sun" } }] },
                { "name": "Comm", "rules": [{ "hardware_index": 0, "target": { "type": "Body", "name": "Earth" } }] },
            ],
            "mode_schedule": schedule,
            "body_tracks": [{ "name": "Earth", "track": earth_track }],
        },
    });
    let cfg: MissionConfig = serde_json::from_value(cfg_json).expect("config should deserialize");
    let errors = mission_planner::config::check_config(&cfg);
    if !errors.is_empty() {
        panic!("check_config failed: {errors:?}");
    }

    println!("\n== Sweep A: gain, null_motion_gain fixed at default ({DEFAULT_NULL_MOTION_GAIN}) ==");
    let gains_grid = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16];
    let mut results: Vec<SweepResult> = gains_grid
        .iter()
        .map(|&g| run_one("gain", g, DEFAULT_NULL_MOTION_GAIN, &cfg, &reference, r0, v0))
        .collect();

    println!("\n== Sweep B: null_motion_gain, gain fixed at default ({DEFAULT_GAIN}) ==");
    let null_grid = [0.0, 0.05, 0.1, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.3, 0.4, 0.6, 0.8];
    results.extend(
        null_grid
            .iter()
            .map(|&ng| run_one("null_motion_gain", DEFAULT_GAIN, ng, &cfg, &reference, r0, v0)),
    );

    let out_dir = "out/cruise_gain_sweep_demo";
    fs::create_dir_all(out_dir).expect("create out dir");
    let mut csv = vec![
        "sweep_axis,gain,null_motion_gain,n_transitions,n_settled,mean_settling_time_s,\
         max_wheel_momentum_nms,max_wheel_sat_frac,rcs_propellant_kg_used,final_pointing_error_deg"
            .to_string(),
    ];
    for r in &results {
        csv.push(format!(
            "{},{:.6},{:.6},{},{},{},{:.6e},{:.6},{:.6e},{:.6}",
            r.sweep_axis, r.gain, r.null_motion_gain, r.n_transitions, r.n_settled,
            r.mean_settling_time_s.map(|s| format!("{s:.3}")).unwrap_or_default(),
            r.max_wheel_momentum_nms, r.max_wheel_sat_frac, r.rcs_propellant_kg_used, r.final_pointing_error_deg,
        ));
    }
    let csv_path = format!("{out_dir}/gain_sweep.csv");
    fs::write(&csv_path, csv.join("\n") + "\n").expect("write csv");
    println!("\n{csv_path}  ({} rows)", results.len());
    println!("\nPlot: python plot/plot_cruise_gain_sweep.py");
}
