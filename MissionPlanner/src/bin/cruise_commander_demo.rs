//! Attitude-commander verification demo — the priority-ordered multi-rule
//! attitude commander (`sim_engine::attitude_commander`, `MissionPlanner::
//! cruise::GncCommander`) flown against the SAME real ANISE Earth->Mars
//! interplanetary leg `cruise_demo.rs` uses, instead of a fixed
//! `SunPointing` hold. A real, periodic comm-pass schedule: `"Cruise"`
//! (a placed SolarPanel's normal -> Sun) most of the time, switching to
//! `"Comm"` (a placed CommAntenna's boresight -> Earth, resolved via a
//! real ANISE-queried Earth position track, not a synthetic fixed point)
//! for a comm pass every 8 h.
//!
//! Exercises exactly the same `run_cruise_streaming` code path
//! `/api/simulate` uses (not a hand-rolled loop), so this is also a real
//! end-to-end check of the full config->commander->result pipeline, not
//! just the commander in isolation (already unit-tested in
//! `cruise::tests`).
//!
//! Run from `MissionPlanner/`: `cargo run -p mission_planner --bin cruise_commander_demo --release`
//! Plot: `python plot/plot_cruise_commander_demo.py`

use std::fs;

use ephemeris::{Almanac, Body, Epoch};
use hifitime::Duration;
use nalgebra::Vector3;
use trajectory_solver::{keplerian::MU_SUN_M3S2, LambertArc};

use mission_planner::config::MissionConfig;
use mission_planner::cruise::{run_cruise_streaming, sample_reference_trajectory_uniform};

// Same real best-arc window `cruise_demo.rs`/`soi_demo.rs` use (Phase 8h).
const DEPARTURE_EPOCH: (i32, u8, u8) = (2026, 10, 28);
const DEP_OFFSET_DAYS: f64 = 3.80;
const TOF_DAYS: f64 = 291.14;

const REFERENCE_SAMPLE_DT_S: f64 = 60.0;
// Same bandwidth-respecting tick `cruise_demo.rs` established for this
// vehicle/gains combination (see its own doc comment for the real incident
// that set this).
const CONTROL_TICK_S: f64 = 10.0;
const DEMO_DURATION_S: f64 = 3.0 * 86_400.0; // 3 days

const COMM_PASS_PERIOD_S: f64 = 8.0 * 3600.0; // one comm pass every 8 h
const COMM_PASS_DURATION_S: f64 = 3600.0; // 1 h per pass

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

    let reference = sample_reference_trajectory_uniform(
        r0, v0, MU_SUN_M3S2, &[], DEMO_DURATION_S, REFERENCE_SAMPLE_DT_S, 1e-10, 1e-3,
    );
    println!("Reference trajectory: {} s span", reference.t_end_s() - reference.t_start_s());

    // Real ANISE-queried Earth position track over the demo window (NOT a
    // synthetic fixed point, unlike the unit-test fixture) -- this is what
    // the Comm mode's "antenna -> Body(Earth)" rule actually resolves
    // against.
    let n_earth_samples = 200;
    let earth_samples: Vec<(f64, [f64; 3], [f64; 3])> = (0..=n_earth_samples)
        .map(|i| {
            let t_s = DEMO_DURATION_S * i as f64 / n_earth_samples as f64;
            let t_abs = dep_epoch + Duration::from_seconds(t_s);
            let st = almanac.body_state_heliocentric(Body::Earth, t_abs).expect("Earth state");
            (
                t_s,
                [st.position.inner.x, st.position.inner.y, st.position.inner.z],
                [st.velocity.inner.x, st.velocity.inner.y, st.velocity.inner.z],
            )
        })
        .collect();
    let earth_track: Vec<serde_json::Value> = earth_samples
        .iter()
        .map(|(t_s, r, v)| serde_json::json!({ "t_s": t_s, "r_m": r, "v_mps": v }))
        .collect();

    let mut schedule = vec![serde_json::json!({ "start_s": 0.0, "end_s": DEMO_DURATION_S, "mode": "Cruise" })];
    let mut t = 0.0_f64;
    while t < DEMO_DURATION_S {
        let pass_start = t;
        let pass_end = (t + COMM_PASS_DURATION_S).min(DEMO_DURATION_S);
        schedule.push(serde_json::json!({ "start_s": pass_start, "end_s": pass_end, "mode": "Comm" }));
        t += COMM_PASS_PERIOD_S;
    }
    // Comm-pass entries are appended AFTER the blanket Cruise entry, so
    // GncCommander's "last matching entry wins" rule correctly gives comm
    // passes priority over the default Cruise coverage.

    let reference_json: Vec<serde_json::Value> = reference
        .points()
        .iter()
        .map(|p| serde_json::json!({ "t_s": p.t_s, "r_m": [p.r_m.x, p.r_m.y, p.r_m.z], "v_mps": [p.v_mps.x, p.v_mps.y, p.v_mps.z] }))
        .collect();

    let cfg_json = serde_json::json!({
        "mission": { "name": "cruise_commander_demo", "objective": "Orbit" },
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
            "output_dir": "out/cruise_commander_demo/",
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

    println!("Flying {:.0} ticks with a real Cruise/Comm mode schedule (comm pass every {:.0} h for {:.0} min)...",
        DEMO_DURATION_S / CONTROL_TICK_S, COMM_PASS_PERIOD_S / 3600.0, COMM_PASS_DURATION_S / 60.0);

    let mut rows_csv = vec![
        "t_s,x_m,y_m,z_m,pointing_error_deg,active_mode,torque_gravity_gradient_nm,torque_srp_nm,\
         wheel_momentum_nms,wheel_sat_frac,max_rule_violation_deg,w1,w2,w3,w4,qw,qx,qy,qz,\
         qcw,qcx,qcy,qcz,tau_cmd_x,tau_cmd_y,tau_cmd_z,tau_del_x,tau_del_y,tau_del_z"
            .to_string(),
    ];
    let mut n_rows = 0usize;
    let result = run_cruise_streaming(&cfg, &mut |row| {
        n_rows += 1;
        rows_csv.push(format!(
            "{:.3},{:.6e},{:.6e},{:.6e},{:.6},{},{:.6e},{:.6e},{:.6e},{:.6},{},{:.4},{:.4},{:.4},{:.4},\
             {:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
            row.t_s, row.r_m.x, row.r_m.y, row.r_m.z, row.pointing_error_deg,
            row.active_mode.as_deref().unwrap_or(""),
            row.torque_gravity_gradient_nm, row.torque_srp_nm, row.wheel_momentum_nms,
            row.wheel_sat_frac,
            row.max_rule_violation_deg.map(|v| format!("{v:.4}")).unwrap_or_default(),
            row.wheel_speeds_radps[0], row.wheel_speeds_radps[1],
            row.wheel_speeds_radps[2], row.wheel_speeds_radps[3],
            row.q[0], row.q[1], row.q[2], row.q[3],
            row.q_cmd[0], row.q_cmd[1], row.q_cmd[2], row.q_cmd[3],
            row.torque_cmd_body_nm.x, row.torque_cmd_body_nm.y, row.torque_cmd_body_nm.z,
            row.torque_delivered_body_nm.x, row.torque_delivered_body_nm.y, row.torque_delivered_body_nm.z,
        ));
        true
    })
    .expect("run_cruise_streaming should succeed");

    println!("\n-- Results --");
    println!("Ticks flown: {n_rows}");
    println!("Final position dispersion: {:.3} m", result.final_dr_m);
    println!("Mode transitions detected: {}", result.mode_transitions.len());
    for t in &result.mode_transitions {
        println!(
            "  t={:.0}s  {} -> {}  settling_time={}  max_wheel_momentum={:.4} N*m*s",
            t.t_s,
            t.from_mode.as_deref().unwrap_or("(none)"),
            t.to_mode,
            t.settling_time_s.map(|s| format!("{s:.0}s")).unwrap_or_else(|| "never".to_string()),
            t.max_wheel_momentum_nms,
        );
    }

    let out_dir = "out/cruise_commander_demo";
    fs::create_dir_all(out_dir).expect("create out dir");
    let traj_path = format!("{out_dir}/cruise_commander_demo.csv");
    fs::write(&traj_path, rows_csv.join("\n") + "\n").expect("write csv");
    println!("\n{traj_path}  ({n_rows} rows)");

    let mut transitions_csv = vec!["t_s,from_mode,to_mode,settling_time_s,rcs_propellant_kg_used,max_wheel_momentum_nms".to_string()];
    for t in &result.mode_transitions {
        transitions_csv.push(format!(
            "{:.3},{},{},{},{:.6e},{:.6e}",
            t.t_s, t.from_mode.as_deref().unwrap_or(""), t.to_mode,
            t.settling_time_s.map(|s| format!("{s:.3}")).unwrap_or_default(),
            t.rcs_propellant_kg_used, t.max_wheel_momentum_nms,
        ));
    }
    let transitions_path = format!("{out_dir}/mode_transitions.csv");
    fs::write(&transitions_path, transitions_csv.join("\n") + "\n").expect("write transitions csv");
    println!("{transitions_path}  ({} rows)", result.mode_transitions.len());

    let mut earth_csv = vec!["t_s,x_m,y_m,z_m".to_string()];
    for (t_s, r, _v) in &earth_samples {
        earth_csv.push(format!("{:.3},{:.6e},{:.6e},{:.6e}", t_s, r[0], r[1], r[2]));
    }
    let earth_path = format!("{out_dir}/earth_track.csv");
    fs::write(&earth_path, earth_csv.join("\n") + "\n").expect("write earth track csv");
    println!("{earth_path}  ({} rows)", earth_samples.len());

    println!("\nPlot: python plot/plot_cruise_commander_demo.py");
    println!("Attitude animation: python plot/plot_cruise_commander_attitude_animation.py");
}
