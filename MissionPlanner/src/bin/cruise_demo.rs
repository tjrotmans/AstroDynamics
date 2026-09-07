//! Phase 5.1 verification demo — the first real end-to-end test of the
//! Phase 13 cruise composition (`mission_planner::cruise::run_cruise_leg`):
//! a real ANISE Earth->Mars interplanetary coast, flown under `step_tick`'s
//! translation+attitude propagation with a real quaternion-PD/wheel
//! attitude-hold controller (SunPointing), while `reference_guidance::
//! dispersion` reports how far the flown state drifts from the Layer-1
//! reference trajectory -- reported, not corrected (no TCM burns in this
//! first cut, see `cruise.rs`'s module doc comment).
//!
//! Reuses the same real best-arc departure window `soi_demo.rs` established
//! for `mars_flyby.toml` (Phase 8h) -- a real, already-verified Earth->Mars
//! Lambert transfer -- rather than re-deriving a new one. The reference
//! trajectory and the flown trajectory are propagated under the IDENTICAL
//! force model (heliocentric gravity only, no SOI candidates) specifically
//! so any reported dispersion is a genuine composition-correctness signal
//! (translation+attitude+control all agreeing with the plan), not an
//! artifact of comparing two different physics models.
//!
//! Flies the FULL transfer (`TOF_DAYS`, ~291 days), not just a leading
//! segment. Originally this binary flew only a short representative window,
//! because `mars_flyby.toml`'s real vehicle (60 kg*m^2 inertia) with the
//! config's own default reaction-wheel PD gains (kp=0.05, kd=0.6) has a
//! closed-loop natural period of ~218 s (omega_n = sqrt(kp/I)), which forces
//! a control tick in the ~11-22 s band ("tick <~ 1/(10-20 x required control
//! bandwidth)", this module's own doc comment) -- a 3600 s (1 h) tick,
//! initially tried under Phase 13's separate "quiescent cruise disturbance
//! torques are nano-N*m -> hundreds-of-seconds ticks are fine" guidance
//! (a DIFFERENT constraint, about disturbance-torque frequency, not control
//! bandwidth -- conflating the two was the original bug), drove the
//! discrete zero-order-hold PD loop genuinely unstable (pointing error to
//! 180 deg, wheel momentum >1e6 N*m*s, MaxNumStepReached). Measured directly
//! rather than assumed: a `CONTROL_TICK_S = 10` s run over the full transfer
//! is ~2.5M ticks and completes in well under a minute in release mode (the
//! rotational/translational integration per tick is cheap), so flying the
//! whole leg is not actually impractical -- it just needs the CSV output
//! decimated (`CSV_DECIMATION`) so the written file stays a reasonable size
//! for pandas/Plotly; the returned `Vec<CruiseTickRow>` and the printed
//! max-stat tracking still cover every tick at full resolution.
//!
//! Run from the `MissionPlanner/` directory: `cargo run -p mission_planner --bin cruise_demo --release`
//! Plots: `python plot/plot_cruise_demo.py`, `python plot/plot_cruise_demo_attitude_animation.py`

use std::fs;

use ephemeris::{Almanac, Body, Epoch};
use hifitime::Duration;
use nalgebra::Vector3;
use trajectory_solver::{keplerian::MU_SUN_M3S2, LambertArc};

use mission_planner::config::MissionConfig;
use mission_planner::cruise::{run_cruise_leg, sample_reference_trajectory_uniform};
use mission_planner::simulate::{
    build_spacecraft_properties, pd_gains_from_cfg, rcs_from_hardware, wheel_cluster_from_hardware,
};
use sim_engine::{ControlMode, CruisePointingMode};

// Real best-arc window for `mars_flyby.toml`, as found by its own porkchop
// scan (Phase 8g/8j) -- same constants `soi_demo.rs` uses (Phase 8h).
const DEPARTURE_EPOCH: (i32, u8, u8) = (2026, 10, 28);
const DEP_OFFSET_DAYS: f64 = 3.80;
const TOF_DAYS: f64 = 291.14;

// Fine enough that ReferenceTrajectory::state_at's linear interpolation
// between samples stays well under the dispersion this demo is trying to
// measure -- found the hard way: a 6 h sample spacing produced a spurious
// ~10,800 km "dispersion" spike from chord-vs-arc interpolation error alone
// (real curvature over a 6 h arc this close to departure, not a real
// flown-vs-reference divergence). Same failure-mode family as the
// `find_inbound_radius_crossing` chord-vs-arc sagitta bug documented in
// the design notes (a real ~600 km Mercury-transit miss-distance error from
// interpolating across a target body's own curvature) -- different
// function, same lesson: don't linearly interpolate across a real orbital
// arc without checking the interpolation interval is fine enough.
const REFERENCE_SAMPLE_DT_S: f64 = 60.0; // 1 min
// Control-bandwidth-respecting tick -- see module doc comment for the
// closed-loop natural-period math this is sized against.
const CONTROL_TICK_S: f64 = 10.0;
// Roughly how many rows to keep when writing CSVs -- the underlying tick-
// by-tick physics still runs at full resolution (CONTROL_TICK_S /
// REFERENCE_SAMPLE_DT_S); this only thins what gets written to disk so the
// files stay a reasonable size for pandas/Plotly over a full ~291-day,
// ~2.5M-tick leg. Max-stat tracking (printed results) is unaffected --
// it's computed from every tick via the `on_row` callback below.
const CSV_TARGET_ROWS: usize = 4000;

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
    println!(
        "Real ANISE Earth->Mars Lambert transfer: dep {dep_epoch}, TOF {TOF_DAYS:.2} d, dv_dep {:.1} m/s",
        (v0 - Vector3::new(v_dep[0], v_dep[1], v_dep[2])).norm(),
    );

    // ── Layer-1 reference trajectory: pure heliocentric coast, no SOI
    // candidates -- the flown leg below uses the IDENTICAL force model so
    // any dispersion reported is a real composition-correctness signal.
    // Propagated over the FULL transfer (tof_s) -- the flown leg below now
    // flies the whole leg, not a short segment. Uniformly resampled (see
    // sample_reference_trajectory_uniform's doc comment) rather than a raw
    // propagate() call -- Sparse-mode output density follows the adaptive
    // integrator's own step choices, not sample_dt_s, which is too coarse
    // for linear interpolation over a smooth coast.
    let reference = sample_reference_trajectory_uniform(
        r0, v0, MU_SUN_M3S2, &[], tof_s, REFERENCE_SAMPLE_DT_S, 1e-10, 1e-3,
    );
    println!(
        "Reference trajectory: {:.0} s span, {REFERENCE_SAMPLE_DT_S:.0} s uniform sampling (of a {tof_s:.0} s full transfer)",
        reference.t_end_s() - reference.t_start_s(),
    );

    // ── Real vehicle, reused from the same TOML this window came from.
    let cfg = MissionConfig::from_file("config/mars_flyby.toml").expect("load mars_flyby.toml");
    let sc = build_spacecraft_properties(&cfg);
    let wheel_cluster = wheel_cluster_from_hardware(&cfg);
    let (rcs_thrusters, _) = rcs_from_hardware(&cfg);
    let gains = pd_gains_from_cfg(&cfg);

    let initial = sim_engine::SixDofState {
        t_s: 0.0,
        r_m: r0,
        v_mps: v0,
        q: sim_engine::reference_guidance::desired_quaternion_cruise(
            CruisePointingMode::SunPointing, &r0, &Vector3::zeros(),
        ),
        omega_radps: Vector3::zeros(),
        wheel_speeds_radps: [0.0; 4],
        mass_kg: sc.mass_kg,
    };

    println!(
        "Vehicle: {:.0} kg, inertia diag {:?} kg*m^2, PD gains kp={:.3} kd={:.3}",
        sc.mass_kg, cfg.spacecraft.inertia_diag_kgm2, gains.kp, gains.kd,
    );
    println!("Flying {:.0} ticks of {CONTROL_TICK_S:.0} s under SunPointing attitude hold, WheelsPrimary allocation...", tof_s / CONTROL_TICK_S);

    let mut max_pointing_error_deg = 0.0_f64;
    let mut max_dr_m = 0.0_f64;
    let mut max_wheel_momentum_nms = 0.0_f64;
    let mut n_rows = 0usize;

    // Demo keeps its single hand-checked PD gain set for every mode (the
    // pre-behavior) — `from_pd` disables activity scaling and
    // scheduling so the numbers printed above are exactly what flies.
    let controls = mission_planner::attitude_tuning::AttitudeControlSet::from_pd(
        gains, sc.inertia_diag_kgm2, sc.mass_kg, wheel_cluster.max_torque,
        sim_engine::rcs_worst_axis_authority_nm(&rcs_thrusters), CONTROL_TICK_S,
    );
    let rows = run_cruise_leg(
        initial, &reference, &[], MU_SUN_M3S2, &sc, &wheel_cluster, &rcs_thrusters,
        &controls, ControlMode::WheelsPrimary,
        sim_engine::MomentumManagementLaw::ThresholdRcs { gain: 0.02, null_motion_gain: 0.05 },
        CruisePointingMode::SunPointing, None, &|_t_s| Vector3::zeros(),
        CONTROL_TICK_S, tof_s, cfg.spacecraft.propellant_mass_kg, None,
        cfg.simulation.rtol, cfg.simulation.atol, Vector3::zeros(), &[],
        &|_, _| None,
        &mut |row| {
            n_rows += 1;
            max_pointing_error_deg = max_pointing_error_deg.max(row.pointing_error_deg);
            max_dr_m = max_dr_m.max(row.dr_m);
            max_wheel_momentum_nms = max_wheel_momentum_nms.max(row.wheel_momentum_nms);
            true
        },
    );

    let last = rows.last().expect("at least one tick");
    println!("\n-- Results --");
    println!("Ticks flown: {n_rows}");
    println!("Final pointing error: {:.4} deg  (max over run: {:.4} deg)", last.pointing_error_deg, max_pointing_error_deg);
    println!("Final position dispersion: {:.3} m  (max over run: {:.3} m)", last.dr_m, max_dr_m);
    println!("Final velocity dispersion: {:.6} m/s", last.dv_mps);
    println!("Final wheel momentum: {:.4} N*m*s  (max over run: {:.4} N*m*s)", last.wheel_momentum_nms, max_wheel_momentum_nms);
    println!("Total RCS propellant used: {:.6} kg", last.rcs_propellant_kg_cum);

    let out_dir = "out/cruise_demo";
    fs::create_dir_all(out_dir).expect("create out dir");

    // Thin what's written to disk -- `rows`/`reference.points()` are already
    // at full tick/sample resolution (millions of entries over a full
    // ~291-day leg); always keep the very last row so the CSV's final state
    // matches the printed "Results" summary above exactly.
    let row_stride = (rows.len() / CSV_TARGET_ROWS).max(1);
    let ref_stride = (reference.points().len() / CSV_TARGET_ROWS).max(1);

    let mut rows_csv = vec![
        "t_s,x_m,y_m,z_m,qw,qx,qy,qz,qcw,qcx,qcy,qcz,dr_m,dv_mps,pointing_error_deg,\
         wheel_momentum_nms,w1,w2,w3,w4,rcs_propellant_kg_cum,wheel_sat_frac,\
         propellant_remaining_kg,torque_gravity_gradient_nm,torque_srp_nm,\
         accel_central_gravity_mps2,accel_third_body_mps2,accel_srp_mps2"
            .to_string(),
    ];
    for (i, row) in rows.iter().enumerate() {
        if i % row_stride != 0 && i != rows.len() - 1 {
            continue;
        }
        rows_csv.push(format!(
            "{:.3},{:.6e},{:.6e},{:.6e},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},\
             {:.6e},{:.6e},{:.6},{:.6},{:.4},{:.4},{:.4},{:.4},{:.6e},{:.6},{:.6e},{:.6e},\
             {:.6e},{:.6e},{:.6e},{:.6e}",
            row.t_s, row.r_m.x, row.r_m.y, row.r_m.z,
            row.q[0], row.q[1], row.q[2], row.q[3],
            row.q_cmd[0], row.q_cmd[1], row.q_cmd[2], row.q_cmd[3],
            row.dr_m, row.dv_mps, row.pointing_error_deg,
            row.wheel_momentum_nms, row.wheel_speeds_radps[0], row.wheel_speeds_radps[1],
            row.wheel_speeds_radps[2], row.wheel_speeds_radps[3], row.rcs_propellant_kg_cum,
            row.wheel_sat_frac, row.propellant_remaining_kg, row.torque_gravity_gradient_nm,
            row.torque_srp_nm, row.accel_central_gravity_mps2, row.accel_third_body_mps2,
            row.accel_srp_mps2,
        ));
    }
    let traj_path = format!("{out_dir}/cruise_demo.csv");
    fs::write(&traj_path, rows_csv.join("\n") + "\n").expect("write cruise_demo.csv");

    let mut ref_csv = vec!["t_s,x_m,y_m,z_m".to_string()];
    let ref_points = reference.points();
    for (i, p) in ref_points.iter().enumerate() {
        if i % ref_stride != 0 && i != ref_points.len() - 1 {
            continue;
        }
        ref_csv.push(format!("{:.3},{:.6e},{:.6e},{:.6e}", p.t_s, p.r_m.x, p.r_m.y, p.r_m.z));
    }
    let ref_path = format!("{out_dir}/reference_trajectory.csv");
    fs::write(&ref_path, ref_csv.join("\n") + "\n").expect("write reference_trajectory.csv");

    // ── Real Earth/Mars heliocentric tracks over the FULL transfer window
    // (dep_epoch -> arr_epoch), not just the 3-day flown segment -- gives the
    // 3D overview plot real solar-system scale/context to place the flown
    // arc against, same convention `design.rs::write_best_arc_bodies` already
    // established for the Layer-1 trajectory-design plots.
    const N_BODY_SAMPLES: usize = 60;
    let mut bodies_csv = vec!["t_s,earth_x_m,earth_y_m,earth_z_m,mars_x_m,mars_y_m,mars_z_m".to_string()];
    for i in 0..N_BODY_SAMPLES {
        let frac = i as f64 / (N_BODY_SAMPLES - 1) as f64;
        let t_s = frac * tof_s;
        let epoch = dep_epoch + Duration::from_seconds(t_s);
        let e = almanac.body_state_heliocentric(Body::Earth, epoch).expect("Earth state");
        let m = almanac.body_state_heliocentric(Body::Mars, epoch).expect("Mars state");
        bodies_csv.push(format!(
            "{t_s:.3},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
            e.position.inner.x, e.position.inner.y, e.position.inner.z,
            m.position.inner.x, m.position.inner.y, m.position.inner.z,
        ));
    }
    let bodies_path = format!("{out_dir}/bodies.csv");
    fs::write(&bodies_path, bodies_csv.join("\n") + "\n").expect("write bodies.csv");

    println!("\n{traj_path}  ({} rows)", rows.len());
    println!("{ref_path}  ({} rows)", reference.points().len());
    println!("{bodies_path}  ({} rows, Earth/Mars over full {tof_s:.0} s transfer)", N_BODY_SAMPLES);
    println!("Plot: python plot/plot_cruise_demo.py");
}
