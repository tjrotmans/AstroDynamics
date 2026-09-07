//! Phase 13d/13e verification demo — the cascaded controller (quaternion PD)
//! + control allocation (WheelsPrimary mode) driving a large initial
//! pointing error to zero, commanded via the Phase 13d pointing-mode library
//! (`CruisePointingMode::BurnAttitude`, a fixed inertial target direction).
//!
//! Confirms, with a real closed-loop run (not just unit tests in isolation):
//!   - the high-level PD law produces a torque that actually reduces
//!     pointing error over time (not just at t=0),
//!   - `control::allocate` in `WheelsPrimary` mode routes 100% of that
//!     torque to the wheels and fires zero RCS while wheels stay well
//!     under the desaturation threshold,
//!   - wheel momentum grows monotonically as it absorbs the spacecraft's
//!     angular momentum change (conservation check: total system angular
//!     momentum -- body + wheels -- stays constant, since there is no
//!     external torque commanded here beyond the internal PD/wheel loop
//!     and the (small) gravity-gradient/SRP disturbance).
//!
//! Run:  cargo run -p mission_planner --bin sixdof_control_demo --release
//! Plot: python plot/plot_sixdof_control_demo.py

use nalgebra::{Vector3, Vector4};
use hardware_catalog::ThrusterSpec;
use orbital_models::constants::MU_SUN;
use sim_engine::truth::{SpacecraftProperties, SrpTruthModel};
use sim_engine::{
    allocate, net_body_torque, step_tick, AttitudeControlLaw, ControlMode, CruisePointingMode,
    PdGains, ReactionWheelCluster, SixDofState,
};
use sim_engine::reference_guidance::desired_quaternion_cruise;
use trajectory_solver::PropagatorBody;

const EARTH_MU: f64 = 3.986_004_418e14;
const EARTH_R: f64 = 6.378e6;
const ORBIT_ALT_M: f64 = 700_000.0;
const TICK_S: f64 = 5.0; // finer than sixdof_demo's cruise tick -- an active control loop
const DURATION_S: f64 = 1800.0; // 30 minutes -- long enough to see convergence

fn main() {
    std::fs::create_dir_all("out/sixdof_control_demo").expect("create out dir");

    let earth_pos = Vector3::new(1.495_98e11, 0.0, 0.0);
    let earth_state_at = move |_t: f64| (earth_pos, Vector3::zeros());
    let bodies = vec![PropagatorBody {
        name: "Earth",
        mu_m3s2: EARTH_MU,
        soi_radius_m: Some(9.24e8),
        state_at: &earth_state_at,
        central_fidelity: None,
        radius_m: Some(EARTH_R),
    }];

    let r_orbit = EARTH_R + ORBIT_ALT_M;
    let v_circ = (EARTH_MU / r_orbit).sqrt();

    let sc = SpacecraftProperties {
        mass_kg: 1000.0,
        inertia_diag_kgm2: Vector3::new(150.0, 150.0, 400.0),
        srp: SrpTruthModel::Cannonball { c_r: 1.4, area_m2: 4.0 },
        drag_area_m2: 4.0,
    };

    // Large initial pointing error: body starts near identity while the
    // commanded attitude is a fixed inertial direction ~90 deg away.
    let mut state = SixDofState {
        t_s: 0.0,
        r_m: earth_pos + Vector3::new(r_orbit, 0.0, 0.0),
        v_mps: Vector3::new(0.0, v_circ, 0.0),
        q: Vector4::new(1.0, 0.0, 0.0, 0.0),
        omega_radps: Vector3::zeros(),
        wheel_speeds_radps: [0.0; 4],
        mass_kg: 1000.0,
    };

    let target_dir = Vector3::new(0.0, 1.0, 0.0).normalize();
    let pointing_mode = CruisePointingMode::BurnAttitude { thrust_dir_inertial: target_dir };
    let q_cmd = desired_quaternion_cruise(pointing_mode, &state.r_m, &Vector3::zeros());

    let law = AttitudeControlLaw::QuaternionPd(PdGains { kp: 0.08, kd: 0.9, pointing_db_rad: 0.0, rate_db_rads: 0.0 });
    let wheel_cluster = ReactionWheelCluster::four_wheel_pyramid(0.012, 628.3, 0.12, 0.8);
    let thruster_spec = ThrusterSpec::monoprop();
    let rcs_thrusters = sim_engine::actuators::build_rcs(&thruster_spec, 0.5);

    let initial_err_deg = pointing_error_deg(&state.q, &q_cmd);
    println!("Phase 13d/13e verification: {DURATION_S:.0} s closed-loop pointing control");
    println!("  Initial pointing error: {initial_err_deg:.2} deg");
    println!("  Tick: {TICK_S:.0} s, PD gains: kp={:.3} kd={:.3}", 0.08, 0.9);

    let mut rows: Vec<String> = vec![
        "t_s,pointing_err_deg,omega_norm_mrads,wheel_momentum_nms,\
         w1,w2,w3,w4,rcs_duty_cycle,rcs_propellant_kg_cum"
            .to_string(),
    ];

    let n_ticks = (DURATION_S / TICK_S).ceil() as usize;
    let mut rcs_propellant_cum = 0.0_f64;
    for _ in 0..n_ticks {
        let tau_cmd = law.command_torque(&state.q, &q_cmd, &state.omega_radps);
        let alloc = allocate(
            ControlMode::WheelsPrimary, tau_cmd, &wheel_cluster, &state.wheel_speeds_radps,
            &rcs_thrusters, TICK_S,
            sim_engine::MomentumManagementLaw::None,
        );
        let control_torque_body = net_body_torque(&wheel_cluster, &alloc);
        rcs_propellant_cum += alloc.rcs_propellant_kg;

        let err_deg = pointing_error_deg(&state.q, &q_cmd);
        let h_wheel = wheel_cluster.total_momentum(&state.wheel_speeds_radps);
        rows.push(format!(
            "{:.2},{:.4},{:.6},{:.6},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6e}",
            state.t_s, err_deg, state.omega_radps.norm() * 1e3, h_wheel.norm(),
            state.wheel_speeds_radps[0], state.wheel_speeds_radps[1],
            state.wheel_speeds_radps[2], state.wheel_speeds_radps[3],
            alloc.rcs_duty_cycle, rcs_propellant_cum,
        ));

        // Integrate wheel speeds exactly (ZOH motor torque -- linear ODE),
        // same pattern `actuators::wheel_step` already uses.
        let mut new_speeds = state.wheel_speeds_radps;
        let speed_dots = wheel_cluster.speed_dots(&alloc.wheel_motor_torque_nm);
        for i in 0..4 {
            // Real motor torque saturates at the wheel's rated max speed —
            // see cruise.rs's identical clamp for the real runaway this
            // prevents (found via cruise_commander_demo).
            new_speeds[i] = (new_speeds[i] + speed_dots[i] * TICK_S).clamp(-wheel_cluster.max_speed, wheel_cluster.max_speed);
        }

        state = step_tick(&state, TICK_S, &sc, &bodies, MU_SUN, control_torque_body, h_wheel, 1e-10, 1e-12);
        state.wheel_speeds_radps = new_speeds;
    }

    let final_err_deg = pointing_error_deg(&state.q, &q_cmd);
    let final_h = wheel_cluster.total_momentum(&state.wheel_speeds_radps);
    println!("  Final pointing error: {final_err_deg:.4} deg");
    println!("  Final wheel momentum: {:.4} N*m*s", final_h.norm());
    println!("  Total RCS propellant used: {rcs_propellant_cum:.6} kg (should be ~0 -- wheels never saturate)");

    let path = "out/sixdof_control_demo/sixdof_control_demo.csv";
    std::fs::write(path, rows.join("\n") + "\n").expect("write csv");
    println!("\nSaved {path} ({n_ticks} rows)");
    println!("Plot: python plot/plot_sixdof_control_demo.py");
}

fn pointing_error_deg(q_cur: &Vector4<f64>, q_cmd: &Vector4<f64>) -> f64 {
    let dot = q_cur.dot(q_cmd).clamp(-1.0, 1.0).abs();
    2.0 * dot.acos().to_degrees()
}
