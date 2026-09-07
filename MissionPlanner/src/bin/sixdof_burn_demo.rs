//! Phase 13f verification demo — a finite burn with a real thrust
//! misalignment offset, fought in closed loop by the Phase 13e PD+wheel
//! controller, propagated through the Phase 13f fully-coupled integrator
//! (`step_tick_with_burn`).
//!
//! Confirms, with a real closed-loop run:
//!   - propellant mass depletes at the Tsiolkovsky rate throughout the burn,
//!   - the misaligned thrust (offset from the CoM) produces a real
//!     disturbance torque the controller must fight to hold pointing,
//!   - translational delta-v accumulates correctly even while attitude is
//!     being actively corrected (the coupled integrator's core claim: thrust
//!     direction follows the CONTROLLED, not nominal, attitude),
//!   - wheel momentum absorbs the disturbance without RCS needed (light
//!     misalignment case).
//!
//! Run:  cargo run -p mission_planner --bin sixdof_burn_demo --release
//! Plot: python plot/plot_sixdof_burn_demo.py

use nalgebra::{Vector3, Vector4};
use hardware_catalog::ThrusterSpec;
use orbital_models::constants::MU_SUN;
use sim_engine::truth::{SpacecraftProperties, SrpTruthModel};
use sim_engine::{
    allocate, net_body_torque, step_tick_with_burn, AttitudeControlLaw, BurnConfig, ControlMode,
    PdGains, ReactionWheelCluster, SixDofState,
};
use trajectory_solver::PropagatorBody;

const TICK_S: f64 = 1.0; // fine tick for an active burn, per the mode-scheduled tick design
const BURN_DURATION_S: f64 = 120.0;

fn main() {
    std::fs::create_dir_all("out/sixdof_burn_demo").expect("create out dir");

    // Deep space -- isolates the burn/control/misalignment physics from
    // orbital dynamics, which this demo isn't trying to verify (already
    // covered by sixdof_demo/sixdof_control_demo).
    let bodies: Vec<PropagatorBody> = Vec::new();

    let sc = SpacecraftProperties {
        mass_kg: 1000.0, // overwritten by state.mass_kg once the burn starts depleting it
        inertia_diag_kgm2: Vector3::new(150.0, 150.0, 400.0),
        srp: SrpTruthModel::Cannonball { c_r: 1.4, area_m2: 4.0 },
        drag_area_m2: 4.0,
    };

    let mut state = SixDofState {
        t_s: 0.0,
        r_m: Vector3::new(1.495_98e11, 0.0, 0.0),
        v_mps: Vector3::zeros(),
        q: Vector4::new(1.0, 0.0, 0.0, 0.0),
        omega_radps: Vector3::zeros(),
        wheel_speeds_radps: [0.0; 4],
        mass_kg: 1000.0,
    };

    let burn = BurnConfig {
        thrust_n: 400.0,
        isp_s: 300.0,
        body_dir: Vector3::new(1.0, 0.0, 0.0),
        // 0.2 mm off-axis -> tau = F x offset = 400 N * 2e-4 m = 0.08 N*m --
        // sized deliberately to stay within this wheel cluster's combined
        // torque authority (~0.39 N*m for a pure z-torque, 4 wheels x
        // 0.12 N*m max x cos(35.26deg) pyramid geometry factor). A first
        // attempt at 5 cm (20 N*m) oversaturated the wheels by ~50x and the
        // spacecraft tumbled to 133 deg -- correct physics (a real
        // disturbance exceeding actuator authority SHOULD cause loss of
        // control), but the wrong choice for a demo meant to show the
        // controller successfully holding attitude against a real but
        // survivable misalignment torque.
        thrust_offset_body_m: Vector3::new(0.0, 2.0e-4, 0.0),
    };

    let q_cmd = Vector4::new(1.0, 0.0, 0.0, 0.0); // hold the initial (burn) attitude
    // Pure PD control has a nonzero STEADY-STATE error against a constant
    // disturbance torque (no integral term) -- with the earlier kp=0.15 the
    // 0.08 N*m misalignment torque settled at ~60 deg residual error
    // (q_err,vec ~ tau_disturbance/kp, a real and correctly-behaving but
    // uninformative result for this demo). Higher gain shrinks that
    // residual; still well within the wheel cluster's combined torque
    // authority (~0.39 N*m) in steady state, so this isn't just masking
    // saturation.
    let law = AttitudeControlLaw::QuaternionPd(PdGains { kp: 1.5, kd: 15.0, pointing_db_rad: 0.0, rate_db_rads: 0.0 });
    let wheel_cluster = ReactionWheelCluster::four_wheel_pyramid(0.012, 628.3, 0.12, 0.8);
    let thruster_spec = ThrusterSpec::monoprop();
    let rcs_thrusters = sim_engine::actuators::build_rcs(&thruster_spec, 0.5);

    println!("Phase 13f verification: {BURN_DURATION_S:.0} s burn, misalignment offset={:?} m", burn.thrust_offset_body_m);
    println!("  Thrust: {:.0} N, Isp: {:.0} s, initial mass: {:.1} kg", burn.thrust_n, burn.isp_s, state.mass_kg);

    let mut rows: Vec<String> = vec![
        "t_s,mass_kg,dv_mps,pointing_err_deg,wheel_momentum_nms,rcs_duty_cycle".to_string(),
    ];

    let n_ticks = (BURN_DURATION_S / TICK_S).ceil() as usize;
    let v0 = state.v_mps;
    for _ in 0..n_ticks {
        let tau_cmd = law.command_torque(&state.q, &q_cmd, &state.omega_radps);
        let alloc = allocate(
            ControlMode::WheelsPrimary, tau_cmd, &wheel_cluster, &state.wheel_speeds_radps,
            &rcs_thrusters, TICK_S,
            sim_engine::MomentumManagementLaw::None,
        );
        let control_torque_body = net_body_torque(&wheel_cluster, &alloc);
        let h_wheel = wheel_cluster.total_momentum(&state.wheel_speeds_radps);

        let dv = (state.v_mps - v0).norm();
        let dot = state.q.dot(&q_cmd).clamp(-1.0, 1.0).abs();
        let pointing_err_deg = 2.0 * dot.acos().to_degrees();
        rows.push(format!(
            "{:.2},{:.4},{:.6},{:.4},{:.4},{:.3}",
            state.t_s, state.mass_kg, dv, pointing_err_deg, h_wheel.norm(), alloc.rcs_duty_cycle,
        ));

        let mut new_speeds = state.wheel_speeds_radps;
        let speed_dots = wheel_cluster.speed_dots(&alloc.wheel_motor_torque_nm);
        for i in 0..4 {
            // Real motor torque saturates at the wheel's rated max speed —
            // see cruise.rs's identical clamp for the real runaway this
            // prevents (found via cruise_commander_demo).
            new_speeds[i] = (new_speeds[i] + speed_dots[i] * TICK_S).clamp(-wheel_cluster.max_speed, wheel_cluster.max_speed);
        }

        state = step_tick_with_burn(
            &state, TICK_S, &sc, &bodies, MU_SUN, Some(&burn), control_torque_body, h_wheel, 1e-11, 1e-13,
        );
        state.wheel_speeds_radps = new_speeds;
    }

    let expected_mass = 1000.0 - burn.thrust_n / (burn.isp_s * orbital_models::constants::G0) * BURN_DURATION_S;
    let final_dv = (state.v_mps - v0).norm();
    let final_err_deg = 2.0 * state.q.dot(&q_cmd).clamp(-1.0, 1.0).abs().acos().to_degrees();
    println!("\n  Final mass: {:.3} kg (Tsiolkovsky prediction: {:.3} kg)", state.mass_kg, expected_mass);
    println!("  Final delta-v: {final_dv:.3} m/s");
    println!("  Final pointing error: {final_err_deg:.4} deg (controller fighting misalignment torque)");

    let path = "out/sixdof_burn_demo/sixdof_burn_demo.csv";
    std::fs::write(path, rows.join("\n") + "\n").expect("write csv");
    println!("\nSaved {path} ({n_ticks} rows)");
    println!("Plot: python plot/plot_sixdof_burn_demo.py");
}
