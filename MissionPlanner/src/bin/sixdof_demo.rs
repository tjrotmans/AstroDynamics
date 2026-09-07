//! Phase 13c verification demo — drives `sim_engine::propagator6dof::step_tick`
//! through several LEO-like orbits around a synthetic Earth-mass body, with
//! ZERO commanded control torque, so the only attitude dynamics visible are
//! the physics themselves: gravity-gradient + flat-plate SRP torque acting
//! on an asymmetric, bus-plus-panels spacecraft, over a translational
//! trajectory that must stay a clean circle (proving the tick-based
//! composition doesn't perturb translation).
//!
//! Uses the flat-plate SRP model (not cannonball) deliberately, so the SRP
//! torque channel in the output is nonzero and worth plotting — a cannonball
//! model would show SRP torque as exactly zero by construction (§4.1),
//! which is correct but not a useful verification case on its own.
//!
//! Per this repo's "every new capability gets a plot matched to what's being
//! verified" rule: this is what proves 13c actually works, not just that its
//! unit tests pass in isolation.
//!
//! Run:   cargo run -p mission_planner --bin sixdof_demo --release
//! Plots: python plot/plot_sixdof_demo.py        (torque/rate/health panels)
//!        python plot/plot_sixdof_demo_3d.py      (interactive 3D orbit + axes)

use nalgebra::{Vector3, Vector4};
use orbital_models::attitude::body_to_inertial;
use orbital_models::constants::MU_SUN;
use orbital_models::Plate;
use sim_engine::truth::{SpacecraftProperties, SrpTruthModel};
use sim_engine::{boresight, disturbance_torque_breakdown, step_tick, SixDofState};
use trajectory_solver::PropagatorBody;

const EARTH_MU: f64 = 3.986_004_418e14;
const EARTH_R: f64 = 6.378e6;
const ORBIT_ALT_M: f64 = 700_000.0; // ~700 km altitude, circular LEO
const TICK_S: f64 = 20.0;
const N_ORBITS: f64 = 3.0;

/// Simple bus (6 faces) + 2 symmetric solar panels, same geometry convention
/// as `MissionPlanner::simulate::build_plates` (Phase 13a) but hardcoded here
/// since this demo has no TOML config to read from.
fn demo_plates() -> Vec<Plate> {
    let (lx, ly, lz) = (2.0, 2.0, 0.63);
    let (bus_rho_s, bus_rho_d) = (0.30, 0.20);
    let (panel_rho_s, panel_rho_d) = (0.08, 0.10);
    let mut plates = vec![
        Plate { normal: Vector3::new(1.0, 0.0, 0.0), area: ly * lz, rho_s: bus_rho_s, rho_d: bus_rho_d, double_sided: false, center_body: Vector3::new(lx / 2.0, 0.0, 0.0) },
        Plate { normal: Vector3::new(-1.0, 0.0, 0.0), area: ly * lz, rho_s: bus_rho_s, rho_d: bus_rho_d, double_sided: false, center_body: Vector3::new(-lx / 2.0, 0.0, 0.0) },
        Plate { normal: Vector3::new(0.0, 1.0, 0.0), area: lx * lz, rho_s: bus_rho_s, rho_d: bus_rho_d, double_sided: false, center_body: Vector3::new(0.0, ly / 2.0, 0.0) },
        Plate { normal: Vector3::new(0.0, -1.0, 0.0), area: lx * lz, rho_s: bus_rho_s, rho_d: bus_rho_d, double_sided: false, center_body: Vector3::new(0.0, -ly / 2.0, 0.0) },
        Plate { normal: Vector3::new(0.0, 0.0, 1.0), area: lx * ly, rho_s: bus_rho_s, rho_d: bus_rho_d, double_sided: false, center_body: Vector3::new(0.0, 0.0, lz / 2.0) },
        Plate { normal: Vector3::new(0.0, 0.0, -1.0), area: lx * ly, rho_s: bus_rho_s, rho_d: bus_rho_d, double_sided: false, center_body: Vector3::new(0.0, 0.0, -lz / 2.0) },
    ];
    // One panel span longer than the other (asymmetric) so SRP produces a
    // real, non-cancelling torque -- a perfectly symmetric ±y panel pair
    // would mostly cancel, understating what a real deployed/asymmetric
    // panel configuration looks like.
    plates.push(Plate { normal: Vector3::new(0.0, 0.0, 1.0), area: 2.0, rho_s: panel_rho_s, rho_d: panel_rho_d, double_sided: true, center_body: Vector3::new(0.0, ly / 2.0 + 1.5, 0.0) });
    plates.push(Plate { normal: Vector3::new(0.0, 0.0, 1.0), area: 1.2, rho_s: panel_rho_s, rho_d: panel_rho_d, double_sided: true, center_body: Vector3::new(0.0, -(ly / 2.0 + 1.0), 0.0) });
    plates
}

fn main() {
    std::fs::create_dir_all("out/sixdof_demo").expect("create out dir");

    let earth_pos = Vector3::new(1.495_98e11, 0.0, 0.0); // fixed heliocentric position
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
    let period_s = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / EARTH_MU).sqrt();
    let duration_s = N_ORBITS * period_s;

    // Asymmetric bus (Izz notably larger than Ixx=Iyy) so gravity-gradient
    // torque has something to act on -- a spherically symmetric body would
    // show exactly zero torque (see propagator6dof's own unit tests).
    let sc = SpacecraftProperties {
        mass_kg: 1000.0,
        inertia_diag_kgm2: Vector3::new(150.0, 150.0, 400.0),
        srp: SrpTruthModel::FlatPlate { plates: demo_plates() },
        drag_area_m2: 4.0,
    };

    // Start with a small tilt away from local-vertical alignment and a small
    // initial spin, so the libration is visibly excited rather than sitting
    // at a (possibly unstable) equilibrium the whole run.
    let mut state = SixDofState {
        t_s: 0.0,
        r_m: earth_pos + Vector3::new(r_orbit, 0.0, 0.0),
        v_mps: Vector3::new(0.0, v_circ, 0.0),
        q: Vector4::new(0.9848, 0.0, 0.1736, 0.0), // ~20 deg tilt about body-y
        omega_radps: Vector3::new(0.0, 0.0, 0.002),
        wheel_speeds_radps: [0.0; 4],
        mass_kg: 1000.0,
    };

    let orbit_alt_km = ORBIT_ALT_M / 1e3;
    println!("Phase 13c verification: {N_ORBITS:.0} orbits @ {orbit_alt_km:.0} km altitude");
    println!("  Orbital period: {:.1} min, tick: {TICK_S:.0} s, total ticks: {:.0}",
             period_s / 60.0, duration_s / TICK_S);

    let mut rows: Vec<String> = vec![
        "t_s,r_x,r_y,r_z,alt_km,qw,qx,qy,qz,q_norm,\
         omega_x,omega_y,omega_z,omega_norm,\
         tau_gg_x,tau_gg_y,tau_gg_z,tau_srp_x,tau_srp_y,tau_srp_z,\
         boresight_x,boresight_y,boresight_z,\
         bodyx_x,bodyx_y,bodyx_z,bodyy_x,bodyy_y,bodyy_z,bodyz_x,bodyz_y,bodyz_z"
            .to_string(),
    ];

    let n_ticks = (duration_s / TICK_S).ceil() as usize;
    for _ in 0..n_ticks {
        let r_rel = state.r_m - earth_pos;
        let alt_km = (r_rel.norm() - EARTH_R) / 1e3;
        let b = boresight(&state);
        let torque = disturbance_torque_breakdown(&state, &sc, &bodies, MU_SUN);
        let bx = body_to_inertial(&state.q, &Vector3::new(1.0, 0.0, 0.0));
        let by = body_to_inertial(&state.q, &Vector3::new(0.0, 1.0, 0.0));
        let bz = body_to_inertial(&state.q, &Vector3::new(0.0, 0.0, 1.0));

        rows.push(format!(
            "{:.2},{:.3},{:.3},{:.3},{:.4},{:.8},{:.8},{:.8},{:.8},{:.10},\
             {:.8},{:.8},{:.8},{:.8},\
             {:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},\
             {:.6},{:.6},{:.6},\
             {:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            state.t_s, r_rel.x, r_rel.y, r_rel.z, alt_km,
            state.q[0], state.q[1], state.q[2], state.q[3], state.q.norm(),
            state.omega_radps.x, state.omega_radps.y, state.omega_radps.z, state.omega_radps.norm(),
            torque.gravity_gradient.x, torque.gravity_gradient.y, torque.gravity_gradient.z,
            torque.srp.x, torque.srp.y, torque.srp.z,
            b.x, b.y, b.z,
            bx.x, bx.y, bx.z, by.x, by.y, by.z, bz.x, bz.y, bz.z,
        ));

        state = step_tick(
            &state, TICK_S, &sc, &bodies, MU_SUN,
            Vector3::zeros(), Vector3::zeros(), // zero commanded torque -- physics only
            1e-10, 1e-12,
        );
    }

    let final_alt_km = ((state.r_m - earth_pos).norm() - EARTH_R) / 1e3;
    println!("  Final altitude: {final_alt_km:.2} km (started at {orbit_alt_km:.2} km)");
    println!("  Final |q|: {:.10} (should be 1.0)", state.q.norm());
    println!("  Final |omega|: {:.6} rad/s", state.omega_radps.norm());

    let path = "out/sixdof_demo/sixdof_demo.csv";
    std::fs::write(path, rows.join("\n") + "\n").expect("write csv");
    println!("\nSaved {path} ({} rows)", n_ticks);
    println!("Plots: python plot/plot_sixdof_demo.py");
    println!("       python plot/plot_sixdof_demo_3d.py");
}
