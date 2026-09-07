//! Phase 13a/13b/13j verification demo — spacecraft geometry (13a) and a
//! per-source translational + rotational force/torque breakdown (13b) over a
//! real heliocentric leg with a nearby (Earth) and a distant (Jupiter)
//! third-body perturber, logged for verification-style plotting (13j).
//!
//! Two passes over the IDENTICAL trajectory, differing only in the SRP model
//! — because the current physics genuinely cannot show a nonzero SRP force
//! on translation and a nonzero SRP torque from the same model at once (see
//! `propagator6dof`'s own doc comment / `docs/MP/MANUAL.md` §7): the
//! decoupled path (`step_tick`) never applies SRP force to translation
//! regardless of model, and Cannonball SRP has no torque by construction
//! (attitude-independent, §4.1). This demo makes that boundary visible
//! rather than hiding it:
//!
//!   - Pass A (Cannonball): a zero-thrust "burn" tick
//!     (`step_tick_with_burn` with `thrust_n: 0.0`) opts into the fully-
//!     coupled integrator so Cannonball SRP genuinely acts on translation —
//!     real nonzero SRP acceleration, zero SRP torque (as expected).
//!   - Pass B (FlatPlate): the ordinary decoupled `step_tick` path — real
//!     nonzero SRP torque (asymmetric bus+panels+dish geometry), zero SRP
//!     force on translation (as expected, and reported honestly, not
//!     estimated).
//!
//! Spacecraft geometry (13a): bus (6 faces) + 2 asymmetric solar panels —
//! logged once to `sixdof_geometry.csv` for a 3D layout plot.
//!
//! Third-body verification (13b): Earth starts close (~0.05 AU offset,
//! outside its own SOI so it never becomes central) and the spacecraft
//! departs radially outward over ~150 days, so Earth's third-body pull
//! should visibly decay; Jupiter is placed far off the departure path so its
//! contribution should stay many orders of magnitude below Earth's and the
//! Sun's central term throughout — both are checkable claims, not just
//! plausible-looking numbers.
//!
//! Run:   cargo run -p mission_planner --bin sixdof_force_breakdown_demo --release
//! Plots: python plot/plot_spacecraft_geometry.py
//!        python plot/plot_force_torque_breakdown.py

use nalgebra::{Vector3, Vector4};
use orbital_models::constants::{MU_EARTH, MU_JUPITER, MU_SUN};
use orbital_models::Plate;
use sim_engine::truth::{SpacecraftProperties, SrpTruthModel};
use sim_engine::{
    disturbance_torque_breakdown, step_tick, step_tick_with_burn, translational_accel_breakdown,
    BurnConfig, SixDofState,
};
use trajectory_solver::PropagatorBody;

const TICK_S: f64 = 86_400.0; // 1 day -- a slow heliocentric coast, coarse ticks are physically fine
const DURATION_DAYS: f64 = 150.0;
const AU: f64 = 1.495_98e11;

/// Bus (6 faces) + 2 asymmetric solar panels — same geometry convention as
/// `MissionPlanner::simulate::build_plates` (Phase 13a), hardcoded here since
/// this demo has no TOML config to read from.
fn demo_plates() -> Vec<Plate> {
    let (lx, ly, lz) = (2.4, 2.0, 1.4);
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
    // Asymmetric panel span (one longer than the other) -- a symmetric pair
    // would mostly cancel torque, understating what a real deployed
    // configuration looks like (same reasoning as the 13c demo).
    plates.push(Plate { normal: Vector3::new(0.0, 0.0, 1.0), area: 3.5, rho_s: panel_rho_s, rho_d: panel_rho_d, double_sided: true, center_body: Vector3::new(0.0, ly / 2.0 + 1.8, 0.0) });
    plates.push(Plate { normal: Vector3::new(0.0, 0.0, 1.0), area: 2.0, rho_s: panel_rho_s, rho_d: panel_rho_d, double_sided: true, center_body: Vector3::new(0.0, -(ly / 2.0 + 1.2), 0.0) });
    plates
}

fn write_geometry_csv(plates: &[Plate], inertia: &Vector3<f64>, mass_kg: f64) {
    let mut rows = vec!["plate_idx,normal_x,normal_y,normal_z,area_m2,rho_s,rho_d,double_sided,\
                          center_x,center_y,center_z"
        .to_string()];
    for (i, p) in plates.iter().enumerate() {
        rows.push(format!(
            "{},{:.6},{:.6},{:.6},{:.4},{:.3},{:.3},{},{:.4},{:.4},{:.4}",
            i, p.normal.x, p.normal.y, p.normal.z, p.area, p.rho_s, p.rho_d,
            p.double_sided as u8, p.center_body.x, p.center_body.y, p.center_body.z,
        ));
    }
    let path = "out/sixdof_force_breakdown_demo/sixdof_geometry.csv";
    std::fs::write(path, rows.join("\n") + "\n").expect("write geometry csv");
    println!("Saved {path} ({} plates)", plates.len());
    println!(
        "Mass: {mass_kg:.1} kg, inertia diag [{:.1}, {:.1}, {:.1}] kg*m^2",
        inertia.x, inertia.y, inertia.z
    );
}

fn run_pass(
    label: &str,
    srp: SrpTruthModel,
    use_coupled_zero_thrust_burn: bool,
    bodies: &[PropagatorBody],
    inertia: Vector3<f64>,
    r0: Vector3<f64>,
    v0: Vector3<f64>,
) -> Vec<String> {
    let sc = SpacecraftProperties { mass_kg: 1000.0, inertia_diag_kgm2: inertia, srp, drag_area_m2: 4.0 };
    let mut state = SixDofState {
        t_s: 0.0, r_m: r0, v_mps: v0,
        q: Vector4::new(0.9848, 0.0, 0.1736, 0.0), // ~20 deg tilt, same convention as 13c demo
        omega_radps: Vector3::new(0.0, 0.0, 0.0005),
        wheel_speeds_radps: [0.0; 4],
        mass_kg: 1000.0,
    };
    let zero_thrust_burn = BurnConfig {
        thrust_n: 0.0,
        isp_s: 300.0, // unused at thrust_n=0.0 (mass_dot = 0), kept physically sane
        body_dir: Vector3::new(1.0, 0.0, 0.0),
        thrust_offset_body_m: Vector3::zeros(),
    };

    let n_ticks = (DURATION_DAYS * 86_400.0 / TICK_S).ceil() as usize;
    let mut rows = Vec::with_capacity(n_ticks);
    for _ in 0..n_ticks {
        let accel = translational_accel_breakdown(&state, &sc, bodies, MU_SUN, use_coupled_zero_thrust_burn);
        let torque = disturbance_torque_breakdown(&state, &sc, bodies, MU_SUN);
        let dist_sun_au = state.r_m.norm() / AU;

        rows.push(format!(
            "{},{:.6},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
            label, dist_sun_au,
            accel.central_gravity.norm(), accel.third_body.norm(), accel.srp.norm(), accel.total().norm(),
            torque.gravity_gradient.norm(), torque.srp.norm(), torque.disturbance_total().norm(),
            state.q.norm(), state.omega_radps.norm(),
            state.r_m.x, state.r_m.y, state.r_m.z,
        ));

        state = if use_coupled_zero_thrust_burn {
            step_tick_with_burn(
                &state, TICK_S, &sc, bodies, MU_SUN, Some(&zero_thrust_burn),
                Vector3::zeros(), Vector3::zeros(), 1e-9, 1e-11,
            )
        } else {
            step_tick(&state, TICK_S, &sc, bodies, MU_SUN, Vector3::zeros(), Vector3::zeros(), 1e-9, 1e-11)
        };
    }
    rows
}

fn main() {
    std::fs::create_dir_all("out/sixdof_force_breakdown_demo").expect("create out dir");

    let inertia = Vector3::new(180.0, 200.0, 420.0); // asymmetric, per the 13c/13f demo convention

    // Earth close to the departure point (outside its own SOI so it stays a
    // third-body perturber, never central) -- Jupiter far off the departure
    // path so its pull should stay negligible throughout, a checkable claim.
    let earth_pos = Vector3::new(AU, 0.0, 0.0);
    let earth_state_at = move |_t: f64| (earth_pos, Vector3::zeros());
    let jupiter_pos = Vector3::new(-5.2 * AU, 0.0, 1.0 * AU);
    let jupiter_state_at = move |_t: f64| (jupiter_pos, Vector3::zeros());
    let bodies = vec![
        PropagatorBody { name: "Earth", mu_m3s2: MU_EARTH, soi_radius_m: Some(9.24e8), state_at: &earth_state_at, central_fidelity: None, radius_m: Some(6.378e6) },
        PropagatorBody { name: "Jupiter", mu_m3s2: MU_JUPITER, soi_radius_m: Some(4.83e10), state_at: &jupiter_state_at, central_fidelity: None, radius_m: Some(7.149e7) },
    ];

    // Departure state: ~0.05 AU outside Earth (well outside its SOI), radial
    // outward velocity added on top of Earth's own circular heliocentric
    // speed, so the spacecraft genuinely departs over the run.
    let v_earth_circ = (MU_SUN / AU).sqrt();
    let r0 = earth_pos + Vector3::new(0.05 * AU, 0.0, 0.0);
    let v0 = Vector3::new(300.0, v_earth_circ * 0.97, 0.0); // small radial kick + slightly sub-circular tangential

    println!("Phase 13a/13b/13j verification: {DURATION_DAYS:.0}-day heliocentric departure leg");
    println!("  Earth offset at t=0: {:.4} AU, Jupiter offset: {:.2} AU", (r0 - earth_pos).norm() / AU, (jupiter_pos - r0).norm() / AU);

    write_geometry_csv(&demo_plates(), &inertia, 1000.0);

    let mut rows = vec![
        "srp_model,dist_sun_au,accel_central_ms2,accel_thirdbody_ms2,accel_srp_ms2,accel_total_ms2,\
         torque_gg_nm,torque_srp_nm,torque_total_nm,q_norm,omega_norm,r_x,r_y,r_z"
            .to_string(),
    ];

    // Pass A: Cannonball, coupled (zero-thrust burn) path -- real SRP accel, zero SRP torque.
    rows.extend(run_pass(
        "cannonball", SrpTruthModel::Cannonball { c_r: 1.4, area_m2: 12.0 }, true,
        &bodies, inertia, r0, v0,
    ));
    // Pass B: FlatPlate, decoupled path -- zero SRP accel (honest), real SRP torque.
    rows.extend(run_pass(
        "flatplate", SrpTruthModel::FlatPlate { plates: demo_plates() }, false,
        &bodies, inertia, r0, v0,
    ));

    let path = "out/sixdof_force_breakdown_demo/force_torque_breakdown.csv";
    std::fs::write(path, rows.join("\n") + "\n").expect("write breakdown csv");
    println!("Saved {path} ({} rows)", rows.len() - 1);
    println!("Plots: python plot/plot_spacecraft_geometry.py");
    println!("       python plot/plot_force_torque_breakdown.py");
}
