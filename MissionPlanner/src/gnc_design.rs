//! GNC design stage — disturbance torque sizing, reaction wheel and RCS
//! sizing, sensor grade selection, and EKF state-dimension sizing from a
//! mission's `[gnc]`/`[spacecraft]` configuration.
//!
//! Mirrors `design.rs`'s shape: a `run(cfg)` entry point, report-printing,
//! and a CSV writer. Hardware is selected from `hardware_catalog`'s concrete
//! specs by linear scan — no trait object, same "concrete types dispatched
//! directly" philosophy as the Phase 2 trajectory solvers.
//!
//! Sizing only runs when the mission defines a closed orbit
//! (`[trajectory.capture].target_orbit_radius_m`) — the per-orbit momentum
//! integral and cyclic disturbance-torque model below don't apply to a
//! hyperbolic flyby, so `Flyby`-objective missions are skipped with a notice
//! rather than treated as an error.
//!
//! Sizing sequence:
//! 1. Worst-case disturbance torque: gravity-gradient (attitude-independent
//!    bound) + SRP (worst-case sun-facing area at the body's heliocentric
//!    distance)
//! 2. Reaction wheel selection: peak torque (with margin) and per-orbit
//!    momentum storage (quarter-period sinusoidal accumulation)
//! 3. RCS desaturation propellant budget: one full desaturation per orbit
//! 4. OpNav camera grade selection from the position accuracy requirement
//!    (range × bearing-noise small-angle relationship)
//! 5. EKF state dimension from SRP model + hardware configuration

use std::fs;

use nalgebra::Vector3;
use orbital_models::gravity_gradient_max;
use hardware_catalog::{OpNavCameraSpec, ReactionWheelSpec, ThrusterSpec};

use crate::config::{HardwareItem, MissionConfig, SrpModel};

/// Heliocentric distances of the major planets [AU] — IAU mean semi-major
/// axes, used only as an SRP-intensity approximation for bodies without
/// explicit Keplerian elements. Not a navigation-grade ephemeris lookup
/// (see `design.rs`'s ANISE-backed body states for that).
const PLANET_AU: &[(&str, f64)] = &[
    ("mercury", 0.387),
    ("venus", 0.723),
    ("earth", 1.000),
    ("moon", 1.000),
    ("mars", 1.524),
    ("jupiter", 5.203),
];

/// Solar radiation pressure intensity exponent's reference distance is 1 AU
/// (`orbital_models::constants::P_SRP`); typical absorptive-to-reflective
/// mixed spacecraft surface coefficient, SMAD (Wertz & Larson) range 1.2–1.5.
const SRP_CR_DEFAULT: f64 = 1.4;

/// Standard preliminary ACS sizing margin on peak disturbance torque (SMAD).
const TORQUE_MARGIN: f64 = 2.0;

/// Typical operational desaturation trigger — fraction of max wheel momentum
/// storage at which desaturation is performed (Wertz & Larson SMAD).
const DESAT_FRACTION: f64 = 0.8;

/// Minimum number of discrete pulses a selected desaturation thruster must
/// be able to deliver the total per-orbit impulse in — added
/// (catalog widening exposed a real gap this constant
/// fixes, see `select_thruster`'s own doc comment). A real desaturation
/// burn needs FINE modulation, not one coarse pulse — too few, too-large
/// pulses risk overshoot/limit-cycling around the target momentum. 20
/// pulses is a conservative, documentable "clearly fine enough" bound, not
/// derived from a specific control-law stability margin.
const MIN_DESAT_PULSES: f64 = 20.0;

/// GNC design stage output — feeds the Phase 4 simulation config and the
/// `/api/design/gnc` HTTP response.
#[derive(Debug, serde::Serialize)]
pub struct GncDesign {
    pub orbit_period_s: f64,
    pub gravity_gradient_torque_nm: f64,
    pub srp_torque_nm: f64,
    pub total_disturbance_torque_nm: f64,
    pub peak_momentum_nms: f64,
    pub selected_wheel: String,
    pub wheel_count: u32,
    pub wheel_torque_margin_ok: bool,
    pub selected_thruster: String,
    pub propellant_per_orbit_kg: f64,
    pub selected_opnav: String,
    pub achieved_position_accuracy_m: f64,
    pub achieved_velocity_accuracy_mps: f64,
    pub ekf_state_dim: usize,
}

// ── Entry point ──────────────────────────────────────────────────────────────

pub fn run(cfg: &MissionConfig) {
    println!("\nGNC design: {}", cfg.mission.name);

    let Some(r) = orbit_radius_m(cfg) else {
        println!(
            "  No [trajectory.capture].target_orbit_radius_m defined — GNC actuator/momentum \
             sizing needs a closed orbit, skipping. (Flyby missions have no cyclic disturbance \
             torque to size against.)"
        );
        return;
    };

    let mu = cfg.target_body.mu_m3s2;
    let inertia = Vector3::new(
        cfg.spacecraft.inertia_diag_kgm2[0],
        cfg.spacecraft.inertia_diag_kgm2[1],
        cfg.spacecraft.inertia_diag_kgm2[2],
    );

    let orbit_period_s = 2.0 * std::f64::consts::PI * (r.powi(3) / mu).sqrt();

    let tau_gg = gravity_gradient_max(mu, r, &inertia);
    let tau_srp = srp_torque_max(cfg);
    let tau_total = tau_gg + tau_srp;

    // Per-orbit peak momentum: cyclic once-per-orbit disturbance torque,
    // accumulated over a quarter period —
    //   ∫₀^(T/4) τ·sin(ωt) dt = τ/ω = τ·T/(2π),  ω = 2π/T
    let peak_momentum_nms = tau_total * orbit_period_s / (2.0 * std::f64::consts::PI);

    println!("\nDisturbance torque  (worst case, attitude-independent)");
    println!("  Orbit radius:        {r:.1} m   Orbit period: {:.1} min", orbit_period_s / 60.0);
    println!("  Gravity-gradient:    {tau_gg:.3e} N·m");
    println!("  SRP:                 {tau_srp:.3e} N·m");
    println!("  Total:               {tau_total:.3e} N·m");
    println!("  Peak momentum/orbit: {peak_momentum_nms:.3e} N·m·s");

    let (wheel, wheel_margin_ok) = select_reaction_wheel(tau_total, peak_momentum_nms);
    println!("\nReaction wheel selection  (4-wheel pyramid, {TORQUE_MARGIN:.0}x torque margin)");
    println!(
        "  {}  τ_max={:.3} N·m  H_max={:.3} N·m·s{}",
        wheel.name, wheel.max_torque_nm, wheel.max_momentum_nms(),
        if wheel_margin_ok { "" } else { "  ⚠ insufficient even at largest catalog size" },
    );

    let (thruster, propellant_per_orbit_kg) = select_thruster(cfg, peak_momentum_nms);
    let orbits_per_day = 86_400.0 / orbit_period_s;
    println!("\nRCS desaturation budget  (1 full desaturation/orbit, conservative)");
    println!(
        "  {}  propellant/orbit: {propellant_per_orbit_kg:.4} kg   ({:.4} kg/day, {orbits_per_day:.1} orbits/day)",
        thruster.name, propellant_per_orbit_kg * orbits_per_day,
    );

    let position_req_m = cfg.gnc.position_accuracy_req_m.unwrap_or(100.0);
    let velocity_req_mps = cfg.gnc.velocity_accuracy_req_mps.unwrap_or(0.01);
    let (opnav, achieved_pos_m) = select_opnav_camera(r, position_req_m);
    let dt_meas_s = cfg.simulation.dt_meas_s;
    // Velocity-from-position finite difference: v = (r2 - r1)/Δt,
    // Var(v) = 2σ_pos²/Δt²  (two independent position errors per difference)
    let achieved_vel_mps = (2.0_f64).sqrt() * achieved_pos_m / dt_meas_s;

    println!("\nNavigation sensor selection");
    println!(
        "  Position req: {position_req_m:.1} m   →  {}  (bearing σ={:.1e} rad, range {r:.0} m → {achieved_pos_m:.2} m)",
        opnav.name, opnav.bearing_noise_rad,
    );
    println!(
        "  Velocity req: {velocity_req_mps:.4} m/s →  achieved ≈ {achieved_vel_mps:.4} m/s  (Δt_meas={dt_meas_s:.0} s){}",
        if achieved_vel_mps > velocity_req_mps { "  ⚠ requirement not met — consider shorter dt_meas_s" } else { "" },
    );

    let ekf_state_dim = ekf_state_dim(cfg);
    println!("\nEKF sizing");
    println!(
        "  State dimension: {ekf_state_dim}  ({})",
        if ekf_state_dim == 10 {
            "r, v, C_SRP, 3x Gauss-Markov stochastic accel — cannonball SRP + OpNav present"
        } else {
            "r, v only — no cannonball SRP + OpNav combination configured"
        }
    );
    println!("  R (bearing):      {:.3e} rad²", opnav.bearing_noise_rad.powi(2));
    println!("  R (angular size): {:.3e} rad²", opnav.angular_size_noise_rad.powi(2));
    // Discrete random-walk variance growth rate: Q ≈ σ_v² · Δt
    let q_pos_rw = velocity_req_mps.powi(2) * dt_meas_s;
    println!("  Q (position RW):  {q_pos_rw:.3e} m²  (σ_v²·Δt_meas)");

    let design = GncDesign {
        orbit_period_s,
        gravity_gradient_torque_nm: tau_gg,
        srp_torque_nm: tau_srp,
        total_disturbance_torque_nm: tau_total,
        peak_momentum_nms,
        selected_wheel: wheel.name.to_string(),
        wheel_count: 4,
        wheel_torque_margin_ok: wheel_margin_ok,
        selected_thruster: thruster.name.to_string(),
        propellant_per_orbit_kg,
        selected_opnav: opnav.name.to_string(),
        achieved_position_accuracy_m: achieved_pos_m,
        achieved_velocity_accuracy_mps: achieved_vel_mps,
        ekf_state_dim,
    };
    write_output(cfg, &design);
}

// ── Disturbance torque ──────────────────────────────────────────────────────

fn heliocentric_distance_au(cfg: &MissionConfig) -> f64 {
    if let Some(k) = &cfg.target_body.keplerian_orbit {
        return k.sma_au;
    }
    let name = cfg.target_body.name.to_lowercase();
    PLANET_AU.iter().find(|(n, _)| *n == name).map(|(_, au)| *au).unwrap_or(1.0)
}

/// Worst-case SRP disturbance torque [N·m] for reaction-wheel/RCS sizing.
///
/// Two modes, selected by whether `cp_cg_offset_m` is explicitly set:
///
/// - **Set**: the original scalar estimate — max sun-facing cross-section
///   (largest bus face + any `SolarPanel` area) times `SRP_CR_DEFAULT`,
///   times the configured center-of-pressure/center-of-mass offset.
/// Unchanged, byte-for-byte, from before this function's 
///   rework — preserved for missions that want to override the derived
///   geometry with a hand-specified worst-case offset (and so the existing
///   `srp_torque_max_matches_hand_calc` fixture, which sets this field,
///   needed no change).
/// - **Unset (default)**: a real, per-plate worst-case torque using the
///   SAME plate geometry (`crate::simulate::build_plates` — bus faces plus
///   any placed `SolarPanel`/`CustomPlate` hardware) and the SAME physics
///   (`orbital_models::flat_plate_torque_body`) `/api/design/vehicle`'s
///   `srp` section already uses (
///   the old scalar estimate always ignored `CustomPlate`
///   geometry entirely, so a builder user placing one saw no effect on this
///   number). Each plate's torque arm is measured from the REAL derived
///   center of mass (`crate::vehicle_properties::compute_vehicle_
///   properties`), not the old fixed `0.1 × max(bus_dim)` SMAD-default
///   offset. Worst case is found by sampling sun direction over a
///   near-uniform Fibonacci-sphere grid and taking the maximum |τ| — with
///   an arbitrary placed plate, the direction maximizing net torque has no
///   general closed form, so this is a direct search over the same
///   validated force law rather than a new approximation.
fn srp_torque_max(cfg: &MissionConfig) -> f64 {
    if let Some(offset_m) = cfg.spacecraft.cp_cg_offset_m {
        let d = cfg.spacecraft.bus_dims_m;
        let bus_max_face = (d[0] * d[1]).max(d[1] * d[2]).max(d[0] * d[2]);
        let panel_area: f64 = cfg
            .spacecraft
            .hardware
            .iter()
            .filter_map(|h| match h {
                HardwareItem::SolarPanel { area_m2, .. } => Some(*area_m2),
                _ => None,
            })
            .sum();
        let area_m2 = bus_max_face + panel_area;

        let d_au = heliocentric_distance_au(cfg);
        let p_local = orbital_models::constants::P_SRP / (d_au * d_au);
        let f_srp = p_local * area_m2 * SRP_CR_DEFAULT;
        return f_srp * offset_m;
    }

    let plates = crate::simulate::build_plates(cfg);
    let com = crate::vehicle_properties::compute_vehicle_properties(cfg).com_m;
    let com_v = Vector3::new(com[0], com[1], com[2]);
    // `build_plates` only pre-shifts `center_body` onto the real CoM when
    // `derive_inertia_from_geometry` is on (see its own doc comment) — do
    // the shift here otherwise, so the torque arm is always CoM-relative
    // regardless of that unrelated flag.
    let already_com_relative = cfg.spacecraft.derive_inertia_from_geometry.unwrap_or(false);
    let com_plates: Vec<sim_engine::Plate> = plates
        .into_iter()
        .map(|mut p| {
            if !already_com_relative {
                p.center_body -= com_v;
            }
            p
        })
        .collect();

    let d_au = heliocentric_distance_au(cfg);
    let p_local = orbital_models::constants::P_SRP / (d_au * d_au);

    const N_SAMPLES: usize = 200;
    let mut tau_max = 0.0_f64;
    for i in 0..N_SAMPLES {
        let sun_hat = fibonacci_sphere_point(i, N_SAMPLES);
        let tau = orbital_models::flat_plate_torque_body(&com_plates, &sun_hat, p_local);
        tau_max = tau_max.max(tau.norm());
    }
    tau_max
}

/// Near-uniform point `i` of `n` on the unit sphere (golden-angle spiral) —
/// used by [`srp_torque_max`]'s sun-direction search.
fn fibonacci_sphere_point(i: usize, n: usize) -> Vector3<f64> {
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let y = 1.0 - 2.0 * (i as f64) / ((n.max(2) - 1) as f64);
    let radius = (1.0 - y * y).max(0.0).sqrt();
    let theta = golden_angle * i as f64;
    Vector3::new(theta.cos() * radius, y, theta.sin() * radius)
}

// ── Reaction wheel selection ────────────────────────────────────────────────

fn select_reaction_wheel(tau_total: f64, peak_momentum_nms: f64) -> (ReactionWheelSpec, bool) {
    let catalog = ReactionWheelSpec::catalog();
    for wheel in catalog {
        if wheel.max_torque_nm >= tau_total * TORQUE_MARGIN
            && wheel.max_momentum_nms() * DESAT_FRACTION >= peak_momentum_nms
        {
            return (wheel, true);
        }
    }
    (catalog[catalog.len() - 1], false)
}

// ── RCS desaturation ─────────────────────────────────────────────────────────

/// Picks the catalog thruster minimizing propellant mass for a one-orbit
/// desaturation impulse, and returns that propellant mass [kg].
///
/// Conservative assumption: one full desaturation (dumping the entire
/// per-orbit peak momentum) per orbit. Moment arm assumed at the bus's
/// largest half-dimension (typical corner-mounted RCS placement).
/// Selects the thruster minimizing propellant for the per-orbit desaturation
/// impulse, subject to a real fitness gate: `min_impulse_bit = thrust_n *
/// min_pulse_s` must be small enough to deliver the total impulse in at
/// least `MIN_DESAT_PULSES` discrete pulses (`min_impulse_bit <= impulse_ns
/// / MIN_DESAT_PULSES`) — a coarse, high-thrust thruster minimizes
/// propellant on Isp alone but cannot MODULATE finely enough for a real
/// desaturation burn without risking overshoot. (Found during the
/// catalog widening): before this gate, `select_thruster` picked
/// purely by Isp with no regard for whether the resulting thruster's own
/// thrust/impulse-bit made physical sense for the desaturation being sized
/// — harmless with the old 2-entry catalog (`Monoprop` was always both the
/// highest-Isp AND a reasonable size), but a real bug once a coarser,
/// slightly-higher-Isp class (`Monoprop-MONARC5class`) was added: it won
/// on Isp alone despite being ~4x `Monoprop`'s thrust, which for a small
/// mission's fine desaturation impulse would genuinely overshoot in
/// practice. Falls back to the smallest-min-impulse-bit catalog entry
/// (finest available control) if NONE qualify — same fallback shape as
/// `select_opnav_camera`'s "falls back to the finest grade" convention.
fn select_thruster(cfg: &MissionConfig, peak_momentum_nms: f64) -> (ThrusterSpec, f64) {
    let moment_arm_m = 0.5 * cfg.spacecraft.bus_dims_m.iter().cloned().fold(0.0_f64, f64::max);
    let impulse_ns = peak_momentum_nms / moment_arm_m;
    let max_impulse_bit_ns = impulse_ns / MIN_DESAT_PULSES;

    let propellant_kg = |t: &ThrusterSpec| impulse_ns / (t.isp_s * orbital_models::constants::G0);

    let qualifying = ThrusterSpec::catalog()
        .into_iter()
        .filter(|t| t.thrust_n * t.min_pulse_s <= max_impulse_bit_ns)
        .map(|t| { let p = propellant_kg(&t); (t, p) })
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

    qualifying.unwrap_or_else(|| {
        let t = ThrusterSpec::catalog()
            .into_iter()
            .min_by(|a, b| (a.thrust_n * a.min_pulse_s).partial_cmp(&(b.thrust_n * b.min_pulse_s)).unwrap())
            .unwrap();
        let p = propellant_kg(&t);
        (t, p)
    })
}

// ── Sensor selection ─────────────────────────────────────────────────────────

/// Selects the coarsest OpNav camera grade meeting the position accuracy
/// requirement via the small-angle relationship σ_position ≈ range·σ_bearing
/// (range approximated as the orbit radius). Falls back to the finest grade
/// if even that doesn't meet the requirement.
fn select_opnav_camera(range_m: f64, position_req_m: f64) -> (OpNavCameraSpec, f64) {
    let catalog = OpNavCameraSpec::catalog(); // fine → medium → coarse
    let mut best = (catalog[0], range_m * catalog[0].bearing_noise_rad);
    for cam in catalog {
        let achieved = range_m * cam.bearing_noise_rad;
        if achieved <= position_req_m {
            best = (cam, achieved);
        }
    }
    best
}

// ── EKF sizing ───────────────────────────────────────────────────────────────

pub(crate) fn ekf_state_dim(cfg: &MissionConfig) -> usize {
    let cannonball = matches!(cfg.spacecraft.srp_model, SrpModel::Cannonball);
    let has_opnav = cfg
        .spacecraft
        .hardware
        .iter()
        .any(|h| matches!(h, HardwareItem::OpNavCamera { .. }));
    if cannonball && has_opnav { 10 } else { 6 }
}

// ── File output ──────────────────────────────────────────────────────────────

fn write_output(cfg: &MissionConfig, d: &GncDesign) {
    let out_dir = format!("{}/design", cfg.simulation.output_dir.trim_end_matches('/'));
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("\nWarning: could not create output dir '{out_dir}': {e}");
        return;
    }

    let path = format!("{out_dir}/gnc_design.csv");
    let csv = format!(
        "orbit_period_s,gravity_gradient_torque_nm,srp_torque_nm,total_disturbance_torque_nm,\
         peak_momentum_nms,selected_wheel,wheel_count,wheel_torque_margin_ok,selected_thruster,\
         propellant_per_orbit_kg,selected_opnav,achieved_position_accuracy_m,\
         achieved_velocity_accuracy_mps,ekf_state_dim\n\
         {:.4},{:.6e},{:.6e},{:.6e},{:.6e},{},{},{},{},{:.6},{},{:.4},{:.6},{}\n",
        d.orbit_period_s,
        d.gravity_gradient_torque_nm,
        d.srp_torque_nm,
        d.total_disturbance_torque_nm,
        d.peak_momentum_nms,
        d.selected_wheel,
        d.wheel_count,
        d.wheel_torque_margin_ok,
        d.selected_thruster,
        d.propellant_per_orbit_kg,
        d.selected_opnav,
        d.achieved_position_accuracy_m,
        d.achieved_velocity_accuracy_mps,
        d.ekf_state_dim,
    );
    match fs::write(&path, csv) {
        Ok(_) => println!("\n  {path}"),
        Err(e) => eprintln!("\n  Warning: could not write gnc_design.csv: {e}"),
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn orbit_radius_m(cfg: &MissionConfig) -> Option<f64> {
    cfg.trajectory.capture.as_ref()?.target_orbit_radius_m
}

// ── API entry point (no I/O, no side-effects) ─────────────────────────────────

/// Run the full GNC sizing chain and return the result without printing or
/// writing files. Used by the `/api/design/gnc` HTTP endpoint.
#[allow(dead_code)]
pub fn compute(cfg: &MissionConfig) -> Result<GncDesign, String> {
    let r = orbit_radius_m(cfg).ok_or_else(|| {
        "no [trajectory.capture].target_orbit_radius_m — GNC sizing requires a closed orbit"
            .to_string()
    })?;

    let mu = cfg.target_body.mu_m3s2;
    let inertia = nalgebra::Vector3::new(
        cfg.spacecraft.inertia_diag_kgm2[0],
        cfg.spacecraft.inertia_diag_kgm2[1],
        cfg.spacecraft.inertia_diag_kgm2[2],
    );
    let orbit_period_s = 2.0 * std::f64::consts::PI * (r.powi(3) / mu).sqrt();
    let tau_gg = orbital_models::gravity_gradient_max(mu, r, &inertia);
    let tau_srp = srp_torque_max(cfg);
    let tau_total = tau_gg + tau_srp;
    let peak_momentum_nms = tau_total * orbit_period_s / (2.0 * std::f64::consts::PI);

    let (wheel, wheel_margin_ok) = select_reaction_wheel(tau_total, peak_momentum_nms);
    let (thruster, propellant_per_orbit_kg) = select_thruster(cfg, peak_momentum_nms);

    let position_req_m = cfg.gnc.position_accuracy_req_m.unwrap_or(100.0);
    let (opnav, achieved_pos_m) = select_opnav_camera(r, position_req_m);
    let achieved_vel_mps = (2.0_f64).sqrt() * achieved_pos_m / cfg.simulation.dt_meas_s;

    Ok(GncDesign {
        orbit_period_s,
        gravity_gradient_torque_nm: tau_gg,
        srp_torque_nm: tau_srp,
        total_disturbance_torque_nm: tau_total,
        peak_momentum_nms,
        selected_wheel: wheel.name.to_string(),
        wheel_count: 4,
        wheel_torque_margin_ok: wheel_margin_ok,
        selected_thruster: thruster.name.to_string(),
        propellant_per_orbit_kg,
        selected_opnav: opnav.name.to_string(),
        achieved_position_accuracy_m: achieved_pos_m,
        achieved_velocity_accuracy_mps: achieved_vel_mps,
        ekf_state_dim: ekf_state_dim(cfg),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AtmosphereModel, CaptureConfig, EphemerisSource, GncConfig, GravityModel, Integrator,
        MissionMeta, MissionObjective, MissionPhase, NavigationFilter, PointingMode,
        SimulationConfig, SpacecraftConfig, TargetBodyConfig, TrajectoryConfig, TrajectorySolver,
        AttitudeController,
    };

    /// Reproduces `config/apophis_orbit.toml` as a Rust struct literal, so the
    /// Phase 3 sizing chain can be asserted against the numbers hand-verified
    /// in session: tau_gg=9.526e-9, tau_srp=1.201e-5, peak_momentum=2.611 N·m·s,
    /// RW-Medium, Monoprop, OpNav-Coarse (0.25 m), EKF state dim 10.
    fn apophis_fixture() -> MissionConfig {
        MissionConfig {
            mission: MissionMeta { name: "Apophis Proximity Orbit".into(), objective: MissionObjective::Orbit },
            target_body: TargetBodyConfig {
                name: "Apophis".into(),
                mu_m3s2: 2.646e-3,
                radius_m: 185.0,
                gravity_model: GravityModel::PointMass,
                atmosphere: AtmosphereModel::None,
                j2: None,
                j3: None,
                j4: None,
                ephemeris: EphemerisSource::Keplerian,
                keplerian_orbit: None,
                third_bodies: vec![],
                pole_ra_deg: None,
                pole_dec_deg: None,
            },
            spacecraft: SpacecraftConfig {
                mass_kg: 1000.0,
                dry_mass_kg: 800.0,
                propellant_mass_kg: 200.0,
                bus_dims_m: [2.0, 2.0, 0.63],
                inertia_diag_kgm2: [366.67, 366.67, 666.67],
                derive_inertia_from_geometry: None,
                srp_model: SrpModel::Cannonball,
                propulsion: None,
                cp_cg_offset_m: Some(0.2),
                reflectivity_cr: None,
                launch_vehicle: None,
                hardware: vec![
                    HardwareItem::SolarPanel {
                        area_m2: 4.0, efficiency: Some(0.29),
                        position_m: None, normal: None, width_m: None, height_m: None,
                        articulation: None, rho_s: None, rho_d: None, mass_kg: None,
                    },
                    HardwareItem::OpNavCamera {
                        bearing_noise_mrad: Some(0.1),
                        angular_size_noise_mrad: Some(0.2),
                        boresight: None, position_m: None, fov_deg: None, mass_kg: None,
                    },
                ],
            },
            trajectory: TrajectoryConfig {
                phases: vec![MissionPhase::Capture],
                solver: TrajectorySolver::Hohmann,
                departure_epoch: None,
                departure_body: "Earth".to_string(),
                departure: None,
                cruise: None,
                capture: Some(CaptureConfig {
                    target_orbit_radius_m: Some(500.0),
                    approach_v_inf_mps: None,
                    terminator_orbit: false,
                    capture_eccentricity: 0.0,
                }),
                landing: None,
                per_phase_radii: std::collections::HashMap::new(),
            },
            gnc: GncConfig {
                navigation_filter: NavigationFilter::EKF,
                pointing_mode: PointingMode::Nadir,
                attitude_controller: AttitudeController::ReactionWheelPD,
                position_accuracy_req_m: Some(10.0),
                velocity_accuracy_req_mps: Some(0.001),
                reaction_wheel_kp: None,
                reaction_wheel_kd: None,
                pointing_deadband_rad: None,
                rate_deadband_radps: None,
                attitude_control: Default::default(),
            },
            simulation: SimulationConfig {
                integrator: Integrator::DormandPrince45,
                rtol: 1.0e-9,
                atol: 1.0e-7,
                dt_truth_s: 10.0,
                dt_meas_s: 120.0,
                monte_carlo_runs: 0,
                output_dir: "out/apophis_orbit_test/".into(),
            },
            optimization: None,
            cruise_seed: None,
        }
    }

    #[test]
    fn srp_torque_max_matches_hand_calc() {
        let cfg = apophis_fixture();
        let tau = srp_torque_max(&cfg);
        // 8 m² (4 m² bus face + 4 m² panel) at d=0.9224 AU is the apophis_orbit.toml
        // case; this fixture has no keplerian_orbit so d defaults to 1.0 AU.
        // P_SRP(1AU)=4.56e-6 * 8 m² * Cr=1.4 * offset=0.2 m = 1.0214e-5 N·m
        assert!((tau - 1.0214e-5).abs() < 1.0e-8, "got {tau:e}");
    }

    /// With `cp_cg_offset_m` unset, a
    /// placed `CustomPlate` must move `srp_torque_max`'s result — the old
    /// scalar (bus-face + panel-area only) estimate always ignored it. A
    /// far-offset, large-area plate should dominate the worst-case torque
    /// over the bare bus-box baseline.
    #[test]
    fn srp_torque_max_reflects_a_placed_custom_plate() {
        let mut cfg = apophis_fixture();
        cfg.spacecraft.cp_cg_offset_m = None;
        cfg.spacecraft.hardware.retain(|h| !matches!(h, HardwareItem::SolarPanel { .. }));

        let baseline = srp_torque_max(&cfg);

        cfg.spacecraft.hardware.push(HardwareItem::CustomPlate {
            normal: [0.0, 0.0, 1.0],
            area_m2: 20.0,
            center_offset_m: [0.0, 3.0, 0.0],
            rho_s: None,
            rho_d: None,
            double_sided: Some(true),
            mass_kg: None,
        });
        let with_plate = srp_torque_max(&cfg);

        assert!(
            with_plate > 5.0 * baseline,
            "a large plate 3 m from the bus should dominate the worst-case torque: baseline={baseline:e}, with_plate={with_plate:e}",
        );
    }

    /// The two `srp_torque_max` modes are independent: an explicit
    /// `cp_cg_offset_m` must keep using the old scalar formula even when
    /// real placed geometry (which the unset-default path would otherwise
    /// use) is also present.
    #[test]
    fn srp_torque_max_override_ignores_placed_geometry() {
        let mut cfg = apophis_fixture(); // cp_cg_offset_m = Some(0.2)
        let without_plate = srp_torque_max(&cfg);

        cfg.spacecraft.hardware.push(HardwareItem::CustomPlate {
            normal: [0.0, 0.0, 1.0],
            area_m2: 20.0,
            center_offset_m: [0.0, 3.0, 0.0],
            rho_s: None,
            rho_d: None,
            double_sided: Some(true),
            mass_kg: None,
        });
        let with_plate = srp_torque_max(&cfg);

        assert!(
            (with_plate - without_plate).abs() < 1.0e-12,
            "explicit cp_cg_offset_m should ignore placed hardware geometry entirely: {without_plate:e} vs {with_plate:e}",
        );
    }

    #[test]
    fn select_reaction_wheel_picks_medium_for_apophis_momentum() {
        let (wheel, ok) = select_reaction_wheel(1.201486e-5, 2.611433);
        assert_eq!(wheel.name, "RW-Medium");
        assert!(ok);
    }

    #[test]
    fn select_reaction_wheel_falls_back_when_no_catalog_entry_fits() {
        let (_wheel, ok) = select_reaction_wheel(1.0, 1.0e6);
        assert!(!ok, "no catalog wheel should satisfy a 1e6 N·m·s momentum requirement");
    }

    #[test]
    fn select_thruster_picks_monoprop_over_cold_gas() {
        let cfg = apophis_fixture();
        let (thruster, propellant_kg) = select_thruster(&cfg, 2.611433);
        assert_eq!(thruster.name, "Monoprop");
        assert!((propellant_kg - 0.001210).abs() < 1.0e-5, "got {propellant_kg}");
    }

    #[test]
    fn select_opnav_camera_picks_coarsest_grade_meeting_requirement() {
        let (cam, achieved_m) = select_opnav_camera(500.0, 10.0);
        assert_eq!(cam.name, "OpNav-Coarse");
        assert!((achieved_m - 0.25).abs() < 1.0e-9);
    }

    #[test]
    fn select_opnav_camera_falls_back_to_finest_when_none_meet_requirement() {
        let (cam, achieved_m) = select_opnav_camera(1.0e9, 1.0e-6);
        assert_eq!(cam.name, "OpNav-Fine", "should fall back to the finest available grade");
        assert!(achieved_m > 1.0e-6, "even the finest grade shouldn't meet an absurd requirement");
    }

    #[test]
    fn ekf_state_dim_is_10_for_cannonball_plus_opnav() {
        let cfg = apophis_fixture();
        assert_eq!(ekf_state_dim(&cfg), 10);
    }

    #[test]
    fn ekf_state_dim_is_6_without_opnav() {
        let mut cfg = apophis_fixture();
        cfg.spacecraft.hardware.retain(|h| !matches!(h, HardwareItem::OpNavCamera { .. }));
        assert_eq!(ekf_state_dim(&cfg), 6);
    }

    #[test]
    fn ekf_state_dim_is_6_for_flat_plate_srp_even_with_opnav() {
        let mut cfg = apophis_fixture();
        cfg.spacecraft.srp_model = SrpModel::FlatPlate;
        assert_eq!(ekf_state_dim(&cfg), 6);
    }
}
