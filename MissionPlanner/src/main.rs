//! AstroDynamics generic mission planning and design tool.
//!
//! Usage:
//!   mission-planner validate  <mission.toml>   # parse + check + print summary
//!   mission-planner design    <mission.toml>   # trajectory design (Phase 2)
//!   mission-planner simulate  <mission.toml>   # high-fidelity sim (Phase 4)

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

mod config;
mod design;
mod gnc_design;
mod lowthrust;
mod mga;
mod mga_scan_run;
mod optimize;
mod sequence_search;
mod simulate;
mod vehicle_properties;
use config::{CaptureConfig, HardwareItem, MissionConfig, MissionObjective};

#[derive(Parser)]
#[command(
    name = "mission-planner",
    version,
    about = "AstroDynamics generic mission planning and design tool"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// CLI-only override for the Phase 9 optimization stage's target — lets one
/// config drive either a Flyby or an Orbit search without maintaining two
/// near-duplicate TOML files (see the design notes).
#[derive(Clone, Copy, Debug, ValueEnum)]
enum TargetModeArg {
    Flyby,
    Orbit,
}

#[derive(Subcommand)]
enum Command {
    /// Parse and validate a mission config; print a human-readable summary.
    Validate {
        /// Path to the TOML mission config file
        config: PathBuf,
    },
    /// Run the trajectory design stage: porkchop scan and best-arc selection.
    Design {
        /// Path to the TOML mission config file
        config: PathBuf,
    },
    /// Run the high-fidelity simulation stage [Phase 4 — not yet implemented].
    Simulate {
        config: PathBuf,
    },
    /// Run the Phase 9 trajectory optimization stage (real propagated
    /// dynamics) — requires an [optimization] section in the config.
    Optimize {
        config: PathBuf,
        /// Override the config's [mission].objective / [trajectory.capture]
        /// target for this run only (does not modify the file). Must be
        /// given together with `--target-distance-m`. Omit both to use
        /// whatever the TOML already specifies.
        #[arg(long, value_enum, requires = "target_distance_m")]
        target_mode: Option<TargetModeArg>,
        /// Flyby closest-approach distance, or orbit insertion radius,
        /// in meters — meaning depends on `--target-mode`. Must be given
        /// together with `--target-mode`.
        #[arg(long, requires = "target_mode")]
        target_distance_m: Option<f64>,
    },
    /// Re-run just the departure-geometry diagnostic from an already-saved
    /// `optimize/<method>_best.csv`, without re-running the GA/PSO search.
    Geometry {
        config: PathBuf,
    },
    /// Re-propagate a sample of individuals' full trajectories from an
    /// already-saved `optimize/<method>_population.csv`, without
    /// re-running the GA/PSO search.
    Population {
        config: PathBuf,
        /// Number of individuals to sample (evenly spaced across the
        /// feasible population).
        #[arg(long, default_value_t = 50)]
        count: usize,
    },
    /// Re-evaluate a sample of individuals from an already-saved
    /// `optimize/<method>_population.csv` to recover their real
    /// arrival/capture geometry (theta_arr/phi_arr/dv_arrival), without
    /// re-running the GA/PSO search. No-op data for a Flyby mission (no
    /// capture burn exists to have geometry).
    ArrivalAngles {
        config: PathBuf,
        /// Number of individuals to sample (evenly spaced by fitness rank).
        #[arg(long, default_value_t = 200)]
        count: usize,
    },
    /// Dense, unbiased grid scan over theta x phi at a fixed dv, using the
    /// real propagated physics -- no GA, no seeding. Ground truth for the
    /// fitness landscape.
    GridScan {
        config: PathBuf,
        #[arg(long)]
        dv_mps: f64,
        #[arg(long, default_value_t = 3.0)]
        theta_step_deg: f64,
        #[arg(long, default_value_t = 3.0)]
        phi_step_deg: f64,
    },
    /// Run only the Tisserand beam search to discover MGA flyby sequences,
    /// print the ranked table, and exit — without running the inner DE optimizer.
    /// Requires an [optimization.mga.sequence_search] section in the config.
    ///
    /// Example:
    ///   cargo run -p mission_planner search-sequence config/venus_saturn_auto.toml
    SearchSequence {
        config: PathBuf,
    },
    /// Re-evaluate the best MGA chromosome from a previously saved
    /// `mga_best_chromosome.csv` and print a detailed per-leg breakdown
    /// (TOF, η, DSM ΔV, flyby periapsis, turn angle, v∞). No optimizer rerun.
    ///
    /// Example:
    ///   cargo run -p mission_planner mga-geometry config/evj_flyby.toml
    MgaGeometry {
        config: PathBuf,
    },
    /// Run multiple-shooting refinement on the best MGA chromosome from a
    /// previously saved `mga_best_chromosome.csv`. Treats each leg's DSM ΔV
    /// vector as a free variable and runs Newton–Raphson to enforce position
    /// continuity at each flyby body and the final target under real Dopri5
    /// dynamics. Writes `mga_refined.csv` and `mga_refined_legs.csv`.
    ///
    /// Example:
    ///   cargo run -p mission_planner mga-refine config/evj_flyby.toml
    MgaRefine {
        config: PathBuf,
    },
    /// Run the Sims-Flanagan low-thrust trajectory optimizer (Phase 9h).
    ///
    /// Divides the TOF into N equal segments and optimizes N×3 thrust vectors
    /// via DE/rand/1/bin so that forward and backward half-arcs meet at a
    /// central match point. Reads propulsion parameters from
    /// [spacecraft.propulsion] (thrust_n, isp_s) and body states from ANISE.
    /// Writes sf_arc.csv and sf_thrust.csv to [simulation].output_dir.
    ///
    /// Example:
    ///   cargo run -p mission_planner low-thrust config/earth_mars_lowthrust.toml
    LowThrust {
        config: PathBuf,
    },
    /// Run the Phase 9w ballistic (no-DSM) powered-flyby MGA grid scan: a
    /// deterministic departure-date x per-leg-TOF window scan using pure
    /// Lambert legs + analytic flyby feasibility, no optimizer. Requires
    /// [optimization.mga] with a fixed `flyby_bodies` sequence and an
    /// [optimization.mga.scan] section. Writes mga_window_scan.csv.
    ///
    /// Example:
    ///   cargo run -p mission_planner mga-scan config/cassini_evvmjs.toml
    MgaScan {
        config: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Validate { config } => validate(&config),
        Command::Design { config } => {
            let cfg = load_config(&config);
            print_summary(&cfg);
            let errs = config::check_config(&cfg);
            if !errs.is_empty() {
                println!("\n✗  Config errors:");
                for e in &errs { println!("   • {e}"); }
                std::process::exit(1);
            }
            design::run(&cfg);
            gnc_design::run(&cfg);
        }
        Command::Simulate { config } => {
            let cfg = load_config(&config);
            let errs = config::check_config(&cfg);
            if !errs.is_empty() {
                println!("✗  Config errors:");
                for e in &errs { println!("   • {e}"); }
                std::process::exit(1);
            }
            simulate::run(&cfg);
        }
        Command::Optimize { config, target_mode, target_distance_m } => {
            let mut cfg = load_config(&config);
            apply_target_mode_override(&mut cfg, target_mode, target_distance_m);
            let errs = config::check_config(&cfg);
            if !errs.is_empty() {
                println!("✗  Config errors:");
                for e in &errs { println!("   • {e}"); }
                std::process::exit(1);
            }
            optimize::run(&cfg);
        }
        Command::Geometry { config } => {
            let cfg = load_config(&config);
            if let Err(e) = optimize::replot_departure_geometry(&cfg) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        Command::Population { config, count } => {
            let cfg = load_config(&config);
            if let Err(e) = optimize::replot_population_sample_trajectories(&cfg, count) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        Command::ArrivalAngles { config, count } => {
            let cfg = load_config(&config);
            if let Err(e) = optimize::replot_population_arrival_angles(&cfg, count) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        Command::GridScan { config, dv_mps, theta_step_deg, phi_step_deg } => {
            let cfg = load_config(&config);
            if let Err(e) = optimize::run_grid_scan(&cfg, dv_mps, theta_step_deg, phi_step_deg) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        Command::SearchSequence { config } => {
            let cfg = load_config(&config);
            let opt = cfg.optimization.as_ref().expect("config requires an [optimization] section");
            let mga_cfg = opt.mga.as_ref().expect("config requires an [optimization.mga] section");
            let ss = mga_cfg.sequence_search.as_ref()
                .expect("config requires an [optimization.mga.sequence_search] section");
            let sequences = sequence_search::run_sequence_search(ss, &opt.departure_body, &opt.target_body);
            sequence_search::print_sequence_table(&sequences, &opt.departure_body, &opt.target_body);
        }
        Command::MgaGeometry { config } => {
            let cfg = load_config(&config);
            let Some(almanac) = design::load_almanac() else {
                std::process::exit(1);
            };
            if let Err(e) = mga::run_mga_geometry(&cfg, &almanac) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        Command::MgaRefine { config } => {
            let cfg = load_config(&config);
            let Some(almanac) = design::load_almanac() else {
                std::process::exit(1);
            };
            if let Err(e) = mga::run_mga_refine(&cfg, &almanac) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        Command::LowThrust { config } => {
            let cfg = load_config(&config);
            let Some(almanac) = design::load_almanac() else {
                std::process::exit(1);
            };
            if let Err(e) = lowthrust::run_low_thrust(&cfg, &almanac) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        Command::MgaScan { config } => {
            let cfg = load_config(&config);
            let Some(almanac) = design::load_almanac() else {
                std::process::exit(1);
            };
            if let Err(e) = mga_scan_run::run_mga_window_scan(&cfg, &almanac) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
    }
}

// ── shared helpers ───────────────────────────────────────────────────────────

fn load_config(path: &std::path::Path) -> MissionConfig {
    match MissionConfig::from_file(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load '{}': {e}", path.display());
            std::process::exit(1);
        }
    }
}

/// Phase 9i: in-memory override of `[mission].objective` and
/// `[trajectory.capture].target_orbit_radius_m` from `--target-mode`/
/// `--target-distance-m`. No-op (and `cfg` unchanged) when both args are
/// `None` — clap's `requires` constraint guarantees they're never given
/// independently, so only the both-`Some`/both-`None` cases are reachable.
fn apply_target_mode_override(
    cfg: &mut MissionConfig,
    target_mode: Option<TargetModeArg>,
    target_distance_m: Option<f64>,
) {
    let (mode, distance_m) = match (target_mode, target_distance_m) {
        (Some(m), Some(d)) => (m, d),
        _ => return,
    };
    cfg.mission.objective = match mode {
        TargetModeArg::Flyby => MissionObjective::Flyby,
        TargetModeArg::Orbit => MissionObjective::Orbit,
    };
    match &mut cfg.trajectory.capture {
        Some(cap) => cap.target_orbit_radius_m = Some(distance_m),
        None => {
            cfg.trajectory.capture = Some(CaptureConfig {
                target_orbit_radius_m: Some(distance_m),
                approach_v_inf_mps: None,
                terminator_orbit: false,
                capture_eccentricity: 0.0,
            });
        }
    }
    println!(
        "  (CLI override: target_mode={mode:?}, target_distance_m={distance_m:.1})"
    );
}

// ── validate command ─────────────────────────────────────────────────────────

fn validate(path: &std::path::Path) {
    let cfg = load_config(path);
    print_summary(&cfg);

    let errors = config::check_config(&cfg);
    if errors.is_empty() {
        println!("\n✓  Config valid.\n");
    } else {
        println!("\n✗  {} error(s):\n", errors.len());
        for e in &errors {
            println!("   • {e}");
        }
        println!();
        std::process::exit(1);
    }
}

fn print_summary(c: &MissionConfig) {
    let sep = "─".repeat(64);

    println!("\n{sep}");
    println!("  {}  ·  {}", c.mission.name, c.mission.objective);
    println!("{sep}");

    // Target body
    let b = &c.target_body;
    println!("\nTarget Body: {}", b.name);
    println!(
        "  μ = {:.4e} m³/s²  |  R = {:.1} m",
        b.mu_m3s2, b.radius_m
    );
    println!(
        "  Gravity: {}  |  Atmosphere: {}",
        b.gravity_model, b.atmosphere
    );
    println!("  Ephemeris: {}", b.ephemeris);
    if let Some(j2) = b.j2 {
        print!("  Harmonics: J2 = {j2:.6e}");
        if let Some(j3) = b.j3 {
            print!("  J3 = {j3:.6e}");
        }
        if let Some(j4) = b.j4 {
            print!("  J4 = {j4:.6e}");
        }
        println!();
    }
    if !b.third_bodies.is_empty() {
        println!("  Third-body perturbers: {}", b.third_bodies.join(", "));
    }

    // Spacecraft
    let sc = &c.spacecraft;
    println!("\nSpacecraft");
    println!(
        "  Mass: {:.1} kg wet  /  {:.1} kg dry  /  {:.1} kg propellant",
        sc.mass_kg, sc.dry_mass_kg, sc.propellant_mass_kg
    );
    println!(
        "  Bus: {:.2} × {:.2} × {:.2} m  |  SRP model: {}",
        sc.bus_dims_m[0], sc.bus_dims_m[1], sc.bus_dims_m[2], sc.srp_model
    );
    println!(
        "  Inertia: [{:.1}, {:.1}, {:.1}] kg·m²",
        sc.inertia_diag_kgm2[0], sc.inertia_diag_kgm2[1], sc.inertia_diag_kgm2[2]
    );
    if let Some(prop) = &sc.propulsion {
        println!(
            "  Propulsion: {}  |  Isp = {:.0} s  |  Thrust = {:.1} N",
            prop.kind, prop.isp_s, prop.thrust_n
        );
        let dv = tsiolkovsky_dv(sc.mass_kg, sc.dry_mass_kg, prop.isp_s);
        println!("  ΔV budget (Tsiolkovsky): {:.1} m/s", dv);
    }
    if sc.hardware.is_empty() {
        println!("  Hardware: (none specified)");
    } else {
        println!("  Hardware ({} items):", sc.hardware.len());
        for hw in &sc.hardware {
            println!("    {}", describe_hardware(hw));
        }
    }

    // Trajectory
    let traj = &c.trajectory;
    let phases: Vec<String> = traj.phases.iter().map(|p| p.to_string()).collect();
    println!("\nTrajectory");
    println!("  Phases: {}", phases.join(" → "));
    println!("  Solver: {}", traj.solver);
    if let Some(epoch) = &traj.departure_epoch {
        println!("  Departure: {epoch}  from {}", traj.departure_body);
    }
    if let Some(cr) = &traj.cruise {
        if let (Some(lo), Some(hi)) = (cr.tof_days_min, cr.tof_days_max) {
            println!("  Cruise TOF: {lo:.0} – {hi:.0} days");
        }
        if let Some(res) = cr.grid_resolution {
            println!("  Grid resolution: {res} × {res}");
        }
    }
    if let Some(cap) = &traj.capture {
        if let Some(r) = cap.target_orbit_radius_m {
            println!("  Capture orbit radius: {:.0} m", r);
        }
    }

    // GNC
    let gnc = &c.gnc;
    println!("\nGNC");
    println!("  Navigation: {}", gnc.navigation_filter);
    println!("  Pointing:   {}", gnc.pointing_mode);
    println!("  Controller: {}", gnc.attitude_controller);
    if let Some(pos_req) = gnc.position_accuracy_req_m {
        println!("  Position accuracy req: {pos_req:.1} m");
    }
    if let Some(vel_req) = gnc.velocity_accuracy_req_mps {
        println!("  Velocity accuracy req: {vel_req:.4} m/s");
    }

    // Simulation
    let sim = &c.simulation;
    println!("\nSimulation");
    println!(
        "  Integrator: {}  (rtol = {:.0e}, atol = {:.0e})",
        sim.integrator, sim.rtol, sim.atol
    );
    println!(
        "  Truth dt: {:.0} s  |  Measurement dt: {:.0} s",
        sim.dt_truth_s, sim.dt_meas_s
    );
    let mc_str = if sim.monte_carlo_runs > 0 {
        format!("{} runs", sim.monte_carlo_runs)
    } else {
        "off".into()
    };
    println!("  Monte Carlo: {mc_str}");
    println!("  Output: {}", sim.output_dir);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Ideal rocket equation: ΔV = Isp · g₀ · ln(m_wet / m_dry) [m/s]
fn tsiolkovsky_dv(mass_wet_kg: f64, mass_dry_kg: f64, isp_s: f64) -> f64 {
    const G0: f64 = 9.80665;
    isp_s * G0 * (mass_wet_kg / mass_dry_kg).ln()
}

/// "  [placed]" when a boresight/normal-style direction field is set, else
/// empty — small formatting helper shared by every `describe_hardware`
/// variant with an optional placement/boresight field (spacecraft-builder
/// schema extension).
fn placement_suffix(direction: &Option<[f64; 3]>) -> &'static str {
    if direction.is_some() { "  [placed]" } else { "" }
}

fn describe_hardware(hw: &HardwareItem) -> String {
    match hw {
        HardwareItem::ReactionWheelCluster { model, count, max_speed_rads, max_torque_nm, .. } => {
            let m = model.as_deref().unwrap_or("generic");
            let spd = max_speed_rads.map_or("?".into(), |v| format!("{v:.0} rad/s"));
            let trq = max_torque_nm.map_or("?".into(), |v| format!("{v:.3} N·m"));
            format!("[Actuator] Reaction Wheel Cluster ×{count}  ({m})  ω_max={spd}  τ_max={trq}")
        }
        HardwareItem::RCS { thrust_n, count, moment_arm_m, .. } => {
            let t = thrust_n.map_or("?".into(), |v| format!("{v:.1} N"));
            let n = count.map_or("?".into(), |v| format!("{v}"));
            let arm = moment_arm_m.map_or("?".into(), |v| format!("{v:.2} m"));
            format!("[Actuator] RCS: {n} thrusters × {t}  arm={arm}")
        }
        HardwareItem::StarTracker { model, noise_rad, boresight, .. } => {
            let m = model.as_deref().unwrap_or("generic");
            let n = noise_rad.map_or("?".into(), |v| format!("{v:.2e} rad"));
            let p = placement_suffix(boresight);
            format!("[Sensor]   Star Tracker: {m}  (σ = {n}){p}")
        }
        HardwareItem::IMU { noise_sigma_mps, .. } => {
            let n = noise_sigma_mps.map_or("?".into(), |v| format!("{v:.3} m/s"));
            format!("[Sensor]   IMU  (ΔV noise σ = {n})")
        }
        HardwareItem::OpNavCamera { bearing_noise_mrad, angular_size_noise_mrad, boresight, .. } => {
            let b = bearing_noise_mrad.map_or("?".into(), |v| format!("{v:.2} mrad"));
            let a = angular_size_noise_mrad.map_or("?".into(), |v| format!("{v:.2} mrad"));
            let p = placement_suffix(boresight);
            format!("[Sensor]   OpNav Camera  (bearing σ={b}, size σ={a}){p}")
        }
        HardwareItem::Lidar { range_noise_m, max_range_m, boresight, .. } => {
            let n = range_noise_m.map_or("?".into(), |v| format!("{v:.1} m"));
            let r = max_range_m.map_or("?".into(), |v| format!("{v:.0} m"));
            let p = placement_suffix(boresight);
            format!("[Sensor]   LIDAR  (range σ={n}, max={r}){p}")
        }
        HardwareItem::RcsThruster { thrust_n, position_m, direction, .. } => {
            format!(
                "[Actuator] RCS Thruster  ({thrust_n:.1} N  pos=[{:.2},{:.2},{:.2}] m  dir=[{:.2},{:.2},{:.2}])",
                position_m[0], position_m[1], position_m[2],
                direction[0], direction[1], direction[2],
            )
        }
        HardwareItem::CommAntenna { boresight, beamwidth_deg, .. } => {
            format!(
                "[Comm]     Antenna  (beamwidth={beamwidth_deg:.1} deg  boresight=[{:.2},{:.2},{:.2}])",
                boresight[0], boresight[1], boresight[2],
            )
        }
        HardwareItem::SolarPanel { area_m2, efficiency, position_m, normal, width_m, height_m, .. } => {
            let e = efficiency.map_or("?".into(), |v| format!("{:.0}%", v * 100.0));
            let area = match (width_m, height_m) {
                (Some(w), Some(h)) => w * h,
                _ => *area_m2,
            };
            let p = placement_suffix(normal);
            let pos = position_m.map_or(String::new(), |v| format!("  pos=[{:.2},{:.2},{:.2}] m", v[0], v[1], v[2]));
            format!("[Power]    Solar Panel  ({area:.1} m²  η={e}){p}{pos}")
        }
        HardwareItem::CustomPlate { normal, area_m2, center_offset_m, rho_s, rho_d, double_sided, .. } => {
            let rs = rho_s.map_or("?".into(), |v| format!("{v:.2}"));
            let rd = rho_d.map_or("?".into(), |v| format!("{v:.2}"));
            let ds = if double_sided.unwrap_or(false) { "double-sided" } else { "single-sided" };
            format!(
                "[Geometry] Custom Plate  ({area_m2:.2} m²  n=[{:.2},{:.2},{:.2}]  \
                 c=[{:.2},{:.2},{:.2}] m  ρs={rs} ρd={rd}  {ds})",
                normal[0], normal[1], normal[2],
                center_offset_m[0], center_offset_m[1], center_offset_m[2],
            )
        }
    }
}
