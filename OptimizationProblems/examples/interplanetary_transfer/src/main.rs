//! Solar Sail Earth-Mars Interplanetary Transfer Optimizer
//!
//! Optimizes a solar sail trajectory from Earth to Mars using Simulated Annealing.
//! The optimizer jointly searches over:
//!   - Sail pointing angles (cone/clock) at N control points
//!   - Time of flight (variable — allows the optimizer to find favorable launch windows)
//!
//! The objective is to minimize the rendezvous error with Mars at arrival
//! (weighted combination of position and velocity mismatch).
//!
//! # Requirements
//!
//! This example requires a JPL ephemeris kernel at `kernels/de440s.bsp`.
//! Download from: https://public-data.nyxspace.com/anise/de440s.bsp
//! Place it at `kernels/de440s.bsp` relative to the working directory.
//!
//! # Running
//!
//! ```bash
//! mkdir kernels
//! curl -O kernels/de440s.bsp https://public-data.nyxspace.com/anise/de440s.bsp
//! cargo run -p interplanetary_transfer --release
//! ```

#![expect(incomplete_features)]
#![feature(inherent_associated_types)]
#![feature(generic_const_exprs)]

use std::sync::{Arc, Mutex};
use uuid::Uuid;

use ephemeris::{Almanac, Body};
use hifitime::Duration;
#[allow(unused_imports)]
use optimization::{
    AlgorithmConfig, CoolingSchedule, GAProblem, InterpolationStrategy, OptimizableProblem,
    OptimizationContext,
};

use optimization::LoggingStrategy;
use interplanetary_transfer::{
    config::MissionConfig,
    control::Angles,
    problem::{TransferInput, TransferProblem},
};

fn main() {
    const DB_NAME: &str = "out/interplanetary_transfer.db";
    const NUM_CONTROL_POINTS: usize = 10;
    const NUM_GENERATIONS: u64 = 50;

    // Load ephemeris kernel — required for Earth and Mars position queries
    println!("Loading ephemeris kernel...");
    let almanac = Almanac::new(&ephemeris::find_kernel("de440s.bsp"))
        .expect("Failed to load de440s.bsp kernel");
    println!("Ephemeris loaded.");

    let config = MissionConfig::optimization_defaults();

    // Query Earth's heliocentric state at departure — this is the spacecraft's initial state
    let initial_state = almanac
        .body_state_heliocentric(Body::Earth, config.departure_epoch)
        .expect("Failed to query Earth heliocentric state");

    println!("Departure epoch: {:?}", config.departure_epoch);
    println!(
        "Earth departure state: r={:.3e} m ({:.3} AU)",
        initial_state.position.norm(),
        initial_state.position.norm() / 1.496e11
    );

    // Query Mars departure state before moving almanac into Arc
    let mars_dep_state = almanac
        .body_state_heliocentric(Body::Mars, config.departure_epoch)
        .expect("Failed to query Mars heliocentric state at departure");

    // Extract raw SVector<f64,3> before initial_state/mars_dep_state are moved.
    // SVector<f64,3> is Copy; FrameState<Heliocentric> is not (Heliocentric lacks Copy).
    let earth_dep_p = initial_state.position.inner;
    let earth_dep_v = initial_state.velocity.inner;
    let mars_dep_p  = mars_dep_state.position.inner;
    let mars_dep_v  = mars_dep_state.velocity.inner;

    let mut problem = TransferProblem::<NUM_CONTROL_POINTS> {
        sail_area: config.sail_area,
        mass: config.mass,
        reflectivity: config.reflectivity,
        initial_state,
        departure_epoch: config.departure_epoch,
        min_tof_days: config.min_tof_days,
        max_tof_days: config.max_tof_days,
        vel_weight: config.vel_weight,
        tof_weight: config.tof_weight,
        departure_window_days: config.departure_window_days,
        almanac: Arc::new(almanac),
        logging_strategy: LoggingStrategy::FinalStateOnly,
    };

    // Initial guess: slight cone to give tangential SRP component for orbit raising, mid-range TOF
    let initial = TransferInput::<NUM_CONTROL_POINTS> {
        angles: core::array::from_fn(|_| Angles { cone: std::f64::consts::PI / 6.0, clock: 0.0 }),
        tof_days: config.max_tof_days,
        departure_offset_days: 180.0,
    };

    // Setup database
    std::fs::create_dir_all("out").unwrap();
    let _ = std::fs::remove_file(DB_NAME);
    let connection = rusqlite::Connection::open(DB_NAME).unwrap();

    let context = OptimizationContext {
        run_id: Uuid::now_v7(),
        connection: Some(Arc::new(Mutex::new(connection))),
        interpolation_strategy: InterpolationStrategy::Linear,
        algorithm_config: AlgorithmConfig::GeneticAlgorithm {
            population_size: 100,
            crossover_rate: 0.85,
            mutation_rate: 0.1,
            elitism_count: 5,
            tournament_size: 3,
        },
    };

    println!("\nStarting optimization: {} control points, {} generations", NUM_CONTROL_POINTS, NUM_GENERATIONS);
    println!("TOF search range: [{}, {}] days", config.min_tof_days, config.max_tof_days);
    println!("Sail: {:.0} m², {:.1} kg, reflectivity {:.2}", config.sail_area, config.mass, config.reflectivity);
    println!();

    // Overwrite weights for initial phase.
    problem.vel_weight = 0.5;   // was: config.vel_weight (2.0)
    problem.tof_weight = 0.5;   // was: config.tof_weight (0.2)
    let opt_best = match &context.algorithm_config {
        AlgorithmConfig::GeneticAlgorithm { .. } => problem.optimize_ga(NUM_GENERATIONS, context.clone()),
        AlgorithmConfig::SimulatedAnnealing { .. } => problem.optimize(initial, NUM_GENERATIONS, context.clone()),
    };

    // Evaluate optimizer result — save energy and TOF for final comparison
    let sa_eval_ctx = optimization::EvaluationContext::new_in_run(&context);
    let sa_energy = problem.energy(&opt_best, &sa_eval_ctx);
    let sa_tof    = opt_best.tof_days;
    let algo_label = match &context.algorithm_config {
        AlgorithmConfig::GeneticAlgorithm { .. } => "GENETIC ALGORITHM",
        AlgorithmConfig::SimulatedAnnealing { .. } => "SIMULATED ANNEALING",
    };
    println!("\n══════════════════════════════════════════════════");
    println!("{} RESULTS", algo_label);
    println!("══════════════════════════════════════════════════");
    println!("  TOF:    {:.1} days", sa_tof);
    println!("  Energy: {:.6}", sa_energy);
    for (i, a) in opt_best.angles.iter().enumerate() {
        println!("    CP[{:2}]: cone={:+.3} rad ({:+.1}°), clock={:.3} rad ({:.1}°)",
            i, a.cone, a.cone.to_degrees(), a.clock, a.clock.to_degrees());
    }

    // --- Local refinement phase ---
    problem.vel_weight = config.vel_weight;
    problem.tof_weight = config.tof_weight;
    println!("\nStarting local refinement (coordinate-wise hill climbing)...");
    let best = problem.refine(opt_best, &context);

    // Evaluate refined result — save for comparison
    let ref_eval_ctx = optimization::EvaluationContext::new_in_run(&context);
    let refined_energy = problem.energy(&best, &ref_eval_ctx);
    println!("\n══════════════════════════════════════════════════");
    println!("AFTER REFINEMENT");
    println!("══════════════════════════════════════════════════");
    println!("  TOF:    {:.1} days", best.tof_days);
    println!("  Energy: {:.6}", refined_energy);
    println!("  Control points:");
    for (i, a) in best.angles.iter().enumerate() {
        println!("    CP[{:2}]: cone={:+.3} rad ({:+.1}°), clock={:.3} rad ({:.1}°)",
            i, a.cone, a.cone.to_degrees(), a.clock, a.clock.to_degrees());
    }

    // --- Comparison ---
    let delta_e = sa_energy - refined_energy;
    let delta_pct = if sa_energy.abs() > 1e-12 { delta_e / sa_energy.abs() * 100.0 } else { 0.0 };
    println!("\n══════════════════════════════════════════════════");
    println!("COMPARISON");
    println!("══════════════════════════════════════════════════");
    println!("  SA best:      TOF={:.1} d   Energy={:.6}", sa_tof, sa_energy);
    println!("  After refine: TOF={:.1} d   Energy={:.6}", best.tof_days, refined_energy);
    println!("  Improvement:  ΔE={:+.6}  ({:+.2}%)", delta_e, delta_pct);

    // --- Visualization pass: log the full refined trajectory for the plotting script ---
    println!("\nLogging refined trajectory for visualization...");
    problem.logging_strategy = LoggingStrategy::AllDownsampled(1);
    let viz_ctx = optimization::EvaluationContext::new_in_run(&context);
    let viz_energy = problem.energy(&best, &viz_ctx);
    println!("Trajectory logged. Energy: {:.6}", viz_energy);

    // Save the eval_id so the plotting script can find this exact trajectory
    if let Some(conn) = context.connection.as_ref() {
        let conn_lock = conn.lock().unwrap();
        conn_lock.execute(
            "CREATE TABLE IF NOT EXISTS solutions (id INTEGER PRIMARY KEY, label TEXT, eval_id TEXT, energy REAL, tof_days REAL)",
            [],
        ).unwrap();
        conn_lock.execute(
            "INSERT INTO solutions (label, eval_id, energy, tof_days) VALUES ('best', ?, ?, ?)",
            [&viz_ctx.eval_id.to_string(), &format!("{:.8}", viz_energy), &format!("{:.3}", best.tof_days)],
        ).unwrap();

        // Store Earth and Mars positions at departure and arrival for the plotting script
        let arrival_epoch = problem.departure_epoch + Duration::from_days(best.tof_days);
        let earth_arr_state = problem.almanac
            .body_state_heliocentric(Body::Earth, arrival_epoch)
            .expect("Failed to query Earth state at arrival");
        let mars_arr_state = problem.almanac
            .body_state_heliocentric(Body::Mars, arrival_epoch)
            .expect("Failed to query Mars state at arrival");

        conn_lock.execute(
            "CREATE TABLE IF NOT EXISTS planet_positions (id INTEGER PRIMARY KEY, body TEXT, epoch_label TEXT, x REAL, y REAL, z REAL, vx REAL, vy REAL, vz REAL)",
            [],
        ).unwrap();
        let mut stmt = conn_lock.prepare(
            "INSERT INTO planet_positions (body, epoch_label, x, y, z, vx, vy, vz) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        ).unwrap();
        // SVector<f64,3> is Copy, so earth_dep_p/mars_dep_p can be used freely here.
        let earth_arr_p = earth_arr_state.position.inner;
        let earth_arr_v = earth_arr_state.velocity.inner;
        let mars_arr_p  = mars_arr_state.position.inner;
        let mars_arr_v  = mars_arr_state.velocity.inner;
        for (body, label, p, v) in [
            ("Earth", "departure", earth_dep_p, earth_dep_v),
            ("Mars",  "departure", mars_dep_p,  mars_dep_v),
            ("Earth", "arrival",   earth_arr_p, earth_arr_v),
            ("Mars",  "arrival",   mars_arr_p,  mars_arr_v),
        ] {
            stmt.insert((body, label, p[0], p[1], p[2], v[0], v[1], v[2])).unwrap();
        }
        drop(stmt);

        // Sample Earth and Mars states along the full transfer — used by the plotting script
        // to draw the actual (non-constant) orbital speed rather than a flat reference line.
        const N_TRACK: usize = 100;
        conn_lock.execute(
            "CREATE TABLE IF NOT EXISTS planet_track \
             (id INTEGER PRIMARY KEY, body TEXT, day_offset REAL, \
              x REAL, y REAL, z REAL, vx REAL, vy REAL, vz REAL)",
            [],
        ).unwrap();
        let mut track_stmt = conn_lock.prepare(
            "INSERT INTO planet_track (body, day_offset, x, y, z, vx, vy, vz) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        ).unwrap();
        for i in 0..=N_TRACK {
            let t_days = best.tof_days * (i as f64) / (N_TRACK as f64);
            let epoch = problem.departure_epoch + Duration::from_days(t_days);
            if let (Ok(es), Ok(ms)) = (
                problem.almanac.body_state_heliocentric(Body::Earth, epoch),
                problem.almanac.body_state_heliocentric(Body::Mars, epoch),
            ) {
                let ep = es.position.inner;
                let ev = es.velocity.inner;
                let mp = ms.position.inner;
                let mv = ms.velocity.inner;
                track_stmt.insert(("Earth", t_days, ep[0], ep[1], ep[2], ev[0], ev[1], ev[2])).unwrap();
                track_stmt.insert(("Mars",  t_days, mp[0], mp[1], mp[2], mv[0], mv[1], mv[2])).unwrap();
            }
        }
        drop(track_stmt);
    }

    // Close database
    println!("\nClosing database...");
    if let Some(conn) = context.connection {
        drop(conn);
    }
    println!("Done! Results saved to {}", DB_NAME);
    println!("Plot with: python examples/interplanetary_transfer/plot_trajectory.py");
}
