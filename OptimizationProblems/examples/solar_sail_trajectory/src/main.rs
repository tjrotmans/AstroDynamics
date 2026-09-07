//! Solar Sail Orbit Raising Trajectory Optimizer
//!
//! This example optimizes the pointing angles of a solar sail to maximize
//! the semi-major axis (raise the orbit) over a fixed time period using
//! Simulated Annealing.
//!
//! The trajectory is represented as piecewise linear interpolation between
//! control points (angle pairs).
//!
//! # Validation
//! 
//! For a simple validation case, consider:
//! - Circular orbit at 500 km altitude
//! - Optimal strategy for circular orbit raising with ideal sail:
//!   * Point sail normal perpendicular to velocity at all times
//!   * This maximizes tangential acceleration → increases velocity → raises orbit
//!
//! Expected behavior:
//! - Semi-major axis should increase monotonically
//! - For small perturbations, can compare against analytical Edelbaum solution
//! - Delta-V for orbit raising: Δv ≈ √(μ/r₁) - √(μ/r₂)

#![expect(incomplete_features)]
#![feature(inherent_associated_types)]
#![feature(generic_const_exprs)]

use std::sync::{Arc, Mutex};
use uuid::Uuid;
use optimization::{OptimizableProblem, OptimizationContext, AlgorithmConfig, InterpolationStrategy};
use optimization::LoggingStrategy;
use solar_sail_trajectory::{
    control::Angles,
    problem::{OrbitRaisingProblem, ControlMode},
    CoolingSchedule,
    MissionConfig,
};

fn main() {
    // Configuration
    const DB_NAME: &str = "out/optimization.db";
    const NUM_CONTROL_POINTS: usize = 10; // Increased for better trajectory control
    const NUM_GENERATIONS: u64 = 1500;

    // Logging strategy - use FinalStateOnly for optimization (only need final SMA)
    // Switch to AllDownsampled(10) when you want to plot full trajectories
    let logging = LoggingStrategy::FinalStateOnly;

    // Setup database
    std::fs::create_dir_all("out").unwrap();
    // Clear old database if it exists
    let _ = std::fs::remove_file(DB_NAME);
    let connection = rusqlite::Connection::open(DB_NAME).unwrap();
    let run_id = Uuid::now_v7();

    // Define the problem
    let config = MissionConfig::optimization_defaults();
    
    let problem = OrbitRaisingProblem::<NUM_CONTROL_POINTS> {
        sail_area: config.sail_area,             // 800 m² sail (reasonably large)
        mass: config.mass,                       // 10 kg spacecraft (high area-to-mass ratio)
        reflectivity: config.reflectivity,       // Typical aluminized Kapton
        drag_coefficient: config.drag_coefficient, // Mostly irrelevant at high altitude
        target_sma: config.target_sma(),          // Target semi-major axis
        target_time: config.target_time,
        max_slew_rate: config.max_slew_rate,      // Maximum attitude change rate
        initial_state: config.initial_orbit(),
        logging_strategy: logging,
        control_mode: ControlMode::ControlPoints,
        local_refinement: true,
        local_refinement_iterations: 10,
    };

    // Generate random initial guess
    // let initial: [Angles; NUM_CONTROL_POINTS] = core::array::from_fn(|_| Angles::random());
    let initial: [Angles; NUM_CONTROL_POINTS] = core::array::from_fn(|_| Angles {
        cone: 0.5,
        clock: 0.0,
    });
    // Setup optimization context
    // Cooling schedule options:
    // - Linear: simple, fast cooling
    // - Exponential { t_initial: 1.0, t_final: 0.01 }: slower initial cooling
    // - Logarithmic { c: 1.0 }: very slow, theoretically optimal
    // - Quadratic: fast initial, slow final
    // - Inverse { t_initial: 1.0, k: 10.0 }: smooth decay
    let context = OptimizationContext {
        run_id,
        connection: Some(Arc::new(Mutex::new(connection))),
        // Interpolation strategy:
        // - Linear: smooth transitions between control points
        // - Piecewise: hold attitude constant until next control point (more realistic)
        interpolation_strategy: InterpolationStrategy::Linear,
        algorithm_config: AlgorithmConfig::SimulatedAnnealing {
            cooling_schedule: CoolingSchedule::Logarithmic { c: 1.0 },
            // Perturbation scale (radians at T=1): controls exploration vs exploitation
            // Lower = finer search, higher = broader exploration
            perturbation_scale: std::f64::consts::PI / 16.0,
        },
    };

    // Run optimization (using Simulated Annealing)
    println!("Starting orbit raising optimization with {} generations...", NUM_GENERATIONS);
    println!("Initial altitude: {} km", problem.initial_state.altitude() / 1000.0);
    println!("Target altitude: {} km", (problem.target_sma - 6.371e6) / 1000.0);
    println!("Mission duration: {} days", problem.target_time / (60.0 * 60.0 * 24.0));
    
    let best = problem.optimize(initial, NUM_GENERATIONS, context.clone());

    // Ensure solutions table exists
    if let Some(conn) = context.connection.as_ref() {
        let conn_lock = conn.lock().unwrap();
        conn_lock.execute(
            "CREATE TABLE IF NOT EXISTS solutions (id INTEGER PRIMARY KEY, run_id TEXT, label TEXT, eval_id TEXT, energy REAL, input TEXT)",
            [],
        ).unwrap();
    }

    // Print SA results
    {
        let eval_ctx = optimization::EvaluationContext::new_in_run(&context);
        let sa_energy = problem.energy(&best, &eval_ctx);
        println!("\n══════════════════════════════════════");
        println!("SIMULATED ANNEALING RESULTS");
        println!("══════════════════════════════════════");
        println!("  Energy: {:.2}", sa_energy);
        println!("  Δmean SMA: {:+.2} m", -sa_energy);
        for (i, a) in best.iter().enumerate() {
            println!("  CP[{}]: cone={:+.2}°, clock={:+.2}°", i, a.cone.to_degrees(), a.clock.to_degrees());
        }

        if let Some(conn) = context.connection.as_ref() {
            let conn_lock = conn.lock().unwrap();
            conn_lock.execute(
                "INSERT INTO solutions (run_id, label, eval_id, energy, input) VALUES (?, 'sa_best', ?, ?, ?)",
                [&context.run_id.to_string(), &eval_ctx.eval_id.to_string(), &format!("{:.6}", sa_energy), &format!("{:?}", best)],
            ).unwrap();
        }
    }

    // Optional local refinement phase
    let _best = if problem.local_refinement {
        println!("\nStarting local refinement phase...");
        let refined = problem.refine(best, &context);

        // Print refinement results
        {
            let eval_ctx = optimization::EvaluationContext::new_in_run(&context);
            let refined_energy = problem.energy(&refined, &eval_ctx);
            println!("\n══════════════════════════════════════");
            println!("LOCAL REFINEMENT RESULTS");
            println!("══════════════════════════════════════");
            println!("  Energy: {:.2}", refined_energy);
            println!("  Δmean SMA: {:+.2} m", -refined_energy);
            for (i, a) in refined.iter().enumerate() {
                println!("  CP[{}]: cone={:+.2}°, clock={:+.2}°", i, a.cone.to_degrees(), a.clock.to_degrees());
            }

            // Tag this eval as "refined_best"
            if let Some(conn) = context.connection.as_ref() {
                let conn_lock = conn.lock().unwrap();
                conn_lock.execute(
                    "INSERT INTO solutions (run_id, label, eval_id, energy, input) VALUES (?, 'refined_best', ?, ?, ?)",
                    [&context.run_id.to_string(), &eval_ctx.eval_id.to_string(), &format!("{:.6}", refined_energy), &format!("{:?}", refined)],
                ).unwrap();
            }
        }

        // Print SA results again after refinement
        {
            let eval_ctx = optimization::EvaluationContext::new_in_run(&context);
            let sa_energy = problem.energy(&best, &eval_ctx);
            println!("\n══════════════════════════════════════");
            println!("SIMULATED ANNEALING RESULTS (before refinement)");
            println!("══════════════════════════════════════");
            println!("  Energy: {:.2}", sa_energy);
            println!("  Δmean SMA: {:+.2} m", -sa_energy);
            for (i, a) in best.iter().enumerate() {
                println!("  CP[{}]: cone={:+.2}°, clock={:+.2}°", i, a.cone.to_degrees(), a.clock.to_degrees());
            }
        }

        refined
    } else {
        best
    };

    // Explicitly close database connection to avoid hanging at program exit
    println!("Closing database...");
    if let Some(conn) = context.connection {
        drop(conn);
    }
    
    println!("Done! Results saved to {}", DB_NAME);
    println!("Run visualization with: python plot/plot_trajectory.py");
    println!("\nFor validation:");
    println!("- Check that SMA increases over time");
    println!("- Optimal strategy should point sail ~perpendicular to velocity");
    println!("- Compare delta-V against analytical solution if possible");
}
