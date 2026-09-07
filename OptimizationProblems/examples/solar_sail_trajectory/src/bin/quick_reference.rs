//! Reference trajectory generator using locally optimal steering laws
//!
//! This generates a reference trajectory using the analytically derived
//! planet-centered locally optimal steering law from McInnes (Sec. 4.4.2.3).
//!
//! For now, this uses a simple approximation: compute the optimal angles
//! at evenly spaced true anomaly points (0°, 36°, 72°, ..., 324°) for one orbit.
//!
//! Usage:
//!   cargo run --release --bin quick_reference

use optimization::LoggingStrategy;
use solar_sail_trajectory::{
    control::Angles,
    problem::OrbitRaisingProblem,
    ControlMode,
    CoolingSchedule,
    MissionConfig,
};
use optimization::{EvaluationContext, OptimizableProblem, OptimizationContext, AlgorithmConfig, InterpolationStrategy};
use std::sync::{Arc, Mutex};
use rusqlite::Connection;

fn main() {
    println!("Reference Trajectory Generator (Locally Optimal Steering)");
    println!("==========================================================\n");

    // Parameters - should match main.rs for comparison
    let mut config = MissionConfig::optimization_defaults();
    config.max_slew_rate = 10.0 * (std::f64::consts::PI / 180.0) / 60.0;

    let target_sma = config.target_sma();
    
    println!("Initial altitude: {:.1} km", config.initial_altitude / 1000.0);
    println!("Target altitude: {:.1} km", config.target_altitude / 1000.0);
    println!("Sail area: {} m²", config.sail_area);
    println!("Mass: {} kg", config.mass);
    println!("Reflectivity: {}", config.reflectivity);
    println!("Simulation time: {:.2} s ({:.4} days)", config.target_time, config.target_time / (24.0 * 3600.0));
    println!("\nUsing: McInnes planet-centered locally optimal steering (Eq. 4.100)\n");

    // Create initial circular orbit
    let initial_orbit = config.initial_orbit();
    
    // Number of control points
    const NUM_CONTROL_POINTS: usize = 30;
    
    let control_angles: [Angles; NUM_CONTROL_POINTS] = [Angles { cone: 0.0, clock: 0.0 }; NUM_CONTROL_POINTS];
    println!("Using realtime locally optimal steering (computed every timestep).");

    // Create problem
    let problem = OrbitRaisingProblem::<NUM_CONTROL_POINTS> {
        sail_area: config.sail_area,
        mass: config.mass,
        reflectivity: config.reflectivity,
        drag_coefficient: config.drag_coefficient,
        target_sma,
        target_time: config.target_time,
        max_slew_rate: config.max_slew_rate,
        initial_state: initial_orbit,
        logging_strategy: LoggingStrategy::AllDownsampled(1),
        control_mode: ControlMode::ControlPoints,
        local_refinement: false,
        local_refinement_iterations: 0,
    };

    // Create database connection
    let db_path = "out/reference.db";
    std::fs::create_dir_all("out").unwrap();
    std::fs::remove_file(db_path).ok();
    
    let connection = Arc::new(Mutex::new(Connection::open(db_path).unwrap()));
    
    // Initialize database
    {
        let conn_lock = connection.lock().unwrap();
        conn_lock
            .execute(
                "CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    eval_id TEXT NOT NULL,
                    run_id TEXT,
                    energy REAL,
                    accepted TEXT
                )",
                [],
            )
            .unwrap();
    }

    let run_id = uuid::Uuid::now_v7();
    let eval_id = uuid::Uuid::now_v7();
    let optimization_context = OptimizationContext {
        run_id,
        connection: Some(connection),
        interpolation_strategy: InterpolationStrategy::Linear,
        algorithm_config: AlgorithmConfig::SimulatedAnnealing {
            cooling_schedule: CoolingSchedule::Linear,
            perturbation_scale: std::f64::consts::PI / 4.0, // Not used for single sim
        },
    };
    
    let eval_context = EvaluationContext {
        eval_id,
        optimization_context: &optimization_context,
    };

    println!("Running simulation with locally optimal steering...");
    println!("Eval ID: {}\n", eval_context.eval_id);
    
    // Run evaluation
    let energy = problem.energy(&control_angles, &eval_context);
    
    // Record evaluation in generations table
    {
        let conn_lock = optimization_context.connection.as_ref().unwrap().lock().unwrap();
        conn_lock.execute(
            "INSERT INTO generations (eval_id, run_id, energy, accepted) VALUES (?, ?, ?, 'true')",
            [eval_context.eval_id.to_string(), optimization_context.run_id.to_string(), energy.to_string()],
        ).unwrap();
    }
    
    println!("\n======================================");
    println!("Reference trajectory complete!");
    println!("Energy (= -Δmean SMA): {:.2}", energy);
    println!("Mean SMA gain: {:+.2} m", -energy);
    println!("\nDatabase saved to: {}", db_path);
    println!("\nTo compare with optimized solution:");
    println!("  DB_PATH=out/quick_sim.db python plot/plot_trajectory.py --plots kepler angles --reference");
}
