//! Quick simulation runner - test a single trajectory without optimization
//!
//! This allows you to quickly test trajectories with specific control angles
//!
//! Usage:
//!   cargo run --release --bin quick_sim           # Run with default angles
//!   cargo run --release --bin quick_sim -- --best # Load best angles from optimization DB

use orbital_models::OrbitalElements;
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

/// Load the best control angles from the optimization database
fn load_best_angles_from_db(db_path: &str) -> Option<Vec<Angles>> {
    let conn = Connection::open(db_path).ok()?;

    // Try to load the refined_best solution first
    let mut stmt = conn.prepare(
        "SELECT input FROM solutions WHERE label = 'refined_best' ORDER BY id DESC LIMIT 1"
    ).ok()?;
    let mut rows = stmt.query([]).ok()?;
    if let Some(row) = rows.next().ok()? {
        let input_str: String = row.get(0).ok()?;
        return parse_angles_from_debug_string(&input_str);
    }

    // Fallback: try to load the sa_best solution
    let mut stmt = conn.prepare(
        "SELECT input FROM solutions WHERE label = 'sa_best' ORDER BY id DESC LIMIT 1"
    ).ok()?;
    let mut rows = stmt.query([]).ok()?;
    if let Some(row) = rows.next().ok()? {
        let input_str: String = row.get(0).ok()?;
        return parse_angles_from_debug_string(&input_str);
    }

    // Fallback: best from generations table (legacy)
    let input_str: String = conn.query_row(
        "SELECT input FROM generations ORDER BY energy ASC LIMIT 1",
        [],
        |row| row.get(0),
    ).ok()?;
    parse_angles_from_debug_string(&input_str)
}

// Helper function to parse angles from the Debug string
fn parse_angles_from_debug_string(input_str: &str) -> Option<Vec<Angles>> {
    let mut angles = Vec::new();
    for part in input_str.split("Angles { ") {
        if part.contains("cone:") {
            let cone_start = part.find("cone: ").map(|i| i + 6)?;
            let cone_end = part[cone_start..].find(',')?;
            let cone: f64 = part[cone_start..cone_start + cone_end].trim().parse().ok()?;

            let clock_start = part.find("clock: ").map(|i| i + 7)?;
            let clock_end = part[clock_start..].find(' ').unwrap_or(part[clock_start..].find('}').unwrap_or(part.len() - clock_start));
            let clock: f64 = part[clock_start..clock_start + clock_end].trim().parse().ok()?;

            angles.push(Angles { cone, clock });
        }
    }
    if angles.is_empty() {
        None
    } else {
        Some(angles)
    }
}

fn main() {
    println!("Quick Solar Sail Trajectory Simulator");
    println!("======================================\n");

    // Check command line args
    let args: Vec<String> = std::env::args().collect();
    let use_best = args.iter().any(|a| a == "--best");
    
    // Number of control points - must match the optimization!
    const NUM_CONTROL_POINTS: usize = 10;
    
    let control_angles: [Angles; NUM_CONTROL_POINTS] = if use_best {
        // Load best angles from optimization database
        let opt_db_path = "out/optimization.db";
        println!("Loading best solution from: {}", opt_db_path);
        
        match load_best_angles_from_db(opt_db_path) {
            Some(angles) => {
                if angles.len() != NUM_CONTROL_POINTS {
                    eprintln!("Error: Found {} angles but expected {}", angles.len(), NUM_CONTROL_POINTS);
                    eprintln!("Make sure NUM_CONTROL_POINTS matches the optimization!");
                    std::process::exit(1);
                }
                
                println!("Loaded {} control angles:", angles.len());
                for (i, a) in angles.iter().enumerate() {
                    println!("  [{}] cone: {:6.2}°, clock: {:6.2}°", 
                        i, a.cone.to_degrees(), a.clock.to_degrees());
                }
                println!();
                
                let mut arr = [Angles { cone: 0.0, clock: 0.0 }; NUM_CONTROL_POINTS];
                for (i, a) in angles.iter().enumerate() {
                    arr[i] = *a;
                }
                arr
            }
            None => {
                eprintln!("Error: Could not load angles from {}", opt_db_path);
                eprintln!("Make sure you have run the optimization first!");
                std::process::exit(1);
            }
        }
    } else {
        // Default: constant attitude
        println!("Using default constant angles (use --best to load from optimization)");
        let default_angles = [
            Angles { cone: 0.25*std::f64::consts::PI, clock: 0.25*std::f64::consts::PI }; NUM_CONTROL_POINTS
        ];
        println!("Control angles: {} points (constant)", NUM_CONTROL_POINTS);
        println!("  Cone angle: {:.1}°", (0.25*std::f64::consts::PI).to_degrees());
        println!("  Clock angle: {:.1}°\n", (0.25*std::f64::consts::PI).to_degrees());
        default_angles
    };

    // ========== PARAMETERS - MUST MATCH main.rs WHEN USING --best ==========
    // When using --best, these should match the optimization parameters exactly
    let default_config = MissionConfig::optimization_defaults();

    let (
        sail_area,
        reflectivity,
        target_time,
        max_slew_rate,
        drag_coefficient,
        mass,
        initial_orbit,
        target_sma,
        initial_altitude,
        target_altitude,
    ) = if use_best {
        // Match main.rs optimization parameters exactly
        (
            default_config.sail_area,
            default_config.reflectivity,
            default_config.target_time,
            default_config.max_slew_rate,
            default_config.drag_coefficient,
            default_config.mass,
            default_config.initial_orbit(),
            default_config.target_sma(),
            default_config.initial_altitude,
            default_config.target_altitude,
        )
    } else {
        // Default test parameters
        let initial_altitude = 2000e3; // 2000 km altitude
        let target_altitude = 2500e3;  // 2500 km target
        let earth_radius = OrbitalElements::EARTH_RADIUS;
        let initial_sma = earth_radius + initial_altitude;
        let target_sma = earth_radius + target_altitude;
        let target_time = 24.0 * 3600.0 * 20.0; // 20 days
        let max_slew_rate = 10.0 * (std::f64::consts::PI / 180.0) / 60.0; // 10 deg/min

        let initial_orbit = OrbitalElements {
            a: initial_sma,
            e: 0.0,
            i: 98.0f64.to_radians(),
            w: 0.0,
            o: 90.0f64.to_radians() - 90.0f64.to_radians(),
            nu: 0.0,
        };

        (
            400.0, // sail_area
            1.8,   // reflectivity
            target_time,
            max_slew_rate,
            2.2,  // drag_coefficient
            10.0, // mass
            initial_orbit,
            target_sma,
            initial_altitude,
            target_altitude,
        )
    };
    println!("{:?}", default_config);

    println!("Initial altitude: {:.1} km", initial_altitude / 1000.0);
    println!("Target altitude: {:.1} km", target_altitude / 1000.0);
    println!("Sail area: {} m²", sail_area);
    println!("Mass: {} kg", mass);
    println!("Reflectivity: {}", reflectivity);
    println!("Simulation time: {:.2} s ({:.4} days)\n", target_time, target_time / (24.0 * 3600.0));

    // Create problem
    let problem = OrbitRaisingProblem::<NUM_CONTROL_POINTS> {
        sail_area,
        mass,
        reflectivity,
        drag_coefficient,
        target_sma,
        target_time,
        max_slew_rate,
        initial_state: initial_orbit,
        logging_strategy: LoggingStrategy::AllDownsampled(10), // Log every 10th timestep (faster)
        control_mode: ControlMode::ControlPoints,
        local_refinement: false,
        local_refinement_iterations: 0,
    };

    // Create database connection
    let db_path = "out/quick_sim.db";
    std::fs::create_dir_all("out").unwrap();
    std::fs::remove_file(db_path).ok(); // Delete old database
    
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
            cooling_schedule: CoolingSchedule::Linear, // Not used for single sim
            perturbation_scale: std::f64::consts::PI / 16.0,
        },
    };

    let eval_context = EvaluationContext {
        eval_id,
        optimization_context: &optimization_context,
    };

    println!("Running simulation...");
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
    println!("Simulation complete!");
    // Energy = -Δmean_sma (negative because we minimize to maximize SMA gain)
    println!("Energy: {:.2} (Δmean SMA: {:+.2} m)", energy, -energy);
    println!("\nDatabase saved to: {}", db_path);
    println!("Run plotting with: DB_PATH={} python plot/plot_trajectory.py", db_path);
}
