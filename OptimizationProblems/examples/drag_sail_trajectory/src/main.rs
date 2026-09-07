//! Solar Sail Deorbitation Trajectory Optimizer
//!
//! This example optimizes the pointing angles of a solar sail to minimize
//! deorbitation time from 500km to 300km altitude using Simulated Annealing.
//!
//! The trajectory is represented as piecewise linear interpolation between
//! control points (angle pairs).

#![expect(incomplete_features)]
#![feature(inherent_associated_types)]
#![feature(generic_const_exprs)]

use std::sync::{Arc, Mutex};
use uuid::Uuid;
use optimization::{OptimizableProblem, OptimizationContext, AlgorithmConfig, CoolingSchedule, InterpolationStrategy};

mod control;
mod problem;
/// Frame-specific type alias for drag sail acceleration model
type DragModel = orbital_models::DragModel<orbital_models::VelocityFrame>;

use orbital_models::OrbitalElements;
use optimization::LoggingStrategy;
use control::Angles;
use problem::DeorbitationProblem;

fn main() {
    // Configuration
    const DB_NAME: &str = "out/drag_sail_trajectory.db";
    const NUM_CONTROL_POINTS: usize = 10;
    const NUM_GENERATIONS: u64 = 1000;

    // Logging strategy - choose one:
    // Option 1: LoggingStrategy::AllDownsampled(100)
    //   - Saves every 100th timestep for ALL trajectories
    //   - Database size: ~90 MB (100x reduction from 9 GB)
    //   - Good for: Visualizing all trajectories with some detail
    //
    // Option 2: LoggingStrategy::BestEveryNthGeneration(20)
    //   - Only saves complete trajectories for best solution every 20 generations
    //   - Database size: ~450 MB (20x reduction)
    //   - Good for: Tracking optimization progress with key trajectories
    //
    // Option 3: LoggingStrategy::FinalStateOnly
    //   - Only saves final position/velocity/time for each evaluation
    //   - Database size: ~1 MB (9000x reduction)
    //   - Good for: Just optimization results, no visualization
    //
    // Option 4: LoggingStrategy::None
    //   - No trajectory data at all, only generations table
    //   - Database size: ~500 KB (18000x reduction)
    //   - Good for: Pure optimization, check results after
    let logging = LoggingStrategy::AllDownsampled(1000);

    // Setup database
    std::fs::create_dir_all("out").unwrap();
    // Clear old database if it exists
    let _ = std::fs::remove_file(DB_NAME);
    let connection = rusqlite::Connection::open(DB_NAME).unwrap();
    let run_id = Uuid::now_v7();

    // Define the problem
    let problem = DeorbitationProblem::<NUM_CONTROL_POINTS> {
        size: 1.1,                              // 1.1 m² sail
        mass: 20.0,                             // 20 kg spacecraft
        drag_coefficient: 2.2,
        _altitude_target: 3000e3,               // Future use
        deorbitation_cutoff: 250e3,             // 300 km
        time_cutoff: 60.0 * 60.0 * 24.0 * 10.0, // 1 year max
        initial_state: OrbitalElements {
            a: 460e3 + 6371e3,                  // 460 km altitude
            e: 0.0,                              // Circular orbit
            i: 88.0f64.to_radians(),            // Near-polar
            w: 0.0,
            o: 0.0,
            nu: 0.0,
        },        logging_strategy: logging,    };

    // Generate random initial guess
    let initial: [Angles; NUM_CONTROL_POINTS] = core::array::from_fn(|_| Angles::random());

    // Setup optimization context
    let context = OptimizationContext {
        run_id,
        connection: Some(Arc::new(Mutex::new(connection))),
        interpolation_strategy: InterpolationStrategy::Linear,
        algorithm_config: AlgorithmConfig::SimulatedAnnealing {
            cooling_schedule: CoolingSchedule::Logarithmic { c: 1.0 },
            perturbation_scale: std::f64::consts::PI / 16.0,
        },
    };

    // Run optimization (using Simulated Annealing)
    println!("Starting optimization with {} generations...", NUM_GENERATIONS);
    let _best = problem.optimize(initial, NUM_GENERATIONS, context.clone());

    // Explicitly close database connection to avoid hanging at program exit
    println!("Closing database...");
    if let Some(conn) = context.connection {
        drop(conn);
    }
    println!("Done! Results saved to {}", DB_NAME);
    println!("Run visualization with: python run_plotter.py");
}
