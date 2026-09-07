//! SA vs GA Comparison for Solar Sail Orbit Raising
//!
//! Runs both Simulated Annealing and Genetic Algorithm on the same problem
//! and compares their performance (best energy, wall time).

#![expect(incomplete_features)]
#![feature(inherent_associated_types)]
#![feature(generic_const_exprs)]

use std::sync::{Arc, Mutex};
use std::time::Instant;
use uuid::Uuid;

use rand::Rng;
use rand_distr::Distribution;

use optimization::{
    AlgorithmConfig, EvaluationContext, GAProblem, InterpolationStrategy,
    OptimizableProblem, OptimizationContext,
};
use optimization::LoggingStrategy;
use solar_sail_trajectory::{
    control::{Angles, normalize_clock, normalize_cone},
    problem::{ControlMode, OrbitRaisingProblem},
    CoolingSchedule,
    MissionConfig,
};

// ============================================================================
// GA wrapper + implementation for OrbitRaisingProblem
// ============================================================================

/// Newtype wrapper to implement GAProblem (orphan rule: both trait and inner
/// type are from foreign crates as seen by this binary).
struct GAProblemWrapper<const N: usize>(OrbitRaisingProblem<N>);

impl<const N: usize> GAProblem for GAProblemWrapper<N> {
    type Individual = [Angles; N];
    type Fitness = f64;

    fn fitness(&self, individual: &Self::Individual, context: &EvaluationContext) -> Self::Fitness {
        // Reuse SA's energy function (lower = better)
        self.0.energy(individual, context)
    }

    fn random_individual(&self) -> Self::Individual {
        core::array::from_fn(|_| Angles::random())
    }

    fn crossover(
        &self,
        parent1: &Self::Individual,
        parent2: &Self::Individual,
        _context: &EvaluationContext,
    ) -> Self::Individual {
        // Two-point crossover: preserves contiguous segments from each parent,
        // which matters because adjacent control points are correlated (slew rate).
        let mut rng = rand::thread_rng();
        let mut pt1 = rng.gen_range(0..N);
        let mut pt2 = rng.gen_range(0..N);
        if pt1 > pt2 {
            std::mem::swap(&mut pt1, &mut pt2);
        }
        core::array::from_fn(|i| {
            if i >= pt1 && i < pt2 {
                parent2[i]
            } else {
                parent1[i]
            }
        })
    }

    fn mutate(
        &self,
        individual: &Self::Individual,
        mutation_rate: f64,
        _context: &EvaluationContext,
    ) -> Self::Individual {
        let mut rng = rand::thread_rng();
        let std_dev = std::f64::consts::PI / 16.0; // Same scale as SA perturbation

        // Time between consecutive control points
        let segment_duration = self.0.target_time / (N - 1) as f64;
        let max_angular_change = self.0.max_slew_rate * segment_duration;

        let mut result = *individual;

        for i in 0..N {
            if rng.gen::<f64>() < mutation_rate {
                let gaussian_cone = rand_distr::Normal::new(result[i].cone, std_dev).unwrap();
                let gaussian_clock = rand_distr::Normal::new(result[i].clock, std_dev).unwrap();
                result[i].cone = normalize_cone(gaussian_cone.sample(&mut rng));
                result[i].clock = normalize_clock(gaussian_clock.sample(&mut rng));
            }
        }

        // Enforce slew rate constraints between consecutive points
        for i in 1..N {
            let mut cone_diff = result[i].cone - result[i - 1].cone;
            let cone_range = std::f64::consts::PI;
            if cone_diff > cone_range / 2.0 {
                cone_diff -= cone_range;
            } else if cone_diff < -cone_range / 2.0 {
                cone_diff += cone_range;
            }
            if cone_diff.abs() > max_angular_change {
                cone_diff = cone_diff.signum() * max_angular_change;
                result[i].cone = normalize_cone(result[i - 1].cone + cone_diff);
            }

            let mut clock_diff = result[i].clock - result[i - 1].clock;
            let clock_range = std::f64::consts::PI;
            if clock_diff > clock_range / 2.0 {
                clock_diff -= clock_range;
            } else if clock_diff < -clock_range / 2.0 {
                clock_diff += clock_range;
            }
            if clock_diff.abs() > max_angular_change {
                clock_diff = clock_diff.signum() * max_angular_change;
                result[i].clock = normalize_clock(result[i - 1].clock + clock_diff);
            }
        }

        result
    }
}

// ============================================================================
// Main comparison
// ============================================================================

fn main() {
    const DB_NAME: &str = "out/compare_sa_ga.db";
    const NUM_CONTROL_POINTS: usize = 10;
    const SA_GENERATIONS: u64 = 3000;
    const GA_GENERATIONS: u64 = 30; // Fewer generations since each evaluates population_size individuals

    let logging = LoggingStrategy::FinalStateOnly;

    // Setup database
    std::fs::create_dir_all("out").unwrap();
    let _ = std::fs::remove_file(DB_NAME);
    let connection = Arc::new(Mutex::new(rusqlite::Connection::open(DB_NAME).unwrap()));

    // Define problem (shared between SA and GA)
    let config = MissionConfig::optimization_defaults();
    let problem = OrbitRaisingProblem::<NUM_CONTROL_POINTS> {
        sail_area: config.sail_area,
        mass: config.mass,
        reflectivity: config.reflectivity,
        drag_coefficient: config.drag_coefficient,
        target_sma: config.target_sma(),
        target_time: config.target_time,
        max_slew_rate: config.max_slew_rate,
        initial_state: config.initial_orbit(),
        logging_strategy: logging,
        control_mode: ControlMode::ControlPoints,
        local_refinement: false,
        local_refinement_iterations: 0,
    };

    println!("==========================================================");
    println!("  SA vs GA Comparison - Solar Sail Orbit Raising");
    println!("==========================================================");
    println!("  Control points:   {}", NUM_CONTROL_POINTS);
    println!("  SA generations:   {}", SA_GENERATIONS);
    println!("  GA generations:   {} (x{} population)", GA_GENERATIONS, 200);
    println!();

    // ── Simulated Annealing ──────────────────────────────────────────────
    let sa_context = OptimizationContext {
        run_id: Uuid::now_v7(),
        connection: Some(Arc::clone(&connection)),
        interpolation_strategy: InterpolationStrategy::Linear,
        algorithm_config: AlgorithmConfig::SimulatedAnnealing {
            cooling_schedule: CoolingSchedule::Logarithmic { c: 1.0 },
            perturbation_scale: std::f64::consts::PI / 16.0,
        },
    };

    let initial: [Angles; NUM_CONTROL_POINTS] = core::array::from_fn(|_| Angles {
        cone: 0.5,
        clock: 0.0,
    });

    println!("Running Simulated Annealing...");
    let sa_start = Instant::now();
    let sa_best = problem.optimize(initial, SA_GENERATIONS, sa_context.clone());
    let sa_elapsed = sa_start.elapsed();

    let sa_eval_ctx = EvaluationContext::new_in_run(&sa_context);
    let sa_energy = problem.energy(&sa_best, &sa_eval_ctx);

    // ── Genetic Algorithm ────────────────────────────────────────────────
    let ga_context = OptimizationContext {
        run_id: Uuid::now_v7(),
        connection: Some(Arc::clone(&connection)),
        interpolation_strategy: InterpolationStrategy::Linear,
        algorithm_config: AlgorithmConfig::GeneticAlgorithm {
            population_size: 100,
            crossover_rate: 0.8,
            mutation_rate: 0.2,
            elitism_count: 5,
            tournament_size: 3,
        },
    };

    let ga_wrapper = GAProblemWrapper(OrbitRaisingProblem::<NUM_CONTROL_POINTS> {
        sail_area: config.sail_area,
        mass: config.mass,
        reflectivity: config.reflectivity,
        drag_coefficient: config.drag_coefficient,
        target_sma: config.target_sma(),
        target_time: config.target_time,
        max_slew_rate: config.max_slew_rate,
        initial_state: config.initial_orbit(),
        logging_strategy: logging,
        control_mode: ControlMode::ControlPoints,
        local_refinement: false,
        local_refinement_iterations: 0,
    });

    println!("\nRunning Genetic Algorithm...");
    let ga_start = Instant::now();
    let ga_best = ga_wrapper.optimize_ga(GA_GENERATIONS, ga_context.clone());
    let ga_elapsed = ga_start.elapsed();

    let ga_eval_ctx = EvaluationContext::new_in_run(&ga_context);
    let ga_energy = ga_wrapper.fitness(&ga_best, &ga_eval_ctx);

    // ── Results ──────────────────────────────────────────────────────────
    println!("\n==========================================================");
    println!("  RESULTS");
    println!("==========================================================");
    println!();
    println!("  {:>25}  {:>12}  {:>12}", "", "SA", "GA");
    println!("  {:>25}  {:>12}  {:>12}", "-".repeat(25), "-".repeat(12), "-".repeat(12));
    println!("  {:>25}  {:>12.2}  {:>12.2}", "Energy", sa_energy, ga_energy);
    println!("  {:>25}  {:>+12.2}  {:>+12.2}", "Delta mean SMA (m)", -sa_energy, -ga_energy);
    println!("  {:>25}  {:>12.1?}  {:>12.1?}", "Wall time", sa_elapsed, ga_elapsed);
    println!();

    if sa_energy < ga_energy {
        println!("  Winner: SA (by {:.2} m delta SMA)", ga_energy - sa_energy);
    } else if ga_energy < sa_energy {
        println!("  Winner: GA (by {:.2} m delta SMA)", sa_energy - ga_energy);
    } else {
        println!("  Tie!");
    }

    println!("\n  Results saved to: {}", DB_NAME);
}
