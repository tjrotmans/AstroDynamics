use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::sa::CoolingSchedule;

/// Algorithm-specific configuration.
///
/// Using an enum ensures invalid states are unrepresentable — you can't
/// accidentally mix SA cooling schedules with GA population parameters.
#[derive(Clone, Debug)]
pub enum AlgorithmConfig {
    SimulatedAnnealing {
        cooling_schedule: CoolingSchedule,
        /// Perturbation scale for neighbour generation (radians at T=1).
        perturbation_scale: f64,
    },
    GeneticAlgorithm {
        population_size: usize,
        crossover_rate: f64,
        mutation_rate: f64,
        elitism_count: usize,
        tournament_size: usize,
    },
}

/// Interpolation strategy for control inputs between control points.
///
/// Determines how the optimizer transitions between defined control points
/// during trajectory evaluation.
#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
pub enum InterpolationStrategy {
    /// Linear interpolation: smoothly transition between control points.
    #[default]
    Linear,

    /// Piecewise constant: hold the previous control point's value until the next one.
    Piecewise,
}

impl InterpolationStrategy {
    /// Interpolate between two values based on the strategy.
    ///
    /// * `prev` - Value at the previous control point
    /// * `next` - Value at the next control point
    /// * `t` - Fractional progress between the two points (0.0 to 1.0)
    pub fn interpolate<T: interpolation::Lerp<Scalar = f64> + Copy>(&self, prev: &T, next: &T, t: f64) -> T {
        match self {
            InterpolationStrategy::Linear => interpolation::lerp(prev, next, &t),
            InterpolationStrategy::Piecewise => *prev,
        }
    }
}

#[derive(Clone)]
pub struct OptimizationContext {
    pub run_id: Uuid,
    pub connection: Option<Arc<Mutex<rusqlite::Connection>>>,
    pub interpolation_strategy: InterpolationStrategy,
    pub algorithm_config: AlgorithmConfig,
}

impl OptimizationContext {
    /// Extract SA-specific config. Panics if not SA.
    pub fn sa_config(&self) -> (&CoolingSchedule, f64) {
        match &self.algorithm_config {
            AlgorithmConfig::SimulatedAnnealing { cooling_schedule, perturbation_scale } => {
                (cooling_schedule, *perturbation_scale)
            }
            _ => panic!("Expected SimulatedAnnealing config"),
        }
    }

    /// Extract GA-specific config. Panics if not GA.
    pub fn ga_config(&self) -> (usize, f64, f64, usize, usize) {
        match &self.algorithm_config {
            AlgorithmConfig::GeneticAlgorithm {
                population_size, crossover_rate, mutation_rate, elitism_count, tournament_size
            } => (*population_size, *crossover_rate, *mutation_rate, *elitism_count, *tournament_size),
            _ => panic!("Expected GeneticAlgorithm config"),
        }
    }
}

#[derive(Clone)]
pub struct EvaluationContext<'optctx> {
    pub eval_id: Uuid,
    pub optimization_context: &'optctx OptimizationContext,
}

impl<'optctx> EvaluationContext<'optctx> {
    pub fn new_in_run(ctx: &'optctx OptimizationContext) -> Self {
        EvaluationContext {
            optimization_context: ctx,
            eval_id: Uuid::now_v7(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_interpolation_midpoint() {
        let strategy = InterpolationStrategy::Linear;
        let result = strategy.interpolate(&1.0_f64, &3.0_f64, 0.5);
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn linear_interpolation_boundaries() {
        let strategy = InterpolationStrategy::Linear;
        let result_start = strategy.interpolate(&1.0_f64, &3.0_f64, 0.0);
        let result_end = strategy.interpolate(&1.0_f64, &3.0_f64, 1.0);
        assert!((result_start - 1.0).abs() < 1e-10);
        assert!((result_end - 3.0).abs() < 1e-10);
    }

    #[test]
    fn piecewise_returns_previous() {
        let strategy = InterpolationStrategy::Piecewise;
        let result = strategy.interpolate(&1.0_f64, &3.0_f64, 0.5);
        assert!((result - 1.0).abs() < 1e-10);
        let result = strategy.interpolate(&1.0_f64, &3.0_f64, 0.99);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sa_config_extraction() {
        let ctx = OptimizationContext {
            run_id: Uuid::now_v7(),
            connection: None,
            interpolation_strategy: InterpolationStrategy::Linear,
            algorithm_config: AlgorithmConfig::SimulatedAnnealing {
                cooling_schedule: CoolingSchedule::Linear,
                perturbation_scale: 0.5,
            },
        };
        let (schedule, scale) = ctx.sa_config();
        assert!(matches!(schedule, CoolingSchedule::Linear));
        assert!((scale - 0.5).abs() < 1e-10);
    }

    #[test]
    fn ga_config_extraction() {
        let ctx = OptimizationContext {
            run_id: Uuid::now_v7(),
            connection: None,
            interpolation_strategy: InterpolationStrategy::Linear,
            algorithm_config: AlgorithmConfig::GeneticAlgorithm {
                population_size: 100,
                crossover_rate: 0.8,
                mutation_rate: 0.2,
                elitism_count: 5,
                tournament_size: 3,
            },
        };
        let (pop, cross, mut_rate, elite, tourn) = ctx.ga_config();
        assert_eq!(pop, 100);
        assert!((cross - 0.8).abs() < 1e-10);
        assert!((mut_rate - 0.2).abs() < 1e-10);
        assert_eq!(elite, 5);
        assert_eq!(tourn, 3);
    }

    #[test]
    #[should_panic(expected = "Expected SimulatedAnnealing config")]
    fn sa_config_panics_on_ga() {
        let ctx = OptimizationContext {
            run_id: Uuid::now_v7(),
            connection: None,
            interpolation_strategy: InterpolationStrategy::Linear,
            algorithm_config: AlgorithmConfig::GeneticAlgorithm {
                population_size: 100,
                crossover_rate: 0.8,
                mutation_rate: 0.2,
                elitism_count: 5,
                tournament_size: 3,
            },
        };
        ctx.sa_config();
    }
}
