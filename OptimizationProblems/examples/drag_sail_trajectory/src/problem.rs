//! Deorbitation problem definition
//!
//! This module implements the solar sail deorbitation optimization problem.
//! It can be solved with different optimization algorithms (Simulated Annealing,
//! Genetic Algorithms, etc.) by implementing the corresponding trait.

use std::{
    collections::HashMap,
};

use maths_traits::analysis::InnerProductMetric;
use numerical_integration::AdaptiveIntegrator;
use rusqlite::Transaction;
use optimization::{EvaluationContext, OptimizableProblem};
use nalgebra::SVector;

use orbital_math::Vector;

use orbital_models::{OrbitalElements, StateVector, GravityModel};

use optimization::LoggingStrategy;
use crate::{
    control::{Angles, normalize_direction, normalize_elevation},
    DragModel,
};

/// Solar sail deorbitation optimization problem
///
/// This problem minimizes the time required to deorbit from an initial altitude
/// down to a deorbitation cutoff altitude using a solar sail.
///
/// # Type Parameters
/// * `SIZE` - Number of control points for the angle trajectory
pub struct DeorbitationProblem<const SIZE: usize> {
    /// Size of the solar sail in m²
    pub size: f64,
    /// Mass of the spacecraft in kg
    pub mass: f64,
    /// Drag coefficient, specific to the spacecraft
    pub drag_coefficient: f64,
    /// Altitude target in meters (unused currently, but reserved for future constraints)
    pub _altitude_target: f64,
    /// Deorbitation cutoff in meters
    pub deorbitation_cutoff: f64,
    /// Time cutoff in seconds (maximum simulation time)
    pub time_cutoff: f64,
    /// Initial state of the spacecraft
    pub initial_state: OrbitalElements,
    /// Logging strategy for trajectory data
    pub logging_strategy: LoggingStrategy,
}

/// Concatenate two arrays into a single array
fn concat_arrays<T: Copy, const N: usize, const M: usize>(a: [T; N], b: [T; M]) -> [T; N + M] {
    let mut result = [a[0]; N + M];
    result[..N].copy_from_slice(&a);
    result[N..].copy_from_slice(&b);
    result
}

/// Integration state snapshot
struct IntegrationState {
    time: f64,
    initial_altitude: f64,
    final_altitude: f64,
    _position: [f64; 3],
    _velocity: [f64; 3],
}

impl<const SIZE: usize> DeorbitationProblem<SIZE> {
    /// Evaluate a control trajectory by numerically integrating the dynamics
    ///
    /// # Arguments
    /// * `input` - Control angle trajectory (array of control points)
    /// * `context` - Evaluation context for logging
    ///
    /// # Returns
    /// Final integration state (time, position, velocity)
    fn evaluate(
        &self,
        input: <Self as OptimizableProblem>::Input,
        context: &EvaluationContext,
    ) -> IntegrationState {
        #[derive(Default, Clone, Debug)]
        struct StepData {
            info: HashMap<&'static str, f64>,
        }

        let eval_function = |_time: f64,
                             current_state: Vector<f64, 6>,
                             angles: Angles,
                             _step_data: &mut StepData| {
            let current_state_vector = StateVector::from(current_state);

            // Compute all forces/accelerations
            let pos: SVector<f64, 3> = current_state_vector.position().into();
            let vel: SVector<f64, 3> = current_state_vector.velocity().into();
            let gravity_accel = GravityModel::compute(&pos, self.mass);

            let drag_accel = DragModel::compute(
                &pos,
                &vel,
                self.size,
                self.mass,
                self.drag_coefficient,
                angles.elevation,
                angles.direction,
                &SVector::<f64, 3>::zeros(),
            );

            // Total acceleration (add more perturbations here as needed)
            gravity_accel + drag_accel
        };

        // Prepare the integrator
        let ds: f64 = 1e-8;
        let integrator = numerical_integration::RK_FELBERG;

        let initial_state = Vector::from(self.initial_state.as_cartesian());
        let initial_altitude = self.initial_state.altitude();

        let interpolation_strategy = context.optimization_context.interpolation_strategy;
        let integration_function = |time, state: Vector<f64, 6>| {
            let mut this_step_data = Default::default();

            // Interpolate control angles from control points
            let progress: f64 = (time * (input.len() as f64 - 2.0)) / self.time_cutoff;
            let prev_index = progress.floor() as usize;
            let prev_angle = input[prev_index];
            let next_angle = input[prev_index + 1];

            let angles: Angles =
                interpolation_strategy.interpolate(&prev_angle, &next_angle, progress - prev_index as f64);

            let accelerations = eval_function(time, state.clone(), angles, &mut this_step_data);

            let new_state =
                concat_arrays(StateVector::from(state).velocity(), accelerations.into());

            (this_step_data, Vector(SVector::<f64, 6>::from(new_state)))
        };

        let mut state0 = integrator.adaptive_init(
            0.0_f64,
            initial_state,
            ds,
            integration_function,
            InnerProductMetric,
        );
        let states = state0.as_mut();

        let connection = context.optimization_context.connection.as_ref().unwrap();

        let mut data_to_log = vec![];
        let mut timestep_index = 0usize;
        
        // Integration loop
        loop {
            let last_state = states.first().unwrap();
            let last_time = last_state.clone().0;
            let last_state_vector = StateVector::from(last_state.2.clone());
            let last_orbit = OrbitalElements::from_state_vector(last_state_vector);
            
            // Stop conditions
            if last_time > self.time_cutoff || last_orbit.altitude() < self.deorbitation_cutoff {
                break;
            }

            // Only log timesteps according to strategy
            if !self.logging_strategy.only_final_state() 
                && self.logging_strategy.should_log_timestep(timestep_index) {
                data_to_log.push((chrono::Utc::now(), states.to_owned()));
            }
            
            timestep_index += 1;
            integrator.adaptive_step(states, ds, integration_function, InnerProductMetric);
        }

        // Always save final state if using FinalStateOnly strategy
        if self.logging_strategy.only_final_state() {
            data_to_log.push((chrono::Utc::now(), states.to_owned()));
        }

        // Log timestep data to database
        let mut connection_lock = connection.lock().unwrap();
        let transaction = Transaction::new(
            &mut connection_lock,
            rusqlite::TransactionBehavior::Deferred,
        )
        .unwrap();
        
        let mut query = transaction
            .prepare(
                "CREATE TABLE IF NOT EXISTS timesteps (id PRIMARY KEY, eval_id TEXT, datetime DATETIME, simtime REAL, x REAL, y REAL, z REAL, vx REAL, vy REAL, vz REAL, step_data TEXT)")
            .unwrap();
        query.execute([]).unwrap();
        drop(query);
        
        let mut query = transaction
                .prepare(
                    "INSERT INTO timesteps (eval_id, datetime, simtime, x, y, z, vx, vy, vz, step_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                .unwrap();
        data_to_log.iter().for_each(|entry| {
            let (time, states) = entry;
            query
                .insert((
                    context.eval_id.to_string(),
                    time.to_string(),
                    states[0].0,
                    states[0].2 .0[0],
                    states[0].2 .0[1],
                    states[0].2 .0[2],
                    states[0].2 .0[3],
                    states[0].2 .0[4],
                    states[0].2 .0[5],
                    format!("{:?}", states[0].1.info),
                ))
                .unwrap();
        });

        drop(query);
        transaction.commit().unwrap();
        drop(connection_lock);

        let last_integrator_state = states.first().unwrap();
        let last_time = last_integrator_state.0;
        
        println!(
            "{} - Deorbitation time: {} seconds",
            context.eval_id, last_time
        );

        let last_state = last_integrator_state.2.clone();
        let last_position = [last_state.0[0], last_state.0[1], last_state.0[2]];
        let last_velocity = [last_state.0[3], last_state.0[4], last_state.0[5]];
        let last_state_vector = StateVector::from(last_state);
        let final_altitude = OrbitalElements::from_state_vector(last_state_vector).altitude();

        IntegrationState {
            time: last_time,
            initial_altitude,
            final_altitude,
            _position: last_position,
            _velocity: last_velocity,
        }
    }
}

// ==============================================================================
// SIMULATED ANNEALING IMPLEMENTATION
// ==============================================================================
// To switch to a different algorithm (e.g., Genetic Algorithm), implement
// the corresponding trait instead or in addition to this one.

impl<const SIZE: usize> OptimizableProblem for DeorbitationProblem<SIZE> {
    type Input = [Angles; SIZE];
    type Energy = f64;

    fn neighbour(
        &self,
        current: &Self::Input,
        temperature: f64,
        context: &EvaluationContext,
    ) -> Self::Input {
        // Generate neighbor using Gaussian perturbation
        // Standard deviation scales with temperature for exploration/exploitation balance
        let (_, perturbation_scale) = context.optimization_context.sa_config();
        let standard_deviation = perturbation_scale * temperature;

        let mut rng = rand::thread_rng();
        let generated_angles = std::array::from_fn(|i| {
            use rand_distr::Distribution;
            
            let gaussian =
                rand_distr::Normal::new(current.get(i).unwrap().direction, standard_deviation)
                    .unwrap();
            let dir = normalize_direction(gaussian.sample(&mut rng));

            let gaussian =
                rand_distr::Normal::new(current.get(i).unwrap().elevation, standard_deviation)
                    .unwrap();
            let elev = normalize_elevation(gaussian.sample(&mut rng));

            Angles {
                elevation: elev,
                direction: dir,
            }
        });
        generated_angles
    }

    fn acceptance(
        &self,
        current: &Self::Energy,
        new: &Self::Energy,
        temperature: f64,
        _context: &EvaluationContext,
    ) -> f64 {
        if new < current {
            // Always accept better solutions
            1.0
        } else {
            // Probabilistic acceptance of worse solutions
            let relative_diff = (new - current) / current;
            // 25% base acceptance rate as a tuning parameter
            (1.0 / relative_diff).ln() * temperature * 0.25
        }
    }

    fn energy(&self, current: &Self::Input, context: &EvaluationContext) -> Self::Energy {
        // Energy = negative average altitude loss rate (minimize)
        // A faster descent means a more negative rate, which is lower energy.
        // This works for both full deorbit (short time, large drop) and
        // partial deorbit (long time, smaller drop) — the optimizer always
        // has a gradient to follow.
        let state = self.evaluate(*current, context);
        let altitude_drop = state.initial_altitude - state.final_altitude;
        -(altitude_drop / state.time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optimization::{AlgorithmConfig, CoolingSchedule, InterpolationStrategy};
    use uuid::Uuid;
    use std::sync::{Arc, Mutex};
    use optimization::OptimizationContext;
    
    /// Create a test OptimizationContext with in-memory SQLite
    fn test_context() -> OptimizationContext {
        let connection = rusqlite::Connection::open_in_memory().unwrap();
        OptimizationContext {
            run_id: Uuid::now_v7(),
            connection: Some(Arc::new(Mutex::new(connection))),
            interpolation_strategy: InterpolationStrategy::Linear,
            algorithm_config: AlgorithmConfig::SimulatedAnnealing {
                cooling_schedule: CoolingSchedule::Logarithmic { c: 1.0 },
                perturbation_scale: std::f64::consts::PI / 16.0,
            },
        }
    }

    /// Regression test: drag sail deorbitation with fixed control angles.
    /// This test catches any changes to physics models, integrator, or drag
    /// computations that would alter the optimization output.
    #[test]
    fn drag_sail_energy_regression() {
        let problem = DeorbitationProblem::<5> {
            size: 1.1,
            mass: 20.0,
            drag_coefficient: 2.2,
            _altitude_target: 3000e3,
            deorbitation_cutoff: 250e3,
            time_cutoff: 60.0 * 60.0 * 24.0 * 10.0,
            initial_state: OrbitalElements {
                a: 460e3 + 6371e3,
                e: 0.0,
                i: 88.0f64.to_radians(),
                w: 0.0,
                o: 0.0,
                nu: 0.0,
            },
            logging_strategy: optimization::LoggingStrategy::FinalStateOnly,
        };

        let fixed_angles: [Angles; 5] = [Angles { elevation: 0.0, direction: 0.0 }; 5];
        let ctx = test_context();
        let eval_ctx = EvaluationContext::new_in_run(&ctx);
        let energy = problem.energy(&fixed_angles, &eval_ctx);

        // Golden value captured from current implementation.
        // If this test fails, it means something in the physics pipeline changed.
        // Update ONLY after verifying the change is intentional.
        let expected = -0.009265632985225269;
        let rel_err = (energy - expected).abs() / expected.abs();
        assert!(rel_err < 1e-6,
            "Drag sail energy regression failed!\n  Expected: {}\n  Got:      {}\n  Rel err:  {:.2e}\n\
             This means physics models, integrator, or drag code changed.",
            expected, energy, rel_err);
    }
}
