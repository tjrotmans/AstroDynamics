//! Solar sail orbit raising problem definition
//!
//! This module implements the solar sail orbit raising optimization problem.
//! The goal is to maximize the semi-major axis (raise the orbit) using
//! solar radiation pressure.

use std::{
    collections::HashMap
};

use maths_traits::analysis::InnerProductMetric;
use numerical_integration::AdaptiveIntegrator;
use rusqlite::Transaction;
use optimization::{OptimizableProblem, EvaluationContext, OptimizationContext};
use nalgebra::SVector;

use orbital_math::Vector;

use orbital_models::{sun_direction_from_position, OrbitalElements, StateVector, GravityModel};
use orbital_models::constants::SUN_POSITION;

use optimization::LoggingStrategy;
use crate::{
    control::{Angles, normalize_clock, normalize_cone, LocallyOptimalSMA},
    SolarPressureModel, DragModel,
};

/// Control mode for generating sail angles during integration.
#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
pub enum ControlMode {
    /// Use the provided control points (optimization mode).
    #[default]
    ControlPoints,
    /// Compute locally optimal angles at every timestep.
    LocallyOptimalRealtime, 
}

/// Solar sail orbit raising optimization problem
///
/// This problem minimizes the difference between achieved and target semi-major axis
/// at a target time using a solar sail.
/// The sail uses solar radiation pressure to gradually raise the orbit.
///
/// # Type Parameters
/// * `SIZE` - Number of control points for the angle trajectory
pub struct OrbitRaisingProblem<const SIZE: usize> {
    /// Size of the solar sail in m²
    pub sail_area: f64,
    /// Mass of the spacecraft in kg
    pub mass: f64,
    /// Reflectivity coefficient (0-2, where 1=perfect absorption, 2=perfect reflection)
    pub reflectivity: f64,
    /// Drag coefficient (for residual atmosphere at low altitudes)
    pub drag_coefficient: f64,
    /// Target semi-major axis in meters
    pub target_sma: f64,
    /// Target mission time in seconds
    pub target_time: f64,
    /// Maximum slew rate in radians per second (e.g., 10 deg/min = 0.00291 rad/s)
    pub max_slew_rate: f64,
    /// Initial state of the spacecraft
    pub initial_state: OrbitalElements,
    /// Logging strategy for trajectory data
    pub logging_strategy: LoggingStrategy,
    /// Control mode for angle generation
    pub control_mode: ControlMode,
    /// After SA finishes, do coordinate-wise hill climbing to polish the solution
    pub local_refinement: bool,
    /// Number of refinement iterations per control point
    pub local_refinement_iterations: usize,
}

/// Concatenate two arrays into a single array
fn concat_arrays<T: Copy, const N: usize, const M: usize>(a: [T; N], b: [T; M]) -> [T; N + M] {
    let mut result = [a[0]; N + M];
    result[..N].copy_from_slice(&a);
    result[N..].copy_from_slice(&b);
    result
}

/// Integration state snapshot
#[allow(dead_code)]
struct IntegrationState {
    time: f64,
    semi_major_axis: f64,
    mean_semi_major_axis: f64,  // Mean SMA over trajectory (more stable for optimization)
    _position: [f64; 3],
    _velocity: [f64; 3],
}

impl<const N: usize> OrbitRaisingProblem<N> {
    /// Evaluate a control trajectory by numerically integrating the dynamics
    ///
    /// # Arguments
    /// * `input` - Control angle trajectory (array of control points)
    /// * `context` - Evaluation context for logging
    ///
    /// # Returns
    /// Final integration state (time, SMA, position, velocity)
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

            // Get position and velocity as SVector for acceleration models
            let position: SVector<f64, 3> = current_state_vector.position().into();
            let velocity: SVector<f64, 3> = current_state_vector.velocity().into();

            // Compute all forces/accelerations using Cartesian coordinates directly
            let gravity_accel = GravityModel::compute(&position, self.mass);
            
            // Solar radiation pressure (main force for orbit raising)
            let sun_pos = nalgebra::SVector::<f64, 3>::from_column_slice(&SUN_POSITION);
            let srp_accel = SolarPressureModel::compute(
                &position,
                &velocity,
                self.sail_area,
                self.mass,
                self.reflectivity,
                angles.cone,
                angles.clock,
                &sun_pos,
            );

            // Drag (negligible at high altitudes, but included for completeness)
            let _drag_accel = DragModel::compute(
                &position,
                &velocity,
                self.sail_area,
                self.mass,
                self.drag_coefficient,
                angles.cone,
                angles.clock,
                &sun_pos,
            );

            // Total acceleration
            let total_accel = gravity_accel + srp_accel ;
            // println!("SRP Accel: {:?}", srp_accel);
            
            total_accel
        };

        // Prepare the integrator
        let ds: f64 = 1e-6;
        let integrator = numerical_integration::RK_FELBERG;

        let initial_state = Vector::from(self.initial_state.as_cartesian());

        let interpolation_strategy = context.optimization_context.interpolation_strategy;
        let control_mode = self.control_mode;
        let compute_angles = |time: f64, state: &Vector<f64, 6>| -> Angles {
            match control_mode {
                ControlMode::ControlPoints => {
                    let progress: f64 = (time * (input.len() as f64 - 2.0)) / self.target_time;
                    let prev_index = progress.floor() as usize;
                    let prev_index = prev_index.min(input.len() - 2);
                    let prev_angle = input[prev_index];
                    let next_angle = input[prev_index + 1];
                    interpolation_strategy.interpolate(&prev_angle, &next_angle, progress - prev_index as f64)
                }
                ControlMode::LocallyOptimalRealtime => {
                    let state_vec = StateVector::from(state.clone());
                    let position = state_vec.position();
                    let velocity = state_vec.velocity();
                    let sun_dir = sun_direction_from_position(position);
                    let orbit = OrbitalElements::from_state_vector(state_vec);
                    let (cone, clock) = LocallyOptimalSMA::compute_angles(&orbit, &sun_dir, &velocity, &position);
                    Angles { cone, clock }
                }
            }
        };
        let integration_function = |time, state: Vector<f64, 6>| {
            let mut this_step_data = Default::default();

            let angles = compute_angles(time, &state);

            let accelerations = eval_function(time, state.clone(), angles, &mut this_step_data);

            let new_state =
                concat_arrays(StateVector::from(state).velocity(), [accelerations[0], accelerations[1], accelerations[2]]);

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
        let mut angle_history = vec![];
        let mut sma_samples: Vec<f64> = vec![];  // Track SMA for mean calculation
        let sma_sample_interval = 100;  // Sample SMA every N steps
        loop {
            let last_state = states.first().unwrap();
            let last_time = last_state.clone().0;
            
            // Check target altitude and sample SMA periodically
            if timestep_index % sma_sample_interval == 0 {
                let current_state_vec = StateVector {
                    vector: [
                        last_state.2.0[0],
                        last_state.2.0[1],
                        last_state.2.0[2],
                        last_state.2.0[3],
                        last_state.2.0[4],
                        last_state.2.0[5],
                    ],
                };
                let current_orbit = OrbitalElements::from_state_vector(current_state_vec);
                
                // Sample SMA for mean calculation
                sma_samples.push(current_orbit.a);
                
                // Stop when target altitude is reached
                if current_orbit.a >= self.target_sma {
                    break;
                }
            }
            
            // Stop when target time is reached
            if last_time >= self.target_time {
                break;
            }

            // Only log timesteps according to strategy
            if !self.logging_strategy.only_final_state() 
                && self.logging_strategy.should_log_timestep(timestep_index) {
                data_to_log.push((chrono::Utc::now(), states.to_owned()));
                
                // Compute control angles for this logged timestep
                let angles = compute_angles(last_time, &states[0].2);
                angle_history.push(angles);
            }
            
            timestep_index += 1;
            integrator.adaptive_step(states, ds, integration_function, InnerProductMetric);
        }

        // Always save final state if using FinalStateOnly strategy
        if self.logging_strategy.only_final_state() {
            data_to_log.push((chrono::Utc::now(), states.to_owned()));
            // Compute final angles
            let last_time = states.first().unwrap().0;
            let angles = compute_angles(last_time, &states[0].2);
            angle_history.push(angles);
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
                "CREATE TABLE IF NOT EXISTS timesteps (id PRIMARY KEY, eval_id TEXT, datetime DATETIME, simtime REAL, x REAL, y REAL, z REAL, vx REAL, vy REAL, vz REAL, cone REAL, clock REAL, step_data TEXT)")
            .unwrap();
        query.execute([]).unwrap();
        drop(query);
        
        let mut query = transaction
                .prepare(
                    "INSERT INTO timesteps (eval_id, datetime, simtime, x, y, z, vx, vy, vz, cone, clock, step_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                .unwrap();
        data_to_log.iter().enumerate().for_each(|(idx, entry)| {
            let (time, states) = entry;
            let angles = angle_history[idx];
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
                    angles.cone,
                    angles.clock,
                    format!("{:?}", states[0].1.info),
                ))
                .unwrap();
        });

        drop(query);
        transaction.commit().unwrap();
        drop(connection_lock);

        let last_integrator_state = states.first().unwrap();
        let last_time = last_integrator_state.0;
        let last_state = last_integrator_state.2.clone();
        let last_state_vector = StateVector {
            vector: [
                last_state.0[0],
                last_state.0[1],
                last_state.0[2],
                last_state.0[3],
                last_state.0[4],
                last_state.0[5],
            ],
        };
        let last_orbit = OrbitalElements::from_state_vector(last_state_vector);
        
        // Add final SMA sample
        sma_samples.push(last_orbit.a);
        
        // Compute mean SMA over trajectory
        let mean_sma = if sma_samples.is_empty() {
            last_orbit.a
        } else {
            sma_samples.iter().sum::<f64>() / sma_samples.len() as f64
        };
        
        println!(
            "{} - Final SMA: {:.1} m, Mean SMA: {:.1} m (Δmean: {:.1} m)",
            context.eval_id, 
            last_orbit.a,
            mean_sma,
            mean_sma - self.initial_state.a
        );

        let last_position = [last_state.0[0], last_state.0[1], last_state.0[2]];
        let last_velocity = [last_state.0[3], last_state.0[4], last_state.0[5]];

        IntegrationState {
            time: last_time,
            semi_major_axis: last_orbit.a,
            mean_semi_major_axis: mean_sma,
            _position: last_position,
            _velocity: last_velocity,
        }
    }

    /// Local coordinate-wise hill climbing refinement.
    /// Perturbs one control point at a time with decreasing step sizes,
    /// only accepting strict improvements.
    pub fn refine(&self, initial: [Angles; N], context: &OptimizationContext) -> [Angles; N] {
        use optimization::EvaluationContext;

        let eval_ctx = EvaluationContext::new_in_run(context);
        let mut best = initial;
        let mut best_energy = self.energy(&best, &eval_ctx);
        println!("Local refinement starting with energy: {:?}", best_energy);

        let step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];

        for (round, &step) in step_sizes.iter().enumerate() {
            let mut improved = true;
            let mut pass = 0;
            while improved && pass < self.local_refinement_iterations {
                improved = false;
                pass += 1;
                for i in 0..N {
                    for field in 0..2 {
                        // Try +step and -step
                        for &direction in &[step, -step] {
                            let mut candidate = best;
                            match field {
                                0 => candidate[i].cone += direction,
                                1 => candidate[i].clock += direction,
                                _ => unreachable!(),
                            }
                            let eval_ctx = EvaluationContext::new_in_run(context);
                            let energy = self.energy(&candidate, &eval_ctx);
                            let acceptance = self.acceptance(&best_energy, &energy, 0.0, &eval_ctx);
                            if acceptance >= 1.0 {
                                println!("  Refinement round {}, pass {}: point {} {} {}{:.4} → energy {:?}",
                                    round, pass, i,
                                    if field == 0 { "cone" } else { "clock" },
                                    if direction > 0.0 { "+" } else { "" }, direction,
                                    energy);
                                best = candidate;
                                best_energy = energy;
                                improved = true;
                            }
                        }
                    }
                }
            }
            println!("Refinement round {} (step={}) done after {} passes, energy: {:?}", round, step, pass, best_energy);
        }

        println!("Local refinement finished with energy: {:?}", best_energy);
        best
    }
}

// ==============================================================================
// SIMULATED ANNEALING IMPLEMENTATION
// ==============================================================================

impl<const N: usize> OptimizableProblem for OrbitRaisingProblem<N> {
    type Input = [Angles; N];
    type Energy = f64;

    fn neighbour(
        &self,
        current: &Self::Input,
        temperature: f64,
        context: &EvaluationContext,
    ) -> Self::Input {
        // Generate neighbor using Gaussian perturbation with slew rate constraint
        // perturbation_scale controls the base standard deviation (at T=1)
        let (_, perturbation_scale) = context.optimization_context.sa_config();
        let standard_deviation = perturbation_scale * temperature;
        
        // Time between consecutive control points
        let segment_duration = self.target_time / (N - 1) as f64;
        // Maximum angular change allowed between consecutive points
        let max_angular_change = self.max_slew_rate * segment_duration;

        let mut rng = rand::thread_rng();
        let mut generated_angles = [Angles { cone: 0.0, clock: 0.0 }; N];
        
        // First control point: perturb freely
        use rand_distr::Distribution;
        let gaussian = rand_distr::Normal::new(current[0].clock, standard_deviation).unwrap();
        generated_angles[0].clock = normalize_clock(gaussian.sample(&mut rng));
        let gaussian = rand_distr::Normal::new(current[0].cone, standard_deviation).unwrap();
        generated_angles[0].cone = normalize_cone(gaussian.sample(&mut rng));
        
        // Subsequent points: ensure slew rate constraint is satisfied
        for i in 1..N {
            // Generate perturbed angles
            let gaussian = rand_distr::Normal::new(current[i].clock, standard_deviation).unwrap();
            let mut new_clock = normalize_clock(gaussian.sample(&mut rng));
            let gaussian = rand_distr::Normal::new(current[i].cone, standard_deviation).unwrap();
            let mut new_cone = normalize_cone(gaussian.sample(&mut rng));
            
            // Calculate angular difference from previous point
            let prev_angles = &generated_angles[i - 1];
            
            // Check clock change (accounting for wrap-around)
            let mut clock_diff = new_clock - prev_angles.clock;
            let clock_range = 2.0 * std::f64::consts::PI;
            if clock_diff > clock_range / 2.0 {
                clock_diff -= clock_range;
            } else if clock_diff < -clock_range / 2.0 {
                clock_diff += clock_range;
            }
            
            // Check cone change (accounting for wrap-around)
            let mut cone_diff = new_cone - prev_angles.cone;
            let cone_range = std::f64::consts::PI;
            if cone_diff > cone_range / 2.0 {
                cone_diff -= cone_range;
            } else if cone_diff < -cone_range / 2.0 {
                cone_diff += cone_range;
            }
            
            // Clamp changes to respect slew rate
            if clock_diff.abs() > max_angular_change {
                clock_diff = clock_diff.signum() * max_angular_change;
                new_clock = normalize_clock(prev_angles.clock + clock_diff);
            }
            if cone_diff.abs() > max_angular_change {
                cone_diff = cone_diff.signum() * max_angular_change;
                new_cone = normalize_cone(prev_angles.cone + cone_diff);
            }
            
            generated_angles[i] = Angles {
                cone: new_cone,
                clock: new_clock,
            };
        }
        
        generated_angles
    }

    fn acceptance(
        &self,
        current: &Self::Energy,
        new: &Self::Energy,
        temperature: f64,
        _context: &EvaluationContext,
    ) -> f64 {
        // Energy = -Δmean_sma, so lower (more negative) is better
        if new <= current {
            // Always accept better or equal solutions
            1.0
        } else {
            // Standard Metropolis-Hastings acceptance criterion
            // P = exp(-ΔE / (scale * T))
            // 
            // Energy is -Δmean_sma in meters. Typical differences are ~10-200 m.
            // Scale should be similar to typical delta_e for meaningful acceptance.
            // At T=1, scale=100: delta_e=100 → P=37%, delta_e=200 → P=14%
            let delta_e = new - current;
            let scale = 100.0;  // Tuned for energy in range ~-1000 to 0
            let exponent = -delta_e / (scale * temperature.max(0.001));
            exponent.exp().min(1.0)  // Clamp to [0, 1]
        }
    }

    fn energy(&self, current: &Self::Input, context: &EvaluationContext) -> Self::Energy {
        // Energy = negative mean SMA change (minimize = maximize SMA increase)
        // Using mean SMA is more stable than final SMA which oscillates with orbital position
        let state = self.evaluate(*current, context);
        let delta_mean_sma = state.mean_semi_major_axis - self.initial_state.a;
        
        // Negative because we want to MAXIMIZE SMA increase
        // Lower energy = better solution = higher SMA gain
        -delta_mean_sma
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use optimization::{AlgorithmConfig, CoolingSchedule, InterpolationStrategy};
    use uuid::Uuid;

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

    /// Regression test: solar sail orbit raising with fixed control angles.
    /// This test catches any changes to physics models, integrator, or acceleration
    /// computations that would alter the optimization output.
    #[test]
    fn solar_sail_energy_regression() {
        let config = crate::config::MissionConfig::optimization_defaults();
        let problem = OrbitRaisingProblem::<5> {
            sail_area: config.sail_area,
            mass: config.mass,
            reflectivity: config.reflectivity,
            drag_coefficient: config.drag_coefficient,
            target_sma: config.target_sma(),
            target_time: config.target_time,
            max_slew_rate: config.max_slew_rate,
            initial_state: config.initial_orbit(),
            logging_strategy: LoggingStrategy::FinalStateOnly,
            control_mode: ControlMode::ControlPoints,
            local_refinement: false,
            local_refinement_iterations: 0,
        };

        let fixed_angles: [Angles; 5] = [Angles { cone: 0.5, clock: 0.0 }; 5];
        let ctx = test_context();
        let eval_ctx = EvaluationContext::new_in_run(&ctx);
        let energy = problem.energy(&fixed_angles, &eval_ctx);

        // Golden value captured from current baseline.
        // If this test fails, it means something in the physics pipeline changed.
        // Update ONLY after verifying the change is intentional.
        let expected = -1516.6575660407543;
        let rel_err = (energy - expected).abs() / expected.abs();
        assert!(rel_err < 1e-6,
            "Solar sail energy regression failed!\n  Expected: {}\n  Got:      {}\n  Rel err:  {:.2e}\n\
             This means physics models, integrator, or acceleration code changed.",
            expected, energy, rel_err);
    }

    /// Regression test with different fixed angles to catch angle-dependent issues
    #[test]
    fn solar_sail_energy_regression_varied_angles() {
        let config = crate::config::MissionConfig::optimization_defaults();
        let problem = OrbitRaisingProblem::<5> {
            sail_area: config.sail_area,
            mass: config.mass,
            reflectivity: config.reflectivity,
            drag_coefficient: config.drag_coefficient,
            target_sma: config.target_sma(),
            target_time: config.target_time,
            max_slew_rate: config.max_slew_rate,
            initial_state: config.initial_orbit(),
            logging_strategy: LoggingStrategy::FinalStateOnly,
            control_mode: ControlMode::ControlPoints,
            local_refinement: false,
            local_refinement_iterations: 0,
        };

        let fixed_angles: [Angles; 5] = [
            Angles { cone: 0.3, clock: 0.1 },
            Angles { cone: 0.5, clock: 0.2 },
            Angles { cone: 0.7, clock: 0.5 },
            Angles { cone: 0.4, clock: 0.8 },
            Angles { cone: 0.2, clock: 0.3 },
        ];
        let ctx = test_context();
        let eval_ctx = EvaluationContext::new_in_run(&ctx);
        let energy = problem.energy(&fixed_angles, &eval_ctx);

        let expected = -1804.2377823023126;
        let rel_err = (energy - expected).abs() / expected.abs();
        assert!(rel_err < 1e-6,
            "Solar sail varied-angles regression failed!\n  Expected: {}\n  Got:      {}\n  Rel err:  {:.2e}",
            expected, energy, rel_err);
    }
}
