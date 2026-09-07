use std::{
    sync::{
        atomic::{AtomicU64, Ordering::SeqCst},
        Arc, Mutex,
        mpsc::{self, Sender, Receiver},
    },
    thread,
    time::Duration,
};
use uuid::Uuid;

use rand::Rng;

use crate::common::{EvaluationContext, OptimizationContext};

/// Cooling schedule for simulated annealing
///
/// Controls how temperature decreases over the optimization process.
/// Higher temperatures allow more exploration (accepting worse solutions),
/// while lower temperatures focus on exploitation (refining good solutions).
#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
pub enum CoolingSchedule {
    /// Linear: T = 1 - progress
    /// Simple and fast cooling, good for quick runs
    #[default]
    Linear,

    /// Exponential: T = T_initial * (T_final/T_initial)^progress
    /// Slower initial cooling, more exploration early on
    /// Parameters: (T_initial, T_final)
    Exponential { t_initial: f64, t_final: f64 },

    /// Logarithmic: T = 1 / (1 + c * ln(1 + progress))
    /// Very slow cooling, theoretically optimal but slow
    /// Parameter: c controls cooling rate (higher = faster cooling)
    Logarithmic { c: f64 },

    /// Quadratic: T = (1 - progress)^2
    /// Fast initial cooling, slower at end
    Quadratic,

    /// Inverse: T = T_initial / (1 + k * progress)
    /// Smooth decay, parameter k controls rate
    Inverse { t_initial: f64, k: f64 },

    /// Adaptive: starts slow, speeds up based on acceptance rate
    /// Not implemented yet - placeholder for future
    Adaptive,
}

impl CoolingSchedule {
    /// Calculate temperature for given progress (0.0 to 1.0)
    pub fn temperature(&self, progress: f64) -> f64 {
        match self {
            CoolingSchedule::Linear => 1.0 - progress,

            CoolingSchedule::Exponential { t_initial, t_final } => {
                t_initial * (t_final / t_initial).powf(progress)
            }

            CoolingSchedule::Logarithmic { c } => {
                1.0 / (1.0 + c * (1.0 + progress).ln())
            }

            CoolingSchedule::Quadratic => (1.0 - progress).powi(2),

            CoolingSchedule::Inverse { t_initial, k } => {
                t_initial / (1.0 + k * progress)
            }

            CoolingSchedule::Adaptive => {
                // Placeholder - just use linear for now
                1.0 - progress
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_schedule_boundaries() {
        let schedule = CoolingSchedule::Linear;
        assert!((schedule.temperature(0.0) - 1.0).abs() < 1e-10);
        assert!((schedule.temperature(1.0) - 0.0).abs() < 1e-10);
        assert!((schedule.temperature(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn exponential_schedule_boundaries() {
        let schedule = CoolingSchedule::Exponential { t_initial: 1.0, t_final: 0.01 };
        assert!((schedule.temperature(0.0) - 1.0).abs() < 1e-10);
        assert!((schedule.temperature(1.0) - 0.01).abs() < 1e-10);
    }

    #[test]
    fn logarithmic_schedule_boundaries() {
        let schedule = CoolingSchedule::Logarithmic { c: 1.0 };
        assert!((schedule.temperature(0.0) - 1.0).abs() < 1e-10);
        // At progress=1: 1/(1 + 1*ln(2)) ≈ 0.5906
        assert!(schedule.temperature(1.0) > 0.0);
        assert!(schedule.temperature(1.0) < 1.0);
    }

    #[test]
    fn quadratic_schedule_boundaries() {
        let schedule = CoolingSchedule::Quadratic;
        assert!((schedule.temperature(0.0) - 1.0).abs() < 1e-10);
        assert!((schedule.temperature(1.0) - 0.0).abs() < 1e-10);
        assert!((schedule.temperature(0.5) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn inverse_schedule_boundaries() {
        let schedule = CoolingSchedule::Inverse { t_initial: 1.0, k: 10.0 };
        assert!((schedule.temperature(0.0) - 1.0).abs() < 1e-10);
        // At progress=1: 1/(1+10) ≈ 0.0909
        let t1 = schedule.temperature(1.0);
        assert!((t1 - 1.0 / 11.0).abs() < 1e-10);
    }

    #[test]
    fn all_schedules_monotonically_decrease() {
        let schedules: Vec<CoolingSchedule> = vec![
            CoolingSchedule::Linear,
            CoolingSchedule::Exponential { t_initial: 1.0, t_final: 0.01 },
            CoolingSchedule::Logarithmic { c: 1.0 },
            CoolingSchedule::Quadratic,
            CoolingSchedule::Inverse { t_initial: 1.0, k: 10.0 },
        ];

        for schedule in &schedules {
            let mut prev_temp = schedule.temperature(0.0);
            for i in 1..=100 {
                let progress = i as f64 / 100.0;
                let temp = schedule.temperature(progress);
                assert!(
                    temp <= prev_temp + 1e-12,
                    "Schedule {:?} not monotonically decreasing at progress={}: {} > {}",
                    schedule, progress, temp, prev_temp
                );
                prev_temp = temp;
            }
        }
    }
}

pub trait SAProblem: Sync {
    type Input: Clone + Send + core::fmt::Debug;
    type Energy: Clone + Send + core::fmt::Debug;

    fn neighbour(
        &self,
        current: &Self::Input,
        temperature: f64,
        context: &EvaluationContext,
    ) -> Self::Input;

    fn acceptance(
        &self,
        current: &Self::Energy,
        new: &Self::Energy,
        temperature: f64,
        context: &EvaluationContext,
    ) -> f64;

    fn energy(&self, current: &Self::Input, context: &EvaluationContext) -> Self::Energy;

    fn optimize(
        &self,
        initial: Self::Input,
        generation_count: u64,
        context: OptimizationContext,
    ) -> Self::Input {
        let (cooling_schedule, _) = context.sa_config();
        let cooling_schedule = *cooling_schedule;

        let eval_context = EvaluationContext::new_in_run(&context);
        let in_energy = self.energy(&initial, &eval_context);
        if let Some(connection) = context.connection.as_ref() {
            let connection_lock = connection.lock().unwrap();

            let mut query = connection_lock
                    .prepare(
                        "CREATE TABLE IF NOT EXISTS optimization_runs (id INTEGER PRIMARY KEY, datetime DATETIME, run_id TEXT, generation_count INTEGER, initial_energy REAL, initial_eval_id TEXT);",
                    )
                    .unwrap();
            query.execute([]).unwrap();
            drop(query);

            let mut query = connection_lock
                    .prepare(
                        "CREATE TABLE IF NOT EXISTS generations (id INTEGER PRIMARY KEY, eval_id TEXT, run_id TEXT, generation INTEGER, temperature REAL, energy REAL, accepted TEXT, input TEXT);",
                    )
                    .unwrap();
            query.execute([]).unwrap();
            drop(query);

            let mut query = connection_lock
                .prepare("INSERT INTO optimization_runs (datetime, run_id, generation_count, initial_energy, initial_eval_id) VALUES (datetime(), ?, ?, ?, ?)").unwrap();
            query
                .insert([
                    context.run_id.to_string(),
                    generation_count.to_string(),
                    format!("{:?}", in_energy),
                    eval_context.eval_id.to_string(),
                ])
                .unwrap();
            drop(query);
            drop(connection_lock);
        }

        let current = Arc::new(Mutex::new((initial.clone(), in_energy.clone())));
        let best = Arc::new(Mutex::new((initial, in_energy)));

        let gen = Arc::new(AtomicU64::new(0));

        // Restart less frequently to avoid disrupting exploration
        let restart_interval: u64 = (generation_count / 5).max(1);

        // Channel-based work queue to prevent deadlocks
        enum WorkMessage {
            Evaluate(u64), // Generation number to evaluate
            Shutdown,
        }

        enum ResultMessage<I, E> {
            Evaluated {
                generation: u64,
                new_input: I,
                new_energy: E,
                accepted: bool,
                eval_id: Uuid,
                temperature: f64,
            },
            Error(String),
        }

        let nb_cpus = num_cpus::get();
        let (work_tx, work_rx): (Sender<WorkMessage>, Receiver<WorkMessage>) = mpsc::channel();
        let (result_tx, result_rx): (Sender<ResultMessage<Self::Input, Self::Energy>>, Receiver<ResultMessage<Self::Input, Self::Energy>>) = mpsc::channel();

        println!("Starting optimization with {} worker threads", nb_cpus);

        // Spawn worker threads
        thread::scope(|s| {
            let work_rx = Arc::new(Mutex::new(work_rx));

            // Worker threads
            for worker_id in 0..nb_cpus {
                let work_rx = Arc::clone(&work_rx);
                let result_tx = result_tx.clone();
                let current = Arc::clone(&current);
                let best = Arc::clone(&best);
                let context = &context;
                let restart_interval = restart_interval;

                s.spawn(move || {
                    loop {
                        // Receive work with timeout to prevent hanging
                        let work_item = match work_rx.lock() {
                            Ok(rx) => rx.recv_timeout(Duration::from_secs(5)),
                            Err(e) => {
                                eprintln!("Worker {} - work_rx lock error: {:?}", worker_id, e);
                                break;
                            }
                        };

                        match work_item {
                            Ok(WorkMessage::Evaluate(gen_num)) => {
                                let eval_context = EvaluationContext::new_in_run(context);
                                let progress = (gen_num as f64) / (generation_count as f64);
                                let temp = cooling_schedule.temperature(progress);

                                // Periodically restart from the best solution
                                if gen_num > 0 && gen_num % restart_interval == 0 {
                                    match (best.lock(), current.lock()) {
                                        (Ok(local_best), Ok(mut local_current)) => {
                                            println!("{} - Worker {} restarting from best solution (gen {})",
                                                eval_context.eval_id, worker_id, gen_num);
                                            *local_current = local_best.clone();
                                        }
                                        _ => eprintln!("Worker {} - Failed to acquire locks for restart", worker_id),
                                    }
                                }

                                // Get current state
                                let (current_input, current_energy) = match current.lock() {
                                    Ok(guard) => guard.clone(),
                                    Err(e) => {
                                        let guard = e.into_inner();
                                        eprintln!("Worker {} - Recovered from poisoned current lock", worker_id);
                                        guard.clone()
                                    }
                                };

                                // Generate neighbor (no locks needed)
                                let new = self.neighbour(&current_input, temp, &eval_context);

                                // Evaluate energy (no locks needed, but may take time)
                                let new_energy = self.energy(&new, &eval_context);
                                println!("Worker {} - Gen {} - Energy: {:?}", worker_id, gen_num, new_energy);

                                // Calculate acceptance
                                let acceptance = self.acceptance(&current_energy, &new_energy, temp, &eval_context);
                                let accepted = acceptance > rand::thread_rng().gen_range(0.0..1.0);

                                // Send result back to main thread (main thread handles all mutations)
                                if let Err(e) = result_tx.send(ResultMessage::Evaluated {
                                    generation: gen_num,
                                    new_input: new,
                                    new_energy: new_energy.clone(),
                                    accepted,
                                    eval_id: eval_context.eval_id,
                                    temperature: temp,
                                }) {
                                    eprintln!("Worker {} - Failed to send result: {:?}", worker_id, e);
                                    break;
                                }
                            }
                            Ok(WorkMessage::Shutdown) => {
                                println!("Worker {} shutting down", worker_id);
                                break;
                            }
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                // Timeout is normal, just continue
                                continue;
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => {
                                println!("Worker {} - Channel disconnected, shutting down", worker_id);
                                break;
                            }
                        }
                    }
                });
            }

            // Drop the original result_tx so only workers hold references
            drop(result_tx);

            // Main coordinator thread
            let mut completed_generations = 0u64;
            let mut pending_work = 0usize;

            // Submit initial batch of work
            for _ in 0..nb_cpus {
                if completed_generations < generation_count {
                    if work_tx.send(WorkMessage::Evaluate(completed_generations)).is_ok() {
                        completed_generations += 1;
                        pending_work += 1;
                    }
                }
            }

            // Process results and submit new work
            while completed_generations < generation_count || pending_work > 0 {
                match result_rx.recv_timeout(Duration::from_secs(30)) {
                    Ok(ResultMessage::Evaluated {
                        generation,
                        new_input,
                        new_energy,
                        accepted,
                        eval_id,
                        temperature,
                    }) => {
                        pending_work = pending_work.saturating_sub(1);

                        // Update generation counter
                        gen.store(generation + 1, SeqCst);

                        // Log to database (non-blocking for workers)
                        if let Some(connection) = context.connection.as_ref() {
                            if let Ok(connection_lock) = connection.lock() {
                                if let Ok(mut query) = connection_lock.prepare(
                                    "INSERT INTO generations (eval_id, run_id, generation, temperature, energy, accepted, input) VALUES (?, ?, ?, ?, ?, ?, ?)"
                                ) {
                                    let _ = query.insert([
                                        eval_id.to_string(),
                                        context.run_id.to_string(),
                                        generation.to_string(),
                                        format!("{:?}", temperature),
                                        format!("{:?}", new_energy),
                                        accepted.to_string(),
                                        format!("{:?}", new_input),
                                    ]);
                                }
                            }
                        }

                        // Update current state if accepted
                        if accepted {
                            if let Ok(mut local_current) = current.lock() {
                                println!("{} - Accepting new energy, {:?} instead of {:?}. Probability was {}%",
                                    eval_id, new_energy, local_current.1,
                                    self.acceptance(&local_current.1, &new_energy, temperature,
                                        &EvaluationContext { eval_id, optimization_context: &context }) * 100.0);
                                *local_current = (new_input.clone(), new_energy.clone());
                            }
                        }

                        // Update best-so-far
                        if let Ok(mut local_best) = best.lock() {
                            let best_acceptance = self.acceptance(&local_best.1, &new_energy, 0.0,
                                &EvaluationContext { eval_id, optimization_context: &context });
                            if best_acceptance >= 1.0 {
                                println!("{} - New best energy: {:?}", eval_id, new_energy);
                                *local_best = (new_input, new_energy);
                            }
                        }

                        // Submit new work if available
                        if completed_generations < generation_count {
                            if work_tx.send(WorkMessage::Evaluate(completed_generations)).is_ok() {
                                println!("Generation: {}/{}", completed_generations, generation_count);
                                completed_generations += 1;
                                pending_work += 1;
                            }
                        }
                    }
                    Ok(ResultMessage::Error(msg)) => {
                        eprintln!("Worker error: {}", msg);
                        pending_work = pending_work.saturating_sub(1);
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        eprintln!("WARNING: No results received for 30 seconds. Pending work: {}", pending_work);
                        if pending_work == 0 {
                            break;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        println!("All workers finished");
                        break;
                    }
                }
            }

            // Shutdown workers
            for _ in 0..nb_cpus {
                let _ = work_tx.send(WorkMessage::Shutdown);
            }

            println!("Optimization complete: {}/{} generations", completed_generations, generation_count);
        });

        let local_best = best.lock().unwrap_or_else(|e| e.into_inner());
        local_best.0.clone()
    }
}
