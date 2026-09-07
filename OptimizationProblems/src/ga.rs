use std::thread;
use uuid::Uuid;

use rand::Rng;

use crate::common::{EvaluationContext, OptimizationContext};

/// Trait for problems solvable by a Genetic Algorithm.
///
/// The GA maintains a population of individuals, evolving them through
/// selection, crossover, and mutation over multiple generations.
/// Fitness is minimized (lower = better), matching SA's energy convention.
pub trait GAProblem: Sync {
    type Individual: Clone + Send + Sync + core::fmt::Debug;
    type Fitness: Clone + Send + Sync + core::fmt::Debug + PartialOrd;

    /// Evaluate the fitness of an individual. Lower is better (minimization).
    fn fitness(&self, individual: &Self::Individual, context: &EvaluationContext) -> Self::Fitness;

    /// Produce a random individual for the initial population.
    fn random_individual(&self) -> Self::Individual;

    /// Crossover two parents to produce one offspring.
    fn crossover(
        &self,
        parent1: &Self::Individual,
        parent2: &Self::Individual,
        context: &EvaluationContext,
    ) -> Self::Individual;

    /// Mutate an individual (returns new individual).
    fn mutate(
        &self,
        individual: &Self::Individual,
        mutation_rate: f64,
        context: &EvaluationContext,
    ) -> Self::Individual;

    /// Run the genetic algorithm. Default implementation provided.
    fn optimize_ga(
        &self,
        generation_count: u64,
        context: OptimizationContext,
    ) -> Self::Individual {
        let (population_size, crossover_rate, mutation_rate, elitism_count, tournament_size) =
            context.ga_config();

        // DB logging: create tables
        if let Some(connection) = context.connection.as_ref() {
            let conn = connection.lock().unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS optimization_runs (
                    id INTEGER PRIMARY KEY, datetime DATETIME, run_id TEXT,
                    generation_count INTEGER, initial_energy REAL, initial_eval_id TEXT
                );
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY, eval_id TEXT, run_id TEXT,
                    generation INTEGER, temperature REAL, energy REAL,
                    accepted TEXT, input TEXT
                );"
            ).unwrap();
            conn.execute(
                "INSERT INTO optimization_runs (datetime, run_id, generation_count, initial_energy, initial_eval_id) VALUES (datetime(), ?, ?, 0.0, '')",
                [&context.run_id.to_string(), &generation_count.to_string()],
            ).unwrap();
        }

        // Initialize population
        let mut population: Vec<Self::Individual> = (0..population_size)
            .map(|_| self.random_individual())
            .collect();

        // Evaluate initial population
        let mut fitnesses = self.evaluate_population(&population, &context);

        println!("Starting GA optimization with {} individuals, {} generations",
            population_size, generation_count);

        // Main GA loop
        for gen in 0..generation_count {
            // Adaptive mutation: cosine decay from mutation_rate to 10% of it
            // Early generations explore broadly, late generations refine
            let progress = gen as f64 / generation_count as f64;
            let decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()); // 1.0 → 0.0
            let min_rate = mutation_rate * 0.1;
            let adaptive_mutation_rate = min_rate + (mutation_rate - min_rate) * decay;

            // Sort by fitness (ascending = best first for minimization)
            let mut indexed: Vec<(usize, &Self::Fitness)> =
                fitnesses.iter().enumerate().collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

            let sorted_indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();

            // Elitism: keep top individuals
            let mut new_population: Vec<Self::Individual> = sorted_indices[..elitism_count.min(population_size)]
                .iter()
                .map(|&i| population[i].clone())
                .collect();

            // Fill rest with crossover + mutation
            while new_population.len() < population_size {
                let mut rng = rand::thread_rng();
                let eval_ctx = EvaluationContext::new_in_run(&context);

                let parent1 = self.tournament_select(&population, &fitnesses, tournament_size);
                let parent2 = self.tournament_select(&population, &fitnesses, tournament_size);

                let child = if rng.gen::<f64>() < crossover_rate {
                    self.crossover(&parent1, &parent2, &eval_ctx)
                } else {
                    parent1
                };

                let child = self.mutate(&child, adaptive_mutation_rate, &eval_ctx);
                new_population.push(child);
            }

            population = new_population;
            fitnesses = self.evaluate_population(&population, &context);

            // Find best in this generation
            let best_idx = fitnesses.iter().enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap();

            println!("Generation {}/{} - Best fitness: {:?} (mutation_rate: {:.4})",
                gen + 1, generation_count, fitnesses[best_idx], adaptive_mutation_rate);

            // Log best of generation to DB
            if let Some(connection) = context.connection.as_ref() {
                if let Ok(conn) = connection.lock() {
                    let eval_id = Uuid::now_v7();
                    let _ = conn.execute(
                        "INSERT INTO generations (eval_id, run_id, generation, temperature, energy, accepted, input) VALUES (?, ?, ?, 0.0, ?, 'true', ?)",
                        [
                            &eval_id.to_string(),
                            &context.run_id.to_string(),
                            &gen.to_string(),
                            &format!("{:?}", fitnesses[best_idx]),
                            &format!("{:?}", population[best_idx]),
                        ],
                    );
                }
            }
        }

        // Return best individual
        let best_idx = fitnesses.iter().enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();

        println!("GA optimization complete. Best fitness: {:?}", fitnesses[best_idx]);
        population[best_idx].clone()
    }

    /// Evaluate all individuals in parallel, bounded to `num_cpus` threads at a time.
    ///
    /// Spawning the entire population at once causes O(population_size) threads to
    /// simultaneously contend for the SQLite write mutex inside `fitness()`, stalling
    /// nearly all threads. Chunking limits concurrency to the number of logical CPUs.
    fn evaluate_population(
        &self,
        population: &[Self::Individual],
        context: &OptimizationContext,
    ) -> Vec<Self::Fitness> {
        let n_threads = num_cpus::get().max(1);
        let mut results = Vec::with_capacity(population.len());
        for chunk in population.chunks(n_threads) {
            thread::scope(|s| {
                let handles: Vec<_> = chunk.iter().map(|individual| {
                    s.spawn(|| {
                        let eval_ctx = EvaluationContext::new_in_run(context);
                        self.fitness(individual, &eval_ctx)
                    })
                }).collect();
                results.extend(handles.into_iter().map(|h| h.join().unwrap()));
            });
        }
        results
    }

    /// Tournament selection: pick `k` random individuals, return the best.
    fn tournament_select(
        &self,
        population: &[Self::Individual],
        fitnesses: &[Self::Fitness],
        k: usize,
    ) -> Self::Individual {
        let mut rng = rand::thread_rng();
        let mut best_idx = rng.gen_range(0..population.len());
        for _ in 1..k {
            let idx = rng.gen_range(0..population.len());
            if fitnesses[idx] < fitnesses[best_idx] {
                best_idx = idx;
            }
        }
        population[best_idx].clone()
    }
}
