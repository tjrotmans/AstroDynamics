//! Database logging configuration
//!
//! Controls how much trajectory data is saved to reduce database size.

/// Logging strategy for trajectory timesteps.
#[derive(Clone, Copy, Debug)]
pub enum LoggingStrategy {
    /// Save all timesteps, downsampled by the given factor (e.g. every 100th step).
    AllDownsampled(usize),
    /// Only save trajectories for the best solution every N generations.
    BestEveryNthGeneration(u64),
    /// Only save the final state — minimal storage.
    FinalStateOnly,
    /// Don't save any trajectory data (only optimization results).
    None,
}

impl LoggingStrategy {
    /// Whether this evaluation's trajectory should be logged at all.
    pub fn should_log_trajectory(&self, generation: u64, is_best_so_far: bool) -> bool {
        match self {
            LoggingStrategy::AllDownsampled(_) => true,
            LoggingStrategy::BestEveryNthGeneration(n) => generation % n == 0 && is_best_so_far,
            LoggingStrategy::FinalStateOnly => true,
            LoggingStrategy::None => false,
        }
    }

    /// Whether a specific timestep index should be written (for downsampling).
    pub fn should_log_timestep(&self, timestep_index: usize) -> bool {
        match self {
            LoggingStrategy::AllDownsampled(factor) => timestep_index % factor == 0,
            LoggingStrategy::BestEveryNthGeneration(_) => true,
            LoggingStrategy::FinalStateOnly => false,
            LoggingStrategy::None => false,
        }
    }

    /// Whether only the final state (not intermediate steps) should be saved.
    pub fn only_final_state(&self) -> bool {
        matches!(self, LoggingStrategy::FinalStateOnly)
    }
}
