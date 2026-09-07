pub mod common;
pub mod logging;
pub mod sa;
pub mod ga;

// Re-export all public types at crate root for convenience
pub use logging::LoggingStrategy;
pub use common::{
    AlgorithmConfig,
    InterpolationStrategy,
    OptimizationContext,
    EvaluationContext,
};
pub use sa::{CoolingSchedule, SAProblem};
pub use ga::GAProblem;

// Backward-compatible alias (downstream code can use either name)
pub use SAProblem as OptimizableProblem;
