//! Ephemeris and reference frame types
//!
//! Provides:
//! - Compile-time frame safety via phantom-typed vectors (`FrameVec<F>`, `FrameState<F>`)
//! - Frame marker types (`ECI`, `Heliocentric`, `LVLH`, `ECEF`)
//! - An `Almanac` wrapper around the ANISE ephemeris engine for querying body positions
//! - Re-export of `hifitime::Epoch` for time handling

pub mod almanac;
pub mod bodies;
pub mod body_track;
pub mod frames;

pub use almanac::Almanac;
pub use bodies::Body;
pub use frames::{FrameState, FrameVec, ECEF, ECI, LVLH, Heliocentric};

/// Re-export of `hifitime::Epoch` — the standard time type used throughout this crate.
///
/// Supports UTC, TAI, TDB, TT, GPS and more. Use `Epoch::from_gregorian_utc(...)` or
/// `Epoch::from_unix_seconds(...)` to construct.
pub use body_track::{BodyTrack, MoonTrack, SunTrack};
pub use hifitime::Epoch;

/// Find a kernel file by searching `kernels/<filename>` from the current
/// directory upward until a match is found.
///
/// This allows a single `kernels/` directory at the workspace root to serve
/// all packages, regardless of which directory `cargo run` or `cargo test`
/// is invoked from.
///
/// # Panics
/// Panics if the file is not found in any parent directory.
pub fn find_kernel(filename: &str) -> String {
    let mut dir = std::env::current_dir().expect("Cannot determine current directory");
    loop {
        let candidate = dir.join("kernels").join(filename);
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
        if !dir.pop() {
            panic!(
                "Kernel '{}' not found in 'kernels/' in the current directory or any parent.\n\
                 Download: https://public-data.nyxspace.com/anise/{}\n\
                 Place at: <workspace-root>/kernels/{}",
                filename, filename, filename
            );
        }
    }
}
