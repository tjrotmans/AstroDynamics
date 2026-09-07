//! Real-kernel verification for the new Phobos/Deimos/Europa/Titan ANISE
//! coverage (frontend-reported gap: `/api/bodies/{name}/state` 422'd for
//! these four because `anise_body()` returned `None` and their satellite
//! kernels weren't loaded). Confirms the chain-loaded satellite kernels
//! (`mar099s.bsp`, `jup365.bsp`, `sat441.bsp`) actually resolve a state for
//! each moon, and that the heliocentric distance from the moon to its
//! primary planet is close to the catalog's `sma_m` (semi-major axis)
//! constant — a coarse but real physical sanity check, not just "it didn't
//! error."
//!
//! Run from the repo root: `cargo run -p mission_planner --bin moon_ephem_demo --release`

use ephemeris::{Almanac, Body, Epoch};

// Catalog semi-major axes (body_models::TargetBody::{phobos,deimos,europa,titan}()),
// reproduced here rather than depending on body_models, since this is a
// standalone ANISE-coverage check, not a mission run.
const PHOBOS_SMA_M: f64 = 9.376e6; // from Mars centre
const DEIMOS_SMA_M: f64 = 2.346_2e7; // from Mars centre (Jacobson 2010)
const EUROPA_SMA_M: f64 = 6.711e8; // from Jupiter centre
const TITAN_SMA_M: f64 = 1.221_870e9; // from Saturn centre

fn find_kernel(filename: &str) -> Option<String> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join("kernels").join(filename);
        if candidate.exists() {
            return Some(candidate.to_string_lossy().into_owned());
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn load_kernel(almanac: Almanac, filename: &str) -> Almanac {
    match find_kernel(filename) {
        Some(path) => match almanac.load(&path) {
            Ok(a) => {
                println!("Loaded {filename} ({path})");
                a
            }
            Err(e) => {
                eprintln!("Failed to load {filename}: {e}");
                std::process::exit(1);
            }
        },
        None => {
            eprintln!(
                "{filename} not found in any kernels/ directory. Download: \
                 https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/{filename}"
            );
            std::process::exit(1);
        }
    }
}

fn check(almanac: &Almanac, name: &str, body: Body, primary: Body, expected_sma_m: f64, epoch: Epoch) {
    let moon = almanac
        .body_state_heliocentric(body, epoch)
        .unwrap_or_else(|e| panic!("{name}: ANISE query failed: {e}"));
    let primary_state = almanac
        .body_state_heliocentric(primary, epoch)
        .unwrap_or_else(|e| panic!("{name}'s primary: ANISE query failed: {e}"));

    // body_state_heliocentric already returns meters (see almanac.rs km_to_m_helio).
    let dr = moon.position.inner - primary_state.position.inner;
    let dist_m = dr.norm();
    let rel_err = (dist_m - expected_sma_m).abs() / expected_sma_m;

    println!(
        "{name:<8} distance from primary: {:>12.1} km   catalog sma: {:>12.1} km   rel err: {:.2}%   {}",
        dist_m / 1000.0,
        expected_sma_m / 1000.0,
        rel_err * 100.0,
        if rel_err < 0.3 { "OK (within eccentricity spread)" } else { "CHECK — outside expected range" }
    );
}

fn main() {
    let de440s = find_kernel("de440s.bsp").expect("kernels/de440s.bsp not found");
    let almanac = Almanac::new(&de440s).expect("failed to load de440s.bsp");
    let almanac = load_kernel(almanac, "mar099s.bsp");
    let almanac = load_kernel(almanac, "jup365.bsp");
    let almanac = load_kernel(almanac, "sat441.bsp");

    let epoch = Epoch::from_gregorian_utc(2026, 7, 2, 0, 0, 0, 0);

    println!("\n=== Moon ephemeris coverage check @ {epoch} ===\n");
    check(&almanac, "Phobos", Body::Phobos, Body::Mars, PHOBOS_SMA_M, epoch);
    check(&almanac, "Deimos", Body::Deimos, Body::Mars, DEIMOS_SMA_M, epoch);
    check(&almanac, "Europa", Body::Europa, Body::Jupiter, EUROPA_SMA_M, epoch);
    check(&almanac, "Titan", Body::Titan, Body::Saturn, TITAN_SMA_M, epoch);
}
