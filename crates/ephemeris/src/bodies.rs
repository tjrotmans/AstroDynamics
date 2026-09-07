//! Solar system body identifiers

/// Solar system bodies supported by the Almanac.
///
/// Only `Sun` and `Moon` are currently wired into `Almanac` queries.
/// The remaining variants are reserved for future point-mass perturbation support.
///
/// `Phobos`/`Deimos`/`Europa`/`Titan` require their own satellite SPK kernel
/// chain-loaded on top of `de440s.bsp` (`mar099s.bsp`, `jup365.bsp`,
/// `sat441.bsp` respectively — see `MissionPlanner::design::load_almanac`);
/// `body_state_heliocentric` returns an `Err` (not a panic) for them if that
/// kernel wasn't loaded.
///
/// `Uranus`/`Neptune` need no extra kernel — their barycenters are already in
/// the base `de440s.bsp` (confirmed: `anise::constants::frames`
/// already defines `URANUS_BARYCENTER_J2000`/`NEPTUNE_BARYCENTER_J2000`, same
/// as Jupiter/Saturn). A prior version of this enum omitted them entirely,
/// which silently made every MGA chromosome targeting Neptune infeasible
/// (`anise_body("neptune")` returned `None`, so every ephemeris lookup for
/// the target body failed) — not a real kernel-coverage gap, just a missing
/// variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Body {
    Sun,
    Moon,
    Earth,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
    Venus,
    Mercury,
    Phobos,
    Deimos,
    Europa,
    Titan,
}
