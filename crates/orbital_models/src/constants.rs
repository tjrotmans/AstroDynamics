//! Physical constants for orbital mechanics

/// Standard gravitational parameter for Earth (m³/s²) — EGM2008 / IERS TN 36
pub const MU_EARTH: f64 = 3.986004418e14;

/// Earth mean radius in meters — IAU 2015
pub const EARTH_RADIUS: f64 = 6.371e6;

/// Newtonian gravitational constant (m³/kg·s²) — CODATA 2018
pub const G: f64 = 6.67430e-11;

/// Earth mass (kg) — derived: MU_EARTH / G
pub const M_EARTH: f64 = 5.972e24;

/// Solar radiation pressure at 1 AU (N/m²) — IERS Conventions 2010, §9.1
pub const P_SRP: f64 = 4.56e-6;

/// Astronomical Unit in meters — IAU 2012 exact definition
pub const AU: f64 = 1.496e11;

/// Fixed Sun position in inertial coordinates (m).
/// Approximation: Earth at 1 AU from Sun in +X direction.
pub const SUN_POSITION: [f64; 3] = [150.0e9, 0.0, 0.0];

/// Standard gravitational parameter for the Moon (m³/s²) — JPL DE430
pub const MU_MOON: f64 = 4.9048695e12;

/// Standard gravitational parameter for the Sun (m³/s²) — JPL DE430
pub const MU_SUN: f64 = 1.327_124_400_18e20;

/// Moon mean radius in meters — IAU 2015
pub const MOON_RADIUS: f64 = 1.7374e6;

/// Standard gravity (m/s²) — ISO 80000-3 exact definition
pub const G0: f64 = 9.80665;

/// Earth equatorial radius [m] — EGM2008
/// Slightly larger than the mean radius (6,371,000 m) due to Earth's oblateness.
pub const EARTH_EQUATORIAL_RADIUS: f64 = 6_378_136.6;

/// J2 zonal harmonic coefficient — EGM2008.
/// Earth's dominant non-spherical gravity term caused by equatorial bulge.
/// At 378 km altitude the J2 acceleration is ~1.3 × 10⁻² m/s² (~1/670 of central gravity).
pub const J2: f64 = 1.082_626_68e-3;

/// J3 zonal harmonic coefficient — EGM2008. ~1/430 of J2.
pub const J3: f64 = -2.532_661_2e-6;

/// J4 zonal harmonic coefficient — EGM2008. ~1/670 of J2.
pub const J4: f64 = -1.619_898_5e-6;

/// Standard gravitational parameter for Venus (m³/s²) — JPL DE430
pub const MU_VENUS: f64 = 3.248_598_96e14;

/// Standard gravitational parameter for Mars barycenter (m³/s²) — JPL DE430
pub const MU_MARS: f64 = 4.282_837_62e13;

/// Standard gravitational parameter for Jupiter barycenter (m³/s²) — JPL DE430
pub const MU_JUPITER: f64 = 1.267_127_678e17;

/// Mean Earth-Moon distance (semi-major axis) [m] — IAU / JPL DE430.
/// Used as the CRTBP characteristic length L★.
pub const EARTH_MOON_DISTANCE: f64 = 384_400_000.0;

/// Moon sidereal period [days] — JPL DE430 mean value
pub const MOON_SIDEREAL_PERIOD_DAYS: f64 = 27.321_661;

/// Julian year [days] — IAU 1976 exact definition.
/// Used to compute the Sun's angular rate relative to the Earth-Moon rotating frame.
pub const JULIAN_YEAR_DAYS: f64 = 365.25;

/// Standard gravitational parameter for Saturn barycenter (m³/s²) — JPL DE430 (Folkner et al. 2014)
pub const MU_SATURN: f64 = 3.793_120_749_865_224e16;

/// Jupiter mean equatorial radius [m] — IAU 2015 Working Group on Cartographic Coordinates
pub const JUPITER_RADIUS: f64 = 71_492_000.0;

/// Saturn mean equatorial radius [m] — IAU 2015 Working Group on Cartographic Coordinates
pub const SATURN_RADIUS: f64 = 60_268_000.0;

/// Standard gravitational parameter for Europa (m³/s²) — Anderson et al. 1998 (Icarus 135, 390)
/// Derived from Galileo spacecraft gravity science during J4 and E12 flybys.
pub const MU_EUROPA: f64 = 3.202_738_774_922_892e12;

/// Europa mean radius [m] — IAU 2015 Working Group on Cartographic Coordinates
pub const EUROPA_RADIUS: f64 = 1_560_800.0;

/// Standard gravitational parameter for Titan (m³/s²) — Iess et al. 2012 (Science 337, 457)
/// Derived from Cassini gravity science flybys.
pub const MU_TITAN: f64 = 8.978_138_376_618_386e12;

/// Titan mean radius [m] — IAU 2015 Working Group on Cartographic Coordinates
pub const TITAN_RADIUS: f64 = 2_575_500.0;

/// Standard gravitational parameter for Phobos (m³/s²) — Jacobson et al. 2014 (AJ 148, 76)
/// Derived from Mars Express HRSC/MCS gravity flybys.
pub const MU_PHOBOS: f64 = 7.087_546e5;

/// Phobos mean radius [m] — Willner et al. 2010 (Planetary and Space Science 58, 1870)
pub const PHOBOS_RADIUS: f64 = 11_100.0;

/// Standard gravitational parameter for Deimos (m³/s²) — Jacobson 2010 (AJ 139, 668)
pub const MU_DEIMOS: f64 = 9.615_9e4;

/// Deimos mean radius [m] — Thomas 1993 (Icarus 105, 326)
pub const DEIMOS_RADIUS: f64 = 6_200.0;

/// Standard gravitational parameter for 433 Eros (m³/s²) — Yeomans et al. 2000 (Science 289, 2085)
/// Derived from NEAR Shoemaker radio science during orbital phase.
pub const MU_EROS: f64 = 4.463e5;

/// 433 Eros mean radius [m] — Thomas et al. 2002 (Icarus 155, 18)
pub const EROS_RADIUS: f64 = 8_416.0;

/// Standard gravitational parameter for 65803 Didymos system (m³/s²) — Scheirich et al. 2022 (PSJ 3, 160)
/// Post-DART value; includes both Didymos primary and Dimorphos moonlet.
pub const MU_DIDYMOS: f64 = 41.0;

/// 65803 Didymos primary mean radius [m] — Naidu et al. 2020 (Icarus 348, 113777)
pub const DIDYMOS_RADIUS: f64 = 390.0;
