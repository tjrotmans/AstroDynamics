//! Simulation configuration — spacecraft, Bennu, sensor noise, EKF tuning.
//!
//! All values are in SI units unless a field name says otherwise.
//! DO NOT modify without explicit permission (protected per project rules).

use nalgebra::Vector3;
use orbital_models::constants::AU;

// ── Physical constants ────────────────────────────────────────────────────────
// MU_SUN, AU, and P_SRP are imported from orbital_models::constants where needed.

/// Gravitational parameter of Bennu [m³/s²]  (Chesley et al. 2020)
pub const MU_BENNU: f64 = 6.674e-11 * 7.329e10; // GM ≈ 4.89 m³/s²

/// Mean radius of Bennu [m]
pub const R_BENNU: f64 = 262.0;

// ── Spacecraft parameters ─────────────────────────────────────────────────────

/// Spacecraft mass [kg]
pub const SC_MASS: f64 = 1000.0;

/// Spacecraft cross-sectional area for cannonball SRP [m²] (filter model)
pub const SC_AREA_CANNONBALL: f64 = 4.0;

/// Reflectivity coefficient C_R (initial guess for EKF state)
pub const C_R_NOMINAL: f64 = 1.4;

/// Inertia tensor diagonal [kg·m²] — simple rectangular box 2×2×1 m, 1000 kg
pub const SC_INERTIA_DIAG: [f64; 3] = [366.67, 366.67, 666.67];

// ── RCS thruster configuration ────────────────────────────────────────────────

/// Thrust per RCS thruster [N]
pub const RCS_THRUST_N: f64 = 1.0;

/// Thruster moment arm from centre of mass [m]
pub const RCS_MOMENT_ARM: f64 = 1.0;

/// Minimum impulse-bit duration [s] — each thruster fires this long per truth step.
pub const RCS_PULSE_DURATION_S: f64 = 0.050;

/// Pointing dead-zone: don't fire for errors smaller than this [rad].
pub const RCS_POINTING_DB_RAD: f64 = 0.5 * std::f64::consts::PI / 180.0;

/// Rate dead-zone: don't fire when angular rate is below this [rad/s].
pub const RCS_RATE_DB_RADS: f64 = 1e-3;

/// Attitude PD proportional gain [N·m/rad] — bang-bang RCS controller.
pub const ATTITUDE_KP: f64 = 1.5;

/// Attitude PD derivative gain [N·m·s/rad] — bang-bang RCS controller.
pub const ATTITUDE_KD: f64 = 50.0;

// ── Spacecraft geometry ───────────────────────────────────────────────────────

/// Main bus full dimensions [m]: (x, y, z) in body frame.
/// Body x = camera/radial, y = solar panel axis, z = along-track.
/// Derived from SC_INERTIA_DIAG assuming solid rectangular box 1000 kg:
///   Ixx = m/12*(ly²+lz²) = 366.67  →  lx=ly=2.0 m, lz≈0.63 m
pub const SC_BUS_DIMS_M: [f64; 3] = [2.0, 2.0, 0.63];

/// Solar panel span (each panel) [m] — extends along ±y body axis from bus edge.
pub const SC_PANEL_SPAN_M: f64 = 2.5;

/// Solar panel chord [m] — extent along x body axis.
pub const SC_PANEL_CHORD_M: f64 = 0.8;

/// Total solar panel area (both panels combined) [m²]
pub const SC_PANEL_AREA_M2: f64 = 2.0 * SC_PANEL_SPAN_M * SC_PANEL_CHORD_M;

/// Centre-of-mass offset from geometric centre in body frame [m].
/// Dense battery pack is slightly off-centre in −z.
pub const COM_OFFSET_BODY: [f64; 3] = [0.02, 0.0, -0.04];

// ── Reaction wheel configuration ──────────────────────────────────────────────

/// Wheel spin-axis moment of inertia [kg·m²].
/// Typical for a ≈20 N·m·s capacity wheel (e.g., Honeywell HR14).
pub const WHEEL_INERTIA_KGM2: f64 = 0.012;

/// Maximum reaction wheel angular speed [rad/s] (≈ 6000 RPM).
pub const WHEEL_MAX_SPEED_RADS: f64 = 628.3;

/// Maximum torque output per wheel [N·m].
pub const WHEEL_MAX_TORQUE_NM: f64 = 0.12;

/// Desaturation trigger threshold — fraction of `WHEEL_MAX_SPEED_RADS`.
pub const WHEEL_DESAT_FRACTION: f64 = 0.80;

// ── Bennu Keplerian elements (J2000 ecliptic, epoch 2182-Sep-24) ─────────────
/// Semi-major axis [m]
pub const BENNU_A: f64 = 1.1264 * AU;
/// Eccentricity
pub const BENNU_E: f64 = 0.2037;
/// Inclination [rad]
pub const BENNU_I: f64 = 6.034_f64 * std::f64::consts::PI / 180.0;
/// RAAN [rad]
pub const BENNU_RAAN: f64 = 2.060_f64 * std::f64::consts::PI / 180.0;
/// Argument of periapsis [rad]
pub const BENNU_AOP: f64 = 66.22_f64 * std::f64::consts::PI / 180.0;
/// Mean anomaly at epoch [rad] — equals 0 at periapsis (epoch is at periapsis passage).
pub const BENNU_M0: f64 = 0.0;
/// Epoch offset from J2000 [s]  (we start the scenario at this epoch)
pub const BENNU_EPOCH_S: f64 = 0.0;

// ── Bennu gravity model (Scheeres et al. 2020) ────────────────────────────────

/// Gravity-field normalising radius R₀ [m] (≠ mean body radius `R_BENNU`).
pub const BENNU_GRAVITY_R0_M: f64 = 290.0;
/// Zonal harmonic J₂ (oblateness).
pub const BENNU_J2: f64 =  1.926101e-2;
/// Zonal harmonic J₃ (north–south pear shape).
pub const BENNU_J3: f64 = -1.221940e-3;
/// Zonal harmonic J₄.
pub const BENNU_J4: f64 = -6.496002e-3;

// ── Initial spacecraft state (Hill frame, Bennu-centred) ─────────────────────

/// Initial relative position of spacecraft w.r.t. Bennu [m]  (5 km along -x)
pub fn sc_r0() -> Vector3<f64> { Vector3::new(-5_000.0, 0.0, 200.0) }

/// Initial relative velocity of spacecraft w.r.t. Bennu [m/s]
/// (+x closes the gap from 5 km; +y gives lateral offset for curved approach)
pub fn sc_v0() -> Vector3<f64> { Vector3::new(0.05, 0.02, 0.005) }

/// Initial spacecraft attitude quaternion [w, x, y, z] — identity (body = inertial)
pub fn sc_q0() -> [f64; 4] { [1.0, 0.0, 0.0, 0.0] }

/// Initial spacecraft angular rate [rad/s]
pub fn sc_omega0() -> Vector3<f64> { Vector3::zeros() }

// ── Simulation parameters ─────────────────────────────────────────────────────

/// Total simulation duration [s]
pub const SIM_DURATION_S: f64 = 86_400.0; // 1 day

/// Integration time step for truth propagation [s]
pub const DT_TRUTH_S: f64 = 1.0;

/// EKF update interval [s]  (how often OpNav + star tracker measurements arrive)
pub const DT_MEAS_S: f64 = 60.0;

// ── Sensor noise parameters ───────────────────────────────────────────────────

/// Star tracker attitude noise, 1-sigma per axis [rad]  (≈ 20 arcsec)
pub const STAR_TRACKER_SIGMA_RAD: f64 = 9.696e-5;

/// OpNav bearing noise, 1-sigma [rad]  (≈ 0.1 pixel at f/1000 px FOV)
pub const OPNAV_BEARING_SIGMA_RAD: f64 = 1e-4;

/// OpNav angular-size noise, 1-sigma [rad]
pub const OPNAV_SIZE_SIGMA_RAD: f64 = 2e-4;

/// LIDAR altimeter range noise, 1-sigma [m]
/// Typical for small planetary altimeters (e.g., mini-LIDAR at ~5 m accuracy).
pub const LIDAR_SIGMA_M: f64 = 5.0;

/// LIDAR operational range limit [m] — returns None beyond this distance.
pub const LIDAR_MAX_RANGE_M: f64 = 5_000.0;

/// IMU ΔV noise, 1-sigma per axis [m/s]
pub const IMU_DV_SIGMA_MPS: f64 = 0.01;

/// Camera half-angle field of view [rad]  (30° full FOV)
pub const CAMERA_FOV_HALF_RAD: f64 = 15.0 * std::f64::consts::PI / 180.0;

// ── EKF tuning ────────────────────────────────────────────────────────────────

/// Initial position uncertainty, 1-sigma [m]
pub const EKF_SIGMA_R0_M: f64 = 500.0;

/// Initial velocity uncertainty, 1-sigma [m/s]
pub const EKF_SIGMA_V0_MPS: f64 = 0.5;

/// Initial C_SRP uncertainty, 1-sigma (dimensionless)
pub const EKF_SIGMA_CR0: f64 = 0.3;

/// Process noise spectral density for position [m²/s³]
/// Tuned for K_lidar ≈ 0.2: with LIDAR every 120 s (R=25 m²), ΔP = 1.2 m²/interval
/// gives steady-state P_rr ≈ 6 m² → σ_r ≈ 2.5 m per axis.
/// Physically: unmodelled gravity harmonics from Bennu's irregular shape.
pub const SNC_QR: f64 = 0.01;

/// Process noise spectral density for velocity [m²/s]
pub const SNC_QV: f64 = 1e-8;

// ── Proximity operations ──────────────────────────────────────────────────────

/// Bennu ephemeris position 1-σ at arrival [m].
/// Represents residual absolute position error after all cruise ground observations.
pub const BENNU_EPHEM_SIGMA_M: f64 = 5_000.0;

/// Target circular orbit radius for proximity operations [m]
pub const TARGET_ORBIT_R_M: f64 = 1_500.0;

/// Station-keeping orbital energy tolerance [J/kg]; correct when |ε − ε_target| exceeds this.
pub const SK_ENERGY_TOL: f64 = 1e-6;

/// Station-keeping check interval [s]
pub const SK_INTERVAL_S: f64 = 3_600.0;

/// DSN tracking pass interval [s]
pub const DSN_UPDATE_INTERVAL_S: f64 = 8.0 * 3_600.0;

/// DSN heliocentric position noise 1-σ (intrinsic, before Bennu ephemeris folded in) [m]
pub const DSN_RANGE_SIGMA_M: f64 = 10.0;

/// DSN heliocentric velocity noise 1-σ (intrinsic) [m/s]
pub const DSN_VELOCITY_SIGMA_MPS: f64 = 1e-4;

/// Bennu velocity uncertainty folded into DSN position update [m/s]
pub const DSN_BENNU_V_SIGMA_MPS: f64 = 0.05;

/// Proximity-ops simulation duration [s] — 7 days
pub const PROX_OPS_DURATION_S: f64 = 7.0 * 86_400.0;

/// Truth integration step for proximity operations [s]
pub const PROX_DT_TRUTH_S: f64 = 10.0;

/// OpNav measurement interval for proximity operations [s]
pub const PROX_DT_MEAS_S: f64 = 120.0;
