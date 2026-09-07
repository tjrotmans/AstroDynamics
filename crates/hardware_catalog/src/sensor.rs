//! Sensor catalog entries — star tracker grades (attitude), OpNav camera
//! grades (bearing, for position knowledge), and the IMU/LIDAR/landmark/DSN
//! sensors already validated by the Bennu mission's GNC simulator.
//!
//! These sensor types size different things: star tracker noise bounds
//! *attitude* knowledge (and therefore pointing budget), while OpNav bearing
//! noise bounds *position* knowledge via the small-angle relationship
//! σ_position ≈ range × σ_bearing — this is the sensor the GNC design stage
//! actually selects against a `position_accuracy_req_m` requirement, matching
//! how the existing Bennu EKF derives position knowledge from OpNav bearing
//! measurements (see `GNC/AutonomousNavigation/src/sensors/opnav.rs`), not
//! from star tracker attitude noise.
//!
//! IMU/LIDAR/landmark/DSN noise parameters below are taken directly from the
//! already-validated Bennu mission GNC model
//! (`GNC/AutonomousNavigation/src/sensors/{imu,opnav,landmark,dsn}.rs` and
//! `src/config.rs`), not invented — each constructor cites the source
//! constant it mirrors. Power and mass figures for these four are
//! representative class estimates (Wertz & Larson, *SMAD*-class figures),
//! since the GNC model only specifies noise/range, not a power/mass budget.

/// A star tracker grade spec — attitude knowledge only.
#[derive(Clone, Copy, Debug)]
pub struct StarTrackerSpec {
    pub name: &'static str,
    /// 1-σ attitude noise per axis [rad]
    pub noise_rad: f64,
    /// Unit mass [kg]
    pub mass_kg: f64,
    /// Nominal operating power draw [W] — representative class estimate.
    pub power_w: f64,
}

impl StarTrackerSpec {
    /// Coarse grade — ~100 arcsec, sun-sensor-adjacent low-cost class.
    pub fn coarse() -> Self {
        Self { name: "StarTracker-Coarse", noise_rad: 4.85e-4, mass_kg: 0.3, power_w: 3.0 }
    }

    /// Medium grade — ~20 arcsec, matches the existing Bennu mission's
    /// Sodern_CT633-class tracker (`STAR_TRACKER_SIGMA_RAD` in
    /// `GNC/AutonomousNavigation/src/config.rs`). Power matches the
    /// CT-633 class (~5-6 W nominal).
    pub fn medium() -> Self {
        Self { name: "StarTracker-Medium", noise_rad: 9.7e-5, mass_kg: 0.7, power_w: 5.0 }
    }

    /// Fine grade — ~2 arcsec, precision science-mission class.
    pub fn fine() -> Self {
        Self { name: "StarTracker-Fine", noise_rad: 9.7e-6, mass_kg: 1.2, power_w: 8.0 }
    }

    /// Finest to coarsest — sizing picks the coarsest grade that still meets
    /// the navigation accuracy requirement.
    pub fn catalog() -> [Self; 3] {
        [Self::fine(), Self::medium(), Self::coarse()]
    }
}

/// An optical navigation camera grade spec — bearing + angular-size
/// measurement of the target body, the sensor that drives position knowledge
/// in proximity operations.
#[derive(Clone, Copy, Debug)]
pub struct OpNavCameraSpec {
    pub name: &'static str,
    /// 1-σ bearing (az/el) noise [rad]
    pub bearing_noise_rad: f64,
    /// 1-σ angular size noise [rad]
    pub angular_size_noise_rad: f64,
    /// Unit mass [kg]
    pub mass_kg: f64,
    /// Nominal operating power draw [W] — representative class estimate.
    pub power_w: f64,
}

impl OpNavCameraSpec {
    /// Coarse grade — wide-FOV low-cost navigation camera.
    pub fn coarse() -> Self {
        Self { name: "OpNav-Coarse", bearing_noise_rad: 5.0e-4, angular_size_noise_rad: 1.0e-3, mass_kg: 1.0, power_w: 3.0 }
    }

    /// Medium grade — matches the existing Bennu mission's OpNav camera
    /// (0.1 mrad bearing / 0.2 mrad angular size; `OPNAV_BEARING_SIGMA_RAD`/
    /// `OPNAV_SIZE_SIGMA_RAD` in `GNC/AutonomousNavigation/src/config.rs`).
    pub fn medium() -> Self {
        Self { name: "OpNav-Medium", bearing_noise_rad: 1.0e-4, angular_size_noise_rad: 2.0e-4, mass_kg: 2.0, power_w: 5.0 }
    }

    /// Fine grade — narrow-FOV precision navigation camera.
    pub fn fine() -> Self {
        Self { name: "OpNav-Fine", bearing_noise_rad: 2.0e-5, angular_size_noise_rad: 4.0e-5, mass_kg: 3.5, power_w: 8.0 }
    }

    /// Finest to coarsest — sizing picks the coarsest grade that still meets
    /// the position accuracy requirement.
    pub fn catalog() -> [Self; 3] {
        [Self::fine(), Self::medium(), Self::coarse()]
    }
}

/// An IMU/accelerometer grade spec — ΔV measurement noise during impulsive
/// manoeuvres (see `GNC/AutonomousNavigation/src/sensors/imu.rs`).
#[derive(Clone, Copy, Debug)]
pub struct ImuSpec {
    pub name: &'static str,
    /// 1-σ ΔV noise per axis [m/s]
    pub dv_noise_mps: f64,
    /// Unit mass [kg]
    pub mass_kg: f64,
    /// Nominal operating power draw [W] — representative class estimate.
    pub power_w: f64,
}

impl ImuSpec {
    /// Coarse grade — low-cost MEMS-class IMU.
    pub fn coarse() -> Self {
        Self { name: "IMU-Coarse", dv_noise_mps: 0.05, mass_kg: 0.3, power_w: 4.0 }
    }

    /// Medium grade — matches the existing Bennu mission's IMU
    /// (`IMU_DV_SIGMA_MPS` in `GNC/AutonomousNavigation/src/config.rs`).
    pub fn medium() -> Self {
        Self { name: "IMU-Medium", dv_noise_mps: 0.01, mass_kg: 0.6, power_w: 8.0 }
    }

    /// Fine grade — navigation-grade IMU, tactical/strategic class.
    pub fn fine() -> Self {
        Self { name: "IMU-Fine", dv_noise_mps: 0.002, mass_kg: 1.2, power_w: 15.0 }
    }

    /// Finest to coarsest — sizing picks the coarsest grade that still meets
    /// the ΔV-knowledge accuracy requirement.
    pub fn catalog() -> [Self; 3] {
        [Self::fine(), Self::medium(), Self::coarse()]
    }
}

/// A LIDAR altimeter grade spec — slant-range measurement to the target
/// body's surface (see `GNC/AutonomousNavigation/src/sensors/opnav.rs`'s
/// `lidar_measure`).
#[derive(Clone, Copy, Debug)]
pub struct LidarSpec {
    pub name: &'static str,
    /// 1-σ range noise [m]
    pub range_noise_m: f64,
    /// Maximum operational range [m] — no return beyond this.
    pub max_range_m: f64,
    /// Unit mass [kg]
    pub mass_kg: f64,
    /// Nominal operating power draw [W] — representative class estimate.
    pub power_w: f64,
}

impl LidarSpec {
    /// Coarse grade — short-range, low-cost rangefinder.
    pub fn coarse() -> Self {
        Self { name: "Lidar-Coarse", range_noise_m: 15.0, max_range_m: 2_000.0, mass_kg: 1.0, power_w: 6.0 }
    }

    /// Medium grade — matches the existing Bennu mission's mini-LIDAR
    /// altimeter (`LIDAR_SIGMA_M` / `LIDAR_MAX_RANGE_M` in
    /// `GNC/AutonomousNavigation/src/config.rs`).
    pub fn medium() -> Self {
        Self { name: "Lidar-Medium", range_noise_m: 5.0, max_range_m: 5_000.0, mass_kg: 2.0, power_w: 10.0 }
    }

    /// Fine grade — long-range, precision planetary altimeter class.
    pub fn fine() -> Self {
        Self { name: "Lidar-Fine", range_noise_m: 1.0, max_range_m: 20_000.0, mass_kg: 4.0, power_w: 18.0 }
    }

    /// Finest to coarsest — sizing picks the coarsest grade that still meets
    /// the range accuracy requirement.
    pub fn catalog() -> [Self; 3] {
        [Self::fine(), Self::medium(), Self::coarse()]
    }
}

/// A landmark-tracking (stereophotoclinometry-analogue) sensor grade spec —
/// bearing-only line-of-sight measurements to catalogued surface features,
/// giving full 3-D position (including range) from several simultaneous
/// landmark sightings without a LIDAR (see
/// `GNC/AutonomousNavigation/src/sensors/landmark.rs`). Reuses the same
/// camera/optics as `OpNavCameraSpec` conceptually, but the catalog keeps it
/// as a distinct hardware item because it is selected/sized against a
/// different requirement (multi-landmark 3-D position knowledge, not single
/// centroid bearing).
#[derive(Clone, Copy, Debug)]
pub struct LandmarkSensorSpec {
    pub name: &'static str,
    /// 1-σ per-landmark bearing noise [rad]
    pub bearing_noise_rad: f64,
    /// Number of catalogued landmarks the processing pipeline can track
    /// (`N_LANDMARKS` in `GNC/AutonomousNavigation/src/sensors/landmark.rs`).
    pub catalog_size: u32,
    /// Unit mass [kg]
    pub mass_kg: f64,
    /// Nominal operating power draw [W] — representative class estimate.
    pub power_w: f64,
}

impl LandmarkSensorSpec {
    /// Coarse grade — fewer catalogued features, wider bearing noise.
    pub fn coarse() -> Self {
        Self { name: "Landmark-Coarse", bearing_noise_rad: 3.0e-4, catalog_size: 50, mass_kg: 1.0, power_w: 4.0 }
    }

    /// Medium grade — matches the existing Bennu mission's landmark tracker
    /// (`OPNAV_BEARING_SIGMA_RAD` for per-landmark noise, `N_LANDMARKS` = 150
    /// in `GNC/AutonomousNavigation/src/sensors/landmark.rs`).
    pub fn medium() -> Self {
        Self { name: "Landmark-Medium", bearing_noise_rad: 1.0e-4, catalog_size: 150, mass_kg: 2.0, power_w: 6.0 }
    }

    /// Fine grade — dense feature catalog, precision stereophotoclinometry
    /// class.
    pub fn fine() -> Self {
        Self { name: "Landmark-Fine", bearing_noise_rad: 3.0e-5, catalog_size: 400, mass_kg: 3.5, power_w: 10.0 }
    }

    /// Finest to coarsest — sizing picks the coarsest grade that still meets
    /// the position accuracy requirement.
    pub fn catalog() -> [Self; 3] {
        [Self::fine(), Self::medium(), Self::coarse()]
    }
}

/// A rate-gyro grade spec — bias random walk (RRW) + angle random walk (ARW)
/// feeding the attitude MEKF (`docs/MP/MANUAL.md` §8.2/§9.1). New sensor
/// type, not present in the Bennu reference (`GNC/AutonomousNavigation`'s IMU
/// model only ever covered the accelerometer/ΔV side).
///
/// Gyro datasheets normally quote **bias stability** [deg/hr, Allan-variance
/// floor] and **ARW** [deg/sqrt(hr)]; the two-state gyro error model this
/// crate's `sim_engine::sensors::gyro` implements (Farrenkopf 1978; Markley &
/// Crassidis, *Fundamentals of Spacecraft Attitude Determination and
/// Control*, §4) instead needs the corresponding diffusion coefficients in SI
/// units [rad/s/sqrt(s)] and [rad/sqrt(s)]. `bias_walk_sigma_rad_s_sqrt_s` is
/// obtained by re-interpreting the published bias-stability figure directly
/// as a rate-random-walk (RRW) coefficient (deg/hr -> deg/hr/sqrt(hr)) — a
/// standard simulation-engineering approximation used when a dedicated RRW
/// number isn't published; each grade below states both the datasheet-style
/// figure and the converted SI value so it can be checked or replaced with a
/// real unit's RRW spec if one becomes available.
#[derive(Clone, Copy, Debug)]
pub struct GyroSpec {
    pub name: &'static str,
    /// Angle random walk (ARW) 1-sigma coefficient [rad/sqrt(s)].
    pub arw_sigma_rad_sqrt_s: f64,
    /// Bias random walk (RRW) 1-sigma coefficient [rad/s/sqrt(s)].
    pub bias_walk_sigma_rad_s_sqrt_s: f64,
    /// Unit mass [kg]
    pub mass_kg: f64,
    /// Nominal operating power draw [W] — representative class estimate.
    pub power_w: f64,
}

impl GyroSpec {
    /// Coarse grade — consumer/low-cost MEMS class.
    /// ARW ≈ 0.3 deg/sqrt(hr) -> 8.73e-5 rad/sqrt(s).
    /// Bias stability ≈ 10 deg/hr -> RRW 8.08e-7 rad/s/sqrt(s).
    pub fn coarse() -> Self {
        Self {
            name: "Gyro-Coarse",
            arw_sigma_rad_sqrt_s: 8.73e-5,
            bias_walk_sigma_rad_s_sqrt_s: 8.08e-7,
            mass_kg: 0.2,
            power_w: 2.0,
        }
    }

    /// Medium grade — tactical-grade MEMS/fiber-optic-gyro (FOG) class.
    /// ARW ≈ 0.05 deg/sqrt(hr) -> 1.45e-5 rad/sqrt(s).
    /// Bias stability ≈ 1 deg/hr -> RRW 8.08e-8 rad/s/sqrt(s).
    pub fn medium() -> Self {
        Self {
            name: "Gyro-Medium",
            arw_sigma_rad_sqrt_s: 1.45e-5,
            bias_walk_sigma_rad_s_sqrt_s: 8.08e-8,
            mass_kg: 0.5,
            power_w: 5.0,
        }
    }

    /// Fine grade — navigation-grade FOG, planetary-mission heritage class.
    /// ARW ≈ 0.002 deg/sqrt(hr) -> 5.82e-7 rad/sqrt(s).
    /// Bias stability ≈ 0.01 deg/hr -> RRW 8.08e-10 rad/s/sqrt(s).
    pub fn fine() -> Self {
        Self {
            name: "Gyro-Fine",
            arw_sigma_rad_sqrt_s: 5.82e-7,
            bias_walk_sigma_rad_s_sqrt_s: 8.08e-10,
            mass_kg: 1.0,
            power_w: 12.0,
        }
    }

    /// Finest to coarsest — sizing picks the coarsest grade that still meets
    /// the attitude-knowledge accuracy requirement.
    pub fn catalog() -> [Self; 3] {
        [Self::fine(), Self::medium(), Self::coarse()]
    }
}

/// A Deep Space Network (DSN) ground-link grade spec — heliocentric uplink
/// accuracy from ground orbit determination (see
/// `GNC/AutonomousNavigation/src/sensors/dsn.rs`'s `GroundOdEkf`). Unlike the
/// other sensors here, "mass"/"power" describe the onboard transponder/HGA
/// terminal, not a ground asset.
#[derive(Clone, Copy, Debug)]
pub struct DsnLinkSpec {
    pub name: &'static str,
    /// 1-σ two-way range noise [m] (`RNG_NOISE_M` in
    /// `GNC/AutonomousNavigation/src/sensors/dsn.rs`).
    pub range_noise_m: f64,
    /// 1-σ range-rate (Doppler) noise [m/s] (`RRATE_NOISE_MPS`).
    pub range_rate_noise_mps: f64,
    /// 1-σ Delta-DOR transverse angular noise [rad] (`DDOR_NOISE_RAD`).
    pub ddor_noise_rad: f64,
    /// Onboard transponder + HGA terminal mass [kg]
    pub mass_kg: f64,
    /// Nominal operating power draw [W] — representative class estimate.
    pub power_w: f64,
}

impl DsnLinkSpec {
    /// Coarse grade — LGA/omni link, coarser ranging, no Delta-DOR support.
    pub fn coarse() -> Self {
        Self {
            name: "DSN-Coarse",
            range_noise_m: 10.0,
            range_rate_noise_mps: 1.0e-3,
            ddor_noise_rad: 50e-9,
            mass_kg: 3.0,
            power_w: 20.0,
        }
    }

    /// Medium grade — matches the existing Bennu mission's ground OD link
    /// (`RNG_NOISE_M` = 2 m, `RRATE_NOISE_MPS` = 1e-4 m/s, `DDOR_NOISE_RAD`
    /// = 10 nrad in `GNC/AutonomousNavigation/src/sensors/dsn.rs`).
    pub fn medium() -> Self {
        Self {
            name: "DSN-Medium",
            range_noise_m: 2.0,
            range_rate_noise_mps: 1.0e-4,
            ddor_noise_rad: 10e-9,
            mass_kg: 6.0,
            power_w: 40.0,
        }
    }

    /// Fine grade — HGA + precision USO link, OSIRIS-REx-class Delta-DOR
    /// (~3 nrad achieved, per `dsn.rs`'s module doc comment).
    pub fn fine() -> Self {
        Self {
            name: "DSN-Fine",
            range_noise_m: 0.5,
            range_rate_noise_mps: 2.0e-5,
            ddor_noise_rad: 3e-9,
            mass_kg: 10.0,
            power_w: 65.0,
        }
    }

    /// Finest to coarsest — sizing picks the coarsest grade that still meets
    /// the navigation accuracy requirement.
    pub fn catalog() -> [Self; 3] {
        [Self::fine(), Self::medium(), Self::coarse()]
    }
}
