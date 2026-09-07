//! Gravity model selection for a target body.

/// Gravity model used for force computation in trajectory design and simulation.
///
/// Variants are ordered by fidelity; higher variants subsume the effects of lower
/// ones. The parameters are dimensionless harmonic coefficients in the body-fixed
/// frame normalised to the body equatorial radius.
#[derive(Clone, Debug)]
pub enum GravityModel {
    /// Acceleration = −μ/r² r̂. Fast; valid for patched-conic preliminary design.
    PointMass,
    /// J2 oblateness only. Adequate for LEO and first-order small-body effects.
    J2 {
        /// Second zonal harmonic (oblateness)
        j2: f64,
    },
    /// J2 through J4 zonal harmonics. Current standard for Earth, Moon, Bennu, Mars.
    J2J3J4 {
        /// Second zonal harmonic (oblateness)
        j2: f64,
        /// Third zonal harmonic (pear shape)
        j3: f64,
        /// Fourth zonal harmonic
        j4: f64,
    },
    // SphericalHarmonic { degree: u32, order: u32 } — Phase 5
}
