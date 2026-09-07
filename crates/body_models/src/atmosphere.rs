//! Atmosphere model for aerodynamic drag computation.

/// Atmospheric drag model for a target body.
///
/// Bodies without significant atmospheres (Moon, Bennu, most asteroids) use
/// `None`. Earth and Mars use `Exponential` for preliminary design; a full
/// empirical model (NRLMSISE) is reserved for Phase 5.
#[derive(Clone, Debug)]
pub enum AtmosphereModel {
    /// No atmosphere — drag acceleration is identically zero.
    None,
    /// Simple exponential: ρ(h) = ρ₀ · exp(−h / H).
    /// Adequate for LEO trajectory design and deorbit sizing.
    Exponential {
        /// Scale height [m]
        scale_height_m: f64,
        /// Sea-level density [kg/m³]
        rho0_kg_m3: f64,
    },
    // NRLMSISE — Phase 5
}
