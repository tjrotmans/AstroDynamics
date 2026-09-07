//! ANISE-backed ephemeris almanac
//!
//! Wraps `anise::prelude::Almanac` to provide frame-typed body positions
//! in SI units (meters, m/s) rather than ANISE's native kilometers.
//!
//! # Kernel Files
//!
//! ANISE requires SPICE-compatible BSP kernel files. Recommended defaults:
//! - `de440s.bsp` — JPL planetary ephemeris 1900–2050 (~18 MB)
//! - `pck08.pca` — planetary constants kernel
//!
//! Both are available from the Nyx Space public CDN:
//! `https://public-data.nyxspace.com/anise/`
//!
//! # Example
//! ```rust,no_run
//! use ephemeris::{Almanac, Epoch};
//!
//! let almanac = Almanac::new(&ephemeris::find_kernel("de440s.bsp")).unwrap();
//! let epoch = Epoch::from_gregorian_utc(2025, 3, 2, 12, 0, 0, 0);
//! let sun_pos = almanac.sun_position(epoch).unwrap();
//! println!("Sun distance from Earth: {:.3e} m", sun_pos.norm());
//! ```

use anise::constants::frames::{
    EARTH_J2000, JUPITER_BARYCENTER_J2000, MARS_BARYCENTER_J2000, MERCURY_J2000,
    MOON_J2000, NEPTUNE_BARYCENTER_J2000, SATURN_BARYCENTER_J2000, SUN_J2000,
    URANUS_BARYCENTER_J2000, VENUS_J2000,
};
use anise::constants::orientations::J2000;
use anise::ephemerides::EphemerisError;
use anise::errors::AlmanacError;
use anise::prelude::{Almanac as AniseAlmanac, Frame};
use hifitime::Epoch;
use nalgebra::SVector;

/// NAIF body IDs for natural satellites — not defined as named constants in
/// `anise::constants::celestial_objects` (only planets/barycenters are), so
/// built directly per the standard NAIF numbering (JPL SPICE Required
/// Reading: NAIF IDs — <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html>).
const PHOBOS_ID: i32 = 401;
const DEIMOS_ID: i32 = 402;
const EUROPA_ID: i32 = 502;
const TITAN_ID: i32 = 606;

const PHOBOS_J2000: Frame = Frame::new(PHOBOS_ID, J2000);
const DEIMOS_J2000: Frame = Frame::new(DEIMOS_ID, J2000);
const EUROPA_J2000: Frame = Frame::new(EUROPA_ID, J2000);
const TITAN_J2000: Frame = Frame::new(TITAN_ID, J2000);

use crate::bodies::Body;
use crate::frames::{FrameState, FrameVec, Heliocentric, ECI};

/// Ephemeris almanac backed by ANISE.
///
/// Provides body positions and states in the ECI (J2000) frame, in SI units (m, m/s).
/// Construct with [`Almanac::new`] pointing at a BSP kernel file.
pub struct Almanac {
    inner: AniseAlmanac,
}

impl Almanac {
    /// Load an almanac from a BSP (or PCK) kernel file.
    ///
    /// # Arguments
    /// * `path` - Path to a SPICE-compatible `.bsp` kernel file
    pub fn new(path: &str) -> Result<Self, AlmanacError> {
        let inner = AniseAlmanac::default().load(path)?;
        Ok(Self { inner })
    }

    /// Chain-load an additional kernel file (BSP, PCK, etc.).
    ///
    /// Returns a new `Almanac` with the additional data loaded.
    pub fn load(self, path: &str) -> Result<Self, AlmanacError> {
        let inner = self.inner.load(path)?;
        Ok(Self { inner })
    }

    /// Position of the Sun relative to Earth center, in ECI (J2000) [m].
    pub fn sun_position(&self, epoch: Epoch) -> Result<FrameVec<ECI>, EphemerisError> {
        let state = self.inner.translate(SUN_J2000, EARTH_J2000, epoch, None)?;
        Ok(km_to_m(state.radius_km.x, state.radius_km.y, state.radius_km.z))
    }

    /// Position of the Moon relative to Earth center, in ECI (J2000) [m].
    pub fn moon_position(&self, epoch: Epoch) -> Result<FrameVec<ECI>, EphemerisError> {
        let state = self.inner.translate(MOON_J2000, EARTH_J2000, epoch, None)?;
        Ok(km_to_m(state.radius_km.x, state.radius_km.y, state.radius_km.z))
    }

    /// Position of Venus relative to Earth center, in ECI (J2000) [m].
    pub fn venus_position(&self, epoch: Epoch) -> Result<FrameVec<ECI>, EphemerisError> {
        let state = self.inner.translate(VENUS_J2000, EARTH_J2000, epoch, None)?;
        Ok(km_to_m(state.radius_km.x, state.radius_km.y, state.radius_km.z))
    }

    /// Position of Mars (barycenter) relative to Earth center, in ECI (J2000) [m].
    pub fn mars_position(&self, epoch: Epoch) -> Result<FrameVec<ECI>, EphemerisError> {
        let state = self.inner.translate(MARS_BARYCENTER_J2000, EARTH_J2000, epoch, None)?;
        Ok(km_to_m(state.radius_km.x, state.radius_km.y, state.radius_km.z))
    }

    /// Position of Jupiter (barycenter) relative to Earth center, in ECI (J2000) [m].
    pub fn jupiter_position(&self, epoch: Epoch) -> Result<FrameVec<ECI>, EphemerisError> {
        let state = self.inner.translate(JUPITER_BARYCENTER_J2000, EARTH_J2000, epoch, None)?;
        Ok(km_to_m(state.radius_km.x, state.radius_km.y, state.radius_km.z))
    }

    /// Full state (position [m] + velocity [m/s]) of a body relative to the Sun, in Heliocentric J2000.
    ///
    /// Supports `Body::Earth`, `Body::Mars`, `Body::Venus`, `Body::Jupiter`, `Body::Saturn`,
    /// `Body::Mercury`, `Body::Moon`, and — provided the relevant satellite kernel was
    /// chain-loaded (see `load_almanac`) — `Body::Phobos`, `Body::Deimos`, `Body::Europa`,
    /// `Body::Titan`. The result is the body's position and velocity expressed in the
    /// Sun-centered, J2000-oriented inertial frame (same axes as ECI, different origin).
    ///
    /// # Panics
    /// Panics if called with a body that is not yet wired up (e.g. `Body::Sun` — no meaning).
    pub fn body_state_heliocentric(&self, body: Body, epoch: Epoch) -> Result<FrameState<Heliocentric>, EphemerisError> {
        let frame = match body {
            Body::Earth => EARTH_J2000,
            Body::Moon  => MOON_J2000,
            Body::Mars    => MARS_BARYCENTER_J2000,
            Body::Venus   => VENUS_J2000, // = Frame::new(VENUS_BARYCENTER, J2000); no moons, so body center == barycenter
            Body::Jupiter => JUPITER_BARYCENTER_J2000,
            Body::Saturn  => SATURN_BARYCENTER_J2000,
            Body::Uranus  => URANUS_BARYCENTER_J2000,
            Body::Neptune => NEPTUNE_BARYCENTER_J2000,
            Body::Mercury => MERCURY_J2000, // = Frame::new(MERCURY_BARYCENTER, J2000); no moons
            // Require their own satellite SPK chain-loaded on top of de440s.bsp
            // (mar099s.bsp / jup365.bsp / sat441.bsp) — see `load_almanac`.
            // `translate` below returns an `EphemerisError`, not a panic, if
            // that kernel wasn't loaded.
            Body::Phobos  => PHOBOS_J2000,
            Body::Deimos  => DEIMOS_J2000,
            Body::Europa  => EUROPA_J2000,
            Body::Titan   => TITAN_J2000,
            other => panic!("{:?} is not yet supported by body_state_heliocentric — add its ANISE frame constant", other),
        };
        let state = self.inner.translate(frame, SUN_J2000, epoch, None)?;
        let pos = km_to_m_helio(state.radius_km.x, state.radius_km.y, state.radius_km.z);
        let vel = km_to_m_helio(state.velocity_km_s.x, state.velocity_km_s.y, state.velocity_km_s.z);
        Ok(FrameState::new(pos, vel))
    }

    /// Full state (position [m] + velocity [m/s]) of a body relative to Earth, in ECI.
    ///
    /// Currently supports `Body::Sun` and `Body::Moon`.
    ///
    /// # Panics
    /// Panics if called with a body that is not yet wired up.
    pub fn body_state_eci(&self, body: Body, epoch: Epoch) -> Result<FrameState<ECI>, EphemerisError> {
        let frame = match body {
            Body::Sun  => SUN_J2000,
            Body::Moon => MOON_J2000,
            other => panic!("{:?} is not yet supported by body_state_eci — add its ANISE frame constant", other),
        };
        let state = self.inner.translate(frame, EARTH_J2000, epoch, None)?;
        let pos = km_to_m(state.radius_km.x, state.radius_km.y, state.radius_km.z);
        let vel = km_to_m(state.velocity_km_s.x, state.velocity_km_s.y, state.velocity_km_s.z);
        Ok(FrameState::new(pos, vel))
    }
}

/// Build a `FrameVec<ECI>` from x/y/z in km, converting to meters.
///
/// Coordinates are extracted individually (x, y, z) rather than passing the
/// nalgebra vector directly, because ANISE compiles against a different version
/// of nalgebra than our workspace — accessing scalar fields is version-agnostic.
fn km_to_m(x_km: f64, y_km: f64, z_km: f64) -> FrameVec<ECI> {
    FrameVec::new(SVector::<f64, 3>::new(
        x_km * 1000.0,
        y_km * 1000.0,
        z_km * 1000.0,
    ))
}

/// Same as `km_to_m` but returns a `FrameVec<Heliocentric>`.
fn km_to_m_helio(x_km: f64, y_km: f64, z_km: f64) -> FrameVec<Heliocentric> {
    FrameVec::new(SVector::<f64, 3>::new(
        x_km * 1000.0,
        y_km * 1000.0,
        z_km * 1000.0,
    ))
}

// ==============================================================================
// Tests
// ==============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Requires `kernels/de440s.bsp` to be present. Run with:
    /// `cargo test -p ephemeris -- --ignored`
    #[test]
    #[ignore = "requires kernels/de440s.bsp — download from https://public-data.nyxspace.com/anise/de440s.bsp"]
    fn mars_heliocentric_distance_reasonable() {
        let almanac = Almanac::new(&crate::find_kernel("de440s.bsp")).unwrap();
        let epoch = Epoch::from_gregorian_utc(2026, 1, 1, 0, 0, 0, 0);
        let mars_state = almanac.body_state_heliocentric(crate::bodies::Body::Mars, epoch).unwrap();

        let au_m = 1.496e11_f64;
        let dist = mars_state.position.norm();
        // Mars semi-major axis is 1.524 AU, but can range ~1.38-1.67 AU
        let dist_au = dist / au_m;
        assert!(dist_au > 1.3 && dist_au < 1.7,
            "Mars heliocentric distance should be 1.38-1.67 AU, got {:.3} AU", dist_au);
    }

    #[test]
    #[ignore = "requires kernels/de440s.bsp — download from https://public-data.nyxspace.com/anise/de440s.bsp"]
    fn sun_position_approximately_one_au() {
        let almanac = Almanac::new(&crate::find_kernel("de440s.bsp")).unwrap();
        let epoch = Epoch::from_gregorian_utc(2025, 3, 2, 12, 0, 0, 0);
        let sun_pos = almanac.sun_position(epoch).unwrap();

        let au_m = 1.496e11_f64;
        let dist = sun_pos.norm();
        let rel_err = (dist - au_m).abs() / au_m;

        assert!(rel_err < 0.05,
            "Sun distance from Earth should be ~1 AU, got {:.3e} m (rel err {:.1}%)",
            dist, rel_err * 100.0);
    }
}
