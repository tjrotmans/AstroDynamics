//! Reference frame marker types and frame-typed vector wrappers
//!
//! Frame markers are zero-sized types (ZSTs) used as phantom type parameters.
//! They carry no runtime data — the frame information exists only at compile time,
//! allowing the compiler to reject accidental mixing of vectors from different frames.
//!
//! # Example
//! ```rust
//! use ephemeris::frames::{ECI, Heliocentric, FrameVec};
//! use nalgebra::SVector;
//!
//! let a: FrameVec<ECI> = FrameVec::new(SVector::<f64, 3>::new(1.0, 0.0, 0.0));
//! let b: FrameVec<ECI> = FrameVec::new(SVector::<f64, 3>::new(0.0, 1.0, 0.0));
//! let c = a + b; // compiles: same frame
//!
//! // let bad: FrameVec<Heliocentric> = FrameVec::new(SVector::<f64, 3>::new(0.0, 0.0, 1.0));
//! // let _ = a + bad; // compile error: ECI ≠ Heliocentric
//! ```

use nalgebra::SVector;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};

// ==============================================================================
// Frame Markers
// ==============================================================================

/// Earth-Centered Inertial (J2000 / ICRF)
///
/// Origin: Earth center of mass.
/// Axes: non-rotating, aligned with ICRS. Z toward north celestial pole at J2000.0,
/// X toward the vernal equinox at J2000.0.
///
/// This is the primary frame for near-Earth orbital mechanics.
pub struct ECI;

/// Heliocentric Inertial (J2000)
///
/// Origin: Sun center of mass.
/// Axes: aligned with ICRS (same orientation as ECI, different origin).
///
/// Used for solar system trajectory design and multi-body problems.
pub struct Heliocentric;

/// Local Vertical, Local Horizontal (LVLH / RSW)
///
/// Origin: spacecraft center of mass.
/// Axes: R (radial, away from Earth), S (along-track), W (cross-track / normal).
///
/// Used for relative motion and attitude description in Earth orbit.
pub struct LVLH;

/// Earth-Centered, Earth-Fixed (ITRF93)
///
/// Origin: Earth center of mass.
/// Axes: rotate with Earth. X toward prime meridian intersection with equator.
///
/// Used for ground station tracking and surface coordinates.
pub struct ECEF;

// ==============================================================================
// FrameVec<F>
// ==============================================================================

/// A 3D vector in a specific reference frame.
///
/// The type parameter `F` is a frame marker (e.g. `ECI`, `Heliocentric`).
/// Arithmetic operations are only defined for vectors in the **same frame**,
/// so accidental mixing of frames produces a compile error rather than a runtime bug.
///
/// The `inner` field is public for interoperability with nalgebra operations,
/// but prefer using the provided methods to preserve frame safety.
#[derive(Debug)]
pub struct FrameVec<F> {
    /// The underlying nalgebra vector [m] (or [m/s] depending on context).
    pub inner: SVector<f64, 3>,
    _frame: PhantomData<F>,
}

impl<F> FrameVec<F> {
    /// Construct a frame-tagged vector from a raw nalgebra vector.
    pub fn new(v: SVector<f64, 3>) -> Self {
        Self { inner: v, _frame: PhantomData }
    }

    /// Euclidean norm of the vector.
    pub fn norm(&self) -> f64 {
        self.inner.norm()
    }

    /// Return a unit vector in the same frame.
    pub fn normalize(&self) -> Self {
        Self::new(self.inner.normalize())
    }

    /// Dot product with another vector in the same frame.
    pub fn dot(&self, other: &Self) -> f64 {
        self.inner.dot(&other.inner)
    }

    /// Cross product with another vector in the same frame.
    pub fn cross(&self, other: &Self) -> Self {
        Self::new(self.inner.cross(&other.inner))
    }
}

// Manual Copy/Clone implementations avoid the overly-conservative F: Copy bound
// that #[derive(Copy)] would add. All our frame markers are ZSTs and are Copy,
// but a manual impl lets FrameVec<F> be Copy unconditionally.
impl<F> Copy for FrameVec<F> {}
impl<F> Clone for FrameVec<F> {
    fn clone(&self) -> Self { *self }
}

impl<F> Add for FrameVec<F> {
    type Output = FrameVec<F>;
    fn add(self, rhs: FrameVec<F>) -> FrameVec<F> {
        FrameVec::new(self.inner + rhs.inner)
    }
}

impl<F> Sub for FrameVec<F> {
    type Output = FrameVec<F>;
    fn sub(self, rhs: FrameVec<F>) -> FrameVec<F> {
        FrameVec::new(self.inner - rhs.inner)
    }
}

impl<F> Mul<f64> for FrameVec<F> {
    type Output = FrameVec<F>;
    fn mul(self, s: f64) -> FrameVec<F> {
        FrameVec::new(self.inner * s)
    }
}

impl<F> Neg for FrameVec<F> {
    type Output = FrameVec<F>;
    fn neg(self) -> FrameVec<F> {
        FrameVec::new(-self.inner)
    }
}

impl<F> From<SVector<f64, 3>> for FrameVec<F> {
    fn from(v: SVector<f64, 3>) -> Self {
        FrameVec::new(v)
    }
}

// ==============================================================================
// FrameState<F>
// ==============================================================================

/// A 6D state vector (position + velocity) in a specific reference frame.
///
/// Both position and velocity are expressed in the same frame `F`.
#[derive(Clone, Copy, Debug)]
pub struct FrameState<F> {
    /// Position vector [m]
    pub position: FrameVec<F>,
    /// Velocity vector [m/s]
    pub velocity: FrameVec<F>,
}

impl<F> FrameState<F> {
    /// Construct from separate position and velocity vectors.
    pub fn new(position: FrameVec<F>, velocity: FrameVec<F>) -> Self {
        Self { position, velocity }
    }
}

// ==============================================================================
// Tests
// ==============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framevec_add_stays_in_frame() {
        let a: FrameVec<ECI> = FrameVec::new(SVector::<f64, 3>::new(1.0, 0.0, 0.0));
        let b: FrameVec<ECI> = FrameVec::new(SVector::<f64, 3>::new(0.0, 2.0, 0.0));
        let c = a + b;
        assert!((c.inner[0] - 1.0).abs() < 1e-15);
        assert!((c.inner[1] - 2.0).abs() < 1e-15);
        assert!((c.inner[2] - 0.0).abs() < 1e-15);
    }

    #[test]
    fn framevec_normalize_is_unit() {
        let v: FrameVec<ECI> = FrameVec::new(SVector::<f64, 3>::new(3.0, 4.0, 0.0));
        let u = v.normalize();
        assert!((u.norm() - 1.0).abs() < 1e-15, "norm = {}", u.norm());
    }

    #[test]
    fn framevec_sub_and_scalar() {
        let a: FrameVec<Heliocentric> = FrameVec::new(SVector::<f64, 3>::new(5.0, 0.0, 0.0));
        let b: FrameVec<Heliocentric> = FrameVec::new(SVector::<f64, 3>::new(2.0, 0.0, 0.0));
        let diff = a - b;
        let scaled = diff * 2.0;
        assert!((scaled.inner[0] - 6.0).abs() < 1e-15);
    }

    #[test]
    fn framevec_dot_and_cross() {
        let x: FrameVec<ECI> = FrameVec::new(SVector::<f64, 3>::new(1.0, 0.0, 0.0));
        let y: FrameVec<ECI> = FrameVec::new(SVector::<f64, 3>::new(0.0, 1.0, 0.0));
        assert!((x.dot(&y) - 0.0).abs() < 1e-15);
        let z = x.cross(&y);
        assert!((z.inner[2] - 1.0).abs() < 1e-15);
    }
}
