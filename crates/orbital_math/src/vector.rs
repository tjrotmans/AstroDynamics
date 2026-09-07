//! Vector wrapper for numerical integration trait compatibility
//!
//! Wraps `nalgebra::SVector` to implement the `maths_traits` traits
//! required by the `numerical_integration` crate.

#[derive(Clone, Debug)]
pub struct Vector<T, const SIZE: usize>(pub nalgebra::SVector<T, SIZE>);

// Impl Div
impl<const SIZE: usize> std::ops::Div<f64> for Vector<f64, SIZE> {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Vector(self.0 / rhs)
    }
}

// Impl DivAssign
impl<const SIZE: usize> std::ops::DivAssign<f64> for Vector<f64, SIZE> {
    fn div_assign(&mut self, rhs: f64) {
        self.0 /= rhs;
    }
}

// Impl Mul
impl<const SIZE: usize> std::ops::Mul<f64> for Vector<f64, SIZE> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Vector(self.0 * rhs)
    }
}

// Impl MulAssign
impl<const SIZE: usize> std::ops::MulAssign<f64> for Vector<f64, SIZE> {
    fn mul_assign(&mut self, rhs: f64) {
        self.0 *= rhs;
    }
}

// Impl Add between two vectors
impl<const SIZE: usize> std::ops::Add<Vector<f64, SIZE>> for Vector<f64, SIZE> {
    type Output = Self;

    fn add(self, rhs: Vector<f64, SIZE>) -> Self::Output {
        Vector(self.0 + rhs.0)
    }
}

// Impl AddAssign between two vectors
impl<const SIZE: usize> std::ops::AddAssign<Vector<f64, SIZE>> for Vector<f64, SIZE> {
    fn add_assign(&mut self, rhs: Vector<f64, SIZE>) {
        self.0 += rhs.0;
    }
}

// Impl Sub between two vectors
impl<const SIZE: usize> std::ops::Sub<Vector<f64, SIZE>> for Vector<f64, SIZE> {
    type Output = Self;

    fn sub(self, rhs: Vector<f64, SIZE>) -> Self::Output {
        Vector(self.0 - rhs.0)
    }
}

// Impl SubAssign between two vectors
impl<const SIZE: usize> std::ops::SubAssign<Vector<f64, SIZE>> for Vector<f64, SIZE> {
    fn sub_assign(&mut self, rhs: Vector<f64, SIZE>) {
        self.0 -= rhs.0;
    }
}

// Impl Neg
impl<const SIZE: usize> std::ops::Neg for Vector<f64, SIZE> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Vector(-self.0)
    }
}

// Impl Distributive
impl<const SIZE: usize> maths_traits::algebra::ring_like::Distributive<f64> for Vector<f64, SIZE> {}

// Impl AddCommutative
impl<const SIZE: usize> maths_traits::algebra::AddCommutative for Vector<f64, SIZE> {}

// Impl AddAssociative
impl<const SIZE: usize> maths_traits::algebra::AddAssociative for Vector<f64, SIZE> {}

// Impl Zero
impl<const SIZE: usize> maths_traits::algebra::Zero for Vector<f64, SIZE> {
    fn zero() -> Self {
        Vector(nalgebra::SVector::zeros())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

// Impl InnerProductSpace
impl<const SIZE: usize> maths_traits::analysis::InnerProductSpace<f64> for Vector<f64, SIZE> {
    fn inner_product(self, rhs: Self) -> f64 {
        self.0.dot(&rhs.0)
    }
}

impl<const SIZE: usize> From<[f64; SIZE]> for Vector<f64, SIZE> {
    fn from(v: [f64; SIZE]) -> Self {
        Vector(nalgebra::SVector::<f64, SIZE>::from(v))
    }
}
