//! Cyclic angle normalization
//!
//! Maps any real number into a cyclic range [min, max].

/// Maps any real number into [min, max] as if it was cyclic
///
/// This function normalizes angles to their principal range without using loops.
///
/// # Examples
/// ```
/// # use orbital_math::normalize;
/// assert_eq!(normalize(0.0, 0.0, 1.0), 0.0);
/// assert_eq!(normalize(1.5, 0.0, 1.0), 0.5);
/// assert_eq!(normalize(-0.5, -1.0, 1.0), -0.5);
/// ```
pub fn normalize(val: f64, min: f64, max: f64) -> f64 {
    let width = max - min;
    let adjusted = val - min;
    let mut modulo = adjusted % width;
    if modulo < 0.0 {
        modulo += width;
    }
    modulo + min
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        // Test positive values
        assert_eq!(normalize(0.0, 0.0, 1.0), 0.0);
        assert_eq!(normalize(0.5, 0.0, 1.0), 0.5);
        assert_eq!(normalize(1.0, 0.0, 1.0), 0.0);
        assert_eq!(normalize(1.5, 0.0, 1.0), 0.5);

        // Test negative values
        assert_eq!(normalize(-0.5, -1.0, 0.0), -0.5);
        assert_eq!(normalize(-1.0, -1.0, 0.0), -1.0);
        assert_eq!(normalize(-1.5, -1.0, 0.0), -0.5);

        // Test positive & negative values together
        assert_eq!(normalize(-0.5, -1.0, 1.0), -0.5);
        assert_eq!(normalize(1.5, -1.0, 1.0), -0.5);
    }
}
