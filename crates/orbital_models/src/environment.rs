//! Atmospheric density models
//!
//! This module provides atmospheric density calculations at various altitudes.
//! Currently implements a simplified model, but can be extended to more sophisticated
//! models like Jacchia-Roberts or NRL-MSISE-00.

/// Atmospheric density calculator
pub struct AtmosphereModel;

impl AtmosphereModel {
    /// Computes the total atmospheric density at a given altitude in meters
    ///
    /// # Arguments
    /// * `altitude` - Altitude above Earth's surface in meters
    ///
    /// # Returns
    /// Atmospheric density in kg/m³
    ///
    /// # Note
    /// This is a very simplified model from https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
    /// For production use, consider implementing:
    /// - Jacchia-Roberts
    /// - NRL-MSISE-00
    /// - Or other more recent atmospheric models
    pub fn density(altitude: f64) -> f64 {
        // Temperature in Celsius
        let temperature = -131.21 + 0.00299 * altitude;
        // Temperature in Kelvin
        let temp_kelvin = temperature + 273.1;

        // Pressure in kPa (from NASA formula)
        let pressure_kpa = 2.488 * ((temp_kelvin / 216.6).powf(-11.388));
        // Convert to Pa
        let pressure_pa = pressure_kpa * 1000.0;

        // Ideal gas law: ρ = P / (R * T)
        // R for air = 287.05 J/(kg·K)
        let r_air = 287.05;
        pressure_pa / (r_air * temp_kelvin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_decreases_with_altitude() {
        let density_100km = AtmosphereModel::density(100e3);
        let density_500km = AtmosphereModel::density(500e3);
        assert!(density_100km > density_500km);
    }
}
