//! Orbital mechanics models
//!
//! This module provides orbital element representations and conversions between
//! Keplerian elements and Cartesian state vectors.

use nalgebra::{Matrix3, SVector, Vector3};
use crate::constants::{MU_EARTH, EARTH_RADIUS};

/// Cartesian state vector (position and velocity)
#[derive(Clone, Copy, Debug)]
pub struct StateVector {
    pub vector: [f64; 6],
}

impl StateVector {
    pub fn new(position: [f64; 3], velocity: [f64; 3]) -> Self {
        StateVector {
            vector: [
                position[0],
                position[1],
                position[2],
                velocity[0],
                velocity[1],
                velocity[2],
            ],
        }
    }

    pub fn position(&self) -> [f64; 3] {
        [self.vector[0], self.vector[1], self.vector[2]]
    }

    pub fn velocity(&self) -> [f64; 3] {
        [self.vector[3], self.vector[4], self.vector[5]]
    }
}

impl From<SVector<f64, 6>> for StateVector {
    fn from(v: SVector<f64, 6>) -> Self {
        StateVector {
            vector: [v[0], v[1], v[2], v[3], v[4], v[5]],
        }
    }
}

impl From<orbital_math::Vector<f64, 6>> for StateVector {
    fn from(v: orbital_math::Vector<f64, 6>) -> Self {
        StateVector {
            vector: [v.0[0], v.0[1], v.0[2], v.0[3], v.0[4], v.0[5]],
        }
    }
}

impl From<StateVector> for orbital_math::Vector<f64, 6> {
    fn from(v: StateVector) -> Self {
        orbital_math::Vector(nalgebra::SVector::<f64, 6>::from_column_slice(&v.vector))
    }
}

/// Keplerian orbital elements
#[derive(Clone, Copy, Debug)]
pub struct OrbitalElements {
    /// Semi-major axis in meters
    pub a: f64,
    /// Eccentricity
    pub e: f64,
    /// Inclination in radians
    pub i: f64,
    /// Argument of periapsis in radians
    pub w: f64,
    /// Longitude of the ascending node in radians
    pub o: f64,
    /// True anomaly in radians
    pub nu: f64,
}

impl OrbitalElements {
    /// Earth radius in meters (kept as associated constant for backward compatibility)
    pub const EARTH_RADIUS: f64 = EARTH_RADIUS;

    /// Convert orbital elements to Cartesian state vector
    pub fn as_cartesian(&self) -> StateVector {
        let a = self.a;
        let e = self.e;
        let i = self.i;
        let w = self.w;
        let o = self.o;
        let nu = self.nu;

        let p = a * (1.0 - e * e);

        // Debug check
        if p.is_nan() || p <= 0.0 {
            eprintln!("ERROR in as_cartesian: invalid p = {} (a={}, e={})", p, a, e);
        }

        // Position vector in PQW frame
        let r_x = p * (nu.cos() / (1.0 + e * nu.cos()));
        let r_y = p * (nu.sin() / (1.0 + e * nu.cos()));
        let r_z = 0.0;
        let r = Vector3::new(r_x, r_y, r_z);

        // Velocity vector in PQW frame
        let v_x = -(MU_EARTH / p).sqrt() * nu.sin();
        let v_y = (MU_EARTH / p).sqrt() * (e + nu.cos());
        let v_z = 0.0;
        let v = Vector3::new(v_x, v_y, v_z);

        // Debug check velocity
        if v.x.is_nan() || v.y.is_nan() {
            eprintln!("ERROR in as_cartesian: NaN velocity");
            eprintln!("  v_x = {}, v_y = {}", v_x, v_y);
            eprintln!("  sqrt(MU/p) = {}", (MU_EARTH / p).sqrt());
            eprintln!("  e = {}, nu = {}", e, nu);
        }

        // Transformation from perifocal to equatorial frame
        let r_pqw_to_eq = Matrix3::new(
            o.cos() * w.cos() - o.sin() * i.cos() * w.sin(),
            -o.cos() * w.sin() - o.sin() * i.cos() * w.cos(),
            o.sin() * i.sin(),
            o.sin() * w.cos() + o.cos() * i.cos() * w.sin(),
            -o.sin() * w.sin() + o.cos() * i.cos() * w.cos(),
            -o.cos() * i.sin(),
            i.sin() * w.sin(),
            i.sin() * w.cos(),
            i.cos(),
        );

        let position_eq = r_pqw_to_eq * r;
        let velocity_eq = r_pqw_to_eq * v;

        StateVector::new(
            [position_eq.x, position_eq.y, position_eq.z],
            [velocity_eq.x, velocity_eq.y, velocity_eq.z],
        )
    }

    /// Convert Cartesian state vector to orbital elements
    pub fn from_cartesian(position: [f64; 3], velocity: [f64; 3]) -> Self {
        let r = Vector3::new(position[0], position[1], position[2]);
        let v = Vector3::new(velocity[0], velocity[1], velocity[2]);

        let h = r.cross(&v);

        let e_vec = ((v.norm_squared() - MU_EARTH / r.norm()) * r - r.dot(&v) * v) / MU_EARTH;
        let e = e_vec.norm();
        let a = 1.0 / (2.0 / r.norm() - v.norm_squared() / MU_EARTH);
        let i = (h[2] / h.norm()).acos();
        let n = Vector3::new(0.0, 0.0, 1.0).cross(&h);

        let o = if n[1] >= 0.0 {
            (n[0] / n.norm()).acos()
        } else {
            2.0 * std::f64::consts::PI - (n[0] / n.norm()).acos()
        };

        let w = if e_vec[2] >= 0.0 {
            (n.dot(&e_vec) / (n.norm() * e_vec.norm())).acos()
        } else {
            2.0 * std::f64::consts::PI - (n.dot(&e_vec) / (n.norm() * e_vec.norm())).acos()
        };

        let nu = if r.dot(&v) >= 0.0 {
            (e_vec.dot(&r) / (e_vec.norm() * r.norm())).acos()
        } else {
            2.0 * std::f64::consts::PI - (e_vec.dot(&r) / (e_vec.norm() * r.norm())).acos()
        };

        OrbitalElements { a, e, i, w, o, nu }
    }

    pub fn from_state_vector(vector: StateVector) -> Self {
        Self::from_cartesian(vector.position(), vector.velocity())
    }

    pub fn position(&self) -> [f64; 3] {
        self.as_cartesian().position()
    }

    pub fn speed(&self) -> [f64; 3] {
        self.as_cartesian().velocity()
    }

    /// Compute instantaneous altitude from Cartesian position
    pub fn altitude(&self) -> f64 {
        let pos = self.as_cartesian().position();
        let r = SVector::<f64, 3>::from(pos).norm();
        r - EARTH_RADIUS
    }
}

/// Compute vector from Sun to spacecraft position
pub fn sun_vector_from_position(position: [f64; 3]) -> [f64; 3] {
    let sun_pos = crate::constants::SUN_POSITION;
    [
        position[0] - sun_pos[0],
        position[1] - sun_pos[1],
        position[2] - sun_pos[2],
    ]
}

/// Compute unit direction vector from Sun to spacecraft position
pub fn sun_direction_from_position(position: [f64; 3]) -> [f64; 3] {
    let v = sun_vector_from_position(position);
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cartesian_and_back() {
        let orbital_elements = OrbitalElements {
            a: 500e3 + 6371e3,
            e: 0.0,
            i: 88.0f64.to_radians(),
            w: 0.0,
            o: 0.0,
            nu: 0.0,
        };
        let state_vector = orbital_elements.as_cartesian();
        let orbital_elements2 = OrbitalElements::from_state_vector(state_vector);

        // Assert that they're equal to within 1e-6
        assert!((orbital_elements.a - orbital_elements2.a).abs() < 1e-6);
        assert!((orbital_elements.e - orbital_elements2.e).abs() < 1e-6);
        assert!((orbital_elements.i - orbital_elements2.i).abs() < 1e-6);
        assert!((orbital_elements.w - orbital_elements2.w).abs() < 1e-6);
        assert!((orbital_elements.o - orbital_elements2.o).abs() < 1e-6);
        assert!((orbital_elements.nu - orbital_elements2.nu).abs() < 1e-6);
    }

    #[test]
    fn altitude_circular_orbit() {
        let orbit = OrbitalElements {
            a: 500e3 + EARTH_RADIUS,
            e: 0.0,
            i: 0.0,
            w: 0.0,
            o: 0.0,
            nu: 0.0,
        };
        let alt = orbit.altitude();
        assert!((alt - 500e3).abs() < 1.0, "Expected ~500km altitude, got {}", alt);
    }

    #[test]
    fn altitude_different_true_anomaly() {
        // For circular orbit, altitude should be the same at any true anomaly
        let alt_0 = OrbitalElements { a: 400e3 + EARTH_RADIUS, e: 0.0, i: 0.0, w: 0.0, o: 0.0, nu: 0.0 }.altitude();
        let alt_pi = OrbitalElements { a: 400e3 + EARTH_RADIUS, e: 0.0, i: 0.0, w: 0.0, o: 0.0, nu: std::f64::consts::PI }.altitude();
        assert!((alt_0 - alt_pi).abs() < 1.0, "Circular orbit altitude varies: {} vs {}", alt_0, alt_pi);
    }
}
