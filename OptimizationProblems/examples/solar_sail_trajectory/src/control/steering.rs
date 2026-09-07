//! Locally Optimal Steering Laws for Solar Sail Orbit Control
//!
//! This module implements McInnes' planet-centered locally optimal steering law
//! for maximizing instantaneous orbital energy gain (and thus orbit raising).
//!
//! In McInnes (Sec. 4.4.2.3), the instantaneous energy gain is proportional to
//! (l dot n)^2 * (n dot v), where:
//! - l is the sun-line unit vector (direction of incoming sunlight),
//! - n is the sail normal unit vector,
//! - v is the velocity direction.
//!
//! Defining psi as the angle between v and l, the optimal pitch (cone)
//! angle is (Eq. 4.100):
//!
//! alpha* = 1/2 [psi - asin(sin(psi)/3)].
//!
//! We choose the sail normal to lie in the plane spanned by the sun-line and the
//! velocity direction, and return the corresponding (cone, clock) in the same
//! sun-pointing frame used by the SRP acceleration model.

use orbital_models::OrbitalElements;

/// Locally optimal steering law (planet-centered) for maximizing instantaneous
/// orbital energy gain.
pub struct LocallyOptimalSMA;

impl LocallyOptimalSMA {
    /// Compute the locally optimal (cone, clock) for planet-centered orbit raising.
    ///
    /// # Arguments
    /// * `orbital_elements` - Currently unused (kept for API compatibility)
    /// * `sun_direction` - Sun-line unit vector in inertial frame.
    ///   Must match the SRP model convention (sun -> spacecraft direction).
    /// * `velocity` - Spacecraft inertial velocity vector (m/s)
    /// * `position` - Spacecraft inertial position vector (m) (used only for robust clock extraction)
    ///
    /// # Returns
    /// (cone, clock) angles (radians) in the sun-pointing frame.
    pub fn compute_angles(
        _orbital_elements: &OrbitalElements,
        sun_direction: &[f64; 3],
        velocity: &[f64; 3],
        position: &[f64; 3],
    ) -> (f64, f64) {
        let l_hat = match normalize(*sun_direction) {
            Some(v) => v,
            None => return (0.0, 0.0),
        };

        let v_hat = match normalize(*velocity) {
            Some(v) => v,
            None => return (0.0, 0.0),
        };

        // psi = angle between v and sun-line l
        let cos_psi = dot(l_hat, v_hat).clamp(-1.0, 1.0);
        let psi = cos_psi.acos();
        let sin_psi = (1.0 - cos_psi * cos_psi).max(0.0).sqrt();

        // Eq. 4.100
        let asin_term = (sin_psi / 3.0).clamp(-1.0, 1.0).asin();
        let mut cone = 0.5 * (psi - asin_term);
        cone = cone.clamp(0.0, std::f64::consts::FRAC_PI_2);

        // If cone ~ 0, clock is irrelevant.
        if cone.sin().abs() < 1e-12 {
            return (cone, 0.0);
        }

        // Build a unit vector in the (l, v) plane perpendicular to l.
        // p_hat points in the direction of v's component perpendicular to l.
        let v_perp = sub(v_hat, scale(l_hat, cos_psi));
        let p_hat = if let Some(p) = normalize(v_perp) {
            p
        } else {
            // v parallel to l: any clock is equivalent.
            // Pick a stable perpendicular direction.
            let fallback = perpendicular_unit(l_hat, *position);
            match fallback {
                Some(p) => p,
                None => return (cone, 0.0),
            }
        };

        // Sail normal in inertial frame: n = cos(alpha) l + sin(alpha) p
        let mut n_hat = add(scale(l_hat, cone.cos()), scale(p_hat, cone.sin()));
        // Front-side convention: if the normal points toward the sun,
        // flip it 180 deg so it points away from the sun.
        if dot(n_hat, l_hat) < 0.0 {
            n_hat = scale(n_hat, -1.0);
        }

        // Convert inertial normal to cone/clock in the SRP sun-frame.
        // Note: we preserve sign of the cone to represent full 2pi azimuth with clock in [0, pi].
        let (cone2, clock) = cone_clock_from_normal(l_hat, n_hat);
        (cone2, clock)
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = norm(v);
    if n < 1e-12 {
        None
    } else {
        Some([v[0] / n, v[1] / n, v[2] / n])
    }
}

/// Create a stable unit vector perpendicular to `l_hat`.
///
/// Uses an inertial axis cross-product fallback, and if needed uses the position
/// vector (to avoid degeneracy in rare alignments).
fn perpendicular_unit(l_hat: [f64; 3], position: [f64; 3]) -> Option<[f64; 3]> {
    let z = [0.0, 0.0, 1.0];
    let x = [1.0, 0.0, 0.0];

    let mut c = cross(z, l_hat);
    if norm(c) < 1e-12 {
        c = cross(x, l_hat);
    }
    if norm(c) < 1e-12 {
        c = cross(position, l_hat);
    }
    normalize(c)
}

/// Convert an inertial sail normal to (cone, clock) using the same sun-frame
/// basis as the SRP model:
/// - s_hat = l_hat
/// - theta_hat = z cross s_hat
/// - phi_hat = s_hat cross theta_hat
fn cone_clock_from_normal(l_hat: [f64; 3], n_hat: [f64; 3]) -> (f64, f64) {
    let cos_cone = dot(l_hat, n_hat).clamp(-1.0, 1.0);
    let mut cone = cos_cone.acos();
    // Physical/front-side convention: cone magnitude in [0, pi/2]
    cone = cone.clamp(0.0, std::f64::consts::FRAC_PI_2);

    // If cone ~ 0, clock is irrelevant.
    if cone.sin().abs() < 1e-12 {
        return (cone, 0.0);
    }

    let z = [0.0, 0.0, 1.0];
    let mut theta_hat = cross(z, l_hat);
    if let Some(t) = normalize(theta_hat) {
        theta_hat = t;
    } else {
        // Rare degenerate case: sun-line aligned with z.
        // Pick any perpendicular axis.
        theta_hat = normalize(cross([1.0, 0.0, 0.0], l_hat)).unwrap_or([0.0, 1.0, 0.0]);
    }
    let phi_hat = normalize(cross(l_hat, theta_hat)).unwrap_or([0.0, 0.0, 0.0]);

    // Perpendicular component direction u in the sun-frame: u = sin(delta) theta_hat + cos(delta) phi_hat
    let proj = sub(n_hat, scale(l_hat, cos_cone));
    let proj_hat = match normalize(proj) {
        Some(p) => p,
        None => return (cone, 0.0),
    };

    let sin_delta = dot(proj_hat, theta_hat);
    let cos_delta = dot(proj_hat, phi_hat);
    let mut clock = sin_delta.atan2(cos_delta); // [-pi, pi]

    // Control convention used in this repo:
    // - clock in [0, pi]
    // - cone in [-pi/2, pi/2]
    // A negative clock can be mapped into [0, pi] by adding pi and flipping the cone sign.
    // This preserves the represented normal because it flips the transverse components.
    if clock < 0.0 {
        clock += std::f64::consts::PI;
        cone = -cone;
    }
    clock = clock.clamp(0.0, std::f64::consts::PI);
    cone = cone.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);

    (cone, clock)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_elements() -> OrbitalElements {
        OrbitalElements {
            a: 7.0e6,
            e: 0.0,
            i: 0.0,
            w: 0.0,
            o: 0.0,
            nu: 0.0,
        }
    }

    #[test]
    fn planet_centered_law_special_cases() {
        let oe = dummy_elements();
        let pos = [7.0e6, 0.0, 0.0];

        // psi = 0 => alpha* = 0
        let l = [1.0, 0.0, 0.0];
        let v = [10.0, 0.0, 0.0];
        let (cone, clock) = LocallyOptimalSMA::compute_angles(&oe, &l, &v, &pos);
        assert!((cone - 0.0).abs() < 1e-10);
        assert!(clock.is_finite());

        // psi = pi/2 should yield a finite cone between 0 and pi/2.
        let v2 = [0.0, 10.0, 0.0];
        let (cone2, clock2) = LocallyOptimalSMA::compute_angles(&oe, &l, &v2, &pos);
        assert!(cone2 > 0.0);
        assert!(cone2 < std::f64::consts::FRAC_PI_2);
        assert!(clock2.is_finite());
    }

    #[test]
    fn eq_4100_is_a_maximum_over_cone() {
        // Objective proportional to: cos^2(alpha) * cos(psi - alpha)
        // where alpha is the cone (pitch) measured from the sun-line in the (l,v) plane.
        fn objective(alpha: f64, psi: f64) -> f64 {
            alpha.cos().powi(2) * (psi - alpha).cos()
        }

        // Sweep psi across [0, pi] and verify the closed-form alpha* is near the numeric maximizer.
        for k in 0..=60 {
            let psi = (k as f64) * std::f64::consts::PI / 60.0;
            let sin_psi = psi.sin();
            let asin_term = (sin_psi / 3.0).clamp(-1.0, 1.0).asin();
            let alpha_star = (0.5 * (psi - asin_term)).clamp(0.0, std::f64::consts::FRAC_PI_2);

            // Brute force maximize on [0, pi/2]
            let mut best_alpha = 0.0;
            let mut best_val = f64::NEG_INFINITY;
            let mut a = 0.0;
            while a <= std::f64::consts::FRAC_PI_2 {
                let val = objective(a, psi);
                if val > best_val {
                    best_val = val;
                    best_alpha = a;
                }
                a += std::f64::consts::PI / 2000.0;
            }

            assert!((best_alpha - alpha_star).abs() < 1e-2);
        }
    }
}
