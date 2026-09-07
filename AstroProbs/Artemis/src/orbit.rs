//! Orbital mechanics helpers for the TLI initial state and burn direction.

use nalgebra::SVector;
use orbital_models::OrbitalElements;
use crate::config::{MissionConfig, EARTH_RADIUS_M};

/// Compute ECI position and velocity at TLI ignition (perigee of parking orbit).
///
/// Places the perigee in the approximate anti-Moon direction at the flyby epoch,
/// then derives RAAN and argument of periapsis so the orbit has the configured
/// inclination and contains that perigee direction.
///
/// # Arguments
/// * `cfg`           – Mission configuration
/// * `moon_flyby_eci` – Moon ECI position at the expected flyby epoch [m]
pub fn compute_tli_initial_state(
    cfg: &MissionConfig,
    moon_flyby_eci: SVector<f64, 3>,
) -> (SVector<f64, 3>, SVector<f64, 3>) {
    if let Some(elements) = cfg.initial_kepler_elements {
        let sv = elements.as_cartesian();
        let p = sv.position();
        let v = sv.velocity();
        return (
            SVector::<f64, 3>::new(p[0], p[1], p[2]),
            SVector::<f64, 3>::new(v[0], v[1], v[2]),
        );
    }

    let r_e  = EARTH_RADIUS_M;
    let r_p  = r_e + cfg.parking_perigee_alt_m;
    let r_a  = r_e + cfg.parking_apogee_alt_m;
    let a    = (r_p + r_a) / 2.0;
    let ecc  = (r_a - r_p) / (r_a + r_p);
    let incl = cfg.inclination_rad;

    // Perigee is anti-Moon: firing prograde at perigee raises apogee toward the Moon.
    let moon_hat = moon_flyby_eci / moon_flyby_eci.norm();
    let p_hat    = -moon_hat;

    // Solve for RAAN (Omega): px*sin(Ω) - py*cos(Ω) = -pz*cot(i)
    let (px, py, pz) = (p_hat[0], p_hat[1], p_hat[2]);
    let cot_i = incl.cos() / incl.sin();
    let big_a = px;
    let big_b = -py;
    let big_c = -pz * cot_i;
    let r_ab  = (big_a * big_a + big_b * big_b).sqrt();
    let phi   = big_a.atan2(big_b);
    let ratio = (big_c / r_ab).clamp(-1.0, 1.0);
    let angle = ratio.acos();

    // Two candidate RAANs — pick the prograde solution (orbital h · z > 0)
    let omega_node = {
        let c1 = phi + angle;
        let c2 = phi - angle;
        let mut best = c1;
        for &cand in &[c1, c2] {
            let n = SVector::<f64, 3>::new(
                cand.sin() * incl.sin(),
               -cand.cos() * incl.sin(),
                incl.cos(),
            );
            let v_approx = n.cross(&p_hat);
            if v_approx[1] * px - v_approx[0] * py > 0.0 {
                best = cand;
                break;
            }
        }
        best
    };

    // Solve for argument of periapsis
    let sin_w = pz / incl.sin();
    let cos_w = px * omega_node.cos() + py * omega_node.sin();
    let omega  = sin_w.atan2(cos_w);

    let elements = OrbitalElements { a, e: ecc, i: incl, o: omega_node, w: omega, nu: 0.0 };
    let sv = elements.as_cartesian();
    let p  = sv.position();
    let v  = sv.velocity();
    (
        SVector::<f64, 3>::new(p[0], p[1], p[2]),
        SVector::<f64, 3>::new(v[0], v[1], v[2]),
    )
}

/// Compute thrust unit vector: prograde at ignition, rotated by pitch and yaw offsets.
///
/// - Pitch: rotation in the orbital plane (positive = toward angular momentum axis)
/// - Yaw: rotation out of the orbital plane
pub fn compute_burn_direction(
    pos: &SVector<f64, 3>,
    vel: &SVector<f64, 3>,
    pitch_rad: f64,
    yaw_rad: f64,
) -> SVector<f64, 3> {
    let v_hat = vel / vel.norm();
    let r_hat = pos / pos.norm();
    let h_hat = r_hat.cross(&v_hat).normalize();

    let after_pitch = pitch_rad.cos() * v_hat + pitch_rad.sin() * h_hat;
    let perp        = after_pitch.cross(&h_hat).normalize();
    yaw_rad.cos() * after_pitch + yaw_rad.sin() * perp
}

/// Find the step with smallest distance to the Moon.
/// Returns (step index, distance [m]).
pub fn find_closest_approach(steps: &[crate::propagator::TrajectoryStep]) -> (usize, f64) {
    let mut best_idx  = 0;
    let mut best_dist = f64::INFINITY;
    for (i, step) in steps.iter().enumerate() {
        let dist = (step.pos - step.moon_pos).norm();
        if dist < best_dist {
            best_dist = dist;
            best_idx  = i;
        }
    }
    (best_idx, best_dist)
}
