//! ECI trajectory propagator for the Artemis 2 lunar transfer
//!
//! Integrates the 7D equations of motion [x, y, z, vx, vy, vz, mass] in the
//! ECI (J2000) frame using the Dormand-Prince RK45 solver from `ode_solvers`
//! with separate relative and absolute tolerances.
//!
//! Forces modeled:
//!   - Earth two-body gravity (point mass)
//!   - Earth zonal harmonics J2, J3, J4 (dominant at TLI perigee, ~378 km)
//!   - Moon third-body perturbation (pre-sampled ECI positions, linearly interpolated)
//!   - Solar radiation pressure (cannonball, Sun position fixed over mission)
//!   - ICPS finite burn during TLI window

use nalgebra::SVector;
use ode_solvers::dopri5::Dopri5;
use ode_solvers::{SVector as OdeVec, System};
use orbital_models::GravityModel;
use orbital_models::constants::P_SRP;

use orbital_models::FiniteBurn;
use orbital_models::constants::G0;
use ephemeris::{MoonTrack, SunTrack};

type State = OdeVec<f64, 7>;

/// One logged state.
#[derive(Clone, Debug)]
pub struct TrajectoryStep {
    pub time_s:     f64,
    pub pos:        SVector<f64, 3>,
    pub vel:        SVector<f64, 3>,
    pub mass_kg:    f64,
    pub moon_pos:   SVector<f64, 3>,
    pub sun_pos:    SVector<f64, 3>,
    pub is_burning: bool,
}

// ── ODE system ────────────────────────────────────────────────────────────────

struct ArtemisOde<'a> {
    burn:              &'a FiniteBurn,
    moon_track:        &'a MoonTrack,
    sun_track:         &'a SunTrack,
    srp_area:          f64,
    reflectivity:      f64,
    /// Abort integration when spacecraft-Moon distance drops below this [m].
    /// Set to 0.0 (or negative) to disable. Used by grid searches to skip
    /// Moon-impact trajectories that would otherwise cause the integrator to
    /// grind through near-singular 1/r² forces at the surface.
    abort_moon_dist_m: f64,
}

impl<'a> System<f64, State> for ArtemisOde<'a> {
    fn system(&self, t: f64, y: &State, dy: &mut State) {
        let pos  = SVector::<f64, 3>::new(y[0], y[1], y[2]);
        let vel  = SVector::<f64, 3>::new(y[3], y[4], y[5]);
        let mass = y[6].max(1.0);

        let moon_p     = self.moon_track.position_at(t);
        let sun_p      = self.sun_track.position_at(t);
        let a_earth    = GravityModel::compute(&pos, mass);
        let a_j2_j4   = GravityModel::zonal_harmonics(&pos);
        let a_moon     = GravityModel::moon_third_body(&pos, &moon_p);
        let a_sun      = GravityModel::sun_third_body(&pos, &sun_p);
        let s_vec      = pos - sun_p;
        let s_hat      = s_vec / s_vec.norm();
        let a_srp      = (P_SRP * self.srp_area * (1.0 + self.reflectivity) / mass) * s_hat;

        let burning    = t >= self.burn.ignition_time_s && t < self.burn.cutoff_time_s;
        let a_thrust   = if burning { (self.burn.thrust_n / mass) * self.burn.direction } else { SVector::zeros() };
        let mass_rate  = if burning { -(self.burn.thrust_n / (self.burn.isp_s * G0)) } else { 0.0 };

        let accel = a_earth + a_j2_j4 + a_moon + a_sun + a_srp + a_thrust;

        dy[0] = vel[0]; dy[1] = vel[1]; dy[2] = vel[2];
        dy[3] = accel[0]; dy[4] = accel[1]; dy[5] = accel[2];
        dy[6] = mass_rate;
    }

    /// Stop integration when the spacecraft is below `abort_moon_dist_m` from
    /// the Moon centre. Returning `true` aborts; `false` continues.
    fn solout(&mut self, t: f64, y: &State, _dy: &State) -> bool {
        if self.abort_moon_dist_m <= 0.0 { return false; }
        let pos  = SVector::<f64, 3>::new(y[0], y[1], y[2]);
        let moon = self.moon_track.position_at(t);
        (pos - moon).norm() < self.abort_moon_dist_m
    }
}

// ── Propagate ─────────────────────────────────────────────────────────────────

/// Propagate the Artemis 2 trajectory from TLI ignition.
///
/// Returns states logged at `log_dt_s` intervals plus the final state.
/// Integration runs to `duration_s` unless aborted early by `abort_moon_dist_m`
/// (set ≤ 0.0 to disable — the normal case for production runs).
pub fn propagate_with_abort(
    initial_pos:       SVector<f64, 3>,
    initial_vel:       SVector<f64, 3>,
    initial_mass:      f64,
    burn:              &FiniteBurn,
    moon_track:        &MoonTrack,
    sun_track:         &SunTrack,
    srp_area_m2:       f64,
    reflectivity:      f64,
    duration_s:        f64,
    log_dt_s:          f64,
    rtol:              f64,
    atol:              f64,
    abort_moon_dist_m: f64,
) -> Vec<TrajectoryStep> {
    let ode = ArtemisOde {
        burn,
        moon_track,
        sun_track,
        srp_area: srp_area_m2,
        reflectivity,
        abort_moon_dist_m,
    };

    #[allow(clippy::useless_conversion)]
    let y0 = State::from_column_slice(&[
        initial_pos[0], initial_pos[1], initial_pos[2],
        initial_vel[0], initial_vel[1], initial_vel[2],
        initial_mass,
    ]);

    let mut stepper = Dopri5::new(
        ode,
        0.0,        // t0
        duration_s, // tf
        log_dt_s,   // initial step size hint
        y0,
        rtol,
        atol,
    );

    // Integrate
    let _stats = stepper.integrate().expect("Integration failed");

    let times  = stepper.x_out();
    let states = stepper.y_out();

    let mut log: Vec<TrajectoryStep> = Vec::with_capacity(times.len());
    for (t, y) in times.iter().zip(states.iter()) {
        let t   = *t;
        let pos = SVector::<f64, 3>::new(y[0], y[1], y[2]);
        let vel = SVector::<f64, 3>::new(y[3], y[4], y[5]);
        log.push(TrajectoryStep {
            time_s:     t,
            pos,
            vel,
            mass_kg:    y[6],
            moon_pos:   moon_track.position_at(t),
            sun_pos:    sun_track.position_at(t),
            is_burning: t >= burn.ignition_time_s && t < burn.cutoff_time_s,
        });
    }

    log
}

/// Propagate without an abort condition. Convenience wrapper for all normal
/// (non-grid-search) callers — signature is unchanged from before.
pub fn propagate(
    initial_pos:  SVector<f64, 3>,
    initial_vel:  SVector<f64, 3>,
    initial_mass: f64,
    burn:         &FiniteBurn,
    moon_track:   &MoonTrack,
    sun_track:    &SunTrack,
    srp_area_m2:  f64,
    reflectivity: f64,
    duration_s:   f64,
    log_dt_s:     f64,
    rtol:         f64,
    atol:         f64,
) -> Vec<TrajectoryStep> {
    propagate_with_abort(
        initial_pos, initial_vel, initial_mass,
        burn, moon_track, sun_track,
        srp_area_m2, reflectivity,
        duration_s, log_dt_s, rtol, atol,
        0.0,  // no abort
    )
}
