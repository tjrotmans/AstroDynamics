//! Solar sail Earth-Mars rendezvous optimization problem
//!
//! Integrates heliocentric equations of motion under Sun gravity and solar radiation pressure.
//! Optimizes sail pointing angles (cone/clock at N control points) plus time-of-flight
//! to minimize the rendezvous error (position + velocity) with Mars at arrival.

use std::sync::Arc;

use maths_traits::analysis::InnerProductMetric;
use nalgebra::SVector;
use numerical_integration::AdaptiveIntegrator;
use orbital_math::Vector;
use rusqlite::Transaction;

use ephemeris::{Almanac, Body, Epoch, FrameState, Heliocentric};
use hifitime::Duration;
use optimization::{EvaluationContext, GAProblem, OptimizableProblem};
use orbital_models::{AttitudeFrame, SunPointingFrame};
use orbital_models::constants::{MU_SUN, P_SRP, AU};

use crate::control::{normalize_clock, normalize_cone, Angles};
use optimization::LoggingStrategy;

// ==============================================================================
// Constants — imported from orbital_models::constants (single source of truth)
// ==============================================================================

const GM_SUN:   f64 = MU_SUN;   // m³/s²
const P_SRP_1AU: f64 = P_SRP;   // N/m²
const AU_M:     f64 = AU;        // m

/// Mars mean orbital speed [m/s] — used to normalize velocity error
const V_MARS_ORBITAL: f64 = 24_130.0;

/// Position error threshold [AU] beyond which a steep penalty is applied.
/// Solutions with pos_err > POS_PENALTY_AU are effectively rejected by the optimizer.
/// Must match POS_PENALTY_AU in plot_trajectory.py to draw the correct threshold circle.
const POS_PENALTY_AU: f64 = 0.1;

/// Velocity error threshold (normalized by V_MARS_ORBITAL) beyond which a steep penalty is applied.
/// 0.04 × V_MARS_ORBITAL ≈ 966 m/s ≈ 1 km/s — the rendezvous requirement.
/// (was 1.1 = 26.5 km/s, which was effectively never active)
const VEL_PENALTY_NORM: f64 = 0.04;

/// Weight for the velocity direction error term.
/// vel_dir_err = (1 − cos θ) / 2 ∈ [0, 1]:
///   0.0 = same direction as Mars, 0.5 = 90° off, 1.0 = anti-parallel (retrograde).
const VEL_DIR_WEIGHT: f64 = 0.4;

/// Direction penalty threshold.
/// (1 − cos θ) / 2 = 0.10 corresponds to θ ≈ 37°.
/// Solutions arriving more than ~37° off Mars's velocity direction incur a steep penalty.
const VEL_DIR_PENALTY_THRESHOLD: f64 = 0.10;

/// Penalty scale per unit of threshold violation.
/// At PENALTY_WEIGHT = 20, being 1 AU outside the pos threshold costs 20 extra energy units,
/// making such solutions ~never accepted by the SA algorithm.
const PENALTY_WEIGHT: f64 = 20.0;

// ==============================================================================
// TransferInput
// ==============================================================================

/// Combined control input: sail angle trajectory + time of flight + (optional) departure offset.
///
/// The type parameter `N` is the number of control points.
/// Angles are interpolated between control points during integration.
#[derive(Clone, Debug)]
pub struct TransferInput<const N: usize> {
    /// Sail pointing angles at N control points
    pub angles: [Angles; N],
    /// Time of flight [days] — a free optimization variable
    pub tof_days: f64,
    /// Departure epoch offset [days] from `TransferProblem::departure_epoch`.
    /// Only perturbed when `TransferProblem::departure_window_days > 0.0`; otherwise 0.0.
    pub departure_offset_days: f64,
}

// Manual Send + Sync: [Angles; N] is Copy + Send + Sync; f64 is Send + Sync.
unsafe impl<const N: usize> Send for TransferInput<N> {}
unsafe impl<const N: usize> Sync for TransferInput<N> {}

// ==============================================================================
// TransferProblem
// ==============================================================================

/// Earth-Mars solar sail rendezvous optimization problem.
///
/// Propagates the spacecraft from Earth's heliocentric state at `departure_epoch`
/// under Sun gravity + SRP for `tof_days` days, then measures the rendezvous error
/// with Mars's heliocentric state at arrival.
///
/// Requires `kernels/de440s.bsp` to run (via the `Almanac`).
pub struct TransferProblem<const N: usize> {
    /// Sail area [m²]
    pub sail_area: f64,
    /// Spacecraft mass [kg]
    pub mass: f64,
    /// Sail reflectivity (0 = perfect absorber, 1 = perfect reflector)
    pub reflectivity: f64,
    /// Heliocentric state of the spacecraft at departure (from Earth ephemeris)
    pub initial_state: FrameState<Heliocentric>,
    /// Departure epoch (used to compute Mars position at arrival)
    pub departure_epoch: Epoch,
    /// Minimum allowed time of flight [days]
    pub min_tof_days: f64,
    /// Maximum allowed time of flight [days]
    pub max_tof_days: f64,
    /// Weight of velocity mismatch in the energy function.
    ///
    /// `energy = pos_err_AU + vel_weight * vel_err_normalized + tof_weight * tof_normalized`
    pub vel_weight: f64,
    /// Weight of time-of-flight in the energy function.
    ///
    /// `tof_normalized = (tof_days - min_tof_days) / (max_tof_days - min_tof_days)` ∈ [0, 1].
    /// Set to 0.0 to optimize for rendezvous accuracy alone.
    pub tof_weight: f64,
    /// Half-window for departure epoch search [days].
    ///
    /// When > 0.0, the optimizer may shift the departure by ±`departure_window_days/2` days.
    /// The actual departure is `departure_epoch + input.departure_offset_days`.
    /// Set to 0.0 (default) to fix the departure at `departure_epoch`.
    pub departure_window_days: f64,
    /// Shared almanac for Mars ephemeris queries (requires kernel file)
    pub almanac: Arc<Almanac>,
    /// How much trajectory data to write to the database
    pub logging_strategy: LoggingStrategy,
}

// ==============================================================================
// Physics helpers (private)
// ==============================================================================

/// Gravitational acceleration from the Sun [m/s²].
fn sun_gravity(pos: &SVector<f64, 3>) -> SVector<f64, 3> {
    let r = pos.norm();
    -(GM_SUN / (r * r)) * (pos / r)
}

/// Solar radiation pressure acceleration [m/s²] in heliocentric frame.
///
/// Uses `HeliocentricSunFrame` from `orbital_models` for the sail normal,
/// which is the cone-clock parameterisation around the Sun→SC direction (`s_hat = pos / |pos|`):
/// - cone = 0: sail directly faces the sun, maximum thrust away from sun
/// - cone = π/2: sail edge-on, zero thrust
///
/// Pressure scales as r⁻² from the sun (the `SolarPressureModel` in `orbital_models`
/// uses a fixed 1-AU pressure; here we apply the correct distance-dependent scaling).
fn srp_acceleration(
    pos: &SVector<f64, 3>,
    vel: &SVector<f64, 3>,
    sail_area: f64,
    mass: f64,
    reflectivity: f64,
    cone: f64,
    clock: f64,
) -> SVector<f64, 3> {
    let r = pos.norm();
    // Pressure scales as (AU/r)² from the sun
    let pressure = P_SRP_1AU * (AU_M / r) * (AU_M / r);

    // Sun is at the origin in heliocentric frame — pass SVector::zeros() as sun position.
    // cone=0 → n_hat = s_hat = pos/|pos| (away from sun), maximum thrust.
    let helio_sun = SVector::<f64, 3>::zeros();
    let n_hat = SunPointingFrame::sail_normal_inertial(cone, clock, pos, vel, &helio_sun);

    // s_hat: from Sun (origin) toward spacecraft
    let s_hat = pos / r;
    // Only the sun-facing side of the sail produces thrust
    let cos_alpha = n_hat.dot(&s_hat).max(0.0);

    let force_mag = pressure * sail_area * (1.0 + reflectivity) * cos_alpha * cos_alpha;
    force_mag * n_hat / mass
}

// ==============================================================================
// TransferProblem::evaluate
// ==============================================================================

/// Final heliocentric state after integration
struct FinalState {
    pos: SVector<f64, 3>,
    vel: SVector<f64, 3>,
    /// State at the point of minimum distance to the linearly-interpolated Mars
    /// position during the trajectory.  Only populated when `mars_interp` is
    /// supplied to `evaluate()`.  Tuple: (time_secs, pos, vel).
    /// Filtered to be at least 1 day before the nominal end so we do not
    /// duplicate the final-state energy computation.
    closest_approach: Option<(f64, SVector<f64, 3>, SVector<f64, 3>)>,
}

impl<const N: usize> TransferProblem<N> {
    /// Propagate the spacecraft trajectory and return the final heliocentric state.
    ///
    /// `mars_interp`: if `Some((dep_pos, arr_pos))`, the integration loop tracks
    /// the state of minimum distance to the linearly-interpolated Mars position
    /// and stores it in `FinalState::closest_approach`.  No almanac calls are
    /// made inside this function; linear interpolation is used as a cheap proxy.
    fn evaluate(
        &self,
        input: &TransferInput<N>,
        context: &EvaluationContext,
        init_pos: SVector<f64, 3>,
        init_vel: SVector<f64, 3>,
        mars_interp: Option<(SVector<f64, 3>, SVector<f64, 3>)>,
    ) -> FinalState {
        #[derive(Default, Clone, Debug)]
        struct StepData;

        let tof_secs = input.tof_days * 86_400.0;
        let sail_area = self.sail_area;
        let mass = self.mass;
        let reflectivity = self.reflectivity;
        let logging = self.logging_strategy;

        // Build the initial 6D state from the provided heliocentric pos/vel
        let p = init_pos;
        let v = init_vel;
        let initial_arr: [f64; 6] = [p[0], p[1], p[2], v[0], v[1], v[2]];
        let initial_vec = Vector::<f64, 6>::from(initial_arr);

        let interpolation_strategy = context.optimization_context.interpolation_strategy;
        let angles_ref = &input.angles;

        let compute_angles = |time: f64| -> Angles {
            let progress = time / tof_secs * (N as f64 - 1.0);
            let idx = (progress.floor() as usize).min(N - 2);
            let t = progress - idx as f64;
            interpolation_strategy.interpolate(&angles_ref[idx], &angles_ref[idx + 1], t)
        };

        let integration_function = |time: f64, state: Vector<f64, 6>| {
            let pos = SVector::<f64, 3>::new(state.0[0], state.0[1], state.0[2]);
            let vel = SVector::<f64, 3>::new(state.0[3], state.0[4], state.0[5]);

            let angles = compute_angles(time);
            let grav = sun_gravity(&pos);
            let srp = srp_acceleration(&pos, &vel, sail_area, mass, reflectivity, angles.cone, angles.clock);
            let accel = grav + srp;

            let deriv = Vector::<f64, 6>::from([
                vel[0], vel[1], vel[2],
                accel[0], accel[1], accel[2],
            ]);
            (StepData, deriv)
        };

        let ds: f64 = 1e-3;
        let integrator = numerical_integration::RK_FELBERG;

        let mut state0 = integrator.adaptive_init(
            0.0_f64,
            initial_vec,
            ds,
            integration_function,
            InnerProductMetric,
        );
        let states = state0.as_mut();

        let connection = context.optimization_context.connection.as_ref().unwrap();
        let mut data_to_log: Vec<(chrono::DateTime<chrono::Utc>, Vec<(f64, StepData, Vector<f64, 6>)>)> = vec![];
        let mut angle_history: Vec<Angles> = vec![];
        let mut step_idx = 0usize;

        // Closest-approach tracking (only when caller provides Mars interpolation data)
        let mut min_mars_dist = f64::INFINITY;
        let mut closest_approach: Option<(f64, SVector<f64, 3>, SVector<f64, 3>)> = None;

        loop {
            let last = states.first().unwrap();
            let last_time = last.0;

            if last_time >= tof_secs {
                break;
            }

            // Track state closest to linearly-interpolated Mars position.
            // Linear interpolation is cheap (no almanac) and accurate enough to
            // identify the approximate closest-approach time.
            if let Some((mars_dep, mars_arr)) = mars_interp {
                let t_frac = last_time / tof_secs;
                let mars_pos_interp = mars_dep + t_frac * (mars_arr - mars_dep);
                let sc_pos = SVector::<f64, 3>::new(last.2.0[0], last.2.0[1], last.2.0[2]);
                let dist = (sc_pos - mars_pos_interp).norm();
                if dist < min_mars_dist {
                    min_mars_dist = dist;
                    let sc_vel = SVector::<f64, 3>::new(last.2.0[3], last.2.0[4], last.2.0[5]);
                    closest_approach = Some((last_time, sc_pos, sc_vel));
                }
            }

            if !logging.only_final_state() && logging.should_log_timestep(step_idx) {
                data_to_log.push((chrono::Utc::now(), states.to_owned()));
                angle_history.push(compute_angles(last_time));
            }

            step_idx += 1;
            integrator.adaptive_step(states, ds, integration_function, InnerProductMetric);
        }

        // Log final state (arrival point)
        if logging.only_final_state() || matches!(logging, LoggingStrategy::AllDownsampled(_)) {
            data_to_log.push((chrono::Utc::now(), states.to_owned()));
            let last_time = states.first().unwrap().0;
            angle_history.push(compute_angles(last_time.min(tof_secs)));
        }

        // Write to database
        let mut conn_lock = connection.lock().unwrap();
        let transaction = Transaction::new(&mut conn_lock, rusqlite::TransactionBehavior::Deferred).unwrap();

        let mut q = transaction
            .prepare("CREATE TABLE IF NOT EXISTS timesteps (id INTEGER PRIMARY KEY, eval_id TEXT, datetime DATETIME, simtime REAL, x REAL, y REAL, z REAL, vx REAL, vy REAL, vz REAL, cone REAL, clock REAL)")
            .unwrap();
        q.execute([]).unwrap();
        drop(q);

        let mut q = transaction
            .prepare("INSERT INTO timesteps (eval_id, datetime, simtime, x, y, z, vx, vy, vz, cone, clock) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
            .unwrap();
        data_to_log.iter().enumerate().for_each(|(i, (dt, s))| {
            let a = angle_history[i];
            q.insert((
                context.eval_id.to_string(),
                dt.to_string(),
                s[0].0,
                s[0].2.0[0], s[0].2.0[1], s[0].2.0[2],
                s[0].2.0[3], s[0].2.0[4], s[0].2.0[5],
                a.cone, a.clock,
            )).unwrap();
        });
        drop(q);
        transaction.commit().unwrap();
        drop(conn_lock);

        let last = states.first().unwrap();
        // Discard closest_approach if it's within 1 day of the nominal end
        // (would just duplicate the final-state energy computation).
        let closest_approach = closest_approach
            .filter(|(ca_t, _, _)| tof_secs - ca_t > 86_400.0);
        FinalState {
            pos: SVector::<f64, 3>::new(last.2.0[0], last.2.0[1], last.2.0[2]),
            vel: SVector::<f64, 3>::new(last.2.0[3], last.2.0[4], last.2.0[5]),
            closest_approach,
        }
    }

    // ==============================================================================
    // Local refinement
    // ==============================================================================

    /// Coordinate-wise hill-climbing refinement starting from the SA best solution.
    ///
    /// Cycles through each control point's cone/clock angles and the TOF,
    /// trying ± step perturbations with progressively finer step sizes.
    /// Only strict improvements are accepted (temperature = 0).
    ///
    /// This fills in detail that SA misses at the end of cooling:
    /// the perturbation scale is coarse relative to the optimal basin.
    pub fn refine(
        &self,
        initial: TransferInput<N>,
        context: &optimization::OptimizationContext,
    ) -> TransferInput<N> {
        use optimization::EvaluationContext;

        let eval_ctx = EvaluationContext::new_in_run(context);
        let mut best = initial;
        let mut best_energy = self.energy(&best, &eval_ctx);
        println!("Refinement start energy: {:.6}", best_energy);

        // Paired step sizes: angles [rad] and TOF [days] shrink together.
        let angle_steps: &[f64] = &[0.10, 0.05, 0.02, 0.01, 0.005, 0.001];
        let tof_steps:   &[f64] = &[5.0,  2.0,  1.0,  0.5,  0.2,   0.1  ];
        // Departure offset steps [days] — only used when departure_window_days > 0
        let dep_steps:   &[f64] = &[10.0, 5.0,  2.0,  1.0,  0.5,   0.2  ];
        const MAX_PASSES: usize = 10;

        for round in 0..angle_steps.len() {
            let a_step = angle_steps[round];
            let t_step = tof_steps[round];
            let d_step = dep_steps[round];
            let mut improved = true;
            let mut pass = 0;

            while improved && pass < MAX_PASSES {
                improved = false;
                pass += 1;

                // ── Angle perturbations ──────────────────────────────────────
                for i in 0..N {
                    for (field, name) in [(0usize, "cone"), (1usize, "clock")] {
                        for &dir in &[a_step, -a_step] {
                            let mut candidate = best.clone();
                            match field {
                                0 => candidate.angles[i].cone =
                                        normalize_cone(candidate.angles[i].cone + dir),
                                _ => candidate.angles[i].clock =
                                        normalize_clock(candidate.angles[i].clock + dir),
                            }
                            let eval_ctx = EvaluationContext::new_in_run(context);
                            let energy = self.energy(&candidate, &eval_ctx);
                            if energy < best_energy {
                                println!(
                                    "  r{round} p{pass}: CP[{i:2}] {name} {:+.4} rad → {energy:.6}",
                                    dir
                                );
                                best = candidate;
                                best_energy = energy;
                                improved = true;
                            }
                        }
                    }
                }

                // ── TOF perturbation ─────────────────────────────────────────
                for &dir in &[t_step, -t_step] {
                    let mut candidate = best.clone();
                    candidate.tof_days =
                        (candidate.tof_days + dir).clamp(self.min_tof_days, self.max_tof_days);
                    let eval_ctx = EvaluationContext::new_in_run(context);
                    let energy = self.energy(&candidate, &eval_ctx);
                    if energy < best_energy {
                        println!(
                            "  r{round} p{pass}: TOF {:+.2} d → {energy:.6}",
                            dir
                        );
                        best = candidate;
                        best_energy = energy;
                        improved = true;
                    }
                }

                // ── Departure offset perturbation (only when window is open) ─
                if self.departure_window_days > 0.0 {
                    let half = self.departure_window_days / 2.0;
                    for &dir in &[d_step, -d_step] {
                        let mut candidate = best.clone();
                        candidate.departure_offset_days =
                            (candidate.departure_offset_days + dir).clamp(-half, half);
                        let eval_ctx = EvaluationContext::new_in_run(context);
                        let energy = self.energy(&candidate, &eval_ctx);
                        if energy < best_energy {
                            println!(
                                "  r{round} p{pass}: dep_offset {:+.2} d → {energy:.6}",
                                dir
                            );
                            best = candidate;
                            best_energy = energy;
                            improved = true;
                        }
                    }
                }
            }
            println!(
                "  Round {round} done (angle±{a_step:.3} rad, TOF±{t_step:.1} d): \
                 {pass} passes, energy={best_energy:.6}"
            );
        }

        println!("Refinement done. Final energy: {:.6}", best_energy);
        best
    }
}

// ==============================================================================
// OptimizableProblem impl
// ==============================================================================

impl<const N: usize> OptimizableProblem for TransferProblem<N> {
    type Input = TransferInput<N>;
    type Energy = f64;

    fn neighbour(
        &self,
        current: &TransferInput<N>,
        temperature: f64,
        context: &EvaluationContext,
    ) -> TransferInput<N> {
        let (_, perturbation_scale) = context.optimization_context.sa_config();
        let std_dev = perturbation_scale * temperature;
        let tof_range = self.max_tof_days - self.min_tof_days;
        let tof_std = tof_range * perturbation_scale * temperature / std::f64::consts::PI;

        let mut rng = rand::thread_rng();
        use rand_distr::Distribution;

        let mut angles = current.angles;
        for i in 0..N {
            let g = rand_distr::Normal::new(current.angles[i].cone, std_dev).unwrap();
            angles[i].cone = normalize_cone(g.sample(&mut rng));
            let g = rand_distr::Normal::new(current.angles[i].clock, std_dev).unwrap();
            angles[i].clock = normalize_clock(g.sample(&mut rng));
        }

        let g = rand_distr::Normal::new(current.tof_days, tof_std).unwrap();
        let tof_days = g.sample(&mut rng).clamp(self.min_tof_days, self.max_tof_days);

        let departure_offset_days = if self.departure_window_days > 0.0 {
            let half = self.departure_window_days / 2.0;
            let dep_std = half * perturbation_scale * temperature / std::f64::consts::PI;
            let g = rand_distr::Normal::new(current.departure_offset_days, dep_std).unwrap();
            g.sample(&mut rng).clamp(-half, half)
        } else {
            0.0
        };

        TransferInput { angles, tof_days, departure_offset_days }
    }

    fn acceptance(
        &self,
        current: &f64,
        new: &f64,
        temperature: f64,
        _context: &EvaluationContext,
    ) -> f64 {
        if new <= current {
            1.0
        } else {
            let delta_e = new - current;
            let scale = 0.5;
            let exponent = -delta_e / (scale * temperature.max(0.001));
            exponent.exp().min(1.0)
        }
    }

    fn energy(&self, input: &TransferInput<N>, context: &EvaluationContext) -> f64 {
        // Actual departure: base epoch + optional offset (0.0 when fixed)
        let actual_departure = self.departure_epoch
            + Duration::from_days(input.departure_offset_days);
        let arrival_epoch = actual_departure + Duration::from_days(input.tof_days);

        // Initial spacecraft state: use precomputed initial_state when epoch is fixed,
        // otherwise query almanac for Earth at the actual (shifted) departure.
        let (init_pos, init_vel) = if self.departure_window_days > 0.0
            && input.departure_offset_days != 0.0
        {
            let s = self.almanac
                .body_state_heliocentric(Body::Earth, actual_departure)
                .expect("Earth ephemeris at actual departure");
            (s.position.inner, s.velocity.inner)
        } else {
            (self.initial_state.position.inner, self.initial_state.velocity.inner)
        };

        // Query Mars at both ends so we can linearly interpolate inside evaluate()
        // (no almanac calls inside the integration loop — just cheap linear math).
        let mars_dep = self.almanac
            .body_state_heliocentric(Body::Mars, actual_departure)
            .expect("Mars ephemeris at departure");
        let mars_arr = self.almanac
            .body_state_heliocentric(Body::Mars, arrival_epoch)
            .expect("Mars ephemeris at arrival");

        let final_state = self.evaluate(
            input, context,
            init_pos, init_vel,
            Some((mars_dep.position.inner, mars_arr.position.inner)),
        );

        // Helper closure: given a spacecraft (pos, vel) and the Mars state at
        // that same instant, compute the scalar energy using the same formula.
        let compute_energy = |sc_pos: SVector<f64, 3>,
                              sc_vel: SVector<f64, 3>,
                              m_pos:  SVector<f64, 3>,
                              m_vel:  SVector<f64, 3>,
                              tof_days: f64| -> f64 {
            let pos_err = (sc_pos - m_pos).norm();
            let vel_err = (sc_vel - m_vel).norm();

            let v_sc_mag  = sc_vel.norm();
            let v_m_mag   = m_vel.norm();
            let vel_dir_err = if v_sc_mag > 1.0 && v_m_mag > 1.0 {
                let cos_theta = (sc_vel.dot(&m_vel) / (v_sc_mag * v_m_mag)).clamp(-1.0, 1.0);
                (1.0 - cos_theta) / 2.0
            } else {
                0.5
            };

            let pos_err_norm = pos_err / AU_M;
            let vel_err_norm = vel_err / V_MARS_ORBITAL;
            let tof_norm = (tof_days - self.min_tof_days)
                / (self.max_tof_days - self.min_tof_days);

            let pos_penalty = (pos_err_norm - POS_PENALTY_AU).max(0.0) * PENALTY_WEIGHT;
            let vel_penalty = (vel_err_norm - VEL_PENALTY_NORM).max(0.0) * PENALTY_WEIGHT;
            let dir_penalty = (vel_dir_err - VEL_DIR_PENALTY_THRESHOLD).max(0.0) * PENALTY_WEIGHT;

            pos_err_norm
                + self.vel_weight * vel_err_norm
                + VEL_DIR_WEIGHT * vel_dir_err
                + self.tof_weight * tof_norm
                + pos_penalty + vel_penalty + dir_penalty
        };

        // Energy at the nominal endpoint (tof_days)
        let energy_final = compute_energy(
            final_state.pos, final_state.vel,
            mars_arr.position.inner, mars_arr.velocity.inner,
            input.tof_days,
        );

        // Energy at closest approach (if it occurred well before the endpoint).
        // We query the almanac once at that exact time for the true Mars state.
        let energy_ca = final_state.closest_approach.and_then(|(ca_secs, ca_pos, ca_vel)| {
            let ca_tof_days = ca_secs / 86_400.0;
            let ca_epoch = actual_departure + Duration::from_days(ca_tof_days);
            self.almanac
                .body_state_heliocentric(Body::Mars, ca_epoch)
                .ok()
                .map(|ms| {
                    compute_energy(
                        ca_pos, ca_vel,
                        ms.position.inner, ms.velocity.inner,
                        ca_tof_days,
                    )
                })
        });

        let energy = match energy_ca {
            Some(e_ca) if e_ca < energy_final => {
                e_ca
            }
            _ => {
                let pos_err = (final_state.pos - mars_arr.position.inner).norm();
                let vel_err = (final_state.vel - mars_arr.velocity.inner).norm();
                let vel_dir_err = {
                    let v_sc = final_state.vel.norm();
                    let v_m  = mars_arr.velocity.inner.norm();
                    if v_sc > 1.0 && v_m > 1.0 {
                        let cos_t = (final_state.vel.dot(&mars_arr.velocity.inner)
                            / (v_sc * v_m)).clamp(-1.0, 1.0);
                        (1.0 - cos_t) / 2.0
                    } else { 0.5 }
                };
                energy_final
            }
        };

        // Log the best energy components for post-processing plots.
        if let Some(conn) = context.optimization_context.connection.as_ref() {
            let conn_lock = conn.lock().unwrap();
            let _ = conn_lock.execute(
                "CREATE TABLE IF NOT EXISTS energy_components \
                 (id INTEGER PRIMARY KEY, eval_id TEXT, \
                  tof_days REAL, pos_err_au REAL, vel_err_kms REAL, vel_dir_err REAL)",
                [],
            );
            // Determine which epoch "won" for logging
            let (log_tof, log_pos, log_vel) =
                if matches!(energy_ca, Some(e) if e < energy_final) {
                    let (ca_secs, ca_pos, ca_vel) = final_state.closest_approach.unwrap();
                    let ca_tof = ca_secs / 86_400.0;
                    let ca_ep  = actual_departure + Duration::from_days(ca_tof);
                    if let Ok(ms) = self.almanac.body_state_heliocentric(Body::Mars, ca_ep) {
                        let pe = (ca_pos - ms.position.inner).norm() / AU_M;
                        let ve = (ca_vel - ms.velocity.inner).norm() / 1000.0;
                        (ca_tof, pe, ve)
                    } else {
                        (input.tof_days,
                         (final_state.pos - mars_arr.position.inner).norm() / AU_M,
                         (final_state.vel - mars_arr.velocity.inner).norm() / 1000.0)
                    }
                } else {
                    (input.tof_days,
                     (final_state.pos - mars_arr.position.inner).norm() / AU_M,
                     (final_state.vel - mars_arr.velocity.inner).norm() / 1000.0)
                };
            let _ = conn_lock.execute(
                "INSERT INTO energy_components \
                 (eval_id, tof_days, pos_err_au, vel_err_kms, vel_dir_err) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    context.eval_id.to_string(),
                    log_tof,
                    log_pos,
                    log_vel,
                    energy,
                ],
            );
        }

        energy
    }
}

// ==============================================================================
// GAProblem impl
// ==============================================================================

impl<const N: usize> GAProblem for TransferProblem<N> {
    type Individual = TransferInput<N>;
    type Fitness = f64;

    fn fitness(&self, individual: &TransferInput<N>, context: &EvaluationContext) -> f64 {
        self.energy(individual, context)
    }

    fn random_individual(&self) -> TransferInput<N> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let departure_offset_days = if self.departure_window_days > 0.0 {
            let half = self.departure_window_days / 2.0;
            rng.gen_range(-half..=half)
        } else {
            0.0
        };
        TransferInput {
            angles: core::array::from_fn(|_| Angles::random()),
            tof_days: rng.gen_range(self.min_tof_days..=self.max_tof_days),
            departure_offset_days,
        }
    }

    fn crossover(
        &self,
        parent1: &TransferInput<N>,
        parent2: &TransferInput<N>,
        _context: &EvaluationContext,
    ) -> TransferInput<N> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        // Two-point crossover on angle control points
        let mut pt1 = rng.gen_range(0..N);
        let mut pt2 = rng.gen_range(0..N);
        if pt1 > pt2 {
            std::mem::swap(&mut pt1, &mut pt2);
        }
        let angles = core::array::from_fn(|i| {
            if i >= pt1 && i < pt2 { parent2.angles[i] } else { parent1.angles[i] }
        });
        // Randomly inherit TOF and departure offset from one parent
        let tof_days = if rng.gen::<bool>() { parent1.tof_days } else { parent2.tof_days };
        let departure_offset_days = if rng.gen::<bool>() {
            parent1.departure_offset_days
        } else {
            parent2.departure_offset_days
        };
        TransferInput { angles, tof_days, departure_offset_days }
    }

    fn mutate(
        &self,
        individual: &TransferInput<N>,
        mutation_rate: f64,
        _context: &EvaluationContext,
    ) -> TransferInput<N> {
        use rand::Rng;
        use rand_distr::Distribution;
        let mut rng = rand::thread_rng();
        let std_dev = std::f64::consts::PI / 16.0;

        let angles = core::array::from_fn(|i| {
            let mut a = individual.angles[i];
            if rng.gen::<f64>() < mutation_rate {
                let nc = rand_distr::Normal::new(a.cone, std_dev).unwrap();
                let nk = rand_distr::Normal::new(a.clock, std_dev).unwrap();
                a.cone = normalize_cone(nc.sample(&mut rng));
                a.clock = normalize_clock(nk.sample(&mut rng));
            }
            a
        });

        let tof_days = if rng.gen::<f64>() < mutation_rate {
            let tof_range = self.max_tof_days - self.min_tof_days;
            let nt = rand_distr::Normal::new(individual.tof_days, tof_range * 0.1).unwrap();
            nt.sample(&mut rng).clamp(self.min_tof_days, self.max_tof_days)
        } else {
            individual.tof_days
        };

        let departure_offset_days = if self.departure_window_days > 0.0
            && rng.gen::<f64>() < mutation_rate
        {
            let half = self.departure_window_days / 2.0;
            let nd = rand_distr::Normal::new(individual.departure_offset_days, half * 0.1).unwrap();
            nd.sample(&mut rng).clamp(-half, half)
        } else {
            individual.departure_offset_days
        };

        TransferInput { angles, tof_days, departure_offset_days }
    }
}

// ==============================================================================
// Tests
// ==============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_inverse_square() {
        let r1 = SVector::<f64, 3>::new(AU_M, 0.0, 0.0);
        let r2 = SVector::<f64, 3>::new(2.0 * AU_M, 0.0, 0.0);
        let a1 = sun_gravity(&r1).norm();
        let a2 = sun_gravity(&r2).norm();
        let ratio = a1 / a2;
        assert!((ratio - 4.0).abs() < 1e-10,
            "Gravity should follow inverse-square: ratio={}", ratio);
    }

    #[test]
    fn srp_zero_cone_sail_normal_away_from_sun() {
        // At cone=0, sail directly faces sun: normal must point AWAY from sun (along s_hat = r_hat)
        // so that SRP pushes the spacecraft away from the sun — the only physically realisable direction.
        let pos = SVector::<f64, 3>::new(AU_M, 0.0, 0.0);
        let vel = SVector::<f64, 3>::new(0.0, 29_780.0, 0.0);
        let n_hat = SunPointingFrame::sail_normal_inertial(0.0, 0.0, &pos, &vel, &SVector::zeros());
        let s_hat = pos / pos.norm(); // from Sun (origin) toward SC
        let dot = n_hat.dot(&s_hat);
        assert!((dot - 1.0).abs() < 1e-10,
            "At cone=0, sail normal should point away from sun (along s_hat), dot={}", dot);
    }

    #[test]
    fn srp_scales_inverse_square_with_distance() {
        // SRP at r vs 2r: should scale as (r/2r)² = 1/4
        let pos1 = SVector::<f64, 3>::new(AU_M, 0.0, 0.0);
        let pos2 = SVector::<f64, 3>::new(2.0 * AU_M, 0.0, 0.0);
        let vel = SVector::<f64, 3>::new(0.0, 29_780.0, 0.0);
        let a1 = srp_acceleration(&pos1, &vel, 100.0, 10.0, 0.9, 0.0, 0.0).norm();
        let a2 = srp_acceleration(&pos2, &vel, 100.0, 10.0, 0.9, 0.0, 0.0).norm();
        let ratio = a1 / a2;
        assert!((ratio - 4.0).abs() < 1e-6,
            "SRP should scale as inverse-square of distance: ratio={}", ratio);
    }

    #[test]
    fn srp_zero_at_edge_on() {
        // At cone = π/2, sail is edge-on: cos(α) = 0, so SRP should be near zero
        let pos = SVector::<f64, 3>::new(AU_M, 0.0, 0.0);
        let vel = SVector::<f64, 3>::new(0.0, 29_780.0, 0.0);
        let a = srp_acceleration(&pos, &vel, 100.0, 10.0, 0.9, std::f64::consts::FRAC_PI_2, 0.0).norm();
        assert!(a < 1e-15,
            "At cone=π/2 (edge-on), SRP should be zero, got a={:.3e}", a);
    }
}
