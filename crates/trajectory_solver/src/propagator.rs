//! 3DOF SOI-patched multi-body trajectory propagator for Layer 1.
//!
//! v1 scope (Phase 7d): point-mass central-body fidelity only, perturbers
//! supplied by the caller as time-varying state callbacks (no
//! user-configurable perturber list/fidelity yet — that's Phase 7k). Central
//! body is resolved by sphere-of-influence (SOI) membership — the smallest
//! SOI containing the spacecraft, falling back to the Sun once outside every
//! candidate body's SOI.
//!
//! Frame is inertial (origin-shift only) throughout, never a rotation:
//! switching central body at a crossing is just `r_new = r_old - r_body(t)`,
//! `v_new = v_old - v_body(t)` (or `+` when switching back to the reference
//! frame) using that body's own ephemeris state. See the design notes "Layer 1
//! Propagator Design — SOI-Patched Multi-Body" for the full design rationale
//! (why inertial-only, why point-mass-only perturbers, why approximate
//! step-boundary switching is acceptable here).

use nalgebra::Vector3;
use ode_solvers::dopri5::Dopri5;
use ode_solvers::{SVector as OdeVec, System};
use orbital_models::GravityModel as AccelGravityModel;

type State = OdeVec<f64, 6>;

/// One sampled point along a propagated trajectory, in whatever frame was
/// central at that instant (see [`propagate`] — points are NOT all in one
/// consistent frame; downstream consumers that need a single frame must
/// convert using the same body-state callbacks).
#[derive(Clone, Copy, Debug)]
pub struct PropagatedPoint {
    pub t_s: f64,
    pub r_m: Vector3<f64>,
    pub v_mps: Vector3<f64>,
    /// Which body was gravitationally central at this point — an index into
    /// whichever `bodies: &[PropagatorBody]` slice was passed to [`propagate`]
    /// for this call, or `None` meaning the Sun/reference frame (outside
    /// every candidate body's SOI). See [`resolve_central_body`].
    pub central_body_index: Option<usize>,
}

/// A candidate central body / third-body perturber.
///
/// `state_at(t)` gives this body's (position, velocity) relative to the
/// reference frame (almost always heliocentric) at absolute mission time
/// `t_s`. The propagator derives body-relative-to-current-central-body
/// states itself when assembling forces or switching frames.
pub struct PropagatorBody<'a> {
    pub name: &'a str,
    pub mu_m3s2: f64,
    /// Laplace SOI radius [m] around this body relative to whatever it itself
    /// orbits. `None` for bodies that are never a central-body candidate
    /// (the Sun — the outermost fallback, not itself "inside" an SOI).
    pub soi_radius_m: Option<f64>,
    pub state_at: &'a dyn Fn(f64) -> (Vector3<f64>, Vector3<f64>),
    /// Zonal-harmonic (J2/J3/J4) fidelity to apply to this body's gravity
    /// *only when it is resolved as the current central body* — has no
    /// effect while this body is acting as a third-body perturber instead.
    /// `None` means point-mass central gravity (either because the mission
    /// config requested point-mass fidelity, or because no citable pole
    /// orientation is available for this body yet — see the design notes
    /// and `body_models::TargetBody::pole_ra_deg`).
    pub central_fidelity: Option<ZonalFidelity>,
    /// Mean physical radius [m] — used only for the collision/close-approach
    /// stop (see [`COLLISION_MARGIN_M`]), never for gravity. `None` disables
    /// the check for this body (no radius known).
    pub radius_m: Option<f64>,
}

/// Safety margin added to a body's physical radius when checking for a
/// literal collision during propagation. Not a science constraint on flyby
/// altitude or orbit-insertion radius — both stay far above this — it's a
/// floor that stops the integrator before it ever has to chase the real
/// 1/r^2 singularity at a body's surface, which is what actually drives a
/// `StepSizeUnderflow` (the local error estimate not shrinking as fast as
/// the step does, near-singularity). 1 km is comfortably above any body's
/// surface roughness/elevation uncertainty without constraining any
/// legitimate trajectory.
pub const COLLISION_MARGIN_M: f64 = 1_000.0;

/// Returns the index of the first body the spacecraft is at or below
/// `radius_m + COLLISION_MARGIN_M` from, in the reference frame, if any.
/// Bodies with `radius_m = None` are never a collision candidate.
///
/// `pub` (Phase 13f) for the same reason [`resolve_central_body`] is —
/// `sim_engine`'s coupled-burn leg loop needs the identical collision check
/// `propagate()` already uses, and re-deriving it there would risk the two
/// silently drifting apart.
pub fn resolve_collision(
    sc_pos_reference_frame: &Vector3<f64>,
    bodies: &[PropagatorBody],
    t_abs_s: f64,
) -> Option<usize> {
    bodies.iter().enumerate().find_map(|(i, b)| {
        let r_m = b.radius_m?;
        let (body_pos, _) = (b.state_at)(t_abs_s);
        let dist = (sc_pos_reference_frame - body_pos).norm();
        (dist <= r_m + COLLISION_MARGIN_M).then_some(i)
    })
}

/// Zonal-harmonic gravity parameters for one central body, including the
/// pole orientation needed by [`orbital_models::GravityModel::zonal_harmonics_body_oriented`]
/// to apply J2/J3/J4 correctly when this propagator's working frame (ICRF,
/// never rotated — see module docs) isn't aligned with the body's true pole.
#[derive(Clone, Copy)]
pub struct ZonalFidelity {
    /// Reference radius for the Jn coefficients [m] (the body's R0/equatorial radius).
    pub r0_m: f64,
    pub j2: f64,
    pub j3: f64,
    pub j4: f64,
    /// Pole right ascension/declination in ICRF [rad], J2000 constant term.
    pub pole_ra_rad: f64,
    pub pole_dec_rad: f64,
}

/// Laplace sphere of influence radius [m].
///
/// Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed. —
/// patched-conic SOI approximation (also in Curtis, *Orbital Mechanics for
/// Engineering Students*). Distinct from the Hill sphere already used
/// elsewhere in this codebase for WSB/CRTBP work — similar idea, different
/// formula, different purpose; do not conflate them.
///
/// `a_m` — body's semi-major axis around whatever it orbits [m]
/// `mass_ratio` — body mass / mass of whatever it orbits
pub fn laplace_soi_radius_m(a_m: f64, mass_ratio: f64) -> f64 {
    a_m * mass_ratio.powf(2.0 / 5.0)
}

/// Resolve which body is "central" given the spacecraft's position in the
/// reference frame at time `t_abs_s`. Returns the index of the smallest SOI
/// containing the spacecraft, or `None` if outside every candidate's SOI
/// (caller falls back to the Sun/reference frame in that case).
///
/// `pub` (not just crate-internal) since `sim_engine`'s 6DOF propagator
/// (Phase 13c) needs the same SOI-membership resolution to select the
/// correct central body/frame for attitude torque sources (gravity-gradient,
/// SRP) — reusing this function instead of re-deriving the same logic keeps
/// SOI resolution as one source of truth across the 3DOF (Layer 1) and 6DOF
/// (Layer 2) propagators.
pub fn resolve_central_body(
    sc_pos_reference_frame: &Vector3<f64>,
    bodies: &[PropagatorBody],
    t_abs_s: f64,
) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, b) in bodies.iter().enumerate() {
        let Some(soi) = b.soi_radius_m else { continue };
        let (body_pos, _) = (b.state_at)(t_abs_s);
        let dist = (sc_pos_reference_frame - body_pos).norm();
        if dist < soi {
            match best {
                Some((_, best_soi)) if soi >= best_soi => {}
                _ => best = Some((i, soi)),
            }
        }
    }
    best.map(|(i, _)| i)
}

/// ODE system for one propagation leg: point-mass central-body gravity plus
/// point-mass third-body perturbers (every other candidate body), all
/// expressed relative to the current central body (`central_index`, or the
/// reference frame if `None`).
struct LegOde<'a, F: FnMut(f64, &Vector3<f64>) -> bool> {
    mu_central: f64,
    bodies: &'a [PropagatorBody<'a>],
    central_index: Option<usize>,
    /// Absolute mission time at the start of this leg — `system`/`solout`
    /// receive leg-local time `t`, so absolute time is `t0_abs_s + t`.
    t0_abs_s: f64,
    should_stop: F,
    /// Records the integrator's true last-visited `(t, state)` on *every*
    /// `solout` call, unconditionally -- independent of `OutputType::Dense`'s
    /// own sampling grid (`sample_dt_s`). Found necessary when
    /// `should_stop` triggers an abort before the integrator reaches even
    /// the *first* dense-output checkpoint (a real, physical scenario for a
    /// low-v-infinity escape, whose distance from the departure body can
    /// dip back inside its SOI determination again within the first
    /// `sample_dt_s` after initially exiting -- confirmed via a real Earth-
    /// Mars case, re-entering Earth's SOI determination ~2.8 hours after
    /// first exiting), `points` (the dense samples) can end up containing
    /// only the leg-start point, making the leg look like zero progress
    /// was made even though the integrator genuinely advanced for hours.
    /// This field is the fix: the true stop point, regardless of whether
    /// dense output happened to sample it.
    last_visited: std::rc::Rc<std::cell::Cell<(f64, State)>>,
}

impl<'a, F: FnMut(f64, &Vector3<f64>) -> bool> System<f64, State> for LegOde<'a, F> {
    fn system(&self, t: f64, y: &State, dy: &mut State) {
        let r = Vector3::new(y[0], y[1], y[2]);
        let t_abs = self.t0_abs_s + t;
        let mut a = AccelGravityModel::point_mass(&r, self.mu_central);

        if let Some(ci) = self.central_index {
            if let Some(zf) = &self.bodies[ci].central_fidelity {
                a += AccelGravityModel::zonal_harmonics_body_oriented(
                    &r, self.mu_central, zf.r0_m, zf.j2, zf.j3, zf.j4, zf.pole_ra_rad, zf.pole_dec_rad,
                );
            }
        }

        let central_pos = self.central_index.map(|ci| (self.bodies[ci].state_at)(t_abs).0);
        for (i, b) in self.bodies.iter().enumerate() {
            if Some(i) == self.central_index {
                continue;
            }
            let (body_pos_ref, _) = (b.state_at)(t_abs);
            let body_pos_relative = match central_pos {
                Some(cp) => body_pos_ref - cp,
                None => body_pos_ref,
            };
            a += AccelGravityModel::third_body(&r, &body_pos_relative, b.mu_m3s2);
        }

        dy[0] = y[3]; dy[1] = y[4]; dy[2] = y[5];
        dy[3] = a.x; dy[4] = a.y; dy[5] = a.z;
    }

    fn solout(&mut self, t: f64, y: &State, _dy: &State) -> bool {
        self.last_visited.set((t, *y));
        let r = Vector3::new(y[0], y[1], y[2]);
        (self.should_stop)(t, &r)
    }
}

/// Propagate one leg under a fixed central body, stopping early (success, not
/// error — same `solout`-abort pattern already used in `AstroProbs/Artemis`)
/// the first time `should_stop` returns `true`. Returns the sampled points
/// (leg-local time, starting at 0) and the final `(t_s leg-local, r, v)`.
#[allow(clippy::too_many_arguments)]
fn integrate_leg<'a>(
    r0: Vector3<f64>,
    v0: Vector3<f64>,
    t0_abs_s: f64,
    duration_s: f64,
    mu_central: f64,
    bodies: &'a [PropagatorBody<'a>],
    central_index: Option<usize>,
    sample_dt_s: f64,
    rtol: f64,
    atol: f64,
    should_stop: impl FnMut(f64, &Vector3<f64>) -> bool,
) -> (Vec<PropagatedPoint>, f64, Vector3<f64>, Vector3<f64>) {
    let y0 = State::from_column_slice(&[r0.x, r0.y, r0.z, v0.x, v0.y, v0.z]);
    let last_visited = std::rc::Rc::new(std::cell::Cell::new((0.0, y0)));
    let ode = LegOde { mu_central, bodies, central_index, t0_abs_s, should_stop, last_visited: last_visited.clone() };

    let duration_s = duration_s.max(1e-6);
    let sample_dt_s = sample_dt_s.max(1e-6);
    // `OutputType::Sparse`, not the `Dopri5::new` default of `Dense`. Found
    // necessary and more serious than the dense-output-drop bug
    // the top-up correction below already works around: in `ode_solvers`'s
    // own `Dopri5::integrate()`, every accepted step calls
    // `solout(self.x, self.results.get().1.last().unwrap(), ...)` --
    // `self.x` is the TRUE, just-advanced time, but for `Dense` output mode
    // `self.results.get().1.last()` is whatever was most recently *dense-
    // sampled* (only updated when an `xd`-multiple of `sample_dt_s` is
    // crossed), which can be several real accepted steps stale. So `solout`
    // -- and therefore `should_stop`'s SOI-crossing check, and this leg's
    // own `last_visited` capture -- can receive a real, advancing time
    // paired with a stale, unmoving position. Confirmed directly: a real
    // low-v-infinity escape's recorded position was bit-for-bit frozen for
    // an entire ~3068 s leg while time and Earth's own ephemeris both
    // advanced normally, producing spurious central-body-switch
    // oscillations with no basis in the actual dynamics. `Sparse` mode's
    // `solution_output` always pushes `(self.x, y_next)` immediately, so
    // `solout` always sees the true, synchronized current state.
    let mut stepper = Dopri5::from_param(
        ode, 0.0, duration_s, sample_dt_s, y0, rtol, atol,
        0.9, 0.04, 0.2, 10.0, duration_s, 0.0, 100_000, 1000,
        ode_solvers::dop_shared::OutputType::Sparse,
    );
    let integrate_result = stepper.integrate();
    if let Err(e) = &integrate_result {
        eprintln!(
            "Warning: propagator leg did not complete cleanly (requested duration {duration_s:.3e} s, \
             mu_central {mu_central:.3e} m^3/s^2): {e:?} — trajectory may be incomplete or inaccurate."
        );
    }

    let times = stepper.x_out();
    let states = stepper.y_out();

    let mut points = Vec::with_capacity(times.len());
    for (t, y) in times.iter().zip(states.iter()) {
        points.push(PropagatedPoint {
            t_s: *t,
            r_m: Vector3::new(y[0], y[1], y[2]),
            v_mps: Vector3::new(y[3], y[4], y[5]),
            central_body_index: central_index,
        });
    }

    // `ode_solvers`'s dense-output sampling (`Dopri5::solution_output`)
    // accumulates `xd += dx` across every accepted step; after enough
    // additions this can overshoot `duration_s` by a sub-ULP margin, which
    // fails the `xd <= x_end` comparison and silently drops the very last
    // sample — even though the integrator itself successfully reached
    // `duration_s` internally. Left unpatched, every propagated arc in this
    // codebase ends up to one `sample_dt_s` short of its true endpoint
    // (confirmed against a real Mars-flyby arc, which ended ~6 days short of
    // its 292-day TOF). Only treat a *small* shortfall (≤ ~1.5 sample
    // intervals) this way — a genuine early stop via `should_stop` (e.g. an
    // SOI crossing) leaves a much larger gap and must NOT be patched over.
    if integrate_result.is_ok() {
        if let Some(last) = points.last().copied() {
            let gap = duration_s - last.t_s;
            if gap > 1e-6 && gap <= 1.5 * sample_dt_s {
                let topup_y0 = State::from_column_slice(&[
                    last.r_m.x, last.r_m.y, last.r_m.z, last.v_mps.x, last.v_mps.y, last.v_mps.z,
                ]);
                let topup_ode = LegOde {
                    mu_central,
                    bodies,
                    central_index,
                    t0_abs_s: t0_abs_s + last.t_s,
                    should_stop: |_t: f64, _r: &Vector3<f64>| false,
                    last_visited: std::rc::Rc::new(std::cell::Cell::new((0.0, topup_y0))),
                };
                // `OutputType::Sparse`, not `Dense`: Sparse records every
                // accepted step directly (no `xd += dx` accumulator), so it
                // can't suffer the same dense-output drop this top-up exists
                // to fix — confirmed needed: with `Dense` here, the top-up's
                // own last sample was itself occasionally dropped the same
                // way, silently no-op'ing the whole fix.
                let mut topup = Dopri5::from_param(
                    topup_ode, 0.0, gap, gap, topup_y0, rtol, atol,
                    0.9, 0.04, 0.2, 10.0, gap, 0.0, 100_000, 1000,
                    ode_solvers::dop_shared::OutputType::Sparse,
                );
                if topup.integrate().is_ok() {
                    if let (Some(t), Some(y)) = (topup.x_out().last(), topup.y_out().last()) {
                        points.push(PropagatedPoint {
                            t_s: last.t_s + *t,
                            r_m: Vector3::new(y[0], y[1], y[2]),
                            v_mps: Vector3::new(y[3], y[4], y[5]),
                            central_body_index: central_index,
                        });
                    }
                }
            }
        }
    }

    let dense_last = points.last().copied();

    // The integrator's true last-visited point (every `solout` call,
    // unconditionally -- see `LegOde::last_visited`'s doc comment) vs.
    // whatever `points` (the dense-output samples) happened to capture. When
    // `should_stop` aborts before the first dense-output checkpoint -- a
    // real scenario, not just numerical noise (confirmed: a
    // low-v-infinity escape can dip back inside the departure body's SOI
    // determination again within the first `sample_dt_s` after first
    // exiting) -- `dense_last` would otherwise silently report zero leg
    // progress even though the integrator genuinely advanced for hours.
    // Use the solout-recorded point whenever it's strictly past whatever
    // dense output captured, and append it to `points` so callers see the
    // trajectory's real endpoint too, not just a bookkeeping-only value
    // with no matching sample.
    let (t_solout, y_solout) = last_visited.get();
    let last = match dense_last {
        Some(dl) if dl.t_s >= t_solout - 1e-6 => dl,
        _ => {
            let solout_point = PropagatedPoint {
                t_s: t_solout,
                r_m: Vector3::new(y_solout[0], y_solout[1], y_solout[2]),
                v_mps: Vector3::new(y_solout[3], y_solout[4], y_solout[5]),
                central_body_index: central_index,
            };
            points.push(solout_point);
            solout_point
        }
    };
    let last = if points.is_empty() {
        PropagatedPoint { t_s: 0.0, r_m: r0, v_mps: v0, central_body_index: central_index }
    } else {
        last
    };
    (points, last.t_s, last.r_m, last.v_mps)
}

/// Propagate `(r0, v0)`, given in the reference frame (heliocentric), from
/// absolute epoch `t0_abs_s` for `duration_s`, switching central body
/// whenever the spacecraft enters/exits a candidate body's SOI. `bodies`
/// lists every SOI-candidate and/or third-body perturber; `reference_mu_m3s2`
/// is the always-available fallback central body (the Sun) used whenever the
/// spacecraft is outside every candidate's SOI.
///
/// Switching is checked at each accepted integrator step (approximate, not
/// exact event-refinement — acceptable per Phase 1's "fast over precise"
/// design goal). Returned points are all converted back into the reference
/// frame, so callers get one consistent frame regardless of how many central-
/// body switches happened internally.
#[allow(clippy::too_many_arguments)]
pub fn propagate(
    r0: Vector3<f64>,
    v0: Vector3<f64>,
    t0_abs_s: f64,
    duration_s: f64,
    reference_mu_m3s2: f64,
    bodies: &[PropagatorBody],
    sample_dt_s: f64,
    rtol: f64,
    atol: f64,
) -> Vec<PropagatedPoint> {
    let mut all_points = Vec::new();
    let mut t_abs = t0_abs_s;
    let mut t_remaining = duration_s;
    // r, v always start each leg in the REFERENCE frame; immediately converted
    // into the resolved central body's frame (if any) before integrating.
    let mut r_ref = r0;
    let mut v_ref = v0;
    // Tracks whether the *previous* leg already used its one-time nudge
    // (below) — re-armed after every genuine (non-degenerate) leg, so each
    // real SOI crossing gets its own retry rather than sharing a single
    // global allowance across the whole propagation.
    let mut nudged_after_last_degenerate_leg = false;

    // Safety cap: never more legs than would be needed to cross every body's
    // SOI boundary twice — guards against a pathological oscillating case
    // looping forever instead of erroring loudly. Raised to a 200-leg floor
    // `bodies.len*4+4` (e.g. 12 for a 2-body Earth+Mars case)
    // is nowhere near enough for a real, low-v-infinity (marginal) escape,
    // which can genuinely oscillate across the departure body's SOI
    // boundary dozens of times (confirmed empirically: 64 transitions, ~16
    // days, for a real Earth->Mars departure burn) before separating
    // cleanly -- the *correct* physical behavior for a near-threshold
    // escape, not a bug. The old cap was silently truncating these real
    // trajectories at whichever leg count it hit first.
    let max_legs = (bodies.len() * 4 + 4).max(200);

    for _ in 0..max_legs {
        if t_remaining <= 1e-6 {
            break;
        }

        let central_index = resolve_central_body(&r_ref, bodies, t_abs);
        let mu_central = match central_index {
            Some(i) => bodies[i].mu_m3s2,
            None => reference_mu_m3s2,
        };
        let (central_pos0, central_vel0) = match central_index {
            Some(i) => (bodies[i].state_at)(t_abs),
            None => (Vector3::zeros(), Vector3::zeros()),
        };
        let r_local0 = r_ref - central_pos0;
        let v_local0 = v_ref - central_vel0;

        // Stop this leg the first time SOI membership changes from what it
        // was at leg start (covers both "left the current body's SOI" and
        // "entered a smaller nested SOI", e.g. Moon inside Earth's). Queries
        // the central body's true time-varying position at each check
        // (not a leg-start snapshot) so the membership test stays correct
        // even if the central body moves meaningfully during a long leg.
        let bodies_for_check = bodies;
        let t0_for_check = t_abs;
        let central_for_check = central_index;
        // Also stop the moment the spacecraft gets within COLLISION_MARGIN_M
        // of any body's surface — independent of the SOI-membership check
        // above, and checked at the same per-step granularity, so the
        // integrator aborts cleanly (via `solout`, same "early stop, not
        // error" path the SOI check already uses) before the adaptive step
        // controller has to chase the real 1/r^2 singularity at the
        // surface. Without this, a GA/PSO candidate that flies close to (or
        // through) a body grinds the step size down toward
        // `StepSizeUnderflow` instead of terminating immediately with a
        // well-defined closest-approach point the caller can penalize.
        let should_stop = move |t_local: f64, r_local: &Vector3<f64>| {
            let t_now = t0_for_check + t_local;
            let central_pos_now = match central_for_check {
                Some(i) => (bodies_for_check[i].state_at)(t_now).0,
                None => Vector3::zeros(),
            };
            let r_ref_now = *r_local + central_pos_now;
            let resolved = resolve_central_body(&r_ref_now, bodies_for_check, t_now);
            resolved != central_for_check || resolve_collision(&r_ref_now, bodies_for_check, t_now).is_some()
        };

        let (points, t_final_local, r_final_local, v_final_local) = integrate_leg(
            r_local0, v_local0, t_abs, t_remaining, mu_central, bodies, central_index,
            sample_dt_s, rtol, atol, should_stop,
        );

        // Convert this leg's points back into the reference frame before
        // appending, so the overall returned trajectory is one consistent frame.
        for p in &points {
            let t_abs_p = t_abs + p.t_s;
            let (cp, cv) = match central_index {
                Some(i) => (bodies[i].state_at)(t_abs_p),
                None => (Vector3::zeros(), Vector3::zeros()),
            };
            all_points.push(PropagatedPoint {
                t_s: t_abs_p,
                r_m: p.r_m + cp,
                v_mps: p.v_mps + cv,
                central_body_index: p.central_body_index,
            });
        }

        t_abs += t_final_local;
        t_remaining -= t_final_local;
        // Convert the leg-final state back to the reference frame using the
        // central body's position at the new t_abs (not the leg-start one).
        let (cp_now, cv_now) = match central_index {
            Some(i) => (bodies[i].state_at)(t_abs),
            None => (Vector3::zeros(), Vector3::zeros()),
        };
        r_ref = r_final_local + cp_now;
        v_ref = v_final_local + cv_now;

        // A collision stop is terminal, not a central-body switch — unlike
        // an SOI-membership change, there's no sensible "next leg" to start
        // from inside/at a body's surface. Stop the whole propagation here;
        // the caller gets a well-defined closest-approach final point
        // instead of either an integrator error or a meaningless leg
        // restarted from inside the body.
        if resolve_collision(&r_ref, bodies, t_abs).is_some() {
            break;
        }

        // No further progress possible (e.g. zero-length leg) — avoid an
        // infinite loop on a degenerate input. Found this fired
        // unconditionally on the *first* such leg, which silently truncated
        // real, physically valid low-v-infinity escapes to just a few days
        // instead of the full requested duration -- a low-v-infinity
        // hyperbola approaches the SOI boundary slowly/asymptotically (vs. a
        // high-energy escape blowing through it decisively), so it's
        // disproportionately likely to land the *next* leg's first
        // should_stop check right back on the noisy boundary it just
        // crossed (numerical noise, not a genuine second event). One tiny
        // forward nudge (a small fraction of `sample_dt_s`, using the
        // already-known local velocity) clears the noisy boundary region
        // and lets the real heliocentric leg actually start; only abort if
        // *that* retry is also degenerate (genuinely stuck, e.g. a true
        // zero-length input). Re-armed after every non-degenerate leg, so
        // each real SOI crossing gets its own one-time retry.
        if t_final_local <= 1e-9 && points.len() <= 1 {
            if !nudged_after_last_degenerate_leg {
                let nudge_s = (sample_dt_s * 1e-3).min(t_remaining.max(0.0));
                r_ref += v_ref * nudge_s;
                t_abs += nudge_s;
                t_remaining -= nudge_s;
                nudged_after_last_degenerate_leg = true;
                continue;
            }
            break;
        }
        nudged_after_last_degenerate_leg = false;
    }

    all_points
}

/// Real escape leg + heliocentric cruise leg, propagated as one continuous,
/// physically real trajectory from a real (finite-radius) parking orbit at
/// the departure body, instead of starting the spacecraft already at v-infinity
/// exactly at the departure body's center (the degenerate `r -> 0` shortcut
/// every caller used before this).
///
/// This is the **shared, generic baseline propagation entry point for every
/// optimization-stage method** in this codebase (today's GA/PSO; the same
/// function is what MultipleShooting/MGA should call too, once implemented
/// — Phase 9d-9g) — not specific to any one caller. It takes only generic
/// orbital-mechanics inputs (body states, masses, a parking radius, a
/// requested v-infinity vector) and returns the real propagated trajectory;
/// it has no knowledge of GA/PSO, fitness functions, or TOML config.
///
/// `bodies[departure_body_index]` must be the departure body's own entry
/// (with a real `soi_radius_m` so SOI-exit can be detected) — the caller is
/// responsible for registering it, same as any other SOI candidate.
///
/// Two phases, both via the existing [`propagate`] SOI-switching machinery
/// (no new propagation mechanism — this only orchestrates two calls to it):
/// 1. **Escape**: propagate from the real injection state (constructed by
///    [`crate::departure::hyperbolic_departure_state`]) for up to
///    `max_escape_search_s`, and take the trajectory up through the first
///    point where the spacecraft is no longer resolved as inside the
///    departure body's SOI. Returns `None` if escape isn't detected within
///    that window (a genuinely infeasible candidate — e.g. too low a v-infinity
///    for the given parking orbit — not a numerical failure to paper over).
/// 2. **Cruise**: propagate onward from that exit state for `cruise_duration_s`
///    more seconds (continuing the same absolute timeline via `t0_abs_s`),
///    under the full configured force model (target body, third-bodies, ...).
///
/// The total mission duration is therefore `escape_duration_s + cruise_duration_s`
/// — escape duration is real, not assumed negligible, and the caller should
/// evaluate target-body state at the *shifted* arrival epoch
/// (`dep_jd + escape_duration_s/86400 + cruise_duration_s/86400`), not the
/// original Lambert-assumed `dep_jd + tof_days`.
/// Upper bound on how long the escape-leg search runs before giving up.
/// Generous for any realistic chemical-propulsion departure (LEO-class
/// parking orbit, any departure C3 a real mission would fly) — escape from
/// a planetary SOI at hyperbolic speed takes hours to a few days, not
/// weeks. Bounds wasted computation per GA/PSO evaluation (this function
/// runs once per candidate).
const MAX_ESCAPE_SEARCH_S: f64 = 5.0 * 86_400.0;
/// Fixed (not duration-derived) sample spacing for the escape leg — tuned
/// for realistic escape timescales (hours to days) so the leg is densely
/// enough sampled for visualization (e.g. a frontend "zoom in on
/// departure" camera stage) regardless of how long the real escape search
/// window is.
const ESCAPE_SAMPLE_DT_S: f64 = 1_800.0;

/// Real departure-leg propagation from a real (finite-radius) parking
/// orbit at the departure body, escaping until SOI exit — replacing the
/// degenerate `r -> 0` shortcut of starting already at v-infinity exactly
/// at the departure body's center.
///
/// **Important caveat for callers doing Lambert-initial-guess-driven
/// targeting** (e.g. the Phase 9 GA/PSO fitness loop): `exit_v_mps` is the
/// *real* velocity at SOI exit, not the fully-converged asymptotic
/// v-infinity `v_inf_vec_ref` requested — a hyperbola's velocity only
/// approaches v-infinity asymptotically as r -> infinity, and a body's SOI
/// radius is reached *long* before that convergence (confirmed empirically:
/// direction within 0.0004 deg needs ~30 days of propagation at typical
/// departure C3s, while SOI exit happens in ~2-3 days). Lambert transfers
/// are highly sensitive to departure velocity, so feeding this residual
/// (typically several percent of v-infinity) into a months-long heliocentric
/// leg amplifies into an arrival error of the same order as the transfer
/// distance itself — confirmed empirically while building this. **Use this
/// function's `points`/`escape_duration_s`/`dv_escape_ms` for visualization
/// and ΔV/timing bookkeeping; do NOT use `exit_v_mps` to seed a
/// Lambert-targeted cruise leg** — re-use the original idealized
/// `v_inf_vec_ref` (added to the departure body's own velocity) for that
/// instead, same as the textbook patched-conic method (idealized
/// instantaneous v-infinity patch for targeting; real propagated escape
/// only for the reasons above). [`propagate_departure_and_cruise`] is
/// provided for callers who *do* want a fully self-consistent continuous
/// trajectory (e.g. forward-validating an already-converged solution,
/// where Lambert-sensitivity doesn't apply) — not used by today's GA/PSO
/// fitness loop, for the reason above.
pub fn propagate_escape_leg(
    r_dep_body_ref: Vector3<f64>,
    v_dep_body_ref: Vector3<f64>,
    departure_body_index: usize,
    r_park_m: f64,
    v_inf_vec_ref: Vector3<f64>,
    reference_mu_m3s2: f64,
    bodies: &[PropagatorBody],
    rtol: f64,
    atol: f64,
) -> Option<EscapeLegResult> {
    let mu_body = bodies.get(departure_body_index)?.mu_m3s2;
    let dep = crate::departure::hyperbolic_departure_state(mu_body, r_park_m, v_inf_vec_ref)?;

    let r0_ref = r_dep_body_ref + dep.r0_m;
    let v0_ref = v_dep_body_ref + dep.v0_mps;

    let escape_points = propagate(
        r0_ref, v0_ref, 0.0, MAX_ESCAPE_SEARCH_S, reference_mu_m3s2, bodies, ESCAPE_SAMPLE_DT_S, rtol, atol,
    );
    let exit_idx = escape_points
        .iter()
        .position(|p| p.central_body_index != Some(departure_body_index))?;
    let exit_point = escape_points[exit_idx];

    Some(EscapeLegResult {
        points: escape_points[..=exit_idx].to_vec(),
        escape_duration_s: exit_point.t_s,
        dv_escape_ms: dep.dv_escape_ms,
        exit_r_m: exit_point.r_m,
        exit_v_mps: exit_point.v_mps,
    })
}

/// Result of [`propagate_escape_leg`].
pub struct EscapeLegResult {
    pub points: Vec<PropagatedPoint>,
    pub escape_duration_s: f64,
    /// The real onboard escape burn from the parking orbit [m/s].
    pub dv_escape_ms: f64,
    /// Real position/velocity at SOI exit — see this function's doc comment
    /// for why `exit_v_mps` should *not* be used to seed a Lambert-targeted
    /// cruise leg.
    pub exit_r_m: Vector3<f64>,
    pub exit_v_mps: Vector3<f64>,
}

/// Escape leg + heliocentric cruise leg, propagated as one continuous, fully
/// self-consistent real trajectory — the cruise leg continues from the
/// escape leg's *actual* SOI-exit state, not an idealized v-infinity patch.
/// Correct for forward-simulating/visualizing an already-chosen trajectory;
/// **not** what a Lambert-initial-guess-driven search (today's GA/PSO
/// fitness loop) should use for targeting — see [`propagate_escape_leg`]'s
/// doc comment for why, and use that function directly (paired with the
/// caller's own idealized-v-infinity cruise propagation) in that case.
#[allow(clippy::too_many_arguments)]
pub fn propagate_departure_and_cruise(
    r_dep_body_ref: Vector3<f64>,
    v_dep_body_ref: Vector3<f64>,
    departure_body_index: usize,
    r_park_m: f64,
    v_inf_vec_ref: Vector3<f64>,
    cruise_duration_s: f64,
    reference_mu_m3s2: f64,
    bodies: &[PropagatorBody],
    cruise_sample_dt_s: f64,
    rtol: f64,
    atol: f64,
) -> Option<DepartureLegResult> {
    let escape = propagate_escape_leg(
        r_dep_body_ref, v_dep_body_ref, departure_body_index, r_park_m, v_inf_vec_ref, reference_mu_m3s2,
        bodies, rtol, atol,
    )?;

    let cruise_points = propagate(
        escape.exit_r_m, escape.exit_v_mps, escape.escape_duration_s, cruise_duration_s, reference_mu_m3s2,
        bodies, cruise_sample_dt_s, rtol, atol,
    );

    let mut points = escape.points;
    points.extend(cruise_points);

    Some(DepartureLegResult {
        points,
        escape_duration_s: escape.escape_duration_s,
        dv_escape_ms: escape.dv_escape_ms,
    })
}

/// Result of [`propagate_departure_and_cruise`].
pub struct DepartureLegResult {
    /// Escape leg + cruise leg, concatenated into one continuous, reference-
    /// frame-consistent trajectory (same guarantee as [`propagate`]'s own
    /// return value).
    pub points: Vec<PropagatedPoint>,
    /// Real time spent escaping the departure body's SOI [s] — add to
    /// `cruise_duration_s` for the true total mission duration, and to
    /// `dep_jd` for the true (shifted) arrival epoch.
    pub escape_duration_s: f64,
    /// The real onboard escape burn from the parking orbit [m/s] — the
    /// caller decides whether to charge this to the spacecraft's own ΔV
    /// budget (e.g. free for an Earth departure via launch vehicle) or not.
    pub dv_escape_ms: f64,
}

/// Real state (relative to `body`) at the first point an already-propagated
/// trajectory crosses inbound through `radius_m` of `body` — e.g. for
/// detecting a real arrival/capture-burn opportunity: where would a tangential
/// insertion burn at a chosen target orbit radius actually happen, given
/// whatever real incoming trajectory the rest of the mission produced (not a
/// separately-targeted approach).
///
/// Linearly interpolates the SPACECRAFT's own state between the two
/// bracketing sample points (smooth, well-sampled, a safe local-linear
/// approximation) for a less sample-spacing-dependent crossing estimate
/// than snapping to the nearest point — but re-queries `body`'s position at
/// the precise interpolated crossing time via `state_at`, rather than
/// linearly interpolating the already-differenced relative vector the way
/// an earlier version of this function did.
///
/// **Why this distinction matters, found from a real API bug
/// report**: interpolating the DIFFERENCED `r_rel` vector implicitly also
/// linearly interpolates `body`'s own position between samples. For a
/// slow-moving body (e.g. Mars, Earth) that's an entirely negligible
/// approximation. For a fast-moving body — Mercury at ~47.9 km/s is the
/// concrete case that surfaced this — the body's real path curves
/// meaningfully between samples (propagator sampling here is on the order
/// of `max_coast_s / 500`, which can be several hours for a realistic
/// transfer), and a chord-vs-arc "sagitta" error of hundreds of km to
/// upward of ~1.5 million m (worked out directly from Mercury's real
/// orbital elements for a ~5-hour sample spacing) leaked into the reported
/// crossing distance/position — enough for a separately re-queried "real"
/// target position (e.g. an API's `target_r_arr_m`, queried fresh via the
/// exact same `state_at`-equivalent ANISE call) to disagree with this
/// function's own `r_rel_m` by more than the target body's physical
/// radius, even though nothing actually collided. Re-querying `body` fresh
/// at the interpolated crossing time removes this error at the source —
/// the SAME real ephemeris call every other consumer of a target body's
/// position already uses, at the SAME instant, so nothing can disagree
/// with it anymore.
///
/// Returns `None` if the trajectory's distance to `body` never drops below
/// `radius_m`.
pub fn find_inbound_radius_crossing(
    points: &[PropagatedPoint],
    body: &PropagatorBody,
    radius_m: f64,
) -> Option<RadiusCrossing> {
    let mut prev: Option<(f64, f64, Vector3<f64>, Vector3<f64>)> = None; // (t_s, dist, r_sc, v_sc)
    for p in points {
        let (body_r, _body_v) = (body.state_at)(p.t_s);
        let dist = (p.r_m - body_r).norm();
        if let Some((prev_t, prev_dist, prev_r_sc, prev_v_sc)) = prev {
            if prev_dist > radius_m && dist <= radius_m {
                let frac = (prev_dist - radius_m) / (prev_dist - dist);
                let t_cross = prev_t + (p.t_s - prev_t) * frac;
                let r_sc_interp = prev_r_sc + (p.r_m - prev_r_sc) * frac;
                let v_sc_interp = prev_v_sc + (p.v_mps - prev_v_sc) * frac;
                let (body_r_cross, body_v_cross) = (body.state_at)(t_cross);
                return Some(RadiusCrossing {
                    t_s: t_cross,
                    r_rel_m: r_sc_interp - body_r_cross,
                    v_rel_mps: v_sc_interp - body_v_cross,
                });
            }
        }
        prev = Some((p.t_s, dist, p.r_m, p.v_mps));
    }
    None
}

/// Result of [`find_inbound_radius_crossing`] — position/velocity relative
/// to the body being approached, at the (interpolated) moment of crossing.
pub struct RadiusCrossing {
    pub t_s: f64,
    pub r_rel_m: Vector3<f64>,
    pub v_rel_mps: Vector3<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end check of `propagate_departure_and_cruise`: a toy
    /// star+planet system (same setup as `switches_central_body_on_soi_entry`
    /// below), departing the planet on a real hyperbolic injection rather
    /// than starting already at v-infinity at the planet's center. Confirms
    /// the two phases actually concatenate correctly: escape leg starts
    /// planet-centered, ends heliocentric, escape duration is real and
    /// within the search window, and the total trajectory continues
    /// outward for the requested cruise duration on top of that.
    #[test]
    fn departure_and_cruise_concatenates_escape_and_heliocentric_legs() {
        const MU_STAR: f64 = 1.327e20;
        const MU_PLANET: f64 = 3.986e14;
        const PLANET_RADIUS_M: f64 = 6_371_000.0;
        let planet_a_m = 1.0e11;
        let mass_ratio = MU_PLANET / MU_STAR;
        let soi = laplace_soi_radius_m(planet_a_m, mass_ratio);

        let planet_pos = Vector3::new(planet_a_m, 0.0, 0.0);
        let planet_vel = Vector3::new(0.0, 29_780.0, 0.0); // Earth-like heliocentric speed
        let state_at = move |_t: f64| (planet_pos, planet_vel);

        let bodies = [PropagatorBody {
            name: "Planet",
            mu_m3s2: MU_PLANET,
            soi_radius_m: Some(soi),
            state_at: &state_at,
            central_fidelity: None,
            radius_m: Some(PLANET_RADIUS_M),
        }];

        let r_park = PLANET_RADIUS_M * 1.5;
        let v_inf_vec = Vector3::new(0.0, 3_000.0, 1_500.0); // arbitrary departure v_infinity

        let cruise_s = 30.0 * 86_400.0;
        let result = propagate_departure_and_cruise(
            planet_pos, planet_vel, 0, r_park, v_inf_vec, cruise_s, MU_STAR, &bodies, 86_400.0, 1e-10, 1e-3,
        )
        .expect("a reasonable v_infinity from a low parking orbit should escape within the search window");

        assert!(
            result.escape_duration_s > 0.0 && result.escape_duration_s < 5.0 * 86_400.0,
            "escape duration should be real and within the search window, got {}",
            result.escape_duration_s
        );

        let v_circ = (MU_PLANET / r_park).sqrt();
        let v_p = (v_inf_vec.norm_squared() + 2.0 * MU_PLANET / r_park).sqrt();
        assert!(
            (result.dv_escape_ms - (v_p - v_circ)).abs() < 1e-3,
            "escape dV should match the analytic patched-conic formula"
        );

        // First point should be planet-centered (just after injection);
        // total trajectory should later resolve heliocentric (None) once
        // past escape, then keep going for the cruise duration.
        assert_eq!(
            result.points.first().unwrap().central_body_index,
            Some(0),
            "trajectory should start inside the departure body's SOI"
        );
        let last = result.points.last().unwrap();
        assert_eq!(last.central_body_index, None, "trajectory should end heliocentric, well past escape");
        assert!(
            (last.t_s - (result.escape_duration_s + cruise_s)).abs() < 86_400.0 * 0.02,
            "total propagated duration should be escape_duration + the requested cruise duration, got t_s={} vs expected {}",
            last.t_s, result.escape_duration_s + cruise_s
        );

        // Sanity: the spacecraft should have moved well clear of the planet
        // by the end of a 30-day heliocentric cruise -- not still sitting at
        // the SOI boundary.
        let final_dist_from_planet = (last.r_m - planet_pos).norm();
        assert!(
            final_dist_from_planet > soi * 2.0,
            "after a 30-day cruise, expected to be well outside the departure body's SOI, got {final_dist_from_planet:.3e} m vs soi {soi:.3e} m"
        );
    }

    /// A circular orbit propagated for exactly one period should return to
    /// (approximately) its starting position — validates the single-central-
    /// body integration path with no perturbers.
    #[test]
    fn circular_orbit_closes_after_one_period() {
        const MU: f64 = 3.986_004_418e14; // Earth
        let r = 7_000_000.0_f64;
        let v_circ = (MU / r).sqrt();
        let r0 = Vector3::new(r, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);
        let period = 2.0 * std::f64::consts::PI * (r.powi(3) / MU).sqrt();

        let (points, _t_final, r_final, _v_final) = integrate_leg(
            r0, v0, 0.0, period, MU, &[], None, period / 50.0, 1e-10, 1e-12, |_, _| false,
        );

        assert!(points.len() > 10, "expected multiple sampled points, got {}", points.len());
        let dist = (r_final - r0).norm();
        assert!(dist < 10.0, "should close to within 10 m after one period, got {dist:.3} m");
    }

    /// Halfway through a circular orbit's period, the spacecraft should be
    /// diametrically opposite its starting point.
    #[test]
    fn circular_orbit_half_period_is_antipodal() {
        const MU: f64 = 3.986_004_418e14;
        let r = 7_000_000.0_f64;
        let v_circ = (MU / r).sqrt();
        let r0 = Vector3::new(r, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);
        let period = 2.0 * std::f64::consts::PI * (r.powi(3) / MU).sqrt();

        let (_points, _t, r_final, _v) = integrate_leg(
            r0, v0, 0.0, period / 2.0, MU, &[], None, period / 20.0, 1e-10, 1e-12, |_, _| false,
        );

        assert!((r_final.x + r).abs() < 10.0, "expected x ~ -r, got {:.3}", r_final.x);
        assert!(r_final.y.abs() < 10.0, "expected y ~ 0, got {:.3}", r_final.y);
    }

    /// Regression test for a real bug found while building Phase 7k: with a
    /// ~292-day TOF sampled at 60 dense-output points (the standard
    /// visualization density — `ARC_SAMPLE_COUNT` in `MissionPlanner`),
    /// `ode_solvers`'s dense-output accumulator (`xd += dx`, repeated 60
    /// times) overshoots `duration_s` by a sub-ULP margin and silently drops
    /// the final sample, leaving the returned arc about one 60th (~1.7%, ~5
    /// days) short of the requested duration. Confirmed against a real
    /// `mars_flyby.toml` run before the `integrate_leg` top-up fix was added
    /// (recorded arc ended at t≈286.2 days instead of the requested 292.27).
    /// This test pins the fix: the returned `t_final` must reach the
    /// requested duration, not stop short of it.
    #[test]
    fn long_heliocentric_leg_reaches_full_requested_duration() {
        const MU_SUN: f64 = 1.327_124_400_18e20;
        let r0 = Vector3::new(1.170236e11, 8.389593e10, 3.636621e10);
        let v0 = Vector3::new(-20_000.0, 25_000.0, 5_000.0);
        let tof_s = 292.27 * 86_400.0;
        let sample_dt_s = tof_s / 60.0;

        let (points, t_final, _r_final, _v_final) = integrate_leg(
            r0, v0, 0.0, tof_s, MU_SUN, &[], None, sample_dt_s, 1e-8, 1e-10, |_, _| false,
        );

        assert!(
            (t_final - tof_s).abs() < 1.0,
            "propagated arc should reach the full requested TOF, got t_final={t_final:.3} vs tof_s={tof_s:.3}"
        );
        assert!(points.len() >= 60, "expected at least the 60 requested dense-output samples, got {}", points.len());
    }

    /// A zero-perturbation Hohmann transfer, numerically propagated from the
    /// analytic solver's own departure state, should arrive at r2 (within the
    /// analytic TOF) and at the analytic apoapsis speed — confirms the new
    /// propagator agrees with the existing, independently-validated analytic
    /// `HohmannSolver` rather than just being internally self-consistent.
    #[test]
    fn matches_analytic_hohmann_solver() {
        use crate::hohmann::HohmannSolver;
        const MU: f64 = 3.986_004_418e14; // Earth
        let r1 = 6_578_000.0; // 200 km LEO
        let r2 = 42_164_000.0; // GEO

        let solver = HohmannSolver { mu_m3s2: MU, r1_m: r1, r2_m: r2 };
        let sol = solver.solve().expect("Hohmann solve should succeed");

        let r0 = Vector3::new(r1, 0.0, 0.0);
        let v0 = Vector3::new(0.0, sol.v_transfer_dep[0], 0.0);

        let (_points, t_final, r_final, v_final) =
            integrate_leg(r0, v0, 0.0, sol.tof_s, MU, &[], None, sol.tof_s / 100.0, 1e-11, 1e-13, |_, _| false);

        assert!((t_final - sol.tof_s).abs() < 1.0, "expected to reach analytic TOF, got t_final={t_final:.1} vs {:.1}", sol.tof_s);
        let r_final_norm = r_final.norm();
        assert!(
            (r_final_norm - r2).abs() / r2 < 1e-4,
            "expected arrival radius ~{r2:.0} m, got {r_final_norm:.1} m"
        );
        let v_final_norm = v_final.norm();
        assert!(
            (v_final_norm - sol.v_transfer_arr[0]).abs() / sol.v_transfer_arr[0] < 1e-4,
            "expected arrival speed ~{:.3} m/s, got {v_final_norm:.3} m/s", sol.v_transfer_arr[0]
        );
    }

    #[test]
    fn laplace_soi_matches_known_earth_value() {
        // Earth: a = 1 AU, mass ratio Earth/Sun ~ 3.003e-6 -> SOI ~ 924,000 km (Vallado)
        const AU_M: f64 = 1.495_978_707e11;
        const EARTH_SUN_MASS_RATIO: f64 = 3.003_48e-6;
        let soi_km = laplace_soi_radius_m(AU_M, EARTH_SUN_MASS_RATIO) / 1000.0;
        assert!((soi_km - 924_000.0).abs() / 924_000.0 < 0.02, "Earth SOI should be ~924,000 km, got {soi_km:.0} km");
    }

    /// `central_fidelity`'s J2 term should reproduce the standard analytic
    /// nodal regression rate (Vallado, *Fundamentals of Astrodynamics and
    /// Applications*, 4th ed., eq. 9-37):
    ///   Ω̇ = -1.5 n J2 (Re/p)² cos(i)
    /// for an inclined circular LEO orbit — confirms the
    /// `zonal_harmonics_body_oriented` wiring through `LegOde::system`
    /// actually changes the propagated dynamics, not just that it compiles.
    /// Pole is ICRF z (RA=0, Dec=90°) here, matching Earth's real pole, so
    /// this also exercises the oriented rotation at the identity case in a
    /// real (non-toy) physical scenario.
    #[test]
    fn j2_central_fidelity_matches_analytic_nodal_regression() {
        const MU: f64 = 3.986_004_418e14; // Earth
        const RE: f64 = 6_378_137.0;
        const J2: f64 = 1.082_626_68e-3; // EGM96, same as body_models::TargetBody::earth()
        let incl_rad: f64 = 51.6_f64.to_radians(); // ISS-like inclination
        let a: f64 = 6_778_000.0; // ~400 km altitude circular orbit
        let n = (MU / a.powi(3)).sqrt();
        let period = 2.0 * std::f64::consts::PI / n;

        // Circular orbit at the given inclination, ascending node at 0°,
        // argument of latitude 0° (i.e. at the ascending node at t=0).
        let r0 = Vector3::new(a, 0.0, 0.0);
        let v_circ = (MU / a).sqrt();
        let v0 = Vector3::new(0.0, v_circ * incl_rad.cos(), v_circ * incl_rad.sin());

        let earth = PropagatorBody {
            name: "Earth",
            mu_m3s2: MU,
            soi_radius_m: None,
            state_at: &|_t: f64| (Vector3::zeros(), Vector3::zeros()),
            central_fidelity: Some(ZonalFidelity {
                r0_m: RE, j2: J2, j3: 0.0, j4: 0.0, pole_ra_rad: 0.0, pole_dec_rad: std::f64::consts::FRAC_PI_2,
            }),
            radius_m: None,
        };
        let bodies = [earth];

        // rtol/atol chosen to be comfortably achievable for LEO-scale state
        // (position ~1e6-1e7 m, velocity ~1e3-1e4 m/s) — atol tighter than
        // ~1e-6 m starts asking the adaptive controller for more precision
        // than is physically meaningful at this magnitude and risks the
        // step size collapsing trying to satisfy it (see the design notes
        // notes on the `ode_solvers` dense-output/step-control caveats this
        // test run uncovered).
        let (_points, _t, r_final, v_final) = integrate_leg(
            r0, v0, 0.0, period, MU, &bodies, Some(0), period / 200.0, 1e-10, 1e-6, |_, _| false,
        );

        // Orbit-normal (specific angular momentum) direction at t=0 and after
        // one period — J2 regresses the node without changing inclination,
        // so h's azimuth (atan2(h_y, h_x)) should drift by ≈ Ω̇·period while
        // h_z (∝ cos i) stays essentially constant.
        let h0 = r0.cross(&v0);
        let h1 = r_final.cross(&v_final);
        let raan_drift_actual = h1.y.atan2(h1.x) - h0.y.atan2(h0.x);

        let p = a; // circular orbit: semi-latus rectum = a
        let raan_dot_analytic = -1.5 * n * J2 * (RE / p).powi(2) * incl_rad.cos();
        let raan_drift_analytic = raan_dot_analytic * period;

        assert!(
            (h1.z - h0.z).abs() / h0.z.abs() < 1e-6,
            "J2 should not change inclination (h_z) to first order: h0.z={:.6e} h1.z={:.6e}", h0.z, h1.z
        );
        assert!(
            (raan_drift_actual - raan_drift_analytic).abs() / raan_drift_analytic.abs() < 0.02,
            "RAAN drift over one period should match analytic nodal regression: actual={raan_drift_actual:.6e} rad, analytic={raan_drift_analytic:.6e} rad"
        );
    }

    /// `should_stop` aborting partway through should produce a final state
    /// well short of the full requested duration — confirms early-stop works
    /// without erroring (the mechanism the SOI-crossing switching loop relies on).
    #[test]
    fn solout_abort_stops_early() {
        const MU: f64 = 3.986_004_418e14;
        let r = 7_000_000.0_f64;
        let v_circ = (MU / r).sqrt();
        let r0 = Vector3::new(r, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);
        let period = 2.0 * std::f64::consts::PI * (r.powi(3) / MU).sqrt();

        let (_points, t_final, _r, _v) = integrate_leg(
            r0, v0, 0.0, period * 10.0, MU, &[], None, period / 50.0, 1e-10, 1e-12,
            move |t, _r| t > period / 4.0,
        );
        assert!(t_final < period / 2.0, "expected early stop near quarter-period, got t_final={t_final:.1}");
        assert!(t_final > 0.0, "expected nonzero progress before stopping");
    }

    /// End-to-end switching test: a toy "planet" with a deliberately huge SOI
    /// orbiting a toy "star". Start the spacecraft just outside the planet's
    /// SOI on a trajectory that enters it, and confirm `propagate` actually
    /// switches central body (not just integrates star-centered throughout)
    /// by checking the trajectory stays bounded near the planet after entry
    /// instead of flying off on an unperturbed star-centered hyperbola.
    #[test]
    fn switches_central_body_on_soi_entry() {
        const MU_STAR: f64 = 1.327e20; // Sun-like
        const MU_PLANET: f64 = 3.986e14; // Earth-like
        let planet_a_m = 1.0e11; // toy semi-major axis
        let mass_ratio = MU_PLANET / MU_STAR;
        let soi = laplace_soi_radius_m(planet_a_m, mass_ratio);

        // Stationary toy planet for simplicity (state_at ignores t).
        let planet_pos = Vector3::new(planet_a_m, 0.0, 0.0);
        let planet_vel = Vector3::zeros();
        let state_at = move |_t: f64| (planet_pos, planet_vel);

        let bodies = [PropagatorBody {
            name: "Planet",
            mu_m3s2: MU_PLANET,
            soi_radius_m: Some(soi),
            state_at: &state_at,
            central_fidelity: None,
            radius_m: None,
        }];

        // Start just outside the SOI, heading straight at the planet fast
        // enough that the approach happens well before the star's radial
        // gravity (~0.0135 m/s^2 here) can decelerate/reverse the drift —
        // at 500 m/s that reversal happens in ~37,000 s, long before the
        // ~31,000 m gap-to-SOI would close; 5000 m/s closes it in ~6,200 s,
        // safely faster than the reversal timescale (~372,000 s at this speed).
        let r0 = planet_pos - Vector3::new(soi * 1.05, 0.0, 0.0);
        let v0 = Vector3::new(5000.0, 0.0, 0.0); // heading toward +x, toward the planet

        let points = propagate(r0, v0, 0.0, 50_000.0, MU_STAR, &bodies, 500.0, 1e-9, 1e-11);
        assert!(points.len() > 5, "expected a real trajectory, got {} points", points.len());

        // After entering the SOI, distance to the planet should be (and stay)
        // well inside the SOI radius — not flying past it on an unperturbed path.
        let min_dist_to_planet = points
            .iter()
            .map(|p| (p.r_m - planet_pos).norm())
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_dist_to_planet < soi,
            "expected spacecraft to approach within the planet's SOI ({soi:.3e} m), got min dist {min_dist_to_planet:.3e} m"
        );

        // `central_body_index` should mark the Sun/reference frame (`None`)
        // before SOI entry and the planet (`Some(0)`, its index in `bodies`)
        // after — not just an unswitched position trajectory that happens to
        // numerically dip inside the SOI.
        assert_eq!(
            points[0].central_body_index, None,
            "expected to start outside every SOI (Sun-centered)"
        );
        assert_eq!(
            points.last().unwrap().central_body_index,
            Some(0),
            "expected to end inside the planet's SOI (planet-centered)"
        );
    }

    /// Focused regression test for the `central_body_index` marker itself
    /// (separate from `switches_central_body_on_soi_entry`'s geometric
    /// bounded-trajectory check above): confirms the marker actually flips
    /// from `None` to `Some(i)` at some point along the trajectory, and that
    /// every point's marker agrees with which body it was actually closest
    /// to relative to that body's own SOI — i.e. the recorded index isn't
    /// just a constant carried over from leg start, but tracks real
    /// switching behavior.
    #[test]
    fn central_body_index_reflects_soi_membership() {
        const MU_STAR: f64 = 1.327e20;
        const MU_PLANET: f64 = 3.986e14;
        let planet_a_m = 1.0e11;
        let mass_ratio = MU_PLANET / MU_STAR;
        let soi = laplace_soi_radius_m(planet_a_m, mass_ratio);

        let planet_pos = Vector3::new(planet_a_m, 0.0, 0.0);
        let planet_vel = Vector3::zeros();
        let state_at = move |_t: f64| (planet_pos, planet_vel);

        let bodies = [PropagatorBody {
            name: "Planet",
            mu_m3s2: MU_PLANET,
            soi_radius_m: Some(soi),
            state_at: &state_at,
            central_fidelity: None,
            radius_m: None,
        }];

        let r0 = planet_pos - Vector3::new(soi * 1.05, 0.0, 0.0);
        let v0 = Vector3::new(5000.0, 0.0, 0.0);

        let points = propagate(r0, v0, 0.0, 50_000.0, MU_STAR, &bodies, 500.0, 1e-9, 1e-11);
        assert!(points.len() > 5, "expected a real trajectory, got {} points", points.len());

        // At least one switch from None -> Some(0) must occur.
        let saw_reference = points.iter().any(|p| p.central_body_index.is_none());
        let saw_planet = points.iter().any(|p| p.central_body_index == Some(0));
        assert!(saw_reference, "expected at least one point still Sun-centered");
        assert!(saw_planet, "expected at least one point switched to planet-centered");

        // Every recorded marker must be consistent with the actual geometry,
        // to within a tolerance band around the SOI boundary: switching is
        // detected only at accepted integrator steps (approximate, not exact
        // event-refinement — see the design notes), so a
        // point right at the boundary can legitimately land a step's worth
        // of travel past the crossing before/after the marker flips. 5% of
        // the SOI radius comfortably covers that step-boundary slop while
        // still catching a marker that's simply wrong (e.g. stuck at the
        // wrong value, or never switching at all).
        for p in &points {
            let dist_to_planet = (p.r_m - planet_pos).norm();
            match p.central_body_index {
                Some(0) => assert!(
                    dist_to_planet < soi * 1.05,
                    "point marked planet-centered should be within (or near) the SOI boundary, \
                     got dist={dist_to_planet:.3e} m vs soi={soi:.3e} m"
                ),
                None => assert!(
                    dist_to_planet > soi * 0.95,
                    "point marked Sun-centered should be outside (or near) the SOI boundary, \
                     got dist={dist_to_planet:.3e} m vs soi={soi:.3e} m"
                ),
                Some(i) => panic!("unexpected central_body_index {i}, only index 0 (Planet) exists"),
            }
        }
    }

    /// A candidate flown straight at a body with `radius_m` set must stop
    /// at (or just past) `radius_m + COLLISION_MARGIN_M`, not grind on
    /// toward the true 1/r^2 singularity at the body's center. This is the
    /// scenario a GA/PSO fitness loop's largely unconstrained candidates can
    /// produce, and what previously surfaced as `StepSizeUnderflow`.
    #[test]
    fn stops_on_collision_instead_of_chasing_singularity() {
        const MU_STAR: f64 = 1.327e20;
        const MU_PLANET: f64 = 3.986e14;
        const PLANET_RADIUS_M: f64 = 6_371_000.0; // Earth-like
        let planet_a_m = 1.0e11;
        let mass_ratio = MU_PLANET / MU_STAR;
        let soi = laplace_soi_radius_m(planet_a_m, mass_ratio);

        let planet_pos = Vector3::new(planet_a_m, 0.0, 0.0);
        let planet_vel = Vector3::zeros();
        let state_at = move |_t: f64| (planet_pos, planet_vel);

        let bodies = [PropagatorBody {
            name: "Planet",
            mu_m3s2: MU_PLANET,
            soi_radius_m: Some(soi),
            state_at: &state_at,
            central_fidelity: None,
            radius_m: Some(PLANET_RADIUS_M),
        }];

        // Start well inside the SOI, heading straight at the planet's center.
        let r0 = planet_pos - Vector3::new(soi * 0.1, 0.0, 0.0);
        let v0 = Vector3::new(5000.0, 0.0, 0.0);

        // Generous duration -- if the collision stop didn't engage, this
        // would either run to completion deep inside the body (unphysical)
        // or hit StepSizeUnderflow trying to integrate through it.
        let points = propagate(r0, v0, 0.0, 100_000.0, MU_STAR, &bodies, 500.0, 1e-9, 1e-11);
        let last = points.last().expect("expected at least one point");
        let final_dist = (last.r_m - planet_pos).norm();

        assert!(
            final_dist <= PLANET_RADIUS_M + COLLISION_MARGIN_M * 1.5,
            "expected propagation to stop near the collision boundary \
             ({} + {} m), got final distance {final_dist:.3e} m",
            PLANET_RADIUS_M, COLLISION_MARGIN_M
        );
        assert!(
            final_dist >= PLANET_RADIUS_M * 0.5,
            "final point should not have tunneled deep inside the body \
             (got dist {final_dist:.3e} m vs radius {PLANET_RADIUS_M:.3e} m) -- \
             the collision stop should have engaged before this"
        );

        // The collision stop is terminal -- propagation should have ended
        // well before the full 100,000 s requested, not run to completion.
        assert!(
            last.t_s < 100_000.0 * 0.9,
            "expected an early stop on collision, got final t={:.3e} s (~ full duration)",
            last.t_s
        );
    }

    /// A spacecraft flying straight at a stationary toy planet should have
    /// its inbound crossing of some intermediate radius detected at
    /// (interpolated) distance very close to that radius -- not just
    /// snapped to the nearest sample, which at this sample spacing would be
    /// off by a non-trivial fraction of the radius itself.
    #[test]
    fn find_inbound_radius_crossing_interpolates_close_to_requested_radius() {
        const MU_PLANET: f64 = 3.986e14;
        let planet_pos = Vector3::new(1.0e11, 0.0, 0.0);
        let planet_vel = Vector3::zeros();
        let state_at = move |_t: f64| (planet_pos, planet_vel);

        let bodies = [PropagatorBody {
            name: "Planet",
            mu_m3s2: MU_PLANET,
            soi_radius_m: Some(5.0e8),
            state_at: &state_at,
            central_fidelity: None,
            radius_m: Some(1.0e5),
        }];

        // Start already inside the planet's SOI (so reference_mu_m3s2 below
        // is never actually exercised -- this toy test isn't about
        // multi-body switching, just the crossing-detection geometry) and
        // offset laterally, not a zero-impact-parameter dive straight at
        // the center (which hits a literal singularity well before reaching
        // any sensible "radius"). Sampling (60 s) fine enough relative to
        // the close, fast encounter that the discrete samples actually
        // bracket the crossing -- confirmed empirically: 3600 s sampling
        // here is too coarse and skips clean over the whole encounter.
        let r0 = planet_pos - Vector3::new(4.0e8, 5.0e5, 0.0);
        let v0 = Vector3::new(5_000.0, 0.0, 0.0);
        let points = propagate(r0, v0, 0.0, 1.0e5, MU_PLANET, &bodies, 60.0, 1e-10, 1e-3);

        let target_radius_m = 2.0e6;
        let crossing = find_inbound_radius_crossing(&points, &bodies[0], target_radius_m)
            .expect("a direct approach should cross the target radius");

        let actual_dist = crossing.r_rel_m.norm();
        // Linear interpolation over a curving path isn't exact -- 60 s of
        // travel at several km/s is ~300-400 m, and the path curves over
        // that step, so a few hundred meters of residual is expected. Still
        // two orders of magnitude tighter than snapping to the nearest raw
        // sample (which could be off by the full sample-to-sample distance).
        assert!(
            (actual_dist - target_radius_m).abs() < 500.0,
            "interpolated crossing should land within ~500 m of the requested radius, got {actual_dist:.3} m vs {target_radius_m} m"
        );
    }

    /// If the trajectory never gets that close, there's no crossing to find.
    #[test]
    fn find_inbound_radius_crossing_returns_none_when_never_reached() {
        const MU_PLANET: f64 = 3.986e14;
        let planet_pos = Vector3::new(1.0e11, 0.0, 0.0);
        let planet_vel = Vector3::zeros();
        let state_at = move |_t: f64| (planet_pos, planet_vel);
        let bodies = [PropagatorBody {
            name: "Planet", mu_m3s2: MU_PLANET, soi_radius_m: Some(5.0e8),
            state_at: &state_at, central_fidelity: None, radius_m: None,
        }];

        // Flies past well outside any radius we'll ask about.
        let r0 = planet_pos + Vector3::new(0.0, 5.0e7, 0.0);
        let v0 = Vector3::new(5_000.0, 0.0, 0.0);
        let points = propagate(r0, v0, 0.0, 1.0e4, MU_PLANET, &bodies, 1000.0, 1e-10, 1e-3);

        assert!(find_inbound_radius_crossing(&points, &bodies[0], 1.0e5).is_none());
    }
}
