//! Mission-phase state machine. Phase transitions are state-machine
//! conditions (energy/geometry thresholds), not fixed time durations — this
//! makes the sim robust to trajectory variations, per the Mission Planner
//! roadmap's architecture rule ("Phase transitions are state-machine
//! conditions, not time conditions").

use nalgebra::Vector3;

use crate::guidance::PointingMode;
use crate::truth::TruthState;

pub trait Phase {
    fn pointing_mode(&self) -> PointingMode;
    /// `Some(radius_m)` when this phase wants a station-keeping burn applied
    /// toward a target orbit radius (see `guidance::station_keeping_dv`);
    /// `None` for phases with no closed-orbit target (e.g. `FlybyPhase`,
    /// `OrbitInsertionPhase`).
    fn target_radius_m(&self) -> Option<f64>;
    /// Called once per truth step with the post-propagation state and the
    /// step size. Returns `true` once this phase's exit condition is met.
    fn step(&mut self, state: &TruthState, dt_s: f64) -> bool;
    fn name(&self) -> &'static str;
    /// Optional one-shot impulsive burn (e.g. LOI at periapsis). Called every
    /// truth step before the EKF update; the implementation fires at most once
    /// and returns `None` on all subsequent calls.
    fn one_time_burn(&mut self, _state: &TruthState, _mu: f64) -> Option<Vector3<f64>> {
        None
    }
}

/// Hyperbolic flyby: declares completion at closest approach, detected online
/// via the sign change of `r . v` (periapsis passage). This is a fresh online
/// condition — `AstroProbs/Artemis/src/orbit.rs::find_closest_approach()` is a
/// post-hoc array scan over an already-complete trajectory and not usable here.
pub struct FlybyPhase {
    prev_rdotv: Option<f64>,
    passed_periapsis: bool,
}

impl FlybyPhase {
    pub fn new() -> Self {
        Self { prev_rdotv: None, passed_periapsis: false }
    }
}

impl Default for FlybyPhase {
    fn default() -> Self {
        Self::new()
    }
}

impl Phase for FlybyPhase {
    fn pointing_mode(&self) -> PointingMode {
        PointingMode::Nadir
    }

    fn target_radius_m(&self) -> Option<f64> {
        None
    }

    fn step(&mut self, state: &TruthState, _dt_s: f64) -> bool {
        let rdotv = state.r_m.dot(&state.v_mps);
        let crossed = matches!(self.prev_rdotv, Some(prev) if prev < 0.0 && rdotv >= 0.0);
        if crossed {
            self.passed_periapsis = true;
        }
        self.prev_rdotv = Some(rdotv);
        self.passed_periapsis
    }

    fn name(&self) -> &'static str {
        "Flyby"
    }
}

/// Maintains a target orbit radius via proportional station-keeping burns for
/// a fixed duration. Generalizes the *spirit* of `proximity_mission`'s 6-phase
/// station-keeping (Capture/Survey/CloseOrbit/Flyover/ScienceHold/RadioScience)
/// without porting every Bennu-specific operational heuristic — multiple named
/// `MissionPhase`s from the TOML map to repeated `OrbitPhase` instances at the
/// same target radius but (for the MVP) a shared duration; see
/// `MissionPlanner/src/simulate.rs`.
pub struct OrbitPhase {
    pub phase_name: &'static str,
    pub target_radius_m: f64,
    pub duration_s: f64,
    pub pointing: PointingMode,
    elapsed_s: f64,
}

impl OrbitPhase {
    pub fn new(phase_name: &'static str, target_radius_m: f64, duration_s: f64, pointing: PointingMode) -> Self {
        Self { phase_name, target_radius_m, duration_s, pointing, elapsed_s: 0.0 }
    }
}

impl Phase for OrbitPhase {
    fn pointing_mode(&self) -> PointingMode {
        self.pointing
    }

    fn target_radius_m(&self) -> Option<f64> {
        Some(self.target_radius_m)
    }

    fn step(&mut self, _state: &TruthState, dt_s: f64) -> bool {
        self.elapsed_s += dt_s;
        self.elapsed_s >= self.duration_s
    }

    fn name(&self) -> &'static str {
        self.phase_name
    }
}

// ── Landing (deorbit + free-fall descent) ────────────────────────────────────

/// Deorbit from a circular parking orbit and free-fall toward the surface.
///
/// On the first truth step, `one_time_burn` fires a single retrograde Hohmann
/// burn that lowers periapsis to the body surface:
///
///   a_transfer = (r_orbit + r_surface) / 2
///   v_transfer = sqrt(mu * (2/r_orbit - 1/a_transfer))
///   ΔV         = v_transfer − v_circ(r_orbit)   [retrograde, so negative]
///
/// No station-keeping is applied during descent — the spacecraft free-falls
/// (or thrusts under powered descent, which is outside this phase's scope).
/// The phase exits when altitude drops below `terminal_altitude_m` (powered-
/// descent handoff trigger) or at surface impact if it's not set.
pub struct LandingPhase {
    pub phase_name: &'static str,
    /// Body mean radius [m] — determines surface and deorbit Hohmann target.
    pub body_radius_m: f64,
    /// Altitude [m] above surface that triggers the phase exit (powered-descent
    /// handoff). `None` means run until surface impact.
    pub terminal_altitude_m: Option<f64>,
    pub pointing: PointingMode,
    deorbit_fired: bool,
    pub touchdown: bool,
}

impl LandingPhase {
    pub fn new(
        phase_name: &'static str,
        body_radius_m: f64,
        terminal_altitude_m: Option<f64>,
        pointing: PointingMode,
    ) -> Self {
        Self {
            phase_name,
            body_radius_m,
            terminal_altitude_m,
            pointing,
            deorbit_fired: false,
            touchdown: false,
        }
    }
}

impl Phase for LandingPhase {
    fn pointing_mode(&self) -> PointingMode {
        self.pointing
    }

    fn target_radius_m(&self) -> Option<f64> {
        None // free-fall — no SK burns during descent
    }

    fn one_time_burn(&mut self, state: &TruthState, mu: f64) -> Option<Vector3<f64>> {
        if self.deorbit_fired {
            return None;
        }
        self.deorbit_fired = true;

        let r = state.r_m.norm();
        let v_mag = state.v_mps.norm();
        if v_mag < 1e-12 {
            return None;
        }

        // Hohmann semi-major axis: apoapsis = current orbit, periapsis = body surface
        let r_surface = self.body_radius_m;
        let a_transfer = (r + r_surface) / 2.0;
        // Speed at apoapsis of transfer ellipse (where the deorbit burn fires)
        let v_transfer_apo = (mu * (2.0 / r - 1.0 / a_transfer)).sqrt();
        let v_circ = (mu / r).sqrt();
        let delta_v = v_transfer_apo - v_circ; // negative → retrograde

        // Apply along velocity direction (retrograde when delta_v < 0)
        let v_hat = state.v_mps / v_mag;
        let dv = delta_v * v_hat;
        if dv.norm() < 1e-15 {
            None
        } else {
            Some(dv)
        }
    }

    fn step(&mut self, state: &TruthState, _dt_s: f64) -> bool {
        let altitude = state.r_m.norm() - self.body_radius_m;
        // Surface impact
        if altitude <= 0.0 {
            self.touchdown = true;
            return true;
        }
        // Powered-descent handoff altitude
        if let Some(term) = self.terminal_altitude_m {
            if altitude < term {
                return true;
            }
        }
        false
    }

    fn name(&self) -> &'static str {
        self.phase_name
    }
}

// ── Hohmann transfer ─────────────────────────────────────────────────────────

/// Physical Hohmann transfer between two circular orbits.
///
/// **Burn 1** fires immediately (at the first truth step) along the current
/// velocity direction. For a circular departure orbit, any point is an
/// acceptable apse, so no phasing wait is needed:
///
///   a_transfer = (r_current + r_target) / 2
///   v_transfer = √(μ · (2/r_current − 1/a_transfer))
///   ΔV₁        = v_transfer − v_circ(r_current)    [−: retrograde, +: prograde]
///
/// The spacecraft then coasts on the transfer ellipse under full perturbed
/// dynamics. Arrival at the opposite apse (periapsis for descending, apoapsis
/// for ascending) is detected by the sign reversal of r·v after the spacecraft
/// has committed to the expected transfer direction.
///
/// **Burn 2** (circularization) fires at arrival:
///
///   ΔV₂ = v_circ(r_actual) − v_tangential_actual
///
/// where r_actual is the true radius at arrival — this absorbs any small
/// perturbation error accumulated during the coast. The phase exits one
/// integration step after the circularization burn.
pub struct HohmannTransferPhase {
    pub phase_name: &'static str,
    /// Target circular orbit radius [m].
    pub r_target_m: f64,
    pub pointing: PointingMode,
    pub mu: f64,
    state: HohmannInternalState,
    /// True for r_target > r_departure (ascending transfer).
    ascending: bool,
    /// True once r·v has moved in the expected transfer direction — guards
    /// against triggering the second burn on the same step as the first.
    committed: bool,
    prev_rdotv: f64,
}

enum HohmannInternalState {
    FireFirst,
    Coasting,
    Done,
}

impl HohmannTransferPhase {
    pub fn new(
        phase_name: &'static str,
        r_target_m: f64,
        pointing: PointingMode,
        mu: f64,
    ) -> Self {
        Self {
            phase_name,
            r_target_m,
            pointing,
            mu,
            state: HohmannInternalState::FireFirst,
            ascending: false,
            committed: false,
            prev_rdotv: 0.0,
        }
    }
}

impl Phase for HohmannTransferPhase {
    fn pointing_mode(&self) -> PointingMode {
        self.pointing
    }

    // No SK burns during the coast — the spacecraft is on the transfer ellipse.
    fn target_radius_m(&self) -> Option<f64> {
        None
    }

    fn name(&self) -> &'static str {
        self.phase_name
    }

    fn one_time_burn(&mut self, state: &TruthState, _mu: f64) -> Option<Vector3<f64>> {
        match self.state {
            HohmannInternalState::FireFirst => {
                let r = state.r_m.norm();
                if r < 1e-3 {
                    return None;
                }
                let v_circ = (self.mu / r).sqrt();
                let a_transfer = (r + self.r_target_m) / 2.0;
                // Speed at the firing point on the transfer ellipse
                let v_transfer = (self.mu * (2.0 / r - 1.0 / a_transfer)).sqrt();
                // Negative for descend (retro), positive for ascend (prograde)
                let delta_speed = v_transfer - v_circ;

                self.ascending = self.r_target_m > r;

                let v_mag = state.v_mps.norm();
                let v_hat = if v_mag > 1e-12 {
                    state.v_mps / v_mag
                } else {
                    // Degenerate: build a perpendicular in the XY plane
                    let r_hat = state.r_m / r;
                    Vector3::new(-r_hat.y, r_hat.x, 0.0).normalize()
                };

                self.state = HohmannInternalState::Coasting;
                self.committed = false;
                self.prev_rdotv = state.r_m.dot(&state.v_mps);

                Some(delta_speed * v_hat)
            }

            HohmannInternalState::Coasting => {
                let rdotv = state.r_m.dot(&state.v_mps);

                // Mark committed once r·v settles into the expected direction
                if !self.committed {
                    if self.ascending && rdotv > 0.0 {
                        self.committed = true;
                    } else if !self.ascending && rdotv < 0.0 {
                        self.committed = true;
                    }
                }

                // Arrival: r·v reverses after committing — opposite-apse crossing
                let arrived = self.committed && (
                    ( self.ascending && self.prev_rdotv > 0.0 && rdotv <= 0.0) ||
                    (!self.ascending && self.prev_rdotv < 0.0 && rdotv >= 0.0)
                );

                self.prev_rdotv = rdotv;

                if arrived {
                    // Circularize at the actual arrival radius (absorbs coast error)
                    let r = state.r_m.norm();
                    let r_hat = state.r_m / r;
                    let v_radial = r_hat * r_hat.dot(&state.v_mps);
                    let v_tang = state.v_mps - v_radial;
                    let v_tang_mag = v_tang.norm();
                    let v_tang_hat = if v_tang_mag > 1e-12 {
                        v_tang / v_tang_mag
                    } else {
                        Vector3::new(-r_hat.y, r_hat.x, 0.0).normalize()
                    };
                    let v_circ_target = (self.mu / r).sqrt();
                    // Kill radial component and match circular tangential speed
                    let dv = (v_circ_target - v_tang_mag) * v_tang_hat - v_radial;
                    self.state = HohmannInternalState::Done;
                    return Some(dv);
                }

                None
            }

            HohmannInternalState::Done => None,
        }
    }

    fn step(&mut self, _state: &TruthState, _dt_s: f64) -> bool {
        matches!(self.state, HohmannInternalState::Done)
    }
}

// ── Orbit insertion ───────────────────────────────────────────────────────────

/// Hyperbolic (or elliptic) approach phase that fires a single retrograde LOI
/// burn at periapsis to capture into a circular orbit at `target_orbit_radius_m`.
///
/// Periapsis is detected online via the sign change of r·v (- to +). At that
/// instant `one_time_burn` returns:
///
///   ΔV = v_circ(r_target) − |v_tangential|, applied in the retro direction
///
/// The exit condition is orbital energy < 0 (spacecraft captured). Once
/// captured the caller should hand off to `OrbitPhase` for station-keeping.
pub struct OrbitInsertionPhase {
    pub phase_name: &'static str,
    pub target_radius_m: f64,
    pub pointing: PointingMode,
    pub mu: f64,
    prev_rdotv: Option<f64>,
    loi_fired: bool,
}

impl OrbitInsertionPhase {
    pub fn new(
        phase_name: &'static str,
        target_radius_m: f64,
        pointing: PointingMode,
        mu: f64,
    ) -> Self {
        Self { phase_name, target_radius_m, pointing, mu, prev_rdotv: None, loi_fired: false }
    }
}

impl Phase for OrbitInsertionPhase {
    fn pointing_mode(&self) -> PointingMode {
        self.pointing
    }

    fn target_radius_m(&self) -> Option<f64> {
        None  // No continuous SK during approach — just the single LOI burn
    }

    fn one_time_burn(&mut self, state: &TruthState, mu: f64) -> Option<Vector3<f64>> {
        if self.loi_fired {
            return None;
        }
        // Detect periapsis: r·v transitions from negative to non-negative
        let rdotv = state.r_m.dot(&state.v_mps);
        let at_periapsis = matches!(self.prev_rdotv, Some(prev) if prev < 0.0 && rdotv >= 0.0);
        self.prev_rdotv = Some(rdotv);
        if !at_periapsis {
            return None;
        }
        self.loi_fired = true;

        let r_norm = state.r_m.norm();
        let r_hat = state.r_m / r_norm;
        let v_radial = r_hat * r_hat.dot(&state.v_mps);
        let v_tang = state.v_mps - v_radial;
        let v_tang_mag = v_tang.norm();

        // Circularize at target radius: retrograde burn to match v_circ(r_target)
        let v_circ = (mu / self.target_radius_m).sqrt();
        let delta_speed = v_circ - v_tang_mag;  // negative (retro) on hyperbolic approach

        let v_hat = if v_tang_mag > 1e-9 {
            v_tang / v_tang_mag
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };

        // Also kill residual radial velocity
        let dv_tang = delta_speed * v_hat;
        let dv_radial = -v_radial;
        let dv = dv_tang + dv_radial;

        if dv.norm() < 1e-15 { None } else { Some(dv) }
    }

    fn step(&mut self, state: &TruthState, _dt_s: f64) -> bool {
        // Exit once orbital energy is negative (captured)
        let e = 0.5 * state.v_mps.norm_squared() - self.mu / state.r_m.norm();
        e < 0.0
    }

    fn name(&self) -> &'static str {
        self.phase_name
    }
}
