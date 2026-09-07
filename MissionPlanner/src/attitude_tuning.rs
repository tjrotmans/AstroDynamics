//! Three-layer attitude-control registry — the MissionPlanner half of the
//! architecture (`docs/MP/MANUAL.md` §10.5; the laws
//! themselves live in `sim_engine::control`):
//!
//! - **Layer 1 (per actuator)**: one [`AttitudeLaw`] for the wheels and one
//!   for the thrusters, each DERIVED by default from that actuator class's
//!   real torque authority (wheels: cluster `max_torque`; thrusters: the
//!   placed RCS layout's worst-axis authority, the same number `/api/design/
//!   vehicle` reports as `rcs_authority_nm`) and the vehicle's real inertia,
//!   with every parameter overridable from `gnc.attitude_control`. The live
//!   loop picks the law by `ControlMode` every tick, so a burn (thrusters
//!   primary) is no longer flown on gains sized for the wheels — the
//! confirmed cause of the tumbling-during-burn report (PD gains
//!   sized for 0.12 N·m wheels commanded ~3% of a 3.54 N·m RCS layout, and
//!   the vehicle rolled through 180° against a 0.059 N·m engine torque).
//! - **Layer 2 (per activity)**: [`Activity`]-dependent tuning on top of the
//!   layer-1 law ([`tuned_law`]).
//! - **Layer 3 (scheduling)**: [`AttitudeControlSet::law_for`] re-derives
//!   from the CURRENT mass properties; the cruise loop calls it whenever the
//!   mode/activity changes or mass drifts past a threshold, and records a
//!   [`GainSchedulePoint`] each time.
//!
//! Everything derived is reported back through [`AttitudeControlEffective`]
//! so a run is reproducible from the values actually used.

use nalgebra::Vector3;
use sim_engine::{
    rcs_axis_authority_nm, rcs_worst_axis_authority_nm, tuned_law, Activity, ActivityTuning, AttitudeLaw,
    ControlMode, PdGains, PhasePlaneParams, ReactionWheelCluster, SpacecraftProperties, Thruster,
};

use crate::config::{ActivityTuningConfig, AttitudeLawConfig, MissionConfig};
use crate::simulate::{derived_pd_gains_with_ceiling, ZETA_TARGET};

/// Plausibility ceiling on the derived ω_n for WHEEL loops [rad/s] — the
/// same 0.05 `simulate::derived_pd_gains` has always used (quiescent
/// cruise pointing).
const WHEEL_OMEGA_N_CEILING_RADPS: f64 = 0.05;
/// Ceiling for THRUSTER-mode loops [rad/s] — a burn-attitude hold against
/// a persistent engine disturbance legitimately runs stiffer than a
/// quiescent wheel loop (powered-flight TVC/RCS attitude loops at 0.1–0.5
/// rad/s: Wie, *Space Vehicle Dynamics and Control*, 2nd ed., Ch. 7).
/// Without a per-class ceiling the mode-scheduled burn tick (§10.5.1)
/// raises the tick cap only for the wheel-sized 0.05 ceiling to re-cap
/// the loop immediately.
const THRUSTER_OMEGA_N_CEILING_RADPS: f64 = 0.2;

/// Default phase-plane deadband [deg] — the same 0.5° the executive's
/// settle gate and `/api/design/slew-test` use, so "settled" and "inside
/// the deadband" mean the same thing.
const DEFAULT_PHASE_PLANE_DEADBAND_DEG: f64 = 0.5;
/// Default minimum thruster on-time [s] — a typical monopropellant valve
/// minimum pulse (Sutton & Biblarz, *Rocket Propulsion Elements*, 9th ed.,
/// Ch. 11: 10–30 ms for small hydrazine thrusters).
const DEFAULT_MIN_ON_TIME_S: f64 = 0.02;
/// Integral time constant as a multiple of the natural period: `T_i =
/// INTEGRAL_TIME_PERIODS · 2π/ω_n`, i.e. `k_i = k_p·ω_n/(2π·N)`. Keeps the
/// integrator's corner well below the loop crossover so it removes bias
/// without eroding phase margin (Franklin, Powell & Emami-Naeini, *Feedback
/// Control of Dynamic Systems*, PID tuning guidance, Ch. 4.3).
const INTEGRAL_TIME_PERIODS: f64 = 2.0;
/// Default layer-3 mass-change trigger (fraction of the reference mass).
const DEFAULT_MASS_CHANGE_FRACTION: f64 = 0.05;

/// Which mass-properties model layer 3 schedules against.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct GainScheduling {
    pub enabled: bool,
    pub mass_change_fraction: f64,
    /// See `GainSchedulingConfig::inertia_scales_with_mass` — off by default
    /// because the truth propagator holds inertia constant.
    pub inertia_scales_with_mass: bool,
}

/// The resolved registry for one vehicle: base laws per actuator class,
/// per-activity tuning, authority numbers, scheduling settings.
#[derive(Clone, Debug)]
pub struct AttitudeControlSet {
    wheels: AttitudeLaw,
    thrusters: AttitudeLaw,
    wheels_source: &'static str,
    thrusters_source: &'static str,
    activities: [(Activity, ActivityTuning); 4],
    pub wheel_authority_nm: f64,
    pub rcs_authority_nm: f64,
    pub rcs_axis_authority_nm: Vector3<f64>,
    pub inertia_ref_kgm2: Vector3<f64>,
    pub mass_ref_kg: f64,
    pub tick_s: f64,
    /// Mode-scheduled tick (`cruise_seed.burn_tick_s`): the
    /// tick the THRUSTER-mode law is derived at — a burn runs at this
    /// shorter tick, lifting the `ω_n ≤ 2π/(15·tick)` discrete-loop cap
    /// exactly where the RCS's larger authority needs it (§10.5.1). Equal
    /// to `tick_s` when no mode-scheduled tick is in use.
    pub thruster_tick_s: f64,
    pub scheduling: GainScheduling,
    warnings: Vec<String>,
}

/// One point of the layer-3 schedule — recorded by the cruise loop every
/// time the active law is (re)derived.
#[derive(Clone, Debug, serde::Serialize)]
pub struct GainSchedulePoint {
    pub t_s: f64,
    pub mass_kg: f64,
    pub control_mode: String,
    pub activity: String,
    pub law: String,
    pub kp: Option<f64>,
    pub kd: Option<f64>,
    pub ki: Option<f64>,
    pub deadband_deg: Option<f64>,
    pub omega_n_radps: Option<f64>,
    pub zeta: Option<f64>,
}

/// Effective (derived or overridden) parameters of one layer-1 law.
#[derive(Clone, Debug, serde::Serialize)]
pub struct AttitudeLawReport {
    pub law: String,
    /// `"derived"` or `"config"`.
    pub source: String,
    pub authority_nm: f64,
    pub kp: Option<f64>,
    pub kd: Option<f64>,
    pub ki: Option<f64>,
    pub integral_limit_rad_s: Option<f64>,
    pub deadband_deg: Option<f64>,
    pub rate_deadband_degs: Option<f64>,
    pub hysteresis_deg: Option<f64>,
    pub min_on_time_s: Option<f64>,
    pub lead_time_s: Option<f64>,
    pub omega_n_radps: Option<f64>,
    pub zeta: Option<f64>,
    pub settling_time_s: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ActivityTuningReport {
    pub activity: String,
    pub bandwidth_scale: f64,
    pub deadband_scale: f64,
}

/// `CruiseResult.attitude_control_effective` — everything the run actually
/// used, for reproducibility and for the frontend's tuning UI.
#[derive(Clone, Debug, serde::Serialize)]
pub struct AttitudeControlEffective {
    pub wheels: AttitudeLawReport,
    pub thrusters: AttitudeLawReport,
    pub activities: Vec<ActivityTuningReport>,
    pub gain_scheduling: GainScheduling,
    pub inertia_ref_kgm2: [f64; 3],
    pub mass_ref_kg: f64,
    pub tick_s: f64,
    pub warnings: Vec<String>,
}

fn max_inertia(inertia: &Vector3<f64>) -> f64 {
    inertia.x.max(inertia.y).max(inertia.z).max(1e-6)
}

/// Derived PD for one actuator class at the given inertia (§10.1 rule,
/// shared with the wheels-only derivation `simulate.rs` already had).
/// `ceiling` is the per-actuator-class ω_n plausibility ceiling
/// (`WHEEL_OMEGA_N_CEILING_RADPS` / `THRUSTER_OMEGA_N_CEILING_RADPS`).
fn derived_pd(inertia: &Vector3<f64>, tau_authority_nm: f64, tick_s: f64, ceiling: f64) -> PdGains {
    let (kp, kd) = derived_pd_gains_with_ceiling(max_inertia(inertia), tau_authority_nm.max(1e-9), tick_s, ceiling);
    PdGains { kp, kd, pointing_db_rad: 0.0, rate_db_rads: 0.0 }
}

/// Derived PID: the PD above plus `k_i = k_p·ω_n/(2π·N)` and an anti-windup
/// limit such that the integral term alone cannot exceed the authority.
fn derived_pid(inertia: &Vector3<f64>, tau_authority_nm: f64, tick_s: f64, ceiling: f64) -> AttitudeLaw {
    let gains = derived_pd(inertia, tau_authority_nm, tick_s, ceiling);
    let omega_n = (gains.kp / (2.0 * max_inertia(inertia))).sqrt();
    let ki = gains.kp * omega_n / (2.0 * std::f64::consts::PI * INTEGRAL_TIME_PERIODS);
    let integral_limit_rad_s = if ki > 0.0 { tau_authority_nm.max(1e-9) / ki } else { 0.0 };
    AttitudeLaw::Pid { gains, ki, integral_limit_rad_s }
}

/// Derived phase-plane: deadband 0.5°, lead time = the derived PD's
/// equivalent rate weighting `2·k_d/k_p` (§10.5.3), hysteresis 0.2·δ, rate
/// deadband δ/T, 20 ms minimum pulse, per-axis RCS authority.
fn derived_phase_plane(inertia: &Vector3<f64>, axis_authority_nm: &Vector3<f64>, tick_s: f64, ceiling: f64) -> PhasePlaneParams {
    let worst = axis_authority_nm.x.min(axis_authority_nm.y).min(axis_authority_nm.z);
    let pd = derived_pd(inertia, worst, tick_s, ceiling);
    let lead_time_s = if pd.kp > 0.0 { 2.0 * pd.kd / pd.kp } else { 10.0 * tick_s };
    let deadband_rad = DEFAULT_PHASE_PLANE_DEADBAND_DEG.to_radians();
    PhasePlaneParams {
        deadband_rad,
        rate_deadband_radps: deadband_rad / lead_time_s.max(1e-9),
        hysteresis_rad: 0.2 * deadband_rad,
        min_on_time_s: DEFAULT_MIN_ON_TIME_S,
        lead_time_s,
        inertia_diag_kgm2: *inertia,
        tau_authority_nm: *axis_authority_nm,
    }
}

/// Resolve one actuator class's law from config (overrides) + derivation.
/// Returns `(law, source)`.
fn resolve_law(
    cfg_law: Option<&AttitudeLawConfig>,
    default_kind: &AttitudeLawConfig,
    inertia: &Vector3<f64>,
    scalar_authority_nm: f64,
    axis_authority_nm: &Vector3<f64>,
    tick_s: f64,
    ceiling: f64,
    legacy_pd_override: Option<(Option<f64>, Option<f64>)>,
) -> (AttitudeLaw, &'static str) {
    let (spec, mut source) = match cfg_law {
        Some(l) => (l, "config"),
        None => (default_kind, "derived"),
    };
    let law = match spec {
        AttitudeLawConfig::Pd { kp, kd } => {
            let mut g = derived_pd(inertia, scalar_authority_nm, tick_s, ceiling);
            // Legacy `gnc.reaction_wheel_kp/kd` (wheels only) applies when
            // no `attitude_control.wheels` block names the law.
            let (lkp, lkd) = if cfg_law.is_none() { legacy_pd_override.unwrap_or((None, None)) } else { (None, None) };
            if let Some(v) = kp.or(lkp) { g.kp = v; source = "config"; }
            if let Some(v) = kd.or(lkd) { g.kd = v; source = "config"; }
            AttitudeLaw::Pd(g)
        }
        AttitudeLawConfig::Pid { kp, ki, kd, integral_limit } => {
            let mut law = derived_pid(inertia, scalar_authority_nm, tick_s, ceiling);
            if let AttitudeLaw::Pid { gains, ki: ki_d, integral_limit_rad_s } = &mut law {
                if let Some(v) = kp { gains.kp = *v; }
                if let Some(v) = kd { gains.kd = *v; }
                if let Some(v) = ki { *ki_d = *v; }
                if let Some(v) = integral_limit { *integral_limit_rad_s = *v; }
            }
            law
        }
        AttitudeLawConfig::PhasePlane { deadband_deg, rate_deadband_degs, hysteresis_deg, min_on_time_s, lead_time_s } => {
            let mut p = derived_phase_plane(inertia, axis_authority_nm, tick_s, ceiling);
            if let Some(v) = lead_time_s { p.lead_time_s = *v; }
            if let Some(v) = deadband_deg {
                p.deadband_rad = v.to_radians();
                p.hysteresis_rad = 0.2 * p.deadband_rad;
                p.rate_deadband_radps = p.deadband_rad / p.lead_time_s.max(1e-9);
            }
            if let Some(v) = rate_deadband_degs { p.rate_deadband_radps = v.to_radians(); }
            if let Some(v) = hysteresis_deg { p.hysteresis_rad = v.to_radians(); }
            if let Some(v) = min_on_time_s { p.min_on_time_s = *v; }
            AttitudeLaw::PhasePlane(p)
        }
    };
    (law, source)
}

fn activity_tuning(activity: Activity, cfg: Option<&ActivityTuningConfig>) -> ActivityTuning {
    let d = activity.default_tuning();
    match cfg {
        None => d,
        Some(c) => ActivityTuning {
            bandwidth_scale: c.bandwidth_scale.unwrap_or(d.bandwidth_scale),
            deadband_scale: c.deadband_scale.unwrap_or(d.deadband_scale),
        },
    }
}

fn law_report(law: &AttitudeLaw, source: &str, authority_nm: f64, inertia_max: f64) -> AttitudeLawReport {
    let (omega_n, zeta) = law.natural_frequency_and_damping(inertia_max).map(|(w, z)| (Some(w), Some(z))).unwrap_or((None, None));
    let mut r = AttitudeLawReport {
        law: law.label().to_string(),
        source: source.to_string(),
        authority_nm,
        kp: None, kd: None, ki: None, integral_limit_rad_s: None,
        deadband_deg: None, rate_deadband_degs: None, hysteresis_deg: None, min_on_time_s: None, lead_time_s: None,
        omega_n_radps: omega_n,
        zeta,
        settling_time_s: law.settling_time_s(inertia_max),
    };
    match law {
        AttitudeLaw::Pd(g) => { r.kp = Some(g.kp); r.kd = Some(g.kd); }
        AttitudeLaw::Pid { gains, ki, integral_limit_rad_s } => {
            r.kp = Some(gains.kp); r.kd = Some(gains.kd); r.ki = Some(*ki); r.integral_limit_rad_s = Some(*integral_limit_rad_s);
        }
        AttitudeLaw::PhasePlane(p) => {
            r.deadband_deg = Some(p.deadband_rad.to_degrees());
            r.rate_deadband_degs = Some(p.rate_deadband_radps.to_degrees());
            r.hysteresis_deg = Some(p.hysteresis_rad.to_degrees());
            r.min_on_time_s = Some(p.min_on_time_s);
            r.lead_time_s = Some(p.lead_time_s);
        }
    }
    r
}

impl AttitudeControlSet {
    /// Build the registry for a mission from its config and the vehicle
    /// actually constructed in Phase 02.
    pub fn from_cfg(
        cfg: &MissionConfig,
        sc: &SpacecraftProperties,
        wheel_cluster: &ReactionWheelCluster,
        rcs_thrusters: &[Thruster],
        tick_s: f64,
        // The tick thruster-mode control actually runs at (`cruise_seed.
        // burn_tick_s` resolved by the caller; pass `tick_s` when no
        // mode-scheduled tick applies, e.g. the slew test).
        thruster_tick_s: f64,
    ) -> Self {
        let ac = &cfg.gnc.attitude_control;
        let inertia = sc.inertia_diag_kgm2;
        let wheel_authority_nm = wheel_cluster.max_torque;
        let rcs_axis = rcs_axis_authority_nm(rcs_thrusters);
        let rcs_authority_nm = rcs_worst_axis_authority_nm(rcs_thrusters);

        let (wheels, wheels_source) = resolve_law(
            ac.wheels.as_ref(),
            &AttitudeLawConfig::Pd { kp: None, kd: None },
            &inertia, wheel_authority_nm, &Vector3::new(wheel_authority_nm, wheel_authority_nm, wheel_authority_nm), tick_s,
            WHEEL_OMEGA_N_CEILING_RADPS,
            Some((cfg.gnc.reaction_wheel_kp, cfg.gnc.reaction_wheel_kd)),
        );
        let mut warnings = Vec::new();
        // Thrusters with zero authority (no RCS placed): fall back to the
        // wheel-derived law so a burn tick still commands SOMETHING sane
        // (the allocator degrades ThrustersPrimary to no-RCS anyway).
        let (thrusters, thrusters_source) = if rcs_authority_nm > 1e-12 {
            resolve_law(
                ac.thrusters.as_ref(),
                &AttitudeLawConfig::Pid { kp: None, ki: None, kd: None, integral_limit: None },
                &inertia, rcs_authority_nm, &rcs_axis, thruster_tick_s,
                THRUSTER_OMEGA_N_CEILING_RADPS, None,
            )
        } else {
            warnings.push(
                "no RCS torque authority on at least one body axis (no thrusters placed, or none aligned) — \
                 thruster-mode law falls back to the wheel-derived law"
                    .to_string(),
            );
            (wheels, "derived")
        };

        // Phase-plane minimum-impulse feasibility (§10.5.3): the limit
        // cycle's rate kick Δω_min = τ_auth·t_min/I moves the switching
        // function by Δω_min·T, which must fit inside the deadband.
        if let AttitudeLaw::PhasePlane(p) = &thrusters {
            let i_min = inertia.x.min(inertia.y).min(inertia.z).max(1e-6);
            let worst_axis_auth = p.tau_authority_nm.x.max(p.tau_authority_nm.y).max(p.tau_authority_nm.z);
            let kick_rad = worst_axis_auth * p.min_on_time_s / i_min * p.lead_time_s;
            if kick_rad > p.deadband_rad {
                warnings.push(format!(
                    "phase-plane minimum impulse bit too coarse for its deadband: one {:.3} s pulse at {:.2} N·m on \
                     I = {:.0} kg·m² moves the switching function by {:.2}° (> deadband {:.2}°) — expect a limit cycle \
                     of that amplitude; shorten min_on_time_s, widen deadband_deg, or use lower-thrust attitude thrusters",
                    p.min_on_time_s, worst_axis_auth, i_min, kick_rad.to_degrees(), p.deadband_rad.to_degrees()
                ));
            }
        }

        let activities = [
            (Activity::Hold, activity_tuning(Activity::Hold, ac.activities.hold.as_ref())),
            (Activity::Slew, activity_tuning(Activity::Slew, ac.activities.slew.as_ref())),
            (Activity::BurnHold, activity_tuning(Activity::BurnHold, ac.activities.burn_hold.as_ref())),
            (Activity::Coast, activity_tuning(Activity::Coast, ac.activities.coast.as_ref())),
        ];
        let gs = &ac.gain_scheduling;
        let scheduling = GainScheduling {
            enabled: gs.enabled.unwrap_or(true),
            mass_change_fraction: gs.mass_change_fraction.unwrap_or(DEFAULT_MASS_CHANGE_FRACTION).max(1e-6),
            inertia_scales_with_mass: gs.inertia_scales_with_mass.unwrap_or(false),
        };
        Self {
            wheels, thrusters, wheels_source, thrusters_source, activities,
            wheel_authority_nm, rcs_authority_nm, rcs_axis_authority_nm: rcs_axis,
            inertia_ref_kgm2: inertia, mass_ref_kg: sc.mass_kg, tick_s, thruster_tick_s, scheduling, warnings,
        }
    }

    /// A registry that flies the supplied PD gains for BOTH actuator
    /// classes with no activity scaling and no scheduling — the exact
    /// pre-behavior, for the demo binaries and tests that hand-
    /// tune one gain set.
    pub fn from_pd(gains: PdGains, inertia_ref_kgm2: Vector3<f64>, mass_ref_kg: f64, wheel_authority_nm: f64, rcs_authority_nm: f64, tick_s: f64) -> Self {
        let unit = ActivityTuning { bandwidth_scale: 1.0, deadband_scale: 1.0 };
        Self {
            wheels: AttitudeLaw::Pd(gains),
            thrusters: AttitudeLaw::Pd(gains),
            wheels_source: "config",
            thrusters_source: "config",
            activities: [(Activity::Hold, unit), (Activity::Slew, unit), (Activity::BurnHold, unit), (Activity::Coast, unit)],
            wheel_authority_nm,
            rcs_authority_nm,
            rcs_axis_authority_nm: Vector3::new(rcs_authority_nm, rcs_authority_nm, rcs_authority_nm),
            inertia_ref_kgm2,
            mass_ref_kg,
            tick_s,
            thruster_tick_s: tick_s,
            scheduling: GainScheduling { enabled: false, mass_change_fraction: DEFAULT_MASS_CHANGE_FRACTION, inertia_scales_with_mass: false },
            warnings: Vec::new(),
        }
    }

    /// The tick a mode's control loop actually runs at (§10.5.1's
    /// mode-scheduled tick) — thruster modes run at `thruster_tick_s`.
    pub fn tick_for(&self, mode: ControlMode) -> f64 {
        match mode {
            ControlMode::WheelsPrimary => self.tick_s,
            ControlMode::ThrustersPrimary | ControlMode::ThrustersOnly => self.thruster_tick_s,
        }
    }

    /// Layer-1 base law for a control mode (no activity/scheduling applied).
    pub fn base_law(&self, mode: ControlMode) -> AttitudeLaw {
        match mode {
            ControlMode::WheelsPrimary => self.wheels,
            ControlMode::ThrustersPrimary | ControlMode::ThrustersOnly => self.thrusters,
        }
    }

    /// Torque authority [N·m] the given mode's primary actuator can deliver
    /// on its worst axis — also what §10.4's slew profile should
    /// deceleration-limit against in that mode.
    pub fn authority_nm(&self, mode: ControlMode) -> f64 {
        match mode {
            ControlMode::WheelsPrimary => self.wheel_authority_nm,
            ControlMode::ThrustersPrimary | ControlMode::ThrustersOnly => {
                if self.rcs_authority_nm > 1e-12 { self.rcs_authority_nm } else { self.wheel_authority_nm }
            }
        }
    }

    pub fn tuning(&self, activity: Activity) -> ActivityTuning {
        self.activities.iter().find(|(a, _)| *a == activity).map(|(_, t)| *t).unwrap_or(activity.default_tuning())
    }

    /// Scheduling inertia at the given mass (§10.5.5).
    pub fn inertia_at(&self, mass_kg: f64) -> Vector3<f64> {
        if self.scheduling.inertia_scales_with_mass && self.mass_ref_kg > 0.0 {
            self.inertia_ref_kgm2 * (mass_kg / self.mass_ref_kg).max(1e-3)
        } else {
            self.inertia_ref_kgm2
        }
    }

    /// The law to run this tick: layer 1 (by mode) → layer 3 (re-derived at
    /// the current mass when the law was derived, not overridden) → layer 2
    /// (activity tuning). Overridden gains are never rescaled by mass —
    /// what the user typed is what flies.
    pub fn law_for(&self, mode: ControlMode, activity: Activity, mass_kg: f64) -> AttitudeLaw {
        let base = self.base_law(mode);
        let source = match mode {
            ControlMode::WheelsPrimary => self.wheels_source,
            _ => self.thrusters_source,
        };
        let scheduled = if self.scheduling.enabled && source == "derived" {
            let inertia = self.inertia_at(mass_kg);
            let auth = self.authority_nm(mode);
            let tick = self.tick_for(mode);
            let ceiling = match mode {
                ControlMode::WheelsPrimary => WHEEL_OMEGA_N_CEILING_RADPS,
                ControlMode::ThrustersPrimary | ControlMode::ThrustersOnly => THRUSTER_OMEGA_N_CEILING_RADPS,
            };
            match base {
                AttitudeLaw::Pd(g) => AttitudeLaw::Pd(PdGains { pointing_db_rad: g.pointing_db_rad, rate_db_rads: g.rate_db_rads, ..derived_pd(&inertia, auth, tick, ceiling) }),
                AttitudeLaw::Pid { .. } => derived_pid(&inertia, auth, tick, ceiling),
                AttitudeLaw::PhasePlane(p) => {
                    let mut np = derived_phase_plane(&inertia, &p.tau_authority_nm, tick, ceiling);
                    np.deadband_rad = p.deadband_rad;
                    np.hysteresis_rad = p.hysteresis_rad;
                    np.min_on_time_s = p.min_on_time_s;
                    AttitudeLaw::PhasePlane(np)
                }
            }
        } else {
            base
        };
        tuned_law(&scheduled, self.tuning(activity))
    }

    /// Should the cruise loop re-derive at this mass? (`true` when the mass
    /// moved by more than `mass_change_fraction` of the reference since
    /// `last_mass_kg`.)
    pub fn mass_trigger(&self, last_mass_kg: f64, mass_kg: f64) -> bool {
        self.scheduling.enabled
            && self.scheduling.inertia_scales_with_mass
            && (mass_kg - last_mass_kg).abs() > self.scheduling.mass_change_fraction * self.mass_ref_kg.max(1e-9)
    }

    /// 2% settling time [s] of the base law for a mode at the reference
    /// inertia — what the burn executive's lead time adds to the kinematic
    /// slew time (§9.2). `None` for phase-plane.
    pub fn settling_time_s(&self, mode: ControlMode) -> Option<f64> {
        self.base_law(mode).settling_time_s(max_inertia(&self.inertia_ref_kgm2))
    }

    pub fn effective(&self) -> AttitudeControlEffective {
        let i_max = max_inertia(&self.inertia_ref_kgm2);
        AttitudeControlEffective {
            wheels: law_report(&self.wheels, self.wheels_source, self.wheel_authority_nm, i_max),
            thrusters: law_report(&self.thrusters, self.thrusters_source, self.rcs_authority_nm, i_max),
            activities: self
                .activities
                .iter()
                .map(|(a, t)| ActivityTuningReport { activity: a.label().to_string(), bandwidth_scale: t.bandwidth_scale, deadband_scale: t.deadband_scale })
                .collect(),
            gain_scheduling: self.scheduling,
            inertia_ref_kgm2: [self.inertia_ref_kgm2.x, self.inertia_ref_kgm2.y, self.inertia_ref_kgm2.z],
            mass_ref_kg: self.mass_ref_kg,
            tick_s: self.tick_s,
            warnings: self.warnings.clone(),
        }
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// One schedule point for telemetry.
    pub fn schedule_point(&self, t_s: f64, mass_kg: f64, mode: ControlMode, activity: Activity, law: &AttitudeLaw) -> GainSchedulePoint {
        let i_max = max_inertia(&self.inertia_at(mass_kg));
        let (omega_n, zeta) = law.natural_frequency_and_damping(i_max).map(|(w, z)| (Some(w), Some(z))).unwrap_or((None, None));
        let (kp, kd) = law.pd_gains().map(|(p, d)| (Some(p), Some(d))).unwrap_or((None, None));
        GainSchedulePoint {
            t_s,
            mass_kg,
            control_mode: format!("{mode:?}"),
            activity: activity.label().to_string(),
            law: law.label().to_string(),
            kp,
            kd,
            ki: match law { AttitudeLaw::Pid { ki, .. } => Some(*ki), _ => None },
            deadband_deg: match law { AttitudeLaw::PhasePlane(p) => Some(p.deadband_rad.to_degrees()), _ => None },
            omega_n_radps: omega_n,
            zeta,
        }
    }
}

/// Damping-ratio target shared with `simulate.rs` — re-exported here so
/// callers reasoning about the derived laws don't reach into `simulate`.
pub const DERIVED_ZETA_TARGET: f64 = ZETA_TARGET;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MissionConfig;

    fn cfg() -> MissionConfig {
        toml::from_str::<MissionConfig>(include_str!("../config/mars_orbit.toml")).expect("preset parses")
    }

    /// Thruster-mode gains must be sized by the RCS authority, not the
    /// wheels' — the root cause. With the reference layout the
    /// RCS authority is far larger than the wheel torque, so the
    /// thruster-mode k_p must exceed the wheel-mode k_p (up to the shared
    /// bandwidth caps) and the loop must be well damped in both.
    #[test]
    fn thruster_law_is_sized_from_rcs_authority_and_well_damped() {
        let cfg = cfg();
        let sc = crate::simulate::build_spacecraft_properties(&cfg);
        let wheels = crate::simulate::wheel_cluster_from_hardware(&cfg);
        let (rcs, _) = crate::simulate::rcs_from_hardware(&cfg);
        let set = AttitudeControlSet::from_cfg(&cfg, &sc, &wheels, &rcs, 10.0, 10.0);
        assert!(set.rcs_authority_nm > set.wheel_authority_nm, "fixture: RCS {} vs wheels {}", set.rcs_authority_nm, set.wheel_authority_nm);
        let eff = set.effective();
        assert_eq!(eff.thrusters.law, "Pid");
        assert_eq!(eff.wheels.law, "Pd");
        assert!(eff.thrusters.kp.unwrap() >= eff.wheels.kp.unwrap());
        for r in [&eff.wheels, &eff.thrusters] {
            let z = r.zeta.unwrap();
            assert!((z - DERIVED_ZETA_TARGET).abs() < 1e-9, "{}: ζ = {z}", r.law);
        }
        // The PID integral limit keeps the integral term within authority.
        let ki = eff.thrusters.ki.unwrap();
        assert!((ki * eff.thrusters.integral_limit_rad_s.unwrap() - set.rcs_authority_nm).abs() < 1e-9);
    }

    /// Mode-scheduled tick (§10.5.1): deriving the thruster law at a 10×
    /// shorter burn tick lifts the `ω_n ≤ 2π/(15·tick)` cap, so the
    /// thruster-mode bandwidth (and k_p) must come out materially higher
    /// than at the cruise tick — this is what makes the RCS authority
    /// usable during a burn.
    #[test]
    fn thruster_law_derived_at_the_burn_tick_gets_higher_bandwidth() {
        let cfg = cfg();
        let sc = crate::simulate::build_spacecraft_properties(&cfg);
        let wheels = crate::simulate::wheel_cluster_from_hardware(&cfg);
        let (rcs, _) = crate::simulate::rcs_from_hardware(&cfg);
        let coarse = AttitudeControlSet::from_cfg(&cfg, &sc, &wheels, &rcs, 10.0, 10.0);
        let fast = AttitudeControlSet::from_cfg(&cfg, &sc, &wheels, &rcs, 10.0, 1.0);
        assert!((fast.tick_for(ControlMode::ThrustersPrimary) - 1.0).abs() < 1e-12);
        assert!((fast.tick_for(ControlMode::WheelsPrimary) - 10.0).abs() < 1e-12);
        let i = sc.inertia_diag_kgm2.x.max(sc.inertia_diag_kgm2.y).max(sc.inertia_diag_kgm2.z);
        let wn_coarse = coarse.effective().thrusters.omega_n_radps.unwrap();
        let wn_fast = fast.effective().thrusters.omega_n_radps.unwrap();
        assert!(wn_fast > 2.0 * wn_coarse, "burn-tick ω_n {wn_fast} should exceed cruise-tick {wn_coarse} substantially");
        // Wheel law unchanged by the burn tick.
        let hold_c = coarse.law_for(ControlMode::WheelsPrimary, Activity::Hold, sc.mass_kg);
        let hold_f = fast.law_for(ControlMode::WheelsPrimary, Activity::Hold, sc.mass_kg);
        assert_eq!(hold_c.pd_gains(), hold_f.pd_gains());
        let _ = i;
    }

    /// Layer 2: BurnHold keeps full bandwidth with a tighter deadband,
    /// Coast halves the bandwidth (ζ invariant) and doubles the deadband.
    #[test]
    fn activity_tuning_applies_on_top_of_the_base_law() {
        let cfg = cfg();
        let sc = crate::simulate::build_spacecraft_properties(&cfg);
        let wheels = crate::simulate::wheel_cluster_from_hardware(&cfg);
        let (rcs, _) = crate::simulate::rcs_from_hardware(&cfg);
        let set = AttitudeControlSet::from_cfg(&cfg, &sc, &wheels, &rcs, 10.0, 10.0);
        let i = sc.inertia_diag_kgm2.x.max(sc.inertia_diag_kgm2.y).max(sc.inertia_diag_kgm2.z);
        let hold = set.law_for(ControlMode::WheelsPrimary, Activity::Hold, sc.mass_kg);
        let coast = set.law_for(ControlMode::WheelsPrimary, Activity::Coast, sc.mass_kg);
        let (wn_h, z_h) = hold.natural_frequency_and_damping(i).unwrap();
        let (wn_c, z_c) = coast.natural_frequency_and_damping(i).unwrap();
        assert!((wn_c / wn_h - 0.5).abs() < 1e-9);
        assert!((z_c - z_h).abs() < 1e-9);
    }

    /// Layer 3: with `inertia_scales_with_mass` off (the default, matching
    /// the constant-inertia truth model) mass changes never trigger a
    /// re-derivation; switched on, a 10% mass drop re-derives and the
    /// gains scale with the modeled inertia (k_p ∝ I at fixed ω_n cap).
    #[test]
    fn gain_scheduling_follows_the_configured_inertia_model() {
        let mut cfg = cfg();
        let sc = crate::simulate::build_spacecraft_properties(&cfg);
        let wheels = crate::simulate::wheel_cluster_from_hardware(&cfg);
        let (rcs, _) = crate::simulate::rcs_from_hardware(&cfg);
        let off = AttitudeControlSet::from_cfg(&cfg, &sc, &wheels, &rcs, 10.0, 10.0);
        assert!(!off.mass_trigger(sc.mass_kg, 0.5 * sc.mass_kg));
        cfg.gnc.attitude_control.gain_scheduling.inertia_scales_with_mass = Some(true);
        let on = AttitudeControlSet::from_cfg(&cfg, &sc, &wheels, &rcs, 10.0, 10.0);
        assert!(on.mass_trigger(sc.mass_kg, 0.9 * sc.mass_kg));
        assert!(!on.mass_trigger(sc.mass_kg, 0.99 * sc.mass_kg));
        let full = on.law_for(ControlMode::ThrustersPrimary, Activity::BurnHold, sc.mass_kg);
        let light = on.law_for(ControlMode::ThrustersPrimary, Activity::BurnHold, 0.5 * sc.mass_kg);
        let (kp_full, _) = full.pd_gains().unwrap();
        let (kp_light, _) = light.pd_gains().unwrap();
        assert!(kp_light < kp_full, "lighter vehicle → smaller derived k_p: {kp_light} vs {kp_full}");
    }

    /// Config overrides win over derivation and are reported as such.
    #[test]
    fn config_override_is_honored_and_reported() {
        let mut cfg = cfg();
        cfg.gnc.attitude_control.thrusters = Some(AttitudeLawConfig::PhasePlane {
            deadband_deg: Some(1.0), rate_deadband_degs: None, hysteresis_deg: None, min_on_time_s: Some(0.05), lead_time_s: None,
        });
        let sc = crate::simulate::build_spacecraft_properties(&cfg);
        let wheels = crate::simulate::wheel_cluster_from_hardware(&cfg);
        let (rcs, _) = crate::simulate::rcs_from_hardware(&cfg);
        let set = AttitudeControlSet::from_cfg(&cfg, &sc, &wheels, &rcs, 10.0, 10.0);
        let eff = set.effective();
        assert_eq!(eff.thrusters.law, "PhasePlane");
        assert_eq!(eff.thrusters.source, "config");
        assert!((eff.thrusters.deadband_deg.unwrap() - 1.0).abs() < 1e-12);
        assert!((eff.thrusters.min_on_time_s.unwrap() - 0.05).abs() < 1e-12);
        assert!(eff.thrusters.lead_time_s.unwrap() > 0.0);
    }
}
