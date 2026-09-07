//! Phase 9 trajectory optimization stage (Layer 1b) — real propagated-dynamics
//! optimization. Distinct from the narrowing stage's closed-form Lambert-proxy
//! GA/PSO/MonteCarlo (`design.rs`): the departure burn here is *not* anchored
//! to a Lambert-solved v-infinity (that would make the search trivially
//! converge to the analytic Hohmann-like minimum — not a meaningful test of
//! global search). Instead the burn is described by independent, freely
//! searchable numbers — see `evaluate_candidate`'s doc comment — and reaching
//! the target at all is a genuine, nontrivial search outcome.
//!
//! `MGA` delegates to `crate::mga::run_mga` (Phase 9d-9g; DE/rand/1/bin inner
//! optimizer, two-phase search). `MultipleShooting` is still a stub.

use ephemeris::{Almanac, Body};
use nalgebra::Vector3;

use crate::mga;
use trajectory_solver::{
    circular_orbit_burn_state, eccentricity, find_inbound_radius_crossing, keplerian::MU_SUN_M3S2, laplace_soi_radius_m,
    orbital_period_s, propagate, semi_major_axis_m, GaSolver, HohmannSolver, PropagatedPoint, PropagatorBody, PsoSolver,
};

use crate::config::{
    BodyRole, DepartureMode, EphemerisSource, GravityModel as CfgGravityModel, MissionConfig, MissionObjective,
    ObjectiveFunction, OptimizationConfig, OptimizationMethod,
};
use crate::design::{
    anise_body, arc_to_api, as_propagator_bodies, body_state, compute_launch_vehicle_check_for, departure_mode,
    departure_onboard_cost_ms, dv_ledger, epoch_to_jd, launch_frame, launch_geometry_api, launch_geometry_for,
    parse_epoch, resolve_parking_orbit_radius_m, ArcApiPoint, DvLedgerApiResult, LaunchGeometryApiResult,
    LaunchVehicleCheckApiResult, PropagatorBodyEntry,
};
use trajectory_solver::{EquatorialFrame, LaunchGeometry};

/// `Launch`-mode departure asymptote from the chromosome's three departure
/// genes (Phase 14b): `v∞ = vinf · (cos DLA cos RLA x̂ + cos DLA sin RLA ŷ +
/// sin DLA ẑ)` in the departure body's equatorial frame — the same
/// `(vinf, θ, φ)` triple MGA's chromosome already carries, so the two
/// searches describe a `Launch`-mode departure identically.
fn launch_v_inf_vec(frame: &EquatorialFrame, vinf_ms: f64, rla_rad: f64, dla_rad: f64) -> Vector3<f64> {
    (frame.x * (dla_rad.cos() * rla_rad.cos()) + frame.y * (dla_rad.cos() * rla_rad.sin()) + frame.z * dla_rad.sin())
        * vinf_ms
}

/// The closed-form launch geometry the single-leg chromosome
/// `[dep_offset, rla_rad, vinf_ms, dla_rad]` implies in `Launch` mode —
/// `None` in `ParkingOrbit` mode (the caller then uses the free burn genes).
fn launch_geometry_for_params(cfg: &MissionConfig, body: &body_models::TargetBody, params: &[f64]) -> Option<LaunchGeometry> {
    if departure_mode(cfg) != DepartureMode::Launch {
        return None;
    }
    let frame = launch_frame(body)?;
    launch_geometry_for(cfg, body, launch_v_inf_vec(&frame, params[2], params[1], params[3]))
}

/// The departure ΔV the SEARCH charges for a candidate (Phase 14b):
/// `ParkingOrbit` mode — the burn gene itself (`params[2]`, the spacecraft's
/// own injection burn); `Launch` mode — only the onboard perigee top-up
/// beyond what the launch vehicle delivers at this mass, for the asymptote
/// `v∞ = params[2]` (`design::departure_onboard_cost_ms`; falls back to the
/// full tangential escape burn if the departure body doesn't resolve).
fn departure_cost_ms(ctx: &FitnessContext, params: &[f64]) -> f64 {
    if !ctx.launch_mode {
        return params[2];
    }
    departure_onboard_cost_ms(ctx.cfg, &ctx.opt.departure_body, params[2]).unwrap_or(params[2])
}

/// Post-capture-burn speed [m/s] at radius `r_m` from the target body for
/// the configured capture orbit — periapsis speed of an ellipse with
/// periapsis `r_m` and eccentricity `[trajectory.capture].capture_eccentricity`
/// (0 = circular): `√(μ(1+e)/r)` (vis-viva at periapsis, `a = r/(1−e)`).
/// The single-leg path's ONE source for the target speed the arrival burn is
/// priced against (`ArrivalCapture::dv_capture_ms`, `dv_capture_any_orbit_
/// ms`) and the post-burn state `post_capture_orbit_arc` is propagated from
/// — Phase 14f (was circular-only, disagreeing with `mga.rs::
/// arrival_dv_ms`/`design.rs::arrival_dv_for_objective_ms`, which honored
/// `e` already; `MANUAL.md` §13.5). Scaling the real arrival speed to
/// this value along the real direction is still always bound: specific
/// energy `v²/2 − μ/r = −μ(1−e)/(2r) < 0` for `e < 1` regardless of
/// direction.
fn capture_target_speed_mps(cfg: &MissionConfig, mu_target: f64, r_m: f64) -> f64 {
    let e = cfg.trajectory.capture.as_ref().map(|c| c.capture_eccentricity).unwrap_or(0.0);
    (mu_target * (1.0 + e) / r_m).sqrt()
}

/// Fixed GA/PSO hyperparameters not exposed in `[optimization.ga]`/`.pso]` —
/// same precedent as `design.rs::ga_solver()`/`pso_solver()`.
const TOURNAMENT_SIZE: usize = 3;
const SEED: u64 = 42;

/// Number of independent, distinctly-seeded phase-1 (flyby-only) restarts —
/// see `run_optimization_with_progress`'s multi-start doc comment. Each
/// restart gets its own fresh random population and a share of phase 1's
/// total generation budget; only the single best individual across all
/// restarts seeds phase 2's narrowed search.
const PHASE1_STARTS: usize = 4;

/// Floor for `[optimization.force_model.atol]`, mirroring `design.rs`'s
/// `PROPAGATOR_ATOL` clamp (Phase 8h). The GA/PSO fitness loop throws largely
/// unconstrained burns at the propagator every evaluation, and some of those
/// genuinely fly very close to a body (especially right around an SOI-patched
/// switch) -- a real near-singular 1/r^2 close encounter, which is what
/// actually drives `StepSizeUnderflow` (the error estimate not shrinking as
/// fast as the step does). Loosening `atol` doesn't fix the close encounter;
/// it gives the adaptive controller enough slack to tolerate the resulting
/// locally-large error instead of spiraling toward the underflow floor on a
/// candidate the GA was going to penalize and discard anyway.
const FORCE_MODEL_ATOL_FLOOR: f64 = 1e-3;

/// One evaluated individual, logged for every generation of every phase —
/// the raw data behind an x/y-scatter-colored-by-generation plot. `phase`
/// is `1` (flyby-only search) or `2` (real-objective refinement); see
/// `run_optimization_with_progress`'s two-phase doc comment.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PopulationLogRow {
    pub phase: u8,
    pub generation: usize,
    pub params: Vec<f64>,
    pub fitness: f64,
    /// This individual's own arrival burn [m/s] (real crossing burn, or the
    /// capture-into-any-orbit burn at its closest approach) — 
    /// recorded as an evaluation side-product for the outcome scatter
    /// plots (departure vs. arrival ΔV). `None` for an infeasible
    /// candidate, and for rows from before this field existed.
    pub dv_arrival_ms: Option<f64>,
    /// This individual's own achieved time of flight [days] (capture time,
    /// or closest-approach time when it never captured) — same 
    /// outcome recording (departure offset vs. coast time scatter).
    pub tof_days: Option<f64>,
}

/// Result of a real-dynamics GA/PSO search — mirrors `design.rs`'s
/// `GaPsoComputeResult`, but fitness was evaluated under real propagation.
pub struct OptimizeComputeResult {
    pub dep_jd_base: f64,
    pub bounds: Vec<(f64, f64)>,
    /// `[dep_offset_days, theta_burn_rad, dv_mps, phi_out_of_plane_rad]` —
    /// see `evaluate_candidate`'s doc comment.
    pub best_params: Vec<f64>,
    pub best_fitness: f64,
    /// Best-fitness-so-far per generation (GA) or iteration (PSO) — for GA
    /// this is phase 2's (the real-objective refinement's) history only;
    /// see `population_log` for both phases' full per-individual record.
    pub history: Vec<f64>,
    /// Phase 1's (flyby-only) best-closest-approach-so-far history [km] —
    /// empty for PSO, which isn't split into phases yet.
    pub phase1_history_km: Vec<f64>,
    /// Every evaluated individual across both phases — empty for PSO.
    pub population_log: Vec<PopulationLogRow>,
    pub miss_km: f64,
    /// The EXACT series the live convergence plot drew, persisted
    /// (so a reload shows the full history including the refined final
    /// fitness — the old `convergence` field is the
    /// internal normalized phase-2-only history, a different series
    /// entirely). One `[step, phase, value]` row per streamed sentinel-free
    /// step: `step` on the unified budget-bounded counter, `phase` 1/2/3
    /// (3 = the refinement entries incl. the final polished value), `value`
    /// in the objective's natural units.
    pub objective_history: Vec<[f64; 3]>,
}

struct FitnessContext<'a> {
    almanac: &'a Almanac,
    cfg: &'a MissionConfig,
    opt: &'a OptimizationConfig,
    dep_anise: Body,
    dep_jd_base: f64,
    /// Normalization references so the objective term combines as a
    /// dimensionless, O(1)-for-a-reasonable-trajectory quantity instead of
    /// raw units. Computed once in `build_context` from a coarse
    /// circular-orbit Hohmann approximation between the departure and
    /// target bodies' real heliocentric distances at `dep_jd_base` -- a
    /// deliberately rough reference scale (real transfers aren't circular,
    /// and this stage's own burns generally won't be Hohmann-like at all),
    /// not a targeting computation.
    refs: ObjectiveReferences,
    /// `[trajectory.departure].mode == Launch` (Phase 14b): the chromosome's
    /// departure slots are `(RLA, v∞, DLA)` and the injection is the closed-
    /// form launch geometry, not `circular_orbit_burn_state`.
    launch_mode: bool,
}

/// See `FitnessContext::refs`.
struct ObjectiveReferences {
    dv_ref_ms: f64,
    tof_ref_days: f64,
    miss_ref_m: f64,
    /// Hohmann departure impulse [m/s] ≈ the departure hyperbolic-excess
    /// speed (v∞) for this transfer, when the Hohmann reference resolved —
    /// kept separately from the aggregate `dv_ref_ms` so the
    /// GA can seed analytically-correct-ENERGY departure-burn candidates
    /// into its initial population (see `hohmann_energy_seeds`), which
    /// needs the departure component alone, not the departure+arrival sum.
    hohmann_dep_vinf_ms: Option<f64>,
}

/// Resolve a catalog body's zonal-harmonic fidelity, given the *requested*
/// fidelity level from `[[optimization.force_model.bodies]]`. The numeric
/// j2/j3/j4 coefficients always come from the `body_models` catalog (the
/// optimization schema only lets the user pick a fidelity *level*, not raw
/// constants) — falls back to point-mass with a warning when the catalog
/// doesn't have matching harmonic or pole data, same rule as
/// `design.rs::build_zonal_fidelity`.
fn zonal_fidelity_for(
    catalog: &body_models::TargetBody,
    requested: CfgGravityModel,
) -> Option<trajectory_solver::ZonalFidelity> {
    let (j2, j3, j4) = match (requested, &catalog.gravity) {
        (CfgGravityModel::PointMass, _) => return None,
        (CfgGravityModel::J2, body_models::GravityModel::J2 { j2 }) => (*j2, 0.0, 0.0),
        (CfgGravityModel::J2, body_models::GravityModel::J2J3J4 { j2, .. }) => (*j2, 0.0, 0.0),
        (CfgGravityModel::J2J3J4, body_models::GravityModel::J2J3J4 { j2, j3, j4 }) => (*j2, *j3, *j4),
        _ => {
            eprintln!(
                "Warning: '{}' requests {requested} gravity fidelity but the body catalog \
                 doesn't have matching harmonic data — using point-mass central gravity instead.",
                catalog.name,
            );
            return None;
        }
    };
    let (Some(ra), Some(dec)) = (catalog.pole_ra_deg, catalog.pole_dec_deg) else {
        eprintln!(
            "Warning: '{}' requests {requested} gravity fidelity but has no pole orientation \
             data in the catalog — propagating its central-body gravity as point-mass.",
            catalog.name,
        );
        return None;
    };
    Some(trajectory_solver::ZonalFidelity {
        r0_m: catalog.radius_m, j2, j3, j4,
        pole_ra_rad: ra.to_radians(), pole_dec_rad: dec.to_radians(),
    })
}

/// Builds one `CentralWhenInSoi` SOI-candidate entry for `name`, sizing its
/// Laplace SOI radius from its real distance to its own primary (Sun, unless
/// the catalog gives a different primary — e.g. Earth for the Moon) at
/// `dep_jd`. Shared by the explicit `[[optimization.force_model.bodies]]`
/// list and the auto-included departure/target bodies below.
fn central_when_in_soi_entry<'a>(
    name: &str,
    fidelity: Option<CfgGravityModel>,
    almanac: &'a Almanac,
    dep_jd: f64,
) -> Option<PropagatorBodyEntry<'a>> {
    let catalog = body_models::TargetBody::by_name(name)?;
    let anise_b = anise_body(&name.to_lowercase())?;
    let (target_r, _) = body_state(almanac, EphemerisSource::Anise, Some(anise_b), &None, dep_jd)?;
    let (primary_mu, primary_r) = match catalog.primary {
        None => (MU_SUN_M3S2, [0.0, 0.0, 0.0]),
        Some(primary_name) => {
            let primary_catalog = body_models::TargetBody::by_name(primary_name)?;
            let primary_anise = anise_body(&primary_name.to_lowercase())?;
            let (pr, _) = body_state(almanac, EphemerisSource::Anise, Some(primary_anise), &None, dep_jd)?;
            (primary_catalog.mu_m3s2, pr)
        }
    };
    let (dx, dy, dz) = (target_r[0] - primary_r[0], target_r[1] - primary_r[1], target_r[2] - primary_r[2]);
    let dist_m = (dx * dx + dy * dy + dz * dz).sqrt();
    let soi_radius_m = trajectory_solver::laplace_soi_radius_m(dist_m, catalog.mu_m3s2 / primary_mu);
    let central_fidelity = fidelity.and_then(|f| zonal_fidelity_for(&catalog, f));
    let name_owned = name.to_string();
    let name_for_closure = name_owned.clone();

    Some(PropagatorBodyEntry {
        name: name_owned,
        mu_m3s2: catalog.mu_m3s2,
        soi_radius_m: Some(soi_radius_m),
        central_fidelity,
        state_at: Box::new(move |t_abs_s: f64| {
            let jd = dep_jd + t_abs_s / 86_400.0;
            let (r, v) = body_state(almanac, EphemerisSource::Anise, Some(anise_b), &None, jd)
                .unwrap_or_else(|| panic!("body '{name_for_closure}' ephemeris unavailable at jd={jd}"));
            (Vector3::new(r[0], r[1], r[2]), Vector3::new(v[0], v[1], v[2]))
        }),
        radius_m: Some(catalog.radius_m),
    })
}

/// Build the propagator's SOI-candidate/third-body perturber list for this
/// leg. The departure body and target body are *always* registered as
/// `CentralWhenInSoi` candidates — not optional, not something the user
/// needs to remember to add to `[[optimization.force_model.bodies]]` — since
/// real departure-leg (escape from a parking orbit) and arrival-leg (SOI-
/// patched approach) propagation both depend on them being present. Anything
/// the user *does* list in `force_model.bodies` is added on top as additional
/// third-body perturbers / SOI candidates (e.g. Jupiter for a Mars transfer);
/// an explicit entry for the departure or target body itself (e.g. to
/// request non-default central gravity fidelity) takes precedence over the
/// automatic point-mass-fidelity one.
///
/// `dep_jd` must be the actual departure JD for this leg, same epoch
/// contract as `design.rs::propagator_body_entries`.
fn force_model_body_entries<'a>(opt: &OptimizationConfig, almanac: &'a Almanac, dep_jd: f64) -> Vec<PropagatorBodyEntry<'a>> {
    let mut entries = Vec::new();
    let mut seen_names: Vec<String> = Vec::new();

    for b in &opt.force_model.bodies {
        let Some(catalog) = body_models::TargetBody::by_name(&b.name) else { continue };
        let Some(anise_b) = anise_body(&b.name.to_lowercase()) else { continue };

        match b.role {
            BodyRole::AlwaysThirdBody => {
                let name_for_closure = b.name.clone();
                entries.push(PropagatorBodyEntry {
                    name: b.name.clone(),
                    mu_m3s2: catalog.mu_m3s2,
                    soi_radius_m: None,
                    central_fidelity: None,
                    state_at: Box::new(move |t_abs_s: f64| {
                        let jd = dep_jd + t_abs_s / 86_400.0;
                        let (r, v) = body_state(almanac, EphemerisSource::Anise, Some(anise_b), &None, jd)
                            .unwrap_or_else(|| panic!("third-body perturber '{name_for_closure}' ephemeris unavailable at jd={jd}"));
                        (Vector3::new(r[0], r[1], r[2]), Vector3::new(v[0], v[1], v[2]))
                    }),
                    radius_m: Some(catalog.radius_m),
                });
                seen_names.push(b.name.to_lowercase());
            }
            BodyRole::CentralWhenInSoi => {
                if let Some(entry) = central_when_in_soi_entry(&b.name, b.fidelity, almanac, dep_jd) {
                    entries.push(entry);
                    seen_names.push(b.name.to_lowercase());
                }
            }
        }
    }

    for auto_name in [&opt.departure_body, &opt.target_body] {
        if seen_names.iter().any(|n| n.eq_ignore_ascii_case(auto_name)) {
            continue;
        }
        if let Some(entry) = central_when_in_soi_entry(auto_name, None, almanac, dep_jd) {
            entries.push(entry);
            seen_names.push(auto_name.to_lowercase());
        }
    }

    entries
}

/// Real arrival/capture burn, computed when the mission actually needs one
/// (anything but `Flyby`) — see `evaluate_candidate`.
#[derive(Clone)]
struct ArrivalCapture {
    /// Tangential insertion ΔV at the configured target orbit radius [m/s] —
    /// `|hyperbolic-approach speed at that radius − capture_target_speed_mps|`
    /// (the configured capture orbit's periapsis speed, eccentricity
    /// included — Phase 14f), the same vis-viva-difference idea as the
    /// departure escape burn, mirrored for arrival.
    dv_capture_ms: f64,
    /// Absolute mission time of the capture burn [s since departure].
    capture_time_s: f64,
    /// Arrival-side mirror of `theta_burn`/`phi_out_of_plane` — see
    /// `arrival_burn_angles`'s doc comment. `None`'d at the `Option<ArrivalCapture>`
    /// level (not here) when no capture happened at all.
    theta_arr_rad: f64,
    phi_arr_rad: f64,
    /// Crossing-point geometry, target-body-centered [m, m/s] — kept here
    /// (not just folded into the angles above) so `write_departure_geometry_csvs`
    /// can draw the real capture-orbit ring and burn vectors without
    /// re-deriving the crossing a second time.
    r_rel_m: Vector3<f64>,
    v_rel_mps: Vector3<f64>,
}

/// Arrival-side mirror of `theta_burn`/`phi_out_of_plane` (`evaluate_candidate`'s
/// departure search parameters) — but *derived* from the real propagated
/// crossing geometry, not a free search variable. Reference plane is the
/// TARGET body's own instantaneous heliocentric orbital plane, deliberately
/// mirroring the departure side's convention exactly (which uses the
/// departure body's own heliocentric orbital plane) — this makes both
/// quantities read as "how much out-of-plane work is this end of the
/// transfer doing," symmetric on both legs.
///
/// Deliberately NOT the local capture-orbit plane (`r_rel x v_rel` at the
/// crossing): that plane is exactly the plane `v_rel` already lies in by
/// construction, so measuring `v_rel`'s "out-of-plane angle" against its own
/// osculating plane would be tautologically zero — not a meaningful
/// diagnostic. (The local capture-orbit plane is still the *right* one to use
/// separately for drawing the resulting orbit ring — a different purpose,
/// see `write_departure_geometry_csvs`.)
fn arrival_burn_angles(
    r_rel_m: Vector3<f64>,
    v_rel_mps: Vector3<f64>,
    target_r_helio: Vector3<f64>,
    target_v_helio: Vector3<f64>,
) -> (f64, f64) {
    let n_hat = target_r_helio.cross(&target_v_helio).normalize();
    // Orthogonalize against n_hat before normalizing -- exact already on the
    // departure side (r_dep is automatically perpendicular to r_dep x v_dep
    // by construction), but not guaranteed here since `r_rel_m`/`v_rel_mps`
    // are body-centered crossing-point quantities, unrelated geometrically
    // to the target's own heliocentric radial direction. Defensive, not a
    // correction for an expected large deviation.
    let ref_raw = target_r_helio.normalize();
    let ref_dir = (ref_raw - n_hat * n_hat.dot(&ref_raw)).normalize();
    let p_hat = r_rel_m.normalize();
    let theta_arr = p_hat.dot(&n_hat.cross(&ref_dir)).atan2(p_hat.dot(&ref_dir));

    let t_hat = n_hat.cross(&p_hat);
    let v_radial = v_rel_mps.dot(&p_hat);
    let v_tangential = v_rel_mps.dot(&t_hat);
    let v_normal = v_rel_mps.dot(&n_hat);
    let phi_arr = v_normal.atan2((v_radial * v_radial + v_tangential * v_tangential).sqrt());

    (theta_arr, phi_arr)
}

/// Number of full orbital periods to propagate and return for a captured
/// orbit (Phase 12c) — "1 or 2 orbits is sufficient" per the Phase 12
/// canonical design statement).
const POST_CAPTURE_ORBIT_PERIODS: f64 = 2.0;

/// Analytic summary of a [`propagate_captured_orbit`] orbit — derived from
/// the REAL post-burn `(r0, v0)` state via standard two-body orbital
/// mechanics (`orbital_math::semi_major_axis_m`/`eccentricity`), not from an
/// idealized/config-assumed shape. Exact (no sampling error): computed from
/// the same closed-form state the propagation itself was seeded with.
struct CapturedOrbitSummary {
    period_s: f64,
    periapsis_m: f64,
    apoapsis_m: f64,
    eccentricity: f64,
    /// Semi-major axis [m] — same `semi_major_axis_m(r0, v0)` the period is
    /// derived from (final-orbit Kepler elements in
    /// the output).
    sma_m: f64,
    /// Inclination [deg] of the orbit plane to the reference frame's own
    /// fundamental plane (the frame every state in this pipeline is
    /// expressed in — J2000), from `acos(h_z/|h|)`.
    inclination_deg: f64,
}

/// Phase 12c; rewritten to fly the REAL post-burn
/// state instead of an idealized re-seed (
/// take the real trajectory, and use the arrival burn from the optimizer to
/// find a real orbit around the target. It can be any kind of orbit (e<1)").
///
/// **What changed and why.** The ORIGINAL version took `(r_hat,
/// plane_normal_hat, r_cap_m, eccentricity)` and RECONSTRUCTED an idealized
/// state from scratch: `v0 = (plane_normal_hat × r_hat) * sqrt(mu*(1+e)/r_cap)`
/// — a purely tangential velocity, as if the real crossing were exactly at
/// periapsis and the configured `eccentricity` were what the burn actually
/// achieved. This discarded the REAL arrival velocity's direction entirely,
/// so the resulting ring was only ever a "what a textbook-perfect capture
/// burn would look like here" illustration — genuinely disconnected from
/// the incoming trajectory whenever the real crossing wasn't near-periapsis
/// or the real burn was large (confirmed as the root cause of a user-
/// reported visible kink/angle between the transfer arc and the arrival
/// orbit, since the idealized velocity direction and the real one only
/// coincide by chance).
///
/// Every call site now passes the REAL post-burn state directly: `r0` is
/// the real propagated crossing position (unchanged, still exactly where
/// the incoming arc left off — no seam), and `v0` is the real crossing
/// velocity's OWN direction, scaled down in magnitude to the local circular
/// speed at that radius (`v0 = v_rel.normalize() * sqrt(mu/r0.norm())`,
/// computed at each call site) — the same single-impulse, minimum-ΔV
/// "just shed the right amount of speed" assumption `dv_capture_ms` already
/// prices (`|v_rel| - v_circ`, unchanged), just applied along the REAL
/// direction instead of a synthetic tangential one. This is provably always
/// a BOUND result regardless of how far the real crossing is from periapis:
/// specific energy `v^2/2 - mu/r` depends only on speed and radius, never on
/// direction, so scaling |v| down to local circular speed always yields
/// negative energy (e < 1) — proven and regression-tested in
/// `orbital_math::kepler::scaling_speed_to_local_circular_speed_is_bound_
/// regardless_of_direction`. The resulting orbit's actual shape (its real
/// eccentricity, no longer tied to any configured value) now comes
/// entirely from the real geometry: a near-tangential real crossing yields
/// a near-circular result (as before); a real crossing with a genuine
/// radial component yields a genuinely elliptical orbit whose periapsis is
/// the REAL crossing point — "any kind of orbit (e<1)," per the request,
/// rather than a fixed idealized shape.
///
/// A caller that still wants the OLD idealized-tangential-at-a-fixed-
/// eccentricity construction (the departure/parking-orbit call sites, which
/// have no "real burn" concept at all, and the MGA no-real-crossing
/// fallback) builds that `(r0, v0)` itself before calling this — the
/// idealized-seed logic itself is unchanged, just moved to the two/three
/// remaining call sites that still need it, since this function's own job
/// is now purely "propagate whatever real or idealized state you hand it
/// and summarize the result," not "decide what state to use."
///
/// Point-mass, target-body-centered only, no perturbers — a captured orbit
/// this close to the body is dominated overwhelmingly by its own gravity,
/// same reasoning already applied to Phase 7k's body-centric LOI arcs
/// (`design.rs`).
fn propagate_captured_orbit(
    r0: Vector3<f64>,
    v0: Vector3<f64>,
    mu_target: f64,
) -> (Vec<PropagatedPoint>, CapturedOrbitSummary) {
    let a = semi_major_axis_m(&r0, &v0, mu_target);
    let ecc = eccentricity(&r0, &v0, mu_target);
    let period_s = orbital_period_s(a, mu_target);
    let duration_s = POST_CAPTURE_ORBIT_PERIODS * period_s;
    let sample_dt_s = (duration_s / 200.0).max(1.0);
    let points = propagate(r0, v0, 0.0, duration_s, mu_target, &[], sample_dt_s, 1e-9, 1e-3);
    let h = r0.cross(&v0);
    let inclination_deg = (h.z / h.norm().max(1e-12)).clamp(-1.0, 1.0).acos().to_degrees();
    let summary = CapturedOrbitSummary {
        period_s,
        periapsis_m: a * (1.0 - ecc),
        apoapsis_m: a * (1.0 + ecc),
        eccentricity: ecc,
        sma_m: a,
        inclination_deg,
    };
    (points, summary)
}

/// Convert `propagate_captured_orbit`'s target-body-RELATIVE points into
/// `ArcApiPoint`s in the SAME heliocentric frame `arc` uses — `origin_m` is
/// the body's own real heliocentric position at the relevant epoch (each
/// call site's own responsibility to supply correctly; see its callers
/// below), added to every point before returning.
///
/// Fixed this function used to return
/// the raw target-body-relative points unmodified, with `arc`'s own
/// heliocentric frame documented as "a SEPARATE set of points, not appended
/// to it." That forced any consumer wanting one continuous path (e.g. the
/// cruise-loop replay's reference trajectory) to either drop these points
/// or mishandle the frame mismatch — confirmed as the real cause of a
/// captured Mercury orbit reading as a flyby in a downstream replay, since
/// the reference builder had no way to safely concatenate a body-relative
/// ring onto a heliocentric transfer arc. `central_body` is still always
/// the target body's own name (unlike `design.rs::arc_to_api`'s
/// heliocentric convention, whose `None` case falls back to `"Sun"`) —
/// that field always meant "which body's gravity is locally dominant here,"
/// never "which frame this position is expressed in," and stays that way;
/// only the position values themselves changed.
/// `origin_v_mps`: the origin body's own real heliocentric
/// velocity at the same epoch `origin_m` was sampled at -- added to each
/// point's body-relative `v_mps` for the same reason `origin_m` is added to
/// `r_m`, so `vx_mps`/`vy_mps`/`vz_mps` come out in the same heliocentric
/// frame as `x_m`/`y_m`/`z_m`.
///
/// **Time-resolved as of **
/// the version translated every point by ONE origin state (the
/// body at a single epoch), so the served "orbit" did not move with its
/// body over its own time span — measured against the real Mercury
/// snapshot, the pre-departure arc spanned 2.95 h while the real Earth
/// moved ~316,000 km (~50 Earth radii), which put a Phase 03 replay's
/// starting state ~316,000 km away from where its own gravity model had
/// the Earth. Each point is now translated by the body's REAL state at
/// that point's own epoch, supplied by `body_state_at(dt_s)` — the offset
/// in seconds from the anchor sample (`t_s == t_s_anchor`, whose epoch the
/// caller knows: `0.0` for a post-capture arc anchored at the capture
/// crossing, `points.last().t_s` for a pre-departure arc anchored at the
/// burn). A closure rather than an `Almanac` so the translation is unit-
/// testable against a synthetic moving body without kernels. When the
/// closure returns `None` (no ephemeris coverage — can't happen for
/// optimize targets today, which are ANISE-gated upstream), the old
/// frozen `fallback_origin_m`/`fallback_origin_v_mps` behavior applies,
/// point-for-point, rather than dropping the arc.
fn captured_orbit_to_api(
    points: Vec<PropagatedPoint>,
    target_body_name: &str,
    t_s_anchor: f64,
    body_state_at: impl Fn(f64) -> Option<(Vector3<f64>, Vector3<f64>)>,
    fallback_origin_m: Vector3<f64>,
    fallback_origin_v_mps: Vector3<f64>,
) -> Vec<ArcApiPoint> {
    points
        .into_iter()
        .map(|p| {
            let (origin_m, origin_v_mps) = body_state_at(p.t_s - t_s_anchor)
                .unwrap_or((fallback_origin_m, fallback_origin_v_mps));
            ArcApiPoint {
                t_s: p.t_s,
                x_m: p.r_m.x + origin_m.x,
                y_m: p.r_m.y + origin_m.y,
                z_m: p.r_m.z + origin_m.z,
                central_body: target_body_name.to_string(),
                leg_idx: None,
                vx_mps: Some(p.v_mps.x + origin_v_mps.x),
                vy_mps: Some(p.v_mps.y + origin_v_mps.y),
                vz_mps: Some(p.v_mps.z + origin_v_mps.z),
            }
        })
        .collect()
}

/// Full real-physics evaluation of one candidate — the single place this
/// happens, reused by the fitness function, the best-point re-evaluation for
/// the API result, and the CLI.
///
/// `params = [dep_offset_days, theta_burn_rad, dv_mps, phi_out_of_plane_rad]`:
/// - `dep_offset_days` — which day to depart (sets the departure/target
///   synodic alignment).
/// - `theta_burn_rad` — true anomaly (orbital phase) of the burn at a
///   circular parking orbit, in the departure body's own instantaneous
///   heliocentric orbital plane (a stand-in for "the ecliptic").
/// - `dv_mps` — burn magnitude.
/// - `phi_out_of_plane_rad` — burn direction within the local
///   tangential/normal plane (full range — handles the real plane change
///   needed to reach the target's own, generally differently-inclined,
///   orbit).
///
/// Deliberately *not* anchored to a Lambert-solved v-infinity: the
/// resulting trajectory is propagated continuously (one [`propagate`] call,
/// covering the real departure-body-centered escape, the heliocentric
/// coast, and — if the target's SOI is entered — the target-body-centered
/// approach) for up to `max_coast_days`, and scored by whatever it actually
/// achieves. Reaching the target at all is therefore a genuine search
/// outcome, not a foregone conclusion baked into the construction (contrast
/// the old Lambert-anchored design, which always converged to the analytic
/// Hohmann-like minimum and wasn't a meaningful test of global search).
struct CandidateEvaluation {
    points: Vec<PropagatedPoint>,
    escape_duration_s: f64,
    /// Real arrival/capture outcome — `None` either because the mission
    /// doesn't need one (`Flyby`) or because *this* candidate didn't
    /// achieve one. Critically, a missing arrival does NOT make the whole
    /// candidate infeasible: `MatchTargetDistance` scores `closest_approach_m`
    /// directly, which is always available regardless of capture, so the GA
    /// can still learn from trajectories that got "relatively close" and
    /// iterate toward an actual capture over generations — the entire point
    /// of using a GA instead of requiring an exact hit immediately. Only
    /// `MinDeltaV`/`MinTof` (which need a real burn/completion time to be
    /// well-defined) treat a needed-but-missing arrival as infeasible, and
    /// they do that themselves in `real_dynamics_fitness`, not here.
    arrival: Option<ArrivalCapture>,
    /// Whether this mission needs a real arrival capture at all (`Flyby`: no).
    needs_capture: bool,
    tof_ref_days: f64,
    /// True minimum distance to the target body over the *entire* coast,
    /// re-querying the target's real position at each sample's own
    /// absolute epoch — not just the distance at the final point. Always
    /// available regardless of capture outcome -- this is the real,
    /// continuous gradient signal `MatchTargetDistance` (and a GA's whole
    /// "learn from near-misses" mechanism) depends on.
    closest_approach_m: f64,
    /// Time of closest approach [days] -- the natural "arrival" moment for
    /// a Flyby, and the fallback "how far did we get" time for a capturing
    /// mission that didn't actually capture.
    closest_approach_days: f64,
    /// Burn to capture into ANY bound orbit at the closest approach
    /// (circularize there, real relative state) [m/s] -- what MinDeltaV
    /// prices as the arrival cost since its radius-free
    /// redefinition. When the approach's osculating periapsis lies BEYOND
    /// the SOI capture ceiling (see `capture_rp_max_m`), this instead holds
    /// the FULL relative-speed match `|v_rel|` -- no fictitious
    /// circular-speed discount at a radius where a target-centric orbit
    /// can't physically exist.
    dv_capture_any_orbit_ms: f64,
    /// Osculating periapsis radius [m] of the closest approach's
    /// target-relative orbit (the radius the any-orbit capture would
    /// happen at). f64::INFINITY when degenerate/unresolvable.
    capture_rp_m: f64,
    /// Maximum physically-meaningful capture radius [m]:
    /// `CAPTURE_MAX_SOI_FRACTION` x the target's Laplace SOI radius
    /// (a MinDeltaV run once "captured" at 27.2M km, outside
    /// Mercury's own SOI -- an orbit that cannot exist, so the maximum
    /// capture radius is capped to a fraction of the SOI).
    /// `None` when the target's SOI radius can't be resolved (no cap
    /// applied -- absence of information must not manufacture
    /// infeasibility).
    capture_rp_max_m: Option<f64>,
}

/// Fraction of the target's Laplace SOI radius inside which a synthesized
/// any-orbit capture is considered physically meaningful. Was 0.5 for a
/// few hours on immediately caught a real 57,000 km Mercury
/// approach (comfortably inside the ~112,000 km SOI) and refused to
/// capture it (
/// within the SOI... You can just make that 1.0*SOI"). 1.0 = the SOI
/// boundary itself is the feasibility line; orbits NEAR the boundary are
/// still tide-fragile in reality, but that's a fidelity judgment for
/// Phase 03's real multi-body replay, not something to silently pre-empt
/// here.
const CAPTURE_MAX_SOI_FRACTION: f64 = 1.0;

/// ΔV pressure weight inside MinDeltaV's hard infeasibility band (the
/// `10 + ln(rp/ceiling)` region for capture-needing candidates beyond the
/// SOI) — small, so approach distance stays the band's primary gradient
/// (the same architecture the old configured-radius gate used), while
/// equally-far candidates still converge through cheap corridors.
const INFEASIBLE_BAND_DV_WEIGHT: f64 = 0.2;

/// Analytic geocentric hyperbolic excess velocity (v-infinity) for a burn at
/// periapsis `(r0, v0)` -- `None` if the orbit isn't actually hyperbolic
/// (`e <= 1`, e.g. an unphysical/degenerate input). `r0 . v0 == 0` is assumed
/// (true for `circular_orbit_burn_state`'s output regardless of `phi`, since
/// both `t_hat` and `n_hat` are perpendicular to the position direction), so
/// the standard eccentricity-vector formula applies directly. Same rotation
/// convention as `departure.rs`'s Rodrigues formula. Shared by
/// `evaluate_candidate` (the net-prograde-heliocentric-energy filter below)
/// and `write_departure_geometry_csvs` (the geometry diagnostic) -- previously
/// duplicated between them.
fn asymptotic_v_infinity_geo(r0: Vector3<f64>, v0: Vector3<f64>, mu_m3s2: f64, r_park_m: f64) -> Option<Vector3<f64>> {
    let v_inf_mag = (v0.norm_squared() - 2.0 * mu_m3s2 / r_park_m).max(0.0).sqrt();
    if v_inf_mag < 1e-3 {
        return None;
    }
    let h_vec = r0.cross(&v0);
    let h_hat = h_vec.normalize();
    let e_vec = v0.cross(&h_vec) / mu_m3s2 - r0.normalize();
    let e = e_vec.norm();
    if e <= 1.0 {
        return None;
    }
    let nu_inf = (-1.0 / e).acos();
    let r0_hat = r0.normalize();
    let v_inf_hat = r0_hat * nu_inf.cos() + h_hat.cross(&r0_hat) * nu_inf.sin();
    Some(v_inf_hat * v_inf_mag)
}

fn evaluate_candidate(ctx: &FitnessContext, params: &[f64]) -> Option<CandidateEvaluation> {
    let dep_offset_days = params[0];
    let theta_burn = params[1];
    let dv_mps = params[2];
    let phi = params[3];

    let dep_jd = ctx.dep_jd_base + dep_offset_days;
    let (r_dep, v_dep) = body_state(ctx.almanac, EphemerisSource::Anise, Some(ctx.dep_anise), &None, dep_jd)?;
    let r_dep_v = Vector3::new(r_dep[0], r_dep[1], r_dep[2]);
    let v_dep_v = Vector3::new(v_dep[0], v_dep[1], v_dep[2]);

    let dep_catalog = body_models::TargetBody::by_name(&ctx.opt.departure_body)?;
    let r_park_m = resolve_parking_orbit_radius_m(ctx.cfg, &dep_catalog);

    // "Ecliptic" proxy: the departure body's own instantaneous heliocentric
    // orbital plane. `r_dep_v` is automatically perpendicular to this
    // normal by construction (cross product), so it's already a valid
    // in-plane zero-point -- no separate reference-frame bookkeeping needed.
    let plane_normal_hat = r_dep_v.cross(&v_dep_v).normalize();
    let reference_dir_hat = r_dep_v.normalize();

    // `Launch` mode (Phase 14b): the three departure genes are (RLA, v∞,
    // DLA) — the asymptote itself — and the injection state is the closed-
    // form launch geometry on the site-feasible plane, not a free burn.
    let burn = if ctx.launch_mode {
        let g = launch_geometry_for_params(ctx.cfg, &dep_catalog, params)?;
        trajectory_solver::CircularOrbitBurn { r0_m: g.injection_r_m, v0_mps: g.injection_v_mps }
    } else {
        circular_orbit_burn_state(dep_catalog.mu_m3s2, r_park_m, plane_normal_hat, reference_dir_hat, theta_burn, dv_mps, phi)
    };

    // Cheap analytic check, before ever propagating: does this burn actually
    // escape the departure body (positive specific energy, hyperbolic), or
    // is it a bound ellipse? A bound orbit can never leave the departure
    // body's SOI through two-body dynamics alone -- propagating it for the
    // full `max_coast_days` anyway means computing potentially thousands of
    // orbital periods, which is both pointless (it will never reach the
    // target) and expensive enough to blow through the integrator's max
    // step count (confirmed empirically: `MaxNumStepReached` after 100,001
    // steps, for what turned out to be a bound orbit going nowhere).
    let specific_energy = 0.5 * burn.v0_mps.norm_squared() - dep_catalog.mu_m3s2 / r_park_m;
    if specific_energy <= 0.0 {
        return None;
    }

    // Second cheap analytic check, before ever propagating: does the real
    // escape asymptote (v-infinity, NOT the periapsis/burn velocity --
    // those differ by the hyperbola's turning angle, found to be
    // a real ~50-60 degree rotation for this binary's typical eccentricities,
    // so the periapsis velocity itself is the wrong thing to check) compose
    // with the departure body's own heliocentric velocity into a *net*
    // heliocentric departure velocity that's still broadly aligned with
    // that body's own motion, or does it point into the opposite
    // hemisphere entirely? The latter can only ever happen if v-infinity's
    // magnitude rivals or exceeds the departure body's own orbital speed
    // (~30 km/s for Earth) -- not reachable at this stage's `dv` bounds in
    // practice, but cheap to guard explicitly rather than rely on that
    // staying true if bounds are ever widened a lot. Rejecting here (an
    // explicit `None`, penalized to `f64::MAX` same as every other
    // infeasible candidate) keeps the search from ever rewarding a
    // pathological "net motion reversed relative to the departure body"
    // candidate, however unlikely.
    if let Some(v_inf_geo) = asymptotic_v_infinity_geo(burn.r0_m, burn.v0_mps, dep_catalog.mu_m3s2, r_park_m) {
        let v_transfer_helio = v_dep_v + v_inf_geo;
        if v_transfer_helio.dot(&v_dep_v) < 0.0 {
            return None;
        }
    }

    let r0_helio = r_dep_v + burn.r0_m;
    let v0_helio = v_dep_v + burn.v0_mps;

    let entries = force_model_body_entries(ctx.opt, ctx.almanac, dep_jd);
    let bodies = as_propagator_bodies(&entries);
    let dep_body_index = entries.iter().position(|e| e.name.eq_ignore_ascii_case(&ctx.opt.departure_body))?;
    let target_body_index = entries.iter().position(|e| e.name.eq_ignore_ascii_case(&ctx.opt.target_body))?;

    let atol = ctx.opt.force_model.atol.max(FORCE_MODEL_ATOL_FLOOR);
    let max_coast_s = ctx.opt.max_coast_days * 86_400.0;
    let sample_dt_s = (max_coast_s / 500.0).max(60.0);
    // NOTE: a low-energy (near-parabolic, small v-infinity) escape can end
    // up almost exactly at the departure body's SOI boundary right when the
    // escape leg ends. `propagate()`'s degenerate-leg guard (existing,
    // shared behavior -- prevents an infinite loop when SOI membership is
    // ambiguous right at the boundary) then ends the *whole* propagation
    // early rather than retrying the heliocentric leg, leaving a short,
    // truncated arc. This doesn't corrupt the result (closest_approach_m
    // below is correctly poor/penalized for a candidate that was barely
    // examined), just under-explores that specific candidate -- a narrow,
    // pre-existing propagator edge case, not something fixed here.
    let points = propagate(
        r0_helio, v0_helio, 0.0, max_coast_s, MU_SUN_M3S2, &bodies, sample_dt_s, ctx.opt.force_model.rtol, atol,
    );

    let escape_duration_s = points
        .iter()
        .find(|p| p.central_body_index != Some(dep_body_index))
        .map(|p| p.t_s)
        .unwrap_or(0.0);

    // One fold yields distance, time, AND index -- the any-orbit capture
    // pricing below reuses the index instead of re-scanning (
    // perf fix: the first cut
    // used a separate min_by whose comparator queried the target ephemeris
    // twice per comparison, ~1,000 redundant state_at calls per candidate
    // evaluation on top of this fold's own ~500).
    let (closest_approach_m, closest_t_s, closest_idx) =
        points.iter().enumerate().fold((f64::INFINITY, 0.0_f64, 0usize), |(min_d, min_t, min_i), (i, p)| {
            let (target_r, _) = (bodies[target_body_index].state_at)(p.t_s);
            let d = (p.r_m - target_r).norm();
            if d < min_d { (d, p.t_s, i) } else { (min_d, min_t, min_i) }
        });

    // Capture-into-ANY-orbit burn at the closest approach [m/s] (
    // MinDeltaV redefinition --
    // it will have total dv as objective, trying to get into any kind of
    // orbit regardless of the radius"): the burn that circularizes at the
    // achieved closest-approach radius, `||v_rel| - v_circ(r_ca)|` -- the
    // same convention `ArrivalCapture::dv_capture_ms` and the fitted
    // post-capture orbit already use, so the number optimized and the
    // orbit displayed agree. Priced at the candidate's OWN closest
    // approach, no configured radius anywhere: physics shapes this on its
    // own (capturing far from the body costs the full relative speed;
    // capturing very deep costs the Oberth-inverted excess -- a real
    // interior optimum exists between them).
    let capture_rp_max_m = bodies[target_body_index].soi_radius_m.map(|soi| soi * CAPTURE_MAX_SOI_FRACTION);
    let (dv_capture_any_orbit_ms, capture_rp_m) = points
        .get(closest_idx)
        .map(|p| {
            let (r_t, v_t) = (bodies[target_body_index].state_at)(p.t_s);
            let r_rel = p.r_m - r_t;
            let v_rel = p.v_mps - v_t;
            let mu_t = bodies[target_body_index].mu_m3s2;
            // Price at the osculating PERIAPSIS of the approach orbit, not
            // at the raw sample (same sampled-point radial-
            // velocity issue as the fitted capture orbit -- see
            // osculating_periapsis_state). Closed form, no extra
            // propagation; falls back to the sampled state when degenerate.
            let (rp, v_at_rp) = match osculating_periapsis_state(r_rel, v_rel, mu_t) {
                Some((r_p, v_p)) => (r_p.norm(), v_p.norm()),
                None => (r_rel.norm().max(1.0), v_rel.norm()),
            };
            // SOI capture ceiling (see capture_rp_max_m's doc
            // comment): beyond it there is no circular-speed discount --
            // a target-centric orbit can't exist there, so the honest
            // "arrival cost" is the full relative-speed match.
            let within_soi = capture_rp_max_m.map_or(true, |cap| rp <= cap);
            let dv = if within_soi { (v_at_rp - capture_target_speed_mps(ctx.cfg, mu_t, rp)).abs() } else { v_rel.norm() };
            (dv, rp)
        })
        .unwrap_or((0.0, f64::INFINITY));

    // A missing arrival does NOT make the candidate infeasible here --
    // see `CandidateEvaluation::arrival`'s doc comment for why. The only
    // thing computed here is "did this candidate happen to achieve one";
    // whether that absence matters is an `optimization.objective`-specific
    // decision made in `real_dynamics_fitness`.
    let needs_capture = !matches!(ctx.cfg.mission.objective, MissionObjective::Flyby);
    let arrival = needs_capture
        .then(|| ctx.cfg.trajectory.capture.as_ref().and_then(|c| c.target_orbit_radius_m))
        .flatten()
        .and_then(|target_radius_m| {
            let crossing = find_inbound_radius_crossing(&points, &bodies[target_body_index], target_radius_m)?;
            // Defensive hard check (real API bug report): a
            // genuine collision must never silently report as a converged
            // capture. `check_config` already rejects a configured
            // `target_radius_m` below the body's real radius, and the
            // propagator's own `resolve_collision` already stops
            // propagation before truly penetrating a body -- so this
            // should be unreachable in practice -- but the interpolation
            // fix above is exactly what closes the one real gap that made
            // it reachable before, so this stays as a real, checked safety
            // net rather than trusting those two guards alone.
            if let Some(real_radius_m) = bodies[target_body_index].radius_m {
                if crossing.r_rel_m.norm() < real_radius_m {
                    return None;
                }
            }
            let v_target = capture_target_speed_mps(ctx.cfg, bodies[target_body_index].mu_m3s2, target_radius_m);
            let (target_r, target_v) = (bodies[target_body_index].state_at)(crossing.t_s);
            let (theta_arr_rad, phi_arr_rad) =
                arrival_burn_angles(crossing.r_rel_m, crossing.v_rel_mps, target_r, target_v);
            Some(ArrivalCapture {
                dv_capture_ms: (crossing.v_rel_mps.norm() - v_target).abs(),
                capture_time_s: crossing.t_s,
                theta_arr_rad,
                phi_arr_rad,
                r_rel_m: crossing.r_rel_m,
                v_rel_mps: crossing.v_rel_mps,
            })
        });

    Some(CandidateEvaluation {
        points,
        escape_duration_s,
        arrival,
        needs_capture,
        tof_ref_days: ctx.refs.tof_ref_days,
        closest_approach_m,
        closest_approach_days: closest_t_s / 86_400.0,
        dv_capture_any_orbit_ms,
        capture_rp_m,
        capture_rp_max_m,
    })
}

/// Phase-1 fitness: minimize closest-approach distance only, ignoring
/// `optimization.objective` and whether the mission needs a real capture.
/// Real propagation gives a genuine but very sparse success signal -- most
/// random `(theta, dv, phi)` combinations either don't escape at all, or
/// escape pointed nowhere near the target, so a from-scratch search under
/// the *real* objective (especially `MinDeltaV`/`MinTof`, which are flatly
/// infeasible until a capture happens at all) has almost nothing to climb
/// early on. This phase finds a good escape direction/timing first --
/// always computable, since it's just a distance -- so phase 2 can start
/// its real-objective search already in the right neighborhood instead of
/// from random noise.
fn flyby_only_fitness(ctx: &FitnessContext, params: &[f64]) -> Option<f64> {
    evaluate_candidate(ctx, params).map(|e| e.closest_approach_m)
}

/// Shrink each parameter's bound to a window of `fraction` of its original
/// width, centered on `center[i]`, clamped back to the original bound.
/// Used to seed phase 2's search near phase 1's discovered region without
/// needing `GaSolver` to support an explicit seeded population. Note: for
/// the periodic angle parameters (theta, phi), this clamps rather than
/// wraps -- a phase-1 best sitting exactly at a 0/2π boundary would narrow
/// to the wrong side of it. Accepted as a minor edge case, not handled.
fn narrow_bounds(bounds: &[(f64, f64)], center: &[f64], fraction: f64) -> Vec<(f64, f64)> {
    bounds
        .iter()
        .zip(center)
        .map(|(&(lo, hi), &c)| {
            let half_width = (hi - lo) * fraction / 2.0;
            ((c - half_width).max(lo), (c + half_width).min(hi))
        })
        .collect()
}

/// How many independent phase-2 refinement runs to launch, each seeded from
/// a different, mutually-distant phase-1 candidate — after a
/// live MinDeltaV run converged to a 61 km/s solution: phase 2 used to
/// refine from phase 1's SINGLE best closest-approach candidate, so the
/// whole real-objective search lived and died inside whichever one basin
/// that candidate happened to sit in, regardless of cost. Same
/// multiple-diverse-seeds lesson the MGA path already learned the hard way
/// (the Cassini-2 "always infeasible without pruning seed diversity"
/// investigation — see the design notes).
const PHASE2_RESTARTS: usize = 3;

/// Minimum normalized (per-bound-span) Euclidean distance between two
/// phase-2 seeds — close-together seeds would just re-run the same basin
/// three times. 0.15 of the (4-dimensional) box diagonal-ish scale is far
/// enough to be a genuinely different search region, near enough that a
/// well-populated phase-1 archive usually has candidates to offer.
const PHASE2_SEED_MIN_NORMALIZED_DISTANCE: f64 = 0.15;

/// Pick up to `k` phase-1 candidates as phase-2 seeds: best-fitness-first,
/// greedily skipping any candidate within
/// [`PHASE2_SEED_MIN_NORMALIZED_DISTANCE`] of an already-picked one. The
/// single best phase-1 candidate is always picked first (it is by
/// construction the first ranked row), so this strictly generalizes the old
/// single-seed behavior rather than replacing it. Pure function over the
/// already-collected population log — unit-testable without an almanac.
fn select_diverse_phase2_seeds(rows: &[PopulationLogRow], bounds: &[(f64, f64)], k: usize) -> Vec<Vec<f64>> {
    let mut ranked: Vec<&PopulationLogRow> = rows
        .iter()
        .filter(|r| r.phase == 1 && r.fitness.is_finite() && r.fitness < f64::MAX && r.params.len() == bounds.len())
        .collect();
    ranked.sort_by(|a, b| a.fitness.total_cmp(&b.fitness));
    // Diversity AMONG CLOSE APPROACHERS only (second revision
    // same day --
    // anymore"): the first version picked mutually-distant seeds from the
    // ENTIRE phase-1 archive, so two of the three phase-2 refinement runs
    // typically spent their budget in basins that never approached the
    // target at all -- a real dilution of the old single-seed structure's
    // near-guaranteed deep refinement. Phase-1 fitness IS closest-approach
    // distance, so truncating the ranked pool to its closest quartile
    // (floor 3k rows) keeps every seed a genuine approacher while still
    // separating distinct approach basins.
    let pool = ranked.len().min((ranked.len() / 4).max(3 * k));
    ranked.truncate(pool);
    let normalized_distance = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .zip(bounds)
            .map(|((x, y), &(lo, hi))| {
                let span = (hi - lo).max(1e-12);
                ((x - y) / span).powi(2)
            })
            .sum::<f64>()
            .sqrt()
    };
    let mut picked: Vec<Vec<f64>> = Vec::new();
    for row in ranked {
        if picked.iter().all(|p| normalized_distance(p, &row.params) > PHASE2_SEED_MIN_NORMALIZED_DISTANCE) {
            picked.push(row.params.clone());
            if picked.len() == k {
                break;
            }
        }
    }
    picked
}

/// Number of evenly-spaced burn-location (theta) values to seed with the
/// analytic Hohmann-energy departure burn — see [`hohmann_energy_seeds`].
const HOHMANN_SEED_THETA_COUNT: usize = 8;

/// Analytically-correct-ENERGY departure-burn seeds for the GA's initial
/// population: the survey/Hohmann reference already knows the
/// transfer's departure v∞, and the parking-orbit injection burn that
/// achieves it is closed-form vis-viva
/// (`dv = sqrt(v∞² + 2μ/r_park) − sqrt(μ/r_park)` — the same Oberth
/// accounting `design.rs::departure_escape_dv_ms` uses). Random
/// initialization has to rediscover this energy scale by luck (observed
/// live: a converged 12.3 km/s departure burn for a transfer whose real
/// injection burn is ~5.5 km/s); these seeds hand it the right ENERGY at
/// [`HOHMANN_SEED_THETA_COUNT`] evenly-spaced burn locations and let the
/// search fix the geometry (theta/phi/date), which is exactly what a
/// population search is good at. Empty when no Hohmann reference resolved
/// (seeding is an accelerant, never a requirement).
fn hohmann_energy_seeds(
    cfg: &MissionConfig,
    opt: &OptimizationConfig,
    refs: &ObjectiveReferences,
    bounds: &[(f64, f64)],
) -> Vec<Vec<f64>> {
    let Some(vinf_ms) = refs.hohmann_dep_vinf_ms else { return Vec::new() };
    let Some(cat) = body_models::TargetBody::by_name(&opt.departure_body) else { return Vec::new() };
    let r_park_m = resolve_parking_orbit_radius_m(cfg, &cat);
    if r_park_m <= 0.0 {
        return Vec::new();
    }
    let v_circ = (cat.mu_m3s2 / r_park_m).sqrt();
    // `Launch` mode (Phase 14b): slot 2 IS the v∞, so seed it directly; the
    // evenly-spaced slot-1 values are then right ascensions of the asymptote
    // rather than burn locations — same role (let the search fix geometry).
    let slot2 = if departure_mode(cfg) == DepartureMode::Launch {
        vinf_ms
    } else {
        (vinf_ms * vinf_ms + 2.0 * cat.mu_m3s2 / r_park_m).sqrt() - v_circ
    };
    (0..HOHMANN_SEED_THETA_COUNT)
        .map(|k| {
            let theta = k as f64 / HOHMANN_SEED_THETA_COUNT as f64 * std::f64::consts::TAU;
            vec![
                0.0, // departure offset: the nominal epoch (clamped into a degenerate window by the GA)
                theta.clamp(bounds[1].0, bounds[1].1),
                slot2.clamp(bounds[2].0, bounds[2].1),
                0.0_f64.clamp(bounds[3].0, bounds[3].1), // in-plane / zero declination
            ]
        })
        .collect()
}

/// Normalized target-distance error: `0` for a candidate whose closest
/// approach (`closest_approach_m`) lands exactly on `miss_ref_m` (the
/// configured `target_orbit_radius_m`, or the target body's own physical
/// radius when unconfigured -- see `build_objective_references`), growing
/// unboundedly for a candidate that lands far from it in either direction
/// (never approached the target at all, or dove well inside the requested
/// distance). This is exactly what `ObjectiveFunction::MatchTargetDistance`
/// itself already scores; factored out (plain `f64` args, not `&CandidateEvaluation`/
/// `&FitnessContext`, so it's unit-testable without a live ANISE almanac) so
/// `real_dynamics_fitness` can fold it into `MinDeltaV`/`MinTof` too (Phase
/// 12a) instead of it only being reachable by picking `MatchTargetDistance`
/// explicitly.
fn target_distance_error(closest_approach_m: f64, miss_ref_m: f64) -> f64 {
    (closest_approach_m - miss_ref_m).abs() / miss_ref_m
}

/// Fitness for one candidate — the single objective the GA/PSO is
/// minimizing, normalized to a dimensionless, O(1)-for-a-reasonable-
/// trajectory scale via `FitnessContext::refs`. Returns `None` (infeasible)
/// when ephemeris is unavailable, or a needed capture is never achieved —
/// `GaSolver`/`PsoSolver` penalize this to `f64::MAX` rather than panicking.
fn real_dynamics_fitness(ctx: &FitnessContext, params: &[f64]) -> Option<f64> {
    let eval = evaluate_candidate(ctx, params)?;
    Some(fitness_from_eval(ctx, params, &eval))
}

/// The post-evaluation half of [`real_dynamics_fitness`] — factored out
/// so a caller that already has this candidate's
/// `CandidateEvaluation` in hand (the GA's outcome-recording fitness
/// closures, which need the eval for the arrival-ΔV/TOF side-products) can
/// score it without paying for a second full propagation.
fn fitness_from_eval(ctx: &FitnessContext, params: &[f64], eval: &CandidateEvaluation) -> f64 {
    // Phase 12a, replacing the narrower Phase 9y
    // fix: a `Flyby` mission's entire point, per the Phase 12 canonical
    // design ("we compute a trajectory getting closest to the flyby
    // periapsis ... regardless of what method"), is that the search always
    // steers toward the configured periapsis distance -- independent of
    // which `optimization.objective` (MinDeltaV/MinTof/MatchTargetDistance)
    // the user separately picked to also minimize. Investigation found the
    // real gap wasn't a missing chromosome dimension (params are unchanged
    // here -- the existing 4 burn parameters already fully determine
    // `closest_approach_m`) but that `MinDeltaV`/`MinTof` had NO periapsis-
    // distance awareness at all for a Flyby mission (`needs_capture` is
    // always false for Flyby, so the capture-crossing-based gates below
    // never fire for it) beyond a coarse "did it enter the target's SOI at
    // all" gate -- SOI radius is orders of magnitude larger than any
    // realistic requested flyby altitude, so that gate let a MinDeltaV/
    // MinTof search converge on the cheapest burn that merely grazed the
    // target's SOI, nowhere near the actual requested periapsis. Compare
    // `mga.rs::arrival_dv_ms`: MGA has no separate `optimization.objective`
    // selector at all -- `mission.objective` alone is the single source of
    // truth for arrival handling. The single-leg path can't drop
    // `optimization.objective` (it genuinely does pick between minimizing
    // ΔV, TOF, or pure distance-matching), so instead `mission.objective ==
    // Flyby` now unconditionally ADDS the same normalized distance-error
    // term `MatchTargetDistance` itself computes into the MinDeltaV/MinTof
    // fitness. Both terms are already O(1)-normalized, so a candidate far
    // from the target periapsis is dominated by the (unboundedly growing)
    // distance term regardless of how cheap its burn is, while a candidate
    // already near the target periapsis is scored almost entirely on real
    // ΔV/TOF -- a continuous, threshold-free handoff from "find the target"
    // to "minimize cost," not a hard gate.
    let flyby_distance_term =
        if matches!(ctx.cfg.mission.objective, MissionObjective::Flyby) { target_distance_error(eval.closest_approach_m, ctx.refs.miss_ref_m) } else { 0.0 };

    match ctx.opt.objective {
        // RADIUS-FREE since (
        // for DV, it will have total dv as objective, trying to get into
        // any kind of orbit regardless of the radius"): total ΔV =
        // departure burn + the burn to capture into ANY bound orbit at the
        // candidate's own closest approach (`dv_capture_any_orbit_ms`,
        // always computable -- no capture gate, no configured radius, no
        // graded non-capture penalty needed anymore; physics itself shapes
        // the arrival term, see that field's doc comment). The old
        // crossing-gated formulation (`noncapture_dv_penalty` + real
        // crossing arrival) is retired for this objective. A Flyby mission
        // (needs_capture false) still prices no arrival burn at all --
        // there is no capture to pay for -- and keeps its distance term.
        ObjectiveFunction::MinDeltaV => {
            // HARD infeasibility band for a capture-needing candidate whose
            // approach periapsis stays beyond the SOI ceiling (
            // third revision same day --
            // found solutions within the soi... it doesnt make sense"):
            // the earlier soft additive ln-pressure quietly demoted the
            // mission's capture requirement from a CONSTRAINT to a
            // preference, and against Mercury's shallow gravity well
            // (capturing costs nearly the full v-infinity at any depth) a
            // cheap near-SOI flyby legitimately out-scored real captures on
            // total ΔV -- so runs converged to infeasible flybys the OLD
            // hard-gated fitness never would have. Restores the old gate's
            // architecture at the SOI line: `10 + ln(rp/ceiling)` keeps
            // every feasible capture (fitness = dv_total/dv_ref, O(1-3))
            // strictly below every infeasible candidate, with a graded
            // approach gradient inside the band and a small ΔV pressure so
            // the band converges through cheap corridors. Radius-freedom
            // INSIDE the SOI is unchanged.
            if eval.needs_capture {
                if let Some(cap) = eval.capture_rp_max_m {
                    if eval.capture_rp_m > cap {
                        return 10.0
                            + (eval.capture_rp_m / cap).ln()
                            + INFEASIBLE_BAND_DV_WEIGHT * (departure_cost_ms(ctx, params) + eval.dv_capture_any_orbit_ms) / ctx.refs.dv_ref_ms;
                    }
                }
            }
            let dv_arrival_ms = if eval.needs_capture { eval.dv_capture_any_orbit_ms } else { 0.0 };
            (departure_cost_ms(ctx, params) + dv_arrival_ms) / ctx.refs.dv_ref_ms + flyby_distance_term
        }
        // Needs a real completion time -- same reasoning as MinDeltaV.
        // Penalty stays distance-only (no ΔV term): time, unlike ΔV, is not
        // an accumulating resource the departure burn alone predicts, so a
        // ΔV term here would bias the approach corridor for the WRONG
        // objective.
        ObjectiveFunction::MinTof => {
            if eval.needs_capture && eval.arrival.is_none() {
                let norm = (eval.closest_approach_m / ctx.refs.miss_ref_m).max(1.0);
                return 10.0 + norm.ln();
            }
            let achieved_tof_days = eval.arrival.as_ref().map(|a| a.capture_time_s / 86_400.0).unwrap_or(eval.closest_approach_days);
            achieved_tof_days / eval.tof_ref_days + flyby_distance_term
        }
        // Always computable from closest_approach_m, regardless of whether
        // a real capture was achieved -- this is the objective that lets
        // the GA learn from candidates that got "relatively close" and
        // iterate toward an actual capture over generations, instead of
        // discarding every non-capturing candidate as equally infeasible
        // (which would remove the GA's only gradient toward the target).
        // (`flyby_distance_term` is redundant here by construction --
        // `target_distance_error` is exactly this branch's own formula --
        // so it's intentionally not added a second time.)
        //
        // + a small ΔV tiebreaker (see `MTD_DV_TIEBREAKER_WEIGHT`):
        // pure distance-matching was completely ΔV-blind, so among the many
        // candidates achieving essentially the same target distance it
        // freely returned energetically absurd ones (observed live: a
        // "converged" 61 km/s total-ΔV Mercury capture). The weight is small
        // enough that any genuine distance improvement still dominates --
        // the objective is still MatchTargetDistance, the tiebreak only
        // orders near-ties by real cost.
        // DEPARTURE burn only in the tiebreaker -- REGRESSION FIXED
        // same day it was introduced (
        // an arrival orbit... The trajectory now does a flyby and
        // continues... You made it worse"): the first version priced
        // `params[2] + dv_arrival_ms`, but `dv_arrival_ms` is only ever
        // COMPUTED for a candidate that actually crosses the capture
        // radius inbound -- an equally-close candidate that barely doesn't
        // cross pays nothing. Near convergence the distance term differs
        // by ~1e-3 between candidates while a real capture burn's
        // tiebreaker share is ~5e-2, so the "tiebreak" dominated and
        // actively selected AGAINST capturing candidates -- runs stopped
        // producing any capture (no arrival burn, no capture orbit, arc
        // coasting through the full budget) at all. The departure burn is
        // priced identically for every candidate, so it breaks energy
        // ties without any capture/non-capture asymmetry.
        ObjectiveFunction::MatchTargetDistance => {
            mtd_fitness(eval.closest_approach_m, ctx.refs.miss_ref_m, departure_cost_ms(ctx, params), ctx.refs.dv_ref_ms)
        }
    }
}

/// A candidate's plottable outcome side-products, recorded once per fitness
/// evaluation (the outcome scatter plots): its own arrival burn
/// [m/s] (the real crossing burn when one was achieved, else the
/// capture-into-any-orbit burn at its closest approach; 0.0 for a Flyby
/// mission, which prices no arrival burn) and its achieved time of flight
/// [days] (capture time, or closest-approach time when it never captured).
fn candidate_outcome(eval: &CandidateEvaluation) -> (f64, f64) {
    let dv_arrival_ms = if eval.needs_capture {
        eval.arrival.as_ref().map(|a| a.dv_capture_ms).unwrap_or(eval.dv_capture_any_orbit_ms)
    } else {
        0.0
    };
    let tof_days = eval.arrival.as_ref().map(|a| a.capture_time_s / 86_400.0).unwrap_or(eval.closest_approach_days);
    (dv_arrival_ms, tof_days)
}

/// The configured objective's value in its NATURAL units for one candidate
/// (
/// optimization. Just show total DV"): total ΔV [m/s] for MinDeltaV,
/// achieved TOF [days] for MinTof, |closest − target| [km] for
/// MatchTargetDistance. What the live convergence plot's y-axis shows —
/// SELECTION still runs on the normalized fitness (which also carries the
/// Flyby distance term etc.); this is display only.
/// `(value, feasible)`: the objective's value in natural units, and whether
/// this candidate is a FEASIBLE capture (inside the SOI ceiling; always
/// true for objectives/missions with no capture concept). Since 
/// (third revision, after a run whose genuinely-feasible final solution
/// never showed on the plot): the value is ALWAYS a real number — for an
/// infeasible MinDeltaV candidate it's the honest lower bound
/// `departure + |v_rel|` — and feasibility travels as a separate flag, so
/// the client can draw an always-present dashed "best so far (no capture
/// yet)" line alongside the solid feasible-capture line, instead of a
/// blank panel whenever the fitness leader happens to sit outside the SOI.
fn objective_display_value(ctx: &FitnessContext, params: &[f64], eval: &CandidateEvaluation) -> (f64, bool) {
    let (dv_arrival_ms, tof_days) = candidate_outcome(eval);
    let feasible = !eval.needs_capture || eval.capture_rp_max_m.map_or(true, |cap| eval.capture_rp_m <= cap);
    match ctx.opt.objective {
        ObjectiveFunction::MinDeltaV => (params[2] + dv_arrival_ms, feasible),
        ObjectiveFunction::MinTof => (tof_days, feasible),
        ObjectiveFunction::MatchTargetDistance => {
            ((eval.closest_approach_m - ctx.refs.miss_ref_m).abs() / 1000.0, feasible)
        }
    }
}

/// Osculating periapsis state of a target-relative two-body orbit, closed
/// form: given ANY point `(r_rel, v_rel)` on the approach
/// orbit, returns the position and velocity AT its periapsis — `r_p = (h²/μ)
/// /(1+e)` along the eccentricity vector, `|v_p| = h/r_p` purely tangential
/// (exact at periapsis for any conic). This is the real fix for the
/// synthesized capture orbit coming out eccentric/off-center/sub-surface
///: the arc's SAMPLED closest-approach point sits
/// up to half a sample interval (~hours at a typical coast budget) from the
/// true periapsis, where the relative velocity has a large RADIAL component
/// — circularizing the speed along that direction produced an ellipse whose
/// real periapsis was far below the sampled radius. Capturing at the
/// osculating periapsis instead is exact regardless of sampling density.
/// `None` for a degenerate near-radial orbit (h ≈ 0).
fn osculating_periapsis_state(r_rel: Vector3<f64>, v_rel: Vector3<f64>, mu: f64) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let h = r_rel.cross(&v_rel);
    let h_norm = h.norm();
    if h_norm < 1e-3 || r_rel.norm() < 1.0 {
        return None;
    }
    let e_vec = v_rel.cross(&h) / mu - r_rel.normalize();
    let e = e_vec.norm();
    let rp = h_norm * h_norm / mu / (1.0 + e);
    if rp < 1.0 {
        return None;
    }
    let e_hat = if e > 1e-9 { e_vec / e } else { r_rel.normalize() };
    let t_hat = h.normalize().cross(&e_hat);
    Some((e_hat * rp, t_hat * (h_norm / rp)))
}

/// Time offset [s] from a state `(r_rel, v_rel)` to its own osculating
/// periapsis, by bisection on the radial-velocity sign flip under two-body
/// Kepler propagation (`r·v` is negative approaching, positive receding,
/// zero exactly at periapsis/apoapsis; near periapsis the flip is unique).
/// Positive = periapsis ahead, negative = behind. `None` when no flip is
/// bracketed within `max_window_s` (state too far from periapsis for the
/// caller's window — fall back to the sampled point) or the Kepler
/// propagation fails to converge.
fn time_to_periapsis_s(r_rel: Vector3<f64>, v_rel: Vector3<f64>, mu: f64, max_window_s: f64) -> Option<f64> {
    let vr_at = |dt: f64| trajectory_solver::propagate_kepler(r_rel, v_rel, dt, mu).map(|(r, v)| r.dot(&v));
    let vr0 = r_rel.dot(&v_rel);
    if vr0.abs() < 1e-6 {
        return Some(0.0);
    }
    let dir = if vr0 < 0.0 { 1.0 } else { -1.0 };
    let mut hi = 60.0_f64;
    loop {
        match vr_at(dir * hi) {
            Some(v) if v * vr0 < 0.0 => break,
            Some(_) => {
                hi *= 2.0;
                if hi > max_window_s {
                    return None;
                }
            }
            None => return None,
        }
    }
    let mut lo = 0.0_f64;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        match vr_at(dir * mid) {
            Some(v) if v * vr0 > 0.0 => lo = mid,
            Some(_) => hi = mid,
            None => return None,
        }
    }
    Some(dir * 0.5 * (lo + hi))
}

/// Weight of `MatchTargetDistance`'s ΔV tiebreaker term — see that fitness
/// branch's own comment. Small: here ΔV is a tiebreak on an objective that
/// is genuinely about distance, not a co-objective. (An earlier same-day
/// `NONCAPTURE_DV_PRESSURE_WEIGHT` mechanism for MinDeltaV was retired
/// hours later when MinDeltaV became radius-free — its whole capture-gate
/// penalty branch no longer exists.)
const MTD_DV_TIEBREAKER_WEIGHT: f64 = 0.05;

/// `MatchTargetDistance`'s full fitness: distance error + a small
/// DEPARTURE-burn tiebreaker. **The signature deliberately does not accept
/// an arrival ΔV** — that is the load-bearing design constraint, not an
/// omission (see the call site's REGRESSION FIXED comment):
/// arrival ΔV only exists for candidates that achieve a capture crossing,
/// so pricing it here makes capturing candidates strictly more expensive
/// than their non-capturing neighbors and drives the search away from
/// capture entirely — live-confirmed as "runs stopped producing any
/// arrival orbit at all" the same day it was tried. Only quantities every
/// candidate is charged identically for may appear in this tiebreaker.
fn mtd_fitness(closest_approach_m: f64, miss_ref_m: f64, dv_departure_ms: f64, dv_ref_ms: f64) -> f64 {
    target_distance_error(closest_approach_m, miss_ref_m) + MTD_DV_TIEBREAKER_WEIGHT * dv_departure_ms / dv_ref_ms
}

/// Re-evaluate the real closest-approach distance [km] at a given point —
/// used to report the best point's miss distance without re-deriving it
/// from the fitness value (whose units vary by `objective`).
fn miss_distance_km(ctx: &FitnessContext, params: &[f64]) -> f64 {
    evaluate_candidate(ctx, params)
        .map(|e| e.closest_approach_m / 1000.0)
        .unwrap_or(f64::NAN)
}

fn build_context<'a>(cfg: &'a MissionConfig, opt: &'a OptimizationConfig, almanac: &'a Almanac) -> Result<FitnessContext<'a>, String> {
    let dep_anise = anise_body(&opt.departure_body.to_lowercase())
        .ok_or_else(|| format!("optimization.departure_body '{}' is not ANISE-covered", opt.departure_body))?;
    let target_anise = anise_body(&opt.target_body.to_lowercase())
        .ok_or_else(|| format!("optimization.target_body '{}' is not ANISE-covered", opt.target_body))?;
    let dep_epoch_str = opt.departure_epoch.as_deref().ok_or("optimization.departure_epoch is required")?;
    let dep_epoch = parse_epoch(dep_epoch_str).map_err(|e| format!("optimization.departure_epoch: {e}"))?;
    let dep_jd_base = epoch_to_jd(dep_epoch);

    // A real capture burn (Orbit/Landing/SampleReturn) needs a target
    // distance to insert at -- checked here too (not just
    // `config::check_config`) since this is the actual run-time entry point
    // and validation isn't guaranteed to have been called first. Flyby
    // needs no arrival burn; Rendezvous needs no capture
    // radius either -- its arrival burn is the full relative-velocity
    // magnitude (`arrival_dv_ms` in mga.rs), not a periapsis-parameterized
    // capture orbit.
    let needs_capture_radius = !matches!(
        cfg.mission.objective,
        MissionObjective::Flyby | MissionObjective::Rendezvous
    );
    if needs_capture_radius {
        let has_target_distance = cfg.trajectory.capture.as_ref().and_then(|c| c.target_orbit_radius_m).is_some();
        if !has_target_distance {
            return Err(
                "mission.objective is not \"Flyby\"/\"Rendezvous\" -- \
                 [trajectory.capture].target_orbit_radius_m is required to \
                 compute a real arrival/capture burn."
                    .into(),
            );
        }
    }

    let refs = build_objective_references(cfg, opt, almanac, dep_anise, target_anise, dep_jd_base);
    let launch_mode = departure_mode(cfg) == DepartureMode::Launch;
    if launch_mode {
        // `check_config` enforces these; re-checked here because this is the
        // run-time entry point (same reasoning as the capture-radius check).
        let body = body_models::TargetBody::by_name(&opt.departure_body)
            .ok_or_else(|| format!("optimization.departure_body '{}' is not in the body catalog", opt.departure_body))?;
        if launch_frame(&body).is_none() {
            return Err(format!(
                "trajectory.departure.mode = \"Launch\": departure body '{}' has no catalog pole", opt.departure_body
            ));
        }
        if cfg.trajectory.departure.as_ref().and_then(|d| d.launch_site.as_ref()).is_none() {
            return Err("trajectory.departure.mode = \"Launch\" requires trajectory.departure.launch_site".into());
        }
    }
    Ok(FitnessContext { almanac, cfg, opt, dep_anise, dep_jd_base, refs, launch_mode })
}

/// Builds the normalization references documented on `FitnessContext::refs`.
/// Falls back to a fixed 1.0 reference (i.e. no normalization) for whichever
/// Hohmann term fails to resolve -- e.g. departure/target ephemeris
/// unavailable exactly at `dep_jd_base` -- rather than failing the whole
/// optimization run over a normalization detail; the real fitness
/// computation downstream still works either way.
fn build_objective_references(
    cfg: &MissionConfig,
    // Unused since the dv_ref fix removed the Earth-departure
    // special case (its only consumer) -- kept in the signature since
    // future reference terms will plausibly need it again.
    _opt: &OptimizationConfig,
    almanac: &Almanac,
    dep_anise: Body,
    target_anise: Body,
    dep_jd_base: f64,
) -> ObjectiveReferences {
    let hohmann = body_state(almanac, EphemerisSource::Anise, Some(dep_anise), &None, dep_jd_base)
        .zip(body_state(almanac, EphemerisSource::Anise, Some(target_anise), &None, dep_jd_base))
        .and_then(|((r_dep, _), (r_target, _))| {
            let r1_m = (r_dep[0].powi(2) + r_dep[1].powi(2) + r_dep[2].powi(2)).sqrt();
            let r2_m = (r_target[0].powi(2) + r_target[1].powi(2) + r_target[2].powi(2)).sqrt();
            HohmannSolver { mu_m3s2: MU_SUN_M3S2, r1_m, r2_m }.solve().ok()
        });

    // Same "is this charged at all" accounting as the real evaluation
    // (`real_dynamics_fitness`): the departure burn is ALWAYS charged
    // (`params[2]`, since the fix — the fitness charges it
    // for Earth departures too, so the normalization reference must
    // include it or the two silently disagree on scale; this line kept the
    // pre-12n "Earth departure is launch-vehicle-free" accounting until
    // a stale mirror of exactly what 12n fixed). No arrival
    // component for a Flyby mission (no capture burn exists — matching the
    // fitness, which prices a Flyby's arrival at 0.0).
    let is_flyby = matches!(cfg.mission.objective, MissionObjective::Flyby);
    let dv_ref_ms = hohmann
        .as_ref()
        .map(|h| {
            let arr_component = if is_flyby { 0.0 } else { h.dv_arrival_ms };
            h.dv_departure_ms + arr_component
        })
        .filter(|v| *v > 0.0)
        .unwrap_or(1.0);
    let tof_ref_days = hohmann.as_ref().map(|h| h.tof_s / 86_400.0).filter(|v| *v > 0.0).unwrap_or(1.0);

    // `[trajectory.capture].target_orbit_radius_m` -- required (validated in
    // `build_context`) for any non-Flyby mission, and for MatchTargetDistance
    // regardless of mission type; otherwise unused, but resolved here
    // regardless so `ObjectiveReferences` doesn't need an `Option`. Falls
    // back to the target body's own physical radius when unset and unused.
    let miss_ref_m = cfg
        .trajectory
        .capture
        .as_ref()
        .and_then(|c| c.target_orbit_radius_m)
        .filter(|v| *v > 0.0)
        .unwrap_or_else(|| cfg.target_body.radius_m.max(1.0));

    ObjectiveReferences {
        dv_ref_ms,
        tof_ref_days,
        miss_ref_m,
        hohmann_dep_vinf_ms: hohmann.as_ref().map(|h| h.dv_departure_ms).filter(|v| *v > 0.0),
    }
}

/// `[dep_offset_days, theta_burn_rad, dv_mps, phi_out_of_plane_rad]` bounds —
/// see `evaluate_candidate`'s doc comment.
///
/// `theta_burn` is left at the full circle: tried restricting it to the
/// "prograde half" (`cos(theta) >= 0`, reasoning that `t_hat(0)` is almost
/// exactly aligned with the departure body's own heliocentric velocity) on
/// the assumption that an outward transfer should never want a net-retrograde
/// departure component. Empirically wrong (scratch_optimize_repro.toml):
/// the unrestricted search's real best (`theta=254 deg`, net retrograde by
/// this same formula) clearly outperformed the restricted search, which
/// pegged against the new boundary. That Hohmann-style intuition only holds
/// near an ideal ~180-degree-transfer-angle departure date; this isn't one
/// (consistent with its long achieved TOF), so the genuinely useful transfer
/// orientation isn't confined to either half.
///
/// `phi` is narrowed to `[-45 deg, +45 deg]`: this still allows diverting up
/// to ~70% of `dv` out-of-plane (`sin(45 deg)`), comfortably more than any of
/// this catalog's target bodies need (Mars's mutual inclination to Earth's
/// orbital plane is ~1.85 deg) -- while guaranteeing a non-trivial tangential
/// contribution always survives. Unlike the `theta` attempt above, this one
/// doesn't presuppose which *direction* is useful, only that the out-of-plane
/// share doesn't need the full range -- lower risk, not (yet) found to
/// exclude a real optimum.
///
/// `Launch` mode (Phase 14b) reuses the four slots as `[dep_offset_days,
/// rla_rad, vinf_ms, dla_rad]`: the θ range becomes the RLA range (full
/// circle by default), the φ range the DLA range (`[−90°, 90°]` by default
/// — the asymptote may point anywhere; a high |DLA| just forces a high-
/// inclination plane), and `dv_min_ms`/`dv_max_ms` are mapped to the v∞
/// range a TANGENTIAL burn of that size from the parking orbit would give,
/// `v∞ = √((v_c + Δv)² − 2μ/r_p)` — so an existing config's burn bounds keep
/// their energy meaning without a second pair of fields.
fn bounds_from(cfg: &MissionConfig, opt: &OptimizationConfig) -> Vec<(f64, f64)> {
    let window = opt.departure_window_days.unwrap_or(0.0);
    // User-configurable angle ranges -- defaults reproduce
    // the previous hardcoded bounds exactly.
    let theta_min = opt.theta_min_deg.unwrap_or(0.0).to_radians();
    let theta_max = opt.theta_max_deg.unwrap_or(360.0).to_radians();
    if departure_mode(cfg) == DepartureMode::Launch {
        let dla_min = opt.phi_min_deg.unwrap_or(-90.0).to_radians();
        let dla_max = opt.phi_max_deg.unwrap_or(90.0).to_radians();
        let (vinf_min, vinf_max) = match body_models::TargetBody::by_name(&opt.departure_body) {
            Some(cat) => {
                let r_p = resolve_parking_orbit_radius_m(cfg, &cat);
                let v_c = (cat.mu_m3s2 / r_p).sqrt();
                let vinf_of = |dv: f64| ((v_c + dv).powi(2) - 2.0 * cat.mu_m3s2 / r_p).max(0.0).sqrt();
                (vinf_of(opt.dv_min_ms), vinf_of(opt.dv_max_ms))
            }
            None => (opt.dv_min_ms, opt.dv_max_ms),
        };
        return vec![(-window / 2.0, window / 2.0), (theta_min, theta_max), (vinf_min, vinf_max), (dla_min, dla_max)];
    }
    let phi_min = opt.phi_min_deg.unwrap_or(-45.0).to_radians();
    let phi_max = opt.phi_max_deg.unwrap_or(45.0).to_radians();
    vec![
        (-window / 2.0, window / 2.0),
        (theta_min, theta_max),
        (opt.dv_min_ms, opt.dv_max_ms),
        (phi_min, phi_max),
    ]
}

/// Run the Phase 9 optimization stage's configured `method` and return its
/// result, with no I/O. `GA`/`PSO` are real-propagation searches (this
/// function); `MultipleShooting`/`MGA` are recognized but not implemented
/// yet (9d-9g — see the design notes), and return a clear `Err` rather than
/// silently falling back to GA/PSO.
pub fn run_optimization(cfg: &MissionConfig, almanac: &Almanac) -> Result<OptimizeComputeResult, String> {
    run_optimization_with_progress(cfg, almanac, |_step, _phase, _best, _feas, _params, _pop, _outcomes| {}, &std::sync::atomic::AtomicBool::new(false))
}

/// Same as [`run_optimization`], but calls
/// `on_step(step_index, phase, best_fitness_so_far)` after each generation/
/// iteration — for the async job's live progress stream (9f). `phase` is `1`
/// (flyby-only closest-approach search, value in km) or `2` (real configured
/// objective, dimensionless normalized fitness) -- the two metrics are on
/// completely different scales and must not be plotted on one shared axis;
/// see `PopulationLogRow::phase`'s doc comment for the same distinction.
/// PSO and the post-search coordinate-descent refinement both report `2`
/// (no flyby-only phase 1 of their own; both already optimize the real
/// objective directly).
///
/// `cancelled` is polled after every generation/iteration (GA/PSO) and
/// every refinement pass; once set, the search stops and returns whatever
/// candidate is best so far rather than continuing to the configured
/// budget (job cancellation, Phase 9k task 4).
pub fn run_optimization_with_progress(
    cfg: &MissionConfig,
    almanac: &Almanac,
    // Fourth argument: the best-so-far PARAMETER VECTOR --
    // empty slice until the first feasible candidate exists. Streams real
    // decision variables for GA/PSO the way MGA always did (the live
    // search-variable convergence plot's data source).
    // Fifth argument (same day): THIS generation's full
    // evaluated population -- the live search-space scatter plots' data
    // source (full population clouds, live, not only
    // after completion). Empty for PSO (its solver callback exposes only
    // the global best) and for the refinement pass.
    //
    // Stream contract rework (
    // two phases, and no one understands why there are more generations on
    // x axis than given in user inputs"): the streamed step index is now a
    // SINGLE unified generation counter that never exceeds the configured
    // generation budget (phase-1 restarts and phase-2 runs consume slices
    // of the SAME budget; their gen-0 initial-population callbacks re-emit
    // the current index instead of inflating it; the post-search
    // refinement pass no longer streams steps at all -- its polish still
    // lands in the returned result). `best_fitness` is now ALWAYS the real
    // configured objective's best-so-far (during the internal
    // closest-approach bootstrap phase, the phase-1 best candidate is
    // re-scored under the real objective once per generation -- one extra
    // evaluation per generation, ~2% of a generation's evaluation cost)
    // -- one plottable series, one unit, whole run.
    // Sixth argument (same day again): per-individual outcome
    // pairs `[dv_arrival_ms, tof_days]`, parallel to (and same length as)
    // the population argument -- both are FEASIBLE-ONLY (infeasible
    // candidates are filtered out of both together, keeping them aligned).
    // The outcome scatter plots' (departure vs arrival ΔV, offset vs coast
    // time) data source.
    mut on_step_raw: impl FnMut(usize, u8, f64, f64, &[f64], &[Vec<f64>], &[[f64; 2]]),
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<OptimizeComputeResult, String> {
    let opt = cfg.optimization.as_ref().ok_or("this config has no [optimization] section")?;
    let ctx = build_context(cfg, opt, almanac)?;
    let bounds = bounds_from(cfg, opt);

    // Record every sentinel-free streamed step into the result's own
    // `objective_history` (see that field's doc comment) -- the wrapper
    // makes the persisted series IDENTICAL to what the live plot drew, by
    // construction rather than by parallel bookkeeping.
    let objective_history_cell: std::cell::RefCell<Vec<[f64; 3]>> = std::cell::RefCell::new(Vec::new());
    let mut on_step = |step: usize, phase: u8, best: f64, feas: f64, params: &[f64], pop: &[Vec<f64>], outs: &[[f64; 2]]| {
        if best < f64::MAX {
            objective_history_cell.borrow_mut().push([step as f64, phase as f64, best]);
        }
        on_step_raw(step, phase, best, feas, params, pop, outs)
    };

    // Two-phase GA: a from-scratch search under the *real* configured
    // objective has almost nothing to climb (escaping at all is already a
    // minority of random burns, and MinDeltaV/MinTof are flatly infeasible
    // until a capture happens to occur by chance too) -- see
    // `flyby_only_fitness`'s doc comment. Phase 1 always searches for the
    // best flyby/closest-approach first, regardless of `mission.objective`
    // or `optimization.objective`; phase 2 then refines under the real
    // objective, starting from a window narrowed around phase 1's result.
    // PSO is not split this way yet -- only GA,.
    //
    // Phase 1 is further split into PHASE1_STARTS independent restarts
    // (distinct seeds, fresh random population each time), and only the
    // single best individual across *all* restarts seeds phase 2. Found
    // empirically (scratch_optimize_repro.toml): a single deep
    // phase-1 run plateaus well before its generation budget is spent (best-
    // so-far stops improving by ~gen 30-47 of 48-49) while sitting in a
    // mediocre basin -- more generations in that same basin don't help, but
    // a real propagated theta/dv grid scan confirmed a substantially better
    // basin exists elsewhere in the *same* bounds. Multiple independent
    // restarts (each free to land in a different basin) directly targets
    // that failure mode; a single longer run does not.
    let (mut best_params, mut best_fitness, mut history, phase1_history_km, mut population_log, final_stream_step, mut best_feasible_so_far) = match opt.method {
        OptimizationMethod::GA => {
            let p = opt.ga.as_ref().ok_or("optimization.ga is required when method = \"GA\"")?;
            let total_generations = (p.generations as usize).max(2);
            // Phase 1 is the harder, from-scratch search (find any good
            // escape direction/timing at all); phase 2 only has to refine
            // within an already-good region, so it gets the smaller share.
            let phase1_generations_total = (total_generations * 3 / 5).max(1);
            let phase2_generations = total_generations.saturating_sub(phase1_generations_total).max(1);
            // Exact split (
            // generation 30 but only to 27"): plain floor division dropped
            // the remainder (e.g. 18/4 restarts = 4 each = 16 of 18), so
            // the unified stream counter fell short of the configured
            // budget. The first `remainder` restarts get one extra
            // generation so the split sums exactly.
            let phase1_split: Vec<usize> = {
                let base = phase1_generations_total / PHASE1_STARTS;
                let rem = phase1_generations_total % PHASE1_STARTS;
                (0..PHASE1_STARTS).map(|i| (base + usize::from(i < rem)).max(1)).collect()
            };

            let mut population_log = Vec::new();
            let mut phase1_history_km = Vec::new();
            let mut running_best_m = f64::MAX;
            let mut global_best_params: Option<Vec<f64>> = None;
            let mut global_best_fitness_m = f64::MAX;

            // Unified stream state (see on_step's own doc comment): one
            // generation counter bounded by the configured budget, one
            // best-so-far REAL-objective series across both phases.
            let mut stream_gen = 0usize;
            let mut stream_best_obj = f64::MAX;
            let mut stream_best_obj_params: Vec<f64> = Vec::new();
            // What the stream's best_fitness field actually carries: the
            // best candidate's objective in NATURAL units (m/s, days, km --
            // see objective_display_value), selection still by normalized
            // fitness.
            let mut stream_best_display = f64::MAX;
            // Best value among FEASIBLE-capture candidates seen anywhere in
            // any population -- NOT just the fitness leader (
            // third revision: a borderline-outside-SOI candidate can hold
            // the fitness lead while feasible solutions exist, which
            // blanked the plotted line for entire runs that ended
            // perfectly feasible).
            let mut stream_best_feasible = f64::MAX;

            // Per-evaluation outcome side-channel (the arrival-
            // ΔV/TOF scatter plots): each fitness call pushes its
            // candidate's (dv_arrival, tof) here in evaluation order (which
            // is population order -- GaSolver re-evaluates the whole
            // population every generation, elites included), and the
            // population callback drains one generation's worth at a time.
            // RefCell because the fitness and progress closures both need
            // it simultaneously.
            let outcome_buf: std::cell::RefCell<Vec<Option<(f64, f64, f64, bool)>>> = std::cell::RefCell::new(Vec::new());

            // Analytic Hohmann-energy departure-burn seeds, injected into
            // EVERY restart's initial population (each restart's random
            // remainder still differs via its own RNG stream) -- see
            // hohmann_energy_seeds' doc comment.
            let energy_seeds = hohmann_energy_seeds(cfg, opt, &ctx.refs, &bounds);

            for start in 0..PHASE1_STARTS {
                // Start 0 is seeded toward `theta in [0 deg, +45 deg]`,
                // `phi in [+15 deg, +45 deg]`, dv/dep_offset left at the full
                // configured bounds. Re-derived after a decisive
                // test: `hyperbolic_departure_state` (exact closed-form
                // injection for a *given* v-infinity vector, no theta/phi
                // approximation at all) applied to the real Lambert solution
                // at `TOF~458 days` gives a genuine 41,429 km real propagated
                // miss -- inside the 50,000 km capture radius -- at
                // `dv~5137 m/s`. The earlier-assumed "minimum energy"
                // solution (`TOF~298d, dv~3637 m/s`) gives a real 32.85M km
                // miss even with the *exact* injection -- not a direction
                // problem at all (confirmed: zero approximation error) but a
                // real escape-duration effect: a low-dv departure is a
                // marginal escape that lingers/oscillates near the SOI
                // boundary for over a week (confirmed separately: 16 days,
                // vs ~2.6 days for a clean high-dv escape) before separating,
                // which the idealized instantaneous-departure Lambert
                // assumption never accounts for. The TOF~458d window's own
                // theta/phi requirement (confirmed via direction-matching
                // sweep) sits at `theta~10-40 deg, phi~20-45 deg` across its
                // neighboring TOF values -- this seed targets that window
                // directly rather than hoping a restart rediscovers it.
                // Starts 1-3 stay fully unbiased, for genuine global search.
                // Restart-0's historical hand-tuned theta/phi bias window
                // (a narrowed [0,45]deg/[15,45]deg box, from a
                // Mars-specific investigation) was REMOVED in favor of a
                // purely random start -- the Hohmann-energy seeds (now
                // injected at generation 1) carry the analytic-guidance
                // role, and every restart's initial population uniformly
                // samples the configured bounds.
                let start_bounds = bounds.clone();
                let _ = start; // restarts differ only by RNG seed now
                let phase1_solver = GaSolver {
                    population_size: p.population_size as usize,
                    generations: phase1_split[start],
                    crossover_rate: p.crossover_rate,
                    mutation_rate: p.mutation_rate,
                    elitism_count: p.elitism_count as usize,
                    tournament_size: TOURNAMENT_SIZE,
                    seed: SEED.wrapping_add(start as u64 * 1_000), // distinct stream per restart
                };
                let r1 = phase1_solver.run_seeded_with_population_progress(
                    &start_bounds,
                    &energy_seeds,
                    |params| {
                        let eval = evaluate_candidate(&ctx, params);
                        outcome_buf.borrow_mut().push(eval.as_ref().map(|e| {
                            let (dv, t) = candidate_outcome(e);
                            let (disp, feas) = objective_display_value(&ctx, params, e);
                            (dv, t, disp, feas)
                        }));
                        eval.map(|e| e.closest_approach_m)
                    },
                    |gen, pop, fits| {
                        let outcomes: Vec<Option<(f64, f64, f64, bool)>> = outcome_buf.borrow_mut().drain(..).collect();
                        for ((ind, &f), out) in pop.iter().zip(fits).zip(&outcomes) {
                            // Same unified stream_gen index the live stream
                            // uses, so post-run population
                            // scatters and the live view share one x-axis
                            // bounded by the configured budget.
                            population_log.push(PopulationLogRow {
                                phase: 1,
                                generation: stream_gen,
                                params: ind.clone(),
                                fitness: f,
                                dv_arrival_ms: out.map(|(dv, _, _, _)| dv),
                                tof_days: out.map(|(_, t, _, _)| t),
                            });
                        }
                        let (gen_min_idx, gen_min_m) = fits
                            .iter()
                            .cloned()
                            .enumerate()
                            .fold((0usize, f64::MAX), |acc, (i, f)| if f < acc.1 { (i, f) } else { acc });
                        if gen_min_m < running_best_m {
                            running_best_m = gen_min_m;
                        }
                        phase1_history_km.push(running_best_m / 1000.0);
                        // Re-score this generation's best under the REAL
                        // configured objective for the unified stream -- one
                        // extra evaluation per generation, see on_step's doc
                        // comment. The internal search still selects on
                        // closest approach; only what's REPORTED changed.
                        // (This direct call bypasses the outcome side-channel
                        // by construction -- it never touches outcome_buf.)
                        if gen_min_m < f64::MAX {
                            if let Some(e) = evaluate_candidate(&ctx, &pop[gen_min_idx]) {
                                let obj = fitness_from_eval(&ctx, &pop[gen_min_idx], &e);
                                if obj < stream_best_obj {
                                    stream_best_obj = obj;
                                    stream_best_obj_params = pop[gen_min_idx].clone();
                                    stream_best_display = objective_display_value(&ctx, &pop[gen_min_idx], &e).0;
                                }
                            }
                        }
                        let (pop_feasible, outcomes_feasible): (Vec<Vec<f64>>, Vec<[f64; 2]>) = pop
                            .iter()
                            .zip(&outcomes)
                            .filter_map(|(ind, out)| out.map(|(dv, t, _, _)| (ind.clone(), [dv, t])))
                            .unzip();
                        for out in &outcomes {
                            if let Some((_, _, d, true)) = out {
                                if *d < stream_best_feasible { stream_best_feasible = *d; }
                            }
                        }
                        on_step(stream_gen, 1, stream_best_display, stream_best_feasible, &stream_best_obj_params, &pop_feasible, &outcomes_feasible);
                        if gen > 0 {
                            stream_gen += 1; // gen-0 re-emits the current index, never inflates it
                        }
                        !cancelled.load(std::sync::atomic::Ordering::Relaxed)
                    },
                );
                if r1.best_fitness < global_best_fitness_m {
                    global_best_fitness_m = r1.best_fitness;
                    global_best_params = Some(r1.best_params.clone());
                }
                if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
            }
            let phase1_best_params = global_best_params.ok_or("phase 1 produced no feasible individual across any restart")?;

            // PHASE2_RESTARTS independent phase-2 runs, each seeded from a
            // different, mutually-distant phase-1 candidate (see
            // select_diverse_phase2_seeds' doc comment for the 61 km/s
            // failure mode this fixes -- one seed = one basin). Each run's
            // bounds narrow around ITS OWN seed, and the seed itself is
            // injected into that run's initial population.
            let mut phase2_seeds = select_diverse_phase2_seeds(&population_log, &bounds, PHASE2_RESTARTS);
            if phase2_seeds.is_empty() {
                phase2_seeds.push(phase1_best_params.clone());
            }
            // Same exact-split treatment as phase 1 (remainder distributed
            // to the first seeds), so phase1+phase2 stream generations sum
            // to the configured budget exactly.
            let phase2_split: Vec<usize> = {
                let n = phase2_seeds.len();
                let base = phase2_generations / n;
                let rem = phase2_generations % n;
                (0..n).map(|i| (base + usize::from(i < rem)).max(1)).collect()
            };

            let mut phase2_best: Option<(Vec<f64>, f64)> = None;
            let mut phase2_history: Vec<f64> = Vec::new();
            for (run_idx, seed) in phase2_seeds.iter().enumerate() {
                let phase2_bounds = narrow_bounds(&bounds, seed, 0.3);
                let phase2_solver = GaSolver {
                    population_size: p.population_size as usize,
                    generations: phase2_split[run_idx],
                    crossover_rate: p.crossover_rate,
                    mutation_rate: p.mutation_rate,
                    elitism_count: p.elitism_count as usize,
                    tournament_size: TOURNAMENT_SIZE,
                    // Distinct stream from every phase-1 restart AND from
                    // the other phase-2 runs.
                    seed: SEED.wrapping_add(999_999 + run_idx as u64 * 7_777),
                };
                let r2 = phase2_solver.run_seeded_with_population_progress(
                    &phase2_bounds,
                    std::slice::from_ref(seed),
                    |params| {
                        let eval = evaluate_candidate(&ctx, params);
                        outcome_buf.borrow_mut().push(eval.as_ref().map(|e| {
                            let (dv, t) = candidate_outcome(e);
                            let (disp, feas) = objective_display_value(&ctx, params, e);
                            (dv, t, disp, feas)
                        }));
                        eval.map(|e| fitness_from_eval(&ctx, params, &e))
                    },
                    |gen, pop, fits| {
                        let outcomes: Vec<Option<(f64, f64, f64, bool)>> = outcome_buf.borrow_mut().drain(..).collect();
                        for ((ind, &f), out) in pop.iter().zip(fits).zip(&outcomes) {
                            population_log.push(PopulationLogRow {
                                phase: 2,
                                generation: stream_gen,
                                params: ind.clone(),
                                fitness: f,
                                dv_arrival_ms: out.map(|(dv, _, _, _)| dv),
                                tof_days: out.map(|(_, t, _, _)| t),
                            });
                        }
                        let (gen_min_idx, gen_min) = fits
                            .iter()
                            .cloned()
                            .enumerate()
                            .fold((0usize, f64::MAX), |acc, (i, f)| if f < acc.1 { (i, f) } else { acc });
                        // Same unified best-so-far series phase 1 already
                        // feeds -- phase-2 fitness IS the real objective, so
                        // no re-scoring needed here.
                        if gen_min < stream_best_obj {
                            stream_best_obj = gen_min;
                            stream_best_obj_params = pop[gen_min_idx].clone();
                            if let Some((_, _, disp, _)) = outcomes[gen_min_idx] {
                                stream_best_display = disp;
                            }
                        }
                        let (pop_feasible, outcomes_feasible): (Vec<Vec<f64>>, Vec<[f64; 2]>) = pop
                            .iter()
                            .zip(&outcomes)
                            .filter_map(|(ind, out)| out.map(|(dv, t, _, _)| (ind.clone(), [dv, t])))
                            .unzip();
                        for out in &outcomes {
                            if let Some((_, _, d, true)) = out {
                                if *d < stream_best_feasible { stream_best_feasible = *d; }
                            }
                        }
                        on_step(stream_gen, 2, stream_best_display, stream_best_feasible, &stream_best_obj_params, &pop_feasible, &outcomes_feasible);
                        if gen > 0 {
                            stream_gen += 1;
                        }
                        !cancelled.load(std::sync::atomic::Ordering::Relaxed)
                    },
                );
                phase2_history.extend(r2.history.iter().copied());
                if phase2_best.as_ref().map_or(true, |(_, bf)| r2.best_fitness < *bf) {
                    phase2_best = Some((r2.best_params, r2.best_fitness));
                }
                if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
            }
            let (best_p, best_f) = phase2_best.expect("at least one phase-2 run always executes");

            (best_p, best_f, phase2_history, phase1_history_km, population_log, stream_gen, stream_best_feasible)
        }
        OptimizationMethod::PSO => {
            let p = opt.pso.as_ref().ok_or("optimization.pso is required when method = \"PSO\"")?;
            let solver = PsoSolver {
                swarm_size: p.swarm_size as usize,
                iterations: p.iterations as usize,
                inertia_max: p.inertia_weight,
                inertia_min: (p.inertia_weight * 0.4).min(p.inertia_weight),
                cognitive_coeff: p.cognitive_weight,
                social_coeff: p.social_weight,
                seed: SEED,
            };
            // Natural-units display value for the stream (see
            // objective_display_value): PSO's solver callback gives only
            // the normalized fitness + best position, so the best is
            // re-evaluated once whenever it CHANGES (not per iteration).
            let mut pso_last_best: Vec<f64> = Vec::new();
            let mut pso_best_display = f64::MAX;
            let mut pso_best_feasible = f64::MAX;
            let r = solver.run_with_progress_params(&bounds, |params| real_dynamics_fitness(&ctx, params), |iter, _best, best_pos| {
                if best_pos != pso_last_best.as_slice() {
                    pso_last_best = best_pos.to_vec();
                    if let Some(e) = evaluate_candidate(&ctx, best_pos) {
                        let (d, feas) = objective_display_value(&ctx, best_pos, &e);
                        pso_best_display = d;
                        if feas && d < pso_best_feasible { pso_best_feasible = d; }
                    }
                }
                // PSO's solver callback exposes only the global best -- no
                // per-iteration swarm positions to stream (a real, known
                // PSO reporting gap, not a bug).
                on_step(iter, 2, pso_best_display, pso_best_feasible, best_pos, &[], &[]);
                !cancelled.load(std::sync::atomic::Ordering::Relaxed)
            });
            (r.best_params, r.best_fitness, r.history, Vec::new(), Vec::new(), p.iterations as usize, pso_best_feasible)
        }
        OptimizationMethod::MultipleShooting => {
            return Err(
                "optimization.method = \"MultipleShooting\" is not implemented yet (Phase 9, \
                 deferred per project roadmap)"
                    .into(),
            );
        }
        OptimizationMethod::MGA => {
            // MGA is dispatched separately in `run_api_optimize` before this
            // function is called — it cannot go through `OptimizeComputeResult`.
            return Err("MGA must be dispatched via run_api_optimize, not run_optimization_with_progress".into());
        }
    };

    // Phase 3: local coordinate-descent refinement, applied after GA/PSO
    // regardless of method -- same shrinking-step-size pattern as
    // OptimizationProblems/examples/interplanetary_transfer's
    // `TransferProblem::refine`. Added a real grid scan (no
    // GA, no seeding) confirmed the search lands in a real, smooth, sharp
    // local optimum -- not a bug, not a discontinuity -- that population
    // search reliably finds the basin of but can't precisely nail with a
    // finite population/generation budget. Coordinate descent from the
    // GA/PSO's own best point is exactly the right tool for that regime.
    // `dep_offset_days` is only perturbed when its bounds are non-degenerate
    // (`departure_window_days > 0`) -- otherwise it's fixed at exactly 0 and
    // perturbing it would just waste evaluations on an always-rejected move.
    // Natural-units display of the refinement's best-so-far -- f64::MAX
    // until the first accepted improvement (the frontend's line filters the
    // sentinel; the probe clouds render regardless).
    let mut refine_best_display = f64::MAX;
    let angle_steps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]; // rad, theta/phi
    let dv_steps = [200.0, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0]; // m/s
    let dep_steps = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]; // days
    const REFINE_MAX_PASSES: usize = 8;
    let dims: &[usize] = if bounds[0].1 > bounds[0].0 { &[0, 1, 2, 3] } else { &[1, 2, 3] };
    // Continues the STREAM's own cumulative step counter (fix:
    // this used to restart at history.len() -- the phase-2 generation count
    // alone -- which is far LESS than the stream's cumulative phase-1+2
    // counter, so refinement steps plotted at earlier x than the
    // generations they followed, part of the "random iterations" axis the
    // live plot showed).
    'refine: for round in 0..angle_steps.len() {
        if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let step_for = |dim: usize| match dim {
            0 => dep_steps[round],
            2 => dv_steps[round],
            _ => angle_steps[round],
        };
        let mut improved = true;
        let mut pass = 0;
        while improved && pass < REFINE_MAX_PASSES {
            if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                break 'refine;
            }
            improved = false;
            pass += 1;
            // Every refinement probe is recorded and streamed as PHASE 3
            // (
            // to the plots... so its clear where the final value came
            // from") -- all at x = the configured budget (the axis's
            // committed end), rendered in a distinct color client-side.
            let mut pass_params: Vec<Vec<f64>> = Vec::new();
            let mut pass_outcomes: Vec<[f64; 2]> = Vec::new();
            for &dim in dims {
                for &dir in &[step_for(dim), -step_for(dim)] {
                    let mut candidate = best_params.clone();
                    candidate[dim] = (candidate[dim] + dir).clamp(bounds[dim].0, bounds[dim].1);
                    if let Some(e) = evaluate_candidate(&ctx, &candidate) {
                        let f = fitness_from_eval(&ctx, &candidate, &e);
                        let (dv_a, tof) = candidate_outcome(&e);
                        population_log.push(PopulationLogRow {
                            phase: 3,
                            generation: final_stream_step,
                            params: candidate.clone(),
                            fitness: f,
                            dv_arrival_ms: Some(dv_a),
                            tof_days: Some(tof),
                        });
                        pass_params.push(candidate.clone());
                        pass_outcomes.push([dv_a, tof]);
                        if f < best_fitness {
                            let (d, feas) = objective_display_value(&ctx, &candidate, &e);
                            refine_best_display = d;
                            if feas && d < best_feasible_so_far { best_feasible_so_far = d; }
                            best_params = candidate;
                            best_fitness = f;
                            improved = true;
                        }
                    }
                }
            }
            history.push(best_fitness);
            on_step(final_stream_step, 3, refine_best_display, best_feasible_so_far, &best_params, &pass_params, &pass_outcomes);
        }
    }

    // ONE post-refinement endpoint at x = the configured budget (
    //
    // when looking at the arrival/dep dv plot" -- the refinement pass can
    // genuinely walk well past the GA's own best, and hiding it entirely
    // made the final stats look disconnected from the plotted convergence).
    if let Some(e) = evaluate_candidate(&ctx, &best_params) {
        let (display, feas) = objective_display_value(&ctx, &best_params, &e);
        if feas && display < best_feasible_so_far {
            best_feasible_so_far = display;
        }
        on_step(final_stream_step, 3, display, best_feasible_so_far, &best_params, &[], &[]);
    }

    let miss_km = miss_distance_km(&ctx, &best_params);
    Ok(OptimizeComputeResult {
        dep_jd_base: ctx.dep_jd_base,
        bounds,
        best_params,
        best_fitness,
        history,
        phase1_history_km,
        population_log,
        miss_km,
        objective_history: objective_history_cell.into_inner(),
    })
}

// ── API result type ───────────────────────────────────────────────────────────

/// Best individual/particle from a Phase 9 real-dynamics GA/PSO search —
/// shaped like `design.rs::OptimizerApiResult` (the narrowing-stage
/// equivalent), with real departure/arrival-leg fields the narrowing stage
/// has no equivalent of (its Lambert-only arc always ends exactly on target
/// by construction, with no real escape or capture leg at all).
#[derive(Debug, serde::Serialize)]
pub struct OptimizeApiResult {
    /// "GA" or "PSO" — short code, also used as the output filename prefix.
    /// Use `method_display` for a human-readable label.
    pub method: String,
    pub method_display: String,
    pub objective: String,
    pub dep_offset_days: f64,
    pub dep_jd: f64,
    /// True anomaly of the departure burn at the parking orbit [rad]. In
    /// `Launch` mode (Phase 14b) this slot is the asymptote's RLA [rad].
    pub theta_burn_rad: f64,
    /// Departure burn magnitude [m/s] — the PHYSICAL injection burn: the
    /// search parameter itself in `ParkingOrbit` mode; in `Launch` mode the
    /// tangential escape burn `√(v∞² + 2μ/r_p) − √(μ/r_p)` the launch
    /// geometry derives (the gene is the v∞ there — see `launch_geometry`).
    /// MGA: the periapsis escape burn either way.
    pub dv_departure_ms: f64,
    /// Out-of-plane angle of the departure burn [rad]. In `Launch` mode
    /// this slot is the asymptote's DLA [rad].
    pub phi_out_of_plane_rad: f64,
    /// `[trajectory.departure].mode`: `"ParkingOrbit"` or `"Launch"` (Phase
    /// 14a/14d).
    pub departure_mode: String,
    /// Who pays `dv_departure_ms` — `dv_ledger.departure_dv_pool`
    /// (`"launcher"` / `"onboard"` / `"split"`), surfaced here so a
    /// consumer seeding Phase 03 can decide `external_stage` without
    /// digging into the ledger. `null` only when the ledger is.
    pub departure_dv_pool: Option<String>,
    /// Epoch of the injection burn [JD] — `dep_jd` (this model fires the
    /// departure at the departure epoch; the parking-orbit coast precedes it).
    pub injection_epoch_jd: f64,
    /// `Launch` mode only (Phase 14c/14d): the closed-form launch geometry
    /// for this result's own departure asymptote — RLA/DLA, the site-
    /// feasible plane, azimuth, coast angle and the exact injection state
    /// (`MANUAL.md` §13.6). `null` in `ParkingOrbit` mode. The
    /// schematic ascent a frontend draws from it is NOT a reference for
    /// Phase 03; `pre_departure_orbit_arc` stays the physical parking coast.
    pub launch_geometry: Option<LaunchGeometryApiResult>,
    /// Real time spent escaping the departure body's SOI [s] -- `arc`'s
    /// first `t_s <= escape_duration_s` points are the departure leg
    /// (departure-body-centered until SOI exit); everything after is the
    /// heliocentric cruise leg.
    pub escape_duration_s: f64,
    /// Time of mission completion (capture burn, or closest approach for a
    /// Flyby) [days] — an output of the search, not a prescribed input.
    pub achieved_tof_days: f64,
    /// Real arrival/capture burn [m/s] -- `0.0` for a Flyby (no capture burn
    /// exists to charge).
    pub dv_arrival_ms: f64,
    /// Absolute time of the capture burn [s since departure], if one was
    /// needed and achieved -- `null` for a Flyby.
    pub capture_time_s: Option<f64>,
    /// Real inertial ΔV VECTOR for the capture burn [m/s] (for
    /// Phase 03's `cruise_seed.planned_burns`) -- direction and
    /// magnitude combined, matching `dv_arrival_ms`'s magnitude exactly by
    /// construction. A tangential (retrograde) insertion burn: opposite the
    /// real incoming relative-velocity direction at the capture crossing
    /// (`ArrivalCapture::v_rel_mps`), same physical assumption
    /// `dv_capture_ms` itself already makes (`|v_rel| - v_circ`), just
    /// exposed as a real vector instead of a magnitude so a downstream
    /// consumer (the cruise-seeded replay) can actually FIRE it instead of
    /// reconstructing a direction client-side. Velocity DIFFERENCES don't
    /// need a frame conversion from body-relative to inertial -- the
    /// target body's own velocity is the same additive term on both sides
    /// of the burn and cancels in the difference. `null` for a Flyby or any
    /// candidate that didn't achieve a real capture, mirroring
    /// `capture_time_s`.
    pub capture_dv_inertial_mps: Option<[f64; 3]>,
    /// Real inertial ΔV VECTOR for the DEPARTURE burn [m/s] (
    /// the departure-side mirror of `capture_dv_inertial_mps`
    /// — without it, `pre_departure_orbit_arc` can only ever be visual-only
    /// in a Phase 03 replay: prepending the parking orbit to the cruise
    /// reference with no ΔV vector to fire at `t_s = arc[0].t_s` would
    /// leave the simulated truth coasting in it forever). As a velocity
    /// DIFFERENCE it needs no body-relative→inertial frame conversion (the
    /// departure body's own velocity cancels). GA/PSO: post-burn minus
    /// pre-burn state from the exact `circular_orbit_burn_state`
    /// construction the search itself used, so `|v|` matches
    /// `dv_departure_ms` by construction. MGA: the periapsis escape burn is
    /// purely tangential (parallel to the injection velocity), so the
    /// vector is `v0_hat × dv_escape` from `mga_departure_injection_state`
    /// — same magnitude convention as `dv_departure_ms`'s own
    /// `departure_escape_dv_ms` pricing. `null` only when the departure
    /// body doesn't resolve (mirrors `pre_departure_orbit_arc`).
    pub departure_dv_inertial_mps: Option<[f64; 3]>,
    /// Arrival-side mirror of `theta_burn_rad`/`phi_out_of_plane_rad` -- see
    /// `arrival_burn_angles`. `null` for a Flyby or any candidate that didn't
    /// achieve a real capture.
    pub theta_arr_rad: Option<f64>,
    pub phi_arr_rad: Option<f64>,
    pub fitness: f64,
    /// Real closest-approach distance to the target body over the whole
    /// arc [km] — the quantity the narrowing stage's GA/PSO has no
    /// equivalent of (its Lambert-only fitness can't miss).
    pub miss_km: f64,
    /// Launch-vehicle check at THIS result's own departure C3 (Phase 14d-lite,
    /// the frontend used the survey's `best_arc.launch_vehicle_
    /// check` as a stand-in before). GA/PSO: C3 from the search's own
    /// injection state (`|v_post|² − 2μ/r_park`); MGA: from the chromosome's
    /// departure v∞ via `mga_departure_injection_state`. `null` for a
    /// non-Earth departure, no `[spacecraft].launch_vehicle`, an unknown
    /// vehicle name, or a departure body that doesn't resolve.
    pub launch_vehicle_check: Option<LaunchVehicleCheckApiResult>,
    /// Two-pool ΔV ledger + propellant feasibility (Phase 14e) for this
    /// result: departure = `dv_departure_ms` (split launcher/onboard per the
    /// check above), DSMs = `mga_dv_dsms_ms` sum (MGA) or 0, arrival =
    /// `dv_arrival_ms`. `null` only when the departure body doesn't resolve.
    pub dv_ledger: Option<DvLedgerApiResult>,
    /// Target body's real heliocentric position at the moment of closest
    /// approach (or capture) [m] — for plotting alongside `arc`.
    pub target_r_arr_m: [f64; 3],
    /// Best-fitness-so-far per generation (GA) or iteration (PSO) — phase
    /// 2's (the real-objective refinement's) history for GA.
    pub convergence: Vec<f64>,
    /// Phase 1's (flyby-only) best-closest-approach-so-far history [km] —
    /// empty for PSO.
    pub phase1_convergence_km: Vec<f64>,
    /// The exact natural-units series the live convergence plot drew --
    /// [step, phase, value] per sentinel-free streamed step, phase 3 = the
    /// post-search refinement entries incl. the final polished value. See
    /// OptimizeComputeResult::objective_history. Empty for MGA.
    pub objective_history: Vec<[f64; 3]>,
    /// Every evaluated individual across both phases — empty for PSO. The
    /// raw data for an x/y scatter of any two search parameters, colored by
    /// generation.
    pub population_log: Vec<PopulationLogRow>,
    pub arc: Vec<ArcApiPoint>,
    /// MGA only: full body-name sequence [departure, flyby0, ..., target].
    /// `null` for GA/PSO.
    pub mga_body_sequence: Option<Vec<String>>,
    /// MGA only: per-leg DSM ΔVs [m/s]. `null` for GA/PSO.
    pub mga_dv_dsms_ms: Option<Vec<f64>>,
    /// MGA only: real inertial ΔV VECTORS for each leg's DSM [m/s] (frontend
    /// ask, same motivation as `OptimizeApiResult::
    /// capture_dv_inertial_mps` — a downstream consumer needs to actually
    /// FIRE these as `cruise_seed.planned_burns` entries, not reconstruct a
    /// direction client-side). `dv_dsms_inertial_mps[k].norm() ==
    /// mga_dv_dsms_ms[k]` by construction. When multiple-shooting refinement
    /// converged (or ran at all — see `mga_ms_converged`), these are the
    /// REAL N-body-corrected vectors (`MultipleShotResult::dv_dsm_corrected`,
    /// already computed internally, previously reduced to magnitude-only
    /// before reaching the API); otherwise (refinement errored) they're
    /// freshly evaluated from the uncorrected search-time chromosome
    /// (`mga_leg.rs::MgaLegResult::v_dsm_after_mps − v_dsm_before_mps`),
    /// matching `mga_dv_dsms_ms`'s own fallback in both cases. `null` for
    /// GA/PSO (no DSMs) or if the fallback evaluation itself fails.
    pub mga_dv_dsms_inertial_mps: Option<Vec<[f64; 3]>>,
    /// MGA only: per-leg DSM heliocentric positions [m].
    /// `dsm_positions_m[k]` is `[x, y, z]` for leg k. `null` for GA/PSO.
    pub mga_dsm_positions_m: Option<Vec<[f64; 3]>>,
    /// MGA only: per-leg DSM epochs [s since departure, same clock as
    /// `arc`'s `t_s`] (the missing epoch
    /// half of `mga_dv_dsms_inertial_mps`, so Phase 03 can populate
    /// `cruise_seed.planned_burns` without inferring epochs by matching
    /// `mga_dsm_positions_m` against `arc`). `dsm_epochs_s[k]` pairs with
    /// `mga_dv_dsms_inertial_mps[k]`/`mga_dsm_positions_m[k]`. Valid for
    /// the multiple-shooting-refined arc too — MS never moves the leg
    /// timing (see `mga::MgaResult::dsm_epochs_s`). `null` for GA/PSO.
    pub mga_dsm_epochs_s: Option<Vec<f64>>,
    /// MGA only: per-leg TOFs [days]. `null` for GA/PSO.
    pub mga_leg_tofs_days: Option<Vec<f64>>,
    /// MGA only: phase 1 (DSM-only) best-fitness history [m/s]. `null` for GA/PSO.
    pub mga_phase1_history: Option<Vec<f64>>,
    /// MGA only (Phase 9k task 3): whether the multiple-shooting refinement
    /// that produced `arc`/`fitness`/`mga_dv_dsms_ms` converged (`Newton–Raphson`
    /// within `MAX_MS_ITER` iterations) or is a best-effort, not-fully-corrected
    /// result — the refinement always returns *some* arc either way, this just
    /// says how much to trust it. `null` for GA/PSO, and also `null` for MGA if
    /// the refinement itself errored out (rare — e.g. the winning chromosome's
    /// initial state was infeasible), in which case `arc`/`mga_dv_dsms_ms` fall
    /// back to the uncorrected search-time values instead.
    pub mga_ms_converged: Option<bool>,
    /// MGA only (user-requested): the real ΔV [m/s] each
    /// intermediate flyby actually delivered, measured directly from the
    /// converged, real-dynamics-propagated trajectory —
    /// `|v_helio_at_SOI_exit − v_helio_at_SOI_entry|` (not an analytic
    /// re-derivation from v∞-in/v∞-out; see `mga.rs::run_multiple_shooting`'s
    /// post-check loop for the exact measurement). Length matches the
    /// number of intermediate flyby bodies (`mga_body_sequence` minus
    /// departure/target). `null` under the same conditions as
    /// `post_capture_orbit_arc`: not MGA, or `mga_ms_converged` is not
    /// `Some(true)` — no genuinely converged flyby passage to measure.
    pub mga_flyby_dv_gained_ms: Option<Vec<f64>>,
    /// Paired with `mga_flyby_dv_gained_ms`: heliocentric speed [m/s] just
    /// before/after each flyby's real SOI entry/exit. `speed_after >
    /// speed_before` means that flyby ADDED heliocentric energy (taken from
    /// the planet's own orbit); `speed_after < speed_before` means it
    /// REMOVED energy (given back to the planet) — the real, checkable
    /// signature of an energy-shedding deceleration flyby. Same length/
    /// gating as `mga_flyby_dv_gained_ms`.
    pub mga_flyby_speed_before_ms: Option<Vec<f64>>,
    pub mga_flyby_speed_after_ms: Option<Vec<f64>>,
    /// Phase 12c: the actual resulting orbit around the target
    /// body after the capture burn, re-seeded with the corrected (circular)
    /// post-burn velocity at the crossing/arrival point and propagated for
    /// `POST_CAPTURE_ORBIT_PERIODS` (2) full periods. Fixed 
    /// (Phase 12k): now returned in the SAME heliocentric frame `arc` uses
    /// (the target body's own real heliocentric position at the capture
    /// epoch has already been added to every point) — it used to be
    /// target-body-relative, documented as "a SEPARATE set of points, not
    /// appended to it," which forced any consumer wanting one continuous
    /// path to drop these points rather than mix frames. `central_body` is
    /// still always the target body's own name — that field means "which
    /// body's gravity is locally dominant here," not "which frame this
    /// position is in." `null` for: `Flyby`/`Rendezvous`/
    /// `SampleReturn` missions (no capture burn / not yet modeled), an
    /// `Orbit`/`Landing` candidate that never achieved a real capture, and —
    /// for MGA specifically — whenever `mga_ms_converged` is `Some(false)` or
    /// `None` (multiple shooting didn't converge, or errored out entirely):
    /// building a "resulting orbit" on top of a physically-uncorrected or
    /// gap-filled trajectory would compound a bad foundation with a
    /// fabricated-looking orbit, so it's deliberately omitted rather than
    /// computed from the uncorrected arc.
    pub post_capture_orbit_arc: Option<Vec<ArcApiPoint>>,
    /// Orbital period [s] of `post_capture_orbit_arc`, computed analytically
    /// from the same inputs used to build the ring (not derived from the
    /// points) — so a frontend can show "period: X hours" directly instead
    /// of re-deriving it from raw geometry. Same presence/null gating as
    /// `post_capture_orbit_arc` itself (user-requested).
    pub post_capture_orbit_period_s: Option<f64>,
    /// Periapsis radius [m] of `post_capture_orbit_arc` — for MGA this is
    /// always the real configured/priced `[trajectory.capture].target_orbit_radius_m`
    /// (or its floor-clamped value); for single-leg, same. Equal to
    /// `post_capture_orbit_apoapsis_m` whenever `post_capture_orbit_eccentricity`
    /// is 0 (circular, the common case).
    pub post_capture_orbit_periapsis_m: Option<f64>,
    /// Apoapsis radius [m] of `post_capture_orbit_arc` — `periapsis_m *
    /// (1+e)/(1-e)`. Equal to `post_capture_orbit_periapsis_m` when circular.
    pub post_capture_orbit_apoapsis_m: Option<f64>,
    /// Eccentricity of `post_capture_orbit_arc` — the REAL post-burn orbit's
    /// own value (`orbital_math::eccentricity` of the propagated seed state).
    /// Both paths (MGA and single-leg GA/PSO/MBH, the latter since Phase 14f,
    ///) price the capture burn against the configured
    /// `[trajectory.capture].capture_eccentricity` periapsis speed
    /// (`capture_target_speed_mps`) and seed the ring from that same speed
    /// along the real crossing direction, so this equals the configured
    /// value for a tangential crossing and reports the real, slightly
    /// different shape when the crossing has a radial component.
    pub post_capture_orbit_eccentricity: Option<f64>,
    /// Semi-major axis [m] of `post_capture_orbit_arc` (
    /// user-requested Kepler-element summary) — derived from the same real
    /// post-burn state the propagated ring itself was seeded with.
    pub post_capture_orbit_sma_m: Option<f64>,
    /// Inclination [deg] of `post_capture_orbit_arc`'s plane to the
    /// reference frame's fundamental plane (the J2000 frame every state in
    /// this pipeline is expressed in), `acos(h_z/|h|)` of the real
    /// post-burn state.
    pub post_capture_orbit_inclination_deg: Option<f64>,
    /// Phase 12 (user-requested): the real parking orbit leading
    /// up to the departure burn, so the frontend can show where the
    /// spacecraft actually started instead of the arc beginning abruptly
    /// mid-escape. Returned in the SAME heliocentric frame `arc` uses since
    /// the departure body's own real heliocentric
    /// position at the departure epoch has already been added to every
    /// point.
    /// - **Single-leg**: the REAL parking orbit the search found —
    ///   `r_park_m`/`theta_burn_rad`/the real orbital plane, ending exactly
    ///   at the real burn point. Free — every number is already computed
    ///   for the real ΔV/escape physics, this just also propagates it
    ///   backward for display. Not an approximation.
    /// - **MGA**: also the real periapsis-injection parking-orbit state
    ///   (`mga::mga_departure_injection_state`, an exact closed-form
    ///   construction from the chromosome's already-optimized departure v∞
    ///   vector — no new chromosome dimension, no approximation), matching
    /// single-leg's fidelity exactly. Superseded the original 
    ///   stand-in (a ring at the departure body's own Laplace SOI radius in
    ///   its heliocentric orbital plane) the same day it was first written
    ///   — this doc comment previously still described that superseded
    /// approach, corrected. Purely additive/display — does not
    ///   change any reported ΔV number.
    pub pre_departure_orbit_arc: Option<Vec<ArcApiPoint>>,
}

/// Convert a slice of [`mga::MgaArcPoint`] to [`ArcApiPoint`], preserving
/// each point's `leg_idx` so the frontend can segment the arc by gravity-assist
/// leg. All MGA arcs are heliocentric; `central_body` is always `"Sun"`.
///
/// `vx_mps`/`vy_mps`/`vz_mps` are real as of (`MgaArcPoint` now
/// carries the velocity its sampling paths always computed and used to
/// discard) — closing the last `null`-velocity case the 
/// `ArcApiPoint` velocity work left open.
fn mga_arc_to_api(mga_pts: &[mga::MgaArcPoint]) -> Vec<ArcApiPoint> {
    mga_pts
        .iter()
        .map(|p| ArcApiPoint {
            t_s: p.t_days * 86_400.0,
            x_m: p.x_m,
            y_m: p.y_m,
            z_m: p.z_m,
            central_body: "Sun".to_string(),
            leg_idx: Some(p.leg_idx as u32),
            vx_mps: Some(p.vx_mps),
            vy_mps: Some(p.vy_mps),
            vz_mps: Some(p.vz_mps),
        })
        .collect()
}

/// Run the optimization stage and return the API-shaped result, including a
/// re-propagated visualization arc at the best point. Used by both the CLI
/// output writer and the `/api/optimize` HTTP endpoint (9f).
pub fn optimize_api(cfg: &MissionConfig, almanac: &Almanac) -> Result<OptimizeApiResult, String> {
    optimize_api_with_progress(
        cfg, almanac,
        |_step, _phase, _best, _feas, _params, _legs, _pop, _outcomes| {},
        |_seq_idx, _seq_count, _flyby_bodies, _is_direct| {},
        &std::sync::atomic::AtomicBool::new(false),
    )
}

/// `on_step(step, phase, best_fitness, best_params_so_far, best_legs_so_far)` —
/// `best_params_so_far` is `Some(..)` for MGA (which always has a chromosome
/// at every generation) and `None` for GA/PSO (Phase 9k task 4's
/// cancellation plumbing is the only consumer of GA/PSO's on_step calls
/// today; live chromosome streaming was only asked for MGA — see the design notes
/// Phase 9w-vii/"Frontend Backlog"). `best_legs_so_far` is the per-leg
/// trajectory state ([`mga::MgaLegStepInfo`]) for `best_params_so_far` —
/// always empty for GA/PSO (no leg concept), populated for MGA (backlog
/// item #18).
///
/// `on_sequence(seq_idx, seq_count, flyby_bodies, is_direct_baseline)` — MGA
/// only (Phase 9k "step-stream context" ask); ignored/never
/// called for GA/PSO. Fires once per candidate sequence BEFORE that
/// sequence's own `on_step` generations start streaming — see
/// `mga::run_mga`'s doc comment. Lets a live-replay client know which real
/// flyby-body sequence the steps it's about to receive belong to, which
/// isn't otherwise derivable client-side under `sequence_search` auto-
/// discovery (several candidate sequences are evaluated in turn, and the
/// request itself only specifies the candidate *pool*, not which one is
/// currently running).
///
/// `cancelled` is honored by the GA/PSO search (checked every generation/
/// iteration, per Phase 9k task 4) but NOT by the MGA branch below — MGA's
/// DE search (`crate::mga::run_mga`) has no cancellation hook yet, since
/// `mga.rs` had uncommitted, unrelated work in flight when this was added.
/// A cancel request against a running MGA job is recorded by the caller
/// (job status reports it) but does not stop the in-flight compute.
pub fn optimize_api_with_progress(
    cfg: &MissionConfig,
    almanac: &Almanac,
    // Sixth argument: the generation's full evaluated
    // population -- GA only (empty for PSO/MGA), see
    // run_optimization_with_progress's own on_step doc comment.
    mut on_step: impl FnMut(usize, u8, f64, Option<f64>, Option<&[f64]>, &[mga::MgaLegStepInfo], &[Vec<f64>], &[[f64; 2]]),
    mut on_sequence: impl FnMut(usize, usize, &[String], bool),
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<OptimizeApiResult, String> {
    let opt = cfg.optimization.as_ref().ok_or("this config has no [optimization] section")?;

    // MGA does all its own I/O and reporting inside run_mga — it is not
    // compatible with the single-leg GA post-processing below (different
    // chromosome format, different evaluator).  Return a minimal result so
    // the CLI/API path doesn't try to call evaluate_candidate on MGA params.
    if opt.method == OptimizationMethod::MGA {
        let r = mga::run_mga(
            cfg, almanac,
            |step, phase, best, params, legs| on_step(step, phase, best, Some(best), Some(params), legs, &[], &[]),
            |seq_idx, seq_count, flyby_bodies, is_direct| on_sequence(seq_idx, seq_count, flyby_bodies, is_direct),
        )?;
        let dep_offset = r.best_params.first().copied().unwrap_or(0.0);
        let dep_jd_val = r.dep_jd;
        // Fallback arc if multiple shooting errors out entirely (rare — see
        // below): the repropagated arc (preferred) or the Keplerian/Lambert
        // arc when re-propagation failed, preserving leg_idx on every point
        // so the frontend can segment by leg.
        let mga_src = if !r.repropagated_arc.is_empty() { &r.repropagated_arc } else { &r.arc };
        // Multiple-shooting refinement (Phase 9k task 3): the
        // real corrected trajectory, not a one-shot uncorrected repropagation
        // — always run for the API result, since it's already an async job
        // (the extra Newton/LM iterations don't block anything) and it always
        // returns *some* arc, converged or not (never fails outright unless
        // the initial state itself is infeasible). `flyby_bodies` here is the
        // intermediate-only slice `run_multiple_shooting` expects (excludes
        // the overall departure_body/target_body endpoints).
        let flyby_bodies: Vec<String> = if r.body_sequence.len() > 2 {
            r.body_sequence[1..r.body_sequence.len() - 1].to_vec()
        } else {
            Vec::new()
        };
        let ms_result = mga::run_multiple_shooting(&r, &flyby_bodies, cfg, almanac);
        let (mga_arc_api, mga_dv_dsms_ms_final, mga_dv_dsms_inertial_mps_final, dv_total_final, mga_ms_converged) = match &ms_result {
            Ok(ms) => (
                mga_arc_to_api(&ms.arc),
                ms.dv_dsm_corrected.iter()
                    .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                    .collect::<Vec<f64>>(),
                // Already real vectors internally -- previously reduced to
                // magnitude-only before this point; now kept as-is too.
                Some(ms.dv_dsm_corrected.clone()),
                ms.dv_total_ms,
                Some(ms.converged),
            ),
            Err(e) => {
                eprintln!(
                    "Warning: MGA multiple-shooting refinement failed ({e}) — API result falls \
                     back to the uncorrected search-time arc/DSMs."
                );
                // Fresh evaluation of the uncorrected search-time chromosome
                // to recover real per-leg DSM vectors (mga_leg.rs::
                // MgaLegResult::v_dsm_after_mps - v_dsm_before_mps) --
                // MgaResult itself only ever kept dv_dsms_ms's magnitudes.
                // None (not a mismatched-length placeholder) if this
                // re-evaluation itself fails, e.g. the same infeasible
                // initial state that made multiple shooting error out.
                let dsm_vectors = mga::evaluate_chromosome_detailed(&r.best_params, cfg, almanac, r.dep_jd_base, &flyby_bodies)
                    .map(|ev| {
                        ev.legs.iter()
                            .map(|leg_eval| {
                                let dv = leg_eval.leg.v_dsm_after_mps - leg_eval.leg.v_dsm_before_mps;
                                [dv.x, dv.y, dv.z]
                            })
                            .collect::<Vec<[f64; 3]>>()
                    });
                (mga_arc_to_api(mga_src), r.dv_dsms_ms.clone(), dsm_vectors, r.dv_total_ms, None)
            }
        };
        // Real target-body position at arrival (Phase 9k-v bug #1 — this
        // used to be left at the origin for every MGA result, which put the
        // target body on top of the Sun in the frontend's 3D view). Computed
        // independently here rather than reading it off `MgaResult` (which
        // has no such field) to avoid touching `mga.rs`, which had
        // uncommitted, unrelated work in flight when this was fixed.
        let target_arr_state = r.body_sequence.last()
            .and_then(|name| anise_body(&name.to_lowercase()))
            .and_then(|target_anise| {
                let arr_jd = r.dep_jd + r.tof_total_days;
                body_state(almanac, EphemerisSource::Anise, Some(target_anise), &None, arr_jd)
            });
        let target_r_arr_m = target_arr_state.map(|(r_arr, _)| r_arr).unwrap_or([0.0; 3]);
        let target_v_arr_mps = target_arr_state.map(|(_, v_arr)| v_arr).unwrap_or([0.0; 3]);

        // Phase 12c, upgraded same day after 9y-h gave the
        // final leg a real, non-degenerate target to converge onto: the
        // actual resulting captured orbit, only for Orbit/Landing missions
        // whose multiple-shooting refinement genuinely converged — see
        // `OptimizeApiResult::post_capture_orbit_arc`'s doc comment for why
        // non-converged/errored MS is deliberately excluded rather than
        // building an orbit on an uncorrected arc.
        //
        // Preferred path: `ms_result`'s `arrival_r_rel_m`/`arrival_v_rel_mps`
        // (`mga.rs::ms_arrival_relative_state`) — a REAL propagated crossing
        // state at the converged final leg's end, the same construction the
        // single-leg path's `eval.arrival.r_rel_m`/`v_rel_mps` already uses.
        // Before 9y-h this didn't exist (the final leg targeted the body's
        // exact center, a degenerate "orbit" with no real geometry), which
        // is why this used to fall back unconditionally to an ecliptic-proxy
        // convention (coplanar with the target body's own heliocentric
        // orbit) — that convention is now only a defensive fallback for the
        // rare case the real state's own re-propagation/ephemeris query
        // fails despite MS reporting `converged: true`.
        // Real config eccentricity is only still needed by the no-real-
        // crossing fallback branch below (an idealized seed, no real burn
        // to apply) -- the primary branch now derives the resulting shape
        // from the REAL crossing velocity direction instead (
        // see `propagate_captured_orbit`'s own header for the full reasoning).
        let capture_eccentricity = cfg.trajectory.capture.as_ref()
            .map(|c| c.capture_eccentricity)
            .unwrap_or(0.0);
        let post_capture_orbit: Option<(Vec<ArcApiPoint>, CapturedOrbitSummary)> = if matches!(cfg.mission.objective, MissionObjective::Orbit | MissionObjective::Landing)
            && mga_ms_converged == Some(true)
        {
            let real_state = ms_result.as_ref().ok().and_then(|ms| {
                let r = ms.arrival_r_rel_m?;
                let v = ms.arrival_v_rel_mps?;
                Some((Vector3::new(r[0], r[1], r[2]), Vector3::new(v[0], v[1], v[2])))
            });
            match real_state {
                // Real crossing state available: apply the real arrival
                // burn (minimum-ΔV speed correction to local circular
                // speed, same magnitude `dv_capture_ms`-style pricing
                // already assumed) along the REAL crossing velocity's own
                // direction -- see `propagate_captured_orbit`'s doc comment
                // for why this is always bound (e<1) regardless of how far
                // the real crossing is from periapsis.
                Some((r_rel, v_rel)) => (|| {
                    let target_name = r.body_sequence.last()?;
                    let cat = body_models::TargetBody::by_name(target_name)?;
                    let r_cap_m = r_rel.norm();
                    // Same periapsis speed `mga.rs::arrival_dv_ms` priced the
                    // capture burn against (eccentricity included) -- the
                    // ring and the ΔV must agree (Phase 14f).
                    let v_post = v_rel.normalize() * capture_target_speed_mps(cfg, cat.mu_m3s2, r_cap_m);
                    let (points, summary) = propagate_captured_orbit(r_rel, v_post, cat.mu_m3s2);
                    let origin_m = Vector3::new(target_r_arr_m[0], target_r_arr_m[1], target_r_arr_m[2]);
                    let origin_v_mps = Vector3::new(target_v_arr_mps[0], target_v_arr_mps[1], target_v_arr_mps[2]);
                    // Time-resolved (frozen-origin fix): anchor
                    // is the arrival crossing (t_s = 0, epoch arr_jd).
                    let target_anise = anise_body(&target_name.to_lowercase());
                    let arr_jd = r.dep_jd + r.tof_total_days;
                    let api_points = captured_orbit_to_api(
                        points, target_name, 0.0,
                        |dt_s| target_anise.and_then(|b| body_state(almanac, EphemerisSource::Anise, Some(b), &None, arr_jd + dt_s / 86_400.0))
                            .map(|(rb, vb)| (Vector3::new(rb[0], rb[1], rb[2]), Vector3::new(vb[0], vb[1], vb[2]))),
                        origin_m, origin_v_mps,
                    );
                    Some((api_points, summary))
                })(),
                // No real crossing state (rare -- MS reported converged but
                // its own re-propagation/ephemeris query failed): no real
                // burn to apply, so this stays the old idealized tangential
                // seed at the configured eccentricity -- an honest
                // approximation for a genuinely missing real state, not a
                // silently-disagreeing shortcut.
                None => target_arr_state.and_then(|(r_t, v_t)| {
                    let target_name = r.body_sequence.last()?;
                    let cat = body_models::TargetBody::by_name(target_name)?;
                    let r_cap_m = cfg.trajectory.capture.as_ref()
                        .and_then(|c| c.target_orbit_radius_m)
                        .unwrap_or(cat.radius_m * 3.0)
                        .max(cat.radius_m);
                    let r_t_v = Vector3::new(r_t[0], r_t[1], r_t[2]);
                    let v_t_v = Vector3::new(v_t[0], v_t[1], v_t[2]);
                    let plane_normal_hat = r_t_v.cross(&v_t_v).normalize();
                    let r_hat = r_t_v.normalize();
                    let t_hat = plane_normal_hat.cross(&r_hat);
                    let v_peri = (cat.mu_m3s2 * (1.0 + capture_eccentricity) / r_cap_m).sqrt();
                    let (points, summary) = propagate_captured_orbit(r_hat * r_cap_m, t_hat * v_peri, cat.mu_m3s2);
                    // Same time-resolved anchoring as the real-state branch
                    // above (anchor: arrival epoch, t_s = 0).
                    let target_anise = anise_body(&target_name.to_lowercase());
                    let arr_jd = r.dep_jd + r.tof_total_days;
                    let api_points = captured_orbit_to_api(
                        points, target_name, 0.0,
                        |dt_s| target_anise.and_then(|b| body_state(almanac, EphemerisSource::Anise, Some(b), &None, arr_jd + dt_s / 86_400.0))
                            .map(|(rb, vb)| (Vector3::new(rb[0], rb[1], rb[2]), Vector3::new(vb[0], vb[1], vb[2]))),
                        r_t_v, v_t_v,
                    );
                    Some((api_points, summary))
                }),
            }
        } else {
            None
        };
        let post_capture_orbit_arc = post_capture_orbit.as_ref().map(|(arc, _)| arc.clone());
        let post_capture_orbit_period_s = post_capture_orbit.as_ref().map(|(_, s)| s.period_s);
        let post_capture_orbit_periapsis_m = post_capture_orbit.as_ref().map(|(_, s)| s.periapsis_m);
        let post_capture_orbit_apoapsis_m = post_capture_orbit.as_ref().map(|(_, s)| s.apoapsis_m);
        let post_capture_orbit_eccentricity = post_capture_orbit.as_ref().map(|(_, s)| s.eccentricity);
        let post_capture_orbit_sma_m = post_capture_orbit.as_ref().map(|(_, s)| s.sma_m);
        let post_capture_orbit_inclination_deg = post_capture_orbit.as_ref().map(|(_, s)| s.inclination_deg);

        // Phase 12: departure-body SOI-radius ring, same
        // approximation convention as the arrival ring just above (MGA has
        // no committed real burn-location/direction to draw from without
        // touching `mga.rs`'s private chromosome-decode helpers -- 12j,
        // still open, real multiple-shooting fix -- see that function's doc
        // comment for why). Updated now uses the REAL periapsis
        // injection state (`mga::mga_departure_injection_state`, an exact
        // closed-form construction from the chromosome's already-optimized
        // departure v∞ vector -- no new chromosome dimension, no
        // approximation) instead of a stand-in SOI-radius ring, matching
        // single-leg's fidelity exactly. Always shown when the departure
        // body resolves (doesn't depend on mission objective or MS
        // convergence -- every MGA mission has a departure leg). Circular
        // (eccentricity 0.0) -- a parking orbit has no configured
        // eccentricity concept the way a capture orbit does.
        // Also yields the real departure ΔV vector: the
        // periapsis escape burn is purely tangential — parallel to the
        // injection velocity `v0_mps` — so the vector is simply
        // `v0_hat × dv_escape_ms`, the same magnitude convention
        // `dv_departure_ms` already prices. A velocity difference needs no
        // body-relative→inertial frame conversion (see the
        // `departure_dv_inertial_mps` field doc).
        let pre_departure = r.body_sequence.first().and_then(|dep_name| {
            let cat = body_models::TargetBody::by_name(dep_name)?;
            let (r0_free, v0_free, dv_escape_ms) =
                mga::mga_departure_injection_state(&r.best_params, cfg, almanac, r.dep_jd_base, &flyby_bodies)?;
            // `Launch` mode (Phase 14b/14d): same v∞, but the injection
            // plane is the site-feasible one from the launch geometry rather
            // than `hyperbolic_departure_state`'s arbitrary plane.
            let mga_launch_geom: Option<LaunchGeometry> = if departure_mode(cfg) == DepartureMode::Launch {
                asymptotic_v_infinity_geo(r0_free, v0_free, cat.mu_m3s2, r0_free.norm())
                    .and_then(|v_inf| launch_geometry_for(cfg, &cat, v_inf))
            } else {
                None
            };
            let (r0_m, v0_mps) = match mga_launch_geom.as_ref() {
                Some(g) => (g.injection_r_m, g.injection_v_mps),
                None => (r0_free, v0_free),
            };
            let mga_launch_geometry_api = mga_launch_geom.as_ref().map(|g| launch_geometry_api(cfg, g));
            let dep_dv_vec = v0_mps.normalize() * dv_escape_ms;
            // Departure C3 [km²/s²] from the real injection state (vis-viva
            // on the escape hyperbola: v∞² = v² − 2μ/r) -- for the launch-
            // vehicle check and ΔV ledger at THIS result's own energy.
            let departure_c3_km2s2 = (v0_mps.norm_squared() - 2.0 * cat.mu_m3s2 / r0_m.norm()) / 1.0e6;
            // Unchanged idealized-circular seed (this is the PARKING orbit
            // before departure, not an arrival capture -- no "real burn to
            // apply" concept here, `v0_mps` from `mga_departure_injection_
            // state` is the real INJECTION state, not the parking-orbit
            // velocity, so it must not be passed to `propagate_captured_
            // orbit` directly). Only the call convention changed (moved
            // inline since the function no longer builds this itself).
            let plane_normal_hat = r0_m.cross(&v0_mps).normalize();
            let r_hat = r0_m.normalize();
            let r_park_m = r0_m.norm();
            let t_hat = plane_normal_hat.cross(&r_hat);
            let v_circ = (cat.mu_m3s2 / r_park_m).sqrt();
            let (points, _summary) = propagate_captured_orbit(r_hat * r_park_m, t_hat * v_circ, cat.mu_m3s2);
            // Real heliocentric departure-body position at the same epoch
            // `mga_departure_injection_state` itself used (r.dep_jd_base) --
            // Phase 12k.
            let dep_anise = anise_body(&dep_name.to_lowercase())?;
            let (r_dep, v_dep) = body_state(almanac, EphemerisSource::Anise, Some(dep_anise), &None, r.dep_jd_base)?;
            let origin_m = Vector3::new(r_dep[0], r_dep[1], r_dep[2]);
            let origin_v_mps = Vector3::new(v_dep[0], v_dep[1], v_dep[2]);
            // Time-resolved (frozen-origin fix): the parking
            // ring leads up to the injection, so anchor = last sample at
            // epoch dep_jd_base (the same epoch
            // mga_departure_injection_state itself used).
            let t_s_anchor = points.last().map(|p| p.t_s).unwrap_or(0.0);
            let api_points = captured_orbit_to_api(
                points, dep_name, t_s_anchor,
                |dt_s| body_state(almanac, EphemerisSource::Anise, Some(dep_anise), &None, r.dep_jd_base + dt_s / 86_400.0)
                    .map(|(rb, vb)| (Vector3::new(rb[0], rb[1], rb[2]), Vector3::new(vb[0], vb[1], vb[2]))),
                origin_m, origin_v_mps,
            );
            Some((
                api_points,
                [dep_dv_vec.x, dep_dv_vec.y, dep_dv_vec.z],
                departure_c3_km2s2,
                mga_launch_geometry_api,
            ))
        });
        let (pre_departure_orbit_arc, departure_dv_inertial_mps, departure_c3_km2s2, mga_launch_geometry) = match pre_departure {
            Some((arc, dv, c3, lg)) => (Some(arc), Some(dv), Some(c3), lg),
            None => (None, None, None, None),
        };
        let launch_vehicle_check = departure_c3_km2s2
            .and_then(|c3| compute_launch_vehicle_check_for(cfg, &opt.departure_body, c3));
        let dsm_sum_ms: f64 = mga_dv_dsms_ms_final.iter().sum();
        let ledger = departure_c3_km2s2
            .map(|c3| dv_ledger(cfg, &opt.departure_body, r.dv_departure_ms, c3, r.dv_arrival_ms, dsm_sum_ms));

        // Population log (Phase 9k task 5): MGA has no per-individual
        // population like GA/PSO (DE/SHADE/MBH's own population isn't
        // exposed here) — this is the best-so-far chromosome per generation,
        // one row per phase1_history/convergence entry. `generation` is
        // 0-indexed WITHIN each phase (unlike GA/PSO's population_log, whose
        // `generation` matches the live-streamed OptimizeStepMsg.step value
        // globally) — a documented deviation, see openapi.json.
        let mga_population_log: Vec<PopulationLogRow> = r.phase1_param_history.iter()
            .zip(r.phase1_history.iter())
            .enumerate()
            .map(|(i, (params, &fitness))| PopulationLogRow { phase: 1, generation: i, params: params.clone(), fitness, dv_arrival_ms: None, tof_days: None })
            .chain(
                r.param_history.iter()
                    .zip(r.convergence.iter())
                    .enumerate()
                    .map(|(i, (params, &fitness))| PopulationLogRow { phase: 2, generation: i, params: params.clone(), fitness, dv_arrival_ms: None, tof_days: None })
            )
            .collect();
        return Ok(OptimizeApiResult {
            method: "MGA".into(),
            method_display: "Multi-Gravity-Assist (MGA-1DSM)".into(),
            objective: format!("{}", opt.objective),
            dep_offset_days: dep_offset,
            dep_jd: dep_jd_val,
            theta_burn_rad: 0.0,
            dv_departure_ms: r.dv_departure_ms,
            phi_out_of_plane_rad: 0.0,
            departure_mode: format!("{:?}", departure_mode(cfg)),
            departure_dv_pool: ledger.as_ref().map(|l| l.departure_dv_pool.clone()),
            injection_epoch_jd: dep_jd_val,
            launch_geometry: mga_launch_geometry,
            escape_duration_s: 0.0,
            achieved_tof_days: r.tof_total_days,
            dv_arrival_ms: r.dv_arrival_ms,
            capture_time_s: None,
            // MGA's own arrival-burn pricing (mga.rs) doesn't compute the
            // same ArrivalCapture crossing GA/PSO does below, so there's no
            // real v_rel_mps here to build a vector from yet -- matches
            // capture_time_s's own None above (a pre-existing MGA scope
            // gap, not new). Real follow-up, not done here.
            capture_dv_inertial_mps: None,
            departure_dv_inertial_mps,
            theta_arr_rad: None,
            phi_arr_rad: None,
            fitness: dv_total_final,
            miss_km: 0.0,
            launch_vehicle_check,
            dv_ledger: ledger,
            target_r_arr_m,
            convergence: r.convergence,
            objective_history: Vec::new(),
            phase1_convergence_km: r.phase1_history.clone(),
            population_log: mga_population_log,
            arc: mga_arc_api,
            mga_body_sequence: Some(r.body_sequence),
            mga_dv_dsms_ms: Some(mga_dv_dsms_ms_final),
            mga_dv_dsms_inertial_mps: mga_dv_dsms_inertial_mps_final,
            mga_dsm_positions_m: Some(r.dsm_positions_m),
            mga_dsm_epochs_s: Some(r.dsm_epochs_s.clone()),
            mga_leg_tofs_days: Some(r.leg_tofs_days),
            mga_phase1_history: Some(r.phase1_history),
            mga_ms_converged,
            mga_flyby_dv_gained_ms: if mga_ms_converged == Some(true) {
                ms_result.as_ref().ok().map(|ms| ms.flyby_dv_gained_ms.clone())
            } else {
                None
            },
            mga_flyby_speed_before_ms: if mga_ms_converged == Some(true) {
                ms_result.as_ref().ok().map(|ms| ms.flyby_speed_before_ms.clone())
            } else {
                None
            },
            mga_flyby_speed_after_ms: if mga_ms_converged == Some(true) {
                ms_result.as_ref().ok().map(|ms| ms.flyby_speed_after_ms.clone())
            } else {
                None
            },
            post_capture_orbit_arc,
            post_capture_orbit_period_s,
            post_capture_orbit_periapsis_m,
            post_capture_orbit_apoapsis_m,
            post_capture_orbit_eccentricity,
            post_capture_orbit_sma_m,
            post_capture_orbit_inclination_deg,
            pre_departure_orbit_arc,
        });
    }

    let r = run_optimization_with_progress(
        cfg,
        almanac,
        // Empty params (pre-first-feasible-candidate) become None on the
        // wire -- the frontend treats a step with no params as "nothing to
        // plot yet", same as MGA's own occasional param-less generations.
        |step, phase, best, feas, params, pop, outcomes| {
            on_step(
                step,
                phase,
                best,
                if feas < f64::MAX { Some(feas) } else { None },
                if params.is_empty() { None } else { Some(params) },
                &[],
                pop,
                outcomes,
            )
        },
        cancelled,
    )?;
    let ctx = build_context(cfg, opt, almanac)?;

    let dep_offset_days = r.best_params[0];
    let dep_jd = r.dep_jd_base + dep_offset_days;

    let eval = evaluate_candidate(&ctx, &r.best_params)
        .ok_or("could not re-evaluate the best point (ephemeris, escape, or capture unavailable)")?;

    let entries = force_model_body_entries(opt, almanac, dep_jd);
    let bodies = as_propagator_bodies(&entries);

    // Reporting-time arrival, two tiers (direct user directive:
    // "the optimizer tries to converge to a solution that is as close as
    // possible to the target radius... the final solution, even when its
    // not close enough, we will still fit an orbit to it even if
    // convergence is bad"):
    // 1. The REAL capture crossing when the best point achieved one
    //    (`eval.arrival`, unchanged).
    // 2. Otherwise, for an Orbit/Landing mission, a synthesized arrival at
    //    the best point's own CLOSEST APPROACH: the real relative state
    //    there, the real burn magnitude a capture at that radius would
    //    need, and the resulting orbit fitted from that state. `miss_km`
    //    still reports how far this sits from the requested radius, so a
    //    poorly-converged result reads as one -- but it now always shows
    //    the orbit its own burn would actually produce, instead of
    //    reporting no arrival at all (which previously left the arc
    //    coasting through the whole budget with no burn, no orbit, and no
    //    way to see what the found solution amounts to).
    // The SEARCH fitness is untouched by this -- it still drives toward
    // the configured radius; this is reporting only.
    let arrival_effective: Option<ArrivalCapture> = eval.arrival.clone().or_else(|| {
        if !matches!(cfg.mission.objective, MissionObjective::Orbit | MissionObjective::Landing) {
            return None;
        }
        let target_body_index = entries.iter().position(|e| e.name.eq_ignore_ascii_case(&opt.target_body))?;
        let (closest_point, _d) = eval.points.iter().fold((None::<&PropagatedPoint>, f64::INFINITY), |acc, p| {
            let (r_t, _) = (bodies[target_body_index].state_at)(p.t_s);
            let d = (p.r_m - r_t).norm();
            if d < acc.1 { (Some(p), d) } else { acc }
        });
        let p = closest_point?;
        let (r_t, v_t) = (bodies[target_body_index].state_at)(p.t_s);
        let r_rel_sample = p.r_m - r_t;
        let v_rel_sample = p.v_mps - v_t;
        if r_rel_sample.norm() < 1.0 {
            return None; // degenerate -- sitting on the body's center
        }
        let mu_t = bodies[target_body_index].mu_m3s2;
        // Refine the SAMPLED closest approach to the true osculating
        // periapsis (
        // too small perigee i think and its not centered around mercury" --
        // see osculating_periapsis_state's own doc comment for the
        // sampled-point radial-velocity mechanism behind that). Falls back
        // to the raw sample when the refinement is degenerate/unbracketed.
        let (r_rel_m, v_rel_mps, capture_time_s) = match (
            osculating_periapsis_state(r_rel_sample, v_rel_sample, mu_t),
            time_to_periapsis_s(r_rel_sample, v_rel_sample, mu_t, 5.0 * 86_400.0),
        ) {
            (Some((r_p, v_p)), Some(dt_s)) => (r_p, v_p, p.t_s + dt_s),
            _ => (r_rel_sample, v_rel_sample, p.t_s),
        };
        // SOI capture ceiling (a seemingly-great MinDeltaV
        // result once had a 27.2M km perigee, OUTSIDE Mercury's own SOI --
        // such an orbit is infeasible and cannot be propagated):
        // no synthesized capture beyond CAPTURE_MAX_SOI_FRACTION x the
        // target's SOI. The two-body ring propagation (and the very notion
        // of a target-centric orbit) is only physically meaningful well
        // inside the SOI; beyond it, report honestly that no capture was
        // achieved (the stats then show the flyby/closest-approach numbers
        // instead of a fictitious orbit).
        if let Some(soi_m) = eval.capture_rp_max_m {
            if r_rel_m.norm() > soi_m {
                return None;
            }
        }
        let v_target = capture_target_speed_mps(cfg, mu_t, r_rel_m.norm());
        let (r_t_cap, v_t_cap) = (bodies[target_body_index].state_at)(capture_time_s);
        let (theta_arr_rad, phi_arr_rad) = arrival_burn_angles(r_rel_m, v_rel_mps, r_t_cap, v_t_cap);
        Some(ArrivalCapture {
            dv_capture_ms: (v_rel_mps.norm() - v_target).abs(),
            capture_time_s,
            theta_arr_rad,
            phi_arr_rad,
            r_rel_m,
            v_rel_mps,
        })
    });

    let achieved_tof_days = arrival_effective.as_ref().map(|a| a.capture_time_s / 86_400.0).unwrap_or(eval.closest_approach_days);
    let dv_arrival_ms = arrival_effective.as_ref().map(|a| a.dv_capture_ms).unwrap_or(0.0);

    // Phase 12c: the actual resulting captured orbit, re-seeded
    // with a post-burn velocity at the REAL propagated crossing point
    // (`eval.arrival.r_rel_m`/`v_rel_mps`) -- unlike the MGA branch above,
    // this path has a genuine propagated crossing state to build the
    // orbital plane from (see `propagate_captured_orbit`'s doc comment for
    // why the plane it preserves is exactly the `dv_capture_ms` model's own
    // minimum-ΔV, plane-preserving assumption). `None` for `Flyby`/
    // `Rendezvous`/`SampleReturn` only -- an Orbit/Landing candidate that
    // never achieved a real crossing now gets the closest-approach-fitted
    // arrival instead (`arrival_effective`, see above), so this is always
    // present for a capture-type mission.
    //
    // Real burn, real direction (- see `propagate_captured_
    // orbit`'s own header). `dv_capture_ms` (`ArrivalCapture` in this file)
    // is priced against `capture_target_speed_mps` -- the configured capture
    // orbit's periapsis speed, eccentricity included since Phase 14f -- and
    // the RESULTING ORBIT SHAPE is derived from the real crossing velocity's
    // own direction scaled to that same magnitude, so the displayed ring and
    // the priced ΔV agree by construction regardless of the real geometry
    // (a tangential real crossing reproduces the configured eccentricity
    // exactly; a radial component makes the real orbit's own eccentricity
    // differ, which `post_capture_orbit_eccentricity` reports honestly).
    let post_capture_orbit = if matches!(cfg.mission.objective, MissionObjective::Orbit | MissionObjective::Landing) {
        arrival_effective.as_ref().and_then(|a| {
            let cat = body_models::TargetBody::by_name(&opt.target_body)?;
            let r_cap_m = a.r_rel_m.norm();
            let v_post = a.v_rel_mps.normalize() * capture_target_speed_mps(cfg, cat.mu_m3s2, r_cap_m);
            let (points, summary) = propagate_captured_orbit(a.r_rel_m, v_post, cat.mu_m3s2);
            // Real heliocentric target-body position at the real capture
            // time (not the nominal dep_jd + achieved_tof_days used
            // elsewhere in this function for the *arrival* re-evaluation --
            // `a.capture_time_s` is this candidate's own actual crossing
            // time, which is what `eval.arrival` itself was built from) --
            // Phase 12k.
            let target_anise = anise_body(&opt.target_body.to_lowercase())?;
            let capture_jd = dep_jd + a.capture_time_s / 86_400.0;
            let (r_tgt, v_tgt) = body_state(ctx.almanac, EphemerisSource::Anise, Some(target_anise), &None, capture_jd)?;
            let origin_m = Vector3::new(r_tgt[0], r_tgt[1], r_tgt[2]);
            let origin_v_mps = Vector3::new(v_tgt[0], v_tgt[1], v_tgt[2]);
            // Time-resolved: each ring sample translated by the
            // target's real state at its own epoch (anchor: the capture
            // crossing, t_s = 0), not the single capture-epoch state.
            let almanac = ctx.almanac;
            let api_points = captured_orbit_to_api(
                points, &opt.target_body, 0.0,
                |dt_s| body_state(almanac, EphemerisSource::Anise, Some(target_anise), &None, capture_jd + dt_s / 86_400.0)
                    .map(|(r, v)| (Vector3::new(r[0], r[1], r[2]), Vector3::new(v[0], v[1], v[2]))),
                origin_m, origin_v_mps,
            );
            Some((api_points, summary))
        })
    } else {
        None
    };
    let post_capture_orbit_arc = post_capture_orbit.as_ref().map(|(arc, _)| arc.clone());
    let post_capture_orbit_period_s = post_capture_orbit.as_ref().map(|(_, s)| s.period_s);
    let post_capture_orbit_periapsis_m = post_capture_orbit.as_ref().map(|(_, s)| s.periapsis_m);
    let post_capture_orbit_apoapsis_m = post_capture_orbit.as_ref().map(|(_, s)| s.apoapsis_m);
    let post_capture_orbit_eccentricity = post_capture_orbit.as_ref().map(|(_, s)| s.eccentricity);
    let post_capture_orbit_sma_m = post_capture_orbit.as_ref().map(|(_, s)| s.sma_m);
    let post_capture_orbit_inclination_deg = post_capture_orbit.as_ref().map(|(_, s)| s.inclination_deg);

    // Phase 12: the real parking orbit leading up to the burn
    // -- recomputing the same plane/burn-point construction `evaluate_candidate`
    // used (cheap, deterministic from cfg + ephemeris), since it isn't
    // itself part of `CandidateEvaluation`. `burn.r0_m` IS the real burn
    // point, so this orbit ends exactly where the returned `arc` begins --
    // no seam, no approximation.
    // `Launch` mode (Phase 14b/14d): the closed-form launch geometry the
    // winning chromosome implies — `None` in `ParkingOrbit` mode.
    let launch_geom: Option<LaunchGeometry> = body_models::TargetBody::by_name(&opt.departure_body)
        .and_then(|b| launch_geometry_for_params(cfg, &b, &r.best_params));
    let launch_geometry_result = launch_geom.as_ref().map(|g| launch_geometry_api(cfg, g));
    // The physical departure burn: the free burn gene in `ParkingOrbit`
    // mode, the launch geometry's tangential injection burn in `Launch` mode
    // (where the gene is the v∞ instead).
    let dv_departure_real_ms = launch_geom.as_ref().map(|g| g.dv_injection_ms).unwrap_or(r.best_params[2]);
    let pre_departure = (|| {
        let (r_dep, v_dep) = body_state(ctx.almanac, EphemerisSource::Anise, Some(ctx.dep_anise), &None, dep_jd)?;
        let r_dep_v = Vector3::new(r_dep[0], r_dep[1], r_dep[2]);
        let v_dep_v = Vector3::new(v_dep[0], v_dep[1], v_dep[2]);
        let dep_catalog = body_models::TargetBody::by_name(&opt.departure_body)?;
        let r_park_m = resolve_parking_orbit_radius_m(cfg, &dep_catalog);
        let plane_normal_hat = r_dep_v.cross(&v_dep_v).normalize();
        let reference_dir_hat = r_dep_v.normalize();
        // Parking-orbit state before the burn, and the post-burn injection
        // state, from the SAME construction the search used:
        // `ParkingOrbit` mode — `circular_orbit_burn_state` on the free burn
        // genes (`burn.r0_m`/`burn.v0_mps` IS the real circular parking-orbit
        // state); `Launch` mode (Phase 14b) — the closed-form
        // launch geometry's parking/injection pair on the site-feasible plane.
        let (r0_park, v_park, v_post) = match launch_geom.as_ref() {
            Some(g) => (g.injection_r_m, g.parking_v_mps, g.injection_v_mps),
            None => {
                let burn = circular_orbit_burn_state(
                    dep_catalog.mu_m3s2, r_park_m, plane_normal_hat, reference_dir_hat, r.best_params[1], 0.0, 0.0,
                );
                let post_burn = circular_orbit_burn_state(
                    dep_catalog.mu_m3s2, r_park_m, plane_normal_hat, reference_dir_hat,
                    r.best_params[1], r.best_params[2], r.best_params[3],
                );
                (burn.r0_m, burn.v0_mps, post_burn.v0_mps)
            }
        };
        let (points, _summary) = propagate_captured_orbit(r0_park, v_park, dep_catalog.mu_m3s2);
        // Real departure ΔV vector: post-burn minus pre-burn
        // velocity from the exact same construction the search itself used
        // (`evaluate_candidate`'s injection state), so `|dv|` equals
        // `dv_departure_ms` by construction — see the
        // `departure_dv_inertial_mps` field doc for the frame reasoning.
        let dep_dv_vec = v_post - v_park;
        // Departure C3 [km²/s²] of the search's own (generally non-
        // tangential) injection: vis-viva on the post-burn state, v∞² = v² −
        // 2μ/r_park -- the energy the launch-vehicle check and ΔV ledger
        // are evaluated at for THIS result.
        let departure_c3_km2s2 = (v_post.norm_squared() - 2.0 * dep_catalog.mu_m3s2 / r0_park.norm()) / 1.0e6;
        // Time-resolved: the parking-orbit ring leads UP TO
        // the burn, so the anchor is its LAST sample (== the burn point,
        // epoch dep_jd); earlier samples map to real epochs BEFORE dep_jd
        // and get the departure body's real state there — the frozen-origin
        // fix (a 2.95 h Earth ring used to ignore ~316,000 km of the
        // Earth's own motion over its span).
        let t_s_anchor = points.last().map(|p| p.t_s).unwrap_or(0.0);
        let almanac = ctx.almanac;
        let dep_anise = ctx.dep_anise;
        let api_points = captured_orbit_to_api(
            points, &opt.departure_body, t_s_anchor,
            |dt_s| body_state(almanac, EphemerisSource::Anise, Some(dep_anise), &None, dep_jd + dt_s / 86_400.0)
                .map(|(r, v)| (Vector3::new(r[0], r[1], r[2]), Vector3::new(v[0], v[1], v[2]))),
            r_dep_v, v_dep_v,
        );
        Some((
            api_points,
            [dep_dv_vec.x, dep_dv_vec.y, dep_dv_vec.z],
            departure_c3_km2s2,
        ))
    })();
    let (pre_departure_orbit_arc, departure_dv_inertial_mps, departure_c3_km2s2) = match pre_departure {
        Some((arc, dv, c3)) => (Some(arc), Some(dv), Some(c3)),
        None => (None, None, None),
    };
    let launch_vehicle_check = departure_c3_km2s2
        .and_then(|c3| compute_launch_vehicle_check_for(cfg, &opt.departure_body, c3));
    let ledger = departure_c3_km2s2
        .map(|c3| dv_ledger(cfg, &opt.departure_body, dv_departure_real_ms, c3, dv_arrival_ms, 0.0));

    let target_anise = anise_body(&opt.target_body.to_lowercase())
        .ok_or_else(|| format!("optimization.target_body '{}' is not ANISE-covered", opt.target_body))?;
    let arr_jd = dep_jd + achieved_tof_days;
    let (r_arr, _) = body_state(almanac, EphemerisSource::Anise, Some(target_anise), &None, arr_jd)
        .ok_or("could not re-evaluate arrival state at the best point")?;

    let arc = arc_to_api(eval.points, &bodies);

    let method = match opt.method {
        OptimizationMethod::GA => "GA",
        OptimizationMethod::PSO => "PSO",
        OptimizationMethod::MultipleShooting => "MultipleShooting",
        OptimizationMethod::MGA => "MGA",
    };

    Ok(OptimizeApiResult {
        method: method.to_string(),
        method_display: format!("{}", opt.method),
        objective: format!("{}", opt.objective),
        dep_offset_days,
        dep_jd,
        theta_burn_rad: r.best_params[1],
        dv_departure_ms: dv_departure_real_ms,
        phi_out_of_plane_rad: r.best_params[3],
        departure_mode: format!("{:?}", departure_mode(cfg)),
        departure_dv_pool: ledger.as_ref().map(|l| l.departure_dv_pool.clone()),
        injection_epoch_jd: dep_jd,
        launch_geometry: launch_geometry_result,
        escape_duration_s: eval.escape_duration_s,
        achieved_tof_days,
        dv_arrival_ms,
        capture_time_s: arrival_effective.as_ref().map(|a| a.capture_time_s),
        // Tangential (retrograde) insertion burn: opposes the real incoming
        // relative-velocity direction, magnitude exactly dv_capture_ms (the
        // same value dv_arrival_ms above already exposes) -- see this
        // field's own doc comment on OptimizeApiResult for why no frame
        // conversion is needed for a velocity DIFFERENCE.
        capture_dv_inertial_mps: arrival_effective.as_ref().map(|a| {
            let dir = a.v_rel_mps.normalize();
            let dv = -dir * a.dv_capture_ms;
            [dv.x, dv.y, dv.z]
        }),
        departure_dv_inertial_mps,
        theta_arr_rad: arrival_effective.as_ref().map(|a| a.theta_arr_rad),
        phi_arr_rad: arrival_effective.as_ref().map(|a| a.phi_arr_rad),
        fitness: r.best_fitness,
        miss_km: r.miss_km,
        launch_vehicle_check,
        dv_ledger: ledger,
        target_r_arr_m: r_arr,
        convergence: r.history,
        objective_history: r.objective_history,
        phase1_convergence_km: r.phase1_history_km,
        population_log: r.population_log,
        arc,
        mga_body_sequence: None,
        mga_dv_dsms_ms: None,
        mga_dv_dsms_inertial_mps: None,
        mga_dsm_positions_m: None,
        mga_dsm_epochs_s: None,
        mga_leg_tofs_days: None,
        mga_phase1_history: None,
        mga_ms_converged: None,
        mga_flyby_dv_gained_ms: None,
        mga_flyby_speed_before_ms: None,
        mga_flyby_speed_after_ms: None,
        post_capture_orbit_arc,
        post_capture_orbit_period_s,
        post_capture_orbit_periapsis_m,
        post_capture_orbit_apoapsis_m,
        post_capture_orbit_eccentricity,
        post_capture_orbit_sma_m,
        post_capture_orbit_inclination_deg,
        pre_departure_orbit_arc,
    })
}

// ── Departure-geometry diagnostic ───────────────────────────────────────────────

/// Writes CSVs that let a human visually check what `theta_burn` and
/// `phi_out_of_plane` actually mean geometrically, using the real
/// departure-body ephemeris at the best point's epoch (the same
/// construction `evaluate_candidate`/`circular_orbit_burn_state` use):
/// - `<prefix>_departure_geometry_orbit.csv`: the full parking-orbit circle
///   (every 5° of theta), so theta=0's reference direction and the burn's
///   actual position on the ring are both visible.
/// - `<prefix>_departure_geometry_vectors.csv`: named start/end points
///   (body-centered, km) for the reference axis (theta=0), the orbital-plane
///   normal axis, the pure-circular velocity vs. the actual departure
///   velocity at the burn point (their angle visually *is* phi, length
/// difference visually *is* dv) -- and, found necessary after
///   a real theta-bound experiment made results *worse*: the periapsis
///   velocity direction is NOT the escape (v-infinity) direction. A
///   meaningfully hyperbolic departure (e well above 1) turns the velocity
///   vector by `asin(1/e)` between periapsis and the outgoing asymptote
///   (same physics as a gravity-assist turn, halved -- a flyby turns twice,
///   once incoming once outgoing). For this binary's typical e~1.3-1.4 that's
///   a real ~45-55 degree rotation, not a rounding-error correction -- a
///   burn whose *periapsis* velocity looks retrograde-ish relative to the
///   departure body's own heliocentric velocity can still produce a usefully
///   prograde *escape* direction once this rotation is accounted for. The
///   `v_infinity_asymptotic` row makes this visible directly.
/// - `<prefix>_departure_geometry_helio.csv`: the same escape asymptote,
///   composed with the departure body's own heliocentric velocity, drawn in
///   the *heliocentric* frame (AU) alongside the Sun, the departure body, and
///   the target body's real position both at departure and at the achieved
///   arrival/closest-approach time -- so the escape direction's alignment
///   with "where the target actually ends up" can be checked visually, not
///   just inferred from the body-centered picture above.
///
/// Returns `None` (and writes nothing) if the departure body isn't
/// ANISE-covered or its ephemeris/catalog lookup fails at this epoch --
/// same fallibility as `evaluate_candidate`'s own burn construction.
fn write_departure_geometry_csvs(
    cfg: &MissionConfig,
    opt: &OptimizationConfig,
    almanac: &Almanac,
    dep_jd: f64,
    theta_burn: f64,
    dv_mps: f64,
    phi: f64,
    target_r_arr_m: [f64; 3],
    out_dir: &str,
    method_prefix: &str,
) -> Option<()> {
    if departure_mode(cfg) == DepartureMode::Launch {
        // The (theta, dv, phi) burn picture this diagnostic draws doesn't
        // exist in `Launch` mode -- the departure is (RLA, v∞, DLA) and its
        // geometry is reported directly on the result (`launch_geometry`).
        println!("  (Launch departure mode: burn-geometry CSVs not applicable, see launch_geometry on the result)");
        return Some(());
    }
    let dep_anise = anise_body(&opt.departure_body.to_lowercase())?;
    let (r_dep, v_dep) = body_state(almanac, EphemerisSource::Anise, Some(dep_anise), &None, dep_jd)?;
    let r_dep_v = Vector3::new(r_dep[0], r_dep[1], r_dep[2]);
    let v_dep_v = Vector3::new(v_dep[0], v_dep[1], v_dep[2]);

    let dep_catalog = body_models::TargetBody::by_name(&opt.departure_body)?;
    let r_park_m = resolve_parking_orbit_radius_m(cfg, &dep_catalog);

    let plane_normal_hat = r_dep_v.cross(&v_dep_v).normalize();
    let reference_dir_hat = r_dep_v.normalize();
    let v_circ = (dep_catalog.mu_m3s2 / r_park_m).sqrt();

    // Parking-orbit ring: theta=0 is the burn body's own instantaneous
    // heliocentric radial direction; theta increases in the body's own
    // prograde sense (same rotation `circular_orbit_burn_state` applies to
    // the burn position).
    let mut orbit_rows = vec!["theta_deg,x_km,y_km,z_km".to_string()];
    for i in 0..=72 {
        let theta = (i as f64) * std::f64::consts::TAU / 72.0;
        let burn = circular_orbit_burn_state(dep_catalog.mu_m3s2, r_park_m, plane_normal_hat, reference_dir_hat, theta, 0.0, 0.0);
        orbit_rows.push(format!(
            "{:.1},{:.3},{:.3},{:.3}",
            theta.to_degrees(), burn.r0_m.x / 1e3, burn.r0_m.y / 1e3, burn.r0_m.z / 1e3
        ));
    }
    let orbit_path = format!("{out_dir}/{method_prefix}_departure_geometry_orbit.csv");
    std::fs::write(&orbit_path, orbit_rows.join("\n") + "\n").ok()?;
    println!("  {orbit_path}");

    let burn_circ = circular_orbit_burn_state(dep_catalog.mu_m3s2, r_park_m, plane_normal_hat, reference_dir_hat, theta_burn, 0.0, 0.0);
    let burn_actual = circular_orbit_burn_state(dep_catalog.mu_m3s2, r_park_m, plane_normal_hat, reference_dir_hat, theta_burn, dv_mps, phi);
    let burn_pos_km = burn_circ.r0_m / 1e3;

    // Velocity arrows are drawn from the burn position, scaled so a vector
    // of magnitude v_circ spans 40% of the parking radius -- a visual scale
    // comparable to the orbit ring itself, not raw km/s next to raw km.
    let vel_scale_km_per_mps = (r_park_m / 1e3) * 0.4 / v_circ;
    let vel_arrow = |v: Vector3<f64>| burn_pos_km + v * vel_scale_km_per_mps;
    let axis_arrow = |dir: Vector3<f64>| dir * (r_park_m / 1e3) * 1.3;

    // Real propagated escape leg, Earth-centered (km) -- found necessary
    // after the burn-position velocity arrows (drawn at parking-
    // orbit scale, ~r_park) caused real confusion about why they don't point
    // "toward" departure_body_velocity_hat: the spacecraft's *geocentric* velocity rotates and
    // shrinks continuously from the burn (periapsis, ~v_circ+dv) toward the
    // asymptotic v_infinity as it climbs out of Earth's gravity well -- it
    // doesn't "reach" v_dep at all (v_dep is Earth's own unrelated absolute
    // motion, never something the geocentric escape converges to). This
    // section runs the real SOI-patched propagator (same physics
    // `evaluate_candidate` uses, just for a short fixed window -- cheap,
    // independent of the full GA search) to find where the leg actually
    // crosses Earth's real SOI and what the real geocentric velocity is
    // there, for direct comparison against the idealized `v_infinity_asymptotic`.
    let soi_radius_m = laplace_soi_radius_m(r_dep_v.norm(), dep_catalog.mu_m3s2 / MU_SUN_M3S2);
    let dep_jd_for_state = dep_jd;
    let earth_state_at = |t_abs_s: f64| {
        let jd = dep_jd_for_state + t_abs_s / 86_400.0;
        let (r, v) = body_state(almanac, EphemerisSource::Anise, Some(dep_anise), &None, jd)
            .unwrap_or((r_dep, v_dep));
        (Vector3::new(r[0], r[1], r[2]), Vector3::new(v[0], v[1], v[2]))
    };
    let dep_body_only = [PropagatorBody {
        name: opt.departure_body.as_str(),
        mu_m3s2: dep_catalog.mu_m3s2,
        soi_radius_m: Some(soi_radius_m),
        state_at: &earth_state_at,
        central_fidelity: None,
        radius_m: Some(dep_catalog.radius_m),
    }];
    let r0_helio = r_dep_v + burn_actual.r0_m;
    let v0_helio = v_dep_v + burn_actual.v0_mps;
    // 20 days is comfortably longer than any realistic escape duration for
    // this catalog's parking-orbit radii (typically 2-5 days) -- cheap
    // regardless, since this is a short window, not the full coast.
    let escape_window_s = 20.0 * 86_400.0;
    let escape_points = propagate(r0_helio, v0_helio, 0.0, escape_window_s, MU_SUN_M3S2, &dep_body_only, 3_600.0, 1e-10, 1e-3);
    let crossing = escape_points.iter().find(|p| p.central_body_index != Some(0));

    let real_leg_path = format!("{out_dir}/{method_prefix}_departure_geometry_real_leg.csv");
    let mut leg_rows = vec!["t_s,x_km,y_km,z_km".to_string()];
    let leg_end_idx = crossing.map(|c| escape_points.iter().position(|p| p.t_s == c.t_s).unwrap() + 1).unwrap_or(escape_points.len());
    for p in &escape_points[..leg_end_idx] {
        let (earth_r, _) = earth_state_at(p.t_s);
        let rel = (p.r_m - earth_r) / 1e3;
        leg_rows.push(format!("{:.3},{:.3},{:.3},{:.3}", p.t_s, rel.x, rel.y, rel.z));
    }
    std::fs::write(&real_leg_path, leg_rows.join("\n") + "\n").ok()?;
    println!("  {real_leg_path}");

    let meta_path = format!("{out_dir}/{method_prefix}_departure_geometry_soi.csv");
    let crossing_info = crossing.map(|c| {
        let (earth_r, earth_v) = earth_state_at(c.t_s);
        (c.t_s, (c.r_m - earth_r) / 1e3, c.v_mps - earth_v)
    });
    let soi_row = match &crossing_info {
        Some((t_s, pos_km, _)) => format!(
            "soi_radius_km,crossed,crossing_t_s,crossing_x_km,crossing_y_km,crossing_z_km\n{:.3},true,{:.3},{:.3},{:.3},{:.3}",
            soi_radius_m / 1e3, t_s, pos_km.x, pos_km.y, pos_km.z,
        ),
        None => format!("soi_radius_km,crossed,crossing_t_s,crossing_x_km,crossing_y_km,crossing_z_km\n{:.3},false,,,,", soi_radius_m / 1e3),
    };
    std::fs::write(&meta_path, soi_row + "\n").ok()?;
    println!("  {meta_path}");

    let vectors_path = format!("{out_dir}/{method_prefix}_departure_geometry_vectors.csv");
    let mut vec_rows = vec!["name,x0_km,y0_km,z0_km,x1_km,y1_km,z1_km".to_string()];
    let mut push = |name: &str, p0: Vector3<f64>, p1: Vector3<f64>| {
        vec_rows.push(format!("{name},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}", p0.x, p0.y, p0.z, p1.x, p1.y, p1.z));
    };
    push("reference_dir_theta0", Vector3::zeros(), axis_arrow(reference_dir_hat));
    push("orbital_plane_normal", Vector3::zeros(), axis_arrow(plane_normal_hat));
    // The departure body's own heliocentric velocity *direction* (drawn from
    // the origin, fixed length like the reference axes above -- its real
    // magnitude, ~30 km/s for Earth, dwarfs every burn-velocity arrow below
    // and would make this panel useless at a shared scale). Added
    // specifically so this frame's orientation can be checked
    // directly against the heliocentric panel's geometry instead of
    // inferred. Named `departure_body_velocity_hat`, not `v_dep_hat` --
    // `v_dep` already means "departure body's velocity" throughout this
    // file's Rust code (`r_dep`/`v_dep`/`dep_jd`), but a bare "v_dep" label
    // on a plotted vector reads the opposite way ("departure burn
    // velocity") -- real confusion this caused, worth avoiding by being
    // unambiguous in anything user-facing.
    push("departure_body_velocity_hat", Vector3::zeros(), axis_arrow(v_dep_v.normalize()));
    push("burn_position", Vector3::zeros(), burn_pos_km);
    push("v_circular_only_ref", burn_pos_km, vel_arrow(burn_circ.v0_mps));
    push("v_departure_actual_best", burn_pos_km, vel_arrow(burn_actual.v0_mps));

    // Asymptotic escape (v-infinity) direction -- the periapsis velocity
    // direction (`burn_actual.v0_mps`, drawn above) rotated by the
    // hyperbola's own turning angle `asin(1/e)` around its angular-momentum
    // axis, NOT the periapsis velocity direction itself.
    let v_inf_vec_geo = asymptotic_v_infinity_geo(burn_actual.r0_m, burn_actual.v0_mps, dep_catalog.mu_m3s2, r_park_m);
    if let Some(v_inf_vec_geo) = v_inf_vec_geo {
        push("v_infinity_asymptotic", burn_pos_km, vel_arrow(v_inf_vec_geo));
        // v_dep + v_infinity composed in this *same* frame, direction only
        // (drawn from the origin like departure_body_velocity_hat) -- this is the patched-conic
        // departure asymptote (`v_transfer`), and should point the same way
        // as `v_infinity_helio` in the heliocentric panel, since both are
        // the same vector just expressed in differently-scaled/centered
        // (but identically-oriented) frames.
        push("v_transfer_hat", Vector3::zeros(), axis_arrow((v_dep_v + v_inf_vec_geo).normalize()));

        // Real SOI-crossing velocity vs. the idealized asymptote, drawn at
        // the *same point* (the real crossing position) and the *same
        // scale* -- this is the direct visual answer to "how close is the
        // idealized v-infinity to what the real propagated escape actually
        // reaches at the real, finite SOI boundary" (known to be nonzero,
        // per `departure_demo.rs`'s own SOI-exit-residual test).
        if let Some((_, pos_km, v_rel)) = &crossing_info {
            let soi_vel_scale = (soi_radius_m / 1e3) * 0.15 / v_inf_vec_geo.norm().max(v_rel.norm()).max(1.0);
            push("v_at_soi_crossing_real", *pos_km, pos_km + v_rel * soi_vel_scale);
            push("v_infinity_asymptotic_at_soi", *pos_km, pos_km + v_inf_vec_geo * soi_vel_scale);
        }
    }
    std::fs::write(&vectors_path, vec_rows.join("\n") + "\n").ok()?;
    println!("  {vectors_path}");

    // Heliocentric panel: the same escape asymptote composed with the
    // departure body's own heliocentric velocity, alongside the Sun, the
    // departure body, and the target body's real position both at departure
    // and at the achieved arrival/closest-approach time.
    let target_anise = anise_body(&opt.target_body.to_lowercase())?;
    let target_r_dep = body_state(almanac, EphemerisSource::Anise, Some(target_anise), &None, dep_jd)?.0;
    let au = 1.495_978_707e11;
    let mut helio_rows = vec!["name,x_au,y_au,z_au".to_string()];
    helio_rows.push(format!("Sun,{:.6},{:.6},{:.6}", 0.0, 0.0, 0.0));
    helio_rows.push(format!("{},{:.6},{:.6},{:.6}", opt.departure_body, r_dep_v.x / au, r_dep_v.y / au, r_dep_v.z / au));
    helio_rows.push(format!(
        "{}_at_departure,{:.6},{:.6},{:.6}", opt.target_body, target_r_dep[0] / au, target_r_dep[1] / au, target_r_dep[2] / au,
    ));
    helio_rows.push(format!(
        "{}_at_arrival,{:.6},{:.6},{:.6}", opt.target_body,
        target_r_arr_m[0] / au, target_r_arr_m[1] / au, target_r_arr_m[2] / au,
    ));
    if let Some(v_inf_vec_geo) = v_inf_vec_geo {
        let v_inf_helio = v_dep_v + v_inf_vec_geo;
        // Drawn from the departure body's own position, scaled to a fixed
        // fraction of an AU so it's visible at this panel's scale (raw
        // m/s next to AU-scale positions would otherwise be invisible).
        let tip = r_dep_v + v_inf_helio.normalize() * 0.3 * au;
        helio_rows.push(format!("v_infinity_helio_origin,{:.6},{:.6},{:.6}", r_dep_v.x / au, r_dep_v.y / au, r_dep_v.z / au));
        helio_rows.push(format!("v_infinity_helio_tip,{:.6},{:.6},{:.6}", tip.x / au, tip.y / au, tip.z / au));
    }
    let helio_path = format!("{out_dir}/{method_prefix}_departure_geometry_helio.csv");
    std::fs::write(&helio_path, helio_rows.join("\n") + "\n").ok()?;
    println!("  {helio_path}");

    // Arrival/capture-side geometry diagnostic, mirroring the departure-side
    // blocks above. Re-derives the real crossing via `evaluate_candidate`
    // (single source of truth, same physics the GA itself used) from the
    // params this function already received, rather than plumbing the
    // crossing through `best.csv` -- written only when this mission actually
    // needs/achieves a capture; a Flyby or a non-capturing best point writes
    // nothing here (Python checks file existence, same convention as
    // `real_leg`/`soi` above).
    match build_context(cfg, opt, almanac) {
        Err(e) => eprintln!("  Warning: arrival-geometry re-evaluation failed: {e}"),
        Ok(ctx) => {
        let dep_offset_days_recovered = dep_jd - ctx.dep_jd_base;
        match evaluate_candidate(&ctx, &[dep_offset_days_recovered, theta_burn, dv_mps, phi]) {
            None => eprintln!("  Warning: arrival-geometry re-evaluation produced no feasible candidate"),
            Some(eval) => {
            // Genuinely expected (not a bug) when replotting from an older
            // `best.csv` written before this stage's params were stored at
            // full precision (Phase 9) -- a razor-thin capture
            // margin can be lost to ASCII round-tripping at low precision.
            // Re-run `optimize` to regenerate `best.csv` at full precision
            // if this fires for a best point that's supposed to have captured.
            if eval.arrival.is_none() && eval.needs_capture {
                eprintln!(
                    "  Note: re-evaluated best point did not achieve a capture (closest approach {:.1} km) \
                     -- if it should have, re-run `optimize` to refresh best.csv at full precision.",
                    eval.closest_approach_m / 1e3,
                );
            }
            if let Some(arrival) = &eval.arrival {
                if let Some(target_catalog) = body_models::TargetBody::by_name(&opt.target_body) {
                    // Local capture-orbit plane -- the actual plane the
                    // resulting circularized orbit lies in (the incoming
                    // relative velocity is exactly in this plane by
                    // construction). Deliberately NOT the reference used for
                    // theta_arr/phi_arr above -- see `arrival_burn_angles`'s
                    // doc comment for why those differ.
                    let capture_normal_hat = arrival.r_rel_m.cross(&arrival.v_rel_mps).normalize();
                    let capture_ref_hat = arrival.r_rel_m.normalize();
                    let target_radius_m = cfg
                        .trajectory
                        .capture
                        .as_ref()
                        .and_then(|c| c.target_orbit_radius_m)
                        .unwrap_or_else(|| arrival.r_rel_m.norm());
                    let v_circ_target = capture_target_speed_mps(cfg, target_catalog.mu_m3s2, target_radius_m);

                    let mut arr_orbit_rows = vec!["theta_deg,x_km,y_km,z_km".to_string()];
                    for i in 0..=72 {
                        let theta = (i as f64) * std::f64::consts::TAU / 72.0;
                        let p = capture_ref_hat * theta.cos() + capture_normal_hat.cross(&capture_ref_hat) * theta.sin();
                        let pos_km = p * target_radius_m / 1e3;
                        arr_orbit_rows.push(format!("{:.1},{:.3},{:.3},{:.3}", theta.to_degrees(), pos_km.x, pos_km.y, pos_km.z));
                    }
                    let arr_orbit_path = format!("{out_dir}/{method_prefix}_arrival_geometry_orbit.csv");
                    if std::fs::write(&arr_orbit_path, arr_orbit_rows.join("\n") + "\n").is_ok() {
                        println!("  {arr_orbit_path}");
                    }

                    let capture_pos_km = arrival.r_rel_m / 1e3;
                    let t_hat_arr = capture_normal_hat.cross(&capture_ref_hat);
                    let v_circ_post_burn = t_hat_arr * v_circ_target;
                    let vel_scale_arr = (target_radius_m / 1e3) * 0.4 / v_circ_target.max(1.0);
                    let mut arr_vec_rows = vec!["name,x0_km,y0_km,z0_km,x1_km,y1_km,z1_km".to_string()];
                    arr_vec_rows.push(format!(
                        "capture_position,{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
                        0.0, 0.0, 0.0, capture_pos_km.x, capture_pos_km.y, capture_pos_km.z
                    ));
                    let push_arr = |rows: &mut Vec<String>, name: &str, p0: Vector3<f64>, p1: Vector3<f64>| {
                        rows.push(format!("{name},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}", p0.x, p0.y, p0.z, p1.x, p1.y, p1.z));
                    };
                    push_arr(&mut arr_vec_rows, "v_incoming_relative", capture_pos_km, capture_pos_km + arrival.v_rel_mps * vel_scale_arr);
                    push_arr(&mut arr_vec_rows, "v_circular_post_burn", capture_pos_km, capture_pos_km + v_circ_post_burn * vel_scale_arr);
                    let arr_vec_path = format!("{out_dir}/{method_prefix}_arrival_geometry_vectors.csv");
                    if std::fs::write(&arr_vec_path, arr_vec_rows.join("\n") + "\n").is_ok() {
                        println!("  {arr_vec_path}");
                    }

                    let meta_path = format!("{out_dir}/{method_prefix}_arrival_geometry_meta.csv");
                    let meta_csv = format!(
                        "theta_arr_deg,phi_arr_deg,dv_capture_ms,capture_time_s,target_orbit_radius_km\n{:.3},{:.3},{:.3},{:.3},{:.3}\n",
                        arrival.theta_arr_rad.to_degrees(), arrival.phi_arr_rad.to_degrees(), arrival.dv_capture_ms,
                        arrival.capture_time_s, target_radius_m / 1e3,
                    );
                    if std::fs::write(&meta_path, meta_csv).is_ok() {
                        println!("  {meta_path}");
                    }
                }
            }
            }
        }
        }
    }

    Some(())
}

/// Re-runs just the departure-geometry diagnostic from an already-saved
/// `<method>_best.csv` (written by a previous `optimize` run), without
/// re-running the GA/PSO search at all -- the diagnostic itself only ever
/// needed the best point's parameters, not the search that found them.
/// Picks whichever of `ga_best.csv`/`pso_best.csv` exists, preferring `ga`
/// (same convention as the plot scripts).
pub fn replot_departure_geometry(cfg: &MissionConfig) -> Result<(), String> {
    let opt = cfg.optimization.as_ref().ok_or("this config has no [optimization] section")?;
    let out_dir = format!("{}/optimize", cfg.simulation.output_dir.trim_end_matches('/'));

    let method_prefix = ["ga", "pso"]
        .into_iter()
        .find(|m| std::path::Path::new(&format!("{out_dir}/{m}_best.csv")).exists())
        .ok_or_else(|| format!("no '{out_dir}/{{ga,pso}}_best.csv' found -- run `optimize` at least once first"))?;

    let best_path = format!("{out_dir}/{method_prefix}_best.csv");
    let content = std::fs::read_to_string(&best_path).map_err(|e| format!("could not read {best_path}: {e}"))?;
    let mut lines = content.lines();
    let header: Vec<&str> = lines.next().ok_or("empty best.csv")?.split(',').collect();
    let values: Vec<&str> = lines.next().ok_or("best.csv has no data row")?.split(',').collect();
    let col = |name: &str| -> Result<f64, String> {
        let idx = header.iter().position(|h| *h == name).ok_or_else(|| format!("best.csv missing column '{name}'"))?;
        values[idx].parse::<f64>().map_err(|e| format!("could not parse column '{name}': {e}"))
    };

    let dep_jd = col("dep_jd")?;
    let theta_burn_rad = col("theta_burn_rad")?;
    let dv_departure_ms = col("dv_departure_ms")?;
    let phi_out_of_plane_rad = col("phi_out_of_plane_rad")?;
    let target_r_arr_m = [col("target_x_m")?, col("target_y_m")?, col("target_z_m")?];

    let Some(almanac) = crate::design::load_almanac() else {
        return Err("could not load ANISE almanac".into());
    };

    write_departure_geometry_csvs(
        cfg, opt, &almanac, dep_jd, theta_burn_rad, dv_departure_ms, phi_out_of_plane_rad,
        target_r_arr_m, &out_dir, method_prefix,
    )
    .ok_or_else(|| "could not write departure-geometry diagnostic (ephemeris/catalog lookup failed)".to_string())
}

/// Re-propagates `sample_count` individuals' full trajectories from an
/// already-saved `<method>_population.csv` (written by a previous `optimize`
/// run), without re-running the GA/PSO search -- same "replot from saved
/// state" precedent as `replot_departure_geometry`. Samples evenly by row
/// index across the *feasible* population in file order (phase 1 then phase
/// 2, chronological within each), not just the best individual, so the
/// result shows the spread of trajectories the search actually considered --
/// same "don't just show the cherry-picked best" precedent as
/// `design.rs`/`plot_porkchop_samples.py`'s narrowing-stage equivalent.
/// Writes `<method>_population_sample_trajectories.csv`
/// (`sample_id,phase,generation,fitness,t_s,x_m,y_m,z_m`, heliocentric).
pub fn replot_population_sample_trajectories(cfg: &MissionConfig, sample_count: usize) -> Result<(), String> {
    let opt = cfg.optimization.as_ref().ok_or("this config has no [optimization] section")?;
    let out_dir = format!("{}/optimize", cfg.simulation.output_dir.trim_end_matches('/'));

    let method_prefix = ["ga", "pso"]
        .into_iter()
        .find(|m| std::path::Path::new(&format!("{out_dir}/{m}_population.csv")).exists())
        .ok_or_else(|| format!("no '{out_dir}/{{ga,pso}}_population.csv' found -- run `optimize` at least once first"))?;

    let pop_path = format!("{out_dir}/{method_prefix}_population.csv");
    let content = std::fs::read_to_string(&pop_path).map_err(|e| format!("could not read {pop_path}: {e}"))?;
    let mut lines = content.lines();
    lines.next().ok_or("empty population.csv")?; // header: phase,generation,dep_offset_days,theta_burn_rad,dv_mps,phi_rad,fitness

    struct Row { phase: u8, generation: usize, params: [f64; 4], fitness_str: String }
    let feasible: Vec<Row> = lines
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() != 7 || f[6] == "inf" {
                return None;
            }
            Some(Row {
                phase: f[0].parse().ok()?,
                generation: f[1].parse().ok()?,
                params: [f[2].parse().ok()?, f[3].parse().ok()?, f[4].parse().ok()?, f[5].parse().ok()?],
                fitness_str: f[6].to_string(),
            })
        })
        .collect();
    if feasible.is_empty() {
        return Err("no feasible individuals found in population.csv".into());
    }

    let Some(almanac) = crate::design::load_almanac() else {
        return Err("could not load ANISE almanac".into());
    };
    let ctx = build_context(cfg, opt, &almanac)?;

    // Sample evenly by *fitness rank* within each phase (always including
    // that phase's single best individual), not evenly by chronological
    // index across the combined list -- found necessary with
    // index-based sampling, the rare elite individuals (a small fraction of
    // a large population) could be skipped entirely by chance, making it
    // look like "no trajectory ever reaches Mars" when the GA's own best
    // result actually gets closer than any of the displayed samples. Phase
    // 1 uses its own fitness units (closest_approach_m); phase 2 uses the
    // real objective's normalized units -- not comparable to each other,
    // but each phase's *internal* ranking is meaningful on its own.
    let mut phase1: Vec<&Row> = feasible.iter().filter(|r| r.phase == 1).collect();
    let mut phase2: Vec<&Row> = feasible.iter().filter(|r| r.phase == 2).collect();
    let parse_fit = |s: &str| s.parse::<f64>().unwrap_or(f64::MAX);
    phase1.sort_by(|a, b| parse_fit(&a.fitness_str).partial_cmp(&parse_fit(&b.fitness_str)).unwrap());
    phase2.sort_by(|a, b| parse_fit(&a.fitness_str).partial_cmp(&parse_fit(&b.fitness_str)).unwrap());

    let total = (phase1.len() + phase2.len()).max(1);
    let n1 = (sample_count * phase1.len() / total).clamp(if phase1.is_empty() { 0 } else { 1 }, phase1.len().max(1));
    let n2 = sample_count.saturating_sub(n1).min(phase2.len().max(1));
    let rank_sample = |sorted: &[&Row], n: usize| -> Vec<usize> {
        if sorted.is_empty() || n == 0 {
            return Vec::new();
        }
        (0..n).map(|i| if n == 1 { 0 } else { i * (sorted.len() - 1) / (n - 1) }).collect()
    };

    let mut out_rows = vec!["sample_id,phase,generation,fitness,theta_burn_rad,dv_mps,phi_rad,t_s,x_m,y_m,z_m".to_string()];
    let mut written = 0usize;
    let mut sample_id = 0usize;
    for (sorted, n) in [(phase1.as_slice(), n1), (phase2.as_slice(), n2)] {
        for idx in rank_sample(sorted, n) {
            let row = sorted[idx];
            let Some(eval) = evaluate_candidate(&ctx, &row.params) else { continue };
            for p in &eval.points {
                out_rows.push(format!(
                    "{sample_id},{},{},{},{:.6},{:.3},{:.6},{:.3},{:.6e},{:.6e},{:.6e}",
                    row.phase, row.generation, row.fitness_str, row.params[1], row.params[2], row.params[3],
                    p.t_s, p.r_m.x, p.r_m.y, p.r_m.z,
                ));
            }
            written += 1;
            sample_id += 1;
        }
    }
    if written == 0 {
        return Err("could not re-evaluate any sampled individual (all failed to re-propagate)".into());
    }

    let traj_path = format!("{out_dir}/{method_prefix}_population_sample_trajectories.csv");
    std::fs::write(&traj_path, out_rows.join("\n") + "\n").map_err(|e| format!("could not write {traj_path}: {e}"))?;
    println!("  {traj_path}  ({written} sampled trajectories, out of {} feasible individuals)", feasible.len());
    Ok(())
}

/// Re-evaluates `sample_count` individuals from an already-saved
/// `<method>_population.csv` to recover their real arrival/capture geometry
/// (`theta_arr_rad`/`phi_arr_rad`/`dv_arrival_ms`) -- same "replot from saved
/// state, no GA rerun" precedent as `replot_population_sample_trajectories`,
/// but writes only per-individual scalars (no trajectory points), so a much
/// larger default sample is affordable for the same propagation cost per
/// individual. A `Flyby` mission, or any sampled individual that didn't
/// achieve a real capture, has no arrival geometry -- those rows still get
/// written with the arrival columns empty (not skipped entirely), so the
/// caller can see what fraction of the sample actually captured.
/// Writes `<method>_population_arrival_sample.csv`
/// (`sample_id,phase,generation,fitness,theta_burn_rad,dv_mps,phi_rad,
///   dv_arrival_ms,theta_arr_rad,phi_arr_rad,dv_total_ms`).
pub fn replot_population_arrival_angles(cfg: &MissionConfig, sample_count: usize) -> Result<(), String> {
    let opt = cfg.optimization.as_ref().ok_or("this config has no [optimization] section")?;
    let out_dir = format!("{}/optimize", cfg.simulation.output_dir.trim_end_matches('/'));

    let method_prefix = ["ga", "pso"]
        .into_iter()
        .find(|m| std::path::Path::new(&format!("{out_dir}/{m}_population.csv")).exists())
        .ok_or_else(|| format!("no '{out_dir}/{{ga,pso}}_population.csv' found -- run `optimize` at least once first"))?;

    let pop_path = format!("{out_dir}/{method_prefix}_population.csv");
    let content = std::fs::read_to_string(&pop_path).map_err(|e| format!("could not read {pop_path}: {e}"))?;
    let mut lines = content.lines();
    lines.next().ok_or("empty population.csv")?;

    struct Row { phase: u8, generation: usize, params: [f64; 4], fitness_str: String }
    let feasible: Vec<Row> = lines
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() != 7 || f[6] == "inf" {
                return None;
            }
            Some(Row {
                phase: f[0].parse().ok()?,
                generation: f[1].parse().ok()?,
                params: [f[2].parse().ok()?, f[3].parse().ok()?, f[4].parse().ok()?, f[5].parse().ok()?],
                fitness_str: f[6].to_string(),
            })
        })
        .collect();
    if feasible.is_empty() {
        return Err("no feasible individuals found in population.csv".into());
    }

    let Some(almanac) = crate::design::load_almanac() else {
        return Err("could not load ANISE almanac".into());
    };
    let ctx = build_context(cfg, opt, &almanac)?;

    // Same fitness-rank sampling convention as `replot_population_sample_trajectories`.
    let mut phase1: Vec<&Row> = feasible.iter().filter(|r| r.phase == 1).collect();
    let mut phase2: Vec<&Row> = feasible.iter().filter(|r| r.phase == 2).collect();
    let parse_fit = |s: &str| s.parse::<f64>().unwrap_or(f64::MAX);
    phase1.sort_by(|a, b| parse_fit(&a.fitness_str).partial_cmp(&parse_fit(&b.fitness_str)).unwrap());
    phase2.sort_by(|a, b| parse_fit(&a.fitness_str).partial_cmp(&parse_fit(&b.fitness_str)).unwrap());

    let total = (phase1.len() + phase2.len()).max(1);
    let n1 = (sample_count * phase1.len() / total).clamp(if phase1.is_empty() { 0 } else { 1 }, phase1.len().max(1));
    let n2 = sample_count.saturating_sub(n1).min(phase2.len().max(1));
    let rank_sample = |sorted: &[&Row], n: usize| -> Vec<usize> {
        if sorted.is_empty() || n == 0 {
            return Vec::new();
        }
        (0..n).map(|i| if n == 1 { 0 } else { i * (sorted.len() - 1) / (n - 1) }).collect()
    };

    let opt_f = |v: Option<f64>| v.map(|x| format!("{x:.6}")).unwrap_or_default();
    let mut out_rows =
        vec!["sample_id,phase,generation,fitness,theta_burn_rad,dv_mps,phi_rad,dv_arrival_ms,theta_arr_rad,phi_arr_rad,dv_total_ms".to_string()];
    let mut written = 0usize;
    let mut captured = 0usize;
    let mut sample_id = 0usize;
    for (sorted, n) in [(phase1.as_slice(), n1), (phase2.as_slice(), n2)] {
        for idx in rank_sample(sorted, n) {
            let row = sorted[idx];
            let Some(eval) = evaluate_candidate(&ctx, &row.params) else { continue };
            let dv_arrival_ms = eval.arrival.as_ref().map(|a| a.dv_capture_ms);
            let theta_arr_rad = eval.arrival.as_ref().map(|a| a.theta_arr_rad);
            let phi_arr_rad = eval.arrival.as_ref().map(|a| a.phi_arr_rad);
            let dv_total_ms = row.params[2] + dv_arrival_ms.unwrap_or(0.0);
            out_rows.push(format!(
                "{sample_id},{},{},{},{:.6},{:.3},{:.6},{},{},{},{:.3}",
                row.phase, row.generation, row.fitness_str, row.params[1], row.params[2], row.params[3],
                opt_f(dv_arrival_ms), opt_f(theta_arr_rad), opt_f(phi_arr_rad), dv_total_ms,
            ));
            if eval.arrival.is_some() {
                captured += 1;
            }
            written += 1;
            sample_id += 1;
        }
    }
    if written == 0 {
        return Err("could not re-evaluate any sampled individual (all failed to re-propagate)".into());
    }

    let arr_path = format!("{out_dir}/{method_prefix}_population_arrival_sample.csv");
    std::fs::write(&arr_path, out_rows.join("\n") + "\n").map_err(|e| format!("could not write {arr_path}: {e}"))?;
    println!("  {arr_path}  ({written} sampled, {captured} achieved a real capture)");
    Ok(())
}

/// Dense, unbiased grid scan over `theta x phi` at a fixed `dv`, using the
/// real propagated physics (`evaluate_candidate`, the same function the
/// GA/PSO fitness loop calls) -- no GA, no seeding, no narrowing. Added
/// in direct response to a real, well-founded doubt about
/// whether the GA's own results (5-generation convergence then a 45+
/// generation flat plateau; zero individuals near Mars's real orbital
/// plane across a supposedly-random search) reflect the true fitness
/// landscape or some artifact of the search itself (premature convergence,
/// a boundary the population is stuck against, or the previously-flagged
/// SOI-truncation discontinuity). This writes the ground truth directly:
/// every grid point's real closest-approach distance, so the landscape can
/// be inspected by eye rather than inferred from a GA's sparse trajectory.
/// Writes `<out_dir>/optimize/grid_scan.csv`
/// (`theta_deg,phi_deg,dv_mps,closest_approach_km,feasible`).
pub fn run_grid_scan(cfg: &MissionConfig, dv_mps: f64, theta_step_deg: f64, phi_step_deg: f64) -> Result<(), String> {
    let opt = cfg.optimization.as_ref().ok_or("this config has no [optimization] section")?;
    let out_dir = format!("{}/optimize", cfg.simulation.output_dir.trim_end_matches('/'));
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("could not create '{out_dir}': {e}"))?;

    let Some(almanac) = crate::design::load_almanac() else {
        return Err("could not load ANISE almanac".into());
    };
    let ctx = build_context(cfg, opt, &almanac)?;

    let n_theta = (360.0 / theta_step_deg).round() as i64;
    let n_phi = (90.0 / phi_step_deg).round() as i64;
    let total = n_theta * (2 * n_phi + 1);
    println!("Grid scan: {n_theta} theta steps x {} phi steps = {total} real propagations (dv={dv_mps:.0} m/s fixed)...", 2 * n_phi + 1);

    let mut rows = vec!["theta_deg,phi_deg,dv_mps,closest_approach_km,feasible".to_string()];
    let mut done = 0usize;
    for ti in 0..n_theta {
        let theta_deg = ti as f64 * theta_step_deg;
        for pi in -n_phi..=n_phi {
            let phi_deg = pi as f64 * phi_step_deg;
            let params = [0.0, theta_deg.to_radians(), dv_mps, phi_deg.to_radians()];
            match evaluate_candidate(&ctx, &params) {
                Some(eval) => rows.push(format!("{theta_deg:.2},{phi_deg:.2},{dv_mps:.1},{:.3},true", eval.closest_approach_m / 1e3)),
                None => rows.push(format!("{theta_deg:.2},{phi_deg:.2},{dv_mps:.1},,false")),
            }
            done += 1;
            if done % 200 == 0 {
                println!("  {done}/{total}...");
            }
        }
    }

    let path = format!("{out_dir}/grid_scan.csv");
    std::fs::write(&path, rows.join("\n") + "\n").map_err(|e| format!("could not write {path}: {e}"))?;
    println!("  {path}  ({total} grid points)");
    Ok(())
}

// ── CLI output ─────────────────────────────────────────────────────────────────

/// CLI entry point: run the optimization stage, print a summary, write
/// `<out_dir>/optimize/<method>_convergence.csv` and `<method>_best.csv` —
/// same shape as `design.rs::write_optimizer_output`, but in a separate
/// `optimize/` subdirectory so this stage's output never collides with the
/// narrowing stage's `design/ga_convergence.csv`/`pso_convergence.csv`.
pub fn run(cfg: &MissionConfig) {
    let Some(opt) = cfg.optimization.as_ref() else {
        println!("No [optimization] section in this config — nothing to do.");
        return;
    };
    let Some(almanac) = crate::design::load_almanac() else {
        std::process::exit(1);
    };

    println!(
        "Optimization stage: method = {}, objective = {}",
        opt.method, opt.objective,
    );

    let result = match optimize_api(cfg, &almanac) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    println!("\n────────────────────────────────────────────────────────────────");
    println!("Optimization result  (best point, step {})", result.convergence.len());
    println!("────────────────────────────────────────────────────────────────");
    println!("  Departure offset:   {:+.2} days from reference epoch", result.dep_offset_days);
    println!(
        "  Departure burn:     {:.2} m/s at theta={:.1} deg, phi={:.1} deg (out-of-plane)",
        result.dv_departure_ms, result.theta_burn_rad.to_degrees(), result.phi_out_of_plane_rad.to_degrees(),
    );
    println!("  Escape duration:    {:.3} days", result.escape_duration_s / 86_400.0);
    println!("  Achieved TOF:       {:.3} days  ({:.2} months)", result.achieved_tof_days, result.achieved_tof_days / 30.4375);
    if let Some(t) = result.capture_time_s {
        println!("  Capture burn:       {:.2} m/s at t={:.3} days", result.dv_arrival_ms, t / 86_400.0);
        if let (Some(theta_arr), Some(phi_arr)) = (result.theta_arr_rad, result.phi_arr_rad) {
            println!(
                "  Capture geometry:   theta_arr={:.1} deg, phi_arr={:.1} deg (out-of-plane)",
                theta_arr.to_degrees(), phi_arr.to_degrees(),
            );
        }
    }
    println!("  Fitness:            {:.4}", result.fitness);
    println!("  Real arrival miss:  {:.1} km", result.miss_km);

    let out_dir = format!("{}/optimize", cfg.simulation.output_dir.trim_end_matches('/'));
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("\nWarning: could not create output dir '{out_dir}': {e}");
        return;
    }

    let method_prefix = result.method.to_lowercase();
    let step_col = if result.method == "GA" { "generation" } else { "iteration" };
    let convergence_path = format!("{out_dir}/{method_prefix}_convergence.csv");
    let mut rows = vec![format!("{step_col},best_fitness_so_far")];
    rows.extend(result.convergence.iter().enumerate().map(|(i, f)| format!("{i},{f:.6}")));
    match std::fs::write(&convergence_path, rows.join("\n") + "\n") {
        Ok(_) => println!("\n  {convergence_path}  ({} {step_col}s)", result.convergence.len()),
        Err(e) => eprintln!("\n  Warning: could not write {method_prefix}_convergence.csv: {e}"),
    }

    // Phase 1 (flyby-only) convergence and the full per-individual
    // population log -- empty for PSO, which isn't split into phases.
    if !result.phase1_convergence_km.is_empty() {
        let phase1_path = format!("{out_dir}/{method_prefix}_phase1_convergence.csv");
        let mut rows = vec!["generation,best_closest_approach_km_so_far".to_string()];
        rows.extend(result.phase1_convergence_km.iter().enumerate().map(|(i, f)| format!("{i},{f:.3}")));
        match std::fs::write(&phase1_path, rows.join("\n") + "\n") {
            Ok(_) => println!("  {phase1_path}  ({} generations)", result.phase1_convergence_km.len()),
            Err(e) => eprintln!("  Warning: could not write {method_prefix}_phase1_convergence.csv: {e}"),
        }
    }
    if !result.population_log.is_empty() {
        let pop_path = format!("{out_dir}/{method_prefix}_population.csv");
        let mut rows = vec!["phase,generation,dep_offset_days,theta_burn_rad,dv_mps,phi_rad,fitness".to_string()];
        rows.extend(result.population_log.iter().map(|row| {
            // `f64::MAX` (the GA's infeasible-point penalty) prints as an
            // unreadable ~300-digit literal and would dominate a plot's
            // color scale -- write "inf" instead, same convention a reader
            // would expect from any other infeasible-point indicator.
            let fitness_str =
                if row.fitness >= 1.0e15 { "inf".to_string() } else { format!("{:.6}", row.fitness) };
            format!(
                "{},{},{:.4},{:.6},{:.3},{:.6},{}",
                row.phase, row.generation, row.params[0], row.params[1], row.params[2], row.params[3], fitness_str,
            )
        }));
        match std::fs::write(&pop_path, rows.join("\n") + "\n") {
            Ok(_) => println!("  {pop_path}  ({} individuals)", result.population_log.len()),
            Err(e) => eprintln!("  Warning: could not write {method_prefix}_population.csv: {e}"),
        }
    }

    let best_path = format!("{out_dir}/{method_prefix}_best.csv");
    let arr_jd = result.dep_jd + result.achieved_tof_days;
    let opt_f = |v: Option<f64>| v.map(|x| format!("{x:.6}")).unwrap_or_default();
    let csv = format!(
        "dep_jd,arr_jd,dep_offset_days,theta_burn_rad,dv_departure_ms,phi_out_of_plane_rad,\
         escape_duration_s,achieved_tof_days,dv_arrival_ms,theta_arr_rad,phi_arr_rad,fitness,miss_km,target_x_m,target_y_m,target_z_m\n\
         {:.9},{:.9},{:.8},{:.9},{:.6},{:.9},{:.3},{:.4},{:.3},{},{},{:.6},{:.3},{:.6e},{:.6e},{:.6e}\n",
        result.dep_jd, arr_jd, result.dep_offset_days, result.theta_burn_rad, result.dv_departure_ms,
        result.phi_out_of_plane_rad, result.escape_duration_s, result.achieved_tof_days, result.dv_arrival_ms,
        opt_f(result.theta_arr_rad), opt_f(result.phi_arr_rad),
        result.fitness, result.miss_km,
        result.target_r_arr_m[0], result.target_r_arr_m[1], result.target_r_arr_m[2],
    );
    match std::fs::write(&best_path, csv) {
        Ok(_) => println!("  {best_path}"),
        Err(e) => eprintln!("  Warning: could not write {method_prefix}_best.csv: {e}"),
    }

    let traj_path = format!("{out_dir}/{method_prefix}_trajectory.csv");
    let mut traj_rows = vec!["t_s,x_m,y_m,z_m,central_body".to_string()];
    traj_rows.extend(
        result.arc.iter().map(|p| format!("{:.3},{:.6e},{:.6e},{:.6e},{}", p.t_s, p.x_m, p.y_m, p.z_m, p.central_body)),
    );
    match std::fs::write(&traj_path, traj_rows.join("\n") + "\n") {
        Ok(_) => println!("  {traj_path}  ({} points)", result.arc.len()),
        Err(e) => eprintln!("  Warning: could not write {method_prefix}_trajectory.csv: {e}"),
    }

    // Departure body's own heliocentric track over the same points -- lets
    // the plot recenter the (otherwise invisible-at-AU-scale) departure leg
    // into a body-centered close-up, same convention as `soi_demo.rs`'s
    // `mars_track.csv` for the arrival side.
    if let Some(dep_anise) = anise_body(&opt.departure_body.to_lowercase()) {
        let mut dep_rows = vec!["t_s,x_m,y_m,z_m".to_string()];
        for p in &result.arc {
            let jd = result.dep_jd + p.t_s / 86_400.0;
            if let Some((r, _)) = body_state(&almanac, EphemerisSource::Anise, Some(dep_anise), &None, jd) {
                dep_rows.push(format!("{:.3},{:.6e},{:.6e},{:.6e}", p.t_s, r[0], r[1], r[2]));
            }
        }
        let dep_track_path = format!("{out_dir}/{method_prefix}_departure_track.csv");
        match std::fs::write(&dep_track_path, dep_rows.join("\n") + "\n") {
            Ok(_) => println!("  {dep_track_path}"),
            Err(e) => eprintln!("  Warning: could not write {method_prefix}_departure_track.csv: {e}"),
        }
    }

    // Target body's own heliocentric track over the same points -- same
    // role as the departure track above, but for a symmetric body-centered
    // close-up of the arrival/capture leg.
    if let Some(target_anise) = anise_body(&opt.target_body.to_lowercase()) {
        let mut target_rows = vec!["t_s,x_m,y_m,z_m".to_string()];
        for p in &result.arc {
            let jd = result.dep_jd + p.t_s / 86_400.0;
            if let Some((r, _)) = body_state(&almanac, EphemerisSource::Anise, Some(target_anise), &None, jd) {
                target_rows.push(format!("{:.3},{:.6e},{:.6e},{:.6e}", p.t_s, r[0], r[1], r[2]));
            }
        }
        let target_track_path = format!("{out_dir}/{method_prefix}_target_track.csv");
        match std::fs::write(&target_track_path, target_rows.join("\n") + "\n") {
            Ok(_) => println!("  {target_track_path}"),
            Err(e) => eprintln!("  Warning: could not write {method_prefix}_target_track.csv: {e}"),
        }
    }

    if write_departure_geometry_csvs(
        cfg, opt, &almanac, result.dep_jd, result.theta_burn_rad, result.dv_departure_ms,
        result.phi_out_of_plane_rad, result.target_r_arr_m, &out_dir, &method_prefix,
    ).is_none() {
        eprintln!("  Warning: could not write departure-geometry diagnostic (ephemeris/catalog lookup failed)");
    }

    // Arrival-angle population sample -- only meaningful for a mission that
    // actually computes a capture burn (Flyby never has arrival angles to
    // sample). Run automatically here (modest default count, same
    // propagation cost per individual as the trajectory sampler above) so
    // the orbit-mode stats plot always has data right after `optimize`
    // finishes, without a separate manual step.
    if !matches!(cfg.mission.objective, MissionObjective::Flyby) {
        if let Err(e) = replot_population_arrival_angles(cfg, 200) {
            eprintln!("  Warning: could not write arrival-angle population sample: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Phase 14b: departure-mode-dependent search bounds. `ParkingOrbit`
    // must reproduce the pre-14b bounds exactly (bit-identity regression);
    // `Launch` maps the burn bounds to v∞ and widens φ to a declination. ──

    fn opt_toml(departure_block: &str) -> MissionConfig {
        toml::from_str(&format!(
            r#"
[mission]
name = "B"
objective = "Orbit"
[target_body]
name = "Mars"
ephemeris = "Anise"
[spacecraft]
mass_kg = 1000.0
dry_mass_kg = 800.0
propellant_mass_kg = 200.0
bus_dims_m = [1.0, 1.0, 1.0]
inertia_diag_kgm2 = [100.0, 100.0, 100.0]
srp_model = "Cannonball"
launch_vehicle = "Falcon9"
[trajectory]
phases = ["Cruise"]
solver = "GridSearch"
departure_body = "Earth"
{departure_block}
[trajectory.capture]
target_orbit_radius_m = 4.0e6
[optimization]
objective = "MinDeltaV"
method = "GA"
departure_body = "Earth"
target_body = "Mars"
departure_window_days = 60.0
dv_min_ms = 3400.0
dv_max_ms = 4200.0
max_coast_days = 400.0
[optimization.force_model]
integrator = "Dopri5"
rtol = 1.0e-9
atol = 1.0e-3
[gnc]
navigation_filter = "EKF"
pointing_mode = "Nadir"
attitude_controller = "ReactionWheelPD"
[simulation]
integrator = "DormandPrince45"
rtol = 1.0e-9
atol = 1.0e-7
dt_truth_s = 10.0
dt_meas_s = 120.0
monte_carlo_runs = 0
output_dir = "out/test/"
"#
        ))
        .unwrap()
    }

    #[test]
    fn parking_orbit_bounds_are_bit_identical_to_the_pre_launch_mode_formula() {
        let cfg = opt_toml("");
        let opt = cfg.optimization.as_ref().unwrap();
        let b = bounds_from(&cfg, opt);
        assert_eq!(b, vec![(-30.0, 30.0), (0.0, 360_f64.to_radians()), (3400.0, 4200.0), ((-45_f64).to_radians(), 45_f64.to_radians())]);
    }

    #[test]
    fn launch_mode_bounds_map_burn_bounds_to_v_infinity_and_widen_declination() {
        let cfg = opt_toml(
            "[trajectory.departure]\nmode = \"Launch\"\n[trajectory.departure.launch_site]\nname = \"KSC\"\nlat_deg = 28.5\n",
        );
        let opt = cfg.optimization.as_ref().unwrap();
        let b = bounds_from(&cfg, opt);
        let earth = body_models::TargetBody::by_name("Earth").unwrap();
        let r_p = resolve_parking_orbit_radius_m(&cfg, &earth);
        let v_c = (earth.mu_m3s2 / r_p).sqrt();
        // A 3.4 km/s burn from 185 km barely escapes (escape is (√2−1)·v_c ≈
        // 3.23 km/s); a burn below that maps to a v∞ floor of 0, not NaN.
        let vinf = |dv: f64| ((v_c + dv).powi(2) - 2.0 * earth.mu_m3s2 / r_p).max(0.0).sqrt();
        assert!(
            (b[2].0 - vinf(3400.0)).abs() < 1e-6 && (b[2].1 - vinf(4200.0)).abs() < 1e-6,
            "bounds {b:?}, expected v_inf ({}, {}), mode {:?}", vinf(3400.0), vinf(4200.0), departure_mode(&cfg)
        );
        assert!(b[2].0 > 0.0);
        assert_eq!(b[3], ((-90_f64).to_radians(), 90_f64.to_radians()));
        assert_eq!(b[1], (0.0, 360_f64.to_radians()));
        // The mapped v∞ range brackets the Earth–Mars C3 band (~4–20 km²/s²).
        assert!((b[2].0 / 1e3).powi(2) < 9.0 && (b[2].1 / 1e3).powi(2) > 9.0);
        // Launch geometry from the three departure genes: exact energy on
        // the site-feasible plane.
        let g = launch_geometry_for_params(&cfg, &earth, &[0.0, 1.0, 3_000.0, 0.2]).unwrap();
        assert!((g.injection_v_mps.norm_squared() - 2.0 * earth.mu_m3s2 / r_p - 9.0e6).abs() < 1e-3);
        assert!((g.dla_rad - 0.2).abs() < 1e-12 && (g.rla_rad - 1.0).abs() < 1e-12);
        // ParkingOrbit mode: no geometry from the same genes.
        assert!(launch_geometry_for_params(&opt_toml(""), &earth, &[0.0, 1.0, 3_000.0, 0.2]).is_none());
    }

    // ── Phase 12a: `target_distance_error` (the primitive that lets
    // MinDeltaV/MinTof fold in Flyby's target-periapsis awareness) ─────────

    #[test]
    fn target_distance_error_is_zero_at_exact_match() {
        assert_eq!(target_distance_error(500_000.0, 500_000.0), 0.0);
    }

    #[test]
    fn target_distance_error_grows_when_overshooting_the_target() {
        // Closest approach twice as far as the requested target distance.
        let err = target_distance_error(1_000_000.0, 500_000.0);
        assert!((err - 1.0).abs() < 1e-9, "expected relative error 1.0, got {err}");
    }

    #[test]
    fn target_distance_error_grows_when_undershooting_the_target() {
        // Closest approach only half the requested target distance (dove
        // too close / inside the requested periapsis) -- the error must
        // grow in this direction too, not just when overshooting, since a
        // Flyby candidate that dives well inside the requested altitude is
        // just as undesirable as one that never got close.
        let err = target_distance_error(250_000.0, 500_000.0);
        assert!((err - 0.5).abs() < 1e-9, "expected relative error 0.5, got {err}");
    }

    #[test]
    fn target_distance_error_grows_unboundedly_far_from_target() {
        // A candidate that merely grazed the target's SOI (hundreds of
        // thousands of km) against a real requested flyby altitude (a few
        // hundred km) -- the pre-Phase-12a gap: this must score enormously
        // worse than a near-target candidate, not merely "pass a coarse
        // SOI-entry gate" the way the superseded Phase 9y fix did.
        let near = target_distance_error(510_000.0, 500_000.0);
        let far = target_distance_error(50_000_000.0, 500_000.0);
        assert!(far > near * 1_000.0, "far={far}, near={near}");
    }

    // ── GA/PSO search-shaping (after a live MinDeltaV run
    // converged to a 61 km/s solution): ΔV pressure in the non-capture
    // penalty, and diverse phase-2 seed selection. Pure functions, no
    // almanac needed. ─────────────────────────────────────────────────────

    /// REGRESSION GUARD (introduced-and-fixed same day): the
    /// MTD tiebreaker must be blind to whether a candidate captured.
    /// Two candidates at the SAME closest approach and SAME departure burn
    /// must score identically -- if one of them achieved a capture crossing
    /// (and therefore has a real arrival burn) and the other didn't, that
    /// difference must NOT appear in the fitness, or the search selects
    /// against capture itself. The pure signature (no arrival-dv parameter)
    /// enforces this by construction; this test additionally pins the
    /// intended ordering properties.
    #[test]
    fn mtd_tiebreaker_prefers_cheap_departure_but_distance_dominates() {
        // Equal distance: cheaper departure burn wins.
        let cheap = mtd_fitness(3.1e6, 3.0e6, 4_000.0, 8_000.0);
        let expensive = mtd_fitness(3.1e6, 3.0e6, 12_000.0, 8_000.0);
        assert!(cheap < expensive);
        // Distance dominates: a meaningfully-closer candidate beats a
        // farther, cheaper one.
        let closer_expensive = mtd_fitness(3.1e6, 3.0e6, 12_000.0, 8_000.0);
        let farther_cheap = mtd_fitness(9.0e6, 3.0e6, 4_000.0, 8_000.0);
        assert!(closer_expensive < farther_cheap);
    }

    /// Diverse seed selection: always includes the single best candidate,
    /// skips near-duplicates of it, and picks the genuinely distant
    /// second basin.
    #[test]
    fn phase2_seed_selection_picks_best_and_distant_not_duplicates() {
        let bounds = [(0.0, 10.0), (0.0, 10.0)];
        let rows = vec![
            PopulationLogRow { phase: 1, generation: 0, params: vec![1.0, 1.0], fitness: 1.0, dv_arrival_ms: None, tof_days: None },  // best
            PopulationLogRow { phase: 1, generation: 0, params: vec![1.1, 1.0], fitness: 2.0, dv_arrival_ms: None, tof_days: None },  // near-duplicate of best
            PopulationLogRow { phase: 1, generation: 0, params: vec![8.0, 9.0], fitness: 5.0, dv_arrival_ms: None, tof_days: None },  // distant second basin
            PopulationLogRow { phase: 2, generation: 1, params: vec![5.0, 5.0], fitness: 0.1, dv_arrival_ms: None, tof_days: None },  // wrong phase -- ignored
            PopulationLogRow { phase: 1, generation: 0, params: vec![4.0, 4.0], fitness: f64::MAX, dv_arrival_ms: None, tof_days: None }, // infeasible -- ignored
        ];
        let seeds = select_diverse_phase2_seeds(&rows, &bounds, 3);
        assert_eq!(seeds.len(), 2, "only two mutually-distant feasible phase-1 basins exist, got {seeds:?}");
        assert_eq!(seeds[0], vec![1.0, 1.0], "the single best candidate must always be picked first");
        assert_eq!(seeds[1], vec![8.0, 9.0]);
    }

    // ── Phase 12k: `captured_orbit_to_api` must translate its
    // body-relative input into the SAME heliocentric frame `arc` uses, by
    // adding `origin_m` -- the real bug this fixes: a downstream consumer
    // concatenating `arc` (heliocentric) with the raw, untranslated ring
    // read a genuine captured orbit as a flyby. ─────────────────────────────

    #[test]
    fn captured_orbit_to_api_translates_to_the_supplied_heliocentric_origin() {
        let mu = 3.986_004_418e14_f64; // Earth-mass central body
        let r_cap_m = 7.0e6_f64;
        let r_hat = Vector3::new(1.0, 0.0, 0.0);
        let plane_normal_hat = Vector3::new(0.0, 0.0, 1.0);
        let t_hat = plane_normal_hat.cross(&r_hat);
        let v_circ = (mu / r_cap_m).sqrt();
        let (points, _summary) = propagate_captured_orbit(r_hat * r_cap_m, t_hat * v_circ, mu);

        // A real heliocentric-scale origin (~1 AU on the x-axis) -- if the
        // translation were missing, the returned points would still be at
        // body-relative (~7,000 km) scale, off by ~4 orders of magnitude
        // from this origin.
        let origin_m = Vector3::new(1.495_98e11, 0.0, 0.0);
        let origin_v_mps = Vector3::new(0.0, 29_800.0, 0.0); // ~Earth's real orbital speed
        // Synthetic MOVING body (frozen-origin fix): linear
        // motion from the anchor state -- exactly the class of motion the
        // frozen version ignored. `body_state_at` receives the offset from
        // the anchor sample (here t_s_anchor = 0.0, a post-capture-style
        // ring).
        let api_points = captured_orbit_to_api(
            points.clone(), "Mercury", 0.0,
            |dt_s| Some((origin_m + origin_v_mps * dt_s, origin_v_mps)),
            origin_m, origin_v_mps,
        );

        assert_eq!(api_points.len(), points.len());
        for (p, api_p) in points.iter().zip(api_points.iter()) {
            // Each point must be translated by the body's state AT ITS OWN
            // EPOCH, not the single anchor-epoch state.
            let o_t = origin_m + origin_v_mps * p.t_s;
            assert!((api_p.x_m - (p.r_m.x + o_t.x)).abs() < 1e-6);
            assert!((api_p.y_m - (p.r_m.y + o_t.y)).abs() < 1e-6);
            assert!((api_p.z_m - (p.r_m.z + o_t.z)).abs() < 1e-6);
            assert!((api_p.vx_mps.unwrap() - (p.v_mps.x + origin_v_mps.x)).abs() < 1e-6);
            assert!((api_p.vy_mps.unwrap() - (p.v_mps.y + origin_v_mps.y)).abs() < 1e-6);
            assert!((api_p.vz_mps.unwrap() - (p.v_mps.z + origin_v_mps.z)).abs() < 1e-6);
            assert_eq!(api_p.central_body, "Mercury");
            // Every point must land near the MOVING body's own position at
            // that epoch (within the ring's own small radius of it) -- the
            // concrete check that the ring travels WITH the body instead of
            // staying frozen at the anchor-epoch position.
            let dist_from_body = ((api_p.x_m - o_t.x).powi(2) + (api_p.y_m - o_t.y).powi(2) + (api_p.z_m - o_t.z).powi(2)).sqrt();
            assert!(dist_from_body < 2.0 * r_cap_m, "point strayed {dist_from_body} m from the moving body, expected within ~{r_cap_m} m");
        }
        // And the frozen behavior would fail this: the LAST point's true
        // body position differs from the anchor-epoch position by far more
        // than the ring radius (the exact live-measured failure shape:
        // ~316,000 km of ignored body motion over a 2.95 h ring span).
        let last = points.last().unwrap();
        let body_motion_over_span = (origin_v_mps * last.t_s).norm();
        assert!(
            body_motion_over_span > 10.0 * r_cap_m,
            "test fixture too short to distinguish time-resolved from frozen translation"
        );
    }

    /// Pre-departure-style anchoring: the ring's LAST sample is the burn
    /// point (offset 0 from the anchor epoch); earlier samples map to
    /// NEGATIVE offsets — the body's state before the anchor.
    #[test]
    fn captured_orbit_to_api_pre_departure_anchors_at_the_last_sample() {
        let mu = 3.986_004_418e14_f64;
        let r_park = 6.678e6_f64;
        let v_circ = (mu / r_park).sqrt();
        let (points, _s) = propagate_captured_orbit(
            Vector3::new(r_park, 0.0, 0.0),
            Vector3::new(0.0, v_circ, 0.0),
            mu,
        );
        let origin_m = Vector3::new(1.495_98e11, 0.0, 0.0);
        let origin_v = Vector3::new(0.0, 29_800.0, 0.0);
        let t_anchor = points.last().unwrap().t_s;
        let api = captured_orbit_to_api(
            points.clone(), "Earth", t_anchor,
            |dt_s| Some((origin_m + origin_v * dt_s, origin_v)),
            origin_m, origin_v,
        );
        // Last sample: dt = 0 exactly — translated by the anchor state itself.
        let last_api = api.last().unwrap();
        let last_rel = points.last().unwrap();
        assert!((last_api.x_m - (last_rel.r_m.x + origin_m.x)).abs() < 1e-6);
        assert!((last_api.y_m - (last_rel.r_m.y + origin_m.y)).abs() < 1e-6);
        // First sample: dt = -t_anchor — the body's position BEFORE the
        // anchor epoch, displaced backward along its motion.
        let first_api = api.first().unwrap();
        let first_rel = points.first().unwrap();
        let o_first = origin_m + origin_v * (first_rel.t_s - t_anchor);
        assert!((first_api.x_m - (first_rel.r_m.x + o_first.x)).abs() < 1e-6);
        assert!((first_api.y_m - (first_rel.r_m.y + o_first.y)).abs() < 1e-6);
        assert!(
            (origin_v * t_anchor).norm() > 10.0 * r_park,
            "fixture too short to distinguish anchoring conventions"
        );
    }

    // ── Phase 12c: `propagate_captured_orbit` (the post-capture-burn orbit
    // re-seed + propagation) -- real physics checks, not just "doesn't
    // panic": constant radius (a genuinely circular orbit) and returning to
    // ~its starting point after one full period. Pure floats/vectors, no
    // ANISE almanac needed. ─────────────────────────────────────────────────

    #[test]
    fn captured_orbit_stays_circular_and_returns_after_one_period() {
        // LEO-like scale around an Earth-mass body: mu = 3.986004418e14
        // m^3/s^2 (standard Earth GM), r_cap = 7,000 km.
        let mu = 3.986_004_418e14_f64;
        let r_cap_m = 7.0e6_f64;
        let r_hat = Vector3::new(1.0, 0.0, 0.0);
        let plane_normal_hat = Vector3::new(0.0, 0.0, 1.0);
        let t_hat = plane_normal_hat.cross(&r_hat);
        let v_circ = (mu / r_cap_m).sqrt();

        let (points, _summary) = propagate_captured_orbit(r_hat * r_cap_m, t_hat * v_circ, mu);
        assert!(points.len() > 50, "expected a well-sampled arc, got {} points", points.len());

        // Radius must stay essentially constant throughout -- the defining
        // property of a circular orbit, and independent of any sampling/
        // phase-alignment concern (unlike the "returns to start" check
        // below).
        for p in &points {
            let r = p.r_m.norm();
            let rel_err = (r - r_cap_m).abs() / r_cap_m;
            assert!(rel_err < 1e-3, "radius drifted: r={r}, r_cap={r_cap_m}, rel_err={rel_err}");
        }

        // The analytic Keplerian period for this circular orbit.
        let period_s = 2.0 * std::f64::consts::PI * (r_cap_m.powi(3) / mu).sqrt();
        assert!((POST_CAPTURE_ORBIT_PERIODS - 2.0).abs() < 1e-9, "test assumes 2 periods are returned");

        // Find the sampled point nearest one full period -- with the
        // propagator's own fixed ~200-point sampling over 2 periods, the
        // worst-case phase misalignment is about half a sample interval
        // (duration / 200 / 2 = period / 200), i.e. an angular error on the
        // order of 2*pi/200 ~ 1.8 degrees, corresponding to a chord-distance
        // error of roughly angle_rad * r_cap -- a few percent of r_cap.
        // Assert well inside that bound (5%) as the "returns to
        // approximately its starting point after one period" physics check.
        let nearest = points
            .iter()
            .min_by(|a, b| (a.t_s - period_s).abs().total_cmp(&(b.t_s - period_s).abs()))
            .expect("non-empty arc");
        let r0 = r_hat * r_cap_m;
        let displacement = (nearest.r_m - r0).norm();
        let rel_displacement = displacement / r_cap_m;
        assert!(
            rel_displacement < 0.05,
            "point nearest t=period (t={}, period={}) did not return near the start: r={:?}, r0={:?}, rel_displacement={}",
            nearest.t_s, period_s, nearest.r_m, r0, rel_displacement
        );

        // Sanity: the arc actually covers both requested periods (last
        // sample time close to 2*period, not truncated early).
        let last_t = points.last().unwrap().t_s;
        assert!((last_t - 2.0 * period_s).abs() / period_s < 0.05, "last_t={last_t}, expected ~{}", 2.0 * period_s);
    }

    #[test]
    fn captured_orbit_prograde_sense_matches_plane_normal_convention() {
        // With r_hat = +x and plane_normal_hat (h_hat) = +z, the resulting
        // tangential direction must be h_hat x r_hat = +y (right-handed:
        // h_hat = r_hat x t_hat) -- i.e. the spacecraft should move toward
        // +y first, not -y. This is the exact convention
        // `propagate_captured_orbit`'s doc comment claims and that the
        // single-leg path's real inbound h_hat = r_rel x v_rel depends on
        // for preserving the true approach's sense of motion.
        let mu = 3.986_004_418e14_f64;
        let r_cap_m = 7.0e6_f64;
        let r_hat = Vector3::new(1.0, 0.0, 0.0);
        let plane_normal_hat = Vector3::new(0.0, 0.0, 1.0);
        let t_hat = plane_normal_hat.cross(&r_hat);
        let v_circ = (mu / r_cap_m).sqrt();
        let (points, _summary) = propagate_captured_orbit(r_hat * r_cap_m, t_hat * v_circ, mu);
        let early = points.get(1).expect("at least two points");
        assert!(early.r_m.y > 0.0, "expected initial motion toward +y, got r_m={:?}", early.r_m);
    }

    // ── Real-burn capture orbit (
    // MUST take the real trajectory, and use the arrival burn from the
    // optimizer to find a real orbit around the target. It can be any kind
    // of orbit (e<1)") -- `propagate_captured_orbit` now flies whatever real
    // state a caller hands it, rather than reconstructing an idealized
    // tangential-at-periapsis one internally. These tests exercise it the
    // way the real call sites now do: scale a real (non-tangential)
    // crossing velocity down to local circular speed, in ITS OWN real
    // direction. ─────────────────────────────────────────────────────────

    /// The real crossing point is a genuine off-periapsis geometry (velocity
    /// 40 deg off local-tangential, i.e. a real radial component) -- the
    /// resulting orbit must be genuinely elliptical (not the old idealized
    /// circle), and its periapsis must be AT OR INSIDE the real crossing
    /// radius (the crossing point is where the burn happened, so it can
    /// never be strictly outside the resulting orbit's own bounds).
    #[test]
    fn real_burn_at_a_non_tangential_crossing_produces_a_genuine_ellipse() {
        let mu = 3.986_004_418e14_f64;
        let r_cap_m = 7.0e6_f64;
        let v_circ_target = (mu / r_cap_m).sqrt();
        let r0 = Vector3::new(r_cap_m, 0.0, 0.0);
        // Real crossing velocity direction: 40 deg off the local tangential
        // (+y) -- a real radial component, unlike every idealized-seed test
        // above.
        let real_dir = Vector3::new(40.0_f64.to_radians().sin(), 40.0_f64.to_radians().cos(), 0.0);
        let v0 = real_dir * v_circ_target;

        let (points, summary) = propagate_captured_orbit(r0, v0, mu);

        assert!(summary.eccentricity > 0.1, "expected a genuinely elliptical result, got e={}", summary.eccentricity);
        assert!(summary.eccentricity < 1.0, "expected a bound orbit, got e={}", summary.eccentricity);
        assert!(
            summary.periapsis_m <= r_cap_m * 1.001,
            "periapsis {} must not exceed the real crossing radius {r_cap_m} (the burn happened there)",
            summary.periapsis_m
        );
        assert!(summary.apoapsis_m > summary.periapsis_m, "expected a real, non-degenerate ellipse");
        // Position continuity: the propagated arc's own first point must be
        // exactly the real crossing state -- "no seam" with the incoming
        // trajectory, the whole point of this fix.
        let first = points.first().expect("non-empty arc");
        assert!((first.r_m - r0).norm() < 1e-6, "arc must start exactly at the real crossing point");
        assert!((first.v_mps - v0).norm() < 1e-6, "arc must start exactly at the real post-burn velocity");
    }

    /// A near-tangential real crossing (the common case for a well-targeted
    /// arrival) must still reduce to a near-circular result, same as the
    /// old idealized construction always assumed -- confirms the new code
    /// doesn't regress the well-behaved case, only generalizes the
    /// mistargeted one.
    #[test]
    fn real_burn_at_a_near_tangential_crossing_is_still_nearly_circular() {
        let mu = 3.986_004_418e14_f64;
        let r_cap_m = 7.0e6_f64;
        let v_circ_target = (mu / r_cap_m).sqrt();
        let r0 = Vector3::new(r_cap_m, 0.0, 0.0);
        // 2 deg off tangential -- a real but small radial component, the
        // kind a well-converged search should typically produce.
        let real_dir = Vector3::new(2.0_f64.to_radians().sin(), 2.0_f64.to_radians().cos(), 0.0);
        let v0 = real_dir * v_circ_target;

        let (_points, summary) = propagate_captured_orbit(r0, v0, mu);
        assert!(summary.eccentricity < 0.05, "expected near-circular for a near-tangential crossing, got e={}", summary.eccentricity);
    }

    // ── mga_dv_dsms_inertial_mps (backend parallel-work ask) ───

    /// Real end-to-end check that `mga_dv_dsms_inertial_mps` is populated
    /// and self-consistent with `mga_dv_dsms_ms` for a genuine multi-leg
    /// MGA result (Earth->Venus->Jupiter, real DSMs) — same small-budget MBH
    /// fixture `mga::tests::mga_evj_smoke_mbh` uses, kernel-gated the same
    /// way. Exercises the REAL code path (`optimize_api_with_progress`),
    /// not just the underlying `MgaLegResult` invariant in isolation.
    #[test]
    fn mga_dv_dsms_inertial_mps_matches_magnitudes_for_a_real_evj_result() {
        let almanac = match ephemeris::Almanac::new("kernels/de440s.bsp")
            .or_else(|_| ephemeris::Almanac::new("../kernels/de440s.bsp"))
        {
            Ok(a) => a,
            Err(_) => {
                eprintln!("[skip] mga_dv_dsms_inertial_mps_matches_magnitudes_for_a_real_evj_result: de440s.bsp not found");
                return;
            }
        };
        let toml_str = std::fs::read_to_string("config/evj_flyby.toml")
            .or_else(|_| std::fs::read_to_string("MissionPlanner/config/evj_flyby.toml"))
            .expect("could not read config/evj_flyby.toml");
        let mut cfg: MissionConfig = toml::from_str(&toml_str).expect("could not parse evj_flyby.toml");
        if let Some(opt) = cfg.optimization.as_mut() {
            if let Some(mga) = opt.mga.as_mut() {
                mga.search_method = crate::config::SearchMethod::Mbh;
                mga.mbh.hops = 15;
                mga.mbh.local_max_iter = 60;
            }
        }
        cfg.simulation.output_dir = std::env::temp_dir()
            .join("mga_dv_dsms_inertial_mps_test")
            .to_string_lossy()
            .into_owned();

        let cancelled = std::sync::atomic::AtomicBool::new(false);
        let result = optimize_api_with_progress(&cfg, &almanac, |_, _, _, _, _, _, _, _| {}, |_, _, _, _| {}, &cancelled)
            .expect("MGA optimize should succeed");

        let mags = result.mga_dv_dsms_ms.expect("expected mga_dv_dsms_ms for an MGA result");
        let vecs = result.mga_dv_dsms_inertial_mps.expect("expected mga_dv_dsms_inertial_mps for an MGA result");
        assert_eq!(mags.len(), vecs.len(), "vector and magnitude arrays must have the same length");
        assert!(!mags.is_empty(), "EVJ has at least one DSM leg");
        for (i, (mag, vec)) in mags.iter().zip(vecs.iter()).enumerate() {
            let vec_norm = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
            let rel_err = (vec_norm - mag).abs() / mag.max(1.0);
            assert!(
                rel_err < 1e-6,
                "leg {i}: vector norm {vec_norm} does not match reported magnitude {mag} (rel_err={rel_err})"
            );
        }

        // ── additions, same run (no extra compute) ──────────────
        // DSM epochs: one per DSM vector, strictly ascending, within the
        // mission's own span.
        let epochs = result.mga_dsm_epochs_s.expect("expected mga_dsm_epochs_s for an MGA result");
        assert_eq!(epochs.len(), vecs.len(), "one DSM epoch per DSM vector");
        let tof_s = result.achieved_tof_days * 86_400.0;
        for w in epochs.windows(2) {
            assert!(w[0] < w[1], "DSM epochs must be strictly ascending: {epochs:?}");
        }
        for (i, e) in epochs.iter().enumerate() {
            assert!(
                *e > 0.0 && *e < tof_s,
                "DSM {i} epoch {e} s outside the mission span (0, {tof_s}) s"
            );
        }

        // Arc velocity: every MGA arc point now carries a real heliocentric
        // velocity in a physically plausible range (heliocentric orbital
        // speeds between Venus- and escape-class: ~1e3..1e5 m/s).
        assert!(!result.arc.is_empty());
        for p in &result.arc {
            let (vx, vy, vz) = (
                p.vx_mps.expect("MGA arc velocity should be populated"),
                p.vy_mps.expect("MGA arc velocity should be populated"),
                p.vz_mps.expect("MGA arc velocity should be populated"),
            );
            let speed = (vx * vx + vy * vy + vz * vz).sqrt();
            assert!(
                (1.0e3..1.0e5).contains(&speed),
                "implausible heliocentric speed {speed} m/s on the MGA arc"
            );
        }

        // Departure ΔV vector: present, and its magnitude matches the
        // escape-burn pricing convention `dv_departure_ms` itself uses
        // (both are sqrt(v_inf^2 + 2 mu/r_p) - v_circ at the same parking
        // radius; small tolerance for the two call paths' independent
        // parking-radius resolution).
        let dep_vec = result.departure_dv_inertial_mps.expect("expected departure_dv_inertial_mps");
        let dep_norm = (dep_vec[0] * dep_vec[0] + dep_vec[1] * dep_vec[1] + dep_vec[2] * dep_vec[2]).sqrt();
        let rel = (dep_norm - result.dv_departure_ms).abs() / result.dv_departure_ms.max(1.0);
        assert!(
            rel < 1e-3,
            "departure vector norm {dep_norm} vs dv_departure_ms {} (rel={rel})",
            result.dv_departure_ms
        );
    }

    /// The GA/PSO `departure_dv_inertial_mps` construction is the difference
    /// of two `circular_orbit_burn_state` calls (post-burn minus pre-burn at
    /// the same theta). By the construction's own algebra the difference is
    /// `t_hat*dv*cos(phi) + n_hat*dv*sin(phi)`, whose norm is exactly `dv`
    /// and whose out-of-plane component is exactly `dv*sin(phi)` — verify
    /// both invariants directly so the API-building code path's algebra has
    /// a fast, ephemeris-free guard.
    #[test]
    fn departure_dv_vector_from_burn_state_difference_matches_magnitude_and_plane_split() {
        const MU: f64 = 3.986_004_418e14;
        let r_park = 6_378_137.0 * 1.5;
        let n_hat = Vector3::new(0.0, 0.0, 1.0);
        let ref_hat = Vector3::new(1.0, 0.0, 0.0);
        let (theta, dv, phi) = (1.234_f64, 3_456.0_f64, 0.4_f64);

        let pre = circular_orbit_burn_state(MU, r_park, n_hat, ref_hat, theta, 0.0, 0.0);
        let post = circular_orbit_burn_state(MU, r_park, n_hat, ref_hat, theta, dv, phi);
        let dv_vec = post.v0_mps - pre.v0_mps;

        assert!((dv_vec.norm() - dv).abs() < 1e-9, "|dv_vec| = {} != dv = {dv}", dv_vec.norm());
        assert!(
            (dv_vec.dot(&n_hat) - dv * phi.sin()).abs() < 1e-9,
            "out-of-plane component should be exactly dv*sin(phi)"
        );
        // Pre-burn state is the pure circular parking orbit — sanity anchor.
        assert!((pre.v0_mps.norm() - (MU / r_park).sqrt()).abs() < 1e-9);
    }
}
