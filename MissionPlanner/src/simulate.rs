//! High-fidelity simulation stage (Phase 4) — drives `sim_engine::SimEngine`
//! through the target-body proximity-ops phases for `Orbit`/`SampleReturn`
//! objectives (station-keeping at a target radius) and a best-effort `Flyby`
//! path. `Landing`/`Rendezvous` are not yet implemented — see the Phase 4
//! backlog in `the design notes`.
//!
//! Mission logic (TOML field interpretation, hardware-list traversal, CSV
//! writing) lives here; all physics stays in `sim_engine` and the crates it
//! depends on — this module's only job is resolving `MissionConfig` into the
//! generic types those crates expect.

use std::fs;
use std::thread;

use nalgebra::{Vector3, Vector4};

use body_models::{AtmosphereModel as BodyAtmosphereModel, GravityModel as BodyGravityModel, TargetBody};
use hardware_catalog::{ReactionWheelSpec, ThrusterSpec};
use sim_engine::ekf::{self, EkfConfig};
use sim_engine::{
    Environment, FlybyPhase, HohmannTransferPhase, LandingPhase, OrbitInsertionPhase, OrbitPhase,
    PdGains, Phase, PointingMode, ReactionWheelCluster, SimEngine, SpacecraftProperties,
    SrpTruthModel, StepTelemetry, Thruster, TruthState,
};

use crate::config::{
    AtmosphereModel as CfgAtmosphereModel, GravityModel as CfgGravityModel, HardwareItem,
    MissionConfig, MissionObjective, MissionPhase, PointingMode as CfgPointingMode, SrpModel,
    TargetBodyConfig,
};
use crate::{design, gnc_design};

// ── Entry point ──────────────────────────────────────────────────────────────

pub fn run(cfg: &MissionConfig) {
    println!("\nHigh-fidelity simulation: {}", cfg.mission.name);

    if matches!(cfg.mission.objective, MissionObjective::Rendezvous) {
        eprintln!(
            "Objective 'Rendezvous' is not yet implemented in the simulate stage."
        );
        std::process::exit(1);
    }

    let body = resolve_target_body(&cfg.target_body);
    let sun_pos_from_body_m = sun_position_from_body(cfg);
    let env = Environment {
        body,
        sun_pos_from_body_m,
        mu_sun_m3s2: orbital_models::constants::MU_SUN,
        extra_perturbers: Vec::new(),
    };
    let hw = build_hardware(cfg);

    let mut noop = |_: &StepTelemetry, _: &Vector4<f64>| true;
    let rows = match cfg.mission.objective {
        MissionObjective::Flyby => run_flyby(cfg, env, hw, &mut noop),
        MissionObjective::Landing => run_landing(cfg, env, hw, &mut noop),
        _ => match run_orbit_chain(cfg, env, hw, &mut noop) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },
    };

    write_telemetry(cfg, &rows);

    // Monte Carlo — run N dispersed simulations after the nominal
    if cfg.simulation.monte_carlo_runs > 0 {
        run_monte_carlo(cfg);
    }
}

/// Streaming variant of [`run`] for the `/api/simulate` HTTP/WebSocket endpoint.
/// Identical physics/phase logic — `on_step` is called after every truth step
/// with the step telemetry and the post-step attitude quaternion; the caller
/// is responsible for throttling (e.g. to `measurement_taken` steps only) so
/// it doesn't flood a live feed at the dense truth-integration rate. CSV
/// output is still written at the end, same as the CLI path.
///
/// Returns `Err` instead of calling `std::process::exit` — unlike the CLI
/// path, this runs inside the long-lived HTTP server process, where exiting
/// would kill every other in-flight request, not just this one.
///
/// `on_step` returns `true` to continue, `false` to stop the run early (job
/// cancellation, Phase 9k task 4) — whatever telemetry was collected up to
/// that point is still written out and returned, just short of completion.
pub fn run_streaming(
    cfg: &MissionConfig,
    on_step: &mut dyn FnMut(&StepTelemetry, &Vector4<f64>) -> bool,
) -> Result<SimResult, String> {
    if matches!(cfg.mission.objective, MissionObjective::Rendezvous) {
        return Err(
            "Objective 'Rendezvous' is not yet implemented in the Phase 4 simulate stage".to_string(),
        );
    }

    let body = resolve_target_body(&cfg.target_body);
    let sun_pos_from_body_m = sun_position_from_body(cfg);
    let env = Environment {
        body,
        sun_pos_from_body_m,
        mu_sun_m3s2: orbital_models::constants::MU_SUN,
        extra_perturbers: Vec::new(),
    };
    let hw = build_hardware(cfg);

    let rows: Vec<StepTelemetry> = match cfg.mission.objective {
        MissionObjective::Flyby => run_flyby(cfg, env, hw, on_step),
        MissionObjective::Landing => run_landing(cfg, env, hw, on_step),
        _ => run_orbit_chain(cfg, env, hw, on_step)?,
    };

    let Some(last) = rows.last() else {
        return Err("Simulation produced no telemetry rows".to_string());
    };

    let out_dir = format!("{}/simulate", cfg.simulation.output_dir.trim_end_matches('/'));
    let mut phases_completed: Vec<String> = Vec::new();
    for row in &rows {
        if phases_completed.last().map(|p| p.as_str() != row.phase_name).unwrap_or(true) {
            phases_completed.push(row.phase_name.to_string());
        }
    }
    let dv_total_mps: f64 = rows.iter().filter_map(|r| r.dv_applied_mps).map(|dv| dv.norm()).sum();

    let result = SimResult {
        phases_completed,
        final_r_m: [last.r_truth_m.x, last.r_truth_m.y, last.r_truth_m.z],
        final_v_mps: [last.v_truth_mps.x, last.v_truth_mps.y, last.v_truth_mps.z],
        dv_total_mps,
        nav_csv_path: format!("{out_dir}/nav.csv"),
        attitude_csv_path: format!("{out_dir}/attitude.csv"),
        maneuvers_csv_path: format!("{out_dir}/maneuvers.csv"),
    };

    write_telemetry(cfg, &rows);
    Ok(result)
}

/// One streamed simulation step, sent over the `/api/simulate/:id/stream` WebSocket.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SimStepMsg {
    pub t_s: f64,
    pub phase_name: String,
    pub r_truth_m: [f64; 3],
    pub v_truth_mps: [f64; 3],
    pub r_ekf_m: [f64; 3],
    pub v_ekf_mps: [f64; 3],
    /// Attitude quaternion [w, x, y, z] (body -> inertial/Hill) — same layout
    /// as `sim_engine::TruthState::q`, NOT the `nalgebra::Vector4::{x,y,z,w}`
    /// accessor names, which are purely positional and don't know about
    /// quaternion semantics.
    pub q: [f64; 4],
    pub sigma_pos_m: f64,
    pub sigma_vel_mps: f64,
    pub wheel_speeds_radps: [f64; 4],
    pub wheel_momentum_nms: f64,
    /// 0-1, fraction of max wheel speed (most-saturated wheel).
    pub wheel_sat_frac: f64,
    pub desat_fired: bool,
    /// `None` when OpNav/LIDAR isn't configured, or didn't measure this step
    /// (only populated on `measurement_taken` steps in the first place, same
    /// as everything else in this struct — see `engine.rs::StepTelemetry`).
    pub bearing_residual_rad: Option<f64>,
    pub angular_size_residual_rad: Option<f64>,
    pub lidar_residual_m: Option<f64>,
}

/// Final summary returned by `GET /api/simulate/:id/result`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SimResult {
    pub phases_completed: Vec<String>,
    pub final_r_m: [f64; 3],
    pub final_v_mps: [f64; 3],
    pub dv_total_mps: f64,
    pub nav_csv_path: String,
    pub attitude_csv_path: String,
    pub maneuvers_csv_path: String,
}

// ── Monte Carlo streaming (async-job path for /api/simulate) ───────────────

/// One body-centric point along a Monte Carlo run's sparse trajectory —
/// `MC_TRAJ_SAMPLE_COUNT`-style point density, deliberately not the dense
/// per-step `SimStepMsg` feed (the design notes "Monte Carlo never gets the dense
/// or even decimated-live per-step stream" rule). Body-centric (this stage's
/// frame is always proximity-ops around the target body, never heliocentric
/// like `design.rs::ArcApiPoint`), so this is its own small type rather than
/// reusing that one.
#[derive(Debug, Clone, serde::Serialize)]
pub struct McTrajPoint {
    pub t_s: f64,
    pub r_truth_m: [f64; 3],
}

/// One streamed message per completed Monte Carlo run, sent over
/// `/api/simulate/:id/stream` when `monte_carlo_runs > 0`. Mirrors
/// `McResult`'s scalar fields plus an optional sparse trajectory for the
/// stride-selected subset — same "scalars for all, trajectory for a subset"
/// split `design.rs::MonteCarloSampleApiResult` already uses for the
/// narrowing-stage Monte Carlo.
#[derive(Debug, Clone, serde::Serialize)]
pub struct McRunMsg {
    pub run_index: usize,
    pub seed: u64,
    pub final_r_m: f64,
    pub min_r_m: f64,
    pub max_r_m: f64,
    pub final_sigma_pos_m: f64,
    pub total_dv_ms: f64,
    pub crashed: bool,
    /// Sparse trajectory (`MC_TRAJ_SAMPLE_COUNT`-equivalent point count),
    /// only for the stride-selected subset of runs — `None` otherwise.
    pub trajectory: Option<Vec<McTrajPoint>>,
}

/// Final summary returned once all Monte Carlo runs complete, analogous to
/// [`SimResult`] for the single-run path.
#[derive(Debug, Clone, serde::Serialize)]
pub struct McSummaryResult {
    pub n_runs: usize,
    pub n_crashed: usize,
    pub mean_final_r_m: f64,
    pub min_r_m: f64,
    pub max_r_m: f64,
    pub mean_dv_total_ms: f64,
    pub mc_summary_csv_path: String,
}

/// Same per-trajectory point density used by `design.rs::MC_TRAJ_SAMPLE_COUNT`
/// for the narrowing-stage Monte Carlo — kept as an independent constant
/// here since `simulate.rs` and `design.rs` are different stages with no
/// shared "MC point density" config today; if a future `[simulation.mc_dispersions]`
/// section is added (per `run_monte_carlo`'s doc comment), this should move there.
const MC_STREAM_TRAJ_POINTS: usize = 18;

/// How many of the `n` runs get a sparse trajectory streamed — a fixed
/// fraction (capped) so payload doesn't scale linearly with `monte_carlo_runs`,
/// matching `design.rs`'s stride-selected-subset approach for its own Monte
/// Carlo trajectories.
const MC_STREAM_TRAJ_SUBSET_MAX: usize = 20;

/// Streaming variant of the CLI's Monte Carlo runner ([`run_monte_carlo`]),
/// for the `/api/simulate` async-job path when `cfg.simulation.monte_carlo_runs
/// > 0`. Runs the same dispersed simulations (same dispersion model, same
/// per-run physics as [`run_single_mc`]) but instead of writing straight to
/// CSV at the end, calls `on_run_done` once per completed run with a sparse
/// summary message — never the dense per-step feed the single-run path uses,
/// per the design notes Monte Carlo streaming rule. `mc_summary.csv` is still
/// written at the end, identical in content to the CLI path.
pub fn run_monte_carlo_streaming(
    cfg: &MissionConfig,
    on_run_done: &mut dyn FnMut(McRunMsg),
) -> Result<McSummaryResult, String> {
    let n = cfg.simulation.monte_carlo_runs as usize;
    if n == 0 {
        return Err("monte_carlo_runs is 0 — nothing to run".to_string());
    }

    // Stride-selected subset gets a sparse trajectory, same selection
    // philosophy as design.rs's MC_TRAJ_COUNT: evenly spaced indices across
    // the full run set, capped so payload doesn't grow with n.
    let subset_n = n.min(MC_STREAM_TRAJ_SUBSET_MAX);
    let stride = (n as f64 / subset_n as f64).max(1.0);
    let traj_indices: std::collections::HashSet<usize> =
        (0..subset_n).map(|k| ((k as f64) * stride) as usize).collect();

    // thread::scope lets scoped threads borrow `cfg` directly — no Clone
    // needed, same pattern as run_monte_carlo. on_run_done is called from the
    // calling thread, in run order, after collecting each handle's result
    // (not from inside the worker threads), so the caller doesn't need to
    // worry about cross-thread Send bounds on the callback itself.
    let mut results: Vec<(usize, McResult, Option<Vec<McTrajPoint>>)> = Vec::with_capacity(n);
    thread::scope(|s| {
        let handles: Vec<_> = (0..n as u64)
            .map(|i| {
                let want_traj = traj_indices.contains(&(i as usize));
                s.spawn(move || {
                    let (r, traj) = run_single_mc_with_traj(cfg, i, want_traj);
                    (i as usize, r, traj)
                })
            })
            .collect();
        results = handles.into_iter().filter_map(|h| h.join().ok()).collect();
    });
    results.sort_by_key(|(idx, ..)| *idx);

    for (run_index, r, traj) in &results {
        on_run_done(McRunMsg {
            run_index: *run_index,
            seed: r.seed,
            final_r_m: r.final_r_m,
            min_r_m: r.min_r_m,
            max_r_m: r.max_r_m,
            final_sigma_pos_m: r.final_sigma_pos_m,
            total_dv_ms: r.total_dv_ms,
            crashed: r.crashed,
            trajectory: traj.clone(),
        });
    }

    let mc_results: Vec<McResult> = results.into_iter().map(|(_, r, _)| r).collect();
    let n_crashed = mc_results.iter().filter(|r| r.crashed).count();
    let mean_final_r: f64 = mc_results.iter().map(|r| r.final_r_m).sum::<f64>() / n as f64;
    let min_r_all: f64 = mc_results.iter().map(|r| r.min_r_m).fold(f64::INFINITY, f64::min);
    let max_r_all: f64 = mc_results.iter().map(|r| r.max_r_m).fold(f64::NEG_INFINITY, f64::max);
    let mean_dv: f64 = mc_results.iter().map(|r| r.total_dv_ms).sum::<f64>() / n as f64;

    write_mc_summary(cfg, &mc_results);
    let out_dir = format!("{}/simulate", cfg.simulation.output_dir.trim_end_matches('/'));

    Ok(McSummaryResult {
        n_runs: n,
        n_crashed,
        mean_final_r_m: mean_final_r,
        min_r_m: min_r_all,
        max_r_m: max_r_all,
        mean_dv_total_ms: mean_dv,
        mc_summary_csv_path: format!("{out_dir}/mc_summary.csv"),
    })
}

// ── Target body resolution ───────────────────────────────────────────────────

fn resolve_target_body(tb: &TargetBodyConfig) -> TargetBody {
    if let Some(mut preset) = TargetBody::by_name(&tb.name) {
        preset.mu_m3s2 = tb.mu_m3s2;
        preset.radius_m = tb.radius_m;
        if let Some(g) = body_gravity_from_cfg(tb) {
            preset.gravity = g;
        }
        preset
    } else {
        let gravity = body_gravity_from_cfg(tb).unwrap_or_else(|| {
            panic!(
                "target_body '{}' is not in the body_models catalog and gravity_model could \
                 not be resolved (missing j2/j3/j4?)",
                tb.name
            )
        });
        let atmosphere = match tb.atmosphere {
            CfgAtmosphereModel::None => BodyAtmosphereModel::None,
            _ => {
                eprintln!(
                    "Warning: atmosphere model '{}' for non-catalog body '{}' has no \
                     scale_height_m/rho0_kg_m3 in the current TOML schema — treating as None \
                     for this simulation. (Known gap: MissionPlanner::config::AtmosphereModel \
                     doesn't yet carry those parameters for custom bodies; flagged for a \
                     follow-up, not blocking — Bennu/Apophis-class missions use None anyway.)",
                    tb.atmosphere, tb.name,
                );
                BodyAtmosphereModel::None
            }
        };
        TargetBody {
            name: tb.name.clone(),
            kind: "Custom",
            mu_m3s2: tb.mu_m3s2,
            radius_m: tb.radius_m,
            gravity,
            atmosphere,
            spin_rate_rads: 0.0,
            pole_ra_deg: tb.pole_ra_deg,
            pole_dec_deg: tb.pole_dec_deg,
            // Irrelevant here: `sim_engine`'s 6DOF simulation is always
            // body-centric on a single fixed target, no SOI switching.
            primary: None,
            // Custom bodies are always used body-centrically; SMA is not needed.
            sma_m: None,
            orbital_elements: None,
        }
    }
}

fn body_gravity_from_cfg(tb: &TargetBodyConfig) -> Option<BodyGravityModel> {
    match tb.gravity_model {
        CfgGravityModel::PointMass => Some(BodyGravityModel::PointMass),
        CfgGravityModel::J2 => tb.j2.map(|j2| BodyGravityModel::J2 { j2 }),
        CfgGravityModel::J2J3J4 => match (tb.j2, tb.j3, tb.j4) {
            (Some(j2), Some(j3), Some(j4)) => Some(BodyGravityModel::J2J3J4 { j2, j3, j4 }),
            _ => None,
        },
        // Not yet a body_models::GravityModel variant (Phase 5) — caller falls back.
        CfgGravityModel::SphericalHarmonic => None,
    }
}

/// Heliocentric position of the target body (Sun -> body) [m] at the epoch the
/// proximity-ops simulation starts from.
fn sun_position_from_body(cfg: &MissionConfig) -> Vector3<f64> {
    match design::keplerian_from_cfg(cfg) {
        Some(kep) => {
            let arr_jd = arrival_epoch_jd(cfg);
            let (r, _v) = kep.state_at_jd(arr_jd);
            Vector3::new(r[0], r[1], r[2])
        }
        None => {
            eprintln!(
                "Warning: target_body.ephemeris is not \"Keplerian\" (or no keplerian_orbit \
                 given) — ANISE-ephemeris target bodies aren't wired into the simulate stage \
                 yet (Phase 5). Assuming a 1 AU heliocentric distance for SRP/tidal purposes."
            );
            Vector3::new(trajectory_solver::keplerian::AU_M, 0.0, 0.0)
        }
    }
}

/// Arrival epoch [JD] for the proximity-ops simulation. Prefers the
/// design-stage's computed arrival JD (`out/<mission>/design/{best_arc,
/// diff_correction}.csv`); falls back to `departure_epoch + cruise TOF
/// midpoint` if no design output exists yet. Precision here only matters for
/// the body's heliocentric distance (SRP magnitude, tidal perturbation), which
/// changes slowly over a multi-week proximity-ops mission — adequate for the
/// physical-sanity validation this stage targets.
fn arrival_epoch_jd(cfg: &MissionConfig) -> f64 {
    if let Some(jd) = read_arr_jd_from_design_output(cfg) {
        return jd;
    }
    eprintln!(
        "Note: no design-stage output found (run `mission-planner design` first for a precise \
         arrival epoch) — approximating arrival epoch from departure_epoch + cruise TOF midpoint."
    );
    let dep_str = cfg.trajectory.departure_epoch.as_deref().unwrap_or("");
    let dep_epoch = design::parse_epoch(dep_str).unwrap_or_else(|e| {
        eprintln!("Error parsing departure_epoch '{dep_str}': {e} — defaulting to J2000.0.");
        ephemeris::Epoch::from_gregorian_utc(2000, 1, 1, 12, 0, 0, 0)
    });
    let dep_jd = design::epoch_to_jd(dep_epoch);
    let tof_days = cfg
        .trajectory
        .cruise
        .as_ref()
        .and_then(|c| match (c.tof_days_min, c.tof_days_max) {
            (Some(lo), Some(hi)) => Some(0.5 * (lo + hi)),
            (None, Some(hi)) => Some(hi),
            (Some(lo), None) => Some(lo),
            _ => None,
        })
        .unwrap_or(0.0);
    dep_jd + tof_days
}

fn read_arr_jd_from_design_output(cfg: &MissionConfig) -> Option<f64> {
    let out_dir = cfg.simulation.output_dir.trim_end_matches('/');
    for fname in ["best_arc.csv", "diff_correction.csv"] {
        let path = format!("{out_dir}/design/{fname}");
        if let Ok(text) = fs::read_to_string(&path) {
            if let Some(jd) = parse_csv_column(&text, "arr_jd") {
                return Some(jd);
            }
        }
    }
    None
}

fn read_v_inf_from_design_output(cfg: &MissionConfig) -> Option<f64> {
    let out_dir = cfg.simulation.output_dir.trim_end_matches('/');
    let path = format!("{out_dir}/design/best_arc.csv");
    let text = fs::read_to_string(path).ok()?;
    parse_csv_column(&text, "v_inf_arr_ms")
}

/// Minimal header-indexed CSV column reader (single data row) — avoids adding
/// a CSV-parsing dependency for this one cross-stage lookup.
fn parse_csv_column(text: &str, column: &str) -> Option<f64> {
    let mut lines = text.lines();
    let header = lines.next()?;
    let idx = header.split(',').position(|c| c == column)?;
    let row = lines.next()?;
    row.split(',').nth(idx)?.trim().parse::<f64>().ok()
}

// ── Hardware resolution ──────────────────────────────────────────────────────

struct Hardware {
    sc: SpacecraftProperties,
    wheel_cluster: ReactionWheelCluster,
    rcs_thrusters: Vec<Thruster>,
    rcs_isp_s: f64,
    pd_gains: PdGains,
    opnav_cfg: Option<(f64, f64)>,
    lidar_cfg: Option<(f64, f64)>,
}

fn build_hardware(cfg: &MissionConfig) -> Hardware {
    let (rcs_thrusters, rcs_isp_s) = rcs_from_hardware(cfg);
    Hardware {
        sc: build_spacecraft_properties(cfg),
        wheel_cluster: wheel_cluster_from_hardware(cfg),
        rcs_thrusters,
        rcs_isp_s,
        pd_gains: pd_gains_from_cfg(cfg),
        opnav_cfg: opnav_cfg_from_hardware(cfg),
        lidar_cfg: lidar_cfg_from_hardware(cfg),
    }
}

fn bus_cross_section_m2(cfg: &MissionConfig) -> f64 {
    let d = cfg.spacecraft.bus_dims_m;
    (d[0] * d[1]).max(d[1] * d[2]).max(d[0] * d[2])
}

/// Sums area only from UNPLACED `SolarPanel` entries — a placed panel
/// (`position_m`+`normal` both set) contributes its own individual `Plate`
/// via `placed_solar_panels_from_hardware` instead of feeding the automatic
/// symmetric ±y pair `build_plates` synthesizes from this total; counting it
/// in both places would double the panel's real SRP area.
fn panel_area_total(cfg: &MissionConfig) -> f64 {
    cfg.spacecraft
        .hardware
        .iter()
        .filter_map(|h| match h {
            HardwareItem::SolarPanel { area_m2, position_m: None, normal: None, .. } => Some(*area_m2),
            _ => None,
        })
        .sum()
}

/// Build one `sim_engine::Plate` per PLACED `HardwareItem::SolarPanel`
/// (`position_m`+`normal` both set — spacecraft-builder placement
/// extension), on top of the automatic ±y pair `build_plates`
/// still synthesizes from every UNPLACED panel's `area_m2` (see
/// `panel_area_total`'s doc comment for why the two never double-count).
/// `width_m`/`height_m`, when both given, override `area_m2` for the plate
/// this panel contributes — the mockup review's "sizing is a first-class
/// play axis" finding needs two independent live dimensions, not a single
/// area scalar. Missing `rho_s`/`rho_d` fall back to the same defaults the
/// automatic panel pair uses (0.08/0.10).
fn placed_solar_panels_from_hardware(cfg: &MissionConfig) -> Vec<sim_engine::Plate> {
    use sim_engine::Plate;
    const PANEL_RHO_S_DEFAULT: f64 = 0.08;
    const PANEL_RHO_D_DEFAULT: f64 = 0.10;

    cfg.spacecraft
        .hardware
        .iter()
        .filter_map(|h| match h {
            HardwareItem::SolarPanel {
                area_m2, position_m: Some(pos), normal: Some(normal), width_m, height_m, rho_s, rho_d, ..
            } => {
                let area = match (width_m, height_m) {
                    (Some(w), Some(hh)) => w * hh,
                    _ => *area_m2,
                };
                let n = Vector3::new(normal[0], normal[1], normal[2]);
                let n = if n.norm() > 1e-12 { n / n.norm() } else { Vector3::new(0.0, 0.0, 1.0) };
                Some(Plate {
                    normal: n,
                    area,
                    rho_s: rho_s.unwrap_or(PANEL_RHO_S_DEFAULT),
                    rho_d: rho_d.unwrap_or(PANEL_RHO_D_DEFAULT),
                    double_sided: true, // deployed panels are double-sided, matching the automatic pair
                    center_body: Vector3::new(pos[0], pos[1], pos[2]),
                })
            }
            _ => None,
        })
        .collect()
}

pub fn build_spacecraft_properties(cfg: &MissionConfig) -> SpacecraftProperties {
    let inertia = if cfg.spacecraft.derive_inertia_from_geometry.unwrap_or(false) {
        // Derived-inertia wiring: diagonal of the real parallel-axis-derived tensor
        // (bus box + every placed/itemized HardwareItem, about the true
        // CoM) instead of the manually-typed field. Off-diagonal terms are
        // dropped here, not ignored upstream -- `crate::vehicle_properties`
        // computes the full Mat3; `SpacecraftProperties` only has room for
        // a diagonal (see `derive_inertia_from_geometry`'s own doc comment).
        let vp = crate::vehicle_properties::compute_vehicle_properties(cfg);
        Vector3::new(vp.inertia_kgm2[0][0], vp.inertia_kgm2[1][1], vp.inertia_kgm2[2][2])
    } else {
        Vector3::new(
            cfg.spacecraft.inertia_diag_kgm2[0],
            cfg.spacecraft.inertia_diag_kgm2[1],
            cfg.spacecraft.inertia_diag_kgm2[2],
        )
    };
    SpacecraftProperties {
        mass_kg: cfg.spacecraft.mass_kg,
        inertia_diag_kgm2: inertia,
        srp: build_srp_model(cfg),
        drag_area_m2: bus_cross_section_m2(cfg),
    }
}

fn build_srp_model(cfg: &MissionConfig) -> SrpTruthModel {
    let c_r = cfg.spacecraft.reflectivity_cr.unwrap_or(1.4);
    match cfg.spacecraft.srp_model {
        SrpModel::Cannonball => SrpTruthModel::Cannonball {
            c_r,
            area_m2: bus_cross_section_m2(cfg) + panel_area_total(cfg),
        },
        SrpModel::FlatPlate => SrpTruthModel::FlatPlate { plates: build_plates(cfg) },
        SrpModel::NPlate => {
            eprintln!(
                "Note: NPlate SRP is not yet implemented in sim_engine — using the same \
                 generic bus+panel FlatPlate model as a stand-in."
            );
            SrpTruthModel::FlatPlate { plates: build_plates(cfg) }
        }
    }
}

/// Generic 6-bus-face + 2-panel plate model, plus any mission-declared
/// `CustomPlate` hardware entries (Phase 13a) for shapes beyond a plain box.
/// Optical properties (rho_s/rho_d) for the bus/panel faces are the same
/// representative MLI-blanket/solar-cell values used by
/// `GNC/AutonomousNavigation/src/dynamics/srp.rs::spacecraft_plates()` — not a
/// mission-specific constant, typical for any small spacecraft bus/panel —
/// since the TOML schema doesn't (yet) carry per-face optical properties for
/// the *automatic* bus faces. `CustomPlate` entries carry their own optical
/// properties (or fall back to the bus defaults — see `custom_plates_from_hardware`).
/// See `docs/MP/MANUAL.md` §4.2 for the governing SRP force/torque law.
///
/// **Torque-arm origin.** Every plate's `center_body`
/// is built directly from the geometric-center-relative `HardwareItem`
/// placement fields (see `vehicle_properties`'s module doc comment) — but
/// `flat_plate_torque_body`'s `tau = center_body x F` is only correct when
/// `center_body` is measured from the TRUE center of mass. Gated on the same
/// `derive_inertia_from_geometry` opt-in as the derived-inertia
/// wiring (both are "use `compute_vehicle_properties`'s real geometry"
/// toggles): when set, every plate returned here — bus faces included — is
/// re-expressed relative to the derived CoM before being handed to the SRP
/// torque law. `None`/`false` (every pre-ask-#13 config) is byte-for-byte
/// the old behavior.
pub fn build_plates(cfg: &MissionConfig) -> Vec<sim_engine::Plate> {
    use sim_engine::Plate;
    const BUS_RHO_S: f64 = 0.30;
    const BUS_RHO_D: f64 = 0.20;
    const PANEL_RHO_S: f64 = 0.08;
    const PANEL_RHO_D: f64 = 0.10;

    let [lx, ly, lz] = cfg.spacecraft.bus_dims_m;
    let mut plates = vec![
        Plate { normal: Vector3::new(1.0, 0.0, 0.0), area: ly * lz, rho_s: BUS_RHO_S, rho_d: BUS_RHO_D, double_sided: false, center_body: Vector3::new(lx / 2.0, 0.0, 0.0) },
        Plate { normal: Vector3::new(-1.0, 0.0, 0.0), area: ly * lz, rho_s: BUS_RHO_S, rho_d: BUS_RHO_D, double_sided: false, center_body: Vector3::new(-lx / 2.0, 0.0, 0.0) },
        Plate { normal: Vector3::new(0.0, 1.0, 0.0), area: lx * lz, rho_s: BUS_RHO_S, rho_d: BUS_RHO_D, double_sided: false, center_body: Vector3::new(0.0, ly / 2.0, 0.0) },
        Plate { normal: Vector3::new(0.0, -1.0, 0.0), area: lx * lz, rho_s: BUS_RHO_S, rho_d: BUS_RHO_D, double_sided: false, center_body: Vector3::new(0.0, -ly / 2.0, 0.0) },
        Plate { normal: Vector3::new(0.0, 0.0, 1.0), area: lx * ly, rho_s: BUS_RHO_S, rho_d: BUS_RHO_D, double_sided: false, center_body: Vector3::new(0.0, 0.0, lz / 2.0) },
        Plate { normal: Vector3::new(0.0, 0.0, -1.0), area: lx * ly, rho_s: BUS_RHO_S, rho_d: BUS_RHO_D, double_sided: false, center_body: Vector3::new(0.0, 0.0, -lz / 2.0) },
    ];

    let panel_area = panel_area_total(cfg);
    if panel_area > 0.0 {
        let half_area = panel_area / 2.0;
        let span = (half_area / lz).max(0.1);
        let offset = ly / 2.0 + span / 2.0;
        plates.push(Plate { normal: Vector3::new(0.0, 0.0, 1.0), area: half_area, rho_s: PANEL_RHO_S, rho_d: PANEL_RHO_D, double_sided: true, center_body: Vector3::new(0.0, offset, 0.0) });
        plates.push(Plate { normal: Vector3::new(0.0, 0.0, 1.0), area: half_area, rho_s: PANEL_RHO_S, rho_d: PANEL_RHO_D, double_sided: true, center_body: Vector3::new(0.0, -offset, 0.0) });
    }

    plates.extend(custom_plates_from_hardware(cfg));
    plates.extend(placed_solar_panels_from_hardware(cfg));

    if cfg.spacecraft.derive_inertia_from_geometry.unwrap_or(false) {
        let com = crate::vehicle_properties::compute_vehicle_properties(cfg).com_m;
        let com_v = Vector3::new(com[0], com[1], com[2]);
        for p in &mut plates {
            p.center_body -= com_v;
        }
    }

    plates
}

/// Build one `sim_engine::Plate` per `HardwareItem::CustomPlate` declared in
/// `cfg.spacecraft.hardware` (Phase 13a) — lets a mission add arbitrary
/// deployables (dish antennas, asymmetric panels, instrument covers, ...)
/// on top of the automatic bus+panel geometry `build_plates` already
/// produces. Missing `rho_s`/`rho_d` fall back to the same bus-face
/// defaults `build_plates` uses for its 6 automatic faces (0.30/0.20) —
/// a reasonable representative default absent mission-specific optics.
/// `normal` is re-normalized here (config validation already rejects a
/// zero/non-finite vector, but does not require unit length).
fn custom_plates_from_hardware(cfg: &MissionConfig) -> Vec<sim_engine::Plate> {
    use sim_engine::Plate;
    const CUSTOM_RHO_S_DEFAULT: f64 = 0.30;
    const CUSTOM_RHO_D_DEFAULT: f64 = 0.20;

    cfg.spacecraft
        .hardware
        .iter()
        .filter_map(|h| match h {
            HardwareItem::CustomPlate { normal, area_m2, center_offset_m, rho_s, rho_d, double_sided, .. } => {
                let n = Vector3::new(normal[0], normal[1], normal[2]);
                let n = if n.norm() > 1e-12 { n / n.norm() } else { Vector3::new(0.0, 0.0, 1.0) };
                Some(Plate {
                    normal: n,
                    area: *area_m2,
                    rho_s: rho_s.unwrap_or(CUSTOM_RHO_S_DEFAULT),
                    rho_d: rho_d.unwrap_or(CUSTOM_RHO_D_DEFAULT),
                    double_sided: double_sided.unwrap_or(false),
                    center_body: Vector3::new(center_offset_m[0], center_offset_m[1], center_offset_m[2]),
                })
            }
            _ => None,
        })
        .collect()
}

pub fn wheel_cluster_from_hardware(cfg: &MissionConfig) -> ReactionWheelCluster {
    let item = cfg.spacecraft.hardware.iter().find_map(|h| match h {
        HardwareItem::ReactionWheelCluster { max_speed_rads, max_torque_nm, inertia_kgm2, .. } => {
            Some((*max_speed_rads, *max_torque_nm, *inertia_kgm2))
        }
        _ => None,
    });
    let default = ReactionWheelSpec::medium();
    let (max_speed, max_torque, inertia) = match item {
        Some((s, t, i)) => (
            s.unwrap_or(default.max_speed_rads),
            t.unwrap_or(default.max_torque_nm),
            i.unwrap_or(default.inertia_kgm2),
        ),
        None => (default.max_speed_rads, default.max_torque_nm, default.inertia_kgm2),
    };
    // Wertz & Larson SMAD — matches gnc_design::DESAT_FRACTION.
    const DESAT_FRACTION: f64 = 0.8;
    ReactionWheelCluster::four_wheel_pyramid(inertia, max_speed, max_torque, DESAT_FRACTION)
}

/// Builds the RCS thruster list the live sim's control allocation
/// (`sim_engine::control::allocate` -> `attitude_control::thruster_selection`,
/// `docs/MP/MANUAL.md` §10.2) actually fires against.
///
/// Placed-thruster wiring: `attitude_control::thruster_selection`
/// already operates generically over any `&[Thruster]` — it was never
/// hardcoded to the idealized symmetric 12-couple layout, that layout was
/// just the only thing anything ever built. So giving individually-placed
/// `HardwareItem::RcsThruster` entries a real path into the sim did not
/// need a new allocation architecture (weighted pseudo-inverse / priority
/// chain / QP, the `MomentumManagementLaw` "N-actuator generalization"
/// extension point in §10.3) — that generalization is for combining
/// MULTIPLE actuator TYPES together (wheels + RCS + magnetorquers + SRP
/// trim), a separate, still-future concern. A single-type allocation over
/// an arbitrary (including asymmetric, incomplete-coverage) thruster list
/// was already correct; it just needed real geometry to run on.
///
/// Precedence: real placed `RcsThruster` entries, when any exist, are used
/// EXACTLY as the user built them — a genuinely bad layout (missing torque
/// authority about some axis, off-axis couples inducing net translational
/// force) shows up as bad, not smoothed into a plausible symmetric
/// equivalent. This is deliberate, not a limitation: it is what makes an
/// "does this layout actually hold attitude" preview meaningful. The old
/// aggregate `RCS` variant (symmetric moment-arm) is the fallback ONLY
/// when no `RcsThruster` entries are present, preserving every existing
/// config's behavior byte-for-byte.
///
/// `RcsThruster.isp_s` gives each placed
/// thruster its own real specific impulse — `None` falls back to
/// `ThrusterSpec::monoprop()`'s Isp, the same default the old aggregate
/// path always used, so an unset value reproduces the exact prior
/// behavior. `sim_engine::control::allocate` prices propellant per-thruster
/// from this, not against one aggregate scalar — a real mixed-class layout
/// (e.g. fine ColdGas thrusters alongside a higher-thrust Monoprop set)
/// now accounts for propellant honestly. The `f64` this function still
/// returns is the aggregate-path Isp, used only by the older body-centric
/// `SixDofEngine` (`engine.rs`), which has its own separate, untouched
/// single-scalar propellant model — it is no longer meaningful for (and no
/// longer accepted by) `run_cruise_leg`'s `allocate()` call.
pub fn rcs_from_hardware(cfg: &MissionConfig) -> (Vec<Thruster>, f64) {
    let default = ThrusterSpec::monoprop();

    let placed: Vec<Thruster> = cfg
        .spacecraft
        .hardware
        .iter()
        .filter_map(|h| match h {
            HardwareItem::RcsThruster { thrust_n, position_m, direction, isp_s, .. } => {
                let dir = Vector3::new(direction[0], direction[1], direction[2]);
                let norm = dir.norm();
                if norm < 1e-12 {
                    return None; // config validation already rejects this; defensive only
                }
                Some(Thruster {
                    dir: dir / norm,
                    pos: Vector3::new(position_m[0], position_m[1], position_m[2]),
                    thrust_n: *thrust_n,
                    isp_s: isp_s.unwrap_or(default.isp_s),
                })
            }
            _ => None,
        })
        .collect();
    if !placed.is_empty() {
        return (placed, default.isp_s);
    }

    let item = cfg.spacecraft.hardware.iter().find_map(|h| match h {
        HardwareItem::RCS { thrust_n, moment_arm_m, .. } => Some((*thrust_n, *moment_arm_m)),
        _ => None,
    });
    let (thrust_n, moment_arm_m) = match item {
        Some((t, a)) => (t.unwrap_or(default.thrust_n), a.unwrap_or(0.5)),
        None => (default.thrust_n, 0.5),
    };
    let thrusters = sim_engine::actuators::build_rcs(&ThrusterSpec { thrust_n, ..default }, moment_arm_m);
    (thrusters, default.isp_s)
}

// Review finding E1: `kp=0.05`/`kd=0.6` (below) were fixed
// constants tuned for one specific reference vehicle (I ~ 367-667 kg*m^2,
// the Bennu vehicle) and applied REGARDLESS of the real Phase-02-built
// vehicle's own inertia. Confirmed by direct live testing to cause exactly
// what the constants' own derivation predicts: a heavier vehicle gets a
// near-undamped loop (natural frequency omega_n = sqrt(kp/I) collapses as I
// grows, so the SAME kd/kp ratio no longer damps it), a lighter or more
// torque-constrained vehicle gets commanded torque the wheels can't deliver
// (immediate saturation, see finding E-ii/E-iv). Derived instead from the
// vehicle's own real inertia (the SAME derived tensor `/api/design/vehicle`
// and Table 2's inertia breakdown already compute) and its real wheel
// torque authority, so the closed loop stays reasonably well-behaved
// (target damping ratio ZETA_TARGET) regardless of what's actually built in
// Phase 02. `cfg.gnc.reaction_wheel_kp/kd` remain real, explicit overrides
// (unchanged) for anyone who wants to hand-tune past the derived values.
pub const ZETA_TARGET: f64 = 0.85;
// A conservative "typical large reorientation" angle used only to size
// kp's torque-vs-angle relationship (kp*theta/2 ~ tau_wheel_max at
// theta=REF_SLEW_ANGLE_RAD, the quaternion-vector-part small-angle
// convention `command_torque` uses -- see this function's own derivation
// comment below) -- NOT a claim about how far this vehicle will actually
// slew. Larger => gentler (lower kp) derived gains.
const REF_SLEW_ANGLE_RAD: f64 = std::f64::consts::FRAC_PI_4; // 45 deg
// Hard ceiling so a tiny, high-torque-authority vehicle doesn't derive an
// implausibly stiff loop (e.g. a cubesat-scale inertia against a
// full-size wheel cluster) -- real spacecraft attitude loops run well
// under this bandwidth.
const OMEGA_N_CEILING_RADPS: f64 = 0.05;
// Nyquist-ish margin matching `cruise_demo.rs`'s own documented incident
// (a 3600 s tick genuinely destabilized a real vehicle/gains combination)
// -- at least this many ticks per natural period.
const MIN_TICKS_PER_NATURAL_PERIOD: f64 = 15.0;
const DEFAULT_TICK_S_FOR_GAIN_DERIVATION: f64 = 30.0;

pub fn pd_gains_from_cfg(cfg: &MissionConfig) -> PdGains {
    let derived = derived_reaction_wheel_pd_gains(cfg);
    // Reaction-wheel PD uses small gains and NO dead-bands.  Dead-bands are
    // appropriate for RCS bang-bang (prevents constant thruster firing) but
    // not for continuous wheel torque.
    PdGains {
        kp: cfg.gnc.reaction_wheel_kp.unwrap_or(derived.0),
        kd: cfg.gnc.reaction_wheel_kd.unwrap_or(derived.1),
        pointing_db_rad: cfg.gnc.pointing_deadband_rad.unwrap_or(0.0),
        rate_db_rads: cfg.gnc.rate_deadband_radps.unwrap_or(0.0),
    }
}

/// (kp, kd) derived from the real vehicle's inertia + wheel torque
/// authority. The reaction-wheel loop is a 2nd-order system driven by
/// `command_torque`'s `tau = -kp*q_err_vec - kd*omega`, where `q_err_vec`
/// is the quaternion's vector part -- for small errors `|q_err_vec| ~
/// theta/2` (half the rotation angle), so the EFFECTIVE proportional
/// relationship between torque and angle is `kp/2`, not `kp` directly:
/// `I*theta_ddot + kd*theta_dot + (kp/2)*theta = 0`, giving
/// `omega_n = sqrt(kp/(2*I))` and `zeta = kd/(2*sqrt((kp/2)*I)) =
/// kd/(2*sqrt(kp*I/2))`. Solving for (kp, kd) at a target (omega_n, zeta):
/// `kp = 2*I*omega_n^2`, `kd = 2*I*zeta*omega_n`.
fn derived_reaction_wheel_pd_gains(cfg: &MissionConfig) -> (f64, f64) {
    let sc = build_spacecraft_properties(cfg);
    // Max principal inertia -- the conservative (slowest-to-respond) axis;
    // using the smallest would under-derive kp/kd for the other two axes.
    let inertia = sc.inertia_diag_kgm2.x.max(sc.inertia_diag_kgm2.y).max(sc.inertia_diag_kgm2.z).max(1e-6);

    let wheels = wheel_cluster_from_hardware(cfg);
    let tau_wheel_max = wheels.max_torque.max(1e-9);

    let tick_s = cfg
        .cruise_seed
        .as_ref()
        .map(|s| s.tick_s)
        .filter(|t| *t > 0.0)
        .unwrap_or(DEFAULT_TICK_S_FOR_GAIN_DERIVATION);
    derived_pd_gains(inertia, tau_wheel_max, tick_s)
}

/// The actuator-agnostic form of the derivation above (three-
/// layer attitude-control architecture, `attitude_tuning.rs`): `(kp, kd)`
/// for a loop about inertia `inertia_kgm2` driven by an actuator whose
/// worst-axis torque authority is `tau_authority_nm`, at control tick
/// `tick_s`. The wheels call it with the cluster `max_torque`; the thruster
/// modes with the placed RCS layout's worst-axis authority — the SAME rule,
/// so a burn is no longer flown on gains sized for a 30× weaker actuator.
pub fn derived_pd_gains(inertia_kgm2: f64, tau_authority_nm: f64, tick_s: f64) -> (f64, f64) {
    derived_pd_gains_with_ceiling(inertia_kgm2, tau_authority_nm, tick_s, OMEGA_N_CEILING_RADPS)
}

/// Same derivation with a caller-chosen plausibility ceiling on `ω_n`
/// (mode-scheduled tick): `OMEGA_N_CEILING_RADPS` (0.05) is
/// sized for quiescent WHEEL loops; a thruster-driven burn-attitude hold
/// legitimately runs stiffer (powered-flight TVC/RCS loops at 0.1–0.5
/// rad/s — Wie, *Space Vehicle Dynamics and Control*, 2nd ed., Ch. 7), so
/// the thruster-mode derivation passes its own, higher ceiling
/// (`attitude_tuning::THRUSTER_OMEGA_N_CEILING_RADPS`). Without this, a
/// short burn tick raises the tick cap but the wheel-sized ceiling
/// immediately re-caps the loop and the RCS authority stays unusable.
pub fn derived_pd_gains_with_ceiling(inertia_kgm2: f64, tau_authority_nm: f64, tick_s: f64, omega_n_ceiling_radps: f64) -> (f64, f64) {
    let inertia = inertia_kgm2.max(1e-6);
    let tau_wheel_max = tau_authority_nm.max(1e-9);
    // omega_n capped by three real limits (most conservative wins):
    // - what the actuator can actually deliver for a REF_SLEW_ANGLE_RAD
    //   correction without exceeding tau_max (kp*theta/2 = tau_max
    //   at theta = REF_SLEW_ANGLE_RAD => kp = 2*tau_max/REF_SLEW_ANGLE_RAD
    //   => omega_n = sqrt(kp/(2*I)) = sqrt(tau_max/(I*REF_SLEW_ANGLE_RAD))).
    // - the control loop's own tick-rate margin.
    // - a hard ceiling against an implausibly stiff derived loop.
    let omega_n_torque = (tau_wheel_max / (inertia * REF_SLEW_ANGLE_RAD)).sqrt();
    // Real bug found running the cruise regression suite: omega_n and 1/T
    // (period) differ by a factor of 2*pi -- `1/(15*tick_s)` bounds the
    // natural period to ~94 ticks, not the intended 15, making this term
    // dominate and silently over-damp/slow every derived loop regardless of
    // real torque authority. `2*pi/(N*tick_s)` is the correct bound for "at
    // least N ticks per natural period."
    let omega_n_tick = 2.0 * std::f64::consts::PI / (MIN_TICKS_PER_NATURAL_PERIOD * tick_s);
    let omega_n = omega_n_torque.min(omega_n_tick).min(omega_n_ceiling_radps.max(1e-6));

    let kp = 2.0 * inertia * omega_n * omega_n;
    let kd = 2.0 * inertia * ZETA_TARGET * omega_n;
    (kp, kd)
}

fn opnav_cfg_from_hardware(cfg: &MissionConfig) -> Option<(f64, f64)> {
    cfg.spacecraft.hardware.iter().find_map(|h| match h {
        HardwareItem::OpNavCamera { bearing_noise_mrad, angular_size_noise_mrad, .. } => Some((
            bearing_noise_mrad.unwrap_or(0.1) * 1.0e-3,
            angular_size_noise_mrad.unwrap_or(0.2) * 1.0e-3,
        )),
        _ => None,
    })
}

fn lidar_cfg_from_hardware(cfg: &MissionConfig) -> Option<(f64, f64)> {
    cfg.spacecraft.hardware.iter().find_map(|h| match h {
        HardwareItem::Lidar { range_noise_m, max_range_m, .. } => {
            Some((range_noise_m.unwrap_or(5.0), max_range_m.unwrap_or(5000.0)))
        }
        _ => None,
    })
}

fn build_ekf_config(cfg: &MissionConfig, env: &Environment, sc: &SpacecraftProperties) -> EkfConfig {
    let state_dim = gnc_design::ekf_state_dim(cfg);
    let area_over_mass = match &sc.srp {
        SrpTruthModel::Cannonball { area_m2, .. } => area_m2 / sc.mass_kg,
        SrpTruthModel::FlatPlate { .. } => bus_cross_section_m2(cfg) / sc.mass_kg,
    };
    let p_srp = orbital_models::pressure_at(env.sun_pos_from_body_m.norm());
    EkfConfig {
        state_dim,
        mu_central_m3s2: env.body.mu_m3s2,
        mu_sun_m3s2: env.mu_sun_m3s2,
        sun_pos_from_body_m: env.sun_pos_from_body_m,
        srp_area_over_mass: area_over_mass,
        p_srp,
        markov_tau_s: ekf::DEFAULT_MARKOV_TAU_S,
        markov_sigma_a: ekf::DEFAULT_MARKOV_SIGMA_A,
        q_r: 0.0,
        // Equivalent white-noise acceleration PSD spanning the same variance
        // budget as the Gauss-Markov term over its correlation time — a filter
        // tuning choice, not a cited physical constant.
        q_v: ekf::DEFAULT_MARKOV_SIGMA_A.powi(2) * ekf::DEFAULT_MARKOV_TAU_S,
        q_cr: 1.0e-10,
    }
}

fn initial_ekf_state(ekf_cfg: &EkfConfig, r0: Vector3<f64>, v0: Vector3<f64>, c_r0: f64) -> ekf::EkfState {
    // Filter-initialization uncertainties — a reasonable starting guess, not
    // physical constants; the EKF converges from here via measurements.
    const SIGMA_R0_FRAC: f64 = 0.01;
    const SIGMA_V0_MPS: f64 = 0.01;
    const SIGMA_CR0: f64 = 0.2;
    let sigma_r0 = (SIGMA_R0_FRAC * r0.norm()).max(1.0);
    ekf::init(ekf_cfg, r0, v0, c_r0, sigma_r0, SIGMA_V0_MPS, SIGMA_CR0, ekf::DEFAULT_MARKOV_SIGMA_A, 0.0)
}

fn pointing_mode_from_cfg(cfg: &MissionConfig) -> PointingMode {
    match &cfg.gnc.pointing_mode {
        CfgPointingMode::Nadir => PointingMode::Nadir,
        CfgPointingMode::VelocityAligned => PointingMode::VelocityAligned,
        other => {
            eprintln!("Note: pointing mode '{other}' not yet supported by sim_engine — using Nadir.");
            PointingMode::Nadir
        }
    }
}

// ── Orbit (station-keeping) phase chain — Orbit / SampleReturn objectives ────

fn phase_label(p: MissionPhase) -> Option<&'static str> {
    match p {
        MissionPhase::Capture => Some("Capture"),
        MissionPhase::Survey => Some("Survey"),
        MissionPhase::CloseOrbit => Some("CloseOrbit"),
        MissionPhase::Flyover => Some("Flyover"),
        MissionPhase::ScienceHold => Some("ScienceHold"),
        MissionPhase::RadioScience => Some("RadioScience"),
        MissionPhase::Proximity => Some("Proximity"),
        MissionPhase::Cruise | MissionPhase::Departure | MissionPhase::Descent | MissionPhase::Landing => None,
    }
}

/// One orbit per phase — enough to exercise station-keeping and let the EKF
/// converge through several measurement updates, without simulating an
/// unrealistically long wall-clock run for an MVP physical-sanity check (this
/// does not reproduce `proximity_mission`'s real per-phase day-counts, which
/// are Bennu-specific operational heuristics, not generic mission data).
fn phase_duration_s(mu_m3s2: f64, target_radius_m: f64) -> f64 {
    2.0 * std::f64::consts::PI * (target_radius_m.powi(3) / mu_m3s2).sqrt()
}

/// Build phases and initial conditions for Orbit/SampleReturn objectives.
///
/// If the TOML phases include `Capture` and the capture config has an
/// `approach_v_inf_mps` value, the Capture phase is modelled as a hyperbolic
/// approach that fires a single LOI burn at periapsis (`OrbitInsertionPhase`).
/// All other orbit phases are `OrbitPhase` with station-keeping.
fn run_orbit_chain(
    cfg: &MissionConfig,
    env: Environment,
    hw: Hardware,
    on_step: &mut dyn FnMut(&StepTelemetry, &Vector4<f64>) -> bool,
) -> Result<Vec<StepTelemetry>, String> {
    let target_radius_m = match cfg.trajectory.capture.as_ref().and_then(|c| c.target_orbit_radius_m) {
        Some(r) => r,
        None => {
            return Err(
                "simulate requires [trajectory.capture].target_orbit_radius_m for \
                 Orbit/SampleReturn objectives"
                    .to_string(),
            );
        }
    };
    let mu = env.body.mu_m3s2;
    let pointing = pointing_mode_from_cfg(cfg);

    // Optional hyperbolic approach: if the Capture phase has v_inf set, start from
    // a hyperbolic IC and use OrbitInsertionPhase; otherwise start in circular orbit.
    let v_inf_mps = cfg.trajectory.capture.as_ref()
        .and_then(|c| c.approach_v_inf_mps);
    let has_insertion = v_inf_mps.is_some()
        && cfg.trajectory.phases.contains(&MissionPhase::Capture);

    // Build the full phase sequence, inserting a physical HohmannTransferPhase
    // between any consecutive orbit phases that change altitude.  prev_r tracks
    // the orbit radius at the end of the last phase so we know when a transfer
    // is needed.
    let mut all_phases: Vec<Box<dyn Phase>> = Vec::new();
    // prev_r starts at the initial circular orbit (or capture orbit for insertion).
    let mut prev_r = if has_insertion { target_radius_m } else { target_radius_m };

    for p in &cfg.trajectory.phases {
        match p {
            MissionPhase::Capture if has_insertion => {
                all_phases.push(Box::new(OrbitInsertionPhase::new("Capture", target_radius_m, pointing, mu)));
                prev_r = target_radius_m;
            }
            _ => {
                if let Some(label) = phase_label(*p) {
                    // Per-phase radius override: TOML [trajectory.per_phase_radii] takes
                    // precedence over the global capture target radius.
                    let r = cfg.trajectory.per_phase_radii
                        .get(label)
                        .copied()
                        .unwrap_or(target_radius_m);

                    // Insert a physical Hohmann transfer arc whenever the altitude changes.
                    if (r - prev_r).abs() > 1.0 {
                        all_phases.push(Box::new(HohmannTransferPhase::new(
                            "Transfer", r, pointing, mu,
                        )));
                    }

                    all_phases.push(Box::new(OrbitPhase::new(
                        label, r, phase_duration_s(mu, r), pointing,
                    )));
                    prev_r = r;
                }
            }
        }
    }

    if all_phases.is_empty() {
        return Err(
            "no simulate-able phases in trajectory.phases — Cruise/Departure/Descent/\
             Landing aren't modeled at the target body yet"
                .to_string(),
        );
    }

    let c_r0 = cfg.spacecraft.reflectivity_cr.unwrap_or(1.4);
    let terminator = cfg.trajectory.capture.as_ref().map(|c| c.terminator_orbit).unwrap_or(false);

    // Initial conditions: hyperbolic approach OR circular orbit.
    // Terminator orbit: orbital pole h aligned with the body→Sun direction so
    // the spacecraft is always at ~90° phase angle (day/night boundary).
    let (r0, v0) = if let Some(v_inf) = v_inf_mps {
        let r_start = 20.0 * target_radius_m;
        let v_start = (v_inf * v_inf + 2.0 * mu / r_start).sqrt();
        (Vector3::new(r_start, 0.0, 0.0), Vector3::new(0.0, -v_start, 0.0))
    } else if terminator {
        terminator_orbit_ic(target_radius_m, (mu / target_radius_m).sqrt(), &env.sun_pos_from_body_m)
    } else {
        let r0 = Vector3::new(target_radius_m, 0.0, 0.0);
        let v0 = Vector3::new(0.0, (mu / target_radius_m).sqrt(), 0.0);
        (r0, v0)
    };
    let q0 = sim_engine::guidance::desired_quaternion(pointing, &r0, &v0);
    let truth0 = TruthState { t_s: 0.0, r_m: r0, v_mps: v0, q: q0, omega_radps: Vector3::zeros(), wheel_speeds_radps: [0.0; 4], c_r: c_r0 };

    let ekf_cfg = build_ekf_config(cfg, &env, &hw.sc);
    let ekf0 = initial_ekf_state(&ekf_cfg, r0, v0, c_r0);

    // SK dead-band: 5% of the capture orbit radius.  Phase-specific tolerances
    // would require passing the tolerance through the Phase trait — deferred.
    let sk_tolerance_m = 0.05 * target_radius_m;

    let mut engine = SimEngine::new(
        truth0, ekf0, hw.sc, env, ekf_cfg, hw.wheel_cluster, hw.rcs_thrusters, hw.rcs_isp_s,
        hw.pd_gains, hw.opnav_cfg, hw.lidar_cfg, cfg.simulation.dt_truth_s, cfg.simulation.dt_meas_s,
        sk_tolerance_m, cfg.spacecraft.propellant_mass_kg, 42,
    );

    let mut rows = Vec::new();
    'phases: for phase in all_phases.iter_mut() {
        let dur_label = match phase.target_radius_m() {
            Some(r) => format!("orbit period {:.2} days", phase_duration_s(mu, r) / 86_400.0),
            None => "coasting transfer arc".to_string(),
        };
        println!("  Phase: {}  ({})", phase.name(), dur_label);
        loop {
            let telem = engine.step(phase.as_mut());
            let keep_going = on_step(&telem, &engine.truth.q);
            let done = phase.step(&engine.truth, cfg.simulation.dt_truth_s);
            rows.push(telem);
            if !keep_going {
                break 'phases;
            }
            if done {
                break;
            }
        }
    }
    println!(
        "  Final orbit radius: {:.1} m (target {target_radius_m:.1} m)  |  EKF sigma_pos: {:.2} m  |  propellant remaining: {:.4} kg",
        engine.truth.r_m.norm(), engine.ekf.sigma_pos_m(), engine.propellant_kg,
    );
    Ok(rows)
}

// ── Terminator orbit IC ───────────────────────────────────────────────────────

/// Compute position and velocity vectors for a circular terminator orbit:
/// the orbital plane is perpendicular to the body→Sun direction so the
/// spacecraft is always at ~90° solar phase angle (flying the terminator).
///
/// Construction: h = r × v must equal ŝ (sun direction).  Given r0_hat ⊥ ŝ,
/// the choice v0_hat = ŝ × r0_hat satisfies r0_hat × v0_hat = ŝ exactly
/// (vector triple product with r0_hat ⊥ ŝ: r0_hat × (ŝ × r0_hat) = ŝ).
fn terminator_orbit_ic(
    r_m: f64,
    v_circ: f64,
    sun_pos_from_body: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let s_hat = sun_pos_from_body.normalize();
    // Build r0_hat perpendicular to ŝ via Gram-Schmidt from a reference axis.
    let reference = if s_hat.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let r0_hat = (reference - s_hat * s_hat.dot(&reference)).normalize();
    // v0_hat = ŝ × r0_hat → h = r0_hat × v0_hat = ŝ
    let v0_hat = s_hat.cross(&r0_hat);   // already unit since s_hat ⊥ r0_hat
    (r0_hat * r_m, v0_hat * v_circ)
}

// ── Flyby — best-effort, validated against real flyby data in a later pass ──

fn run_flyby(
    cfg: &MissionConfig,
    env: Environment,
    hw: Hardware,
    on_step: &mut dyn FnMut(&StepTelemetry, &Vector4<f64>) -> bool,
) -> Vec<StepTelemetry> {
    eprintln!(
        "Note: Flyby initial-condition construction from the design-stage v_inf is best-effort \
         — full validation against real flyby data is deferred to Phase 5."
    );
    let mu = env.body.mu_m3s2;
    let v_inf_ms = read_v_inf_from_design_output(cfg).unwrap_or(1000.0);
    let periapsis_m = 5.0 * env.body.radius_m.max(1.0);
    let energy = 0.5 * v_inf_ms * v_inf_ms;
    let v_p = (2.0 * (energy + mu / periapsis_m)).max(0.0).sqrt();

    let pointing = PointingMode::Nadir;
    let c_r0 = cfg.spacecraft.reflectivity_cr.unwrap_or(1.4);
    let r0 = Vector3::new(periapsis_m, 0.0, 0.0);
    // Small inward radial component superimposed on the periapsis tangential
    // speed, purely so `r . v` starts negative and `FlybyPhase`'s periapsis-
    // crossing condition has a transition to find — not a precise hyperbolic
    // asymptote state (see the best-effort note above).
    let v0 = Vector3::new(-0.02 * v_p, v_p, 0.0);
    let q0 = sim_engine::guidance::desired_quaternion(pointing, &r0, &v0);
    let truth0 = TruthState { t_s: 0.0, r_m: r0, v_mps: v0, q: q0, omega_radps: Vector3::zeros(), wheel_speeds_radps: [0.0; 4], c_r: c_r0 };

    let ekf_cfg = build_ekf_config(cfg, &env, &hw.sc);
    let ekf0 = initial_ekf_state(&ekf_cfg, r0, v0, c_r0);

    let mut engine = SimEngine::new(
        truth0, ekf0, hw.sc, env, ekf_cfg, hw.wheel_cluster, hw.rcs_thrusters, hw.rcs_isp_s,
        hw.pd_gains, hw.opnav_cfg, hw.lidar_cfg, cfg.simulation.dt_truth_s, cfg.simulation.dt_meas_s,
        0.0, cfg.spacecraft.propellant_mass_kg, 42,
    );

    let mut phase = FlybyPhase::new();
    let mut rows = Vec::new();
    println!("  Phase: {}", phase.name());
    loop {
        let telem = engine.step(&mut phase);
        let keep_going = on_step(&telem, &engine.truth.q);
        let done = phase.step(&engine.truth, cfg.simulation.dt_truth_s);
        rows.push(telem);
        if done || !keep_going || rows.len() > 200_000 {
            break;
        }
    }
    rows
}

// ── Landing — deorbit from parking orbit to terminal altitude ─────────────────

fn run_landing(
    cfg: &MissionConfig,
    env: Environment,
    hw: Hardware,
    on_step: &mut dyn FnMut(&StepTelemetry, &Vector4<f64>) -> bool,
) -> Vec<StepTelemetry> {
    let mu = env.body.mu_m3s2;
    let body_radius_m = env.body.radius_m;
    let pointing = pointing_mode_from_cfg(cfg);
    let c_r0 = cfg.spacecraft.reflectivity_cr.unwrap_or(1.4);

    // Deorbit radius: explicit config → capture target → 10× body radius fallback
    let deorbit_r = cfg.trajectory.landing.as_ref()
        .and_then(|l| l.deorbit_radius_m)
        .or_else(|| cfg.trajectory.capture.as_ref()?.target_orbit_radius_m)
        .unwrap_or(10.0 * body_radius_m);
    let terminal_alt = cfg.trajectory.landing.as_ref()
        .and_then(|l| l.terminal_altitude_m)
        .unwrap_or(50.0);

    // Circular parking orbit IC
    let r0 = Vector3::new(deorbit_r, 0.0, 0.0);
    let v0 = Vector3::new(0.0, (mu / deorbit_r).sqrt(), 0.0);
    let q0 = sim_engine::guidance::desired_quaternion(pointing, &r0, &v0);
    let truth0 = TruthState {
        t_s: 0.0, r_m: r0, v_mps: v0, q: q0,
        omega_radps: Vector3::zeros(), wheel_speeds_radps: [0.0; 4], c_r: c_r0,
    };

    let ekf_cfg = build_ekf_config(cfg, &env, &hw.sc);
    let ekf0 = initial_ekf_state(&ekf_cfg, r0, v0, c_r0);

    let mut engine = SimEngine::new(
        truth0, ekf0, hw.sc, env, ekf_cfg, hw.wheel_cluster, hw.rcs_thrusters, hw.rcs_isp_s,
        hw.pd_gains, hw.opnav_cfg, hw.lidar_cfg, cfg.simulation.dt_truth_s, cfg.simulation.dt_meas_s,
        0.0, cfg.spacecraft.propellant_mass_kg, 42,
    );

    let mut phase = LandingPhase::new("Descent", body_radius_m, Some(terminal_alt), pointing);

    // Safety cap: deorbit arc is at most one full orbit period so a mis-configured
    // terminal altitude never produces an infinite loop.
    let max_steps = (2.0 * std::f64::consts::PI * (deorbit_r.powi(3) / mu).sqrt()
        / cfg.simulation.dt_truth_s) as usize
        + 1;

    println!(
        "  Phase: Descent  (deorbit from {deorbit_r:.0} m, terminal alt {terminal_alt:.0} m)"
    );
    let mut rows = Vec::new();
    loop {
        let telem = engine.step(&mut phase);
        let keep_going = on_step(&telem, &engine.truth.q);
        let done = phase.step(&engine.truth, cfg.simulation.dt_truth_s);
        rows.push(telem);
        if done || !keep_going || rows.len() >= max_steps {
            break;
        }
    }

    let final_alt = engine.truth.r_m.norm() - body_radius_m;
    let v_final = engine.truth.v_mps.norm();
    if phase.touchdown {
        println!("  Touchdown: altitude {final_alt:.1} m  |  impact speed {v_final:.3} m/s");
    } else {
        println!(
            "  Powered-descent handoff: altitude {final_alt:.1} m  |  speed {v_final:.4} m/s  \
             |  EKF σ_r {:.2} m",
            engine.ekf.sigma_pos_m()
        );
    }
    rows
}

// ── Monte Carlo runner ───────────────────────────────────────────────────────

/// Per-run summary statistics collected by the Monte Carlo runner.
struct McResult {
    seed: u64,
    final_r_m: f64,
    min_r_m: f64,
    max_r_m: f64,
    final_sigma_pos_m: f64,
    total_dv_ms: f64,
    crashed: bool,
}

/// Run `cfg.simulation.monte_carlo_runs` dispersed orbit simulations in
/// parallel threads, collect per-run statistics, and write `mc_summary.csv`
/// to the simulate output directory. CLI-only entry point — the `/api/simulate`
/// HTTP/WebSocket path uses [`run_monte_carlo_streaming`] instead, which
/// shares the same per-run physics via [`run_single_mc_with_traj`] but
/// streams a sparse per-run summary as each run completes.
///
/// Dispersions applied to the nominal circular initial state:
///   - Position: Gaussian, σ = 1% of target radius (navigation uncertainty)
///   - Velocity: Gaussian, σ = 5% of circular speed (nav + burn execution)
/// These represent realistic proximity-ops arrival errors; mission-specific
/// values belong in a future `[simulation.mc_dispersions]` TOML section.
fn run_monte_carlo(cfg: &MissionConfig) {
    let n = cfg.simulation.monte_carlo_runs as usize;
    println!("\nMonte Carlo: {n} runs");

    // thread::scope lets scoped threads borrow `cfg` directly — no Clone needed.
    let mut results: Vec<McResult> = Vec::with_capacity(n);
    thread::scope(|s| {
        let handles: Vec<_> = (0..n as u64)
            .map(|i| s.spawn(move || run_single_mc(cfg, i)))
            .collect();
        results = handles.into_iter().filter_map(|h| h.join().ok()).collect();
    });
    results.sort_by(|a, b| a.seed.cmp(&b.seed));

    let n_crashed = results.iter().filter(|r| r.crashed).count();
    let mean_final_r: f64 = results.iter().map(|r| r.final_r_m).sum::<f64>() / n as f64;
    let min_r_all: f64 = results.iter().map(|r| r.min_r_m).fold(f64::INFINITY, f64::min);
    let max_r_all: f64 = results.iter().map(|r| r.max_r_m).fold(f64::NEG_INFINITY, f64::max);
    let mean_dv: f64 = results.iter().map(|r| r.total_dv_ms).sum::<f64>() / n as f64;

    println!(
        "  Crashed: {n_crashed}/{n}  |  mean final r: {mean_final_r:.0} m  |  \
         orbit range: [{min_r_all:.0}, {max_r_all:.0}] m  |  mean ΔV: {mean_dv:.4} m/s"
    );

    write_mc_summary(cfg, &results);
}

fn run_single_mc(cfg: &MissionConfig, seed: u64) -> McResult {
    run_single_mc_with_traj(cfg, seed, false).0
}

/// Core single-run Monte Carlo worker shared by the CLI path ([`run_single_mc`])
/// and the streaming path ([`run_monte_carlo_streaming`]). Identical physics/
/// dispersion model either way; `want_traj` only controls whether a sparse
/// trajectory (`MC_STREAM_TRAJ_POINTS`, stride-sampled from the full per-step
/// telemetry) is collected alongside the scalar `McResult` — collecting it
/// costs a small amount of extra bookkeeping per step, so the CLI path (which
/// never streams anything) skips it via `run_single_mc`'s `false`.
fn run_single_mc_with_traj(cfg: &MissionConfig, seed: u64, want_traj: bool) -> (McResult, Option<Vec<McTrajPoint>>) {
    // Box-Muller Gaussian from SplitMix64 — no external RNG dependency
    let mut sm = seed.wrapping_add(0x9e3779b97f4a7c15);

    let rng_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_add(0x9e3779b97f4a7c15);
        let z = {
            let mut z = *s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        (z as i64 as f64) / (i64::MAX as f64)
    };

    let gauss = |s: &mut u64| -> f64 {
        let u = (rng_f64(s) + 1.0) * 0.5; // [0,1)
        let v = rng_f64(s);
        let r = (-2.0 * u.max(1e-15).ln()).sqrt();
        r * (2.0 * std::f64::consts::PI * v).cos()
    };

    let target_radius_m = match cfg.trajectory.capture.as_ref().and_then(|c| c.target_orbit_radius_m) {
        Some(r) => r,
        None => {
            let result = McResult { seed, final_r_m: 0.0, min_r_m: 0.0, max_r_m: 0.0, final_sigma_pos_m: 0.0, total_dv_ms: 0.0, crashed: true };
            return (result, None);
        }
    };
    let mu = cfg.target_body.mu_m3s2;
    let v_circ = (mu / target_radius_m).sqrt();

    // Disperse initial conditions around the nominal circular orbit
    let sigma_r = 0.01 * target_radius_m;
    let sigma_v = 0.05 * v_circ;
    let r0 = Vector3::new(
        target_radius_m + sigma_r * gauss(&mut sm),
        sigma_r * gauss(&mut sm),
        sigma_r * gauss(&mut sm),
    );
    let v0 = Vector3::new(
        sigma_v * gauss(&mut sm),
        v_circ + sigma_v * gauss(&mut sm),
        sigma_v * gauss(&mut sm),
    );

    let body = resolve_target_body(&cfg.target_body);
    let sun_pos_from_body_m = sun_position_from_body(cfg);
    let env = Environment {
        body,
        sun_pos_from_body_m,
        mu_sun_m3s2: orbital_models::constants::MU_SUN,
        extra_perturbers: Vec::new(),
    };
    let hw = build_hardware(cfg);
    let pointing = pointing_mode_from_cfg(cfg);

    let c_r0 = cfg.spacecraft.reflectivity_cr.unwrap_or(1.4);
    let q0 = sim_engine::guidance::desired_quaternion(pointing, &r0, &v0);
    let truth0 = TruthState {
        t_s: 0.0, r_m: r0, v_mps: v0, q: q0,
        omega_radps: Vector3::zeros(), wheel_speeds_radps: [0.0; 4], c_r: c_r0,
    };

    let ekf_cfg = build_ekf_config(cfg, &env, &hw.sc);
    let ekf0 = initial_ekf_state(&ekf_cfg, r0, v0, c_r0);
    let sk_tolerance_m = 0.05 * target_radius_m;

    let mut engine = SimEngine::new(
        truth0, ekf0, hw.sc, env, ekf_cfg, hw.wheel_cluster, hw.rcs_thrusters, hw.rcs_isp_s,
        hw.pd_gains, hw.opnav_cfg, hw.lidar_cfg, cfg.simulation.dt_truth_s, cfg.simulation.dt_meas_s,
        sk_tolerance_m, cfg.spacecraft.propellant_mass_kg, seed + 1,
    );

    let phase_duration = phase_duration_s(mu, target_radius_m);
    let mut orbit_phase = OrbitPhase::new("MC", target_radius_m, phase_duration, pointing);

    let mut min_r = f64::INFINITY;
    let mut max_r = f64::NEG_INFINITY;
    let mut total_dv = 0.0;
    let mut crashed = false;
    let body_radius = cfg.target_body.radius_m;

    // Time-based stride sampling: record a sparse point roughly every
    // phase_duration / (MC_STREAM_TRAJ_POINTS - 1) seconds. Time-based, not
    // step-count-based, since the loop can end early on a crash and the
    // total step count isn't known ahead of time — same fixed-point-budget
    // rationale as design.rs's MC_TRAJ_SAMPLE_COUNT, just adapted to a
    // step-driven loop instead of a fixed-duration propagator call.
    let sample_interval_s = (phase_duration / (MC_STREAM_TRAJ_POINTS.max(2) - 1) as f64).max(1.0);
    let mut traj: Vec<McTrajPoint> = Vec::new();
    let mut next_sample_t = 0.0_f64;

    loop {
        let telem = engine.step(&mut orbit_phase);
        let r = telem.r_truth_m.norm();
        if r < min_r { min_r = r; }
        if r > max_r { max_r = r; }
        if let Some(dv) = telem.dv_applied_mps { total_dv += dv.norm(); }
        if want_traj && telem.t_s >= next_sample_t {
            traj.push(McTrajPoint { t_s: telem.t_s, r_truth_m: [telem.r_truth_m.x, telem.r_truth_m.y, telem.r_truth_m.z] });
            next_sample_t += sample_interval_s;
        }
        if r < body_radius {
            crashed = true;
            break;
        }
        if orbit_phase.step(&engine.truth, cfg.simulation.dt_truth_s) {
            break;
        }
    }
    if want_traj {
        // Always include the true final point, even if it falls between strides.
        let last_t = engine.truth.t_s;
        if traj.last().map(|p| p.t_s != last_t).unwrap_or(true) {
            traj.push(McTrajPoint { t_s: last_t, r_truth_m: [engine.truth.r_m.x, engine.truth.r_m.y, engine.truth.r_m.z] });
        }
    }

    let result = McResult {
        seed,
        final_r_m: engine.truth.r_m.norm(),
        min_r_m: min_r,
        max_r_m: max_r,
        final_sigma_pos_m: engine.ekf.sigma_pos_m(),
        total_dv_ms: total_dv,
        crashed,
    };
    (result, if want_traj { Some(traj) } else { None })
}

fn write_mc_summary(cfg: &MissionConfig, results: &[McResult]) {
    let out_dir = format!("{}/simulate", cfg.simulation.output_dir.trim_end_matches('/'));
    let _ = fs::create_dir_all(&out_dir);
    let path = format!("{out_dir}/mc_summary.csv");

    let mut rows = vec!["seed,final_r_m,min_r_m,max_r_m,sigma_pos_m,total_dv_ms,crashed".to_string()];
    for r in results {
        rows.push(format!(
            "{},{:.1},{:.1},{:.1},{:.3},{:.6},{}",
            r.seed, r.final_r_m, r.min_r_m, r.max_r_m, r.final_sigma_pos_m, r.total_dv_ms,
            r.crashed as u8,
        ));
    }
    write_csv(&path, &rows);
    println!("  MC summary: {path}  ({} rows)", rows.len() - 1);
}

// ── Telemetry output ─────────────────────────────────────────────────────────

fn write_telemetry(cfg: &MissionConfig, rows: &[StepTelemetry]) {
    let out_dir = format!("{}/simulate", cfg.simulation.output_dir.trim_end_matches('/'));
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("\nWarning: could not create output dir '{out_dir}': {e}");
        return;
    }

    let mut nav = vec!["time_s,phase,tx_m,ty_m,tz_m,tvx,tvy,tvz,ex_m,ey_m,ez_m,evx,evy,evz,sigma_r_m,sigma_v_mps".to_string()];
    let mut attitude = vec!["time_s,phase,omega_norm_rads,pointing_err_mrad,w1_rads,w2_rads,w3_rads,w4_rads,h_w_norm_nms,wheel_sat_pct,desat".to_string()];
    let mut maneuvers = vec!["time_s,dv_x_ms,dv_y_ms,dv_z_ms,dv_mag_ms,label".to_string()];

    for row in rows {
        if row.measurement_taken {
            nav.push(format!(
                "{:.1},{},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.4},{:.6}",
                row.t_s, row.phase_name,
                row.r_truth_m.x, row.r_truth_m.y, row.r_truth_m.z,
                row.v_truth_mps.x, row.v_truth_mps.y, row.v_truth_mps.z,
                row.r_ekf_m.x, row.r_ekf_m.y, row.r_ekf_m.z,
                row.v_ekf_mps.x, row.v_ekf_mps.y, row.v_ekf_mps.z,
                row.sigma_pos_m, row.sigma_vel_mps,
            ));
        }
        attitude.push(format!(
            "{:.1},{},{:.6},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.4},{}",
            row.t_s, row.phase_name, row.omega_norm_radps, row.pointing_err_rad * 1000.0,
            row.wheel_speeds_radps[0], row.wheel_speeds_radps[1], row.wheel_speeds_radps[2], row.wheel_speeds_radps[3],
            row.wheel_momentum_nms, row.wheel_sat_frac * 100.0, row.desat_fired as u8,
        ));
        if let Some(dv) = row.dv_applied_mps {
            maneuvers.push(format!("{:.1},{:.6},{:.6},{:.6},{:.6},SK", row.t_s, dv.x, dv.y, dv.z, dv.norm()));
        }
    }

    let nav_path = format!("{out_dir}/nav.csv");
    let attitude_path = format!("{out_dir}/attitude.csv");
    let maneuvers_path = format!("{out_dir}/maneuvers.csv");
    write_csv(&nav_path, &nav);
    write_csv(&attitude_path, &attitude);
    write_csv(&maneuvers_path, &maneuvers);

    println!("\nOutput:");
    println!("  {nav_path}  ({} rows)", nav.len() - 1);
    println!("  {attitude_path}  ({} rows)", attitude.len() - 1);
    println!("  {maneuvers_path}  ({} rows)", maneuvers.len() - 1);
}

fn write_csv(path: &str, rows: &[String]) {
    if let Err(e) = fs::write(path, rows.join("\n") + "\n") {
        eprintln!("  Warning: could not write {path}: {e}");
    }
}

// ── Tests (Phase 13a — custom plate geometry) ─────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use orbital_models::{flat_plate_force_body, flat_plate_torque_body};

    /// Minimal valid mission TOML (mirrors `config.rs`'s own test helper),
    /// with an injectable `[[spacecraft.hardware]]` block for CustomPlate cases.
    fn mission_toml_with_hardware(hardware_block: &str) -> String {
        format!(
            r#"
[mission]
name = "Test"
objective = "Orbit"

[target_body]
name = "Bennu"
ephemeris = "Keplerian"

[spacecraft]
mass_kg = 1000.0
dry_mass_kg = 800.0
propellant_mass_kg = 200.0
bus_dims_m = [2.0, 2.0, 0.63]
inertia_diag_kgm2 = [366.67, 366.67, 666.67]
srp_model = "FlatPlate"

{hardware_block}

[trajectory]
phases = ["Cruise"]
solver = "Hohmann"
departure_body = "Earth"

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
        )
    }

    #[test]
    fn custom_plate_round_trips_through_srp_force_and_torque() {
        let toml = mission_toml_with_hardware(
            r#"[[spacecraft.hardware]]
type = "CustomPlate"
normal = [0.0, 0.0, 1.0]
area_m2 = 4.0
center_offset_m = [0.0, 2.0, 0.0]
rho_s = 0.0
rho_d = 0.0
double_sided = false"#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should parse");
        let plates = custom_plates_from_hardware(&cfg);
        assert_eq!(plates.len(), 1);
        let plate = plates[0];
        assert!((plate.normal - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-12);
        assert_eq!(plate.area, 4.0);
        assert_eq!(plate.center_body, Vector3::new(0.0, 2.0, 0.0));

        // Sun straight along the plate normal, fully absorbing (rho_s=rho_d=0):
        // force should be -P*A*cos(theta)*sun_hat (anti-sun), torque = center x F.
        let sun_hat = Vector3::new(0.0, 0.0, 1.0);
        let p_srp = 1.0;
        let f = flat_plate_force_body(&[plate], &sun_hat, p_srp);
        let expected_f = Vector3::new(0.0, 0.0, -4.0); // -P*A*cos*sun_hat, cos=1, A=4
        assert!((f - expected_f).norm() < 1e-9, "force mismatch: {:?}", f);

        let tau = flat_plate_torque_body(&[plate], &sun_hat, p_srp);
        let expected_tau = plate.center_body.cross(&expected_f);
        assert!((tau - expected_tau).norm() < 1e-9, "torque mismatch: {:?}", tau);
    }

    #[test]
    fn custom_plate_defaults_when_optics_omitted() {
        let toml = mission_toml_with_hardware(
            r#"[[spacecraft.hardware]]
type = "CustomPlate"
normal = [1.0, 0.0, 0.0]
area_m2 = 1.5
center_offset_m = [0.5, 0.0, 0.0]"#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should parse with defaults");
        let plates = custom_plates_from_hardware(&cfg);
        assert_eq!(plates.len(), 1);
        assert_eq!(plates[0].rho_s, 0.30);
        assert_eq!(plates[0].rho_d, 0.20);
        assert!(!plates[0].double_sided);
    }

    #[test]
    fn build_plates_includes_bus_panels_and_custom_plate() {
        let toml = mission_toml_with_hardware(
            r#"[[spacecraft.hardware]]
type = "SolarPanel"
area_m2 = 8.0

[[spacecraft.hardware]]
type = "CustomPlate"
normal = [1.0, 0.0, 0.0]
area_m2 = 0.5
center_offset_m = [3.0, 0.0, 0.0]"#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should parse");
        let plates = build_plates(&cfg);
        // 6 automatic bus faces + 2 panel halves + 1 custom plate = 9
        assert_eq!(plates.len(), 9);
    }

    /// A PLACED SolarPanel (position_m + normal both set) must contribute
    /// its OWN plate via `placed_solar_panels_from_hardware`, and must NOT
    /// also feed the automatic symmetric +/-y pair `build_plates` derives
    /// from `panel_area_total` — the real double-counting risk this pair
    /// of functions was written to avoid (spacecraft-builder placement
    /// extension).
    #[test]
    fn build_plates_placed_solar_panel_contributes_own_plate_not_the_aggregate_pair() {
        let toml = mission_toml_with_hardware(
            r#"[[spacecraft.hardware]]
type = "SolarPanel"
area_m2 = 6.0
position_m = [0.0, 2.0, 0.0]
normal = [0.0, 1.0, 0.0]
width_m = 3.0
height_m = 2.0"#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should parse");
        let plates = build_plates(&cfg);
        // 6 automatic bus faces + 1 placed panel plate, NO automatic +/-y pair
        assert_eq!(plates.len(), 7, "placed panel should not also trigger the automatic pair: {plates:?}");
        let panel = plates.iter().find(|p| p.double_sided).expect("placed panel plate should be double-sided");
        // width_m * height_m = 6.0, overriding area_m2's own value (also 6.0
        // here, deliberately, so this assertion exercises the override path
        // rather than passing by coincidence against area_m2).
        assert!((panel.area - 6.0).abs() < 1e-9, "area should come from width_m*height_m: {}", panel.area);
        assert!((panel.center_body.y - 2.0).abs() < 1e-9, "plate center should come from position_m");
    }

    #[test]
    fn custom_plate_normal_is_renormalized() {
        let toml = mission_toml_with_hardware(
            r#"[[spacecraft.hardware]]
type = "CustomPlate"
normal = [0.0, 0.0, 3.0]
area_m2 = 1.0
center_offset_m = [0.0, 0.0, 0.0]"#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should parse");
        let plates = custom_plates_from_hardware(&cfg);
        assert!((plates[0].normal.norm() - 1.0).abs() < 1e-12, "normal should be unit length: {:?}", plates[0].normal);
    }

    /// Torque-arm re-centering: flag absent/false leaves `build_plates`
    /// byte-for-byte unchanged -- every plate's `center_body` stays
    /// geometric-center-relative, matching `custom_plate_round_trips_
    /// through_srp_force_and_torque`'s own direct assertion on the
    /// unshifted value.
    #[test]
    fn build_plates_keeps_geometric_center_origin_by_default() {
        let toml = mission_toml_with_hardware(
            r#"[[spacecraft.hardware]]
type = "CustomPlate"
normal = [1.0, 0.0, 0.0]
area_m2 = 0.5
center_offset_m = [3.0, 0.0, 0.0]
mass_kg = 400.0"#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should parse");
        assert!(cfg.spacecraft.derive_inertia_from_geometry.is_none());
        let plates = build_plates(&cfg);
        let custom = plates.iter().find(|p| (p.center_body.x - 3.0).abs() < 1e-9);
        assert!(custom.is_some(), "custom plate should be unshifted at x=3.0: {plates:?}");
    }

    /// Torque-arm re-centering: with the flag set, a heavy off-center
    /// `CustomPlate` pulls the true CoM away from the geometric-center
    /// origin, and every returned plate -- including the bus faces, whose
    /// config-declared position never itself changes -- must be
    /// re-expressed relative to that real CoM before the SRP torque law
    /// uses `center_body` as an `r x F` arm.
    #[test]
    fn build_plates_recenters_on_derived_com_when_flag_is_set() {
        let toml = mission_toml_with_hardware(
            r#"derive_inertia_from_geometry = true

[[spacecraft.hardware]]
type = "CustomPlate"
normal = [1.0, 0.0, 0.0]
area_m2 = 0.5
center_offset_m = [3.0, 0.0, 0.0]
mass_kg = 400.0"#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should parse");
        let vp = crate::vehicle_properties::compute_vehicle_properties(&cfg);
        assert!(vp.com_m[0].abs() > 1e-6, "a 400 kg plate 3 m off-axis on a 1000 kg bus should pull the CoM off zero: {:?}", vp.com_m);

        let plates = build_plates(&cfg);
        // The custom plate itself: config position (3.0) minus the derived CoM.
        let custom = plates
            .iter()
            .find(|p| (p.center_body.x - (3.0 - vp.com_m[0])).abs() < 1e-6)
            .unwrap_or_else(|| panic!("expected custom plate shifted by -com_m[0]={}: {plates:?}", vp.com_m[0]));
        assert!((custom.center_body.y - (0.0 - vp.com_m[1])).abs() < 1e-9);
        assert!((custom.center_body.z - (0.0 - vp.com_m[2])).abs() < 1e-9);

        // A bus face (+x, at lx/2 = 1.0 in the geometric-center frame) must
        // ALSO be shifted by the same -com_m, even though its own
        // config-declared position never changed -- proves the shift is
        // applied uniformly across every plate, not just the placed one.
        let bus_plus_x = plates
            .iter()
            .find(|p| (p.normal - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-9 && p.area == 2.0 * 0.63)
            .expect("expected the +x bus face plate");
        assert!((bus_plus_x.center_body.x - (1.0 - vp.com_m[0])).abs() < 1e-6, "bus +x face should also be recentered: {:?} vs expected {}", bus_plus_x.center_body, 1.0 - vp.com_m[0]);
    }

    /// Derived inertia: flag absent/false must be byte-for-byte the
    /// old behavior -- the manually-typed `inertia_diag_kgm2` stays
    /// authoritative regardless of any placed hardware.
    #[test]
    fn build_spacecraft_properties_uses_manual_inertia_by_default() {
        let toml = mission_toml_with_hardware(
            r#"[[spacecraft.hardware]]
type = "RcsThruster"
thrust_n = 5.0
position_m = [1.0, 0.0, 0.0]
direction = [0.0, 0.0, 1.0]
mass_kg = 50.0"#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should parse");
        assert!(cfg.spacecraft.derive_inertia_from_geometry.is_none());
        let sc = build_spacecraft_properties(&cfg);
        let manual = Vector3::new(366.67, 366.67, 666.67);
        assert!((sc.inertia_diag_kgm2 - manual).norm() < 1e-9, "got {:?}", sc.inertia_diag_kgm2);
    }

    /// Derived inertia: `derive_inertia_from_geometry = true` switches the live
    /// sim's inertia to the diagonal of `vehicle_properties::
    /// compute_vehicle_properties`'s real parallel-axis tensor -- proven by
    /// checking the result DIFFERS from the manually-typed value once real
    /// off-center mass is placed (a thruster 1 m off-axis has a real,
    /// non-trivial parallel-axis contribution) and MATCHES the independently
    /// computed derived value exactly.
    #[test]
    fn derive_inertia_from_geometry_true_uses_the_real_derived_diagonal() {
        let toml = mission_toml_with_hardware(
            r#"derive_inertia_from_geometry = true

[[spacecraft.hardware]]
type = "RcsThruster"
thrust_n = 5.0
position_m = [1.0, 0.0, 0.0]
direction = [0.0, 0.0, 1.0]
mass_kg = 50.0"#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should parse");
        assert_eq!(cfg.spacecraft.derive_inertia_from_geometry, Some(true));
        let sc = build_spacecraft_properties(&cfg);
        let manual = Vector3::new(366.67, 366.67, 666.67);
        assert!((sc.inertia_diag_kgm2 - manual).norm() > 1.0, "derived value should differ from the manual one, got {:?}", sc.inertia_diag_kgm2);

        let vp = crate::vehicle_properties::compute_vehicle_properties(&cfg);
        let expected = Vector3::new(vp.inertia_kgm2[0][0], vp.inertia_kgm2[1][1], vp.inertia_kgm2[2][2]);
        assert!((sc.inertia_diag_kgm2 - expected).norm() < 1e-9, "got {:?}, expected {:?}", sc.inertia_diag_kgm2, expected);
    }

    /// Placed thrusters: individually-placed `RcsThruster` entries reach the sim's
    /// real actuator model, exactly as placed (no smoothing into a
    /// symmetric-cluster approximation) -- and take precedence over an
    /// aggregate `RCS` entry present in the same config.
    #[test]
    fn rcs_from_hardware_uses_placed_thrusters_over_the_aggregate_variant() {
        let toml = mission_toml_with_hardware(
            r#"[[spacecraft.hardware]]
type = "RCS"
thrust_n = 1.0
count = 12
moment_arm_m = 0.5

[[spacecraft.hardware]]
type = "RcsThruster"
thrust_n = 3.0
position_m = [0.5, 0.2, -0.1]
direction = [0.0, 1.0, 0.0]

[[spacecraft.hardware]]
type = "RcsThruster"
thrust_n = 4.0
position_m = [-0.5, 0.2, -0.1]
direction = [0.0, -1.0, 0.0]"#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should parse");
        let (thrusters, _isp_s) = rcs_from_hardware(&cfg);
        assert_eq!(thrusters.len(), 2, "should use the 2 placed thrusters, not the aggregate RCS's 12");
        assert_eq!(thrusters[0].thrust_n, 3.0);
        assert!((thrusters[0].pos - Vector3::new(0.5, 0.2, -0.1)).norm() < 1e-12);
        assert!((thrusters[0].dir - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-12);
        assert_eq!(thrusters[1].thrust_n, 4.0);
    }

    /// Placed thrusters: a real, badly-asymmetric layout must reach the sim as-is --
    /// this is what makes an "attitude-stability preview" honest. A single
    /// thruster with no counterpart cannot produce a torque-only couple; its
    /// own torque/force are exactly what `Thruster::torque()`/`force()`
    /// compute from real geometry, not zeroed or corrected.
    #[test]
    fn rcs_from_hardware_reflects_a_genuinely_unbalanced_layout_honestly() {
        let toml = mission_toml_with_hardware(
            r#"[[spacecraft.hardware]]
type = "RcsThruster"
thrust_n = 10.0
position_m = [1.0, 0.0, 0.0]
direction = [0.0, 0.0, 1.0]"#,
        );
        let cfg: MissionConfig = toml::from_str(&toml).expect("should parse");
        let (thrusters, _isp_s) = rcs_from_hardware(&cfg);
        assert_eq!(thrusters.len(), 1);
        let t = &thrusters[0];
        // A single off-axis thruster produces a real net FORCE (not just a
        // torque) -- exactly the "this layout can't produce pure attitude
        // authority" failure the stability preview is meant to surface.
        assert!((t.force() - Vector3::new(0.0, 0.0, 10.0)).norm() < 1e-9);
        let expected_torque = Vector3::new(1.0, 0.0, 0.0).cross(&Vector3::new(0.0, 0.0, 10.0));
        assert!((t.torque() - expected_torque).norm() < 1e-9);
    }
}
