//! Trajectory design stage — dispatches to the solver chosen in `[trajectory].solver`.
//!
//! - `Hohmann`                — analytical sizing estimate (no ephemeris needed)
//! - `Lambert` / `GridSearch` — full departure × TOF porkchop scan via [`PorkchopGrid`]
//! - `LambertThenDiffCorrect` — porkchop scan, then Newton-refine the TOF at the
//!   best departure offset to exactly hit the s/c ΔV budget
//! - `DiffCorrection`         — multi-start Newton solve directly (no porkchop pre-scan):
//!   finds the fastest TOF that exactly uses the ΔV budget
//! - `MonteCarlo`             — Gaussian scatter of (departure offset, TOF) around the
//!   porkchop best, reporting ΔV dispersion statistics
//! - `GA` / `PSO`            — global/heuristic search over (departure offset, TOF)
//!   via [`GaSolver`]/[`PsoSolver`], verified by converging to the same point the
//!   `Lambert`/`GridSearch` porkchop scan finds independently (Phase 8j)
//! - `SA` / `ManifoldStitch` / `WSB` — not yet wired into the generic design
//!   stage (SA lives in `OptimizationProblems/examples`, ManifoldStitch/WSB in
//!   `AstroProbs/LunarTrajectories`); reported as not-yet-implemented.
//!
//! Body state sources:
//! - Departure body (`[trajectory].departure_body`, defaults to Earth, always
//!   ANISE — see `resolve_departure_body`): `Almanac::body_state_heliocentric`
//! - Arrival body, `ephemeris = "Anise"`:     same almanac call (Mars, Venus, ...)
//! - Arrival body, `ephemeris = "Keplerian"`: Keplerian propagation from TOML elements
//!
//! `crates/trajectory_solver` stays ephemeris-free — all ANISE/Keplerian body-state
//! lookups live here, and solvers never write files themselves; this module ranks
//! and writes all CSV output.

use std::fs;

use ephemeris::{Almanac, Body, Epoch};
use orbital_models::constants::G0;
use trajectory_solver::{
    keplerian::{KeplerianElements, AU_M, MU_SUN_M3S2},
    propagate, DiffCorrectionResult, DiffCorrectionSolver, GaSolver, HohmannSolver,
    LambertArc, MonteCarloSample, MonteCarloSolver, PorkchopGrid, PorkchopPoint,
    PsoSolver, TrajectorySolution,
};

/// Target sample spacing for a visualized transfer arc — ~60 points across
/// the leg, matching the previous analytic-resampling density. Phase 7 v1
/// scope: point-mass only, no perturbers (`&[]`) — see the design notes
/// Propagator Design — SOI-Patched Multi-Body".
const ARC_SAMPLE_COUNT: f64 = 60.0;
/// Dopri5 tolerances for arc visualization propagation — matches the WSB
/// pipeline's standard tolerances), looser than the cross-checked
/// unit-test tolerances since this is for display, not targeting.
const PROPAGATOR_RTOL: f64 = 1e-8;
/// `1e-10` m (the value used before Phase 8h) is only meaningful at
/// AU-scale (~1e11 m) interplanetary positions, where `rtol`'s
/// `rtol·|y| ≈ 1e-8 · 1e11 = 1e3` term already dwarfs it — atol was
/// practically inert for every Mars/Jupiter/asteroid case verified so far.
/// It becomes a real problem once a Moon/Europa/Titan/Phobos/Deimos-scale
/// central body engages (positions ~1e6-1e8 m): a real `StepSizeUnderflow`
/// was found running a Moon-target case through this exact constant.
/// `1e-3` m is still far tighter than this propagator needs anywhere and
/// is achievable across both scales — confirmed no change in the Mars
/// case's results (expected, since atol was already irrelevant there).
const PROPAGATOR_ATOL: f64 = 1e-3;
/// Per-trajectory point density for Monte Carlo's sparse trajectory output
/// (Phase 8i, absorbs 7h) — deliberately much sparser than `ARC_SAMPLE_COUNT`:
/// payload scales with sample count, and a dispersion cloud needs many
/// trajectories at once, not one finely-sampled one. 18 sits in the design notes
/// specified 15-20 point range.
const MC_TRAJ_SAMPLE_COUNT: f64 = 18.0;
/// How many of the Monte Carlo samples get a propagated trajectory, out of
/// `run_monte_carlo`'s full `N_SAMPLES` scatter — a representative subset
/// for the dispersion-cloud plot, not every sample (same "sparse, not dense"
/// rationale as `MC_TRAJ_SAMPLE_COUNT`).
const MC_TRAJ_COUNT: usize = 30;
/// Full Monte Carlo scatter size — shared by the CLI path (`run_monte_carlo`)
/// and the API path (`monte_carlo_api`, Phase 7j) so both report statistics
/// over the same sample count.
const MC_N_SAMPLES: usize = 500;

use crate::config::{DepartureMode, EphemerisSource, MissionConfig, MissionObjective, TrajectorySolver};

// ── Entry point ──────────────────────────────────────────────────────────────

pub fn run(cfg: &MissionConfig) {
    println!("\nTrajectory design: {}", cfg.mission.name);
    println!("Target body:  {}  ({})", cfg.target_body.name, cfg.target_body.ephemeris);
    println!("Solver:       {}\n", cfg.trajectory.solver);

    // Hohmann is purely analytical — skip ephemeris entirely
    if let TrajectorySolver::Hohmann = cfg.trajectory.solver {
        run_hohmann(cfg);
        return;
    }

    // All other solvers need Earth's (and the arrival body's) state from ANISE/Keplerian
    let almanac = match load_almanac() {
        Some(a) => a,
        None    => std::process::exit(1),
    };

    let kep = keplerian_from_cfg(cfg);
    if matches!(cfg.target_body.ephemeris, EphemerisSource::Keplerian) && kep.is_none() {
        eprintln!(
            "Error: ephemeris = \"Keplerian\" requires a [target_body.keplerian_orbit] \
             section in the TOML."
        );
        std::process::exit(1);
    }

    match cfg.trajectory.solver {
        TrajectorySolver::Hohmann => unreachable!("handled above"),

        TrajectorySolver::Lambert | TrajectorySolver::GridSearch => {
            let (points, dep_jd_base) = run_porkchop_scan(cfg, &almanac, &kep);
            report_porkchop_best(cfg, &points, dep_jd_base, &almanac, &kep);
        }

        TrajectorySolver::LambertThenDiffCorrect => {
            let (points, dep_jd_base) = run_porkchop_scan(cfg, &almanac, &kep);
            let best = report_porkchop_best(cfg, &points, dep_jd_base, &almanac, &kep);
            run_diff_correct_refine(cfg, &almanac, &kep, dep_jd_base, best);
        }

        TrajectorySolver::DiffCorrection => {
            run_diff_correct_standalone(cfg, &almanac, &kep);
        }

        TrajectorySolver::MonteCarlo => {
            let (points, dep_jd_base) = run_porkchop_scan(cfg, &almanac, &kep);
            let best = report_porkchop_best(cfg, &points, dep_jd_base, &almanac, &kep);
            run_monte_carlo(cfg, &almanac, &kep, dep_jd_base, best);
        }

        TrajectorySolver::GA => {
            run_ga(cfg, &almanac, &kep);
        }

        TrajectorySolver::PSO => {
            run_pso(cfg, &almanac, &kep);
        }

        TrajectorySolver::SA | TrajectorySolver::ManifoldStitch | TrajectorySolver::WSB => {
            eprintln!(
                "Solver '{}' is not yet implemented in the generic MissionPlanner design stage.\n\
                 SA lives in OptimizationProblems/examples; ManifoldStitch/WSB live in \
                 AstroProbs/LunarTrajectories.",
                cfg.trajectory.solver
            );
            std::process::exit(1);
        }
    }
}

// ── Hohmann sizing ───────────────────────────────────────────────────────────

fn run_hohmann(cfg: &MissionConfig) {
    let r2_m = match cfg.target_body.ephemeris {
        EphemerisSource::Keplerian => cfg
            .target_body
            .keplerian_orbit
            .as_ref()
            .map(|k| k.sma_au * AU_M)
            .unwrap_or(AU_M),
        EphemerisSource::Anise => match cfg.target_body.name.to_lowercase().as_str() {
            "mars"    => 1.524  * AU_M,
            "venus"   => 0.7233 * AU_M,
            "jupiter" => 5.203  * AU_M,
            "saturn"  => 9.537  * AU_M,
            _         => AU_M,
        },
        EphemerisSource::Custom => AU_M,
    };

    // Only print interplanetary cruise Hohmann when the target is not the Earth itself
    // and the heliocentric distance makes sense (r2 meaningfully ≠ r1).
    let r1_m = AU_M;
    if (r2_m - r1_m).abs() > 1e9 {
        let solver = HohmannSolver { mu_m3s2: MU_SUN_M3S2, r1_m, r2_m };
        match solver.solve() {
            Ok(s) => {
                println!("Interplanetary cruise Hohmann  (heliocentric, circular/coplanar)");
                println!("  r1 = 1.000 AU  →  r2 = {:.4} AU", r2_m / AU_M);
                println!("  TOF:       {:.1} days", s.tof_s / 86_400.0);
                println!("  ΔV dep:    {:.0} m/s  ({:.3} km/s)", s.dv_departure_ms, s.dv_departure_ms / 1e3);
                println!("  ΔV arr:    {:.0} m/s  ({:.3} km/s)", s.dv_arrival_ms,   s.dv_arrival_ms   / 1e3);
                println!("  ΔV total:  {:.0} m/s  ({:.3} km/s)", s.dv_total_ms,     s.dv_total_ms     / 1e3);
                println!("  C3:        {:.3} km²/s²", s.c3_km2s2);
            }
            Err(e) => eprintln!("Hohmann solve failed: {e}"),
        }
    }

    // Body-centric orbit sizing — uses the target body's own μ.
    // Active whenever [trajectory.capture] specifies target_orbit_radius_m.
    // The parking orbit is estimated from the body atmosphere:
    //   atmospheric bodies  → body_radius + 200 km (safe above most drag)
    //   airless bodies      → 1.5 × r_cap (moderate approach orbit)
    if let Some(cap) = &cfg.trajectory.capture {
        if let Some(r_cap) = cap.target_orbit_radius_m {
            let has_atm = !matches!(cfg.target_body.atmosphere, crate::config::AtmosphereModel::None);
            let r_park = if has_atm {
                cfg.target_body.radius_m + 200_000.0
            } else {
                r_cap * 1.5
            };
            let bdy_solver = HohmannSolver {
                mu_m3s2: cfg.target_body.mu_m3s2,
                r1_m:    r_park,
                r2_m:    r_cap,
            };
            match bdy_solver.solve() {
                Ok(s) => {
                    println!("\nBody-centric orbit sizing  (μ = {} = {:.4e} m³/s²)", cfg.target_body.name, cfg.target_body.mu_m3s2);
                    println!("  r_park = {:10.1} km  ({})",
                        r_park / 1e3,
                        if has_atm { "body radius + 200 km" } else { "1.5 × target orbit" },
                    );
                    println!("  r_cap  = {:10.1} km  (target orbit)", r_cap / 1e3);
                    println!("  TOF:       {:.1} min", s.tof_s / 60.0);
                    println!("  ΔV dep:    {:.0} m/s", s.dv_departure_ms);
                    println!("  ΔV arr:    {:.0} m/s", s.dv_arrival_ms);
                    println!("  ΔV total:  {:.0} m/s  ({:.3} km/s)", s.dv_total_ms, s.dv_total_ms / 1e3);
                }
                Err(e) => eprintln!("Body-centric Hohmann failed: {e}"),
            }
        }
    }
}

// ── Porkchop scan (Lambert / GridSearch / seed for LambertThenDiffCorrect / MonteCarlo) ───

/// Run the departure-offset × TOF porkchop scan from the TOML `[trajectory.cruise]`
/// parameters. Exits the process if no valid Lambert solutions are found anywhere
/// on the grid (bad TOF range, body unavailable, or ephemeris gap).
fn run_porkchop_scan(
    cfg: &MissionConfig,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
) -> (Vec<PorkchopPoint>, f64) {
    let cruise     = cfg.trajectory.cruise.as_ref();
    let tof_min    = cruise.and_then(|c| c.tof_days_min).unwrap_or(100.0);
    let tof_max    = cruise.and_then(|c| c.tof_days_max).unwrap_or(400.0);
    let n_grid     = cruise.and_then(|c| c.grid_resolution).unwrap_or(30) as usize;
    let dep_window = cruise.and_then(|c| c.departure_window_days).unwrap_or(60.0);

    let dep_epoch = match require_departure_epoch(cfg) {
        Ok(e) => e,
        Err(msg) => {
            eprintln!("Error: {msg}");
            std::process::exit(1);
        }
    };
    let dep_jd_base = epoch_to_jd(dep_epoch);

    let n_dep = n_grid.max(2);
    let n_tof = n_grid.max(2);
    let dep_offsets: Vec<f64> = (0..n_dep)
        .map(|i| -dep_window / 2.0 + i as f64 * dep_window / (n_dep - 1) as f64)
        .collect();
    let tof_grid: Vec<f64> = (0..n_tof)
        .map(|i| tof_min + i as f64 * (tof_max - tof_min) / (n_tof - 1) as f64)
        .collect();

    println!(
        "Departure window: {:+.0} to {:+.0} days around {dep_epoch}",
        dep_offsets.first().copied().unwrap_or(0.0),
        dep_offsets.last().copied().unwrap_or(0.0),
    );
    println!("TOF range:        {tof_min:.0} – {tof_max:.0} days");
    println!("Grid:             {n_dep} × {n_tof} = {} Lambert evaluations", n_dep * n_tof);

    let body_name = cfg.target_body.name.to_lowercase();
    let ephemeris  = cfg.target_body.ephemeris;
    let body = resolve_body(ephemeris, &body_name);
    let dep_body = resolve_departure_body(cfg);

    let grid = PorkchopGrid {
        mu:               MU_SUN_M3S2,
        dep_offsets_days: dep_offsets,
        tof_days_grid:    tof_grid,
    };

    let points = grid.evaluate(|dep_off, tof_d| {
        transfer_states(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_off, tof_d)
    });

    if points.is_empty() {
        eprintln!(
            "\nNo valid Lambert solutions found.\n\
             Check: TOF range, body availability, and whether de440s.bsp covers the departure epoch."
        );
        std::process::exit(1);
    }

    (points, dep_jd_base)
}

/// Picks the porkchop grid's "best" point for the mission's objective.
///
/// `Flyby` never burns to arrive — minimizing `dv_total_ms` (departure C3
/// plus arrival v∞, summed as if the v∞ were a real capture burn) biases
/// the search toward slow, low-arrival-speed transfers no flyby mission
/// would actually fly. Flyby instead minimizes `c3_km2s2` alone, the real
/// cost driver (launch vehicle performance / injected mass). Objectives
/// that do burn on arrival (`Orbit`, `Landing`, `Rendezvous`,
/// `SampleReturn`) keep minimizing `dv_total_ms` — correct there, since the
/// arrival burn is real.
fn best_point_for_objective(
    points: &[PorkchopPoint],
    objective: MissionObjective,
) -> Option<&PorkchopPoint> {
    match objective {
        MissionObjective::Flyby => points
            .iter()
            .min_by(|a, b| a.c3_km2s2.partial_cmp(&b.c3_km2s2).unwrap()),
        _ => PorkchopGrid::best(points),
    }
}

/// Objective-aware arrival ΔV for the narrowing-stage (Lambert/porkchop/
/// DiffCorrection) API results (frontend backlog items #14/#15,
///). `PorkchopGrid`/`LambertArc` (crate-level, objective-agnostic
/// per the design notes "Shared Crates Rules" — no application-specific logic
/// belongs there) always price the raw arrival v∞ as if it were itself a
/// real capture burn — correct only for objectives that actually do burn a
/// full v∞ match on arrival. This corrects that at the API boundary, one
/// mission objective at a time:
///
/// - `Flyby` never burns to arrive at all — always `0.0`, matching
///   `OptimizeApiResult.dv_arrival_ms`'s existing convention on the Phase 9
///   real-dynamics path (`optimize.rs`), which already gets this right.
/// - `Orbit` / `Landing` burn a real vis-viva capture,
///   `√(v∞² + 2μ/r_p) − √(μ(1+e)/r_p)`, with `r_p` floor-clamped to the
///   target body's own physical radius (Phase 9y) — the exact formula and
///   floor `mga_scan_run.rs::capture_dv_ms` and `mga.rs::arrival_dv_ms`
///   already use, applied here so the narrowing stage and MGA are finally
///   comparable. Only computed when `[trajectory.capture].target_orbit_radius_m`
///   is actually configured; DiffCorrection/porkchop callers have never
///   required that section to run at all, so falls back to the raw v∞
///   (the pre-fix behaviour) rather than fail outright when it's absent.
/// - `Rendezvous` never burns a periapsis-radius capture — its arrival
///   burn is deliberately the full relative-velocity match (see
///   `config.rs::check_config`'s `needs_capture_radius` comment) — raw v∞
///   unchanged.
/// - `SampleReturn` has no dedicated narrowing-stage handling to correct
///   here — raw v∞ unchanged.
fn arrival_dv_for_objective_ms(cfg: &MissionConfig, raw_v_inf_arr_ms: f64) -> f64 {
    match cfg.mission.objective {
        MissionObjective::Flyby => 0.0,
        MissionObjective::Orbit | MissionObjective::Landing => {
            match cfg.trajectory.capture.as_ref().and_then(|cap| cap.target_orbit_radius_m) {
                Some(r_cap_configured) => {
                    let cap = cfg.trajectory.capture.as_ref().unwrap();
                    // Phase 9y floor-clamp — never a vis-viva burn for an
                    // "orbit" inside the target body's own surface.
                    let r_cap = r_cap_configured.max(cfg.target_body.radius_m);
                    let mu = cfg.target_body.mu_m3s2;
                    let v_peri = (mu * (1.0 + cap.capture_eccentricity) / r_cap).sqrt();
                    let v_hyp = (raw_v_inf_arr_ms * raw_v_inf_arr_ms + 2.0 * mu / r_cap).sqrt();
                    v_hyp - v_peri
                }
                None => raw_v_inf_arr_ms,
            }
        }
        MissionObjective::Rendezvous | MissionObjective::SampleReturn => raw_v_inf_arr_ms,
    }
}

/// Applies [`arrival_dv_for_objective_ms`] to a raw `(dv_dep_ms, v_inf_arr_ms)`
/// pair and returns `(dv_arr_ms, dv_total_ms)` — the two API fields that
/// need objective-aware repricing. `v_inf_arr_ms` itself (a physical
/// quantity, not a ΔV) is never touched.
fn objective_priced_arrival(cfg: &MissionConfig, dv_dep_ms: f64, raw_v_inf_arr_ms: f64) -> (f64, f64) {
    let dv_arr_ms = arrival_dv_for_objective_ms(cfg, raw_v_inf_arr_ms);
    (dv_arr_ms, dv_dep_ms + dv_arr_ms)
}

/// Print and write the porkchop best-arc outputs; returns the best point so
/// callers (`LambertThenDiffCorrect`, `MonteCarlo`) can seed further work from it.
fn report_porkchop_best<'a>(
    cfg: &MissionConfig,
    points: &'a [PorkchopPoint],
    dep_jd_base: f64,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
) -> &'a PorkchopPoint {
    let best = best_point_for_objective(points, cfg.mission.objective).unwrap();
    print_best(cfg, best, dep_jd_base);
    write_outputs(cfg, points, best, dep_jd_base);
    write_best_arc_trajectory(cfg, best, dep_jd_base, almanac, kep);
    write_porkchop_sample_trajectories(cfg, points, dep_jd_base, almanac, kep);
    best
}

// ── Result printing ──────────────────────────────────────────────────────────

fn print_best(cfg: &MissionConfig, best: &PorkchopPoint, dep_jd_base: f64) {
    let dep_jd = dep_jd_base + best.dep_offset_days;
    let is_flyby = matches!(cfg.mission.objective, MissionObjective::Flyby);

    println!("\n────────────────────────────────────────────────────────────────");
    if is_flyby {
        println!("Best arc  (minimum C3 — Flyby has no arrival burn)");
    } else {
        println!("Best arc  (minimum total ΔV from porkchop grid points)");
    }
    println!("────────────────────────────────────────────────────────────────");
    println!("  Departure offset:  {:+.2} days from reference epoch", best.dep_offset_days);
    println!("  Departure JD:      {dep_jd:.4}  (≈ {})", epoch_approx_str(dep_jd));
    println!("  Time of flight:    {:.2} days  ({:.2} months)", best.tof_days, best.tof_days / 30.4375);
    println!("  Arrival JD:        {:.4}", dep_jd + best.tof_days);
    println!();
    println!("  C3 at departure:   {:.4} km²/s²", best.c3_km2s2);
    println!("  v∞ at arrival:     {:.2} m/s  ({:.4} km/s)", best.v_inf_arr_ms, best.v_inf_arr_ms / 1e3);
    println!("  ΔV departure:      {:.2} m/s  ({:.4} km/s)", best.dv_dep_ms,    best.dv_dep_ms    / 1e3);
    if is_flyby {
        println!("  v∞ at arrival is a flyby speed, not a burn — no capture ΔV is charged.");
    } else {
        println!("  ΔV arrival (cap):  {:.2} m/s  ({:.4} km/s)", best.dv_arr_ms,    best.dv_arr_ms    / 1e3);
        println!("  ΔV total:          {:.2} m/s  ({:.4} km/s)", best.dv_total_ms,  best.dv_total_ms  / 1e3);
    }

    if let (Some(prop), Some(budget)) = (cfg.spacecraft.propulsion.as_ref(), dv_budget_ms(cfg)) {
        println!();
        println!("  s/c ΔV budget:     {budget:.2} m/s  (Tsiolkovsky, Isp={:.0} s)", prop.isp_s);
        let dep_name = cfg.trajectory.departure_body.as_str();
        if let Some((_, r_park)) = departure_escape_dv_ms(cfg, dep_name, best.dv_dep_ms) {
            let configured = cfg.trajectory.departure.as_ref().and_then(|d| d.parking_orbit_radius_m).is_some();
            println!(
                "  Parking orbit:     {dep_name} at {:.1} km radius ({})",
                r_park / 1e3,
                if configured { "configured" } else { "default heuristic" },
            );
        }
        match narrowing_dv_ledger(cfg, dep_name, best.dv_dep_ms, best.v_inf_arr_ms) {
            Some(ledger) => print_dv_ledger(&ledger),
            None => eprintln!(
                "  Warning: could not compute onboard ΔV required — departure body '{dep_name}' not in body_models catalog.",
            ),
        }
    }
    print_launch_vehicle_check(cfg, best.c3_km2s2);
}

// ── Differential correction (DiffCorrection / LambertThenDiffCorrect) ─────────

/// Onboard ΔV available from the Tsiolkovsky rocket equation,
/// `Isp·g₀·ln(m_wet/m_dry)` (MANUAL.md §13.5), or `None` if no propulsion
/// is configured. Single source of truth — used by the ΔV ledger's margin
/// and as the DiffCorrection target residual.
pub(crate) fn dv_budget_ms(cfg: &MissionConfig) -> Option<f64> {
    let prop = cfg.spacecraft.propulsion.as_ref()?;
    Some(prop.isp_s * G0 * (cfg.spacecraft.mass_kg / cfg.spacecraft.dry_mass_kg).ln())
}

/// Onboard escape-burn ΔV from a circular parking orbit at a non-Earth
/// departure body, via the energy/vis-viva relation (Curtis §8, Vallado
/// §10): `ΔV = v_hyperbolic(r_park) − v_circular(r_park)`. Parking orbit
/// radius is `[trajectory.departure].parking_orbit_radius_m` when set
/// (Phase 8e), otherwise the same atmospheric/airless heuristic
/// `run_hohmann`/`hohmann_api` already use on the arrival side
/// (`body_radius + 200 km` if atmospheric, `1.5 × body_radius` if airless).
/// Returns `(ΔV, r_park used)`; `None` if the body isn't in the
/// `body_models` catalog.
pub(crate) fn departure_escape_dv_ms(cfg: &MissionConfig, dep_body_name: &str, v_inf_ms: f64) -> Option<(f64, f64)> {
    let body = body_models::TargetBody::by_name(dep_body_name)?;
    let r_park = resolve_parking_orbit_radius_m(cfg, &body);
    let v_circ = (body.mu_m3s2 / r_park).sqrt();
    let v_hyp = (v_inf_ms * v_inf_ms + 2.0 * body.mu_m3s2 / r_park).sqrt();
    Some((v_hyp - v_circ, r_park))
}

/// Resolves the parking orbit radius a departure escape burn (or, for the
/// Phase 9 optimizer, a real numerically-propagated escape leg) is sized
/// from: `[trajectory.departure].parking_orbit_radius_m` when set (Phase
/// 8e), otherwise the same atmospheric/airless heuristic
/// `run_hohmann`/`hohmann_api` already use on the arrival side
/// (`body_radius + 200 km` if atmospheric, `1.5 × body_radius` if airless).
/// Shared by `departure_escape_dv_ms` (narrowing stage, analytic) and
/// `optimize.rs` (Phase 9, real propagation) so both use the exact same
/// parking-orbit convention.
pub(crate) fn resolve_parking_orbit_radius_m(cfg: &MissionConfig, body: &body_models::TargetBody) -> f64 {
    let dep = cfg.trajectory.departure.as_ref();
    dep.and_then(|d| d.parking_orbit_radius_m).unwrap_or_else(|| {
        if departure_mode(cfg) == DepartureMode::Launch {
            // Launch mode (Phase 14a): the launcher's ascent reaches a
            // parking orbit at `parking_altitude_m` (default 185 km, the
            // customary ~100 n.mi. injection altitude) before the upper-
            // stage burn.
            return body.radius_m + dep.and_then(|d| d.parking_altitude_m).unwrap_or(LAUNCH_PARKING_ALTITUDE_DEFAULT_M);
        }
        let has_atm = !matches!(body.atmosphere, body_models::AtmosphereModel::None);
        if has_atm {
            body.radius_m + 200_000.0
        } else {
            body.radius_m * 1.5
        }
    })
}

/// Default `Launch`-mode parking-orbit altitude [m] — 185 km (~100 n.mi.),
/// the customary direct-ascent injection altitude (Sergeyevsky et al., JPL
/// 82-43 quote C3 curves for a 185 km parking orbit).
pub(crate) const LAUNCH_PARKING_ALTITUDE_DEFAULT_M: f64 = 185_000.0;

/// `[trajectory.departure].mode`, `ParkingOrbit` when the section is absent.
pub(crate) fn departure_mode(cfg: &MissionConfig) -> DepartureMode {
    cfg.trajectory.departure.as_ref().map(|d| d.mode).unwrap_or_default()
}

/// The departure ΔV the SPACECRAFT pays for a departure asymptote of excess
/// speed `v_inf_ms` — the pool-aware departure cost a search fitness should
/// charge (Phase 14b):
/// - `ParkingOrbit` mode: the full tangential escape burn
///   (`departure_escape_dv_ms`), whoever ends up paying it in the ledger.
/// - `Launch` mode: only the perigee top-up beyond what the launch vehicle
///   delivers at this wet mass (`LaunchVehicleCheckApiResult::
///   onboard_departure_dv_ms` — zero when the launcher covers the C3, the
///   full escape burn when it cannot lift the mass at all). This is what
///   makes a `Launch`-mode search prefer asymptotes the launcher can
///   actually reach, graded rather than gated.
/// `None` only if the departure body isn't in the `body_models` catalog.
pub(crate) fn departure_onboard_cost_ms(cfg: &MissionConfig, dep_name: &str, v_inf_ms: f64) -> Option<f64> {
    let (full_burn, _) = departure_escape_dv_ms(cfg, dep_name, v_inf_ms)?;
    if departure_mode(cfg) != DepartureMode::Launch {
        return Some(full_burn);
    }
    let c3_km2s2 = (v_inf_ms / 1_000.0).powi(2);
    Some(compute_launch_vehicle_check_for(cfg, dep_name, c3_km2s2)
        .map(|c| c.onboard_departure_dv_ms)
        .unwrap_or(full_burn))
}

/// Body equatorial frame for RLA/DLA and the launch plane — `None` when the
/// body has no catalog pole (which `check_config` already rejects for
/// `Launch` mode).
pub(crate) fn launch_frame(body: &body_models::TargetBody) -> Option<trajectory_solver::EquatorialFrame> {
    let (ra, dec) = (body.pole_ra_deg?, body.pole_dec_deg?);
    Some(trajectory_solver::EquatorialFrame::from_pole(ra.to_radians(), dec.to_radians()))
}

/// Closed-form launch geometry for a departure asymptote in `Launch` mode
/// (Phase 14c wiring): site latitude and optional explicit inclination from
/// `[trajectory.departure]`, parking radius from `resolve_parking_orbit_
/// radius_m`, frame from the body's catalog pole. `None` outside `Launch`
/// mode, without a site, without a pole, or for a zero `v∞`.
pub(crate) fn launch_geometry_for(
    cfg: &MissionConfig,
    body: &body_models::TargetBody,
    v_inf_vec_ms: nalgebra::Vector3<f64>,
) -> Option<trajectory_solver::LaunchGeometry> {
    if departure_mode(cfg) != DepartureMode::Launch {
        return None;
    }
    let dep = cfg.trajectory.departure.as_ref()?;
    let site = dep.launch_site.as_ref()?;
    let frame = launch_frame(body)?;
    trajectory_solver::launch_geometry(
        body.mu_m3s2,
        resolve_parking_orbit_radius_m(cfg, body),
        v_inf_vec_ms,
        &frame,
        site.lat_deg.to_radians(),
        dep.inclination_deg.map(f64::to_radians),
        false,
    )
}

/// API shape of a [`trajectory_solver::LaunchGeometry`] (Phase 14d):
/// angles in degrees, vectors body-relative inertial [m, m/s].
#[derive(Debug, Clone, serde::Serialize)]
pub struct LaunchGeometryApiResult {
    pub site_name: String,
    pub site_lat_deg: f64,
    pub site_lon_deg: f64,
    pub c3_km2s2: f64,
    pub rla_deg: f64,
    pub dla_deg: f64,
    pub inclination_deg: f64,
    pub min_inclination_deg: f64,
    pub feasible_no_dogleg: bool,
    pub launch_azimuth_deg: f64,
    pub raan_deg: f64,
    pub coast_angle_deg: f64,
    pub parking_orbit_radius_m: f64,
    pub injection_r_m: [f64; 3],
    pub injection_v_mps: [f64; 3],
    pub parking_v_mps: [f64; 3],
    pub dv_injection_ms: f64,
}

pub(crate) fn launch_geometry_api(cfg: &MissionConfig, g: &trajectory_solver::LaunchGeometry) -> LaunchGeometryApiResult {
    let site = cfg.trajectory.departure.as_ref().and_then(|d| d.launch_site.as_ref());
    LaunchGeometryApiResult {
        site_name: site.map(|s| s.name.clone()).unwrap_or_default(),
        site_lat_deg: site.map(|s| s.lat_deg).unwrap_or(0.0),
        site_lon_deg: site.map(|s| s.lon_deg).unwrap_or(0.0),
        c3_km2s2: g.c3_km2s2,
        rla_deg: g.rla_rad.to_degrees(),
        dla_deg: g.dla_rad.to_degrees(),
        inclination_deg: g.inclination_rad.to_degrees(),
        min_inclination_deg: g.min_inclination_rad.to_degrees(),
        feasible_no_dogleg: g.feasible_no_dogleg,
        launch_azimuth_deg: g.launch_azimuth_rad.to_degrees(),
        raan_deg: g.raan_rad.to_degrees(),
        coast_angle_deg: g.coast_angle_rad.to_degrees(),
        parking_orbit_radius_m: g.injection_r_m.norm(),
        injection_r_m: [g.injection_r_m.x, g.injection_r_m.y, g.injection_r_m.z],
        injection_v_mps: [g.injection_v_mps.x, g.injection_v_mps.y, g.injection_v_mps.z],
        parking_v_mps: [g.parking_v_mps.x, g.parking_v_mps.y, g.parking_v_mps.z],
        dv_injection_ms: g.dv_injection_ms,
    }
}

/// Two-pool ΔV ledger for one trajectory (Phase 14e;
/// MANUAL.md §13.5). Every ΔV a mission needs is paid from one of two
/// pools, and the same trajectory can be flyable from one and unflyable from
/// the other:
/// - the **launcher pool** — the launch vehicle's upper stage delivers (part
///   of) the departure injection, priced against its verified C3-vs-mass
///   curve (`compute_launch_vehicle_check`). Only ever non-zero for an Earth
///   departure with a catalog vehicle configured.
/// - the **onboard pool** — the spacecraft's own tank, `Isp·g₀·ln(m_wet/
///   m_dry)`, which pays for whatever the launcher does not: the departure
///   remainder (all of it for a non-Earth departure, or when no vehicle is
///   configured), every deep-space maneuver, and ALWAYS the arrival burn.
///
/// Partial launcher coverage: a vehicle that cannot
/// reach the required C3 still gives its maximum at this mass — see
/// `LaunchVehicleCheckApiResult::max_c3_at_mass_km2s2` — and the spacecraft
/// tops up from the same periapsis. Here `departure_dv_ms` is the FULL
/// departure burn the caller's own model charged (the tangential
/// parking-orbit escape burn for the narrowing stage and MGA; the search's
/// own, generally non-tangential, `dv_departure_ms` for the single-leg
/// GA/PSO), and it is split as: launcher fully covers the C3 → the whole
/// burn is the launcher's; partial → the launcher's tangential share from
/// the check, remainder onboard; no launcher → all onboard.
///
/// Distinct from `dv_total_ms`-style fields, which report the trajectory's
/// physics (departure + arrival ΔV) regardless of who pays for which piece.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DvLedgerApiResult {
    /// Full departure burn [m/s] from the parking orbit, before any split.
    pub departure_dv_ms: f64,
    /// Who pays the departure burn: `"launcher"` (fully covered),
    /// `"onboard"` (no launcher share at all), or `"split"` (the launcher
    /// delivers its maximum C3 at this mass and the spacecraft tops up).
    pub departure_dv_pool: String,
    /// Departure ΔV [m/s] delivered by the launch vehicle — `0.0` without one.
    pub launcher_dv_ms: f64,
    /// Departure ΔV [m/s] the spacecraft's own engine must add.
    pub onboard_departure_dv_ms: f64,
    /// Arrival/capture burn [m/s] — always onboard; `0.0` for a Flyby.
    pub onboard_arrival_dv_ms: f64,
    /// Deep-space maneuvers [m/s] (MGA DSM sum) — always onboard; `0.0` for
    /// a single-leg transfer.
    pub onboard_dsm_dv_ms: f64,
    /// `onboard_departure_dv_ms + onboard_dsm_dv_ms + onboard_arrival_dv_ms`.
    pub onboard_dv_required_ms: f64,
    /// `Isp·g₀·ln(m_wet/m_dry)` [m/s] — `None` when no `[spacecraft.
    /// propulsion]` is configured (the three fields below are then `None`
    /// too: without an Isp the rocket equation has nothing to evaluate).
    pub onboard_dv_available_ms: Option<f64>,
    /// Propellant the onboard requirement consumes, `m_wet·(1 − e^{−Δv/(Isp·g₀)})`
    /// [kg] — compare with `[spacecraft].propellant_mass_kg`.
    pub onboard_propellant_required_kg: Option<f64>,
    /// `m_wet/m_dry` the onboard requirement demands, `e^{Δv/(Isp·g₀)}` —
    /// a scale-free feasibility figure (chemical stages top out around 10–15).
    pub mass_ratio_required: Option<f64>,
    /// `onboard_dv_available_ms >= onboard_dv_required_ms`.
    pub propellant_feasible: Option<bool>,
    /// `onboard_dv_available_ms − onboard_dv_required_ms` [m/s] — positive =
    /// onboard margin remaining. The same number the result types' legacy
    /// `budget_margin_ms` field carries.
    pub budget_margin_ms: Option<f64>,
}

/// Builds the [`DvLedgerApiResult`] for a trajectory whose departure burn,
/// departure C3, arrival burn and DSM sum the caller has already priced with
/// its own model — the ONE place the launcher/onboard split and the rocket-
/// equation feasibility numbers are computed, for every result type.
/// `arrival_dv_ms` must already be the objective-priced burn (see
/// `arrival_dv_for_objective_ms`); a `Flyby` is forced to `0.0` here anyway
/// as a safety net, since Flyby never burns to arrive.
pub(crate) fn dv_ledger(
    cfg: &MissionConfig,
    dep_name: &str,
    departure_dv_ms: f64,
    c3_km2s2: f64,
    arrival_dv_ms: f64,
    dsm_dv_ms: f64,
) -> DvLedgerApiResult {
    let onboard_arrival_dv_ms = if matches!(cfg.mission.objective, MissionObjective::Flyby) { 0.0 } else { arrival_dv_ms };
    let check = compute_launch_vehicle_check_for(cfg, dep_name, c3_km2s2);
    let (launcher_dv_ms, onboard_departure_dv_ms) = match &check {
        // Launcher reaches the required C3 at this mass: the whole injection
        // is the upper stage's, however the caller's model shaped the burn.
        Some(c) if c.onboard_departure_dv_ms <= 0.0 => (departure_dv_ms, 0.0),
        // Partial: the launcher's tangential share, the rest onboard.
        Some(c) => {
            let l = c.launcher_departure_dv_ms.clamp(0.0, departure_dv_ms);
            (l, departure_dv_ms - l)
        }
        None => (0.0, departure_dv_ms),
    };
    let departure_dv_pool = if launcher_dv_ms <= 0.0 {
        "onboard"
    } else if onboard_departure_dv_ms <= 1e-9 {
        "launcher"
    } else {
        "split"
    };
    let onboard_dv_required_ms = onboard_departure_dv_ms + dsm_dv_ms + onboard_arrival_dv_ms;
    let onboard_dv_available_ms = dv_budget_ms(cfg);
    let exhaust_velocity_mps = cfg.spacecraft.propulsion.as_ref().map(|p| p.isp_s * G0);
    let mass_ratio_required = exhaust_velocity_mps.map(|ve| (onboard_dv_required_ms / ve).exp());
    let onboard_propellant_required_kg = mass_ratio_required.map(|mr| cfg.spacecraft.mass_kg * (1.0 - 1.0 / mr));
    DvLedgerApiResult {
        departure_dv_ms,
        departure_dv_pool: departure_dv_pool.to_string(),
        launcher_dv_ms,
        onboard_departure_dv_ms,
        onboard_arrival_dv_ms,
        onboard_dsm_dv_ms: dsm_dv_ms,
        onboard_dv_required_ms,
        onboard_dv_available_ms,
        onboard_propellant_required_kg,
        mass_ratio_required,
        propellant_feasible: onboard_dv_available_ms.map(|a| a >= onboard_dv_required_ms),
        budget_margin_ms: onboard_dv_available_ms.map(|a| a - onboard_dv_required_ms),
    }
}

/// [`dv_ledger`] for the narrowing stage (Lambert/porkchop/DiffCorrection),
/// whose crate-level solutions carry only the raw departure and arrival
/// hyperbolic excess speeds: the departure burn is the tangential parking-
/// orbit escape burn `departure_escape_dv_ms` for EVERY departure body
/// (Earth included — before an Earth departure with no covering
/// launcher was charged the bare v∞, which is neither a launcher-provided
/// injection nor the real escape burn from the orbit the spacecraft sits
/// in), `C3 = v∞²`, and the arrival burn is `arrival_dv_for_objective_ms`
/// (before the raw arrival v∞ was charged as if it were the
/// capture burn — the sign error behind a reported −967 m/s margin where
/// +832 m/s was expected). `None` only if the departure body isn't in the
/// `body_models` catalog (the escape burn can't be computed).
pub(crate) fn narrowing_dv_ledger(cfg: &MissionConfig, dep_name: &str, dep_v_inf_ms: f64, arr_v_inf_ms: f64) -> Option<DvLedgerApiResult> {
    let (departure_dv_ms, _r_park) = departure_escape_dv_ms(cfg, dep_name, dep_v_inf_ms)?;
    let c3_km2s2 = (dep_v_inf_ms / 1_000.0).powi(2);
    let arrival_dv_ms = arrival_dv_for_objective_ms(cfg, arr_v_inf_ms);
    Some(dv_ledger(cfg, dep_name, departure_dv_ms, c3_km2s2, arrival_dv_ms, 0.0))
}

/// Onboard ΔV the spacecraft's own propulsion must actually provide for a
/// narrowing-stage trajectory — `narrowing_dv_ledger(..).onboard_dv_required_ms`
/// (Phase 8c origin, now pool-aware and
/// ledger-based). Both ΔV arguments are the RAW hyperbolic excess speeds the
/// crate-level `TrajectorySolution`/`PorkchopPoint` report (their
/// `dv_departure_ms`/`dv_arr_ms` ARE those v∞ values); pricing happens
/// inside. `None` only if the departure body isn't in the `body_models`
/// catalog — distinct from a legitimate `Some(0.0)`.
pub(crate) fn onboard_dv_required_ms(cfg: &MissionConfig, dep_name: &str, dep_v_inf_ms: f64, arr_v_inf_ms: f64) -> Option<f64> {
    narrowing_dv_ledger(cfg, dep_name, dep_v_inf_ms, arr_v_inf_ms).map(|l| l.onboard_dv_required_ms)
}

/// CLI printer for a [`DvLedgerApiResult`] — the same numbers the API
/// serializes, so the CLI and frontend ledgers can never disagree.
fn print_dv_ledger(ledger: &DvLedgerApiResult) {
    match ledger.departure_dv_pool.as_str() {
        "launcher" => println!("  Departure burn:    {:.2} m/s  — launch vehicle (launcher pool)", ledger.departure_dv_ms),
        "split" => println!(
            "  Departure burn:    {:.2} m/s  — launcher {:.2} m/s + onboard {:.2} m/s (launcher falls short of the required C3)",
            ledger.departure_dv_ms, ledger.launcher_dv_ms, ledger.onboard_departure_dv_ms
        ),
        _ => println!("  Departure burn:    {:.2} m/s  — onboard (no launch vehicle covers it)", ledger.departure_dv_ms),
    }
    if ledger.onboard_dsm_dv_ms > 0.0 {
        println!("  DSMs:              {:.2} m/s  — onboard", ledger.onboard_dsm_dv_ms);
    }
    println!("  Arrival burn:      {:.2} m/s  — onboard", ledger.onboard_arrival_dv_ms);
    println!("  Onboard required:  {:.2} m/s", ledger.onboard_dv_required_ms);
    if let (Some(avail), Some(prop_kg), Some(mr), Some(margin)) = (
        ledger.onboard_dv_available_ms,
        ledger.onboard_propellant_required_kg,
        ledger.mass_ratio_required,
        ledger.budget_margin_ms,
    ) {
        println!("  Onboard available: {avail:.2} m/s  (propellant needed {prop_kg:.1} kg, mass ratio {mr:.2})");
        if margin >= 0.0 {
            println!("  Margin:           +{margin:.2} m/s  ✓");
        } else {
            println!("  Margin:            {margin:.2} m/s  ✗  (onboard ΔV exceeded)");
        }
    }
}

/// Structured result of checking a trajectory's departure C3 against the
/// configured launch vehicle's (`[spacecraft].launch_vehicle`) real
/// (C3, injected mass) performance curve — the API-facing counterpart of
/// what `print_launch_vehicle_check`/`print_best` used to only `println!`.
/// Only ever constructed for an Earth departure with a launch vehicle
/// actually configured and found in the catalog; every other case (non-Earth
/// departure, no vehicle configured, unknown vehicle name) is `None` at the
/// call site rather than a variant of this struct, so a `Some` here always
/// means "a real feasibility number was computed."
#[allow(dead_code)]
#[derive(Debug, serde::Serialize)]
pub struct LaunchVehicleCheckApiResult {
    /// Catalog name of the configured vehicle (`[spacecraft].launch_vehicle`).
    pub vehicle: String,
    /// This trajectory's required departure C3 [km²/s²].
    pub required_c3_km2s2: f64,
    /// Max mass [kg] `vehicle` can inject at `required_c3_km2s2`, from its
    /// real flown-mission performance curve — `None` if `required_c3_km2s2`
    /// exceeds every verified data point for this vehicle (see
    /// `LaunchVehicleSpec::injected_mass_kg`'s "no extrapolation" rule).
    pub max_injected_mass_kg: Option<f64>,
    /// Spacecraft wet mass [kg] (`[spacecraft].mass_kg`), for comparison.
    pub spacecraft_mass_kg: f64,
    /// `max_injected_mass_kg - spacecraft_mass_kg` — positive = margin
    /// remaining. `None` whenever `max_injected_mass_kg` is `None`.
    pub margin_kg: Option<f64>,
    /// `margin_kg.is_some_and(|m| m >= 0.0)` — true only when the vehicle has
    /// verified performance data at this C3 *and* can lift this wet mass.
    pub feasible: bool,
    /// Partial coverage (Phase 14e): the maximum C3 [km²/s²] the
    /// vehicle gives THIS wet mass, from the inverse of its verified curve
    /// (`LaunchVehicleSpec::max_c3_at_mass_km2s2`, clamped to the verified
    /// range). `None` when no verified point shows the vehicle lifting this
    /// mass to any escape energy. May be negative in principle (an ellipse
    /// short of escape), never below the parking orbit's own energy.
    pub max_c3_at_mass_km2s2: Option<f64>,
    /// Parking-orbit radius [m] the ΔV split below is referred to
    /// (`resolve_parking_orbit_radius_m`).
    pub parking_orbit_radius_m: f64,
    /// Departure ΔV [m/s] the launcher delivers from that parking orbit,
    /// `√(C3_L + 2μ/r_p) − √(μ/r_p)` with `C3_L = min(max_c3_at_mass,
    /// C3_required)` — the full tangential escape burn when it covers the
    /// C3, `0.0` when it can't lift the mass at all.
    pub launcher_departure_dv_ms: f64,
    /// Perigee top-up [m/s] the spacecraft adds from the same periapsis,
    /// `√(C3_req + 2μ/r_p) − √(C3_L + 2μ/r_p)` — `0.0` when the launcher
    /// covers the C3, the whole escape burn when it can't lift the mass.
    pub onboard_departure_dv_ms: f64,
    /// Why `feasible` is false, when it is — distinguishes "required C3 is
    /// beyond the vehicle's verified performance range" from "the vehicle
    /// cannot lift this wet mass at the required C3". `None` when feasible.
    pub reason: Option<String>,
}

/// Checks the configured launch vehicle (`[spacecraft].launch_vehicle`)
/// against this trajectory's required departure C3, structured for both the
/// CLI printer and the HTTP API. Only meaningful for an Earth departure — a
/// non-Earth departure has no rocket at all (see `onboard_dv_required_ms`)
/// and always gets `None` here, same as "no vehicle configured" or "vehicle
/// name not in the catalog" (those are real gaps, not feasibility failures,
/// so they're not reported as `feasible: false` — there's nothing to be
/// infeasible *about* without a real vehicle to check). Phase 8c origin;
/// pool-aware framing added per the design notes "Launch Vehicle
/// Selection".
pub(crate) fn compute_launch_vehicle_check(cfg: &MissionConfig, c3_km2s2: f64) -> Option<LaunchVehicleCheckApiResult> {
    compute_launch_vehicle_check_for(cfg, &cfg.trajectory.departure_body, c3_km2s2)
}

/// [`compute_launch_vehicle_check`] with the departure body named explicitly
/// — the Phase 9 optimizer has its own `[optimization].departure_body`,
/// which need not equal `[trajectory].departure_body`.
pub(crate) fn compute_launch_vehicle_check_for(cfg: &MissionConfig, dep_name: &str, c3_km2s2: f64) -> Option<LaunchVehicleCheckApiResult> {
    if !dep_name.eq_ignore_ascii_case("earth") {
        return None;
    }
    let lv_name = cfg.spacecraft.launch_vehicle.as_deref()?;
    let lv = hardware_catalog::LaunchVehicleSpec::by_name(lv_name)?;
    let body = body_models::TargetBody::by_name(dep_name)?;
    let mass_kg = cfg.spacecraft.mass_kg;
    let max_injected_mass_kg = lv.injected_mass_kg(c3_km2s2);
    let margin_kg = max_injected_mass_kg.map(|m| m - mass_kg);
    let feasible = margin_kg.is_some_and(|m| m >= 0.0);

    // Partial-coverage split at the parking-orbit periapsis (MANUAL.md
    // §13.5): the launcher's hyperbola and the required one share r_p, so
    // the top-up is the difference of the two periapsis speeds.
    let r_p = resolve_parking_orbit_radius_m(cfg, &body);
    let mu = body.mu_m3s2;
    let v_circ = (mu / r_p).sqrt();
    let c3_req_m2s2 = c3_km2s2 * 1.0e6;
    let v_peri_required = (c3_req_m2s2 + 2.0 * mu / r_p).sqrt();
    let max_c3_at_mass_km2s2 = lv.max_c3_at_mass_km2s2(mass_kg);
    // Never below the parking orbit's own energy (C3 = −μ/r_p is circular),
    // never above what this trajectory actually needs.
    let c3_launcher_m2s2 = max_c3_at_mass_km2s2.map(|c3| (c3 * 1.0e6).clamp(-mu / r_p, c3_req_m2s2.max(-mu / r_p)));
    let launcher_departure_dv_ms = c3_launcher_m2s2.map(|c3| (c3 + 2.0 * mu / r_p).sqrt() - v_circ).unwrap_or(0.0);
    let onboard_departure_dv_ms = (v_peri_required - v_circ - launcher_departure_dv_ms).max(0.0);

    let reason = if feasible {
        None
    } else {
        Some(match (max_injected_mass_kg, lv.performance_points.last()) {
            (_, None) => format!("{} has no verified interplanetary performance data", lv.name),
            (None, Some(&(c3_max, _))) => format!(
                "required C3 {c3_km2s2:.2} km²/s² exceeds {}'s verified performance range (highest verified point {c3_max:.2} km²/s²)",
                lv.name
            ),
            (Some(m_max), _) => format!(
                "{} can inject at most {m_max:.0} kg at C3 {c3_km2s2:.2} km²/s²; spacecraft wet mass is {mass_kg:.0} kg",
                lv.name
            ),
        })
    };
    Some(LaunchVehicleCheckApiResult {
        vehicle: lv.name.to_string(),
        required_c3_km2s2: c3_km2s2,
        max_injected_mass_kg,
        spacecraft_mass_kg: mass_kg,
        margin_kg,
        feasible,
        max_c3_at_mass_km2s2,
        parking_orbit_radius_m: r_p,
        launcher_departure_dv_ms,
        onboard_departure_dv_ms,
        reason,
    })
}

/// Prints whether the configured launch vehicle (`[spacecraft].launch_vehicle`)
/// can deliver this trajectory's departure C3 at the spacecraft's wet mass.
/// Thin formatting wrapper around `compute_launch_vehicle_check` — the CLI
/// path's own diagnostics (no vehicle configured / vehicle name not found)
/// stay here since the API path reports those as a plain `None` rather than
/// an error string. Only meaningful for an Earth departure.
fn print_launch_vehicle_check(cfg: &MissionConfig, c3_km2s2: f64) {
    let dep_name = cfg.trajectory.departure_body.as_str();
    if !dep_name.eq_ignore_ascii_case("earth") {
        return;
    }
    let Some(lv_name) = cfg.spacecraft.launch_vehicle.as_deref() else {
        println!(
            "\n  No [spacecraft].launch_vehicle selected — departure C3 ({c3_km2s2:.2} km²/s²) \
             not checked against any vehicle's capability."
        );
        return;
    };
    if hardware_catalog::LaunchVehicleSpec::by_name(lv_name).is_none() {
        println!("\n  Warning: launch_vehicle '{lv_name}' not found in the catalog — capability not checked.");
        return;
    }
    let Some(check) = compute_launch_vehicle_check(cfg, c3_km2s2) else {
        return; // unreachable given the two guards above, but stay defensive
    };
    println!("\n  Launch vehicle:    {}  (departure C3 = {c3_km2s2:.2} km²/s²)", check.vehicle);
    match check.max_injected_mass_kg {
        Some(max_mass) => {
            let margin = check.margin_kg.unwrap();
            println!("  Max injected mass: {max_mass:.0} kg  (s/c wet mass {:.0} kg)", check.spacecraft_mass_kg);
            if margin >= 0.0 {
                println!("  Margin:           +{margin:.0} kg  ✓");
            } else {
                println!("  Margin:            {margin:.0} kg  ✗  (vehicle cannot deliver this C3 at this mass)");
            }
        }
        None => {
            println!(
                "  ✗ No verified performance data for {} at or above this C3 — likely cannot \
                 reach it at all with this catalog's data.",
                check.vehicle
            );
        }
    }
    if !check.feasible {
        match check.max_c3_at_mass_km2s2 {
            Some(c3_max) => println!(
                "  Launcher still delivers C3 = {c3_max:.2} km²/s² at this mass → {:.2} m/s of the escape burn; \
                 onboard top-up {:.2} m/s from the {:.0} km parking orbit",
                check.launcher_departure_dv_ms, check.onboard_departure_dv_ms, check.parking_orbit_radius_m / 1e3
            ),
            None => println!("  Launcher cannot lift this mass to any verified C3 — the full escape burn is onboard."),
        }
    }
}

fn diff_correction_solver() -> DiffCorrectionSolver {
    DiffCorrectionSolver {
        max_iter: 60,
        fd_step:  vec![0.1],   // days
        tol:      vec![1.0],   // m/s
        damping:  0.8,
        max_step: vec![30.0],  // days per iteration
    }
}

/// Standalone DiffCorrection (no porkchop pre-scan): multi-start over a small
/// grid of (departure offset, TOF) seeds, each targeting `dv_total(tof) == budget`
/// as a 1-parameter Newton root-find (free variable: TOF; departure offset fixed
/// per seed). Among all converged seeds, reports the one with the smallest TOF —
/// the fastest transfer that exactly uses the available ΔV budget. Two roots of
/// `dv_total(tof) = budget` generally exist (the porkchop ΔV-vs-TOF curve is
/// bowl-shaped at fixed departure date); the shorter-TOF root is the operationally
/// useful one, so the selection is explicit here rather than left to the solver's
/// generic residual-norm tie-break.
fn run_diff_correct_standalone(cfg: &MissionConfig, almanac: &Almanac, kep: &Option<KeplerianElements>) {
    let Some(dv_budget) = dv_budget_ms(cfg) else {
        eprintln!("Error: DiffCorrection solver requires [spacecraft.propulsion] (isp_s) to define a ΔV target.");
        std::process::exit(1);
    };

    let cruise     = cfg.trajectory.cruise.as_ref();
    let tof_min    = cruise.and_then(|c| c.tof_days_min).unwrap_or(100.0);
    let tof_max    = cruise.and_then(|c| c.tof_days_max).unwrap_or(400.0);
    let dep_window = cruise.and_then(|c| c.departure_window_days).unwrap_or(60.0);

    let dep_epoch = match require_departure_epoch(cfg) {
        Ok(e) => e,
        Err(msg) => {
            eprintln!("Error: {msg}");
            std::process::exit(1);
        }
    };
    let dep_jd_base = epoch_to_jd(dep_epoch);

    let body_name = cfg.target_body.name.to_lowercase();
    let ephemeris  = cfg.target_body.ephemeris;
    let body = resolve_body(ephemeris, &body_name);
    let dep_body = resolve_departure_body(cfg);

    let dep_offset_seeds: Vec<f64> = if dep_window > 0.0 {
        vec![-dep_window / 2.0, 0.0, dep_window / 2.0]
    } else {
        vec![0.0]
    };
    let tof_seeds = [tof_min, 0.5 * (tof_min + tof_max), tof_max];

    println!("Differential correction: targeting ΔV budget = {dv_budget:.2} m/s by adjusting TOF");
    println!("  TOF search range:        {tof_min:.0} – {tof_max:.0} days");
    println!("  Departure offset seeds:  {dep_offset_seeds:?} days\n");

    let solver = diff_correction_solver();
    let mut converged: Vec<(f64, DiffCorrectionResult)> = Vec::new();

    for &dep_offset in &dep_offset_seeds {
        for &tof_seed in &tof_seeds {
            let eval = |params: &[f64]| -> Vec<f64> {
                let tof = params[0];
                match eval_dv(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_offset, tof) {
                    Some(sol) => match onboard_dv_required_ms(cfg, &cfg.trajectory.departure_body, sol.dv_departure_ms, sol.dv_arrival_ms) {
                        Some(required) => vec![required - dv_budget],
                        None           => vec![1.0e9], // departure body not in body_models catalog
                    },
                    None      => vec![1.0e9], // infeasible epoch/body lookup — push the corrector away
                }
            };
            let result = solver.solve(&[tof_seed], eval);
            println!(
                "  seed dep_offset={dep_offset:+.1}d tof={tof_seed:.1}d  →  tof={:.2}d  residual={:.3} m/s  {}",
                result.params[0], result.residuals[0],
                if result.converged { "converged" } else { "no convergence" },
            );
            if result.converged {
                converged.push((dep_offset, result));
            }
        }
    }

    let Some((dep_offset, result)) = converged
        .into_iter()
        .min_by(|(_, a), (_, b)| a.params[0].partial_cmp(&b.params[0]).unwrap())
    else {
        eprintln!("\nDifferential correction failed to converge from any seed.");
        std::process::exit(1);
    };

    let tof = result.params[0];
    let sol = eval_dv(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_offset, tof);

    println!("\n────────────────────────────────────────────────────────────────");
    println!("Differential correction result  (fastest TOF meeting ΔV budget)");
    println!("────────────────────────────────────────────────────────────────");
    println!("  Departure offset:  {dep_offset:+.2} days from reference epoch");
    println!("  Time of flight:    {tof:.3} days  ({:.2} months)", tof / 30.4375);
    println!("  Residual:          {:.4} m/s  ({} iterations)", result.residuals[0], result.iterations);
    if let Some(s) = &sol {
        println!("  ΔV total:          {:.2} m/s", s.dv_total_ms);
        if let Some(required) = onboard_dv_required_ms(cfg, &cfg.trajectory.departure_body, s.dv_departure_ms, s.dv_arrival_ms) {
            println!("  Onboard required:  {required:.2} m/s  (budget {dv_budget:.2} m/s)");
        }
        println!("  C3 at departure:   {:.4} km²/s²", s.c3_km2s2);
        println!("  v∞ at arrival:     {:.2} m/s", s.v_inf_arr_ms);
        print_launch_vehicle_check(cfg, s.c3_km2s2);
    }

    write_diff_correction_output(cfg, dep_jd_base, dep_offset, tof, sol.as_ref(), result.residuals[0]);
}

/// LambertThenDiffCorrect: seed a single Newton solve from the porkchop best
/// point (already a good guess, so no multi-start needed) and refine the TOF
/// to exactly hit the ΔV budget. Departure offset is held fixed at the
/// porkchop-best value.
fn run_diff_correct_refine(
    cfg: &MissionConfig,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
    dep_jd_base: f64,
    best: &PorkchopPoint,
) {
    let Some(dv_budget) = dv_budget_ms(cfg) else {
        println!(
            "\n(LambertThenDiffCorrect: no [spacecraft.propulsion] configured — \
             skipping Newton refinement, porkchop best stands.)"
        );
        return;
    };

    let body_name  = cfg.target_body.name.to_lowercase();
    let ephemeris  = cfg.target_body.ephemeris;
    let body       = resolve_body(ephemeris, &body_name);
    let dep_body   = resolve_departure_body(cfg);
    let dep_offset = best.dep_offset_days;

    let solver = diff_correction_solver();
    let eval = |params: &[f64]| -> Vec<f64> {
        let tof = params[0];
        match eval_dv(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_offset, tof) {
            Some(sol) => match onboard_dv_required_ms(cfg, &cfg.trajectory.departure_body, sol.dv_departure_ms, sol.dv_arrival_ms) {
                Some(required) => vec![required - dv_budget],
                None           => vec![1.0e9], // departure body not in body_models catalog
            },
            None      => vec![1.0e9],
        }
    };
    let result = solver.solve(&[best.tof_days], eval);

    println!("\n────────────────────────────────────────────────────────────────");
    println!("Newton refinement  (seeded from porkchop best, targeting ΔV budget = {dv_budget:.2} m/s)");
    println!("────────────────────────────────────────────────────────────────");
    if !result.converged {
        println!(
            "  Did not converge within {} iterations (residual {:.3} m/s) — porkchop best stands.",
            result.iterations, result.residuals[0]
        );
        return;
    }

    let tof = result.params[0];
    let sol = eval_dv(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_offset, tof);
    println!("  Departure offset:  {dep_offset:+.2} days from reference epoch (unchanged from porkchop)");
    println!("  Time of flight:    {tof:.3} days  (porkchop seed was {:.3} days)", best.tof_days);
    println!("  Residual:          {:.4} m/s  ({} iterations)", result.residuals[0], result.iterations);
    if let Some(s) = &sol {
        println!("  ΔV total:          {:.2} m/s", s.dv_total_ms);
        if let Some(required) = onboard_dv_required_ms(cfg, &cfg.trajectory.departure_body, s.dv_departure_ms, s.dv_arrival_ms) {
            println!("  Onboard required:  {required:.2} m/s  (budget {dv_budget:.2} m/s)");
        }
        print_launch_vehicle_check(cfg, s.c3_km2s2);
    }

    write_diff_correction_output(cfg, dep_jd_base, dep_offset, tof, sol.as_ref(), result.residuals[0]);
}

fn write_diff_correction_output(
    cfg: &MissionConfig,
    dep_jd_base: f64,
    dep_offset_days: f64,
    tof_days: f64,
    sol: Option<&TrajectorySolution>,
    residual_ms: f64,
) {
    let out_dir = design_out_dir(cfg);
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("\nWarning: could not create output dir '{out_dir}': {e}");
        return;
    }

    let dep_jd = dep_jd_base + dep_offset_days;
    let arr_jd = dep_jd + tof_days;
    let (c3, v_inf, dv_total) = sol
        .map(|s| (s.c3_km2s2, s.v_inf_arr_ms, s.dv_total_ms))
        .unwrap_or((f64::NAN, f64::NAN, f64::NAN));

    let path = format!("{out_dir}/diff_correction.csv");
    let csv = format!(
        "dep_jd,arr_jd,dep_offset_days,tof_days,c3_km2s2,v_inf_arr_ms,dv_total_ms,residual_ms\n\
         {dep_jd:.6},{arr_jd:.6},{dep_offset_days:.4},{tof_days:.4},{c3:.6},{v_inf:.4},{dv_total:.4},{residual_ms:.4}\n"
    );
    match fs::write(&path, csv) {
        Ok(_)  => println!("\n  {path}"),
        Err(e) => eprintln!("\n  Warning: could not write diff_correction.csv: {e}"),
    }
}

// ── Monte Carlo (local scatter around the porkchop best) ──────────────────────

/// Gaussian scatter of (departure offset, TOF) around the porkchop best point.
/// Phase-2 scope: a local robustness probe around an already-found solution,
/// not the Phase-4 full-mission dispersion/Monte Carlo (initial state, hardware
/// noise, mass properties) that will live in `sim_engine`.
fn run_monte_carlo(
    cfg: &MissionConfig,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
    dep_jd_base: f64,
    best: &PorkchopPoint,
) {
    let (ephemeris, body, dep_body) = resolve_bodies(cfg);
    let (samples, sigma_dep_offset, sigma_tof) = monte_carlo_scatter(cfg, almanac, kep, dep_jd_base, best);

    println!(
        "Monte Carlo: {MC_N_SAMPLES} samples scattered around porkchop best  \
         (σ_dep={sigma_dep_offset:.2}d, σ_tof={sigma_tof:.2}d)"
    );

    let dvs: Vec<f64> = samples.iter().filter_map(|s| s.value.as_ref().map(|v| v.dv_total_ms)).collect();
    if dvs.is_empty() {
        eprintln!("\nMonte Carlo: no samples produced a valid Lambert solution.");
        return;
    }

    let n = dvs.len() as f64;
    let mean = dvs.iter().sum::<f64>() / n;
    let var  = dvs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n;
    let std  = var.sqrt();
    let min  = dvs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max  = dvs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\n────────────────────────────────────────────────────────────────");
    println!("Monte Carlo ΔV statistics  ({} / {} samples valid)", dvs.len(), samples.len());
    println!("────────────────────────────────────────────────────────────────");
    println!("  Mean ΔV total:     {mean:.2} m/s");
    println!("  Std dev:           {std:.2} m/s");
    println!("  Min / Max:         {min:.2} / {max:.2} m/s");
    println!("  Porkchop best:     {:.2} m/s  (reference)", best.dv_total_ms);

    write_monte_carlo_output(cfg, dep_jd_base, &samples);
    write_monte_carlo_trajectories(cfg, &samples, dep_jd_base, almanac, ephemeris, body, dep_body, kep);
}

/// Gaussian scatter of (departure offset, TOF) around the porkchop best
/// point — the actual Monte Carlo computation, with no I/O. Shared by the
/// CLI path (`run_monte_carlo`, which prints + writes CSVs) and the API path
/// (`monte_carlo_api`, Phase 7j). Returns `(samples, sigma_dep_offset_days,
/// sigma_tof_days)` — the sigmas are returned alongside since both callers
/// report them (CLI in a println, API in [`MonteCarloApiResult`]).
fn monte_carlo_scatter(
    cfg: &MissionConfig,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
    dep_jd_base: f64,
    best: &PorkchopPoint,
) -> (Vec<MonteCarloSample<Option<TrajectorySolution>>>, f64, f64) {
    let (ephemeris, body, dep_body) = resolve_bodies(cfg);

    let cruise = cfg.trajectory.cruise.as_ref();
    let tof_min = cruise.and_then(|c| c.tof_days_min).unwrap_or(100.0);
    let tof_max = cruise.and_then(|c| c.tof_days_max).unwrap_or(400.0);
    let tof_span = tof_max - tof_min;

    let sigma_dep_offset = 2.0_f64;          // days, 1-sigma scatter around porkchop best
    let sigma_tof        = 0.02 * tof_span;  // days, 2% of the configured TOF span

    let solver = MonteCarloSolver { sigma: vec![sigma_dep_offset, sigma_tof], n_samples: MC_N_SAMPLES, seed: 42 };
    let reference = [best.dep_offset_days, best.tof_days];

    let samples = solver.run(&reference, |p| {
        eval_dv(almanac, dep_body, ephemeris, body, kep, dep_jd_base, p[0], p[1])
    });

    (samples, sigma_dep_offset, sigma_tof)
}

/// API-shaped Monte Carlo result (Phase 7j) — statistics over the full
/// scatter, plus a stride-selected subset (`MC_TRAJ_COUNT`) carrying a
/// sparse propagated trajectory, mirroring `monte_carlo.csv` +
/// `monte_carlo_trajectories.csv`'s CLI output combined into one shape.
fn monte_carlo_api(
    cfg: &MissionConfig,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
    dep_jd_base: f64,
    best: &PorkchopPoint,
) -> MonteCarloApiResult {
    let (ephemeris, body, dep_body) = resolve_bodies(cfg);
    let (samples, sigma_dep_offset, sigma_tof) = monte_carlo_scatter(cfg, almanac, kep, dep_jd_base, best);

    let dvs: Vec<f64> = samples.iter().filter_map(|s| s.value.as_ref().map(|v| v.dv_total_ms)).collect();
    let (mean, std, min, max) = if dvs.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    } else {
        let n = dvs.len() as f64;
        let mean = dvs.iter().sum::<f64>() / n;
        let var = dvs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n;
        (mean, var.sqrt(), dvs.iter().cloned().fold(f64::INFINITY, f64::min), dvs.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    };

    // Stride-selected subset gets a sparse propagated trajectory — same
    // selection as `write_monte_carlo_trajectories`'s CLI output.
    let stride = (samples.len() / MC_TRAJ_COUNT).max(1);
    let traj_indices: std::collections::HashSet<usize> =
        (0..samples.len()).step_by(stride).take(MC_TRAJ_COUNT).collect();

    let api_samples = samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let dep_offset = s.params[0];
            let tof = s.params[1];
            match &s.value {
                Some(sol) => {
                    let trajectory = if traj_indices.contains(&i) {
                        // Entries built per-sample, at this sample's actual
                        // departure epoch — each MC sample has a different
                        // dep_offset, so a shared/hoisted entry set built at
                        // dep_jd_base alone would put every registered body
                        // at the wrong epoch for all but one sample.
                        let body_entries = propagator_body_entries(cfg, almanac, dep_jd_base + dep_offset);
                        let propagator_bodies = as_propagator_bodies(&body_entries);
                        transfer_states(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_offset, tof)
                            .map(|(r_dep, ..)| propagate_arc_n(r_dep, sol.v_transfer_dep, MU_SUN_M3S2, tof * 86_400.0, MC_TRAJ_SAMPLE_COUNT, &propagator_bodies))
                    } else {
                        None
                    };
                    MonteCarloSampleApiResult {
                        dep_offset_days: dep_offset,
                        tof_days: tof,
                        valid: true,
                        dv_total_ms: Some(sol.dv_total_ms),
                        c3_km2s2: Some(sol.c3_km2s2),
                        v_inf_arr_ms: Some(sol.v_inf_arr_ms),
                        trajectory,
                    }
                }
                None => MonteCarloSampleApiResult {
                    dep_offset_days: dep_offset,
                    tof_days: tof,
                    valid: false,
                    dv_total_ms: None,
                    c3_km2s2: None,
                    v_inf_arr_ms: None,
                    trajectory: None,
                },
            }
        })
        .collect();

    MonteCarloApiResult {
        n_samples: samples.len(),
        n_valid: dvs.len(),
        sigma_dep_offset_days: sigma_dep_offset,
        sigma_tof_days: sigma_tof,
        mean_dv_total_ms: mean,
        std_dv_total_ms: std,
        min_dv_total_ms: min,
        max_dv_total_ms: max,
        reference_dv_total_ms: best.dv_total_ms,
        samples: api_samples,
    }
}

/// Sparse-sampled propagated trajectory for a representative subset of the
/// Monte Carlo scatter (Phase 8i, absorbs 7h) — lets the dispersion cloud be
/// plotted and visually confirmed to scatter sensibly around the reference
/// trajectory, not just described by scalar ΔV statistics. Deliberately a
/// stride-selected subset (`MC_TRAJ_COUNT` of the full `N_SAMPLES`) at a
/// sparse point density (`MC_TRAJ_SAMPLE_COUNT`) — payload scales with
/// sample count × points per sample, and a cloud needs many trajectories,
/// not one finely-sampled one (the design notes "Monte Carlo never gets the
/// dense... stream" rule, applied to the per-sample density too).
fn write_monte_carlo_trajectories(
    cfg: &MissionConfig,
    samples: &[MonteCarloSample<Option<TrajectorySolution>>],
    dep_jd_base: f64,
    almanac: &Almanac,
    ephemeris: EphemerisSource,
    body: Option<Body>,
    dep_body: Option<Body>,
    kep: &Option<KeplerianElements>,
) {
    let stride = (samples.len() / MC_TRAJ_COUNT).max(1);
    let selected: Vec<&MonteCarloSample<Option<TrajectorySolution>>> =
        samples.iter().step_by(stride).take(MC_TRAJ_COUNT).collect();

    let mut rows = vec!["sample_id,dep_offset_days,tof_days,dv_total_ms,t_s,x_m,y_m,z_m".to_string()];
    let mut n_written = 0usize;
    for (i, s) in selected.iter().enumerate() {
        let Some(sol) = &s.value else { continue };
        let dep_offset = s.params[0];
        let tof = s.params[1];
        // Built per-sample, at this sample's actual departure epoch — see
        // the matching comment in `monte_carlo_api`.
        let body_entries = propagator_body_entries(cfg, almanac, dep_jd_base + dep_offset);
        let propagator_bodies = as_propagator_bodies(&body_entries);
        let Some((r_dep, ..)) =
            transfer_states(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_offset, tof)
        else { continue };
        let arc = propagate_arc_n(r_dep, sol.v_transfer_dep, MU_SUN_M3S2, tof * 86_400.0, MC_TRAJ_SAMPLE_COUNT, &propagator_bodies);
        for point in &arc {
            rows.push(format!(
                "{i},{dep_offset:.4},{tof:.4},{:.4},{:.3},{:.6e},{:.6e},{:.6e}",
                sol.dv_total_ms, point.t_s, point.x_m, point.y_m, point.z_m,
            ));
        }
        n_written += 1;
    }

    let out_dir = design_out_dir(cfg);
    let path = format!("{out_dir}/monte_carlo_trajectories.csv");
    match fs::write(&path, rows.join("\n") + "\n") {
        Ok(_) => println!("  {path}  ({n_written} sample trajectories, {MC_TRAJ_SAMPLE_COUNT:.0} pts each)"),
        Err(e) => eprintln!("  Warning: could not write monte_carlo_trajectories.csv: {e}"),
    }
}

fn write_monte_carlo_output(
    cfg: &MissionConfig,
    dep_jd_base: f64,
    samples: &[MonteCarloSample<Option<TrajectorySolution>>],
) {
    let out_dir = design_out_dir(cfg);
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("\nWarning: could not create output dir '{out_dir}': {e}");
        return;
    }

    let path = format!("{out_dir}/monte_carlo.csv");
    let mut rows = vec!["dep_jd,dep_offset_days,tof_days,valid,dv_total_ms,c3_km2s2,v_inf_arr_ms".to_string()];
    for s in samples {
        let dep_jd = dep_jd_base + s.params[0];
        match &s.value {
            Some(sol) => rows.push(format!(
                "{dep_jd:.6},{:.4},{:.4},1,{:.4},{:.6},{:.4}",
                s.params[0], s.params[1], sol.dv_total_ms, sol.c3_km2s2, sol.v_inf_arr_ms
            )),
            None => rows.push(format!("{dep_jd:.6},{:.4},{:.4},0,,,", s.params[0], s.params[1])),
        }
    }
    match fs::write(&path, rows.join("\n") + "\n") {
        Ok(_)  => println!("\n  {path}  ({} samples)", samples.len()),
        Err(e) => eprintln!("\n  Warning: could not write monte_carlo.csv: {e}"),
    }
}

// ── Genetic algorithm (Phase 8j) ───────────────────────────────────────────────

/// Fixed GA hyperparameters — same "safe to modify, hardcoded for now"
/// precedent as `diff_correction_solver()`'s fixed Newton-solver settings;
/// no `[trajectory.ga]` TOML block exists yet, consistent with the existing
/// solvers in this file.
fn ga_solver() -> GaSolver {
    GaSolver {
        population_size: 60,
        generations: 100,
        crossover_rate: 0.8,
        mutation_rate: 0.3,
        elitism_count: 2,
        tournament_size: 3,
        seed: 42,
    }
}

/// Fitness value `best_point_for_objective` would also select on — using the
/// *same* metric here is what makes "did the GA converge to the grid
/// search's already-known best point" a meaningful, apples-to-apples check
/// (Phase 8j's verification approach), not a comparison between two
/// different optimization targets.
fn ga_fitness(cfg: &MissionConfig, sol: &TrajectorySolution) -> f64 {
    if matches!(cfg.mission.objective, MissionObjective::Flyby) {
        sol.c3_km2s2
    } else {
        sol.dv_total_ms
    }
}

/// Result of a GA or PSO search — no I/O, shared by the CLI path
/// (`run_ga`/`run_pso`, which print + write CSVs) and the API path
/// (`ga_api`/`pso_api`, Phase 7j).
struct GaPsoComputeResult {
    dep_jd_base: f64,
    bounds: [(f64, f64); 2],
    /// [departure_offset_days, tof_days]
    best_params: [f64; 2],
    best_fitness: f64,
    /// Best-fitness-so-far per generation (GA) or iteration (PSO).
    history: Vec<f64>,
    solution: Option<TrajectorySolution>,
}

/// Departure-offset × TOF search box and reference epoch shared by GA and
/// PSO — same box `run_porkchop_scan`'s grid covers, so a result landing on
/// (or very near) the grid search's independently-found best point is real
/// cross-validation, not a search-space coincidence (Phase 8j).
fn optimizer_bounds_and_epoch(cfg: &MissionConfig) -> Result<([(f64, f64); 2], f64), String> {
    let cruise = cfg.trajectory.cruise.as_ref();
    let tof_min = cruise.and_then(|c| c.tof_days_min).unwrap_or(100.0);
    let tof_max = cruise.and_then(|c| c.tof_days_max).unwrap_or(400.0);
    let dep_window = cruise.and_then(|c| c.departure_window_days).unwrap_or(60.0);

    let dep_epoch = require_departure_epoch(cfg)?;
    let dep_jd_base = epoch_to_jd(dep_epoch);

    Ok(([(-dep_window / 2.0, dep_window / 2.0), (tof_min, tof_max)], dep_jd_base))
}

fn ga_compute(
    cfg: &MissionConfig,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
) -> Result<GaPsoComputeResult, String> {
    let (bounds, dep_jd_base) = optimizer_bounds_and_epoch(cfg)?;
    let (ephemeris, body, dep_body) = resolve_bodies(cfg);

    let result = ga_solver().run(&bounds, |params| {
        eval_dv(almanac, dep_body, ephemeris, body, kep, dep_jd_base, params[0], params[1]).map(|sol| ga_fitness(cfg, &sol))
    });

    let dep_offset = result.best_params[0];
    let tof = result.best_params[1];
    let solution = eval_dv(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_offset, tof);

    Ok(GaPsoComputeResult {
        dep_jd_base,
        bounds,
        best_params: [dep_offset, tof],
        best_fitness: result.best_fitness,
        history: result.history,
        solution,
    })
}

fn pso_compute(
    cfg: &MissionConfig,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
) -> Result<GaPsoComputeResult, String> {
    let (bounds, dep_jd_base) = optimizer_bounds_and_epoch(cfg)?;
    let (ephemeris, body, dep_body) = resolve_bodies(cfg);

    let result = pso_solver().run(&bounds, |params| {
        eval_dv(almanac, dep_body, ephemeris, body, kep, dep_jd_base, params[0], params[1]).map(|sol| ga_fitness(cfg, &sol))
    });

    let dep_offset = result.best_params[0];
    let tof = result.best_params[1];
    let solution = eval_dv(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_offset, tof);

    Ok(GaPsoComputeResult {
        dep_jd_base,
        bounds,
        best_params: [dep_offset, tof],
        best_fitness: result.best_fitness,
        history: result.history,
        solution,
    })
}

fn run_ga(cfg: &MissionConfig, almanac: &Almanac, kep: &Option<KeplerianElements>) {
    let r = match ga_compute(cfg, almanac, kep) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    let is_flyby = matches!(cfg.mission.objective, MissionObjective::Flyby);

    println!(
        "Genetic algorithm: {} individuals × {} generations, search box dep_offset∈[{:+.1},{:+.1}]d tof∈[{:.1},{:.1}]d  \
         (fitness = {})",
        ga_solver().population_size, ga_solver().generations,
        r.bounds[0].0, r.bounds[0].1, r.bounds[1].0, r.bounds[1].1,
        if is_flyby { "C3" } else { "ΔV total" },
    );

    let dep_offset = r.best_params[0];
    let tof = r.best_params[1];

    println!("\n────────────────────────────────────────────────────────────────");
    println!("GA result  (best individual, generation {})", r.history.len());
    println!("────────────────────────────────────────────────────────────────");
    println!("  Departure offset:  {dep_offset:+.2} days from reference epoch");
    println!("  Time of flight:    {tof:.3} days  ({:.2} months)", tof / 30.4375);
    println!("  Fitness:           {:.4} {}", r.best_fitness, if is_flyby { "km²/s²" } else { "m/s" });
    if let Some(s) = &r.solution {
        println!("  C3 at departure:   {:.4} km²/s²", s.c3_km2s2);
        println!("  v∞ at arrival:     {:.2} m/s", s.v_inf_arr_ms);
        println!("  ΔV total:          {:.2} m/s", s.dv_total_ms);
        print_launch_vehicle_check(cfg, s.c3_km2s2);
    }

    write_optimizer_output(cfg, "ga", &r.history, r.dep_jd_base, dep_offset, tof, r.solution.as_ref());
}

// ── Particle swarm optimization (Phase 8j follow-up) ───────────────────────────

/// Fixed PSO hyperparameters — same "safe to modify, hardcoded for now"
/// precedent as `ga_solver()`; no `[trajectory.pso]` TOML block exists yet.
fn pso_solver() -> PsoSolver {
    PsoSolver {
        swarm_size: 40,
        iterations: 100,
        inertia_max: 0.9,
        inertia_min: 0.4,
        cognitive_coeff: 1.5,
        social_coeff: 1.5,
        seed: 42,
    }
}

fn run_pso(cfg: &MissionConfig, almanac: &Almanac, kep: &Option<KeplerianElements>) {
    let r = match pso_compute(cfg, almanac, kep) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    let is_flyby = matches!(cfg.mission.objective, MissionObjective::Flyby);

    println!(
        "Particle swarm: {} particles × {} iterations, search box dep_offset∈[{:+.1},{:+.1}]d tof∈[{:.1},{:.1}]d  \
         (fitness = {})",
        pso_solver().swarm_size, pso_solver().iterations,
        r.bounds[0].0, r.bounds[0].1, r.bounds[1].0, r.bounds[1].1,
        if is_flyby { "C3" } else { "ΔV total" },
    );

    let dep_offset = r.best_params[0];
    let tof = r.best_params[1];

    println!("\n────────────────────────────────────────────────────────────────");
    println!("PSO result  (best particle, iteration {})", r.history.len());
    println!("────────────────────────────────────────────────────────────────");
    println!("  Departure offset:  {dep_offset:+.2} days from reference epoch");
    println!("  Time of flight:    {tof:.3} days  ({:.2} months)", tof / 30.4375);
    println!("  Fitness:           {:.4} {}", r.best_fitness, if is_flyby { "km²/s²" } else { "m/s" });
    if let Some(s) = &r.solution {
        println!("  C3 at departure:   {:.4} km²/s²", s.c3_km2s2);
        println!("  v∞ at arrival:     {:.2} m/s", s.v_inf_arr_ms);
        println!("  ΔV total:          {:.2} m/s", s.dv_total_ms);
        print_launch_vehicle_check(cfg, s.c3_km2s2);
    }

    write_optimizer_output(cfg, "pso", &r.history, r.dep_jd_base, dep_offset, tof, r.solution.as_ref());
}

/// Write `<solver>_convergence.csv` + `<solver>_best.csv` — shared by GA and
/// PSO, which write identically-shaped output (only the convergence column
/// header differs: "generation" vs "iteration").
fn write_optimizer_output(
    cfg: &MissionConfig,
    solver_prefix: &str,
    history: &[f64],
    dep_jd_base: f64,
    dep_offset_days: f64,
    tof_days: f64,
    sol: Option<&TrajectorySolution>,
) {
    let out_dir = design_out_dir(cfg);
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("\nWarning: could not create output dir '{out_dir}': {e}");
        return;
    }

    let step_col = if solver_prefix == "ga" { "generation" } else { "iteration" };
    let convergence_path = format!("{out_dir}/{solver_prefix}_convergence.csv");
    let mut rows = vec![format!("{step_col},best_fitness_so_far")];
    rows.extend(history.iter().enumerate().map(|(i, f)| format!("{i},{f:.6}")));
    match fs::write(&convergence_path, rows.join("\n") + "\n") {
        Ok(_)  => println!("\n  {convergence_path}  ({} {step_col}s)", history.len()),
        Err(e) => eprintln!("\n  Warning: could not write {solver_prefix}_convergence.csv: {e}"),
    }

    let dep_jd = dep_jd_base + dep_offset_days;
    let arr_jd = dep_jd + tof_days;
    let (c3, v_inf, dv_total) = sol
        .map(|s| (s.c3_km2s2, s.v_inf_arr_ms, s.dv_total_ms))
        .unwrap_or((f64::NAN, f64::NAN, f64::NAN));
    let best_fitness = history.last().copied().unwrap_or(f64::NAN);
    let best_path = format!("{out_dir}/{solver_prefix}_best.csv");
    let csv = format!(
        "dep_jd,arr_jd,dep_offset_days,tof_days,c3_km2s2,v_inf_arr_ms,dv_total_ms,best_fitness\n\
         {dep_jd:.6},{arr_jd:.6},{dep_offset_days:.4},{tof_days:.4},{c3:.6},{v_inf:.4},{dv_total:.4},{best_fitness:.6}\n",
    );
    match fs::write(&best_path, csv) {
        Ok(_)  => println!("  {best_path}"),
        Err(e) => eprintln!("  Warning: could not write {solver_prefix}_best.csv: {e}"),
    }
}

/// Build the API-shaped GA/PSO result — re-solves the Lambert problem at the
/// best (departure offset, TOF) point for the visualization arc, same as
/// `build_best_arc_api` does for the porkchop-based solvers (Phase 7j).
fn build_optimizer_api_result(
    cfg: &MissionConfig,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
    solver_name: &str,
    r: GaPsoComputeResult,
) -> Result<OptimizerApiResult, String> {
    let dep_offset = r.best_params[0];
    let tof = r.best_params[1];
    let is_flyby = matches!(cfg.mission.objective, MissionObjective::Flyby);

    let (ephemeris, body, dep_body) = resolve_bodies(cfg);
    let arc = lambert_visualization_arc(cfg, almanac, dep_body, ephemeris, body, kep, r.dep_jd_base, dep_offset, tof);

    let sol = r.solution.ok_or_else(|| {
        "Optimizer converged to a point with no valid Lambert solution".to_string()
    })?;
    let ledger = narrowing_dv_ledger(cfg, &cfg.trajectory.departure_body, sol.dv_departure_ms, sol.v_inf_arr_ms);

    Ok(OptimizerApiResult {
        solver: solver_name.to_string(),
        dep_offset_days: dep_offset,
        dep_jd: r.dep_jd_base + dep_offset,
        tof_days: tof,
        fitness: r.best_fitness,
        fitness_units: if is_flyby { "km2/s2".into() } else { "m/s".into() },
        c3_km2s2: sol.c3_km2s2,
        v_inf_arr_ms: sol.v_inf_arr_ms,
        dv_total_ms: sol.dv_total_ms,
        dv_budget_ms: dv_budget_ms(cfg),
        budget_margin_ms: ledger.as_ref().and_then(|l| l.budget_margin_ms),
        dv_ledger: ledger,
        launch_vehicle_check: compute_launch_vehicle_check(cfg, sol.c3_km2s2),
        convergence: r.history,
        arc,
    })
}

#[allow(dead_code)]
fn ga_api(cfg: &MissionConfig, almanac: &Almanac, kep: &Option<KeplerianElements>) -> Result<OptimizerApiResult, String> {
    let r = ga_compute(cfg, almanac, kep)?;
    build_optimizer_api_result(cfg, almanac, kep, "GA", r)
}

#[allow(dead_code)]
fn pso_api(cfg: &MissionConfig, almanac: &Almanac, kep: &Option<KeplerianElements>) -> Result<OptimizerApiResult, String> {
    let r = pso_compute(cfg, almanac, kep)?;
    build_optimizer_api_result(cfg, almanac, kep, "PSO", r)
}

// ── File output ──────────────────────────────────────────────────────────────

fn design_out_dir(cfg: &MissionConfig) -> String {
    format!("{}/design", cfg.simulation.output_dir.trim_end_matches('/'))
}

fn write_outputs(cfg: &MissionConfig, points: &[PorkchopPoint], best: &PorkchopPoint, dep_jd_base: f64) {
    let out_dir = design_out_dir(cfg);
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("\nWarning: could not create output dir '{out_dir}': {e}");
        return;
    }

    // porkchop.csv
    let pork_path = format!("{out_dir}/porkchop.csv");
    let mut rows = vec![PorkchopPoint::csv_header().to_string()];
    rows.extend(points.iter().map(|p| p.to_csv_row()));
    match fs::write(&pork_path, rows.join("\n") + "\n") {
        Ok(_)  => println!("\nOutput:"),
        Err(e) => { eprintln!("\nWarning: could not write porkchop.csv: {e}"); return; }
    }
    println!("  {pork_path}  ({} points)", points.len());

    // best_arc.csv
    let dep_jd = dep_jd_base + best.dep_offset_days;
    let arr_jd = dep_jd + best.tof_days;
    let best_path = format!("{out_dir}/best_arc.csv");
    let best_csv = format!(
        "dep_jd,arr_jd,tof_days,c3_km2s2,v_inf_arr_ms,dv_dep_ms,dv_arr_ms,dv_total_ms\n\
         {dep_jd:.6},{arr_jd:.6},{:.4},{:.6},{:.4},{:.4},{:.4},{:.4}\n",
        best.tof_days, best.c3_km2s2, best.v_inf_arr_ms,
        best.dv_dep_ms, best.dv_arr_ms, best.dv_total_ms,
    );
    match fs::write(&best_path, best_csv) {
        Ok(_)  => println!("  {best_path}"),
        Err(e) => eprintln!("  Warning: could not write best_arc.csv: {e}"),
    }
}

/// Write the best arc's real propagated trajectory (Phase 7) to CSV, for
/// plotting — `plot/plot_trajectory.py` reads this. Re-solves the Lambert arc
/// at the best (dep_offset, tof) point (cheap — one extra Lambert solve, not
/// the whole grid) to get the departure state needed to propagate.
fn write_best_arc_trajectory(
    cfg: &MissionConfig,
    best: &PorkchopPoint,
    dep_jd_base: f64,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
) {
    let body_name = cfg.target_body.name.to_lowercase();
    let ephemeris = cfg.target_body.ephemeris;
    let body = resolve_body(ephemeris, &body_name);
    let dep_body = resolve_departure_body(cfg);

    let Some((r_dep, v_dep, r_arr, v_arr)) =
        transfer_states(almanac, dep_body, ephemeris, body, kep, dep_jd_base, best.dep_offset_days, best.tof_days)
    else {
        eprintln!("  Warning: could not re-evaluate transfer states for trajectory CSV");
        return;
    };
    let lambert = LambertArc { r_dep, v_dep, r_arr, v_arr, tof_s: best.tof_days * 86_400.0, mu: MU_SUN_M3S2 };
    let Ok(sol) = lambert.solve() else {
        eprintln!("  Warning: could not re-solve Lambert arc for trajectory CSV");
        return;
    };

    let out_dir = design_out_dir(cfg);
    let body_entries = propagator_body_entries(cfg, almanac, dep_jd_base + best.dep_offset_days);
    let propagator_bodies = as_propagator_bodies(&body_entries);
    let arc = propagate_arc(r_dep, sol.v_transfer_dep, MU_SUN_M3S2, best.tof_days * 86_400.0, &propagator_bodies);
    let traj_path = format!("{out_dir}/best_arc_trajectory.csv");
    let mut rows = vec!["t_s,x_m,y_m,z_m".to_string()];
    rows.extend(arc.iter().map(|p| format!("{:.3},{:.6e},{:.6e},{:.6e}", p.t_s, p.x_m, p.y_m, p.z_m)));
    match fs::write(&traj_path, rows.join("\n") + "\n") {
        Ok(_) => println!("  {traj_path}  ({} points)", arc.len()),
        Err(e) => eprintln!("  Warning: could not write best_arc_trajectory.csv: {e}"),
    }

    write_best_arc_bodies(cfg, best, dep_jd_base, almanac, kep, ephemeris, body, dep_body);
}

/// Sample the departure body's and the target body's heliocentric positions
/// over the transfer window, for the plot to show alongside the spacecraft's
/// arc — lets you visually confirm the target body is actually where the
/// spacecraft arrives, not just trust the printed numbers.
fn write_best_arc_bodies(
    cfg: &MissionConfig,
    best: &PorkchopPoint,
    dep_jd_base: f64,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
    ephemeris: EphemerisSource,
    body: Option<Body>,
    dep_body: Option<Body>,
) {
    let dep_jd = dep_jd_base + best.dep_offset_days;
    let dep_name = cfg.trajectory.departure_body.as_str();
    const N: usize = 60;
    let mut rows = vec!["t_s,dep_x_m,dep_y_m,dep_z_m,target_x_m,target_y_m,target_z_m".to_string()];
    for i in 0..N {
        let frac = i as f64 / (N - 1) as f64;
        let jd = dep_jd + frac * best.tof_days;
        let t_s = frac * best.tof_days * 86_400.0;
        let Some((dep_r, _)) = body_state(almanac, EphemerisSource::Anise, dep_body, &None, jd) else { continue };
        let Some((target_r, _)) = body_state(almanac, ephemeris, body, kep, jd) else { continue };
        rows.push(format!(
            "{t_s:.3},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
            dep_r[0], dep_r[1], dep_r[2], target_r[0], target_r[1], target_r[2],
        ));
    }

    let out_dir = design_out_dir(cfg);
    let path = format!("{out_dir}/best_arc_bodies.csv");
    let csv = format!("# target_name={}\n# dep_name={dep_name}\n{}\n", cfg.target_body.name, rows.join("\n"));
    match fs::write(&path, csv) {
        Ok(_) => println!("  {path}  ({} points, target={})", rows.len() - 1, cfg.target_body.name),
        Err(e) => eprintln!("  Warning: could not write best_arc_bodies.csv: {e}"),
    }
}

/// Propagate a spread of porkchop grid points (sampled across the ΔV-ranked
/// distribution — best through worst, not a random/grid-order subset) so the
/// whole considered search space can be visually sanity-checked, not just
/// the cherry-picked best point. `N_SAMPLES` real propagations is cheap
/// (each is one Lambert re-solve + one Dopri5 integration, same cost as the
/// single best-arc case) compared to the porkchop scan itself.
fn write_porkchop_sample_trajectories(
    cfg: &MissionConfig,
    points: &[PorkchopPoint],
    dep_jd_base: f64,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
) {
    const N_SAMPLES: usize = 8;
    let body_name = cfg.target_body.name.to_lowercase();
    let ephemeris = cfg.target_body.ephemeris;
    let body = resolve_body(ephemeris, &body_name);
    let dep_body = resolve_departure_body(cfg);

    let mut sorted: Vec<&PorkchopPoint> = points.iter().collect();
    sorted.sort_by(|a, b| a.dv_total_ms.partial_cmp(&b.dv_total_ms).unwrap());

    let mut rows = vec!["sample_rank_pct,dep_offset_days,tof_days,dv_total_ms,t_s,x_m,y_m,z_m".to_string()];
    for i in 0..N_SAMPLES {
        let idx = (i * (sorted.len() - 1)) / (N_SAMPLES - 1);
        let p = sorted[idx];
        let rank_pct = 100.0 * idx as f64 / (sorted.len() - 1) as f64;
        // Built per-sample, at this sample's actual departure epoch — each
        // porkchop sample point has a different dep_offset_days.
        let body_entries = propagator_body_entries(cfg, almanac, dep_jd_base + p.dep_offset_days);
        let propagator_bodies = as_propagator_bodies(&body_entries);

        let Some((r_dep, v_dep, r_arr, v_arr)) =
            transfer_states(almanac, dep_body, ephemeris, body, kep, dep_jd_base, p.dep_offset_days, p.tof_days)
        else { continue };
        let lambert = LambertArc { r_dep, v_dep, r_arr, v_arr, tof_s: p.tof_days * 86_400.0, mu: MU_SUN_M3S2 };
        let Ok(sol) = lambert.solve() else { continue };
        let arc = propagate_arc(r_dep, sol.v_transfer_dep, MU_SUN_M3S2, p.tof_days * 86_400.0, &propagator_bodies);

        for point in &arc {
            rows.push(format!(
                "{rank_pct:.1},{:.4},{:.4},{:.4},{:.3},{:.6e},{:.6e},{:.6e}",
                p.dep_offset_days, p.tof_days, p.dv_total_ms, point.t_s, point.x_m, point.y_m, point.z_m,
            ));
        }
    }

    let out_dir = design_out_dir(cfg);
    let path = format!("{out_dir}/porkchop_samples.csv");
    match fs::write(&path, rows.join("\n") + "\n") {
        Ok(_) => println!("  {path}  ({N_SAMPLES} sample trajectories, ranked best-to-worst by ΔV)"),
        Err(e) => eprintln!("  Warning: could not write porkchop_samples.csv: {e}"),
    }
}

// ── Ephemeris / Keplerian body-state helpers ───────────────────────────────────

/// Body state in the heliocentric frame at a given Julian Date, dispatched by
/// `ephemeris` source. The one ephemeris-aware chokepoint that both the porkchop
/// scan and the DiffCorrection/MonteCarlo evaluators call through.
pub(crate) fn body_state(
    almanac: &Almanac,
    ephemeris: EphemerisSource,
    body: Option<Body>,
    kep: &Option<KeplerianElements>,
    jd: f64,
) -> Option<([f64; 3], [f64; 3])> {
    match ephemeris {
        EphemerisSource::Anise => {
            let b = body?;
            let ep = jd_to_epoch(jd);
            let st = almanac.body_state_heliocentric(b, ep).ok()?;
            Some((sv3_to_arr(st.position.inner), sv3_to_arr(st.velocity.inner)))
        }
        EphemerisSource::Keplerian => {
            let (r, v) = kep.as_ref()?.state_at_jd(jd);
            Some((r, v))
        }
        EphemerisSource::Custom => None,
    }
}

/// Auto-populates `cruise_seed.body_tracks` entries for `target_body.
/// third_bodies` — Phase 01/02 consistency ask. Before this,
/// Phase 01's trajectory design used `third_bodies` as real gravitational
/// perturbers (`propagator_body_entries`), but Phase 02's cruise loop only
/// ever saw perturbers the CLIENT happened to also put in `cruise_seed.
/// body_tracks` — nothing kept the two in sync, so a mission configured
/// with e.g. `third_bodies = ["Jupiter"]` in Phase 01 could silently lose
/// Jupiter's pull the moment it moved to Phase 02.
///
/// Called from `server::routes::simulate::start()` before dispatching a
/// cruise job, so it never needs to be threaded through `cruise.rs` itself
/// (deliberately kept ANISE-free, see that module's own doc comment) —
/// this function does the one-time ANISE work up front and hands `cruise::
/// build_body_track_perturbers` plain precomputed tracks, same as any
/// client-supplied `body_tracks` entry.
///
/// Any name ALREADY present in `existing` is left untouched (an explicit
/// client-supplied track — e.g. a hand-tuned or synthetic one — always
/// wins over this auto-population, never silently overwritten). Returns
/// `vec![]` (not an error) when `trajectory.departure_epoch` is unset or a
/// name doesn't resolve via ANISE — this is a best-effort convenience
/// layer, not a hard requirement; a mission that wants guaranteed
/// perturbers should still supply `body_tracks` explicitly.
pub(crate) fn build_third_body_tracks(
    cfg: &MissionConfig,
    almanac: &Almanac,
    duration_s: f64,
    existing: &[crate::config::BodyTrackConfig],
) -> Vec<crate::config::BodyTrackConfig> {
    let Ok(dep_epoch) = require_departure_epoch(cfg) else { return Vec::new() };
    let dep_jd = epoch_to_jd(dep_epoch);
    const N_SAMPLES: usize = 8;

    cfg.target_body
        .third_bodies
        .iter()
        .filter(|name| !existing.iter().any(|t| &t.name == *name))
        .filter_map(|name| {
            let anise = anise_body(&name.to_lowercase())?;
            let mu_m3s2 = body_models::TargetBody::by_name(name).map(|b| b.mu_m3s2);
            let track: Vec<crate::config::CruiseReferencePointConfig> = (0..N_SAMPLES)
                .filter_map(|i| {
                    let t_s = duration_s * i as f64 / (N_SAMPLES - 1) as f64;
                    let jd = dep_jd + t_s / 86_400.0;
                    let (r_m, v_mps) = body_state(almanac, EphemerisSource::Anise, Some(anise), &None, jd)?;
                    Some(crate::config::CruiseReferencePointConfig { t_s, r_m, v_mps })
                })
                .collect();
            if track.len() < 2 {
                return None;
            }
            // These are always third_bodies-list perturbers (never the
            // mission's own target/capture body), so soi_capture stays
            // false -- a config wanting real central-body switching for
            // its capture body must supply that body_track explicitly with
            // soi_capture: true, per BodyTrackConfig's own doc comment.
            Some(crate::config::BodyTrackConfig { name: name.clone(), track, epoch_jd: Some(dep_jd), mu_m3s2, soi_capture: false })
        })
        .collect()
}

/// Review B2 — server-resolved body tracks. Expands every
/// `cruise_seed.body_tracks` entry whose `track` is EMPTY into a real
/// ANISE-sampled track over the leg's duration, so a client can request a
/// perturber/pointing/SOI body by `{ name, epoch_jd }` alone instead of
/// shipping ephemeris samples through JSON at client-chosen density (the
/// backend owns the kernels — `/api/bodies/{name}/state` already proves
/// it). Entries with a non-empty `track` are left completely untouched
/// (explicit client data always wins, same convention as
/// [`build_third_body_tracks`]).
///
/// Density: 4 samples/day of leg duration, clamped to [64, 2000] points —
/// consumed through `ReferenceTrajectory`'s cubic Hermite interpolation
/// (MANUAL.md §9.1), whose O(Δt⁴) error at 6 h spacing is far below any
/// physical effect these tracks drive (pointing, third-body perturbation,
/// SOI switching). Errors (not silently degrades) when a named body has no
/// ANISE coverage or no epoch anchors the sampling — an empty track the
/// client explicitly asked the server to fill must never quietly become
/// "no perturbation at all."
///
/// Called from `server::routes::simulate::start()` BEFORE `check_config`,
/// so validation sees fully-resolved tracks and its ≥2-points rule doubles
/// as the safety net. Deliberately not threaded into `cruise.rs` itself
/// (ANISE-free by design, same reasoning as [`build_third_body_tracks`]).
pub(crate) fn resolve_named_body_tracks(
    cfg: &mut MissionConfig,
    almanac: &Almanac,
) -> Result<(), String> {
    let dep_jd_default = require_departure_epoch(cfg).ok().map(epoch_to_jd);
    let Some(seed) = cfg.cruise_seed.as_mut() else { return Ok(()) };
    let duration_s = seed.duration_s;
    for track in seed.body_tracks.iter_mut().filter(|t| t.track.is_empty()) {
        let anise = anise_body(&track.name.to_lowercase()).ok_or_else(|| {
            format!(
                "cruise_seed.body_tracks['{0}']: track is empty (server-resolved form) but '{0}' has \
                 no ANISE coverage on this server — supply the track samples explicitly",
                track.name
            )
        })?;
        let epoch_jd = track.epoch_jd.or(dep_jd_default).ok_or_else(|| {
            format!(
                "cruise_seed.body_tracks['{}']: track is empty (server-resolved form) but neither \
                 epoch_jd nor trajectory.departure_epoch is set — nothing anchors the sampling",
                track.name
            )
        })?;
        let n = ((duration_s / 86_400.0 * 4.0).ceil() as usize).clamp(64, 2000);
        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            let t_s = duration_s * i as f64 / (n - 1) as f64;
            let jd = epoch_jd + t_s / 86_400.0;
            let (r_m, v_mps) = body_state(almanac, EphemerisSource::Anise, Some(anise), &None, jd)
                .ok_or_else(|| {
                    format!("cruise_seed.body_tracks['{}']: ANISE query failed at JD {jd:.3}", track.name)
                })?;
            pts.push(crate::config::CruiseReferencePointConfig { t_s, r_m, v_mps });
        }
        track.track = pts;
        if track.mu_m3s2.is_none() {
            track.mu_m3s2 = body_models::TargetBody::by_name(&track.name).map(|b| b.mu_m3s2);
        }
    }
    Ok(())
}

/// Departure (always ANISE — see `resolve_departure_body`) and arrival body
/// states for one Lambert evaluation at a given departure offset and time of
/// flight.
fn transfer_states(
    almanac: &Almanac,
    dep_body: Option<Body>,
    ephemeris: EphemerisSource,
    arrival_body: Option<Body>,
    kep: &Option<KeplerianElements>,
    dep_jd_base: f64,
    dep_offset_days: f64,
    tof_days: f64,
) -> Option<([f64; 3], [f64; 3], [f64; 3], [f64; 3])> {
    let dep_jd = dep_jd_base + dep_offset_days;
    let arr_jd = dep_jd + tof_days;
    let (r_dep, v_dep) = body_state(almanac, EphemerisSource::Anise, dep_body, &None, dep_jd)?;
    let (r_arr, v_arr) = body_state(almanac, ephemeris, arrival_body, kep, arr_jd)?;
    Some((r_dep, v_dep, r_arr, v_arr))
}

/// Solve a single Lambert arc for the given departure offset and TOF.
fn eval_dv(
    almanac: &Almanac,
    dep_body: Option<Body>,
    ephemeris: EphemerisSource,
    arrival_body: Option<Body>,
    kep: &Option<KeplerianElements>,
    dep_jd_base: f64,
    dep_offset_days: f64,
    tof_days: f64,
) -> Option<TrajectorySolution> {
    let (r_dep, v_dep, r_arr, v_arr) =
        transfer_states(almanac, dep_body, ephemeris, arrival_body, kep, dep_jd_base, dep_offset_days, tof_days)?;
    let arc = LambertArc { r_dep, v_dep, r_arr, v_arr, tof_s: tof_days * 86_400.0, mu: MU_SUN_M3S2 };
    arc.solve().ok()
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Resolve `(ephemeris, arrival_body, departure_body)` from config — the same
/// four-line lookup repeated at most call sites in this file; factored out
/// for the new MC/GA/PSO API paths (Phase 7j) so they don't each re-duplicate
/// it again on top of the existing duplication.
fn resolve_bodies(cfg: &MissionConfig) -> (EphemerisSource, Option<Body>, Option<Body>) {
    let body_name = cfg.target_body.name.to_lowercase();
    let ephemeris = cfg.target_body.ephemeris;
    let body = resolve_body(ephemeris, &body_name);
    let dep_body = resolve_departure_body(cfg);
    (ephemeris, body, dep_body)
}

/// One SOI-candidate or third-body-perturber entry for the propagator
/// (Phase 7k), with its almanac-backed state closure already built. Kept
/// separate from `trajectory_solver::PropagatorBody` because that type's
/// `state_at` is a borrowed `&dyn Fn` that must outlive the `PropagatorBody`
/// built from it — callers hold a `Vec<PropagatorBodyEntry>` alive for the
/// duration of the `propagate*` call and build the borrowed view via
/// [`as_propagator_bodies`] immediately before using it.
pub(crate) struct PropagatorBodyEntry<'a> {
    pub(crate) name: String,
    pub(crate) mu_m3s2: f64,
    pub(crate) soi_radius_m: Option<f64>,
    pub(crate) central_fidelity: Option<trajectory_solver::ZonalFidelity>,
    pub(crate) state_at: Box<dyn Fn(f64) -> (nalgebra::Vector3<f64>, nalgebra::Vector3<f64>) + 'a>,
    /// Mean physical radius [m] — threaded through to `PropagatorBody::radius_m`
    /// for the propagator's collision/close-approach stop. See the design notes.
    pub(crate) radius_m: Option<f64>,
}

pub(crate) fn as_propagator_bodies<'a>(entries: &'a [PropagatorBodyEntry<'a>]) -> Vec<trajectory_solver::PropagatorBody<'a>> {
    entries
        .iter()
        .map(|e| trajectory_solver::PropagatorBody {
            name: &e.name,
            mu_m3s2: e.mu_m3s2,
            soi_radius_m: e.soi_radius_m,
            state_at: e.state_at.as_ref(),
            central_fidelity: e.central_fidelity,
            radius_m: e.radius_m,
        })
        .collect()
}

/// Resolve a target body's zonal-harmonic central-body fidelity from its
/// config, falling back to point-mass (with a warning) when the requested
/// fidelity needs data the body doesn't have: pole orientation (only
/// Earth/Moon/Mars/Jupiter/Europa/Titan have a citable one so far — see
/// `body_models::TargetBody::pole_ra_deg`) or a `SphericalHarmonic` model
/// this propagator doesn't implement yet.
fn build_zonal_fidelity(body: &crate::config::TargetBodyConfig) -> Option<trajectory_solver::ZonalFidelity> {
    use crate::config::GravityModel as CfgGravityModel;
    let (j2, j3, j4) = match body.gravity_model {
        CfgGravityModel::PointMass => return None,
        CfgGravityModel::J2 => (body.j2?, 0.0, 0.0),
        CfgGravityModel::J2J3J4 => (body.j2?, body.j3?, body.j4?),
        CfgGravityModel::SphericalHarmonic => {
            eprintln!(
                "Warning: '{}' requests SphericalHarmonic gravity, which the Layer 1 propagator \
                 doesn't implement yet — using point-mass central gravity instead.",
                body.name,
            );
            return None;
        }
    };
    let (Some(ra), Some(dec)) = (body.pole_ra_deg, body.pole_dec_deg) else {
        eprintln!(
            "Warning: '{}' requests {} gravity fidelity but has no pole orientation data \
             (pole_ra_deg/pole_dec_deg) — propagating its central-body gravity as point-mass. \
             Add pole_ra_deg/pole_dec_deg to [target_body] to enable real fidelity.",
            body.name, body.gravity_model,
        );
        return None;
    };
    Some(trajectory_solver::ZonalFidelity {
        r0_m: body.radius_m, j2, j3, j4,
        pole_ra_rad: ra.to_radians(), pole_dec_rad: dec.to_radians(),
    })
}

/// Builds a `PropagatorBodyEntry` for a catalog body acting as a real SOI
/// candidate, given its real distance from `primary_mu_m3s2`'s body at
/// `dep_jd_base` (already resolved by the caller — this function only
/// assembles the entry, it doesn't decide what a body's primary is).
/// `state_at` always returns the body's position in the REFERENCE frame
/// (heliocentric — see `PropagatorBody`'s contract), regardless of which
/// primary was used to size its SOI radius.
fn primary_relative_entry<'a>(
    almanac: &'a Almanac,
    body_name: String,
    body_mu_m3s2: f64,
    body_radius_m: f64,
    body_anise: Body,
    dep_jd_base: f64,
    dist_from_primary_m: f64,
    primary_mu_m3s2: f64,
    central_fidelity: Option<trajectory_solver::ZonalFidelity>,
) -> PropagatorBodyEntry<'a> {
    let mass_ratio = body_mu_m3s2 / primary_mu_m3s2;
    let soi_radius_m = trajectory_solver::laplace_soi_radius_m(dist_from_primary_m, mass_ratio);
    let name_for_closure = body_name.clone();
    PropagatorBodyEntry {
        name: body_name,
        mu_m3s2: body_mu_m3s2,
        soi_radius_m: Some(soi_radius_m),
        central_fidelity,
        state_at: Box::new(move |t_abs_s: f64| {
            let jd = dep_jd_base + t_abs_s / 86_400.0;
            let (r, v) = body_state(almanac, EphemerisSource::Anise, Some(body_anise), &None, jd)
                .unwrap_or_else(|| panic!("body '{name_for_closure}' ephemeris unavailable at jd={jd}"));
            (nalgebra::Vector3::new(r[0], r[1], r[2]), nalgebra::Vector3::new(v[0], v[1], v[2]))
        }),
        radius_m: Some(body_radius_m),
    }
}

/// Builds the propagator's SOI-candidate (target body, and its real
/// orbital primary if not the Sun) and third-body perturber list for a
/// heliocentric leg (Phase 7k, extended Phase 8h). Only registers
/// candidates when `ephemeris = Anise` — `Keplerian`/`Custom` target bodies
/// have no almanac state to query, so those legs stay point-mass-only
/// exactly as before 7k (no regression, just no new fidelity).
///
/// **Why a body's primary matters**: the Laplace SOI formula needs the
/// body's *actual* primary, not always the Sun. Mars/Jupiter/asteroids
/// orbit the Sun directly, so heliocentric distance is correct for them —
/// unchanged here. But the Moon orbits Earth, Europa orbits Jupiter, Titan
/// orbits Saturn, Phobos/Deimos orbit Mars (`body_models::TargetBody::primary`)
/// — for those, the SOI radius must use the body's distance from *that*
/// primary, and the primary itself must ALSO be registered as a (real,
/// Sun-relative) SOI candidate, so the propagator's existing nested-SOI
/// logic (smallest enclosing SOI wins) correctly resolves "inside both
/// Earth's and the Moon's SOI" to the Moon, not Earth. Bodies outside the
/// catalog (custom TOML bodies) have no known primary and fall back to the
/// Sun-relative convention — a real limitation for non-catalog bodies, not
/// fixed generically (see the design notes).
///
/// `dep_jd` must be the *actual* departure Julian Date for this specific
/// leg (`dep_jd_base + dep_offset_days`) — every `state_at` closure treats
/// leg-local `t_abs_s = 0` as `dep_jd`, which must match the epoch the
/// caller's own `r0`/`v0` were evaluated at. Passing the bare window
/// reference epoch (`dep_jd_base`) instead is a real bug found while
/// verifying this fix on the Moon (Phase 8h): a several-day epoch
/// mismatch put Earth/Moon's registered positions millions of km from
/// where the spacecraft actually started, so the Moon's SOI never
/// registered as entered even on an arc that geometrically passed deep
/// inside it.
fn propagator_body_entries<'a>(
    cfg: &MissionConfig,
    almanac: &'a Almanac,
    dep_jd: f64,
) -> Vec<PropagatorBodyEntry<'a>> {
    let dep_jd_base = dep_jd;
    let mut entries = Vec::new();

    if matches!(cfg.target_body.ephemeris, EphemerisSource::Anise) {
        let name = cfg.target_body.name.to_lowercase();
        if let Some(target_anise_body) = anise_body(&name) {
            if let Some((target_r, _)) = body_state(almanac, EphemerisSource::Anise, Some(target_anise_body), &None, dep_jd_base) {
                let catalog_primary = body_models::TargetBody::by_name(&cfg.target_body.name).and_then(|b| b.primary);

                match catalog_primary {
                    None => {
                        // Orbits the Sun directly (or not in the catalog) —
                        // heliocentric distance is the correct "a" for the
                        // Laplace formula, same as before this fix.
                        let r_dep_norm = (target_r[0].powi(2) + target_r[1].powi(2) + target_r[2].powi(2)).sqrt();
                        entries.push(primary_relative_entry(
                            almanac, cfg.target_body.name.clone(), cfg.target_body.mu_m3s2, cfg.target_body.radius_m, target_anise_body,
                            dep_jd_base, r_dep_norm, MU_SUN_M3S2, build_zonal_fidelity(&cfg.target_body),
                        ));
                    }
                    Some(primary_name) => {
                        // Orbits something other than the Sun (e.g. Moon/Earth) —
                        // size the target's SOI using its real distance from
                        // that primary, not the Sun.
                        //
                        // Deliberately NOT also registering the primary itself
                        // as a separate central-body candidate here, even
                        // though that would be the "textbook nested-SOI"
                        // thing to do (Earth's own real SOI, with the Moon
                        // nested inside it). Tried it (Phase 8h) and it
                        // produces a literal singularity: this architecture's
                        // departure abstraction starts the spacecraft AT the
                        // departure body's own center (`r_dep` = the
                        // departure body's heliocentric position, with the
                        // patched-conic v∞ added) — and the departure body is
                        // overwhelmingly Earth. Registering Earth as a
                        // point-mass candidate exactly where the spacecraft
                        // starts makes `r → 0` in the point-mass formula at
                        // t=0, which is a real `StepSizeUnderflow`, not a
                        // tuning issue. Properly supporting "spacecraft
                        // actually near the departure body" needs the
                        // generic cruise phase tracked separately in
                        // the design notes (Phase 10) — out of scope for a SOI-
                        // radius-value fix. The target body's OWN entry below
                        // is unaffected by this and is exactly what makes a
                        // real lunar/Europa/Titan/Phobos/Deimos flyby
                        // correctly switch central body on arrival.
                        if let Some(primary_catalog) = body_models::TargetBody::by_name(primary_name) {
                            if let Some(primary_anise) = anise_body(&primary_name.to_lowercase()) {
                                if let Some((primary_r, _)) =
                                    body_state(almanac, EphemerisSource::Anise, Some(primary_anise), &None, dep_jd_base)
                                {
                                    let rel_dx = target_r[0] - primary_r[0];
                                    let rel_dy = target_r[1] - primary_r[1];
                                    let rel_dz = target_r[2] - primary_r[2];
                                    let rel_dist = (rel_dx * rel_dx + rel_dy * rel_dy + rel_dz * rel_dz).sqrt();
                                    entries.push(primary_relative_entry(
                                        almanac, cfg.target_body.name.clone(), cfg.target_body.mu_m3s2, cfg.target_body.radius_m, target_anise_body,
                                        dep_jd_base, rel_dist, primary_catalog.mu_m3s2, build_zonal_fidelity(&cfg.target_body),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for third_body_name in &cfg.target_body.third_bodies {
        let Some(anise_b) = anise_body(&third_body_name.to_lowercase()) else { continue };
        let Some(catalog) = body_models::TargetBody::by_name(third_body_name) else {
            eprintln!("Warning: third-body perturber '{third_body_name}' has no mu_m3s2 in the body_models catalog — skipping.");
            continue;
        };
        let mu = catalog.mu_m3s2;
        let perturber_name = third_body_name.clone();
        entries.push(PropagatorBodyEntry {
            name: perturber_name.clone(),
            mu_m3s2: mu,
            soi_radius_m: None,
            central_fidelity: None,
            state_at: Box::new(move |t_abs_s: f64| {
                let jd = dep_jd_base + t_abs_s / 86_400.0;
                let (r, v) = body_state(almanac, EphemerisSource::Anise, Some(anise_b), &None, jd)
                    .unwrap_or_else(|| panic!("third-body perturber '{perturber_name}' ephemeris unavailable at jd={jd}"));
                (nalgebra::Vector3::new(r[0], r[1], r[2]), nalgebra::Vector3::new(v[0], v[1], v[2]))
            }),
            radius_m: Some(catalog.radius_m),
        });
    }

    entries
}

/// Propagate the visualized transfer arc for a given (departure offset, TOF)
/// pair by re-solving the Lambert problem at real ephemeris-derived endpoint
/// states. Shared by the Lambert/GridSearch best-arc, Monte Carlo's
/// reference arc, and the GA/PSO best-individual arc (Phase 7j) — previously
/// duplicated inline in `compute_trajectory`'s Lambert arm only.
fn lambert_visualization_arc(
    cfg: &MissionConfig,
    almanac: &Almanac,
    dep_body: Option<Body>,
    ephemeris: EphemerisSource,
    body: Option<Body>,
    kep: &Option<KeplerianElements>,
    dep_jd_base: f64,
    dep_offset_days: f64,
    tof_days: f64,
) -> Vec<ArcApiPoint> {
    let body_entries = propagator_body_entries(cfg, almanac, dep_jd_base + dep_offset_days);
    let bodies = as_propagator_bodies(&body_entries);
    transfer_states(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_offset_days, tof_days)
        .and_then(|(r_dep, v_dep, r_arr, v_arr)| {
            let lambert = LambertArc { r_dep, v_dep, r_arr, v_arr, tof_s: tof_days * 86_400.0, mu: MU_SUN_M3S2 };
            lambert.solve().ok().map(|sol| propagate_arc(r_dep, sol.v_transfer_dep, MU_SUN_M3S2, tof_days * 86_400.0, &bodies))
        })
        .unwrap_or_default()
}

/// Walk from cwd upward looking for `kernels/<filename>`, without panicking.
fn try_find_kernel(filename: &str) -> Option<String> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join("kernels").join(filename);
        if candidate.exists() {
            return Some(candidate.to_string_lossy().into_owned());
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Satellite SPK kernels chain-loaded on top of `de440s.bsp`, each optional —
/// missing one only drops ANISE coverage for that system's moons
/// (`anise_body`'s `Phobos`/`Deimos`/`Europa`/`Titan` arms), it never fails
/// `load_almanac` itself. Filenames/sizes/source: NAIF generic_kernels
/// (`https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/`).
const SATELLITE_KERNELS: &[(&str, &str)] = &[
    ("mar099s.bsp", "Phobos/Deimos (Mars satellites, reduced/short-span, ~68 MB)"),
    ("jup365.bsp", "Europa (Jupiter satellites — only kernel with individual moon states, ~1.1 GB)"),
    ("sat441.bsp", "Titan (Saturn satellites — only kernel with individual moon states, ~660 MB)"),
];

/// Process-wide cache, built once, shared by every caller (found
/// via a real live server session that `load_almanac()` was being called
/// dozens of times — once per `/api/optimize` job, once per pruning/backfit
/// sub-search, once per `/api/bodies/{name}/state` request, etc. — with NO
/// caching at all, each call re-reading and re-parsing all four kernel
/// files (~1.83 GB: `de440s.bsp` + the three satellite kernels) from disk
/// from scratch. 32 calls in one session meant ~58 GB of redundant I/O —
/// easily tens of minutes of wall-clock cost, invisible to a caller watching
/// a live step-stream (kernel loading isn't a search generation, so no
/// progress indicator moves during it, making a search look "stuck" when
/// it's actually burning most of its time on disk I/O). `Almanac`'s loaded
/// context is immutable after construction (already established elsewhere
/// in this codebase, `mga.rs`'s `Almanac: Sync` proof) — safe to share via
/// `Arc` across every concurrent caller, same `OnceLock`-cache pattern
/// `mga.rs::ACTIVE_EPH_CACHE` already uses for the analogous ephemeris-query
/// cost problem.
static ALMANAC_CACHE: std::sync::OnceLock<Option<std::sync::Arc<Almanac>>> = std::sync::OnceLock::new();

/// Load (once per process) or return the cached ANISE almanac. See
/// [`ALMANAC_CACHE`]'s doc comment for why this exists. `None` if kernel
/// loading fails — cached too, so a caller doesn't pay the failed-lookup
/// cost (directory walk + missing-file errors) repeatedly either.
pub(crate) fn load_almanac() -> Option<std::sync::Arc<Almanac>> {
    ALMANAC_CACHE.get_or_init(load_almanac_uncached).clone()
}

fn load_almanac_uncached() -> Option<std::sync::Arc<Almanac>> {
    load_almanac_inner().map(std::sync::Arc::new)
}

fn load_almanac_inner() -> Option<Almanac> {
    let path = match try_find_kernel("de440s.bsp") {
        Some(p) => p,
        None => {
            eprintln!("Error: de440s.bsp not found in any kernels/ directory.");
            eprintln!("  Download: https://public-data.nyxspace.com/anise/de440s.bsp");
            eprintln!("  Place at: <workspace-root>/kernels/de440s.bsp");
            return None;
        }
    };
    let mut almanac = match Almanac::new(&path) {
        Ok(a) => {
            println!("Loaded ANISE kernel: {path}");
            a
        }
        Err(e) => {
            eprintln!("Error loading de440s.bsp from '{path}': {e}");
            return None;
        }
    };

    for (filename, coverage) in SATELLITE_KERNELS {
        let Some(sat_path) = try_find_kernel(filename) else {
            eprintln!(
                "Note: {filename} not found in any kernels/ directory — {coverage} unavailable. \
                 Download: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/{filename}"
            );
            continue;
        };
        almanac = match almanac.load(&sat_path) {
            Ok(a) => {
                println!("Loaded ANISE kernel: {sat_path} ({coverage})");
                a
            }
            Err(e) => {
                eprintln!("Error loading {filename} from '{sat_path}': {e} — {coverage} unavailable.");
                // `Almanac::load` consumes `self` even on failure (the ANISE error
                // variant doesn't hand the receiver back), so the in-progress
                // almanac — including any satellite kernels loaded in an earlier
                // iteration of this loop — is unrecoverable here. Rebuild fresh
                // from de440s.bsp and keep trying the remaining satellite kernels
                // rather than aborting the whole load over one bad file.
                match Almanac::new(&path) {
                    Ok(a) => a,
                    Err(e2) => {
                        eprintln!("Error re-loading de440s.bsp from '{path}': {e2}");
                        return None;
                    }
                }
            }
        };
    }

    Some(almanac)
}

pub(crate) fn keplerian_from_cfg(cfg: &MissionConfig) -> Option<KeplerianElements> {
    cfg.target_body.keplerian_orbit.as_ref().map(|k| KeplerianElements {
        sma_m:              k.sma_au * AU_M,
        eccentricity:       k.eccentricity,
        inclination_rad:    k.inclination_deg.to_radians(),
        raan_rad:           k.raan_deg.to_radians(),
        aop_rad:            k.aop_deg.to_radians(),
        mean_anomaly_0_rad: k.mean_anomaly_deg.to_radians(),
        epoch_jd:           k.epoch_jd,
        mu_central:         MU_SUN_M3S2,
    })
}

pub(crate) fn anise_body(name: &str) -> Option<Body> {
    match name {
        "earth"   => Some(Body::Earth),
        "moon"    => Some(Body::Moon),
        "mars"    => Some(Body::Mars),
        "venus"   => Some(Body::Venus),
        "jupiter" => Some(Body::Jupiter),
        "saturn"  => Some(Body::Saturn),
        "mercury" => Some(Body::Mercury),
        // Barycenters already in the base de440s.bsp — no extra kernel needed.
        // Added previously missing entirely (not a real coverage
        // gap — anise::constants::frames already has
        // URANUS_BARYCENTER_J2000/NEPTUNE_BARYCENTER_J2000, same as
        // Jupiter/Saturn), which silently made every MGA chromosome
        // targeting Neptune infeasible (ephemeris lookup for the target body
        // always failed, so every fitness evaluation returned None).
        "uranus"  => Some(Body::Uranus),
        "neptune" => Some(Body::Neptune),
        // Require the relevant satellite SPK chain-loaded on top of de440s.bsp
        // (mar099s.bsp / jup365.bsp / sat441.bsp — see `load_almanac`). If that
        // kernel isn't present, `state_for`'s `body_state_heliocentric` call
        // returns a normal `Err` (500 with the ANISE error message), not a panic.
        "phobos"  => Some(Body::Phobos),
        "deimos"  => Some(Body::Deimos),
        "europa"  => Some(Body::Europa),
        "titan"   => Some(Body::Titan),
        // Everything else — including Sun (heliocentric origin, no lookup needed)
        // and small bodies (Apophis, Bennu, Eros, Didymos, Pluto…) — returns
        // None silently. The caller's 422 response already describes the
        // coverage gap to the client; a server-side eprintln adds only noise.
        _ => None,
    }
}

/// Satellite kernel filename required for a moon covered by `anise_body`, if
/// any — `None` for bodies covered directly by the always-present
/// `de440s.bsp` (or not ANISE-covered at all). Used to report `anise_covered`
/// (and the `/state` 422) truthfully based on what's actually deployed on
/// this server, not just whether the `Body` enum has a variant for it.
fn required_satellite_kernel(name: &str) -> Option<&'static str> {
    match name {
        "phobos" | "deimos" => Some("mar099s.bsp"),
        "europa" => Some("jup365.bsp"),
        "titan" => Some("sat441.bsp"),
        _ => None,
    }
}

/// True if `/api/bodies/{name}/state` can actually answer for this body on
/// this server right now — i.e. `anise_body` covers it AND, for the four
/// moons that need a satellite kernel beyond `de440s.bsp`, that kernel file
/// is present.
pub(crate) fn anise_body_available(name: &str) -> bool {
    let lower = name.to_lowercase();
    anise_body(&lower).is_some()
        && required_satellite_kernel(&lower).is_none_or(|k| try_find_kernel(k).is_some())
}

/// Resolve the arrival body's ANISE frame only when `ephemeris = "Anise"` —
/// `Keplerian`/`Custom` sources never touch the almanac, so they must not
/// trigger `anise_body`'s "not available via ANISE" warning.
fn resolve_body(ephemeris: EphemerisSource, name: &str) -> Option<Body> {
    match ephemeris {
        EphemerisSource::Anise => anise_body(name),
        EphemerisSource::Keplerian | EphemerisSource::Custom => None,
    }
}

/// Resolve `[trajectory].departure_body`, defaulting to Earth when unset —
/// every existing config that doesn't set the field keeps departing from
/// Earth exactly as before. Departure is always looked up via ANISE (no
/// Keplerian-departure support yet — every catalog body `anise_body` knows
/// about is ANISE-covered).
fn resolve_departure_body(cfg: &MissionConfig) -> Option<Body> {
    let name = cfg.trajectory.departure_body.as_str().to_lowercase();
    anise_body(&name)
}

/// Extract `[f64; 3]` from any indexable value (e.g. `nalgebra::SVector<f64, 3>`).
///
/// This avoids a direct nalgebra import: `FrameVec::inner` is `SVector<f64, 3>`,
/// which implements `Index<usize, Output = f64>`, so the conversion works without
/// naming the nalgebra type.
#[inline]
fn sv3_to_arr<V: std::ops::Index<usize, Output = f64>>(v: V) -> [f64; 3] {
    [v[0], v[1], v[2]]
}

/// Parse "YYYY-MM-DDTHH:MM:SS UTC" (or without the trailing " UTC").
pub fn parse_epoch(s: &str) -> Result<Epoch, String> {
    let s = s.trim().trim_end_matches(" UTC").trim();
    let (date_part, time_part) = s
        .split_once('T')
        .ok_or_else(|| format!("expected YYYY-MM-DDTHH:MM:SS, got '{s}'"))?;

    let d: Vec<i32> = date_part
        .split('-')
        .map(|x| x.parse::<i32>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("date parse: {e}"))?;

    let t: Vec<u8> = time_part
        .split(':')
        .map(|x| x.parse::<u8>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("time parse: {e}"))?;

    if d.len() < 3 || t.len() < 3 {
        return Err(format!("incomplete date/time in '{s}'"));
    }
    Ok(Epoch::from_gregorian_utc(d[0], d[1] as u8, d[2] as u8, t[0], t[1], t[2], 0))
}

/// `Epoch` → Julian Date (UTC).  Unix epoch = JD 2 440 587.5
pub fn epoch_to_jd(epoch: Epoch) -> f64 {
    epoch.to_unix_seconds() / 86_400.0 + 2_440_587.5
}

/// Julian Date (UTC) → `Epoch`.
fn jd_to_epoch(jd: f64) -> Epoch {
    Epoch::from_unix_seconds((jd - 2_440_587.5) * 86_400.0)
}

/// Resolve `[trajectory].departure_epoch` to an `Epoch`, with a clear,
/// actionable error when it's missing. Every real-ephemeris solver needs a
/// concrete epoch to query ANISE at — there is no physically meaningful
/// default (silently picking "now" would produce a real ΔV/TOF result at an
/// epoch the user never chose, with no indication anything was defaulted).
/// Previously each call site defaulted the missing case to `""` before
/// parsing, producing the confusing "expected YYYY-MM-DDTHH:MM:SS, got ''"
/// instead of saying what's actually wrong.
pub(crate) fn require_departure_epoch(cfg: &MissionConfig) -> Result<Epoch, String> {
    let Some(s) = cfg.trajectory.departure_epoch.as_deref() else {
        return Err(format!(
            "trajectory.departure_epoch is required for solver \"{}\" — set it to an ISO 8601 \
             string, e.g. \"2026-09-15T00:00:00 UTC\"",
            cfg.trajectory.solver,
        ));
    };
    parse_epoch(s).map_err(|e| format!("trajectory.departure_epoch '{s}': {e}"))
}

/// Calendar string from JD for display (uses hifitime Display impl).
fn epoch_approx_str(jd: f64) -> String {
    format!("{}", jd_to_epoch(jd))
}

// ── API result types ──────────────────────────────────────────────────────────

/// One sampled point along a propagated transfer arc, for 3D visualization.
/// Mirrors `trajectory_solver::ArcPoint` — kept separate so `serde` stays out
/// of the shared physics crate.
#[allow(dead_code)]
#[derive(Debug, Clone, serde::Serialize)]
pub struct ArcApiPoint {
    /// Time since departure [s]
    pub t_s: f64,
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
    /// Which body was gravitationally central at this point (Phase 7k
    /// SOI-switching) — `"Sun"` when the spacecraft was outside every
    /// candidate body's SOI (the reference frame), else the name of the
    /// `PropagatorBody` whose SOI it was inside.
    pub central_body: String,
    /// For MGA results: which leg (0-indexed) this point belongs to.
    /// `null` for single-leg GA/PSO results.
    pub leg_idx: Option<u32>,
    /// Real velocity at this point, same frame as x_m/y_m/z_m.
    /// The propagator (`PropagatedPoint::v_mps`) already computes this at
    /// every sample -- it was simply never serialized before, forcing
    /// downstream consumers (Phase 03's cruise reference) to approximate it
    /// via finite-differencing adjacent positions instead of using the real
    /// value. Always `Some` as of GA/PSO's `arc`, every
    /// `pre_departure_orbit_arc`/`post_capture_orbit_arc` (both search
    /// methods), AND MGA's own multi-leg `arc` (`mga::MgaArcPoint` now
    /// keeps the velocity its sampling always computed). Kept `Option`
    /// only so results serialized before the respective field landed still
    /// deserialize.
    pub vx_mps: Option<f64>,
    pub vy_mps: Option<f64>,
    pub vz_mps: Option<f64>,
}

/// Resolve a `trajectory_solver::PropagatedPoint`'s `central_body_index`
/// against the same `bodies` slice that was passed to `propagate()` for that
/// call, into the owned body name `ArcApiPoint` carries across the API
/// boundary. `None` (outside every candidate's SOI) maps to `"Sun"`, the
/// always-available reference-frame fallback.
pub(crate) fn arc_to_api(
    points: Vec<trajectory_solver::PropagatedPoint>,
    bodies: &[trajectory_solver::PropagatorBody],
) -> Vec<ArcApiPoint> {
    points
        .into_iter()
        .map(|p| ArcApiPoint {
            t_s: p.t_s,
            x_m: p.r_m.x,
            y_m: p.r_m.y,
            z_m: p.r_m.z,
            central_body: match p.central_body_index {
                Some(i) => bodies[i].name.to_string(),
                None => "Sun".to_string(),
            },
            leg_idx: None,
            vx_mps: Some(p.v_mps.x),
            vy_mps: Some(p.v_mps.y),
            vz_mps: Some(p.v_mps.z),
        })
        .collect()
}

/// Propagate a visualized transfer arc at the standard ~60-point density,
/// with the given SOI-candidate/third-body perturber list (Phase 7k — pass
/// `&[]` for a point-mass-only leg, e.g. the synthetic-orientation Hohmann
/// arcs). See the design notes "Layer 1 Propagator Design".
pub(crate) fn propagate_arc(r0: [f64; 3], v0: [f64; 3], mu: f64, tof_s: f64, bodies: &[trajectory_solver::PropagatorBody]) -> Vec<ArcApiPoint> {
    propagate_arc_n(r0, v0, mu, tof_s, ARC_SAMPLE_COUNT, bodies)
}

/// Propagate a visualized transfer arc at a caller-chosen point density.
/// Monte Carlo trajectories use a much sparser count than the standard
/// best-arc visualization (`MC_TRAJ_SAMPLE_COUNT`, Phase 8i/7h) — payload
/// scales with sample count, and a dispersion cloud needs many trajectories,
/// not one finely-sampled one.
pub(crate) fn propagate_arc_n(r0: [f64; 3], v0: [f64; 3], mu: f64, tof_s: f64, n_points: f64, bodies: &[trajectory_solver::PropagatorBody]) -> Vec<ArcApiPoint> {
    let r0 = nalgebra::Vector3::new(r0[0], r0[1], r0[2]);
    let v0 = nalgebra::Vector3::new(v0[0], v0[1], v0[2]);
    let sample_dt_s = (tof_s / n_points).max(1.0);
    arc_to_api(propagate(r0, v0, 0.0, tof_s, mu, bodies, sample_dt_s, PROPAGATOR_RTOL, PROPAGATOR_ATOL), bodies)
}

/// Analytical Hohmann transfer result — interplanetary cruise or body-centric LOI.
#[allow(dead_code)]
#[derive(Debug, serde::Serialize)]
pub struct HohmannApiResult {
    /// "cruise_heliocentric" or "body_centric_loi"
    pub label: String,
    pub r1_km: f64,
    pub r2_km: f64,
    pub tof_s: f64,
    pub dv_dep_ms: f64,
    pub dv_arr_ms: f64,
    pub dv_total_ms: f64,
    pub c3_km2s2: f64,
    /// Launch vehicle feasibility check against `c3_km2s2`, only for the
    /// `"cruise_heliocentric"` entry (the departure leg) — always `None` for
    /// `"body_centric_loi"` (an arrival/capture burn, not a departure) or
    /// when `[trajectory].departure_body` isn't Earth, no vehicle is
    /// configured, or the configured name isn't in the catalog. Added
    /// see the design notes.
    pub launch_vehicle_check: Option<LaunchVehicleCheckApiResult>,
    /// Sampled transfer-ellipse path, for 3D visualization. NOTE: the absolute
    /// orientation is synthetic (departure placed along +x) — Hohmann's
    /// analytical model has no real inertial-frame context, unlike the
    /// Lambert/DiffCorrection arcs below, which use real departure/arrival
    /// position vectors. Shape and timing are physically correct; the
    /// orientation is not tied to actual heliocentric/body-fixed directions.
    pub arc: Vec<ArcApiPoint>,
}

/// Best arc from porkchop grid or differential correction.
#[allow(dead_code)]
#[derive(Debug, serde::Serialize)]
pub struct BestArcApiResult {
    pub dep_offset_days: f64,
    pub dep_jd: f64,
    pub tof_days: f64,
    pub dv_dep_ms: f64,
    pub dv_arr_ms: f64,
    pub dv_total_ms: f64,
    pub c3_km2s2: f64,
    pub v_inf_arr_ms: f64,
    /// Tsiolkovsky ΔV budget [m/s], if propulsion is configured. Pool-aware
    /// as of this is the *onboard* propellant pool only (see
    /// `onboard_dv_required_ms`) — it never includes launcher-provided ΔV.
    pub dv_budget_ms: Option<f64>,
    /// `dv_budget_ms - onboard_dv_required_ms(...)` — positive = onboard
    /// margin remaining. Distinct from `launch_vehicle_check`'s own
    /// feasibility, which is a separate pool entirely. Same number as
    /// `dv_ledger.budget_margin_ms` (kept for backward compatibility).
    pub budget_margin_ms: Option<f64>,
    /// Two-pool ΔV ledger (Phase 14e) — `None` only when the departure body
    /// isn't in the `body_models` catalog.
    pub dv_ledger: Option<DvLedgerApiResult>,
    /// Launch vehicle feasibility check against this arc's departure C3 —
    /// `None` whenever `[trajectory].departure_body` isn't Earth, no vehicle
    /// is configured, or the configured name isn't in the catalog. Added
    /// see the design notes.
    pub launch_vehicle_check: Option<LaunchVehicleCheckApiResult>,
    /// Sampled transfer-arc path (real heliocentric departure/arrival
    /// positions — not synthetic, unlike the Hohmann case above).
    pub arc: Vec<ArcApiPoint>,
}

/// One point from the porkchop grid.
#[allow(dead_code)]
#[derive(Debug, serde::Serialize)]
pub struct PorkchopApiPoint {
    pub dep_offset_days: f64,
    pub tof_days: f64,
    pub dv_total_ms: f64,
    pub dv_dep_ms: f64,
    pub dv_arr_ms: f64,
    pub c3_km2s2: f64,
    pub v_inf_arr_ms: f64,
}

/// One Monte Carlo sample, with a sparse propagated trajectory only for the
/// stride-selected subset (`MC_TRAJ_COUNT` of the full scatter) — mirrors
/// `monte_carlo_trajectories.csv`'s CLI output. Samples outside that subset
/// still appear in [`MonteCarloApiResult::samples`] but with `trajectory: None`,
/// matching `monte_carlo.csv`'s scalar-only rows.
#[allow(dead_code)]
#[derive(Debug, serde::Serialize)]
pub struct MonteCarloSampleApiResult {
    pub dep_offset_days: f64,
    pub tof_days: f64,
    pub valid: bool,
    pub dv_total_ms: Option<f64>,
    pub c3_km2s2: Option<f64>,
    pub v_inf_arr_ms: Option<f64>,
    /// Sparse propagated trajectory (`MC_TRAJ_SAMPLE_COUNT` points), only for
    /// the stride-selected subset — never the full scatter, per the design notes
    /// "Monte Carlo never gets the dense stream" rule.
    pub trajectory: Option<Vec<ArcApiPoint>>,
}

/// Local Gaussian-scatter robustness probe around the porkchop best point
/// (Phase 2 scope — not a full state/hardware Monte Carlo). Mirrors the CLI's
/// `monte_carlo.csv` (full scatter, scalar) + `monte_carlo_trajectories.csv`
/// (sparse subset with trajectories), combined into `samples`.
#[allow(dead_code)]
#[derive(Debug, serde::Serialize)]
pub struct MonteCarloApiResult {
    pub n_samples: usize,
    pub n_valid: usize,
    pub sigma_dep_offset_days: f64,
    pub sigma_tof_days: f64,
    pub mean_dv_total_ms: f64,
    pub std_dv_total_ms: f64,
    pub min_dv_total_ms: f64,
    pub max_dv_total_ms: f64,
    /// Porkchop-best ΔV total this scatter is centered on, for comparison.
    pub reference_dv_total_ms: f64,
    /// All samples (scalar fields always populated when valid); only the
    /// stride-selected subset carries a `trajectory`.
    pub samples: Vec<MonteCarloSampleApiResult>,
}

/// Best individual/particle from a GA or PSO search (Phase 8j solvers, wired
/// into the API at Phase 7j). Shared shape for both — they search the same
/// (departure offset, TOF) space with the same fitness metric
/// (`ga_fitness`), differing only in algorithm family.
#[allow(dead_code)]
#[derive(Debug, serde::Serialize)]
pub struct OptimizerApiResult {
    /// "GA" or "PSO"
    pub solver: String,
    pub dep_offset_days: f64,
    pub dep_jd: f64,
    pub tof_days: f64,
    /// Fitness value the search minimized: C3 [km²/s²] for Flyby, ΔV total
    /// [m/s] otherwise — same metric `best_point_for_objective` selects on.
    pub fitness: f64,
    pub fitness_units: String,
    pub c3_km2s2: f64,
    pub v_inf_arr_ms: f64,
    pub dv_total_ms: f64,
    /// Onboard propellant pool only (see `onboard_dv_required_ms`) — never
    /// includes launcher-provided ΔV.
    pub dv_budget_ms: Option<f64>,
    /// Same number as `dv_ledger.budget_margin_ms` (kept for backward compatibility).
    pub budget_margin_ms: Option<f64>,
    /// Two-pool ΔV ledger (Phase 14e) — `None` only when the departure body
    /// isn't in the `body_models` catalog.
    pub dv_ledger: Option<DvLedgerApiResult>,
    /// Launch vehicle feasibility check against `c3_km2s2` — `None` whenever
    /// `[trajectory].departure_body` isn't Earth, no vehicle is configured,
    /// or the configured name isn't in the catalog. See
    /// the design notes "Launch Vehicle Selection."
    pub launch_vehicle_check: Option<LaunchVehicleCheckApiResult>,
    /// Best-fitness-so-far per generation (GA) or iteration (PSO).
    pub convergence: Vec<f64>,
    pub arc: Vec<ArcApiPoint>,
}

/// Full trajectory design result returned by [`compute_trajectory`].
#[allow(dead_code)]
#[derive(Debug, serde::Serialize)]
pub struct TrajectoryApiResult {
    pub solver: String,
    pub warnings: Vec<String>,
    /// Hohmann analytical estimates (0–2 entries: cruise + body-centric LOI).
    pub hohmann: Vec<HohmannApiResult>,
    /// Best arc (porkchop minimum or diff-correction result), if available.
    pub best_arc: Option<BestArcApiResult>,
    /// Full porkchop grid (empty for Hohmann/DiffCorrection solvers).
    pub porkchop: Vec<PorkchopApiPoint>,
    /// Populated only for the `MonteCarlo` solver (Phase 7j).
    pub monte_carlo: Option<MonteCarloApiResult>,
    /// Populated only for the `GA`/`PSO` solvers (Phase 7j).
    pub optimizer: Option<OptimizerApiResult>,
}

// ── API entry point (no I/O, no side-effects) ─────────────────────────────────

/// Run the trajectory design stage and return structured data without printing
/// or writing files. Used by the `/api/design/trajectory` HTTP endpoint.
#[allow(dead_code)]
pub fn compute_trajectory(cfg: &MissionConfig) -> Result<TrajectoryApiResult, String> {
    match cfg.trajectory.solver {
        TrajectorySolver::Hohmann => Ok(TrajectoryApiResult {
            solver: "Hohmann".into(),
            warnings: Vec::new(),
            hohmann: hohmann_api(cfg),
            best_arc: None,
            porkchop: Vec::new(),
            monte_carlo: None,
            optimizer: None,
        }),

        TrajectorySolver::Lambert
        | TrajectorySolver::GridSearch
        | TrajectorySolver::LambertThenDiffCorrect => {
            let almanac = load_almanac()
                .ok_or_else(|| "Could not load ANISE kernel — de440s.bsp not found".to_string())?;
            let kep = keplerian_from_cfg(cfg);
            if matches!(cfg.target_body.ephemeris, EphemerisSource::Keplerian) && kep.is_none() {
                return Err(
                    "ephemeris = \"Keplerian\" requires a [target_body.keplerian_orbit] section"
                        .into(),
                );
            }
            let (points, dep_jd_base) = porkchop_data(cfg, &almanac, &kep)?;
            let best = best_point_for_objective(&points, cfg.mission.objective)
                .ok_or_else(|| "No valid Lambert solutions found".to_string())?;

            let best_arc = build_best_arc_api(cfg, &almanac, &kep, dep_jd_base, best);
            let porkchop = porkchop_points_to_api(cfg, &points);
            Ok(TrajectoryApiResult {
                solver: cfg.trajectory.solver.to_string(),
                warnings: Vec::new(),
                hohmann: Vec::new(),
                best_arc: Some(best_arc),
                porkchop,
                monte_carlo: None,
                optimizer: None,
            })
        }

        TrajectorySolver::DiffCorrection => {
            let almanac = load_almanac()
                .ok_or_else(|| "Could not load ANISE kernel — de440s.bsp not found".to_string())?;
            let kep = keplerian_from_cfg(cfg);
            let dv_budget = dv_budget_ms(cfg).ok_or_else(|| {
                "DiffCorrection solver requires [spacecraft.propulsion] (isp_s) to set a ΔV target"
                    .to_string()
            })?;
            let best_arc = diff_correct_api(cfg, &almanac, &kep, dv_budget)?;
            Ok(TrajectoryApiResult {
                solver: "DiffCorrection".into(),
                warnings: Vec::new(),
                hohmann: Vec::new(),
                best_arc: Some(best_arc),
                porkchop: Vec::new(),
                monte_carlo: None,
                optimizer: None,
            })
        }

        TrajectorySolver::MonteCarlo => {
            let almanac = load_almanac()
                .ok_or_else(|| "Could not load ANISE kernel — de440s.bsp not found".to_string())?;
            let kep = keplerian_from_cfg(cfg);
            if matches!(cfg.target_body.ephemeris, EphemerisSource::Keplerian) && kep.is_none() {
                return Err(
                    "ephemeris = \"Keplerian\" requires a [target_body.keplerian_orbit] section"
                        .into(),
                );
            }
            let (points, dep_jd_base) = porkchop_data(cfg, &almanac, &kep)?;
            let best = best_point_for_objective(&points, cfg.mission.objective)
                .ok_or_else(|| "No valid Lambert solutions found".to_string())?;

            let best_arc = build_best_arc_api(cfg, &almanac, &kep, dep_jd_base, best);
            let porkchop = porkchop_points_to_api(cfg, &points);
            let monte_carlo = monte_carlo_api(cfg, &almanac, &kep, dep_jd_base, best);
            Ok(TrajectoryApiResult {
                solver: "MonteCarlo".into(),
                warnings: Vec::new(),
                hohmann: Vec::new(),
                best_arc: Some(best_arc),
                porkchop,
                monte_carlo: Some(monte_carlo),
                optimizer: None,
            })
        }

        TrajectorySolver::GA | TrajectorySolver::PSO => {
            let almanac = load_almanac()
                .ok_or_else(|| "Could not load ANISE kernel — de440s.bsp not found".to_string())?;
            let kep = keplerian_from_cfg(cfg);
            if matches!(cfg.target_body.ephemeris, EphemerisSource::Keplerian) && kep.is_none() {
                return Err(
                    "ephemeris = \"Keplerian\" requires a [target_body.keplerian_orbit] section"
                        .into(),
                );
            }
            let solver_name = if matches!(cfg.trajectory.solver, TrajectorySolver::GA) { "GA" } else { "PSO" };
            let optimizer = if solver_name == "GA" {
                ga_api(cfg, &almanac, &kep)?
            } else {
                pso_api(cfg, &almanac, &kep)?
            };
            Ok(TrajectoryApiResult {
                solver: solver_name.into(),
                warnings: Vec::new(),
                hohmann: Vec::new(),
                best_arc: None,
                porkchop: Vec::new(),
                monte_carlo: None,
                optimizer: Some(optimizer),
            })
        }

        TrajectorySolver::SA | TrajectorySolver::ManifoldStitch | TrajectorySolver::WSB => Err(format!(
            "Solver '{}' is not available via the API \
             (SA is in OptimizationProblems; ManifoldStitch/WSB are in AstroProbs/LunarTrajectories)",
            cfg.trajectory.solver
        )),
    }
}

/// Map porkchop grid points to their API shape — shared by the
/// Lambert/GridSearch/LambertThenDiffCorrect arm and the MonteCarlo arm
/// (Phase 7j), both of which surface the full evaluated grid. `dv_arr_ms`/
/// `dv_total_ms` are repriced per `cfg.mission.objective` via
/// [`objective_priced_arrival`] (frontend backlog #14/#15) — `PorkchopPoint`
/// itself stays the crate's raw, objective-agnostic output.
fn porkchop_points_to_api(cfg: &MissionConfig, points: &[PorkchopPoint]) -> Vec<PorkchopApiPoint> {
    points
        .iter()
        .map(|p| {
            let (dv_arr_ms, dv_total_ms) = objective_priced_arrival(cfg, p.dv_dep_ms, p.v_inf_arr_ms);
            PorkchopApiPoint {
                dep_offset_days: p.dep_offset_days,
                tof_days: p.tof_days,
                dv_total_ms,
                dv_dep_ms: p.dv_dep_ms,
                dv_arr_ms,
                c3_km2s2: p.c3_km2s2,
                v_inf_arr_ms: p.v_inf_arr_ms,
            }
        })
        .collect()
}

/// Build the porkchop-best [`BestArcApiResult`] — shared by the
/// Lambert/GridSearch/LambertThenDiffCorrect arm and the MonteCarlo arm
/// (Phase 7j; MonteCarlo's scatter is centered on this same reference point).
fn build_best_arc_api(
    cfg: &MissionConfig,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
    dep_jd_base: f64,
    best: &PorkchopPoint,
) -> BestArcApiResult {
    let (ephemeris, body, dep_body) = resolve_bodies(cfg);
    let arc = lambert_visualization_arc(
        cfg, almanac, dep_body, ephemeris, body, kep, dep_jd_base, best.dep_offset_days, best.tof_days,
    );
    let (dv_arr_ms, dv_total_ms) = objective_priced_arrival(cfg, best.dv_dep_ms, best.v_inf_arr_ms);
    let ledger = narrowing_dv_ledger(cfg, &cfg.trajectory.departure_body, best.dv_dep_ms, best.v_inf_arr_ms);
    BestArcApiResult {
        dep_offset_days: best.dep_offset_days,
        dep_jd: dep_jd_base + best.dep_offset_days,
        tof_days: best.tof_days,
        dv_dep_ms: best.dv_dep_ms,
        dv_arr_ms,
        dv_total_ms,
        c3_km2s2: best.c3_km2s2,
        v_inf_arr_ms: best.v_inf_arr_ms,
        dv_budget_ms: dv_budget_ms(cfg),
        budget_margin_ms: ledger.as_ref().and_then(|l| l.budget_margin_ms),
        dv_ledger: ledger,
        launch_vehicle_check: compute_launch_vehicle_check(cfg, best.c3_km2s2),
        arc,
    }
}

#[allow(dead_code)]
fn hohmann_api(cfg: &MissionConfig) -> Vec<HohmannApiResult> {
    use crate::config::AtmosphereModel;
    let mut results = Vec::new();

    let r2_m = match cfg.target_body.ephemeris {
        EphemerisSource::Keplerian => cfg
            .target_body
            .keplerian_orbit
            .as_ref()
            .map(|k| k.sma_au * AU_M)
            .unwrap_or(AU_M),
        EphemerisSource::Anise => match cfg.target_body.name.to_lowercase().as_str() {
            "mars"    => 1.524  * AU_M,
            "venus"   => 0.7233 * AU_M,
            "jupiter" => 5.203  * AU_M,
            "saturn"  => 9.537  * AU_M,
            _         => AU_M,
        },
        EphemerisSource::Custom => AU_M,
    };
    let r1_m = AU_M;
    if (r2_m - r1_m).abs() > 1e9 {
        let solver = HohmannSolver { mu_m3s2: MU_SUN_M3S2, r1_m, r2_m };
        if let Ok(s) = solver.solve() {
            let r0 = [r1_m, 0.0, 0.0];
            let v0 = [0.0, s.v_transfer_dep[0], 0.0];
            // Synthetic orientation (departure placed along +x, no real
            // epoch correspondence) — SOI-candidate registration needs a
            // real departure epoch to query ephemeris, so this stays
            // point-mass-only, same as before 7k (see `HohmannApiResult.arc` doc comment).
            let arc = propagate_arc(r0, v0, MU_SUN_M3S2, s.tof_s, &[]);
            results.push(HohmannApiResult {
                label: "cruise_heliocentric".into(),
                r1_km: r1_m / 1e3,
                r2_km: r2_m / 1e3,
                tof_s: s.tof_s,
                dv_dep_ms: s.dv_departure_ms,
                dv_arr_ms: s.dv_arrival_ms,
                dv_total_ms: s.dv_total_ms,
                c3_km2s2: s.c3_km2s2,
                launch_vehicle_check: compute_launch_vehicle_check(cfg, s.c3_km2s2),
                arc,
            });
        }
    }

    if let Some(cap) = &cfg.trajectory.capture {
        if let Some(r_cap) = cap.target_orbit_radius_m {
            let has_atm = !matches!(cfg.target_body.atmosphere, AtmosphereModel::None);
            let r_park = if has_atm {
                cfg.target_body.radius_m + 200_000.0
            } else {
                r_cap * 1.5
            };
            let solver = HohmannSolver {
                mu_m3s2: cfg.target_body.mu_m3s2,
                r1_m: r_park,
                r2_m: r_cap,
            };
            if let Ok(s) = solver.solve() {
                let r0 = [r_park, 0.0, 0.0];
                let v0 = [0.0, s.v_transfer_dep[0], 0.0];
                // Already body-centric by construction (no SOI switching
                // needed) — adding the target's own zonal fidelity here is a
                // smaller, separate enhancement than 7k's SOI-switching
                // scope; left point-mass for now, same as before 7k.
                let arc = propagate_arc(r0, v0, cfg.target_body.mu_m3s2, s.tof_s, &[]);
                results.push(HohmannApiResult {
                    label: "body_centric_loi".into(),
                    r1_km: r_park / 1e3,
                    r2_km: r_cap / 1e3,
                    tof_s: s.tof_s,
                    dv_dep_ms: s.dv_departure_ms,
                    dv_arr_ms: s.dv_arrival_ms,
                    dv_total_ms: s.dv_total_ms,
                    c3_km2s2: s.c3_km2s2,
                    // Arrival/capture burn, not a departure — never checked
                    // against a launch vehicle.
                    launch_vehicle_check: None,
                    arc,
                });
            }
        }
    }

    results
}

#[allow(dead_code)]
fn porkchop_data(
    cfg: &MissionConfig,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
) -> Result<(Vec<PorkchopPoint>, f64), String> {
    let cruise = cfg.trajectory.cruise.as_ref();
    let tof_min = cruise.and_then(|c| c.tof_days_min).unwrap_or(100.0);
    let tof_max = cruise.and_then(|c| c.tof_days_max).unwrap_or(400.0);
    let n_grid = cruise.and_then(|c| c.grid_resolution).unwrap_or(30) as usize;
    let dep_window = cruise.and_then(|c| c.departure_window_days).unwrap_or(60.0);

    let dep_epoch = require_departure_epoch(cfg)?;
    let dep_jd_base = epoch_to_jd(dep_epoch);

    let n_dep = n_grid.max(2);
    let n_tof = n_grid.max(2);
    let dep_offsets: Vec<f64> = (0..n_dep)
        .map(|i| -dep_window / 2.0 + i as f64 * dep_window / (n_dep - 1) as f64)
        .collect();
    let tof_grid: Vec<f64> = (0..n_tof)
        .map(|i| tof_min + i as f64 * (tof_max - tof_min) / (n_tof - 1) as f64)
        .collect();

    let body_name = cfg.target_body.name.to_lowercase();
    let ephemeris = cfg.target_body.ephemeris;
    let body = resolve_body(ephemeris, &body_name);
    let dep_body = resolve_departure_body(cfg);

    let grid = PorkchopGrid {
        mu: MU_SUN_M3S2,
        dep_offsets_days: dep_offsets,
        tof_days_grid: tof_grid,
    };
    let points = grid.evaluate(|dep_off, tof_d| {
        transfer_states(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_off, tof_d)
    });

    if points.is_empty() {
        return Err(
            "No valid Lambert solutions found. Check TOF range, departure epoch, and that \
             de440s.bsp covers the requested epoch."
                .into(),
        );
    }
    Ok((points, dep_jd_base))
}

#[allow(dead_code)]
fn diff_correct_api(
    cfg: &MissionConfig,
    almanac: &Almanac,
    kep: &Option<KeplerianElements>,
    dv_budget: f64,
) -> Result<BestArcApiResult, String> {
    let cruise = cfg.trajectory.cruise.as_ref();
    let tof_min = cruise.and_then(|c| c.tof_days_min).unwrap_or(100.0);
    let tof_max = cruise.and_then(|c| c.tof_days_max).unwrap_or(400.0);
    let dep_window = cruise.and_then(|c| c.departure_window_days).unwrap_or(60.0);

    let dep_epoch = require_departure_epoch(cfg)?;
    let dep_jd_base = epoch_to_jd(dep_epoch);

    let body_name = cfg.target_body.name.to_lowercase();
    let ephemeris = cfg.target_body.ephemeris;
    let body = resolve_body(ephemeris, &body_name);
    let dep_body = resolve_departure_body(cfg);

    let dep_seeds: Vec<f64> = if dep_window > 0.0 {
        vec![-dep_window / 2.0, 0.0, dep_window / 2.0]
    } else {
        vec![0.0]
    };
    let tof_seeds = [tof_min, 0.5 * (tof_min + tof_max), tof_max];
    let solver = diff_correction_solver();

    let mut converged: Vec<(f64, DiffCorrectionResult)> = Vec::new();
    for &dep_off in &dep_seeds {
        for &tof_seed in &tof_seeds {
            let eval = |params: &[f64]| -> Vec<f64> {
                match eval_dv(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_off, params[0]) {
                    Some(sol) => match onboard_dv_required_ms(cfg, &cfg.trajectory.departure_body, sol.dv_departure_ms, sol.dv_arrival_ms) {
                        Some(required) => vec![required - dv_budget],
                        None           => vec![1.0e9], // departure body not in body_models catalog
                    },
                    None => vec![1.0e9],
                }
            };
            let res = solver.solve(&[tof_seed], eval);
            if res.converged {
                converged.push((dep_off, res));
            }
        }
    }

    let (dep_offset, result) = converged
        .into_iter()
        .min_by(|(_, a), (_, b)| a.params[0].partial_cmp(&b.params[0]).unwrap())
        .ok_or_else(|| {
            "Differential correction failed to converge from any seed".to_string()
        })?;

    let tof = result.params[0];
    let sol = eval_dv(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_offset, tof)
        .ok_or_else(|| "Could not evaluate Lambert arc at converged solution".to_string())?;

    let body_entries = propagator_body_entries(cfg, almanac, dep_jd_base + dep_offset);
    let propagator_bodies = as_propagator_bodies(&body_entries);
    let arc = transfer_states(almanac, dep_body, ephemeris, body, kep, dep_jd_base, dep_offset, tof)
        .map(|(r_dep, ..)| propagate_arc(r_dep, sol.v_transfer_dep, MU_SUN_M3S2, tof * 86_400.0, &propagator_bodies))
        .unwrap_or_default();

    let (dv_arr_ms, dv_total_ms) = objective_priced_arrival(cfg, sol.dv_departure_ms, sol.v_inf_arr_ms);
    let ledger = narrowing_dv_ledger(cfg, &cfg.trajectory.departure_body, sol.dv_departure_ms, sol.v_inf_arr_ms);
    Ok(BestArcApiResult {
        dep_offset_days: dep_offset,
        dep_jd: dep_jd_base + dep_offset,
        tof_days: tof,
        dv_dep_ms: sol.dv_departure_ms,
        dv_arr_ms,
        dv_total_ms,
        c3_km2s2: sol.c3_km2s2,
        v_inf_arr_ms: sol.v_inf_arr_ms,
        dv_budget_ms: Some(dv_budget),
        budget_margin_ms: ledger.as_ref().and_then(|l| l.budget_margin_ms),
        dv_ledger: ledger,
        launch_vehicle_check: compute_launch_vehicle_check(cfg, sol.c3_km2s2),
        arc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid mission TOML for exercising [`arrival_dv_for_objective_ms`]
    /// / [`objective_priced_arrival`] in isolation — no almanac, no propagation.
    /// `objective` and `capture_block` are parameterised; `mu_m3s2`/`radius_m`
    /// are fixed Mars-like values shared by every case so the vis-viva numbers
    /// below are reproducible.
    fn mission_toml(objective: &str, capture_block: &str) -> String {
        format!(
            r#"
[mission]
name = "Test"
objective = "{objective}"

[target_body]
name = "Mars"
mu_m3s2 = 4.282837e13
radius_m = 3.3895e6
gravity_model = "PointMass"
atmosphere = "None"
ephemeris = "Anise"

[spacecraft]
mass_kg = 1000.0
dry_mass_kg = 800.0
propellant_mass_kg = 200.0
bus_dims_m = [2.0, 2.0, 0.63]
inertia_diag_kgm2 = [366.67, 366.67, 666.67]
srp_model = "Cannonball"

[trajectory]
phases = ["Cruise"]
solver = "Lambert"
departure_body = "Earth"
{capture_block}

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

    /// Fixture for the two-pool ΔV ledger (Phase 14e): Earth departure with
    /// real propulsion, an optional catalog launch vehicle, and a Mars Orbit
    /// capture at 4,000 km. `mass_kg`/`dry_mass_kg` are parameterised so the
    /// launcher curve can be probed on either side of its verified points.
    fn ledger_toml(objective: &str, launch_vehicle: Option<&str>, mass_kg: f64, dry_mass_kg: f64) -> String {
        let lv_line = launch_vehicle.map(|v| format!("launch_vehicle = \"{v}\"")).unwrap_or_default();
        format!(
            r#"
[mission]
name = "Ledger"
objective = "{objective}"

[target_body]
name = "Mars"
mu_m3s2 = 4.282837e13
radius_m = 3.3895e6
gravity_model = "PointMass"
atmosphere = "None"
ephemeris = "Anise"

[spacecraft]
mass_kg = {mass_kg}
dry_mass_kg = {dry_mass_kg}
propellant_mass_kg = {prop_kg}
bus_dims_m = [2.0, 2.0, 0.63]
inertia_diag_kgm2 = [366.67, 366.67, 666.67]
srp_model = "Cannonball"
{lv_line}

[spacecraft.propulsion]
type = "Biprop"
isp_s = 300.0
thrust_n = 400.0

[trajectory]
phases = ["Cruise"]
solver = "Lambert"
departure_body = "Earth"

[trajectory.capture]
target_orbit_radius_m = 4.0e6

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
"#,
            prop_kg = mass_kg - dry_mass_kg,
        )
    }

    /// Tangential escape burn from the Earth parking orbit the ledger prices
    /// against, recomputed independently from the catalog constants.
    fn earth_escape_burn_ms(cfg: &MissionConfig, v_inf_ms: f64) -> (f64, f64, f64) {
        let earth = body_models::TargetBody::by_name("Earth").unwrap();
        let r_p = resolve_parking_orbit_radius_m(cfg, &earth);
        let mu = earth.mu_m3s2;
        let v_c = (mu / r_p).sqrt();
        ((v_inf_ms * v_inf_ms + 2.0 * mu / r_p).sqrt() - v_c, mu, r_p)
    }

    #[test]
    fn ledger_earth_without_launcher_charges_the_full_escape_burn_onboard() {
        let cfg: MissionConfig = toml::from_str(&ledger_toml("Orbit", None, 1000.0, 800.0)).unwrap();
        let (dep_burn, _, _) = earth_escape_burn_ms(&cfg, 3_000.0);
        let arrival = arrival_dv_for_objective_ms(&cfg, 3_000.0);
        let l = narrowing_dv_ledger(&cfg, "Earth", 3_000.0, 3_000.0).unwrap();
        assert_eq!(l.departure_dv_pool, "onboard");
        assert!((l.departure_dv_ms - dep_burn).abs() < 1e-9);
        assert_eq!(l.launcher_dv_ms, 0.0);
        assert!((l.onboard_departure_dv_ms - dep_burn).abs() < 1e-9);
        assert!((l.onboard_arrival_dv_ms - arrival).abs() < 1e-9);
        assert!((l.onboard_dv_required_ms - (dep_burn + arrival)).abs() < 1e-9);
        // Rocket equation, independently: available = Isp g0 ln(m_wet/m_dry).
        let avail = 300.0 * G0 * (1000.0_f64 / 800.0).ln();
        assert!((l.onboard_dv_available_ms.unwrap() - avail).abs() < 1e-9);
        assert!((l.budget_margin_ms.unwrap() - (avail - l.onboard_dv_required_ms)).abs() < 1e-9);
        let mr = (l.onboard_dv_required_ms / (300.0 * G0)).exp();
        assert!((l.mass_ratio_required.unwrap() - mr).abs() < 1e-12);
        assert!((l.onboard_propellant_required_kg.unwrap() - 1000.0 * (1.0 - 1.0 / mr)).abs() < 1e-9);
        // ~3.6 km/s escape + ~2.1 km/s capture against a 0.66 km/s tank.
        assert_eq!(l.propellant_feasible, Some(false));
        // The legacy wrapper reports the same onboard requirement.
        assert_eq!(onboard_dv_required_ms(&cfg, "Earth", 3_000.0, 3_000.0).unwrap(), l.onboard_dv_required_ms);
    }

    #[test]
    fn ledger_launcher_covering_the_c3_puts_the_whole_departure_in_the_launcher_pool() {
        // AtlasV401 injects 2,454 kg at C3 <= 11.84; v_inf = 3 km/s is C3 = 9.
        let cfg: MissionConfig = toml::from_str(&ledger_toml("Orbit", Some("AtlasV401"), 2000.0, 1600.0)).unwrap();
        let (dep_burn, _, _) = earth_escape_burn_ms(&cfg, 3_000.0);
        let arrival = arrival_dv_for_objective_ms(&cfg, 3_000.0);
        let check = compute_launch_vehicle_check(&cfg, 9.0).unwrap();
        assert!(check.feasible);
        assert_eq!(check.reason, None);
        assert!((check.launcher_departure_dv_ms - dep_burn).abs() < 1e-6);
        assert_eq!(check.onboard_departure_dv_ms, 0.0);
        let l = narrowing_dv_ledger(&cfg, "Earth", 3_000.0, 3_000.0).unwrap();
        assert_eq!(l.departure_dv_pool, "launcher");
        assert!((l.launcher_dv_ms - dep_burn).abs() < 1e-9);
        assert_eq!(l.onboard_departure_dv_ms, 0.0);
        // Onboard pays the capture burn only -- the margin is budget minus
        // the PRICED capture burn, not minus the raw arrival v_inf (the sign
        // error a frontend report caught: -967 m/s where +832 was expected).
        assert!((l.onboard_dv_required_ms - arrival).abs() < 1e-9);
        let avail = 300.0 * G0 * (2000.0_f64 / 1600.0).ln();
        assert!((l.budget_margin_ms.unwrap() - (avail - arrival)).abs() < 1e-9);
        assert!(l.budget_margin_ms.unwrap() > avail - 3_000.0, "raw v_inf must not be what gets charged");
    }

    #[test]
    fn ledger_partial_launcher_coverage_splits_the_departure_burn_at_the_same_periapsis() {
        // 2,300 kg sits between AtlasV401's two verified points; at C3 = 25
        // the curve gives ~2,191 kg -> infeasible, but the inverse curve says
        // the vehicle still reaches C3 ~19.5 with this mass.
        let cfg: MissionConfig = toml::from_str(&ledger_toml("Orbit", Some("AtlasV401"), 2300.0, 1900.0)).unwrap();
        let v_inf_ms = 5_000.0; // C3 = 25
        let (dep_burn, mu, r_p) = earth_escape_burn_ms(&cfg, v_inf_ms);
        let check = compute_launch_vehicle_check(&cfg, 25.0).unwrap();
        assert!(!check.feasible);
        assert!(check.reason.as_deref().unwrap().contains("can inject at most"));
        let c3_l = check.max_c3_at_mass_km2s2.unwrap();
        let expected_c3_l = 11.84 + (2454.0 - 2300.0) / (2454.0 - 2105.0) * (29.29678 - 11.84);
        assert!((c3_l - expected_c3_l).abs() < 1e-9);
        let v_c = (mu / r_p).sqrt();
        let launcher = (c3_l * 1e6 + 2.0 * mu / r_p).sqrt() - v_c;
        let onboard = (25.0e6 + 2.0 * mu / r_p).sqrt() - (c3_l * 1e6 + 2.0 * mu / r_p).sqrt();
        assert!((check.launcher_departure_dv_ms - launcher).abs() < 1e-6);
        assert!((check.onboard_departure_dv_ms - onboard).abs() < 1e-6);
        // The two shares add up to the full tangential escape burn.
        assert!((launcher + onboard - dep_burn).abs() < 1e-6);
        let l = narrowing_dv_ledger(&cfg, "Earth", v_inf_ms, 3_000.0).unwrap();
        assert_eq!(l.departure_dv_pool, "split");
        assert!((l.launcher_dv_ms - launcher).abs() < 1e-6);
        assert!((l.onboard_departure_dv_ms - onboard).abs() < 1e-6);
        assert!((l.onboard_dv_required_ms - (onboard + arrival_dv_for_objective_ms(&cfg, 3_000.0))).abs() < 1e-6);
    }

    #[test]
    fn launch_check_reports_a_c3_beyond_the_verified_range_distinctly() {
        // Light payload, C3 above AtlasV401's highest verified point (29.3):
        // no injected-mass figure exists, but the inverse curve still gives
        // the highest verified C3 as what the launcher delivers.
        let cfg: MissionConfig = toml::from_str(&ledger_toml("Orbit", Some("AtlasV401"), 500.0, 400.0)).unwrap();
        let check = compute_launch_vehicle_check(&cfg, 40.0).unwrap();
        assert!(!check.feasible);
        assert_eq!(check.max_injected_mass_kg, None);
        assert!(check.reason.as_deref().unwrap().contains("verified performance range"));
        assert_eq!(check.max_c3_at_mass_km2s2, Some(29.29678));
        assert!(check.launcher_departure_dv_ms > 0.0);
        assert!(check.onboard_departure_dv_ms > 0.0);
        let (dep_burn, _, _) = earth_escape_burn_ms(&cfg, 40.0_f64.sqrt() * 1e3);
        assert!((check.launcher_departure_dv_ms + check.onboard_departure_dv_ms - dep_burn).abs() < 1e-6);
    }

    /// `Launch`-mode fixture (Phase 14a): the ledger fixture plus a
    /// `[trajectory.departure]` block in `Launch` mode from Cape Canaveral.
    fn launch_toml(launch_vehicle: Option<&str>, mass_kg: f64, dry_mass_kg: f64, site: bool) -> String {
        let site_block = if site {
            "[trajectory.departure.launch_site]\nname = \"Cape Canaveral\"\nlat_deg = 28.5\nlon_deg = -80.6\n"
        } else {
            ""
        };
        ledger_toml("Orbit", launch_vehicle, mass_kg, dry_mass_kg).replace(
            "[trajectory.capture]",
            &format!("[trajectory.departure]\nmode = \"Launch\"\n{site_block}\n[trajectory.capture]"),
        )
    }

    #[test]
    fn launch_mode_defaults_the_parking_orbit_to_185_km_and_prices_only_the_top_up() {
        let cfg: MissionConfig = toml::from_str(&launch_toml(Some("AtlasV401"), 2000.0, 1600.0, true)).unwrap();
        assert_eq!(departure_mode(&cfg), DepartureMode::Launch);
        let earth = body_models::TargetBody::by_name("Earth").unwrap();
        assert!((resolve_parking_orbit_radius_m(&cfg, &earth) - (earth.radius_m + LAUNCH_PARKING_ALTITUDE_DEFAULT_M)).abs() < 1e-9);
        // Launcher covers C3 = 9 at 2,000 kg -> the spacecraft pays nothing.
        assert_eq!(departure_onboard_cost_ms(&cfg, "Earth", 3_000.0), Some(0.0));
        // Beyond the vehicle's verified range the top-up is what's charged,
        // and it equals the launch check's own number.
        let check = compute_launch_vehicle_check(&cfg, 40.0).unwrap();
        let cost = departure_onboard_cost_ms(&cfg, "Earth", 40.0_f64.sqrt() * 1e3).unwrap();
        assert!((cost - check.onboard_departure_dv_ms).abs() < 1e-9);
        assert!(cost > 0.0);
        // ParkingOrbit mode charges the full escape burn regardless of launcher.
        let cfg_po: MissionConfig = toml::from_str(&ledger_toml("Orbit", Some("AtlasV401"), 2000.0, 1600.0)).unwrap();
        let (full, _) = departure_escape_dv_ms(&cfg_po, "Earth", 3_000.0).unwrap();
        assert_eq!(departure_onboard_cost_ms(&cfg_po, "Earth", 3_000.0), Some(full));
        // The launch geometry resolves from the catalog pole + site.
        let g = launch_geometry_for(&cfg, &earth, nalgebra::Vector3::new(0.6, 0.8, 0.0) * 3_000.0).unwrap();
        assert!((g.inclination_rad - 28.5_f64.to_radians()).abs() < 1e-12, "DLA = 0 here, so the site sets the plane");
        assert!(g.feasible_no_dogleg);
        let api = launch_geometry_api(&cfg, &g);
        assert_eq!(api.site_name, "Cape Canaveral");
        assert!((api.parking_orbit_radius_m - (earth.radius_m + LAUNCH_PARKING_ALTITUDE_DEFAULT_M)).abs() < 1e-3);
        // ParkingOrbit mode never yields a launch geometry.
        assert!(launch_geometry_for(&cfg_po, &earth, nalgebra::Vector3::new(0.6, 0.8, 0.0) * 3_000.0).is_none());
    }

    #[test]
    fn check_config_enforces_launch_mode_prerequisites() {
        // No launch vehicle, no site: both reported.
        let cfg: MissionConfig = toml::from_str(&launch_toml(None, 1000.0, 800.0, false)).unwrap();
        let errs = crate::config::check_config(&cfg);
        assert!(errs.iter().any(|e| e.contains("requires spacecraft.launch_vehicle")), "{errs:?}");
        assert!(errs.iter().any(|e| e.contains("requires trajectory.departure.launch_site")), "{errs:?}");
        // Complete Launch config passes the departure checks.
        let ok: MissionConfig = toml::from_str(&launch_toml(Some("Falcon9"), 1000.0, 800.0, true)).unwrap();
        let errs = crate::config::check_config(&ok);
        assert!(!errs.iter().any(|e| e.contains("trajectory.departure")), "{errs:?}");
    }

    #[test]
    fn ledger_flyby_never_charges_an_arrival_burn() {
        let cfg: MissionConfig = toml::from_str(&ledger_toml("Flyby", None, 1000.0, 800.0)).unwrap();
        let l = dv_ledger(&cfg, "Earth", 3_600.0, 9.0, 2_500.0, 0.0);
        assert_eq!(l.onboard_arrival_dv_ms, 0.0);
        assert!((l.onboard_dv_required_ms - 3_600.0).abs() < 1e-12);
    }

    #[test]
    fn flyby_arrival_dv_is_always_zero() {
        // No [trajectory.capture] section at all -- Flyby never needs one.
        let cfg: MissionConfig = toml::from_str(&mission_toml("Flyby", "")).unwrap();
        assert_eq!(arrival_dv_for_objective_ms(&cfg, 3_500.0), 0.0);
        let (dv_arr, dv_total) = objective_priced_arrival(&cfg, 4_200.0, 3_500.0);
        assert_eq!(dv_arr, 0.0);
        assert_eq!(dv_total, 4_200.0);
    }

    #[test]
    fn flyby_arrival_dv_is_zero_even_with_capture_configured() {
        // A stray [trajectory.capture] section (e.g. copy-pasted from an
        // Orbit config) must not resurrect an arrival burn for Flyby.
        let cfg: MissionConfig = toml::from_str(&mission_toml(
            "Flyby",
            "[trajectory.capture]\ntarget_orbit_radius_m = 4.0e6",
        ))
        .unwrap();
        assert_eq!(arrival_dv_for_objective_ms(&cfg, 3_500.0), 0.0);
    }

    #[test]
    fn orbit_arrival_dv_is_real_vis_viva_capture_burn() {
        // Mars mu = 4.282837e13 m^3/s^2, r_cap = 4.0e6 m (above Mars' 3.3895e6 m
        // radius, so the Phase 9y floor-clamp is inert here), circular capture
        // (capture_eccentricity defaults to 0.0), v_inf = 3000 m/s.
        let cfg: MissionConfig = toml::from_str(&mission_toml(
            "Orbit",
            "[trajectory.capture]\ntarget_orbit_radius_m = 4.0e6",
        ))
        .unwrap();
        let mu = 4.282837e13_f64;
        let r_cap = 4.0e6_f64;
        let v_inf = 3_000.0_f64;
        let expected = (v_inf * v_inf + 2.0 * mu / r_cap).sqrt() - (mu / r_cap).sqrt();
        let got = arrival_dv_for_objective_ms(&cfg, v_inf);
        assert!((got - expected).abs() < 1e-6, "got {got}, expected {expected}");
        // Real capture burn must differ from (be smaller than) the raw v_inf
        // this replaces -- otherwise the fix isn't doing anything.
        assert!(got < v_inf);

        let (dv_arr, dv_total) = objective_priced_arrival(&cfg, 4_200.0, v_inf);
        assert_eq!(dv_arr, expected);
        assert_eq!(dv_total, 4_200.0 + expected);
    }

    #[test]
    fn landing_arrival_dv_matches_orbit_formula() {
        // Landing uses the same capture-burn formula as Orbit (both are
        // "burn to match a capture orbit" from the narrowing stage's
        // perspective -- descent-specific ΔV is a later stage's concern).
        let cfg_orbit: MissionConfig = toml::from_str(&mission_toml(
            "Orbit",
            "[trajectory.capture]\ntarget_orbit_radius_m = 4.0e6",
        ))
        .unwrap();
        let cfg_landing: MissionConfig = toml::from_str(&mission_toml(
            "Landing",
            "[trajectory.capture]\ntarget_orbit_radius_m = 4.0e6",
        ))
        .unwrap();
        assert_eq!(
            arrival_dv_for_objective_ms(&cfg_orbit, 3_000.0),
            arrival_dv_for_objective_ms(&cfg_landing, 3_000.0),
        );
    }

    #[test]
    fn orbit_capture_radius_floor_clamped_to_body_radius() {
        // A configured target_orbit_radius_m below the target body's own
        // physical radius (3.3895e6 m for this Mars fixture) must be
        // floor-clamped, not fed straight into vis-viva (Phase 9y).
        let cfg: MissionConfig = toml::from_str(&mission_toml(
            "Orbit",
            "[trajectory.capture]\ntarget_orbit_radius_m = 1.0e6",
        ))
        .unwrap();
        let mu = 4.282837e13_f64;
        let r_cap = 3.3895e6_f64; // floor = target_body.radius_m, not the configured 1.0e6
        let v_inf = 3_000.0_f64;
        let expected = (v_inf * v_inf + 2.0 * mu / r_cap).sqrt() - (mu / r_cap).sqrt();
        let got = arrival_dv_for_objective_ms(&cfg, v_inf);
        assert!((got - expected).abs() < 1e-6, "got {got}, expected {expected}");
    }

    #[test]
    fn orbit_without_capture_config_falls_back_to_raw_v_inf() {
        // No [trajectory.capture] section at all -- DiffCorrection/porkchop
        // callers have never required one to run, so there's no real burn
        // to price; fall back to the pre-fix raw-v_inf behaviour rather
        // than fail outright.
        let cfg: MissionConfig = toml::from_str(&mission_toml("Orbit", "")).unwrap();
        assert_eq!(arrival_dv_for_objective_ms(&cfg, 3_000.0), 3_000.0);
    }

    #[test]
    fn rendezvous_and_sample_return_keep_raw_v_inf_unchanged() {
        // Neither objective is in scope for this fix -- both must be
        // provably unaffected by it, capture configured or not.
        for objective in ["Rendezvous", "SampleReturn"] {
            let cfg: MissionConfig = toml::from_str(&mission_toml(
                objective,
                "[trajectory.capture]\ntarget_orbit_radius_m = 4.0e6",
            ))
            .unwrap();
            assert_eq!(arrival_dv_for_objective_ms(&cfg, 3_000.0), 3_000.0);
        }
    }

    /// Phase 01/02 consistency ask: `target_body.third_bodies`
    /// must produce a real, usable `body_tracks` entry when a departure
    /// epoch is available.
    #[test]
    fn build_third_body_tracks_resolves_a_catalog_body() {
        let almanac = match ephemeris::Almanac::new("kernels/de440s.bsp")
            .or_else(|_| ephemeris::Almanac::new("../kernels/de440s.bsp"))
        {
            Ok(a) => a,
            Err(_) => {
                eprintln!("[skip] build_third_body_tracks_resolves_a_catalog_body: de440s.bsp not found");
                return;
            }
        };

        let toml_str = format!(
            r#"
[mission]
name = "Test"
objective = "Orbit"

[target_body]
name = "Mars"
mu_m3s2 = 4.282837e13
radius_m = 3.3895e6
gravity_model = "PointMass"
atmosphere = "None"
ephemeris = "Anise"
third_bodies = ["Jupiter"]

[spacecraft]
mass_kg = 1000.0
dry_mass_kg = 800.0
propellant_mass_kg = 200.0
bus_dims_m = [2.0, 2.0, 0.63]
inertia_diag_kgm2 = [366.67, 366.67, 666.67]
srp_model = "Cannonball"

[trajectory]
phases = ["Cruise"]
solver = "Lambert"
departure_body = "Earth"
departure_epoch = "2026-09-15T00:00:00 UTC"

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
        );
        let cfg: MissionConfig = toml::from_str(&toml_str).unwrap();

        let tracks = build_third_body_tracks(&cfg, &almanac, 3600.0, &[]);
        assert_eq!(tracks.len(), 1, "Jupiter should resolve to exactly one track");
        assert_eq!(tracks[0].name, "Jupiter");
        assert!(tracks[0].track.len() >= 2);
        assert!(tracks[0].mu_m3s2.is_some(), "Jupiter should resolve a real catalog mu_m3s2");
        // A real heliocentric position, not a placeholder -- Jupiter sits
        // roughly 5 AU out, so its distance from the origin should be well
        // above 1 AU regardless of the exact epoch.
        let r0 = tracks[0].track[0].r_m;
        let dist = (r0[0].powi(2) + r0[1].powi(2) + r0[2].powi(2)).sqrt();
        assert!(dist > 3.0e11, "Jupiter's real distance should be several AU, got {dist:e} m");
    }

    /// Review B2: an empty `track` on a named ANISE body is
    /// resolved server-side into a real sampled track; an explicit track
    /// is left byte-for-byte untouched; an unknown name is a hard error.
    #[test]
    fn resolve_named_body_tracks_fills_empty_tracks_and_rejects_unknown_names() {
        let almanac = match ephemeris::Almanac::new("kernels/de440s.bsp")
            .or_else(|_| ephemeris::Almanac::new("../kernels/de440s.bsp"))
        {
            Ok(a) => a,
            Err(_) => {
                eprintln!("[skip] resolve_named_body_tracks_fills_empty_tracks_and_rejects_unknown_names: de440s.bsp not found");
                return;
            }
        };

        let toml_str = r#"
[mission]
name = "Test"
objective = "Orbit"

[target_body]
name = "Mars"
mu_m3s2 = 4.282837e13
radius_m = 3.3895e6
gravity_model = "PointMass"
atmosphere = "None"
ephemeris = "Anise"

[spacecraft]
mass_kg = 1000.0
dry_mass_kg = 800.0
propellant_mass_kg = 200.0
bus_dims_m = [2.0, 2.0, 0.63]
inertia_diag_kgm2 = [366.67, 366.67, 666.67]
srp_model = "Cannonball"

[trajectory]
phases = ["Cruise"]
solver = "Lambert"
departure_body = "Earth"
departure_epoch = "2026-09-15T00:00:00 UTC"

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

[cruise_seed]
r0_m = [1.5e11, 0.0, 0.0]
v0_m = [0.0, 29800.0, 0.0]
duration_s = 864000.0
tick_s = 60.0
reference = [
  { t_s = 0.0, r_m = [1.5e11, 0.0, 0.0], v_mps = [0.0, 29800.0, 0.0] },
  { t_s = 864000.0, r_m = [1.49e11, 2.6e10, 0.0], v_mps = [0.0, 29800.0, 0.0] },
]

[[cruise_seed.body_tracks]]
name = "Jupiter"
"#;
        let mut cfg: MissionConfig = toml::from_str(toml_str).unwrap();
        assert!(cfg.cruise_seed.as_ref().unwrap().body_tracks[0].track.is_empty(), "fixture precondition");

        resolve_named_body_tracks(&mut cfg, &almanac).expect("Jupiter should resolve server-side");
        let seed = cfg.cruise_seed.as_ref().unwrap();
        let t = &seed.body_tracks[0];
        assert!(t.track.len() >= 64, "expected a real sampled track, got {} points", t.track.len());
        assert!((t.track.last().unwrap().t_s - 864_000.0).abs() < 1e-6, "track must span the leg duration");
        assert!(t.mu_m3s2.is_some(), "catalog mu should be auto-filled");
        let r0 = t.track[0].r_m;
        let dist = (r0[0].powi(2) + r0[1].powi(2) + r0[2].powi(2)).sqrt();
        assert!(dist > 3.0e11, "Jupiter should be several AU out, got {dist:e} m");
        // Samples must actually MOVE (a frozen track would defeat the point).
        let r_last = t.track.last().unwrap().r_m;
        let moved = ((r_last[0] - r0[0]).powi(2) + (r_last[1] - r0[1]).powi(2) + (r_last[2] - r0[2]).powi(2)).sqrt();
        assert!(moved > 1.0e9, "Jupiter should move >1e6 km over 10 days, got {moved:e} m");

        // Unknown name with an empty track: hard error, never silence.
        let mut bad = cfg.clone();
        bad.cruise_seed.as_mut().unwrap().body_tracks.push(crate::config::BodyTrackConfig {
            name: "NotABody".into(),
            track: vec![],
            epoch_jd: None,
            mu_m3s2: None,
            soi_capture: false,
        });
        let err = resolve_named_body_tracks(&mut bad, &almanac).unwrap_err();
        assert!(err.contains("NotABody"), "error should name the unresolvable body: {err}");
    }

    /// A name already present in `existing` must be left untouched -- an
    /// explicit client-supplied track always wins over auto-population.
    #[test]
    fn build_third_body_tracks_skips_names_already_present() {
        let almanac = match ephemeris::Almanac::new("kernels/de440s.bsp")
            .or_else(|_| ephemeris::Almanac::new("../kernels/de440s.bsp"))
        {
            Ok(a) => a,
            Err(_) => {
                eprintln!("[skip] build_third_body_tracks_skips_names_already_present: de440s.bsp not found");
                return;
            }
        };

        let toml_str = format!(
            r#"
[mission]
name = "Test"
objective = "Orbit"

[target_body]
name = "Mars"
mu_m3s2 = 4.282837e13
radius_m = 3.3895e6
gravity_model = "PointMass"
atmosphere = "None"
ephemeris = "Anise"
third_bodies = ["Jupiter"]

[spacecraft]
mass_kg = 1000.0
dry_mass_kg = 800.0
propellant_mass_kg = 200.0
bus_dims_m = [2.0, 2.0, 0.63]
inertia_diag_kgm2 = [366.67, 366.67, 666.67]
srp_model = "Cannonball"

[trajectory]
phases = ["Cruise"]
solver = "Lambert"
departure_body = "Earth"
departure_epoch = "2026-09-15T00:00:00 UTC"

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
        );
        let cfg: MissionConfig = toml::from_str(&toml_str).unwrap();

        let existing = vec![crate::config::BodyTrackConfig {
            name: "Jupiter".to_string(),
            track: vec![
                crate::config::CruiseReferencePointConfig { t_s: 0.0, r_m: [1.0, 2.0, 3.0], v_mps: [0.0, 0.0, 0.0] },
                crate::config::CruiseReferencePointConfig { t_s: 10.0, r_m: [1.0, 2.0, 3.0], v_mps: [0.0, 0.0, 0.0] },
            ],
            epoch_jd: None,
            mu_m3s2: None,
            soi_capture: false,
        }];
        let tracks = build_third_body_tracks(&cfg, &almanac, 3600.0, &existing);
        assert!(tracks.is_empty(), "an already-present name should not be re-added");
    }

    /// No `departure_epoch` -> no ANISE queries possible -> empty, not an
    /// error (best-effort convenience layer, see the function's own doc
    /// comment).
    #[test]
    fn build_third_body_tracks_empty_without_departure_epoch() {
        let almanac = match ephemeris::Almanac::new("kernels/de440s.bsp")
            .or_else(|_| ephemeris::Almanac::new("../kernels/de440s.bsp"))
        {
            Ok(a) => a,
            Err(_) => {
                eprintln!("[skip] build_third_body_tracks_empty_without_departure_epoch: de440s.bsp not found");
                return;
            }
        };
        let cfg: MissionConfig = toml::from_str(&mission_toml("Orbit", "")).unwrap();
        assert!(cfg.trajectory.departure_epoch.is_none());
        let tracks = build_third_body_tracks(&cfg, &almanac, 3600.0, &[]);
        assert!(tracks.is_empty());
    }
}
