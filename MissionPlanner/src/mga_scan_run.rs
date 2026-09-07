//! Phase 9w-iii — wire the ballistic MGA grid scan (`trajectory_solver::mga_scan`)
//! into an ANISE-aware CLI command: build the body sequence, departure-date
//! and per-leg TOF grids from `[optimization.mga]`/`[optimization.mga.scan]`,
//! run the scan, and write `mga_window_scan.csv`.
//!
//! This module contains no scan algorithm — that lives in the generic
//! `trajectory_solver` crate (no ANISE dependency, per the shared-crates
//! rule). Everything here is ephemeris plumbing and file I/O, matching the
//! existing `mga.rs`/`sequence_search.rs` split between generic search logic
//! and mission-specific wiring.

use ephemeris::{Almanac, Body, Epoch};
use nalgebra::Vector3;
use trajectory_solver::{
    keplerian::MU_SUN_M3S2, mga_scan::{run_mga_scan, MgaScanConfig, ScanBody, ScanRecord},
};

use crate::config::MissionConfig;
use crate::design::{anise_body, epoch_to_jd, parse_epoch};

/// Resolve a catalog body name to its ANISE state-query closure input, plus
/// its μ and minimum flyby periapsis (body radius — the scan has no
/// separate safety-margin config; `flyby_min_periapsis_m` from
/// `[optimization.mga]` is used uniformly, matching the DE search's own
/// floor).
struct ResolvedBody {
    anise: Body,
    mu_m3s2: f64,
    radius_m: f64,
    min_periapsis_m: f64,
}

fn resolve_body(name: &str, min_periapsis_m: f64) -> Result<ResolvedBody, String> {
    let anise = anise_body(&name.to_lowercase())
        .ok_or_else(|| format!("mga-scan: '{name}' has no ANISE coverage — the scan requires real ephemeris"))?;
    let cat = body_models::TargetBody::by_name(name)
        .ok_or_else(|| format!("mga-scan: '{name}' is not in the body catalog"))?;
    Ok(ResolvedBody { anise, mu_m3s2: cat.mu_m3s2, radius_m: cat.radius_m, min_periapsis_m })
}

/// Vis-viva capture burn from an arrival hyperbola (`vinf_arr_ms`) into the
/// configured `[trajectory.capture]` orbit around the target body — same
/// formula as `mga.rs::arrival_dv_ms`'s Orbit/Landing branch, reimplemented
/// here in terms of the scan's own (mu, radius) inputs since the ballistic
/// scan has no `ChromosomeEval` to read them from. `None` when
/// `[trajectory.capture].target_orbit_radius_m` isn't configured — there's
/// no orbit to compute a burn into, so the scan can only ever report the
/// flyby-style cost (departure + flyby ΔVs, no arrival burn) for that
/// mission. This is what lets the porkchop plots show either the Flyby-style
/// cost (no arrival burn) or the Orbit-style cost (with a real capture burn)
/// for the same scan, per the frontend's "we may want to know either" ask.
fn capture_dv_ms(cfg: &MissionConfig, vinf_arr_ms: f64, mu_target_m3s2: f64, body_radius_m: f64) -> Option<f64> {
    let cap = cfg.trajectory.capture.as_ref()?;
    let r_cap_configured = cap.target_orbit_radius_m?;
    // Same floor-clamp as mga.rs::arrival_dv_ms (Phase 9y) — never compute a
    // vis-viva burn for an "orbit" inside the target body's own surface.
    let r_cap = r_cap_configured.max(body_radius_m);
    let v_peri = (mu_target_m3s2 * (1.0 + cap.capture_eccentricity) / r_cap).sqrt();
    let v_hyp = (vinf_arr_ms * vinf_arr_ms + 2.0 * mu_target_m3s2 / r_cap).sqrt();
    Some(v_hyp - v_peri)
}

/// JD → Epoch, matching `design.rs::jd_to_epoch` (private there).
fn jd_to_epoch(jd: f64) -> Epoch {
    Epoch::from_unix_seconds((jd - 2_440_587.5) * 86_400.0)
}

fn anise_state_at(almanac: &Almanac, body: Body, jd: f64) -> (Vector3<f64>, Vector3<f64>) {
    let st = almanac
        .body_state_heliocentric(body, jd_to_epoch(jd))
        .unwrap_or_else(|e| panic!("mga-scan: ANISE query failed for {body:?} at JD {jd:.1}: {e}"));
    (
        Vector3::new(st.position.inner[0], st.position.inner[1], st.position.inner[2]),
        Vector3::new(st.velocity.inner[0], st.velocity.inner[1], st.velocity.inner[2]),
    )
}

/// Run the Phase 9w ballistic grid scan for `cfg`'s `[optimization]` +
/// `[optimization.mga]` + `[optimization.mga.scan]` sections, and write
/// `mga_window_scan.csv` to `[simulation].output_dir`.
///
/// Uses the fixed `flyby_bodies` sequence from `[optimization.mga]` — run
/// the Tisserand `search-sequence` command first to pick a sequence for an
/// auto-discovery mission, then paste it in here (same pattern the DE search
/// already uses via `run_mga_fixed_sequence`).
pub fn run_mga_window_scan(cfg: &MissionConfig, almanac: &Almanac) -> Result<(), String> {
    let (records, capture_dvs, names, n_legs_evaluated, n_records_dropped, elapsed_s) = compute_mga_scan(cfg, almanac)?;
    println!(
        "  {} legs evaluated, {} feasible complete branches found ({} dropped by max_records) in {:.1}s",
        n_legs_evaluated, records.len(), n_records_dropped, elapsed_s
    );
    write_scan_csv(cfg, &records, &capture_dvs, &names)?;
    Ok(())
}

/// Same scan as [`run_mga_window_scan`], but returns the records as an
/// API-shaped JSON-serializable result instead of writing only a CSV — for
/// `POST /api/mga-scan` (Phase 9w-vii). Still writes `mga_window_scan.csv`
/// as a side effect (same as the CLI path — every other async-job endpoint
/// in this server also leaves its CSV output on disk), so this is additive,
/// not a replacement code path.
pub fn mga_scan_api(cfg: &MissionConfig, almanac: &Almanac) -> Result<MgaScanApiResult, String> {
    let (records, capture_dvs, names, n_legs_evaluated, n_records_dropped, _elapsed_s) = compute_mga_scan(cfg, almanac)?;
    write_scan_csv(cfg, &records, &capture_dvs, &names)?;

    let api_records = records.iter().zip(capture_dvs.iter()).map(|(rec, &capture_dv_ms)| {
        let total_tof_days: f64 = rec.leg_tofs_s.iter().sum::<f64>() / 86_400.0;
        MgaScanRecordApi {
            dep_mjd2000: rec.dep_epoch_s / 86_400.0,
            vinf_dep_ms: rec.vinf_dep_ms,
            vinf_arr_ms: rec.vinf_arr_ms,
            sum_flyby_dv_ms: rec.sum_flyby_dv_ms,
            capture_dv_ms,
            total_tof_days,
            leg_tofs_days: rec.leg_tofs_s.iter().map(|s| s / 86_400.0).collect(),
            flyby_dvs_ms: rec.flyby_dvs_ms.clone(),
            flyby_rp_km: rec.flyby_rp_m.iter().map(|m| m / 1e3).collect(),
            flyby_turn_deg: rec.flyby_turn_rad.iter().map(|r| r.to_degrees()).collect(),
        }
    }).collect();

    Ok(MgaScanApiResult {
        body_sequence: names,
        n_legs_evaluated,
        n_records_dropped,
        records: api_records,
    })
}

/// One row of `mga_window_scan.csv`, JSON-shaped — see `write_scan_csv` for
/// the CSV column layout this mirrors.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MgaScanRecordApi {
    pub dep_mjd2000: f64,
    pub vinf_dep_ms: f64,
    pub vinf_arr_ms: f64,
    pub sum_flyby_dv_ms: f64,
    /// Vis-viva capture burn into `[trajectory.capture]`'s configured orbit
    /// — `None` when that section/field isn't set (no orbit to capture
    /// into, e.g. a Flyby mission). `cost_ms` = vinf_dep_ms + sum_flyby_dv_ms
    /// (Flyby-style, no arrival burn) or + capture_dv_ms (Orbit-style) — the
    /// caller picks, since both are legitimate depending on mission type.
    pub capture_dv_ms: Option<f64>,
    pub total_tof_days: f64,
    /// One entry per leg, leg order (departure_body -> flyby_bodies... -> target_body).
    pub leg_tofs_days: Vec<f64>,
    /// One entry per intermediate flyby body (`len == body_sequence.len() - 2`).
    pub flyby_dvs_ms: Vec<f64>,
    pub flyby_rp_km: Vec<f64>,
    pub flyby_turn_deg: Vec<f64>,
}

/// `POST /api/mga-scan` result: `mga_window_scan.csv`'s rows as JSON, plus
/// the resolved body sequence and scan summary stats.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MgaScanApiResult {
    /// [departure_body, flyby_bodies..., target_body], in visit order.
    pub body_sequence: Vec<String>,
    pub n_legs_evaluated: u64,
    pub n_records_dropped: u64,
    pub records: Vec<MgaScanRecordApi>,
}

/// Shared scan core for both [`run_mga_window_scan`] (CLI) and
/// [`mga_scan_api`] (`/api/mga-scan`) — resolves the flyby sequence (fixed
/// or Tisserand-discovered) from config, then delegates to
/// [`compute_mga_scan_for_sequence`] for the actual grid build + scan run.
fn compute_mga_scan(
    cfg: &MissionConfig,
    almanac: &Almanac,
) -> Result<(Vec<ScanRecord>, Vec<Option<f64>>, Vec<String>, u64, u64, f64), String> {
    let opt = cfg.optimization.as_ref()
        .ok_or("mga-scan requires an [optimization] section")?;
    let mga = opt.mga.as_ref()
        .ok_or("mga-scan requires an [optimization.mga] section")?;

    // Auto-discover the flyby sequence via the Tisserand beam search when
    // [optimization.mga.sequence_search] is configured — same source of
    // truth `run_mga`'s own sequence_search branch uses, so `mga-scan`
    // doesn't require the user to already know (or have separately run
    // `search-sequence` to find) a fixed flyby_bodies list. Takes the
    // single top-ranked sequence — the scan is cheap enough per-sequence
    // that a caller wanting several candidates can just run this multiple
    // times with different [optimization.mga.sequence_search] pools, but
    // "the code decides the one sequence to analyze" is the actual ask.
    let effective_flyby_bodies: Vec<String> = if let Some(ss_cfg) = mga.sequence_search.as_ref() {
        let sequences = crate::sequence_search::run_sequence_search(ss_cfg, &opt.departure_body, &opt.target_body);
        let top = sequences.first().ok_or(
            "mga-scan: Tisserand beam search found no feasible sequences for this window — \
             try relaxing sequence_search.max_legs or sequence_search.beam_width"
        )?;
        println!(
            "  Auto-selected flyby sequence via Tisserand search: {} → {} → {}  \
             (score {:.3}, estimated arrival v∞ {:.0} m/s)",
            opt.departure_body,
            if top.flyby_bodies.is_empty() { "(direct)".to_string() } else { top.flyby_bodies.join(" → ") },
            opt.target_body,
            top.tisserand_score, top.estimated_vinf_arr_ms,
        );
        top.flyby_bodies.clone()
    } else {
        mga.flyby_bodies.clone()
    };

    compute_mga_scan_for_sequence(cfg, almanac, &effective_flyby_bodies)
}

/// Runs the ballistic scan for an EXPLICIT flyby-body sequence — no
/// sequence_search resolution, the caller already has a concrete sequence
/// in hand. Shared by [`compute_mga_scan`] (CLI/API, which resolves the
/// sequence from config first) and Phase 9w-vi's scan-informed window
/// derivation in `mga.rs::run_mga_fixed_sequence` (which already knows its
/// own resolved sequence — it's the function's own parameter).
pub fn compute_mga_scan_for_sequence(
    cfg: &MissionConfig,
    almanac: &Almanac,
    effective_flyby_bodies: &[String],
) -> Result<(Vec<ScanRecord>, Vec<Option<f64>>, Vec<String>, u64, u64, f64), String> {
    let opt = cfg.optimization.as_ref()
        .ok_or("mga-scan requires an [optimization] section")?;
    let mga = opt.mga.as_ref()
        .ok_or("mga-scan requires an [optimization.mga] section")?;
    let scan_cfg = mga.scan.as_ref()
        .ok_or("mga-scan requires an [optimization.mga.scan] section")?;

    if mga.leg_tof_days.is_empty() {
        return Err("mga-scan requires at least one [optimization.mga].leg_tof_days entry".into());
    }
    let n_legs = effective_flyby_bodies.len() + 1;

    let dep_epoch_str = opt.departure_epoch.as_deref()
        .ok_or("mga-scan requires [optimization].departure_epoch")?;
    let dep_epoch_center = parse_epoch(dep_epoch_str)?;
    let dep_jd_center = epoch_to_jd(dep_epoch_center);

    // Body sequence: departure_body, flyby_bodies..., target_body.
    let mut names: Vec<String> = vec![opt.departure_body.clone()];
    names.extend(effective_flyby_bodies.iter().cloned());
    names.push(opt.target_body.clone());

    let resolved: Vec<ResolvedBody> = names.iter()
        .map(|n| resolve_body(n, mga.flyby_min_periapsis_m))
        .collect::<Result<_, _>>()?;

    // `+ Sync`: `run_mga_scan` (Phase 9w parallelization) shares `&[ScanBody]`
    // across worker threads, so `ScanBody::state_at` must be `Sync`. `almanac`
    // is captured by move (a `&Almanac`, `Copy`) with no interior mutability,
    // and `Almanac` is already queried concurrently elsewhere in this
    // codebase (MGA's MBH worker threads, `mga.rs`) — safe to widen.
    let state_fns: Vec<Box<dyn Fn(f64) -> (Vector3<f64>, Vector3<f64>) + Sync + '_>> = resolved.iter()
        .map(|rb| -> Box<dyn Fn(f64) -> (Vector3<f64>, Vector3<f64>) + Sync> {
            Box::new(move |t_abs_s: f64| anise_state_at(almanac, rb.anise, 2_451_544.5 + t_abs_s / 86_400.0))
        })
        .collect();

    let scan_bodies: Vec<ScanBody> = resolved.iter().zip(state_fns.iter())
        .map(|(rb, f)| ScanBody {
            name: "", // filled from `names` when writing output, not needed by the solver
            mu_m3s2: rb.mu_m3s2,
            min_periapsis_m: rb.min_periapsis_m,
            state_at: f.as_ref(),
        })
        .collect();

    // Departure-epoch grid: ± horizon_years/2 around the config epoch,
    // absolute mission time in seconds since MJD2000 epoch (JD 2451544.5) —
    // matching the `t_abs_s` convention `state_fns` above assumes.
    const DAYS_PER_YEAR: f64 = 365.25;
    let half_span_days = 0.5 * scan_cfg.horizon_years * DAYS_PER_YEAR;
    let dep_jd_start = dep_jd_center - half_span_days;
    let dep_jd_end = dep_jd_center + half_span_days;
    let n_dep_steps = ((dep_jd_end - dep_jd_start) / scan_cfg.departure_step_days).floor() as usize + 1;
    let departure_epochs_s: Vec<f64> = (0..n_dep_steps)
        .map(|i| (dep_jd_start + i as f64 * scan_cfg.departure_step_days - 2_451_544.5) * 86_400.0)
        .collect();

    // Per-leg TOF bounds, by ARRAY POSITION with the last entry as a
    // fallback when the discovered/configured sequence has more legs than
    // leg_tof_days entries — same convention `mga.rs`'s DE search already
    // uses for auto-discovered sequences (see its `.get(k).or_else(||
    // .last())` in the chromosome-bounds builder), so a config written for
    // sequence_search doesn't need to predict the discovered leg count.
    let leg_tof_grids_s: Vec<Vec<f64>> = (0..n_legs)
        .map(|k| {
            let [lo, hi] = mga.leg_tof_days.get(k).or_else(|| mga.leg_tof_days.last())
                .expect("checked non-empty above");
            (0..scan_cfg.tof_grid_points_per_leg)
                .map(|i| {
                    let frac = i as f64 / (scan_cfg.tof_grid_points_per_leg - 1).max(1) as f64;
                    (lo + frac * (hi - lo)) * 86_400.0
                })
                .collect()
        })
        .collect();

    let scan_config = MgaScanConfig {
        mu_sun: MU_SUN_M3S2,
        departure_epochs_s,
        leg_tof_grids_s,
        vinf_dep_max_ms: mga.departure_vinf_max_ms,
        flyby_dv_max_ms: scan_cfg.flyby_dv_max_ms,
        vinf_arr_max_ms: mga.arrival_vinf_max_ms,
        max_records: scan_cfg.max_records,
    };

    println!(
        "Running Phase 9w ballistic MGA scan: {} → {} ({} legs), {} departure dates x {} TOF points/leg",
        opt.departure_body, opt.target_body, n_legs, n_dep_steps, scan_cfg.tof_grid_points_per_leg
    );
    let t_start = std::time::Instant::now();
    let out = run_mga_scan(&scan_bodies, &scan_config);
    let elapsed_s = t_start.elapsed().as_secs_f64();

    // Capture ΔV per record (Orbit-style cost) — `None` for every record
    // when [trajectory.capture].target_orbit_radius_m isn't configured, in
    // which case only the Flyby-style cost (no arrival burn) is available.
    let target = resolved.last().expect("names always has >= 2 entries (departure + target)");
    let capture_dvs: Vec<Option<f64>> = out.records.iter()
        .map(|rec| capture_dv_ms(cfg, rec.vinf_arr_ms, target.mu_m3s2, target.radius_m))
        .collect();

    Ok((out.records, capture_dvs, names, out.n_legs_evaluated, out.n_records_dropped, elapsed_s))
}

fn write_scan_csv(cfg: &MissionConfig, records: &[ScanRecord], capture_dvs: &[Option<f64>], names: &[String]) -> Result<(), String> {
    let out_dir = &cfg.simulation.output_dir;
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let path = format!("{out_dir}/mga_window_scan.csv");

    let n_flybys = names.len() - 2;
    let mut header = String::from("dep_mjd2000,vinf_dep_ms,vinf_arr_ms,sum_flyby_dv_ms,total_tof_days");
    for i in 0..n_flybys {
        header.push_str(&format!(",leg{i}_tof_days,flyby{i}_dv_ms,flyby{i}_rp_km,flyby{i}_turn_deg"));
    }
    header.push_str(&format!(",leg{n_flybys}_tof_days,capture_dv_ms\n"));

    let mut body = String::new();
    for (rec, capture_dv) in records.iter().zip(capture_dvs.iter()) {
        let dep_mjd2000 = rec.dep_epoch_s / 86_400.0;
        let total_tof_days: f64 = rec.leg_tofs_s.iter().sum::<f64>() / 86_400.0;
        body.push_str(&format!(
            "{:.4},{:.2},{:.2},{:.2},{:.3}",
            dep_mjd2000, rec.vinf_dep_ms, rec.vinf_arr_ms, rec.sum_flyby_dv_ms, total_tof_days
        ));
        for i in 0..n_flybys {
            body.push_str(&format!(
                ",{:.3},{:.2},{:.1},{:.3}",
                rec.leg_tofs_s[i] / 86_400.0,
                rec.flyby_dvs_ms[i],
                rec.flyby_rp_m[i] / 1e3,
                rec.flyby_turn_rad[i].to_degrees(),
            ));
        }
        body.push_str(&format!(",{:.3}", rec.leg_tofs_s[n_flybys] / 86_400.0));
        match capture_dv {
            Some(dv) => body.push_str(&format!(",{dv:.2}\n")),
            None => body.push_str(",\n"),
        }
    }

    std::fs::write(&path, header + &body).map_err(|e| e.to_string())?;
    println!("  Written: {path}  (plot: py plot/plot_mga_porkchop.py <mission>)");
    Ok(())
}
