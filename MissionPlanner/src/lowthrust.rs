//! Phase 9h — Sims-Flanagan low-thrust trajectory optimization.
//!
//! Reads departure and arrival body states from ANISE, builds a
//! `SimsFlanagan` transcription problem, runs the DE/rand/1/bin optimizer,
//! and writes results to CSV.
//!
//! The Sims-Flanagan method divides the TOF into N segments. Each segment
//! has a constant thrust vector (the decision variable). Forward and backward
//! half-arcs are propagated from departure and arrival respectively; the
//! six-component match-point defect (position + velocity mismatch at the
//! mid-time) is included as a penalty in the fitness function. The DE
//! optimizer minimizes control effort × match-point penalty + throttle
//! violation penalty.
//!
//! # Reference
//! Sims, J.A. and Flanagan, S.N. (1997): "Preliminary Design of Low-Thrust
//! Interplanetary Missions", AAS/AIAA Astrodynamics Specialist Conference,
//! paper AAS 97-636.

use std::fs;
use std::path::Path;

use ephemeris::Almanac;
use nalgebra::Vector3;
use trajectory_solver::{
    keplerian::MU_SUN_M3S2,
    sims_flanagan::{ArcHalf, SFResult, SimsFlanagan},
};

use crate::config::{EphemerisSource, MissionConfig};
use crate::design::{anise_body, body_state, epoch_to_jd, parse_epoch};

// ── Defaults for the Sims-Flanagan DE optimizer ───────────────────────────────

/// DE population size. Chromosome has 3 × N_SEGMENTS = 60 dimensions; rule
/// of thumb ≥ 10× gives 600, but 300 is a good balance of speed vs. coverage.
const DEFAULT_POP_SIZE: usize = 300;
/// DE generation count. More generations mean a better-converged solution at
/// the cost of wall-clock time.
const DEFAULT_GENERATIONS: usize = 1_000;
/// DE mutation scale factor F ∈ [0.4, 1.0] — 0.8 is the standard
/// recommendation for continuous, coupled parameters (Storn & Price 1997).
const DEFAULT_F_WEIGHT: f64 = 0.8;
/// DE crossover probability CR — 0.9 is recommended for tightly-correlated
/// parameter spaces (segment thrust vectors are interdependent due to the
/// match-point constraint).
const DEFAULT_CR: f64 = 0.9;
/// Number of equal thrust segments. Must be even.
const DEFAULT_N_SEGMENTS: usize = 20;
/// RNG seed for reproducibility.
const DEFAULT_SEED: u64 = 42;

// ── Entry point ───────────────────────────────────────────────────────────────

/// Run the Sims-Flanagan low-thrust optimizer for the mission described by
/// `cfg`.
///
/// Reads `[optimization]` for departure/arrival bodies and epoch, reads
/// `[spacecraft.propulsion]` for thrust and Isp. If `[optimization]` is
/// absent, falls back to `[trajectory]` for the departure epoch and uses
/// a default TOF of 300 days to Mars (or the config's cruise max-TOF when
/// present).
///
/// Writes:
/// - `<output_dir>/sf_arc.csv`    — all arc points (forward + backward)
/// - `<output_dir>/sf_thrust.csv` — per-segment thrust vectors and magnitudes
pub fn run_low_thrust(cfg: &MissionConfig, almanac: &Almanac) -> Result<(), String> {
    // ── Spacecraft propulsion parameters ─────────────────────────────────────

    let prop = cfg.spacecraft.propulsion.as_ref().ok_or(
        "low-thrust requires [spacecraft.propulsion] with type, isp_s, and thrust_n",
    )?;
    let t_max_n = prop.thrust_n;
    let isp_s = prop.isp_s;
    let m0_kg = cfg.spacecraft.mass_kg;

    println!("Propulsion:  type={}, Isp={:.0} s, T_max={:.3} N, m0={:.1} kg",
        prop.kind, isp_s, t_max_n, m0_kg);

    // ── Departure and arrival body names ─────────────────────────────────────

    let (dep_name, target_name, dep_epoch_str, tof_days) = resolve_transfer_params(cfg)?;

    println!("Departure:   {} → {}  at  {}  TOF = {:.1} days",
        dep_name, target_name, dep_epoch_str, tof_days);

    // ── ANISE body lookup ─────────────────────────────────────────────────────

    let dep_anise = anise_body(&dep_name.to_lowercase())
        .ok_or_else(|| format!("departure body '{}' is not ANISE-covered (must be a major planet)", dep_name))?;
    let arr_anise = anise_body(&target_name.to_lowercase())
        .ok_or_else(|| format!("target body '{}' is not ANISE-covered (must be a major planet)", target_name))?;

    let dep_epoch = parse_epoch(&dep_epoch_str)
        .map_err(|e| format!("departure epoch '{}': {e}", dep_epoch_str))?;
    let dep_jd = epoch_to_jd(dep_epoch);
    let arr_jd = dep_jd + tof_days;

    let (r0_arr, v0_arr) = body_state(almanac, EphemerisSource::Anise, Some(dep_anise), &None, dep_jd)
        .ok_or_else(|| format!("ANISE failed to return state for '{}' at departure JD {dep_jd:.2}", dep_name))?;
    let (rf_arr, vf_arr) = body_state(almanac, EphemerisSource::Anise, Some(arr_anise), &None, arr_jd)
        .ok_or_else(|| format!("ANISE failed to return state for '{}' at arrival JD {arr_jd:.2}", target_name))?;

    let r0 = Vector3::new(r0_arr[0], r0_arr[1], r0_arr[2]);
    let v0 = Vector3::new(v0_arr[0], v0_arr[1], v0_arr[2]);
    let rf = Vector3::new(rf_arr[0], rf_arr[1], rf_arr[2]);
    let vf = Vector3::new(vf_arr[0], vf_arr[1], vf_arr[2]);

    println!("r0 = [{:.3e}, {:.3e}, {:.3e}] m", r0[0], r0[1], r0[2]);
    println!("rf = [{:.3e}, {:.3e}, {:.3e}] m", rf[0], rf[1], rf[2]);

    // ── Build and run the Sims-Flanagan problem ───────────────────────────────

    let sf = SimsFlanagan {
        r0,
        v0,
        rf,
        vf,
        mu: MU_SUN_M3S2,
        tof_s: tof_days * 86_400.0,
        n_segments: DEFAULT_N_SEGMENTS,
        m0_kg,
        t_max_n,
        isp_s,
    };

    println!(
        "\nSims-Flanagan:  N={} segments,  pop={},  gen={},  F={},  CR={}",
        DEFAULT_N_SEGMENTS, DEFAULT_POP_SIZE, DEFAULT_GENERATIONS, DEFAULT_F_WEIGHT, DEFAULT_CR
    );
    println!("Running DE optimizer…");

    let t_start = std::time::Instant::now();
    let result = sf.run(
        DEFAULT_POP_SIZE,
        DEFAULT_GENERATIONS,
        DEFAULT_F_WEIGHT,
        DEFAULT_CR,
        DEFAULT_SEED,
    );
    let elapsed = t_start.elapsed();

    // ── Print match-point residuals ───────────────────────────────────────────

    println!("\n--- Sims-Flanagan Result ---");
    println!("  Elapsed:           {:.1} s", elapsed.as_secs_f64());
    println!("  Best fitness:      {:.6e}", result.best_fitness);
    println!("  Match-point Δr:    {:.3e} m  ({:.1} km)", result.match_point_r_err_m, result.match_point_r_err_m / 1e3);
    println!("  Match-point Δv:    {:.3e} m/s", result.match_point_v_err_ms);
    println!("  Total ΔV (proxy):  {:.1} m/s  ({:.3} km/s)", result.dv_total_ms, result.dv_total_ms / 1e3);
    println!("  Converged:         {}", result.converged);

    // ── Write output files ────────────────────────────────────────────────────

    let out_dir = &cfg.simulation.output_dir;
    fs::create_dir_all(out_dir)
        .map_err(|e| format!("failed to create output directory '{}': {e}", out_dir))?;

    write_arc_csv(out_dir, &result, tof_days)?;
    write_thrust_csv(out_dir, &result, tof_days)?;

    println!("\nOutput files:");
    println!("  {out_dir}/sf_arc.csv");
    println!("  {out_dir}/sf_thrust.csv");

    Ok(())
}

// ── Parameter resolution ──────────────────────────────────────────────────────

/// Resolve (departure_body, target_body, departure_epoch_str, tof_days) from
/// the config. Prefers `[optimization]` fields; falls back to `[trajectory]`
/// when `[optimization]` is absent.
fn resolve_transfer_params(cfg: &MissionConfig) -> Result<(String, String, String, f64), String> {
    if let Some(opt) = &cfg.optimization {
        let dep = opt.departure_body.clone();
        let tgt = opt.target_body.clone();
        let epoch = opt
            .departure_epoch
            .clone()
            .ok_or("optimization.departure_epoch is required for low-thrust")?;
        // Use max_coast_days as the TOF — a reasonable upper bound.
        let tof = opt.max_coast_days;
        return Ok((dep, tgt, epoch, tof));
    }

    // Fall back to [trajectory] fields.
    let dep = cfg.trajectory.departure_body.clone();
    let tgt = cfg.target_body.name.clone();
    let epoch = cfg.trajectory.departure_epoch.clone()
        .ok_or("[trajectory].departure_epoch is required for low-thrust when [optimization] is absent")?;
    let tof = cfg
        .trajectory
        .cruise
        .as_ref()
        .and_then(|c| c.tof_days_max)
        .unwrap_or(300.0);

    Ok((dep, tgt, epoch, tof))
}

// ── CSV writers ───────────────────────────────────────────────────────────────

/// Write all arc points (forward + backward) to `<out_dir>/sf_arc.csv`.
fn write_arc_csv(out_dir: &str, result: &SFResult, tof_days: f64) -> Result<(), String> {
    let path = Path::new(out_dir).join("sf_arc.csv");
    let mut buf = String::from("arc,t_days,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps\n");

    // Combine forward then backward arcs, deduplicate the shared match-point
    // (it appears as the last point of arc_fwd and the first of arc_bwd).
    let all_fwd = &result.arc_fwd;
    let all_bwd = &result.arc_bwd;

    for pt in all_fwd.iter() {
        buf.push_str(&arc_row(pt, tof_days));
    }
    // Skip arc_bwd[0] (duplicate of arc_fwd's last point, the match point).
    for pt in all_bwd.iter().skip(1) {
        buf.push_str(&arc_row(pt, tof_days));
    }

    fs::write(&path, buf)
        .map_err(|e| format!("failed to write {}: {e}", path.display()))
}

fn arc_row(pt: &trajectory_solver::sims_flanagan::SFPoint, _tof_days: f64) -> String {
    let label = match pt.arc {
        ArcHalf::Forward => "fwd",
        ArcHalf::Backward => "bwd",
    };
    format!(
        "{},{:.6},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n",
        label,
        pt.t_s / 86_400.0,
        pt.r_m[0], pt.r_m[1], pt.r_m[2],
        pt.v_mps[0], pt.v_mps[1], pt.v_mps[2],
    )
}

/// Write per-segment thrust vectors to `<out_dir>/sf_thrust.csv`.
fn write_thrust_csv(out_dir: &str, result: &SFResult, tof_days: f64) -> Result<(), String> {
    let path = Path::new(out_dir).join("sf_thrust.csv");
    let mut buf = String::from("seg,t_mid_days,ux_ms2,uy_ms2,uz_ms2,u_mag_ms2\n");

    let n = result.thrust_vectors.len();
    let dt_days = tof_days / n as f64;

    for (k, &[ux, uy, uz]) in result.thrust_vectors.iter().enumerate() {
        let u_mag = (ux * ux + uy * uy + uz * uz).sqrt();
        let t_mid = (k as f64 + 0.5) * dt_days;
        buf.push_str(&format!(
            "{},{:.6},{:.6e},{:.6e},{:.6e},{:.6e}\n",
            k, t_mid, ux, uy, uz, u_mag
        ));
    }

    fs::write(&path, buf)
        .map_err(|e| format!("failed to write {}: {e}", path.display()))
}
