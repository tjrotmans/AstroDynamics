//! wsb_search — Ballistic Lunar Transfer search following Circi & Teofilatto (2001)
//!              and Belbruno & Carrico (2000).
//!
//! # Strategy
//!
//! The search runs in four sequential phases, each building on the last:
//!
//! ## Phase 1 — Backward propagation from Moon (Circi §2)
//!
//! Start at the Moon with a grid of selenocentric orbital parameters
//! (periselene height, eccentricity, inclination, RAAN).  Integrate backward
//! in BCR4BP until the trajectory either reaches Earth-perigee altitude or
//! exceeds the maximum propagation time.  Two filters are applied:
//!
//!   (a) β-angle filter  — the angle between the Earth–Moon and Earth–Sun
//!       directions at periselene must fall inside one of the two ~60° windows
//!       identified by Circi (Fig. 6).  Only β ∈ [60°,150°] ∪ [240°,330°]
//!       produce valid WSB transfers.
//!
//!   (b) Earth perigee filter — backward integration must reach a perigee in
//!       the 200–1000 km altitude band, closing the loop to a real parking orbit.
//!
//! ## Phase 2 — α-angle forward screen (Circi §3.1)
//!
//! For every backward solution found in Phase 1, propagate forward from the
//! recovered Earth perigee.  Locate the apogee of the outgoing arc and compute
//! the α angle (Earth-to-apogee vs Earth-to-Sun).  Per Circi Fig. 8, the Sun
//! pumps energy into the orbit only when α is in quadrant II (90°–180°) or
//! quadrant IV (270°–360°).  Trajectories whose α falls in quadrant I or III
//! are rejected — the Sun kicks the spacecraft *back* toward the Moon too soon.
//!
//! ## Phase 3 — Forward refinement from Earth parking orbit (Belbruno §2.1)
//!
//! Survivors of Phase 2 are re-propagated in BCR4BP from the Earth parking
//! orbit.  The Belbruno forward algorithm sweeps injection angle θ and a
//! fine grid of Sun phases θ_sun concentrated in the two valid α windows.
//! Injection speed is derived from vis-viva targeting apogee ≈ 3.9 nd
//! (≈ 1.5×10⁶ km, the Sun–Earth WSB / SE-L1 region).  All captures are
//! collected — deduplication and TOP_N filtering happen only in Phase 4.
//!
//! ## Phase 4 — Envelope sweep + family analysis
//!
//! Every Phase 3 capture seed is used to launch a dense local refinement:
//! a fine (θ ± ENVELOPE_DTHETA) × (θ_sun ± ENVELOPE_DSUN) grid is swept
//! around each winner, and also a sweep of apogee distances around
//! R_APOGEE_TARGET to map the ΔV–time trade-off.  All surviving captures
//! are then:
//!
//!   1. Classified into families by α quadrant (Q2 = "direct-sun" family,
//!      Q4 = "anti-sun" family) and by transfer time bracket.
//!   2. Placed on a Pareto front in (ΔV, transfer_time) space — a solution
//!      is Pareto-optimal if no other solution is both faster AND cheaper.
//!   3. Written to a family CSV with one row per solution, suitable for
//!      scatter-plotting ΔV vs transfer time coloured by family.
//!
//! # Outputs  (all in out/wsb_search/)
//!
//!   backward_solutions.csv   — Phase 1 trajectories (full backward arc)
//!   forward_screen.csv       — Phase 2 Earth-perigee ICs that pass α filter
//!   blt_candidates.csv       — Phase 3+4 full transfer arcs (truncated at Hill)
//!   family_analysis.csv      — one row per solution: θ, θ_sun, ΔV, t_transfer,
//!                              capture_dur, alpha, family, pareto_rank
//!   blt_info.txt             — ranked candidate summary + Pareto table
//!
//! Usage:  cargo run -p lunar_trajectories --bin wsb_search --release

use std::f64::consts::PI;
use std::fs;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use rayon::prelude::*;

use lunar_trajectories::crtbp::CrtbpParams;
use lunar_trajectories::propagator::{propagate_bcr4bp, Bcr4bpParams, Step3d};
use lunar_trajectories::transfers::{
    c_lagrange, lunar_hill_radius, detect_capture, tli_injection_ic,
};

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                         USER CONFIGURATION                                  ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ── Parking orbit ─────────────────────────────────────────────────────────────

/// Earth parking orbit radius [nd].  378 km altitude (Artemis II perigee).
const R_PARK: f64 = (6_371.0 + 378.0) / 384_400.0;

/// Perigee altitude band accepted as a valid Earth departure orbit [nd].
/// Corresponds to roughly 200–1000 km altitude.
const R_PERIGEE_MIN: f64 = (6_371.0 + 200.0)  / 384_400.0;
const R_PERIGEE_MAX: f64 = (6_371.0 + 1_000.0) / 384_400.0;

// ── Phase 1 — backward grid from Moon ─────────────────────────────────────────

/// Periselene heights to try [km].  Circi uses 200 km; we sweep a small band.
const PERI_HEIGHTS_KM: &[f64] = &[200.0, 500.0, 1_000.0];

/// Selenocentric eccentricities at periselene.  Circi starts at e_M = 0.92
/// and iterates upward; we sweep a coarse grid and rely on the perigee filter
/// to select the geometrically consistent ones.
const N_ECC: usize = 16;
const ECC_MIN: f64 = 0.92;
const ECC_MAX: f64 = 1.05;   // slightly hyperbolic — still quasi-ballistic capture

/// Inclinations of the selenocentric departure orbit [deg].
const INCL_DEG: &[f64] = &[0.0, 30.0, 60.0, 90.0];

/// RAAN of the selenocentric departure orbit [deg].  Circi sweeps Ω_M.
const N_RAAN: usize = 12;

/// Sun phase angles to try in the backward propagation.  We concentrate
/// samples inside the two β windows (see β filter below) and add a coarse
/// global sweep so we don't miss edge cases.
const N_SUN_COARSE: usize = 18;
const N_SUN_FINE:   usize = 20;

/// Backward propagation time [nd].  15π ≈ 210 days — long enough to reach
/// Earth perigee from Moon on a WSB arc (Circi: typical transfer 60–100 days).
const T_BACKWARD: f64 = 15.0 * PI;

// ── Phase 2 — α-angle forward screen ─────────────────────────────────────────

/// Forward propagation time for the α-angle screen [nd].  6π ≈ 84 days —
/// enough to reach apogee (~45 days) and confirm the quadrant.
const T_SCREEN: f64 = 6.0 * PI;

// ── Phase 3 — Belbruno forward sweep ─────────────────────────────────────────

/// Target apogee for TLI [nd].  Belbruno & Carrico §3.2: apogee ≈ 1.5×10⁶ km
/// = 3.9 nd.  The Sun–Earth WSB sits at ~3.9 nd from Earth.
///
/// NOTE: we do NOT hard-code a Jacobi constant C here.  In the EM rotating
/// frame C = 2Ω(r,θ) − v² depends strongly on position, so the same C value
/// gives a completely different apogee at different injection angles and does
/// NOT correspond to a fixed energy in the inertial frame.  Instead we derive
/// the injection speed directly from the vis-viva relation:
///
///   a   = (R_park + R_apogee) / 2          semi-major axis of transfer ellipse
///   v²  = (1 − mu) · (2/R_park − 1/a)      vis-viva, Earth point-mass gravity
///
/// This is what Belbruno & Carrico actually do (their C3 = −μ/a, eq. 4).
/// The Jacobi constant C then takes whatever value it has at the resulting IC —
/// the BCR4BP integrator only needs the state vector.
const R_APOGEE_TARGET: f64 = 3.9;   // nd ≈ 1.5×10⁶ km — near SE-L1 / Earth-Sun WSB

/// Injection angle sweep: 360 values over [0, 2π).
const N_THETA: usize = 360;

/// Sun phase sweep — two fine windows corresponding to Q2 and Q4 α geometry,
/// plus a coarse global pass to catch any outliers.
const N_SUN_P3_COARSE: usize = 36;
const N_SUN_P3_FINE:   usize = 48;

/// Forward propagation time for Phase 3 [nd].  20π ≈ 220 days.
const T_PROP: f64 = 20.0 * PI;

/// Extended propagation time for the top-N output trajectories [nd].
/// 20π ≈ 280 days — same budget as search but enough to see capture orbits.
const T_PROP_FINAL: f64 = T_PROP;

/// One lunar orbital period in non-dimensional time [nd].  = 2π by definition.
const LUNAR_PERIOD_ND: f64 = 2.0 * PI;

/// Minimum Hill sphere dwell time to count as capture [nd].  ≈ 2.4 days.
const MIN_CAPTURE_TIME: f64 = 0.15;

/// Target: at least this many continuous lunar orbits inside the Hill sphere.
const MIN_CAPTURE_ORBITS: f64 = 3.0;

/// Hill sphere entry must occur after this time — rejects fast direct transfers.
/// 5π ≈ 70 days is the minimum a true WSB arc takes.
const T_MIN_HILL: f64 = 5.0 * PI;

/// Must reach this far from Earth to confirm SE-L1 excursion [nd].  ~3.5 nd.
const R_MIN_EXCURSION: f64 = 3.0;

/// Score = est_capture_orbits − LAMBDA × min_lunar_dist_nd + speed bonus.
/// Orbit count dominates so solutions that stay around the Moon rank highest.
const LAMBDA: f64 = 1.0;

/// Bonus weight applied to fast transfers (< T_FAST_TARGET) in the score.
/// Tapers linearly from SPEED_BONUS_WEIGHT at t=0 to 0 at T_FAST_TARGET.
const SPEED_BONUS_WEIGHT: f64 = 0.5;

/// Transfer time threshold for the speed bonus [nd].  ~80 days.
const T_FAST_TARGET: f64 = 6.0 * PI;

/// Maximum candidates to keep after deduplication.
const TOP_N: usize = 5;

/// Minimum angular separation (θ) between saved candidates [deg].
const MIN_THETA_SEP: f64 = 5.0;

/// Maximum total candidates carried into family classification and Pareto ranking.
/// The dense 8×8 envelope produces a reliable landscape map for the island GA.
const MAX_CANDIDATES: usize = 10_000;

// ── Phase 4 — Envelope sweep + family analysis ────────────────────────────────

/// Half-width of the local θ refinement around each Phase 3 seed [deg].
/// A ±ENVELOPE_DTHETA × ±ENVELOPE_DSUN box is swept at finer resolution.
const ENVELOPE_DTHETA: f64 = 8.0;   // deg — ±8° around seed injection angle
const ENVELOPE_DSUN:   f64 = 10.0;  // deg — ±10° around seed Sun phase

/// Number of steps across each envelope dimension.
/// Dense 8×8 grid maps the family landscape reliably; the island GA then
/// exploits each family's θ window instead of re-discovering it.
const ENVELOPE_N_THETA: usize = 8;
const ENVELOPE_N_SUN:   usize = 8;

/// Apogee distances to sweep in the envelope [nd].
/// Maps ΔV vs transfer-time trade-off: lower apogee = shorter time but higher ΔV.
const ENVELOPE_APOGEES: &[f64] = &[2.8, 3.2, 3.5, 3.7, 3.9, 4.1, 4.3, 4.6];

/// Transfer time budget for family classification boundaries [nd].
///   "fast"   < T_FAMILY_FAST        (roughly < 80 days)
///   "medium" < T_FAMILY_MEDIUM      (roughly 80–140 days)
///   "slow"   ≥ T_FAMILY_MEDIUM      (roughly > 140 days)
const T_FAMILY_FAST:   f64 = 6.0  * PI;   // ~84 days
const T_FAMILY_MEDIUM: f64 = 10.0 * PI;   // ~141 days

// ── Integrator ────────────────────────────────────────────────────────────────
const LOG_DT: f64 = 0.06;
const RTOL:   f64 = 1e-8;
const ATOL:   f64 = 1e-8;

// ── Output ────────────────────────────────────────────────────────────────────
const OUT_DIR: &str = "out/wsb";

// ── Runtime config (overrideable via env var WSB_R_PARK_ND / WSB_R_APOGEE_ND) ─
//
// Set by wsb_pipeline before invoking this binary.  Falls back to the compile-time
// constants above when running standalone.
static _CFG_R_PARK:   OnceLock<f64> = OnceLock::new();
static _CFG_R_APOGEE: OnceLock<f64> = OnceLock::new();

fn cfg_r_park() -> f64 {
    *_CFG_R_PARK.get_or_init(|| {
        std::env::var("WSB_R_PARK_ND").ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(R_PARK)
    })
}
fn cfg_r_apogee() -> f64 {
    *_CFG_R_APOGEE.get_or_init(|| {
        std::env::var("WSB_R_APOGEE_ND").ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(R_APOGEE_TARGET)
    })
}

// ════════════════════════════════════════════════════════════════════════════════

/// A backward solution that survived both the β filter and the Earth perigee filter.
#[allow(dead_code)]
struct BackwardSolution {
    /// Selenocentric parameters used to seed the backward propagation.
    peri_km:       f64,
    ecc:           f64,
    incl_deg:      f64,
    raan_deg:      f64,
    theta_sun_deg: f64,
    /// β angle at periselene [deg] — Earth–Moon vs Earth–Sun direction.
    beta_deg:      f64,
    /// Minimum distance from Earth centre reached during backward arc [nd].
    min_earth_dist: f64,
    /// The full backward trajectory (Moon → Earth).
    traj:          Vec<Step3d>,
}

/// A capture candidate — produced by Phase 3 or Phase 4.
struct Candidate {
    theta_deg:             f64,
    theta_sun_deg:         f64,
    dv_nd:                 f64,
    dv_km_s:               f64,
    /// Time of first Hill sphere entry [nd] — the actual transfer time.
    t_transfer_nd:         f64,
    t_transfer_days:       f64,
    capture_duration_nd:   f64,
    capture_duration_days: f64,
    n_entries:             usize,
    min_lunar_dist_nd:     f64,
    min_lunar_alt_km:      f64,
    max_interval_nd:       f64,
    max_interval_days:     f64,
    /// Estimated number of continuous lunar orbits inside the Hill sphere.
    /// = max_capture_interval / LUNAR_PERIOD_ND.  ≥ 3 is the quality target.
    est_capture_orbits:    f64,
    /// True if the minimum lunar distance dips below the Moon's surface.
    /// Such solutions are flagged but NOT discarded — a nearby trajectory in
    /// (θ, θ_sun) space may thread the needle without impacting.
    crashed_moon:          bool,
    /// α angle at apogee [deg] — determines which solar-assist family this is.
    alpha_deg:             f64,
    /// Maximum distance from Earth reached [nd] — proxy for actual apogee.
    apogee_nd:             f64,
    /// Target apogee used to compute this TLI [nd].
    r_apogee_target:       f64,
    /// Family label assigned in Phase 4: "Q2-fast", "Q2-medium", "Q2-slow",
    /// "Q4-fast", "Q4-medium", "Q4-slow".
    family:                String,
    /// Pareto rank in (ΔV, transfer_time) space.  1 = Pareto-optimal.
    pareto_rank:           usize,
    score:                 f64,
    /// Injection IC — stored so we can re-propagate for longer output trajectories.
    ic:                    [f64; 6],
    traj:                  Vec<Step3d>,
}

// ════════════════════════════════════════════════════════════════════════════════

fn main() {
    // Read runtime config (overrideable via env var from wsb_pipeline)
    let r_park          = cfg_r_park();
    let r_apogee_target = cfg_r_apogee();

    let params    = CrtbpParams::earth_moon();
    let mu        = params.mu;
    let l_km      = params.l_star / 1e3;
    let v_km_s    = params.v_star / 1e3;
    let r_moon_km = 1_737.4_f64;
    let moon_x    = 1.0 - mu;
    let x_earth   = -mu;

    let c_l1   = c_lagrange(mu, 1);
    let c_l2   = c_lagrange(mu, 2);
    let r_hill = lunar_hill_radius(mu);

    fs::create_dir_all(OUT_DIR).unwrap();

    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!("║        WSB Ballistic Lunar Transfer Search           ║");
    eprintln!("╚══════════════════════════════════════════════════════╝");
    eprintln!("  μ      = {mu:.6}");
    eprintln!("  C_L1   = {c_l1:.6}   C_L2 = {c_l2:.6}");
    eprintln!("  r_Hill = {r_hill:.5} nd  = {:.0} km", r_hill * l_km);
    eprintln!("  R_park = {r_park:.5} nd  ({:.0} km alt)",
        r_park * l_km - 6_371.0);
    eprintln!();

    // ── Phase 1 ──────────────────────────────────────────────────────────────
    eprintln!("── Phase 1 : Backward propagation from Moon (Circi §2) ──────────");
    let bwd_solutions = phase1_backward(mu, &params, moon_x, x_earth, r_hill, l_km);
    eprintln!("  Phase 1 complete — {} solutions passed β + perigee filters",
        bwd_solutions.len());
    save_backward_csv(&bwd_solutions);
    eprintln!();

    // ── Phase 2 ──────────────────────────────────────────────────────────────
    eprintln!("── Phase 2 : α-angle forward screen (Circi §3.1) ───────────────");
    let screened = phase2_alpha_screen(mu, &params, x_earth, &bwd_solutions);
    // Free backward trajectories — no longer needed after Phase 2
    drop(bwd_solutions);
    eprintln!("  Phase 2 complete — {} solutions have α in Q2 or Q4", screened.len());
    save_screen_csv(&screened);
    eprintln!();

    // ── Phase 3 ──────────────────────────────────────────────────────────────
    eprintln!("── Phase 3 : Belbruno forward sweep (Belbruno §2.1) ────────────");
    let phase3_candidates = phase3_forward_sweep(mu, &params, x_earth, moon_x, r_hill,
                                                  l_km, v_km_s, r_moon_km, r_park,
                                                  r_apogee_target, &screened);
    eprintln!("  Phase 3 complete — {} raw captures", phase3_candidates.len());
    eprintln!();

    // ── Phase 4 ──────────────────────────────────────────────────────────────
    eprintln!("── Phase 4 : Envelope sweep + family analysis ──────────────────");
    let mut all_candidates = phase4_envelope(mu, &params, x_earth, moon_x, r_hill,
                                              l_km, v_km_s, r_moon_km, r_park,
                                              phase3_candidates);

    // Trim to MAX_CANDIDATES by score before O(n log n) Pareto ranking.
    // Keeps the best solutions across the full (ΔV, time) trade-off space.
    if all_candidates.len() > MAX_CANDIDATES {
        all_candidates.sort_unstable_by(|a, b|
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_candidates.truncate(MAX_CANDIDATES);
        eprintln!("  Trimmed to top {MAX_CANDIDATES} by score");
    }

    classify_families(&mut all_candidates);
    assign_pareto_ranks(&mut all_candidates);
    eprintln!("  Phase 4 complete — {} total solutions classified", all_candidates.len());
    eprintln!();

    save_candidates(mu, &params, l_km, r_hill, all_candidates);
}

// ════════════════════════════════════════════════════════════════════════════════
// Phase 1 — backward propagation from Moon
// ════════════════════════════════════════════════════════════════════════════════

fn phase1_backward(
    mu:      f64,
    params:  &CrtbpParams,
    moon_x:  f64,
    x_earth: f64,
    _r_hill: f64,
    _l_km:   f64,
) -> Vec<BackwardSolution> {

    let l_nd       = params.l_star / 1e3;
    let sun_phases = build_sun_phase_grid_p1();

    let total = PERI_HEIGHTS_KM.len() * N_ECC * INCL_DEG.len() * N_RAAN * sun_phases.len();
    eprintln!("  grid: {} peri × {} ecc × {} incl × {} RAAN × {} θ_sun = {} propagations",
        PERI_HEIGHTS_KM.len(), N_ECC, INCL_DEG.len(), N_RAAN, sun_phases.len(), total);

    let n_propagated = AtomicUsize::new(0);
    let n_beta_pass  = AtomicUsize::new(0);

    // Build flat work list: (peri_km, ie, incl_deg, ir)
    let work: Vec<(f64, usize, f64, usize)> = PERI_HEIGHTS_KM.iter()
        .flat_map(|&p| (0..N_ECC).flat_map(move |ie|
            INCL_DEG.iter().flat_map(move |&inc|
                (0..N_RAAN).map(move |ir| (p, ie, inc, ir)))))
        .collect();

    let solutions: Vec<BackwardSolution> = work
        .into_par_iter()
        .flat_map(|(peri_km, ie, incl_deg, ir)| {
            let r_peri   = (1_737.4 + peri_km) / l_nd;
            let ecc      = ECC_MIN + (ECC_MAX - ECC_MIN) * ie as f64 / (N_ECC - 1) as f64;
            let incl     = incl_deg.to_radians();
            let raan_deg = 360.0 * ir as f64 / N_RAAN as f64;
            let raan     = raan_deg.to_radians();

            let v_peri = (mu * (1.0 + ecc) / r_peri).sqrt();
            let px = moon_x + r_peri * raan.cos();
            let py =          r_peri * raan.sin();
            let vx = -v_peri * incl.cos() * raan.sin();
            let vy =  v_peri * incl.cos() * raan.cos();
            let vz =  v_peri * incl.sin();
            let ic: [f64; 6] = [px, py, 0.0, vx, vy, vz];

            let mut local: Vec<BackwardSolution> = Vec::new();
            for &theta_sun in &sun_phases {
                let prev = n_propagated.fetch_add(1, Ordering::Relaxed);
                if prev % 5_000 == 0 {
                    eprintln!("  [P1 {}/{total}]  β-pass={}  solutions={}",
                        prev, n_beta_pass.load(Ordering::Relaxed), local.len());
                }

                // β = angle of Sun relative to Earth-Moon direction.
                // In the EM rotating frame the Earth-Moon line is always +x (angle = 0),
                // so β = −θ_sun mod 360°.  atan2(y=0, x=moon_x−x_earth) = 0 correctly.
                let em_angle: f64 = 0.0;   // Earth-Moon direction = +x in rotating frame
                let beta_deg = (em_angle - theta_sun).rem_euclid(2.0 * PI).to_degrees();
                if !beta_in_valid_window(beta_deg) { continue; }
                n_beta_pass.fetch_add(1, Ordering::Relaxed);

                let bcr    = Bcr4bpParams::earth_moon_sun(theta_sun);
                let ic_bwd = [ic[0], ic[1], ic[2], -ic[3], -ic[4], -ic[5]];
                let traj   = propagate_bcr4bp(mu, bcr, ic_bwd, T_BACKWARD, LOG_DT, RTOL, ATOL);

                let min_earth_dist = traj.iter().map(|s| {
                    let dx = s.x - x_earth;
                    (dx*dx + s.y*s.y + s.z*s.z).sqrt()
                }).fold(f64::MAX, f64::min);

                if min_earth_dist < R_PERIGEE_MIN || min_earth_dist > R_PERIGEE_MAX { continue; }

                local.push(BackwardSolution {
                    peri_km, ecc, incl_deg, raan_deg,
                    theta_sun_deg: theta_sun.to_degrees(),
                    beta_deg, min_earth_dist, traj,
                });
            }
            local
        })
        .collect();

    eprintln!("  propagated: {}   β-pass: {}   perigee-pass: {}",
        n_propagated.load(Ordering::Relaxed),
        n_beta_pass.load(Ordering::Relaxed),
        solutions.len());
    solutions
}

// ════════════════════════════════════════════════════════════════════════════════
// Phase 2 — α-angle forward screen
// ════════════════════════════════════════════════════════════════════════════════

/// A backward solution that also passed the α-angle screen.
/// Carries the recovered Earth-perigee IC for Phase 3 seeding.
/// Fields from BackwardSolution are inlined so we don't need to move the Vec.
#[allow(dead_code)]
struct ScreenedSolution {
    // ── provenance from Phase 1 ───────────────────────────────────────────────
    peri_km:       f64,
    ecc:           f64,
    incl_deg:      f64,
    raan_deg:      f64,
    theta_sun_deg: f64,
    beta_deg:      f64,
    // ── Phase 2 result ────────────────────────────────────────────────────────
    alpha_deg:     f64,
    /// State at Earth perigee (closest approach point on backward arc),
    /// with velocity sign restored to forward direction.
    ic_earth:      [f64; 6],
}

fn phase2_alpha_screen(
    mu:       f64,
    _params:  &CrtbpParams,
    x_earth:  f64,
    bwd:     &[BackwardSolution],
) -> Vec<ScreenedSolution> {

    eprintln!("  screening {} backward solutions for α quadrant …", bwd.len());

    let mut screened: Vec<ScreenedSolution> = Vec::new();

    for sol in bwd {
        // Locate the Earth-closest point on the backward arc — this is the
        // Earth-perigee state we will use as the forward IC.
        let peri_state = match sol.traj.iter().min_by(|a, b| {
            let ra = ((a.x - x_earth).powi(2) + a.y.powi(2) + a.z.powi(2)).sqrt();
            let rb = ((b.x - x_earth).powi(2) + b.y.powi(2) + b.z.powi(2)).sqrt();
            ra.partial_cmp(&rb).unwrap()
        }) {
            Some(s) => s,
            None    => continue,
        };

        // Forward IC at Earth perigee: propagate *forward* from here
        // (the backward arc had reversed velocity, so we reverse back).
        let ic_earth: [f64; 6] = [
            peri_state.x,  peri_state.y,  peri_state.z,
           -peri_state.vx, -peri_state.vy, -peri_state.vz,
        ];

        // Propagate forward to find apogee and compute α
        let theta_sun = sol.theta_sun_deg.to_radians();
        let bcr       = Bcr4bpParams::earth_moon_sun(theta_sun);
        let fwd       = propagate_bcr4bp(mu, bcr, ic_earth, T_SCREEN, LOG_DT, RTOL, ATOL);

        let alpha_deg = match compute_alpha_at_apogee(&fwd, x_earth, theta_sun, bcr.omega_s) {
            Some(a) => a,
            None    => continue,
        };

        // Circi §3.1: α must be in Q2 (70°–180°) or Q4 (250°–360°).
        if !alpha_in_valid_quadrant(alpha_deg) { continue; }

        eprintln!("  α-pass: peri={:.0}km  e={:.3}  β={:.1}°  α={:.1}°  θ_sun={:.1}°",
            sol.peri_km, sol.ecc, sol.beta_deg, alpha_deg, sol.theta_sun_deg);

        screened.push(ScreenedSolution {
            peri_km:       sol.peri_km,
            ecc:           sol.ecc,
            incl_deg:      sol.incl_deg,
            raan_deg:      sol.raan_deg,
            theta_sun_deg: sol.theta_sun_deg,
            beta_deg:      sol.beta_deg,
            alpha_deg,
            ic_earth,
        });
    }

    screened
}

// ════════════════════════════════════════════════════════════════════════════════
// Phase 3 — Belbruno forward sweep from Earth parking orbit
// ════════════════════════════════════════════════════════════════════════════════

fn phase3_forward_sweep(
    mu:             f64,
    params:         &CrtbpParams,
    x_earth:        f64,
    moon_x:         f64,
    r_hill:         f64,
    l_km:           f64,
    v_km_s:         f64,
    r_moon_km:      f64,
    r_park:         f64,
    r_apogee_target: f64,
    screened:       &[ScreenedSolution],
) -> Vec<Candidate> {

    // Build the θ_sun grid for Phase 3: coarse global + fine in Q2/Q4 windows
    let sun_phases = build_sun_phase_grid_p3();

    let total = N_THETA * sun_phases.len();
    eprintln!("  grid: {N_THETA}θ × {} θ_sun = {total} propagations  \
               (apogee target={r_apogee_target:.1} nd = {:.0} km)",
        sun_phases.len(), r_apogee_target * 384_400.0);

    // Collect θ_sun hints from Phase 2 survivors to bias the sweep
    let hint_suns: Vec<f64> = screened.iter()
        .map(|s| s.theta_sun_deg.to_radians())
        .collect();

    let n_propagated = AtomicUsize::new(0);
    let n_excursion  = AtomicUsize::new(0);

    let mu_earth = 1.0 - mu;
    let v_circ   = (mu_earth / r_park).sqrt();
    let a_base   = (r_park + r_apogee_target) / 2.0;
    let v_inj    = (mu_earth * (2.0 / r_park - 1.0 / a_base)).sqrt();
    let dv_nd    = v_inj - v_circ;
    let dv_kms   = dv_nd * v_km_s;

    let candidates: Vec<Candidate> = (0..N_THETA)
        .into_par_iter()
        .flat_map(|ti| {
            let theta = 2.0 * PI * ti as f64 / N_THETA as f64;
            let Some(ic_base) = tli_injection_ic(mu, r_park, r_apogee_target, theta)
                else { return vec![]; };

            // Merge Phase-2-hinted sun phases into this θ's sweep
            let mut this_suns = sun_phases.clone();
            this_suns.extend_from_slice(&hint_suns);
            this_suns.sort_by(|a, b| a.partial_cmp(b).unwrap());
            this_suns.dedup_by(|a, b| (*a - *b).abs() < 1e-4);

            let mut local: Vec<Candidate> = Vec::new();
            for theta_sun in &this_suns {
                let prev = n_propagated.fetch_add(1, Ordering::Relaxed);
                if prev % 5_000 == 0 {
                    eprintln!("  [P3 {prev}/{total}]  θ={:.0}°  excursions={}  captures={}",
                        theta.to_degrees(), n_excursion.load(Ordering::Relaxed), local.len());
                }

                let bcr  = Bcr4bpParams::earth_moon_sun(*theta_sun);
                let traj = propagate_bcr4bp(mu, bcr, ic_base, T_PROP, LOG_DT, RTOL, ATOL);

                let reaches = traj.iter().any(|s| {
                    let dx = s.x - x_earth;
                    (dx*dx + s.y*s.y + s.z*s.z).sqrt() > R_MIN_EXCURSION
                });
                if !reaches { continue; }
                n_excursion.fetch_add(1, Ordering::Relaxed);

                let alpha_deg = match compute_alpha_at_apogee(&traj, x_earth, *theta_sun, bcr.omega_s) {
                    Some(a) => a,
                    None    => continue,
                };
                if !alpha_in_valid_quadrant(alpha_deg) { continue; }

                let hill_entry = traj.iter().find(|s| {
                    let dx = s.x - moon_x;
                    (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
                });
                let Some(hill_state) = hill_entry else { continue };
                if hill_state.time < T_MIN_HILL { continue; }

                let capture = detect_capture(&traj, mu, MIN_CAPTURE_TIME);
                if capture.n_entries == 0 { continue; }

                let t_hill_nd    = hill_state.time;
                let alt_km       = capture.min_lunar_dist * l_km - r_moon_km;
                let est_orbits   = capture.max_capture_interval / LUNAR_PERIOD_ND;
                let crashed_moon = alt_km < 0.0;
                let speed_bonus  = if t_hill_nd < T_FAST_TARGET {
                    SPEED_BONUS_WEIGHT * (1.0 - t_hill_nd / T_FAST_TARGET)
                } else { 0.0 };
                let score = (est_orbits - LAMBDA * capture.min_lunar_dist + speed_bonus)
                    * if crashed_moon { 0.3 } else { 1.0 };
                let apogee_nd = traj.iter().map(|s| {
                    let dx = s.x - x_earth;
                    (dx*dx + s.y*s.y + s.z*s.z).sqrt()
                }).fold(0.0_f64, f64::max);

                eprintln!("  -> CAPTURE  θ={:.1}°  θ_sun={:.1}°  α={:.1}°  \
                           t={:.0}d  orbits≈{:.1}  alt={:.0}km  ΔV={dv_kms:.3}km/s{}{}",
                    theta.to_degrees(), theta_sun.to_degrees(), alpha_deg,
                    t_hill_nd * params.t_star / 86_400.0, est_orbits, alt_km,
                    if est_orbits >= MIN_CAPTURE_ORBITS { "  ★" } else { "" },
                    if crashed_moon { "  ☠CRASH" } else { "" });

                local.push(Candidate {
                    theta_deg:             theta.to_degrees(),
                    theta_sun_deg:         theta_sun.to_degrees(),
                    dv_nd, dv_km_s: dv_kms,
                    t_transfer_nd:         t_hill_nd,
                    t_transfer_days:       t_hill_nd * params.t_star / 86_400.0,
                    capture_duration_nd:   capture.capture_duration,
                    capture_duration_days: params.dim_time_days(capture.capture_duration),
                    n_entries:             capture.n_entries,
                    min_lunar_dist_nd:     capture.min_lunar_dist,
                    min_lunar_alt_km:      alt_km,
                    max_interval_nd:       capture.max_capture_interval,
                    max_interval_days:     params.dim_time_days(capture.max_capture_interval),
                    est_capture_orbits:    est_orbits,
                    crashed_moon, alpha_deg, apogee_nd,
                    r_apogee_target:       r_apogee_target,
                    family:                String::new(),
                    pareto_rank:           0,
                    score,
                    ic:                    ic_base,
                    traj:                  Vec::new(),
                });
            }
            local
        })
        .collect();

    eprintln!("  propagated: {}   SE-L1 excursions: {}   raw captures: {}",
        n_propagated.load(Ordering::Relaxed),
        n_excursion.load(Ordering::Relaxed),
        candidates.len());

    candidates
}

// ════════════════════════════════════════════════════════════════════════════════
// Phase 4 — Envelope sweep + family analysis
// ════════════════════════════════════════════════════════════════════════════════

/// Seed a dense local sweep around every Phase 3 capture, and also vary the
/// apogee target to map the ΔV–time trade-off.  Returns ALL survivors merged
/// with the input Phase 3 candidates (duplicates kept — Pareto ranking will
/// surface the best ones).
fn phase4_envelope(
    mu:          f64,
    params:      &CrtbpParams,
    x_earth:     f64,
    moon_x:      f64,
    r_hill:      f64,
    l_km:        f64,
    v_km_s:      f64,
    r_moon_km:   f64,
    r_park:      f64,
    mut seeds:   Vec<Candidate>,
) -> Vec<Candidate> {

    let mu_earth    = 1.0 - mu;
    let v_circ      = (mu_earth / r_park).sqrt();

    // Collect seed (θ, θ_sun) pairs — one envelope per seed
    let seed_params: Vec<(f64, f64)> = seeds.iter()
        .map(|c| (c.theta_deg.to_radians(), c.theta_sun_deg.to_radians()))
        .collect();

    // Drop any stale trajectories on the incoming seeds — not needed beyond metrics
    for s in seeds.iter_mut() { s.traj = Vec::new(); }

    let n_envelope = seed_params.len()
        * ENVELOPE_N_THETA * ENVELOPE_N_SUN
        * ENVELOPE_APOGEES.len();
    eprintln!("  {} seeds × {}θ × {}θ_sun × {} apogees = {} envelope propagations",
        seed_params.len(), ENVELOPE_N_THETA, ENVELOPE_N_SUN,
        ENVELOPE_APOGEES.len(), n_envelope);

    let dtheta  = ENVELOPE_DTHETA.to_radians();
    let dsun    = ENVELOPE_DSUN.to_radians();

    let n_prop = AtomicUsize::new(0);

    let new_candidates: Vec<Candidate> = seed_params
        .into_par_iter()
        .flat_map(|(seed_theta, seed_sun)| {
            let mut local: Vec<Candidate> = Vec::new();

            for &r_apogee in ENVELOPE_APOGEES {
                let a      = (r_park + r_apogee) / 2.0;
                let v_inj  = (mu_earth * (2.0 / r_park - 1.0 / a)).sqrt();
                let dv_nd  = v_inj - v_circ;
                let dv_kms = dv_nd * v_km_s;

                for it in 0..ENVELOPE_N_THETA {
                    let theta = seed_theta - dtheta
                        + 2.0 * dtheta * it as f64 / (ENVELOPE_N_THETA - 1) as f64;

                    let Some(ic) = tli_injection_ic(mu, r_park, r_apogee, theta)
                        else { continue };

                    for is in 0..ENVELOPE_N_SUN {
                        let theta_sun = seed_sun - dsun
                            + 2.0 * dsun * is as f64 / (ENVELOPE_N_SUN - 1) as f64;

                        let prev = n_prop.fetch_add(1, Ordering::Relaxed);
                        if prev % 10_000 == 0 {
                            eprintln!("  [P4 {}/{n_envelope}]  new_captures={}",
                                prev, local.len());
                        }

                        let bcr  = Bcr4bpParams::earth_moon_sun(theta_sun);
                        let traj = propagate_bcr4bp(mu, bcr, ic, T_PROP, LOG_DT, RTOL, ATOL);

                        let reaches = traj.iter().any(|s| {
                            let dx = s.x - x_earth;
                            (dx*dx + s.y*s.y + s.z*s.z).sqrt() > R_MIN_EXCURSION
                        });
                        if !reaches { continue; }

                        let alpha_deg = match compute_alpha_at_apogee(&traj, x_earth, theta_sun, bcr.omega_s) {
                            Some(a) => a,
                            None    => continue,
                        };
                        if !alpha_in_valid_quadrant(alpha_deg) { continue; }

                        let hill = traj.iter().find(|s| {
                            let dx = s.x - moon_x;
                            (dx*dx + s.y*s.y + s.z*s.z).sqrt() < r_hill
                        });
                        let Some(hill_s) = hill else { continue };
                        if hill_s.time < T_MIN_HILL { continue; }

                        let capture = detect_capture(&traj, mu, MIN_CAPTURE_TIME);
                        if capture.n_entries == 0 { continue; }

                        let t_hill_nd    = hill_s.time;
                        let alt_km       = capture.min_lunar_dist * l_km - r_moon_km;
                        let est_orbits   = capture.max_capture_interval / LUNAR_PERIOD_ND;
                        let crashed_moon = alt_km < 0.0;
                        let speed_bonus  = if t_hill_nd < T_FAST_TARGET {
                            SPEED_BONUS_WEIGHT * (1.0 - t_hill_nd / T_FAST_TARGET)
                        } else { 0.0 };
                        let score = (est_orbits - LAMBDA * capture.min_lunar_dist + speed_bonus)
                            * if crashed_moon { 0.3 } else { 1.0 };
                        let apogee_nd = traj.iter().map(|s| {
                            let dx = s.x - x_earth;
                            (dx*dx + s.y*s.y + s.z*s.z).sqrt()
                        }).fold(0.0_f64, f64::max);

                        local.push(Candidate {
                            theta_deg:             theta.to_degrees(),
                            theta_sun_deg:         theta_sun.to_degrees(),
                            dv_nd, dv_km_s: dv_kms,
                            t_transfer_nd:         t_hill_nd,
                            t_transfer_days:       t_hill_nd * params.t_star / 86_400.0,
                            capture_duration_nd:   capture.capture_duration,
                            capture_duration_days: params.dim_time_days(capture.capture_duration),
                            n_entries:             capture.n_entries,
                            min_lunar_dist_nd:     capture.min_lunar_dist,
                            min_lunar_alt_km:      alt_km,
                            max_interval_nd:       capture.max_capture_interval,
                            max_interval_days:     params.dim_time_days(capture.max_capture_interval),
                            est_capture_orbits:    est_orbits,
                            crashed_moon, alpha_deg, apogee_nd,
                            r_apogee_target:       r_apogee,
                            family:                String::new(),
                            pareto_rank:           0,
                            score, ic,
                            traj:                  Vec::new(),
                        });
                    }
                }
            }
            local
        })
        .collect();

    let n_new = new_candidates.len();
    seeds.extend(new_candidates);
    eprintln!("  envelope: {} propagations → {n_new} new captures",
        n_prop.load(Ordering::Relaxed));
    seeds
}

/// Assign a family label to every candidate based on α quadrant and transfer time.
///
/// Family naming:  "{quadrant}-{speed}"
///   quadrant = "Q2" if α ∈ [70°,180°]   (apogee toward Sun side)
///              "Q4" if α ∈ [250°,360°]  (apogee away from Sun — Q3 rejected)
///   speed    = "fast"   if t_transfer < T_FAMILY_FAST   (~84 days)
///              "medium" if t_transfer < T_FAMILY_MEDIUM (~141 days)
///              "slow"   otherwise
fn classify_families(candidates: &mut [Candidate]) {
    for c in candidates.iter_mut() {
        let quadrant = if c.alpha_deg <= 180.0 { "Q2" } else { "Q4" };
        let speed = if c.t_transfer_nd < T_FAMILY_FAST {
            "fast"
        } else if c.t_transfer_nd < T_FAMILY_MEDIUM {
            "medium"
        } else {
            "slow"
        };
        c.family = format!("{quadrant}-{speed}");
    }
}

/// Assign Pareto ranks in the (ΔV, transfer_time) objective space.  O(n log n).
///
/// Rank 1 = Pareto-optimal (no other solution is both cheaper AND faster).
/// Rank 2 = Pareto-optimal after removing rank-1 solutions.  Etc.
///
/// Algorithm for 2 objectives:
///   1. Sort by (dv asc, t asc).
///   2. Process candidates in equal-dv groups.  Within each group, intra-group
///      dominance is determined by t alone (r_group = count of group members
///      with strictly smaller t).  Cross-group dominance is tracked via
///      `rank_fronts[r]` = minimum t on rank-(r+1) seen so far from lower-dv
///      groups (a non-decreasing sequence, valid for binary search).
///   3. rank(i) = max(r_prev, r_group) + 1, then rank_fronts is updated.
fn assign_pareto_ranks(candidates: &mut Vec<Candidate>) {
    let n = candidates.len();
    if n == 0 { return; }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        candidates[a].dv_nd.partial_cmp(&candidates[b].dv_nd).unwrap()
            .then(candidates[a].t_transfer_nd.partial_cmp(&candidates[b].t_transfer_nd).unwrap())
    });

    // rank_fronts[r] = minimum t_transfer on rank-(r+1) front from all groups
    // with strictly smaller dv than the current group.  Non-decreasing by the
    // Pareto structure, so partition_point is valid.
    let mut rank_fronts: Vec<f64> = Vec::new();

    let mut i = 0;
    while i < n {
        // Collect all indices with the same dv value
        let dv_val = candidates[order[i]].dv_nd;
        let group_len = order[i..].partition_point(|&k| candidates[k].dv_nd == dv_val);
        let group = &order[i .. i + group_len];

        // Snapshot rank_fronts so intra-group updates don't affect each other
        let snapshot = rank_fronts.clone();
        let mut group_min_t: Vec<f64> = Vec::new();

        for &idx in group {
            let t = candidates[idx].t_transfer_nd;
            // Dominators from lower-dv groups: count fronts with min_t <= t
            let r_prev  = snapshot.partition_point(|&ft| ft <= t);
            // Dominators within this group: count members with strictly smaller t
            let r_group = group.partition_point(|&k| candidates[k].t_transfer_nd < t);
            let rank = r_prev.max(r_group) + 1;
            candidates[idx].pareto_rank = rank;

            let ri = rank - 1;
            if ri >= group_min_t.len() { group_min_t.resize(ri + 1, f64::MAX); }
            group_min_t[ri] = group_min_t[ri].min(t);
        }

        // Merge this group's per-rank minimums into rank_fronts
        for (r, &min_t) in group_min_t.iter().enumerate() {
            if min_t < f64::MAX {
                if r >= rank_fronts.len() { rank_fronts.resize(r + 1, f64::MAX); }
                rank_fronts[r] = rank_fronts[r].min(min_t);
            }
        }

        i += group_len;
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Helper functions
// ════════════════════════════════════════════════════════════════════════════════

/// Build the θ_sun grid for Phase 1.
/// Coarse global sweep + fine samples inside the two β windows.
fn build_sun_phase_grid_p1() -> Vec<f64> {
    let mut phases: Vec<f64> = Vec::new();

    // Coarse global pass
    for i in 0..N_SUN_COARSE {
        phases.push(2.0 * PI * i as f64 / N_SUN_COARSE as f64);
    }

    // Fine pass inside β window 1: Sun at ~[30°,120°] → β ∈ [60°,150°]
    let w1_center = 75.0_f64.to_radians();
    let w1_half   = 45.0_f64.to_radians();
    for i in 0..N_SUN_FINE {
        let t = w1_center - w1_half + 2.0 * w1_half * i as f64 / (N_SUN_FINE - 1) as f64;
        phases.push(t);
    }

    // Fine pass inside β window 2: Sun at ~[210°,300°] → β ∈ [240°,330°]
    let w2_center = 255.0_f64.to_radians();
    let w2_half   = 45.0_f64.to_radians();
    for i in 0..N_SUN_FINE {
        let t = w2_center - w2_half + 2.0 * w2_half * i as f64 / (N_SUN_FINE - 1) as f64;
        phases.push(t);
    }

    // Deduplicate (within 0.5°)
    phases.sort_by(|a, b| a.partial_cmp(b).unwrap());
    phases.dedup_by(|a, b| (*a - *b).abs() < 0.5_f64.to_radians());
    phases
}

/// Build the θ_sun grid for Phase 3.
/// Coarse global sweep + fine samples in the Q2/Q4 α windows.
fn build_sun_phase_grid_p3() -> Vec<f64> {
    let mut phases: Vec<f64> = Vec::new();

    // Coarse global
    for i in 0..N_SUN_P3_COARSE {
        phases.push(2.0 * PI * i as f64 / N_SUN_P3_COARSE as f64);
    }

    // Fine in α Q2 window: θ_sun ≈ [0°,90°] maps α into Q2 for typical θ
    let q2_center = 45.0_f64.to_radians();
    let q2_half   = 50.0_f64.to_radians();
    for i in 0..N_SUN_P3_FINE {
        let t = q2_center - q2_half + 2.0 * q2_half * i as f64 / (N_SUN_P3_FINE - 1) as f64;
        phases.push(t.rem_euclid(2.0 * PI));
    }

    // Fine sweep near the θ_sun window that tends to produce Q4 α (apogee anti-Sun).
    let q4_center = 225.0_f64.to_radians();
    let q4_half   = 50.0_f64.to_radians();
    for i in 0..N_SUN_P3_FINE {
        let t = q4_center - q4_half + 2.0 * q4_half * i as f64 / (N_SUN_P3_FINE - 1) as f64;
        phases.push(t.rem_euclid(2.0 * PI));
    }

    phases.sort_by(|a, b| a.partial_cmp(b).unwrap());
    phases.dedup_by(|a, b| (*a - *b).abs() < 0.5_f64.to_radians());
    phases
}

/// Returns `true` if β (Earth–Moon vs Earth–Sun at periselene) falls inside
/// one of the two valid transfer windows identified in Circi Fig. 6:
///   window 1: β ∈ [60°, 150°]
///   window 2: β ∈ [240°, 330°]
fn beta_in_valid_window(beta_deg: f64) -> bool {
    let b = beta_deg.rem_euclid(360.0);
    (b >= 60.0 && b <= 150.0) || (b >= 240.0 && b <= 330.0)
}

/// Compute the α angle at apogee: angle of (Earth→apogee) relative to the
/// Sun direction at apogee time, in degrees [0°, 360°).
///
/// Per Circi §3.1 and Fig. 8, energy is pumped into the orbit when α is in
/// Q2 (90°–180°) or Q4 (270°–360°).
///
/// The Sun's actual angle at apogee is `theta_sun_0 + omega_s * t_apogee`.
/// Using the initial theta_sun_0 is wrong: with omega_s ≈ -0.9252 nd/nd the
/// Sun moves ~190° during a typical 43-day transfer, completely scrambling α.
fn compute_alpha_at_apogee(
    traj:      &[Step3d],
    x_earth:   f64,
    theta_sun: f64,   // initial Sun angle at TLI [rad]
    omega_s:   f64,   // Sun angular velocity in synodic frame [nd/nd]
) -> Option<f64> {
    // Find apogee = maximum distance from Earth
    let apogee = traj.iter().max_by(|a, b| {
        let ra = ((a.x - x_earth).powi(2) + a.y.powi(2) + a.z.powi(2)).sqrt();
        let rb = ((b.x - x_earth).powi(2) + b.y.powi(2) + b.z.powi(2)).sqrt();
        ra.partial_cmp(&rb).unwrap()
    })?;

    // Sun's actual direction at the moment the spacecraft reaches apogee
    let theta_sun_at_apo = theta_sun + omega_s * apogee.time;

    // α = angle of (Earth→apogee) relative to Sun direction at apogee time
    let apogee_angle = apogee.y.atan2(apogee.x - x_earth);
    let alpha = (apogee_angle - theta_sun_at_apo).rem_euclid(2.0 * PI);
    Some(alpha.to_degrees())
}

/// Returns `true` if α falls in Q2 (70°–180°) or Q4 (250°–360°).
///
/// Per Circi Fig. 8, the Sun pumps energy into the orbit only in Q2 and Q4.
/// Q3 (180°–270°) is NOT valid — it is symmetric to Q1, not to Q2.
/// ±20° tolerance on each quadrant boundary (Sun-assist doesn't cut off sharply).
fn alpha_in_valid_quadrant(alpha_deg: f64) -> bool {
    let a = alpha_deg.rem_euclid(360.0);
    // Q2: 70°–180°  (apogee toward Sun side)
    // Q4: 250°–360° (apogee away from Sun — 270° ± 20°)
    (a >= 70.0 && a <= 180.0) || (a >= 250.0 && a <= 360.0)
}


// ════════════════════════════════════════════════════════════════════════════════
// Output helpers
// ════════════════════════════════════════════════════════════════════════════════

fn save_backward_csv(solutions: &[BackwardSolution]) {
    let path = format!("{OUT_DIR}/backward_solutions.csv");
    let mut w = BufWriter::new(fs::File::create(&path).expect("backward csv create failed"));
    writeln!(w, "time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,\
                 sol_id,peri_km,ecc,incl_deg,raan_deg,theta_sun_deg,beta_deg").unwrap();
    for (id, sol) in solutions.iter().enumerate() {
        let sid = id + 1;
        for s in &sol.traj {
            writeln!(w,
                "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},\
                 {sid},{:.0},{:.4},{:.1},{:.1},{:.2},{:.2}",
                s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz,
                sol.peri_km, sol.ecc, sol.incl_deg, sol.raan_deg,
                sol.theta_sun_deg, sol.beta_deg,
            ).unwrap();
        }
        writeln!(w,
            "NaN,NaN,NaN,NaN,NaN,NaN,NaN,\
             {sid},{:.0},{:.4},{:.1},{:.1},{:.2},{:.2}",
            sol.peri_km, sol.ecc, sol.incl_deg, sol.raan_deg,
            sol.theta_sun_deg, sol.beta_deg,
        ).unwrap();
    }
    eprintln!("  Saved {path}  ({} arcs)", solutions.len());
}

fn save_screen_csv(screened: &[ScreenedSolution]) {
    let path = format!("{OUT_DIR}/forward_screen.csv");
    let mut w = BufWriter::new(fs::File::create(&path).expect("screen csv create failed"));
    writeln!(w, "time_nd,x_nd,y_nd,z_nd,sol_id,alpha_deg,beta_deg,theta_sun_deg").unwrap();
    // We don't re-propagate here — just record the Earth-perigee IC as a point
    for (id, sol) in screened.iter().enumerate() {
        let sid = id + 1;
        let ic  = &sol.ic_earth;
        writeln!(w,
            "{:.6},{:.8},{:.8},{:.8},{sid},{:.2},{:.2},{:.2}",
            0.0_f64, ic[0], ic[1], ic[2],
            sol.alpha_deg, sol.beta_deg, sol.theta_sun_deg,
        ).unwrap();
    }
    eprintln!("  Saved {path}  ({} screened ICs)", screened.len());
}

fn save_candidates(
    mu:         f64,
    params:     &CrtbpParams,
    l_km:       f64,
    r_hill:     f64,
    mut candidates: Vec<Candidate>,
) {
    let c_l1 = c_lagrange(mu, 1);
    let c_l2 = c_lagrange(mu, 2);

    // Sort by score for the trajectory CSV; Pareto rank is already assigned
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score)
        .unwrap_or(std::cmp::Ordering::Equal));

    // ── trajectory CSV (top TOP_N by score, deduplicated) ────────────────────
    // Collect top indices now (score-sorted order), then write both CSVs
    // before re-using `top_indices` for the summary — avoids a borrow conflict
    // with the second sort below.
    let top_indices: Vec<usize> = {
        let deduped = deduplicate_ref(&candidates, MIN_THETA_SEP);
        deduped.into_iter().take(TOP_N).map(|c| {
            candidates.iter().position(|x| std::ptr::eq(x, c)).unwrap()
        }).collect()
    };

    let traj_path = format!("{OUT_DIR}/blt_candidates.csv");
    {
        let mut w = BufWriter::new(fs::File::create(&traj_path).expect("candidates csv create failed"));
        writeln!(w, "time_nd,x_nd,y_nd,z_nd,vx_nd,vy_nd,vz_nd,cand_id,family,pareto_rank").unwrap();
        for (k, &idx) in top_indices.iter().enumerate() {
            let cand = &candidates[idx];
            let cid  = k + 1;
            let theta_sun = cand.theta_sun_deg.to_radians();
            let bcr       = Bcr4bpParams::earth_moon_sun(theta_sun);
            eprintln!("  Re-propagating candidate {cid} for {:.0} days …",
                T_PROP_FINAL * params.t_star / 86_400.0);
            let long_traj = propagate_bcr4bp(mu, bcr, cand.ic, T_PROP_FINAL, LOG_DT, RTOL, ATOL);
            for s in &long_traj {
                writeln!(w,
                    "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{cid},{},{}",
                    s.time, s.x, s.y, s.z, s.vx, s.vy, s.vz,
                    cand.family, cand.pareto_rank,
                ).unwrap();
            }
            writeln!(w, "NaN,NaN,NaN,NaN,NaN,NaN,NaN,{cid},{},{}",
                cand.family, cand.pareto_rank).unwrap();
        }
    }
    eprintln!("  Saved {traj_path}  ({} arcs)", top_indices.len());

    // ── family analysis CSV (all solutions) ───────────────────────────────────
    // Sort by Pareto rank then score for the analysis CSV
    candidates.sort_by(|a, b| {
        a.pareto_rank.cmp(&b.pareto_rank)
            .then(b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal))
    });

    let fam_path = format!("{OUT_DIR}/family_analysis.csv");
    {
        let mut w = BufWriter::new(fs::File::create(&fam_path).expect("family csv create failed"));
        writeln!(w, "theta_deg,theta_sun_deg,dv_kms,t_transfer_days,capture_dur_days,\
                     min_alt_km,alpha_deg,apogee_nd,r_apogee_target_nd,\
                     n_entries,est_capture_orbits,family,pareto_rank,score,crashed_moon").unwrap();
        for c in &candidates {
            writeln!(w,
                "{:.3},{:.3},{:.5},{:.2},{:.2},{:.1},{:.2},{:.4},{:.2},{},{:.2},{},{},{:.4},{}",
                c.theta_deg, c.theta_sun_deg,
                c.dv_km_s, c.t_transfer_days, c.capture_duration_days,
                c.min_lunar_alt_km, c.alpha_deg, c.apogee_nd, c.r_apogee_target,
                c.n_entries, c.est_capture_orbits, c.family, c.pareto_rank,
                c.score, if c.crashed_moon { 1 } else { 0 },
            ).unwrap();
        }
    }
    let n_quality = candidates.iter().filter(|c| c.est_capture_orbits >= MIN_CAPTURE_ORBITS).count();
    let n_crashed = candidates.iter().filter(|c| c.crashed_moon).count();
    eprintln!("  Saved {fam_path}  ({} solutions, {} with ≥{:.0} orbits, {} moon-crash flagged)",
        candidates.len(), n_quality, MIN_CAPTURE_ORBITS, n_crashed);

    // ── summary text ─────────────────────────────────────────────────────────
    let info_path = format!("{OUT_DIR}/blt_info.txt");
    // Tee output: write to file and stdout simultaneously
    let mut info_file = BufWriter::new(fs::File::create(&info_path).expect("info write failed"));
    macro_rules! tee {
        ($($arg:tt)*) => {{
            let line = format!($($arg)*);
            writeln!(info_file, "{line}").unwrap();
            println!("{line}");
        }};
    }

    tee!("=== WSB Ballistic Lunar Transfer — Results ===\n");
    tee!("Method   : Phase1 backward (Circi §2) + α-screen (§3.1) \
                    + Belbruno forward + envelope sweep");
    let r_park_out = cfg_r_park();
    let r_apogee_out = cfg_r_apogee();
    tee!("r_park   = {r_park_out:.5} nd  ({:.0} km alt)", r_park_out * l_km - 6_371.0);
    tee!("Apogee   = {r_apogee_out:.1} nd  ({:.0} km, SE-WSB target)",
        r_apogee_out * 384_400.0);
    tee!("C_L1     = {c_l1:.6}   C_L2 = {c_l2:.6}");
    tee!("r_Hill   = {r_hill:.5} nd  = {:.0} km\n", r_hill * l_km);
    tee!("Total solutions : {}", candidates.len());

    // Family breakdown
    let families = ["Q2-fast", "Q2-medium", "Q2-slow", "Q4-fast", "Q4-medium", "Q4-slow"];
    tee!("\n── Family breakdown ─────────────────────────────────");
    for fam in &families {
        let members: Vec<&Candidate> = candidates.iter()
            .filter(|c| c.family == *fam).collect();
        if members.is_empty() { continue; }
        let best_dv  = members.iter().map(|c| c.dv_km_s).fold(f64::MAX, f64::min);
        let best_t   = members.iter().map(|c| c.t_transfer_days).fold(f64::MAX, f64::min);
        let best_cap = members.iter().map(|c| c.capture_duration_days).fold(0.0_f64, f64::max);
        tee!("  {fam:<12}  n={:3}  best ΔV={best_dv:.4} km/s  \
                        fastest={best_t:.0} days  longest_cap={best_cap:.1} days",
            members.len());
    }

    // Pareto front (rank 1)
    let pareto1: Vec<&Candidate> = candidates.iter()
        .filter(|c| c.pareto_rank == 1).collect();
    tee!("\n── Pareto front  (rank 1, {} solutions) ─────────────", pareto1.len());
    tee!("  {:>8}  {:>10}  {:>12}  {:>10}  {:>10}  {}",
        "θ (°)", "θ_sun (°)", "ΔV (km/s)", "t (days)", "cap (days)", "family");
    let mut pf = pareto1;
    pf.sort_by(|a, b| a.t_transfer_days.partial_cmp(&b.t_transfer_days).unwrap());
    for c in &pf {
        tee!("  {:>8.2}  {:>10.2}  {:>12.5}  {:>10.1}  {:>10.2}  {}",
            c.theta_deg, c.theta_sun_deg, c.dv_km_s,
            c.t_transfer_days, c.capture_duration_days, c.family);
    }

    // Top TOP_N candidates by score
    tee!("\n── Top {TOP_N} by score ──────────────────────────────────");
    for (k, &idx) in top_indices.iter().enumerate() {
        let cand = &candidates[idx];
        tee!("\n--- Candidate {} --- [{}]  Pareto rank {}", k + 1, cand.family, cand.pareto_rank);
        tee!("  θ_inject       : {:.2}°",  cand.theta_deg);
        tee!("  θ_sun (at TLI) : {:.2}°",  cand.theta_sun_deg);
        tee!("  α at apogee    : {:.1}°  ({})", cand.alpha_deg,
            if cand.alpha_deg <= 180.0 { "Q2 — apogee toward Sun" }
            else                        { "Q4 — apogee away from Sun" });
        tee!("  apogee reached : {:.3} nd  = {:.0} km",
            cand.apogee_nd, cand.apogee_nd * 384_400.0);
        tee!("  TLI ΔV         : {:.5} nd  =  {:.5} km/s", cand.dv_nd, cand.dv_km_s);
        tee!("  transfer time  : {:.3} nd  =  {:.1} days",
            cand.t_transfer_nd, cand.t_transfer_days);
        tee!("  capture time   : {:.3} nd  =  {:.2} days",
            cand.capture_duration_nd, cand.capture_duration_days);
        tee!("  max interval   : {:.3} nd  =  {:.2} days",
            cand.max_interval_nd, cand.max_interval_days);
        tee!("  est. orbits    : {:.2}  {}", cand.est_capture_orbits,
            if cand.est_capture_orbits >= MIN_CAPTURE_ORBITS { "★ meets 3-orbit target" }
            else { "(below 3-orbit target)" });
        tee!("  Hill entries   : {}", cand.n_entries);
        tee!("  min r_Moon     : {:.5} nd  =  {:.1} km alt{}",
            cand.min_lunar_dist_nd, cand.min_lunar_alt_km,
            if cand.crashed_moon { "  ☠ MOON CRASH" } else { "" });
        tee!("  score          : {:.4}", cand.score);
    }

    eprintln!("  Saved {info_path}");
}

/// Non-consuming version of deduplicate — returns references into `v`.
fn deduplicate_ref<'a>(v: &'a [Candidate], min_sep_deg: f64) -> Vec<&'a Candidate> {
    let mut out: Vec<&Candidate> = Vec::new();
    'outer: for c in v {
        for o in &out {
            let diff = (c.theta_deg - o.theta_deg).abs().rem_euclid(360.0);
            let sep  = diff.min(360.0 - diff);
            if sep < min_sep_deg { continue 'outer; }
        }
        out.push(c);
    }
    out
}