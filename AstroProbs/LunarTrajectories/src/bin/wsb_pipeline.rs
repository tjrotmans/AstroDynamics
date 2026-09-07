//! wsb_pipeline — unified WSB transfer search pipeline.
//!
//! Single entry point for the complete weak-stability-boundary workflow.
//! Edit the `CONFIG` block below, then run:
//!
//!   cargo run -p lunar_trajectories --bin wsb_pipeline --release
//!
//! All step binaries must be built first (the pipeline does this automatically):
//!
//!   cargo build -p lunar_trajectories --release
//!
//! ═══════════════════════════════════════════════════════════════════════════════
//! PIPELINE OVERVIEW
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//!  wsb_search
//!    → wsb_optimize (island GA)
//!        → wsb_refine (Monte Carlo polish)
//!            → wsb_maxhifi (high-fidelity repropagation)
//!                → wsb_circularize (LOI burn + ECI frame)
//!
//!  Optional diagnostic branches (run independently after wsb_refine):
//!    wsb_sensitivity          — 200-sample Gaussian perturbation ensemble
//!    wsb_sensitivity_individual — one-at-a-time (OAT) 4×50 parameter sweep
//!    wsb_stats                — 20 000-sample sigma-sweep Monte Carlo statistics
//!    wsb_dense_traj           — 10 diverse solutions, maximum time resolution
//!    wsb_basin                — 2-D capture-basin grid (θ × θ_sun)
//!
//!  All outputs land in out/wsb/.
//!  Python plots are orchestrated by plot/wsb_plots.py.
//!
//! ═══════════════════════════════════════════════════════════════════════════════
//! RUST BINARIES — INPUTS, OUTPUTS, PURPOSE
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! ── wsb_search ──────────────────────────────────────────────────────────────
//! Purpose : Four-phase global search for ballistic lunar transfer (BLT) seeds.
//!           Implements the Circi & Teofilatto backward/forward screening method.
//! Inputs  : R_park_nd and r_apogee_nd via env vars WSB_R_PARK_ND / WSB_R_APOGEE_ND
//!           (injected by wsb_pipeline; set as constants in the binary otherwise).
//! Phases  :
//!   1. Backward from Moon: β-angle filter (60–150° ∪ 240–330°), Earth perigee
//!      altitude filter (200–1000 km).
//!   2. Forward α-angle screen: keeps quadrants II and IV (direct-sun / anti-sun
//!      families).
//!   3. Forward sweep from Earth parking orbit: (θ, θ_sun) injection grid.
//!   4. Dense 8×8 envelope scan around seeds; family and Pareto analysis.
//! Outputs : out/wsb/backward_solutions.csv
//!           out/wsb/forward_screen.csv
//!           out/wsb/blt_candidates.csv      — Phase 3+4 full transfer arcs
//!           out/wsb/family_analysis.csv     — Pareto-ranked candidates with
//!                                             family tag (Q2/Q4 × speed class)
//!           out/wsb/blt_info.txt            — ranked summary
//! Pre-run : none
//!
//! ── wsb_optimize ────────────────────────────────────────────────────────────
//! Purpose : Island genetic algorithm (GA) over (θ, θ_sun, r_apogee).
//!           One independent GA island per trajectory family; runs in parallel.
//!           Minimises ΔV_total = ΔV_TLI + ΔV_LOI_min; only Hill-entering
//!           solutions (ΔV_LOI < 1.0 km/s) are kept.
//! Inputs  : out/wsb/family_analysis.csv  (per-family θ bounds, warm-start seeds)
//! Config  : ISLAND_POP=150, ISLAND_GENERATIONS=150, GA_TOP_N=30
//! Outputs : out/wsb/ga_solutions.csv  — all valid Hill-entering GA solutions
//!           out/wsb/ga_best.csv       — top GA_TOP_N=30 ranked by ΔV_total
//! Pre-run : wsb_search
//!
//! ── wsb_refine ──────────────────────────────────────────────────────────────
//! Purpose : Tight 3-D Monte Carlo (MC) local polish around GA solutions.
//!           N_MC_PER_SEED=1000 uniform random samples per GA seed within a
//!           ±3° θ, ±3° θ_sun, ±0.15 nd r_apogee window.
//! Inputs  : out/wsb/ga_best.csv  (top 30 GA solutions)
//! Outputs : out/wsb/mc_solutions.csv     — all captured MC solutions by ΔV
//!           out/wsb/refine_summary.csv   — wsb_maxhifi-compatible; one row per
//!                                          seed (source=seed) and one per
//!                                          refined solution (source=refined),
//!                                          with columns: seed_id, source,
//!                                          theta_deg, theta_sun_deg, r_apogee_nd,
//!                                          dv_kms, t_transfer_days,
//!                                          est_capture_orbits, min_alt_km,
//!                                          dtheta_deg, dsun_deg, dr_apogee_nd,
//!                                          entered_hill, max_hill_dwell_nd,
//!                                          score, crashed_seed
//!           out/wsb/mc_traj.csv          — top N_TOP_TRAJ solution trajectories
//!                                          for plotting (time_nd, x_nd, y_nd, ...)
//! CLI     : --reprop [N]  re-propagate hit N from refine_summary.csv
//! Pre-run : wsb_optimize
//!
//! ── wsb_maxhifi ─────────────────────────────────────────────────────────────
//! Purpose : Single maximum-fidelity Dopri5 repropagation of the best hit
//!           (highest est_capture_orbits in refine_summary.csv).
//!           Uses sparse output mode (every accepted adaptive step) for
//!           maximum temporal resolution near lunar periapsis.
//! Inputs  : out/wsb/refine_summary.csv  (reads top hit, or --hit N)
//! Config  : RTOL=1e-10, ATOL=1e-12, H_INIT=1e-4, T_PROP=20π nd (~300 days)
//! Outputs : out/wsb/solution_hifi.csv  — metadata row (IC + orbit quality)
//!           out/wsb/maxhifi.csv        — full trajectory, every adaptive step
//!                                        (time_nd, x_nd, y_nd, z_nd, vx_nd, ...)
//!           out/wsb/maxhifi_info.txt   — IC summary + step-size statistics
//! CLI     : --hit N
//! Pre-run : wsb_refine
//!
//! ── wsb_circularize ─────────────────────────────────────────────────────────
//! Purpose : Post-ballistic-capture lunar orbit insertion (LOI) burn and
//!           conversion from BCR4BP rotating frame to ECI (J2000).
//!           Reads DE440S-based Moon ephemeris for accurate frame conversion.
//! Inputs  : out/wsb/maxhifi.csv        (BCR4BP trajectory)
//!           out/wsb/solution_hifi.csv  (theta_sun_deg for epoch)
//!           ../Artemis/out/moon_ephem.csv  (Moon ephemeris)
//! Algorithm:
//!   1. Find first periapsis inside Hill sphere.
//!   2. Compute circularisation ΔV in rotating-frame coords.
//!   3. Convert to ECI using theta_sun_deg + DE440S ephemeris.
//!   4. Propagate post-LOI orbit (2-body Moon-centred, km/s).
//! Outputs : out/wsb/capture.csv            — pre-burn BCR4BP trajectory
//!           out/wsb/loi_orbit.csv          — post-burn ECI orbit
//!                                            (time_s, x_km, y_km, z_km,
//!                                             moon_x_km, moon_y_km, moon_z_km)
//!           out/wsb/epoch_info.txt         — WSB departure epoch + R0_wsb
//!                                            rotation matrix for ECI frame
//!                                            (needed by all animation scripts)
//!           out/wsb/circularize_info.txt   — LOI summary (altitude, ΔV, etc.)
//! Pre-run : wsb_maxhifi
//!
//! ── wsb_sensitivity ─────────────────────────────────────────────────────────
//! Purpose : N_PERTURB=200 trajectories with Gaussian (Box-Muller) perturbations
//!           on the nominal IC — visualises first-order sensitivity to injection
//!           errors.
//! Inputs  : out/wsb/refine_summary.csv  (one hit, by rank) or CLI override
//! Perturbations (1σ):
//!   ΔV magnitude  : 1e-3  (±0.1 % thrust)
//!   Pointing      : 1.75e-3 rad (±0.1°, pitch and yaw independently)
//!   Burn timing   : 0.2°  (≈ ±3 s on 92-min LEO)
//!   Launch window : 3.0°  (≈ ±5.9 hr Sun angle)
//! Outputs : out/wsb/sensitivity_ensemble[_TAG].csv
//!               columns: run_id, time_nd, x_nd, y_nd, z_nd, vx_nd, vy_nd,
//!                        vz_nd, outcome, n_orbits, est_orbits, min_alt_km,
//!                        is_nominal, dmag_frac, dpitch_rad, dyaw_rad,
//!                        dtheta_deg, dtsun_deg
//!               one NaN row per run (metadata), rest are trajectory points
//!           out/wsb/sensitivity_summary[_TAG].txt   — human-readable breakdown
//! CLI     : --hifi  --hit N  --tag NAME
//!           --theta X --r-apogee Y --theta-sun Z  (bypass refine_summary)
//! Pre-run : wsb_refine (or wsb_maxhifi for --hifi mode)
//!
//! ── wsb_sensitivity_individual ──────────────────────────────────────────────
//! Purpose : One-At-A-Time (OAT) sensitivity sweep: each of the 4 perturbation
//!           parameters swept from −3σ to +3σ independently.  Useful to isolate
//!           which parameter drives sensitivity.
//! Inputs  : Same as wsb_sensitivity
//! Grid    : N_PER_PARAM=50 steps × 4 params + 1 nominal = 201 trajectories.
//!           run_id groups: 0=nominal, 1–50=ΔV mag, 51–100=pointing,
//!           101–150=burn timing, 151–200=launch window.
//! Outputs : Same format as wsb_sensitivity (sensitivity_ensemble compatible)
//!           out/wsb/sensitivity_ensemble[_TAG].csv
//!           out/wsb/sensitivity_summary[_TAG].txt
//! Pre-run : wsb_refine
//!
//! ── wsb_stats ───────────────────────────────────────────────────────────────
//! Purpose : Full Gaussian MC sigma-sweep (20 000 samples) for statistical
//!           outcome fractions.  Does NOT save individual trajectory data
//!           during the MC run to save memory and time; instead re-propagates
//!           a stratified subset at animation resolution afterwards.
//! Inputs  : out/wsb/refine_summary.csv  or CLI override
//! Config  : N_PER_LEVEL=20 000, SIGMA_SCALES=[1.0], T_PROP=25π nd
//! Classification (windowed to hill_entry + ANIM_SAVE_WIN):
//!   EarthCrash   — any step inside r_earth_nd
//!   MoonCrash    — any step inside r_moon_nd (within show window)
//!   Captured     — est_orbits ≥ capture_est_thresh (3.5 days ≈ 0.128)
//!   LateTransfer — Captured but hill_entry > nominal + 90 days
//!   Escaped      — everything else
//! Outputs : out/wsb/stats_samples[_TAG].csv   — one row per trajectory
//!               (sigma_scale, run_id, perturbations, outcome, n_orbits,
//!                est_orbits, min_alt_km, hill_entry_nd)
//!           out/wsb/stats_sweep[_TAG].csv     — aggregated per sigma level
//!               (n_total, n_captured, n_flyby, n_moon_crash, n_earth_crash,
//!                n_escaped, fractions)
//!           out/wsb/sensitivity_ensemble[_TAG].csv
//!               — stratified animation export:
//!                 EXPORT_N_CRASH=3 (moon crashes confirmed at ANIM resolution)
//!                 EXPORT_N_FLYBY=10 (captured, est_orbits < threshold)
//!                 EXPORT_N_CAP=3   (captured, est_orbits ≥ threshold)
//!                 EXPORT_N_MISS=184 (escaped)
//!                 All re-propagated at ANIM_LOG_DT=0.002 nd for smooth video.
//!                 Timing gate: all must enter Hill sphere within 10 days of
//!                 nominal (same window as shown in animation).
//! CLI     : --hifi  --hit N  --tag NAME  --seed N
//!           --theta X --r-apogee Y --theta-sun Z
//! Pre-run : wsb_refine
//!
//! ── wsb_dense_traj ──────────────────────────────────────────────────────────
//! Purpose : Re-propagate 10 hand-picked diverse solutions with Dopri5 sparse
//!           output (every accepted adaptive step) for the smoothest possible
//!           lunar periapsis arcs in the multi-solution animation.
//! Inputs  : Hardcoded SOLUTIONS array (10 rows of theta, theta_sun, r_apogee).
//! Config  : H_MAX=0.02 nd, T_PROP=20π nd, RTOL=1e-8, ATOL=1e-10
//! Outputs : out/wsb/dense_traj.csv
//!               columns: time_nd, x_nd, y_nd, z_nd, vx_nd, vy_nd, vz_nd,
//!                        run_id, seed_id, theta_deg, theta_sun_deg,
//!                        r_apogee_nd, dv_total_kms, est_capture_orbits
//! Pre-run : none (standalone; hardcoded ICs)
//!
//! ── wsb_basin ───────────────────────────────────────────────────────────────
//! Purpose : 2-D/3-D grid sweep classifying each (θ, θ_sun) combination at
//!           one or more r_apogee slices.  Produces the capture-basin map.
//! Config  : THETA_STEPS=1500, SUN_STEPS=1500, APO_SLICES=[3.9]
//!           T_PROP=8π nd (~112 days), LOG_DT=0.01 nd
//! Classification:
//!   captured   — Hill entry + est_orbits ≥ MIN_CAPTURE_ORBITS=0.5
//!   capturable — Hill entry + est_orbits < 0.5 but ΔV_LOI < 0.4 km/s
//!   moon_crash / escaped
//! Outputs : out/wsb/basin_sweep.csv
//!               columns: r_apogee_nd, theta_deg, theta_sun_deg,
//!                        outcome, est_orbits, min_loi_kms
//! Runtime : ~2–3 min per r_apogee slice at 1500×1500 resolution
//! Pre-run : none (standalone)
//!
//! ═══════════════════════════════════════════════════════════════════════════════
//! PYTHON PLOT SCRIPTS — INPUTS, OUTPUTS, PRE-RUNS
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! All scripts live in plot/ and are invoked from the crate root.
//! Shared constants and helpers are in plot/wsb_style.py (imported by all).
//!
//! ── plot/wsb_plots.py ──────────────────────────────────────────────────────
//! Purpose : Unified plot orchestrator — toggle groups on/off via CONFIG dict.
//! Runs    : Each group calls its own script as a subprocess.
//! Invocation: python plot/wsb_plots.py
//!
//! ── plot/plot_wsb_search.py ────────────────────────────────────────────────
//! Reads   : out/wsb/blt_candidates.csv
//!           out/wsb/blt_info.txt
//!           out/wsb/family_analysis.csv
//! Writes  : out/wsb/wsb_search.html    — inertial-frame trajectory viewer
//!           out/wsb/wsb_pareto.html    — ΔV vs transfer-time Pareto front
//!           out/wsb/wsb_heatmap.html   — (θ × θ_sun) capture-quality heatmap
//!           out/wsb/wsb_scatter.html   — orbits vs transfer time
//! Pre-run : wsb_search
//!
//! ── plot/plot_wsb_phases.py ────────────────────────────────────────────────
//! Reads   : out/wsb/backward_solutions.csv, out/wsb/forward_screen.csv,
//!           out/wsb/blt_candidates.csv, out/wsb/family_analysis.csv
//! Writes  : out/wsb/wsb_phase1_capture.html  — Phase 1 backward arcs
//!           out/wsb/wsb_phase2_alpha.html    — Phase 2 α-filter results
//!           out/wsb/wsb_phase3_captures.html — Phase 3+4 forward transfers
//! Pre-run : wsb_search
//!
//! ── plot/plot_wsb_anim.py ──────────────────────────────────────────────────
//! Reads   : out/wsb/blt_candidates.csv
//!           out/wsb/blt_info.txt
//!           out/wsb/epoch_info.txt  (optional; 23.4° ecliptic tilt fallback)
//! Writes  : out/wsb/wsb_anim.html  — 2-panel Plotly animation
//!           Left: 3D ECI, Right: Moon-centred, daily frames
//! Pre-run : wsb_search
//! Note    : For epoch_info.txt run wsb_circularize first.
//!
//! ── plot/plot_wsb_solutions.py ─────────────────────────────────────────────
//! Reads   : out/wsb/mc_solutions.csv
//!           out/wsb/ga_solutions.csv
//!           out/wsb/mc_traj.csv
//! Writes  : out/wsb/wsb_solutions.html
//!           (5-panel: Pareto front, ΔV breakdown, rotating frame, inertial,
//!            Moon-centred)
//! Pre-run : wsb_refine + wsb_optimize
//!
//! ── plot/plot_wsb_solution_anim.py ─────────────────────────────────────────
//! Reads   : out/wsb/solution_hifi.csv
//!           out/wsb/maxhifi.csv
//!           out/wsb/epoch_info.txt  (optional)
//! Writes  : out/wsb/wsb_solution_anim.html  — animated 2-panel best solution
//! CLI     : --fps N  --step HOURS
//! Pre-run : wsb_maxhifi  (+ wsb_circularize for exact ECI frame)
//!
//! ── plot/plot_wsb_multi_anim.py ────────────────────────────────────────────
//! Reads   : out/wsb/mc_traj.csv
//!           out/wsb/mc_solutions.csv
//!           out/wsb/epoch_info.txt  (optional)
//! Writes  : out/wsb/wsb_multi_anim.html  — N fastest MC solutions animated
//! CLI     : --n N  --fps FPS  --step HOURS  --maxpts N
//! Pre-run : wsb_refine
//!
//! ── plot/plot_wsb_multi_diverse.py ─────────────────────────────────────────
//! Reads   : out/wsb/dense_traj.csv  (preferred)
//!           or out/wsb/all_refinements.csv  (fallback)
//!           out/wsb/epoch_info.txt  (optional)
//! Writes  : out/wsb/wsb_multi_diverse.html
//! CLI     : --n N  --fps FPS  --step HOURS  --video
//! Pre-run : wsb_dense_traj  (or wsb_refine as fallback)
//!
//! ── plot/plot_wsb_covariance.py ────────────────────────────────────────────
//! Reads   : out/wsb/mc_solutions.csv, out/wsb/ga_solutions.csv
//! Writes  : out/wsb/wsb_param_sensitivity.html
//!           (2×2: θ/θ_sun/r_apo/α vs ΔV, MC circles + GA diamonds)
//! Pre-run : wsb_refine + wsb_optimize
//!
//! ── plot/plot_wsb_basin_anim.py ────────────────────────────────────────────
//! Reads   : out/wsb/basin_sweep.csv  (preferred)
//!           or out/wsb/family_analysis.csv  (fallback)
//!           out/wsb/mc_solutions.csv  (optional overlay)
//! Writes  : out/wsb/wsb_basin_anim.html  — animated (θ × θ_sun) basin map
//! CLI     : --source basin|family
//! Pre-run : wsb_basin  (or wsb_search as fallback)
//!
//! ── plot/plot_wsb_sensitivity_anim.py ──────────────────────────────────────
//! Purpose : Animated comet-tail ensemble of perturbed trajectories.
//!           Left panel: 3D ECI (Earth sphere, Moon moving).
//!           Right panel: Moon-centred Hill-sphere region (2D, orthographic).
//!           Outcome colours: nominal=cyan, captured=green (#1AE870),
//!           flyby=purple, moon_crash=red, miss=grey.
//!           flyby/capture split: est_orbits threshold = 3.5 days (≈ 0.128 EM periods).
//! Reads   : out/wsb/sensitivity_ensemble[_TAG].csv
//!           out/wsb/sensitivity_summary[_TAG].txt  (optional, title metadata)
//!           out/wsb/epoch_info.txt  (optional; 23.4° tilt fallback)
//!           out/wsb/stats_sweep[_TAG].csv  (optional; shown in HUD legend)
//! Writes  : out/wsb/wsb_sensitivity_anim[_TAG].html  (always — Plotly)
//!           out/wsb/wsb_sensitivity_anim[_TAG].mp4   (--mp4, via PyVista+ffmpeg)
//!           out/wsb/wsb_sensitivity_anim[_TAG].gif   (--gif, via PyVista+Pillow)
//! CLI     : --tag TAG  --fps N  --step HOURS  --tail DAYS  --mp4  --gif
//!           --vid-n N  (cap trajectories in mp4/gif; 0 = all)
//!           --preview  (render single mid-frame PNG, fast sanity check)
//!           --stride N  (use every N-th CSV point, default 2)
//! Pre-run : wsb_stats  (preferred — guarantees outcome diversity via 20k MC)
//!           or wsb_sensitivity / wsb_sensitivity_individual  (for HTML only)
//! Note    : wsb_stats writes sensitivity_ensemble.csv with stratified selection
//!           (guaranteed crashes/flybys/captures from 20k run).  wsb_sensitivity
//!           writes the same file from 200 samples (less diverse, fine for HTML).
//!
//! ── plot/plot_wsb_sensitivity_png.py ───────────────────────────────────────
//! Reads   : out/wsb/sensitivity_ensemble[_TAG].csv
//!           out/wsb/epoch_info.txt  (optional)
//! Writes  : out/wsb/wsb_sensitivity_png[_TAG].png
//!           (matplotlib, ECI top-down, coloured by dominant input perturbation)
//! Pre-run : wsb_sensitivity or wsb_sensitivity_individual
//!
//! ── plot/plot_wsb_sensitivity_3d.py ────────────────────────────────────────
//! Reads   : out/wsb/sensitivity_ensemble[_TAG].csv
//!           out/wsb/epoch_info.txt  (optional)
//! Writes  : out/wsb/wsb_sensitivity_3d[_TAG].png
//!           (matplotlib 3D, shows out-of-plane yaw dispersion)
//! CLI     : --tag TAG  --view KM  --elev DEG  --azim DEG
//! Pre-run : wsb_sensitivity or wsb_sensitivity_individual
//!
//! ── plot/plot_wsb_stats.py ─────────────────────────────────────────────────
//! Reads   : out/wsb/stats_samples[_TAG].csv
//!           out/wsb/stats_sweep[_TAG].csv
//! Writes  : out/wsb/wsb_stats[_TAG].png
//!           (stacked-bar outcome fractions + violin perturbation distributions)
//! Pre-run : wsb_stats
//!
//! ── plot/plot_wsb_final_refinement.py ──────────────────────────────────────
//! Reads   : out/wsb/sensitivity_ensemble.csv
//!           out/wsb/sensitivity_summary.txt  (optional)
//! Writes  : out/wsb/wsb_sensitivity.html
//!           (3D rotating frame + ECI 2D, outcome colouring)
//! Pre-run : wsb_sensitivity or wsb_sensitivity_individual
//!
//! ── plot/plot_wsb_vs_artemis.py ────────────────────────────────────────────
//! Reads   : out/wsb/solution_hifi.csv, out/wsb/maxhifi.csv
//!           out/wsb/capture.csv, out/wsb/epoch_info.txt  (optional)
//!           out/wsb/loi_orbit.csv  (optional)
//!           ../Artemis/out/artemis2_trajectory.csv
//!           ../Artemis/out/moon_ephem.csv
//! Writes  : out/wsb/wsb_vs_artemis_3d.html
//!           out/wsb/wsb_vs_artemis_portrait.png
//!           out/wsb/wsb_vs_artemis.mp4  (requires ffmpeg)
//! Pre-run : wsb_maxhifi + wsb_circularize + Artemis pipeline
//!
//! ═══════════════════════════════════════════════════════════════════════════════
//! DATA FLOW SUMMARY
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//!  wsb_search
//!    ↓ family_analysis.csv
//!  wsb_optimize
//!    ↓ ga_best.csv
//!  wsb_refine
//!    ↓ refine_summary.csv   ←── read by: wsb_maxhifi, wsb_sensitivity,
//!    ↓ mc_solutions.csv          wsb_sensitivity_individual, wsb_stats
//!    ↓ mc_traj.csv
//!  wsb_maxhifi
//!    ↓ maxhifi.csv, solution_hifi.csv
//!  wsb_circularize
//!    ↓ capture.csv, loi_orbit.csv, epoch_info.txt (needed by all anim scripts)
//!
//!  [independent after wsb_refine]
//!  wsb_stats ──────────────────→ sensitivity_ensemble.csv (for animation)
//!                                 stats_samples.csv, stats_sweep.csv
//!  wsb_sensitivity ────────────→ sensitivity_ensemble.csv (200 samples)
//!  wsb_sensitivity_individual →  sensitivity_ensemble.csv (OAT grid)
//!  wsb_dense_traj ─────────────→ dense_traj.csv
//!  wsb_basin ──────────────────→ basin_sweep.csv
//!
//! # Config
//!
//! All physically meaningful parameters live in the `Config` struct below.
//! Step-specific tuning knobs (grid density, tolerances, sample counts) live
//! inside each individual binary source file.

use std::path::PathBuf;
use std::process::Command;

/// Absolute path to the crate root (the directory containing Cargo.toml).
/// Baked in at compile time — used to set the working directory for all
/// subprocesses so that relative paths like `out/wsb/` resolve correctly
/// regardless of where the user invokes cargo from.
const CRATE_DIR: &str = env!("CARGO_MANIFEST_DIR");

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                      USER CONFIGURATION — edit here                         ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

struct Config {
    // ── Physics ───────────────────────────────────────────────────────────────

    /// Earth parking orbit altitude [km].  Artemis II: 378 km.
    earth_alt_km: f64,

    /// Target apogee distance from Earth [nd].
    /// WSB sits near the Sun–Earth L1 at ≈ 3.9 nd ≈ 1.5 × 10⁶ km.
    r_apogee_nd: f64,

    // ── Pipeline control ──────────────────────────────────────────────────────

    /// Run wsb_search (Phase 1–4 backward/forward sweep).
    run_search: bool,

    /// Run wsb_refine (MC sampling, ΔV_TLI + ΔV_LOI objective).
    run_refine: bool,

    /// Run wsb_optimize (GA optimizer over [θ, θ_sun, r_apogee]).
    run_optimize: bool,

    /// Run wsb_maxhifi (Dopri5 sparse max-fidelity re-propagation of best solution).
    run_maxhifi: bool,

    /// Run wsb_circularize (LOI burn + circularised orbit output).
    run_circularize: bool,

    /// Run wsb_sensitivity (micro-perturbation ensemble, optional).
    run_sensitivity: bool,

    /// Run wsb_basin (capture basin 2-D sweep at multiple r_apogee, optional).
    run_basin: bool,

    /// Call Python plot scripts after each step (requires Python + plotly).
    run_plots: bool,
}

const CONFIG: Config = Config {
    earth_alt_km:    378.0,
    r_apogee_nd:     3.9,

    run_search:      true,
    run_refine:      true,
    run_optimize:    true,     // GA global search: reads family_analysis.csv, writes ga_best.csv
    run_maxhifi:     true,
    run_circularize: true,
    run_sensitivity: false,    // optional diagnostics
    run_basin:       false,    // optional — ~2880 trajectories, ~2-3 min
    run_plots:       true,
};

// ════════════════════════════════════════════════════════════════════════════════

fn main() {
    let r_park_nd = (6_371.0 + CONFIG.earth_alt_km) / 384_400.0;

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║           WSB Pipeline — unified workflow            ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  Earth alt  : {:.0} km  →  R_park = {r_park_nd:.6} nd",
        CONFIG.earth_alt_km);
    println!("  R_apogee   : {:.2} nd  ({:.0} km)",
        CONFIG.r_apogee_nd, CONFIG.r_apogee_nd * 384_400.0);
    println!();

    // Ensure output directory exists
    std::fs::create_dir_all("out/wsb").expect("Cannot create out/wsb/");

    // Rebuild all step binaries so any source changes take effect before running
    println!("── Build : cargo build --release ─────────────────────────");
    let build_status = std::process::Command::new("cargo")
        .args(["build", "-p", "lunar_trajectories", "--release"])
        .current_dir(CRATE_DIR)
        .status()
        .expect("Failed to invoke cargo build");
    if !build_status.success() {
        eprintln!("  [pipeline] Build failed — aborting.");
        std::process::exit(build_status.code().unwrap_or(1));
    }
    println!("  [pipeline] Build OK.");
    println!();

    // Find sibling binaries (same target directory as this pipeline binary)
    let bin_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from));

    let env_vars: Vec<(&str, String)> = vec![
        ("WSB_R_PARK_ND",  format!("{:.10}", r_park_nd)),
        ("WSB_R_APOGEE_ND", format!("{:.6}", CONFIG.r_apogee_nd)),
    ];

    if CONFIG.run_search {
        println!("── Step 1 : wsb_search ────────────────────────────────");
        run_step("wsb_search", &bin_dir, &env_vars);
    }

    if CONFIG.run_optimize {
        println!("── Step 2 : wsb_optimize (GA — global search) ─────────");
        run_step("wsb_optimize", &bin_dir, &env_vars);
    }

    if CONFIG.run_refine {
        println!("── Step 3 : wsb_refine (MC — local polish) ────────────");
        run_step("wsb_refine", &bin_dir, &env_vars);
    }

    if CONFIG.run_maxhifi {
        println!("── Step 4 : wsb_maxhifi ───────────────────────────────");
        run_step("wsb_maxhifi", &bin_dir, &env_vars);
    }

    if CONFIG.run_circularize {
        println!("── Step 5 : wsb_circularize ───────────────────────────");
        run_step("wsb_circularize", &bin_dir, &env_vars);
    }

    if CONFIG.run_sensitivity {
        println!("── Optional: wsb_sensitivity ──────────────────────────");
        run_step("wsb_sensitivity", &bin_dir, &env_vars);
    }

    if CONFIG.run_basin {
        println!("── Optional: wsb_basin ────────────────────────────────");
        run_step("wsb_basin", &bin_dir, &env_vars);
    }

    // All plots go through the unified plotter — configure toggles in wsb_plots.py
    if CONFIG.run_plots {
        println!("── Plots : wsb_plots.py ───────────────────────────────");
        run_python("plot/wsb_plots.py");
    }

    println!();
    println!("=== Pipeline complete ===");
    println!("  Outputs    : out/wsb/");
    println!("  Archive    : out/wsb_solutions.csv");
    if CONFIG.run_plots {
        println!("  Plots      : out/wsb/*.html");
        println!("  Replot     : python plot/wsb_plots.py");
    }
}

/// Run a WSB step binary.
///
/// Strategy:
///   1. Look for a pre-built sibling in the same target/release dir as this binary.
///   2. If not found, invoke `cargo run -p lunar_trajectories --release --bin <name>`.
///
/// In both cases the subprocess runs with `CRATE_DIR` as its working directory,
/// so relative paths like `out/wsb/` always resolve to the right place.
fn run_step(name: &str, bin_dir: &Option<PathBuf>, env_vars: &[(&str, String)]) {
    let exe_path = bin_dir
        .as_ref()
        .map(|d| d.join(name))
        .filter(|p| p.exists());

    let mut cmd = if let Some(exe) = exe_path {
        println!("  [pipeline] Using pre-built binary: {}", exe.display());
        Command::new(exe)
    } else {
        println!("  [pipeline] Binary not pre-built — invoking cargo run …");
        let mut c = Command::new("cargo");
        c.args(["run", "--release", "-p", "lunar_trajectories", "--bin", name]);
        c
    };

    // Always run from the crate root so out/wsb/ paths resolve correctly
    cmd.current_dir(CRATE_DIR);

    for (key, val) in env_vars {
        cmd.env(key, val);
    }

    let status = cmd.status().unwrap_or_else(|e| {
        eprintln!("  [pipeline] Failed to launch {name}: {e}");
        std::process::exit(1);
    });

    if !status.success() {
        eprintln!("  [pipeline] {name} exited with code {:?}", status.code());
        std::process::exit(status.code().unwrap_or(1));
    }
    println!("  [pipeline] {name} completed successfully.");
    println!();
}

/// Run a Python plot script from the crate root (non-fatal if Python unavailable).
///
/// Tries `python` first (Anaconda / conda env which has numpy/plotly),
/// then falls back to `python3` (system Python).
fn run_python(script: &str) {
    // Try python (Anaconda) first, then python3 (system fallback)
    for interpreter in ["python", "python3"] {
        match Command::new(interpreter)
            .arg(script)
            .current_dir(CRATE_DIR)
            .status()
        {
            Err(_) => continue,   // interpreter not found — try next
            Ok(s) if s.success() => {
                println!("  [pipeline] {script} done.");
                return;
            }
            Ok(s) => {
                eprintln!("  [pipeline] {script} failed with {interpreter} (code {:?}, non-fatal)",
                    s.code());
                return;
            }
        }
    }
    eprintln!("  [pipeline] Cannot run {script}: no Python interpreter found (skipping)");
}
