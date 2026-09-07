//! Ballistic (no-DSM) powered-flyby MGA grid scan — STOUR-class launch-window
//! scanner (Phase 9w-i).
//!
//! The middle tier of the three-tier MGA pipeline: takes a candidate body
//! sequence (from the Tisserand beam search) and scans a departure-date ×
//! per-leg time-of-flight grid with a pruned depth-first tree search. Every
//! evaluation is a Lambert solve plus an analytic powered-flyby feasibility
//! check — no numerical propagation, no optimizer — so multi-decade horizons
//! scan in seconds-to-minutes and the launch-window structure comes out sharp
//! (DSMs deliberately excluded: they blur phasing; the MGA-1DSM DE adds them
//! back in tier 3).
//!
//! # Model
//! Each leg is a pure ballistic Lambert arc between body positions. At each
//! intermediate flyby the incoming and outgoing hyperbolic excess velocities
//! generally disagree in both direction and magnitude; the mismatch is priced
//! as a single tangential impulse at the shared periapsis of the incoming and
//! outgoing hyperbolas (see [`powered_flyby`]). A branch is *infeasible* —
//! pruned, never penalty-blurred — when the required turn angle exceeds the
//! maximum achievable above the body's minimum periapsis radius. This is
//! exactly the GTOP **Cassini-1** problem formulation, which serves as this
//! module's validation benchmark (Phase 9w-ii).
//!
//! The scan reports raw components (departure v∞, per-flyby powered ΔVs,
//! arrival v∞); converting departure v∞ to a parking-orbit escape burn and
//! arrival v∞ to an insertion burn is the caller's job — those models are
//! mission-specific and live in `MissionPlanner` (`design.rs`), not here.
//!
//! # References
//! - Rinderle (1986), "Galileo User's Guide, Mission Design: Satellite Tour
//!   Analysis and Design Program (STOUR)", JPL D-263.
//! - Longuski & Williams (1991), "Automated Design of Gravity-Assist
//!   Trajectories to Mars and the Outer Planets", Celest. Mech. Dyn. Astron.
//!   52:207–220 (the pruned depth-first grid search this scan follows).
//! - Izzo & Vinkó, ESA GTOP database — Cassini-1 problem (powered-flyby
//!   MGA model; the periapsis-burn matching used here).
//! - Battin (1999) §6.3 — hyperbolic flyby turn-angle geometry.

use nalgebra::Vector3;
use orbital_math::lambert::lambert;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// One body in the scan sequence (departure body, intermediate flyby bodies,
/// target body — in visit order).
///
/// `state_at(t)` gives the body's heliocentric (position, velocity) at
/// absolute mission time `t_s` [s], same convention as
/// [`crate::propagator::PropagatorBody`]. `mu_m3s2` and `min_periapsis_m`
/// are only used when the body acts as an intermediate flyby body; they are
/// ignored for the first and last entries.
pub struct ScanBody<'a> {
    pub name: &'a str,
    /// Gravitational parameter [m³/s²] — flyby turn geometry.
    pub mu_m3s2: f64,
    /// Minimum allowed flyby periapsis radius [m] (body radius + safety margin).
    pub min_periapsis_m: f64,
    /// Heliocentric (position [m], velocity [m/s]) at absolute epoch `t_s` [s].
    ///
    /// `+ Sync` (widened from a plain `dyn Fn` for the parallel scan below):
    /// `run_mga_scan` shares `bodies: &[ScanBody]` across worker threads, so
    /// every field — including this closure — must be safely readable from
    /// multiple threads at once. Callers' closures (e.g.
    /// `MissionPlanner/src/mga_scan_run.rs`'s ANISE query, which captures
    /// `&Almanac` by move) already satisfy this: they hold no interior
    /// mutability, and `Almanac` is already queried concurrently elsewhere in
    /// this codebase (MGA's MBH worker threads, `mga.rs`).
    pub state_at: &'a (dyn Fn(f64) -> (Vector3<f64>, Vector3<f64>) + Sync),
}

/// Grid definition and pruning thresholds for one scan run.
pub struct MgaScanConfig {
    /// Sun's gravitational parameter [m³/s²].
    pub mu_sun: f64,
    /// Departure epoch grid, absolute mission time [s]. The caller converts
    /// from calendar dates / Julian Dates — this module has no time system.
    pub departure_epochs_s: Vec<f64>,
    /// One TOF grid [s] per leg; `leg_tof_grids_s.len()` must equal
    /// `bodies.len() - 1`. Grids are indexed by leg position (leg 0 departs
    /// the departure body), matching `MgaParams.leg_tof_days` convention.
    pub leg_tof_grids_s: Vec<Vec<f64>>,
    /// Prune branches whose departure v∞ exceeds this [m/s].
    pub vinf_dep_max_ms: f64,
    /// Prune branches whose powered-flyby matching burn exceeds this [m/s]
    /// at any single flyby. This is the "v∞-match tolerance" expressed as a
    /// ΔV cost rather than a hard magnitude-equality test.
    pub flyby_dv_max_ms: f64,
    /// Prune complete branches whose arrival v∞ exceeds this [m/s].
    /// `f64::INFINITY` disables the check (e.g. pure flyby missions).
    pub vinf_arr_max_ms: f64,
    /// Hard cap on stored records — a safety valve against a pathological
    /// grid/threshold combination flooding memory. Branches beyond the cap
    /// are still counted (see [`MgaScanOutput::n_records_dropped`]) so the
    /// caller knows the output is truncated and can tighten the pruning.
    pub max_records: usize,
}

/// One feasible complete branch (departure → all flybys → target).
#[derive(Clone, Debug)]
pub struct ScanRecord {
    /// Departure epoch, absolute mission time [s].
    pub dep_epoch_s: f64,
    /// Per-leg times of flight [s], in leg order.
    pub leg_tofs_s: Vec<f64>,
    /// Departure hyperbolic excess speed [m/s].
    pub vinf_dep_ms: f64,
    /// Arrival hyperbolic excess speed at the target [m/s].
    pub vinf_arr_ms: f64,
    /// Powered-flyby matching burn at each intermediate body [m/s].
    pub flyby_dvs_ms: Vec<f64>,
    /// Solved shared periapsis radius at each flyby [m].
    pub flyby_rp_m: Vec<f64>,
    /// Total turn angle achieved at each flyby [rad].
    pub flyby_turn_rad: Vec<f64>,
    /// Sum of `flyby_dvs_ms` [m/s] — the branch's total powered-flyby cost.
    /// Departure-escape and arrival-insertion burns are NOT included; the
    /// caller prices those with its own mission-specific models.
    pub sum_flyby_dv_ms: f64,
}

/// Scan output: all feasible branches plus bookkeeping counters.
pub struct MgaScanOutput {
    pub records: Vec<ScanRecord>,
    /// Total Lambert-leg evaluations performed (pruned + kept).
    pub n_legs_evaluated: u64,
    /// Feasible complete branches dropped because `max_records` was hit.
    pub n_records_dropped: u64,
}

/// Result of the powered-flyby periapsis-burn match.
#[derive(Clone, Copy, Debug)]
pub struct PoweredFlyby {
    /// Tangential matching burn at periapsis [m/s].
    pub dv_ms: f64,
    /// Shared periapsis radius of the incoming/outgoing hyperbolas [m].
    pub rp_m: f64,
    /// Total turn angle δ_in + δ_out = α [rad].
    pub turn_rad: f64,
}

/// Maximum turn angle a single unpowered hyperbolic passage can achieve at
/// periapsis radius `rp_m` for excess speed `vinf_ms`:
/// `δ_max = 2·arcsin(1 / (1 + r_p·v_∞²/μ))` (Battin 1999 §6.3).
pub fn max_turn_angle_rad(vinf_ms: f64, rp_m: f64, mu_body: f64) -> f64 {
    let e = 1.0 + rp_m * vinf_ms * vinf_ms / mu_body;
    2.0 * (1.0 / e).asin()
}

/// Solve the powered-flyby match: connect an incoming hyperbola (excess speed
/// `vinf_in_ms`) to an outgoing hyperbola (excess speed `vinf_out_ms`) turned
/// by `alpha_rad`, with a single tangential burn at their shared periapsis.
///
/// The shared periapsis radius `r_p` satisfies
/// `arcsin(1/e_in(r_p)) + arcsin(1/e_out(r_p)) = α` with
/// `e_i = 1 + r_p·v_∞i²/μ` — each hyperbola contributes half-turn `δ_i/2 =
/// arcsin(1/e_i)`. The left side is strictly decreasing in `r_p`, so the root
/// is found by bisection. The matching burn is then
/// `ΔV = |√(v_∞out² + 2μ/r_p) − √(v_∞in² + 2μ/r_p)|` (vis-viva at periapsis).
///
/// Returns `None` when the required turn exceeds the maximum achievable at
/// `rp_min_m` — the branch is infeasible, per this scan's feasible-only rule.
/// This is the GTOP Cassini-1 `PowSwingByInv` model.
pub fn powered_flyby(
    vinf_in_ms:  f64,
    vinf_out_ms: f64,
    alpha_rad:   f64,
    mu_body:     f64,
    rp_min_m:    f64,
) -> Option<PoweredFlyby> {
    let half_turns = |rp: f64| -> f64 {
        let e_in  = 1.0 + rp * vinf_in_ms * vinf_in_ms / mu_body;
        let e_out = 1.0 + rp * vinf_out_ms * vinf_out_ms / mu_body;
        (1.0 / e_in).asin() + (1.0 / e_out).asin()
    };

    // Feasibility: the deepest allowed passage must achieve at least α.
    if half_turns(rp_min_m) < alpha_rad {
        return None;
    }

    // Bracket the root: half_turns is strictly decreasing in rp, positive at
    // rp_min, and → 0 as rp → ∞. Expand geometrically until it drops below α,
    // capped at RP_MAX — for a near-zero α the root runs off to infinity where
    // the burn tends to the plain |v_∞out − v_∞in| magnitude match; clamping
    // at RP_MAX loses nothing physical (2μ/r_p is negligible there).
    const RP_MAX_M: f64 = 1.0e14;
    let rp = if alpha_rad <= 0.0 {
        RP_MAX_M
    } else {
        let mut lo = rp_min_m;
        let mut hi = rp_min_m * 2.0;
        while half_turns(hi) > alpha_rad && hi < RP_MAX_M {
            lo = hi;
            hi *= 2.0;
        }
        if hi >= RP_MAX_M && half_turns(RP_MAX_M) > alpha_rad {
            RP_MAX_M
        } else {
            for _ in 0..100 {
                let mid = 0.5 * (lo + hi);
                if half_turns(mid) > alpha_rad { lo = mid; } else { hi = mid; }
                if (hi - lo) / hi < 1e-14 { break; }
            }
            0.5 * (lo + hi)
        }
    };

    let v_peri_in  = (vinf_in_ms * vinf_in_ms + 2.0 * mu_body / rp).sqrt();
    let v_peri_out = (vinf_out_ms * vinf_out_ms + 2.0 * mu_body / rp).sqrt();
    Some(PoweredFlyby {
        dv_ms:    (v_peri_out - v_peri_in).abs(),
        rp_m:     rp,
        turn_rad: alpha_rad,
    })
}

/// Run the pruned depth-first ballistic MGA grid scan.
///
/// `bodies` is the full visit sequence (departure body first, target last —
/// at least 2 entries). Returns every feasible complete branch, up to
/// `cfg.max_records`. The caller ranks/bins the records (e.g. minimum cost
/// per departure date for the porkchop) and writes any files — this function
/// does neither, per the crate's no-I/O rule.
///
/// Internally parallel over `cfg.departure_epochs_s`: each departure epoch is
/// a fully independent depth-first tree search (no shared mutable state
/// across epochs except the final merged output), so worker threads
/// work-steal epochs off a shared index — same idiom as `MbhSolver::run`'s
/// chain parallelism (`mbh.rs`), plain `std::thread::scope`, no new
/// dependency.
pub fn run_mga_scan(bodies: &[ScanBody], cfg: &MgaScanConfig) -> MgaScanOutput {
    assert!(bodies.len() >= 2, "scan needs at least departure + target body");
    assert_eq!(
        cfg.leg_tof_grids_s.len(),
        bodies.len() - 1,
        "one TOF grid per leg required ({} bodies → {} legs)",
        bodies.len(),
        bodies.len() - 1,
    );

    let n_deps = cfg.departure_epochs_s.len();
    let workers = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(1)
        .min(n_deps.max(1));

    let next_dep = AtomicUsize::new(0);
    // One local accumulator per worker slot; each worker's `scan_leg` calls
    // cap at `cfg.max_records` independently (per the task's memory-bound
    // design — worst case total memory across `workers` threads is
    // `workers * max_records`, still bounded), so no cross-thread locking is
    // needed during the scan itself, only at the final merge.
    let worker_outs: Vec<Mutex<MgaScanOutput>> = (0..workers)
        .map(|_| Mutex::new(MgaScanOutput { records: Vec::new(), n_legs_evaluated: 0, n_records_dropped: 0 }))
        .collect();

    std::thread::scope(|s| {
        for w in 0..workers {
            let next_dep = &next_dep;
            let worker_outs = &worker_outs;
            s.spawn(move || {
                let mut local = MgaScanOutput {
                    records:           Vec::new(),
                    n_legs_evaluated:  0,
                    n_records_dropped: 0,
                };
                let mut branch = BranchState {
                    leg_tofs_s:      Vec::with_capacity(bodies.len() - 1),
                    flyby_dvs_ms:    Vec::with_capacity(bodies.len().saturating_sub(2)),
                    flyby_rp_m:      Vec::with_capacity(bodies.len().saturating_sub(2)),
                    flyby_turn_rad:  Vec::with_capacity(bodies.len().saturating_sub(2)),
                    vinf_dep_ms:     0.0,
                };

                loop {
                    let idx = next_dep.fetch_add(1, Ordering::Relaxed);
                    if idx >= n_deps { break; }
                    let t0 = cfg.departure_epochs_s[idx];
                    scan_leg(bodies, cfg, t0, t0, 0, None, &mut branch, &mut local);
                }

                *worker_outs[w].lock().unwrap() = local;
            });
        }
    });

    // Merge: sum evaluation/drop counters exactly (unaffected by any cap —
    // every leg evaluation is counted whether kept or pruned), concatenate
    // every worker's local records. If the GLOBAL cap is exceeded once
    // merged, sort by departure epoch (stable sort preserves each worker's
    // own within-epoch insertion order, matching the original single-
    // threaded scan's chronological ordering) and truncate, folding the
    // newly-truncated count into `n_records_dropped` alongside whatever each
    // worker already dropped locally.
    let mut out = MgaScanOutput {
        records:           Vec::new(),
        n_legs_evaluated:  0,
        n_records_dropped: 0,
    };
    for wo in worker_outs {
        let local = wo.into_inner().unwrap();
        out.n_legs_evaluated += local.n_legs_evaluated;
        out.n_records_dropped += local.n_records_dropped;
        out.records.extend(local.records);
    }

    if out.records.len() > cfg.max_records {
        out.records.sort_by(|a, b| a.dep_epoch_s.partial_cmp(&b.dep_epoch_s).unwrap());
        let newly_dropped = out.records.len() - cfg.max_records;
        out.records.truncate(cfg.max_records);
        out.n_records_dropped += newly_dropped as u64;
    }

    out
}

/// Sequential reference implementation of [`run_mga_scan`], kept for
/// apples-to-apples correctness testing against the parallel version above —
/// not part of the public API surface used by callers.
#[cfg(test)]
fn run_mga_scan_sequential(bodies: &[ScanBody], cfg: &MgaScanConfig) -> MgaScanOutput {
    assert!(bodies.len() >= 2, "scan needs at least departure + target body");
    assert_eq!(cfg.leg_tof_grids_s.len(), bodies.len() - 1);

    let mut out = MgaScanOutput {
        records:           Vec::new(),
        n_legs_evaluated:  0,
        n_records_dropped: 0,
    };
    let mut branch = BranchState {
        leg_tofs_s:      Vec::with_capacity(bodies.len() - 1),
        flyby_dvs_ms:    Vec::with_capacity(bodies.len().saturating_sub(2)),
        flyby_rp_m:      Vec::with_capacity(bodies.len().saturating_sub(2)),
        flyby_turn_rad:  Vec::with_capacity(bodies.len().saturating_sub(2)),
        vinf_dep_ms:     0.0,
    };
    for &t0 in &cfg.departure_epochs_s {
        scan_leg(bodies, cfg, t0, t0, 0, None, &mut branch, &mut out);
    }
    out
}

/// Mutable per-branch accumulator threaded through the depth-first recursion.
struct BranchState {
    leg_tofs_s:     Vec<f64>,
    flyby_dvs_ms:   Vec<f64>,
    flyby_rp_m:     Vec<f64>,
    flyby_turn_rad: Vec<f64>,
    vinf_dep_ms:    f64,
}

/// Depth-first recursion over leg `leg_idx` (departing `bodies[leg_idx]` at
/// absolute epoch `t_s`). `vinf_in` is the incoming heliocentric-frame excess
/// velocity vector at the current body (`None` only at the departure body).
#[allow(clippy::too_many_arguments)]
fn scan_leg(
    bodies:  &[ScanBody],
    cfg:     &MgaScanConfig,
    dep_epoch_s: f64,
    t_s:     f64,
    leg_idx: usize,
    vinf_in: Option<Vector3<f64>>,
    branch:  &mut BranchState,
    out:     &mut MgaScanOutput,
) {
    let body_a = &bodies[leg_idx];
    let body_b = &bodies[leg_idx + 1];
    let (r_a, v_a) = (body_a.state_at)(t_s);
    let is_last_leg = leg_idx + 2 == bodies.len();

    for &tof_s in &cfg.leg_tof_grids_s[leg_idx] {
        let t_arr = t_s + tof_s;
        let (r_b, v_b) = (body_b.state_at)(t_arr);
        out.n_legs_evaluated += 1;

        // All single-rev Lambert solutions, both transfer senses. Retrograde
        // heliocentric branches are almost always pruned by the v∞ caps, but
        // they are legitimate distinct branches, so the tree explores them.
        for prograde in [true, false] {
            for (v1_arr3, v2_arr3) in lambert(
                [r_a.x, r_a.y, r_a.z],
                [r_b.x, r_b.y, r_b.z],
                tof_s,
                prograde,
                cfg.mu_sun,
            ) {
                let v1 = Vector3::new(v1_arr3[0], v1_arr3[1], v1_arr3[2]);
                let v2 = Vector3::new(v2_arr3[0], v2_arr3[1], v2_arr3[2]);
                let vinf_out = v1 - v_a;

                // Connect this leg's departure to the branch so far.
                let flyby: Option<PoweredFlyby> = match vinf_in {
                    None => {
                        // Departure body: prune on departure v∞ cap.
                        if vinf_out.norm() > cfg.vinf_dep_max_ms { continue; }
                        None
                    }
                    Some(vin) => {
                        let cos_a = (vin.dot(&vinf_out)
                            / (vin.norm() * vinf_out.norm()))
                            .clamp(-1.0, 1.0);
                        let fb = match powered_flyby(
                            vin.norm(),
                            vinf_out.norm(),
                            cos_a.acos(),
                            body_a.mu_m3s2,
                            body_a.min_periapsis_m,
                        ) {
                            Some(fb) => fb,
                            None => continue, // turn infeasible above rp_min
                        };
                        if fb.dv_ms > cfg.flyby_dv_max_ms { continue; }
                        Some(fb)
                    }
                };

                let vinf_arr = v2 - v_b;
                if is_last_leg && vinf_arr.norm() > cfg.vinf_arr_max_ms {
                    continue;
                }

                // Push this leg onto the branch accumulator.
                branch.leg_tofs_s.push(tof_s);
                if let Some(fb) = flyby {
                    branch.flyby_dvs_ms.push(fb.dv_ms);
                    branch.flyby_rp_m.push(fb.rp_m);
                    branch.flyby_turn_rad.push(fb.turn_rad);
                } else {
                    branch.vinf_dep_ms = vinf_out.norm();
                }

                if is_last_leg {
                    if out.records.len() < cfg.max_records {
                        let sum_dv: f64 = branch.flyby_dvs_ms.iter().sum();
                        out.records.push(ScanRecord {
                            dep_epoch_s:    dep_epoch_s,
                            leg_tofs_s:     branch.leg_tofs_s.clone(),
                            vinf_dep_ms:    branch.vinf_dep_ms,
                            vinf_arr_ms:    vinf_arr.norm(),
                            flyby_dvs_ms:   branch.flyby_dvs_ms.clone(),
                            flyby_rp_m:     branch.flyby_rp_m.clone(),
                            flyby_turn_rad: branch.flyby_turn_rad.clone(),
                            sum_flyby_dv_ms: sum_dv,
                        });
                    } else {
                        out.n_records_dropped += 1;
                    }
                } else {
                    scan_leg(
                        bodies, cfg, dep_epoch_s, t_arr, leg_idx + 1,
                        Some(vinf_arr), branch, out,
                    );
                }

                // Pop this leg before trying the next Lambert branch.
                branch.leg_tofs_s.pop();
                if flyby.is_some() {
                    branch.flyby_dvs_ms.pop();
                    branch.flyby_rp_m.pop();
                    branch.flyby_turn_rad.pop();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sun's gravitational parameter [m³/s²] — JPL DE430
    const MU_SUN: f64 = 1.327_124_400_18e20;
    /// Earth's gravitational parameter [m³/s²] — IERS 2010
    const MU_EARTH: f64 = 3.986_004_418e14;
    const AU: f64 = 1.496e11;

    /// Equal in/out excess speeds with an achievable turn → the periapsis
    /// speeds are identical, so the matching burn is exactly zero.
    #[test]
    fn powered_flyby_symmetric_speeds_zero_dv() {
        let vinf = 6_000.0;
        let rp_min = 6_678e3; // ~300 km Earth altitude
        let alpha = 0.5 * max_turn_angle_rad(vinf, rp_min, MU_EARTH);
        let fb = powered_flyby(vinf, vinf, alpha, MU_EARTH, rp_min)
            .expect("achievable turn must be feasible");
        assert!(fb.dv_ms < 1e-6, "symmetric flyby ΔV should be 0, got {:.3e}", fb.dv_ms);
        assert!(fb.rp_m >= rp_min, "solved rp {:.3e} below rp_min", fb.rp_m);
    }

    /// A turn larger than the maximum achievable at rp_min is infeasible.
    #[test]
    fn powered_flyby_excessive_turn_is_infeasible() {
        let vinf = 6_000.0;
        let rp_min = 6_678e3;
        let alpha = 1.1 * max_turn_angle_rad(vinf, rp_min, MU_EARTH);
        assert!(powered_flyby(vinf, vinf, alpha, MU_EARTH, rp_min).is_none());
    }

    /// Near-zero turn → periapsis runs to infinity → the burn tends to the
    /// plain excess-speed magnitude difference |v_∞out − v_∞in|.
    #[test]
    fn powered_flyby_zero_turn_matches_speed_difference() {
        let (vin, vout) = (5_000.0, 5_400.0);
        let fb = powered_flyby(vin, vout, 0.0, MU_EARTH, 6_678e3)
            .expect("zero turn is always feasible");
        let err = (fb.dv_ms - (vout - vin)).abs();
        assert!(err < 1.0, "zero-turn ΔV should be ≈ 400 m/s, err = {err:.3} m/s");
    }

    /// The solved periapsis must reproduce the requested turn angle exactly:
    /// arcsin(1/e_in) + arcsin(1/e_out) = α at the returned rp.
    #[test]
    fn powered_flyby_rp_solves_turn_equation() {
        let (vin, vout): (f64, f64) = (7_000.0, 4_500.0);
        let rp_min = 6_678e3;
        let alpha = 0.7 * max_turn_angle_rad(vin.min(vout), rp_min, MU_EARTH);
        let fb = powered_flyby(vin, vout, alpha, MU_EARTH, rp_min).expect("feasible");
        let e_in  = 1.0 + fb.rp_m * vin * vin / MU_EARTH;
        let e_out = 1.0 + fb.rp_m * vout * vout / MU_EARTH;
        let achieved = (1.0_f64 / e_in).asin() + (1.0_f64 / e_out).asin();
        assert!((achieved - alpha).abs() < 1e-9,
            "turn equation residual {:.3e} rad", (achieved - alpha).abs());
    }

    /// Circular coplanar Earth/Mars analogue at exact Hohmann phasing: the
    /// scan's best direct branch must reproduce the analytic Hohmann departure
    /// and arrival excess speeds.
    #[test]
    fn direct_scan_recovers_hohmann_transfer() {
        let a_e = 1.0 * AU;
        let a_m = 1.524 * AU;
        let n_e = (MU_SUN / a_e.powi(3)).sqrt();
        let n_m = (MU_SUN / a_m.powi(3)).sqrt();

        let a_x   = 0.5 * (a_e + a_m);
        let tof_h = std::f64::consts::PI * (a_x.powi(3) / MU_SUN).sqrt();
        // Mars phase angle at departure such that it reaches the antipode of
        // Earth's departure position after exactly the Hohmann half-period.
        let phase0 = std::f64::consts::PI - n_m * tof_h;

        let circ = |a: f64, n: f64, th0: f64| {
            move |t: f64| -> (Vector3<f64>, Vector3<f64>) {
                let th = th0 + n * t;
                let v = (MU_SUN / a).sqrt();
                (
                    Vector3::new(a * th.cos(), a * th.sin(), 0.0),
                    Vector3::new(-v * th.sin(), v * th.cos(), 0.0),
                )
            }
        };
        let earth_state = circ(a_e, n_e, 0.0);
        let mars_state  = circ(a_m, n_m, phase0);

        let bodies = [
            ScanBody { name: "Earth", mu_m3s2: MU_EARTH, min_periapsis_m: 6_678e3, state_at: &earth_state },
            ScanBody { name: "Mars",  mu_m3s2: 4.2828e13, min_periapsis_m: 3_689e3, state_at: &mars_state },
        ];
        // TOF grid straddling the Hohmann TOF (avoid the exact 180° transfer,
        // which Lambert's plane definition rejects as degenerate).
        let tofs: Vec<f64> = (-6..=6).map(|k| tof_h + k as f64 * 5.0 * 86_400.0).collect();
        let cfg = MgaScanConfig {
            mu_sun: MU_SUN,
            departure_epochs_s: vec![0.0],
            leg_tof_grids_s: vec![tofs],
            vinf_dep_max_ms: 10_000.0,
            flyby_dv_max_ms: f64::INFINITY,
            vinf_arr_max_ms: f64::INFINITY,
            max_records: 10_000,
        };
        let out = run_mga_scan(&bodies, &cfg);
        assert!(!out.records.is_empty(), "no feasible direct branch found");

        let best = out.records.iter()
            .min_by(|a, b| a.vinf_dep_ms.partial_cmp(&b.vinf_dep_ms).unwrap())
            .unwrap();

        // Analytic Hohmann excess speeds for these radii:
        // v∞_dep = √(μ(2/a_e − 1/a_x)) − v_circ(a_e) ≈ 2.94 km/s
        // v∞_arr = v_circ(a_m) − √(μ(2/a_m − 1/a_x)) ≈ 2.65 km/s
        let vinf_dep_h = (MU_SUN * (2.0 / a_e - 1.0 / a_x)).sqrt() - (MU_SUN / a_e).sqrt();
        let vinf_arr_h = (MU_SUN / a_m).sqrt() - (MU_SUN * (2.0 / a_m - 1.0 / a_x)).sqrt();
        // Grid points are ±5-day offsets around (but excluding) the exact
        // Hohmann geometry, so agreement is to grid resolution, not exact.
        assert!((best.vinf_dep_ms - vinf_dep_h).abs() < 300.0,
            "departure v∞ {:.1} m/s vs Hohmann {:.1} m/s", best.vinf_dep_ms, vinf_dep_h);
        assert!((best.vinf_arr_ms - vinf_arr_h).abs() < 300.0,
            "arrival v∞ {:.1} m/s vs Hohmann {:.1} m/s", best.vinf_arr_ms, vinf_arr_h);
        assert!(best.flyby_dvs_ms.is_empty(), "direct branch must have no flybys");
    }

    /// Three-body chain on synthetic circular orbits: tightening the per-flyby
    /// ΔV cap must monotonically shrink (never grow) the feasible record set,
    /// and every surviving record must respect the cap and the rp floor.
    #[test]
    fn flyby_dv_cap_prunes_monotonically() {
        let a_e = 1.0 * AU;
        let a_v = 0.723 * AU;
        let a_m = 1.524 * AU;
        let mk = |a: f64, th0: f64| {
            let n = (MU_SUN / a.powi(3)).sqrt();
            move |t: f64| -> (Vector3<f64>, Vector3<f64>) {
                let th = th0 + n * t;
                let v = (MU_SUN / a).sqrt();
                (
                    Vector3::new(a * th.cos(), a * th.sin(), 0.0),
                    Vector3::new(-v * th.sin(), v * th.cos(), 0.0),
                )
            }
        };
        let e_state = mk(a_e, 0.0);
        let v_state = mk(a_v, 1.0);
        let m_state = mk(a_m, 2.5);
        let bodies = [
            ScanBody { name: "Earth", mu_m3s2: MU_EARTH,   min_periapsis_m: 6_678e3, state_at: &e_state },
            ScanBody { name: "Venus", mu_m3s2: 3.2486e14,  min_periapsis_m: 6_352e3, state_at: &v_state },
            ScanBody { name: "Mars",  mu_m3s2: 4.2828e13,  min_periapsis_m: 3_689e3, state_at: &m_state },
        ];
        let day = 86_400.0;
        let dep_grid: Vec<f64> = (0..16).map(|k| k as f64 * 30.0 * day).collect();
        let tof1: Vec<f64> = (3..=14).map(|k| k as f64 * 15.0 * day).collect();
        let tof2: Vec<f64> = (4..=20).map(|k| k as f64 * 15.0 * day).collect();

        let loose_cap = 50_000.0;
        let run = |cap: f64| {
            run_mga_scan(&bodies, &MgaScanConfig {
                mu_sun: MU_SUN,
                departure_epochs_s: dep_grid.clone(),
                leg_tof_grids_s: vec![tof1.clone(), tof2.clone()],
                vinf_dep_max_ms: 15_000.0,
                flyby_dv_max_ms: cap,
                vinf_arr_max_ms: f64::INFINITY,
                max_records: 1_000_000,
            })
        };
        let loose = run(loose_cap);
        let tight = run(1_000.0);
        assert!(!loose.records.is_empty(), "loose cap should find feasible chains");
        assert!(tight.records.len() < loose.records.len(),
            "tighter cap must strictly shrink the record set here");
        for rec in loose.records.iter().chain(tight.records.iter()) {
            assert_eq!(rec.flyby_dvs_ms.len(), 1, "one intermediate flyby expected");
            assert!(rec.flyby_dvs_ms[0] <= loose_cap + 1e-9);
            assert!(rec.flyby_rp_m[0] >= 6_352e3 - 1e-3, "rp below Venus floor");
            assert!((rec.sum_flyby_dv_ms - rec.flyby_dvs_ms[0]).abs() < 1e-9);
        }
        for rec in &tight.records {
            assert!(rec.flyby_dvs_ms[0] <= 1_000.0 + 1e-9);
        }
    }

    /// Parallelizing `run_mga_scan` over departure epochs must not change
    /// what it finds: same total leg-evaluation count and the exact same set
    /// of feasible branches as the sequential reference implementation
    /// (`run_mga_scan_sequential`), with a generous `max_records` so no
    /// global truncation is in play.
    #[test]
    fn parallel_scan_matches_sequential_reference() {
        let a_e = 1.0 * AU;
        let a_v = 0.723 * AU;
        let a_m = 1.524 * AU;
        let mk = |a: f64, th0: f64| {
            let n = (MU_SUN / a.powi(3)).sqrt();
            move |t: f64| -> (Vector3<f64>, Vector3<f64>) {
                let th = th0 + n * t;
                let v = (MU_SUN / a).sqrt();
                (
                    Vector3::new(a * th.cos(), a * th.sin(), 0.0),
                    Vector3::new(-v * th.sin(), v * th.cos(), 0.0),
                )
            }
        };
        let e_state = mk(a_e, 0.0);
        let v_state = mk(a_v, 1.0);
        let m_state = mk(a_m, 2.5);
        let bodies = [
            ScanBody { name: "Earth", mu_m3s2: MU_EARTH,  min_periapsis_m: 6_678e3, state_at: &e_state },
            ScanBody { name: "Venus", mu_m3s2: 3.2486e14, min_periapsis_m: 6_352e3, state_at: &v_state },
            ScanBody { name: "Mars",  mu_m3s2: 4.2828e13, min_periapsis_m: 3_689e3, state_at: &m_state },
        ];
        let day = 86_400.0;
        // 18 departure epochs, modest per-leg TOF grids — enough branching
        // to spread real work across however many threads the test machine
        // has, small enough to stay a fast unit test.
        let dep_grid: Vec<f64> = (0..18).map(|k| k as f64 * 25.0 * day).collect();
        let tof1: Vec<f64> = (3..=10).map(|k| k as f64 * 15.0 * day).collect();
        let tof2: Vec<f64> = (4..=14).map(|k| k as f64 * 15.0 * day).collect();

        let make_cfg = || MgaScanConfig {
            mu_sun: MU_SUN,
            departure_epochs_s: dep_grid.clone(),
            leg_tof_grids_s: vec![tof1.clone(), tof2.clone()],
            vinf_dep_max_ms: 15_000.0,
            flyby_dv_max_ms: 50_000.0,
            vinf_arr_max_ms: f64::INFINITY,
            max_records: 1_000_000, // generous: no global truncation expected
        };

        let par = run_mga_scan(&bodies, &make_cfg());
        let seq = run_mga_scan_sequential(&bodies, &make_cfg());

        assert_eq!(par.n_legs_evaluated, seq.n_legs_evaluated,
            "leg-evaluation count must be identical regardless of thread distribution");
        assert_eq!(par.n_records_dropped, 0, "generous max_records should drop nothing");
        assert_eq!(seq.n_records_dropped, 0, "generous max_records should drop nothing (sequential)");
        assert_eq!(par.records.len(), seq.records.len(), "same total feasible branch count expected");
        assert!(!par.records.is_empty(), "test setup should find at least some feasible chains");

        // Sort both by (dep_epoch, leg_tofs, vinf_dep) — a stable tie-break
        // in case a (dep_epoch, tofs) pair has both a prograde and a
        // retrograde surviving solution — then compare field-by-field within
        // a tight float tolerance (both paths run the identical
        // Lambert/powered-flyby math, just in a different evaluation order).
        let key = |r: &ScanRecord| {
            (r.dep_epoch_s, r.leg_tofs_s.clone(), (r.vinf_dep_ms * 1e6).round() as i64)
        };
        let mut par_sorted = par.records.clone();
        let mut seq_sorted = seq.records.clone();
        par_sorted.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
        seq_sorted.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
        for (p, s) in par_sorted.iter().zip(seq_sorted.iter()) {
            assert!((p.dep_epoch_s - s.dep_epoch_s).abs() < 1e-6);
            assert_eq!(p.leg_tofs_s.len(), s.leg_tofs_s.len());
            for (pt, st) in p.leg_tofs_s.iter().zip(s.leg_tofs_s.iter()) {
                assert!((pt - st).abs() < 1e-6);
            }
            assert!((p.vinf_dep_ms - s.vinf_dep_ms).abs() < 1e-6);
            assert!((p.vinf_arr_ms - s.vinf_arr_ms).abs() < 1e-6);
            assert!((p.sum_flyby_dv_ms - s.sum_flyby_dv_ms).abs() < 1e-6);
        }
    }

    /// Global-cap reconciliation: however work is distributed across worker
    /// threads (each capping its own local accumulator at `max_records`
    /// independently, per the parallel design), `records.len() +
    /// n_records_dropped` must equal the TRUE total number of feasible
    /// branches — independent of `max_records` — verified against the same
    /// scan run with a deliberately generous cap.
    #[test]
    fn parallel_scan_reconciles_dropped_count_under_tight_cap() {
        let a_e = 1.0 * AU;
        let a_v = 0.723 * AU;
        let a_m = 1.524 * AU;
        let mk = |a: f64, th0: f64| {
            let n = (MU_SUN / a.powi(3)).sqrt();
            move |t: f64| -> (Vector3<f64>, Vector3<f64>) {
                let th = th0 + n * t;
                let v = (MU_SUN / a).sqrt();
                (
                    Vector3::new(a * th.cos(), a * th.sin(), 0.0),
                    Vector3::new(-v * th.sin(), v * th.cos(), 0.0),
                )
            }
        };
        let e_state = mk(a_e, 0.0);
        let v_state = mk(a_v, 1.0);
        let m_state = mk(a_m, 2.5);
        let bodies = [
            ScanBody { name: "Earth", mu_m3s2: MU_EARTH,  min_periapsis_m: 6_678e3, state_at: &e_state },
            ScanBody { name: "Venus", mu_m3s2: 3.2486e14, min_periapsis_m: 6_352e3, state_at: &v_state },
            ScanBody { name: "Mars",  mu_m3s2: 4.2828e13, min_periapsis_m: 3_689e3, state_at: &m_state },
        ];
        let day = 86_400.0;
        let dep_grid: Vec<f64> = (0..24).map(|k| k as f64 * 20.0 * day).collect();
        let tof1: Vec<f64> = (3..=12).map(|k| k as f64 * 15.0 * day).collect();
        let tof2: Vec<f64> = (4..=16).map(|k| k as f64 * 15.0 * day).collect();

        let make_cfg = |max_records: usize| MgaScanConfig {
            mu_sun: MU_SUN,
            departure_epochs_s: dep_grid.clone(),
            leg_tof_grids_s: vec![tof1.clone(), tof2.clone()],
            vinf_dep_max_ms: 15_000.0,
            flyby_dv_max_ms: 50_000.0,
            vinf_arr_max_ms: f64::INFINITY,
            max_records,
        };

        // Reference: generous cap, no truncation, gives the TRUE total
        // feasible branch count.
        let reference = run_mga_scan(&bodies, &make_cfg(1_000_000));
        let total_feasible = reference.records.len() as u64 + reference.n_records_dropped;
        assert_eq!(reference.n_records_dropped, 0);
        assert!(total_feasible > 20, "test setup should find a healthy number of feasible chains");

        // Tight cap, deliberately small enough that overflow is likely
        // across however many worker threads run this, and definitely once
        // all workers' local records are merged.
        let tight_cap = (total_feasible / 4).max(1) as usize;
        let capped = run_mga_scan(&bodies, &make_cfg(tight_cap));

        assert!(capped.records.len() <= tight_cap, "global cap must be honored after merge");
        assert_eq!(
            capped.records.len() as u64 + capped.n_records_dropped,
            total_feasible,
            "kept + dropped must reconcile to the true total feasible count regardless of cap"
        );
        assert_eq!(
            capped.n_legs_evaluated, reference.n_legs_evaluated,
            "leg-evaluation count is independent of max_records"
        );
    }
}
