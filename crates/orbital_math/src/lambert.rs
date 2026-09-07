//! Universal-variable Lambert solver (Battin / Curtis §5.3).
//!
//! Finds the velocity vectors at the departure and arrival positions for a
//! given time-of-flight.  Handles both elliptic (z > 0) and hyperbolic
//! (z < 0) transfers; the exactly-parabolic boundary (z = 0) is covered by
//! the Stumpff series expansions.
//!
//! # Usage
//! ```ignore
//! let solutions = lambert(r1, r2, tof_s, true);   // prograde
//! for (v1, v2) in solutions { ... }
//! ```

/// 3-component position / velocity vector in SI units.
pub type V3 = [f64; 3];

// ── Stumpff functions ─────────────────────────────────────────────────────────

/// Stumpff C function (also called c₂): `C(z) = (1 − cos√z)/z` for z > 0,
/// extended analytically to z < 0 and z ≈ 0.
/// Used in the universal-variable Lambert and Kepler propagator formulations.
/// Reference: Battin (1999) §4.4; Vallado (2013) §2.2.
pub(crate) fn stumpff_c(z: f64) -> f64 {
    if z > 1e-6 {
        // Half-angle form 1 − cos s = 2·sin²(s/2): algebraically identical,
        // but keeps full precision as s → 2π (resonant-return Lambert legs,
        // Phase 9w), where the direct 1 − cos s loses ~8 digits.
        let s = z.sqrt();
        let h = (0.5 * s).sin();
        2.0 * h * h / z
    } else if z < -1e-6 {
        let q = (-z).sqrt();
        let h = (0.5 * q).sinh();
        2.0 * h * h / (-z)
    } else {
        0.5
    }
}

/// Stumpff S function (also called c₃): `S(z) = (√z − sin√z)/(√z)³` for z > 0,
/// extended analytically to z < 0 and z ≈ 0.
/// Reference: Battin (1999) §4.4; Vallado (2013) §2.2.
/// `1 − z·S(z)`, computed without cancellation.
///
/// Algebraic identity: `z·S(z) = (√z − sin√z)/√z = 1 − sin(√z)/√z`, so
/// `1 − z·S(z) = sin(√z)/√z` exactly (and `sinh(√−z)/√−z` for z < 0).
/// The direct form differences two O(1) numbers and loses all significance
/// as √z → 2π — the resonant-return corner (Phase 9w) lives exactly there.
pub(crate) fn one_minus_z_s(z: f64) -> f64 {
    if z > 1e-6 {
        let s = z.sqrt();
        s.sin() / s
    } else if z < -1e-6 {
        let q = (-z).sqrt();
        q.sinh() / q
    } else {
        1.0 - z / 6.0
    }
}

pub(crate) fn stumpff_s(z: f64) -> f64 {
    if z > 1e-6 {
        let sq = z.sqrt();
        (sq - sq.sin()) / sq.powi(3)
    } else if z < -1e-6 {
        let sq = (-z).sqrt();
        (sq.sinh() - sq) / sq.powi(3)
    } else {
        1.0 / 6.0
    }
}

// ── V3 helpers (private) ──────────────────────────────────────────────────────

#[inline] fn dot(a: V3, b: V3) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
#[inline] fn norm(a: V3) -> f64 { dot(a, a).sqrt() }
#[inline] fn sub(a: V3, b: V3) -> V3 { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
#[inline] fn add(a: V3, b: V3) -> V3 { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
#[inline] fn scale(a: V3, s: f64) -> V3 { [a[0]*s, a[1]*s, a[2]*s] }

/// Reject transfer angles within ~3° of 0° or 180°.
///
/// Near these angles the transfer plane (defined by `r1 x r2`) is nearly
/// degenerate — Vallado §5.3 notes single-revolution Lambert solutions are
/// ill-conditioned there. The old threshold (`1e-9`, ~180.00006°) only
/// caught the exact mathematical singularity, not the surrounding band where
/// `dnu`'s `acos` round-trip has already lost enough precision to produce a
/// numerically garbage (but not literally infinite) root. Confirmed
/// empirically: an Earth-Jupiter porkchop scan returned C3 = 2245 km²/s² at
/// a transfer angle of ~180.1°, next to a clean ~100 km²/s² neighborhood
/// just a few degrees away — see `MissionPlanner/config/jupiter_flyby.toml`.
const MIN_SIN_TRANSFER_ANGLE: f64 = 0.05;

/// Transfer angle (Δν) from `r1` to `r2` in the chosen direction, in `[0, 2π)`.
///
/// `prograde = true` treats +z as the direction of motion (the standard
/// convention for near-ecliptic heliocentric transfers, where +z is
/// ecliptic north). Shared by [`lambert_with_min_transfer_angle`] and the
/// VILM boundary-value solver (`trajectory_solver::vilm`) — same geometric
/// convention, single source rather than two independent implementations.
pub fn transfer_angle_rad(r1: V3, r2: V3, prograde: bool) -> f64 {
    let mag_r1 = norm(r1);
    let mag_r2 = norm(r2);
    let cos_dnu = (dot(r1, r2) / (mag_r1 * mag_r2)).clamp(-1.0, 1.0);
    let mut dnu = cos_dnu.acos();

    let cross_z = r1[0] * r2[1] - r1[1] * r2[0];
    if cross_z < 0.0 {
        if prograde { dnu = 2.0 * std::f64::consts::PI - dnu; }
    } else if !prograde {
        dnu = 2.0 * std::f64::consts::PI - dnu;
    }
    dnu
}

// ── Public API ────────────────────────────────────────────────────────────────

/// All single-revolution Lambert solutions for the given geometry.
///
/// `r1`, `r2` — departure / arrival position vectors [m]
/// `tof`      — time of flight [s]
/// `prograde` — `true` selects short-arc (Δν < π), `false` selects long-arc.
///
/// Returns a list of `(v1, v2)` velocity pairs [m/s].  The list is usually
/// length 0 (degenerate geometry) or 1 (one elliptic solution found).
///
/// The central-body gravitational parameter `mu` [m³/s²] is taken at the
/// call site by normalising `tof`: pass `tof * sqrt(mu)` for a unit-mu
/// formulation, or use SI units throughout.
pub fn lambert(r1: V3, r2: V3, tof: f64, prograde: bool, mu: f64) -> Vec<(V3, V3)> {
    lambert_with_min_transfer_angle(r1, r2, tof, prograde, mu, MIN_SIN_TRANSFER_ANGLE)
}

/// [`lambert`] with a caller-controlled near-0°/360° transfer-angle guard.
///
/// The two degenerate bands are physically different and are guarded
/// differently here:
/// - **Near 180°** the transfer plane (`r1 × r2`) genuinely flips sign and
///   solutions go numerically wild (the empirical C3 = 2245 km²/s² case in
///   [`MIN_SIN_TRANSFER_ANGLE`]'s comment) — that band always keeps the
///   0.05 guard regardless of `min_sin_dnu`.
/// - **Near 0°/360°** the geometry is a resonant return (same-point
///   re-encounter after the body completes full revolutions — Cassini's
///   Venus→Venus 1:2 pair, VEEGA's Earth→Earth hops). The solution family
///   degrades gracefully toward the resonant orbit rather than diverging,
///   so ballistic MGA work (Phase 9w) must be able to relax this band.
///   `min_sin_dnu` sets the rejection threshold there; the exactly-parallel
///   singularity itself is still rejected via the `1 - cos Δν` guard.
pub fn lambert_with_min_transfer_angle(
    r1: V3, r2: V3, tof: f64, prograde: bool, mu: f64,
    min_sin_dnu: f64,
) -> Vec<(V3, V3)> {
    let mag_r1 = norm(r1);
    let mag_r2 = norm(r2);
    // Raw dot-product cosine (not round-tripped through the acos/cos in
    // transfer_angle_rad below) — preserves the exact original numerics
    // this function's tight tolerances depend on.
    let cos_dnu = (dot(r1, r2) / (mag_r1 * mag_r2)).clamp(-1.0, 1.0);
    let dnu = transfer_angle_rad(r1, r2, prograde);
    let sin_dnu = dnu.sin();
    if cos_dnu < 0.0 {
        // Near-180° band: always the empirical 0.05 guard (see above).
        if sin_dnu.abs() < MIN_SIN_TRANSFER_ANGLE { return Vec::new(); }
    } else if sin_dnu.abs() < min_sin_dnu {
        return Vec::new();
    }

    let one_minus_cos = (1.0 - cos_dnu).abs();
    if one_minus_cos < 1e-10 { return Vec::new(); }
    let a_param = sin_dnu * (mag_r1 * mag_r2 / one_minus_cos).sqrt();

    let y_of_z = |z: f64| -> f64 {
        let cz = stumpff_c(z);
        if cz > 1e-12 {
            mag_r1 + mag_r2 - a_param * one_minus_z_s(z) / cz.sqrt()
        } else {
            f64::NAN
        }
    };

    let tof_of_z = |z: f64| -> f64 {
        let y = y_of_z(z);
        if y.is_nan() || y < 0.0 { return f64::NAN; }
        let c = stumpff_c(z);
        if c < 1e-12 { return f64::NAN; }
        let t = (y / c).powf(1.5) * stumpff_s(z) / mu.sqrt()
              + a_param * y.sqrt() / mu.sqrt();
        if t <= 0.0 { f64::NAN } else { t }
    };

    let residual = |z: f64| -> f64 { tof_of_z(z) - tof };

    // Scan for sign-change brackets in the elliptic region (z > 0).
    let z_lo = 1e-3_f64;
    let z_hi = (2.0 * std::f64::consts::PI).powi(2) - 1e-4;
    let n_scan = 1_000usize;

    let mut brackets: Vec<(f64, f64)> = Vec::new();
    let mut tof_min_seen = f64::INFINITY;
    let mut tof_max_seen = f64::NEG_INFINITY;
    {
        let mut z_prev = f64::NAN;
        let mut r_prev = f64::NAN;
        for k in 0..n_scan {
            let z = z_lo + (z_hi - z_lo) * k as f64 / (n_scan - 1) as f64;
            let t_here = tof_of_z(z);
            if t_here.is_finite() {
                tof_min_seen = tof_min_seen.min(t_here);
                tof_max_seen = tof_max_seen.max(t_here);
            }
            let r = residual(z);
            if r.is_finite() {
                if r_prev.is_finite() && r_prev * r < 0.0 {
                    brackets.push((z_prev, z));
                }
                z_prev = z; r_prev = r;
            }
        }
    }
    // Fine geometric tail scan as z → (2π)², the full-revolution limit —
    // resonant-return geometries (Δν → 2π: Cassini's Venus→Venus 1:2 pair,
    // VEEGA's Earth→Earth hops) put the root within the last ~0.01 of z,
    // far sharper than the uniform scan spacing above can resolve. Under
    // the default transfer-angle guard these geometries are rejected before
    // reaching this scan at all (sin Δν < 0.05 there), so this refinement
    // only ever activates for relaxed-guard callers
    // (`lambert_with_min_transfer_angle`) and cannot change existing results.
    {
        let z_top = (2.0 * std::f64::consts::PI).powi(2);
        let mut z_prev = f64::NAN;
        let mut r_prev = f64::NAN;
        let mut gap = 0.05_f64;
        while gap > 1e-9 {
            let z = z_top - gap;
            let r = residual(z);
            if r.is_finite() {
                if r_prev.is_finite() && r_prev * r < 0.0 {
                    brackets.push((z_prev, z));
                }
                z_prev = z;
                r_prev = r;
            }
            gap *= 0.7;
        }
    }

    // Hyperbolic branch (z < 0). T(z) is monotone increasing through the
    // parabolic boundary (Curtis §5.3, Battin §7.4), so when the requested
    // TOF is shorter than the near-parabolic TOF there is exactly one root
    // at negative z — e.g. a post-gravity-assist Jupiter→Neptune leg, which
    // is heliocentric hyperbolic (Voyager 2 was). Bracket it by geometric
    // expansion; T → 0⁺ as y(z) → 0⁺, so if y goes negative (NaN residual)
    // the root lies between the last feasible point and there — shrink back.
    let r_parabolic = residual(0.0);
    if r_parabolic.is_finite() && r_parabolic > 0.0 {
        let mut za = 0.0_f64;
        let mut zb = -1.0_f64;
        for _ in 0..64 {
            let rb = residual(zb);
            if rb.is_finite() {
                if rb < 0.0 { brackets.push((za, zb)); break; }
                za = zb;
                zb *= 2.0;
                // cosh(√-z) overflows f64 near z ≈ -5e5; TOF there is
                // already astronomically small, so nothing physical is lost.
                if zb < -2.0e5 { break; }
            } else {
                zb = 0.5 * (za + zb);
            }
        }
    }

    if std::env::var("LAMBERT_DEBUG").is_ok() && brackets.is_empty() {
        eprintln!(
            "[LAMBERT_DEBUG] no bracket: requested_tof={:.1}d elliptic_scan_range=[{:.1}d, {:.1}d] dnu_deg={:.2} prograde={}",
            tof/86400.0, tof_min_seen/86400.0, tof_max_seen/86400.0, dnu.to_degrees(), prograde,
        );
    }

    // Bisect each bracket to get root, then convert to velocity vectors.
    let mut solutions: Vec<(V3, V3)> = Vec::new();
    for (mut za, mut zb) in brackets {
        for _ in 0..60 {
            let zm = 0.5 * (za + zb);
            let rm = residual(zm);
            if !rm.is_finite() { break; }
            if (zb - za).abs() < 1e-10 || rm.abs() * mu.sqrt() < 1.0 { za = zm; break; }
            if residual(za) * rm < 0.0 { zb = zm; } else { za = zm; }
        }
        let y = y_of_z(za);
        if y <= 0.0 || y.is_nan() { continue; }

        let g = a_param * (y / mu).sqrt();
        if g.abs() < 1e-12 { continue; }

        // Lagrange-coefficient velocities, factored through Δr = r2 − r1:
        //   v1 = (r2 − f·r1)/g      with f    = 1 − y/|r1|
        //      = (Δr + (y/|r1|)·r1)/g
        //   v2 = (ġ·r2 − r1)/g      with ġ    = 1 − y/|r2|
        //      = (Δr − (y/|r2|)·r2)/g
        // Algebraically identical to the direct f/ġ form, but avoids the
        // catastrophic cancellation in r2 − f·r1 when f → 1 and r2 ≈ r1 —
        // the resonant-return geometry (Δν → 2π, Phase 9w), where the
        // direct form produced ~5×10⁷ km propagation misses.
        let dr = sub(r2, r1);
        let v1 = scale(add(dr, scale(r1, y / mag_r1)), 1.0 / g);
        let v2 = scale(add(dr, scale(r2, -y / mag_r2)), 1.0 / g);
        solutions.push((v1, v2));
    }
    solutions
}

/// Minimum-ΔV solution from all Lambert arcs (prograde + retrograde).
///
/// Returns `(v1, v2)` for the transfer with lowest total ΔV relative to the
/// given departure velocity `v_dep` and arrival velocity `v_arr`.
pub fn lambert_min_dv(
    r1: V3, r2: V3, tof: f64, mu: f64,
    v_dep: V3, v_arr: V3,
) -> Option<(V3, V3)> {
    let mut best: Option<(f64, V3, V3)> = None;
    for prograde in [true, false] {
        for (v1, v2) in lambert(r1, r2, tof, prograde, mu) {
            let dv = norm(sub(v1, v_dep)) + norm(sub(v2, v_arr));
            if best.map_or(true, |(b, _, _)| dv < b) {
                best = Some((dv, v1, v2));
            }
        }
    }
    best.map(|(_, v1, v2)| (v1, v2))
}

/// Multi-revolution Lambert: like [`lambert_with_min_transfer_angle`] but
/// also searches solutions where the transfer orbit completes `n_rev`
/// additional full revolutions before arrival — required for genuine
/// resonant-return legs (e.g. VEEGA's ~2-year Earth→Earth leg, Cassini-class
/// Venus→Venus resonant returns) where the real trajectory wraps around the
/// central body more than once. Every OTHER Lambert-based tool in this
/// codebase (`lambert`, `lambert_min_dv`, and everything built on them —
/// Hohmann/porkchop/GridSearch/Sims-Flanagan/`mga_scan`) remains N=0-only;
/// this function is additive, not a replacement.
///
/// `n_rev=0` delegates to [`lambert_with_min_transfer_angle`] exactly
/// (identical results, zero behavioural change for existing callers).
///
/// For `n_rev>=1`: only the elliptic branch applies (a multi-revolution
/// transfer orbit must be periodic/closed — hyperbolic orbits have no
/// period). The key fact that makes this simple: the universal anomaly
/// `χ` (and `z = χ²/a`) ALREADY encodes the full swept angle including any
/// number of extra revolutions — the time-of-flight formula itself is
/// unchanged, only the z-DOMAIN searched changes. The Stumpff C-function
/// `C(z) = (1-cos√z)/z` has zeros at `z = (2πN)²` for every integer N ≥ 1,
/// which — confirmed numerically — cleanly partitions the z-axis into
/// disjoint bands `((2πN)², (2π(N+1))²)`, one per revolution count: N=0's
/// existing solver already scans the first band `(0, (2π)²)`; this function
/// scans the SAME (unmodified) `y(z)`/time formula within the N-th band
/// instead. (An earlier version of this function tried adding an explicit
/// `+ N · period` term on top of the N=0 formula instead of changing the
/// scan domain — that double-counts the swept angle and was caught by this
/// function's own round-trip test returning a solution 40 km/s off the
/// true departure velocity; reverted in favour of the domain-shift approach
/// derived here, which passes the same test to < 1 m/s.) Each band is
/// generically still U-shaped (a minimum-TOF point partway through), so for
/// any requested `tof` above that minimum there are generically TWO roots
/// (a "low-a"/faster and "high-a"/slower branch) — both are returned when
/// found, unlike N=0's usual single root.
///
/// # References
/// - Curtis (2013), *Orbital Mechanics for Engineering Students*, 3rd ed.,
///   §5.3 (multiple-revolution Lambert extension).
/// - Battin (1999), §6.9 (multi-revolution transfer time equation).
pub fn lambert_n_rev(
    r1: V3, r2: V3, tof: f64, prograde: bool, mu: f64, n_rev: u32, min_sin_dnu: f64,
) -> Vec<(V3, V3)> {
    if n_rev == 0 {
        return lambert_with_min_transfer_angle(r1, r2, tof, prograde, mu, min_sin_dnu);
    }

    let mag_r1 = norm(r1);
    let mag_r2 = norm(r2);
    let cos_dnu = (dot(r1, r2) / (mag_r1 * mag_r2)).clamp(-1.0, 1.0);
    let mut dnu = cos_dnu.acos();

    let cross_z = r1[0] * r2[1] - r1[1] * r2[0];
    if cross_z < 0.0 {
        if prograde { dnu = 2.0 * std::f64::consts::PI - dnu; }
    } else if !prograde {
        dnu = 2.0 * std::f64::consts::PI - dnu;
    }

    let sin_dnu = dnu.sin();
    if cos_dnu < 0.0 {
        if sin_dnu.abs() < MIN_SIN_TRANSFER_ANGLE { return Vec::new(); }
    } else if sin_dnu.abs() < min_sin_dnu {
        return Vec::new();
    }

    let one_minus_cos = (1.0 - cos_dnu).abs();
    if one_minus_cos < 1e-10 { return Vec::new(); }
    let a_param = sin_dnu * (mag_r1 * mag_r2 / one_minus_cos).sqrt();

    let y_of_z = |z: f64| -> f64 {
        let cz = stumpff_c(z);
        if cz > 1e-12 {
            mag_r1 + mag_r2 - a_param * one_minus_z_s(z) / cz.sqrt()
        } else {
            f64::NAN
        }
    };

    let tof_n_of_z = |z: f64| -> f64 {
        let y = y_of_z(z);
        if y.is_nan() || y < 0.0 { return f64::NAN; }
        let c = stumpff_c(z);
        if c < 1e-12 { return f64::NAN; }
        let t = (y / c).powf(1.5) * stumpff_s(z) / mu.sqrt()
              + a_param * y.sqrt() / mu.sqrt();
        if t <= 0.0 { f64::NAN } else { t }
    };

    let residual = |z: f64| -> f64 { tof_n_of_z(z) - tof };

    // The N-th revolution band: (2*pi*N)^2 < z < (2*pi*(N+1))^2, both ends
    // are genuine C(z)=0 singularities (confirmed numerically) so stay a
    // small epsilon inside them, matching the N=0 solver's own z_lo/z_hi
    // margins around its band's 0 and (2*pi)^2 endpoints.
    let two_pi = 2.0 * std::f64::consts::PI;
    let z_lo = (two_pi * n_rev as f64).powi(2) + 1e-3;
    let z_hi = (two_pi * (n_rev as f64 + 1.0)).powi(2) - 1e-3;
    let n_scan = 2_000usize; // finer than N=0's 1000: higher bands are wider in z

    let mut brackets: Vec<(f64, f64)> = Vec::new();
    {
        let mut z_prev = f64::NAN;
        let mut r_prev = f64::NAN;
        for k in 0..n_scan {
            let z = z_lo + (z_hi - z_lo) * k as f64 / (n_scan - 1) as f64;
            let r = residual(z);
            if r.is_finite() {
                if r_prev.is_finite() && r_prev * r < 0.0 {
                    brackets.push((z_prev, z));
                }
                z_prev = z; r_prev = r;
            }
        }
    }

    if std::env::var("LAMBERT_DEBUG").is_ok() && brackets.is_empty() {
        eprintln!(
            "[LAMBERT_DEBUG n_rev={n_rev}] no bracket: requested_tof={:.1}d dnu_deg={:.2} prograde={}",
            tof/86400.0, dnu.to_degrees(), prograde,
        );
    }

    let mut solutions: Vec<(V3, V3)> = Vec::new();
    for (mut za, mut zb) in brackets {
        for _ in 0..60 {
            let zm = 0.5 * (za + zb);
            let rm = residual(zm);
            if !rm.is_finite() { break; }
            if (zb - za).abs() < 1e-10 || rm.abs() * mu.sqrt() < 1.0 { za = zm; break; }
            if residual(za) * rm < 0.0 { zb = zm; } else { za = zm; }
        }
        let y = y_of_z(za);
        if y <= 0.0 || y.is_nan() { continue; }

        let g = a_param * (y / mu).sqrt();
        if g.abs() < 1e-12 { continue; }

        let dr = sub(r2, r1);
        let v1 = scale(add(dr, scale(r1, y / mag_r1)), 1.0 / g);
        let v2 = scale(add(dr, scale(r2, -y / mag_r2)), 1.0 / g);
        solutions.push((v1, v2));
    }
    solutions
}

/// Minimum-ΔV Lambert solution at EXACTLY `n_rev` extra revolutions
/// (prograde + retrograde searched; the revolution count itself is NOT
/// searched). Added for the N-as-chromosome-gene MGA fix: the
/// greedy branch selection in [`lambert_min_dv_multi_rev`] (min leg-LOCAL
/// ΔV over all N) was found to wreck fixed-budget DE searches — it makes
/// the fitness discontinuous in the chromosome (a tiny TOF/η change flips
/// which N wins, teleporting the downstream v∞ geometry) and judges the
/// branch by one leg's cost when its real consequences land on every
/// DOWNSTREAM leg (confirmed on GTOP Cassini-2: 15.5–19.2 km/s across all
/// seeds greedy, vs. 5.4–5.9 km/s with the branch dimension removed — see
/// the design notes Phase 9x-iv). With `n_rev` supplied by the caller (an explicit
/// optimizer gene), each chromosome evaluates ONE smooth trajectory family
/// and the branch choice is judged by full-mission fitness instead.
pub fn lambert_min_dv_at_n_rev(
    r1: V3, r2: V3, tof: f64, mu: f64,
    v_dep: V3, v_arr: V3, n_rev: u32,
) -> Option<(V3, V3)> {
    let mut best: Option<(f64, V3, V3)> = None;
    for prograde in [true, false] {
        for (v1, v2) in lambert_n_rev(r1, r2, tof, prograde, mu, n_rev, MIN_SIN_TRANSFER_ANGLE) {
            let dv = norm(sub(v1, v_dep)) + norm(sub(v2, v_arr));
            if best.map_or(true, |(b, ..)| dv < b) {
                best = Some((dv, v1, v2));
            }
        }
    }
    best.map(|(_, v1, v2)| (v1, v2))
}

/// Minimum-ΔV solution searching `n_rev = 0..=max_n_rev` (both direct-arc
/// N=0 and genuine multi-revolution branches) and both prograde/retrograde
/// — the multi-revolution-aware counterpart of [`lambert_min_dv`]. Returns
/// `(v1, v2, n_rev)` so the caller can tell when a multi-rev branch won.
///
/// ⚠ Do NOT use this inside an optimizer's fitness loop — the greedy
/// leg-local branch pick is exactly what regressed Cassini-2 (see
/// [`lambert_min_dv_at_n_rev`]'s doc comment). It remains for diagnostic
/// tools (`galileo_*` binaries, standalone leg probes) where a single
/// point is being inspected, not a landscape searched.
pub fn lambert_min_dv_multi_rev(
    r1: V3, r2: V3, tof: f64, mu: f64,
    v_dep: V3, v_arr: V3, max_n_rev: u32,
) -> Option<(V3, V3, u32)> {
    let mut best: Option<(f64, V3, V3, u32)> = None;
    for n_rev in 0..=max_n_rev {
        for prograde in [true, false] {
            for (v1, v2) in lambert_n_rev(r1, r2, tof, prograde, mu, n_rev, MIN_SIN_TRANSFER_ANGLE) {
                let dv = norm(sub(v1, v_dep)) + norm(sub(v2, v_arr));
                if best.map_or(true, |(b, ..)| dv < b) {
                    best = Some((dv, v1, v2, n_rev));
                }
            }
        }
    }
    best.map(|(_, v1, v2, n_rev)| (v1, v2, n_rev))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kepler::propagate_kepler;
    use nalgebra::Vector3;

    /// Sun's gravitational parameter [m³/s²] — JPL DE430
    const MU_SUN: f64 = 1.327_124_400_18e20;

    fn specific_energy(r: V3, v: V3, mu: f64) -> f64 {
        0.5 * dot(v, v) - mu / norm(r)
    }

    /// Round-trip on a known hyperbolic orbit: propagate a heliocentric
    /// hyperbola with the universal-variable Kepler propagator, then ask
    /// Lambert to recover the departure velocity from the endpoint positions
    /// and TOF alone.
    #[test]
    fn hyperbolic_round_trip_matches_kepler_propagation() {
        // Jupiter-distance departure at 1.2× local escape speed — a
        // post-gravity-assist heliocentric hyperbola.
        let r0: Vector3<f64> = Vector3::new(7.78e11, 0.0, 0.0); // 5.2 AU
        let v_esc = (2.0 * MU_SUN / r0.norm()).sqrt();
        let v0 = Vector3::new(2.0e3, 1.2 * v_esc, 1.0e3);
        assert!(specific_energy([r0.x, r0.y, r0.z], [v0.x, v0.y, v0.z], MU_SUN) > 0.0,
            "test orbit must be hyperbolic");

        let tof = 3000.0 * 86_400.0;
        let (r1, v1) = propagate_kepler(r0, v0, tof, MU_SUN).expect("propagation failed");

        let sols = lambert([r0.x, r0.y, r0.z], [r1.x, r1.y, r1.z], tof, true, MU_SUN);
        assert!(!sols.is_empty(), "no Lambert solution on a known hyperbolic transfer");

        let (v1_sol, v2_sol) = sols.iter()
            .min_by(|a, b| {
                let ea = norm(sub(a.0, [v0.x, v0.y, v0.z]));
                let eb = norm(sub(b.0, [v0.x, v0.y, v0.z]));
                ea.partial_cmp(&eb).unwrap()
            })
            .copied().unwrap();
        let err_v1 = norm(sub(v1_sol, [v0.x, v0.y, v0.z]));
        let err_v2 = norm(sub(v2_sol, [v1.x, v1.y, v1.z]));
        assert!(err_v1 < 0.5, "departure velocity error {:.3e} m/s", err_v1);
        assert!(err_v2 < 0.5, "arrival velocity error {:.3e} m/s", err_v2);
    }

    /// Real-geometry Voyager-2-class leg: Jupiter's position at the Voyager 2
    /// flyby epoch (1979-07-09) to Neptune's position at the Voyager 2
    /// arrival epoch (1989-08-25), TOF = 3700.0 days exactly.
    ///
    /// State vectors from JPL Horizons (heliocentric, ecliptic J2000,
    /// queried). A ballistic Jupiter→Neptune leg this fast is
    /// heliocentric hyperbolic — the elliptic-only solver returned nothing
    /// here. Verified by propagating the returned departure state forward
    /// with the universal Kepler propagator and checking it hits Neptune.
    #[test]
    fn voyager2_class_jupiter_to_neptune_leg_is_hyperbolic() {
        // Jupiter, 1979-07-09 00:00 TDB [m]
        let r1 = [-5.881826407390529e11, 5.382726227896252e11, 1.096298777704048e10];
        // Neptune, 1989-08-25 00:00 TDB [m]
        let r2 = [8.975581634402157e11, -4.429309408005850e12, 7.053007037235570e10];
        let tof = 3700.0 * 86_400.0;

        let sols = lambert(r1, r2, tof, true, MU_SUN);
        assert!(!sols.is_empty(), "hyperbolic Jupiter→Neptune leg not found");

        let (v1, _v2) = sols[0];
        let eps = specific_energy(r1, v1, MU_SUN);
        assert!(eps > 0.0, "transfer should be heliocentric hyperbolic, eps = {:.3e}", eps);

        // Independent check: propagate the Lambert departure state forward
        // the full TOF and confirm it arrives at Neptune's position.
        let (r_end, _) = propagate_kepler(
            Vector3::new(r1[0], r1[1], r1[2]),
            Vector3::new(v1[0], v1[1], v1[2]),
            tof, MU_SUN,
        ).expect("propagation failed");
        let miss = (r_end - Vector3::new(r2[0], r2[1], r2[2])).norm();
        assert!(miss < 1.0e6, "arrival miss {:.3e} m (> 1000 km)", miss);
    }

    /// Regression: the elliptic branch must be unaffected — a Hohmann-class
    /// Earth→Mars TOF still returns a bound (negative-energy) transfer.
    #[test]
    fn elliptic_branch_unaffected() {
        let r1 = [1.496e11, 0.0, 0.0];
        let ang = 140.0_f64.to_radians();
        let r2 = [2.279e11 * ang.cos(), 2.279e11 * ang.sin(), 0.0]; // Mars distance, 140° ahead
        let tof = 260.0 * 86_400.0;

        let sols = lambert(r1, r2, tof, true, MU_SUN);
        assert!(!sols.is_empty(), "elliptic Earth→Mars transfer not found");
        let (v1, _) = sols[0];
        let eps = specific_energy(r1, v1, MU_SUN);
        assert!(eps < 0.0, "Hohmann-class transfer should be elliptic, eps = {:.3e}", eps);
    }

    /// `lambert_n_rev(n_rev=0)` must be byte-identical in behaviour to
    /// `lambert_with_min_transfer_angle` — the zero-regression-risk guarantee
    /// the doc comment promises for every existing N=0 caller.
    #[test]
    fn n_rev_zero_matches_existing_solver_exactly() {
        let r1 = [1.496e11, 0.0, 0.0];
        let ang = 140.0_f64.to_radians();
        let r2 = [2.279e11 * ang.cos(), 2.279e11 * ang.sin(), 0.0];
        let tof = 260.0 * 86_400.0;

        let baseline = lambert(r1, r2, tof, true, MU_SUN);
        let via_n_rev = lambert_n_rev(r1, r2, tof, true, MU_SUN, 0, MIN_SIN_TRANSFER_ANGLE);
        assert_eq!(baseline.len(), via_n_rev.len());
        for ((v1a, v2a), (v1b, v2b)) in baseline.iter().zip(via_n_rev.iter()) {
            assert_eq!(*v1a, *v1b);
            assert_eq!(*v2a, *v2b);
        }
    }

    /// Round-trip on a genuine multi-revolution (N=2) resonant-return
    /// geometry, directly modelling the real VEEGA Earth→Earth leg case
    /// that motivated this function (Phase 9x-iv): endpoints
    /// close together in angle (a near-circular orbit's start/end after
    /// completing whole extra revolutions), TOF long enough to wrap the Sun
    /// twice. The N=0 solver cannot represent this at all (near-0° transfer
    /// angle at a multi-year TOF has no single-revolution solution); the
    /// N=2 branch must recover the known departure velocity.
    #[test]
    fn multi_rev_round_trip_matches_kepler_propagation() {
        let r0: Vector3<f64> = Vector3::new(1.496e11, 0.0, 0.0); // 1 AU
        let v_circ = (MU_SUN / r0.norm()).sqrt();
        // Mildly eccentric (2% faster than circular) so the transfer angle
        // and period are well-defined and non-degenerate.
        let v0 = Vector3::new(0.0, v_circ * 1.02, 0.0);
        let eps = specific_energy([r0.x, r0.y, r0.z], [v0.x, v0.y, v0.z], MU_SUN);
        assert!(eps < 0.0, "test orbit must be elliptic");
        let a = -MU_SUN / (2.0 * eps);
        let period = 2.0 * std::f64::consts::PI * (a.powi(3) / MU_SUN).sqrt();

        // 2 full revolutions plus a SMALL residual (0.13% of the period,
        // matching the real VEEGA leg's "2.0013 Earth orbits in 731 days" —
        // deliberately small so r0/r1 stay close in angle, same as the real
        // near-circular-orbit case). A large residual fraction would sweep
        // a big angle near periapsis (this test orbit isn't exactly
        // circular) and wouldn't exercise the near-0° degenerate geometry
        // this function exists for.
        let n_rev = 2u32;
        let tof = n_rev as f64 * period + 0.0013 * period;

        let (r1, _v1_truth) = propagate_kepler(r0, v0, tof, MU_SUN).expect("propagation failed");

        // N=0 must NOT find this (that's the whole point of the gap).
        let n0 = lambert([r0.x, r0.y, r0.z], [r1.x, r1.y, r1.z], tof, true, MU_SUN);
        assert!(n0.is_empty(), "N=0 unexpectedly found a solution for a genuine N=2 geometry");

        // N=2 must find it and reproduce the true departure velocity.
        let sols = lambert_n_rev([r0.x, r0.y, r0.z], [r1.x, r1.y, r1.z], tof, true, MU_SUN, n_rev, 1e-6);
        assert!(!sols.is_empty(), "no N=2 Lambert solution found for a known N=2 geometry");

        let (v1_sol, v2_sol) = sols.iter()
            .min_by(|a, b| {
                let ea = norm(sub(a.0, [v0.x, v0.y, v0.z]));
                let eb = norm(sub(b.0, [v0.x, v0.y, v0.z]));
                ea.partial_cmp(&eb).unwrap()
            })
            .copied().unwrap();
        let err_v1 = norm(sub(v1_sol, [v0.x, v0.y, v0.z]));
        assert!(err_v1 < 1.0, "departure velocity error {:.3e} m/s", err_v1);

        // Independent check: propagate the recovered departure state forward
        // the full (multi-revolution) TOF and confirm it arrives at r1.
        let (r_end, _) = propagate_kepler(
            r0,
            Vector3::new(v1_sol[0], v1_sol[1], v1_sol[2]),
            tof, MU_SUN,
        ).expect("propagation failed");
        let miss = (r_end - r1).norm();
        assert!(miss < 1.0e6, "arrival miss {:.3e} m (> 1000 km)", miss);
        let _ = v2_sol;
    }
}
