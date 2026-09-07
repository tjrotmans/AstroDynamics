//! GTOP Cassini-2 published-solution cross-check (Phase 9v-v).
//!
//! Evaluates the PUBLISHED best-known GTOP Cassini-2 decision vector with
//! this repo's own MGA-1DSM building blocks (`propagate_kepler` + `lambert`
//! via `evaluate_mga_leg`), using pykep/GTOP's exact conventions for the
//! parts where our optimizer's parameterisation deliberately differs:
//!
//! - v∞ departure direction from (u, v): θ = 2πu, φ = acos(2v−1) − π/2,
//!   expressed in the heliocentric ECLIPTIC J2000 frame (GTOP's jpl_lp
//!   ephemerides are ecliptic) and rotated into ICRF for ANISE.
//! - Flyby B-plane basis from the PLANET VELOCITY (pykep `fb_prop`):
//!   î = v̂∞_in, ĵ = (î × v_planet)/|·|, k̂ = î × ĵ,
//!   v∞_out = |v∞_in|·(cosδ·î + cosβ·sinδ·ĵ + sinβ·sinδ·k̂).
//!   Our own `flyby_turn` uses an arbitrary fixed reference vector for the
//!   B-plane basis instead — same reachable set, different β convention —
//!   so the published β values are only meaningful under pykep's basis.
//!
//! Published reference (ESA GTOP archive, retrieved):
//!   https://www.esa.int/gsp/ACT/projects/gtop/cassini2/
//!   Best known solution (May 2009): objective 8.383 km/s.
//!   (Schlueter et al. 2017 later improved this to 6.399 km/s, but the full
//!   decision vector published on the archive page is the 2009 one — that is
//!   what this binary reproduces.)
//!
//! Known, accepted differences vs. GTOP's own evaluator:
//! - Ephemeris: DE440s (ANISE) vs. GTOP's analytic low-precision Keplerian
//!   elements — planet positions differ by O(10³–10⁴ km), so DSM magnitudes
//!   shift by O(10 m/s).
//! - Time scale: t0 is MJD2000; we convert via JD with a UTC-based Epoch
//!   (~69 s offset from TT) — negligible against the ephemeris difference.
//!
//! Pass criterion: total ΔV within a few percent of 8 383 m/s. A km/s-scale
//! mismatch means a convention error (frame, β basis, leg chaining), which
//! is exactly what this check exists to catch.
//!
//! Run:
//!   cargo run -p mission_planner --bin gtop_cassini2_check --release

use ephemeris::{Almanac, Body, Epoch};
use nalgebra::Vector3;
use trajectory_solver::{evaluate_mga_leg, keplerian::MU_SUN_M3S2};

/// JD to Epoch using the same formula as `design.rs::jd_to_epoch`.
fn jd_to_epoch(jd: f64) -> Epoch {
    Epoch::from_unix_seconds((jd - 2_440_587.5) * 86_400.0)
}

fn body_state_vec3(almanac: &Almanac, body: Body, jd: f64) -> (Vector3<f64>, Vector3<f64>) {
    let state = almanac
        .body_state_heliocentric(body, jd_to_epoch(jd))
        .unwrap_or_else(|e| panic!("ANISE query failed for {body:?} at JD {jd:.1}: {e}"));
    // ephemeris::Almanac already converts ANISE's km to metres internally
    // (see crates/ephemeris/src/almanac.rs::km_to_m_helio) — no scaling here.
    let r = state.position.inner;
    let v = state.velocity.inner;
    (
        Vector3::new(r[0], r[1], r[2]),
        Vector3::new(v[0], v[1], v[2]),
    )
}

/// Mean obliquity of the ecliptic at J2000 [rad] — IAU 2006 (Hilton et al.).
/// Same constant `plot/mga_frame_utils.py` uses for the inverse rotation.
const OBLIQUITY_J2000_RAD: f64 = 23.439_291_11_f64 * std::f64::consts::PI / 180.0;

/// Rotate a vector from heliocentric ecliptic J2000 into equatorial ICRF
/// (rotation by +ε about the x-axis).
fn ecliptic_to_icrf(v: Vector3<f64>) -> Vector3<f64> {
    let (s, c) = OBLIQUITY_J2000_RAD.sin_cos();
    Vector3::new(v.x, c * v.y - s * v.z, s * v.y + c * v.z)
}

/// pykep `fb_prop` unpowered gravity-assist: B-plane basis built from the
/// planet's heliocentric velocity (see module doc). Returns v∞_out.
fn fb_prop_pykep(
    v_inf_in: Vector3<f64>,
    v_planet: Vector3<f64>,
    r_p_m: f64,
    beta_rad: f64,
    mu_body: f64,
) -> Vector3<f64> {
    let v_mag = v_inf_in.norm();
    let e = 1.0 + r_p_m * v_mag * v_mag / mu_body;
    let delta = 2.0 * (1.0 / e).asin();

    let i_hat = v_inf_in / v_mag;
    let j_hat = i_hat.cross(&v_planet).normalize();
    let k_hat = i_hat.cross(&j_hat);

    v_mag
        * (delta.cos() * i_hat
            + beta_rad.cos() * delta.sin() * j_hat
            + beta_rad.sin() * delta.sin() * k_hat)
}

// ── GTOP-style low-precision analytic ephemeris (GTOP_LP=1) ─────────────────
// Keplerian elements + linear centennial rates, heliocentric ecliptic J2000
// (Standish, "Keplerian Elements for Approximate Positions of the Major
// Planets", JPL, 1800 AD–2050 AD table — the same family of low-precision
// elements GTOP's pleph_an / pykep's jpl_lp use). Purpose: evaluate the
// published GTOP chromosome against the SAME KIND of ephemeris it was
// optimized on, removing the DE440s-vs-analytic confound that the resonant
// V-V / short V-E legs amplify.
// Elements per body: [a_AU, e, I_deg, L_deg, peri_deg (ϖ), node_deg (Ω)]
// and their per-Julian-century rates.
struct LpElements {
    kep0: [f64; 6],
    rate: [f64; 6],
}

fn lp_elements(body: Body) -> LpElements {
    match body {
        Body::Venus => LpElements {
            kep0: [0.723_335_66, 0.006_776_72, 3.394_676_05, 181.979_099_50, 131.602_467_18, 76.679_842_55],
            rate: [0.000_003_90, -0.000_041_07, -0.000_788_90, 58_517.815_387_29, 0.002_683_29, -0.277_694_18],
        },
        Body::Earth => LpElements {
            // Earth-Moon barycenter — the same simplification GTOP makes.
            kep0: [1.000_002_61, 0.016_711_23, -0.000_015_31, 100.464_571_66, 102.937_681_93, 0.0],
            rate: [0.000_005_62, -0.000_043_92, -0.012_946_68, 35_999.372_449_81, 0.323_273_64, 0.0],
        },
        Body::Jupiter => LpElements {
            kep0: [5.202_887_00, 0.048_386_24, 1.304_396_95, 34.396_440_51, 14.728_479_83, 100.473_909_09],
            rate: [-0.000_116_07, -0.000_132_53, -0.001_837_14, 3_034.746_127_75, 0.212_526_68, 0.204_691_06],
        },
        Body::Saturn => LpElements {
            kep0: [9.536_675_94, 0.053_861_79, 2.485_991_87, 49.954_244_23, 92.598_878_31, 113.662_424_48],
            rate: [-0.001_250_60, -0.000_509_91, 0.001_936_09, 1_222.493_622_01, -0.418_972_16, -0.288_677_94],
        },
        other => panic!("lp_elements: no low-precision elements for {other:?}"),
    }
}

/// Heliocentric ecliptic-J2000 position [m] from the low-precision elements
/// at Julian date `jd` (TDB ≈ TT — sub-minute timescale differences are far
/// below the accuracy of these elements).
fn lp_position_ecliptic(el: &LpElements, jd: f64) -> Vector3<f64> {
    const AU_M: f64 = 1.495_978_707e11;
    let t_cy = (jd - 2_451_545.0) / 36_525.0;
    let a = (el.kep0[0] + el.rate[0] * t_cy) * AU_M;
    let e = el.kep0[1] + el.rate[1] * t_cy;
    let i = (el.kep0[2] + el.rate[2] * t_cy).to_radians();
    let l = (el.kep0[3] + el.rate[3] * t_cy).to_radians();
    let peri = (el.kep0[4] + el.rate[4] * t_cy).to_radians();
    let node = (el.kep0[5] + el.rate[5] * t_cy).to_radians();

    let omega = peri - node;        // argument of periapsis
    let m = (l - peri) % (2.0 * std::f64::consts::PI); // mean anomaly

    // Kepler solve (Newton).
    let mut ea = m;
    for _ in 0..30 {
        let f = ea - e * ea.sin() - m;
        ea -= f / (1.0 - e * ea.cos());
    }
    let x_pf = a * (ea.cos() - e);
    let y_pf = a * (1.0 - e * e).sqrt() * ea.sin();

    let (so, co) = omega.sin_cos();
    let (sn, cn) = node.sin_cos();
    let (si, ci) = i.sin_cos();
    Vector3::new(
        (co * cn - so * sn * ci) * x_pf + (-so * cn - co * sn * ci) * y_pf,
        (co * sn + so * cn * ci) * x_pf + (-so * sn + co * cn * ci) * y_pf,
        (so * si) * x_pf + (co * si) * y_pf,
    )
}

/// Low-precision body state in the same frame the rest of this binary uses
/// (ICRF equatorial, matching ANISE): analytic ecliptic position rotated to
/// ICRF, velocity by central finite difference (±60 s).
fn lp_state_icrf(body: Body, jd: f64) -> (Vector3<f64>, Vector3<f64>) {
    let el = lp_elements(body);
    let r = ecliptic_to_icrf(lp_position_ecliptic(&el, jd));
    let dt_days = 60.0 / 86_400.0;
    let r_m = ecliptic_to_icrf(lp_position_ecliptic(&el, jd - dt_days));
    let r_p = ecliptic_to_icrf(lp_position_ecliptic(&el, jd + dt_days));
    let v = (r_p - r_m) / 120.0;
    (r, v)
}

fn main() {
    let almanac = Almanac::new("kernels/de440s.bsp")
        .or_else(|_| Almanac::new("../kernels/de440s.bsp"))
        .expect("de440s.bsp not found — download from https://public-data.nyxspace.com/anise/de440s.bsp");

    // ── Published GTOP Cassini-2 best-known decision vector (May 2009) ──────
    // https://www.esa.int/gsp/ACT/projects/gtop/cassini2/  (objective 8.383 km/s)
    // Layout: [t0(MJD2000), Vinf(km/s), u, v, T1..T5(days), eta1..eta5,
    //          rp1..rp4 (planet radii), beta1..beta4 (rad)]
    let t0_mjd2000 = -779.046_753_814_506_f64;
    let vinf_kms = 3.259_114_468_323_45_f64;
    let u = 0.525_976_214_695_235_f64;
    let v = 0.380_864_964_586_57_f64;
    let tof_days = [
        167.378_952_534_645_f64,
        424.028_254_165_204,
        53.289_740_976_920_5,
        589.766_954_923_325,
        2_200.0,
    ];
    let eta = [
        0.769_483_451_363_201_f64,
        0.513_289_529_822_621,
        0.027_417_536_226_402_4,
        0.263_985_256_705_873,
        0.599_984_695_281_461,
    ];
    let rp_norm = [1.348_779_686_571_76_f64, 1.05, 1.307_302_783_720_17, 69.809_014_299_349_5];
    let beta = [-1.593_737_112_119_1_f64, -1.959_525_122_324_47, -1.554_988_592_830_59, -1.513_462_529_967_4];

    // Body sequence: Earth → Venus → Venus → Earth → Jupiter → Saturn.
    let bodies = [Body::Earth, Body::Venus, Body::Venus, Body::Earth, Body::Jupiter, Body::Saturn];
    // μ [m³/s²] and radius [m] per flyby body (GTOP constants; the catalog's
    // values differ only in trailing digits): Venus, Venus, Earth, Jupiter.
    let flyby_mu = [3.248_59e14_f64, 3.248_59e14, 3.986_004_418e14, 1.266_865_34e17];
    let flyby_radius_m = [6_052_000.0_f64, 6_052_000.0, 6_378_000.0, 71_492_000.0];

    // Saturn capture (real GTOP problem spec): r_p = 108 950 km, e = 0.98.
    let mu_saturn = 3.793_118_7e16_f64;
    let r_cap = 108_950_000.0_f64;
    let e_cap = 0.98_f64;

    // MJD2000 epoch = 2000-01-01 00:00 = JD 2451544.5.
    let dep_jd = 2_451_544.5 + t0_mjd2000;

    // Ephemeris selector: DE440s (default) or GTOP-style low-precision
    // analytic elements (GTOP_LP=1) — see lp_elements() above.
    let use_lp = std::env::var("GTOP_LP").is_ok();
    let bstate = |b: Body, jd: f64| -> (Vector3<f64>, Vector3<f64>) {
        if use_lp { lp_state_icrf(b, jd) } else { body_state_vec3(&almanac, b, jd) }
    };
    if use_lp {
        // Sanity print: LP-vs-DE440s position difference at departure for
        // each body — a memory-slip in the element table would show up here
        // as a huge (≫1e5 km) discrepancy.
        println!("[lp] analytic-elements ephemeris active; sanity vs DE440s at departure JD:");
        for b in [Body::Venus, Body::Earth, Body::Jupiter, Body::Saturn] {
            let (r_lp, _) = lp_state_icrf(b, dep_jd);
            let (r_de, _) = body_state_vec3(&almanac, b, dep_jd);
            println!("  {b:?}: |Δr| = {:.0} km", (r_lp - r_de).norm() / 1e3);
        }
        println!();
    }

    // Departure v∞ direction: GTOP's original MATLAB (`mga_dsm.m`, ESA ACT)
    // expresses (θ, φ) in a LOCAL orbital frame at the departure planet —
    //   î = v̂_planet (along-track), ẑ = (r × v)̂ (orbit normal), ĵ = ẑ × î —
    //   v∞ = Vinf·(cosθ·cosφ·î + sinθ·cosφ·ĵ + sinφ·ẑ)
    // NOT a global inertial frame. (Verified here empirically: a global
    // ecliptic decode puts the departure 135° away from the leg-0 DSM-ΔV
    // minimum; the local-frame decode lands on it — see GTOP_SCAN=1.)
    let theta = 2.0 * std::f64::consts::PI * u;
    let phi = (2.0 * v - 1.0).acos() - std::f64::consts::FRAC_PI_2;
    let vinf_ms = vinf_kms * 1e3;

    println!("=== GTOP Cassini-2 published-solution cross-check (9v-v) ===\n");
    println!("Departure JD: {dep_jd:.4}  (t0 = {t0_mjd2000:.3} MJD2000, ~1997-11-13)");
    println!("Departure v∞: {vinf_ms:.1} m/s  θ={:.4} rad  φ={:.4} rad (local orbital frame)\n", theta, phi);

    // Encounter epochs.
    let mut jd = dep_jd;
    let mut encounter_jd = vec![dep_jd];
    for t in tof_days {
        jd += t;
        encounter_jd.push(jd);
    }

    // Departure state.
    let (r_dep, v_dep) = bstate(bodies[0], dep_jd);
    let i_hat = v_dep.normalize();
    let z_hat = r_dep.cross(&v_dep).normalize();
    let j_hat = z_hat.cross(&i_hat);
    let mut v_inf_dep = vinf_ms
        * (theta.cos() * phi.cos() * i_hat
            + theta.sin() * phi.cos() * j_hat
            + phi.sin() * z_hat);

    // GTOP_BEST_CHAIN=1: instead of the published direction/β values, use
    // the leg-0-DSM-optimal departure direction and, at each flyby, the
    // next-leg-DSM-optimal β — i.e. locally re-optimize the continuous
    // pointing variables under OUR dynamics while keeping the published
    // Vinf/T/η/rp structure. The published values were tuned against GTOP's
    // analytic ephemeris; the resonant V-V and 53-day V-E legs amplify the
    // small DE440s-vs-analytic state differences, so this mode shows what
    // the same solution shape costs under our ephemeris.
    let best_chain = std::env::var("GTOP_BEST_CHAIN").is_ok();
    if best_chain {
        let (r_v1, v_v1) = bstate(bodies[1], encounter_jd[1]);
        let mut best = (f64::MAX, v_inf_dep);
        // Coarse 1° grid, then 0.05° refinement around the optimum.
        let mut center = (0.0_f64, 0.0_f64);
        for (step_deg, half_range_deg) in [(1.0_f64, 180.0_f64), (0.05, 1.5)] {
            let n = (2.0 * half_range_deg / step_deg) as i64;
            for it in 0..=n {
                for ip in 0..=n.min(((2.0 * 45.0_f64.min(half_range_deg)) / step_deg) as i64) {
                    let th = center.0 + (it as f64 * step_deg - half_range_deg).to_radians();
                    let ph = center.1
                        + (ip as f64 * step_deg - 45.0_f64.min(half_range_deg)).to_radians();
                    if ph.abs() > std::f64::consts::FRAC_PI_2 { continue; }
                    let vv = vinf_ms * Vector3::new(
                        ph.cos() * th.cos(), ph.cos() * th.sin(), ph.sin());
                    if let Some(l) = evaluate_mga_leg(
                        r_dep, v_dep + vv, eta[0], tof_days[0] * 86_400.0,
                        r_v1, v_v1, MU_SUN_M3S2,
                    ) {
                        if l.dv_dsm_ms < best.0 { best = (l.dv_dsm_ms, vv); }
                    }
                }
            }
            let b = best.1 / vinf_ms;
            center = (b.y.atan2(b.x), b.z.asin());
        }
        println!("[best-chain] leg-0 optimal departure: DSM = {:.1} m/s\n", best.0);
        v_inf_dep = best.1;
    }
    let mut r_sc = r_dep;
    let mut v_sc = v_dep + v_inf_dep;

    // ── Diagnostic: scan leg-0 departure direction (GTOP_SCAN=1) ────────────
    // Finds the (θ, φ) in the ICRF frame that minimises leg-0 DSM ΔV at the
    // published Vinf/η/T — reveals the correct direction decode when the
    // published (u, v) mapping produces garbage.
    if std::env::var("GTOP_SCAN").is_ok() {
        let (r_v1, v_v1) = bstate(bodies[1], encounter_jd[1]);
        let mut best = (f64::MAX, 0.0, 0.0);
        for it in 0..360 {
            for ip in 0..91 {
                let th = it as f64 * std::f64::consts::PI / 180.0;
                let ph = (ip as f64 - 45.0) * std::f64::consts::PI / 180.0;
                let vv = Vector3::new(
                    vinf_ms * ph.cos() * th.cos(),
                    vinf_ms * ph.cos() * th.sin(),
                    vinf_ms * ph.sin(),
                );
                if let Some(l) = evaluate_mga_leg(
                    r_dep, v_dep + vv, eta[0], tof_days[0] * 86_400.0, r_v1, v_v1, MU_SUN_M3S2,
                ) {
                    if l.dv_dsm_ms < best.0 {
                        best = (l.dv_dsm_ms, th, ph);
                    }
                }
            }
        }
        println!(
            "[scan] leg-0 min DSM = {:.1} m/s at θ={:.4} rad ({:.1}°), φ={:.4} rad ({:.1}°) [ICRF]",
            best.0, best.1, best.1.to_degrees(), best.2, best.2.to_degrees()
        );
        println!(
            "[scan] published decode gave θ={:.4} rad ({:.1}°), φ={:.4} rad ({:.1}°) [ecl] → ICRF vec [{:.1}, {:.1}, {:.1}]",
            theta, theta.to_degrees(), phi, phi.to_degrees(),
            v_inf_dep.x, v_inf_dep.y, v_inf_dep.z
        );
    }

    let mut dv_dsm_total = 0.0;
    let mut v_inf_arr_mag = 0.0;

    // Arc samples for plotting: (t_days since departure, r [m], leg_idx) —
    // written in plot_mga.py's mga_best.csv format so the standard MGA plot
    // renders this reproduction directly (per the every-trajectory-gets-a-
    // plot rule). Both sub-arcs sampled by Kepler propagation: forward from
    // the leg start, and backward from the arrival state (v_body + v∞_arr).
    let mut arc_rows: Vec<String> =
        vec!["t_days,x_m,y_m,z_m,leg_idx".to_string()];

    for k in 0..5 {
        let (r_next, v_next) = bstate(bodies[k + 1], encounter_jd[k + 1]);
        println!(
            "  [dbg] leg {k}: |r_sc|={:.4} AU |v_sc|={:.1} m/s  |r_next|={:.4} AU |v_next|={:.1} m/s  jd_next={:.2}",
            r_sc.norm() / 1.496e11, v_sc.norm(),
            r_next.norm() / 1.496e11, v_next.norm(), encounter_jd[k + 1]
        );
        let leg = evaluate_mga_leg(
            r_sc, v_sc, eta[k], tof_days[k] * 86_400.0, r_next, v_next, MU_SUN_M3S2,
        )
        .unwrap_or_else(|| panic!("leg {k} infeasible — convention error upstream?"));

        println!(
            "Leg {k} ({:?} → {:?}): TOF={:7.1} d  η={:.3}  DSM ΔV={:8.2} m/s  v∞_arr={:8.1} m/s",
            bodies[k], bodies[k + 1], tof_days[k], eta[k],
            leg.dv_dsm_ms, leg.v_inf_arr_mps.norm()
        );
        dv_dsm_total += leg.dv_dsm_ms;
        v_inf_arr_mag = leg.v_inf_arr_mps.norm();

        // Sample both sub-arcs for the plot (40 points each).
        {
            use trajectory_solver::propagate_kepler;
            let t_leg_start_days = encounter_jd[k] - dep_jd;
            let dt_a = eta[k] * tof_days[k] * 86_400.0;
            let dt_b = (1.0 - eta[k]) * tof_days[k] * 86_400.0;
            let v_sc_arr = v_next + leg.v_inf_arr_mps;
            for i in 0..=40 {
                let f = i as f64 / 40.0;
                if let Some((r, _)) = propagate_kepler(r_sc, v_sc, f * dt_a, MU_SUN_M3S2) {
                    arc_rows.push(format!(
                        "{:.6},{:.3},{:.3},{:.3},{k}",
                        t_leg_start_days + f * dt_a / 86_400.0, r.x, r.y, r.z
                    ));
                }
            }
            for i in 0..=40 {
                let f = i as f64 / 40.0;
                // Backward from arrival covers the post-DSM Lambert sub-arc.
                if let Some((r, _)) =
                    propagate_kepler(r_next, v_sc_arr, -(1.0 - f) * dt_b, MU_SUN_M3S2)
                {
                    arc_rows.push(format!(
                        "{:.6},{:.3},{:.3},{:.3},{k}",
                        t_leg_start_days + (eta[k] * tof_days[k]) + f * dt_b / 86_400.0,
                        r.x, r.y, r.z
                    ));
                }
            }
        }

        if k < 4 {
            // Scan β for the value minimising the NEXT leg's DSM — as a
            // diagnostic (GTOP_SCAN=1, reveals convention offsets) and as
            // the value actually used in best-chain mode.
            let mut beta_used = beta[k];
            if std::env::var("GTOP_SCAN").is_ok() || best_chain {
                let (r_n2, v_n2) = bstate(bodies[k + 2], encounter_jd[k + 2]);
                let mut best = (f64::MAX, 0.0);
                for ib in 0..7200 {
                    let b = -std::f64::consts::PI + ib as f64 * std::f64::consts::PI / 3600.0;
                    let vo = fb_prop_pykep(
                        leg.v_inf_arr_mps, v_next, rp_norm[k] * flyby_radius_m[k], b, flyby_mu[k],
                    );
                    if let Some(l2) = evaluate_mga_leg(
                        r_next, v_next + vo, eta[k + 1], tof_days[k + 1] * 86_400.0,
                        r_n2, v_n2, MU_SUN_M3S2,
                    ) {
                        if l2.dv_dsm_ms < best.0 { best = (l2.dv_dsm_ms, b); }
                    }
                }
                println!(
                    "        [scan] flyby {k}: min next-leg DSM {:.1} m/s at β*={:+.4} rad (published {:+.4}, Δ={:+.4})",
                    best.0, best.1, beta[k], best.1 - beta[k]
                );
                if best_chain { beta_used = best.1; }
            }
            // pykep-convention flyby at bodies[k+1].
            let v_inf_out = fb_prop_pykep(
                leg.v_inf_arr_mps,
                v_next,
                rp_norm[k] * flyby_radius_m[k],
                beta_used,
                flyby_mu[k],
            );
            let turn_deg = (v_inf_out.dot(&leg.v_inf_arr_mps)
                / (v_inf_out.norm() * leg.v_inf_arr_mps.norm()))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
            println!(
                "        flyby: rp={:.0} km  β={:+.3} rad  turn={:.2}°  |v∞| conserved to {:.1e}",
                rp_norm[k] * flyby_radius_m[k] * 1e-3,
                beta[k],
                turn_deg,
                (v_inf_out.norm() - leg.v_inf_arr_mps.norm()).abs() / leg.v_inf_arr_mps.norm()
            );
            r_sc = r_next;
            v_sc = v_next + v_inf_out;
        }
    }

    // Saturn insertion into the GTOP capture ellipse (burn at periapsis).
    let v_hyp = (v_inf_arr_mag * v_inf_arr_mag + 2.0 * mu_saturn / r_cap).sqrt();
    let v_peri = (mu_saturn * (1.0 + e_cap) / r_cap).sqrt();
    let dv_insertion = v_hyp - v_peri;

    println!("\nSaturn insertion (r_p = {:.0} km, e = {e_cap}):", r_cap * 1e-3);
    println!("  v∞ = {v_inf_arr_mag:.1} m/s → ΔV = {dv_insertion:.2} m/s");

    let total_with_launch_vinf = vinf_ms + dv_dsm_total + dv_insertion;
    let total_dsm_only = dv_dsm_total + dv_insertion;

    println!("\n=== Totals ===");
    println!("  Launch v∞ (raw):          {vinf_ms:9.2} m/s");
    println!("  DSM sum:                  {dv_dsm_total:9.2} m/s");
    println!("  Saturn insertion:         {dv_insertion:9.2} m/s");
    println!("  Total (v∞ + DSM + ins.):  {total_with_launch_vinf:9.2} m/s");
    println!("  Total (DSM + insertion):  {total_dsm_only:9.2} m/s");
    println!("\n  GTOP published objective:  8383.00 m/s (May 2009 best known)");
    println!(
        "  Δ vs published (v∞ conv.): {:+9.2} m/s  ({:+.2}%)",
        total_with_launch_vinf - 8_383.0,
        (total_with_launch_vinf - 8_383.0) / 8_383.0 * 100.0
    );

    // Write the sampled arc so `py plot/plot_mga.py cassini2_check` renders it.
    let out_dir = "out/cassini2_check";
    let _ = std::fs::create_dir_all(out_dir);
    let path = format!("{out_dir}/mga_best.csv");
    match std::fs::write(&path, arc_rows.join("\n") + "\n") {
        Ok(()) => println!("\n  Arc written: {path}  (plot: py plot/plot_mga.py cassini2_check)"),
        Err(e) => eprintln!("Warning: could not write {path}: {e}"),
    }
}
