//! Standalone Cassini-2 GTOP fitness evaluator, callable per-decision-vector
//! from an external process (built to let a real pagmo/pygmo
//! run evaluate the SAME validated physics as `gtop_cassini2_check.rs`,
//! since modern pygmo no longer ships the classic GTOP benchmark problems
//! built in). Reuses that binary's exact formulas (Lambert legs via
//! `evaluate_mga_leg`, pykep-convention flyby, GTOP low-precision analytic
//! ephemeris — no ANISE kernel needed, so this starts fast enough to be
//! called thousands of times from an external optimizer) — see that
//! binary's own doc comment for the full convention notes (departure
//! direction decoded in the departure body's LOCAL frame; flyby β in
//! pykep's B-plane basis). Only the input (a fixed published vector there
//! vs. an arbitrary chromosome here) differs.
//!
//! Usage: reads one line from stdin — 22 comma-separated floats, GTOP's own
//! Cassini-2 decision-vector layout `[t0(MJD2000), Vinf(km/s), u, v,
//! T1..T5(days), eta1..eta5, rp1..rp4, beta1..beta4]` — and prints the
//! total ΔV (v∞ + DSM sum + Saturn-insertion burn, m/s) to stdout. Prints
//! `inf` for an infeasible chromosome (a leg's Lambert solve fails).
//!
//! cargo run -p mission_planner --bin gtop_cassini2_eval --release

use nalgebra::Vector3;
use trajectory_solver::{evaluate_mga_leg, keplerian::MU_SUN_M3S2};

#[derive(Clone, Copy, Debug)]
enum Body {
    Venus,
    Earth,
    Jupiter,
    Saturn,
}

const OBLIQUITY_J2000_RAD: f64 = 23.439_291_11_f64 * std::f64::consts::PI / 180.0;

fn ecliptic_to_icrf(v: Vector3<f64>) -> Vector3<f64> {
    let (s, c) = OBLIQUITY_J2000_RAD.sin_cos();
    Vector3::new(v.x, c * v.y - s * v.z, s * v.y + c * v.z)
}

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
    }
}

fn lp_position_ecliptic(el: &LpElements, jd: f64) -> Vector3<f64> {
    const AU_M: f64 = 1.495_978_707e11;
    let t_cy = (jd - 2_451_545.0) / 36_525.0;
    let a = (el.kep0[0] + el.rate[0] * t_cy) * AU_M;
    let e = el.kep0[1] + el.rate[1] * t_cy;
    let i = (el.kep0[2] + el.rate[2] * t_cy).to_radians();
    let l = (el.kep0[3] + el.rate[3] * t_cy).to_radians();
    let peri = (el.kep0[4] + el.rate[4] * t_cy).to_radians();
    let node = (el.kep0[5] + el.rate[5] * t_cy).to_radians();

    let omega = peri - node;
    let m = (l - peri) % (2.0 * std::f64::consts::PI);

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

fn lp_state_icrf(body: Body, jd: f64) -> (Vector3<f64>, Vector3<f64>) {
    let el = lp_elements(body);
    let r = ecliptic_to_icrf(lp_position_ecliptic(&el, jd));
    let dt_days = 60.0 / 86_400.0;
    let r_m = ecliptic_to_icrf(lp_position_ecliptic(&el, jd - dt_days));
    let r_p = ecliptic_to_icrf(lp_position_ecliptic(&el, jd + dt_days));
    let v = (r_p - r_m) / 120.0;
    (r, v)
}

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

/// Evaluates the GTOP Cassini-2 decision vector `x` (22 values, GTOP layout).
/// Returns `None` if any leg's Lambert solve is infeasible.
fn evaluate(x: &[f64]) -> Option<f64> {
    if x.len() != 22 {
        return None;
    }
    let t0_mjd2000 = x[0];
    let vinf_kms = x[1];
    let u = x[2];
    let v = x[3];
    let tof_days = [x[4], x[5], x[6], x[7], x[8]];
    let eta = [x[9], x[10], x[11], x[12], x[13]];
    let rp_norm = [x[14], x[15], x[16], x[17]];
    let beta = [x[18], x[19], x[20], x[21]];

    let bodies = [Body::Venus, Body::Venus, Body::Earth, Body::Jupiter];
    let flyby_mu = [3.248_59e14_f64, 3.248_59e14, 3.986_004_418e14, 1.266_865_34e17];
    let flyby_radius_m = [6_052_000.0_f64, 6_052_000.0, 6_378_000.0, 71_492_000.0];

    let dep_jd = 2_451_544.5 + t0_mjd2000;
    let bstate = |b: Body, jd: f64| lp_state_icrf(b, jd);

    let theta = 2.0 * std::f64::consts::PI * u;
    let phi = (2.0 * v - 1.0).acos() - std::f64::consts::FRAC_PI_2;
    let vinf_ms = vinf_kms * 1e3;

    let mut jd = dep_jd;
    let mut encounter_jd = vec![dep_jd];
    for t in tof_days {
        jd += t;
        encounter_jd.push(jd);
    }

    let (r_dep, v_dep) = bstate(Body::Earth, dep_jd);
    let i_hat = v_dep.normalize();
    let z_hat = r_dep.cross(&v_dep).normalize();
    let j_hat = z_hat.cross(&i_hat);
    let v_inf_dep = vinf_ms
        * (theta.cos() * phi.cos() * i_hat
            + theta.sin() * phi.cos() * j_hat
            + phi.sin() * z_hat);

    let mut r_sc = r_dep;
    let mut v_sc = v_dep + v_inf_dep;

    let mut dv_dsm_total = 0.0;
    let mut v_inf_arr_mag = 0.0;

    for k in 0..5 {
        let (r_next, v_next) = if k + 1 == 5 {
            bstate(Body::Saturn, encounter_jd[k + 1])
        } else {
            bstate(bodies[k], encounter_jd[k + 1])
        };
        let leg = evaluate_mga_leg(
            r_sc, v_sc, eta[k], tof_days[k] * 86_400.0, r_next, v_next, MU_SUN_M3S2,
        )?;
        dv_dsm_total += leg.dv_dsm_ms;
        v_inf_arr_mag = leg.v_inf_arr_mps.norm();

        if k < 4 {
            let v_inf_out = fb_prop_pykep(
                leg.v_inf_arr_mps, v_next, rp_norm[k] * flyby_radius_m[k], beta[k], flyby_mu[k],
            );
            r_sc = r_next;
            v_sc = v_next + v_inf_out;
        }
    }

    // NOTE: deliberately NOT the Oberth-reduced eccentric-
    // capture formula `gtop_cassini2_check.rs` uses to match the historical
    // published 8,383 m/s record -- that record's own arrival-burn
    // convention was later found (BENCHMARKS.md, "SECOND REAL METHODOLOGY
    // GAP") to disagree with the real GTOPtoolbox source's own
    // `cassini2()` definition (`problem.type = total_DV_rndv`: the FULL
    // arrival relative-velocity magnitude, no reduction at all). This
    // binary exists to compare a real pagmo run against OUR OWN project's
    // actual optimized objective (`MissionObjective::Rendezvous` in
    // `mga.rs`), so it must use the SAME convention our own search does --
    // otherwise a "does further search improve this seed" comparison would
    // silently be comparing two different objective functions.
    Some(vinf_ms + dv_dsm_total + v_inf_arr_mag)
}

fn main() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("failed to read stdin");
    let x: Vec<f64> = line
        .trim()
        .split(',')
        .map(|s| s.parse::<f64>().expect("failed to parse chromosome value"))
        .collect();

    match evaluate(&x) {
        Some(f) if f.is_finite() => println!("{f}"),
        _ => println!("inf"),
    }
}
