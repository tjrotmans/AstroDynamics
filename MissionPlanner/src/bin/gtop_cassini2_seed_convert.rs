//! Converts one of OUR OWN saved MGA chromosomes (27-value layout, this
//! project's own parameterisation) into GTOP's native Cassini-2 decision
//! vector layout (22 values), for seeding a real pagmo run with a chromosome
//! our own pruning/bidirectional-backfit pipeline actually produced.
//!
//! Most fields transfer directly (same units, frame-independent scalars:
//! departure epoch offset, all 5 leg TOFs, all 5 etas, all 4 flyby periapsis
//! radii). The departure direction (`u,v`) and the 4 flyby `beta` angles do
//! NOT transfer directly -- our project decodes these in a FIXED GLOBAL
//! frame (`mga.rs:612-616`: raw spherical coordinates) and a fixed-reference-
//! vector B-plane basis (`mga_leg.rs::flyby_turn`), while GTOP decodes them
//! in the departure body's own LOCAL orbital frame and a planet-velocity-
//! based B-plane basis (`gtop_cassini2_eval.rs`, matching pykep's `fb_prop`)
//! -- same numeric values, physically DIFFERENT directions. Built 
//! after a direct-copy attempt gave a wildly worse result (33,557 vs.
//! 13,605 m/s) confirming this isn't a minor detail.
//!
//! This tool instead computes the REAL PHYSICAL VECTOR our chromosome
//! produces (via our own formulas), then re-projects that same vector onto
//! GTOP's basis -- a change of basis, not a re-derivation of physics.
//! Self-validated by round-trip: re-decoding the derived (u,v)/beta through
//! GTOP's OWN formula must reproduce the original physical vector to
//! floating-point precision; the tool prints this residual for every
//! converted quantity so a real bug shows up as a loud numerical mismatch,
//! not something that could look subtly fine.
//!
//! Usage: cargo run -p mission_planner --bin gtop_cassini2_seed_convert --release
//!   -- <our_27_value_chromosome_csv> <departure_reference_jd>
//!
//! Prints the converted 22-value GTOP-layout chromosome (ready for
//! `gtop_cassini2_eval`/pygmo) plus round-trip validation diagnostics.

use ephemeris::{Almanac, Body as AniseBody, Epoch};
use nalgebra::Vector3;
use trajectory_solver::{evaluate_mga_leg, flyby_turn, keplerian::MU_SUN_M3S2};

fn jd_to_epoch(jd: f64) -> Epoch {
    Epoch::from_unix_seconds((jd - 2_440_587.5) * 86_400.0)
}

fn body_state_vec3(almanac: &Almanac, body: AniseBody, jd: f64) -> (Vector3<f64>, Vector3<f64>) {
    let state = almanac
        .body_state_heliocentric(body, jd_to_epoch(jd))
        .unwrap_or_else(|e| panic!("ANISE query failed for {body:?} at JD {jd:.1}: {e}"));
    let r = state.position.inner;
    let v = state.velocity.inner;
    (Vector3::new(r[0], r[1], r[2]), Vector3::new(v[0], v[1], v[2]))
}

/// GTOP's local departure-orbital-frame basis (same construction as
/// `gtop_cassini2_eval.rs`): i_hat along velocity, z_hat along orbit
/// normal, j_hat completes the right-handed set.
fn gtop_departure_basis(r_dep: Vector3<f64>, v_dep: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    let i_hat = v_dep.normalize();
    let z_hat = r_dep.cross(&v_dep).normalize();
    let j_hat = z_hat.cross(&i_hat);
    (i_hat, j_hat, z_hat)
}

/// Projects a physical v∞ vector onto GTOP's local departure basis and
/// recovers (theta, phi), then (u, v). Returns (u, v, theta, phi).
fn encode_gtop_departure(v_inf: Vector3<f64>, r_dep: Vector3<f64>, v_dep: Vector3<f64>) -> (f64, f64, f64, f64) {
    let (i_hat, j_hat, z_hat) = gtop_departure_basis(r_dep, v_dep);
    let vinf_mag = v_inf.norm();
    let vi = v_inf.dot(&i_hat);
    let vj = v_inf.dot(&j_hat);
    let vz = v_inf.dot(&z_hat);
    let theta = vj.atan2(vi);
    let phi = (vz / vinf_mag).clamp(-1.0, 1.0).asin();
    let u = {
        let mut uu = theta / (2.0 * std::f64::consts::PI);
        if uu < 0.0 { uu += 1.0; }
        uu
    };
    let v = ((phi + std::f64::consts::FRAC_PI_2).cos() + 1.0) / 2.0;
    (u, v, theta, phi)
}

/// Decodes GTOP's (u, v) back into a physical v∞ vector, for round-trip
/// validation -- must reproduce the ORIGINAL vector passed to
/// `encode_gtop_departure`.
fn decode_gtop_departure(u: f64, v: f64, vinf_mag: f64, r_dep: Vector3<f64>, v_dep: Vector3<f64>) -> Vector3<f64> {
    let (i_hat, j_hat, z_hat) = gtop_departure_basis(r_dep, v_dep);
    let theta = 2.0 * std::f64::consts::PI * u;
    let phi = (2.0 * v - 1.0).clamp(-1.0, 1.0).acos() - std::f64::consts::FRAC_PI_2;
    vinf_mag * (theta.cos() * phi.cos() * i_hat + theta.sin() * phi.cos() * j_hat + phi.sin() * z_hat)
}

/// pykep `fb_prop` B-plane basis (same as `gtop_cassini2_eval.rs`).
fn pykep_flyby_basis(v_inf_in: Vector3<f64>, v_planet: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    let i_hat = v_inf_in.normalize();
    let j_hat = i_hat.cross(&v_planet).normalize();
    let k_hat = i_hat.cross(&j_hat);
    (i_hat, j_hat, k_hat)
}

/// Projects our own physical v∞_out (from `flyby_turn`) onto pykep's basis
/// and recovers beta. Returns (beta, delta).
fn encode_pykep_beta(v_inf_in: Vector3<f64>, v_inf_out: Vector3<f64>, v_planet: Vector3<f64>) -> f64 {
    let (i_hat, j_hat, k_hat) = pykep_flyby_basis(v_inf_in, v_planet);
    let vmag = v_inf_in.norm();
    let delta = (v_inf_out.dot(&i_hat) / vmag).clamp(-1.0, 1.0).acos();
    let sin_delta = delta.sin();
    if sin_delta.abs() < 1e-9 {
        return 0.0; // degenerate: no meaningful turn, beta unconstrained
    }
    let cos_b = v_inf_out.dot(&j_hat) / (vmag * sin_delta);
    let sin_b = v_inf_out.dot(&k_hat) / (vmag * sin_delta);
    sin_b.atan2(cos_b)
}

/// Reconstructs v∞_out from pykep's beta, for round-trip validation
/// (mirrors `fb_prop_pykep` in `gtop_cassini2_eval.rs`/`gtop_cassini2_check.rs`).
fn decode_pykep_beta(v_inf_in: Vector3<f64>, v_planet: Vector3<f64>, beta: f64, r_p_m: f64, mu_body: f64) -> Vector3<f64> {
    let v_mag = v_inf_in.norm();
    let e = 1.0 + r_p_m * v_mag * v_mag / mu_body;
    let delta = 2.0 * (1.0 / e).asin();
    let (i_hat, j_hat, k_hat) = pykep_flyby_basis(v_inf_in, v_planet);
    v_mag * (delta.cos() * i_hat + beta.cos() * delta.sin() * j_hat + beta.sin() * delta.sin() * k_hat)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: gtop_cassini2_seed_convert <our_27_value_chromosome_csv> <departure_reference_jd>");
        std::process::exit(1);
    }
    let p: Vec<f64> = args[1].split(',').map(|s| s.parse().expect("bad chromosome value")).collect();
    let dep_jd_base: f64 = args[2].parse().expect("bad reference JD");
    assert!(p.len() >= 22, "expected at least 22 chromosome values, got {}", p.len());

    let almanac = Almanac::new("kernels/de440s.bsp")
        .or_else(|_| Almanac::new("../kernels/de440s.bsp"))
        .expect("de440s.bsp not found");

    // Our own chromosome layout (mga.rs): p0=t0_offset, p1=vinf_ms,
    // p2=theta/u, p3=phi/v, then interleaved [T_k, eta_k] per leg,
    // then interleaved [rp_j, beta_j] per flyby.
    let t0_offset = p[0];
    let vinf_ms = p[1];
    let u_ours = p[2];
    let v_ours = p[3];
    let tof_days = [p[4], p[6], p[8], p[10], p[12]];
    let eta = [p[5], p[7], p[9], p[11], p[13]];
    let rp_norm = [p[14], p[16], p[18], p[20]];
    let beta_ours = [p[15], p[17], p[19], p[21]];

    let bodies = [AniseBody::Venus, AniseBody::Venus, AniseBody::Earth, AniseBody::Jupiter, AniseBody::Saturn];
    let flyby_mu = [3.248_59e14_f64, 3.248_59e14, 3.986_004_418e14, 1.266_865_34e17];
    let flyby_radius_m = [6_052_000.0_f64, 6_052_000.0, 6_378_000.0, 71_492_000.0];

    let dep_jd = dep_jd_base + t0_offset;
    let t0_mjd2000 = dep_jd - 2_451_544.5;

    let mut encounter_jd = vec![dep_jd];
    let mut jd = dep_jd;
    for t in tof_days {
        jd += t;
        encounter_jd.push(jd);
    }

    // ── Departure direction conversion ──────────────────────────────────
    let (r_dep, v_dep) = body_state_vec3(&almanac, AniseBody::Earth, dep_jd);
    // Our OWN physical v_inf vector (mga.rs:612-616 formula, global frame).
    let theta_ours = 2.0 * std::f64::consts::PI * u_ours; // MGA_GTOP_UNIFORM_DEPARTURE convention
    let phi_ours = (2.0 * v_ours - 1.0).clamp(-1.0, 1.0).acos() - std::f64::consts::FRAC_PI_2;
    let v_inf_ours = vinf_ms * Vector3::new(
        phi_ours.cos() * theta_ours.cos(),
        phi_ours.cos() * theta_ours.sin(),
        phi_ours.sin(),
    );

    let (u_gtop, v_gtop, _, _) = encode_gtop_departure(v_inf_ours, r_dep, v_dep);
    let v_check = decode_gtop_departure(u_gtop, v_gtop, vinf_ms, r_dep, v_dep);
    let dep_residual = (v_check - v_inf_ours).norm() / vinf_ms;
    eprintln!("[validate] departure round-trip relative residual: {dep_residual:.3e}");

    // ── Propagate legs, converting each flyby's beta ────────────────────
    let mut r_sc = r_dep;
    let mut v_sc = v_dep + v_inf_ours;
    let mut beta_gtop = [0.0_f64; 4];

    for k in 0..5 {
        let (r_next, v_next) = body_state_vec3(&almanac, bodies[k], encounter_jd[k + 1]);
        let leg = evaluate_mga_leg(r_sc, v_sc, eta[k], tof_days[k] * 86_400.0, r_next, v_next, MU_SUN_M3S2)
            .unwrap_or_else(|| panic!("leg {k} infeasible for this chromosome"));

        if k < 4 {
            let v_inf_out_ours = flyby_turn(
                leg.v_inf_arr_mps, rp_norm[k] * flyby_radius_m[k], beta_ours[k], flyby_mu[k],
            );
            let b_gtop = encode_pykep_beta(leg.v_inf_arr_mps, v_inf_out_ours, v_next);
            let v_check = decode_pykep_beta(leg.v_inf_arr_mps, v_next, b_gtop, rp_norm[k] * flyby_radius_m[k], flyby_mu[k]);
            let residual = (v_check - v_inf_out_ours).norm() / leg.v_inf_arr_mps.norm();
            eprintln!("[validate] flyby {k} beta round-trip relative residual: {residual:.3e}");
            beta_gtop[k] = b_gtop;

            r_sc = r_next;
            v_sc = v_next + v_inf_out_ours;
        }
    }

    let gtop_x = [
        t0_mjd2000, vinf_ms / 1e3, u_gtop, v_gtop,
        tof_days[0], tof_days[1], tof_days[2], tof_days[3], tof_days[4],
        eta[0], eta[1], eta[2], eta[3], eta[4],
        rp_norm[0], rp_norm[1], rp_norm[2], rp_norm[3],
        beta_gtop[0], beta_gtop[1], beta_gtop[2], beta_gtop[3],
    ];
    println!("{}", gtop_x.iter().map(|v| format!("{v:.15e}")).collect::<Vec<_>>().join(","));
}
