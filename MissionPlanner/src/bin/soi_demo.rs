//! Phase 8h: real SOI-crossing verification.
//!
//! The toy star+planet unit test in `crates/trajectory_solver/src/propagator.rs`
//! (`switches_central_body_on_soi_entry`) confirms the SOI-switching mechanism
//! works in isolation, but until Phase 7k wired real `PropagatorBody` candidates
//! into `design.rs`, no *real* mission ever exercised it. This binary propagates
//! the actual `mars_flyby.toml` best-arc Lambert solution (real ANISE ephemeris,
//! real Laplace SOI radius, real Mars J2/J3/J4 fidelity) a few days past its
//! nominal TOF, so the spacecraft's real flyby-speed pass through Mars' sphere
//! of influence is fully visible — entry, closest approach, and exit — not just
//! the single instant where the Lambert arc happens to coincide with Mars.
//!
//! Run from the repo root: `cargo run -p mission_planner --bin soi_demo --release`

use std::fs;

use ephemeris::{Almanac, Body, Epoch};
use hifitime::Duration;
use trajectory_solver::{
    keplerian::MU_SUN_M3S2, laplace_soi_radius_m, propagate, LambertArc, PropagatorBody,
    ZonalFidelity,
};

// Mars physical constants — same values as `MissionPlanner/config/mars_flyby.toml`
// and `body_models::TargetBody::mars()`.
const MARS_MU: f64 = 4.282_837_362_069_909e13;
const MARS_R0: f64 = 3_396_200.0;
const MARS_J2: f64 = 1.960_45e-3;
const MARS_J3: f64 = 3.142_5e-5;
const MARS_J4: f64 = -1.538_5e-5;
// Archinal et al. 2018 (IAU WGCCRE 2015) Mars pole, J2000 constant term.
const MARS_POLE_RA_DEG: f64 = 317.269_202;
const MARS_POLE_DEC_DEG: f64 = 54.432_516;

// Real best-arc window for `mars_flyby.toml`, as found by its own porkchop
// scan (Phase 8g/8j) — reproduced here directly rather than re-parsing the
// TOML, since this is a standalone debug tool, not a config-driven run.
const DEPARTURE_EPOCH: (i32, u8, u8) = (2026, 10, 28);
const DEP_OFFSET_DAYS: f64 = 3.80;
const TOF_DAYS: f64 = 291.14;
// How far past the nominal arrival to keep propagating, so the flyby's
// SOI entry/exit are both visible, not just the single coincident instant.
const POST_ARRIVAL_BUFFER_DAYS: f64 = 4.0;

fn find_kernel() -> Option<String> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join("kernels").join("de440s.bsp");
        if candidate.exists() {
            return Some(candidate.to_string_lossy().into_owned());
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn main() {
    let kernel = find_kernel().expect("kernels/de440s.bsp not found — see the README for download instructions");
    let almanac = Almanac::new(&kernel).expect("failed to load de440s.bsp");

    let (y, m, d) = DEPARTURE_EPOCH;
    let epoch0 = Epoch::from_gregorian_utc(y, m, d, 0, 0, 0, 0);
    let dep_epoch = epoch0 + Duration::from_days(DEP_OFFSET_DAYS);
    let arr_epoch = dep_epoch + Duration::from_days(TOF_DAYS);

    let earth_dep = almanac.body_state_heliocentric(Body::Earth, dep_epoch).expect("Earth state");
    let mars_arr = almanac.body_state_heliocentric(Body::Mars, arr_epoch).expect("Mars state");

    // Real Laplace SOI radius from Mars' actual heliocentric distance at
    // departure (same convention `design.rs::propagator_body_entries` uses).
    let mars_dep = almanac.body_state_heliocentric(Body::Mars, dep_epoch).expect("Mars state");
    let mars_dep_dist = (mars_dep.position.inner.x.powi(2)
        + mars_dep.position.inner.y.powi(2)
        + mars_dep.position.inner.z.powi(2))
    .sqrt();
    let soi_radius_m = laplace_soi_radius_m(mars_dep_dist, MARS_MU / MU_SUN_M3S2);

    let r_dep = [earth_dep.position.inner.x, earth_dep.position.inner.y, earth_dep.position.inner.z];
    let v_dep = [earth_dep.velocity.inner.x, earth_dep.velocity.inner.y, earth_dep.velocity.inner.z];

    // Target a B-plane-style aim point offset from Mars' center (along its
    // own velocity direction at arrival), not Mars' exact center — real
    // flyby targeting always aims for a specific periapsis, not body-center
    // impact. Offset magnitude is comfortably inside the real SOI radius
    // (~40% of it) so the *propagated* (not just analytic) trajectory is
    // guaranteed to cross the SOI boundary with margin to spare, regardless
    // of the few-hundred-thousand-km integration drift a ~291-day Lambert
    // transfer accumulates by arrival (the same effect any real flyby
    // targeting process — which this demo isn't attempting to reproduce —
    // would correct for with an aim-point search).
    let mars_v = [mars_arr.velocity.inner.x, mars_arr.velocity.inner.y, mars_arr.velocity.inner.z];
    let mars_v_norm = (mars_v[0].powi(2) + mars_v[1].powi(2) + mars_v[2].powi(2)).sqrt();
    let offset_m = 0.4 * soi_radius_m;
    let r_arr = [
        mars_arr.position.inner.x + offset_m * mars_v[0] / mars_v_norm,
        mars_arr.position.inner.y + offset_m * mars_v[1] / mars_v_norm,
        mars_arr.position.inner.z + offset_m * mars_v[2] / mars_v_norm,
    ];
    let v_arr = mars_v;

    let lambert = LambertArc { r_dep, v_dep, r_arr, v_arr, tof_s: TOF_DAYS * 86_400.0, mu: MU_SUN_M3S2 };
    let sol = lambert.solve().expect("Lambert solve failed");

    let almanac_ref = &almanac;
    let mars_state_at = move |t_abs_s: f64| {
        let t = dep_epoch + Duration::from_seconds(t_abs_s);
        let st = almanac_ref.body_state_heliocentric(Body::Mars, t).expect("Mars state");
        (
            nalgebra::Vector3::new(st.position.inner.x, st.position.inner.y, st.position.inner.z),
            nalgebra::Vector3::new(st.velocity.inner.x, st.velocity.inner.y, st.velocity.inner.z),
        )
    };

    let mars_body = PropagatorBody {
        name: "Mars",
        mu_m3s2: MARS_MU,
        soi_radius_m: Some(soi_radius_m),
        state_at: &mars_state_at,
        central_fidelity: Some(ZonalFidelity {
            r0_m: MARS_R0,
            j2: MARS_J2,
            j3: MARS_J3,
            j4: MARS_J4,
            pole_ra_rad: MARS_POLE_RA_DEG.to_radians(),
            pole_dec_rad: MARS_POLE_DEC_DEG.to_radians(),
        }),
        radius_m: Some(MARS_R0),
    };

    let r0 = nalgebra::Vector3::new(r_dep[0], r_dep[1], r_dep[2]);
    let v0 = nalgebra::Vector3::new(sol.v_transfer_dep[0], sol.v_transfer_dep[1], sol.v_transfer_dep[2]);
    let duration_s = (TOF_DAYS + POST_ARRIVAL_BUFFER_DAYS) * 86_400.0;
    // ~3-hour sampling — fine enough to resolve the few-day SOI passage in
    // detail while keeping the CSV a manageable size for the whole arc.
    let sample_dt_s = 3.0 * 3600.0;

    println!("Mars SOI radius: {:.0} km (real Laplace formula, Mars dist {:.4} AU at departure)", soi_radius_m / 1e3, mars_dep_dist / 1.495_978_707e11);

    let points = propagate(r0, v0, 0.0, duration_s, MU_SUN_M3S2, &[mars_body], sample_dt_s, 1e-8, 1e-10);

    let mut rows = vec!["t_s,x_m,y_m,z_m,dist_to_mars_km,inside_soi".to_string()];
    let mut entered: Option<f64> = None;
    let mut exited: Option<f64> = None;
    let mut min_dist_km = f64::INFINITY;
    let mut was_inside = false;

    for p in &points {
        let t_abs = dep_epoch + Duration::from_seconds(p.t_s);
        let mars_now = almanac.body_state_heliocentric(Body::Mars, t_abs).expect("Mars state");
        let mars_pos = nalgebra::Vector3::new(mars_now.position.inner.x, mars_now.position.inner.y, mars_now.position.inner.z);
        let dist_m = (p.r_m - mars_pos).norm();
        let dist_km = dist_m / 1e3;
        min_dist_km = min_dist_km.min(dist_km);
        let inside = dist_m < soi_radius_m;

        if inside && !was_inside {
            entered = Some(p.t_s);
        }
        if !inside && was_inside {
            exited = Some(p.t_s);
        }
        was_inside = inside;

        rows.push(format!(
            "{:.3},{:.6e},{:.6e},{:.6e},{:.3},{}",
            p.t_s, p.r_m.x, p.r_m.y, p.r_m.z, dist_km, if inside { 1 } else { 0 },
        ));
    }

    println!("Closest approach to Mars: {min_dist_km:.0} km  (SOI radius: {:.0} km)", soi_radius_m / 1e3);
    match (entered, exited) {
        (Some(t_in), Some(t_out)) => println!(
            "SOI entry at t={:.2} days, exit at t={:.2} days  (real central-body switch confirmed over a {:.2}-day passage)",
            t_in / 86_400.0, t_out / 86_400.0, (t_out - t_in) / 86_400.0,
        ),
        (Some(t_in), None) => println!("SOI entry at t={:.2} days, still inside at end of propagated arc", t_in / 86_400.0),
        _ => println!("WARNING: spacecraft never entered Mars' SOI over this arc — increase POST_ARRIVAL_BUFFER_DAYS or re-check the window"),
    }

    let out_dir = "out/soi_demo";
    fs::create_dir_all(out_dir).expect("create out dir");
    let traj_path = format!("{out_dir}/trajectory.csv");
    fs::write(&traj_path, rows.join("\n") + "\n").expect("write trajectory.csv");
    println!("{traj_path}  ({} points)", points.len());

    // Mars' own track over the same window, plus the SOI radius, for the plot
    // to draw Mars' path and a sphere at closest approach.
    let mut mars_rows = vec!["t_s,x_m,y_m,z_m".to_string()];
    for p in &points {
        let t_abs = dep_epoch + Duration::from_seconds(p.t_s);
        let mars_now = almanac.body_state_heliocentric(Body::Mars, t_abs).expect("Mars state");
        mars_rows.push(format!(
            "{:.3},{:.6e},{:.6e},{:.6e}",
            p.t_s, mars_now.position.inner.x, mars_now.position.inner.y, mars_now.position.inner.z,
        ));
    }
    let mars_path = format!("{out_dir}/mars_track.csv");
    fs::write(&mars_path, mars_rows.join("\n") + "\n").expect("write mars_track.csv");

    let meta_path = format!("{out_dir}/meta.csv");
    fs::write(&meta_path, format!("soi_radius_m\n{soi_radius_m:.6e}\n")).expect("write meta.csv");
    println!("{mars_path}, {meta_path}");
}
