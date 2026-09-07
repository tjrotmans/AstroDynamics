//! Phase 8h follow-up: real lunar SOI-crossing verification.
//!
//! Companion to `soi_demo.rs` (Mars). A lunar transfer is fundamentally an
//! Earth-centered two-body problem, not heliocentric — the Moon's real
//! sphere of influence is relative to Earth (it orbits Earth, not the Sun),
//! not the Sun. `design.rs::propagator_body_entries` (Phase 7k) hardcodes
//! the Sun as the SOI primary for every target body, which is correct for
//! Mars/Jupiter/asteroids (orbit the Sun directly) but wrong for moons —
//! flagged as a known follow-up in the design notes, not fixed here. This binary
//! computes the Moon's SOI the physically correct way (Earth as primary)
//! directly, rather than reusing the (for this case, wrong) design.rs path.
//!
//! Run from the repo root: `cargo run -p mission_planner --bin soi_demo_moon --release`

use std::fs;

use ephemeris::{Almanac, Body, Epoch};
use hifitime::Duration;
use trajectory_solver::{laplace_soi_radius_m, propagate, LambertArc, PropagatorBody, ZonalFidelity};

const MU_EARTH: f64 = 3.986_004_418e14;
const MU_MOON: f64 = 4.902_800_066e12; // DE430, Folkner et al. 2014 — body_models::TargetBody::moon()
const MOON_R0: f64 = 1_737_400.0;
const MOON_J2: f64 = 2.032_3e-4;
const MOON_J3: f64 = -8.468e-6;
const MOON_J4: f64 = -9.009e-6;
// Archinal et al. 2018 (IAU WGCCRE 2015) lunar mean pole, J2000 constant term.
const MOON_POLE_RA_DEG: f64 = 269.994_9;
const MOON_POLE_DEC_DEG: f64 = 66.539_2;

const DEPARTURE_EPOCH: (i32, u8, u8) = (2026, 7, 1);
const LEO_ALTITUDE_M: f64 = 200_000.0;
const EARTH_RADIUS_M: f64 = 6_378_137.0;
const TOF_DAYS: f64 = 4.0; // typical translunar transfer duration
const POST_ARRIVAL_BUFFER_DAYS: f64 = 3.0;

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
    let kernel = find_kernel().expect("kernels/de440s.bsp not found");
    let almanac = Almanac::new(&kernel).expect("failed to load de440s.bsp");

    let (y, m, d) = DEPARTURE_EPOCH;
    let dep_epoch = Epoch::from_gregorian_utc(y, m, d, 0, 0, 0, 0);
    let arr_epoch = dep_epoch + Duration::from_days(TOF_DAYS);

    // Moon's position/velocity relative to EARTH (not the Sun) — the
    // correct frame for a translunar transfer.
    let moon_geocentric_at = |almanac: &Almanac, epoch: Epoch| -> (nalgebra::Vector3<f64>, nalgebra::Vector3<f64>) {
        let earth = almanac.body_state_heliocentric(Body::Earth, epoch).expect("Earth state");
        let moon = almanac.body_state_heliocentric(Body::Moon, epoch).expect("Moon state");
        (
            nalgebra::Vector3::new(
                moon.position.inner.x - earth.position.inner.x,
                moon.position.inner.y - earth.position.inner.y,
                moon.position.inner.z - earth.position.inner.z,
            ),
            nalgebra::Vector3::new(
                moon.velocity.inner.x - earth.velocity.inner.x,
                moon.velocity.inner.y - earth.velocity.inner.y,
                moon.velocity.inner.z - earth.velocity.inner.z,
            ),
        )
    };

    let (moon_r_dep, _) = moon_geocentric_at(&almanac, dep_epoch);
    let (moon_r_arr, moon_v_arr) = moon_geocentric_at(&almanac, arr_epoch);

    // Real Laplace SOI radius for the Moon, using EARTH as the primary
    // (mass ratio Moon/Earth, orbit radius = Moon's actual Earth-relative
    // distance at departure) — not the Sun, unlike `design.rs`'s current
    // (Mars/Jupiter/asteroid-only-correct) convention.
    let moon_dist_from_earth = moon_r_dep.norm();
    let soi_radius_m = laplace_soi_radius_m(moon_dist_from_earth, MU_MOON / MU_EARTH);
    println!(
        "Moon SOI radius (Earth-relative): {:.0} km  (Moon distance {:.0} km at departure)",
        soi_radius_m / 1e3, moon_dist_from_earth / 1e3,
    );

    // Departure: low Earth orbit, geocentric. Lambert (mu = Earth, not Sun
    // — this is a geocentric two-body problem) solves for the actual
    // translunar injection velocity needed to reach the Moon's real
    // position at arrival, exactly the same pattern `design.rs` uses
    // heliocentrically for interplanetary departures.
    // Target a B-plane-style aim point offset from the Moon's center, not
    // the center itself — letting Lambert aim dead-center produced an
    // uncontrolled, very deep (~3,640 km) flyby whose Moon-centered
    // dynamics turned out to be numerically extreme (`StepSizeUnderflow`
    // mid-leg, twice). A real flyby targets a specific periapsis, not a
    // body-center impact; this offset (60% of the SOI radius) keeps the
    // encounter comfortably gentle while still well inside the SOI.
    let moon_v_norm = moon_v_arr.norm();
    let offset_m = 0.6 * soi_radius_m;
    let r_arr = [
        moon_r_arr.x + offset_m * moon_v_arr.x / moon_v_norm,
        moon_r_arr.y + offset_m * moon_v_arr.y / moon_v_norm,
        moon_r_arr.z + offset_m * moon_v_arr.z / moon_v_norm,
    ];

    let r_dep = [EARTH_RADIUS_M + LEO_ALTITUDE_M, 0.0, 0.0];
    let v_dep = [0.0, 0.0, 0.0]; // unused for ΔV display here
    let v_arr = [moon_v_arr.x, moon_v_arr.y, moon_v_arr.z];
    let lambert = LambertArc { r_dep, v_dep, r_arr, v_arr, tof_s: TOF_DAYS * 86_400.0, mu: MU_EARTH };
    let sol = lambert.solve().expect("Lambert solve failed");
    println!("TLI injection ΔV (LEO circular -> transfer): {:.0} m/s", {
        let v_circ = (MU_EARTH / r_dep[0]).sqrt();
        (sol.v_transfer_dep[1] - v_circ).abs().max((sol.v_transfer_dep[0]).abs())
    });

    let almanac_ref = &almanac;
    let moon_state_at = move |t_abs_s: f64| {
        let t = dep_epoch + Duration::from_seconds(t_abs_s);
        moon_geocentric_at(almanac_ref, t)
    };

    let moon_body = PropagatorBody {
        name: "Moon",
        mu_m3s2: MU_MOON,
        soi_radius_m: Some(soi_radius_m),
        state_at: &moon_state_at,
        central_fidelity: Some(ZonalFidelity {
            r0_m: MOON_R0,
            j2: MOON_J2,
            j3: MOON_J3,
            j4: MOON_J4,
            pole_ra_rad: MOON_POLE_RA_DEG.to_radians(),
            pole_dec_rad: MOON_POLE_DEC_DEG.to_radians(),
        }),
        radius_m: Some(MOON_R0),
    };

    let r0 = nalgebra::Vector3::new(r_dep[0], r_dep[1], r_dep[2]);
    let v0 = nalgebra::Vector3::new(sol.v_transfer_dep[0], sol.v_transfer_dep[1], sol.v_transfer_dep[2]);
    let duration_s = (TOF_DAYS + POST_ARRIVAL_BUFFER_DAYS) * 86_400.0;
    let sample_dt_s = 1800.0; // 30 min — fine enough to resolve a multi-hour SOI passage

    // reference_mu_m3s2 = Earth (not the Sun): outside the Moon's SOI, this
    // is a geocentric two-body problem by construction for this demo.
    // atol=1e-10 m (the interplanetary convention `design.rs` uses, where
    // positions are AU-scale, ~1e11 m) is unreasonably tight once the
    // central body switches to the Moon and positions shrink to
    // lunar-orbit scale (~1e6-1e7 m) — found by running this demo with that
    // value: it produced a real `StepSizeUnderflow`. 1e-3 m is still far
    // tighter than this demo needs and avoids forcing the step size down
    // trying to satisfy an unreachable absolute precision.
    let points = propagate(r0, v0, 0.0, duration_s, MU_EARTH, &[moon_body], sample_dt_s, 1e-8, 1e-3);

    let mut rows = vec!["t_s,x_m,y_m,z_m,dist_to_moon_km,inside_soi".to_string()];
    let mut entered: Option<f64> = None;
    let mut exited: Option<f64> = None;
    let mut min_dist_km = f64::INFINITY;
    let mut was_inside = false;

    for p in &points {
        let t_abs = dep_epoch + Duration::from_seconds(p.t_s);
        let (moon_pos, _) = moon_geocentric_at(&almanac, t_abs);
        let dist_m = (p.r_m - moon_pos).norm();
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

    println!("Closest approach to Moon: {min_dist_km:.0} km  (SOI radius: {:.0} km)", soi_radius_m / 1e3);
    match (entered, exited) {
        (Some(t_in), Some(t_out)) => println!(
            "Moon SOI entry at t={:.2} days, exit at t={:.2} days  (real central-body switch confirmed over a {:.2}-day passage)",
            t_in / 86_400.0, t_out / 86_400.0, (t_out - t_in) / 86_400.0,
        ),
        (Some(t_in), None) => println!("Moon SOI entry at t={:.2} days, still inside at end of propagated arc", t_in / 86_400.0),
        _ => println!("WARNING: spacecraft never entered the Moon's SOI over this arc — increase POST_ARRIVAL_BUFFER_DAYS or re-check the window"),
    }

    let out_dir = "out/soi_demo_moon";
    fs::create_dir_all(out_dir).expect("create out dir");
    let traj_path = format!("{out_dir}/trajectory.csv");
    fs::write(&traj_path, rows.join("\n") + "\n").expect("write trajectory.csv");
    println!("{traj_path}  ({} points)", points.len());

    let mut moon_rows = vec!["t_s,x_m,y_m,z_m".to_string()];
    for p in &points {
        let t_abs = dep_epoch + Duration::from_seconds(p.t_s);
        let (moon_pos, _) = moon_geocentric_at(&almanac, t_abs);
        moon_rows.push(format!("{:.3},{:.6e},{:.6e},{:.6e}", p.t_s, moon_pos.x, moon_pos.y, moon_pos.z));
    }
    let moon_path = format!("{out_dir}/moon_track.csv");
    fs::write(&moon_path, moon_rows.join("\n") + "\n").expect("write moon_track.csv");

    let meta_path = format!("{out_dir}/meta.csv");
    fs::write(&meta_path, format!("soi_radius_m\n{soi_radius_m:.6e}\n")).expect("write meta.csv");
    println!("{moon_path}, {meta_path}");
}
