//! Real patched-conic hyperbolic departure leg verification.
//!
//! `crates/trajectory_solver/src/departure.rs`'s unit tests confirm the
//! injection-state construction in isolation (toy bodies, hand-picked
//! v-infinity vectors). This binary exercises it against a real mission:
//! a real Earth->Mars Lambert solve (same window as `soi_demo.rs`) supplies
//! the real 3D v-infinity vector, which then drives a real numerically-
//! propagated escape from a real parking orbit, out through Earth's actual
//! Laplace SOI — replacing the degenerate "start already at v-infinity at
//! Earth's center" shortcut every caller used before Phase 9's departure-leg
//! work.
//!
//! Run from the repo root: `cargo run -p mission_planner --bin departure_demo --release`

use std::fs;

use ephemeris::{Almanac, Body, Epoch};
use hifitime::Duration;
use trajectory_solver::{
    keplerian::MU_SUN_M3S2, laplace_soi_radius_m, propagate_escape_leg, LambertArc, PropagatorBody,
};

// Earth physical constants (EGM2008 mu, IAU mean equatorial radius).
const EARTH_MU: f64 = 3.986_004_418e14;
const EARTH_RADIUS_M: f64 = 6_378_137.0;

// Same real Earth->Mars window as `soi_demo.rs` (`mars_flyby.toml`'s own
// verified porkchop minimum) — reused here only as a source of a real,
// physically reasonable v-infinity vector, not because the arrival side
// matters for this demo.
const DEPARTURE_EPOCH: (i32, u8, u8) = (2026, 10, 28);
const DEP_OFFSET_DAYS: f64 = 3.80;
const TOF_DAYS: f64 = 291.14;

// Parking orbit radius: 1.5x Earth's radius, matching
// `design.rs::resolve_parking_orbit_radius_m`'s airless-body heuristic
// (Earth has an atmosphere, so a real mission would use radius+200km, but
// 1.5x keeps this demo's numbers round and the difference is immaterial to
// what's being verified here).
const R_PARK_M: f64 = EARTH_RADIUS_M * 1.5;

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

    let r_dep = [earth_dep.position.inner.x, earth_dep.position.inner.y, earth_dep.position.inner.z];
    let v_dep = [earth_dep.velocity.inner.x, earth_dep.velocity.inner.y, earth_dep.velocity.inner.z];
    let r_arr = [mars_arr.position.inner.x, mars_arr.position.inner.y, mars_arr.position.inner.z];
    let v_arr = [mars_arr.velocity.inner.x, mars_arr.velocity.inner.y, mars_arr.velocity.inner.z];

    let lambert = LambertArc { r_dep, v_dep, r_arr, v_arr, tof_s: TOF_DAYS * 86_400.0, mu: MU_SUN_M3S2 };
    let sol = lambert.solve().expect("Lambert solve failed");

    let r_dep_v = nalgebra::Vector3::new(r_dep[0], r_dep[1], r_dep[2]);
    let v_dep_v = nalgebra::Vector3::new(v_dep[0], v_dep[1], v_dep[2]);
    let v_transfer_dep_v = nalgebra::Vector3::new(sol.v_transfer_dep[0], sol.v_transfer_dep[1], sol.v_transfer_dep[2]);
    let v_inf_vec = v_transfer_dep_v - v_dep_v;

    println!(
        "Real Lambert-derived v-infinity at departure: {:.3} km/s  (C3 = {:.3} km^2/s^2)",
        v_inf_vec.norm() / 1e3, sol.c3_km2s2,
    );

    // Earth's own Laplace SOI radius relative to the Sun (real heliocentric
    // distance at departure) — same convention `design.rs::propagator_body_entries`
    // uses for any body whose primary is the Sun.
    let earth_dist_m = r_dep_v.norm();
    let soi_radius_m = laplace_soi_radius_m(earth_dist_m, EARTH_MU / MU_SUN_M3S2);
    println!("Earth's SOI radius: {:.0} km (at {:.4} AU from the Sun)", soi_radius_m / 1e3, earth_dist_m / 1.495_978_707e11);

    let almanac_ref = &almanac;
    let earth_state_at = move |t_abs_s: f64| {
        let t = dep_epoch + Duration::from_seconds(t_abs_s);
        let st = almanac_ref.body_state_heliocentric(Body::Earth, t).expect("Earth state");
        (
            nalgebra::Vector3::new(st.position.inner.x, st.position.inner.y, st.position.inner.z),
            nalgebra::Vector3::new(st.velocity.inner.x, st.velocity.inner.y, st.velocity.inner.z),
        )
    };

    let earth_body = PropagatorBody {
        name: "Earth",
        mu_m3s2: EARTH_MU,
        soi_radius_m: Some(soi_radius_m),
        state_at: &earth_state_at,
        central_fidelity: None,
        radius_m: Some(EARTH_RADIUS_M),
    };

    let escape = propagate_escape_leg(
        r_dep_v, v_dep_v, 0, R_PARK_M, v_inf_vec, MU_SUN_M3S2, &[earth_body], 1e-12, 1e-3,
    )
    .expect("escape leg should clear Earth's SOI within the search window for a real interplanetary departure C3");

    let v_circ = (EARTH_MU / R_PARK_M).sqrt();
    println!(
        "\nParking orbit: r_park = {:.1} km, v_circ = {:.3} km/s",
        R_PARK_M / 1e3, v_circ / 1e3,
    );
    println!(
        "Escape burn (real patched-conic, this demo's construction): {:.3} km/s",
        escape.dv_escape_ms / 1e3,
    );
    println!(
        "Real escape duration (parking orbit -> Earth SOI exit): {:.3} days ({} points)",
        escape.escape_duration_s / 86_400.0, escape.points.len(),
    );

    // Confirm the SOI-exit velocity direction/magnitude is in the right
    // ballpark of (but, per propagate_escape_leg's doc comment, not exactly
    // equal to) the idealized v-infinity -- a real, finite-radius SOI exit
    // happens well before full asymptotic convergence.
    let exit_v_helio = escape.exit_v_mps - earth_state_at(escape.escape_duration_s).1;
    let speed_residual = (exit_v_helio.norm() - v_inf_vec.norm()).abs();
    let angle_residual_deg = exit_v_helio.normalize().dot(&v_inf_vec.normalize()).clamp(-1.0, 1.0).acos().to_degrees();
    println!(
        "SOI-exit velocity (Earth-relative) vs. idealized v-infinity: speed residual {:.1} m/s, angle residual {:.4} deg \
         (expected to be nonzero -- this is exactly why the optimizer's cruise leg re-uses the idealized v-infinity, \
         not this exit velocity, for Lambert-targeted propagation)",
        speed_residual, angle_residual_deg,
    );

    let out_dir = "out/departure_demo";
    fs::create_dir_all(out_dir).expect("create out dir");

    // Earth-centered coordinates for the close-up visualization -- at
    // heliocentric scale the whole escape leg is an invisible point next to
    // Earth's ~1 AU position, so the meaningful frame for plotting "zoom in
    // on departure" is relative to Earth itself, not the Sun.
    let mut rows = vec!["t_s,x_m,y_m,z_m,dist_from_earth_km".to_string()];
    for p in &escape.points {
        let (earth_now, _) = earth_state_at(p.t_s);
        let r_rel = p.r_m - earth_now;
        rows.push(format!("{:.3},{:.6e},{:.6e},{:.6e},{:.3}", p.t_s, r_rel.x, r_rel.y, r_rel.z, r_rel.norm() / 1e3));
    }
    let traj_path = format!("{out_dir}/escape_trajectory.csv");
    fs::write(&traj_path, rows.join("\n") + "\n").expect("write escape_trajectory.csv");
    println!("\n{traj_path}  ({} points, Earth-centered)", escape.points.len());

    let meta_path = format!("{out_dir}/meta.csv");
    fs::write(
        &meta_path,
        format!(
            "r_park_m,soi_radius_m,escape_duration_s,dv_escape_ms\n{R_PARK_M:.6e},{soi_radius_m:.6e},{:.3},{:.3}\n",
            escape.escape_duration_s, escape.dv_escape_ms,
        ),
    )
    .expect("write meta.csv");
    println!("{meta_path}");
}
