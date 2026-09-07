//! POST /api/design/vehicle — derived mass properties + per-plate SRP force
//! vectors for a built spacecraft configuration.
//!
//! Part of the spacecraft configuration builder. See
//! `crate::vehicle_properties` for the mass/CoM/inertia computation itself
//! and `docs/MP/MANUAL.md` §6.4 for the governing math — this route is
//! thin plumbing (JSON in/out) around that module plus
//! `orbital_models::flat_plate_force_per_plate` for the SRP vectors.
//!
//! Request body: `{ config: MissionConfig, sun_hat_body: [f64;3] | null,
//! sun_distance_m: number | null }`. `sun_hat_body` is the spacecraft→Sun
//! unit direction ALREADY IN THE BODY FRAME — this route does not resolve
//! attitude or heliocentric geometry itself, it takes the direction as
//! given (the caller — e.g. the frontend's mission-timeline scrubber —
//! already has that context). Omitting it (`null`) skips the `srp` section
//! of the response entirely (mass properties are still returned).
//! `sun_distance_m` defaults to 1 AU when omitted (only affects radiation
//! pressure magnitude, not direction).

use axum::{http::StatusCode, response::{IntoResponse, Response}, Json};
use nalgebra::Vector3;
use serde::Deserialize;
use serde_json::json;

use crate::config::MissionConfig;
use crate::vehicle_properties::compute_vehicle_properties;

#[derive(Deserialize)]
pub struct VehicleRequest {
    config: MissionConfig,
    sun_hat_body: Option<[f64; 3]>,
    sun_distance_m: Option<f64>,
}

pub async fn vehicle(
    body: Result<Json<VehicleRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let req = match body {
        Ok(Json(r)) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("JSON parse error: {e}") })),
            )
                .into_response();
        }
    };

    let vp = compute_vehicle_properties(&req.config);

    let contributions: Vec<_> = vp
        .contributions
        .iter()
        .map(|c| {
            json!({
                "label": c.label,
                "mass_kg": c.mass_kg,
                "position_m": c.position_m,
                "inertia_about_com_kgm2": c.inertia_about_com_kgm2,
            })
        })
        .collect();

    let srp = req.sun_hat_body.map(|sun_hat| {
        let plates = crate::simulate::build_plates(&req.config);
        let sun_hat_body = {
            let v = Vector3::new(sun_hat[0], sun_hat[1], sun_hat[2]);
            let n = v.norm();
            if n > 1e-9 { v / n } else { Vector3::new(1.0, 0.0, 0.0) }
        };
        let dist_m = req.sun_distance_m.unwrap_or(orbital_models::constants::AU);
        let p_srp = orbital_models::pressure_at(dist_m);
        let forces = orbital_models::flat_plate_force_per_plate(&plates, &sun_hat_body, p_srp);
        let total: Vector3<f64> = forces.iter().sum();

        // Center of pressure: the force-magnitude-weighted centroid of
        // illuminated plates, for THIS sun direction — this, not a static
        // geometric centroid, is the point the SRP torque arm (CoP - CoM)
        // is actually measured from; it moves as the sun direction changes.
        // Unilluminated plates (force_n == 0) contribute zero weight, so
        // they drop out naturally rather than needing an explicit filter.
        let weight_total: f64 = forces.iter().map(|f| f.norm()).sum();
        let cop_m = if weight_total > 1e-12 {
            let mut c = Vector3::zeros();
            for (pl, f) in plates.iter().zip(forces.iter()) {
                c += f.norm() * pl.center_body;
            }
            c /= weight_total;
            Some([c.x, c.y, c.z])
        } else {
            None
        };

        let plates_json: Vec<_> = plates
            .iter()
            .zip(forces.iter())
            .map(|(pl, f)| {
                json!({
                    "normal": [pl.normal.x, pl.normal.y, pl.normal.z],
                    "area_m2": pl.area,
                    "center_body_m": [pl.center_body.x, pl.center_body.y, pl.center_body.z],
                    "force_n": [f.x, f.y, f.z],
                })
            })
            .collect();

        json!({
            "sun_hat_body": [sun_hat_body.x, sun_hat_body.y, sun_hat_body.z],
            "p_srp_nm2": p_srp,
            "cop_m": cop_m,
            "plates": plates_json,
            "total_force_n": [total.x, total.y, total.z],
        })
    });

    // Review E5, backend half: static main-engine-vs-RCS
    // feasibility. During a burn the main engine (body +X, mounted at the
    // -X face — `cruise.rs::BurnConfig`'s fixed convention) produces a real
    // disturbance torque `τ_dist = r_offset × F_thrust` about the CoM (the
    // SAME arm the live simulation flies with,
    // `cruise::main_engine_thrust_offset_body_m`). RCS must cancel it to
    // hold burn attitude (`ControlMode::ThrustersPrimary`). If the placed
    // layout's achievable torque along the cancellation axis is below the
    // disturbance, every burn tumbles the vehicle BY PHYSICS, not by bug —
    // this check surfaces that before a run, for the Phase 02 builder and
    // the pre-Phase-03 gate. Static geometry only (no dynamics): the same
    // honesty tier as the rest of this route.
    let main_engine_torque_check = req.config.spacecraft.propulsion.as_ref().map(|prop| {
        let offset = crate::cruise::main_engine_thrust_offset_body_m(&req.config);
        let thrust_force_body = Vector3::new(prop.thrust_n, 0.0, 0.0);
        let tau_dist = offset.cross(&thrust_force_body);
        let tau_dist_nm = tau_dist.norm();
        let (rcs_thrusters, _) = crate::simulate::rcs_from_hardware(&req.config);
        // Achievable RCS torque along the CANCELLATION direction (−τ̂):
        // select the aligned thruster set exactly the way the live
        // allocator does, then project its full-duty net torque onto that
        // axis (off-axis components don't help hold the burn attitude).
        let rcs_authority_nm = if tau_dist_nm > 1e-12 && !rcs_thrusters.is_empty() {
            let cancel_dir = -tau_dist / tau_dist_nm;
            let (_f, tau_full, _t) = sim_engine::thruster_selection(&cancel_dir, &rcs_thrusters);
            tau_full.dot(&cancel_dir).max(0.0)
        } else if !rcs_thrusters.is_empty() {
            // No disturbance to cancel — report the layout's weakest-axis
            // authority anyway (min over ±x/±y/±z probes) as a general
            // capability number.
            let probes = [
                Vector3::x(), -Vector3::x(), Vector3::y(), -Vector3::y(), Vector3::z(), -Vector3::z(),
            ];
            probes
                .iter()
                .map(|d| sim_engine::thruster_selection(d, &rcs_thrusters).1.dot(d).max(0.0))
                .fold(f64::INFINITY, f64::min)
        } else {
            0.0
        };
        let ratio = if rcs_authority_nm > 1e-12 { Some(tau_dist_nm / rcs_authority_nm) } else { None };
        let feasible = tau_dist_nm <= 1e-12 || ratio.is_some_and(|r| r <= 1.0);
        json!({
            "thrust_offset_body_m": [offset.x, offset.y, offset.z],
            "disturbance_torque_nm": tau_dist_nm,
            "rcs_authority_nm": rcs_authority_nm,
            // disturbance / authority — > 1.0 means this engine/RCS
            // combination cannot hold attitude during a burn. null when
            // the layout has zero authority along the needed axis (worse
            // than any finite ratio) while a real disturbance exists.
            "ratio": ratio,
            "feasible": feasible,
        })
    });

    Json(json!({
        "mass_kg": vp.total_mass_kg,
        "com_m": vp.com_m,
        "inertia_kgm2": vp.inertia_kgm2,
        "contributions": contributions,
        "warnings": vp.warnings,
        "srp": srp,
        "main_engine_torque_check": main_engine_torque_check,
    }))
    .into_response()
}
