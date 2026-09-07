//! GET /api/bodies — return the full body catalog as JSON.
//! GET /api/bodies/{name}/state?epoch=... — body's heliocentric state vector
//! at a given epoch (ANISE-covered bodies only — see `state`).

use axum::extract::{Path, Query};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use ephemeris::Almanac;
use crate::design::anise_body_available;

pub async fn bodies() -> Json<Value> {
    let catalog: Vec<Value> = body_models::TargetBody::catalog()
        .into_iter()
        .map(|b| {
            let (gravity_model, j2, j3, j4) = match &b.gravity {
                body_models::GravityModel::PointMass => ("PointMass", None, None, None),
                body_models::GravityModel::J2 { j2 } => ("J2", Some(*j2), None, None),
                body_models::GravityModel::J2J3J4 { j2, j3, j4 } => {
                    ("J2J3J4", Some(*j2), Some(*j3), Some(*j4))
                }
            };
            let atmosphere = match &b.atmosphere {
                body_models::AtmosphereModel::None => "None",
                body_models::AtmosphereModel::Exponential { .. } => "Exponential",
            };
            // Heliocentric orbital elements:
            // real catalog-served orbit shapes so the frontend can delete
            // its hardcoded client-side element table and draw real rings
            // for small bodies too. `null` for the Sun and for moons
            // (primary-relative elements are a different contract — see
            // `body_models::TargetBody::orbital_elements`'s doc comment).
            let orbital_elements = b.orbital_elements.map(|oe| json!({
                "sma_m":             oe.sma_m,
                "eccentricity":      oe.eccentricity,
                "inclination_deg":   oe.inclination_deg,
                "raan_deg":          oe.raan_deg,
                "arg_periapsis_deg": oe.arg_periapsis_deg,
                "mean_anomaly_deg":  oe.mean_anomaly_deg,
                "epoch_jd":          oe.epoch_jd,
            }));
            json!({
                "name":            b.name,
                "kind":            b.kind,
                "anise_covered":   anise_body_available(&b.name.to_lowercase()),
                "mu_m3s2":         b.mu_m3s2,
                "radius_m":        b.radius_m,
                "gravity_model":   gravity_model,
                "j2":              j2,
                "j3":              j3,
                "j4":              j4,
                "atmosphere":      atmosphere,
                "spin_rate_rads":  b.spin_rate_rads,
                "pole_ra_deg":     b.pole_ra_deg,
                "pole_dec_deg":    b.pole_dec_deg,
                "primary":         b.primary,
                "sma_m":           b.sma_m,
                "orbital_elements": orbital_elements,
            })
        })
        .collect();

    Json(json!({ "bodies": catalog }))
}

#[derive(Debug, Deserialize)]
pub struct StateQuery {
    /// "YYYY-MM-DDTHH:MM:SS UTC" (or without the trailing " UTC") — same
    /// format `crate::design::parse_epoch` already parses for
    /// `[trajectory].departure_epoch`.
    pub epoch: String,
}

/// GET /api/bodies/{name}/state?epoch=... — heliocentric state vector (SI
/// units) for a catalog body at a given epoch.
///
/// Bodies covered by the always-present `de440s.bsp` (Sun, planets, Moon)
/// always have a real ephemeris-backed answer. Phobos, Deimos, Europa, and
/// Titan additionally require their own satellite kernel to be deployed on
/// this server (`mar099s.bsp` / `jup365.bsp` / `sat441.bsp` — see
/// `design::load_almanac`); `anise_body_available` checks for that file, so
/// a deployment missing one of those kernels gets a 422 for the affected
/// moon rather than a 500. Small bodies in the catalog (Bennu, Apophis,
/// Ryugu, Eros, Didymos, ...) have no DE440S coverage and no fixed orbital
/// elements in `body_models` (their position is epoch-dependent and only
/// known if the requesting mission's own TOML supplies
/// `[target_body.keplerian_orbit]`, which this generic endpoint has no
/// access to) — those return a clear 422 rather than silently guessing.
pub async fn state(Path(name): Path<String>, Query(q): Query<StateQuery>) -> Response {
    if body_models::TargetBody::by_name(&name).is_none() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": format!(
                    "'{name}' is not in the body catalog. See GET /api/bodies for the full list."
                )
            })),
        )
            .into_response();
    }

    let epoch = match crate::design::parse_epoch(&q.epoch) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("invalid epoch '{}': {e}", q.epoch) })),
            )
                .into_response();
        }
    };

    if !anise_body_available(&name) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({
                "error": format!(
                    "'{name}' has no ANISE ephemeris coverage on this server. Major bodies \
                     (Sun, Mercury, Venus, Earth, Moon, Mars, Jupiter, Saturn) always have a \
                     generically-available state; Phobos, Deimos, Europa, and Titan are \
                     covered only when their satellite kernel is present on the server \
                     (mar099s.bsp / jup365.bsp / sat441.bsp — see load_almanac). Other \
                     small bodies (Bennu, Apophis, Ryugu, Eros, Didymos, ...) only have a \
                     position when the requesting mission's own TOML supplies \
                     [target_body.keplerian_orbit], which this endpoint does not have \
                     access to."
                )
            })),
        )
            .into_response();
    }
    // Safe: anise_body_available already confirmed anise_body(name) is Some.
    let anise_name = crate::design::anise_body(&name.to_lowercase()).unwrap();

    let almanac = match crate::design::load_almanac() {
        Some(a) => a,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "ANISE kernel (de440s.bsp) failed to load on the server." })),
            )
                .into_response();
        }
    };

    match state_for(&almanac, anise_name, epoch) {
        Ok((r, v)) => Json(json!({
            "name":    name,
            "epoch":   q.epoch,
            "x_m":     r[0],
            "y_m":     r[1],
            "z_m":     r[2],
            "vx_mps":  v[0],
            "vy_mps":  v[1],
            "vz_mps":  v[2],
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("ANISE query failed: {e}") })),
        )
            .into_response(),
    }
}

/// Heliocentric position/velocity [m], [m/s] for an ANISE body at an epoch.
fn state_for(
    almanac: &Almanac,
    body: ephemeris::Body,
    epoch: ephemeris::Epoch,
) -> Result<([f64; 3], [f64; 3]), String> {
    let st = almanac.body_state_heliocentric(body, epoch).map_err(|e| e.to_string())?;
    let r = [st.position.inner[0], st.position.inner[1], st.position.inner[2]];
    let v = [st.velocity.inner[0], st.velocity.inner[1], st.velocity.inner[2]];
    Ok((r, v))
}
