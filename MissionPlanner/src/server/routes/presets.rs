//! GET /api/presets — example mission config library embedded at compile time.
//!
//! All TOML files from `MissionPlanner/config/` are baked in via `include_str!`,
//! parsed as `toml::Value`, and re-serialised to JSON so the frontend can
//! populate a preset-picker without hand-duplicating config in TypeScript.

use axum::Json;
use serde_json::{json, Value};

/// (id, raw TOML source). Embedded at compile time — no runtime filesystem dependency.
static PRESET_SOURCES: &[(&str, &str)] = &[
    ("apophis_orbit",             include_str!("../../../config/apophis_orbit.toml")),
    ("bennu_landing",             include_str!("../../../config/bennu_landing.toml")),
    ("bennu_sample_return",       include_str!("../../../config/bennu_sample_return.toml")),
    ("cassini2_gtop",             include_str!("../../../config/cassini2_gtop.toml")),
    ("custom_asteroid",           include_str!("../../../config/custom_asteroid.toml")),
    ("earth_leo",                 include_str!("../../../config/earth_leo.toml")),
    ("earth_neptune_auto",        include_str!("../../../config/earth_neptune_auto.toml")),
    ("europa_orbit",              include_str!("../../../config/europa_orbit.toml")),
    ("evj_flyby",                 include_str!("../../../config/evj_flyby.toml")),
    ("jupiter_flyby",             include_str!("../../../config/jupiter_flyby.toml")),
    ("lunar_orbit",               include_str!("../../../config/lunar_orbit.toml")),
    ("mars_flyby",                include_str!("../../../config/mars_flyby.toml")),
    ("mars_ga",                   include_str!("../../../config/mars_ga.toml")),
    ("mars_jupiter_flyby",        include_str!("../../../config/mars_jupiter_flyby.toml")),
    ("mars_monte_carlo",          include_str!("../../../config/mars_monte_carlo.toml")),
    ("mars_orbit",                include_str!("../../../config/mars_orbit.toml")),
    ("mars_orbiter_launch",       include_str!("../../../config/mars_orbiter_launch.toml")),
    ("mars_pso",                  include_str!("../../../config/mars_pso.toml")),
    ("mars_round_trip_outbound",  include_str!("../../../config/mars_round_trip_outbound.toml")),
    ("mars_round_trip_return",    include_str!("../../../config/mars_round_trip_return.toml")),
    ("veega_flyby",               include_str!("../../../config/veega_flyby.toml")),
    ("venus_orbit",               include_str!("../../../config/venus_orbit.toml")),
    ("venus_saturn_auto",         include_str!("../../../config/venus_saturn_auto.toml")),
];

pub async fn presets() -> Json<Value> {
    let entries: Vec<Value> = PRESET_SOURCES
        .iter()
        .filter_map(|(id, src)| {
            let tv: toml::Value = toml::from_str(src).ok()?;
            let cfg: Value = serde_json::to_value(&tv).ok()?;
            let name = cfg["mission"]["name"].as_str().unwrap_or(id).to_string();
            let description = build_description(&cfg);
            Some(json!({
                "id":          id,
                "name":        name,
                "description": description,
                "config":      cfg,
            }))
        })
        .collect();

    Json(json!({ "presets": entries }))
}

/// Derive a one-line description from the parsed config fields.
///
/// `[trajectory].solver` is the narrowing-stage solver (Hohmann/Lambert/GA/
/// PSO/...) and is present on every config, but it's vestigial/misleading
/// for a config whose real method is Phase 9's `[optimization].method` (GA/
/// PSO/MGA under real propagated dynamics) — an MGA benchmark preset like
/// `veega_flyby.toml` still carries `solver = "Lambert"` from its narrowing-
/// stage section, which would otherwise show as "(Lambert)" for a mission
/// that's actually MGA. Prefer `[optimization].method` when present.
fn build_description(cfg: &Value) -> String {
    let objective = cfg["mission"]["objective"].as_str().unwrap_or("Unknown");
    let target    = cfg["target_body"]["name"].as_str().unwrap_or("Unknown");
    let dep_body  = cfg["trajectory"]["departure_body"].as_str().unwrap_or("Earth");
    let solver = cfg["optimization"]["method"].as_str()
        .unwrap_or_else(|| cfg["trajectory"]["solver"].as_str().unwrap_or("Lambert"));
    format!("{objective} — {dep_body} → {target} ({solver})")
}
