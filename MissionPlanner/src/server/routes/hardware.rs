//! GET /api/hardware — return the hardware catalog (RW, thruster, sensors).

use axum::Json;
use serde_json::{json, Value};
use hardware_catalog::{
    DsnLinkSpec, ImuSpec, LandmarkSensorSpec, LaunchVehicleSpec, LidarSpec, OpNavCameraSpec,
    ReactionWheelSpec, StarTrackerSpec, ThrusterSpec,
};

pub async fn hardware() -> Json<Value> {
    let reaction_wheels: Vec<Value> = ReactionWheelSpec::catalog()
        .into_iter()
        .map(|w| json!({
            "name":             w.name,
            "max_torque_nm":    w.max_torque_nm,
            "max_momentum_nms": w.max_momentum_nms(),
            "max_speed_rads":   w.max_speed_rads,
            "inertia_kgm2":     w.inertia_kgm2,
            "mass_kg":          w.mass_kg,
            "power_w":          w.power_w,
        }))
        .collect();

    let thrusters: Vec<Value> = ThrusterSpec::catalog()
        .into_iter()
        .map(|t| json!({
            "name":        t.name,
            "thrust_n":    t.thrust_n,
            "isp_s":       t.isp_s,
            "min_pulse_s": t.min_pulse_s,
            "power_w":     t.power_w,
        }))
        .collect();

    let star_trackers: Vec<Value> = StarTrackerSpec::catalog()
        .into_iter()
        .map(|s| json!({
            "name":      s.name,
            "noise_rad": s.noise_rad,
            "mass_kg":   s.mass_kg,
            "power_w":   s.power_w,
        }))
        .collect();

    let opnav_cameras: Vec<Value> = OpNavCameraSpec::catalog()
        .into_iter()
        .map(|c| json!({
            "name":                   c.name,
            "bearing_noise_rad":      c.bearing_noise_rad,
            "angular_size_noise_rad": c.angular_size_noise_rad,
            "mass_kg":                c.mass_kg,
            "power_w":                c.power_w,
        }))
        .collect();

    let imus: Vec<Value> = ImuSpec::catalog()
        .into_iter()
        .map(|i| json!({
            "name":          i.name,
            "dv_noise_mps":  i.dv_noise_mps,
            "mass_kg":       i.mass_kg,
            "power_w":       i.power_w,
        }))
        .collect();

    let lidars: Vec<Value> = LidarSpec::catalog()
        .into_iter()
        .map(|l| json!({
            "name":           l.name,
            "range_noise_m":  l.range_noise_m,
            "max_range_m":    l.max_range_m,
            "mass_kg":        l.mass_kg,
            "power_w":        l.power_w,
        }))
        .collect();

    let landmark_sensors: Vec<Value> = LandmarkSensorSpec::catalog()
        .into_iter()
        .map(|l| json!({
            "name":              l.name,
            "bearing_noise_rad": l.bearing_noise_rad,
            "catalog_size":      l.catalog_size,
            "mass_kg":           l.mass_kg,
            "power_w":           l.power_w,
        }))
        .collect();

    let dsn_links: Vec<Value> = DsnLinkSpec::catalog()
        .into_iter()
        .map(|d| json!({
            "name":                  d.name,
            "range_noise_m":         d.range_noise_m,
            "range_rate_noise_mps":  d.range_rate_noise_mps,
            "ddor_noise_rad":        d.ddor_noise_rad,
            "mass_kg":               d.mass_kg,
            "power_w":               d.power_w,
        }))
        .collect();

    // Launch vehicles — only meaningful for an Earth departure (see
    // the design notes "Launch Vehicle Selection"). `performance_points` is the raw
    // (C3 [km²/s²], injected mass [kg]) curve this catalog's
    // `injected_mass_kg()` interpolates over — exposed as-is rather than
    // pre-sampled, so the frontend can render the real verified points
    // (e.g. as a curve/markers) instead of a derived approximation.
    let launch_vehicles: Vec<Value> = LaunchVehicleSpec::catalog()
        .into_iter()
        .map(|lv| json!({
            "name": lv.name,
            "performance_points": lv.performance_points
                .iter()
                .map(|&(c3_km2s2, mass_kg)| json!({ "c3_km2s2": c3_km2s2, "injected_mass_kg": mass_kg }))
                .collect::<Vec<_>>(),
        }))
        .collect();

    Json(json!({
        "reaction_wheels":  reaction_wheels,
        "thrusters":        thrusters,
        "star_trackers":    star_trackers,
        "opnav_cameras":    opnav_cameras,
        "imus":             imus,
        "lidars":           lidars,
        "landmark_sensors": landmark_sensors,
        "dsn_links":        dsn_links,
        "launch_vehicles":  launch_vehicles,
    }))
}
