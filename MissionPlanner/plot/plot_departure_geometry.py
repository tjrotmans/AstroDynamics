"""
MissionPlanner Phase 9 optimizer — departure burn geometry diagnostic.

Three panels, three views:
- Left: burn close-up, body-centered (km), origin at the departure body,
  parking-orbit scale (~r_park). Visualizes exactly what `theta_burn` and
  `phi_out_of_plane` mean in
  `crates/trajectory_solver/src/departure.rs::circular_orbit_burn_state` --
  the parking-orbit ring, `reference_dir` (theta=0, the departure body's own
  instantaneous heliocentric radial direction), `orbital_plane_normal`,
  `departure_body_velocity_hat` (the departure body's own heliocentric velocity direction --
  NOT something the spacecraft's geocentric velocity converges to; it's an
  unrelated reference vector, drawn here only so this frame's orientation
  can be checked directly against the heliocentric panel), the pure-circular
  velocity vs. the actual departure velocity at the burn point (their angle
  *is* phi, length difference *is* dv), `v_infinity_asymptotic` (the real
  escape direction -- the periapsis velocity rotated by the hyperbola's
  turning angle `asin(1/e)`, NOT the periapsis velocity itself), and
  `v_transfer_hat` (`v_dep + v_infinity`, direction only -- this is the one
  that should match the heliocentric panel's `v_infinity_helio`, not
  `v_departure_actual_best`, which is a different, geocentric-only quantity).
- Middle: departure leg + real SOI crossing, body-centered (km), SOI scale
  (~10^5-10^6 km, much larger than the burn close-up). The actual propagated
  escape trajectory (same physics `evaluate_candidate` uses, run standalone
  for a short window -- this is why `mission-planner geometry` doesn't need
  to re-run the GA/PSO search at all), the real Laplace SOI sphere, the
  actual crossing point, and the real geocentric velocity there vs. the
  idealized `v_infinity_asymptotic` evaluated at the same point and scale --
  the residual between them (known nonzero, see `departure_demo.rs`) is
  directly visible here, not just asserted.
- Right: heliocentric (AU), origin at the Sun. The same escape asymptote
  composed with the departure body's own heliocentric velocity
  (`v_infinity_helio`), alongside the Sun, the departure body, and the
  target body's real position both at departure and at the achieved
  arrival/closest-approach time.

Reads: MissionPlanner/out/<mission>/optimize/{ga,pso}_departure_geometry_orbit.csv
       MissionPlanner/out/<mission>/optimize/{ga,pso}_departure_geometry_vectors.csv
       MissionPlanner/out/<mission>/optimize/{ga,pso}_departure_geometry_helio.csv
       MissionPlanner/out/<mission>/optimize/{ga,pso}_departure_geometry_real_leg.csv
       MissionPlanner/out/<mission>/optimize/{ga,pso}_departure_geometry_soi.csv

Produces (into the same directory): <method>_departure_geometry.html

Usage (from MissionPlanner/ directory):
  python plot/plot_departure_geometry.py scratch_optimize_repro

To regenerate the underlying data without re-running the GA/PSO search:
  cargo run --bin mission-planner --release -- geometry config/<mission>.toml
"""

import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

mission = sys.argv[1] if len(sys.argv) > 1 else "mars_flyby"

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / mission / "optimize"

method = None
for m in ("ga", "pso"):
    if (DATA / f"{m}_departure_geometry_orbit.csv").exists():
        method = m
        break

if method is None:
    print(f"Not found: {DATA}/{{ga,pso}}_departure_geometry_orbit.csv")
    print("Run: cargo run --bin mission-planner --release -- optimize config/<mission>.toml")
    sys.exit(1)

orbit = pd.read_csv(DATA / f"{method}_departure_geometry_orbit.csv")
vectors = pd.read_csv(DATA / f"{method}_departure_geometry_vectors.csv")
helio_path = DATA / f"{method}_departure_geometry_helio.csv"
helio = pd.read_csv(helio_path) if helio_path.exists() else None
real_leg_path = DATA / f"{method}_departure_geometry_real_leg.csv"
real_leg = pd.read_csv(real_leg_path) if real_leg_path.exists() else None
soi_path = DATA / f"{method}_departure_geometry_soi.csv"
soi = pd.read_csv(soi_path).iloc[0] if soi_path.exists() else None

BG = "#0f0f0f"

fig = make_subplots(
    rows=1, cols=3,
    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
    subplot_titles=("Burn close-up (body-centered, km)", "Departure leg + SOI crossing (body-centered, km)", "Heliocentric (AU)"),
)

fig.add_trace(go.Scatter3d(
    x=orbit.x_km, y=orbit.y_km, z=orbit.z_km, mode="lines",
    line=dict(color="#555555", width=3), name="Parking orbit ring",
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode="markers", marker=dict(size=8, color="#3a7bd5"), name="Departure body",
), row=1, col=1)

colors = {
    "reference_dir_theta0": "#FFD700",
    "orbital_plane_normal": "#7ED957",
    "departure_body_velocity_hat": "#FF00FF",
    "burn_position": "#FF6B35",
    "v_circular_only_ref": "#00d4ff",
    "v_departure_actual_best": "#FFA94D",
    "v_infinity_asymptotic": "#FF3366",
    "v_transfer_hat": "#B0FFB0",
    "v_at_soi_crossing_real": "#00d4ff",
    "v_infinity_asymptotic_at_soi": "#FF3366",
}

# Heliocentric J2000/ICRF reference axes (X toward vernal equinox, Z toward
# the north celestial pole at J2000.0 -- see crates/ephemeris/src/frames.rs).
# This panel's origin is shifted to the departure body's center (per
# the design notes "frame transformation is an inertial origin shift only, never a
# rotation" rule), so these axes are *not* rotated relative to the true
# heliocentric frame -- they're the same fixed directions, just drawn from a
# different origin. Scaled a bit longer than the orbit ring so they're
# visually distinguishable from reference_dir/orbital_plane_normal.
axis_len_km = orbit.x_km.abs().max() * 1.6 if len(orbit) else 1.0
axis_colors = {"X": "#aa3355", "Y": "#3355aa", "Z": "#55aa33"}
for name, vec in (("X", (1, 0, 0)), ("Y", (0, 1, 0)), ("Z", (0, 0, 1))):
    x1, y1, z1 = (c * axis_len_km for c in vec)
    fig.add_trace(go.Scatter3d(
        x=[0, x1], y=[0, y1], z=[0, z1], mode="lines+text",
        line=dict(color=axis_colors[name], width=4, dash="dot"),
        text=["", f"{name} (J2000/ICRF)"], textposition="top center",
        name=f"{name} (J2000/ICRF)",
    ), row=1, col=1)

# `_at_soi` rows belong to the SOI-scale middle panel (col=2), not the
# parking-orbit-scale burn close-up (col=1) -- their coordinates are at the
# SOI's ~10^5-10^6 km scale and would make the close-up panel unreadable.
soi_scale_names = {"v_at_soi_crossing_real", "v_infinity_asymptotic_at_soi"}
for _, row in vectors.iterrows():
    target_col = 2 if row["name"] in soi_scale_names else 1
    color = colors.get(row["name"], "#dddddd")
    if row["name"] == "burn_position":
        fig.add_trace(go.Scatter3d(
            x=[row.x1_km], y=[row.y1_km], z=[row.z1_km], mode="markers+text",
            marker=dict(size=6, color=color), text=["Burn position"], textposition="top center",
            name=row["name"],
        ), row=1, col=target_col)
        continue
    fig.add_trace(go.Scatter3d(
        x=[row.x0_km, row.x1_km], y=[row.y0_km, row.y1_km], z=[row.z0_km, row.z1_km],
        mode="lines+markers+text",
        line=dict(color=color, width=6),
        marker=dict(size=[2, 5], color=color),
        text=["", row["name"]], textposition="top center",
        name=row["name"],
    ), row=1, col=target_col)

# Middle panel: real propagated escape leg, SOI wireframe sphere, and the
# real crossing point -- all body-centered (km), at SOI scale.
if real_leg is not None:
    fig.add_trace(go.Scatter3d(
        x=real_leg.x_km, y=real_leg.y_km, z=real_leg.z_km, mode="lines",
        line=dict(color="#00d4ff", width=4), name="Real propagated escape leg",
    ), row=1, col=2)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers+text", marker=dict(size=8, color="#3a7bd5"),
        text=["Departure body"], textposition="bottom center", name="Departure body (SOI panel)",
    ), row=1, col=2)
if soi is not None:
    soi_radius_km = soi.soi_radius_km
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    sx = soi_radius_km * np.cos(u) * np.sin(v)
    sy = soi_radius_km * np.sin(u) * np.sin(v)
    sz = soi_radius_km * np.cos(v)
    fig.add_trace(go.Surface(
        x=sx, y=sy, z=sz, opacity=0.10, colorscale=[[0, "#00d4ff"], [1, "#00d4ff"]],
        showscale=False, name=f"SOI ({soi_radius_km:.0f} km)",
    ), row=1, col=2)
    if bool(soi.crossed):
        fig.add_trace(go.Scatter3d(
            x=[soi.crossing_x_km], y=[soi.crossing_y_km], z=[soi.crossing_z_km],
            mode="markers+text", marker=dict(size=6, color="#FFD700"),
            text=[f"SOI crossing (t={soi.crossing_t_s / 86400:.2f} d)"], textposition="top center",
            name="SOI crossing",
        ), row=1, col=2)

# Heliocentric panel: Sun, departure body, target body at departure and at
# the achieved arrival/closest-approach epoch, and the escape asymptote
# (v_infinity, composed with the departure body's own heliocentric
# velocity) drawn from the departure body's position.
if helio is not None:
    helio_idx = helio.set_index("name")
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers", marker=dict(size=10, color="#FFD700"), name="Sun",
    ), row=1, col=3)
    point_names = [n for n in helio_idx.index if not n.startswith("v_infinity_helio")]
    point_colors = {"_at_departure": "#3a7bd5", "_at_arrival": "#FF6B35"}
    for name in point_names:
        r = helio_idx.loc[name]
        color = "#3a7bd5"
        for suffix, c in point_colors.items():
            if name.endswith(suffix):
                color = c
        fig.add_trace(go.Scatter3d(
            x=[r.x_au], y=[r.y_au], z=[r.z_au], mode="markers+text",
            marker=dict(size=6, color=color), text=[name], textposition="top center", name=name,
        ), row=1, col=3)
    if "v_infinity_helio_origin" in helio_idx.index and "v_infinity_helio_tip" in helio_idx.index:
        o = helio_idx.loc["v_infinity_helio_origin"]
        t = helio_idx.loc["v_infinity_helio_tip"]
        fig.add_trace(go.Scatter3d(
            x=[o.x_au, t.x_au], y=[o.y_au, t.y_au], z=[o.z_au, t.z_au],
            mode="lines+markers+text", line=dict(color="#FF3366", width=6),
            marker=dict(size=[2, 5], color="#FF3366"),
            text=["", "v_infinity_helio (direction only, fixed-length arrow)"], textposition="top center",
            name="v_infinity_helio",
        ), row=1, col=3)
else:
    fig.add_annotation(
        text="No heliocentric geometry data (helio.csv not found)", showarrow=False,
        x=0.92, y=0.5, xref="paper", yref="paper", font=dict(color="#888888"),
    )

fig.update_layout(
    title=f"{mission} — departure burn geometry (theta_burn/phi_out_of_plane reference)",
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#dddddd"),
    height=700,
    scene=dict(
        xaxis_title="x [km]", yaxis_title="y [km]", zaxis_title="z [km]",
        xaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        yaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        zaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        aspectmode="data",
    ),
    scene2=dict(
        xaxis_title="x [km]", yaxis_title="y [km]", zaxis_title="z [km]",
        xaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        yaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        zaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        aspectmode="data",
    ),
    scene3=dict(
        xaxis_title="x [AU]", yaxis_title="y [AU]", zaxis_title="z [AU]",
        xaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        yaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        zaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        aspectmode="data",
    ),
    annotations=[dict(
        text="Left/middle frames: J2000/ICRF inertial axes, origin shifted to the departure body's center (not rotated; "
             "different scales -- left is parking-orbit scale, middle is SOI scale). Right frame: heliocentric J2000/ICRF, origin at the Sun.",
        xref="paper", yref="paper", x=0.5, y=-0.04, showarrow=False,
        font=dict(color="#999999", size=12),
    )],
)

out_path = DATA / f"{method}_departure_geometry.html"
fig.write_html(str(out_path))
print(f"Wrote {out_path}")
webbrowser.open(out_path.as_uri())
