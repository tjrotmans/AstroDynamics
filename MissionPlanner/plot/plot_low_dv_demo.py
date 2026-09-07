"""
TEMPORARY debug scratch -- not part of the project structure.

Plots the low-dv (TOF=298d, dv=3637 m/s) exact-Lambert-injection candidate
post propagator-bug-fix. Three panels:
- Left: Earth-centered close-up (km), real Earth SOI wireframe, departure.
- Middle: Mars-centered close-up (km), real Mars SOI wireframe, arrival/
  closest-approach -- same convention as the departure panel, mirrored for
  the arrival side.
- Right: heliocentric overview (AU), real miss vs Mars.

Reads: MissionPlanner/out/scratch_optimize_repro/optimize/low_dv_demo_*.csv
  (written by `cargo run -p mission_planner --bin debug_plot_low_dv --release`)

Usage (from MissionPlanner/ directory):
  python plot/plot_low_dv_demo.py
"""

import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / "scratch_optimize_repro" / "optimize"

traj = pd.read_csv(DATA / "low_dv_demo_trajectory.csv")
earth_track = pd.read_csv(DATA / "low_dv_demo_earth_track.csv")
mars_track = pd.read_csv(DATA / "low_dv_demo_mars_track.csv")
meta = pd.read_csv(DATA / "low_dv_demo_meta.csv").iloc[0]

AU = 1.495978707e11
BG = "#0f0f0f"

fig = make_subplots(
    rows=1, cols=3,
    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
    subplot_titles=(
        "Earth-centered close-up (km) -- departure",
        "Mars-centered close-up (km) -- closest approach",
        "Heliocentric overview (AU)",
    ),
)

# --- Left: Earth-centered close-up (first 25 days) ---
rel_x = (traj.x_m - earth_track.x_m) / 1e3
rel_y = (traj.y_m - earth_track.y_m) / 1e3
rel_z = (traj.z_m - earth_track.z_m) / 1e3

close_mask = traj.t_s < 25 * 86400
inside = close_mask & (traj.central_body == "Earth")
outside = close_mask & (traj.central_body != "Earth")

fig.add_trace(go.Scatter3d(
    x=rel_x[inside], y=rel_y[inside], z=rel_z[inside], mode="markers+lines",
    marker=dict(size=2, color="#FF6B35"), line=dict(color="#FF6B35", width=3),
    name="Earth-centered (inside SOI)",
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=rel_x[outside], y=rel_y[outside], z=rel_z[outside], mode="markers+lines",
    marker=dict(size=2, color="#00d4ff"), line=dict(color="#00d4ff", width=3),
    name="Heliocentric (outside Earth SOI)",
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode="markers+text", marker=dict(size=8, color="#3a7bd5"),
    text=["Earth"], textposition="bottom center", name="Earth",
), row=1, col=1)

earth_soi_km = meta.earth_soi_km
u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
fig.add_trace(go.Surface(
    x=earth_soi_km * np.cos(u) * np.sin(v), y=earth_soi_km * np.sin(u) * np.sin(v), z=earth_soi_km * np.cos(v),
    opacity=0.12, colorscale=[[0, "#888888"], [1, "#888888"]], showscale=False,
    name=f"Earth SOI ({earth_soi_km:.0f} km)",
), row=1, col=1)

# --- Middle: Mars-centered close-up, around the actual closest-approach time ---
mars_rel_x = (traj.x_m - mars_track.x_m) / 1e3
mars_rel_y = (traj.y_m - mars_track.y_m) / 1e3
mars_rel_z = (traj.z_m - mars_track.z_m) / 1e3
closest_t_s = meta.closest_t_s

# Window: +-15 days around closest approach, same "zoom into what's being
# verified" rationale as the departure panel.
arrival_mask = (traj.t_s > closest_t_s - 15 * 86400) & (traj.t_s < closest_t_s + 15 * 86400)
mars_soi_km = meta.mars_soi_km
inside_mars = arrival_mask & (traj.central_body == "Mars")
outside_mars = arrival_mask & (traj.central_body != "Mars")

fig.add_trace(go.Scatter3d(
    x=mars_rel_x[outside_mars], y=mars_rel_y[outside_mars], z=mars_rel_z[outside_mars], mode="markers+lines",
    marker=dict(size=2, color="#00d4ff"), line=dict(color="#00d4ff", width=3),
    name="Heliocentric (outside Mars SOI)",
), row=1, col=2)
fig.add_trace(go.Scatter3d(
    x=mars_rel_x[inside_mars], y=mars_rel_y[inside_mars], z=mars_rel_z[inside_mars], mode="markers+lines",
    marker=dict(size=3, color="#FF6B35"), line=dict(color="#FF6B35", width=4),
    name="Mars-centered (inside SOI)",
), row=1, col=2)
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode="markers+text", marker=dict(size=8, color="#C1440E"),
    text=["Mars"], textposition="bottom center", name="Mars",
), row=1, col=2)
fig.add_trace(go.Surface(
    x=mars_soi_km * np.cos(u) * np.sin(v), y=mars_soi_km * np.sin(u) * np.sin(v), z=mars_soi_km * np.cos(v),
    opacity=0.12, colorscale=[[0, "#888888"], [1, "#888888"]], showscale=False,
    name=f"Mars SOI ({mars_soi_km:.0f} km)",
), row=1, col=2)

closest_idx = (traj.t_s - closest_t_s).abs().idxmin()
closest_dist_km = np.sqrt(
    mars_rel_x[closest_idx] ** 2 + mars_rel_y[closest_idx] ** 2 + mars_rel_z[closest_idx] ** 2
)
fig.add_trace(go.Scatter3d(
    x=[mars_rel_x[closest_idx]], y=[mars_rel_y[closest_idx]], z=[mars_rel_z[closest_idx]],
    mode="markers+text", marker=dict(size=6, color="#FFD700"),
    text=[f"Closest approach: {closest_dist_km:,.0f} km"], textposition="top center", name="Closest approach",
), row=1, col=2)

# --- Right: heliocentric overview ---
fig.add_trace(go.Scatter3d(
    x=traj.x_m / AU, y=traj.y_m / AU, z=traj.z_m / AU, mode="lines",
    line=dict(color="#00d4ff", width=4), name="Spacecraft trajectory",
), row=1, col=3)
fig.add_trace(go.Scatter3d(
    x=earth_track.x_m / AU, y=earth_track.y_m / AU, z=earth_track.z_m / AU, mode="lines",
    line=dict(color="#3a7bd5", width=2, dash="dot"), name="Earth track",
), row=1, col=3)
fig.add_trace(go.Scatter3d(
    x=mars_track.x_m / AU, y=mars_track.y_m / AU, z=mars_track.z_m / AU, mode="lines",
    line=dict(color="#FF6B35", width=2, dash="dot"), name="Mars track",
), row=1, col=3)
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode="markers", marker=dict(size=10, color="#FFD700"), name="Sun",
), row=1, col=3)
fig.add_trace(go.Scatter3d(
    x=[traj.x_m[closest_idx] / AU], y=[traj.y_m[closest_idx] / AU], z=[traj.z_m[closest_idx] / AU],
    mode="markers+text", marker=dict(size=6, color="#FFD700"),
    text=[f"Closest approach: {closest_dist_km:,.0f} km at t={closest_t_s/86400:.0f}d"],
    textposition="top center", name="Closest approach (helio)",
), row=1, col=3)
fig.add_trace(go.Scatter3d(
    x=[mars_track.x_m[closest_idx] / AU], y=[mars_track.y_m[closest_idx] / AU], z=[mars_track.z_m[closest_idx] / AU],
    mode="markers+text", marker=dict(size=6, color="#FF6B35", symbol="diamond"),
    text=["Mars (at closest-approach time)"], textposition="bottom center", name="Mars (at closest-approach time)",
), row=1, col=3)

fig.update_layout(
    title=f"Exact Lambert injection (TOF=298d, dv=3637 m/s): real closest approach = {closest_dist_km:,.0f} km",
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
)

out_path = DATA / "low_dv_demo.html"
fig.write_html(str(out_path))
print(f"Wrote {out_path}")
webbrowser.open(out_path.as_uri())
