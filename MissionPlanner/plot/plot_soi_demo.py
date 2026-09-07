"""
Phase 8h — real SOI-crossing verification plot.

Reads (written by `cargo run -p mission_planner --bin soi_demo[_moon] --release`,
run from the repo root):
  out/soi_demo[_moon]/trajectory.csv  (t_s, x_m, y_m, z_m, dist_to_<body>_km, inside_soi)
  out/soi_demo[_moon]/<body>_track.csv  (t_s, x_m, y_m, z_m — the body's own track,
    heliocentric for Mars, geocentric for the Moon — same frame `trajectory.csv` is in)
  out/soi_demo[_moon]/meta.csv        (soi_radius_m)

Unlike `plot_trajectory.py`'s heliocentric overview, this plots the
spacecraft's position *relative to the target body* (body-centered, km) —
that's the frame in which "does the trajectory actually cross the SOI
boundary" is visible at all; at heliocentric/AU scale the SOI sphere would
be an invisible dot next to the much larger transfer arc.

Produces: out/soi_demo[_moon]/soi_crossing.html — interactive Plotly 3D plot.

Usage (from the repo root):
  python MissionPlanner/plot/plot_soi_demo.py            # Mars (default)
  python MissionPlanner/plot/plot_soi_demo.py moon        # Moon
"""

import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

body = sys.argv[1].lower() if len(sys.argv) > 1 else "mars"
if body == "moon":
    out_dir, body_name, body_color, track_file = "soi_demo_moon", "Moon", "#AAAAAA", "moon_track.csv"
else:
    out_dir, body_name, body_color, track_file = "soi_demo", "Mars", "#C1440E", "mars_track.csv"

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "out" / out_dir
TRAJ_CSV = DATA / "trajectory.csv"
BODY_CSV = DATA / track_file
META_CSV = DATA / "meta.csv"

if not TRAJ_CSV.exists():
    print(f"Not found: {TRAJ_CSV}")
    print(f"Run: cargo run -p mission_planner --bin soi_demo{'_moon' if body == 'moon' else ''} --release")
    sys.exit(1)

traj = pd.read_csv(TRAJ_CSV)
body_track = pd.read_csv(BODY_CSV)
soi_radius_m = pd.read_csv(META_CSV).soi_radius_m.iloc[0]
soi_radius_km = soi_radius_m / 1e3
dist_col = f"dist_to_{body}_km" if f"dist_to_{body}_km" in traj.columns else "dist_to_mars_km" if "dist_to_mars_km" in traj.columns else "dist_to_moon_km"

# Body-centered relative position [km] — same row order/t_s as `traj`,
# written in lockstep by the Rust binary.
rel_x = (traj.x_m - body_track.x_m) / 1e3
rel_y = (traj.y_m - body_track.y_m) / 1e3
rel_z = (traj.z_m - body_track.z_m) / 1e3

BG = "#0f0f0f"
ACCENT = "#00d4ff"
INSIDE_COLOR = "#FF6B35"

fig = go.Figure()

outside = traj.inside_soi == 0
inside = traj.inside_soi == 1

fig.add_trace(go.Scatter3d(
    x=rel_x[outside], y=rel_y[outside], z=rel_z[outside],
    mode="lines", line=dict(color=ACCENT, width=4),
    name=f"Outside {body_name} SOI",
))
fig.add_trace(go.Scatter3d(
    x=rel_x[inside], y=rel_y[inside], z=rel_z[inside],
    mode="markers+lines", line=dict(color=INSIDE_COLOR, width=6),
    marker=dict(size=3, color=INSIDE_COLOR),
    name=f"Inside {body_name} SOI (real central-body switch)",
))

closest_idx = traj[dist_col].idxmin()
fig.add_trace(go.Scatter3d(
    x=[rel_x[closest_idx]], y=[rel_y[closest_idx]], z=[rel_z[closest_idx]],
    mode="markers+text", marker=dict(size=6, color="#FFD700"),
    text=[f"Closest approach: {traj[dist_col][closest_idx]:.0f} km"],
    textposition="top center", name="Closest approach",
))

# Wireframe sphere at the body's real Laplace SOI radius (origin, since
# this is the body-centered frame).
u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
sx = soi_radius_km * np.cos(u) * np.sin(v)
sy = soi_radius_km * np.sin(u) * np.sin(v)
sz = soi_radius_km * np.cos(v)
fig.add_trace(go.Surface(
    x=sx, y=sy, z=sz,
    opacity=0.12, colorscale=[[0, ACCENT], [1, ACCENT]], showscale=False,
    name=f"{body_name} SOI ({soi_radius_km:.0f} km)",
))

fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode="markers+text", marker=dict(size=8, color=body_color),
    text=[body_name], textposition="bottom center", name=body_name,
))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor=BG, plot_bgcolor=BG,
    title=f"Phase 8h — real SOI-crossing verification ({body_name}-centered, km; SOI = {soi_radius_km:.0f} km)",
    scene=dict(
        xaxis_title="x [km]", yaxis_title="y [km]", zaxis_title="z [km]",
        aspectmode="data",
        bgcolor=BG,
    ),
    legend=dict(bgcolor="rgba(0,0,0,0.5)"),
)

out_path = DATA / "soi_crossing.html"
fig.write_html(out_path)
print(f"Wrote {out_path}")
webbrowser.open(out_path.as_uri())
