"""
Phase 13c verification plot — interactive 3D orbit with spacecraft body axes
and the inertial reference frame.

Reads (written by `cargo run -p mission_planner --bin sixdof_demo --release`):
  out/sixdof_demo/sixdof_demo.csv

Shows the body-centered orbit path plus periodic body-axis triads (red =
body +x / boresight, green = body +y, blue = body +z — standard RGB<->XYZ
convention) so the attitude motion (gravity-gradient + SRP libration, no
control) is visible directly against the orbit geometry, not just as
abstract angle-vs-time curves. A fixed inertial-frame triad at the origin
(dashed, dimmer) gives a non-rotating reference to judge the body axes
against.

Usage (from the repo root):
  python MissionPlanner/plot/plot_sixdof_demo_3d.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent.parent
CSV = ROOT / "out" / "sixdof_demo" / "sixdof_demo.csv"

df = pd.read_csv(CSV)

# Orbit path in km, body-centered.
r = df[["r_x", "r_y", "r_z"]].to_numpy() / 1e3

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=r[:, 0], y=r[:, 1], z=r[:, 2],
    mode="lines", line=dict(color="#00d4ff", width=3),
    name="orbit (body-centered)",
))

# Central body, drawn to scale (Earth-like radius used by the demo binary).
earth_r_km = 6378.0
u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
ex = earth_r_km * np.cos(u) * np.sin(v)
ey = earth_r_km * np.sin(u) * np.sin(v)
ez = earth_r_km * np.cos(v)
fig.add_trace(go.Surface(
    x=ex, y=ey, z=ez, showscale=False, opacity=0.35,
    colorscale=[[0, "#3366aa"], [1, "#3366aa"]], name="central body",
))

# ── Sample body-axis triads along the orbit, evenly spaced ──────────────────
n_triads = 40
idx = np.linspace(0, len(df) - 1, n_triads).astype(int)
axis_len_km = 400.0  # visible fraction of the ~7078 km orbit radius

axis_specs = [
    ("bodyx", "#ff4444", "body +x (boresight)"),
    ("bodyy", "#44ff44", "body +y"),
    ("bodyz", "#4488ff", "body +z"),
]
for prefix, color, label in axis_specs:
    xs, ys, zs = [], [], []
    for i in idx:
        p0 = r[i]
        d = df.loc[i, [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]].to_numpy(dtype=float)
        p1 = p0 + d * axis_len_km
        xs += [p0[0], p1[0], None]
        ys += [p0[1], p1[1], None]
        zs += [p0[2], p1[2], None]
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=4),
        name=label,
    ))

# ── Fixed inertial-frame reference triad at the origin ──────────────────────
inertial_len_km = 2500.0
inertial_axes = [
    ((inertial_len_km, 0, 0), "#888888", "inertial X"),
    ((0, inertial_len_km, 0), "#888888", "inertial Y"),
    ((0, 0, inertial_len_km), "#888888", "inertial Z"),
]
for (dx, dy, dz), color, label in inertial_axes:
    fig.add_trace(go.Scatter3d(
        x=[0, dx], y=[0, dy], z=[0, dz], mode="lines+text",
        line=dict(color=color, width=3, dash="dash"),
        text=["", label], textposition="top center",
        textfont=dict(color=color, size=10),
        name=label, showlegend=False,
    ))

fig.update_layout(
    title="Phase 13c — 6DOF orbit with spacecraft body axes and inertial frame",
    scene=dict(
        xaxis_title="x [km]", yaxis_title="y [km]", zaxis_title="z [km]",
        aspectmode="data",
        bgcolor="#0f0f0f",
    ),
    paper_bgcolor="#0f0f0f",
    font=dict(color="#dddddd"),
    legend=dict(bgcolor="rgba(0,0,0,0.5)"),
)

out_path = ROOT / "out" / "sixdof_demo" / "sixdof_demo_3d.html"
fig.write_html(str(out_path))
print(f"Saved {out_path}")
fig.show()
