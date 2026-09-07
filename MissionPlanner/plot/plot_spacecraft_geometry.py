"""
Phase 13a verification plot — spacecraft plate geometry (13a's "design a
spacecraft with more accuracy than a cube" goal), rendered as a real 3D
layout rather than just a CSV of numbers.

Reads (written by `cargo run -p mission_planner --bin sixdof_force_breakdown_demo
--release`, run from the repo root):
  out/sixdof_force_breakdown_demo/sixdof_geometry.csv

Each row is one `orbital_models::Plate` (the same struct
`SrpTruthModel::FlatPlate` and `[[spacecraft.hardware]] type = "CustomPlate"`
both build from — see `MissionPlanner/src/simulate.rs::build_plates`). This
plot draws every plate as a real oriented quad (sized by sqrt(area), centered
at `center_body`) with its outward normal as an arrow, in the spacecraft BODY
frame — so asymmetric panels, an off-axis dish, and the 6 bus faces are all
visually distinguishable, not just numbers in a table. This is the check that
actually proves the geometry a mission config describes is what the SRP
force/torque model sees, not an assumption.

Usage (from the repo root):
  python MissionPlanner/plot/plot_spacecraft_geometry.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent.parent
CSV = ROOT / "out" / "sixdof_force_breakdown_demo" / "sixdof_geometry.csv"

df = pd.read_csv(CSV)

fig = go.Figure()

BUS_COLOR = "#00d4ff"
PANEL_COLOR = "#ffaa00"
OTHER_COLOR = "#ff6688"
NORMAL_COLOR = "#77dd77"


def plate_color(row):
    # Heuristic classification purely for coloring, mirroring
    # `build_plates`' own conventions: bus faces sit flush on a face-center
    # offset with rho_s=0.30 typical, panels are double-sided, everything
    # else (a CustomPlate-style deployable) gets its own color.
    if row["double_sided"]:
        return PANEL_COLOR
    if abs(row["rho_s"] - 0.30) < 1e-6 and abs(row["rho_d"] - 0.20) < 1e-6:
        return BUS_COLOR
    return OTHER_COLOR


for _, row in df.iterrows():
    n = np.array([row["normal_x"], row["normal_y"], row["normal_z"]])
    n = n / np.linalg.norm(n)
    c = np.array([row["center_x"], row["center_y"], row["center_z"]])

    # In-plane basis for a square patch of side sqrt(area), centered at c.
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    half = np.sqrt(row["area_m2"]) / 2.0

    corners = [c + half * u + half * v, c + half * u - half * v,
               c - half * u - half * v, c - half * u + half * v]
    xs = [pt[0] for pt in corners]
    ys = [pt[1] for pt in corners]
    zs = [pt[2] for pt in corners]
    color = plate_color(row)

    fig.add_trace(go.Mesh3d(
        x=xs, y=ys, z=zs, i=[0, 0], j=[1, 2], k=[2, 3],
        color=color, opacity=0.55, flatshading=True, showlegend=False,
        hovertext=f"plate {int(row['plate_idx'])}: area={row['area_m2']:.2f} m^2, "
                   f"rho_s={row['rho_s']:.2f}, rho_d={row['rho_d']:.2f}, "
                   f"double_sided={bool(row['double_sided'])}",
        hoverinfo="text",
    ))
    # Outward normal arrow, length scaled with sqrt(area) so small/large
    # plates both show a visible but proportionate arrow.
    arrow_len = 0.6 + 0.4 * np.sqrt(row["area_m2"])
    tip = c + n * arrow_len
    fig.add_trace(go.Scatter3d(
        x=[c[0], tip[0]], y=[c[1], tip[1]], z=[c[2], tip[2]],
        mode="lines", line=dict(color=NORMAL_COLOR, width=4), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Cone(
        x=[tip[0]], y=[tip[1]], z=[tip[2]], u=[n[0]], v=[n[1]], w=[n[2]],
        sizemode="absolute", sizeref=0.3, anchor="tail",
        colorscale=[[0, NORMAL_COLOR], [1, NORMAL_COLOR]], showscale=False, hoverinfo="skip",
    ))

# Body-frame axes triad at the CoM, for orientation reference.
axis_len = 1.5
for axis, color, name in [([1, 0, 0], "#ff4444", "+x"), ([0, 1, 0], "#44ff44", "+y"), ([0, 0, 1], "#4488ff", "+z")]:
    fig.add_trace(go.Scatter3d(
        x=[0, axis[0] * axis_len], y=[0, axis[1] * axis_len], z=[0, axis[2] * axis_len],
        mode="lines+text", line=dict(color=color, width=6), text=["", name],
        textposition="top center", showlegend=False, hoverinfo="skip",
    ))

n_bus = int(((df["double_sided"] == 0) & (abs(df["rho_s"] - 0.30) < 1e-6)).sum())
n_panel = int((df["double_sided"] == 1).sum())
n_other = len(df) - n_bus - n_panel
print(f"Plates: {n_bus} bus faces, {n_panel} panel halves, {n_other} other (deployables/custom)")

fig.update_layout(
    title="Phase 13a — spacecraft plate geometry (body frame)<br>"
          f"<sub>{BUS_COLOR}-colored bus faces &nbsp;|&nbsp; {PANEL_COLOR}-colored solar panels &nbsp;|&nbsp; "
          f"{OTHER_COLOR}-colored deployables &nbsp;|&nbsp; {NORMAL_COLOR}-colored outward normals</sub>",
    template="plotly_dark",
    paper_bgcolor="#0f0f0f",
    font=dict(color="#dddddd"),
    scene=dict(
        xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
        aspectmode="data",
        bgcolor="#0f0f0f",
    ),
    height=900, width=1100,
)

out_path = ROOT / "out" / "sixdof_force_breakdown_demo" / "spacecraft_geometry.html"
fig.write_html(str(out_path))
print(f"Saved {out_path}")
fig.show()
