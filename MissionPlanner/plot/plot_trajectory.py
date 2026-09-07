"""
MissionPlanner trajectory design — 3D heliocentric transfer arc visualisation.

Reads:
  MissionPlanner/out/<mission>/design/best_arc_trajectory.csv
    (t_s, x_m, y_m, z_m — Phase 7 real propagated ephemeris, not an
    analytic shortcut; an analytic two-body shortcut)
  MissionPlanner/out/<mission>/design/best_arc_bodies.csv
    (t_s, dep_x/y/z_m, target_x/y/z_m — the departure body's (Earth unless
    [trajectory].departure_body is set) and the target body's real
    heliocentric position over the same transfer window, so you can
    visually confirm the target body is actually where the spacecraft
    arrives, not just trust the printed numbers)

Produces (into the same directory): best_arc_trajectory.html — interactive
Plotly 3D plot, dark theme, opens directly in a browser.

Usage (from MissionPlanner/ directory):
  python plot/plot_trajectory.py                 # default: mars_flyby
  python plot/plot_trajectory.py bennu_sample_return
"""

import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

mission = sys.argv[1] if len(sys.argv) > 1 else "mars_flyby"

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / mission / "design"
CSV = DATA / "best_arc_trajectory.csv"
BODIES_CSV = DATA / "best_arc_bodies.csv"

if not CSV.exists():
    print(f"Not found: {CSV}")
    print("Run: cargo run --bin mission-planner --release -- design config/<mission>.toml")
    sys.exit(1)

df = pd.read_csv(CSV)
AU = 1.495978707e11
x_au, y_au, z_au = df.x_m / AU, df.y_m / AU, df.z_m / AU

BG = "#0f0f0f"
ACCENT = "#00d4ff"

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x_au, y=y_au, z=z_au,
    mode="lines",
    line=dict(color=ACCENT, width=4),
    name="Transfer arc",
))
fig.add_trace(go.Scatter3d(
    x=[x_au.iloc[0]], y=[y_au.iloc[0]], z=[z_au.iloc[0]],
    mode="markers+text", marker=dict(size=6, color="#4FC3F7"),
    text=["Departure"], textposition="top center", name="Departure",
))
fig.add_trace(go.Scatter3d(
    x=[x_au.iloc[-1]], y=[y_au.iloc[-1]], z=[z_au.iloc[-1]],
    mode="markers+text", marker=dict(size=6, color="#FF6B35"),
    text=["Arrival"], textposition="top center", name="Arrival",
))
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode="markers", marker=dict(size=10, color="#FFD700"),
    name="Sun",
))

target_name = "Target"
dep_name = "Earth"
if BODIES_CSV.exists():
    with open(BODIES_CSV) as f:
        header_lines = [f.readline(), f.readline()]
    for line in header_lines:
        if line.startswith("# target_name="):
            target_name = line.split("=", 1)[1].strip()
        elif line.startswith("# dep_name="):
            dep_name = line.split("=", 1)[1].strip()

    bdf = pd.read_csv(BODIES_CSV, comment="#")
    ex_au, ey_au, ez_au = bdf.dep_x_m / AU, bdf.dep_y_m / AU, bdf.dep_z_m / AU
    tx_au, ty_au, tz_au = bdf.target_x_m / AU, bdf.target_y_m / AU, bdf.target_z_m / AU

    fig.add_trace(go.Scatter3d(
        x=ex_au, y=ey_au, z=ez_au,
        mode="lines", line=dict(color="#4FC3F7", width=2, dash="dot"),
        name=f"{dep_name} orbit (over transfer window)",
    ))
    fig.add_trace(go.Scatter3d(
        x=tx_au, y=ty_au, z=tz_au,
        mode="lines", line=dict(color="#FF6B35", width=2, dash="dot"),
        name=f"{target_name} orbit (over transfer window)",
    ))
    # Target body's position at arrival time — should coincide with the
    # spacecraft's arrival marker if the transfer is correctly targeted.
    fig.add_trace(go.Scatter3d(
        x=[tx_au.iloc[-1]], y=[ty_au.iloc[-1]], z=[tz_au.iloc[-1]],
        mode="markers+text", marker=dict(size=5, color="#FF6B35", symbol="diamond"),
        text=[target_name], textposition="bottom center", name=f"{target_name} at arrival",
    ))
else:
    print(f"Note: {BODIES_CSV} not found — showing spacecraft arc only, no body orbits")

r_dep_au = (x_au.iloc[0] ** 2 + y_au.iloc[0] ** 2 + z_au.iloc[0] ** 2) ** 0.5
r_arr_au = (x_au.iloc[-1] ** 2 + y_au.iloc[-1] ** 2 + z_au.iloc[-1] ** 2) ** 0.5
tof_days = df.t_s.iloc[-1] / 86400.0

fig.update_layout(
    title=f"{mission} — {dep_name} → {target_name} transfer (r_dep={r_dep_au:.3f} AU, "
          f"r_arr={r_arr_au:.3f} AU, TOF={tof_days:.1f} days, {len(df)} pts)",
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color="#dddddd"),
    scene=dict(
        xaxis_title="x [AU]", yaxis_title="y [AU]", zaxis_title="z [AU]",
        xaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        yaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        zaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        aspectmode="data",
    ),
)

out_path = DATA / "best_arc_trajectory.html"
fig.write_html(str(out_path))
print(f"Wrote {out_path}")
webbrowser.open(out_path.as_uri())
