"""
Phase 9 — real patched-conic hyperbolic departure leg verification plot.

Reads (written by `cargo run -p mission_planner --bin departure_demo --release`,
run from the repo root):
  out/departure_demo/escape_trajectory.csv  (t_s, x_m, y_m, z_m, dist_from_earth_km
    -- already Earth-centered, written that way by the Rust binary)
  out/departure_demo/meta.csv               (r_park_m, soi_radius_m,
    escape_duration_s, dv_escape_ms)

Earth-centered, not heliocentric -- at heliocentric/AU scale the whole
escape leg is an invisible point next to Earth's ~1 AU position. This is
the close-up departure view, the same role `plot_soi_demo.py` plays for
arrival-side SOI crossings, but for the new departure leg instead.

Produces: out/departure_demo/departure_leg.html -- interactive Plotly 3D plot.

Usage (from the repo root):
  python MissionPlanner/plot/plot_departure_demo.py
"""

import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "out" / "departure_demo"
TRAJ_CSV = DATA / "escape_trajectory.csv"
META_CSV = DATA / "meta.csv"

if not TRAJ_CSV.exists():
    print(f"Not found: {TRAJ_CSV}")
    print("Run: cargo run -p mission_planner --bin departure_demo --release")
    sys.exit(1)

traj = pd.read_csv(TRAJ_CSV)
meta = pd.read_csv(META_CSV).iloc[0]
r_park_km = meta.r_park_m / 1e3
soi_radius_km = meta.soi_radius_m / 1e3
escape_days = meta.escape_duration_s / 86_400.0
dv_escape_kms = meta.dv_escape_ms / 1e3

x_km = traj.x_m / 1e3
y_km = traj.y_m / 1e3
z_km = traj.z_m / 1e3

BG = "#0f0f0f"
ACCENT = "#00d4ff"

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x_km, y=y_km, z=z_km,
    mode="lines", line=dict(color=ACCENT, width=4),
    name="Real escape trajectory (parking orbit -> SOI exit)",
))

fig.add_trace(go.Scatter3d(
    x=[x_km.iloc[0]], y=[y_km.iloc[0]], z=[z_km.iloc[0]],
    mode="markers+text", marker=dict(size=5, color="#FFD700"),
    text=["Injection burn"], textposition="top center", name="Injection (parking orbit)",
))
fig.add_trace(go.Scatter3d(
    x=[x_km.iloc[-1]], y=[y_km.iloc[-1]], z=[z_km.iloc[-1]],
    mode="markers+text", marker=dict(size=5, color="#FF6B35"),
    text=[f"SOI exit ({escape_days:.2f} days)"], textposition="top center", name="SOI exit",
))

# Wireframe sphere at Earth's real Laplace SOI radius (origin, Earth-centered frame).
u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
sx = soi_radius_km * np.cos(u) * np.sin(v)
sy = soi_radius_km * np.sin(u) * np.sin(v)
sz = soi_radius_km * np.cos(v)
fig.add_trace(go.Surface(
    x=sx, y=sy, z=sz,
    opacity=0.08, colorscale=[[0, ACCENT], [1, ACCENT]], showscale=False,
    name=f"Earth SOI ({soi_radius_km:,.0f} km)",
))

# Small wireframe sphere at the parking orbit radius, for scale reference
# against the (much larger) SOI sphere above.
sx_p = r_park_km * np.cos(u) * np.sin(v)
sy_p = r_park_km * np.sin(u) * np.sin(v)
sz_p = r_park_km * np.cos(v)
fig.add_trace(go.Surface(
    x=sx_p, y=sy_p, z=sz_p,
    opacity=0.25, colorscale=[[0, "#4488ff"], [1, "#4488ff"]], showscale=False,
    name=f"Parking orbit ({r_park_km:,.0f} km)",
))

fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode="markers+text", marker=dict(size=8, color="#3a7bd5"),
    text=["Earth"], textposition="bottom center", name="Earth",
))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor=BG, plot_bgcolor=BG,
    title=(
        f"Real patched-conic departure leg (Earth-centered, km) -- "
        f"escape ΔV {dv_escape_kms:.3f} km/s, duration {escape_days:.2f} days"
    ),
    scene=dict(
        xaxis_title="x [km]", yaxis_title="y [km]", zaxis_title="z [km]",
        aspectmode="data",
        bgcolor=BG,
    ),
    legend=dict(bgcolor="rgba(0,0,0,0.5)"),
)

out_path = DATA / "departure_leg.html"
fig.write_html(out_path)
print(f"Wrote {out_path}")
webbrowser.open(out_path.as_uri())
