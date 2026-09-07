"""
MissionPlanner trajectory design — heliocentric RADIUS vs TIME comparison.

Different plot type from plot_trajectory.py's 3D position view, deliberately —
this one is for checking energy/ΔV efficiency against the idealized analytic
Hohmann minimum-energy transfer, which a 3D position overlay wouldn't show
clearly (a real ANISE-based grid-search result and an idealized two-body
Hohmann ellipse don't share an inertial orientation, so overlaying their 3D
shapes directly looks misleading; comparing r(t) profiles is the meaningful
check here instead).

Reads: MissionPlanner/out/<mission>/design/best_arc_trajectory.csv

Produces: best_arc_radius_vs_time.html — interactive Plotly 2D plot, dark
theme. Overlays the actual propagated r(t) against the idealized Hohmann
r(t) profile for the SAME r1/r2 endpoints (computed analytically in this
script via vis-viva + Kepler's equation — not a Rust round-trip, this is a
display-only reference curve).

Usage (from MissionPlanner/ directory):
  python plot/plot_radius_vs_time.py jupiter_flyby
"""

import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

mission = sys.argv[1] if len(sys.argv) > 1 else "jupiter_flyby"

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / mission / "design"
CSV = DATA / "best_arc_trajectory.csv"

if not CSV.exists():
    print(f"Not found: {CSV}")
    sys.exit(1)

df = pd.read_csv(CSV)
AU = 1.495978707e11
MU_SUN = 1.32712440018e20

r_m = np.sqrt(df.x_m**2 + df.y_m**2 + df.z_m**2)
r1_m, r2_m = r_m.iloc[0], r_m.iloc[-1]
t_days = df.t_s / 86400.0

# Idealized two-body Hohmann r(t) for the SAME r1/r2 (not the same TOF —
# the Hohmann TOF is whatever minimum-energy transfer between these radii
# actually takes; comparing the two TOFs is itself part of the verification).
a_t = 0.5 * (r1_m + r2_m)
e_t = (r2_m - r1_m) / (r2_m + r1_m)
n_mean = np.sqrt(MU_SUN / a_t**3)
hohmann_tof_days = np.pi / n_mean / 86400.0

m_anom = np.linspace(0, np.pi, 200)  # periapsis (dep) to apoapsis (arr)
ea = m_anom.copy()
for _ in range(50):
    ea = ea - (ea - e_t * np.sin(ea) - m_anom) / (1 - e_t * np.cos(ea))
r_hohmann_m = a_t * (1 - e_t * np.cos(ea))
t_hohmann_days = m_anom / n_mean / 86400.0

BG = "#0f0f0f"

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t_days, y=r_m / AU, mode="lines",
    line=dict(color="#00d4ff", width=3),
    name=f"Actual propagated trajectory (TOF={t_days.iloc[-1]:.1f} d)",
))
fig.add_trace(go.Scatter(
    x=t_hohmann_days, y=r_hohmann_m / AU, mode="lines",
    line=dict(color="#FFD166", width=2, dash="dash"),
    name=f"Idealized Hohmann, same r1/r2 (TOF={hohmann_tof_days:.1f} d)",
))
fig.add_hline(y=r1_m / AU, line=dict(color="#4FC3F7", width=1, dash="dot"),
              annotation_text=f"r1 = {r1_m/AU:.3f} AU", annotation_position="bottom right")
fig.add_hline(y=r2_m / AU, line=dict(color="#FF6B35", width=1, dash="dot"),
              annotation_text=f"r2 = {r2_m/AU:.3f} AU", annotation_position="top right")

fig.update_layout(
    title=f"{mission} — heliocentric radius vs time (actual vs idealized Hohmann)",
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#dddddd"),
    xaxis=dict(title="Time since departure [days]", gridcolor="#333333"),
    yaxis=dict(title="Heliocentric distance [AU]", gridcolor="#333333"),
    legend=dict(bgcolor="rgba(20,20,20,0.7)"),
)

out_path = DATA / "best_arc_radius_vs_time.html"
fig.write_html(str(out_path))
print(f"Wrote {out_path}")
print(f"Actual TOF: {t_days.iloc[-1]:.1f} d  |  Idealized Hohmann TOF: {hohmann_tof_days:.1f} d  "
      f"({100*(t_days.iloc[-1]/hohmann_tof_days - 1):+.1f}%)")
webbrowser.open(out_path.as_uri())
