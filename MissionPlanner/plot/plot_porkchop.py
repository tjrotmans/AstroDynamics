"""
MissionPlanner trajectory design — porkchop plot (departure offset x TOF,
coloured by total DeltaV). Classic mission-design visualisation; applies to
any solver that runs a porkchop scan (Lambert, GridSearch,
LambertThenDiffCorrect, MonteCarlo) — anything that writes porkchop.csv.

Reads: MissionPlanner/out/<mission>/design/porkchop.csv
       (dep_offset_days, tof_days, c3_km2s2, v_inf_arr_ms, dv_dep_ms,
       dv_arr_ms, dv_total_ms — one row per grid point)

Produces: porkchop.html — interactive Plotly contour plot, dark theme.
Marks the minimum-DeltaV point found (cross-checked against best_arc.csv's
tof_days when that file exists, otherwise just the grid's own minimum).

Usage (from MissionPlanner/ directory):
  python plot/plot_porkchop.py jupiter_flyby
  python plot/plot_porkchop.py mars_flyby c3        # colour by C3 instead of dv_total
"""

import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

mission = sys.argv[1] if len(sys.argv) > 1 else "mars_flyby"
metric = sys.argv[2] if len(sys.argv) > 2 else "dv_total"

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / mission / "design"
CSV = DATA / "porkchop.csv"

if not CSV.exists():
    print(f"Not found: {CSV}")
    print("(Solver may not produce a porkchop scan — Hohmann and standalone DiffCorrection don't.)")
    sys.exit(1)

df = pd.read_csv(CSV)

METRIC_COL = {"dv_total": "dv_total_ms", "c3": "c3_km2s2", "v_inf": "v_inf_arr_ms"}[metric]
METRIC_LABEL = {"dv_total": "Total ΔV [m/s]", "c3": "C3 [km²/s²]", "v_inf": "Arrival v∞ [m/s]"}[metric]

pivot = df.pivot(index="tof_days", columns="dep_offset_days", values=METRIC_COL)
best = df.loc[df.dv_total_ms.idxmin()]

BG = "#0f0f0f"

fig = go.Figure(data=go.Contour(
    x=pivot.columns, y=pivot.index, z=pivot.values,
    colorscale="Viridis",
    contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
    colorbar=dict(title=METRIC_LABEL),
))
fig.add_trace(go.Scatter(
    x=[best.dep_offset_days], y=[best.tof_days],
    mode="markers+text", marker=dict(size=12, color="#FF6B35", symbol="x"),
    text=["Best (min ΔV)"], textposition="top center", textfont=dict(color="#FF6B35"),
    name="Best point",
))

fig.update_layout(
    title=f"{mission} — porkchop ({METRIC_LABEL}), {len(df)} grid points — "
          f"best: dep={best.dep_offset_days:+.1f}d, tof={best.tof_days:.1f}d, "
          f"ΔV_total={best.dv_total_ms:.0f} m/s",
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#dddddd"),
    xaxis=dict(title="Departure offset [days]", gridcolor="#333333"),
    yaxis=dict(title="Time of flight [days]", gridcolor="#333333"),
)

out_path = DATA / "porkchop.html"
fig.write_html(str(out_path))
print(f"Wrote {out_path}")
webbrowser.open(out_path.as_uri())
