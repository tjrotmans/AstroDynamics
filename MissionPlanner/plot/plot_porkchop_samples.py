"""
MissionPlanner trajectory design — multiple grid-search trajectories in one
3D view, spread across the whole ΔV-ranked distribution (best to worst), not
just the single cherry-picked best arc. Lets you sanity-check the whole
considered search space at a glance — do all candidate trajectories look
like physically reasonable transfers, or does something in the family look
wrong even though the "best" one alone looked fine.

Reads: MissionPlanner/out/<mission>/design/porkchop_samples.csv
       (sample_rank_pct, dep_offset_days, tof_days, dv_total_ms, t_s, x_m, y_m, z_m)
Also overlays departure-body/target-body orbits from best_arc_bodies.csv, if present.

Produces: porkchop_samples.html — interactive Plotly 3D plot, dark theme,
one colour per sample (cool = low ΔV/best, warm = high ΔV/worst).

Usage (from MissionPlanner/ directory):
  python plot/plot_porkchop_samples.py jupiter_flyby
"""

import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

mission = sys.argv[1] if len(sys.argv) > 1 else "mars_flyby"

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / mission / "design"
CSV = DATA / "porkchop_samples.csv"
BODIES_CSV = DATA / "best_arc_bodies.csv"

if not CSV.exists():
    print(f"Not found: {CSV}")
    sys.exit(1)

df = pd.read_csv(CSV)
AU = 1.495978707e11

BG = "#0f0f0f"
fig = go.Figure()

ranks = sorted(df.sample_rank_pct.unique())
# Hand-picked palette, not a sampled continuous colorscale — Turbo/Viridis/
# Plasma all have a dark purple/blue low end that's nearly invisible against
# the #0f0f0f background. Every colour here is bright enough to read on black,
# ordered best (cyan) -> worst (magenta) the same way Turbo would be.
BRIGHT_PALETTE = ["#00d4ff", "#00ff9f", "#aaff00", "#ffee00", "#ffaa00", "#ff6b35", "#ff3366", "#ff00cc"]
colors = [BRIGHT_PALETTE[i % len(BRIGHT_PALETTE)] for i in range(len(ranks))]

for rank, color in zip(ranks, colors):
    sub = df[df.sample_rank_pct == rank]
    dv = sub.dv_total_ms.iloc[0]
    dep = sub.dep_offset_days.iloc[0]
    tof = sub.tof_days.iloc[0]
    fig.add_trace(go.Scatter3d(
        x=sub.x_m / AU, y=sub.y_m / AU, z=sub.z_m / AU,
        mode="lines",
        line=dict(color=color, width=4 if rank == 0 else 2),
        name=f"{rank:.0f}%ile  ΔV={dv:.0f} m/s  (dep={dep:+.0f}d, tof={tof:.0f}d)",
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
    fig.add_trace(go.Scatter3d(
        x=bdf.dep_x_m / AU, y=bdf.dep_y_m / AU, z=bdf.dep_z_m / AU,
        mode="lines", line=dict(color="#4FC3F7", width=1, dash="dot"),
        name=f"{dep_name} (best-arc window)", opacity=0.8,
    ))
    fig.add_trace(go.Scatter3d(
        x=bdf.target_x_m / AU, y=bdf.target_y_m / AU, z=bdf.target_z_m / AU,
        mode="lines", line=dict(color="#FF6B35", width=1, dash="dot"),
        name=f"{target_name} (best-arc window)", opacity=0.8,
    ))

fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode="markers", marker=dict(size=10, color="#FFD700"), name="Sun",
))

fig.update_layout(
    title=f"{mission} — {len(ranks)} porkchop samples, best→worst ΔV (Turbo colourscale)",
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#dddddd"),
    scene=dict(
        xaxis_title="x [AU]", yaxis_title="y [AU]", zaxis_title="z [AU]",
        xaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        yaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        zaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        aspectmode="data",
    ),
)

out_path = DATA / "porkchop_samples.html"
fig.write_html(str(out_path))
print(f"Wrote {out_path}")
webbrowser.open(out_path.as_uri())
