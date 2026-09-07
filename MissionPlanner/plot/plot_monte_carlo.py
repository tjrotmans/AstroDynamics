"""
MissionPlanner trajectory design — Monte Carlo dispersion cloud (Phase 8i,
absorbs 7h).

Different question from plot_porkchop_samples.py's best-to-worst spread: that
plot deliberately shows the *full considered search space*, edge to edge.
This one checks whether a *local* Gaussian scatter around an already-found
solution looks like a sensible, tight cluster centered on the reference —
not a wide search, so the right visual check is "does the cloud hang
together," not "does it span the whole box." Trajectories are colour-binned
by ΔV quintile (relative to this scatter's own distribution, not the full
porkchop grid) so any spatial split between low/high-ΔV samples is visible;
the reference (porkchop-best) trajectory is drawn as a bold white line so
the cloud's centering on it is obvious at a glance.

Reads:
  MissionPlanner/out/<mission>/design/monte_carlo_trajectories.csv
    (sample_id, dep_offset_days, tof_days, dv_total_ms, t_s, x_m, y_m, z_m)
  MissionPlanner/out/<mission>/design/best_arc_trajectory.csv  (reference)
  MissionPlanner/out/<mission>/design/best_arc_bodies.csv      (body orbits, if present)

Produces: monte_carlo_cloud.html — interactive Plotly 3D plot, dark theme.

Usage (from MissionPlanner/ directory):
  python plot/plot_monte_carlo.py mars_monte_carlo
"""

import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

mission = sys.argv[1] if len(sys.argv) > 1 else "mars_monte_carlo"

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / mission / "design"
CSV = DATA / "monte_carlo_trajectories.csv"
REF_CSV = DATA / "best_arc_trajectory.csv"
BODIES_CSV = DATA / "best_arc_bodies.csv"

if not CSV.exists():
    print(f"Not found: {CSV}")
    print("Run: cargo run --bin mission-planner --release -- design config/<mission>.toml  (solver = \"MonteCarlo\")")
    sys.exit(1)

df = pd.read_csv(CSV)
AU = 1.495978707e11

BG = "#0f0f0f"
fig = go.Figure()

# Quintile color-bin by this scatter's own ΔV distribution — a genuinely
# different question from porkchop_samples' full-grid best-to-worst rank.
per_sample_dv = df.groupby("sample_id").dv_total_ms.first().sort_values()
n_bins = 5
bin_edges = pd.qcut(per_sample_dv, n_bins, labels=False, duplicates="drop")
BRIGHT_PALETTE = ["#00d4ff", "#00ff9f", "#ffee00", "#ff6b35", "#ff3366"]

for sample_id, sub in df.groupby("sample_id"):
    bin_idx = int(bin_edges.loc[sample_id])
    color = BRIGHT_PALETTE[bin_idx % len(BRIGHT_PALETTE)]
    dv = sub.dv_total_ms.iloc[0]
    fig.add_trace(go.Scatter3d(
        x=sub.x_m / AU, y=sub.y_m / AU, z=sub.z_m / AU,
        mode="lines",
        line=dict(color=color, width=1.5),
        opacity=0.55,
        name=f"sample {sample_id}  ΔV={dv:.0f} m/s",
        showlegend=False,
    ))

# Dummy traces just to get one legend entry per quintile bin (real traces
# above all have showlegend=False to avoid 30 separate legend rows).
dv_min, dv_max = per_sample_dv.min(), per_sample_dv.max()
for i, color in enumerate(BRIGHT_PALETTE[:n_bins]):
    lo = dv_min + i * (dv_max - dv_min) / n_bins
    hi = dv_min + (i + 1) * (dv_max - dv_min) / n_bins
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None], mode="lines",
        line=dict(color=color, width=4),
        name=f"ΔV quintile {i+1}  ({lo:.0f}-{hi:.0f} m/s)",
    ))

if REF_CSV.exists():
    ref = pd.read_csv(REF_CSV)
    fig.add_trace(go.Scatter3d(
        x=ref.x_m / AU, y=ref.y_m / AU, z=ref.z_m / AU,
        mode="lines", line=dict(color="#ffffff", width=5),
        name="Reference (porkchop best)",
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
        name=f"{dep_name} (window)", opacity=0.8,
    ))
    fig.add_trace(go.Scatter3d(
        x=bdf.target_x_m / AU, y=bdf.target_y_m / AU, z=bdf.target_z_m / AU,
        mode="lines", line=dict(color="#FF6B35", width=1, dash="dot"),
        name=f"{target_name} (window)", opacity=0.8,
    ))

fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode="markers", marker=dict(size=10, color="#FFD700"), name="Sun",
))

n_samples = df.sample_id.nunique()
fig.update_layout(
    title=f"{mission} — Monte Carlo dispersion cloud ({n_samples} samples around porkchop best)",
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#dddddd"),
    scene=dict(
        xaxis_title="x [AU]", yaxis_title="y [AU]", zaxis_title="z [AU]",
        xaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        yaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        zaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        aspectmode="data",
    ),
)

out_path = DATA / "monte_carlo_cloud.html"
fig.write_html(str(out_path))
print(f"Wrote {out_path}")
webbrowser.open(out_path.as_uri())
