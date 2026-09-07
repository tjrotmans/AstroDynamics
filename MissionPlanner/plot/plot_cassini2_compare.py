"""plot_cassini2_compare.py -- our Cassini-2 solution vs. ESA's published one.

Overlays two trajectories in the top-down ecliptic view:
  - ESA GTOP published best-known solution (May 2009, 8.383 km/s), as
    reproduced by `gtop_cassini2_check` -> out/cassini2_check/mga_best.csv
  - Our DE optimizer's best (out/cassini2_gtop/mga_best.csv)

Both files are ICRF [m] arcs in plot_mga.py's mga_best.csv format; rendered
in the ecliptic frame via mga_frame_utils (single-source-of-truth rule).

Usage (from MissionPlanner/):
  python plot/plot_cassini2_compare.py
"""

import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from mga_frame_utils import add_ecliptic_columns, ring_xyz, body_color

BG, GRID = "#0f0f0f", "#222"
LEG_COLORS_OURS = ["#00d4ff", "#ff6b35", "#7fba00", "#bf5fff", "#ffd24a"]

ours_path = Path("out/cassini2_gtop/mga_best.csv")
esa_path = Path("out/cassini2_check/mga_best.csv")
for p in (ours_path, esa_path):
    if not p.exists():
        raise SystemExit(
            f"{p} not found -- run the optimizer (optimize config/cassini2_gtop.toml) "
            "and the cross-check (GTOP_LP=1 cargo run --bin gtop_cassini2_check --release) first."
        )

ours = add_ecliptic_columns(pd.read_csv(ours_path))
esa = add_ecliptic_columns(pd.read_csv(esa_path))

fig = go.Figure()

# Planet orbit rings.
for name, sma_au in [("Venus", 0.723), ("Earth", 1.000), ("Jupiter", 5.203), ("Saturn", 9.537)]:
    rx, ry, _ = ring_xyz(sma_au)
    fig.add_trace(go.Scatter(
        x=rx, y=ry, mode="lines",
        line=dict(color=body_color(name), width=1, dash="dot"),
        opacity=0.5, name=f"{name} orbit", hoverinfo="skip",
    ))

# ESA published solution: one white dashed line (it is one continuous tour).
fig.add_trace(go.Scatter(
    x=esa["x_ecl_au"], y=esa["y_ecl_au"], mode="lines",
    line=dict(color="#ffffff", width=2, dash="dash"),
    name="ESA GTOP best (8.38 km/s raw-v∞ total)",
))

# Our solution: per-leg colors.
for leg in sorted(ours["leg_idx"].unique()):
    seg = ours[ours["leg_idx"] == leg]
    fig.add_trace(go.Scatter(
        x=seg["x_ecl_au"], y=seg["y_ecl_au"], mode="lines",
        line=dict(color=LEG_COLORS_OURS[int(leg) % len(LEG_COLORS_OURS)], width=2.5),
        name=f"ours — leg {int(leg)}",
    ))

# Start/end markers.
for df, label, color in [(esa, "ESA", "#ffffff"), (ours, "ours", "#00d4ff")]:
    fig.add_trace(go.Scatter(
        x=[df["x_ecl_au"].iloc[0]], y=[df["y_ecl_au"].iloc[0]],
        mode="markers", marker=dict(symbol="circle", size=9, color=color),
        name=f"{label} departure",
    ))
    fig.add_trace(go.Scatter(
        x=[df["x_ecl_au"].iloc[-1]], y=[df["y_ecl_au"].iloc[-1]],
        mode="markers", marker=dict(symbol="x", size=10, color=color),
        name=f"{label} arrival (Saturn)",
    ))

fig.add_trace(go.Scatter(
    x=[0], y=[0], mode="markers",
    marker=dict(size=12, color="#ffffc0"), name="Sun",
))

fig.update_layout(
    title="Cassini-2 (E-V-V-E-J-S): our DE solution vs. ESA GTOP published best "
          "— top-down ecliptic view",
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#ddd"),
    xaxis=dict(title="x [AU, ecliptic]", gridcolor=GRID, zerolinecolor=GRID,
               scaleanchor="y", scaleratio=1),
    yaxis=dict(title="y [AU, ecliptic]", gridcolor=GRID, zerolinecolor=GRID),
    legend=dict(bgcolor="#181818"),
)

out_html = Path("out/cassini2_gtop/cassini2_compare.html")
fig.write_html(out_html, include_plotlyjs="cdn")
print(f"Saved: {out_html}")
webbrowser.open(out_html.resolve().as_uri())
