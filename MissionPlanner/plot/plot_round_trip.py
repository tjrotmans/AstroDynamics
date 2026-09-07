"""
MissionPlanner trajectory design — round-trip overlay (Phase 8l).

Distinct check from plot_trajectory.py's single-leg geometry verification:
that script confirms one arc actually arrives where its target body is.
This script instead confirms the *pair* of legs forms a sensible loop —
outbound arc ends near the target body, the return arc starts from
wherever the outbound leg left off (after a stay), and the return arc ends
back near the original departure body. Neither leg's own plot can show
that relationship; only an overlay of both can.

Reads two mission output dirs (default: mars_round_trip_outbound,
mars_round_trip_return), each one's:
  out/<mission>/design/best_arc_trajectory.csv  (t_s, x_m, y_m, z_m)
  out/<mission>/design/best_arc_bodies.csv      (t_s, dep_x/y/z_m, target_x/y/z_m)

Produces: out/<outbound_mission>/design/round_trip.html

Usage (from MissionPlanner/ directory):
  python plot/plot_round_trip.py
  python plot/plot_round_trip.py mars_round_trip_outbound mars_round_trip_return
"""

import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

outbound_mission = sys.argv[1] if len(sys.argv) > 1 else "mars_round_trip_outbound"
return_mission = sys.argv[2] if len(sys.argv) > 2 else "mars_round_trip_return"

ROOT = Path(__file__).parent.parent
AU = 1.495978707e11
BG = "#0f0f0f"


def load_leg(mission):
    data = ROOT / "out" / mission / "design"
    csv = data / "best_arc_trajectory.csv"
    bodies_csv = data / "best_arc_bodies.csv"
    if not csv.exists():
        print(f"Not found: {csv}")
        print(f"Run: cargo run --bin mission-planner --release -- design config/{mission}.toml")
        sys.exit(1)

    df = pd.read_csv(csv)
    dep_name, target_name = "Departure body", "Target body"
    bdf = None
    if bodies_csv.exists():
        with open(bodies_csv) as f:
            header_lines = [f.readline(), f.readline()]
        for line in header_lines:
            if line.startswith("# target_name="):
                target_name = line.split("=", 1)[1].strip()
            elif line.startswith("# dep_name="):
                dep_name = line.split("=", 1)[1].strip()
        bdf = pd.read_csv(bodies_csv, comment="#")

    return df, bdf, dep_name, target_name


outbound_df, outbound_bdf, dep_name, mid_name = load_leg(outbound_mission)
return_df, return_bdf, mid_name2, home_name = load_leg(return_mission)

fig = go.Figure()

# Sun
fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                            marker=dict(size=10, color="#FFD700"), name="Sun"))

# Outbound arc
ox, oy, oz = outbound_df.x_m / AU, outbound_df.y_m / AU, outbound_df.z_m / AU
fig.add_trace(go.Scatter3d(x=ox, y=oy, z=oz, mode="lines",
                            line=dict(color="#00d4ff", width=4),
                            name=f"Outbound ({dep_name} → {mid_name})"))
fig.add_trace(go.Scatter3d(x=[ox.iloc[0]], y=[oy.iloc[0]], z=[oz.iloc[0]],
                            mode="markers+text", marker=dict(size=6, color="#4FC3F7"),
                            text=[f"Depart {dep_name}"], textposition="top center",
                            name=f"Depart {dep_name}"))

# Return arc
rx, ry, rz = return_df.x_m / AU, return_df.y_m / AU, return_df.z_m / AU
fig.add_trace(go.Scatter3d(x=rx, y=ry, z=rz, mode="lines",
                            line=dict(color="#FF6B35", width=4),
                            name=f"Return ({mid_name2} → {home_name})"))
fig.add_trace(go.Scatter3d(x=[rx.iloc[0]], y=[ry.iloc[0]], z=[rz.iloc[0]],
                            mode="markers+text", marker=dict(size=6, color="#FFA552"),
                            text=[f"Depart {mid_name2}"], textposition="top center",
                            name=f"Depart {mid_name2}"))
fig.add_trace(go.Scatter3d(x=[rx.iloc[-1]], y=[ry.iloc[-1]], z=[rz.iloc[-1]],
                            mode="markers+text", marker=dict(size=6, color="#76FF03"),
                            text=[f"Arrive {home_name}"], textposition="top center",
                            name=f"Arrive {home_name}"))

# Body orbit traces — combine whichever legs have them, dedupe by name.
seen = set()
for bdf, names, colors in (
    (outbound_bdf, (dep_name, mid_name), ("#4FC3F7", "#FF6B35")),
    (return_bdf, (mid_name2, home_name), ("#FF6B35", "#4FC3F7")),
):
    if bdf is None:
        continue
    for col_prefix, name, color in (("dep", names[0], colors[0]), ("target", names[1], colors[1])):
        if name in seen:
            continue
        seen.add(name)
        bx = bdf[f"{col_prefix}_x_m"] / AU
        by = bdf[f"{col_prefix}_y_m"] / AU
        bz = bdf[f"{col_prefix}_z_m"] / AU
        fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="lines",
                                    line=dict(color=color, width=2, dash="dot"),
                                    name=f"{name} orbit"))

tof_out_days = outbound_df.t_s.iloc[-1] / 86400.0
tof_ret_days = return_df.t_s.iloc[-1] / 86400.0

fig.update_layout(
    title=f"{outbound_mission} + {return_mission} — round trip "
          f"({dep_name} → {mid_name}: {tof_out_days:.1f}d, {mid_name2} → {home_name}: {tof_ret_days:.1f}d)",
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#dddddd"),
    scene=dict(
        xaxis_title="x [AU]", yaxis_title="y [AU]", zaxis_title="z [AU]",
        xaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        yaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        zaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        aspectmode="data",
    ),
)

out_path = ROOT / "out" / outbound_mission / "design" / "round_trip.html"
fig.write_html(str(out_path))
print(f"Wrote {out_path}")
webbrowser.open(out_path.as_uri())
