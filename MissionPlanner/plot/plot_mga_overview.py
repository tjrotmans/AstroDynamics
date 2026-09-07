"""
MGA-1DSM 2D top-down ecliptic view -- Phase 9 backend visualisation.

A clean, publication-style top-down view of the solar system showing:
  - Reference circular orbits for all bodies in the sequence (grey rings)
  - Transfer arcs for each leg, color-coded by leg index
  - DSM positions as × markers on each arc
  - Body positions at each encounter epoch (colored dots with labels)
  - The Sun at the origin

This complements plot_mga.py (which has a 3D view) by offering a 2D view
that is easier to read for confirming orbital geometry (periapsis/apoapsis
alignment, flyby geometry, relative phasing of bodies).

Usage (from MissionPlanner/ directory):
  python plot/plot_mga_overview.py evj_flyby
  python plot/plot_mga_overview.py venus_saturn_auto

Reads (from out/<mission>/):
  mga_best.csv      -- arc points: t_days, x_m, y_m, z_m, leg_idx
  mga_legs.csv      -- per-leg: body_dep, body_arr, t_dep_jd, t_arr_jd,
                       vinf_dep_ms, vinf_arr_ms, turn_deg, rp_km,
                       x_dsm_m, y_dsm_m, z_dsm_m
"""

import sys
import math
import webbrowser
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────────────────────

mission = sys.argv[1] if len(sys.argv) > 1 else "evj_flyby"
out_dir = Path("out") / mission

AU     = 1.495_978_707e11       # m
MU_SUN = 1.327_124_400_18e20   # m³/s²

BG    = "#0f0f0f"
GRID  = "#1a1a1a"
TEXT  = "#cccccc"
LEG_COLORS = [
    "#00d4ff", "#ff6b35", "#7fba00", "#bf5fff",
    "#ffcf00", "#ff4081", "#00e676", "#ff9800",
]
BODY_COLORS = {
    "Sun":     "#ffff00",
    "Mercury": "#aaaaaa",
    "Venus":   "#e8c97e",
    "Earth":   "#4fc3f7",
    "Mars":    "#ef5350",
    "Jupiter": "#ffb74d",
    "Saturn":  "#f9a825",
    "Uranus":  "#80cbc4",
    "Neptune": "#1565c0",
}
BODY_SMA_AU = {
    "Mercury": 0.387, "Venus": 0.723, "Earth": 1.000,
    "Mars": 1.524,    "Jupiter": 5.203, "Saturn": 9.537,
    "Uranus": 19.19,  "Neptune": 30.07,
}

def body_color(name: str) -> str:
    return BODY_COLORS.get(name, "#ffffff")

def orbital_ring(sma_au: float, n_pts: int = 200):
    """XY coords of a circular orbit ring [AU]."""
    theta = np.linspace(0, 2 * math.pi, n_pts)
    return sma_au * np.cos(theta), sma_au * np.sin(theta)


# ── Load data ─────────────────────────────────────────────────────────────────

def load(name: str):
    p = out_dir / name
    if not p.exists():
        print(f"[warn] {p} not found")
        return None
    return pd.read_csv(p)

arc  = load("mga_best.csv")
legs = load("mga_legs.csv")

if arc is None:
    print(f"No mga_best.csv in {out_dir}. Run: cargo run -p mission_planner optimize config/{mission}.toml")
    sys.exit(1)

arc["x_au"] = arc["x_m"] / AU
arc["y_au"] = arc["y_m"] / AU

# Determine which bodies to draw rings for.
ring_bodies = set()
if legs is not None:
    for _, r in legs.iterrows():
        ring_bodies.add(r["body_dep"].strip())
        ring_bodies.add(r["body_arr"].strip())

# ── Figure ────────────────────────────────────────────────────────────────────

fig = go.Figure()
fig.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=TEXT, family="monospace", size=12),
    title=dict(
        text=f"MGA-1DSM ecliptic view — {mission}",
        font=dict(color="#00d4ff", size=18),
        x=0.5,
    ),
    xaxis=dict(title="X [AU]", gridcolor=GRID, color="#888",
               scaleanchor="y", scaleratio=1),
    yaxis=dict(title="Y [AU]", gridcolor=GRID, color="#888"),
    height=750, width=800,
    legend=dict(bgcolor="#111", bordercolor="#333", borderwidth=1, font=dict(size=11)),
)

# ── Orbital rings ─────────────────────────────────────────────────────────────

for body in ring_bodies:
    if body in BODY_SMA_AU:
        rx, ry = orbital_ring(BODY_SMA_AU[body])
        fig.add_trace(go.Scatter(
            x=rx, y=ry, mode="lines",
            line=dict(color="#2a2a2a", width=1, dash="dot"),
            name=f"{body} orbit",
            showlegend=False,
        ))

# ── Transfer arcs ─────────────────────────────────────────────────────────────

n_legs = int(arc["leg_idx"].max()) + 1
for k in range(n_legs):
    seg = arc[arc["leg_idx"] == k]
    col = LEG_COLORS[k % len(LEG_COLORS)]
    fig.add_trace(go.Scatter(
        x=seg["x_au"], y=seg["y_au"],
        mode="lines",
        line=dict(color=col, width=2.5),
        name=f"Leg {k}",
    ))

# ── DSM positions and encounter bodies ────────────────────────────────────────

if legs is not None:
    body_seq = []
    if not legs.empty:
        body_seq.append(legs.iloc[0]["body_dep"].strip())
        for _, r in legs.iterrows():
            body_seq.append(r["body_arr"].strip())

    for i, (_, leg) in enumerate(legs.iterrows()):
        col = LEG_COLORS[i % len(LEG_COLORS)]

        # DSM position (from propagated Kepler arc + Lambert).
        dsm_x = float(leg["x_dsm_m"]) / AU
        dsm_y = float(leg["y_dsm_m"]) / AU
        fig.add_trace(go.Scatter(
            x=[dsm_x], y=[dsm_y],
            mode="markers",
            marker=dict(color=col, size=8, symbol="x",
                        line=dict(color="white", width=1.5)),
            name=f"DSM {i}",
            showlegend=True,
        ))

    # Encounter body positions at each epoch (departure of each leg = position of body_dep at t_dep).
    # We read body positions from the first/last arc points of each leg as a proxy.
    for i, (_, leg) in enumerate(legs.iterrows()):
        # Arc start of this leg = spacecraft at departure body position at t_dep.
        leg_arc = arc[arc["leg_idx"] == i]
        if leg_arc.empty:
            continue

        # Departure body position: first point of this leg's arc.
        x0 = float(leg_arc.iloc[0]["x_au"])
        y0 = float(leg_arc.iloc[0]["y_au"])
        bdep = leg["body_dep"].strip()
        fig.add_trace(go.Scatter(
            x=[x0], y=[y0],
            mode="markers+text",
            marker=dict(color=body_color(bdep), size=12, symbol="circle",
                        line=dict(color="white", width=1)),
            text=[bdep],
            textposition="top right",
            textfont=dict(color=body_color(bdep), size=10),
            name=bdep,
            showlegend=(i == 0),
            legendgroup=bdep,
        ))

        # Final body (last leg): also mark the arrival position.
        if i == n_legs - 1:
            x1 = float(leg_arc.iloc[-1]["x_au"])
            y1 = float(leg_arc.iloc[-1]["y_au"])
            barr = leg["body_arr"].strip()
            fig.add_trace(go.Scatter(
                x=[x1], y=[y1],
                mode="markers+text",
                marker=dict(color=body_color(barr), size=12, symbol="circle",
                            line=dict(color="white", width=1)),
                text=[barr],
                textposition="top right",
                textfont=dict(color=body_color(barr), size=10),
                name=barr,
                showlegend=True,
            ))

# ── Sun ───────────────────────────────────────────────────────────────────────

fig.add_trace(go.Scatter(
    x=[0], y=[0],
    mode="markers+text",
    marker=dict(color="#ffff00", size=16, symbol="circle",
                line=dict(color="#ffaa00", width=2)),
    text=["Sun"],
    textposition="bottom right",
    textfont=dict(color="#ffff00", size=11),
    name="Sun",
))

# ── Info annotation from legs data ────────────────────────────────────────────

if legs is not None and not legs.empty:
    dv_dsm_total = legs["dv_dsm_ms"].sum()
    seq_str = " → ".join(body_seq)
    tof_total = legs["tof_days"].sum()
    info_text = (
        f"<b>Sequence:</b> {seq_str}<br>"
        f"<b>Total TOF:</b> {tof_total:.1f} days<br>"
        f"<b>Total DSM ΔV:</b> {dv_dsm_total:.1f} m/s"
    )
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=info_text,
        showarrow=False,
        align="left",
        font=dict(color="#aaa", size=11),
        bgcolor="#111",
        bordercolor="#333",
        borderwidth=1,
    )

# ── Save ──────────────────────────────────────────────────────────────────────

out_html = out_dir / "mga_overview.html"
fig.write_html(str(out_html))
print(f"Saved: {out_html}")
webbrowser.open(str(out_html))
