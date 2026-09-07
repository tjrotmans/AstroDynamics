"""
MGA-1DSM animated trajectory -- Phase 9 backend visualisation.

Time-animated top-down ecliptic view of an MGA trajectory.  Each frame
advances time, showing:
  - Spacecraft moving along the transfer arc
  - Planets moving along their circular orbits
  - Encounter events (flyby, departure, arrival) highlighted

The planet positions at each epoch are derived from:
  1. The spacecraft arc's first point = departure body position at t=0
  2. Each planet is propagated forward from that position using a circular
     orbit approximation: theta(t) = theta_0 + omega * t
     where omega = sqrt(MU_SUN / a^3).

This is approximate (planets are on elliptical, inclined orbits) but
is visually accurate for the 2D ecliptic projection used here, and avoids
needing a separate ANISE call from Python.

Usage (from MissionPlanner/ directory):
  python plot/plot_mga_animation.py evj_flyby
  python plot/plot_mga_animation.py venus_saturn_auto --frames 120

Reads (from out/<mission>/):
  mga_best.csv      -- arc points: t_days, x_m, y_m, z_m, leg_idx
  mga_legs.csv      -- per-leg: body_dep, body_arr, t_dep_jd, t_arr_jd,
                       tof_days, eta, dv_dsm_ms, vinf_dep_ms, vinf_arr_ms,
                       turn_deg, rp_km, rp_norm, x_dsm_m, y_dsm_m, z_dsm_m

Output:
  out/<mission>/mga_animation.html
"""

import sys
import math
import argparse
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Constants ─────────────────────────────────────────────────────────────────

MU_SUN = 1.327_124_400_18e20   # m³/s² (IAU 2012)
AU     = 1.495_978_707e11       # m

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


# ── Orbital mechanics helpers ─────────────────────────────────────────────────

def circular_omega(sma_m: float) -> float:
    """Mean motion [rad/s] for a circular orbit."""
    return math.sqrt(MU_SUN / sma_m**3)


def planet_position_au(theta0_rad: float, sma_au: float, t_days: float) -> tuple[float, float]:
    """
    XY position [AU] of a planet on a circular orbit at t_days after its
    reference epoch.  theta0_rad is the planet's angle at t=0.
    """
    sma_m  = sma_au * AU
    omega  = circular_omega(sma_m)
    theta  = theta0_rad + omega * t_days * 86400.0
    return sma_au * math.cos(theta), sma_au * math.sin(theta)


def infer_theta0(pos_x_m: float, pos_y_m: float) -> float:
    """
    Infer the angle of a body from its known x,y position at t=0.
    Returns angle in [0, 2π).
    """
    return math.atan2(pos_y_m, pos_x_m)


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Animated MGA trajectory")
parser.add_argument("mission", nargs="?", default="evj_flyby",
                    help="Mission name (subdirectory under out/)")
parser.add_argument("--frames", type=int, default=360,
                    help="Number of animation frames (default 360)")
parser.add_argument("--output", default=None,
                    help="Save HTML to this path instead of auto-naming")
args = parser.parse_args()

mission = args.mission
out_dir = Path("out") / mission
n_frames = max(30, args.frames)

# ── Load data ─────────────────────────────────────────────────────────────────

arc_path  = out_dir / "mga_best.csv"
legs_path = out_dir / "mga_legs.csv"

if not arc_path.exists():
    print(f"[ERROR] {arc_path} not found.")
    print(f"Run: cargo run -p mission_planner -- optimize config/{mission}.toml")
    sys.exit(1)

arc  = pd.read_csv(arc_path)
legs = pd.read_csv(legs_path) if legs_path.exists() else None

arc["x_au"] = arc["x_m"] / AU
arc["y_au"] = arc["y_m"] / AU

t_min = float(arc["t_days"].min())
t_max = float(arc["t_days"].max())

# Build ordered body sequence.
body_seq = []
if legs is not None and not legs.empty:
    body_seq.append(legs.iloc[0]["body_dep"].strip())
    for _, r in legs.iterrows():
        body_seq.append(r["body_arr"].strip())

# ── Determine planet theta0 from arc data ─────────────────────────────────────
# The first arc point is the spacecraft at the departure body's position.
# Use that to pin the departure body's orbital angle at t=0.
# For other bodies in the sequence, use their known encounter positions.

body_theta0 = {}   # body_name -> theta0 [rad]

# Departure body: first arc point.
x0_au = float(arc.iloc[0]["x_au"])
y0_au = float(arc.iloc[0]["y_au"])
if body_seq:
    dep_body = body_seq[0]
    body_theta0[dep_body] = infer_theta0(x0_au * AU, y0_au * AU)

# Intermediate and arrival bodies: first arc point of the leg departing from them.
n_legs = int(arc["leg_idx"].max()) + 1
if legs is not None:
    for i, (_, leg) in enumerate(legs.iterrows()):
        barr = leg["body_arr"].strip()
        leg_arc = arc[arc["leg_idx"] == i]
        if leg_arc.empty:
            continue
        # Arrival body position = last point of this leg.
        xarr = float(leg_arc.iloc[-1]["x_au"])
        yarr = float(leg_arc.iloc[-1]["y_au"])
        t_arr = float(leg_arc.iloc[-1]["t_days"])
        # Back-project to find theta0 at t=0.
        if barr in BODY_SMA_AU:
            sma_m = BODY_SMA_AU[barr] * AU
            omega  = circular_omega(sma_m)
            theta_at_arr = infer_theta0(xarr * AU, yarr * AU)
            theta0 = theta_at_arr - omega * t_arr * 86400.0
            body_theta0[barr] = theta0

# Propagate orbits for bodies not yet in body_theta0 (those not in the sequence
# but relevant for context: only plot bodies actually in the sequence).
ring_bodies = sorted(set(b for b in body_seq if b in BODY_SMA_AU))

# ── Build animation frames ────────────────────────────────────────────────────

# Sample times for frames (not uniformly spaced — spend more frames on the
# middle portion to show slow evolution near Jupiter; end-compress for Saturn).
t_sample = np.linspace(t_min, t_max, n_frames)

# Static traces: orbital rings, Sun, full arc (faded background).
static_traces = []

# Orbital rings (static backdrop).
for body in ring_bodies:
    sma = BODY_SMA_AU[body]
    theta_ring = np.linspace(0, 2 * math.pi, 300)
    rx = sma * np.cos(theta_ring)
    ry = sma * np.sin(theta_ring)
    static_traces.append(go.Scatter(
        x=rx, y=ry, mode="lines",
        line=dict(color="#1e1e1e", width=1, dash="dot"),
        name=f"{body} orbit",
        showlegend=False,
        hoverinfo="skip",
    ))

# Full arc as faded background.
for k in range(n_legs):
    seg = arc[arc["leg_idx"] == k]
    col = LEG_COLORS[k % len(LEG_COLORS)]
    static_traces.append(go.Scatter(
        x=seg["x_au"].values, y=seg["y_au"].values,
        mode="lines",
        line=dict(color=col, width=1, dash="dot"),
        opacity=0.2,
        name=f"Leg {k} (full)",
        showlegend=False,
        hoverinfo="skip",
    ))

# DSM positions (static markers).
if legs is not None:
    for i, (_, leg) in enumerate(legs.iterrows()):
        col = LEG_COLORS[i % len(LEG_COLORS)]
        dsm_x = float(leg["x_dsm_m"]) / AU
        dsm_y = float(leg["y_dsm_m"]) / AU
        static_traces.append(go.Scatter(
            x=[dsm_x], y=[dsm_y],
            mode="markers",
            marker=dict(color=col, size=9, symbol="x",
                        line=dict(color="white", width=1.5)),
            name=f"DSM {i}  ΔV={float(leg['dv_dsm_ms']):.0f} m/s",
            showlegend=True,
        ))

# Sun.
static_traces.append(go.Scatter(
    x=[0], y=[0],
    mode="markers+text",
    marker=dict(color="#ffff00", size=16, symbol="circle",
                line=dict(color="#ffaa00", width=2)),
    text=["Sun"], textposition="bottom right",
    textfont=dict(color="#ffff00", size=11),
    name="Sun",
))

# ── Per-frame traces: spacecraft + planet positions ───────────────────────────

def build_frame(t_days: float):
    traces = []

    # Trajectory so far (spacecraft trail).
    arc_so_far = arc[arc["t_days"] <= t_days]
    for k in range(n_legs):
        seg = arc_so_far[arc_so_far["leg_idx"] == k]
        if seg.empty:
            continue
        col = LEG_COLORS[k % len(LEG_COLORS)]
        traces.append(go.Scatter(
            x=seg["x_au"].values, y=seg["y_au"].values,
            mode="lines",
            line=dict(color=col, width=2.5),
            name=f"Leg {k}",
            showlegend=bool(t_days == t_sample[0]),
        ))

    # Current spacecraft position + short comet tail (last 5% of elapsed arc).
    closest_idx = (arc["t_days"] - t_days).abs().argmin()
    closest = arc.iloc[closest_idx]
    sc_x = float(closest["x_au"])
    sc_y = float(closest["y_au"])
    leg_k = int(closest["leg_idx"])

    # Comet tail: last ~3% of total arc duration behind the spacecraft
    tail_start = max(t_min, t_days - (t_max - t_min) * 0.03)
    tail_seg = arc[(arc["t_days"] >= tail_start) & (arc["t_days"] <= t_days)]
    if len(tail_seg) >= 2:
        traces.append(go.Scatter(
            x=tail_seg["x_au"].values, y=tail_seg["y_au"].values,
            mode="lines",
            line=dict(color="#ffffff", width=3),
            opacity=0.6,
            showlegend=False,
            hoverinfo="skip",
        ))

    traces.append(go.Scatter(
        x=[sc_x], y=[sc_y],
        mode="markers",
        marker=dict(color="#ffffff", size=14, symbol="triangle-up",
                    line=dict(color=LEG_COLORS[leg_k % len(LEG_COLORS)], width=2.5)),
        name="Spacecraft",
        showlegend=True,
    ))

    # Planet positions at this time.
    for body in ring_bodies:
        if body not in body_theta0:
            continue
        sma = BODY_SMA_AU[body]
        theta0 = body_theta0[body]
        px, py = planet_position_au(theta0, sma, t_days)
        col = body_color(body)
        traces.append(go.Scatter(
            x=[px], y=[py],
            mode="markers+text",
            marker=dict(color=col, size=11, symbol="circle",
                        line=dict(color="white", width=1)),
            text=[body],
            textposition="top right",
            textfont=dict(color=col, size=9),
            name=body,
            showlegend=False,
        ))

    return traces


# Build all frames.
frames = []
for t in t_sample:
    frame_traces = build_frame(t)
    frames.append(go.Frame(
        data=frame_traces,
        name=f"{t:.1f}d",
        layout=go.Layout(
            title=dict(
                text=f"MGA-1DSM — {mission}  |  T+{t:.0f} days",
            )
        ),
    ))

# ── Initial figure ────────────────────────────────────────────────────────────

# Axis range from arc extent + padding.
x_ext = arc["x_au"].abs().max()
y_ext = arc["y_au"].abs().max()
ext   = max(x_ext, y_ext) * 1.15

sequence_str = " → ".join(body_seq) if body_seq else mission

fig = go.Figure(
    data=static_traces + build_frame(t_sample[0]),
    layout=go.Layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="monospace", size=12),
        title=dict(
            text=f"MGA-1DSM — {sequence_str}  |  T+0 days",
            font=dict(color="#00d4ff", size=17),
            x=0.5,
        ),
        xaxis=dict(
            title="X [AU]", range=[-ext, ext],
            gridcolor=GRID, color="#888",
            scaleanchor="y", scaleratio=1,
        ),
        yaxis=dict(
            title="Y [AU]", range=[-ext, ext],
            gridcolor=GRID, color="#888",
        ),
        height=800, width=950,
        legend=dict(bgcolor="#111", bordercolor="#333", borderwidth=1, font=dict(size=10)),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.0, x=1.02,
            xanchor="left", yanchor="top",
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": max(30, 6000 // n_frames), "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    }],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            pad={"t": 50},
            steps=[
                dict(
                    method="animate",
                    label=f"{t:.0f}d",
                    args=[[f"{t:.1f}d"], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                )
                for t in t_sample[::max(1, n_frames // 30)]  # every ~30 ticks on slider
            ],
        )],
    ),
    frames=frames,
)

# ── Save ──────────────────────────────────────────────────────────────────────

out_path = args.output if args.output else str(out_dir / "mga_animation.html")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
fig.write_html(out_path, include_plotlyjs=True)
print(f"Saved: {out_path}")
webbrowser.open(out_path)
