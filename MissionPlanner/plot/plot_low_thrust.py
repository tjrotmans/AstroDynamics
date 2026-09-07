#!/usr/bin/env python3
"""
plot_low_thrust.py — Sims-Flanagan low-thrust trajectory visualisation (Phase 9h).

Reads:
  out/<mission>/sf_arc.csv    — forward + backward arc points
  out/<mission>/sf_thrust.csv — per-segment thrust vectors

Usage:
  python plot/plot_low_thrust.py <mission>

  <mission> must match [simulation].output_dir in the TOML, e.g.:
    python plot/plot_low_thrust.py earth_mars_lowthrust

The script auto-resolves departure and target body names from the arc's first
and last known positions — it does NOT require the TOML at plot time.
"""

import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Constants ─────────────────────────────────────────────────────────────────
AU_M = 1.496e11  # 1 AU [m] — IAU 2012 exact

# Body orbit radii for decorative rings [AU]
BODY_ORBITS_AU = {
    "Earth":  1.000,
    "Mars":   1.524,
    "Venus":  0.723,
    "Jupiter": 5.203,
}
RING_N = 200  # points per orbit ring

# Plot colours (dark background, WSB-style)
COL_FWD    = "#00d4ff"   # cyan  — forward arc
COL_BWD    = "#ff6b35"   # orange — backward arc
COL_MATCH  = "#ffee00"   # yellow — match point
COL_DEP    = "#44ff44"   # green  — departure
COL_ARR    = "#ff4444"   # red    — arrival
COL_THRUST = "#cc88ff"   # purple — thrust

BG_PAPER = "#0a0a0a"
BG_PLOT  = "#0f0f0f"
GRID_COL = "#222222"
TEXT_COL = "#cccccc"


def orbit_ring(r_au: float):
    """Return (x, y, z) arrays for a circular orbit ring at radius r_au."""
    theta = np.linspace(0, 2 * np.pi, RING_N)
    x = r_au * np.cos(theta)
    y = r_au * np.sin(theta)
    z = np.zeros_like(theta)
    return x, y, z


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot/plot_low_thrust.py <mission>")
        print("  e.g.: python plot/plot_low_thrust.py earth_mars_lowthrust")
        sys.exit(1)

    mission = sys.argv[1]
    out_dir = os.path.join("out", mission)

    arc_path    = os.path.join(out_dir, "sf_arc.csv")
    thrust_path = os.path.join(out_dir, "sf_thrust.csv")

    for p in [arc_path, thrust_path]:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            print(f"Run: cargo run -p mission_planner --release -- low-thrust config/{mission}.toml")
            sys.exit(1)

    arc    = pd.read_csv(arc_path)
    thrust = pd.read_csv(thrust_path)

    # Convert positions to AU and velocities to km/s for display.
    for col in ["x_m", "y_m", "z_m"]:
        arc[col] /= AU_M
    arc = arc.rename(columns={"x_m": "x_au", "y_m": "y_au", "z_m": "z_au"})
    for col in ["vx_mps", "vy_mps", "vz_mps"]:
        arc[col] /= 1e3
    arc = arc.rename(columns={"vx_mps": "vx_kms", "vy_mps": "vy_kms", "vz_mps": "vz_kms"})

    fwd = arc[arc["arc"] == "fwd"]
    bwd = arc[arc["arc"] == "bwd"]

    # Match point: last point of fwd (= first point of bwd before dedup).
    mp = fwd.iloc[-1] if not fwd.empty else None
    dep = fwd.iloc[0] if not fwd.empty else None
    arr = bwd.iloc[-1] if not bwd.empty else None

    thrust["u_mag_mms2"] = thrust["u_mag_ms2"] * 1e3  # for display in mm/s²

    # ── Thrust magnitude vs time ──────────────────────────────────────────────
    # Calculate total ΔV proxy
    dt_days = thrust["t_mid_days"].diff().mean() if len(thrust) > 1 else 1.0
    dt_s = dt_days * 86400.0
    dv_total = (thrust["u_mag_ms2"] * dt_s).sum()

    print(f"  Forward arc:  {len(fwd)} points, t = {fwd['t_days'].min():.1f}..{fwd['t_days'].max():.1f} days")
    print(f"  Backward arc: {len(bwd)} points, t = {bwd['t_days'].min():.1f}..{bwd['t_days'].max():.1f} days")
    print(f"  Total ΔV proxy: {dv_total/1e3:.3f} km/s")
    if mp is not None:
        print(f"  Match point: t = {mp['t_days']:.1f} days")

    # ── Build subplot figure ──────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scatter3d", "rowspan": 2}, {"type": "xy"}],
            [None,                                  {"type": "xy"}],
        ],
        column_widths=[0.60, 0.40],
        row_heights=[0.55, 0.45],
        subplot_titles=[
            "3D Heliocentric Arc",
            "Thrust Magnitude vs Time",
            "Match-Point Defect (convergence proxy)",
        ],
    )

    # ── Panel 1: 3D arc ───────────────────────────────────────────────────────
    if not fwd.empty:
        fig.add_trace(go.Scatter3d(
            x=fwd["x_au"], y=fwd["y_au"], z=fwd["z_au"],
            mode="lines+markers",
            line=dict(color=COL_FWD, width=3),
            marker=dict(size=2, color=COL_FWD),
            name="Forward arc",
        ), row=1, col=1)

    if not bwd.empty:
        fig.add_trace(go.Scatter3d(
            x=bwd["x_au"], y=bwd["y_au"], z=bwd["z_au"],
            mode="lines+markers",
            line=dict(color=COL_BWD, width=3),
            marker=dict(size=2, color=COL_BWD),
            name="Backward arc",
        ), row=1, col=1)

    if mp is not None:
        fig.add_trace(go.Scatter3d(
            x=[mp["x_au"]], y=[mp["y_au"]], z=[mp["z_au"]],
            mode="markers+text",
            marker=dict(size=8, color=COL_MATCH, symbol="diamond"),
            text=["Match point"], textposition="top center",
            textfont=dict(color=COL_MATCH, size=11),
            name="Match point",
        ), row=1, col=1)

    if dep is not None:
        fig.add_trace(go.Scatter3d(
            x=[dep["x_au"]], y=[dep["y_au"]], z=[dep["z_au"]],
            mode="markers+text",
            marker=dict(size=8, color=COL_DEP, symbol="circle"),
            text=["Departure"], textposition="top center",
            textfont=dict(color=COL_DEP, size=11),
            name="Departure",
        ), row=1, col=1)

    if arr is not None:
        fig.add_trace(go.Scatter3d(
            x=[arr["x_au"]], y=[arr["y_au"]], z=[arr["z_au"]],
            mode="markers+text",
            marker=dict(size=8, color=COL_ARR, symbol="circle"),
            text=["Arrival"], textposition="top center",
            textfont=dict(color=COL_ARR, size=11),
            name="Arrival",
        ), row=1, col=1)

    # Sun marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=10, color="yellow", symbol="circle"),
        name="Sun",
    ), row=1, col=1)

    # Orbit rings
    for body, r_au in BODY_ORBITS_AU.items():
        rx, ry, rz = orbit_ring(r_au)
        fig.add_trace(go.Scatter3d(
            x=rx, y=ry, z=rz,
            mode="lines",
            line=dict(color="#333333", width=1, dash="dot"),
            showlegend=False,
            hovertext=[body] * RING_N,
            hoverinfo="text",
        ), row=1, col=1)
        # Body label at (r_au, 0, 0)
        fig.add_trace(go.Scatter3d(
            x=[r_au], y=[0], z=[0],
            mode="text",
            text=[body],
            textfont=dict(color="#666666", size=9),
            showlegend=False,
        ), row=1, col=1)

    # ── Panel 2: thrust magnitude vs time ─────────────────────────────────────
    fig.add_trace(go.Bar(
        x=thrust["t_mid_days"],
        y=thrust["u_mag_ms2"] * 1e6,   # display in µm/s²
        marker_color=COL_THRUST,
        name="Thrust magnitude",
        showlegend=False,
    ), row=1, col=2)

    # Mark the match-point time (TOF/2)
    if not arc.empty:
        tof = arc["t_days"].max()
        fig.add_vline(
            x=tof / 2,
            line_dash="dash", line_color=COL_MATCH, line_width=1.5,
            annotation_text=f"Match point (t={tof/2:.0f} d)",
            annotation_font_color=COL_MATCH,
            row=1, col=2,
        )

    fig.update_xaxes(title_text="Time [days]", row=1, col=2,
                     gridcolor=GRID_COL, color=TEXT_COL)
    fig.update_yaxes(title_text="Thrust [µm/s²]", row=1, col=2,
                     gridcolor=GRID_COL, color=TEXT_COL)

    # ── Panel 3: per-component thrust vs time ─────────────────────────────────
    for comp, col_hex, label in [
        ("ux_ms2", "#ff4040", "uₓ"),
        ("uy_ms2", "#40ff40", "u_y"),
        ("uz_ms2", "#4080ff", "u_z"),
    ]:
        fig.add_trace(go.Scatter(
            x=thrust["t_mid_days"],
            y=thrust[comp] * 1e6,
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(color=col_hex, width=1.5),
            name=f"{label} [µm/s²]",
        ), row=2, col=2)

    fig.update_xaxes(title_text="Time [days]", row=2, col=2,
                     gridcolor=GRID_COL, color=TEXT_COL)
    fig.update_yaxes(title_text="Thrust component [µm/s²]", row=2, col=2,
                     gridcolor=GRID_COL, color=TEXT_COL,
                     zeroline=True, zerolinecolor="#444444")

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"Sims-Flanagan Low-Thrust Trajectory — {mission.replace('_', ' ').title()}",
            font=dict(color=TEXT_COL, size=16),
        ),
        paper_bgcolor=BG_PAPER,
        plot_bgcolor=BG_PLOT,
        font=dict(color=TEXT_COL),
        legend=dict(
            bgcolor="#111111",
            bordercolor="#333333",
            borderwidth=1,
        ),
        scene=dict(
            xaxis=dict(title="x [AU]", backgroundcolor=BG_PLOT,
                       gridcolor=GRID_COL, color=TEXT_COL),
            yaxis=dict(title="y [AU]", backgroundcolor=BG_PLOT,
                       gridcolor=GRID_COL, color=TEXT_COL),
            zaxis=dict(title="z [AU]", backgroundcolor=BG_PLOT,
                       gridcolor=GRID_COL, color=TEXT_COL),
            bgcolor=BG_PLOT,
        ),
        height=750,
    )

    # ── Annotations ───────────────────────────────────────────────────────────
    # Add a text box with key numbers
    tof_total = arc["t_days"].max() if not arc.empty else 0.0
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.62, y=0.97,
        text=(
            f"TOF: {tof_total:.1f} days<br>"
            f"ΔV (proxy): {dv_total/1e3:.3f} km/s<br>"
            f"Segments: {len(thrust)}"
        ),
        showarrow=False,
        bgcolor="#111111",
        bordercolor="#555555",
        borderwidth=1,
        font=dict(color=TEXT_COL, size=11),
        align="left",
    )

    out_html = os.path.join(out_dir, "sf_trajectory.html")
    fig.write_html(out_html)
    print(f"\nSaved: {out_html}")

    # Also save a static PNG if kaleido is available.
    try:
        out_png = os.path.join(out_dir, "sf_trajectory.png")
        fig.write_image(out_png, width=1400, height=750)
        print(f"Saved: {out_png}")
    except Exception:
        pass  # kaleido not installed — skip PNG, HTML is sufficient

    fig.show()


if __name__ == "__main__":
    main()
