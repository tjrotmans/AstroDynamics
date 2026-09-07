#!/usr/bin/env python3
"""
Artemis 2 — Earth-Moon rotating frame dashboard

Transforms the ECI trajectory into the co-rotating Earth-Moon frame and
creates an interactive Plotly dashboard with multiple views.

Run from AstroProbs/Artemis after cargo run -p artemis --release:
    python plot/plot_rotating.py
"""

from __future__ import annotations

import argparse
import os
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "out", "artemis2_trajectory.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")
EARTH_RADIUS_KM = 6_371.0
MOON_RADIUS_KM = 1_737.4


def load_eci_trajectory(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def rotating_frame(sc_pos: np.ndarray, moon_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform spacecraft ECI positions to the Earth-Moon rotating frame.

    The z-axis is estimated once from the mean angular momentum of the Moon's
    orbit (cross product of consecutive Moon positions, averaged over the first
    N pairs) — stable because the Moon's orbital plane barely changes over 10 days.

    Parameters
    ----------
    sc_pos   : (N, 3) ndarray — spacecraft ECI [km]
    moon_pos : (N, 3) ndarray — Moon ECI [km]

    Returns
    -------
    sc_rot   : (N, 3) ndarray — spacecraft in rotating frame [km]
    moon_rot : (N,)   ndarray — Moon x-coordinate in rotating frame [km]
               (by construction the Moon is always at (moon_rot, 0, 0))
    """
    n_avg = min(50, len(moon_pos) - 1)
    h_vecs = np.cross(moon_pos[:n_avg], moon_pos[1:n_avg + 1])
    h_norms = np.linalg.norm(h_vecs, axis=1, keepdims=True)
    z_hat = (h_vecs / np.where(h_norms > 0, h_norms, 1)).mean(axis=0)
    z_hat /= np.linalg.norm(z_hat)

    n = len(sc_pos)
    sc_rot = np.zeros((n, 3))
    moon_rot = np.zeros(n)

    for i in range(n):
        m_norm = np.linalg.norm(moon_pos[i])
        if m_norm < 1e-6:
            continue
        x_hat = moon_pos[i] / m_norm
        y_hat = np.cross(z_hat, x_hat)
        y_norm = np.linalg.norm(y_hat)
        if y_norm < 1e-10:
            continue
        y_hat /= y_norm
        R = np.array([x_hat, y_hat, z_hat])   # rows = frame axes
        sc_rot[i] = R @ sc_pos[i]
        moon_rot[i] = m_norm   # Moon is on the +x axis by construction

    return sc_rot, moon_rot


def make_dark_layout(title: str) -> dict[str, Any]:
    return {
        "template": "plotly_dark",
        "title": {"text": title, "x": 0.01, "xanchor": "left"},
        "paper_bgcolor": "rgb(15,15,25)",
        "plot_bgcolor": "rgb(15,15,25)",
        "font": {"color": "white", "size": 12},
        "legend": {"bgcolor": "rgba(20,20,35,0.8)", "bordercolor": "rgba(255,255,255,0.15)"},
    }


def build_rotating_3d_figure(
    sc_rot: np.ndarray,
    moon_rot: np.ndarray,
    burn: np.ndarray,
    t_days: np.ndarray,
    ca_idx: int,
) -> go.Figure:
    fig = go.Figure()

    # Trajectory color-coded by time
    fig.add_trace(go.Scatter3d(
        x=sc_rot[:, 0], y=sc_rot[:, 1], z=sc_rot[:, 2],
        mode="lines",
        line={"color": t_days, "colorscale": "Viridis", "width": 4},
        name="Spacecraft trajectory",
        hovertemplate="Day %{customdata:.2f}<br>X=%{x:.0f} km<br>Y=%{y:.0f} km<br>Z=%{z:.0f} km<extra></extra>",
        customdata=t_days,
    ))

    if burn.any():
        fig.add_trace(go.Scatter3d(
            x=sc_rot[burn, 0], y=sc_rot[burn, 1], z=sc_rot[burn, 2],
            mode="lines",
            line={"color": "#EF5350", "width": 5},
            name="TLI burn",
        ))

    # Markers
    fig.add_trace(go.Scatter3d(
        x=[sc_rot[0, 0]], y=[sc_rot[0, 1]], z=[sc_rot[0, 2]],
        mode="markers",
        marker={"size": 7, "color": "#69F0AE"},
        name="TLI ignition",
    ))

    fig.add_trace(go.Scatter3d(
        x=[sc_rot[ca_idx, 0]], y=[sc_rot[ca_idx, 1]], z=[sc_rot[ca_idx, 2]],
        mode="markers",
        marker={"size": 8, "color": "#FF6F00"},
        name="Closest approach",
    ))

    fig.add_trace(go.Scatter3d(
        x=[sc_rot[-1, 0]], y=[sc_rot[-1, 1]], z=[sc_rot[-1, 2]],
        mode="markers",
        marker={"size": 8, "symbol": "x", "color": "#EF5350"},
        name="End",
    ))

    # Earth sphere
    phi = np.linspace(0, np.pi, 24)
    theta = np.linspace(0, 2 * np.pi, 48)
    phi, theta = np.meshgrid(phi, theta)
    x = EARTH_RADIUS_KM * np.sin(phi) * np.cos(theta)
    y = EARTH_RADIUS_KM * np.sin(phi) * np.sin(theta)
    z = EARTH_RADIUS_KM * np.cos(phi)
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        surfacecolor=np.zeros_like(x),
        colorscale=[[0, "rgb(21,101,192)"], [1, "rgb(21,101,192)"]],
        showscale=False,
        opacity=0.5,
        name="Earth",
    ))

    # Moon sphere (at mean position)
    moon_x_mean = moon_rot.mean()
    x = moon_x_mean + MOON_RADIUS_KM * 2 * np.sin(phi) * np.cos(theta)
    y = MOON_RADIUS_KM * 2 * np.sin(phi) * np.sin(theta)
    z = MOON_RADIUS_KM * 2 * np.cos(phi)
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        surfacecolor=np.ones_like(x) * 0.5,
        colorscale=[[0, "rgb(128,128,128)"], [1, "rgb(128,128,128)"]],
        showscale=False,
        opacity=0.4,
        name="Moon",
    ))

    fig.update_layout(
        **make_dark_layout("Artemis 2 — Earth-Moon Rotating Frame (3D)"),
        scene={
            "xaxis": {"title": "x (Earth→Moon) [km]"},
            "yaxis": {"title": "y (in-plane) [km]"},
            "zaxis": {"title": "z (out-of-plane) [km]"},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )

    return fig


def build_rotating_panels_figure(
    sc_rot: np.ndarray,
    moon_rot: np.ndarray,
    burn: np.ndarray,
    t_days: np.ndarray,
    ca_idx: int,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("XY — Orbital Plane", "XZ — Out-of-Plane", "YZ — Cross-Track"),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )

    moon_x_mean = moon_rot.mean()
    ca_alt_km = np.linalg.norm(sc_rot[ca_idx] - np.array([moon_x_mean, 0, 0])) - MOON_RADIUS_KM

    # Panel configurations: (xi, yi, xlabel, ylabel, row, col)
    panels = [
        (0, 1, "x (Earth→Moon) [km]", "y (in-plane) [km]", 1, 1),
        (0, 2, "x (Earth→Moon) [km]", "z (out-of-plane) [km]", 1, 2),
        (1, 2, "y (in-plane) [km]", "z (out-of-plane) [km]", 2, 1),
    ]

    for xi, yi, xlabel, ylabel, row, col in panels:
        # Earth circle
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=EARTH_RADIUS_KM * np.cos(theta),
            y=EARTH_RADIUS_KM * np.sin(theta),
            mode="lines",
            line={"color": "rgb(21,101,192)", "width": 2},
            fill="toself",
            fillcolor="rgba(21,101,192,0.3)",
            name="Earth",
            showlegend=(row == 1 and col == 1),
        ), row=row, col=col)

        # Moon circle
        moon_cx = moon_x_mean if xi == 0 else 0.0
        moon_cy = 0.0
        fig.add_trace(go.Scatter(
            x=moon_cx + MOON_RADIUS_KM * 3 * np.cos(theta),
            y=moon_cy + MOON_RADIUS_KM * 3 * np.sin(theta),
            mode="lines",
            line={"color": "gray", "width": 2},
            fill="toself",
            fillcolor="rgba(128,128,128,0.2)",
            name="Moon",
            showlegend=(row == 1 and col == 1),
        ), row=row, col=col)

        # Trajectory segments color-coded by time
        for i in range(len(sc_rot) - 1):
            if not burn[i]:
                frac = t_days[i] / t_days[-1]
                color = f"rgba({int(frac*255)},{int((1-frac)*255)},255,0.8)"
                fig.add_trace(go.Scatter(
                    x=sc_rot[i:i+2, xi],
                    y=sc_rot[i:i+2, yi],
                    mode="lines",
                    line={"color": color, "width": 2},
                    showlegend=False,
                ), row=row, col=col)

        # TLI burn
        if burn.any():
            fig.add_trace(go.Scatter(
                x=sc_rot[burn, xi],
                y=sc_rot[burn, yi],
                mode="lines",
                line={"color": "#EF5350", "width": 4},
                name="TLI burn",
                showlegend=(row == 1 and col == 1),
            ), row=row, col=col)

        # Markers
        fig.add_trace(go.Scatter(
            x=[sc_rot[0, xi]], y=[sc_rot[0, yi]],
            mode="markers",
            marker={"color": "#69F0AE", "size": 8},
            name="TLI ignition",
            showlegend=(row == 1 and col == 1),
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=[sc_rot[ca_idx, xi]], y=[sc_rot[ca_idx, yi]],
            mode="markers",
            marker={"color": "#FF6F00", "size": 10},
            name=f"CA: {ca_alt_km:.0f} km alt",
            showlegend=(row == 1 and col == 1),
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=[sc_rot[-1, xi]], y=[sc_rot[-1, yi]],
            mode="markers",
            marker={"color": "#EF5350", "symbol": "x", "size": 10},
            name="End",
            showlegend=(row == 1 and col == 1),
        ), row=row, col=col)

        fig.update_xaxes(title_text=xlabel, row=row, col=col)
        fig.update_yaxes(title_text=ylabel, row=row, col=col, scaleanchor=f"x{row}{col}", scaleratio=1)

    # Colorbar for time
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker={"colorscale": "Viridis", "showscale": True,
                "colorbar": {"title": "Mission elapsed time [days]",
                           "titleside": "right"}},
        showlegend=False,
    ), row=2, col=2)

    fig.update_layout(
        **make_dark_layout("Artemis 2 — Earth-Moon Rotating Frame Panels"),
        showlegend=True,
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Artemis 2 rotating frame dashboard")
    parser.add_argument("--no-open", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        print("Run: cargo run -p artemis --release")
        return

    df = pd.read_csv(CSV_PATH)
    km = 1e-3

    sc_pos = df[["x_m", "y_m", "z_m"]].values * km
    moon_pos = df[["moon_x_m", "moon_y_m", "moon_z_m"]].values * km
    burn = df["is_burn"].values.astype(bool)
    t_days = df["time_s"].values / 86_400.0

    sc_rot, moon_rot = rotating_frame(sc_pos, moon_pos)

    sc_moon_dist = np.linalg.norm(sc_pos - moon_pos, axis=1)
    ca_idx = np.argmin(sc_moon_dist)
    ca_alt_km = sc_moon_dist[ca_idx] - MOON_RADIUS_KM

    print(f"Closest approach: {ca_alt_km:.0f} km altitude (target: 6,513 km) at T+{t_days[ca_idx]:.2f} days")

    # Create dashboard with 3D and panels
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("3D Rotating Frame", "2D Panel Views"),
        specs=[[{"type": "scene"}],
               [{"type": "xy"}]],
        vertical_spacing=0.1,
    )

    # Add 3D rotating frame
    rot_3d = build_rotating_3d_figure(sc_rot, moon_rot, burn, t_days, ca_idx)
    for trace in rot_3d.data:
        fig.add_trace(trace, row=1, col=1)

    # Add panels (we'll just show one main panel for the dashboard)
    panels = build_rotating_panels_figure(sc_rot, moon_rot, burn, t_days, ca_idx)
    # Add the XY orbital plane panel
    for trace in panels.data[:10]:  # First subplot traces
        if hasattr(trace, 'x') and trace.x is not None:
            fig.add_trace(trace, row=2, col=1)

    # Update layout
    fig.update_layout(
        **make_dark_layout("Artemis 2 — Earth-Moon Rotating Frame Dashboard"),
        scene={
            "xaxis": {"title": "x (Earth→Moon) [km]"},
            "yaxis": {"title": "y (in-plane) [km]"},
            "zaxis": {"title": "z (out-of-plane) [km]"},
            "aspectmode": "data",
            "domain": {"x": [0.0, 1.0], "y": [0.3, 1.0]},
        },
        xaxis2={"title": "x (Earth→Moon) [km]", "domain": [0.0, 1.0]},
        yaxis2={"title": "y (in-plane) [km]", "domain": [0.0, 0.25]},
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )

    # Save and open
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "artemis_rotating_dashboard.html")
    fig.write_html(output_path)
    print(f"Saved dashboard to {output_path}")

    if not args.no_open:
        # Use pathlib to create proper file:// URL
        file_path = Path(output_path).resolve()
        file_url = file_path.as_uri()
        webbrowser.open(file_url)


if __name__ == "__main__":
    main()