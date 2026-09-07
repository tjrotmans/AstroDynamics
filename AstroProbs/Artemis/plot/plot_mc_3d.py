#!/usr/bin/env python3
"""
Monte Carlo final positions overlaid on 3D ECI trajectory dashboard

Reads out/artemis2_trajectory.csv and out/mc_final_positions.csv and creates an interactive
Plotly dashboard showing:
  - Spacecraft trajectory colour-coded by mission elapsed time
  - Earth sphere at origin
  - Moon positions along the track (gray dots)
  - TLI burn segment highlighted in red
  - Closest approach to Moon annotated
  - MC final positions as small cyan dots

Run from AstroProbs/Artemis after:
  cargo run -p artemis --release
  cargo run -p artemis --bin MC --release
  python plot/plot_mc_3d.py
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

TRAJ_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "out", "artemis2_trajectory.csv")
MC_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "out", "mc_top_100_positions.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")
EARTH_RADIUS_KM = 6_371.0
MOON_RADIUS_KM = 1_737.4
MOON_MEAN_KM = 384_400.0


def load_trajectory_data(traj_path: str, mc_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory CSV not found: {traj_path}")
    if not os.path.exists(mc_path):
        raise FileNotFoundError(f"MC CSV not found: {mc_path}")

    df_traj = pd.read_csv(traj_path)
    df_mc = pd.read_csv(mc_path)
    return df_traj, df_mc


def make_dark_layout(title: str) -> dict[str, Any]:
    return {
        "template": "plotly_dark",
        "title": {"text": title, "x": 0.01, "xanchor": "left"},
        "paper_bgcolor": "rgb(15,15,25)",
        "plot_bgcolor": "rgb(15,15,25)",
        "font": {"color": "white", "size": 12},
        "legend": {"bgcolor": "rgba(20,20,35,0.8)", "bordercolor": "rgba(255,255,255,0.15)"},
    }


def build_eci_mc_3d_figure(
    sc_pos: np.ndarray,
    moon_pos: np.ndarray,
    mc_pos: np.ndarray,
    burn: np.ndarray,
    t_days: np.ndarray,
    ca_idx: int,
    mc_alt: np.ndarray,
) -> go.Figure:
    fig = go.Figure()

    # Trajectory color-coded by time
    fig.add_trace(go.Scatter3d(
        x=sc_pos[:, 0], y=sc_pos[:, 1], z=sc_pos[:, 2],
        mode="lines",
        line={"color": t_days, "colorscale": "Viridis", "width": 4},
        name="Spacecraft trajectory",
        hovertemplate="Day %{customdata:.2f}<br>X=%{x:.0f} km<br>Y=%{y:.0f} km<br>Z=%{z:.0f} km<extra></extra>",
        customdata=t_days,
    ))

    # TLI burn
    if burn.any():
        fig.add_trace(go.Scatter3d(
            x=sc_pos[burn, 0], y=sc_pos[burn, 1], z=sc_pos[burn, 2],
            mode="lines",
            line={"color": "#EF5350", "width": 5},
            name="TLI burn",
        ))

    # Moon track
    fig.add_trace(go.Scatter3d(
        x=moon_pos[::10, 0], y=moon_pos[::10, 1], z=moon_pos[::10, 2],
        mode="markers",
        marker={"size": 3, "color": "gray", "opacity": 0.3},
        name="Moon orbit track",
    ))

    # Moon at closest approach
    ca_alt_km = np.linalg.norm(sc_pos[ca_idx] - moon_pos[ca_idx]) - MOON_RADIUS_KM
    fig.add_trace(go.Scatter3d(
        x=[moon_pos[ca_idx, 0]], y=[moon_pos[ca_idx, 1]], z=[moon_pos[ca_idx, 2]],
        mode="markers",
        marker={"size": 8, "color": "white", "line": {"color": "#888", "width": 1}},
        name=f"Moon at CA (T+{t_days[ca_idx]:.2f} d)",
    ))

    # MC final positions
    fig.add_trace(go.Scatter3d(
        x=mc_pos[:, 0], y=mc_pos[:, 1], z=mc_pos[:, 2],
        mode="markers",
        marker={"size": 4, "color": "cyan", "opacity": 0.6},
        name=f"MC final positions (n={len(mc_pos)})",
        hovertemplate="Alt: %{customdata:.0f} km<extra></extra>",
        customdata=mc_alt,
    ))

    # Markers
    fig.add_trace(go.Scatter3d(
        x=[sc_pos[0, 0]], y=[sc_pos[0, 1]], z=[sc_pos[0, 2]],
        mode="markers",
        marker={"size": 7, "color": "#69F0AE"},
        name="TLI ignition",
    ))

    fig.add_trace(go.Scatter3d(
        x=[sc_pos[ca_idx, 0]], y=[sc_pos[ca_idx, 1]], z=[sc_pos[ca_idx, 2]],
        mode="markers",
        marker={"size": 8, "color": "#FF6F00"},
        name=f"Closest approach ({ca_alt_km:.0f} km alt)",
    ))

    fig.add_trace(go.Scatter3d(
        x=[sc_pos[-1, 0]], y=[sc_pos[-1, 1]], z=[sc_pos[-1, 2]],
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

    fig.update_layout(
        **make_dark_layout("Artemis 2 — ECI Frame with MC Solutions (J2000)"),
        scene={
            "xaxis": {"title": "X ECI [km]"},
            "yaxis": {"title": "Y ECI [km]"},
            "zaxis": {"title": "Z ECI [km]"},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )

    return fig


def build_mc_statistics_figure(mc_alt: np.ndarray) -> go.Figure:
    fig = go.Figure()

    # Histogram of MC altitudes
    fig.add_trace(go.Histogram(
        x=mc_alt,
        nbinsx=50,
        marker_color="cyan",
        opacity=0.7,
        name="MC Altitudes",
    ))

    fig.update_layout(
        **make_dark_layout("Monte Carlo Altitude Distribution"),
        xaxis={"title": "Altitude [km]"},
        yaxis={"title": "Count"},
        bargap=0.1,
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Artemis 2 MC 3D dashboard")
    parser.add_argument("--no-open", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    try:
        df_traj, df_mc = load_trajectory_data(TRAJ_CSV_PATH, MC_CSV_PATH)
    except FileNotFoundError as e:
        print(str(e))
        print("Run: cargo run -p artemis --release")
        print("Then: cargo run -p artemis --bin MC --release")
        return

    km = 1e-3

    sc_pos = df_traj[["x_m", "y_m", "z_m"]].values * km  # km, ECI
    moon_pos = df_traj[["moon_x_m", "moon_y_m", "moon_z_m"]].values * km
    burn = df_traj["is_burn"].values.astype(bool)
    t_days = df_traj["time_s"].values / 86_400.0

    # MC final positions
    mc_pos = df_mc[["x_m", "y_m", "z_m"]].values * km  # km, ECI
    mc_alt = df_mc["lunar_alt_km"].values

    sc_moon_dist = np.linalg.norm(sc_pos - moon_pos, axis=1)
    ca_idx = np.argmin(sc_moon_dist)
    ca_alt_km = sc_moon_dist[ca_idx] - MOON_RADIUS_KM

    print(f"Closest approach: {ca_alt_km:.0f} km altitude")
    print(f"MC solutions: {len(mc_pos)} (altitude range: {mc_alt.min():.0f}–{mc_alt.max():.0f} km)")

    # Create dashboard with 3D view and statistics
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("3D ECI Trajectory with MC Solutions", "MC Altitude Distribution"),
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        horizontal_spacing=0.1,
    )

    # Add 3D trajectory
    eci_3d = build_eci_mc_3d_figure(sc_pos, moon_pos, mc_pos, burn, t_days, ca_idx, mc_alt)
    for trace in eci_3d.data:
        fig.add_trace(trace, row=1, col=1)

    # Add statistics
    stats = build_mc_statistics_figure(mc_alt)
    for trace in stats.data:
        fig.add_trace(trace, row=1, col=2)

    # Update layout
    fig.update_layout(
        **make_dark_layout("Artemis 2 — ECI Frame with Monte Carlo Solutions Dashboard"),
        scene={
            "xaxis": {"title": "X ECI [km]"},
            "yaxis": {"title": "Y ECI [km]"},
            "zaxis": {"title": "Z ECI [km]"},
            "aspectmode": "data",
            "domain": {"x": [0.0, 0.6], "y": [0.0, 1.0]},
        },
        xaxis2={"title": "Altitude [km]", "domain": [0.65, 1.0]},
        yaxis2={"title": "Count", "domain": [0.0, 1.0]},
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )

    # Save and open
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "artemis_mc_3d_dashboard.html")
    fig.write_html(output_path)
    print(f"Saved dashboard to {output_path}")

    if not args.no_open:
        # Use pathlib to create proper file:// URL
        file_path = Path(output_path).resolve()
        file_url = file_path.as_uri()
        webbrowser.open(file_url)


if __name__ == "__main__":
    main()