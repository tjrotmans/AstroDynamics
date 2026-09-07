"""
Artemis 2 — ECI trajectory dashboard

Reads out/artemis2_trajectory.csv and reference CCSDS OEM ASC data, then
creates Plotly dashboard panels.

Run from AstroProbs/Artemis after cargo run -p artemis --release:
    python plot/plot_eci_dashboard.py
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
ASC_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "Artemis_II_OEM_2026_04_04_to_EI.asc")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")
REF_PLOT_MAX_POINTS = 300
EARTH_RADIUS_KM = 6_371.0
MOON_RADIUS_KM = 1_737.4
MOON_MEAN_KM = 384_400.0

# ── Animation ──────────────────────────────────────────────────────────────────
N_FRAMES = 60          # number of animation frames
FRAME_DURATION_MS = 80 # ms per frame during playback


def load_ccsds_oem_asc(path: str) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if not os.path.exists(path):
        return None, None, None

    ref_frame = None
    positions = []
    velocities = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("COMMENT"):
                continue
            if stripped in ("META_START", "META_STOP"):
                continue
            if "=" in stripped:
                key, value = [part.strip() for part in stripped.split("=", 1)]
                if key == "REF_FRAME":
                    ref_frame = value
                continue

            tokens = stripped.split()
            if len(tokens) < 7:
                continue

            positions.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            velocities.append([float(tokens[4]), float(tokens[5]), float(tokens[6])])

    return np.array(positions, dtype=float), np.array(velocities, dtype=float), ref_frame


def load_eci_trajectory(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def downsample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, max_points, dtype=int))


def make_dark_layout(title: str) -> dict[str, Any]:
    return {
        "template": "plotly_dark",
        "title": {"text": title, "x": 0.01, "xanchor": "left"},
        "paper_bgcolor": "rgb(15,15,25)",
        "plot_bgcolor": "rgb(15,15,25)",
        "font": {"color": "white", "size": 12},
        "legend": {"bgcolor": "rgba(20,20,35,0.8)", "bordercolor": "rgba(255,255,255,0.15)"},
    }


# ── Sphere geometry helpers ────────────────────────────────────────────────────

def _sphere_xyz(
    radius: float,
    cx: float = 0.0,
    cy: float = 0.0,
    cz: float = 0.0,
    n_lat: int = 32,
    n_lon: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return meshgrid (x, y, z) arrays for a sphere of given radius centred at (cx,cy,cz)."""
    phi = np.linspace(0.0, np.pi, n_lat)
    theta = np.linspace(0.0, 2.0 * np.pi, n_lon)
    phi, theta = np.meshgrid(phi, theta)
    x = cx + radius * np.sin(phi) * np.cos(theta)
    y = cy + radius * np.sin(phi) * np.sin(theta)
    z = cz + radius * np.cos(phi)
    return x, y, z


def _make_earth_traces() -> list[go.Surface]:
    """Return Surface traces for Earth body + atmosphere glow ring."""
    # ── Earth body: blue-ocean / green-land colorscale with lighting ──────────
    x, y, z = _sphere_xyz(EARTH_RADIUS_KM)
    # Use latitude to drive a "land vs ocean" pattern (purely aesthetic)
    lat = np.degrees(np.arcsin(z / EARTH_RADIUS_KM))        # -90 … +90
    lon = np.degrees(np.arctan2(y, x))                       # -180 … +180
    # Combine lat/lon into a surface-colour value in [0,1]
    surface_color = (
        0.45
        + 0.35 * np.sin(np.radians(lat) * 3.0)
        + 0.20 * np.cos(np.radians(lon) * 5.0) * np.cos(np.radians(lat))
    )
    surface_color = np.clip(surface_color, 0.0, 1.0)

    earth = go.Surface(
        x=x, y=y, z=z,
        surfacecolor=surface_color,
        colorscale=[
            [0.00, "rgb(10, 40, 100)"],    # deep ocean
            [0.30, "rgb(30, 90, 180)"],    # shallow ocean
            [0.52, "rgb(45, 110, 50)"],    # low land
            [0.70, "rgb(80, 140, 60)"],    # highlands
            [0.88, "rgb(160, 130, 100)"],  # mountains
            [1.00, "rgb(220, 220, 230)"],  # ice / peaks
        ],
        showscale=False,
        opacity=1.0,
        name="Earth",
        hoverinfo="skip",
        lighting={
            "ambient": 0.35,
            "diffuse": 0.85,
            "specular": 0.25,
            "roughness": 0.7,
            "fresnel": 0.2,
        },
        lightposition={"x": 100000, "y": 50000, "z": 25000},
    )

    # ── Atmosphere glow: slightly larger transparent pale-blue sphere ──────────
    xa, ya, za = _sphere_xyz(EARTH_RADIUS_KM * 1.035)
    atmo = go.Surface(
        x=xa, y=ya, z=za,
        surfacecolor=np.ones_like(xa),
        colorscale=[[0, "rgba(100,180,255,0.0)"], [1, "rgba(130,200,255,0.12)"]],
        showscale=False,
        opacity=0.18,
        name="Atmosphere",
        hoverinfo="skip",
    )

    return [earth, atmo]


def _make_moon_surface_trace(
    moon_cx: float,
    moon_cy: float,
    moon_cz: float,
    name: str = "Moon",
    opacity: float = 1.0,
) -> go.Surface:
    """Gray sphere for the Moon at the given centre coordinates."""
    x, y, z = _sphere_xyz(MOON_RADIUS_KM, moon_cx, moon_cy, moon_cz, n_lat=24, n_lon=48)
    surface_color = (
        0.45
        + 0.30 * np.cos(np.linspace(0, np.pi * 4, x.shape[0])[:, None] * np.ones(x.shape[1]))
        + 0.15 * np.sin(np.linspace(0, np.pi * 3, x.shape[1])[None, :] * np.ones(x.shape[0])[:, None])
    )
    surface_color = np.clip(surface_color, 0.0, 1.0)
    return go.Surface(
        x=x, y=y, z=z,
        surfacecolor=surface_color,
        colorscale=[
            [0.0, "rgb(60, 60, 65)"],
            [0.4, "rgb(110, 108, 112)"],
            [0.7, "rgb(155, 150, 155)"],
            [1.0, "rgb(200, 195, 200)"],
        ],
        showscale=False,
        opacity=opacity,
        name=name,
        hoverinfo="skip",
        lighting={
            "ambient": 0.40,
            "diffuse": 0.80,
            "specular": 0.15,
            "roughness": 0.9,
        },
        lightposition={"x": 100000, "y": 50000, "z": 25000},
    )




def _trajectory_glow_traces(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    color_vals: np.ndarray,
    colorscale: str = "Plasma",
    name: str = "Trajectory",
) -> list[go.Scatter3d]:
    """Two-pass trajectory rendering: thick transparent glow + thin opaque line."""
    # Glow pass — thick, semi-transparent
    glow = go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line={"color": color_vals, "colorscale": colorscale, "width": 12},
        opacity=0.20,
        name=name + " glow",
        showlegend=False,
        hoverinfo="skip",
    )
    # Opaque pass — thinner, full colour
    trail = go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line={"color": color_vals, "colorscale": colorscale, "width": 3,
              "colorbar": {"title": "Day", "thickness": 10, "len": 0.4, "x": 1.0}},
        opacity=1.0,
        name=name,
        hovertemplate=(
            "Day %{customdata:.2f}<br>"
            "X=%{x:.0f} km<br>Y=%{y:.0f} km<br>Z=%{z:.0f} km"
            "<extra></extra>"
        ),
        customdata=color_vals,
    )
    return [glow, trail]


# ── Main 3-D figure builder ────────────────────────────────────────────────────

def build_eci_3d_figure(
    sc_pos: np.ndarray,      # shape (N,3) in km
    moon_pos: np.ndarray,    # shape (N,3) in km
    burn: np.ndarray,        # shape (N,) bool
    t_days: np.ndarray,      # shape (N,) float
    ref_pos_km: np.ndarray | None,
    ca_idx: int,
) -> go.Figure:
    """
    Build a visually rich ECI 3-D trajectory figure with:
    - Deep-space starfield background
    - Lit Earth sphere with atmosphere glow
    - Lit Moon sphere at closest-approach position (static), animated in frames
    - Two-pass (glow + opaque) Plasma-coloured trajectory trail
    - Burn segments highlighted in bright orange-red
    - Reference OEM in gold dashed line
    - Event markers: TLI Ignition, Lunar Flyby, Earth Return
    - Plotly frame animation (Play/Pause + time slider)
    """

    N = len(sc_pos)

    # ── 1. Normalised time colour values ──────────────────────────────────────
    t_norm = (t_days - t_days[0]) / max(t_days[-1] - t_days[0], 1e-9)

    # ── 2. Identify key event indices ─────────────────────────────────────────
    tli_idx   = 0
    ca_label  = "Lunar Flyby"
    end_idx   = N - 1

    # ── 2b. Fixed axis ranges — computed once so grid never rescales ──────────
    all_x = np.concatenate([sc_pos[:, 0], moon_pos[:, 0]])
    all_y = np.concatenate([sc_pos[:, 1], moon_pos[:, 1]])
    all_z = np.concatenate([sc_pos[:, 2], moon_pos[:, 2]])
    cx = (all_x.min() + all_x.max()) * 0.5
    cy = (all_y.min() + all_y.max()) * 0.5
    cz = (all_z.min() + all_z.max()) * 0.5
    half_span = max(all_x.max() - all_x.min(),
                    all_y.max() - all_y.min(),
                    all_z.max() - all_z.min()) * 0.55
    ax_range_x = [cx - half_span, cx + half_span]
    ax_range_y = [cy - half_span, cy + half_span]
    ax_range_z = [cz - half_span, cz + half_span]

    # ── 3. Build the static (full-mission) figure ──────────────────────────────
    fig = go.Figure()

    # Earth
    for trace in _make_earth_traces():
        fig.add_trace(trace)

    # Reference OEM — gold dashed line (plotted below main trajectory so it sits "behind")
    if ref_pos_km is not None:
        ref_idx  = downsample_indices(len(ref_pos_km), REF_PLOT_MAX_POINTS)
        ref_plot = ref_pos_km[ref_idx]
        fig.add_trace(go.Scatter3d(
            x=ref_plot[:, 0], y=ref_plot[:, 1], z=ref_plot[:, 2],
            mode="lines",
            line={"color": "gold", "width": 2, "dash": "dash"},
            opacity=0.75,
            name="Reference OEM",
        ))
        fig.add_trace(go.Scatter3d(
            x=[ref_pos_km[0, 0]], y=[ref_pos_km[0, 1]], z=[ref_pos_km[0, 2]],
            mode="markers",
            marker={"size": 5, "color": "gold", "symbol": "diamond"},
            name="OEM start",
        ))
        fig.add_trace(go.Scatter3d(
            x=[ref_pos_km[-1, 0]], y=[ref_pos_km[-1, 1]], z=[ref_pos_km[-1, 2]],
            mode="markers",
            marker={"size": 5, "color": "gold", "symbol": "cross"},
            name="OEM end",
        ))

    # Main trajectory — two-pass (glow + opaque)
    for trace in _trajectory_glow_traces(
        sc_pos[:, 0], sc_pos[:, 1], sc_pos[:, 2],
        t_days, colorscale="Plasma", name="Artemis trajectory",
    ):
        fig.add_trace(trace)

    # Burn segments — thick, bright orange-red (plotted after main trail)
    if burn.any():
        # Glow pass
        fig.add_trace(go.Scatter3d(
            x=sc_pos[burn, 0], y=sc_pos[burn, 1], z=sc_pos[burn, 2],
            mode="lines",
            line={"color": "#FF6600", "width": 16},
            opacity=0.28,
            name="TLI burn glow",
            showlegend=False,
            hoverinfo="skip",
        ))
        # Opaque pass
        fig.add_trace(go.Scatter3d(
            x=sc_pos[burn, 0], y=sc_pos[burn, 1], z=sc_pos[burn, 2],
            mode="lines",
            line={"color": "#FF4500", "width": 5},
            opacity=1.0,
            name="TLI burn",
        ))

    # Event markers
    _event_markers = [
        (tli_idx,  "#69F0AE", "diamond",  "TLI Ignition",  10),
        (ca_idx,   "#FF6F00", "circle",   ca_label,        11),
        (end_idx,  "#EF5350", "x",        "Earth Return",  10),
    ]
    for idx, color, symbol, label, sz in _event_markers:
        fig.add_trace(go.Scatter3d(
            x=[sc_pos[idx, 0]], y=[sc_pos[idx, 1]], z=[sc_pos[idx, 2]],
            mode="markers",
            marker={"size": sz, "color": color, "symbol": symbol,
                    "line": {"color": "white", "width": 1}},
            name=label,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Day {t_days[idx]:.2f}<br>"
                "X=%{x:.0f} km<br>Y=%{y:.0f} km<br>Z=%{z:.0f} km"
                "<extra></extra>"
            ),
        ))

    # ── 4. Build animation frames ──────────────────────────────────────────────
    frame_indices = np.unique(
        np.linspace(0, N - 1, N_FRAMES, dtype=int)
    )

    frames: list[go.Frame] = []
    slider_steps: list[dict] = []

    # Trace indices in the static figure that we will update per frame.
    # We locate them by their names so the order doesn't matter.
    trace_names = [tr.name for tr in fig.data]

    def _tidx(name: str) -> int:
        """Return the data-array index of the first trace matching name."""
        for i, n in enumerate(trace_names):
            if n == name:
                return i
        return -1

    traj_glow_idx  = _tidx("Artemis trajectory glow")
    traj_line_idx  = _tidx("Artemis trajectory")

    # Pre-compute moon sphere geometry for every frame (avoid repeated trig in loop)
    MOON_N_LAT, MOON_N_LON = 12, 24
    _moon_surf_color = np.full((MOON_N_LON, MOON_N_LAT), 0.5)
    frame_moon_xyz = [
        _sphere_xyz(MOON_RADIUS_KM,
                    moon_pos[int(ni)][0], moon_pos[int(ni)][1], moon_pos[int(ni)][2],
                    n_lat=MOON_N_LAT, n_lon=MOON_N_LON)
        for ni in frame_indices
    ]

    for fi, ni in enumerate(frame_indices):
        ni = int(ni)
        trail_x = sc_pos[: ni + 1, 0]
        trail_y = sc_pos[: ni + 1, 1]
        trail_z = sc_pos[: ni + 1, 2]
        trail_t = t_days[: ni + 1]

        # Moon geometry pre-computed above
        mx, my, mz = frame_moon_xyz[fi]

        frame_data: list[Any] = []

        # Trajectory glow pass
        frame_data.append(go.Scatter3d(
            x=trail_x, y=trail_y, z=trail_z,
            mode="lines",
            line={"color": trail_t, "colorscale": "Plasma", "width": 12},
            opacity=0.20,
        ))
        # Trajectory opaque pass
        frame_data.append(go.Scatter3d(
            x=trail_x, y=trail_y, z=trail_z,
            mode="lines",
            line={"color": trail_t, "colorscale": "Plasma", "width": 3},
            opacity=1.0,
        ))

        # Spacecraft marker (cone-like: large marker at head of trail)
        frame_data.append(go.Scatter3d(
            x=[sc_pos[ni, 0]], y=[sc_pos[ni, 1]], z=[sc_pos[ni, 2]],
            mode="markers",
            marker={"size": 9, "color": "#FFFFFF",
                    "symbol": "circle",
                    "line": {"color": "#00CFFF", "width": 2}},
        ))

        # Moon sphere (flat color — minimal data per frame)
        frame_data.append(go.Surface(
            x=mx, y=my, z=mz,
            surfacecolor=_moon_surf_color,
            colorscale=[[0.0, "rgb(110,108,112)"], [1.0, "rgb(165,160,165)"]],
            showscale=False,
            opacity=1.0,
        ))

        # Trace indices for the frame (must match data array order in fig.data)
        traces_to_update = [traj_glow_idx, traj_line_idx]

        # Append spacecraft marker and Moon as new traces at fixed positions
        # Strategy: we add a "SC marker" and "Moon anim" trace once to fig.data
        # and update them per-frame.  We do this after the first frame loop.
        # See below for the two-step approach.

        frames.append(go.Frame(
            data=frame_data,
            name=str(fi),
            traces=[traj_glow_idx, traj_line_idx, len(fig.data), len(fig.data) + 1],
        ))

        slider_steps.append({
            "args": [[str(fi)], {"frame": {"duration": FRAME_DURATION_MS, "redraw": True},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
            "label": f"Day {t_days[ni]:.1f}",
            "method": "animate",
        })

    # Add the two placeholder traces that frames will update
    # (spacecraft marker at t=0, Moon at t=0)
    mx0, my0, mz0 = frame_moon_xyz[0]

    fig.add_trace(go.Scatter3d(
        x=[sc_pos[0, 0]], y=[sc_pos[0, 1]], z=[sc_pos[0, 2]],
        mode="markers",
        marker={"size": 9, "color": "#FFFFFF",
                "symbol": "circle",
                "line": {"color": "#00CFFF", "width": 2}},
        name="Spacecraft",
        showlegend=True,
    ))
    fig.add_trace(go.Surface(
        x=mx0, y=my0, z=mz0,
        surfacecolor=_moon_surf_color,
        colorscale=[[0.0, "rgb(110,108,112)"], [1.0, "rgb(165,160,165)"]],
        showscale=False,
        opacity=1.0,
        name="Moon (animated)",
        hoverinfo="skip",
    ))

    fig.frames = frames

    # ── 5. Layout ──────────────────────────────────────────────────────────────
    updatemenus = [
        {
            "type": "buttons",
            "showactive": False,
            "x": 0.01,
            "xanchor": "left",
            "y": -0.04,
            "yanchor": "top",
            "bgcolor": "rgba(20,20,40,0.85)",
            "font": {"color": "white", "size": 13},
            "buttons": [
                {
                    "label": "▶  Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": FRAME_DURATION_MS, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                            "mode": "immediate",
                        },
                    ],
                },
                {
                    "label": "⏸  Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ]

    sliders = [
        {
            "active": 0,
            "currentvalue": {
                "prefix": "Time: ",
                "visible": True,
                "xanchor": "left",
                "font": {"size": 12, "color": "white"},
            },
            "pad": {"b": 10, "t": 10},
            "len": 0.88,
            "x": 0.10,
            "xanchor": "left",
            "y": -0.04,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.08)",
            "bordercolor": "rgba(255,255,255,0.2)",
            "tickcolor": "rgba(255,255,255,0.4)",
            "font": {"color": "white", "size": 10},
            "steps": slider_steps,
            "transition": {"duration": 0},
        }
    ]

    _ax_common = {
        "backgroundcolor": "rgba(0,0,0,0)",
        "gridcolor": "rgba(255,255,255,0.06)",
        "showbackground": True,
    }
    fig.update_layout(
        **make_dark_layout("Artemis 2 — ECI Trajectory"),
        scene={
            "xaxis": {"title": "X ECI [km]", "range": ax_range_x, **_ax_common},
            "yaxis": {"title": "Y ECI [km]", "range": ax_range_y, **_ax_common},
            "zaxis": {"title": "Z ECI [km]", "range": ax_range_z, **_ax_common},
            "aspectmode": "cube",
            "bgcolor": "rgb(5,5,15)",
            "uirevision": "constant",
            "camera": {
                "eye": {"x": 0.55, "y": -1.10, "z": 0.60},
                "up":  {"x": 0.0,  "y": 0.0,   "z": 1.0},
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        },
        margin={"l": 0, "r": 0, "t": 60, "b": 90},
        updatemenus=updatemenus,
        sliders=sliders,
    )
    # Override legend separately to avoid conflict with make_dark_layout's legend key
    fig.update_layout(legend={
        "bgcolor": "rgba(10,10,25,0.80)",
        "bordercolor": "rgba(255,255,255,0.15)",
        "borderwidth": 1,
        "font": {"size": 11},
    })

    return fig


def build_physics_figure(
    t_days: np.ndarray,
    alt_km: np.ndarray,
    speed_kms: np.ndarray,
    v_radial: np.ndarray,
    v_transverse: np.ndarray,
    sc_moon_dist: np.ndarray,
    earth_moon_dist: np.ndarray,
    mass_kg: np.ndarray,
    burn: np.ndarray,
    ca_idx: int,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Altitude above Earth",
            "Velocity components",
            "SC–Moon & Earth–Moon distance",
            "Spacecraft mass",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    fig.add_trace(go.Scatter(x=t_days, y=alt_km, mode="lines", line={"color": "#42A5F5", "width": 2}, name="Altitude"), row=1, col=1)
    fig.add_hline(y=MOON_MEAN_KM, line_dash="dash", line_color="gray", annotation_text="Moon mean dist", row=1, col=1)
    fig.add_vline(x=t_days[ca_idx], line_dash="dot", line_color="#FF6F00", row=1, col=1)
    if burn.any():
        fig.add_vrect(x0=t_days[burn][0], x1=t_days[burn][-1], fillcolor="rgba(239,83,80,0.1)", line_width=0, row=1, col=1)

    fig.add_trace(go.Scatter(x=t_days, y=speed_kms, mode="lines", line={"color": "#283593", "width": 2}, name="|v| total"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t_days, y=v_transverse, mode="lines", line={"color": "#2E7D32", "dash": "dash", "width": 2}, name="Transverse v⊥"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t_days, y=v_radial, mode="lines", line={"color": "#FB8C00", "dash": "dot", "width": 2}, name="Radial vᵣ"), row=1, col=2)
    fig.add_vline(x=t_days[ca_idx], line_dash="dot", line_color="#FF6F00", row=1, col=2)
    if burn.any():
        fig.add_vrect(x0=t_days[burn][0], x1=t_days[burn][-1], fillcolor="rgba(239,83,80,0.1)", line_width=0, row=1, col=2)

    fig.add_trace(go.Scatter(x=t_days, y=sc_moon_dist, mode="lines", line={"color": "#8E24AA", "width": 2}, name="SC–Moon dist"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_days, y=earth_moon_dist, mode="lines", line={"color": "gray", "dash": "dash", "width": 1.5}, name="Earth–Moon dist"), row=2, col=1)
    fig.add_hline(y=MOON_RADIUS_KM, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_trace(go.Scatter(x=[t_days[ca_idx]], y=[sc_moon_dist[ca_idx]], mode="markers", marker={"color": "#FF6F00", "size": 9}, name="Closest approach"), row=2, col=1)
    if burn.any():
        fig.add_vrect(x0=t_days[burn][0], x1=t_days[burn][-1], fillcolor="rgba(239,83,80,0.1)", line_width=0, row=2, col=1)

    fig.add_trace(go.Scatter(x=t_days, y=mass_kg, mode="lines", line={"color": "#00ACC1", "width": 2}, name="Mass"), row=2, col=2)
    if burn.any():
        fig.add_vrect(x0=t_days[burn][0], x1=t_days[burn][-1], fillcolor="rgba(239,83,80,0.1)", line_width=0, row=2, col=2)
        dm = mass_kg[burn][0] - mass_kg[burn][-1]
        fig.add_annotation(
            x=t_days[burn][-1],
            y=mass_kg[burn][-1],
            text=f"ΔM = {dm:.0f} kg",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
            font={"color": "#FF7043"},
            row=2,
            col=2,
        )

    fig.update_xaxes(title_text="Mission elapsed time [days]", row=1, col=1)
    fig.update_xaxes(title_text="Mission elapsed time [days]", row=1, col=2)
    fig.update_xaxes(title_text="Mission elapsed time [days]", row=2, col=1)
    fig.update_xaxes(title_text="Mission elapsed time [days]", row=2, col=2)
    fig.update_yaxes(title_text="Altitude [km]", row=1, col=1)
    fig.update_yaxes(title_text="Speed [km/s]", row=1, col=2)
    fig.update_yaxes(title_text="Distance [km]", row=2, col=1)
    fig.update_yaxes(title_text="Mass [kg]", row=2, col=2)
    fig.update_layout(
        **make_dark_layout("Artemis 2 — Physics Summary"),
        margin={"l": 50, "r": 20, "t": 80, "b": 40},
        height=720,
    )

    return fig


def plot_dashboard(
    df: pd.DataFrame,
    ref_pos_km: np.ndarray | None,
    ref_frame: str | None,
) -> dict[str, go.Figure]:
    sc_pos = df[["x_m", "y_m", "z_m"]].values * 1e-3
    sc_vel = df[["vx_ms", "vy_ms", "vz_ms"]].values * 1e-3
    moon_pos = df[["moon_x_m", "moon_y_m", "moon_z_m"]].values * 1e-3
    burn = df["is_burn"].values.astype(bool)
    t_days = df["time_s"].values / 86_400.0
    alt_km = np.linalg.norm(sc_pos, axis=1) - EARTH_RADIUS_KM
    speed_kms = np.linalg.norm(sc_vel, axis=1)
    r_hat = sc_pos / np.linalg.norm(sc_pos, axis=1, keepdims=True)
    v_radial = np.einsum("ij,ij->i", sc_vel, r_hat)
    v_transverse = np.sqrt(np.maximum(0, speed_kms**2 - v_radial**2))
    sc_moon_dist = np.linalg.norm(sc_pos - moon_pos, axis=1)
    earth_moon_dist = np.linalg.norm(moon_pos, axis=1)
    ca_idx = np.argmin(sc_moon_dist)

    panels = {
        "orbit_3d": build_eci_3d_figure(sc_pos, moon_pos, burn, t_days, ref_pos_km, ca_idx),
        "physics": build_physics_figure(
            t_days,
            alt_km,
            speed_kms,
            v_radial,
            v_transverse,
            sc_moon_dist,
            earth_moon_dist,
            df["mass_kg"].values,
            burn,
            ca_idx,
        ),
    }

    print("Loaded trajectory and built dashboard panels.")
    print(f"Reference frame: {ref_frame}")
    if ref_frame not in (None, "EME2000"):
        print("WARNING: Reference ASC frame is not EME2000. Verify frame alignment.")

    return panels


def save_dashboard(panels: dict[str, go.Figure], output_dir: str = OUTPUT_DIR, prefix: str = "trajectory_eci") -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    saved_files: list[str] = []
    for key, fig in panels.items():
        fname = os.path.join(output_dir, f"{prefix}_{key}.html")
        fig.write_html(fname, include_plotlyjs="cdn")
        saved_files.append(fname)
        print(f"Saved {fname}")
    return saved_files


def open_dashboard_files(file_paths: list[str]) -> None:
    for path in file_paths:
        # Use pathlib to create proper file:// URL
        file_path = Path(path).resolve()
        file_url = file_path.as_uri()
        webbrowser.open(file_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Artemis ECI trajectory dashboard.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory to save HTML outputs")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Do not open the generated panels in the browser")
    parser.set_defaults(show=True)
    args = parser.parse_args()

    df = load_eci_trajectory(CSV_PATH)
    ref_pos_km, _, ref_frame = load_ccsds_oem_asc(ASC_REFERENCE_PATH)
    panels = plot_dashboard(df, ref_pos_km, ref_frame)
    saved_files = save_dashboard(panels, args.output_dir)

    if args.show:
        open_dashboard_files(saved_files)


if __name__ == "__main__":
    main()
