#!/usr/bin/env python3
"""
Artemis 2 Trajectory Visualization Dashboard

Plot Monte Carlo or grid search trajectories in interactive 3D Plotly.
Trajectories are colored by their perturbation group (ΔV/timing direction).

Modes:
  python plot/plot_top_100_trajectories.py         — top 100 MC (default)
  python plot/plot_top_100_trajectories.py --all    — all 1000 MC
  python plot/plot_top_100_trajectories.py --grid   — 1000 grid search (OAT sensitivity)

After running the appropriate Artemis binary:
  cargo run -p artemis --bin mc --release      (for MC modes)
  cargo run -p artemis --bin grid --release    (for grid mode)
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

TOP100_TRAJ_CSV    = os.path.join(os.path.dirname(__file__), "..", "out", "mc_top_100_trajectories.csv")
ALL_TRAJ_CSV       = os.path.join(os.path.dirname(__file__), "..", "out", "mc_all_trajectories.csv")
GRID_TRAJ_CSV      = os.path.join(os.path.dirname(__file__), "..", "out", "grid_trajectories.csv")
GRID_POS_CSV       = os.path.join(os.path.dirname(__file__), "..", "out", "grid_positions.csv")
TOP_POS_CSV_PATH   = os.path.join(os.path.dirname(__file__), "..", "out", "mc_top_100_positions.csv")
MOON_TRACK_CSV     = os.path.join(os.path.dirname(__file__), "..", "out", "artemis2_trajectory.csv")
ASC_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "Artemis_II_OEM_2026_04_04_to_EI.asc")
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), "..", "out")
REF_PLOT_MAX_POINTS = 300
EARTH_RADIUS_KM = 6_371.0
MOON_RADIUS_KM  = 1_737.4
MOON_MEAN_KM    = 384_400.0

# Group highlight config
GROUP_COLORS = {
    "early_burn": "#FF8C00",   # orange  — early ignition
    "late_burn":  "#BF5FFF",   # purple  — late ignition
    "low_dv":     "#00E5FF",   # cyan    — under-performance
    "high_dv":    "#59FF40",   # rose    — over-performance
}
GROUP_LABELS = {
    "early_burn": "Early ignition",
    "late_burn":  "Late ignition",
    "low_dv":     "Low ΔV",
    "high_dv":    "High ΔV",
}


def load_ccsds_oem_asc(path: str) -> np.ndarray | None:
    """Parse a CCSDS OEM .asc file and return positions in km, or None if missing."""
    if not os.path.exists(path):
        return None
    positions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("COMMENT"):
                continue
            if stripped in ("META_START", "META_STOP") or "=" in stripped:
                continue
            tokens = stripped.split()
            if len(tokens) < 7:
                continue
            positions.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
    return np.array(positions, dtype=float) if positions else None


def load_trajectory_data(use_all: bool = False, use_grid: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None, pd.DataFrame | None]:
    if use_grid:
        traj_path = GRID_TRAJ_CSV
        pos_path = GRID_POS_CSV
    else:
        traj_path = ALL_TRAJ_CSV if use_all else TOP100_TRAJ_CSV
        pos_path = TOP_POS_CSV_PATH

    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectories CSV not found: {traj_path}")
    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"Positions CSV not found: {pos_path}")

    df_traj = pd.read_csv(traj_path)
    df_pos  = pd.read_csv(pos_path)
    mode_str = 'grid' if use_grid else ('all' if use_all else 'top-100')
    print(f"Loaded {mode_str} trajectories: "
          f"{df_traj['solution_idx'].nunique()} solutions, {len(df_traj)} rows")

    # Load reference trajectory (moon track + reference positions)
    moon_pos = None
    df_ref = None
    if os.path.exists(MOON_TRACK_CSV):
        df_ref = pd.read_csv(MOON_TRACK_CSV)
        moon_pos = df_ref[["moon_x_m", "moon_y_m", "moon_z_m"]].values * 1e-3  # km
        print(f"Loaded reference trajectory with {len(df_ref)} steps")

    return df_traj, df_pos, moon_pos, df_ref


def make_dark_layout(title: str) -> dict[str, Any]:
    return {
        "template": "plotly_dark",
        "title": {"text": title, "x": 0.01, "xanchor": "left"},
        "paper_bgcolor": "rgb(15,15,25)",
        "plot_bgcolor": "rgb(15,15,25)",
        "font": {"color": "white", "size": 12},
        "legend": {"bgcolor": "rgba(20,20,35,0.8)", "bordercolor": "rgba(255,255,255,0.15)"},
    }


def earth_ca_mask(pos_km: np.ndarray, time_s: np.ndarray, max_time_s: float | None = None) -> np.ndarray:
    """Boolean mask that keeps rows up to and including Earth closest approach
    (minimum distance from origin after T+6 days), or up to max_time_s if specified."""
    if max_time_s is not None:
        return np.searchsorted(time_s, max_time_s) > np.arange(len(time_s))
    dist     = np.linalg.norm(pos_km, axis=1)
    start    = int(np.searchsorted(time_s, 6 * 86_400))
    ca_local = int(np.argmin(dist[start:]))
    ca_idx   = start + ca_local
    return np.arange(len(pos_km)) <= ca_idx


def earth_ca_mask_safe(pos_km: np.ndarray, time_s: np.ndarray, max_dist_km: float = 80_000.0, max_time_s: float | None = None) -> np.ndarray:
    """Like earth_ca_mask but returns the full trajectory if the spacecraft never
    comes within *max_dist_km* of Earth after T+6 days (i.e. it missed entirely).
    If max_time_s is specified, clip at that time instead of at Earth CA."""
    if max_time_s is not None:
        return np.searchsorted(time_s, max_time_s) > np.arange(len(time_s))
    dist  = np.linalg.norm(pos_km, axis=1)
    start = int(np.searchsorted(time_s, 6 * 86_400))
    if start >= len(pos_km):
        return np.ones(len(pos_km), dtype=bool)
    min_dist = dist[start:].min()
    if min_dist > max_dist_km:
        return np.ones(len(pos_km), dtype=bool)  # missed — show full trajectory
    ca_local = int(np.argmin(dist[start:]))
    ca_idx   = start + ca_local
    return np.arange(len(pos_km)) <= ca_idx


def _hex_alpha(hex_color: str, alpha: float) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_trajectories_3d_figure(
    df_traj: pd.DataFrame,
    df_pos: pd.DataFrame,
    moon_pos: np.ndarray | None,
    df_ref: pd.DataFrame | None = None,
    oem_pos_km: np.ndarray | None = None,
    use_all: bool = False,
    use_grid: bool = False,
    max_plot_time_s: float | None = None,
) -> go.Figure:
    fig = go.Figure()

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
        opacity=0.8,
        name="Earth",
    ))

    # Moon orbit track (from reference trajectory)
    if moon_pos is not None:
        fig.add_trace(go.Scatter3d(
            x=moon_pos[::20, 0], y=moon_pos[::20, 1], z=moon_pos[::20, 2],
            mode="markers",
            marker={"size": 4, "color": "gray", "opacity": 0.8},
            name="Moon orbit track",
        ))

    # ── Trajectories — colored by perturbation group ────────────────────────────
    # For --all and --grid modes, use safe mask (some trajectories may miss Earth).
    # Alpha is lower when rendering many trajectories to avoid visual noise.
    clip_fn   = earth_ca_mask_safe if (use_all or use_grid) else earth_ca_mask
    coast_alpha = 0.12 if (use_all or use_grid) else 0.30
    burn_alpha  = 0.20 if (use_all or use_grid) else 0.50

    solution_groups = df_traj.groupby("solution_idx")
    scores          = df_pos["score"].values
    shown_groups: set[str] = set()
    label_points: dict[str, np.ndarray] = {}

    # Opaque legend entries — one dummy marker per group added before the lines
    for group_name, color in GROUP_COLORS.items():
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="lines",
            line={"color": color, "width": 10},
            name=GROUP_LABELS[group_name],
            legendgroup=group_name,
            showlegend=True,
        ))

    for _sol_idx, sol_df in solution_groups:
        group_name = sol_df["group"].iloc[0] if "group" in sol_df.columns else "high_dv"
        color      = GROUP_COLORS.get(group_name, "#8888FF")

        pos_km = sol_df[["x_m", "y_m", "z_m"]].values * 1e-3
        time_s = sol_df["time_s"].values
        burn   = sol_df["is_burn"].values.astype(bool)
        mask   = clip_fn(pos_km, time_s, max_time_s=max_plot_time_s)
        pos_km = pos_km[mask]
        burn   = burn[mask]

        first = group_name not in shown_groups
        if (~burn).any():
            fig.add_trace(go.Scatter3d(
                x=pos_km[~burn, 0], y=pos_km[~burn, 1], z=pos_km[~burn, 2],
                mode="lines",
                line={"color": _hex_alpha(color, coast_alpha), "width": 1},
                showlegend=False,
                legendgroup=group_name,
            ))
            if first:
                label_points[group_name] = pos_km[-1]
        if burn.any():
            fig.add_trace(go.Scatter3d(
                x=pos_km[burn, 0], y=pos_km[burn, 1], z=pos_km[burn, 2],
                mode="lines",
                line={"color": _hex_alpha(color, burn_alpha), "width": 2},
                showlegend=False,
                legendgroup=group_name,
            ))
        shown_groups.add(group_name)

    # ── Best MC solution (top-100 mode only) ─────────────────────────────────
    if not use_all and not use_grid:
        best_sol_df = solution_groups.get_group(int(np.argmin(scores)))
        best_pos    = best_sol_df[["x_m", "y_m", "z_m"]].values * 1e-3
        best_time   = best_sol_df["time_s"].values
        best_burn   = best_sol_df["is_burn"].values.astype(bool)
        best_mask   = earth_ca_mask(best_pos, best_time, max_time_s=max_plot_time_s)
        best_pos    = best_pos[best_mask]
        best_burn   = best_burn[best_mask]
        if (~best_burn).any():
            fig.add_trace(go.Scatter3d(
                x=best_pos[~best_burn, 0], y=best_pos[~best_burn, 1], z=best_pos[~best_burn, 2],
                mode="lines",
                line={"color": "white", "width": 3},
                name="Best MC solution",
            ))

    # ── Reference trajectory (nominal simulation) ─────────────────────────────
    if df_ref is not None:
        ref_km   = df_ref[["x_m", "y_m", "z_m"]].values * 1e-3
        ref_time = df_ref["time_s"].values
        ref_burn = df_ref["is_burn"].values.astype(bool)
        ref_mask = earth_ca_mask(ref_km, ref_time, max_time_s=max_plot_time_s)
        ref_km   = ref_km[ref_mask]
        ref_burn = ref_burn[ref_mask]

        if (~ref_burn).any():
            fig.add_trace(go.Scatter3d(
                x=ref_km[~ref_burn, 0], y=ref_km[~ref_burn, 1], z=ref_km[~ref_burn, 2],
                mode="lines",
                line={"color": "#FF0000", "width": 10},
                name="Reference (nominal)",
            ))
        # if ref_burn.any():
        #     fig.add_trace(go.Scatter3d(
        #         x=ref_km[ref_burn, 0], y=ref_km[ref_burn, 1], z=ref_km[ref_burn, 2],
        #         mode="lines",
        #         line={"color": "#FFA500", "width": 7},
        #     ))

    # ── OEM reference trajectory ──────────────────────────────────────────────
    if oem_pos_km is not None:
        # OEM already ends at EI — just downsample for rendering
        oem_idx  = np.unique(np.linspace(0, len(oem_pos_km) - 1, REF_PLOT_MAX_POINTS, dtype=int))
        oem_plot = oem_pos_km[oem_idx]
        fig.add_trace(go.Scatter3d(
            x=oem_plot[:, 0], y=oem_plot[:, 1], z=oem_plot[:, 2],
            mode="lines",
            line={"color": "white", "width": 4, "dash": "dash"},
            name="OEM reference",
        ))

    n_traj = df_traj["solution_idx"].nunique()
    if use_grid:
        title_str = f"Artemis sensitivity plot (N={n_traj}) — ECI Frame (J2000)"
    elif use_all:
        title_str = f"Monte Carlo Trajectories (N={n_traj}, all) — ECI Frame (J2000)"
    else:
        title_str = f"Monte Carlo Trajectories (N={n_traj}, top-100) — ECI Frame (J2000)"

    layout_dict = make_dark_layout(title_str)
    layout_dict.update({
        "width": 1280,
        "height": 720,
        "margin": {"l": 0, "r": 0, "t": 80, "b": 0},
        "legend": {
            "orientation": "h",
            "y": 1.06,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "bottom",
        },
        "scene": {
            "xaxis": {"title": "X ECI [km]"},
            "yaxis": {"title": "Y ECI [km]"},
            "zaxis": {"title": "Z ECI [km]"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.8, "y": 1.2, "z": 0.9},
            "camera": {"eye": {"x": 2.0, "y": 1.6, "z": 1.0}},
        },
    })
    fig.update_layout(**layout_dict)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MC or grid search trajectories in 3D")
    parser.add_argument("--no-open", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--all", dest="use_all", action="store_true",
                        help="Plot all MC trajectories instead of top 100")
    parser.add_argument("--grid", dest="use_grid", action="store_true",
                        help="Plot grid search (one-at-a-time sensitivity) trajectories")
    parser.add_argument("--max-plot-days", type=float, default=None,
                        help="Clip trajectories at N days for plotting (e.g. 8.0 for 8 days)")
    parser.add_argument("--hq", action="store_true",
                        help="Export high-quality PNG (1920x1440, requires kaleido)")
    args = parser.parse_args()

    max_plot_time_s = args.max_plot_days * 86_400.0 if args.max_plot_days else None

    try:
        df_traj, df_pos, moon_pos, df_ref = load_trajectory_data(use_all=args.use_all, use_grid=args.use_grid)
    except FileNotFoundError as e:
        print(str(e))
        print("Run: cargo run -p artemis --bin mc --release")
        print("  or: cargo run -p artemis --bin grid --release")
        return

    oem_pos_km = load_ccsds_oem_asc(ASC_REFERENCE_PATH)
    if oem_pos_km is not None:
        print(f"Loaded OEM reference with {len(oem_pos_km)} points")
    else:
        print("OEM reference not found, skipping.")

    lunar_alts = df_pos["lunar_alt_km"].values
    earth_alts = df_pos["earth_alt_km"].values
    scores     = df_pos["score"].values
    best_idx   = np.argmin(scores)

    print(f"Trajectory range: Lunar {lunar_alts.min():.0f}–{lunar_alts.max():.0f} km")
    print(f"Best solution: Lunar {lunar_alts[best_idx]:.0f} km, Earth {earth_alts[best_idx]:.0f} km")

    fig = build_trajectories_3d_figure(df_traj, df_pos, moon_pos, df_ref, oem_pos_km,
                                       use_all=args.use_all, use_grid=args.use_grid,
                                       max_plot_time_s=max_plot_time_s)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "artemis_top_100_trajectories_dashboard.html")
    fig.write_html(output_path)
    print(f"Saved → {output_path}")

    if args.hq:
        try:
            png_path = output_path.replace(".html", ".png")
            fig.write_image(png_path, width=1080, height=1920, scale=2)
            print(f"Saved HQ PNG → {png_path}")
        except Exception as e:
            print(f"HQ export failed (install kaleido): {e}")

    if not args.no_open:
        webbrowser.open(Path(output_path).resolve().as_uri())


if __name__ == "__main__":
    main()