#!/usr/bin/env python3
"""
Artemis 2 Trajectory Uncertainty Visualisation

Focuses on the two mission-critical events:
  - Lunar flyby (T+~3.86 days)
  - Earth return / reentry (T+~9 days)

Reads:
  out/artemis2_trajectory.csv      — nominal trajectory (ECI, metres)
  out/mc_position_snapshots.csv    — N runs × 8 epochs, positions in km
  out/mc_all_solutions.csv         — N runs of scalar mission metrics

Produces two HTML dashboards:
  out/mc_trajectory_uncertainty_3d.html  — 3D: nominal arc clipped at Earth CA,
                                            scatter clouds + 3σ ellipsoids at the
                                            two key epochs only
  out/mc_trajectory_uncertainty_2d.html  — 2×2: lunar flyby scatter, Earth return
                                            scatter, σ growth, XY view with focused
                                            ellipses at the two key epochs only

Run from AstroProbs/Artemis/:
  cargo run --bin artemis --release
  cargo run --bin mc      --release
  python plot/plot_trajectory_uncertainty.py
"""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(__file__)
NOM_CSV  = os.path.join(_HERE, "..", "out", "artemis2_trajectory.csv")
SNAP_CSV = os.path.join(_HERE, "..", "out", "mc_position_snapshots.csv")
SOL_CSV  = os.path.join(_HERE, "..", "out", "mc_all_solutions.csv")
OUT_DIR  = os.path.join(_HERE, "..", "out")
OUT_3D   = os.path.join(OUT_DIR, "mc_trajectory_uncertainty_3d.html")
OUT_2D   = os.path.join(OUT_DIR, "mc_trajectory_uncertainty_2d.html")

EARTH_R_KM = 6_371.0
MOON_R_KM  = 1_737.4


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Style ─────────────────────────────────────────────────────────────────────
DARK: dict[str, Any] = {
    "template":      "plotly_dark",
    "paper_bgcolor": "rgb(15,15,25)",
    "plot_bgcolor":  "rgb(15,15,25)",
    "font":          {"color": "white", "size": 11},
    "legend":        {"bgcolor": "rgba(20,20,35,0.8)",
                      "bordercolor": "rgba(255,255,255,0.15)"},
}

COL_MOON  = "#C0C0D0"   # Moon flyby accent
COL_EARTH = "#4FC3F7"   # Earth return accent
COL_NOM   = "#FFD700"   # nominal trajectory


# ── Geometry helpers ──────────────────────────────────────────────────────────

def ellipsoid_surface(
    center: np.ndarray, cov3: np.ndarray,
    n_std: float = 3.0, n_u: int = 24, n_v: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, Y, Z) mesh for a covariance ellipsoid at n_std sigma."""
    vals, vecs = np.linalg.eigh(cov3)
    scale = n_std * np.sqrt(np.abs(vals))
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, n_u), np.linspace(0, np.pi, n_v))
    sphere = np.stack([np.sin(v)*np.cos(u), np.sin(v)*np.sin(u), np.cos(v)])
    pts = (vecs @ (np.diag(scale) @ sphere.reshape(3, -1))).reshape(3, *v.shape)
    return pts[0] + center[0], pts[1] + center[1], pts[2] + center[2]


def ellipse_2d(
    center: np.ndarray, cov2: np.ndarray, n_std: float, n_pts: int = 160,
) -> tuple[np.ndarray, np.ndarray]:
    """(ex, ey) for a 2-D covariance ellipse at n_std sigma."""
    vals, vecs = np.linalg.eigh(cov2)
    t = np.linspace(0, 2*np.pi, n_pts)
    ell = vecs @ (np.diag(n_std * np.sqrt(np.abs(vals))) @ np.stack([np.cos(t), np.sin(t)]))
    return ell[0] + center[0], ell[1] + center[1]


# ── Data loading and pre-processing ──────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in (NOM_CSV, SNAP_CSV, SOL_CSV):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "Run:  cargo run --bin artemis --release\n"
                "      cargo run --bin mc      --release"
            )
    return pd.read_csv(NOM_CSV), pd.read_csv(SNAP_CSV), pd.read_csv(SOL_CSV)


def clip_at_earth_ca(nom: pd.DataFrame) -> tuple[pd.DataFrame, int, float]:
    """Return nominal trajectory clipped at Earth closest approach (after 6 days)."""
    km  = 1e-3
    pos = nom[["x_m", "y_m", "z_m"]].values * km
    t   = nom["time_s"].values / 86_400.0
    mask = t >= 6.0
    idx_offset = np.where(mask)[0][0]
    earth_dist = np.linalg.norm(pos[mask], axis=1)
    ca_local   = int(np.argmin(earth_dist))
    ca_idx     = idx_offset + ca_local
    ca_time    = t[ca_idx]
    return nom.iloc[:ca_idx + 1].copy(), ca_idx, ca_time


def find_lunar_ca(nom: pd.DataFrame) -> tuple[int, float]:
    km      = 1e-3
    sc_pos  = nom[["x_m", "y_m", "z_m"]].values * km
    mn_pos  = nom[["moon_x_m", "moon_y_m", "moon_z_m"]].values * km
    dist    = np.linalg.norm(sc_pos - mn_pos, axis=1)
    ca_idx  = int(np.argmin(dist))
    ca_time = nom["time_s"].values[ca_idx] / 86_400.0
    return ca_idx, ca_time


def epoch_stats(snap: pd.DataFrame, t_day: float) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and 3×3 covariance [km] for the snapshot epoch nearest to t_day."""
    epochs    = snap["time_days"].unique()
    nearest   = epochs[np.argmin(np.abs(epochs - t_day))]
    pts       = snap[np.isclose(snap["time_days"], nearest)][["x_km","y_km","z_km"]].values
    return pts.mean(axis=0), np.cov(pts.T)


# ── Figure 1: 3D ─────────────────────────────────────────────────────────────

def build_3d_figure(
    nom_clip: pd.DataFrame,
    snap: pd.DataFrame,
    lunar_ca_time: float,
    earth_ca_time: float,
) -> go.Figure:
    fig = go.Figure()
    km  = 1e-3
    sc  = nom_clip[["x_m","y_m","z_m"]].values * km
    mn  = nom_clip[["moon_x_m","moon_y_m","moon_z_m"]].values * km
    t   = nom_clip["time_s"].values / 86_400.0
    burn = nom_clip["is_burn"].values.astype(bool)

    # Earth sphere
    phi, th = np.meshgrid(np.linspace(0,np.pi,20), np.linspace(0,2*np.pi,40))
    fig.add_trace(go.Surface(
        x=EARTH_R_KM*np.sin(phi)*np.cos(th),
        y=EARTH_R_KM*np.sin(phi)*np.sin(th),
        z=EARTH_R_KM*np.cos(phi),
        colorscale=[[0,"rgb(21,101,192)"],[1,"rgb(21,101,192)"]],
        showscale=False, opacity=0.5, name="Earth",
    ))

    # Moon track
    fig.add_trace(go.Scatter3d(
        x=mn[::12,0], y=mn[::12,1], z=mn[::12,2],
        mode="markers",
        marker={"size":2, "color":"gray", "opacity":0.25},
        name="Moon track",
    ))

    # Nominal trajectory
    fig.add_trace(go.Scatter3d(
        x=sc[~burn,0], y=sc[~burn,1], z=sc[~burn,2],
        mode="lines", line={"color":COL_NOM, "width":3},
        name="Nominal trajectory",
    ))
    if burn.any():
        fig.add_trace(go.Scatter3d(
            x=sc[burn,0], y=sc[burn,1], z=sc[burn,2],
            mode="lines", line={"color":"#EF5350","width":5},
            name="TLI burn",
        ))

    # ── Two key epochs: lunar CA and Earth return ─────────────────────────────
    events = [
        (lunar_ca_time, COL_MOON,  "Lunar flyby",   "rgba(192,192,208,"),
        (earth_ca_time, COL_EARTH, "Earth return",  "rgba(79,195,247,"),
    ]

    for t_key, color, label, rgba_prefix in events:
        cen, C = epoch_stats(snap, t_key)
        pts    = snap[np.isclose(snap["time_days"],
                                 snap["time_days"].unique()[
                                     np.argmin(np.abs(snap["time_days"].unique() - t_key))
                                 ])][["x_km","y_km","z_km"]].values

        # Scatter cloud (all runs)
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode="markers",
            marker={"size":2, "color":color, "opacity":0.4},
            name=f"MC cloud — {label}",
        ))

        # 1σ and 3σ ellipsoids
        for n_std, opacity in ((1, 0.30), (3, 0.12)):
            X, Y, Z = ellipsoid_surface(cen, C, n_std=n_std)
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale=[[0, rgba_prefix+"0.0)"],[1, rgba_prefix+"1.0)"]],
                showscale=False, opacity=opacity,
                name=f"{n_std}σ — {label}",
            ))

        # Centroid marker
        fig.add_trace(go.Scatter3d(
            x=[cen[0]], y=[cen[1]], z=[cen[2]],
            mode="markers",
            marker={"size":7, "color":color, "symbol":"cross"},
            name=f"Centroid — {label}",
        ))

    # Moon sphere at lunar CA
    luna_ca_idx = int(np.argmin(np.abs(t - lunar_ca_time)))
    fig.add_trace(go.Scatter3d(
        x=[mn[luna_ca_idx,0]], y=[mn[luna_ca_idx,1]], z=[mn[luna_ca_idx,2]],
        mode="markers",
        marker={"size":10, "color":"white", "line":{"color":"#888","width":1}},
        name=f"Moon at CA (T+{lunar_ca_time:.2f} d)",
    ))

    fig.update_layout(
        **DARK,
        title={"text": "Artemis 2 — Position Uncertainty at Lunar Flyby & Earth Return  "
                       f"(trajectory clipped at T+{earth_ca_time:.1f} d)",
               "x":0.01, "xanchor":"left"},
        scene={"xaxis":{"title":"X [km]"}, "yaxis":{"title":"Y [km]"},
               "zaxis":{"title":"Z [km]"}, "aspectmode":"data"},
        height=800, margin={"l":0,"r":0,"t":60,"b":0},
    )
    return fig


# ── Figure 2: 2×2 focused dashboard ──────────────────────────────────────────

def build_2d_figure(
    nom_clip: pd.DataFrame,
    snap: pd.DataFrame,
    sol: pd.DataFrame,
    lunar_ca_time: float,
    earth_ca_time: float,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Lunar flyby uncertainty  (all MC runs)",
            "Earth return uncertainty  (all MC runs)",
            "Position uncertainty growth  (1σ)",
            "XY view — uncertainty at lunar flyby & Earth return",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    km    = 1e-3
    sc    = nom_clip[["x_m","y_m","z_m"]].values * km
    mn    = nom_clip[["moon_x_m","moon_y_m","moon_z_m"]].values * km
    t_nom = nom_clip["time_s"].values / 86_400.0

    # Nominal scalars (from nominal trajectory CSV)
    nom_lunar_t   = lunar_ca_time               # days
    nom_lunar_alt = (np.linalg.norm(
        sc[np.argmin(np.abs(t_nom - lunar_ca_time))] -
        mn[np.argmin(np.abs(t_nom - lunar_ca_time))]
    ) - MOON_R_KM)

    nom_earth_t   = earth_ca_time               # days
    nom_earth_alt = (np.linalg.norm(
        sc[np.argmin(np.abs(t_nom - earth_ca_time))]
    ) - EARTH_R_KM)

    # ── Panel 1: Lunar flyby scatter ──────────────────────────────────────────
    # x = time offset from nominal [hours], y = altitude [km], color = Earth return alt
    lunar_dt_h = (sol["lunar_ca_time_days"] - nom_lunar_t) * 24.0

    fig.add_trace(go.Scatter(
        x=lunar_dt_h,
        y=sol["lunar_alt_km"],
        mode="markers",
        marker={"color": sol["earth_alt_km"], "colorscale": "Plasma",
                "size": 5, "opacity": 0.6,
                "colorbar": {"title": "Earth return<br>alt [km]",
                             "x": 0.46, "len": 0.45, "y": 0.78,
                             "thickness": 12}},
        showlegend=False,
    ), row=1, col=1)

    # Nominal star
    fig.add_trace(go.Scatter(
        x=[0], y=[nom_lunar_alt],
        mode="markers",
        marker={"color": COL_NOM, "size": 14, "symbol": "star"},
        name="Nominal",
    ), row=1, col=1)

    # Target altitude line
    fig.add_hline(y=6513, line={"color":COL_MOON,"dash":"dot","width":1.5},
                  annotation_text="Target 6513 km", annotation_font_color=COL_MOON,
                  row=1, col=1)

    fig.update_xaxes(title_text="Lunar CA time offset [h]", row=1, col=1)
    fig.update_yaxes(title_text="Lunar CA altitude [km]",   row=1, col=1)

    # ── Panel 2: Earth return scatter ─────────────────────────────────────────
    earth_dt_h = (sol["earth_ca_time_days"] - nom_earth_t) * 24.0

    fig.add_trace(go.Scatter(
        x=earth_dt_h,
        y=sol["earth_alt_km"],
        mode="markers",
        marker={"color": sol["lunar_alt_km"], "colorscale": "Viridis",
                "size": 5, "opacity": 0.6,
                "colorbar": {"title": "Lunar CA<br>alt [km]",
                             "x": 1.01, "len": 0.45, "y": 0.78,
                             "thickness": 12}},
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[0], y=[nom_earth_alt],
        mode="markers",
        marker={"color": COL_NOM, "size": 14, "symbol": "star"},
        name="Nominal", showlegend=False,
    ), row=1, col=2)

    fig.add_hline(y=130,  line={"color":"#EF5350","dash":"dot","width":1.5},
                  annotation_text="Reentry limit 130 km",
                  annotation_font_color="#EF5350", row=1, col=2)
    fig.add_hline(y=60,   line={"color":"#FF6F00","dash":"dot","width":1.5},
                  annotation_text="Target 60 km",
                  annotation_font_color="#FF6F00",  row=1, col=2)

    fig.update_xaxes(title_text="Earth return time offset [h]", row=1, col=2)
    fig.update_yaxes(title_text="Earth return altitude [km]",   row=1, col=2)

    # ── Panel 3: σ_3D growth over time ───────────────────────────────────────
    epochs = np.sort(snap["time_days"].unique())
    sigmas = []
    for t in epochs:
        pts = snap[np.isclose(snap["time_days"], t)][["x_km","y_km","z_km"]].values
        sigmas.append(np.sqrt(np.trace(np.cov(pts.T))))

    fig.add_trace(go.Scatter(
        x=epochs, y=sigmas,
        mode="lines+markers",
        line={"color":"#69F0AE","width":2},
        marker={"size":6},
        fill="tozeroy", fillcolor="rgba(105,240,174,0.08)",
        name="σ₃D",
    ), row=2, col=1)

    fig.add_vline(x=lunar_ca_time,
                  line={"color":COL_MOON, "dash":"dot","width":1.5},
                  annotation_text="Lunar CA",
                  annotation_font_color=COL_MOON, row=2, col=1)
    fig.add_vline(x=earth_ca_time,
                  line={"color":COL_EARTH,"dash":"dot","width":1.5},
                  annotation_text="Earth CA",
                  annotation_font_color=COL_EARTH, row=2, col=1)

    fig.update_xaxes(title_text="Mission time [days]",           row=2, col=1)
    fig.update_yaxes(title_text="1σ position uncertainty [km]",  row=2, col=1)

    # ── Panel 4: XY view with ellipses at the two key epochs only ────────────
    fig.add_trace(go.Scatter(
        x=sc[:,0], y=sc[:,1],
        mode="lines", line={"color":COL_NOM,"width":2},
        name="Nominal trajectory", showlegend=False,
    ), row=2, col=2)

    # Start and end markers
    fig.add_trace(go.Scatter(
        x=[sc[0,0]], y=[sc[0,1]], mode="markers",
        marker={"color":"#69F0AE","size":9,"symbol":"circle"},
        name="TLI", showlegend=False,
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=[sc[-1,0]], y=[sc[-1,1]], mode="markers",
        marker={"color":COL_EARTH,"size":9,"symbol":"x"},
        name="Earth CA", showlegend=False,
    ), row=2, col=2)

    for t_key, color, label in (
        (lunar_ca_time, COL_MOON,  "Lunar CA"),
        (earth_ca_time, COL_EARTH, "Earth return"),
    ):
        cen, C = epoch_stats(snap, t_key)
        cen2   = cen[:2]
        C2     = C[:2, :2]

        # Scatter cloud
        pts = snap[np.isclose(snap["time_days"],
                               snap["time_days"].unique()[
                                   np.argmin(np.abs(snap["time_days"].unique() - t_key))
                               ])][["x_km","y_km"]].values
        fig.add_trace(go.Scatter(
            x=pts[:,0], y=pts[:,1],
            mode="markers",
            marker={"size":3, "color":color, "opacity":0.35},
            name=f"MC — {label}", showlegend=False,
        ), row=2, col=2)

        # 1σ / 2σ / 3σ ellipses
        for n_std, width, opacity in ((1, 2.0, 0.95), (2, 1.2, 0.55), (3, 0.7, 0.30)):
            ex, ey = ellipse_2d(cen2, C2, n_std)
            fig.add_trace(go.Scatter(
                x=ex, y=ey, mode="lines",
                line={"color": hex_to_rgba(color, opacity), "width": width},
                name=f"{n_std}σ {label}" if n_std == 1 else None,
                showlegend=(n_std == 1),
            ), row=2, col=2)

        # Centroid
        fig.add_trace(go.Scatter(
            x=[cen2[0]], y=[cen2[1]], mode="markers",
            marker={"color":color,"size":8,"symbol":"cross"},
            showlegend=False,
        ), row=2, col=2)

    fig.update_xaxes(title_text="X ECI [km]", row=2, col=2)
    fig.update_yaxes(title_text="Y ECI [km]", row=2, col=2,
                     scaleanchor="x4", scaleratio=1)

    n = len(sol)
    fig.update_layout(
        **DARK,
        title={"text": f"Artemis 2 — Lunar Flyby & Earth Return Uncertainty  (N={n} MC runs)",
               "x":0.01, "xanchor":"left"},
        height=950,
        margin={"l":60,"r":80,"t":80,"b":40},
    )
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    try:
        nom, snap, sol = load_data()
    except FileNotFoundError as e:
        print(e)
        return

    nom_clip, _, earth_ca_time = clip_at_earth_ca(nom)
    _, lunar_ca_time           = find_lunar_ca(nom_clip)

    print(f"Nominal lunar CA:   T+{lunar_ca_time:.3f} days")
    print(f"Nominal Earth CA:   T+{earth_ca_time:.3f} days  (trajectory clipped here)")
    print(f"MC runs:            {len(sol)}")
    print(f"Snapshot epochs:    {snap['time_days'].nunique()}")

    # σ at the two key epochs
    for t_key, label in ((lunar_ca_time, "Lunar CA"), (earth_ca_time, "Earth return")):
        cen, C = epoch_stats(snap, t_key)
        print(f"  {label:<16} σ₃D = {np.sqrt(np.trace(C)):.1f} km")

    os.makedirs(OUT_DIR, exist_ok=True)

    fig3d = build_3d_figure(nom_clip, snap, lunar_ca_time, earth_ca_time)
    fig3d.write_html(OUT_3D)
    print(f"Saved → {OUT_3D}")

    fig2d = build_2d_figure(nom_clip, snap, sol, lunar_ca_time, earth_ca_time)
    fig2d.write_html(OUT_2D)
    print(f"Saved → {OUT_2D}")

    for path in (OUT_3D, OUT_2D):
        webbrowser.open(Path(path).resolve().as_uri())


if __name__ == "__main__":
    main()
