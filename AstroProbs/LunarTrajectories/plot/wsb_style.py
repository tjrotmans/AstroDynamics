"""
wsb_style.py — shared constants, colours, and frame-conversion helpers
for WSB trajectory plots.

Physical constants and frame-rotation functions are imported from the
workspace-level Python "crate" (crates/python/astrodynamics.py), which
mirrors crates/orbital_models/src/constants.rs exactly.  Do not redefine
constants here — change them in astrodynamics.py only.

Imported by all plot_wsb_*.py scripts.
"""
from __future__ import annotations
import pathlib
import sys
import numpy as np

# ── Import from workspace Python "crate" ──────────────────────────────────────
# Go up 4 levels: plot/ → LunarTrajectories/ → AstroProbs/ → AstroDynamics/
_CRATES_PY = pathlib.Path(__file__).resolve().parents[3] / "crates" / "python"
if str(_CRATES_PY) not in sys.path:
    sys.path.insert(0, str(_CRATES_PY))

from astrodynamics import (
    # CRTBP normalisation
    MU_ND, X_M, L_KM, T_STAR, V_STAR, V_STAR_KM,
    R_HILL_ND, R_HILL_KM,
    # Physical radii (km)
    EARTH_RADIUS_KM as R_EARTH,
    MOON_RADIUS_KM  as R_MOON,
    # Frame rotation functions (canonical implementations)
    rot_em_to_eci, moon_em_to_eci, orbital_plane_r3d, eci_to_rot_em,
)

# Expose MU as the name used throughout the WSB plot scripts
MU = MU_ND

# ---- Style -------------------------------------------------------------------
BG        = "#0F0F19"
COL_WSB   = "#00E5FF"
COL_ART   = "#FF8C00"
COL_EARTH = "#1565C0"
COL_MOON  = "#9E9E9E"
COL_PARK  = "#FFD700"
COL_HILL  = "#BA7517"
COL_CAPT  = "#FF6B35"    # capture arc (post-Hill sphere entry)

MULTI_COLORS = [
    "#00E5FF", "#FF8C00", "#69FF47", "#FF6B6B", "#C77DFF",
    "#4FC3F7", "#FFD166", "#06D6A0", "#EF476F", "#118AB2",
]

GRID_COLOR = "rgba(80,80,160,0.18)"
ZERO_COLOR = "rgba(150,150,255,0.25)"
TICK_COLOR = "#aaaaaa"
BG_3D      = "rgb(12,12,22)"
PAPER_BG   = "rgb(10,10,18)"


def load_epoch_info(path: pathlib.Path) -> tuple[float, np.ndarray]:
    """Read epoch_info.txt from wsb_circularize. Returns (dep_offset_days, R0_wsb 3×3)."""
    info: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                key, _, val = line.partition(":")
                info[key.strip()] = val.strip()
    dep_offset = float(info["dep_offset_days"])
    c0 = np.array([float(v) for v in info["R0_wsb_col0"].split()])
    c1 = np.array([float(v) for v in info["R0_wsb_col1"].split()])
    c2 = np.array([float(v) for v in info["R0_wsb_col2"].split()])
    return dep_offset, np.column_stack([c0, c1, c2])


def sphere_surface(r: float, nu: int = 30, nv: int = 16):
    """Sphere surface mesh of radius r [km], for go.Surface."""
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(nu), np.cos(v))
    return x, y, z


def downsample(arr: np.ndarray, max_pts: int) -> np.ndarray:
    """Uniformly subsample 1-D array to at most max_pts entries."""
    if len(arr) <= max_pts:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_pts, dtype=int)
    return arr[idx]


def dark_3d_axis(title: str, rng: list) -> dict:
    """Plotly 3D scene axis dict for dark theme."""
    return dict(
        title=title, range=rng,
        showgrid=True, gridcolor=GRID_COLOR,
        zeroline=False, tickfont=dict(size=8),
    )


def dark_soi_layout(view_km: float) -> dict:
    """Returns xaxis/yaxis dicts for the 2D Moon-centred SOI panel."""
    axis = dict(
        range=[-view_km, view_km],
        showgrid=True, gridcolor=GRID_COLOR,
        zeroline=True, zerolinecolor=ZERO_COLOR,
        tickfont=dict(size=9, color=TICK_COLOR),
    )
    return (
        dict(title="X from Moon [km]", scaleanchor="y", scaleratio=1, **axis),
        dict(title="Y from Moon [km]", **axis),
    )


def play_pause_buttons(frame_ms: int) -> dict:
    """Plotly updatemenus entry for Play/Pause buttons (dark theme)."""
    return {
        "type": "buttons", "showactive": False,
        "x": 0.5, "xanchor": "center", "y": -0.08, "yanchor": "top",
        "buttons": [
            {"label": "Play", "method": "animate",
             "args": [None, {"frame": {"duration": frame_ms, "redraw": True},
                             "fromcurrent": True, "transition": {"duration": 0}}]},
            {"label": "Pause", "method": "animate",
             "args": [[None], {"frame": {"duration": 0, "redraw": False},
                               "mode": "immediate", "transition": {"duration": 0}}]},
        ],
        "font": {"color": "white"},
        "bgcolor": "rgba(40,40,60,0.9)",
        "bordercolor": "rgba(100,100,150,0.5)",
    }


def dark_slider(steps: list) -> dict:
    """Plotly sliders entry for dark theme."""
    return {
        "active": 0, "x": 0.05, "len": 0.90,
        "y": -0.02, "yanchor": "top",
        "currentvalue": {
            "prefix": "Day: ", "visible": True, "xanchor": "center",
            "font": {"color": "white", "size": 11},
        },
        "transition": {"duration": 0},
        "bgcolor": "rgba(40,40,60,0.8)",
        "bordercolor": "rgba(100,100,150,0.4)",
        "tickcolor": "rgba(200,200,255,0.5)",
        "font": {"color": "white", "size": 9},
        "steps": steps,
    }
