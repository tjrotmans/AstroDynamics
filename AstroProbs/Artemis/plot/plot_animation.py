#!/usr/bin/env python3
"""
Artemis 2 — ECI vs Rotating Frame Side-by-Side Animation

Left:  ECI (J2000) inertial frame — Earth fixed at origin, Moon moves
Right: Earth-Moon rotating frame  — Earth at origin, Moon fixed on +x axis

The rotating frame co-rotates with the Moon: at each time t, the frame is
rotated by -θ(t) where θ(t) = atan2(moon_y_eci, moon_x_eci).

Reads:  out/artemis2_trajectory.csv
Output: out/artemis_animation.html

Run from AstroProbs/Artemis/:
  python plot/plot_animation.py
"""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_HERE    = os.path.dirname(__file__)
NOM_CSV  = os.path.join(_HERE, "..", "out", "artemis2_trajectory.csv")
OUT_HTML = os.path.join(_HERE, "..", "out", "artemis_animation.html")

EARTH_R_KM = 6_371.0
MOON_R_KM  = 1_737.4
N_FRAMES   = 250       # animation frames (more = smoother but heavier)
FRAME_MS   = 25        # ms per frame → ~6 s total at 250 frames


def clip_at_earth_ca(t: np.ndarray, pos: np.ndarray) -> int:
    """Index of Earth closest approach after T+6 days."""
    mask = t >= 6.0
    offset = int(np.where(mask)[0][0])
    return offset + int(np.argmin(np.linalg.norm(pos[mask], axis=1)))


def circle_xy(cx: float, cy: float, r: float, n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    a = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(a), cy + r * np.sin(a)


def main() -> None:
    if not os.path.exists(NOM_CSV):
        print(f"Missing: {NOM_CSV}\nRun: cargo run --bin artemis --release")
        return

    df   = pd.read_csv(NOM_CSV)
    km   = 1e-3
    sc   = df[["x_m", "y_m", "z_m"]].values * km
    mn   = df[["moon_x_m", "moon_y_m", "moon_z_m"]].values * km
    t    = df["time_s"].values / 86_400.0

    # Clip at Earth CA
    end  = clip_at_earth_ca(t, sc)
    sc, mn, t = sc[:end+1], mn[:end+1], t[:end+1]

    # ── Rotating frame transform ──────────────────────────────────────────────
    # Proper Earth-Moon rotating frame that accounts for the Moon's inclination.
    # At each step:
    #   x-hat = radial (Earth→Moon 3D unit vector)
    #   z-hat = fixed Moon orbital angular momentum direction
    #   y-hat = z-hat × x-hat  (in-plane tangential)
    #
    # Projecting onto xy gives the Moon's orbital plane view, so the spacecraft
    # 2D distance to the Moon matches its true 3D distance (no inclination error).

    # Fixed z-hat: mean orbital angular momentum of the Moon over the arc
    dt    = np.gradient(t) * 86_400.0              # time steps in seconds
    mn_v  = np.gradient(mn, axis=0) / dt[:, None]  # Moon velocity [km/s]
    h_vec = np.cross(mn, mn_v)
    z_hat = h_vec.mean(axis=0)
    z_hat /= np.linalg.norm(z_hat)

    sc_r = np.zeros((len(t), 2))
    mn_r = np.zeros((len(t), 2))
    for i in range(len(t)):
        x_hat = mn[i] / np.linalg.norm(mn[i])
        y_hat = np.cross(z_hat, x_hat)
        y_hat /= np.linalg.norm(y_hat)
        sc_r[i] = [np.dot(x_hat, sc[i]), np.dot(y_hat, sc[i])]
        mn_r[i] = [np.dot(x_hat, mn[i]), np.dot(y_hat, mn[i])]  # ≈ (|mn|, 0)

    moon_dist = float(np.mean(mn_r[:, 0]))   # ~384 400 km

    # ── Downsample to N_FRAMES ────────────────────────────────────────────────
    idx   = np.round(np.linspace(0, len(t) - 1, N_FRAMES)).astype(int)
    sc_a  = sc[idx, :2]      # animated SC positions  — ECI
    mn_a  = mn[idx, :2]      # animated Moon positions — ECI
    scr_a = sc_r[idx]        # animated SC positions  — rotating

    # ── Static body circles ───────────────────────────────────────────────────
    EARTH_DRAW_R = EARTH_R_KM * 0.3    # display radius — real scale would be invisible
    MOON_DRAW_R  = MOON_R_KM  * 0.6
    ex,  ey  = circle_xy(0,          0, EARTH_DRAW_R)
    mrx, mry = circle_xy(moon_dist,  0, MOON_DRAW_R)

    # ── Build figure with two subplots ────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["ECI (inertial) frame", "Earth–Moon rotating frame"],
        horizontal_spacing=0.06,
    )

    # Trace indices (order matters for frame updates):
    #  0  ECI  — trajectory path    (static)
    #  1  ECI  — spacecraft dot     (animated)
    #  2  ECI  — Moon dot           (animated)
    #  3  ECI  — Earth circle       (static)
    #  4  ROT  — trajectory path    (static)
    #  5  ROT  — spacecraft dot     (animated)
    #  6  ROT  — Moon circle        (static)
    #  7  ROT  — Earth circle       (static)

    ANIMATED = [1, 2, 5]   # indices updated each frame

    def add(x, y, row, col, color, mode="lines", size=6, width=1,
            fill=None, fillcolor=None):
        kw: dict = {"x": x, "y": y, "mode": mode, "showlegend": False}
        if "lines" in mode:
            kw["line"] = {"color": color, "width": width}
        if "markers" in mode:
            kw["marker"] = {"color": color, "size": size}
        if fill:
            kw["fill"] = fill
            kw["fillcolor"] = fillcolor
        fig.add_trace(go.Scatter(**kw), row=row, col=col)

    # ECI
    add(sc[:, 0], sc[:, 1],   1, 1, "rgba(255,215,0,0.25)", width=1)  # 0 path
    add([sc_a[0, 0]], [sc_a[0, 1]], 1, 1, "white", "markers", size=8) # 1 SC
    add([mn_a[0, 0]], [mn_a[0, 1]], 1, 1, "#C8C8C8", "markers", size=9) # 2 Moon
    add(ex, ey, 1, 1, "rgba(33,150,243,0.5)", fill="toself",
        fillcolor="rgba(33,150,243,0.10)")                              # 3 Earth

    # Rotating
    add(sc_r[:, 0], sc_r[:, 1], 1, 2, "rgba(255,215,0,0.25)", width=1) # 4 path
    add([scr_a[0, 0]], [scr_a[0, 1]], 1, 2, "white", "markers", size=8) # 5 SC
    add(mrx, mry, 1, 2, "rgba(200,200,200,0.45)", fill="toself",
        fillcolor="rgba(200,200,200,0.10)")                              # 6 Moon
    add(ex, ey, 1, 2, "rgba(33,150,243,0.5)", fill="toself",
        fillcolor="rgba(33,150,243,0.10)")                              # 7 Earth

    # ── Animation frames ──────────────────────────────────────────────────────
    fig.frames = [
        go.Frame(
            data=[
                go.Scatter(x=[sc_a[i, 0]],  y=[sc_a[i, 1]]),
                go.Scatter(x=[mn_a[i, 0]],  y=[mn_a[i, 1]]),
                go.Scatter(x=[scr_a[i, 0]], y=[scr_a[i, 1]]),
            ],
            traces=ANIMATED,
            name=str(i),
        )
        for i in range(N_FRAMES)
    ]

    # ── Slider steps ─────────────────────────────────────────────────────────
    slider_steps = [
        {
            "args": [[str(i)], {"frame": {"duration": FRAME_MS, "redraw": False},
                                "mode": "immediate", "transition": {"duration": 0}}],
            "label": f"{t[idx[i]]:.1f}",
            "method": "animate",
        }
        for i in range(0, N_FRAMES, max(1, N_FRAMES // 50))   # ~50 labelled ticks
    ]

    # ── Layout ────────────────────────────────────────────────────────────────
    r_eci = float(np.max(np.abs(sc[:, :2]))) * 1.08
    r_rot_x_lo = float(sc_r[:, 0].min()) * 1.08
    r_rot_x_hi = (moon_dist + MOON_R_KM * 4) * 1.05
    r_rot_y    = float(np.max(np.abs(sc_r[:, 1]))) * 1.12

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgb(10,10,18)",
        plot_bgcolor="rgb(10,10,18)",
        font={"color": "white", "size": 11},
        title={"text": "Artemis 2 — Trajectory in ECI vs Earth–Moon Rotating Frame",
               "x": 0.01, "xanchor": "left"},
        annotations=[
            # ECI frame — Earth label
            {"text": "Earth", "x": 0, "y": EARTH_DRAW_R * 1.6,
             "xref": "x", "yref": "y", "showarrow": False,
             "font": {"color": "#64B5F6", "size": 11}},
            # Rotating frame — Earth label
            {"text": "Earth", "x": 0, "y": EARTH_DRAW_R * 1.6,
             "xref": "x2", "yref": "y2", "showarrow": False,
             "font": {"color": "#64B5F6", "size": 11}},
            # Rotating frame — Moon label
            {"text": "Moon", "x": moon_dist, "y": MOON_DRAW_R * 1.8,
             "xref": "x2", "yref": "y2", "showarrow": False,
             "font": {"color": "#C8C8C8", "size": 11}},
        ],
        height=560,
        margin={"l": 40, "r": 40, "t": 60, "b": 80},
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.5, "xanchor": "center",
            "y": -0.12, "yanchor": "top",
            "buttons": [
                {"label": "▶  Play",
                 "method": "animate",
                 "args": [None, {"frame": {"duration": FRAME_MS, "redraw": False},
                                 "fromcurrent": True,
                                 "transition": {"duration": 0}}]},
                {"label": "⏸  Pause",
                 "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate",
                                   "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.05, "len": 0.90,
            "y": -0.04, "yanchor": "top",
            "currentvalue": {"prefix": "T+", "suffix": " days",
                             "visible": True, "xanchor": "center"},
            "transition": {"duration": 0},
            "steps": slider_steps,
        }],
    )

    # ECI axes: equal scale, centred on Earth
    fig.update_xaxes(range=[-r_eci, r_eci], showgrid=False, zeroline=False,
                     showticklabels=False, row=1, col=1)
    fig.update_yaxes(range=[-r_eci, r_eci], showgrid=False, zeroline=False,
                     showticklabels=False, scaleanchor="x", scaleratio=1,
                     row=1, col=1)

    # Rotating axes: x covers Earth→Moon, y shows cross-track spread
    fig.update_xaxes(range=[r_rot_x_lo, r_rot_x_hi], showgrid=False,
                     zeroline=False, showticklabels=False, row=1, col=2)
    fig.update_yaxes(range=[-r_rot_y, r_rot_y], showgrid=False,
                     zeroline=False, showticklabels=False,
                     scaleanchor="x2", scaleratio=1, row=1, col=2)

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    fig.write_html(OUT_HTML, auto_play=False)
    print(f"Saved → {OUT_HTML}")
    webbrowser.open(Path(OUT_HTML).resolve().as_uri())


if __name__ == "__main__":
    main()
