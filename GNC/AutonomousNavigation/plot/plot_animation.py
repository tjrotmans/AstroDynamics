"""
Autonomous Navigation â€” interactive Plotly animation.

Left panel  : Hill-frame xâ€“y view (radial vs along-track).
              Truth trail fades in; animated dot shows spacecraft.
              EKF estimate shown as a second dot.
              Camera FOV wedge animates with the real boresight direction.
Right panel : Range to Bennu + EKF position error over time,
              with a vertical time cursor moving in sync.

Output: out/nav_animation.html  (auto-opens in browser)
Run from GNC/AutonomousNavigation/:
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

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

HERE     = Path(__file__).parent
ROOT     = HERE.parent
OUT_HTML = ROOT / "out" / "nav_animation.html"

# â”€â”€ Animation settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

N_FRAMES  = 300      # animation frames
FRAME_MS  = 40       # ms per frame â†’ ~12 s total playback
FOV_HALF_DEG = 15.0  # camera half-angle (must match config.rs)

# â”€â”€ Colours â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

C_TRUTH  = "#FFD700"   # gold  â€” truth spacecraft
C_EKF    = "#E63946"   # red   â€” EKF estimate
C_BENNU  = "#C8A84B"   # amber â€” Bennu
C_TRAIL  = "rgba(255,215,0,0.18)"
C_EKF_TR = "rgba(230,57,70,0.15)"
C_FOV    = "rgba(0,200,255,0.25)"
C_FOV_ED = "rgba(0,200,255,0.6)"
C_RANGE  = "#4FC3F7"
C_ERR    = "#FF7043"


def fov_wedge_xy(
    sc_x: float, sc_y: float,
    bore_x: float, bore_y: float,
    half_deg: float,
    length: float,
    n_arc: int = 20,
) -> tuple[list[float], list[float]]:
    """Return (xs, ys) for a closed FOV wedge polygon in 2-D."""
    # Project boresight onto x-y and normalise
    b = np.array([bore_x, bore_y])
    bn = np.linalg.norm(b)
    if bn < 1e-9:
        return [sc_x], [sc_y]
    b /= bn

    half_rad = np.radians(half_deg)
    angle0   = np.arctan2(b[1], b[0])

    # Arc at distance `length`
    arcs  = np.linspace(angle0 - half_rad, angle0 + half_rad, n_arc)
    arc_x = sc_x + length * np.cos(arcs)
    arc_y = sc_y + length * np.sin(arcs)

    xs = [sc_x] + arc_x.tolist() + [sc_x]
    ys = [sc_y] + arc_y.tolist() + [sc_y]
    return xs, ys


def main() -> None:
    # â”€â”€ Load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    truth_f = ROOT / "out" / "truth.csv"
    ekf_f   = ROOT / "out" / "ekf_est.csv"
    if not truth_f.exists():
        print(f"Missing {truth_f}\nRun:  cargo run --bin autonav --release")
        return

    truth = pd.read_csv(truth_f)
    ekf   = pd.read_csv(ekf_f)

    # Coordinates in km
    tx = truth["x_m"].values / 1e3
    ty = truth["y_m"].values / 1e3
    tz = truth["z_m"].values / 1e3
    ex = ekf["x_m"].values / 1e3
    ey = ekf["y_m"].values / 1e3

    t_h   = truth["time_s"].values / 3600.0
    rng   = np.sqrt(tx**2 + ty**2 + tz**2)           # km
    err_m = np.sqrt((truth["x_m"].values - ekf["x_m"].values)**2 +
                    (truth["y_m"].values - ekf["y_m"].values)**2 +
                    (truth["z_m"].values - ekf["z_m"].values)**2)

    has_bore = "bx" in truth.columns
    bx = truth["bx"].values if has_bore else -tx / rng
    by = truth["by"].values if has_bore else -ty / rng

    # â”€â”€ Downsample â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    idx   = np.round(np.linspace(0, len(t_h) - 1, N_FRAMES)).astype(int)
    t_a   = t_h[idx]
    tx_a  = tx[idx];  ty_a = ty[idx]
    ex_a  = ex[idx];  ey_a = ey[idx]
    rng_a = rng[idx]; err_a = err_m[idx]
    bx_a  = bx[idx];  by_a  = by[idx]

    # FOV cone length = distance to Bennu (capped for readability)
    fov_len = np.minimum(rng_a, 5.0)

    # â”€â”€ Figure layout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.62, 0.38],
        row_heights=[0.55, 0.45],
        specs=[
            [{"rowspan": 2}, {}],
            [None,           {}],
        ],
        subplot_titles=[
            "Hill Frame â€” x (radial) vs y (along-track)",
            "Range to Bennu [km]",
            "",
            "EKF Position Error [m]",
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    # â”€â”€ Static traces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Trace index map (order must match frame update list):
    #   0  truth trail    (static)
    #   1  EKF trail      (static)
    #   2  Bennu          (static)
    #   3  FOV wedge      (animated)
    #   4  truth SC dot   (animated)
    #   5  EKF SC dot     (animated)
    #   6  range line     (static)
    #   7  range cursor   (animated)
    #   8  error line     (static)
    #   9  error cursor   (animated)

    ANIMATED = [3, 4, 5, 7, 9]

    # 0 â€” truth trail
    fig.add_trace(go.Scatter(
        x=tx, y=ty, mode="lines",
        line=dict(color=C_TRAIL, width=1.5),
        showlegend=False, name="Truth trail",
    ), row=1, col=1)

    # 1 â€” EKF trail
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(color=C_EKF_TR, width=1.2, dash="dot"),
        showlegend=False, name="EKF trail",
    ), row=1, col=1)

    # 2 â€” Bennu body
    bennu_r_km = 0.000262   # 262 m radius
    phi = np.linspace(0, 2 * np.pi, 60)
    fig.add_trace(go.Scatter(
        x=bennu_r_km * np.cos(phi),
        y=bennu_r_km * np.sin(phi),
        mode="lines", fill="toself",
        line=dict(color=C_BENNU, width=1),
        fillcolor="rgba(200,168,75,0.35)",
        showlegend=False, name="Bennu",
    ), row=1, col=1)

    # 3 â€” FOV wedge (animated, first frame)
    fwx, fwy = fov_wedge_xy(tx_a[0], ty_a[0], bx_a[0], by_a[0],
                             FOV_HALF_DEG, fov_len[0])
    fig.add_trace(go.Scatter(
        x=fwx, y=fwy, mode="lines", fill="toself",
        line=dict(color=C_FOV_ED, width=1),
        fillcolor=C_FOV,
        showlegend=False, name="Camera FOV",
    ), row=1, col=1)

    # 4 â€” truth SC dot
    fig.add_trace(go.Scatter(
        x=[tx_a[0]], y=[ty_a[0]], mode="markers+text",
        marker=dict(color=C_TRUTH, size=10, symbol="diamond"),
        text=["S/C"], textposition="top right",
        textfont=dict(color=C_TRUTH, size=9),
        showlegend=True, name="Truth",
    ), row=1, col=1)

    # 5 â€” EKF SC dot
    fig.add_trace(go.Scatter(
        x=[ex_a[0]], y=[ey_a[0]], mode="markers+text",
        marker=dict(color=C_EKF, size=8, symbol="circle"),
        text=["EKF"], textposition="bottom right",
        textfont=dict(color=C_EKF, size=9),
        showlegend=True, name="EKF estimate",
    ), row=1, col=1)

    # 6 â€” range line (static)
    fig.add_trace(go.Scatter(
        x=t_h, y=rng,
        mode="lines", line=dict(color=C_RANGE, width=1.5),
        showlegend=False,
    ), row=1, col=2)

    # 7 â€” range cursor (animated)
    r0 = float(rng_a[0])
    fig.add_trace(go.Scatter(
        x=[t_a[0], t_a[0]], y=[0, r0],
        mode="lines", line=dict(color=C_RANGE, width=1, dash="dash"),
        showlegend=False,
    ), row=1, col=2)

    # 8 â€” error line (static)
    fig.add_trace(go.Scatter(
        x=t_h, y=err_m,
        mode="lines", line=dict(color=C_ERR, width=1.5),
        showlegend=False,
    ), row=2, col=2)

    # 9 â€” error cursor (animated)
    e0 = float(err_a[0])
    fig.add_trace(go.Scatter(
        x=[t_a[0], t_a[0]], y=[0, e0],
        mode="lines", line=dict(color=C_ERR, width=1, dash="dash"),
        showlegend=False,
    ), row=2, col=2)

    # â”€â”€ Animation frames â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    frames = []
    for i in range(N_FRAMES):
        fwx, fwy = fov_wedge_xy(tx_a[i], ty_a[i], bx_a[i], by_a[i],
                                 FOV_HALF_DEG, fov_len[i])
        ri = float(rng_a[i])
        ei = float(err_a[i])
        frames.append(go.Frame(
            data=[
                go.Scatter(x=fwx, y=fwy),                         # 3 FOV
                go.Scatter(x=[tx_a[i]], y=[ty_a[i]]),             # 4 truth dot
                go.Scatter(x=[ex_a[i]], y=[ey_a[i]]),             # 5 EKF dot
                go.Scatter(x=[t_a[i], t_a[i]], y=[0, ri]),        # 7 range cursor
                go.Scatter(x=[t_a[i], t_a[i]], y=[0, ei]),        # 9 error cursor
            ],
            traces=ANIMATED,
            name=str(i),
        ))
    fig.frames = frames

    # â”€â”€ Slider â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tick_every = max(1, N_FRAMES // 50)
    slider_steps = [
        {
            "args": [[str(i)], {"frame": {"duration": FRAME_MS, "redraw": False},
                                "mode": "immediate", "transition": {"duration": 0}}],
            "label": f"{t_a[i]:.1f}",
            "method": "animate",
        }
        for i in range(0, N_FRAMES, tick_every)
    ]

    # â”€â”€ Layout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    pad_xy = 0.15
    xlo = min(tx.min(), ex.min()) - pad_xy
    xhi = max(tx.max(), ex.max()) + pad_xy
    ylo = min(ty.min(), ey.min()) - pad_xy
    yhi = max(ty.max(), ey.max()) + pad_xy

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgb(10,10,18)",
        plot_bgcolor="rgb(10,10,18)",
        font=dict(color="white", size=11),
        title=dict(
            text="Autonomous Navigation â€” Bennu Proximity Approach",
            x=0.01, xanchor="left",
        ),
        height=620,
        margin=dict(l=50, r=30, t=60, b=90),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0.4)",
                    bordercolor="rgba(255,255,255,0.2)", borderwidth=1),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.28, "xanchor": "center",
            "y": -0.13, "yanchor": "top",
            "buttons": [
                {"label": "â–¶  Play",
                 "method": "animate",
                 "args": [None, {"frame": {"duration": FRAME_MS, "redraw": False},
                                 "fromcurrent": True,
                                 "transition": {"duration": 0}}]},
                {"label": "â¸  Pause",
                 "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate",
                                   "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.05, "len": 0.90,
            "y": -0.06, "yanchor": "top",
            "currentvalue": {"prefix": "T+", "suffix": " h",
                             "visible": True, "xanchor": "center"},
            "transition": {"duration": 0},
            "steps": slider_steps,
        }],
    )

    # Trajectory panel axes (independent scaling, km)
    fig.update_xaxes(range=[xlo, xhi], showgrid=True,
                     gridcolor="rgba(255,255,255,0.06)",
                     zeroline=False, title_text="x [km]  (radial)", row=1, col=1)
    fig.update_yaxes(range=[ylo, yhi], showgrid=True,
                     gridcolor="rgba(255,255,255,0.06)",
                     zeroline=False, title_text="y [km]  (along-track)", row=1, col=1)

    # Range panel
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     zeroline=False, title_text="Time [h]", row=1, col=2)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     zeroline=False, title_text="km", row=1, col=2)

    # Error panel
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     zeroline=False, title_text="Time [h]", row=2, col=2)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     zeroline=False, title_text="m", row=2, col=2)

    # Bennu label annotation
    fig.add_annotation(
        x=0, y=0.004, text="Bennu",
        showarrow=False, font=dict(color=C_BENNU, size=10),
        xref="x", yref="y",
    )

    # â”€â”€ Save and open â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    OUT_HTML.parent.mkdir(exist_ok=True)
    fig.write_html(str(OUT_HTML), auto_play=False)
    print(f"Saved: {OUT_HTML}")
    webbrowser.open(OUT_HTML.resolve().as_uri())


if __name__ == "__main__":
    main()
