"""
Bennu proximity mission — interactive Plotly HTML animation.

Left panel  : Hill-frame x–z view (radial vs out-of-plane, coloured by phase).
              Truth trail with spacecraft dot animated.
              EKF estimate shown as secondary animated dot.
              Bennu body shown at origin.
Right panels: Range to Bennu + EKF position error, with animated time cursors.

Phase legend and pointing-mode indicator also animated.

Output: out/mission/mission_anim.html  (auto-opens in browser)
Run from GNC/AutonomousNavigation/:
    python plot/plot_mission_anim.py
"""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ─────────────────────────────────────────────────────────────────────

HERE     = Path(__file__).parent
ROOT     = HERE.parent
DATA     = ROOT / "out" / "mission"
OUT_HTML = DATA / "mission_anim.html"

# ── Animation settings ────────────────────────────────────────────────────────

N_FRAMES  = 400      # animation frames
FRAME_MS  = 35       # ms per frame → ~14 s total playback

# ── Colours ───────────────────────────────────────────────────────────────────

PHASE_COLORS = {
    "Capture":     "#f06292",
    "Survey":      "#4fc3f7",
    "CloseOrbit":  "#ab47bc",
    "Flyover":     "#ff7043",
    "ScienceHold": "#66bb6a",
}
C_TRUTH  = "#FFD700"
C_EKF    = "#E63946"
C_BENNU  = "#C8A84B"
C_TRAIL  = "rgba(255,215,0,0.15)"
C_EKF_TR = "rgba(230,57,70,0.12)"
C_RANGE  = "#4FC3F7"
C_ERR    = "#FF7043"


def main() -> None:
    # ── Load data ──────────────────────────────────────────────────────────────
    nav_f = DATA / "nav.csv"
    att_f = DATA / "attitude.csv"
    if not nav_f.exists():
        print(f"Missing {nav_f}")
        print("Run:  cargo run --bin proximity_mission --release")
        return

    nav = pd.read_csv(nav_f)
    att = pd.read_csv(att_f) if att_f.exists() else None

    t0_s  = nav["time_s"].iloc[0]
    t_h   = (nav["time_s"] - t0_s) / 3600.0

    # Coordinates in km
    tx = nav["tx_m"].values / 1e3
    ty = nav["ty_m"].values / 1e3
    tz = nav["tz_m"].values / 1e3
    ex = nav["ex_m"].values / 1e3
    ey = nav["ey_m"].values / 1e3
    ez = nav["ez_m"].values / 1e3

    rng_km = nav["range_km"].values
    err_m  = np.sqrt((nav["tx_m"] - nav["ex_m"])**2 +
                     (nav["ty_m"] - nav["ey_m"])**2 +
                     (nav["tz_m"] - nav["ez_m"])**2).values
    sigma_m = np.maximum(nav["sigma_r_m"].values, 0.0)
    phases   = nav["phase"].values

    # Pointing mode from attitude CSV (if available)
    if att is not None and "pointing_mode" in att.columns:
        att_t_h = (att["time_s"] - t0_s) / 3600.0
        pt_mode = att["pointing_mode"].values
    else:
        att_t_h = t_h
        pt_mode = np.where(phases == "ScienceHold", "VelocityAligned", "Nadir")

    # ── Downsample ─────────────────────────────────────────────────────────────
    idx   = np.round(np.linspace(0, len(t_h) - 1, N_FRAMES)).astype(int)
    t_a   = t_h.values[idx]
    tx_a  = tx[idx];  ty_a = ty[idx];  tz_a = tz[idx]
    ex_a  = ex[idx];  ey_a = ey[idx];  ez_a = ez[idx]
    rng_a = rng_km[idx]
    err_a = err_m[idx]
    sig_a = sigma_m[idx]
    ph_a  = phases[idx]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.60, 0.40],
        row_heights=[0.55, 0.45],
        specs=[
            [{"rowspan": 2, "type": "scene"}, {"type": "scatter"}],
            [None,                              {"type": "scatter"}],
        ],
        subplot_titles=[
            "Orbit — Hill Frame 3-D  [km]  (drag to rotate)",
            "Range to Bennu [km]",
            "",
            "EKF Position Error [m]",
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    # ── Static traces ─────────────────────────────────────────────────────────
    # Draw truth trail segment-by-segment, coloured by phase
    uniq_phases = list(dict.fromkeys(phases))
    for ph in uniq_phases:
        mask = phases == ph
        col  = PHASE_COLORS.get(ph, "gray")
        fig.add_trace(go.Scatter3d(
            x=tx[mask], y=ty[mask], z=tz[mask],
            mode="lines",
            line=dict(color=col, width=3),
            name=ph, legendgroup=ph,
            showlegend=True,
        ), row=1, col=1)

    # EKF trail (static, translucent)
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color=C_EKF_TR, width=1),
        showlegend=True, name="EKF estimate",
    ), row=1, col=1)

    # Bennu body (marker at origin)
    fig.add_trace(go.Scatter3d(
        x=[0.0], y=[0.0], z=[0.0], mode="markers",
        marker=dict(color=C_BENNU, size=12, symbol="circle",
                    line=dict(color="#8d6e40", width=1)),
        showlegend=True, name="Bennu",
    ), row=1, col=1)

    # Animated: truth spacecraft dot
    fig.add_trace(go.Scatter3d(
        x=[tx_a[0]], y=[ty_a[0]], z=[tz_a[0]], mode="markers+text",
        marker=dict(color=C_TRUTH, size=6, symbol="diamond"),
        text=["S/C"], textposition="top right",
        textfont=dict(color=C_TRUTH, size=9),
        showlegend=True, name="Truth S/C",
    ), row=1, col=1)

    # Animated: EKF dot
    fig.add_trace(go.Scatter3d(
        x=[ex_a[0]], y=[ey_a[0]], z=[ez_a[0]], mode="markers+text",
        marker=dict(color=C_EKF, size=5, symbol="circle"),
        text=["EKF"], textposition="bottom right",
        textfont=dict(color=C_EKF, size=9),
        showlegend=True, name="EKF dot",
    ), row=1, col=1)

    # Static: range line
    fig.add_trace(go.Scatter(
        x=t_h, y=rng_km, mode="lines",
        line=dict(color=C_RANGE, width=1.5),
        showlegend=False, name="Range",
    ), row=1, col=2)

    # Animated: range cursor
    fig.add_trace(go.Scatter(
        x=[t_a[0], t_a[0]], y=[0, float(rng_a[0])],
        mode="lines", line=dict(color=C_RANGE, width=1, dash="dash"),
        showlegend=False,
    ), row=1, col=2)

    # Static: error + 3σ
    fig.add_trace(go.Scatter(
        x=t_h, y=err_m, mode="lines",
        line=dict(color=C_ERR, width=1.5),
        showlegend=False, name="|Δr|",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=t_h, y=3*sigma_m, mode="lines",
        line=dict(color="#ffa726", width=1.0, dash="dash"),
        showlegend=False, name="3σ",
    ), row=2, col=2)

    # Animated: error cursor
    fig.add_trace(go.Scatter(
        x=[t_a[0], t_a[0]], y=[0, float(err_a[0])],
        mode="lines", line=dict(color=C_ERR, width=1, dash="dash"),
        showlegend=False,
    ), row=2, col=2)

    # Trace indices of animated elements (match order of add_trace above):
    #  0..len(uniq_phases)-1  : phase trail lines        (static)
    #  len(uniq_phases)       : EKF trail                (static)
    #  len(uniq_phases)+1     : Bennu circle             (static)  n_static = len+2
    #  n_static + 0           : truth spacecraft dot     (animated)
    #  n_static + 1           : EKF dot                  (animated)
    #  n_static + 2           : range line               (static)
    #  n_static + 3           : range cursor             (animated)
    #  n_static + 4           : error line               (static)
    #  n_static + 5           : 3-sigma line             (static)
    #  n_static + 6           : error cursor             (animated)
    n_static = len(uniq_phases) + 2
    IDX_TRUTH_DOT  = n_static
    IDX_EKF_DOT    = n_static + 1
    IDX_RNG_CURSOR = n_static + 3
    IDX_ERR_CURSOR = n_static + 6
    ANIMATED = [IDX_TRUTH_DOT, IDX_EKF_DOT, IDX_RNG_CURSOR, IDX_ERR_CURSOR]

    # ── Animation frames ───────────────────────────────────────────────────────
    frames = []
    for i in range(N_FRAMES):
        phase_col = PHASE_COLORS.get(str(ph_a[i]), "gray")
        ri = float(rng_a[i])
        ei = float(err_a[i])
        frame_data = [
            go.Scatter3d(x=[tx_a[i]], y=[ty_a[i]], z=[tz_a[i]],    # truth dot
                         marker=dict(color=phase_col, size=6, symbol="diamond")),
            go.Scatter3d(x=[ex_a[i]], y=[ey_a[i]], z=[ez_a[i]]),   # EKF dot
            go.Scatter(x=[t_a[i], t_a[i]], y=[0, ri]),    # range cursor
            go.Scatter(x=[t_a[i], t_a[i]], y=[0, ei]),    # error cursor
        ]
        frames.append(go.Frame(
            data=frame_data,
            traces=ANIMATED,
            name=str(i),
            layout=go.Layout(
                annotations=[dict(
                    x=0.01, y=0.01, xref="paper", yref="paper",
                    text=(f"t = {t_a[i]:.1f} h  |  Phase: {ph_a[i]}"
                          f"  |  Range: {ri:.2f} km  |  Nav err: {ei:.0f} m"),
                    showarrow=False, font=dict(size=11, color="white"),
                    bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1,
                )]
            ),
        ))

    fig.frames = frames

    # ── Static reference vectors ───────────────────────────────────────────────
    r_scale = float(np.percentile(rng_km, 75)) * 1.8   # arrow length ≈ 1.8× orbit r

    # Sun direction (from srp.csv, mean over mission — sun moves slowly)
    srp_path = DATA / "srp.csv"
    if srp_path.exists():
        srp_df  = pd.read_csv(srp_path)
        sun_hat = srp_df[["sunx", "suny", "sunz"]].mean().values
        sun_hat = sun_hat / np.linalg.norm(sun_hat)
        sv = sun_hat * r_scale
        fig.add_trace(go.Scatter3d(
            x=[0.0, float(sv[0])], y=[0.0, float(sv[1])], z=[0.0, float(sv[2])],
            mode="lines+text",
            line=dict(color="#FFD700", width=3),
            text=["", "Sun"], textfont=dict(color="#FFD700", size=11),
            textposition="top center",
            showlegend=True, name="Sun direction",
        ), row=1, col=1)

    # Orbit normal h = r x v (representative Survey step)
    sur_mask = nav["phase"].values == "Survey"
    if sur_mask.any():
        i0   = int(np.where(sur_mask)[0][0])
        r_v  = np.array([tx[i0], ty[i0], tz[i0]])
        v_v  = nav[["tvx", "tvy", "tvz"]].values[i0]
        hvec = np.cross(r_v, v_v)
        hhat = hvec / np.linalg.norm(hvec)
        hv   = hhat * r_scale
        fig.add_trace(go.Scatter3d(
            x=[0.0, float(hv[0])], y=[0.0, float(hv[1])], z=[0.0, float(hv[2])],
            mode="lines+text",
            line=dict(color="#80cbc4", width=2),
            text=["", "h (orbit normal)"], textfont=dict(color="#80cbc4", size=10),
            textposition="top center",
            showlegend=True, name="Orbit normal",
        ), row=1, col=1)

    # Bennu-centred ecliptic frame axes (+x = vernal equinox, +z = ecliptic north)
    ax_len = float(rng_km.mean()) * 0.45
    for ax_dir, ax_col, ax_lbl in [
        ([1, 0, 0], "#FF5555", "+x (vernal equinox)"),
        ([0, 1, 0], "#55FF77", "+y"),
        ([0, 0, 1], "#5599FF", "+z (ecliptic N)"),
    ]:
        e0, e1, e2 = ax_dir[0]*ax_len, ax_dir[1]*ax_len, ax_dir[2]*ax_len
        fig.add_trace(go.Scatter3d(
            x=[0.0, e0], y=[0.0, e1], z=[0.0, e2],
            mode="lines+text",
            line=dict(color=ax_col, width=2),
            text=["", ax_lbl], textfont=dict(color=ax_col, size=9),
            textposition="top center",
            showlegend=False,
        ), row=1, col=1)

    # ── Slider + buttons ──────────────────────────────────────────────────────
    sliders = [dict(
        steps=[dict(method="animate",
                    args=[[f.name], dict(mode="immediate",
                                         frame=dict(duration=FRAME_MS, redraw=True),
                                         transition=dict(duration=0))],
                    label="") for f in frames],
        transition=dict(duration=0),
        x=0.0, y=0.0,
        currentvalue=dict(visible=False),
        len=1.0,
    )]
    buttons = [dict(
        label="▶  Play",
        method="animate",
        args=[None, dict(frame=dict(duration=FRAME_MS, redraw=True),
                         fromcurrent=True, transition=dict(duration=0))],
    ), dict(
        label="⏸  Pause",
        method="animate",
        args=[[None], dict(frame=dict(duration=0, redraw=False),
                           mode="immediate", transition=dict(duration=0))],
    )]

    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False,
                          y=1.05, x=0.0, xanchor="left",
                          buttons=buttons)],
        sliders=sliders,
        title=dict(
            text="Bennu Proximity Mission — Interactive Animation",
            font=dict(size=15, color="white"), x=0.5,
        ),
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0a0a1a",
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(20,20,40,0.8)", bordercolor="#444",
            font=dict(size=10, color="white"),
        ),
        height=700,
        margin=dict(l=50, r=30, t=90, b=80),
        scene=dict(
            bgcolor="#0a0a1a",
            uirevision="static",
            xaxis=dict(title="x [km]", gridcolor="#222", zeroline=False,
                       showbackground=True, backgroundcolor="#0a0a1a",
                       tickfont=dict(color="#888", size=8)),
            yaxis=dict(title="y [km]", gridcolor="#222", zeroline=False,
                       showbackground=True, backgroundcolor="#0a0a1a",
                       tickfont=dict(color="#888", size=8)),
            zaxis=dict(title="z [km]", gridcolor="#222", zeroline=False,
                       showbackground=True, backgroundcolor="#0a0a1a",
                       tickfont=dict(color="#888", size=8)),
            aspectmode="cube",
        ),
    )

    # Axis styling (2-D panels only)
    fig.update_xaxes(gridcolor="#222", zeroline=False, showgrid=True)
    fig.update_yaxes(gridcolor="#222", zeroline=True,  showgrid=True,
                     zerolinecolor="#444", zerolinewidth=1)
    fig.update_xaxes(title_text="Elapsed time [h]",      row=1, col=2)
    fig.update_yaxes(title_text="Range [km]",             row=1, col=2)
    fig.update_xaxes(title_text="Elapsed time [h]",      row=2, col=2)
    fig.update_yaxes(title_text="|Δr| [m]",              row=2, col=2, type="log")

    # Phase colour legend entries
    for ph, col in PHASE_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color=col, symbol="square"),
            name=ph, showlegend=True,
        ))

    # ── Write & open ──────────────────────────────────────────────────────────
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT_HTML), auto_open=False)
    print(f"Saved: {OUT_HTML}")
    webbrowser.open(OUT_HTML.as_uri())


if __name__ == "__main__":
    main()
