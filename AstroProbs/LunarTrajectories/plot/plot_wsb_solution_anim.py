"""
plot_wsb_solution_anim.py — animated 3D ECI + Moon-centred SOI view of a captured solution.

Left  : 3D ECI — Earth sphere, Moon orbit ring, full transfer + capture trajectory.
        Trail switches cyan (transfer) -> orange (capture) at Hill sphere entry.
Right : Moon-centred EM-inertial top-down view — Hill sphere approach.

WORKFLOW:
  1. cargo run --bin wsb_refine --release -- --reprop
     Writes: out/wsb/solution_hifi.csv

  2. cargo run --bin wsb_circularize --release
     Writes: out/wsb/epoch_info.txt   (required for correct ECI tilt)

  3. python plot/plot_wsb_solution_anim.py [--fps N] [--step HOURS]

If epoch_info.txt is missing the trajectory is shown in EM-inertial (R0=I).

Reads:  out/wsb/solution_hifi.csv
        out/wsb/epoch_info.txt   (optional — for ECI frame)
Saves:  out/wsb/wsb_solution_anim.html
"""
import argparse, pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wsb_style import (
    MU, X_M, L_KM, T_STAR, R_EARTH, R_MOON, R_HILL_ND, R_HILL_KM,
    COL_WSB, COL_EARTH, COL_MOON, COL_HILL, COL_CAPT,
    PAPER_BG, BG_3D,
    rot_em_to_eci, moon_em_to_eci, load_epoch_info, sphere_surface, downsample,
    dark_3d_axis, dark_soi_layout, play_pause_buttons, dark_slider,
)

ROOT       = pathlib.Path(__file__).parent.parent
SREF       = ROOT / "out" / "wsb"
SOL_CSV    = SREF / "solution_hifi.csv"   # metadata
MAXI_CSV   = SREF / "maxhifi.csv"         # full BCR4BP trajectory
EPOCH_TXT  = SREF / "epoch_info.txt"
OUT_HTML   = SREF / "wsb_solution_anim.html"

MAX_TRAIL = 600


def load_solution():
    # Load metadata from solution_hifi.csv (first row has hit_id, orbits, etc.)
    if not SOL_CSV.exists():
        raise FileNotFoundError(
            f"{SOL_CSV} not found.\n"
            "Run: cargo run ... --bin wsb_refine --release -- --reprop"
        )
    sol = pd.read_csv(SOL_CSV)
    info = sol.iloc[0]

    def _f(col, cast=float):
        try:
            return cast(info[col]) if col in info.index else None
        except (ValueError, TypeError):
            return None

    orbits  = _f("est_capture_orbits") or 0.0
    hit_id  = int(_f("hit_id") or 0)
    theta_s = _f("theta_sun_deg")
    r_apo   = _f("r_apogee_nd")

    def fmt(v, spec=".3f"):
        return f"{v:{spec}}" if v is not None else "N/A"

    meta = (f"theta_sun={fmt(theta_s)} deg  r_apo={fmt(r_apo)} nd  |  "
            f"hit={hit_id}  orbits={orbits:.2f}")

    # Load trajectory from maxhifi.csv (full high-fidelity BCR4BP propagation)
    if not MAXI_CSV.exists():
        raise FileNotFoundError(
            f"{MAXI_CSV} not found.\n"
            "Run: cargo run ... --bin wsb_refine --release -- --reprop"
        )
    df = pd.read_csv(MAXI_CSV).dropna(subset=["x_nd"]).sort_values("time_nd")
    return df, orbits, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps",  type=int,   default=24,
                        help="playback frames per second (default 24)")
    parser.add_argument("--step", type=float, default=24.0,
                        help="hours per animation frame (default 24)")
    args = parser.parse_args()
    frame_ms = int(1000 / args.fps)
    step_nd  = args.step * 3600.0 / T_STAR

    print("Loading solution_hifi.csv ...")
    df, orbits, meta = load_solution()
    t_nd = df.time_nd.values
    x_nd = df.x_nd.values
    y_nd = df.y_nd.values
    z_nd = df.z_nd.values if "z_nd" in df.columns else np.zeros_like(t_nd)

    if EPOCH_TXT.exists():
        _, R0_wsb = load_epoch_info(EPOCH_TXT)
        print("  ECI frame from epoch_info.txt")
    else:
        R0_wsb = np.eye(3)
        print("  WARNING: epoch_info.txt not found — using EM-inertial (R0=I)")

    # ---- ECI coordinates ----
    xi, yi, zi = rot_em_to_eci(x_nd, y_nd, z_nd, t_nd, R0_wsb)

    # BCR4BP circular Moon in ECI (for 3D animated dot)
    mxi_full, myi_full, mzi_full = moon_em_to_eci(t_nd, R0_wsb)

    # ---- EM-inertial (barycentre-centred) for SOI panel ----
    xi_em  = (x_nd * np.cos(t_nd) - y_nd * np.sin(t_nd)) * L_KM
    yi_em  = (x_nd * np.sin(t_nd) + y_nd * np.cos(t_nd)) * L_KM
    mxi_em = X_M * np.cos(t_nd) * L_KM
    myi_em = X_M * np.sin(t_nd) * L_KM
    xi_mc  = xi_em - mxi_em    # Moon-centred EM-inertial
    yi_mc  = yi_em - myi_em

    # ---- Hill sphere entry ----
    r_moon = np.sqrt((x_nd - X_M)**2 + y_nd**2)
    inside = r_moon < R_HILL_ND
    hill_t = t_nd[inside][0] if inside.any() else t_nd[-1]
    hill_d = hill_t * T_STAR / 86400.0
    t_max  = t_nd[-1]
    t_total_d = t_max * T_STAR / 86400.0

    print(f"  Transfer: {t_total_d:.1f} d  |  Hill entry: day {hill_d:.1f}  |  "
          f"{orbits:.2f} orbits")

    # ---- Scene limits (computed from actual trajectory extent) ----
    r_eci_lim = max(np.max(np.abs(xi)), np.max(np.abs(yi)), np.max(np.abs(zi))) * 1.2
    r_eci_lim = max(r_eci_lim, R_HILL_KM * 2.0)  # at least 2x Hill radius
    VIEW_SOI  = R_HILL_KM * 2.8

    # ---- Animation frame grid ----
    t_frames  = np.arange(0.0, t_max + step_nd * 0.5, step_nd)
    n_frames  = len(t_frames)
    step_d    = step_nd * T_STAR / 86400.0

    # ---- Static geometry ----
    sx, sy, sz = sphere_surface(R_EARTH, nu=30, nv=16)

    # Moon orbit ring (one full BCR4BP circular orbit)
    t_ring = np.linspace(0, 2 * np.pi, 200)
    mxr, myr, mzr = moon_em_to_eci(t_ring, R0_wsb)

    # SOI panel static shapes
    phi  = np.linspace(0, 2 * np.pi, 200)
    hs_x = (R_HILL_KM * np.cos(phi)).tolist()
    hs_y = (R_HILL_KM * np.sin(phi)).tolist()
    md_x = (R_MOON * 5 * np.cos(phi)).tolist()    # Moon disc enlarged 5x
    md_y = (R_MOON * 5 * np.sin(phi)).tolist()

    # ---- Build figure ----
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=["ECI Frame — Full Transfer",
                        "Moon-Centred — Hill Sphere Approach"],
        column_widths=[0.56, 0.44],
        horizontal_spacing=0.06,
    )

    # ---- Static LEFT (3D ECI) ----
    fig.add_trace(go.Surface(
        x=sx, y=sy, z=sz,
        colorscale=[[0, "#1A237E"], [0.5, COL_EARTH], [1, "#64B5F6"]],
        showscale=False, opacity=0.9,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.4),
        name="Earth", hoverinfo="skip",
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=mxr.tolist(), y=myr.tolist(), z=mzr.tolist(), mode="lines",
        line=dict(color="rgba(180,180,180,0.25)", width=2),
        name="Moon orbit", showlegend=True, hoverinfo="skip",
    ), row=1, col=1)

    # ---- Static RIGHT (2D Moon-centred) ----
    fig.add_trace(go.Scatter(
        x=hs_x, y=hs_y, mode="lines",
        line=dict(color=COL_HILL, width=1.5, dash="dot"),
        name=f"Hill sphere ({R_HILL_KM/1e3:.0f}e3 km)", showlegend=True,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=md_x, y=md_y, mode="lines",
        fill="toself", fillcolor="rgba(160,160,160,0.35)",
        line=dict(color="#AAAAAA", width=1),
        name="Moon", showlegend=False,
    ), row=1, col=2)

    # ---- Animated traces ----
    n0_eci = len(fig.data)

    fig.add_trace(go.Scatter3d(
        x=[float(xi[0])], y=[float(yi[0])], z=[float(zi[0])],
        mode="markers",
        marker=dict(color=COL_WSB, size=6, symbol="circle",
                    line=dict(color="white", width=0.8)),
        name="Spacecraft", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[float(mxi_full[0])], y=[float(myi_full[0])], z=[float(mzi_full[0])],
        mode="markers+text",
        marker=dict(color=COL_MOON, size=7),
        text=["Moon"], textposition="top center",
        textfont=dict(color=COL_MOON, size=9),
        name="Moon", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="lines",
        line=dict(color=COL_WSB, width=4),
        name="Transfer trail", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="lines",
        line=dict(color=COL_CAPT, width=4),
        name="Capture arc", showlegend=True,
    ), row=1, col=1)

    n0_soi = len(fig.data)

    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines",
        line=dict(color=COL_WSB, width=2.5),
        name="Transfer", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines",
        line=dict(color=COL_CAPT, width=2.5),
        name="Capture", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[float(xi_mc[0])], y=[float(yi_mc[0])],
        mode="markers",
        marker=dict(color=COL_WSB, size=8, symbol="circle",
                    line=dict(color="white", width=0.8)),
        name="SC SOI", showlegend=False,
    ), row=1, col=2)

    ANIM_TRACES = [n0_eci, n0_eci+1, n0_eci+2, n0_eci+3,
                   n0_soi, n0_soi+1, n0_soi+2]

    # ---- Build frames ----
    print(f"  Building {n_frames} frames ...", end="", flush=True)
    frames = []
    for fi, t_f in enumerate(t_frames):
        if fi % 50 == 0:
            print(".", end="", flush=True)
        day   = t_f * T_STAR / 86400.0
        phase = "capturing" if t_f >= hill_t else "transfer"

        sc_xi = float(np.interp(t_f, t_nd, xi))
        sc_yi = float(np.interp(t_f, t_nd, yi))
        sc_zi = float(np.interp(t_f, t_nd, zi))

        m_arr = moon_em_to_eci(np.array([t_f]), R0_wsb)
        m_xi = float(m_arr[0][0]); m_yi = float(m_arr[1][0]); m_zi = float(m_arr[2][0])

        mask_pre  = t_nd <= min(t_f, hill_t)
        mask_post = (t_nd > hill_t) & (t_nd <= t_f)

        pre_xi = xi[mask_pre];  pre_yi = yi[mask_pre];  pre_zi = zi[mask_pre]
        cap_xi = xi[mask_post]; cap_yi = yi[mask_post]; cap_zi = zi[mask_post]

        if len(pre_xi) > MAX_TRAIL:
            idx = downsample(np.arange(len(pre_xi)), MAX_TRAIL)
            pre_xi, pre_yi, pre_zi = pre_xi[idx], pre_yi[idx], pre_zi[idx]
        if len(cap_xi) > MAX_TRAIL:
            idx = downsample(np.arange(len(cap_xi)), MAX_TRAIL)
            cap_xi, cap_yi, cap_zi = cap_xi[idx], cap_yi[idx], cap_zi[idx]

        pre_mx = xi_mc[mask_pre];  pre_my = yi_mc[mask_pre]
        cap_mx = xi_mc[mask_post]; cap_my = yi_mc[mask_post]
        sc_mc_x = float(np.interp(t_f, t_nd, xi_mc))
        sc_mc_y = float(np.interp(t_f, t_nd, yi_mc))
        if len(pre_mx) > MAX_TRAIL:
            idx = downsample(np.arange(len(pre_mx)), MAX_TRAIL)
            pre_mx, pre_my = pre_mx[idx], pre_my[idx]

        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=[sc_xi], y=[sc_yi], z=[sc_zi]),
                go.Scatter3d(x=[m_xi], y=[m_yi], z=[m_zi],
                             text=["Moon"], textposition="top center"),
                go.Scatter3d(x=pre_xi.tolist(), y=pre_yi.tolist(), z=pre_zi.tolist()),
                go.Scatter3d(x=cap_xi.tolist(), y=cap_yi.tolist(), z=cap_zi.tolist()),
                go.Scatter(x=pre_mx.tolist(), y=pre_my.tolist()),
                go.Scatter(x=cap_mx.tolist(), y=cap_my.tolist()),
                go.Scatter(x=[sc_mc_x], y=[sc_mc_y]),
            ],
            traces=ANIM_TRACES, name=str(fi),
            layout=go.Layout(title_text=(
                f"WSB Captured Solution — Day {day:.1f} / {t_total_d:.1f}  "
                f"[{phase}]  |  Hill entry: day {hill_d:.1f}  |  "
                f"{orbits:.2f} orbits<br><sup>{meta}</sup>"
            )),
        ))
    print(" done.")
    fig.frames = frames

    # ---- Slider ----
    label_every = max(1, int(5.0 / step_d))
    slider_steps = [
        dict(
            args=[[str(fi)],
                  dict(frame=dict(duration=0, redraw=True), mode="immediate",
                       transition=dict(duration=0))],
            label=f"{fi*step_d:.0f}d" if fi % label_every == 0 else "",
            method="animate",
        )
        for fi in range(n_frames)
    ]

    # ---- Layout ----
    xax, yax = dark_soi_layout(VIEW_SOI)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        scene=dict(
            bgcolor=BG_3D,
            xaxis=dark_3d_axis("X [km]", [-r_eci_lim, r_eci_lim]),
            yaxis=dark_3d_axis("Y [km]", [-r_eci_lim, r_eci_lim]),
            zaxis=dark_3d_axis("Z [km]", [-r_eci_lim * 0.4, r_eci_lim * 0.4]),
            camera=dict(eye=dict(x=0.9, y=-1.4, z=0.7)),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.4),
        ),
        xaxis=xax,
        yaxis=yax,
        plot_bgcolor=BG_3D,
        font=dict(color="white", size=11),
        title=dict(
            text=(
                f"WSB Captured Solution — Day 0 / {t_total_d:.1f}  [transfer]  |  "
                f"Hill entry: day {hill_d:.1f}  |  {orbits:.2f} orbits<br>"
                f"<sup>{meta}</sup>"
            ),
            x=0.01, xanchor="left", font=dict(size=12),
        ),
        legend=dict(x=0.01, y=0.98, font=dict(size=10),
                    bgcolor="rgba(15,15,25,0.7)"),
        height=720,
        margin=dict(l=0, r=0, t=95, b=80),
        updatemenus=[play_pause_buttons(frame_ms)],
        sliders=[dark_slider(slider_steps)],
    )

    fig.write_html(str(OUT_HTML), auto_play=False)
    print(f"\nSaved {OUT_HTML}")
    print("Open in browser — press Play or drag the slider.")


if __name__ == "__main__":
    main()
