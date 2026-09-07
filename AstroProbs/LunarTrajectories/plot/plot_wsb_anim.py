"""
plot_wsb_anim.py — animated 3D ECI + Moon-centred SOI view of the best BLT search candidate.

Left  : 3D ECI — Earth sphere, Moon orbit ring, full transfer arc.
Right : Moon-centred EM-inertial top-down view — Hill sphere approach.

This is the wsb_search candidate (before refinement / maxhifi).
For the high-fidelity refined solution see plot_wsb_solution_anim.py.

Run:  python plot/plot_wsb_anim.py
Reads: out/wsb/blt_candidates.csv
       out/wsb/blt_info.txt
       out/wsb/epoch_info.txt   (optional — for ECI tilt; R0=I if missing)
Saves: out/wsb/wsb_anim.html
"""
import re, pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wsb_style import (
    MU, X_M, L_KM, T_STAR, R_EARTH, R_MOON, R_HILL_ND, R_HILL_KM,
    COL_WSB, COL_EARTH, COL_MOON, COL_HILL,
    PAPER_BG, BG_3D,
    rot_em_to_eci, moon_em_to_eci, load_epoch_info, sphere_surface, downsample,
    dark_3d_axis, dark_soi_layout, play_pause_buttons, dark_slider,
)

ROOT      = pathlib.Path(__file__).parent.parent
OUT       = ROOT / "out" / "wsb"
EPOCH_TXT = OUT / "epoch_info.txt"
OUT_HTML  = OUT / "wsb_anim.html"

DAY_ND    = 86400.0 / T_STAR
MAX_TRAIL = 800

# ---- Load data ---------------------------------------------------------------
cand_df = pd.read_csv(OUT / "blt_candidates.csv")

theta_sun_map: dict[int, float] = {}
info_path = OUT / "blt_info.txt"
if info_path.exists():
    parts = re.split(r"--- Candidate (\d+) ---",
                     info_path.read_text(encoding="utf-8", errors="replace"))
    for i in range(1, len(parts), 2):
        cid = int(parts[i])
        m   = re.search(r"theta_sun[^:]*:\s*([\d.]+)", parts[i + 1], re.IGNORECASE)
        if not m:
            m = re.search(r"[θΘ]_sun[^:]*:\s*([\d.]+)", parts[i + 1])
        if m:
            theta_sun_map[cid] = float(m.group(1))

cid  = 1
sub  = cand_df[cand_df["cand_id"] == cid].dropna(subset=["x_nd", "y_nd"])
theta_sun_deg = theta_sun_map.get(cid, 0.0)

t_raw = sub["time_nd"].values
x_raw = sub["x_nd"].values
y_raw = sub["y_nd"].values

# Clip at Hill sphere entry
r_moon_raw = np.sqrt((x_raw - X_M)**2 + y_raw**2)
in_hill    = r_moon_raw < R_HILL_ND
t_hill     = float(t_raw[in_hill][0]) if in_hill.any() else t_raw[-1]
keep       = t_raw <= t_hill
t_nd = t_raw[keep]; x_nd = x_raw[keep]; y_nd = y_raw[keep]
z_nd = np.zeros_like(t_nd)

hill_d  = t_hill * T_STAR / 86400.0
t_max   = t_nd[-1]
t_total = t_max * T_STAR / 86400.0
print(f"Best candidate: {t_total:.0f} d to SOI  theta_sun={theta_sun_deg:.1f} deg")

# ---- ECI frame ---------------------------------------------------------------
if EPOCH_TXT.exists():
    _, R0_wsb = load_epoch_info(EPOCH_TXT)
    print("  ECI frame from epoch_info.txt")
else:
    R0_wsb = np.eye(3)
    print("  WARNING: epoch_info.txt not found — using EM-inertial (R0=I)")

xi, yi, zi = rot_em_to_eci(x_nd, y_nd, z_nd, t_nd, R0_wsb)
mxi_full, myi_full, mzi_full = moon_em_to_eci(t_nd, R0_wsb)

# ---- SOI panel (Moon-centred EM-inertial) ------------------------------------
xi_em  = (x_nd * np.cos(t_nd) - y_nd * np.sin(t_nd)) * L_KM
yi_em  = (x_nd * np.sin(t_nd) + y_nd * np.cos(t_nd)) * L_KM
mxi_em = X_M * np.cos(t_nd) * L_KM
myi_em = X_M * np.sin(t_nd) * L_KM
xi_mc  = xi_em - mxi_em
yi_mc  = yi_em - myi_em

# ---- Scene limits (dynamic — fit actual trajectory extent) ------------------
r_eci_lim = max(np.max(np.abs(xi)), np.max(np.abs(yi)), np.max(np.abs(zi))) * 1.2
r_eci_lim = max(r_eci_lim, R_HILL_KM * 2.0)
VIEW_SOI  = R_HILL_KM * 2.8

# ---- Animation frame grid (every 1 day) --------------------------------------
t_frames = np.arange(0.0, t_max + DAY_ND, DAY_ND)
n_frames = len(t_frames)
print(f"Building {n_frames} frames ...", end="", flush=True)

# ---- Static geometry --------------------------------------------------------
sx, sy, sz = sphere_surface(R_EARTH, nu=30, nv=16)
t_ring = np.linspace(0, 2 * np.pi, 200)
mxr, myr, mzr = moon_em_to_eci(t_ring, R0_wsb)

phi  = np.linspace(0, 2 * np.pi, 200)
hs_x = (R_HILL_KM * np.cos(phi)).tolist()
hs_y = (R_HILL_KM * np.sin(phi)).tolist()
md_x = (R_MOON * 5 * np.cos(phi)).tolist()
md_y = (R_MOON * 5 * np.sin(phi)).tolist()

# ---- Figure ------------------------------------------------------------------
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "xy"}]],
    subplot_titles=["ECI Frame — BLT Search Candidate",
                    "Moon-Centred — Hill Sphere Approach"],
    column_widths=[0.56, 0.44],
    horizontal_spacing=0.06,
)

# Static LEFT (3D ECI)
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
# Static RIGHT (2D Moon-centred)
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

# Animated traces
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
    mode="markers+text", marker=dict(color=COL_MOON, size=7),
    text=["Moon"], textposition="top center",
    textfont=dict(color=COL_MOON, size=9),
    name="Moon", showlegend=False,
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=[], y=[], z=[], mode="lines",
    line=dict(color=COL_WSB, width=4),
    name="Trail", showlegend=True,
), row=1, col=1)

n0_soi = len(fig.data)
fig.add_trace(go.Scatter(
    x=[], y=[], mode="lines",
    line=dict(color=COL_WSB, width=2.5),
    name="Trail SOI", showlegend=False,
), row=1, col=2)
fig.add_trace(go.Scatter(
    x=[float(xi_mc[0])], y=[float(yi_mc[0])],
    mode="markers",
    marker=dict(color=COL_WSB, size=8, symbol="circle",
                line=dict(color="white", width=0.8)),
    name="SC SOI", showlegend=False,
), row=1, col=2)

ANIM_TRACES = [n0_eci, n0_eci+1, n0_eci+2, n0_soi, n0_soi+1]

# ---- Build frames ------------------------------------------------------------
frames = []
for fi, t_f in enumerate(t_frames):
    if fi % 50 == 0:
        print(".", end="", flush=True)
    day = t_f * T_STAR / 86400.0

    sc_xi = float(np.interp(t_f, t_nd, xi))
    sc_yi = float(np.interp(t_f, t_nd, yi))
    sc_zi = float(np.interp(t_f, t_nd, zi))

    m_arr = moon_em_to_eci(np.array([t_f]), R0_wsb)
    m_xi = float(m_arr[0][0]); m_yi = float(m_arr[1][0]); m_zi = float(m_arr[2][0])

    mask = t_nd <= t_f
    tr_xi = xi[mask]; tr_yi = yi[mask]; tr_zi = zi[mask]
    if len(tr_xi) > MAX_TRAIL:
        idx = downsample(np.arange(len(tr_xi)), MAX_TRAIL)
        tr_xi, tr_yi, tr_zi = tr_xi[idx], tr_yi[idx], tr_zi[idx]

    tr_mx = xi_mc[mask]; tr_my = yi_mc[mask]
    if len(tr_mx) > MAX_TRAIL:
        idx = downsample(np.arange(len(tr_mx)), MAX_TRAIL)
        tr_mx, tr_my = tr_mx[idx], tr_my[idx]
    sc_mc_x = float(np.interp(t_f, t_nd, xi_mc))
    sc_mc_y = float(np.interp(t_f, t_nd, yi_mc))

    frames.append(go.Frame(
        data=[
            go.Scatter3d(x=[sc_xi], y=[sc_yi], z=[sc_zi]),
            go.Scatter3d(x=[m_xi], y=[m_yi], z=[m_zi],
                         text=["Moon"], textposition="top center"),
            go.Scatter3d(x=tr_xi.tolist(), y=tr_yi.tolist(), z=tr_zi.tolist()),
            go.Scatter(x=tr_mx.tolist(), y=tr_my.tolist()),
            go.Scatter(x=[sc_mc_x], y=[sc_mc_y]),
        ],
        traces=ANIM_TRACES, name=str(fi),
        layout=go.Layout(title_text=(
            f"WSB BLT Search Candidate — Day {day:.0f} / {t_total:.0f}  "
            f"(Hill entry: day {hill_d:.0f})"
        )),
    ))
print(" done.")
fig.frames = frames

slider_steps = [
    dict(
        args=[[str(fi)],
              dict(frame=dict(duration=0, redraw=True), mode="immediate",
                   transition=dict(duration=0))],
        label=f"{fi:.0f}d" if fi % 10 == 0 else "",
        method="animate",
    )
    for fi in range(n_frames)
]

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
            f"WSB BLT Search Candidate — Day 0 / {t_total:.0f}  "
            f"(pre-refinement)  |  Hill entry: day {hill_d:.0f}<br>"
            f"<sup>theta_sun={theta_sun_deg:.1f} deg  — "
            f"For maxhifi solution see wsb_solution_anim.html</sup>"
        ),
        x=0.01, xanchor="left", font=dict(size=12),
    ),
    legend=dict(x=0.01, y=0.98, font=dict(size=10),
                bgcolor="rgba(15,15,25,0.7)"),
    height=720,
    margin=dict(l=0, r=0, t=95, b=80),
    updatemenus=[play_pause_buttons(60)],
    sliders=[dark_slider(slider_steps)],
)

fig.write_html(str(OUT_HTML), auto_play=False)
print(f"\nSaved {OUT_HTML}")
