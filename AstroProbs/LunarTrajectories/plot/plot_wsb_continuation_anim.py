"""
plot_wsb_continuation_anim.py — animated IC convergence in real ephemeris.

Each corrected IC (one per lambda step) is propagated at full lambda=1 forces.
The animation sweeps through time showing comet tails for all 5 ICs simultaneously,
so you can see which initial conditions actually reach the real Moon.

Left  : 3D ECI  — Earth sphere, real ANISE Moon orbit, comet trails.
Right : Real Moon-centred SOI — all trajectories relative to the ANISE Moon.

Reads : out/wsb/continuation_eval.csv
Saves : out/wsb/wsb_continuation_anim.html

CLI:
  --step HOURS   hours per animation frame (default 24)
  --tail DAYS    comet-tail length in days  (default 10)
  --fps  FPS     play speed in fps          (default 10)
  --maxpts N     max trail points per IC per frame (default 300)
"""
import argparse, pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wsb_style import (
    R_EARTH, R_MOON, R_HILL_KM,
    COL_EARTH, COL_MOON, COL_HILL, PAPER_BG, BG_3D,
    sphere_surface, downsample,
    dark_3d_axis, dark_soi_layout, play_pause_buttons, dark_slider,
)

ROOT    = pathlib.Path(__file__).parent.parent
OUT_DIR = ROOT / "out" / "wsb"
CSV     = OUT_DIR / "continuation_eval.csv"
OUT     = OUT_DIR / "wsb_continuation_anim.html"

LAM_COLORS = {
    0.00: "#00E5FF",
    0.25: "#4FC3F7",
    0.50: "#69FF47",
    0.75: "#FFD166",
    1.00: "#FF8C00",
}
LAM_LABELS = {
    0.00: "IC λ=0.00  (BCR4BP-equiv.)",
    0.25: "IC λ=0.25",
    0.50: "IC λ=0.50",
    0.75: "IC λ=0.75",
    1.00: "IC λ=1.00  (final solution)",
}

if not CSV.exists():
    print(f"Not found: {CSV}")
    print("Run: cargo run -p lunar_trajectories --bin wsb_continuation_corrected --release")
    raise SystemExit(1)

# ---- CLI ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--step",   type=float, default=24,  help="hours per animation frame")
parser.add_argument("--tail",   type=float, default=10,  help="comet-tail length in days")
parser.add_argument("--fps",    type=float, default=10,  help="frames per second")
parser.add_argument("--maxpts", type=int,   default=300, help="max trail pts per IC per frame")
args = parser.parse_args()

TAIL_S = args.tail * 86400.0

# ---- Load data ---------------------------------------------------------------
print(f"Loading {CSV.name} ...", flush=True)
df = pd.read_csv(CSV)
# Column is lambda_origin in eval CSV
df = df.rename(columns={"lambda_origin": "lambda"})
lambdas = sorted(df["lambda"].unique())
print(f"  lambdas: {lambdas}  |  {len(df)} rows")

# ---- Pre-compute per-IC arrays -----------------------------------------------
lam_data: dict = {}
for lam in lambdas:
    sub = df[df["lambda"] == lam].sort_values("t_s").reset_index(drop=True)
    lam_data[lam] = dict(
        t_s    = sub["t_s"].values.astype(np.float64),
        x_km   = sub["x_km"].values.astype(np.float64),
        y_km   = sub["y_km"].values.astype(np.float64),
        z_km   = sub["z_km"].values.astype(np.float64),
        moon_x = sub["moon_x_km"].values.astype(np.float64),
        moon_y = sub["moon_y_km"].values.astype(np.float64),
        moon_z = sub["moon_z_km"].values.astype(np.float64),
        r_moon = sub["r_moon_km"].values.astype(np.float64),
    )

# Reference Moon track: use lambda=1 (real ANISE) for ALL Moon-centred coords
# and for the animated Moon dot and orbit ring.
d_hi = lam_data[max(lambdas)]

# ---- Scene bounds ------------------------------------------------------------
_all_r = np.concatenate([
    np.sqrt(d["x_km"]**2 + d["y_km"]**2 + d["z_km"]**2)
    for d in lam_data.values()
])
r_eci_lim = float(np.percentile(_all_r, 95)) * 1.25
r_eci_lim = max(r_eci_lim, R_HILL_KM * 2.0)
VIEW_SOI  = R_HILL_KM * 1.8

# ---- Animation time grid -----------------------------------------------------
t_max_s = max(d["t_s"][-1] for d in lam_data.values())
dt_s    = args.step * 3600.0
t_anim  = np.arange(0, t_max_s + dt_s * 0.5, dt_s)
print(f"  {len(t_anim)} frames  (step={args.step:.0f} h, tail={args.tail:.0f} d)")

# ---- Static geometry ---------------------------------------------------------
sx, sy, sz = sphere_surface(R_EARTH, nu=30, nv=16)

# Real ANISE Moon orbit ring — trace the actual Moon positions from the λ=1 data.
# This is an ellipse in 3D, inclined to the equator, matching the animated Moon dot.
mxr = d_hi["moon_x"].tolist()
myr = d_hi["moon_y"].tolist()
mzr = d_hi["moon_z"].tolist()

phi   = np.linspace(0, 2 * np.pi, 200)
hs_x  = (R_HILL_KM * np.cos(phi)).tolist()
hs_y  = (R_HILL_KM * np.sin(phi)).tolist()
md_x  = (R_MOON * 5 * np.cos(phi)).tolist()
md_y  = (R_MOON * 5 * np.sin(phi)).tolist()
md_x_true = (R_MOON * np.cos(phi)).tolist()
md_y_true = (R_MOON * np.sin(phi)).tolist()

N = len(lambdas)

# ---- Figure ------------------------------------------------------------------
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "xy"}]],
    subplot_titles=[
        "ECI — same forces (λ=1 real ephemeris), ICs from each correction step",
        "Real Moon-Centred — IC convergence (ANISE Moon reference)",
    ],
    column_widths=[0.56, 0.44],
    horizontal_spacing=0.06,
)

# Static LEFT: Earth + real Moon orbit ring
fig.add_trace(go.Surface(
    x=sx, y=sy, z=sz,
    colorscale=[[0, "#1A237E"], [0.5, COL_EARTH], [1, "#64B5F6"]],
    showscale=False, opacity=0.9,
    lighting=dict(ambient=0.6, diffuse=0.8, specular=0.4),
    name="Earth", hoverinfo="skip",
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=mxr, y=myr, z=mzr, mode="lines",
    line=dict(color="rgba(180,180,180,0.35)", width=2),
    name="Moon orbit (ANISE)", showlegend=True, hoverinfo="skip",
), row=1, col=1)

# Static RIGHT: Hill sphere + Moon discs
fig.add_trace(go.Scatter(
    x=hs_x, y=hs_y, mode="lines",
    line=dict(color=COL_HILL, width=1.5, dash="dot"),
    name=f"Hill sphere ({R_HILL_KM/1e3:.0f}e3 km)", showlegend=True,
), row=1, col=2)
fig.add_trace(go.Scatter(
    x=md_x, y=md_y, mode="lines",
    fill="toself", fillcolor="rgba(160,160,160,0.30)",
    line=dict(color="#AAAAAA", width=1),
    name="Moon (5× visual)", showlegend=True,
), row=1, col=2)
fig.add_trace(go.Scatter(
    x=md_x_true, y=md_y_true, mode="lines",
    line=dict(color="#FFFFFF", width=1, dash="dot"),
    name=f"Moon surface ({R_MOON:.0f} km)", showlegend=True,
), row=1, col=2)

# Animated: Moon dot (3D) + N×(trail 3D, dot 3D) + N×trail 2D
n0 = len(fig.data)

fig.add_trace(go.Scatter3d(
    x=[float(d_hi["moon_x"][0])],
    y=[float(d_hi["moon_y"][0])],
    z=[float(d_hi["moon_z"][0])],
    mode="markers+text",
    marker=dict(color=COL_MOON, size=7),
    text=["Moon"], textposition="top center",
    textfont=dict(color=COL_MOON, size=9),
    name="Moon", showlegend=False,
), row=1, col=1)

for lam in lambdas:
    col = LAM_COLORS.get(lam, "#888888")
    lbl = LAM_LABELS.get(lam, f"IC λ={lam:.2f}")
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="lines",
        line=dict(color=col, width=3), opacity=0.9,
        name=lbl, legendgroup=f"lam{lam}", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="markers",
        marker=dict(color=col, size=6),
        legendgroup=f"lam{lam}", showlegend=False,
    ), row=1, col=1)

for lam in lambdas:
    col = LAM_COLORS.get(lam, "#888888")
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines",
        line=dict(color=col, width=2), opacity=0.9,
        legendgroup=f"lam{lam}", showlegend=False,
    ), row=1, col=2)

# ANIM_TRACES: Moon dot + N×(trail3d + dot3d) + N×trail2d
ANIM_TRACES = list(range(n0, n0 + 1 + 3 * N))

# ---- Build frames ------------------------------------------------------------
print(f"Building {len(t_anim)} frames ...", end="", flush=True)
frames       = []
slider_steps = []

for fi, t_now in enumerate(t_anim):
    if fi % 50 == 0:
        print(".", end="", flush=True)
    t_days = t_now / 86400.0

    # Real Moon position at t_now (interpolated from λ=1 ANISE track)
    m_xi = float(np.interp(t_now, d_hi["t_s"], d_hi["moon_x"]))
    m_yi = float(np.interp(t_now, d_hi["t_s"], d_hi["moon_y"]))
    m_zi = float(np.interp(t_now, d_hi["t_s"], d_hi["moon_z"]))

    data: list = [go.Scatter3d(x=[m_xi], y=[m_yi], z=[m_zi],
                               text=["Moon"], textposition="top center")]

    for lam in lambdas:
        s    = lam_data[lam]
        mask = (s["t_s"] > t_now - TAIL_S) & (s["t_s"] <= t_now)

        tx = s["x_km"][mask]; ty = s["y_km"][mask]; tz = s["z_km"][mask]
        if len(tx) > args.maxpts:
            idx = downsample(np.arange(len(tx)), args.maxpts)
            tx, ty, tz = tx[idx], ty[idx], tz[idx]

        data.append(go.Scatter3d(x=tx.tolist(), y=ty.tolist(), z=tz.tolist()))
        if mask.any():
            data.append(go.Scatter3d(
                x=[float(s["x_km"][mask][-1])],
                y=[float(s["y_km"][mask][-1])],
                z=[float(s["z_km"][mask][-1])],
            ))
        else:
            data.append(go.Scatter3d(x=[], y=[], z=[]))

    for lam in lambdas:
        s    = lam_data[lam]
        mask = (s["t_s"] > t_now - TAIL_S) & (s["t_s"] <= t_now)
        # Moon-centred using real ANISE Moon — interpolate at each masked t
        t_masked   = s["t_s"][mask]
        moon_x_ref = np.interp(t_masked, d_hi["t_s"], d_hi["moon_x"])
        moon_y_ref = np.interp(t_masked, d_hi["t_s"], d_hi["moon_y"])
        mc_x = s["x_km"][mask] - moon_x_ref
        mc_y = s["y_km"][mask] - moon_y_ref
        if len(mc_x) > args.maxpts:
            idx  = downsample(np.arange(len(mc_x)), args.maxpts)
            mc_x, mc_y = mc_x[idx], mc_y[idx]
        data.append(go.Scatter(x=mc_x.tolist(), y=mc_y.tolist()))

    frames.append(go.Frame(
        data=data, traces=ANIM_TRACES, name=str(fi),
        layout=go.Layout(title_text=(
            f"IC convergence — all propagated at λ=1  ·  t = {t_days:.1f} days"
        )),
    ))
    slider_steps.append(dict(
        args=[[str(fi)],
              {"frame": {"duration": 0}, "mode": "immediate",
               "transition": {"duration": 0}}],
        label=f"{t_days:.0f}d",
        method="animate",
    ))

print(" done.")
fig.frames = frames

# ---- Layout ------------------------------------------------------------------
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
            "WSB IC convergence  ·  Each colour = corrected IC from that λ step, "
            "all propagated at λ=1 (full real ephemeris)<br>"
            "<sup>Left: ECI 3D  |  Right: real Moon-centred (ANISE reference)  |  "
            "Orange = final solution</sup>"
        ),
        x=0.01, xanchor="left", font=dict(size=12),
    ),
    legend=dict(x=0.01, y=0.98, font=dict(size=10),
                bgcolor="rgba(15,15,25,0.7)"),
    height=720,
    margin=dict(l=0, r=0, t=95, b=80),
    updatemenus=[play_pause_buttons(int(1000 / args.fps))],
    sliders=[dark_slider(slider_steps)],
)

fig.write_html(str(OUT), auto_play=False)
print(f"Saved {OUT}")
