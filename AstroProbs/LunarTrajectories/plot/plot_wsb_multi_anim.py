"""
plot_wsb_multi_anim.py — animated 3D ECI + Moon-centred SOI view of the N fastest MC solutions.

Left  : 3D ECI — Earth sphere, Moon orbit ring, N solution trajectories (each its own colour).
Right : Moon-centred EM-inertial top-down view — Hill sphere approach for each solution.

Each trajectory is clipped at its own Hill sphere entry.  Solutions are ordered
fastest-first (shortest transfer time to Hill entry).

Reads  : out/wsb/mc_traj.csv
         out/wsb/mc_solutions.csv
         out/wsb/epoch_info.txt   (optional — for ECI tilt; R0=I if missing)
Saves  : out/wsb/wsb_multi_anim.html

CLI options:
  --n    N      max solutions to animate  (default 5)
  --fps  FPS    frames per second         (default 10)
  --step HOURS  time advance per frame    (default 24 h)
  --maxpts N    max trajectory points per solution for downsampling (default 300)
"""
import argparse, pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wsb_style import (
    MU, X_M, L_KM, T_STAR, R_EARTH, R_MOON, R_HILL_ND, R_HILL_KM,
    COL_EARTH, COL_MOON, COL_HILL,
    MULTI_COLORS, PAPER_BG, BG_3D,
    rot_em_to_eci, moon_em_to_eci, load_epoch_info, sphere_surface, downsample,
    dark_3d_axis, dark_soi_layout, play_pause_buttons, dark_slider,
)

ROOT      = pathlib.Path(__file__).parent.parent
OUT_DIR   = ROOT / "out" / "wsb"
TRAJ_CSV  = OUT_DIR / "mc_traj.csv"
SOL_CSV   = OUT_DIR / "mc_solutions.csv"
EPOCH_TXT = OUT_DIR / "epoch_info.txt"
OUT_HTML  = OUT_DIR / "wsb_multi_anim.html"

MAX_TRAIL = 600

if not TRAJ_CSV.exists():
    print(f"Not found: {TRAJ_CSV}")
    print("Run: cargo run -p lunar_trajectories --bin wsb_refine --release")
    raise SystemExit(1)

# ---- CLI ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n",      type=int,   default=5,   help="max solutions to show")
parser.add_argument("--fps",    type=float, default=10,  help="frames per second")
parser.add_argument("--step",   type=float, default=24,  help="hours per frame")
parser.add_argument("--maxpts", type=int,   default=300, help="max traj points per solution")
parser.add_argument("--video",  action="store_true",     help="also export GIF via matplotlib")
args = parser.parse_args()

# ---- ECI frame ---------------------------------------------------------------
if EPOCH_TXT.exists():
    _, R0_wsb = load_epoch_info(EPOCH_TXT)
    print("  ECI frame from epoch_info.txt")
else:
    R0_wsb = np.eye(3)
    print("  WARNING: epoch_info.txt not found — using EM-inertial (R0=I)")

# ---- Load trajectory data ----------------------------------------------------
df_raw  = pd.read_csv(TRAJ_CSV)
df_traj = df_raw[df_raw["x_nd"].notna()].copy()
sol_df  = pd.read_csv(SOL_CSV) if SOL_CSV.exists() else pd.DataFrame()

ranks = sorted(df_traj["rank"].unique())
print(f"Loaded {len(ranks)} solutions from {TRAJ_CSV.name}")

# ---- Pre-compute per-solution ECI and SOI data --------------------------------
solutions: dict = {}
for rank in ranks:
    sub = (df_traj[df_traj["rank"] == rank]
           .sort_values("time_nd").reset_index(drop=True))
    t = sub["time_nd"].values
    x = sub["x_nd"].values
    y = sub["y_nd"].values
    z = np.zeros_like(t)

    r_moon  = np.sqrt((x - X_M)**2 + y**2)
    in_hill = r_moon < R_HILL_ND
    t_hill  = float(t[in_hill][0]) if in_hill.any() else float(t[-1])

    # Clip at Hill entry, then downsample
    clip = t <= t_hill
    t_c, x_c, y_c, z_c = t[clip], x[clip], y[clip], z[clip]
    if len(t_c) > args.maxpts:
        idx = np.linspace(0, len(t_c) - 1, args.maxpts, dtype=int)
        t_c, x_c, y_c, z_c = t_c[idx], x_c[idx], y_c[idx], z_c[idx]

    # ECI (3D left panel)
    xi3, yi3, zi3 = rot_em_to_eci(x_c, y_c, z_c, t_c, R0_wsb)

    # Moon-centred EM-inertial (2D right panel)
    xi_em = (x_c * np.cos(t_c) - y_c * np.sin(t_c)) * L_KM
    yi_em = (x_c * np.sin(t_c) + y_c * np.cos(t_c)) * L_KM
    xi_mc = xi_em - X_M * np.cos(t_c) * L_KM
    yi_mc = yi_em - X_M * np.sin(t_c) * L_KM

    solutions[rank] = dict(
        t=t_c,
        xi=xi3, yi=yi3, zi=zi3,
        xi_mc=xi_mc, yi_mc=yi_mc,
        t_hill=t_hill,
        t_days=t_hill * T_STAR / 86400.0,
        dv_total=float(sub["dv_total_kms"].iloc[0]),
    )

ranked  = sorted(solutions.items(), key=lambda kv: kv[1]["t_hill"])[:args.n]
sol_ids = [rank for rank, _ in ranked]
print(f"  Top {len(sol_ids)} fastest solutions (Hill entry time):")
for rank, s in ranked:
    print(f"    rank {rank:2d}: t_hill={s['t_days']:.1f} d  DV={s['dv_total']:.4f} km/s")

t_global_max = max(s["t_hill"] for _, s in ranked)
dt_nd  = args.step / (T_STAR / 3600.0)
t_anim = np.arange(0, t_global_max + dt_nd * 0.5, dt_nd)
print(f"  Animation: {len(t_anim)} frames  (step={args.step:.0f} h  fps={args.fps})")

N = len(sol_ids)
r_eci_lim = max(
    max(np.max(np.abs(solutions[r]["xi"])) for r in sol_ids),
    max(np.max(np.abs(solutions[r]["yi"])) for r in sol_ids),
    max(np.max(np.abs(solutions[r]["zi"])) for r in sol_ids),
) * 1.2
r_eci_lim = max(r_eci_lim, R_HILL_KM * 2.0)
VIEW_SOI  = R_HILL_KM * 2.8

# ---- Static geometry ---------------------------------------------------------
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
    subplot_titles=[f"ECI Frame — {N} Fastest MC Solutions",
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

# Animated traces: Moon dot, N 3D trails, N SOI trails
n0 = len(fig.data)

mxi0, myi0, mzi0 = moon_em_to_eci(np.array([0.0]), R0_wsb)
fig.add_trace(go.Scatter3d(
    x=[float(mxi0[0])], y=[float(myi0[0])], z=[float(mzi0[0])],
    mode="markers+text", marker=dict(color=COL_MOON, size=7),
    text=["Moon"], textposition="top center",
    textfont=dict(color=COL_MOON, size=9),
    name="Moon", showlegend=False,
), row=1, col=1)

for k, rank in enumerate(sol_ids):
    s   = solutions[rank]
    col = MULTI_COLORS[k % len(MULTI_COLORS)]
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="lines",
        line=dict(color=col, width=3), opacity=0.9,
        name=f"#{rank} DV={s['dv_total']:.3f} t={s['t_days']:.0f}d",
        legendgroup=str(rank), showlegend=True,
    ), row=1, col=1)

for k, rank in enumerate(sol_ids):
    col = MULTI_COLORS[k % len(MULTI_COLORS)]
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines",
        line=dict(color=col, width=2), opacity=0.9,
        legendgroup=str(rank), showlegend=False,
    ), row=1, col=2)

# ANIM_TRACES: Moon dot + N 3D trails + N SOI trails
ANIM_TRACES = list(range(n0, n0 + 1 + 2 * N))

# ---- Build frames ------------------------------------------------------------
print(f"Building {len(t_anim)} frames ...", end="", flush=True)
frames = []
slider_steps = []

for fi, t_now in enumerate(t_anim):
    if fi % 50 == 0:
        print(".", end="", flush=True)
    t_days = t_now * T_STAR / 86400.0

    m_arr = moon_em_to_eci(np.array([t_now]), R0_wsb)
    m_xi = float(m_arr[0][0]); m_yi = float(m_arr[1][0]); m_zi = float(m_arr[2][0])

    data: list = [go.Scatter3d(x=[m_xi], y=[m_yi], z=[m_zi],
                               text=["Moon"], textposition="top center")]

    for rank in sol_ids:
        s    = solutions[rank]
        mask = s["t"] <= t_now
        tx = s["xi"][mask]; ty = s["yi"][mask]; tz = s["zi"][mask]
        if len(tx) > MAX_TRAIL:
            idx = downsample(np.arange(len(tx)), MAX_TRAIL)
            tx, ty, tz = tx[idx], ty[idx], tz[idx]
        data.append(go.Scatter3d(x=tx.tolist(), y=ty.tolist(), z=tz.tolist()))

    for rank in sol_ids:
        s    = solutions[rank]
        mask = s["t"] <= t_now
        mx = s["xi_mc"][mask]; my = s["yi_mc"][mask]
        if len(mx) > MAX_TRAIL:
            idx = downsample(np.arange(len(mx)), MAX_TRAIL)
            mx, my = mx[idx], my[idx]
        data.append(go.Scatter(x=mx.tolist(), y=my.tolist()))

    frames.append(go.Frame(
        data=data, traces=ANIM_TRACES, name=str(fi),
        layout=go.Layout(title_text=(
            f"WSB MC Solutions — t = {t_days:.1f} days  "
            f"({N} fastest, sorted by transfer time)"
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
            f"WSB MC Solutions — {N} fastest, clipped at Hill sphere entry<br>"
            "<sup>Left: ECI frame (3D)  |  Right: Moon-centred EM-inertial  |  "
            "Sorted by transfer time (fastest first)</sup>"
        ),
        x=0.01, xanchor="left", font=dict(size=13),
    ),
    legend=dict(x=0.01, y=0.98, font=dict(size=10),
                bgcolor="rgba(15,15,25,0.7)"),
    height=720,
    margin=dict(l=0, r=0, t=95, b=80),
    updatemenus=[play_pause_buttons(int(1000 / args.fps))],
    sliders=[dark_slider(slider_steps)],
)

fig.write_html(str(OUT_HTML), auto_play=False)
print(f"Saved {OUT_HTML}")

# ---- Optional GIF export via matplotlib --------------------------------------
if args.video:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation, PillowWriter

    GIF_PATH = OUT_HTML.with_suffix(".gif")
    MCOLORS_MPL = [
        "#00E5FF", "#FF8C00", "#69FF47", "#FF6B6B", "#C77DFF",
        "#4FC3F7", "#FFD166", "#06D6A0", "#EF476F", "#118AB2",
    ]
    BG = "#0F0F19"
    lim_eci = r_eci_lim / 1e3   # convert to x10^3 km for labels

    fig_v, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 7),
                                        facecolor=BG)
    for ax in (ax_l, ax_r):
        ax.set_facecolor(BG)
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor((80/255, 80/255, 160/255, 0.4))

    ax_l.set_xlim(-lim_eci, lim_eci)
    ax_l.set_ylim(-lim_eci, lim_eci)
    ax_l.set_aspect("equal")
    ax_l.set_xlabel("X [×10³ km]", color="#aaaaaa")
    ax_l.set_ylabel("Y [×10³ km]", color="#aaaaaa")
    ax_l.set_title("ECI frame (XY proj.)", color="white", fontsize=10)

    soi_km = R_HILL_KM / 1e3
    ax_r.set_xlim(-soi_km * 2.8, soi_km * 2.8)
    ax_r.set_ylim(-soi_km * 2.8, soi_km * 2.8)
    ax_r.set_aspect("equal")
    ax_r.set_xlabel("X from Moon [×10³ km]", color="#aaaaaa")
    ax_r.set_ylabel("Y from Moon [×10³ km]", color="#aaaaaa")
    ax_r.set_title("Moon-centred SOI", color="white", fontsize=10)

    # Static: Earth circle, Moon orbit ring, Hill sphere, Moon disc
    earth_c = plt.Circle((0, 0), 6_371 / 1e3, color="#1565C0", zorder=3)
    ax_l.add_patch(earth_c)
    moon_ring_t = np.linspace(0, 2 * np.pi, 200)
    ax_l.plot(X_M * L_KM * np.cos(moon_ring_t) / 1e3,
              X_M * L_KM * np.sin(moon_ring_t) / 1e3,
              color=(180/255, 180/255, 180/255, 0.2), lw=0.8)

    hs_circ = plt.Circle((0, 0), soi_km, fill=False,
                          edgecolor="#BA7517", linestyle="--", lw=1.2)
    ax_r.add_patch(hs_circ)
    moon_disc = plt.Circle((0, 0), R_MOON * 5 / 1e3,
                            color=(160/255, 160/255, 160/255, 0.35), zorder=3)
    ax_r.add_patch(moon_disc)

    moon_dot_l, = ax_l.plot([], [], "o", color="#9E9E9E", ms=6, zorder=5)
    trail_lines_l = [ax_l.plot([], [], lw=1.5, color=MCOLORS_MPL[k])[0]
                     for k in range(N)]
    trail_lines_r = [ax_r.plot([], [], lw=1.5, color=MCOLORS_MPL[k])[0]
                     for k in range(N)]
    time_txt = fig_v.text(0.5, 0.97, "", ha="center", va="top",
                          color="white", fontsize=11)

    legend_patches = [
        mpatches.Patch(color=MCOLORS_MPL[k],
                       label=f"#{sol_ids[k]} DV={solutions[sol_ids[k]]['dv_total']:.3f} "
                             f"t={solutions[sol_ids[k]]['t_days']:.0f}d")
        for k in range(N)
    ]
    fig_v.legend(handles=legend_patches, loc="lower center", ncol=N,
                 fontsize=8, facecolor=BG, labelcolor="white",
                 framealpha=0.7)

    def _update(fi):
        t_now = t_anim[fi]
        t_days = t_now * T_STAR / 86400.0
        m_arr = moon_em_to_eci(np.array([t_now]), R0_wsb)
        moon_dot_l.set_data([float(m_arr[0][0]) / 1e3], [float(m_arr[1][0]) / 1e3])
        for k, rank in enumerate(sol_ids):
            s = solutions[rank]
            mask = s["t"] <= t_now
            tx = s["xi"][mask] / 1e3; ty = s["yi"][mask] / 1e3
            if len(tx) > MAX_TRAIL:
                idx = downsample(np.arange(len(tx)), MAX_TRAIL)
                tx, ty = tx[idx], ty[idx]
            trail_lines_l[k].set_data(tx, ty)
            mx = s["xi_mc"][mask] / 1e3; my = s["yi_mc"][mask] / 1e3
            if len(mx) > MAX_TRAIL:
                idx = downsample(np.arange(len(mx)), MAX_TRAIL)
                mx, my = mx[idx], my[idx]
            trail_lines_r[k].set_data(mx, my)
        time_txt.set_text(f"t = {t_days:.1f} days")
        return [moon_dot_l] + trail_lines_l + trail_lines_r + [time_txt]

    # Subsample frames for GIF to keep file manageable
    gif_step = max(1, len(t_anim) // 120)
    gif_frames = list(range(0, len(t_anim), gif_step))
    print(f"Exporting GIF: {len(gif_frames)} frames at {args.fps:.0f} fps ...",
          end="", flush=True)
    anim = FuncAnimation(fig_v, _update, frames=gif_frames, blit=True)
    anim.save(str(GIF_PATH), writer=PillowWriter(fps=int(args.fps)),
              dpi=120)
    plt.close(fig_v)
    print(f" done.\nSaved {GIF_PATH}")
