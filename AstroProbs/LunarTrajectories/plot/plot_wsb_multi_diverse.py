"""
plot_wsb_multi_diverse.py — animated 3D ECI + Moon-centred SOI view of
N diverse WSB solutions spanning different synodic-period windows.

Diversity is defined by theta_sun_deg: one best-capture solution per 30° bin.

NOTE — elapsed-day counter (not absolute datetime):
  Each diverse solution has a different theta_sun_deg and therefore a different
  real departure epoch (they span different synodic windows). Displaying a single
  absolute datetime would only be accurate for one solution and misleading for all
  others. "Day X.X" is elapsed simulation time from injection and is the same for
  every solution in the ensemble, making it the honest choice here.
Reads  : out/wsb/dense_traj.csv  (preferred — adaptive-step dense log)
         out/wsb/all_refinements.csv  (fallback)
         out/wsb/epoch_info.txt  (optional — for ECI tilt; R0=I if missing)
Saves  : out/wsb/wsb_multi_diverse.html
         out/wsb/wsb_multi_diverse.gif  (only with --video, via PyVista)
         out/wsb/wsb_multi_diverse.mp4  (only with --video, requires ffmpeg)

CLI options:
  --n      N      max diverse solutions (default 10)
  --fps    FPS    frames per second     (default 10)
  --step   HOURS  time advance per frame (default 24 h)
  --stride N      use every N-th CSV row (default 1 = all points)
  --tail    DAYS  length of comet tail in days (default 15)
  --video         also export GIF + MP4 via PyVista
  --compact       SOI panel as inset in bottom-right of ECI panel (phone-friendly)
"""
import argparse, pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wsb_style import (
    MU, X_M, L_KM, T_STAR, R_EARTH, R_MOON, R_HILL_ND, R_HILL_KM,
    COL_EARTH, COL_MOON, COL_HILL,
    PAPER_BG, BG_3D,
    rot_em_to_eci, moon_em_to_eci, load_epoch_info, sphere_surface,
    dark_3d_axis, dark_soi_layout, play_pause_buttons, dark_slider,
)


ROOT       = pathlib.Path(__file__).parent.parent
OUT_DIR    = ROOT / "out" / "wsb"
# Prefer the densely-logged re-propagation (wsb_dense_traj) when available;
# fall back to all_refinements.csv from the MC search (coarser timesteps).
_DENSE_CSV  = OUT_DIR / "dense_traj.csv"
REFINE_CSV  = _DENSE_CSV if _DENSE_CSV.exists() else OUT_DIR / "all_refinements.csv"
EPOCH_TXT  = OUT_DIR / "epoch_info.txt"
OUT_HTML   = OUT_DIR / "wsb_multi_diverse.html"

COL_TRAJ = "rgba(20, 70, 170, 0.92)"     # dark navy-blue trajectory line
COL_DOT  = "rgba(0, 229, 255, 0.95)"    # cyan s/c dot (matches WSB palette)

if not REFINE_CSV.exists():
    print(f"Not found: {REFINE_CSV}")
    print("Run wsb_refine to generate all_refinements.csv first.")
    raise SystemExit(1)

# ---- CLI ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n",      type=int,   default=10,  help="max diverse solutions")
parser.add_argument("--fps",    type=float, default=10,  help="frames per second")
parser.add_argument("--step",   type=float, default=24,  help="hours per frame")
parser.add_argument("--stride", type=int,   default=1,   help="use every N-th CSV row (default 1 = all points)")
parser.add_argument("--tail",   type=float, default=15,  help="comet-tail length in days (default 15)")
parser.add_argument("--video",   action="store_true",     help="also export GIF + MP4 via PyVista (3D ECI + SOI)")
parser.add_argument("--preview", action="store_true",     help="render a single mid-point frame as PNG (fast check)")
parser.add_argument("--compact", action="store_true",     help="SOI inset in bottom-right of ECI panel (phone-friendly)")
args = parser.parse_args()

TAIL_ND = args.tail * 86400.0 / T_STAR   # tail length in non-dimensional time

# ---- ECI frame ---------------------------------------------------------------
if EPOCH_TXT.exists():
    _, R0_wsb = load_epoch_info(EPOCH_TXT)
    print("  ECI frame from epoch_info.txt")
else:
    _inc = np.radians(23.4)
    R0_wsb = np.array([
        [1.0,           0.0,            0.0],
        [0.0,  np.cos(_inc), -np.sin(_inc)],
        [0.0,  np.sin(_inc),  np.cos(_inc)],
    ])
    print("  WARNING: epoch_info.txt not found — using approximate ECI tilt (~23.4°)")
    print("  Run wsb_circularize to get the exact frame.")

# ---- Load and select diverse solutions ---------------------------------------
print(f"Loading {REFINE_CSV.name} ...", flush=True)
df_raw = pd.read_csv(REFINE_CSV)
print(f"  {len(df_raw)} rows loaded")

# Filter to Hill-entering solutions
if "entered_hill" in df_raw.columns:
    df_hill = df_raw[df_raw["entered_hill"].astype(bool)].copy()
else:
    r_moon_col = np.sqrt((df_raw["x_nd"] - X_M)**2 + df_raw["y_nd"]**2)
    entering = df_raw[r_moon_col < R_HILL_ND]["run_id"].unique()
    df_hill = df_raw[df_raw["run_id"].isin(entering)].copy()

print(f"  {df_hill['run_id'].nunique()} Hill-entering runs")

# Best solution per 30° theta_sun_deg bin
meta = (df_hill.groupby("run_id")
        .agg(theta_sun_deg=("theta_sun_deg", "first"),
             est_capture_orbits=("est_capture_orbits", "first"))
        .reset_index())
meta["theta_bin"] = (meta["theta_sun_deg"] // 30).astype(int)
best_per_bin = meta.loc[meta.groupby("theta_bin")["est_capture_orbits"].idxmax()]
top = best_per_bin.nlargest(args.n, "est_capture_orbits")

selected_run_ids = top["run_id"].tolist()
print(f"  Selected {len(selected_run_ids)} diverse solutions:")
for _, row in top.iterrows():
    print(f"    run_id={int(row['run_id'])}  bin={int(row['theta_bin'])*30}-{int(row['theta_bin'])*30+30}°  "
          f"theta_sun={row['theta_sun_deg']:.1f}°  orbits={row['est_capture_orbits']:.2f}")

# Load trajectory rows for selected runs only
df_traj = df_raw[df_raw["run_id"].isin(selected_run_ids)].dropna(subset=["x_nd"]).copy()
print(f"  {len(df_traj)} trajectory rows for {len(selected_run_ids)} solutions")

# ---- Pre-compute per-solution ECI and SOI data -------------------------------
print(f"  Pre-computing coordinates (stride={args.stride}) ...", flush=True)
solutions: dict = {}
for run_id in selected_run_ids:
    sub = (df_traj[df_traj["run_id"] == run_id]
           .sort_values("time_nd").reset_index(drop=True))
    if sub.empty:
        continue
    t_raw = sub["time_nd"].values
    x_raw = sub["x_nd"].values
    y_raw = sub["y_nd"].values
    z_raw = sub["z_nd"].values if "z_nd" in sub.columns else np.zeros_like(t_raw)

    # Hill-sphere entry and exit times (on raw data before stride)
    r_moon_arr = np.sqrt((x_raw - X_M)**2 + y_raw**2)
    in_hill    = r_moon_arr < R_HILL_ND
    t_hill     = float(t_raw[in_hill][0]) if in_hill.any() else float(t_raw[-1])

    # First exit after entry: clip trajectory there so the tail stops cleanly
    if in_hill.any():
        entry_idx = int(np.argmax(in_hill))
        post_exit = np.where(~in_hill[entry_idx:])[0]
        t_exit    = float(t_raw[entry_idx + post_exit[0]]) if len(post_exit) else float(t_raw[-1])
    else:
        entry_idx = len(t_raw)
        t_exit    = float(t_raw[-1])

    # Apply stride and clip to t_exit so stored arrays end at SOI departure
    sl   = slice(None, None, args.stride)
    t    = t_raw[sl]; x = x_raw[sl]; y = y_raw[sl]; z = z_raw[sl]
    clip = t <= t_exit
    t = t[clip]; x = x[clip]; y = y[clip]; z = z[clip]

    xi3, yi3, zi3 = rot_em_to_eci(x, y, z, t, R0_wsb)

    xi_em = (x * np.cos(t) - y * np.sin(t)) * L_KM
    yi_em = (x * np.sin(t) + y * np.cos(t)) * L_KM
    xi_mc = xi_em - X_M * np.cos(t) * L_KM
    yi_mc = yi_em - X_M * np.sin(t) * L_KM

    # Unwrapped Moon-centred angle for live orbit counting
    unwrapped_mc = np.unwrap(np.arctan2(yi_mc, xi_mc))

    solutions[run_id] = dict(
        t=t,
        xi=xi3, yi=yi3, zi=zi3,
        xi_mc=xi_mc, yi_mc=yi_mc,
        unwrapped_mc=unwrapped_mc,
        t_hill=t_hill,
        t_days=t_hill * T_STAR / 86400.0,
        orbits=float(sub["est_capture_orbits"].iloc[0]),
        theta_sun=float(sub["theta_sun_deg"].iloc[0]),
    )

sol_ids = [r for r in selected_run_ids if r in solutions]
N = len(sol_ids)

# Animation runs until the last trajectory's data ends; after that each tail
# naturally slides off screen.
t_global_min = min(solutions[r]["t"][0] for r in sol_ids)
t_global_max = max(solutions[r]["t"][-1] for r in sol_ids) + TAIL_ND
dt_nd  = args.step * 3600.0 / T_STAR
t_anim = np.arange(t_global_min, t_global_max + dt_nd * 0.5, dt_nd)
print(f"  Animation: {len(t_anim)} frames  "
      f"(step={args.step:.0f} h  fps={args.fps}  stride={args.stride}  tail={args.tail:.0f} d)")

r_eci_lim = max(
    max(np.max(np.abs(solutions[r]["xi"])) for r in sol_ids),
    max(np.max(np.abs(solutions[r]["yi"])) for r in sol_ids),
    max(np.max(np.abs(solutions[r]["zi"])) for r in sol_ids),
) * 1.2
r_eci_lim = max(r_eci_lim, R_HILL_KM * 2.0)
VIEW_SOI  = R_HILL_KM * 1.6

# ---- Static geometry ---------------------------------------------------------
sx, sy, sz = sphere_surface(R_EARTH, nu=30, nv=16)
t_ring = np.linspace(0, 2 * np.pi, 200)
mxr, myr, mzr = moon_em_to_eci(t_ring, R0_wsb)

phi  = np.linspace(0, 2 * np.pi, 200)
hs_x = (R_HILL_KM * np.cos(phi)).tolist()
hs_y = (R_HILL_KM * np.sin(phi)).tolist()
md_x = (R_MOON * np.cos(phi)).tolist()
md_y = (R_MOON * np.sin(phi)).tolist()

# ---- Figure ------------------------------------------------------------------
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "xy"}]],
    subplot_titles=[f"{N} different solutions (ECI)",
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
# Moon drawn as a layout shape so it always renders above all trajectory traces,
# cleanly hiding any sub-Moon interpolation segment.
MOON_SHAPE = dict(
    type="circle",
    xref="x", yref="y",
    x0=-R_MOON, y0=-R_MOON, x1=R_MOON, y1=R_MOON,
    fillcolor="rgba(160,160,160,0.55)",
    line=dict(color="#AAAAAA", width=1),
    layer="above",
)

# ---- Animated traces ---------------------------------------------------------
# Layout:
#   n0+0        : Moon dot (3D)
#   n0+1..N     : comet-tail 3D trails
#   n0+N+1..2N  : s/c dots 3D
#   n0+2N+1..3N : comet-tail 2D trails
#   n0+3N+1..4N : s/c dots 2D
#   n0+4N+1     : orbit-count text (2D right panel)
n0 = len(fig.data)

mxi0, myi0, mzi0 = moon_em_to_eci(np.array([0.0]), R0_wsb)
fig.add_trace(go.Scatter3d(
    x=[float(mxi0[0])], y=[float(myi0[0])], z=[float(mzi0[0])],
    mode="markers+text", marker=dict(color=COL_MOON, size=7),
    text=["Moon"], textposition="top center",
    textfont=dict(color=COL_MOON, size=9),
    name="Moon", showlegend=False,
), row=1, col=1)

for _ in sol_ids:
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="lines",
        line=dict(color=COL_TRAJ, width=3),
        showlegend=False,
    ), row=1, col=1)

# S/C dots 3D
for _ in sol_ids:
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="markers",
        marker=dict(color=COL_DOT, size=5,
                    line=dict(color=COL_TRAJ, width=2)),
        showlegend=False,
    ), row=1, col=1)

for _ in sol_ids:
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines",
        line=dict(color=COL_TRAJ, width=2),
        showlegend=False,
    ), row=1, col=2)

# S/C dots 2D
for _ in sol_ids:
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="markers",
        marker=dict(color=COL_DOT, size=8,
                    line=dict(color=COL_TRAJ, width=2)),
        showlegend=False,
    ), row=1, col=2)

# Orbit-count text follows the tail head in the SOI panel
fig.add_trace(go.Scatter(
    x=[], y=[], mode="text", text=[],
    textposition="top center",
    textfont=dict(color=COL_TRAJ, size=9),
    showlegend=False,
), row=1, col=2)

ANIM_TRACES = list(range(n0, n0 + 4 * N + 2))

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

    # Pre-compute tail mask once per solution: vanishes immediately when t_now
    # exceeds the trajectory's last data point (t_exit / SOI departure).
    masks = {}
    for run_id in sol_ids:
        s = solutions[run_id]
        if t_now > s["t"][-1]:
            masks[run_id] = np.zeros(len(s["t"]), dtype=bool)
        else:
            masks[run_id] = (s["t"] > t_now - TAIL_ND) & (s["t"] <= t_now)

    # 3D comet tails
    for run_id in sol_ids:
        s = solutions[run_id]; mask = masks[run_id]
        tx = s["xi"][mask]; ty = s["yi"][mask]; tz = s["zi"][mask]
        data.append(go.Scatter3d(x=tx.tolist(), y=ty.tolist(), z=tz.tolist()))

    # 3D s/c dots (head of tail)
    for run_id in sol_ids:
        s = solutions[run_id]; mask = masks[run_id]
        if mask.any():
            data.append(go.Scatter3d(x=[float(s["xi"][mask][-1])],
                                     y=[float(s["yi"][mask][-1])],
                                     z=[float(s["zi"][mask][-1])]))
        else:
            data.append(go.Scatter3d(x=[], y=[], z=[]))

    # 2D comet tails
    for run_id in sol_ids:
        s = solutions[run_id]; mask = masks[run_id]
        mx = s["xi_mc"][mask]; my = s["yi_mc"][mask]
        data.append(go.Scatter(x=mx.tolist(), y=my.tolist()))

    # 2D s/c dots (head of tail)
    for run_id in sol_ids:
        s = solutions[run_id]; mask = masks[run_id]
        if mask.any():
            data.append(go.Scatter(x=[float(s["xi_mc"][mask][-1])],
                                   y=[float(s["yi_mc"][mask][-1])]))
        else:
            data.append(go.Scatter(x=[], y=[]))

    # Orbit-count text: one label per trajectory whose tail is inside the SOI
    orb_x, orb_y, orb_txt = [], [], []
    for run_id in sol_ids:
        s    = solutions[run_id]
        mask = masks[run_id]
        if not mask.any() or t_now < s["t_hill"]:
            continue
        mask_post = (s["t"] >= s["t_hill"]) & (s["t"] <= t_now)
        if mask_post.sum() >= 2:
            idx0  = int(np.argmax(s["t"] >= s["t_hill"]))
            delta = s["unwrapped_mc"][mask_post][-1] - s["unwrapped_mc"][idx0]
            n_orb = abs(delta) / (2.0 * np.pi)
        else:
            n_orb = 0.0
        # Anchor at tail head (most recent point) in SOI coordinates
        orb_x.append(float(s["xi_mc"][mask][-1]))
        orb_y.append(float(s["yi_mc"][mask][-1]))
        orb_txt.append(f"{n_orb:.1f}orb")
    data.append(go.Scatter(x=orb_x, y=orb_y, text=orb_txt))

    frames.append(go.Frame(
        data=data, traces=ANIM_TRACES, name=str(fi),
        layout=go.Layout(title_text=(
            f"WSB Diverse Solutions — t = {t_days:.1f} days  "
            f"({N} solutions, one per 30° θ_sun bin)"
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
            f"WSB Diverse Solutions — {N} families, one per 30° θ_sun bin<br>"
            "<sup>Left: ECI frame (3D)  |  Right: Moon-centred EM-inertial  |  "
            f"Comet tail: {args.tail:.0f} days  |  Best capture-orbit solution from each synodic window</sup>"
        ),
        x=0.01, xanchor="left", font=dict(size=13),
    ),
    legend=dict(x=0.01, y=0.98, font=dict(size=10),
                bgcolor="rgba(15,15,25,0.7)"),
    height=720,
    margin=dict(l=0, r=0, t=95, b=80),
    updatemenus=[play_pause_buttons(int(1000 / args.fps))],
    sliders=[dark_slider(slider_steps)],
    shapes=[MOON_SHAPE],
)

fig.write_html(str(OUT_HTML), auto_play=False)
print(f"Saved {OUT_HTML}")

# ---- Optional GIF + MP4 export via PyVista (3-D ECI + 2-D SOI) ──────────────
# Left panel : textured Earth/Moon spheres, tube trajectories, ECI frame.
# Right panel: orthographic Moon-centred Hill-sphere view.
# Colours match the HTML palette (COL_TRAJ / COL_DOT / etc.).
if args.video or args.preview:
    import shutil as _shutil, glob as _glob, os as _os
    import tempfile as _tmpfile, subprocess as _subp
    import urllib.request as _urlreq
    from datetime import datetime, timezone, timedelta
    try:
        import pyvista as pv
    except ImportError:
        raise SystemExit("PyVista not installed — run: pip install pyvista vtk")
    try:
        from PIL import Image as _PILImage
    except ImportError:
        raise SystemExit("Pillow not installed — run: pip install Pillow")

    def _find_ffmpeg():
        if _shutil.which("ffmpeg"):
            return "ffmpeg"
        for pat in [
            r"C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg*\bin\ffmpeg.exe",
        ]:
            hits = _glob.glob(pat)
            if hits:
                return hits[0]
        return None

    _ffmpeg_path = _find_ffmpeg()
    GIF_PATH = OUT_HTML.with_suffix(".gif")
    MP4_PATH = OUT_HTML.with_suffix(".mp4")
    TEX_DIR  = OUT_DIR / "textures"
    TEX_DIR.mkdir(exist_ok=True)

    # ── Epoch ──────────────────────────────────────────────────────────────────
    _TLI_EPOCH = datetime(2026, 4, 2, 23, 49, 0, tzinfo=timezone.utc)
    _dep_t_s   = 0.0
    if EPOCH_TXT.exists():
        with open(EPOCH_TXT) as _ef:
            for _line in _ef:
                if _line.startswith("dep_t_s"):
                    _dep_t_s = float(_line.partition(":")[2].strip())
                    break
    _t0_dt = _TLI_EPOCH + timedelta(seconds=_dep_t_s)

    # ── Colours (0–1 floats, match HTML palette) ───────────────────────────────
    _BG3D  = (22/255,  22/255,  45/255)
    _TRAJ  = (32/255,  92/255, 195/255)
    _DOT   = (20/255, 105/255, 215/255)
    _MCOL  = (195/255, 195/255, 195/255)
    _ECOL  = ( 21/255, 101/255, 192/255)
    _HCOL  = (198/255, 130/255,  30/255)

    def _hex(c):
        return "#{:02x}{:02x}{:02x}".format(*(int(v * 255) for v in c))

    # ── Texture auto-download ──────────────────────────────────────────────────
    _EARTH_URL = "https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg"
    _MOON_URL  = "https://www.solarsystemscope.com/textures/download/2k_moon.jpg"
    _EARTH_JPG = TEX_DIR / "earth_daymap.jpg"
    _MOON_JPG  = TEX_DIR / "moon_1k.jpg"

    def _dl_texture(url, path):
        if path.exists():
            print(f"  {path.name} already cached.")
            return True
        print(f"  Downloading {path.name} ...", end="", flush=True)
        try:
            _urlreq.urlretrieve(url, str(path))
            print(" done.")
            return True
        except Exception as exc:
            print(f" FAILED ({exc}) — solid-colour fallback")
            return False

    _earth_ok  = _dl_texture(_EARTH_URL, _EARTH_JPG)
    _moon_ok   = _dl_texture(_MOON_URL,  _MOON_JPG)
    _earth_tex = pv.read_texture(str(_EARTH_JPG)) if _earth_ok else None
    _moon_tex  = pv.read_texture(str(_MOON_JPG))  if _moon_ok  else None

    # ── Low-level PolyData builders ────────────────────────────────────────────
    def _line_poly(pts_arr):
        n = len(pts_arr)
        if n < 2:
            return pv.PolyData()
        poly = pv.PolyData()
        poly.points = np.asarray(pts_arr, dtype=np.float64)
        poly.lines  = np.array([n] + list(range(n)), dtype=np.int64)
        return poly

    def _ring_poly(radius, z=0.0, n=200):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([radius * np.cos(phi),
                               radius * np.sin(phi),
                               np.full(n, z)])
        poly = pv.PolyData()
        poly.points = pts.astype(np.float64)
        poly.lines  = np.array([n + 1] + list(range(n)) + [0], dtype=np.int64)
        return poly

    def _grid_poly(extent, step):
        """XY-plane grid as a single PolyData (no bounds side-effects)."""
        vals = np.arange(-extent, extent + step * 0.01, step)
        pts, cells, pid = [], [], 0
        for v in vals:
            if abs(v) > extent + 1:
                continue
            pts += [[-extent, v, 0.0], [extent, v, 0.0]]
            cells += [2, pid, pid + 1]; pid += 2
            pts += [[v, -extent, 0.0], [v, extent, 0.0]]
            cells += [2, pid, pid + 1]; pid += 2
        poly = pv.PolyData()
        poly.points = np.array(pts, dtype=np.float64)
        poly.lines  = np.array(cells, dtype=np.int64)
        return poly

    def _overlay_hud(arr, pl, compact=False):
        """Composite legend + axis labels/distance marks onto a screenshot via PIL."""
        from PIL import ImageDraw as _PILDraw, ImageFont as _PILFont
        img  = _PILImage.fromarray(arr).convert("RGBA")
        W, H = img.size

        # ── Font setup — scale with render resolution ─────────────────────────
        _fs    = max(30, int(26 * W / 1920))   # main label font
        _fs_sm = max(28, int(22 * W / 1920))   # small label font
        def _load_fnt(size):
            for name in ("cour.ttf", "consola.ttf", "lucon.ttf"):
                try:
                    return _PILFont.truetype(name, size)
                except Exception:
                    pass
            return None
        _fnt    = _load_fnt(_fs)
        _fnt_sm = _load_fnt(_fs_sm)
        _kw    = {"font": _fnt}    if _fnt    else {}
        _kw_sm = {"font": _fnt_sm} if _fnt_sm else {}
        _grey    = (220, 220, 220, 240)
        _grey_sm = (175, 175, 175, 210)

        # ── Legend box (bottom-left of left panel) ────────────────────────────
        entries = [
            ("Moon orbit",   _MCOL,  "line"),
            ("Hill sphere",  _HCOL,  "dashed"),
            ("Spacecraft",   _DOT,   "square"),
        ]
        _sc = W / 1920                               # scale factor vs 1920 px
        pad   = int(18 * _sc);  row_h = int(56 * _sc);  icon_w = int(46 * _sc)
        bx0 = int(12 * _sc);    bx1   = int(330 * _sc)
        by0 = H - (pad * 2 + row_h * len(entries) + int(10 * _sc))
        by1 = H - int(10 * _sc)
        box = _PILImage.new("RGBA", (bx1 - bx0, by1 - by0), (12, 12, 32, 200))
        img.paste(box, (bx0, by0), mask=box)
        draw = _PILDraw.Draw(img)
        _lbl_dy = int(_fs * 0.45)   # vertical centre-offset for label text
        for i, (label, col_f, kind) in enumerate(entries):
            col_i = tuple(int(c * 255) for c in col_f) + (255,)
            cy  = by0 + pad + i * row_h + row_h // 2
            ix0 = bx0 + pad
            if kind == "line":
                draw.line([(ix0, cy), (ix0 + icon_w, cy)], fill=col_i, width=3)
            elif kind == "dashed":
                x = ix0
                while x < ix0 + icon_w:
                    draw.line([(x, cy), (min(x + 7, ix0 + icon_w), cy)],
                              fill=col_i, width=3)
                    x += 12
            elif kind == "square":
                s = int(7 * _sc)
                mx = ix0 + icon_w // 2
                draw.rectangle([mx - s, cy - s, mx + s, cy + s], fill=col_i)
            else:
                r = 6
                draw.ellipse([ix0, cy - r, ix0 + r * 2, cy + r], fill=col_i)
            draw.text((ix0 + icon_w + int(8 * _sc), cy - _lbl_dy), label,
                      fill=(215, 215, 215, 255), **_kw)

        # ── VTK world→screen: LEFT panel only ───────────────────────────────
        def _w2px_eci(x, y, z):
            ren = pl.renderers[0]
            ren.SetWorldPoint(x, y, z, 1.0)
            ren.WorldToDisplay()
            sx, sy = ren.GetDisplayPoint()[:2]
            return int(round(sx)), int(round(H - sy))

        # Analytical parallel projection: RIGHT panel (standard) or stacked below (compact)
        if compact:
            _soi_y0 = int((1 - _STK_SOI_FRAC) * H)   # top of SOI panel in PIL coords
            _soi_cx = W / 2.0
            _soi_cy = (_soi_y0 + H) / 2.0
            _soi_hw = W / 2.0
            _soi_hh = (H - _soi_y0) / 2.0
            _soi_ar = _soi_hw / _soi_hh
            def _w2px_soi(wx, wy):
                sx = int(round(_soi_cx + _soi_hw * wx / (VIEW_SOI * _soi_ar)))
                sy = int(round(_soi_cy - _soi_hh * wy / VIEW_SOI))
                return sx, sy
            _Lmax = W - int(6 * _sc)   # ECI spans full width in stacked mode
        else:
            _ar = (W / 2.0) / H
            def _w2px_soi(wx, wy):
                W_half = W // 2
                sx = W_half + int(round(W_half / 2.0 * (1.0 + wx / (VIEW_SOI * _ar))))
                sy = int(round(H / 2.0 * (1.0 - wy / VIEW_SOI)))
                return sx, sy
            _Lmax = W // 2 - int(6 * _sc)   # stay in left half

        _ax_r  = r_eci_lim * 0.85

        # ── LEFT panel: ±X / ±Y / +Z tip labels ──────────────────────────────
        for wpt, lbl, ox, oy in [
            (( _ax_r,  0,          0),    "+X",  4, -9),
            ((-_ax_r,  0,          0),    "-X", -22, -9),
            (( 0,      _ax_r,      0),    "+Y",  4, -9),
            (( 0,     -_ax_r,      0),    "-Y",  4,  4),
            (( 0,      0,  _ax_r * 0.4), "+Z",  4, -9),
        ]:
            try:
                px, py = _w2px_eci(*wpt)
                if px + ox < _Lmax:
                    draw.text((px + ox, py + oy), lbl, fill=_grey, **_kw)
                    if lbl == "-Y":
                        try:
                            _ux, _uy = _w2px_eci(50_000, -1_400_000, 0)
                            draw.text((_ux + int(6 * _sc), _uy - int(_fs_sm * 0.5)),
                                      "×10³ km", fill=_grey_sm, **_kw_sm)
                        except Exception:
                            pass
            except Exception:
                pass

        # Distance labels at every _g_step along X, Y, and Z (both ±)
        for _axis_wpt, _axis_lbl_off, _axis_max in [
            (lambda d: (d, 0, 0),   lambda px, py: (px - int(12*_sc), py + int(5*_sc)),  _ax_r),
            (lambda d: (0, d, 0),   lambda px, py: (px + int(5*_sc),  py - int(7*_sc)),  _ax_r),
            (lambda d: (0, 0, d),   lambda px, py: (px + int(6*_sc),  py - int(5*_sc)),  _ax_r * 0.4),
        ]:
            _td = _g_step
            while _td < _axis_max:
                lbl = f"{int(_td // 1000)}"
                for _sign in (+1, -1):
                    try:
                        px, py = _w2px_eci(*_axis_wpt(_sign * _td))
                        tx, ty = _axis_lbl_off(px, py)
                        if tx < _Lmax:
                            draw.text((tx, ty), lbl, fill=_grey_sm, **_kw)
                    except Exception:
                        pass
                _td += _g_step

        # ── RIGHT panel: distance marks only (no +X/+Y direction labels) ─────
        _tk = int(5 * _sc)
        for _axis in ("x", "y"):
            _td = _soi_step
            while _td < VIEW_SOI * 0.92:
                lbl = f"{int(_td / 1000):g}"
                for _sign in (+1, -1):
                    if _axis == "x":
                        px, py = _w2px_soi(_sign * _td, 0)
                        draw.line([(px, py - _tk), (px, py + _tk)],
                                  fill=_grey_sm, width=1)
                        draw.text((px - int(14*_sc), py + int(6*_sc)),
                                  lbl, fill=_grey_sm, **_kw_sm)
                    else:
                        px, py = _w2px_soi(0, _sign * _td)
                        draw.line([(px - _tk, py), (px + _tk, py)],
                                  fill=_grey_sm, width=1)
                        draw.text((px + int(6*_sc), py - int(8*_sc)),
                                  lbl, fill=_grey_sm, **_kw_sm)
                _td += _soi_step

        # ×10³ km unit label — next to SOI -Y axis, just below the bottommost tick
        _sy_end = _soi_step * max(1, int(VIEW_SOI * 0.88 / _soi_step))
        try:
            _pu, _pv = _w2px_soi(0, -_sy_end-10_000)
            draw.text((_pu + int(8 * _sc), _pv + int(42 * _sc)),
                      "×10³ km", fill=_grey_sm, **_kw_sm)
        except Exception:
            pass

        return np.array(img.convert("RGB"))

    # ── Visual radii (scaled up from physical for display clarity) ────────────
    _R_EARTH_VIS    = max(R_EARTH * 3.2, r_eci_lim * 0.016)
    _R_MOON_ECI_VIS = _R_EARTH_VIS * 0.30

    # ── Camera constants ───────────────────────────────────────────────────────
    _eye_s   = r_eci_lim * 1.05
    _C3_POS  = ( 0.9 * _eye_s, -1.4 * _eye_s, 0.7 * _eye_s)
    _C3_FOC  = (0.0, 0.0, 0.0)
    _C3_UP   = (0.0, 0.0, 1.0)
    _C2_POS  = (0.0, 0.0, VIEW_SOI * 10.0)
    _C2_FOC  = (0.0, 0.0, 0.0)
    _C2_UP   = (0.0, 1.0, 0.0)
    _STK_SOI_FRAC = 0.40   # compact: bottom fraction of window for SOI panel
    _STK_VID_W    = 1080   # compact: portrait video width
    _STK_VID_H    = 1620   # compact: portrait video height

    # ── Grid / render quality settings ────────────────────────────────────────
    _g_step   = 400_000                        # ECI grid/label step [km]
    _soi_step = round(R_HILL_KM / 2.0, -3)    # SOI grid/label step ≈ 31 k km
    _VID_W = _STK_VID_W if args.compact else 2560
    _VID_H = _STK_VID_H if args.compact else 1440
    _hifi     = False                          # set True before video render

    # ── Plotter factory ────────────────────────────────────────────────────────
    def _build_pl(w, h, hifi=False, compact=False):
        pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[w, h],
                        line_smoothing=True, point_smoothing=True)

        # LEFT — 3-D ECI ───────────────────────────────────────────────────────
        pl.subplot(0, 0)
        pl.set_background(_BG3D, all_renderers=False)

        # Earth: static mesh, actor rotation applied per-frame in _upd.
        # start_phi/end_phi avoids the polar-singularity distortion artefact.
        _sph_t = 90 if hifi else 180
        _sph_p = 45 if hifi else 90
        _es = pv.Sphere(radius=_R_EARTH_VIS,
                        theta_resolution=_sph_t, phi_resolution=_sph_p,
                        start_phi=0.1, end_phi=179.9)
        # Override UV with explicit equirectangular formula (same fix as sensitivity_anim).
        _epts = _es.points
        _er   = np.sqrt(np.sum(_epts**2, axis=1))
        _elat = np.arcsin(np.clip(_epts[:, 2] / np.maximum(_er, 1e-10), -1.0, 1.0))
        _elon = np.arctan2(_epts[:, 1], _epts[:, 0])
        _es.active_texture_coordinates = np.column_stack([
            (_elon + np.pi) / (2.0 * np.pi),
            (_elat + np.pi / 2.0) / np.pi,
        ]).astype(np.float32)
        if _earth_tex is not None:
            pl._earth_actor = pl.add_mesh(_es, texture=_earth_tex, smooth_shading=True,
                                          lighting=False, name="earth")
        else:
            pl._earth_actor = pl.add_mesh(_es, color=_ECOL, smooth_shading=True,
                                          opacity=0.92, name="earth")

        # Moon orbit ring
        pl.add_mesh(
            _line_poly(np.column_stack([mxr, myr, mzr])),
            color=_MCOL, opacity=0.80, line_width=1.0, name="moon_orbit",
        )

        # XY floor grid
        pl.add_mesh(_grid_poly(r_eci_lim, _g_step),
                    color=(80/255, 80/255, 80/255), opacity=0.10,
                    line_width=3.2, name="eci_grid")

        # ECI axis lines — grey, full ±, no arrowhead dots (labels via PIL overlay)
        _ax_r   = r_eci_lim * 0.85
        _ax_col = (160/255, 160/255, 160/255)
        for _ap, _an in [
            (np.array([[-_ax_r,  0,           0], [_ax_r,  0,          0]]),  "ax_x"),
            (np.array([[0,      -_ax_r,        0], [0,      _ax_r,      0]]),  "ax_y"),
            (np.array([[0,       0, -_ax_r*0.4],   [0,      0, _ax_r*0.4]]), "ax_z"),
        ]:
            pl.add_mesh(_line_poly(_ap), color=_ax_col, line_width=4.5,
                        opacity=0.25, name=_an)
        # Tick marks at every _g_step, both ± directions, along X (perp in Y) and Y (perp in X)
        _tk_h   = r_eci_lim * 0.012
        _tk_pts, _tk_cells, _tkp = [], [], 0
        _td = _g_step
        while _td < _ax_r:
            for _sign in (+1, -1):
                # X-axis tick (perpendicular in Y)
                _tk_pts  += [[_sign * _td, -_tk_h, 0], [_sign * _td, _tk_h, 0]]
                _tk_cells += [2, _tkp, _tkp + 1]; _tkp += 2
                # Y-axis tick (perpendicular in X)
                _tk_pts  += [[-_tk_h, _sign * _td, 0], [_tk_h, _sign * _td, 0]]
                _tk_cells += [2, _tkp, _tkp + 1]; _tkp += 2
                # Z-axis tick (perpendicular in X), only within Z extent
                if _td <= _ax_r * 0.4:
                    _tk_pts  += [[-_tk_h, 0, _sign * _td], [_tk_h, 0, _sign * _td]]
                    _tk_cells += [2, _tkp, _tkp + 1]; _tkp += 2
            _td += _g_step
        if _tk_pts:
            _tkpoly        = pv.PolyData()
            _tkpoly.points = np.array(_tk_pts, dtype=np.float64)
            _tkpoly.lines  = np.array(_tk_cells, dtype=np.int64)
            pl.add_mesh(_tkpoly, color=_ax_col, opacity=0.25,
                        line_width=4.5, name="ax_ticks")

        # Star field
        _rng_s = np.random.default_rng(42)
        _sph_s = np.arccos(_rng_s.uniform(-1, 1, 1000))
        _sth_s = _rng_s.uniform(0, 2 * np.pi, 1000)
        _sr_s  = r_eci_lim * 4.5
        _spts  = np.column_stack([
            _sr_s * np.sin(_sph_s) * np.cos(_sth_s),
            _sr_s * np.sin(_sph_s) * np.sin(_sth_s),
            _sr_s * np.cos(_sph_s) * 0.6,
        ])
        pl.add_mesh(pv.PolyData(_spts), style="points", point_size=2.2,
                    color="white", opacity=0.32, name="stars3d")

        pl.camera.position    = _C3_POS
        pl.camera.focal_point = _C3_FOC
        pl.camera.up          = _C3_UP
        pl.camera_set = True

        # RIGHT — Moon-centred orthographic ────────────────────────────────────
        pl.subplot(0, 1)
        pl.set_background(_BG3D, all_renderers=False)
        pl.enable_parallel_projection()

        # Background grid + crosshair zero lines
        _soi_ext = VIEW_SOI * 1.15
        pl.add_mesh(_grid_poly(_soi_ext, _soi_step),
                    color=(80/255, 80/255, 80/255), opacity=0.10,
                    line_width=3.2, name="soi_grid")
        pl.add_mesh(_line_poly([[-_soi_ext, 0, 0], [_soi_ext, 0, 0]]),
                    color=(150/255, 150/255, 150/255), opacity=0.42,
                    line_width=1.5, name="xhair_x")
        pl.add_mesh(_line_poly([[0, -_soi_ext, 0], [0, _soi_ext, 0]]),
                    color=(150/255, 150/255, 150/255), opacity=0.42,
                    line_width=1.5, name="xhair_y")

        pl.add_mesh(_ring_poly(R_HILL_KM), color=_HCOL, line_width=1.5,
                    name="hill_ring")

        # Textured Moon sphere (scaled up from physical for visibility at SOI scale)
        _ms2_r = R_MOON * 1.5
        _ms2_t, _ms2_p = (48, 32) if hifi else (36, 24)
        _ms2 = pv.Sphere(radius=_ms2_r, center=(0.0, 0.0, 0.0),
                         theta_resolution=_ms2_t, phi_resolution=_ms2_p,
                         start_phi=0.1, end_phi=179.9)
        if _moon_tex is not None:
            if not any("coord" in k.lower() for k in _ms2.point_data.keys()):
                _ms2 = _ms2.texture_map_to_sphere()
            pl.add_mesh(_ms2, texture=_moon_tex, smooth_shading=True,
                        lighting=False, name="moon_soi")
        else:
            pl.add_mesh(_ms2, color=_MCOL, opacity=0.85, smooth_shading=True,
                        name="moon_soi")

        pl.camera.position       = _C2_POS
        pl.camera.focal_point    = _C2_FOC
        pl.camera.up             = _C2_UP
        pl.camera.parallel_scale = VIEW_SOI
        pl.camera_set = True

        if compact:
            # Stacked portrait layout: ECI on top, SOI panel below.
            pl.renderers[0].SetViewport(0.0, _STK_SOI_FRAC, 1.0, 1.0)
            pl.renderers[1].SetViewport(0.0, 0.0, 1.0, _STK_SOI_FRAC)

        return pl

    # ── Per-frame update ───────────────────────────────────────────────────────
    # Earth physical rotation: 360° × T_STAR / 86164 °/ND.
    # Absolute formula (_EARTH_PHYS_DEG_PER_ND * t_now) % 360 gives the
    # physically correct orientation at each frame without aliasing.
    _EARTH_PHYS_DEG_PER_ND = 360.0 * T_STAR / 86164.0

    _EMPTY  = pv.PolyData(np.zeros((1, 3), dtype=np.float64))

    def _upd(pl, fi):
        t_now  = t_anim[fi]
        sim_day = float(t_now) * T_STAR / 86400.0

        # — Left panel — (lock camera first so add_mesh never auto-resets it)
        pl.subplot(0, 0)
        pl.camera.position    = _C3_POS
        pl.camera.focal_point = _C3_FOC
        pl.camera.up          = _C3_UP
        pl.camera_set = True

        # Absolute physical orientation — correct regardless of step size.
        _earth_rot_deg = (_EARTH_PHYS_DEG_PER_ND * t_now) % 360.0
        _ea = getattr(pl, '_earth_actor', None)
        if _ea is not None:
            try:
                _ea.orientation = (0.0, 0.0, _earth_rot_deg)
            except AttributeError:
                _ea.SetOrientation(0.0, 0.0, _earth_rot_deg)

        m_arr = moon_em_to_eci(np.array([t_now]), R0_wsb)
        mx, my, mz = float(m_arr[0][0]), float(m_arr[1][0]), float(m_arr[2][0])
        _ms_t = 36 if _hifi else 24
        _ms_p = 24 if _hifi else 16
        _ms = pv.Sphere(radius=_R_MOON_ECI_VIS, center=(mx, my, mz),
                        theta_resolution=_ms_t, phi_resolution=_ms_p,
                        start_phi=0.1, end_phi=179.9)
        if _moon_tex is not None:
            if not any("coord" in k.lower() for k in _ms.point_data.keys()):
                _ms = _ms.texture_map_to_sphere()
            pl.add_mesh(_ms, texture=_moon_tex, smooth_shading=True,
                        lighting=False, name="moon3d")
        else:
            pl.add_mesh(_ms, color=_MCOL, opacity=0.85, smooth_shading=True,
                        name="moon3d")

        for k, run_id in enumerate(sol_ids):
            s    = solutions[run_id]
            mask = (np.zeros(len(s["t"]), dtype=bool) if t_now > s["t"][-1]
                    else (s["t"] > t_now - TAIL_ND) & (s["t"] <= t_now))
            if mask.sum() >= 2:
                _tp = _line_poly(np.column_stack(
                    [s["xi"][mask], s["yi"][mask], s["zi"][mask]]))
                _n3 = mask.sum()
                _rgb3 = np.array([int(c*255) for c in _TRAJ], dtype=np.uint8)
                _rgba3 = np.zeros((_n3, 4), dtype=np.uint8)
                _rgba3[:, :3] = _rgb3
                _rgba3[:, 3] = np.linspace(0, 240, _n3, dtype=np.uint8)
                _tp.point_data["colors"] = _rgba3
                pl.add_mesh(_tp, scalars="colors", rgba=True,
                            line_width=4.5, name=f"trl3_{k}")
            else:
                pl.add_mesh(_EMPTY, name=f"trl3_{k}", opacity=0.0)
            if mask.any():
                _cx3, _cy3, _cz3 = (s["xi"][mask][-1],
                                    s["yi"][mask][-1],
                                    s["zi"][mask][-1])
                _sz3 = _R_EARTH_VIS * 0.18
                _cube = pv.Box(bounds=[_cx3-_sz3, _cx3+_sz3,
                                       _cy3-_sz3, _cy3+_sz3,
                                       _cz3-_sz3, _cz3+_sz3])
                pl.add_mesh(_cube, color=_DOT, name=f"dot3_{k}")
            else:
                pl.add_mesh(_EMPTY, name=f"dot3_{k}", opacity=0.0)

        # Day counter (not datetime) — solutions span different real departure epochs
        # so a single absolute date would be misleading. See module docstring.
        pl.add_text(
            f"{N} solutions (ECI)   Day {sim_day:.1f}",
            position="upper_edge", font_size=15, color="white",
            name="ttl3", font="courier",
        )
        pl.camera.position    = _C3_POS
        pl.camera.focal_point = _C3_FOC
        pl.camera.up          = _C3_UP
        pl.camera_set = True

        # — Right panel — (lock camera before mesh loop)
        pl.subplot(0, 1)
        pl.enable_parallel_projection()
        pl.camera.position       = _C2_POS
        pl.camera.focal_point    = _C2_FOC
        pl.camera.up             = _C2_UP
        pl.camera.parallel_scale = VIEW_SOI
        pl.camera_set = True

        for k, run_id in enumerate(sol_ids):
            s    = solutions[run_id]
            mask = (np.zeros(len(s["t"]), dtype=bool) if t_now > s["t"][-1]
                    else (s["t"] > t_now - TAIL_ND) & (s["t"] <= t_now))
            if mask.sum() >= 2:
                _n2 = mask.sum()
                _tp2 = _line_poly(np.column_stack(
                    [s["xi_mc"][mask], s["yi_mc"][mask], np.zeros(_n2)]))
                _rgb2 = np.array([int(c*255) for c in _TRAJ], dtype=np.uint8)
                _rgba2 = np.zeros((_n2, 4), dtype=np.uint8)
                _rgba2[:, :3] = _rgb2
                _rgba2[:, 3] = np.linspace(0, 240, _n2, dtype=np.uint8)
                _tp2.point_data["colors"] = _rgba2
                pl.add_mesh(_tp2, scalars="colors", rgba=True,
                            line_width=3.0, name=f"trl2_{k}")
            else:
                pl.add_mesh(_EMPTY, name=f"trl2_{k}", opacity=0.0)
            if mask.any():
                _cx2, _cy2 = s["xi_mc"][mask][-1], s["yi_mc"][mask][-1]
                _sz2 = R_HILL_KM * 0.018
                _sq = pv.Plane(center=(_cx2, _cy2, 0.0),
                               i_size=_sz2 * 2, j_size=_sz2 * 2,
                               i_resolution=1, j_resolution=1)
                pl.add_mesh(_sq, color=_DOT, name=f"dot2_{k}")
            else:
                pl.add_mesh(_EMPTY, name=f"dot2_{k}", opacity=0.0)

        _ttl2 = "Hill Sphere" if args.compact else "Moon-Centred (Hill Sphere)"
        _fs2  = 14 if args.compact else 18
        pl.add_text(_ttl2, position="upper_edge", font_size=_fs2, color="white",
                    name="ttl2", font="courier")
        pl.camera.position       = _C2_POS
        pl.camera.focal_point    = _C2_FOC
        pl.camera.up             = _C2_UP
        pl.camera.parallel_scale = VIEW_SOI
        pl.camera_set = True

    # ── Preview: single mid-point frame → PNG ─────────────────────────────────
    if args.preview:
        _prev_fi   = len(t_anim) // 2
        _prev_path = OUT_HTML.with_suffix(".preview.png")
        print(f"Rendering preview (frame {_prev_fi}/{len(t_anim)}) ...", flush=True)
        _prev_w = _STK_VID_W if args.compact else 1920
        _prev_h = _STK_VID_H if args.compact else 1080
        _pl_prev = _build_pl(_prev_w, _prev_h, compact=args.compact)
        _upd(_pl_prev, _prev_fi)
        _prev_arr = _pl_prev.screenshot(return_img=True)
        _prev_arr = _overlay_hud(_prev_arr, _pl_prev, compact=args.compact)
        _PILImage.fromarray(_prev_arr).save(str(_prev_path))
        _pl_prev.close()
        print(f"Saved {_prev_path}")

    # ── Render loop (only when --video) ───────────────────────────────────────
    if args.video:
        mp4_step  = max(1, len(t_anim) // 400)
        mp4_idxs  = list(range(0, len(t_anim), mp4_step))
        # trim last 10 sim-days — fixed in simulation time, independent of fps/step
        _trim_anim = int(60 * 86400.0 / T_STAR / dt_nd)
        _trim_mp4  = max(0, _trim_anim // mp4_step)
        mp4_idxs   = mp4_idxs[: max(10, len(mp4_idxs) - _trim_mp4)]
        gif_every = max(1, len(mp4_idxs) // 120)

        _hifi = True   # higher sphere tessellation for final render
        print(f"Rendering {len(mp4_idxs)} frames via PyVista ({_VID_W}×{_VID_H}) ...",
              flush=True)
        _pl  = _build_pl(_VID_W, _VID_H, hifi=True, compact=args.compact)
        _tmp = _tmpfile.mkdtemp(prefix="wsb_pv_")
        _gif_imgs = []

        try:
            for _fi, fi in enumerate(mp4_idxs):
                if _fi % 20 == 0:
                    print(f"  frame {_fi}/{len(mp4_idxs)}", flush=True)
                _upd(_pl, fi)
                _arr = _pl.screenshot(return_img=True)
                _arr = _overlay_hud(_arr, _pl, compact=args.compact)
                _PILImage.fromarray(_arr).save(
                    _os.path.join(_tmp, f"frame_{_fi:06d}.png"))
                if _fi % gif_every == 0:
                    _gw = 540 if args.compact else 1920
                    _gh = 810 if args.compact else 1080
                    _gif_imgs.append(
                        _PILImage.fromarray(_arr).resize(
                            (_gw, _gh), _PILImage.LANCZOS))
            _pl.close()

            # GIF
            print(f"Saving GIF ({len(_gif_imgs)} frames) ...", end="", flush=True)
            _gif_imgs[0].save(
                str(GIF_PATH), save_all=True, append_images=_gif_imgs[1:],
                duration=int(1000 / args.fps), loop=0, optimize=False,
            )
            print(f" done.\nSaved {GIF_PATH}")

            # MP4
            if _ffmpeg_path:
                print(f"Encoding MP4 ({len(mp4_idxs)} frames, {_VID_W}×{_VID_H}) ...",
                      end="", flush=True)
                _subp.run(
                    [_ffmpeg_path, "-y",
                     "-framerate", str(int(args.fps)),
                     "-i", _os.path.join(_tmp, "frame_%06d.png"),
                     "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "15", "-preset", "slow",
                     str(MP4_PATH)],
                    check=True, capture_output=True,
                )
                print(f" done.\nSaved {MP4_PATH}")
            else:
                print("Skipping MP4 — ffmpeg not found (winget install Gyan.FFmpeg).")

        finally:
            _shutil.rmtree(_tmp, ignore_errors=True)
