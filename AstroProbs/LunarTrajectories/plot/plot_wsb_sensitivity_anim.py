"""
plot_wsb_sensitivity_anim.py — animated sensitivity ensemble.

Reads   : out/wsb/sensitivity_ensemble[_<tag>].csv
          out/wsb/sensitivity_summary[_<tag>].txt   (optional, for title)
          out/wsb/epoch_info.txt                     (optional; R0=I if missing)
Saves   : out/wsb/wsb_sensitivity_anim[_<tag>].html
          out/wsb/wsb_sensitivity_anim[_<tag>].gif   (only with --gif)

Left  : 3D ECI — full Earth-to-Moon transfer.
Right : Moon-centred EM-inertial — Hill sphere approach.

Trajectories coloured by outcome: captured / moon_crash / escaped.
Nominal drawn last, thicker, in bright cyan.
All trajectories shown (no stratified sampling cap).
Trajectories clipped at first SOI exit after Hill-sphere entry.
Comet-tail animation: last --tail days of data shown per frame.
"""
import argparse, pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wsb_style import (
    X_M, L_KM, T_STAR, R_EARTH, R_MOON, R_HILL_ND, R_HILL_KM,
    COL_EARTH, COL_MOON, COL_HILL, PAPER_BG, BG_3D,
    rot_em_to_eci, moon_em_to_eci, load_epoch_info, sphere_surface,
    dark_3d_axis, dark_soi_layout, play_pause_buttons, dark_slider,
)

ROOT      = pathlib.Path(__file__).parent.parent
OUT_DIR   = ROOT / "out" / "wsb"
EPOCH_TXT = OUT_DIR / "epoch_info.txt"

MAX_HTML_FRAMES     = 120
MAX_GIF_FRAMES      = 400
MAX_TRAIL_GIF       = 2000
MAX_HTML_TRAIL_PTS  = 80   # max trail points per traj per HTML frame (limits file size)

OUTCOME_COLOR = {
    "captured":   "#1AE870",
    "flyby":      "#C77DFF",
    "moon_crash": "#E63946",
    "miss":       "#666677",
}
OUTCOME_ALPHA    = {"captured": 0.75, "flyby": 0.65, "moon_crash": 0.65, "miss": 0.35}
OUTCOME_WIDTH_3D = {"captured": 6,    "flyby": 5,    "moon_crash": 5,    "miss": 3}
OUTCOME_WIDTH_2D = {"captured": 4.5,  "flyby": 3.5,  "moon_crash": 3.5,  "miss": 2.5}

# Classification threshold for Rust-labelled "captured" trajectories.
# est_orbits = max Hill-sphere dwell / Earth-Moon orbital period (≈ 27.3 days).
# 5 days expressed as a fraction of that period.
import math as _math
CAPTURE_EST_ORBITS_THRESH = 3.5 * 86400.0 / (T_STAR * 2.0 * _math.pi)  # ≈ 0.128

# ---- CLI ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag",    default="",   help="tag used with wsb_sensitivity --tag")
parser.add_argument("--seed",   type=int,   default=0xDEAD_BEEF_0000_0000, help="base seed for reproducible randomness")
parser.add_argument("--fps",    type=float, default=12,  help="frames per second for GIF")
parser.add_argument("--step",   type=float, default=24,  help="hours per animation frame")
parser.add_argument("--skip",   type=int,   default=1,   help="plot every N-th perturbed run")
parser.add_argument("--gif",    action="store_true",     help="export GIF via matplotlib")
parser.add_argument("--mp4",    action="store_true",     help="export MP4 via matplotlib + ffmpeg")
parser.add_argument("--stride", type=int,   default=2,   help="use every N-th CSV point (default 2)")
parser.add_argument("--tail",   type=float, default=15,  help="comet-tail length in days (default 15)")
parser.add_argument("--vid-n",  type=int,   default=0,   help="cap trajectories in GIF/MP4 (0=all)")
parser.add_argument("--preview", action="store_true",    help="render a single mid-point frame to PNG (fast check)")
args = parser.parse_args()

suffix   = f"_{args.tag}" if args.tag else ""
csv_path = OUT_DIR / f"sensitivity_ensemble{suffix}.csv"
txt_path = OUT_DIR / f"sensitivity_summary{suffix}.txt"
html_out = OUT_DIR / f"wsb_sensitivity_anim{suffix}.html"
gif_out  = OUT_DIR / f"wsb_sensitivity_anim{suffix}.gif"
mp4_out  = OUT_DIR / f"wsb_sensitivity_anim{suffix}.mp4"

if not csv_path.exists():
    print(f"Not found: {csv_path}")
    raise SystemExit(1)

TAIL_ND = args.tail * 86400.0 / T_STAR

# ---- ECI frame ---------------------------------------------------------------
if EPOCH_TXT.exists():
    _, R0_wsb = load_epoch_info(EPOCH_TXT)
    print("  ECI frame from epoch_info.txt")
else:
    # Approximate ECI tilt: ecliptic obliquity ~23.4° around X axis.
    # Not astronomically precise (ignores Moon's orbital inclination to ecliptic
    # and ascending-node precession), but far better than flat R0=I.
    # Run wsb_circularize to get the exact epoch-based R0_wsb in epoch_info.txt.
    _inc = np.radians(23.4)
    R0_wsb = np.array([
        [1.0,           0.0,            0.0],
        [0.0,  np.cos(_inc), -np.sin(_inc)],
        [0.0,  np.sin(_inc),  np.cos(_inc)],
    ])
    print("  WARNING: epoch_info.txt not found — using approximate ECI tilt (~23.4°)")
    print("  Run wsb_circularize to get the exact frame.")

# ---- Load summary text -------------------------------------------------------
title_meta = ""
if txt_path.exists():
    for line in txt_path.read_text().splitlines():
        if any(k in line for k in ("θ_sun", "theta_sun", "Nominal IC", "IC source",
                                   "Nominal orbits", "est_orbits", "Tag")):
            title_meta += line.strip() + "  |  "
title_meta = title_meta.rstrip("  |  ")

# ---- Load CSV ----------------------------------------------------------------
print(f"Loading {csv_path.name} ...", flush=True)
df_raw = pd.read_csv(csv_path)
print(f"  {len(df_raw)} rows, {df_raw['run_id'].nunique()} trajectories")

df_traj = df_raw[df_raw["x_nd"].notna()].copy()
df_meta = (df_raw[df_raw["x_nd"].isna()]
           .groupby("run_id").first().reset_index()
           [["run_id", "outcome", "n_orbits", "est_orbits", "is_nominal"]])

# ---- Select run_ids (all trajectories — no stratified sampling cap) ----------
nominal_id = (int(df_meta[df_meta["is_nominal"] == 1]["run_id"].iloc[0])
              if "is_nominal" in df_meta.columns else 0)
perturbed  = df_meta[df_meta["run_id"] != nominal_id]["run_id"].values
if args.skip > 1:
    perturbed = perturbed[::args.skip]

outcome_map    = {int(k): v       for k, v in zip(df_meta["run_id"], df_meta["outcome"])}
orbits_map     = {int(k): int(v)  for k, v in zip(df_meta["run_id"], df_meta["n_orbits"])}
est_orbits_map = {int(k): float(v) for k, v in zip(df_meta["run_id"], df_meta["est_orbits"])}

# Perturbed first so nominal is drawn on top
run_ids_all = [int(r) for r in perturbed] + [nominal_id]

# Display-level outcome: finer classification than the raw Rust label.
#   "captured" — Rust "captured" with est_orbits >= CAPTURE_EST_ORBITS_THRESH
#   "flyby"    — Rust "captured" with est_orbits < CAPTURE_EST_ORBITS_THRESH
#   "miss"     — Rust "escaped"  (never entered Hill sphere)
def _disp(run_id: int) -> str:
    if run_id == nominal_id:
        return "nominal"
    raw = outcome_map.get(run_id, "escaped")
    if raw == "captured":
        est = est_orbits_map.get(run_id, 0.0)
        if est < CAPTURE_EST_ORBITS_THRESH:
            return "flyby"
        return "captured"
    if raw == "escaped":
        return "miss"
    return raw   # moon_crash unchanged

display_outcome_map = {int(r): _disp(int(r)) for r in run_ids_all}

n_tot = len(run_ids_all)
n_cap = sum(1 for r in run_ids_all if display_outcome_map.get(r) == "captured")
n_fly = sum(1 for r in run_ids_all if display_outcome_map.get(r) == "flyby")
n_cra = sum(1 for r in run_ids_all if display_outcome_map.get(r) == "moon_crash")
n_mis = sum(1 for r in run_ids_all if display_outcome_map.get(r) == "miss")

# ---- Pre-compute coordinates -------------------------------------------------
R_MOON_ND = R_MOON / L_KM

print("  Pre-computing coordinates ...", flush=True)
trajs: dict = {}
for run_id in run_ids_all:
    sub = df_traj[df_traj["run_id"] == run_id].sort_values("time_nd")
    if sub.empty:
        continue
    t_raw = sub["time_nd"].values
    x_raw = sub["x_nd"].values
    y_raw = sub["y_nd"].values
    z_raw = sub["z_nd"].values if "z_nd" in sub.columns else np.zeros_like(t_raw)

    outcome = outcome_map.get(run_id, "escaped")

    # SOI exit truncation — outcome-dependent:
    #   captured / nominal : show through the FINAL Hill-sphere exit (all orbits visible)
    #   flyby              : clip at FIRST exit (brief encounter, then remove)
    #   miss / moon_crash  : no SOI gate → full data (miss) or Moon-surface clip (crash)
    r_moon_arr = np.sqrt((x_raw - X_M)**2 + y_raw**2 + z_raw**2)
    in_hill    = r_moon_arr < R_HILL_ND
    t_hill_raw = float(t_raw[in_hill][0]) if in_hill.any() else float(t_raw[-1])
    disp_out   = display_outcome_map.get(run_id, "miss")
    if in_hill.any():
        entry_idx = int(np.argmax(in_hill))
        # First exit after first Hill entry — always computed for video cutoff
        post_first        = np.where(~in_hill[entry_idx:])[0]
        t_first_hill_exit = (float(t_raw[entry_idx + post_first[0]])
                              if len(post_first) else float(t_raw[-1]))
        if disp_out in ("captured", "nominal"):
            # Find last moment inside Hill, then the first departure after that
            last_in_idx = int(np.where(in_hill)[0][-1])
            post_final  = np.where(~in_hill[last_in_idx:])[0]
            t_exit = (float(t_raw[last_in_idx + post_final[0]])
                      if len(post_final) else float(t_raw[-1]))
        else:
            # flyby / moon_crash: clip at first exit after first entry
            t_exit = t_first_hill_exit
    else:
        t_first_hill_exit = float(t_raw[-1])
        t_exit            = float(t_raw[-1])

    # Moon crash: clip at interpolated Moon-surface crossing.
    # Rust saves up to (and including) the first sub-surface step, so t_raw[-1]
    # can be inside the Moon.  We find the first inside-Moon index with argmax,
    # then interpolate back to the exact surface crossing.
    # Fallback: if Python's R_MOON_ND misses the transition due to floating-point
    # differences with Rust's r_moon_nd, clip at the second-to-last raw step so
    # at minimum the inside-Moon point is excluded.
    if outcome == "moon_crash":
        inside = r_moon_arr < R_MOON_ND
        if inside.any():
            first_in = int(np.argmax(inside))
            if first_in > 0:
                r0, r1  = r_moon_arr[first_in - 1], r_moon_arr[first_in]
                frac    = (R_MOON_ND - r0) / (r1 - r0)
                t_crash = float(t_raw[first_in - 1]
                                + frac * (t_raw[first_in] - t_raw[first_in - 1]))
            else:
                t_crash = float(t_raw[0])
            t_exit = t_crash   # confirmed crash: clip at Moon surface
        else:
            # Rust clipped trajectory just before Moon surface (segment-level crossing
            # so no saved step is inside the Moon).  t_raw[-1] is the last step before
            # the surface — use it if it's still inside the Hill sphere.
            t_crash_est = float(t_raw[-1])
            if t_crash_est < t_first_hill_exit:
                t_exit = t_crash_est
            # else: 2D-detection false positive — keep t_exit = t_first_hill_exit

    # Apply stride and clip to t_exit
    sl   = slice(None, None, args.stride)
    t    = t_raw[sl]; x = x_raw[sl]; y = y_raw[sl]; z = z_raw[sl]
    clip = t <= t_exit
    t = t[clip]; x = x[clip]; y = y[clip]; z = z[clip]
    if len(t) == 0:
        continue

    # ECI (for 3D HTML left panel)
    xi3, yi3, zi3 = rot_em_to_eci(x, y, z, t, R0_wsb)

    # EM-inertial in km (for GIF left panel)
    xi_em = (x * np.cos(t) - y * np.sin(t)) * L_KM
    yi_em = (x * np.sin(t) + y * np.cos(t)) * L_KM

    # Moon-centred EM-inertial in km (for right panel, both HTML and GIF)
    xi_mc = xi_em - X_M * np.cos(t) * L_KM
    yi_mc = yi_em - X_M * np.sin(t) * L_KM
    # Frame-invariant Moon-centred distance (km) — used to clip visual Moon penetration.
    r_mc_km = np.sqrt(xi_mc**2 + yi_mc**2)

    trajs[run_id] = dict(
        t=t,
        xi=xi3, yi=yi3, zi=zi3,
        xi_em=xi_em, yi_em=yi_em,
        xi_mc=xi_mc, yi_mc=yi_mc,
        r_mc_km=r_mc_km,
        t_hill=t_hill_raw,
        t_exit=t_exit,                        # SOI / crash / data-end gate for immediate hide
        t_first_hill_exit=t_first_hill_exit,  # first Hill departure (video cutoff reference)
    )

# Only trajectories with valid data (consistent trace indexing)
run_ids = [r for r in run_ids_all if r in trajs]
N_traj  = len(run_ids)

# Animation ends when the nominal trajectory's comet tail fades out completely
t_nom_end   = trajs[nominal_id]["t"][-1] if nominal_id in trajs else max(v["t"][-1] for v in trajs.values())
dt_nd       = args.step * 3600.0 / T_STAR
t_anim_full = np.arange(0, t_nom_end + TAIL_ND + dt_nd * 0.5, dt_nd)

# Both HTML and MP4 share the same truncated time range (0 → nominal's first Hill exit)
# so "Day X" labels in HTML and MP4 always refer to the same simulation time.
_t_vid_end  = (trajs[nominal_id]["t_first_hill_exit"]
               if nominal_id in trajs else t_nom_end)
_t_full_vid = t_anim_full[t_anim_full <= _t_vid_end + dt_nd * 0.5]
if len(_t_full_vid) == 0:
    _t_full_vid = t_anim_full[:1]

# Compute the variable-speed MP4 frame grid here (top-level, before HTML rendering)
# so HTML can be defined as a strict SUBSET of these time points.
# Guarantee: any "Day X" label in HTML maps to the exact same t_now as in the MP4.
_VID_TIME_SPLIT  = 0.70   # first this fraction of mission time …
_VID_FRAME_SPLIT = 0.267  # … gets only this fraction of frames  (fast outward leg, ×1.5 vs slow end)
_n_vid   = min(MAX_GIF_FRAMES, len(_t_full_vid))
_n_early = max(1, int(_n_vid * _VID_FRAME_SPLIT))
_n_late  = max(1, _n_vid - _n_early)
_t_split  = _t_vid_end * _VID_TIME_SPLIT
_tv_early = _t_full_vid[_t_full_vid <= _t_split]
_tv_late  = _t_full_vid[_t_full_vid >  _t_split]
_ei = np.linspace(0, len(_tv_early) - 1, min(_n_early, max(1, len(_tv_early))), dtype=int)
_li = np.linspace(0, len(_tv_late)  - 1, min(_n_late,  max(1, len(_tv_late))),  dtype=int)
t_anim_gif = (np.concatenate([_tv_early[_ei], _tv_late[_li]])
              if len(_tv_late) > 0 else _tv_early[_ei])

# HTML is a uniform subset of the SAME t_anim_gif points → identical "Day X" = identical t_now
html_idx    = np.linspace(0, len(t_anim_gif) - 1,
                          min(MAX_HTML_FRAMES, len(t_anim_gif)), dtype=int)
t_anim_html = t_anim_gif[html_idx]

# ECI axis range — computed from Hill-entering trajectories only.
# Miss trajectories now have T_PROP worth of data and would drift millions of km
# from Earth, completely blowing out the scene scale if included.
_traj_maxd = [
    float(np.max(np.sqrt(trajs[r]["xi"]**2 + trajs[r]["yi"]**2 + trajs[r]["zi"]**2)))
    for r in run_ids
    if len(trajs[r]["xi"]) > 0 and display_outcome_map.get(r) != "miss"
]
if not _traj_maxd:   # fallback if somehow all are miss
    _traj_maxd = [
        float(np.max(np.sqrt(trajs[r]["xi"]**2 + trajs[r]["yi"]**2 + trajs[r]["zi"]**2)))
        for r in run_ids if len(trajs[r]["xi"]) > 0
    ]
r_eci_lim = float(np.percentile(_traj_maxd, 90)) * 1.35 if _traj_maxd else R_HILL_KM * 2.0
r_eci_lim = max(r_eci_lim, R_HILL_KM * 2.0)
VIEW_SOI  = R_HILL_KM * 1.6

print(f"  {N_traj} trajectories  |  "
      f"captured={n_cap}  flyby={n_fly}  crash={n_cra}  miss={n_mis}")
print(f"  HTML: {N_traj} traj × {len(t_anim_html)} frames  |  "
      f"GIF: {N_traj} traj × {len(t_anim_gif)} frames")

# ---- Plotly: static geometry -------------------------------------------------
sx, sy, sz     = sphere_surface(R_EARTH, nu=30, nv=16)
t_ring         = np.linspace(0, 2 * np.pi, 200)
mxr, myr, mzr  = moon_em_to_eci(t_ring, R0_wsb)

phi  = np.linspace(0, 2 * np.pi, 200)
hs_x = (R_HILL_KM * np.cos(phi)).tolist()
hs_y = (R_HILL_KM * np.sin(phi)).tolist()

# Moon rendered as a layout shape so it is always above all trajectory traces
MOON_SHAPE = dict(
    type="circle",
    xref="x", yref="y",
    x0=-R_MOON, y0=-R_MOON, x1=R_MOON, y1=R_MOON,
    fillcolor="rgba(160,160,160,0.55)",
    line=dict(color="#AAAAAA", width=1),
    layer="above",
)

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "xy"}]],
    subplot_titles=["ECI Frame — Full Transfer",
                    "Moon-Centred — Hill Sphere Zoom"],
    column_widths=[0.56, 0.44],
    horizontal_spacing=0.06,
)

# Static: Earth sphere (3D left panel)
fig.add_trace(go.Surface(
    x=sx, y=sy, z=sz,
    colorscale=[[0, "#1A237E"], [0.5, COL_EARTH], [1, "#64B5F6"]],
    showscale=False, opacity=0.9,
    lighting=dict(ambient=0.6, diffuse=0.8, specular=0.4),
    name="Earth", hoverinfo="skip",
), row=1, col=1)

# Static: Moon orbit ring (3D left panel)
fig.add_trace(go.Scatter3d(
    x=mxr.tolist(), y=myr.tolist(), z=mzr.tolist(), mode="lines",
    line=dict(color="rgba(210,210,210,0.55)", width=2),
    name="Lunar orbit", showlegend=True, hoverinfo="skip",
), row=1, col=1)

# World point for the "Lunar orbit" MP4 label: rightmost point on the ring
# in ECI (maximum mxr).  From camera (0.9, -1.4, 0.7) this projects to the
# right edge of the ring ellipse, exactly where a label belongs.
_lbl_ring_x = float(mxr[int(np.argmax(mxr))])
_lbl_ring_y = float(myr[int(np.argmax(mxr))])
_lbl_ring_z = float(mzr[int(np.argmax(mxr))])

# Static: Hill sphere ring (2D right panel, dashed-dot)
fig.add_trace(go.Scatter(
    x=hs_x, y=hs_y, mode="lines",
    line=dict(color=COL_HILL, width=1.5, dash="dot"),
    name=f"Hill sphere ({R_HILL_KM/1e3:.0f}e3 km)", showlegend=True,
), row=1, col=2)

# ---- Static legend anchors ---------------------------------------------------
# One invisible trace per outcome added BEFORE the animated traces.
# They are never in ANIM_TRACES, so they survive empty-data frames and keep
# the legend entries visible throughout the entire animation.
_LEGEND_DEFS = [
    ("nominal",    "#FFD700", 6, f"Nominal"),
    ("captured",   "#1AE870", 3, f"Captured ({n_cap})"),
    ("flyby",      "#C77DFF", 2, f"Flyby ({n_fly})"),
    ("moon_crash", "#E63946", 2, f"Moon crash ({n_cra})"),
    ("miss",       "#666677", 1, f"Miss ({n_mis})"),
]
for _lg, _lc, _lw, _ll in _LEGEND_DEFS:
    fig.add_trace(go.Scatter3d(
        x=[1e15], y=[1e15], z=[1e15], mode="lines",
        line=dict(color=_lc, width=_lw),
        name=_ll, legendgroup=_lg, showlegend=True, hoverinfo="skip",
    ), row=1, col=1)

# ---- Animated traces ---------------------------------------------------------
# Layout per trajectory:  trail3d | dot3d | trail2d | dot2d
# Preceded by one Moon-dot (3D).  All showlegend=False — legend comes from anchors.
n_anim = len(fig.data)

mxi0, myi0, mzi0 = moon_em_to_eci(np.array([0.0]), R0_wsb)
fig.add_trace(go.Scatter3d(
    x=[float(mxi0[0])], y=[float(myi0[0])], z=[float(mzi0[0])],
    mode="markers+text", marker=dict(color=COL_MOON, size=7),
    text=["Moon"], textposition="top center",
    textfont=dict(color=COL_MOON, size=9),
    name="Moon", showlegend=False,
), row=1, col=1)

for run_id in run_ids:
    is_nom = (run_id == nominal_id)
    disp   = display_outcome_map.get(run_id, "miss")
    if is_nom:
        col = "#FFD700"; w3d = 8; w2d = 5.5; alpha = 1.0; ds3 = 10; ds2 = 14
    else:
        col   = OUTCOME_COLOR.get(disp, "#888888")
        w3d   = OUTCOME_WIDTH_3D.get(disp, 1)
        w2d   = OUTCOME_WIDTH_2D.get(disp, 0.8)
        alpha = OUTCOME_ALPHA.get(disp, 0.3)
        ds3   = 4; ds2 = 6
    lg = "nominal" if is_nom else disp

    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="lines",
        line=dict(color=col, width=w3d), opacity=alpha,
        legendgroup=lg, showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="markers",
        marker=dict(color=col, size=ds3), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines",
        line=dict(color=col, width=w2d), opacity=alpha,
        legendgroup=lg, showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="markers",
        marker=dict(color=col, size=ds2), showlegend=False,
    ), row=1, col=2)

ANIM_TRACES = list(range(n_anim, n_anim + 1 + 4 * N_traj))

def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)

# ---- Build Plotly frames -----------------------------------------------------
print(f"Building {len(t_anim_html)} Plotly frames "
      f"({N_traj} traj × comet tail {args.tail:.0f} d) ...",
      end="", flush=True)
frames       = []
slider_steps = []

for fi, t_now in enumerate(t_anim_html):
    if fi % 20 == 0:
        print(".", end="", flush=True)
    t_days = t_now * T_STAR / 86400.0

    m_arr = moon_em_to_eci(np.array([t_now]), R0_wsb)
    m_xi  = float(m_arr[0][0])
    m_yi  = float(m_arr[1][0])
    m_zi  = float(m_arr[2][0])

    data: list = [go.Scatter3d(x=[m_xi], y=[m_yi], z=[m_zi],
                               text=["Moon"], textposition="top center")]

    for run_id in run_ids:
        s    = trajs[run_id]
        disp = display_outcome_map.get(run_id, "miss")
        if disp not in ("nominal", "miss") and t_now > s["t_exit"]:
            mask = np.zeros(len(s["t"]), dtype=bool)
        else:
            mask = (s["t"] > t_now - TAIL_ND) & (s["t"] <= t_now)

        # Downsample HTML trail to cap file size (MAX_HTML_TRAIL_PTS points per traj)
        xi_h = s["xi"][mask]; yi_h = s["yi"][mask]; zi_h = s["zi"][mask]
        xmc_h = s["xi_mc"][mask]; ymc_h = s["yi_mc"][mask]
        if len(xi_h) > MAX_HTML_TRAIL_PTS:
            _hi = np.linspace(0, len(xi_h) - 1, MAX_HTML_TRAIL_PTS, dtype=int)
            xi_h = xi_h[_hi]; yi_h = yi_h[_hi]; zi_h = zi_h[_hi]
            xmc_h = xmc_h[_hi]; ymc_h = ymc_h[_hi]

        # Trail 3D
        data.append(go.Scatter3d(
            x=xi_h.tolist(), y=yi_h.tolist(), z=zi_h.tolist(),
        ))
        # Dot 3D
        if mask.any():
            data.append(go.Scatter3d(
                x=[float(s["xi"][mask][-1])],
                y=[float(s["yi"][mask][-1])],
                z=[float(s["zi"][mask][-1])],
            ))
        else:
            data.append(go.Scatter3d(x=[], y=[], z=[]))
        # Trail 2D
        data.append(go.Scatter(x=xmc_h.tolist(), y=ymc_h.tolist()))
        # Dot 2D
        if mask.any():
            data.append(go.Scatter(
                x=[float(s["xi_mc"][mask][-1])],
                y=[float(s["yi_mc"][mask][-1])],
            ))
        else:
            data.append(go.Scatter(x=[], y=[]))

    frames.append(go.Frame(
        data=data, traces=ANIM_TRACES, name=str(fi),
        layout=go.Layout(title_text=f"Sensitivity Ensemble — t = {t_days:.1f} days"),
    ))
    slider_steps.append(dict(
        args=[[str(fi)], {"frame": {"duration": 0}, "mode": "immediate",
                          "transition": {"duration": 0}}],
        label=f"{t_days:.1f}d", method="animate",
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
            f"WSB Sensitivity Ensemble — N={n_tot}  |  "
            f"captured={n_cap}  flyby={n_fly}  crash={n_cra}  miss={n_mis}<br>"
            f"<sup>{title_meta}</sup>"
        ),
        x=0.01, xanchor="left", font=dict(size=12),
    ),
    legend=dict(x=1.01, y=1, font=dict(size=10),
                bgcolor="rgba(15,15,25,0.7)"),
    height=720,
    margin=dict(l=0, r=120, t=95, b=80),
    updatemenus=[play_pause_buttons(int(1000 / args.fps))],
    sliders=[dark_slider(slider_steps)],
    shapes=[MOON_SHAPE],
)

fig.write_html(str(html_out), auto_play=False)
print(f"Saved {html_out}")

# ---- GIF / MP4 export via PyVista — same 3-D frame as plot_wsb_multi_diverse --
# Left  : 3-D ECI — textured Earth/Moon spheres, tube trails, auto-zoom camera.
# Right : orthographic Moon-centred Hill-sphere panel.
# Colours: outcome-based (captured / moon_crash / escaped / nominal).
if args.gif or args.mp4 or args.preview:
    import shutil as _shutil, glob as _glob, os as _os
    import tempfile as _tmpfile, subprocess as _subp
    import urllib.request as _urlreq
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
        for _pat in [
            r"C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg*\bin\ffmpeg.exe",
        ]:
            hits = _glob.glob(_pat)
            if hits:
                return hits[0]
        return None

    _ffmpeg_path = _find_ffmpeg()

    # ── Colours (0–1 floats) ──────────────────────────────────────────────────
    _BG3D   = (22/255,  22/255,  45/255)
    _MCOL   = (195/255, 195/255, 195/255)   # Moon/orbit ring
    _ECOL   = ( 21/255, 101/255, 192/255)   # Earth solid fallback
    _HCOL   = (198/255, 130/255,  30/255)   # Hill sphere ring

    # Per-outcome trajectory colours (RGB 0–1)
    _TRAJ_COL = {
        "nominal":    (255/255, 215/255,   0/255),   # #FFD700  gold
        "captured":   ( 26/255, 232/255, 112/255),   # #1AE870  green
        "flyby":      (210/255, 150/255, 255/255),   # #D296FF  bright purple
        "moon_crash": (255/255,  80/255,  90/255),   # #FF505A  bright red
        "miss":       (120/255, 120/255, 140/255),   # #78788C  lighter grey
    }
    _DOT_COL = {k: v for k, v in _TRAJ_COL.items()}
    _TRAJ_LW = {"nominal": 9.0, "captured": 7.5, "flyby": 6.0, "moon_crash": 7.0, "miss": 4.5}
    _TRAJ_A  = {"nominal": 240, "captured": 220, "flyby": 200, "moon_crash": 200, "miss": 100}

    # ── Texture download ──────────────────────────────────────────────────────
    TEX_DIR  = OUT_DIR / "textures"
    TEX_DIR.mkdir(exist_ok=True)
    _EARTH_URL = "https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg"
    _MOON_URL  = "https://www.solarsystemscope.com/textures/download/2k_moon.jpg"
    _EARTH_JPG = TEX_DIR / "earth_daymap.jpg"
    _MOON_JPG  = TEX_DIR / "moon_1k.jpg"

    def _dl_tex(url, path):
        if path.exists():
            return True
        print(f"  Downloading {path.name} ...", end="", flush=True)
        try:
            _urlreq.urlretrieve(url, str(path)); print(" done."); return True
        except Exception as exc:
            print(f" FAILED ({exc}) — solid fallback"); return False

    _earth_ok  = _dl_tex(_EARTH_URL, _EARTH_JPG)
    _moon_ok   = _dl_tex(_MOON_URL,  _MOON_JPG)
    _earth_tex = pv.read_texture(str(_EARTH_JPG)) if _earth_ok else None
    _moon_tex  = pv.read_texture(str(_MOON_JPG))  if _moon_ok  else None

    # ── PolyData helpers ──────────────────────────────────────────────────────
    def _line_poly(pts_arr):
        n = len(pts_arr)
        if n < 2:
            return pv.PolyData()
        poly        = pv.PolyData()
        poly.points = np.asarray(pts_arr, dtype=np.float64)
        poly.lines  = np.array([n] + list(range(n)), dtype=np.int64)
        return poly

    def _ring_poly(radius, z=0.0, n=200):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([radius * np.cos(phi), radius * np.sin(phi), np.full(n, z)])
        poly        = pv.PolyData()
        poly.points = pts.astype(np.float64)
        poly.lines  = np.array([n + 1] + list(range(n)) + [0], dtype=np.int64)
        return poly

    def _grid_poly(extent, step):
        vals = np.arange(-extent, extent + step * 0.01, step)
        pts, cells, pid = [], [], 0
        for v in vals:
            if abs(v) > extent + 1:
                continue
            pts += [[-extent, v, 0.0], [extent, v, 0.0]]
            cells += [2, pid, pid + 1]; pid += 2
            pts += [[v, -extent, 0.0], [v, extent, 0.0]]
            cells += [2, pid, pid + 1]; pid += 2
        poly        = pv.PolyData()
        poly.points = np.array(pts, dtype=np.float64)
        poly.lines  = np.array(cells, dtype=np.int64)
        return poly

    # ── Camera / scene constants ──────────────────────────────────────────────
    _R_EARTH_VIS    = max(R_EARTH * 3.2, r_eci_lim * 0.016)
    _R_MOON_ECI_VIS = _R_EARTH_VIS * 0.6
    # Un-normalized direction matches plot_wsb_multi_diverse exactly:
    #   _C3_POS = (0.9*_eye_s, -1.4*_eye_s, 0.7*_eye_s) → |r| ≈ 1.806*_eye_s
    # Using 1.35 (vs multi_diverse's 1.05) gives extra room to see the outward leg.
    _EYE_DIR        = np.array([0.9, -1.4, 0.7])   # NOT normalised
    _eye_s_full     = r_eci_lim * 1.5   # wide enough to watch the outward leg  ← tune this
    _eye_s_near     = _eye_s_full * 0.70  # tighter zoom at start/end            ← tune this ratio
    _C2_POS         = (0.0, 0.0, VIEW_SOI * 10.0)
    _C2_FOC         = (0.0, 0.0, 0.0)
    # 90° CW rotation: world +Y goes right, world +X goes down (matches ECI x/y screen alignment)
    _C2_UP    = (-1.0, 0.0, 0.0)
    _C2_UP_2D = (-1.0, 0.0)
    _g_step         = 400_000
    _soi_step       = round(R_HILL_KM / 2.0, -3)
    _VID_W, _VID_H  = 1080, 1920
    # Stacked layout fractions (VTK y=0 is bottom, y=1 is top):
    #   top (1 - _STK_SOI_END) fraction → ECI 3D panel (full width)
    #   middle _STK_SOI_FRAC fraction   → SOI 2D panel (full width)
    #   bottom _STK_STATS_FRAC fraction → stats bar (PIL overlay only, no renderer)
    _STK_STATS_FRAC = 0.0
    _STK_SOI_FRAC   = 0.35
    _STK_SOI_END    = _STK_STATS_FRAC + _STK_SOI_FRAC   # 0.35 — top of SOI = bottom of ECI
    _OFFSCREEN      = np.array([[r_eci_lim * 100, 0.0, 0.0]], dtype=np.float64)
    _DUMMY_PTS      = np.zeros((2, 3), dtype=np.float64)
    _DUMMY_LINES    = np.array([2, 0, 1], dtype=np.int64)

    # Camera zoom schedule (non-dimensional time):
    #   0 → _t_zoom_in   : hold at full view (full outward + deep space leg)
    #   _t_zoom_in → _t_vid_end : smooth zoom IN to near (Moon approach / capture)
    # Start zoom 20% of the Hill-approach window before actual Hill entry so the
    # camera is already tightening as the trajectories converge toward the Moon.
    _t_hill_nom = (trajs[nominal_id]["t_hill"] if nominal_id in trajs else t_nom_end * 0.65)
    _t_zoom_in  = _t_hill_nom * 0.80

    def _eye_pos(t_now: float):
        if t_now < _t_zoom_in:
            s = _eye_s_full
        elif t_now <= _t_vid_end:
            frac = _smoothstep(
                min(1.0, (t_now - _t_zoom_in) / max(_t_vid_end - _t_zoom_in, dt_nd)))
            s = _eye_s_full + (_eye_s_near - _eye_s_full) * frac
        else:
            s = _eye_s_near
        ev = _EYE_DIR * s
        return (float(ev[0]), float(ev[1]), float(ev[2]))

    _EARTH_PHYS_DEG_PER_ND = 360.0 * T_STAR / 86164.0
    # Use absolute physical orientation: rotation = deg_per_ND * t_now (mod 360).
    # Variable-speed frames are 1.5–9.5 h apart (22–142° per step), both well
    # below the 180° Nyquist limit, so no aliasing and no cap needed.

    # ── Video trajectory subset — stratified by outcome ───────────────────────
    # Guaranteed composition: ≥10 moon_crash, ≥20 flyby, ≥10 captured, rest
    # filled with miss up to --vid-n cap (0 = no cap → include all of each type).
    # Nominal is always appended last (drawn on top).
    _perturbed_vid = [r for r in run_ids if r != nominal_id]
    _by_oc: dict = {}
    for _r in _perturbed_vid:
        _k = display_outcome_map.get(_r, "miss")
        _by_oc.setdefault(_k, []).append(_r)

    _MIN_PER_OC = [("moon_crash", 10), ("flyby", 20), ("captured", 10)]

    _strat: list = []
    for _oc, _mn in _MIN_PER_OC:
        _pool = _by_oc.get(_oc, [])
        _n_tk = min(_mn, len(_pool))
        if _n_tk > 0:
            _ii = np.linspace(0, len(_pool) - 1, _n_tk, dtype=int)
            _strat.extend(_pool[int(i)] for i in _ii)

    _strat_set = set(_strat)
    _miss_all  = [r for r in _by_oc.get("miss", []) if r not in _strat_set]

    if args.vid_n > 0:
        _miss_budget = max(0, args.vid_n - 1 - len(_strat))   # -1 for nominal
        if _miss_budget < len(_miss_all):
            _mi = np.linspace(0, len(_miss_all) - 1, _miss_budget, dtype=int)
            _miss_sel = [_miss_all[int(i)] for i in _mi]
        else:
            _miss_sel = _miss_all
    else:
        _miss_sel = _miss_all

    vid_ids = _strat + _miss_sel + [nominal_id]
    _vc = {k: sum(1 for r in vid_ids if display_outcome_map.get(r) == k)
           for k in ("moon_crash", "flyby", "captured", "miss")}
    print(f"  Video trajectories: {len(vid_ids)}  "
          f"crash={_vc['moon_crash']}  flyby={_vc['flyby']}  "
          f"cap={_vc['captured']}  miss={_vc['miss']}  nom=1  (HTML has {len(run_ids)})")

    # ── Plotter factory — pre-allocates all trajectory actors ─────────────────
    # Returns (plotter, pd) where pd["tr3/dt3/tr2/dt2"][run_id] are PolyData
    # objects whose data is mutated in-place each frame — no add_mesh overhead.
    def _build_pl(w, h, hifi=False):
        pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[w, h],
                        line_smoothing=True, point_smoothing=True)
        pl.enable_anti_aliasing("ssaa")

        # Stacked layout: ECI on top (full width), SOI below (full width).
        # Stats bar is PIL-only — no renderer needed for it.
        pl.renderers[0].SetViewport(0.0, _STK_SOI_END,    1.0, 1.0)
        pl.renderers[1].SetViewport(0.0, _STK_STATS_FRAC, 1.0, _STK_SOI_END)

        # LEFT — 3-D ECI (static geometry) ────────────────────────────────────
        pl.subplot(0, 0)
        pl.set_background(_BG3D, all_renderers=False)

        _sph_t = 180 if hifi else 90
        _sph_p = 90  if hifi else 45
        _es = pv.Sphere(radius=_R_EARTH_VIS,
                        theta_resolution=_sph_t, phi_resolution=_sph_p,
                        start_phi=0.1, end_phi=179.9)
        # Override UV with explicit equirectangular formula.
        # vtkSphereSource auto-UV and texture_map_to_sphere() both conflict with
        # standard equirectangular textures: VTK flips loaded JPEGs vertically
        # (origin at bottom), so north-pole geometry must map to v=0.
        # Formula: u=0 at 180°W date line, v=0 at north pole after VTK flip.
        _epts = _es.points
        _er   = np.sqrt(np.sum(_epts**2, axis=1))
        _elat = np.arcsin(np.clip(_epts[:, 2] / np.maximum(_er, 1e-10), -1.0, 1.0))
        _elon = np.arctan2(_epts[:, 1], _epts[:, 0])
        _es.active_texture_coordinates = np.column_stack([
            (_elon + np.pi) / (2.0 * np.pi),           # u ∈ [0,1], 0 at 180°W
            (_elat + np.pi / 2.0) / np.pi,               # v ∈ [0,1], 0 at south pole, 1 at north pole
        ]).astype(np.float32)
        if _earth_tex is not None:
            pl._earth_actor = pl.add_mesh(_es, texture=_earth_tex,
                                          smooth_shading=True, lighting=False, name="earth")
        else:
            pl._earth_actor = pl.add_mesh(_es, color=_ECOL,
                                          smooth_shading=True, opacity=0.92, name="earth")

        pl.add_mesh(_line_poly(np.column_stack([mxr, myr, mzr])),
                    color=(215/255, 215/255, 215/255), opacity=1.0, line_width=2.5)
        pl.add_mesh(_grid_poly(r_eci_lim, _g_step),
                    color=(80/255, 80/255, 80/255), opacity=0.18, line_width=4.5)
        _ax_r   = r_eci_lim * 0.85
        _ax_col = (160/255, 160/255, 160/255)
        for _ap, _an in [
            (np.array([[-_ax_r, 0, 0], [_ax_r, 0, 0]]),           "ax_x"),
            (np.array([[0, -_ax_r, 0], [0, _ax_r, 0]]),            "ax_y"),
            (np.array([[0, 0, -_ax_r*0.4], [0, 0, _ax_r*0.4]]),   "ax_z"),
        ]:
            pl.add_mesh(_line_poly(_ap), color=_ax_col, line_width=6.5,
                        opacity=0.30, name=_an)
        # Tick marks at every _g_step — copied from wsb_multi_diverse
        _tk_h   = r_eci_lim * 0.012
        _tk_pts, _tk_cells, _tkp = [], [], 0
        _td = _g_step
        while _td < _ax_r:
            for _sign in (+1, -1):
                _tk_pts  += [[_sign * _td, -_tk_h, 0], [_sign * _td, _tk_h, 0]]
                _tk_cells += [2, _tkp, _tkp + 1]; _tkp += 2
                _tk_pts  += [[-_tk_h, _sign * _td, 0], [_tk_h, _sign * _td, 0]]
                _tk_cells += [2, _tkp, _tkp + 1]; _tkp += 2
                if _td <= _ax_r * 0.4:
                    _tk_pts  += [[-_tk_h, 0, _sign * _td], [_tk_h, 0, _sign * _td]]
                    _tk_cells += [2, _tkp, _tkp + 1]; _tkp += 2
            _td += _g_step
        if _tk_pts:
            _tkpoly        = pv.PolyData()
            _tkpoly.points = np.array(_tk_pts, dtype=np.float64)
            _tkpoly.lines  = np.array(_tk_cells, dtype=np.int64)
            pl.add_mesh(_tkpoly, color=_ax_col, opacity=0.30,
                        line_width=6.5, name="ax_ticks")
        _rng_s = np.random.default_rng(42)
        _sph_s = np.arccos(_rng_s.uniform(-1, 1, 1000))
        _sth_s = _rng_s.uniform(0, 2 * np.pi, 1000)
        _sr_s  = r_eci_lim * 4.5
        pl.add_mesh(pv.PolyData(np.column_stack([
            _sr_s * np.sin(_sph_s) * np.cos(_sth_s),
            _sr_s * np.sin(_sph_s) * np.sin(_sth_s),
            _sr_s * np.cos(_sph_s) * 0.6,
        ])), style="points", point_size=3.5, color="white", opacity=0.55)

        # Pre-allocate trajectory actors (added ONCE — updated in-place later)
        _tr3, _dt3 = {}, {}
        for run_id in vid_ids:
            is_nom  = (run_id == nominal_id)
            outcome = "nominal" if is_nom else display_outcome_map.get(run_id, "miss")
            _c3f = _TRAJ_COL[outcome]; _lw3 = _TRAJ_LW[outcome]
            tp = pv.PolyData()
            tp.points = _DUMMY_PTS.copy(); tp.lines = _DUMMY_LINES.copy()
            tp.point_data["colors"] = np.zeros((2, 4), dtype=np.uint8)
            pl.add_mesh(tp, scalars="colors", rgba=True, line_width=_lw3)
            _tr3[run_id] = tp
            dp = pv.PolyData(_OFFSCREEN.copy())
            pl.add_mesh(dp, style="points", point_size=_lw3 * 3.0,
                        color=_DOT_COL[outcome], render_points_as_spheres=True)
            _dt3[run_id] = dp

        pl.camera.position    = _eye_pos(0.0)
        pl.camera.focal_point = (0.0, 0.0, 0.0)
        pl.camera.up          = (0.0, 0.0, 1.0)
        pl.camera_set = True

        # RIGHT — Moon-centred orthographic (static geometry) ─────────────────
        pl.subplot(0, 1)
        pl.set_background(_BG3D, all_renderers=False)
        pl.enable_parallel_projection()

        _soi_ext = VIEW_SOI * 1.15
        pl.add_mesh(_grid_poly(_soi_ext, _soi_step),
                    color=(80/255, 80/255, 80/255), opacity=0.18, line_width=4.5)
        pl.add_mesh(_line_poly([[-_soi_ext, 0, 0], [_soi_ext, 0, 0]]),
                    color=(150/255, 150/255, 150/255), opacity=0.50, line_width=2.0)
        pl.add_mesh(_line_poly([[0, -_soi_ext, 0], [0, _soi_ext, 0]]),
                    color=(150/255, 150/255, 150/255), opacity=0.50, line_width=2.0)
        pl.add_mesh(_ring_poly(R_HILL_KM), color=_HCOL, line_width=2.5)

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
            pl.add_mesh(_ms2, color=_MCOL, opacity=0.85,
                        smooth_shading=True, name="moon_soi")

        # Pre-allocate SOI trajectory actors
        _tr2, _dt2 = {}, {}
        _offscreen_2d = np.array([[VIEW_SOI * 100, 0.0, 0.0]], dtype=np.float64)
        for run_id in vid_ids:
            is_nom  = (run_id == nominal_id)
            outcome = "nominal" if is_nom else display_outcome_map.get(run_id, "miss")
            _c2f = _TRAJ_COL[outcome]; _lw2 = _TRAJ_LW[outcome] * 0.7
            tp2 = pv.PolyData()
            tp2.points = _DUMMY_PTS.copy(); tp2.lines = _DUMMY_LINES.copy()
            tp2.point_data["colors"] = np.zeros((2, 4), dtype=np.uint8)
            pl.add_mesh(tp2, scalars="colors", rgba=True, line_width=_lw2)
            _tr2[run_id] = tp2
            dp2 = pv.PolyData(_offscreen_2d.copy())
            pl.add_mesh(dp2, style="points", point_size=_lw2 * 4.5,
                        color=_DOT_COL[outcome], render_points_as_spheres=True)
            _dt2[run_id] = dp2

        pl.camera.position       = _C2_POS
        pl.camera.focal_point    = _C2_FOC
        pl.camera.up             = _C2_UP
        pl.camera.parallel_scale = VIEW_SOI
        pl.camera_set = True

        pd = dict(tr3=_tr3, dt3=_dt3, tr2=_tr2, dt2=_dt2,
                  offscreen_2d=_offscreen_2d,
                  r_moon_eci_vis=_R_MOON_ECI_VIS,
                  r_moon_soi_vis=_ms2_r)
        return pl, pd

    # Fixed screen position for "Lunar orbit" label — computed once at frame 0
    # from the initial wide-shot camera via _w2px_eci, then frozen for all frames.
    # ── PIL HUD overlay ───────────────────────────────────────────────────────
    def _overlay_hud(arr, t_days, fi, pl):
        from PIL import ImageDraw as _PILDraw, ImageFont as _PILFont
        img  = _PILImage.fromarray(arr).convert("RGBA")
        W, H = img.size
        _sc  = W / 1920          # base scale matched to plot_wsb_multi_diverse
        _fs  = max(36, int(30 * _sc))

        def _load_fnt(sz):
            for nm in ("cour.ttf", "consola.ttf", "lucon.ttf"):
                try:
                    return _PILFont.truetype(nm, sz)
                except Exception:
                    pass
            return None

        _fnt    = _load_fnt(_fs)
        _fnt_sm = _load_fnt(max(30, int(26 * _sc)))
        _kw     = {"font": _fnt}    if _fnt    else {}
        _kw_sm  = {"font": _fnt_sm} if _fnt_sm else {}
        draw    = _PILDraw.Draw(img)

        # SOI panel geometry in PIL coords (computed early — used by legend and SOI drawing).
        _soi_h_pil   = int(_STK_SOI_FRAC * H)
        _soi_top_pil = int((1.0 - _STK_SOI_END) * H)   # top of SOI in PIL (y=0 at top)
        _soi_cy_pil  = _soi_top_pil + _soi_h_pil // 2
        _ar_soi      = W / max(1, _soi_h_pil)

        # Time label — top-centre of ECI panel (full width)
        _ttl = f"Sensitivity sweep   Day {t_days:.1f}"
        try:
            _ttl_w = _fnt.getlength(_ttl) if _fnt else len(_ttl) * int(_fs * 0.58)
        except Exception:
            _ttl_w = len(_ttl) * int(_fs * 0.58)
        _ttl_y = int(12 * _sc)
        draw.text((max(4, int((W - _ttl_w) // 2)), _ttl_y),
                  _ttl, fill=(220, 220, 220, 240), **_kw)
        # Subtitle — sample count
        _n_shown = len(vid_ids)
        _sub = (f"(showing 200 of {_stats_n:,} samples)"
                if _stats_n else f"(showing {_n_shown:,} samples)")
        try:
            _sub_w = _fnt_sm.getlength(_sub) if _fnt_sm else len(_sub) * int(max(30, int(26 * _sc)) * 0.58)
        except Exception:
            _sub_w = len(_sub) * int(max(30, int(26 * _sc)) * 0.58)
        draw.text((max(4, int((W - _sub_w) // 2)), _ttl_y + _fs + int(4 * _sc)),
                  _sub, fill=(160, 160, 190, 200), **_kw_sm)
        # SOI panel title — top-centre of SOI strip
        _soi_ttl = "Moon-Centred zoom"
        try:
            _soi_ttl_w = _fnt.getlength(_soi_ttl) if _fnt else len(_soi_ttl) * int(_fs * 0.58)
        except Exception:
            _soi_ttl_w = len(_soi_ttl) * int(_fs * 0.58)
        draw.text((max(4, int((W - _soi_ttl_w) // 2)), _soi_top_pil + int(8 * _sc)),
                  _soi_ttl, fill=(220, 220, 220, 240), **_kw)

        # ── Axis label style — wsb_multi_diverse exact copy ─────────────────
        _grey    = (220, 220, 220, 240)
        _grey_sm = (175, 175, 175, 210)
        _ax_r    = r_eci_lim * 0.85
        _kw_ax   = _kw
        _kw_ax_sm = _kw_sm

        # VTK world→screen for left panel
        def _w2px_eci(x, y, z):
            ren = pl.renderers[0]
            ren.SetWorldPoint(x, y, z, 1.0)
            ren.WorldToDisplay()
            sx, sy = ren.GetDisplayPoint()[:2]
            return int(round(sx)), int(round(H - sy))

        # Analytical parallel projection for the SOI panel.
        # SOI renderer viewport: [0, _STK_STATS_FRAC, 1, _STK_SOI_END] (full width).
        # _soi_h_pil / _soi_top_pil / _soi_cy_pil / _ar_soi computed above.
        # Camera up = _C2_UP_2D = (ux, uy); right = (uy, -ux).
        # half-width world = VIEW_SOI * _ar_soi; half-height world = VIEW_SOI.
        _ux_soi, _uy_soi = _C2_UP_2D
        def _w2px_soi(wx, wy):
            screen_right = wx * _uy_soi - wy * _ux_soi   # component along (uy, -ux)
            screen_up    = wx * _ux_soi + wy * _uy_soi   # component along (ux,  uy)
            sx = int(round(W / 2 + (W / 2) * screen_right / (VIEW_SOI * _ar_soi)))
            sy = int(round(_soi_cy_pil - (_soi_h_pil // 2) * screen_up / VIEW_SOI))
            return sx, sy
        _Lmax = W - int(6 * _sc)   # ECI now spans full width

        # ±X / ±Y / +Z axis tip labels
        for wpt, lbl, ox, oy in [
            (( _ax_r,  0,           0),   "+X",  4, -9),
            ((-_ax_r,  0,           0),   "-X", -22, -9),
            (( 0,      _ax_r,       0),   "+Y",  4, -9),
            (( 0,     -_ax_r,       0),   "-Y",  4,  4),
            (( 0,      0,  _ax_r * 0.4), "+Z",  4, -9),
        ]:
            try:
                px, py = _w2px_eci(*wpt)
                if px + ox < _Lmax:
                    draw.text((px + ox, py + oy), lbl, fill=_grey, **_kw_ax)
                    if lbl == "-Y":
                        try:
                            _ux, _uy = _w2px_eci(50_000, -1_400_000, 0)
                            draw.text((_ux + int(6 * _sc), _uy - int(_fnt_sm.size * 0.5
                                                                      if hasattr(_fnt_sm, 'size')
                                                                      else 14)),
                                      "×10³ km", fill=_grey_sm, **_kw_ax_sm)
                        except Exception:
                            pass
            except Exception:
                pass

        # Distance numbers along X, Y, Z at every _g_step
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
                            draw.text((tx, ty), lbl, fill=_grey_sm, **_kw_ax_sm)
                    except Exception:
                        pass
                _td += _g_step

        # SOI panel: distance tick marks + numbers along X (vertical) and Y (horizontal).
        # After 90° CW rotation: world +X goes DOWN on screen, world +Y goes RIGHT.
        # x-axis is vertical → ticks are horizontal; y-axis is horizontal → ticks are vertical.
        _tk = int(5 * _sc)
        for _axis in ("x", "y"):
            _td = _soi_step
            while _td < VIEW_SOI * 0.92:
                lbl = f"{int(_td / 1000):g}"
                for _sign in (+1, -1):
                    if _axis == "x":
                        # x-axis is now vertical — draw horizontal tick marks, label to the right
                        px, py = _w2px_soi(_sign * _td, 0)
                        draw.line([(px - _tk, py), (px + _tk, py)],
                                  fill=_grey_sm, width=1)
                        draw.text((px + int(6*_sc), py - int(8*_sc)),
                                  lbl, fill=_grey_sm, **_kw_ax_sm)
                    else:
                        # y-axis is now horizontal — draw vertical tick marks, label below
                        px, py = _w2px_soi(0, _sign * _td)
                        draw.line([(px, py - _tk), (px, py + _tk)],
                                  fill=_grey_sm, width=1)
                        draw.text((px - int(14*_sc), py + int(6*_sc)),
                                  lbl, fill=_grey_sm, **_kw_ax_sm)
                _td += _soi_step

        # ×10³ km unit label — after CW rotation the -Y axis is at the left edge (horizontal axis)
        _sy_end = -_soi_step * max(1, int(VIEW_SOI * 0.95 / _soi_step))
        try:
            _pu, _pv2 = _w2px_soi(0, -_sy_end)
            draw.text((_pu + int(6 * _sc), _pv2 - int(20 * _sc)),
                      "×10³ km", fill=_grey_sm, **_kw_ax_sm)
        except Exception:
            pass

        # ── ECI / SOI separator ─────────────────────────────────────────────
        draw.line([(0, _soi_top_pil), (W, _soi_top_pil)], fill=(60, 60, 100, 255), width=2)

        # ── Floating horizontal legend at ECI/SOI separator ──────────────────
        # No background panel — transparent overlay straddling the junction line.
        # Each outcome: color swatch + label + percentage + count from MC run.
        _fnt_lg  = _load_fnt(max(30, int(25 * _sc)))
        _fnt_lg2 = _load_fnt(max(24, int(20 * _sc)))
        _kw_lg   = {"font": _fnt_lg}  if _fnt_lg  else {}
        _kw_lg2  = {"font": _fnt_lg2} if _fnt_lg2 else {}

        _slbl_map = {"captured": "Captured", "flyby": "Flyby",
                     "moon_crash": "Moon crash", "escaped": "Miss"}
        _scol_map = {"captured": "captured", "flyby": "flyby",
                     "moon_crash": "moon_crash", "escaped": "miss"}

        _leg_items = [("nominal", "Nominal", "1 sample")]
        if _stats_fracs is not None:
            for _lk in ("captured", "flyby", "moon_crash", "escaped"):
                _lf = _stats_fracs.get(_lk, 0.0)
                if _lf < 5e-4:
                    continue
                _lcnt     = int(round(_stats_n * _lf)) if _stats_n else 0
                _lcnt_str = f"{_lcnt/1000:.1f}k" if _lcnt >= 1000 else str(_lcnt)
                _leg_items.append((_lk, _slbl_map[_lk], f"{_lf:.0%} ({_lcnt_str})"))

        _n_leg     = len(_leg_items)
        _lg_sc     = W / 1080            # portrait-fixed scale for legend pixel dims
        _item_w    = W // max(1, _n_leg) # fill full width evenly
        _sw_rect_h = max(int(13 * _lg_sc), 9)
        _sw_rect_w = max(int(30 * _lg_sc), 20)
        _sw_pad    = max(int(7 * _lg_sc), 5)
        _lbl_top   = _soi_top_pil - int(58 * _lg_sc)
        _stat_y    = _soi_top_pil - int(26 * _lg_sc)
        _swatch_y  = (_lbl_top + _stat_y) // 2

        _item_x_nudge = { "escaped": int(18 * _lg_sc)}

        for _ii, (_ik, _ilbl, _istat) in enumerate(_leg_items):
            _icol_f = (_TRAJ_COL["nominal"] if _ik == "nominal"
                       else _TRAJ_COL[_scol_map[_ik]])
            _icol_i = tuple(int(c * 255) for c in _icol_f) + (255,)
            _icol_d = (_icol_i[0], _icol_i[1], _icol_i[2], 195)

            _ix0 = _item_w * _ii + int(8 * _lg_sc) + _item_x_nudge.get(_ik, 0)

            if _ik == "nominal":
                draw.line([(_ix0, _swatch_y), (_ix0 + _sw_rect_w, _swatch_y)],
                          fill=_icol_i, width=max(5, int(5 * _lg_sc)))
            else:
                draw.rectangle([(_ix0, _swatch_y - _sw_rect_h // 2),
                                 (_ix0 + _sw_rect_w, _swatch_y + _sw_rect_h // 2)],
                                fill=_icol_i)

            _tx = _ix0 + _sw_rect_w + _sw_pad
            draw.text((_tx, _lbl_top), _ilbl, fill=(220, 220, 220, 255), **_kw_lg)
            draw.text((_tx, _stat_y),  _istat, fill=_icol_d,              **_kw_lg2)

        return np.array(img.convert("RGB"))

    # ── Per-frame update — in-place PolyData mutation, no add_mesh overhead ────
    # Trajectories were added to the scene once in _build_pl. Here we only
    # mutate their points/lines/scalars and call Modified() so VTK re-renders.
    # Only the moving Moon sphere still uses add_mesh (1 call/frame).
    def _upd_pv(pl, pd, fi, t_arr):
        t_now = t_arr[fi]
        _R_ECI_MOON_VIS_KM = pd["r_moon_eci_vis"]   # visual Moon radius in ECI panel (km)
        _R_SOI_MOON_VIS_KM = pd["r_moon_soi_vis"]   # visual Moon radius in SOI panel (km)

        # LEFT panel ───────────────────────────────────────────────────────────
        pl.subplot(0, 0)
        c3 = _eye_pos(t_now)
        pl.camera.position    = c3
        pl.camera.focal_point = (0.0, 0.0, 0.0)
        pl.camera.up          = (0.0, 0.0, 1.0)
        pl.camera_set = True

        # Earth rotation (actor transform — free, no add_mesh)
        _ea = getattr(pl, "_earth_actor", None)
        if _ea is not None:
            _rot = (_EARTH_PHYS_DEG_PER_ND * t_now) % 360.0
            try:
                _ea.orientation = (0.0, 0.0, _rot)
            except AttributeError:
                _ea.SetOrientation(0.0, 0.0, _rot)

        # Moon sphere — moves each frame, so one add_mesh per frame is fine
        m_arr = moon_em_to_eci(np.array([t_now]), R0_wsb)
        mx3, my3, mz3 = float(m_arr[0][0]), float(m_arr[1][0]), float(m_arr[2][0])
        _ms = pv.Sphere(radius=_R_MOON_ECI_VIS, center=(mx3, my3, mz3),
                        theta_resolution=24, phi_resolution=16,
                        start_phi=0.1, end_phi=179.9)
        if _moon_tex is not None:
            if not any("coord" in k.lower() for k in _ms.point_data.keys()):
                _ms = _ms.texture_map_to_sphere()
            pl.add_mesh(_ms, texture=_moon_tex, smooth_shading=True,
                        lighting=False, name="moon3d")
        else:
            pl.add_mesh(_ms, color=_MCOL, opacity=0.85,
                        smooth_shading=True, name="moon3d")

        # Trajectories — mutate pre-allocated PolyData in-place
        for run_id in vid_ids:
            s       = trajs[run_id]
            is_nom  = (run_id == nominal_id)
            outcome = "nominal" if is_nom else display_outcome_map.get(run_id, "miss")
            if outcome not in ("nominal", "miss") and t_now > s["t_exit"]:
                mask = np.zeros(len(s["t"]), dtype=bool)
            else:
                mask = (s["t"] > t_now - TAIL_ND) & (s["t"] <= t_now)
            # Clip at visual Moon sphere radius so crash trajectories stop exactly at
            # the displayed Moon surface (physical clip radius < visual display radius).
            if outcome == "moon_crash" and mask.any():
                mask = mask & (s["r_mc_km"] > _R_ECI_MOON_VIS_KM)
            _a3  = _TRAJ_A[outcome]
            _rgb = np.array([int(c * 255) for c in _TRAJ_COL[outcome]], dtype=np.uint8)
            tp   = pd["tr3"][run_id]
            dp   = pd["dt3"][run_id]

            if mask.sum() >= 2:
                pts  = np.column_stack([s["xi"][mask], s["yi"][mask], s["zi"][mask]])
                n    = len(pts)
                rgba = np.zeros((n, 4), dtype=np.uint8)
                rgba[:, :3] = _rgb
                rgba[:, 3]  = np.linspace(max(30, _a3 // 4), _a3, n, dtype=np.uint8)
                tp.points = pts
                tp.lines  = np.array([n] + list(range(n)), dtype=np.int64)
                tp.point_data["colors"] = rgba
                tp.Modified()
                dp.points = np.array([[s["xi"][mask][-1],
                                       s["yi"][mask][-1],
                                       s["zi"][mask][-1]]], dtype=np.float64)
                dp.Modified()
            else:
                tp.points = _DUMMY_PTS.copy()
                tp.lines  = _DUMMY_LINES.copy()
                tp.point_data["colors"] = np.zeros((2, 4), dtype=np.uint8)
                tp.Modified()
                dp.points = _OFFSCREEN.copy()
                dp.Modified()

        pl.camera.position    = c3
        pl.camera.focal_point = (0.0, 0.0, 0.0)
        pl.camera.up          = (0.0, 0.0, 1.0)
        pl.camera_set = True

        # RIGHT panel ──────────────────────────────────────────────────────────
        pl.subplot(0, 1)
        pl.enable_parallel_projection()
        pl.camera.position       = _C2_POS
        pl.camera.focal_point    = _C2_FOC
        pl.camera.up             = _C2_UP
        pl.camera.parallel_scale = VIEW_SOI
        pl.camera_set = True

        _off2d = pd["offscreen_2d"]
        for run_id in vid_ids:
            s       = trajs[run_id]
            is_nom  = (run_id == nominal_id)
            outcome = "nominal" if is_nom else display_outcome_map.get(run_id, "miss")
            if outcome not in ("nominal", "miss") and t_now > s["t_exit"]:
                mask = np.zeros(len(s["t"]), dtype=bool)
            else:
                mask = (s["t"] > t_now - TAIL_ND) & (s["t"] <= t_now)
            # Clip at visual Moon sphere radius — Moon in SOI display is larger than physical.
            if outcome == "moon_crash" and mask.any():
                mask = mask & (s["r_mc_km"] > _R_SOI_MOON_VIS_KM)
            _a2  = _TRAJ_A[outcome]
            _rgb2 = np.array([int(c * 255) for c in _TRAJ_COL[outcome]], dtype=np.uint8)
            tp2  = pd["tr2"][run_id]
            dp2  = pd["dt2"][run_id]

            if mask.sum() >= 2:
                n2   = mask.sum()
                pts2 = np.column_stack(
                    [s["xi_mc"][mask], s["yi_mc"][mask], np.zeros(n2)])
                rgba2 = np.zeros((n2, 4), dtype=np.uint8)
                rgba2[:, :3] = _rgb2
                rgba2[:, 3]  = np.linspace(max(30, _a2 // 4), _a2, n2, dtype=np.uint8)
                tp2.points = pts2
                tp2.lines  = np.array([n2] + list(range(n2)), dtype=np.int64)
                tp2.point_data["colors"] = rgba2
                tp2.Modified()
                dp2.points = np.array([[s["xi_mc"][mask][-1],
                                        s["yi_mc"][mask][-1], 0.0]], dtype=np.float64)
                dp2.Modified()
            else:
                tp2.points = _DUMMY_PTS.copy()
                tp2.lines  = _DUMMY_LINES.copy()
                tp2.point_data["colors"] = np.zeros((2, 4), dtype=np.uint8)
                tp2.Modified()
                dp2.points = _off2d.copy()
                dp2.Modified()

        pl.camera.position       = _C2_POS
        pl.camera.focal_point    = _C2_FOC
        pl.camera.up             = _C2_UP
        pl.camera.parallel_scale = VIEW_SOI
        pl.camera_set = True

    # ── Render loop ───────────────────────────────────────────────────────────
    # t_anim_gif is already computed at top level (variable-speed, shared with HTML)
    gif_every = max(1, len(t_anim_gif) // 120)
    # Load stats bar data from stats_sweep.csv when available (falls back to ensemble).
    _stats_fracs = None
    _stats_n     = None
    try:
        _sw = pd.read_csv(OUT_DIR / "stats_sweep.csv")
        _row = _sw[np.abs(_sw["sigma_scale"] - 1.0) < 1e-6]
        if len(_row):
            _r = _row.iloc[0]
            _stats_n     = int(_r["n_total"])
            _stats_fracs = {k: float(_r[f"frac_{k}"])
                            for k in ("captured", "flyby", "moon_crash", "escaped")}
    except Exception:
        pass
    if _stats_fracs is None:
        _n_ens = n_cap + n_fly + n_cra + n_mis
        if _n_ens > 0:
            _stats_n = _n_ens
            _stats_fracs = {
                "captured":   n_cap   / _n_ens,
                "flyby":      n_fly   / _n_ens,
                "moon_crash": n_cra   / _n_ens,
                "escaped":    n_mis   / _n_ens,
            }

    # ── Preview: single frame → PNG (fast sanity check) ──────────────────────
    if args.preview:
        _prev_fi   = len(t_anim_gif) // 2
        _prev_path = mp4_out.with_suffix(".preview.png")
        print(f"Rendering preview frame {_prev_fi}/{len(t_anim_gif)} "
              f"(Day {float(t_anim_gif[_prev_fi]) * T_STAR / 86400.0:.1f}) ...", flush=True)
        _pl_prev, _pd_prev = _build_pl(_VID_W, _VID_H, hifi=True)
        _upd_pv(_pl_prev, _pd_prev, _prev_fi, t_anim_gif)
        _pl_prev.render()
        _prev_arr = _pl_prev.screenshot(return_img=True)
        _prev_t_d = float(t_anim_gif[_prev_fi]) * T_STAR / 86400.0
        _prev_arr = _overlay_hud(_prev_arr, _prev_t_d, _prev_fi, _pl_prev)
        _PILImage.fromarray(_prev_arr).save(str(_prev_path))
        _pl_prev.close()
        print(f"Saved {_prev_path}")
        if not (args.gif or args.mp4):
            import sys as _sys; _sys.exit(0)

    print(f"Rendering {len(t_anim_gif)} frames via PyVista ({_VID_W}×{_VID_H})  "
          f"[{len(vid_ids)} traj, {len(t_anim_gif)} frames] ...", flush=True)
    _pl, _pd = _build_pl(_VID_W, _VID_H, hifi=True)
    _tmp = _tmpfile.mkdtemp(prefix="wsb_sens_pv_")
    _gif_imgs = []

    try:
        for _fi, _t_idx in enumerate(range(len(t_anim_gif))):
            if _fi % 20 == 0:
                print(f"  frame {_fi}/{len(t_anim_gif)}", flush=True)
            _upd_pv(_pl, _pd, _fi, t_anim_gif)
            _pl.render()   # flush PolyData mutations before capture
            _arr = _pl.screenshot(return_img=True)
            _t_days = float(t_anim_gif[_fi]) * T_STAR / 86400.0
            _arr = _overlay_hud(_arr, _t_days, _fi, _pl)
            _PILImage.fromarray(_arr).save(
                _os.path.join(_tmp, f"frame_{_fi:06d}.png"))
            if _fi % gif_every == 0:
                _gif_imgs.append(
                    _PILImage.fromarray(_arr).resize((1920, 1080), _PILImage.LANCZOS))
        _pl.close()

        if args.gif:
            print(f"Saving GIF ({len(_gif_imgs)} frames) ...", end="", flush=True)
            _gif_imgs[0].save(
                str(gif_out), save_all=True, append_images=_gif_imgs[1:],
                duration=int(1000 / args.fps), loop=0, optimize=False,
            )
            print(f" done.\nSaved {gif_out}")

        if args.mp4:
            if _ffmpeg_path:
                print(f"Encoding MP4 ({len(t_anim_gif)} frames, {_VID_W}×{_VID_H}) ...",
                      end="", flush=True)
                _subp.run(
                    [_ffmpeg_path, "-y",
                     "-framerate", str(int(args.fps)),
                     "-i", _os.path.join(_tmp, "frame_%06d.png"),
                     "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "15", "-preset", "slow",
                     str(mp4_out)],
                    check=True, capture_output=True,
                )
                print(f" done.\nSaved {mp4_out}")
            else:
                print("MP4 skipped — ffmpeg not found (winget install Gyan.FFmpeg)")
    finally:
        _shutil.rmtree(_tmp, ignore_errors=True)
