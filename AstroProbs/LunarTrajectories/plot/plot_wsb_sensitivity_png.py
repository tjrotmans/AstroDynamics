#!/usr/bin/env python3
"""
plot_wsb_sensitivity_png.py — static ECI top-down PNG of the WSB sensitivity ensemble.

Trajectories are coloured by DOMINANT INPUT PERTURBATION (ΔV magnitude / pointing /
burn timing / launch window), mirroring the Artemis sensitivity portrait style.
Falls back to outcome colouring if the CSV predates the perturbation columns.

All trajectories (including nominal) are stopped at the time the nominal first
reaches the Hill sphere (SOI).  The Moon dot is placed at the Moon's ECI position
at that same moment.  Alpha per group is normalised so every group has equal
visual weight regardless of how many trajectories fall in each group.

Reads:
  out/wsb/sensitivity_ensemble[_TAG].csv
  out/wsb/epoch_info.txt              (optional — identity R0 if missing)

Outputs:
  out/wsb/wsb_sensitivity_png[_TAG].png

Usage:
  python plot/plot_wsb_sensitivity_png.py [--tag TAG]
"""
from __future__ import annotations
import argparse, math, pathlib, sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from wsb_style import (
    MU, X_M, R_EARTH, R_HILL_ND, R_HILL_KM,
    rot_em_to_eci, moon_em_to_eci, load_epoch_info,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT  = ROOT / "out" / "wsb"

# ── Dark theme ────────────────────────────────────────────────────────────────
BG      = "#0F0F19"
GRID_C  = "#1A1A30"
SPINE_C = "#252540"
TEXT_C  = "#CCCCEE"
TITLE_C = "#E0E0FF"

# ── Input-perturbation groups (Artemis-style colouring) ───────────────────────
GROUP_COLS = {
    "dv_mag":        "#00E5FF",
    "pointing":      "#FF8C00",
    "burn_timing":   "#BF5FFF",
    "launch_window": "#59FF40",
}
GROUP_LABELS = {
    "dv_mag":        "ΔV magnitude",
    "pointing":      "Pointing error",
    "burn_timing":   "Burn timing",
    "launch_window": "Launch window",
}
GROUP_ORDER = ["dv_mag", "pointing", "burn_timing", "launch_window"]

# ── Fallback: outcome colouring ───────────────────────────────────────────────
FLYBY_THRESH = 3
OUTCOME_COLS = {
    "captured":   "#3DFF8F",
    "flyby":      "#C77DFF",
    "moon_crash": "#FF4455",
    "escaped":    "#778899",
}
OUTCOME_LABELS = {
    "captured":   "Captured",
    "flyby":      "Flyby",
    "moon_crash": "Moon crash",
    "escaped":    "Miss",
}
OUTCOME_ORDER = ["captured", "flyby", "moon_crash", "escaped"]

# ── Sigma values (must match wsb_sensitivity.rs) ──────────────────────────────
_DV_MAG_SIGMA    = 1e-3
_DV_DIR_SIGMA    = 1.75e-3
_THETA_SIGMA     = 0.2    # deg — burn timing (≈3 s on 92-min orbit)
_THETA_SUN_SIGMA = 3.0    # deg — launch window (≈5.9 hr)

_DV_PCT      = _DV_MAG_SIGMA * 100.0
_POINT_DEG   = math.degrees(_DV_DIR_SIGMA)
_OMEGA_DEG_S = math.sqrt(398_600.4418 / (6_371.0 + 378.0) ** 3) * (180.0 / math.pi)
_TIMING_S    = _THETA_SIGMA / _OMEGA_DEG_S
_SUN_DEG_HR  = 360.0 / (29.530_589 * 24.0)   # Sun angular rate [deg/hr]
_WIN_HR      = _THETA_SUN_SIGMA / _SUN_DEG_HR

# Alpha normalisation: each group targets this many "equivalent solid trajectories"
# so every group is equally readable regardless of member count.
_TARGET_ALPHA_SUM = 5.0
_ALPHA_MIN = 0.06
_ALPHA_MAX = 0.45


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify_outcome(df: pd.DataFrame) -> pd.Series:
    is_flyby = (df["outcome"] == "captured") & (df["n_orbits"] < FLYBY_THRESH)
    oc = df["outcome"].copy()
    oc[is_flyby] = "flyby"
    return oc


def _assign_group(row: pd.Series) -> str:
    scores = {
        "dv_mag":        abs(row["dmag_frac"])   / _DV_MAG_SIGMA,
        "pointing":      max(abs(row["dpitch_rad"]), abs(row["dyaw_rad"])) / _DV_DIR_SIGMA,
        "burn_timing":   abs(row["dtheta_deg"])  / _THETA_SIGMA,
        "launch_window": abs(row["dtsun_deg"])   / _THETA_SUN_SIGMA,
    }
    return max(scores, key=scores.get)


def _group_alpha(n: int) -> float:
    """Normalised alpha so n stacked lines have the same total weight for every group."""
    return float(np.clip(_TARGET_ALPHA_SUM / max(n, 1), _ALPHA_MIN, _ALPHA_MAX))


def _hill_entry_time(grp: pd.DataFrame) -> float | None:
    """First time_nd the trajectory enters the Hill sphere (EM rotating frame)."""
    dx = grp["x_nd"].values - X_M
    dy = grp["y_nd"].values
    dz = grp["z_nd"].values
    inside = np.sqrt(dx**2 + dy**2 + dz**2) < R_HILL_ND
    idx = int(np.argmax(inside))
    return float(grp["time_nd"].values[idx]) if inside[idx] else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--post-soi", type=float, default=0.0, metavar="ND",
        help="Extend each trajectory this many nd past nominal SOI arrival "
             "(useful for OAT data where divergence happens near the Moon). "
             "E.g. --post-soi 6.28 for +1 lunar period.",
    )
    ap.add_argument(
        "--view", type=float, default=1_600_000.0, metavar="KM",
        help="Half-width of the ECI view in km (default 1,600,000).",
    )
    ap.add_argument("--video", action="store_true",
                    help="Also render an MP4 animation (same time range as PNG).")
    ap.add_argument("--fps",    type=int, default=30,  metavar="FPS",
                    help="Video frame rate (default 30).")
    ap.add_argument("--frames", type=int, default=450, metavar="N",
                    help="Number of animation frames (default 450 → 15 s at 30 fps).")
    args = ap.parse_args()

    sfx        = f"_{args.tag}" if args.tag else ""
    post_soi   = args.post_soi
    view_km    = args.view
    csv_path   = OUT / f"sensitivity_ensemble{sfx}.csv"
    epoch_path = OUT / "epoch_info.txt"

    if not csv_path.exists():
        sys.exit(
            f"[error] {csv_path} not found — run wsb_sensitivity first.\n"
            "  cargo run -p lunar_trajectories --bin wsb_sensitivity --release -- --hifi"
        )

    # ── Data ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    has_perturbs = "dmag_frac" in df.columns
    df["oc"] = _classify_outcome(df)

    df_clean  = df.dropna(subset=["x_nd"])
    run_first = df_clean.groupby("run_id").first().reset_index()
    n_total   = len(run_first)

    if has_perturbs:
        run_first["group"] = run_first.apply(_assign_group, axis=1)
        group_counts  = {g: int((run_first["group"] == g).sum()) for g in GROUP_ORDER}
        group_alpha_m = {g: _group_alpha(group_counts[g]) for g in GROUP_ORDER}
        col_map       = GROUP_COLS
        alpha_m       = group_alpha_m
    else:
        oc_counts  = {oc: int((run_first["oc"] == oc).sum()) for oc in OUTCOME_ORDER}
        oc_alpha_m = {oc: _group_alpha(oc_counts[oc]) for oc in OUTCOME_ORDER}
        col_map    = OUTCOME_COLS
        alpha_m    = oc_alpha_m

    # ── SOI arrival time: nominal's first Hill entry ──────────────────────────
    nom_grp = df_clean[df_clean["is_nominal"] == 1].sort_values("time_nd")
    t_soi   = _hill_entry_time(nom_grp)
    if t_soi is None:
        # Nominal never enters Hill sphere — use end of trajectory as fallback
        t_soi = float(nom_grp["time_nd"].max())
        print("[warn] nominal never entered Hill sphere; clipping at end of data")

    # ── ECI rotation matrix ───────────────────────────────────────────────────
    R0 = np.eye(3)
    if epoch_path.exists():
        _, R0 = load_epoch_info(epoch_path)

    # ── Rotate trajectories to ECI, all clipped at t_soi ─────────────────────
    if has_perturbs:
        group_map  = run_first.set_index("run_id")["group"].to_dict()
    is_nom_map = run_first.set_index("run_id")["is_nominal"].to_dict()
    oc_map     = run_first.set_index("run_id")["oc"].to_dict()

    trajs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, str, bool]] = {}
    for run_id, grp in df_clean.groupby("run_id", sort=False):
        g      = grp.sort_values("time_nd")
        g      = g[g["time_nd"] <= t_soi + post_soi]
        if g.empty:
            continue
        is_nom = bool(is_nom_map.get(run_id, 0))
        t_arr  = g["time_nd"].values

        xe, ye, _ = rot_em_to_eci(
            g["x_nd"].values, g["y_nd"].values, g["z_nd"].values,
            g["time_nd"].values, R0,
        )

        color_key = (group_map.get(run_id, "dv_mag") if has_perturbs
                     else oc_map.get(run_id, "escaped"))
        trajs[run_id] = (xe, ye, t_arr, color_key, is_nom)

    # ── Moon orbit track (full orbit for context) ─────────────────────────────
    t_orb = np.linspace(0.0, 2.0 * math.pi, 360)
    mx_orb, my_orb, _ = moon_em_to_eci(t_orb, R0)

    # Moon position at SOI arrival
    m_soi_x = float(moon_em_to_eci(np.array([t_soi]), R0)[0][0])
    m_soi_y = float(moon_em_to_eci(np.array([t_soi]), R0)[1][0])

    # Hill sphere centred on Moon's SOI position
    hs_th = np.linspace(0.0, 2.0 * math.pi, 256)
    hs_x  = m_soi_x + R_HILL_KM * np.cos(hs_th)
    hs_y  = m_soi_y + R_HILL_KM * np.sin(hs_th)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 10), facecolor=BG)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.83, bottom=0.09)

    ax.set_facecolor(BG)
    ax.set_aspect("equal", adjustable="box")
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax.tick_params(colors=TEXT_C, labelsize=9)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.grid(color=GRID_C, linewidth=0.4, zorder=0)
    ax.set_xlabel("X ECI [km]", fontsize=9)
    ax.set_ylabel("Y ECI [km]", fontsize=9)
    ax.set_title("XY — top-down view (ECI frame)", fontsize=8.5, color=TEXT_C, pad=5)
    ax.ticklabel_format(style="sci", axis="both", scilimits=(3, 3))
    ax.xaxis.offsetText.set_color(TEXT_C)
    ax.yaxis.offsetText.set_color(TEXT_C)

    # Moon orbit track
    ax.scatter(mx_orb[::2], my_orb[::2], s=0.8, c="#55556A", alpha=0.55,
               zorder=1, linewidths=0)

    # Hill sphere
    ax.plot(hs_x, hs_y, color="#BA7517", linewidth=0.9, linestyle="--",
            alpha=0.65, zorder=2)
    hs_lbl_ang = math.radians(135)
    ax.text(
        m_soi_x + R_HILL_KM * 1.12 * math.cos(hs_lbl_ang),
        m_soi_y + R_HILL_KM * 1.12 * math.sin(hs_lbl_ang),
        "Hill sphere", color="#BA7517", fontsize=7.5,
        ha="center", va="center", alpha=0.85, zorder=5,
    )

    # Perturbed trajectories — per-group normalised alpha, uniform linewidth
    # Collect endpoints per group for dot markers (same style as Artemis portrait)
    endpoints: dict[str, list[tuple[float, float]]] = {k: [] for k in col_map}
    for _, (xe, ye, _t, color_key, is_nom) in trajs.items():
        if is_nom:
            continue
        c = col_map.get(color_key, "#888888")
        a = (group_alpha_m.get(color_key, 0.15) if has_perturbs
             else oc_alpha_m.get(color_key, 0.15))
        ax.plot(xe, ye, color=c, linewidth=0.5, alpha=a,
                zorder=3, rasterized=True)
        if len(xe) > 0:
            endpoints.setdefault(color_key, []).append((float(xe[-1]), float(ye[-1])))

    # Endpoint dots
    for color_key, pts in endpoints.items():
        if not pts:
            continue
        ep = np.array(pts)
        c  = col_map.get(color_key, "#888888")
        ax.scatter(ep[:, 0], ep[:, 1], s=3.5, color=c, alpha=0.85,
                   zorder=4, linewidths=0)

    # Nominal on top
    nom_xe, nom_ye = next(
        (xe, ye) for xe, ye, _t, _c, is_nom in trajs.values() if is_nom
    )
    ax.plot(nom_xe, nom_ye, color="#FFD700", linewidth=1.6, alpha=0.95,
            zorder=10, solid_capstyle="round")
    ax.scatter(nom_xe[0], nom_ye[0], color="#FFD700", s=28,
               zorder=11, edgecolors="none")
    ax.annotate(
        "TLI injection",
        xy=(nom_xe[0], nom_ye[0]),
        xytext=(0.64, 0.88), textcoords="axes fraction",
        color="#FFD700", fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#FFD700", lw=1.0, alpha=0.75),
        zorder=12,
    )

    # Earth
    ax.scatter(0, 0, s=75, color="#1565C0", zorder=12,
               edgecolors="#4FC3F7", linewidths=0.8)
    ax.text(0, R_EARTH * 5, "Earth", color="white", fontsize=10,
            ha="center", va="bottom", fontweight="bold", zorder=13)

    # Moon at nominal SOI arrival
    ax.scatter(m_soi_x, m_soi_y, s=35, color="#9E9E9E", zorder=12, edgecolors="none")
    ax.text(m_soi_x, m_soi_y + R_HILL_KM * 0.22, "Moon", color="white", fontsize=10,
            ha="center", va="bottom", fontweight="bold", zorder=13)

    ax.set_xlim(-view_km, view_km)
    ax.set_ylim(-view_km, view_km)

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = []
    if has_perturbs:
        for g in GROUP_ORDER:
            n   = group_counts[g]
            pct = n / n_total * 100.0
            handles.append(Line2D(
                [0], [0], color=GROUP_COLS[g], linewidth=2.2,
                label=f"{GROUP_LABELS[g]}   {pct:.0f}%  ({n})",
            ))
    else:
        for oc in OUTCOME_ORDER:
            n = oc_counts[oc]
            if n == 0:
                continue
            pct = n / n_total * 100.0
            handles.append(Line2D(
                [0], [0], color=OUTCOME_COLS[oc], linewidth=2.2,
                label=f"{OUTCOME_LABELS[oc]}   {pct:.0f}%  ({n})",
            ))

    handles += [
        Line2D([0], [0], color="#FFD700", linewidth=2.0,
               label="Nominal trajectory"),
        Line2D([0], [0], color="#55556A", marker=".", markersize=6,
               linestyle="None", label="Moon orbit track"),
        Line2D([0], [0], color="#BA7517", linewidth=1.2, linestyle="--",
               label="Hill sphere"),
    ]

    fig.legend(
        handles=handles,
        loc="upper center", bbox_to_anchor=(0.5, 0.983),
        ncol=4, fontsize=9.5,
        framealpha=0.0, frameon=False,
        labelcolor="white", handlelength=1.8,
        handletextpad=0.5, columnspacing=1.2,
    )

    # ── Titles ───────────────────────────────────────────────────────────────
    t_soi_days      = t_soi * 375_700.0 / 86_400.0
    post_soi_days   = post_soi * 375_700.0 / 86_400.0
    clip_label      = (f"T+{t_soi_days:.1f} d + {post_soi_days:.1f} d post-SOI"
                       if post_soi > 0 else f"T+{t_soi_days:.1f} d (SOI arrival)")
    fig.text(
        0.5, 0.998,
        "WSB Transfer Sensitivity — ECI Top-Down View",
        ha="center", va="top", fontsize=13, color=TITLE_C, fontweight="bold",
    )
    fig.text(
        0.5, 0.873,
        f"1σ:  ΔV ±{_DV_PCT:.2f}%   ·   pointing ±{_POINT_DEG:.3f}°   ·   "
        f"burn timing ±{_TIMING_S:.1f} s   ·   launch window ±{_WIN_HR:.1f} hr"
        f"   ·   N = {n_total:,}   ·   shown to {clip_label}",
        ha="center", va="top", fontsize=8, color="#888899", fontstyle="italic",
    )

    # ── Save PNG ─────────────────────────────────────────────────────────────
    out_path = OUT / f"wsb_sensitivity_png{sfx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved {out_path}")
    plt.close(fig)

    # ── Video animation ───────────────────────────────────────────────────────
    if not args.video:
        return

    import matplotlib.animation as animation

    t_end_anim = t_soi + post_soi
    t_frames   = np.linspace(0.0, t_end_anim, args.frames)
    hs_th_anim = np.linspace(0.0, 2.0 * math.pi, 256)
    _T_STAR    = 375_700.0

    fig_v, ax_v = plt.subplots(figsize=(11, 10), facecolor=BG)
    fig_v.subplots_adjust(left=0.09, right=0.97, top=0.83, bottom=0.09)
    ax_v.set_facecolor(BG)
    ax_v.set_aspect("equal", adjustable="box")
    for sp in ax_v.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax_v.tick_params(colors=TEXT_C, labelsize=9)
    ax_v.xaxis.label.set_color(TEXT_C)
    ax_v.yaxis.label.set_color(TEXT_C)
    ax_v.grid(color=GRID_C, linewidth=0.4, zorder=0)
    ax_v.set_xlabel("X ECI [km]", fontsize=9)
    ax_v.set_ylabel("Y ECI [km]", fontsize=9)
    ax_v.set_title("XY — top-down view (ECI frame)", fontsize=8.5, color=TEXT_C, pad=5)
    ax_v.ticklabel_format(style="sci", axis="both", scilimits=(3, 3))
    ax_v.xaxis.offsetText.set_color(TEXT_C)
    ax_v.yaxis.offsetText.set_color(TEXT_C)
    ax_v.set_xlim(-view_km, view_km)
    ax_v.set_ylim(-view_km, view_km)

    # Static background elements
    ax_v.scatter(mx_orb[::2], my_orb[::2], s=0.8, c="#55556A", alpha=0.55,
                 zorder=1, linewidths=0)
    ax_v.scatter(0, 0, s=75, color="#1565C0", zorder=12,
                 edgecolors="#4FC3F7", linewidths=0.8)
    ax_v.text(0, R_EARTH * 5, "Earth", color="white", fontsize=10,
              ha="center", va="bottom", fontweight="bold", zorder=13)

    # TLI injection dot (static)
    nom_xe_v, nom_ye_v = next(
        (xe, ye) for xe, ye, _t, _c, is_nom in trajs.values() if is_nom
    )
    ax_v.scatter(nom_xe_v[0], nom_ye_v[0], color="#FFD700", s=28,
                 zorder=11, edgecolors="none")
    ax_v.annotate(
        "TLI injection",
        xy=(nom_xe_v[0], nom_ye_v[0]),
        xytext=(0.64, 0.88), textcoords="axes fraction",
        color="#FFD700", fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#FFD700", lw=1.0, alpha=0.75),
        zorder=12,
    )

    # Legend and titles (same as PNG)
    handles_v = []
    if has_perturbs:
        for g in GROUP_ORDER:
            n_g  = group_counts[g]
            pct  = n_g / n_total * 100.0
            handles_v.append(Line2D([0], [0], color=GROUP_COLS[g], linewidth=2.2,
                                    label=f"{GROUP_LABELS[g]}   {pct:.0f}%  ({n_g})"))
    else:
        for oc in OUTCOME_ORDER:
            n_oc = oc_counts[oc]
            if n_oc == 0:
                continue
            pct = n_oc / n_total * 100.0
            handles_v.append(Line2D([0], [0], color=OUTCOME_COLS[oc], linewidth=2.2,
                                    label=f"{OUTCOME_LABELS[oc]}   {pct:.0f}%  ({n_oc})"))
    handles_v += [
        Line2D([0], [0], color="#FFD700", linewidth=2.0, label="Nominal trajectory"),
        Line2D([0], [0], color="#55556A", marker=".", markersize=6,
               linestyle="None", label="Moon orbit track"),
        Line2D([0], [0], color="#BA7517", linewidth=1.2, linestyle="--",
               label="Hill sphere"),
    ]
    fig_v.legend(handles=handles_v, loc="upper center", bbox_to_anchor=(0.5, 0.983),
                 ncol=4, fontsize=9.5, framealpha=0.0, frameon=False,
                 labelcolor="white", handlelength=1.8,
                 handletextpad=0.5, columnspacing=1.2)
    fig_v.text(0.5, 0.998, "WSB Transfer Sensitivity — ECI Top-Down View",
               ha="center", va="top", fontsize=13, color=TITLE_C, fontweight="bold")
    fig_v.text(
        0.5, 0.873,
        f"1σ:  ΔV ±{_DV_PCT:.2f}%   ·   pointing ±{_POINT_DEG:.3f}°   ·   "
        f"burn timing ±{_TIMING_S:.1f} s   ·   launch window ±{_WIN_HR:.1f} hr"
        f"   ·   N = {n_total:,}   ·   shown to {clip_label}",
        ha="center", va="top", fontsize=8, color="#888899", fontstyle="italic",
    )

    # Pre-create a line object per trajectory
    line_objs: list[tuple] = []
    for xe, ye, t_arr, color_key, is_nom in trajs.values():
        c  = col_map.get(color_key, "#888888")
        a  = alpha_m.get(color_key, 0.15)
        lw = 2.2 if is_nom else 1.2
        zo = 10  if is_nom else 3
        ln, = ax_v.plot([], [], color=c, linewidth=lw, alpha=a, zorder=zo)
        line_objs.append((ln, xe, ye, t_arr))

    # Pre-compute endpoint (xe[-1], ye[-1], t_end) per non-nominal trajectory
    ep_info: list[tuple[float, float, float, str]] = []
    for xe, ye, t_arr, color_key, is_nom in trajs.values():
        if is_nom or len(xe) == 0:
            continue
        ep_info.append((float(xe[-1]), float(ye[-1]), float(t_arr[-1]), color_key))

    # One scatter artist per color group — starts empty, fills as trajectories complete
    ep_scatters: dict[str, object] = {}
    for ck in col_map:
        ep_scatters[ck] = ax_v.scatter(
            [], [], s=3.5, color=col_map[ck], alpha=0.85, zorder=4, linewidths=0,
        )

    # Dynamic artists: Moon dot, Hill sphere, Moon label, Hill label, time readout
    moon_dot_v, = ax_v.plot([], [], "o", color="#9E9E9E", markersize=8,
                             zorder=12, markeredgecolor="none")
    moon_lbl_v  = ax_v.text(0.0, 0.0, "Moon", color="white", fontsize=10,
                             ha="center", va="bottom", fontweight="bold", zorder=13)
    hill_ln_v,  = ax_v.plot([], [], color="#BA7517", linewidth=0.9,
                             linestyle="--", alpha=0.65, zorder=2)
    hill_lbl_v  = ax_v.text(0.0, 0.0, "Hill sphere", color="#BA7517", fontsize=7.5,
                             ha="center", va="center", alpha=0.85, zorder=5)
    time_lbl_v  = ax_v.text(0.02, 0.05, "", transform=ax_v.transAxes,
                             color=TEXT_C, fontsize=11, fontweight="bold", zorder=20)

    def _update(i: int) -> list:
        t_curr = float(t_frames[i])

        for ln, xe, ye, t_arr in line_objs:
            mask = t_arr <= t_curr
            ln.set_data(xe[mask], ye[mask])

        mx_c = float(moon_em_to_eci(np.array([t_curr]), R0)[0][0])
        my_c = float(moon_em_to_eci(np.array([t_curr]), R0)[1][0])

        moon_dot_v.set_data([mx_c], [my_c])
        moon_lbl_v.set_position((mx_c, my_c + R_HILL_KM * 0.22))

        hx = mx_c + R_HILL_KM * np.cos(hs_th_anim)
        hy = my_c + R_HILL_KM * np.sin(hs_th_anim)
        hill_ln_v.set_data(hx, hy)
        lbl_ang = math.radians(135)
        hill_lbl_v.set_position((
            mx_c + R_HILL_KM * 1.12 * math.cos(lbl_ang),
            my_c + R_HILL_KM * 1.12 * math.sin(lbl_ang),
        ))

        t_days = t_curr * _T_STAR / 86_400.0
        time_lbl_v.set_text(f"T + {t_days:.1f} d")

        # Endpoint dots: appear once each trajectory's arc is fully drawn
        by_group: dict[str, list] = {ck: [] for ck in col_map}
        for xe_e, ye_e, t_e, ck in ep_info:
            if t_e <= t_curr:
                by_group[ck].append((xe_e, ye_e))
        for ck, sc in ep_scatters.items():
            pts = by_group[ck]
            sc.set_offsets(np.array(pts) if pts else np.empty((0, 2)))

        return ([ln for ln, *_ in line_objs]
                + list(ep_scatters.values())
                + [moon_dot_v, moon_lbl_v, hill_ln_v, hill_lbl_v, time_lbl_v])

    ani = animation.FuncAnimation(
        fig_v, _update, frames=args.frames,
        interval=1000 // args.fps, blit=False,
    )

    vid_path = OUT / f"wsb_sensitivity_video{sfx}.mp4"
    try:
        writer = animation.FFMpegWriter(
            fps=args.fps, bitrate=2000,
            extra_args=["-pix_fmt", "yuv420p"],
        )
        print(f"Rendering {args.frames} frames → {vid_path} …")
        ani.save(str(vid_path), writer=writer, dpi=120, savefig_kwargs={"facecolor": BG})
        print(f"Saved {vid_path}")
    except Exception as exc:
        gif_path = vid_path.with_suffix(".gif")
        print(f"[warn] FFMpeg unavailable ({exc}); falling back to GIF → {gif_path}")
        ani.save(str(gif_path), writer=animation.PillowWriter(fps=args.fps),
                 dpi=100, savefig_kwargs={"facecolor": BG})
        print(f"Saved {gif_path}")
    finally:
        plt.close(fig_v)


if __name__ == "__main__":
    main()
