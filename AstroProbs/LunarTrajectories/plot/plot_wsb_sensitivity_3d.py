#!/usr/bin/env python3
"""
plot_wsb_sensitivity_3d.py — 3-D ECI view of the WSB sensitivity ensemble.

Same data pipeline and colouring as plot_wsb_sensitivity_png.py but rendered
in three dimensions so out-of-plane motion from yaw (pointing) perturbations
is visible.

Note: the z-axis is set to the same ±VIEW km as x/y to preserve geometry.
Out-of-plane motion is physically small; use --view 200000 to zoom into the
Moon-vicinity region where z-spread is most visible.

Usage:
  python plot/plot_wsb_sensitivity_3d.py [--tag TAG]
         [--post-soi ND] [--view KM] [--elev DEG] [--azim DEG]
"""
from __future__ import annotations
import argparse, math, pathlib, sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  — registers '3d' projection
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

# ── Input-perturbation group colours ─────────────────────────────────────────
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
_DV_DIR_SIGMA    = 3.5e-3
_THETA_SIGMA     = 0.2
_THETA_SUN_SIGMA = 3.0

_DV_PCT      = _DV_MAG_SIGMA * 100.0
_POINT_DEG   = math.degrees(_DV_DIR_SIGMA)
_OMEGA_DEG_S = math.sqrt(398_600.4418 / (6_371.0 + 378.0) ** 3) * (180.0 / math.pi)
_TIMING_S    = _THETA_SIGMA / _OMEGA_DEG_S
_SUN_DEG_HR  = 360.0 / (29.530_589 * 24.0)
_WIN_HR      = _THETA_SUN_SIGMA / _SUN_DEG_HR

_TARGET_ALPHA_SUM = 5.0
_ALPHA_MIN = 0.06
_ALPHA_MAX = 0.45


# ── Helpers (identical to 2-D script) ────────────────────────────────────────

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
    return float(np.clip(_TARGET_ALPHA_SUM / max(n, 1), _ALPHA_MIN, _ALPHA_MAX))


def _hill_entry_time(grp: pd.DataFrame) -> float | None:
    dx = grp["x_nd"].values - X_M
    dy = grp["y_nd"].values
    dz = grp["z_nd"].values
    inside = np.sqrt(dx**2 + dy**2 + dz**2) < R_HILL_ND
    idx = int(np.argmax(inside))
    return float(grp["time_nd"].values[idx]) if inside[idx] else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",      default="")
    ap.add_argument("--post-soi", type=float, default=0.0,       metavar="ND")
    ap.add_argument("--view",     type=float, default=1_600_000., metavar="KM",
                    help="Half-width of all three ECI axes [km]")
    ap.add_argument("--elev",     type=float, default=20.,        metavar="DEG",
                    help="Camera elevation (default 20)")
    ap.add_argument("--azim",     type=float, default=220.,       metavar="DEG",
                    help="Camera azimuth (default 220)")
    args = ap.parse_args()

    sfx      = f"_{args.tag}" if args.tag else ""
    post_soi = args.post_soi
    view_km  = args.view

    csv_path   = OUT / f"sensitivity_ensemble{sfx}.csv"
    epoch_path = OUT / "epoch_info.txt"

    if not csv_path.exists():
        sys.exit(
            f"[error] {csv_path} not found — run wsb_sensitivity first.\n"
            "  cargo run -p lunar_trajectories --bin wsb_sensitivity --release -- --hifi"
        )

    # ── Load data ─────────────────────────────────────────────────────────────
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
    else:
        oc_counts  = {oc: int((run_first["oc"] == oc).sum()) for oc in OUTCOME_ORDER}
        oc_alpha_m = {oc: _group_alpha(oc_counts[oc]) for oc in OUTCOME_ORDER}
        col_map    = OUTCOME_COLS

    # ── Nominal SOI clip time ─────────────────────────────────────────────────
    nom_grp = df_clean[df_clean["is_nominal"] == 1].sort_values("time_nd")
    t_soi   = _hill_entry_time(nom_grp)
    if t_soi is None:
        t_soi = float(nom_grp["time_nd"].max())
        print("[warn] nominal never entered Hill sphere; clipping at end of data")

    # ── ECI rotation matrix ───────────────────────────────────────────────────
    R0 = np.eye(3)
    if epoch_path.exists():
        _, R0 = load_epoch_info(epoch_path)

    # ── Rotate all trajectories to 3-D ECI, clipped at t_soi ─────────────────
    if has_perturbs:
        group_map = run_first.set_index("run_id")["group"].to_dict()
    is_nom_map = run_first.set_index("run_id")["is_nominal"].to_dict()
    oc_map     = run_first.set_index("run_id")["oc"].to_dict()

    # (xe, ye, ze, color_key, is_nom)
    trajs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, str, bool]] = {}
    for run_id, grp in df_clean.groupby("run_id", sort=False):
        g = grp.sort_values("time_nd")
        g = g[g["time_nd"] <= t_soi + post_soi]
        if g.empty:
            continue
        is_nom = bool(is_nom_map.get(run_id, 0))
        xe, ye, ze = rot_em_to_eci(
            g["x_nd"].values, g["y_nd"].values, g["z_nd"].values,
            g["time_nd"].values, R0,
        )
        color_key = (group_map.get(run_id, "dv_mag") if has_perturbs
                     else oc_map.get(run_id, "escaped"))
        trajs[run_id] = (xe, ye, ze, color_key, is_nom)

    # ── Moon orbit track (full circle in ECI) ────────────────────────────────
    t_orb = np.linspace(0.0, 2.0 * math.pi, 360)
    mx_orb, my_orb, mz_orb = moon_em_to_eci(t_orb, R0)

    # Moon position at nominal SOI arrival
    m_soi_arr = moon_em_to_eci(np.array([t_soi]), R0)
    m_soi_x = float(m_soi_arr[0][0])
    m_soi_y = float(m_soi_arr[1][0])
    m_soi_z = float(m_soi_arr[2][0])

    # Hill sphere wireframe (sphere centred on Moon at t_soi)
    hs_u = np.linspace(0, 2 * math.pi, 36)
    hs_v = np.linspace(0, math.pi, 18)
    hs_sx = m_soi_x + R_HILL_KM * np.outer(np.cos(hs_u), np.sin(hs_v))
    hs_sy = m_soi_y + R_HILL_KM * np.outer(np.sin(hs_u), np.sin(hs_v))
    hs_sz = m_soi_z + R_HILL_KM * np.outer(np.ones(36),  np.cos(hs_v))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10), facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.86, bottom=0.0)

    # Dark pane backgrounds
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor(BG)
        pane.set_edgecolor(SPINE_C)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_C, labelsize=7.5)
    for lbl in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        lbl.set_color(TEXT_C)
    ax.set_xlabel("X ECI [km]", fontsize=8, labelpad=8)
    ax.set_ylabel("Y ECI [km]", fontsize=8, labelpad=8)
    ax.set_zlabel("Z ECI [km]", fontsize=8, labelpad=8)
    ax.view_init(elev=args.elev, azim=args.azim)

    # Moon orbit track
    ax.plot(mx_orb[::3], my_orb[::3], mz_orb[::3],
            ".", ms=0.8, color="#55556A", alpha=0.45, zorder=1)

    # Hill sphere
    ax.plot_wireframe(hs_sx, hs_sy, hs_sz,
                      color="#BA7517", alpha=0.10, linewidth=0.25, zorder=2)

    # Perturbed trajectories + endpoint dots
    endpoints: dict[str, list[tuple[float, float, float]]] = {k: [] for k in col_map}
    for _, (xe, ye, ze, color_key, is_nom) in trajs.items():
        if is_nom:
            continue
        c = col_map.get(color_key, "#888888")
        a = (group_alpha_m.get(color_key, 0.15) if has_perturbs
             else oc_alpha_m.get(color_key, 0.15))
        ax.plot(xe, ye, ze, color=c, linewidth=0.5, alpha=a, zorder=3)
        if len(xe) > 0:
            endpoints.setdefault(color_key, []).append(
                (float(xe[-1]), float(ye[-1]), float(ze[-1])))

    for color_key, pts in endpoints.items():
        if not pts:
            continue
        ep = np.array(pts)
        c  = col_map.get(color_key, "#888888")
        ax.scatter(ep[:, 0], ep[:, 1], ep[:, 2],
                   s=4.5, color=c, alpha=0.85, zorder=4, depthshade=False)

    # Nominal on top
    nom_xe, nom_ye, nom_ze = next(
        (xe, ye, ze) for xe, ye, ze, _, is_nom in trajs.values() if is_nom
    )
    ax.plot(nom_xe, nom_ye, nom_ze,
            color="#FFD700", linewidth=1.8, alpha=0.95, zorder=10)
    ax.scatter([nom_xe[0]], [nom_ye[0]], [nom_ze[0]],
               color="#FFD700", s=32, zorder=11, depthshade=False)

    # Earth
    ax.scatter([0], [0], [0], s=90, color="#1565C0",
               edgecolors="#4FC3F7", linewidths=0.8, zorder=12, depthshade=False)
    ax.text(0, 0, R_EARTH * 6, "Earth", color="white", fontsize=9,
            ha="center", va="bottom", fontweight="bold", zorder=13)

    # Moon at nominal SOI arrival
    ax.scatter([m_soi_x], [m_soi_y], [m_soi_z],
               s=40, color="#9E9E9E", zorder=12, depthshade=False)
    ax.text(m_soi_x, m_soi_y, m_soi_z + R_HILL_KM * 0.3,
            "Moon", color="white", fontsize=9,
            ha="center", va="bottom", fontweight="bold", zorder=13)

    ax.set_xlim(-view_km, view_km)
    ax.set_ylim(-view_km, view_km)
    ax.set_zlim(-view_km, view_km)

    # Scientific notation on tick labels
    try:
        for axis_name in ("x", "y", "z"):
            ax.ticklabel_format(style="sci", axis=axis_name, scilimits=(3, 3))
    except Exception:
        pass

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = []
    if has_perturbs:
        for g in GROUP_ORDER:
            n   = group_counts[g]
            pct = n / n_total * 100.0
            handles.append(Line2D([0], [0], color=GROUP_COLS[g], linewidth=2.2,
                label=f"{GROUP_LABELS[g]}   {pct:.0f}%  ({n})"))
    else:
        for oc in OUTCOME_ORDER:
            n = oc_counts[oc]
            if n == 0:
                continue
            pct = n / n_total * 100.0
            handles.append(Line2D([0], [0], color=OUTCOME_COLS[oc], linewidth=2.2,
                label=f"{OUTCOME_LABELS[oc]}   {pct:.0f}%  ({n})"))

    handles += [
        Line2D([0], [0], color="#FFD700", linewidth=2.0, label="Nominal trajectory"),
        Line2D([0], [0], color="#55556A", marker=".", markersize=6,
               linestyle="None", label="Moon orbit track"),
        Line2D([0], [0], color="#BA7517", linewidth=1.2, linestyle="--",
               label="Hill sphere"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=4, fontsize=9.5, framealpha=0.0, frameon=False,
               labelcolor="white", handlelength=1.8,
               handletextpad=0.5, columnspacing=1.2)

    # ── Titles ────────────────────────────────────────────────────────────────
    t_soi_days    = t_soi * 375_700.0 / 86_400.0
    post_soi_days = post_soi * 375_700.0 / 86_400.0
    clip_label    = (f"T+{t_soi_days:.1f} d + {post_soi_days:.1f} d post-SOI"
                     if post_soi > 0 else f"T+{t_soi_days:.1f} d (SOI arrival)")
    fig.text(0.5, 0.998,
             "WSB Transfer Sensitivity — ECI 3D View",
             ha="center", va="top", fontsize=13, color=TITLE_C, fontweight="bold")
    fig.text(0.5, 0.888,
             f"1σ:  ΔV ±{_DV_PCT:.2f}%   ·   pointing ±{_POINT_DEG:.3f}°   ·   "
             f"burn timing ±{_TIMING_S:.1f} s   ·   launch window ±{_WIN_HR:.1f} hr"
             f"   ·   N = {n_total:,}   ·   shown to {clip_label}"
             f"   ·   elev={args.elev:.0f}°  az={args.azim:.0f}°",
             ha="center", va="top", fontsize=7.5, color="#888899", fontstyle="italic")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = OUT / f"wsb_sensitivity_3d{sfx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
