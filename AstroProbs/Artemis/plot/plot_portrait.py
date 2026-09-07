#!/usr/bin/env python3
"""
Artemis 2 — Portrait static PNG export (matplotlib)

Produces two portrait PNGs (9:16):
  artemis_portrait_3d.png  — annotated 3D perspective
  artemis_portrait_2d.png  — XY + XZ 2D projections stacked vertically

Modes (same as plot_top_100_trajectories.py):
  python plot/plot_portrait.py            — top-100 MC (default)
  python plot/plot_portrait.py --all      — all MC trajectories
  python plot/plot_portrait.py --grid     — grid search (OAT sensitivity)
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
TOP100_TRAJ_CSV    = os.path.join(_HERE, "..", "out", "mc_top_100_trajectories.csv")
TOP_POS_CSV_PATH   = os.path.join(_HERE, "..", "out", "mc_top_100_positions.csv")
ALL_TRAJ_CSV       = os.path.join(_HERE, "..", "out", "mc_all_trajectories.csv")
GRID_TRAJ_CSV      = os.path.join(_HERE, "..", "out", "grid_trajectories.csv")
GRID_POS_CSV       = os.path.join(_HERE, "..", "out", "grid_positions.csv")
MOON_TRACK_CSV     = os.path.join(_HERE, "..", "out", "artemis2_trajectory.csv")
ASC_REFERENCE_PATH = os.path.join(_HERE, "..", "src", "Artemis_II_OEM_2026_04_04_to_EI.asc")
OUTPUT_DIR         = os.path.join(_HERE, "..", "out")

EARTH_RADIUS_KM = 6_371.0

# ── Style ─────────────────────────────────────────────────────────────────────
BG        = "#0F0F19"
PANE_EDGE = "#252540"
GRID_C    = "#1A1A30"

GROUP_COLORS = {
    "early_burn": "#FF8C00",
    "late_burn":  "#BF5FFF",
    "low_dv":     "#00E5FF",
    "high_dv":    "#59FF40",
}
GROUP_LABELS = {
    "early_burn": "Early ignition",
    "late_burn":  "Late ignition",
    "low_dv":     "Low \u0394V",
    "high_dv":    "High \u0394V",
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_oem(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    rows: list[list[float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s or s.startswith("COMMENT")
                    or s in ("META_START", "META_STOP") or "=" in s):
                continue
            t = s.split()
            if len(t) >= 7:
                rows.append([float(t[1]), float(t[2]), float(t[3])])
    return np.array(rows, dtype=float) if rows else None


def ca_mask(pos: np.ndarray, t: np.ndarray, max_miss_km: float = 80_000.0) -> np.ndarray:
    """Keep rows up to Earth closest-approach (after T+6 days), or all if missed."""
    dist  = np.linalg.norm(pos, axis=1)
    start = int(np.searchsorted(t, 6 * 86_400))
    if start >= len(pos) or dist[start:].min() > max_miss_km:
        return np.ones(len(pos), dtype=bool)
    ci = start + int(np.argmin(dist[start:]))
    return np.arange(len(pos)) <= ci


def load_data(use_all: bool, use_grid: bool):
    traj_path = GRID_TRAJ_CSV if use_grid else (ALL_TRAJ_CSV if use_all else TOP100_TRAJ_CSV)
    pos_path  = GRID_POS_CSV  if use_grid else TOP_POS_CSV_PATH

    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Missing: {traj_path}\nRun the appropriate cargo binary first.")

    df_traj = pd.read_csv(traj_path)
    df_pos  = pd.read_csv(pos_path)

    ref_km: np.ndarray | None        = None
    ref_burn_m: np.ndarray | None    = None
    moon_pos: np.ndarray | None      = None
    moon_at_flyby: np.ndarray | None = None

    if os.path.exists(MOON_TRACK_CSV):
        df_ref    = pd.read_csv(MOON_TRACK_CSV)
        moon_pos  = df_ref[["moon_x_m", "moon_y_m", "moon_z_m"]].values * 1e-3
        _rk       = df_ref[["x_m", "y_m", "z_m"]].values * 1e-3
        _rt       = df_ref["time_s"].values
        _rb       = df_ref["is_burn"].values.astype(bool)
        _rm       = ca_mask(_rk, _rt)
        ref_km    = _rk[_rm]
        ref_burn_m = _rb[_rm]
        # Moon position at closest spacecraft approach (flyby)
        dist_moon     = np.linalg.norm(_rk - moon_pos, axis=1)
        moon_at_flyby = moon_pos[int(np.argmin(dist_moon))]

    return df_traj, df_pos, ref_km, ref_burn_m, moon_pos, moon_at_flyby


def preprocess(df_traj: pd.DataFrame, df_pos: pd.DataFrame):
    """Returns traj_data, tli_point, park_point."""
    sol_groups = df_traj.groupby("solution_idx")
    traj_data: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {g: [] for g in GROUP_COLORS}

    # Best solution for annotation points
    scores    = df_pos["score"].values
    best_idx  = int(np.argmin(scores))
    best_df   = sol_groups.get_group(best_idx)
    best_pos  = best_df[["x_m", "y_m", "z_m"]].values * 1e-3
    best_time = best_df["time_s"].values
    best_burn = best_df["is_burn"].values.astype(bool)
    best_mask = ca_mask(best_pos, best_time)
    best_pos  = best_pos[best_mask]
    best_burn = best_burn[best_mask]

    first_coast_idx = np.where(~best_burn)[0]
    park_point: np.ndarray | None = None
    if len(first_coast_idx) > 0:
        pick = first_coast_idx[len(first_coast_idx) // 6]
        park_point = best_pos[pick]

    burn_idx  = np.where(best_burn)[0]
    tli_point: np.ndarray | None = best_pos[burn_idx[0]] if len(burn_idx) else None

    for _, sol_df in sol_groups:
        grp = sol_df["group"].iloc[0] if "group" in sol_df.columns else "high_dv"
        if grp not in GROUP_COLORS:
            continue
        pos  = sol_df[["x_m", "y_m", "z_m"]].values * 1e-3
        t    = sol_df["time_s"].values
        burn = sol_df["is_burn"].values.astype(bool)
        m    = ca_mask(pos, t)
        pos, burn = pos[m], burn[m]
        traj_data[grp].append((pos[~burn], pos[burn], pos[-1]))

    return traj_data, tli_point, park_point


# ── Style helpers ─────────────────────────────────────────────────────────────

def style_3d(ax) -> None:
    ax.set_facecolor(BG)
    for attr in ("xaxis", "yaxis", "zaxis"):
        pane = getattr(ax, attr).pane
        pane.fill = False
        pane.set_edgecolor(PANE_EDGE)
        getattr(ax, attr)._axinfo["grid"]["color"] = GRID_C
        getattr(ax, attr).label.set_color("white")
        getattr(ax, attr).label.set_fontsize(7)
    ax.tick_params(colors="white", labelsize=5.5, pad=0)


def style_2d(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(BG)
    ax.set_xlabel(xlabel, color="white", fontsize=12, labelpad=4)
    ax.set_ylabel(ylabel, color="white", fontsize=12, labelpad=4)
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.tick_params(colors="white", labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(PANE_EDGE)
    ax.grid(color=GRID_C, linewidth=0.35, zorder=0)
    ax.ticklabel_format(style="sci", axis="both", scilimits=(3, 3))
    ax.xaxis.offsetText.set_color("white")
    ax.yaxis.offsetText.set_color("white")


def _earth_sphere(r: float, nu: int = 32, nv: int = 16):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    return (
        r * np.outer(np.cos(u), np.sin(v)),
        r * np.outer(np.sin(u), np.sin(v)),
        r * np.outer(np.ones(nu), np.cos(v)),
    )



def _annotate_3d(ax, point: np.ndarray, label: str, color: str,
                 offset: np.ndarray, fontsize: float = 7.0,
                 fontweight: str = "normal", fontstyle: str = "normal") -> None:
    lp = point + offset
    ax.plot([point[0], lp[0]], [point[1], lp[1]], [point[2], lp[2]],
            color=color, linewidth=0.9, alpha=0.9)
    ax.text(lp[0], lp[1], lp[2], label, color=color, fontsize=fontsize,
            ha="center", va="center", fontweight=fontweight, fontstyle=fontstyle)


def build_legend_handles(ref_km, oem, moon_pos) -> list:
    handles = [
        Line2D([0], [0], color=GROUP_COLORS[k], linewidth=2.5, label=GROUP_LABELS[k])
        for k in GROUP_COLORS
    ]
    if ref_km is not None:
        handles.append(Line2D([0], [0], color="#FF2222", linewidth=1.5, label="Reference (nominal)"))
    if oem is not None:
        handles.append(Line2D([0], [0], color="white", linewidth=1.0,
                               linestyle="--", alpha=0.75, label="OEM reference"))
    if moon_pos is not None:
        handles.append(Line2D([0], [0], color="gray", marker=".", markersize=6,
                               linestyle="None", label="Moon orbit track"))
    return handles


def build_legend_handles_2d(ref_km, moon_pos) -> list:
    """Legend for 2D sensitivity plot — ordered for column-first legend filling.

    matplotlib fills ncol=3 legends column-by-column, so the handle list must be
    interleaved to produce the desired visual row layout:
      Row 1: Early ignition | Low ΔV | Artemis II Nominal Trajectory
      Row 2: Late ignition  | High ΔV | Moon orbit track
    The required handle order is therefore: col0-row0, col0-row1, col1-row0, ...
    i.e. [early, late, low_dv, high_dv, ref, moon].
    """
    handles = [
        Line2D([0], [0], color=GROUP_COLORS["early_burn"], linewidth=2.5,
               label=GROUP_LABELS["early_burn"]),
        Line2D([0], [0], color=GROUP_COLORS["late_burn"],  linewidth=2.5,
               label=GROUP_LABELS["late_burn"]),
        Line2D([0], [0], color=GROUP_COLORS["low_dv"],     linewidth=2.5,
               label=GROUP_LABELS["low_dv"]),
        Line2D([0], [0], color=GROUP_COLORS["high_dv"],    linewidth=2.5,
               label=GROUP_LABELS["high_dv"]),
    ]
    if ref_km is not None:
        handles.append(Line2D([0], [0], color="#FF2222", linewidth=1.5,
                               label="Artemis II Nominal Trajectory"))
    if moon_pos is not None:
        handles.append(Line2D([0], [0], color="gray", marker=".", markersize=6,
                               linestyle="None", label="Moon orbit track"))
    return handles


# ── 3D figure ─────────────────────────────────────────────────────────────────

def plot_3d(
    traj_data, ref_km, ref_burn_m, moon_pos, oem,
    tli_point, park_point,
    title_str: str, out_path: str,
    use_many: bool = False,
) -> None:
    coast_alpha = 0.15 if use_many else 0.30
    burn_alpha  = 0.35 if use_many else 0.60

    fig = plt.figure(figsize=(9, 16), dpi=200, facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d")
    style_3d(ax)

    # Title + legend at top
    fig.text(0.5, 0.975, title_str, color="white", fontsize=9,
             ha="center", va="top", fontweight="bold")
    handles = build_legend_handles(ref_km, oem, moon_pos)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.972),
               ncol=3, fontsize=6.5, framealpha=0.35,
               facecolor="#14141E", edgecolor="#333355",
               labelcolor="white", handlelength=1.5, columnspacing=1.0)

    # Earth sphere
    sx, sy, sz = _earth_sphere(EARTH_RADIUS_KM)
    ax.plot_surface(sx, sy, sz, color="#1565C0", alpha=0.85, linewidth=0, zorder=2)
    ax.text(0, 0, EARTH_RADIUS_KM * 1.4, "Earth",
            color="white", fontsize=8, ha="center", va="bottom", fontweight="bold")

    # Moon orbit track
    if moon_pos is not None:
        mp = moon_pos[::20]
        ax.scatter(mp[:, 0], mp[:, 1], mp[:, 2], s=0.6, c="gray", alpha=0.35, zorder=1)

    # MC/grid trajectories — grouped
    for grp, tlist in traj_data.items():
        c = GROUP_COLORS[grp]
        for coast, burns, _ in tlist:
            if len(coast) > 1:
                ax.plot(coast[:, 0], coast[:, 1], coast[:, 2],
                        color=c, alpha=coast_alpha, linewidth=0.5, zorder=3)
            if len(burns) > 1:
                ax.plot(burns[:, 0], burns[:, 1], burns[:, 2],
                        color=c, alpha=burn_alpha, linewidth=1.0, zorder=4)

    # Reference trajectory
    if ref_km is not None:
        coast_ref = ref_km[~ref_burn_m]
        if len(coast_ref) > 1:
            ax.plot(coast_ref[:, 0], coast_ref[:, 1], coast_ref[:, 2],
                    color="#FF2222", linewidth=2.0, zorder=10)

    # OEM reference
    if oem is not None:
        idx = np.unique(np.linspace(0, len(oem) - 1, 300, dtype=int))
        ax.plot(oem[idx, 0], oem[idx, 1], oem[idx, 2],
                color="white", linewidth=1.0, linestyle="--", alpha=0.75, zorder=11)

    # Parking orbit annotation
    if park_point is not None:
        r         = np.linalg.norm(park_point)
        direction = park_point / r
        offset    = direction * 30_000 + np.array([0, 0, 18_000])
        ax.scatter(*park_point, color="#AAAAFF", s=15, zorder=12)
        _annotate_3d(ax, park_point, "Parking orbit", "#AAAAFF",
                     offset=offset, fontstyle="italic")

    # TLI burn annotation
    if tli_point is not None:
        ax.scatter(*tli_point, color="#FFD700", s=25, zorder=13)
        _annotate_3d(ax, tli_point, "TLI burn", "#FFD700",
                     offset=np.array([0, -45_000, 30_000]), fontweight="bold")

    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect([1.8, 1.2, 0.9])
    ax.set_xlabel("X [km]", labelpad=1)
    ax.set_ylabel("Y [km]", labelpad=1)
    ax.set_zlabel("Z [km]", labelpad=1)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.02)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    jpg_path = os.path.splitext(out_path)[0] + ".jpg"
    fig.savefig(jpg_path, dpi=200, facecolor=BG, format="jpeg")
    banner_path = os.path.splitext(out_path)[0] + "_banner.png"
    fig.set_size_inches(6.27, 12)
    fig.savefig(banner_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"Saved 3D → {out_path}")
    print(f"Saved 3D → {jpg_path}")
    print(f"Saved 3D → {banner_path}")


# ── 2D figure ─────────────────────────────────────────────────────────────────

def plot_2d(
    traj_data, ref_km, ref_burn_m,
    tli_point, moon_pos, moon_at_flyby,
    title_str: str, out_path: str,
    use_many: bool = False,
) -> None:
    coast_alpha = 0.12 if use_many else 0.22
    burn_alpha  = 0.30 if use_many else 0.50

    fig = plt.figure(figsize=(9, 9), dpi=200, facecolor=BG)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.82, bottom=0.08)
    ax = fig.add_subplot(111)

    style_2d(ax, "X [km]", "Y [km]", "XY \u2014 top-down view (ECI J2000)")

    # ── Main title ───────────────────────────────────────────────────────────
    fig.text(0.5, 0.965, title_str, color="white", fontsize=11,
             ha="center", va="top", fontweight="bold")

    # ── Legend (2 rows × 3 cols, ordered to match reference image) ───────────
    handles = build_legend_handles_2d(ref_km=ref_km, moon_pos=moon_pos)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.944),
               ncol=3, fontsize=11, framealpha=0.0, frameon=False,
               labelcolor="white", handlelength=1.8, columnspacing=1.2,
               handletextpad=0.5)

    # ── Subtitle ─────────────────────────────────────────────────────────────
    fig.text(0.5, 0.863,
             "\u00b7 dots mark end of 10 day simulation    "
             "Max. \u0394V\u2009offset = +/-\u20091% of total,  "
             "Max. TLI offset = +/- 2 min",
             color="#888899", fontsize=9, ha="center", va="top",
             fontstyle="italic")

    # ── MC / grid trajectories ───────────────────────────────────────────────
    for grp, tlist in traj_data.items():
        c = GROUP_COLORS[grp]
        endpoints = []
        for coast, burns, endpoint in tlist:
            if len(coast) > 1:
                ax.plot(coast[:, 0], coast[:, 1],
                        color=c, alpha=coast_alpha, linewidth=0.4)
            if len(burns) > 1:
                ax.plot(burns[:, 0], burns[:, 1],
                        color=c, alpha=burn_alpha, linewidth=0.8)
            endpoints.append(endpoint)
        if endpoints:
            ep = np.array(endpoints)
            ax.scatter(ep[:, 0], ep[:, 1], s=0.5, color=c, alpha=0.8, zorder=5)

    # ── Reference (nominal) trajectory ───────────────────────────────────────
    if ref_km is not None:
        c_ref = ref_km[~ref_burn_m]
        if len(c_ref) > 1:
            ax.plot(c_ref[:, 0], c_ref[:, 1],
                    color="#FF2222", linewidth=1.4, zorder=10)

    # ── Moon orbit track ─────────────────────────────────────────────────────
    if moon_pos is not None:
        mp = moon_pos[::20]
        ax.scatter(mp[:, 0], mp[:, 1], s=0.8, c="gray", alpha=0.45, zorder=1)

    # ── Derive axis spans after data is plotted ───────────────────────────────
    ax.autoscale_view()
    xlims  = ax.get_xlim()
    ylims  = ax.get_ylim()
    xspan  = xlims[1] - xlims[0]
    yspan  = ylims[1] - ylims[0]

    # ── Earth dot ────────────────────────────────────────────────────────────
    ax.scatter(0, 0, color="#1565C0", s=50, zorder=15, edgecolors="none")
    ax.text(0, yspan * 0.025, "Earth",
            color="white", fontsize=11, ha="center", va="bottom",
            fontweight="bold", zorder=16)

    # ── Moon dot + "Moon" label + "Lunar Flyby" annotation ───────────────────
    if moon_at_flyby is not None:
        mf = moon_at_flyby
        ax.scatter(mf[0], mf[1], color="#CCCCCC", s=35, zorder=15, edgecolors="none")
        ax.text(mf[0], mf[1] + yspan * 0.026, "Moon",
                color="white", fontsize=11, ha="center", va="bottom",
                fontweight="bold", zorder=16)
        ax.text(mf[0] - xspan * 0.02, mf[1] - yspan * 0.03,
                "Lunar Flyby \u2014 6 Apr 2026",
                color="white", fontsize=10, ha="right", va="top",
                fontstyle="italic", zorder=16)

    # ── TLI annotation (text anchored to axes fraction to stay inside plot) ──
    if tli_point is not None:
        ax.scatter(tli_point[0], tli_point[1], color="#FFD700", s=22, zorder=11)
        ax.annotate(
            "TLI \u2014 2 Apr 2026, 23:49 UTC",
            xy=(tli_point[0], tli_point[1]),
            xycoords="data",
            xytext=(0.62, 0.90),
            textcoords="axes fraction",
            color="#FFD700", fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#FFD700", lw=1.1),
            zorder=12,
        )

    fig.savefig(out_path, dpi=200, facecolor=BG)
    jpg_path = os.path.splitext(out_path)[0] + ".jpg"
    fig.savefig(jpg_path, dpi=200, facecolor=BG, format="jpeg")
    banner_path = os.path.splitext(out_path)[0] + "_banner.png"
    fig.set_size_inches(6.27, 12)
    fig.savefig(banner_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"Saved 2D \u2192 {out_path}")
    print(f"Saved 2D \u2192 {jpg_path}")
    print(f"Saved 2D \u2192 {banner_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Portrait static PNG export")
    parser.add_argument("--all",  dest="use_all",  action="store_true",
                        help="Plot all MC trajectories")
    parser.add_argument("--grid", dest="use_grid", action="store_true",
                        help="Plot grid search (OAT sensitivity) trajectories")
    args = parser.parse_args()

    df_traj, df_pos, ref_km, ref_burn_m, moon_pos, moon_at_flyby = load_data(args.use_all, args.use_grid)
    oem = load_oem(ASC_REFERENCE_PATH)

    n_traj = df_traj["solution_idx"].nunique()
    if args.use_grid:
        tag       = "grid"
        title_str = "Artemis II Trajectory \u2014 Sensitivity Plot"
    elif args.use_all:
        tag       = "all"
        title_str = f"Monte Carlo Trajectories (N={n_traj}, all) — ECI Frame (J2000)"
    else:
        tag       = "top100"
        title_str = f"Monte Carlo Trajectories (N={n_traj}, top-100) — ECI Frame (J2000)"

    use_many = args.use_all or args.use_grid
    traj_data, tli_point, park_point = preprocess(df_traj, df_pos)

    grp_counts = {g: len(v) for g, v in traj_data.items()}
    print(f"Groups: { {GROUP_LABELS[k]: v for k, v in grp_counts.items()} }")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_3d = os.path.join(OUTPUT_DIR, f"artemis_portrait_3d_{tag}.png")
    out_2d = os.path.join(OUTPUT_DIR, f"artemis_portrait_2d_{tag}.png")

    plot_3d(traj_data, ref_km, ref_burn_m, moon_pos, oem,
            tli_point, park_point, title_str, out_3d, use_many=use_many)

    plot_2d(traj_data, ref_km, ref_burn_m, tli_point, moon_pos, moon_at_flyby,
            title_str, out_2d, use_many=use_many)


if __name__ == "__main__":
    main()
