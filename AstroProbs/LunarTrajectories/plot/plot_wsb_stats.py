#!/usr/bin/env python3
"""
plot_wsb_stats.py — 1σ Monte Carlo statistics for WSB sensitivity.

Row 1  Horizontal stacked bar — outcome fractions at σ×1.0 with counts
Row 2  Violin distributions by outcome — ΔV error, pointing error, launch window

Reads:
  out/wsb/stats_samples[_TAG].csv
  out/wsb/stats_sweep[_TAG].csv

Outputs:
  out/wsb/wsb_stats[_TAG].png

Usage:
  python plot/plot_wsb_stats.py [--tag TAG]
"""
from __future__ import annotations
import argparse, math, pathlib, sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT  = ROOT / "out" / "wsb"

# ── Dark theme ─────────────────────────────────────────────────────────────────
BG        = "#0F0F19"
AX_BG     = "#13132A"
GRID_CLR  = "#2A2A4A"
TEXT_CLR  = "#CCCCEE"
TITLE_CLR = "#E0E0FF"

# ── Outcome colours / ordering ─────────────────────────────────────────────────
FLYBY_THRESH_ORBITS = 3   # n_orbits (spacecraft periapsis passages) < this → flyby

OUTCOME_COLS = {
    "captured":   "#3DFF8F",
    "flyby":      "#C77DFF",
    "moon_crash": "#FF4455",
    "escaped":    "#778899",
}
OUTCOME_ORDER  = ["captured", "flyby", "moon_crash", "escaped"]
OUTCOME_LABELS = {
    "captured":   "Captured",
    "flyby":      "Flyby",
    "moon_crash": "Moon crash",
    "escaped":    "Escaped",
}
SWEEP_COL = {
    "captured":   "frac_captured",
    "flyby":      "frac_flyby",
    "moon_crash": "frac_moon_crash",
    "escaped":    "frac_escaped",
}

# ── Physical unit conversions ──────────────────────────────────────────────────
_R_PARK_KM         = 6_371.0 + 378.0           # km
_GM_EARTH          = 398_600.4418              # km³ s⁻²
_OMEGA_PARK_DEG_S  = (math.sqrt(_GM_EARTH / _R_PARK_KM**3)
                      * (180.0 / math.pi))      # °/s  ≈ 0.0653
_SYNODIC_DAYS      = 29.530_589
_SUN_DEG_PER_MIN   = 360.0 / (_SYNODIC_DAYS * 24.0 * 60.0)  # °/min

_S_DV_PCT    = 1e-3 * 100.0
_S_POINT_DEG = math.degrees(1.75e-3)
_S_TIMING_S  = 0.05 / _OMEGA_PARK_DEG_S
_S_WIN_MIN   = 0.05 / _SUN_DEG_PER_MIN


# ── Helpers ────────────────────────────────────────────────────────────────────

def _setup_ax(ax, *, title="", xlabel="", ylabel=""):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#3A3A5A")
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TITLE_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.7)
    if title:  ax.set_title(title, fontsize=9, pad=5)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)


def _classify(df: pd.DataFrame) -> pd.Series:
    is_flyby = (df["outcome"] == "captured") & (df["n_orbits"] < FLYBY_THRESH_ORBITS)
    oc = df["outcome"].copy()
    oc[is_flyby] = "flyby"
    return oc


def _phys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dv_pct"]     = df["dv_mag_actual"] * 100.0
    df["point_deg"]  = df["dv_dir_actual_rad"] * (180.0 / math.pi)
    df["window_min"] = df["theta_sun_actual_deg"] / _SUN_DEG_PER_MIN
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    sfx          = f"_{args.tag}" if args.tag else ""
    samples_path = OUT / f"stats_samples{sfx}.csv"
    sweep_path   = OUT / f"stats_sweep{sfx}.csv"

    if not samples_path.exists():
        sys.exit(f"[error] {samples_path} not found — run wsb_stats first.\n"
                 "  cargo run -p lunar_trajectories --bin wsb_stats --release -- --hifi")

    df = pd.read_csv(samples_path)
    sw = pd.read_csv(sweep_path)
    df["outcome_class"] = _classify(df)
    df = _phys(df)

    # Use the σ=1.0 row from sweep; fall back to computing from samples
    row1 = sw[np.abs(sw["sigma_scale"] - 1.0) < 1e-6]
    if len(row1):
        fracs = {k: float(row1[SWEEP_COL[k]].iloc[0]) for k in OUTCOME_ORDER}
    else:
        n = len(df)
        fracs = {k: (df["outcome_class"] == k).sum() / n for k in OUTCOME_ORDER}

    n_total = len(df)
    counts  = {k: int(round(fracs[k] * n_total)) for k in OUTCOME_ORDER}

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9), facecolor=BG)
    gs  = fig.add_gridspec(
        2, 3,
        height_ratios=[0.55, 1.0],
        hspace=0.50, wspace=0.38,
        left=0.07, right=0.97, top=0.92, bottom=0.09,
    )
    ax_bar = fig.add_subplot(gs[0, :])
    ax_v   = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # ─────────────────────────────────────────────────────────────────────────
    # Row 1 — horizontal stacked outcome bar
    # ─────────────────────────────────────────────────────────────────────────
    ax_bar.set_facecolor(AX_BG)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor("#3A3A5A")

    left = 0.0
    for key in OUTCOME_ORDER:
        f = fracs[key]
        if f < 1e-6:
            continue
        ax_bar.barh(0, f, left=left, height=0.55,
                    color=OUTCOME_COLS[key], alpha=0.92,
                    linewidth=0.4, edgecolor="black")
        if f > 0.025:
            ax_bar.text(
                left + f / 2, 0,
                f"{OUTCOME_LABELS[key]}\n{f:.1%}\n({counts[key]:,})",
                ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold",
                linespacing=1.4,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="black")],
            )
        left += f

    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.6, 0.6)
    ax_bar.set_yticks([])
    ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_bar.tick_params(colors=TEXT_CLR, labelsize=8)
    ax_bar.xaxis.label.set_color(TEXT_CLR)
    ax_bar.title.set_color(TITLE_CLR)
    ax_bar.set_title(
        f"Outcome fractions at σ×1.0  "
        f"(n = {n_total:,} Gaussian samples)",
        fontsize=10, pad=6,
    )
    ax_bar.set_xlabel(
        f"1σ:  ΔV ±{_S_DV_PCT:.2f}%   ·   pointing ±{_S_POINT_DEG:.3f}°   ·   "
        f"burn timing ±{_S_TIMING_S:.2f} s   ·   launch window ±{_S_WIN_MIN:.1f} min",
        fontsize=7.5, style="italic",
    )
    ax_bar.grid(False)

    # ─────────────────────────────────────────────────────────────────────────
    # Row 2 — violin distributions by outcome (ΔV, pointing, launch window)
    # ─────────────────────────────────────────────────────────────────────────
    violin_cfg = [
        ("dv_pct",    "ΔV error",      "%"),
        ("point_deg", "Pointing error", "deg"),
        ("window_min","Launch window", "min"),
    ]
    for ax, (col, label, unit) in zip(ax_v, violin_cfg):
        nonempty = [
            (df.loc[df["outcome_class"] == oc, col].values, oc)
            for oc in OUTCOME_ORDER
            if (df["outcome_class"] == oc).sum() > 1
        ]
        data     = [g for g, _ in nonempty]
        x_labels = [OUTCOME_LABELS[oc] for _, oc in nonempty]
        cols     = [OUTCOME_COLS[oc]   for _, oc in nonempty]

        if data:
            vp = ax.violinplot(data, showmedians=True, showextrema=False)
            for body, c in zip(vp["bodies"], cols):
                body.set_facecolor(c)
                body.set_alpha(0.55)
                body.set_edgecolor("white")
                body.set_linewidth(0.55)
            vp["cmedians"].set_color("white")
            vp["cmedians"].set_linewidth(1.3)
            all_vals = np.concatenate(data)
            if all_vals.min() < 0 < all_vals.max():
                ax.axhline(0, color=TEXT_CLR, lw=0.6, alpha=0.35)

        ax.set_xticks(range(1, len(nonempty) + 1))
        ax.set_xticklabels(x_labels, fontsize=7.5, rotation=18, ha="right")
        _setup_ax(ax,
                  title=f"{label}  by outcome  (σ×1.0)",
                  ylabel=unit)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "WSB Transfer Sensitivity — 1σ Monte Carlo",
        fontsize=12, color=TITLE_CLR, y=0.975,
    )

    out_path = OUT / f"wsb_stats{sfx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
