#!/usr/bin/env python3
"""
Artemis 2 — Social Media Animation Export

Renders the ECI vs Earth-Moon rotating frame animation as:
  MP4  (1920×1080, 30 fps)  — requires ffmpeg on PATH
  GIF  (960×540,   20 fps)  — fallback, requires Pillow

Reads:  out/artemis2_trajectory.csv
Output: out/artemis_animation.mp4  (or .gif)

Usage:
  python plot/export_animation.py
  python plot/export_animation.py --gif     # force GIF
  python plot/export_animation.py --frames 400  # more frames → smoother
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as manim
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(__file__)
NOM_CSV = os.path.join(_HERE, "..", "out", "artemis2_trajectory.csv")
OUT_DIR = os.path.join(_HERE, "..", "out")

EARTH_R_KM = 6_371.0
MOON_R_KM  = 1_737.4
TAIL_LEN   = 18    # number of animation steps shown as trailing line

BG      = "#0a0a12"
C_PATH  = "#FFD700"
C_SC    = "white"
C_MOON  = "#C8C8C8"
C_EARTH = "#1565C0"


# ── Data helpers ──────────────────────────────────────────────────────────────

def clip_at_earth_ca(t: np.ndarray, pos: np.ndarray) -> int:
    mask   = t >= 6.0
    offset = int(np.where(mask)[0][0])
    return offset + int(np.argmin(np.linalg.norm(pos[mask], axis=1)))


def rotating_frame(sc: np.ndarray, mn: np.ndarray, t: np.ndarray):
    """
    Project sc and mn into the Earth-Moon rotating orbital-plane frame.
    Returns sc_r (N,2), mn_r (N,2), moon_dist (float).
    """
    dt    = np.gradient(t) * 86_400.0
    mn_v  = np.gradient(mn, axis=0) / dt[:, None]
    h_vec = np.cross(mn, mn_v)
    z_hat = h_vec.mean(axis=0)
    z_hat /= np.linalg.norm(z_hat)

    sc_r = np.zeros((len(t), 2))
    mn_r = np.zeros((len(t), 2))
    for i in range(len(t)):
        x_hat = mn[i] / np.linalg.norm(mn[i])
        y_hat = np.cross(z_hat, x_hat)
        y_hat /= np.linalg.norm(y_hat)
        sc_r[i] = [np.dot(x_hat, sc[i]), np.dot(y_hat, sc[i])]
        mn_r[i] = [np.dot(x_hat, mn[i]), np.dot(y_hat, mn[i])]

    return sc_r, mn_r, float(np.mean(mn_r[:, 0]))


# ── Build animation ───────────────────────────────────────────────────────────

def build(n_frames: int, fps: int, dpi: int, figsize: tuple):
    if not os.path.exists(NOM_CSV):
        sys.exit(f"Missing {NOM_CSV}\nRun: cargo run --bin artemis --release")

    df  = pd.read_csv(NOM_CSV)
    km  = 1e-3
    sc  = df[["x_m", "y_m", "z_m"]].values * km
    mn  = df[["moon_x_m", "moon_y_m", "moon_z_m"]].values * km
    t   = df["time_s"].values / 86_400.0

    end    = clip_at_earth_ca(t, sc)
    sc, mn, t = sc[:end+1], mn[:end+1], t[:end+1]

    sc_r, mn_r, moon_dist = rotating_frame(sc, mn, t)

    # Downsample to n_frames
    idx   = np.round(np.linspace(0, len(t) - 1, n_frames)).astype(int)
    sc_a  = sc[idx, :2]
    mn_a  = mn[idx, :2]
    scr_a = sc_r[idx]
    t_a   = t[idx]

    # Draw radii (slightly larger than real for visibility)
    EARTH_DRAW = EARTH_R_KM * 0.3
    MOON_DRAW  = MOON_R_KM  * 0.6

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize,
        facecolor=BG,
        gridspec_kw={"wspace": 0.04},
    )

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.set_aspect("equal")
        ax.axis("off")

    # ── Axis ranges ───────────────────────────────────────────────────────────
    r_eci = float(np.max(np.abs(sc[:, :2]))) * 1.08
    ax1.set_xlim(-r_eci, r_eci)
    ax1.set_ylim(-r_eci, r_eci)

    r_rx_lo = float(sc_r[:, 0].min()) * 1.10
    r_rx_hi = (moon_dist + MOON_DRAW * 3) * 1.05
    r_ry    = float(np.max(np.abs(sc_r[:, 1]))) * 1.12
    # Keep equal aspect — expand tighter axis to match wider one
    x_span  = r_rx_hi - r_rx_lo
    y_span  = 2 * r_ry
    if y_span < x_span:
        pad   = (x_span - y_span) / 2
        r_ry += pad
    ax2.set_xlim(r_rx_lo, r_rx_hi)
    ax2.set_ylim(-r_ry, r_ry)

    # ── Static elements ───────────────────────────────────────────────────────
    # Full trajectory paths (dim)
    ax1.plot(sc[:, 0], sc[:, 1], color=C_PATH, alpha=0.20, lw=0.8, zorder=1)
    ax2.plot(sc_r[:, 0], sc_r[:, 1], color=C_PATH, alpha=0.20, lw=0.8, zorder=1)

    # Earth circles
    for ax in (ax1, ax2):
        ax.add_patch(Circle((0, 0), EARTH_DRAW,
                            color=C_EARTH, alpha=0.85, zorder=3))

    # Moon circle in rotating frame (fixed)
    ax2.add_patch(Circle((moon_dist, 0), MOON_DRAW,
                         color=C_MOON, alpha=0.70, zorder=3))

    # Labels
    lbl_kw = dict(fontsize=9, ha="center", va="bottom", zorder=5)
    ax1.text(0,  EARTH_DRAW * 1.5, "Earth", color=C_EARTH, **lbl_kw)
    ax2.text(0,  EARTH_DRAW * 1.5, "Earth", color=C_EARTH, **lbl_kw)
    ax2.text(moon_dist, MOON_DRAW * 1.6, "Moon",  color=C_MOON,  **lbl_kw)

    # Subplot titles
    title_kw = dict(color="white", fontsize=11, fontweight="bold",
                    transform=ax1.transAxes, ha="center")
    ax1.text(0.5, 1.01, "Inertial frame (ECI)",          **title_kw)
    ax2.text(0.5, 1.01, "Earth–Moon rotating frame",
             **{**title_kw, "transform": ax2.transAxes})

    # Main title
    fig.text(0.5, 0.97, "Artemis II — Free-Return Trajectory",
             color="white", fontsize=13, fontweight="bold", ha="center", va="top")

    # ── Animated artists ──────────────────────────────────────────────────────
    tail1,  = ax1.plot([], [], "-", color="white", alpha=0.55, lw=1.5, zorder=4)
    tail2,  = ax2.plot([], [], "-", color="white", alpha=0.55, lw=1.5, zorder=4)
    sc_dot1, = ax1.plot([], [], "o", color=C_SC,   ms=5, zorder=6)
    sc_dot2, = ax2.plot([], [], "o", color=C_SC,   ms=5, zorder=6)
    moon_dot, = ax1.plot([], [], "o", color=C_MOON, ms=5, zorder=5)

    time_txt = fig.text(0.5, 0.02, "", color="white", fontsize=10,
                        ha="center", va="bottom", alpha=0.75)

    artists = (tail1, tail2, sc_dot1, sc_dot2, moon_dot, time_txt)

    def init():
        for a in artists:
            if hasattr(a, "set_data"):
                a.set_data([], [])
        time_txt.set_text("")
        return artists

    def update(i):
        # Tail (last TAIL_LEN frames)
        lo = max(0, i - TAIL_LEN)
        tail1.set_data(sc_a[lo:i+1, 0],  sc_a[lo:i+1, 1])
        tail2.set_data(scr_a[lo:i+1, 0], scr_a[lo:i+1, 1])

        sc_dot1.set_data([sc_a[i, 0]],  [sc_a[i, 1]])
        sc_dot2.set_data([scr_a[i, 0]], [scr_a[i, 1]])
        moon_dot.set_data([mn_a[i, 0]], [mn_a[i, 1]])

        time_txt.set_text(f"T + {t_a[i]:.2f} days")
        return artists

    anim = manim.FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, blit=True, interval=1000 // fps,
    )

    return fig, anim


# ── Export ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif",    action="store_true", help="Force GIF output")
    parser.add_argument("--frames", type=int, default=300, help="Number of animation frames")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    force_gif   = args.gif
    has_ffmpeg  = shutil.which("ffmpeg") is not None
    use_mp4     = has_ffmpeg and not force_gif

    if use_mp4:
        fps, dpi      = 30, 120          # 1920×1080
        figsize       = (16, 9)
        out_path      = os.path.join(OUT_DIR, "artemis_animation.mp4")
        writer        = manim.FFMpegWriter(fps=fps, bitrate=6000,
                                           extra_args=["-vcodec", "libx264",
                                                       "-pix_fmt", "yuv420p"])
    else:
        if not force_gif and not has_ffmpeg:
            print("ffmpeg not found — falling back to GIF (install ffmpeg for MP4)")
        fps, dpi      = 20, 80           # 960×540 — manageable GIF size
        figsize       = (12, 6.75)
        out_path      = os.path.join(OUT_DIR, "artemis_animation.gif")
        writer        = manim.PillowWriter(fps=fps)

    print(f"Rendering {args.frames} frames → {out_path}  ({dpi} dpi, {fps} fps)")
    fig, anim = build(args.frames, fps, dpi, figsize)

    anim.save(out_path, writer=writer, dpi=dpi,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
