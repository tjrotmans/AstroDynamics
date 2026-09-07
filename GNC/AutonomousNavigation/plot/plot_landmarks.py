"""
Landmark OpNav tracking visualisation.

Reads outputs from `cargo run --bin proximity_mission`:
  out/mission/landmark_frames.csv  -- per-frame spacecraft + camera + Sun geometry
  out/mission/landmark_obs.csv     -- per-landmark state within each frame

Produces an animation with two panels:
  LEFT  -- 3-D Hill frame: Bennu (rotating), surface landmarks colour-coded by
           state, the spacecraft, and its NavCam boresight cone.
  RIGHT -- synthetic NavCam frame: the FOV circle with the landmarks the filter
           currently recognises, marked with reticles.

Landmark state colours:
  green  = Tracked          (sunlit, near-side, in FOV -> used by the EKF)
  grey   = Visible, off-FOV (sunlit, near-side, outside the camera)
  dim    = Dark             (near-side but unlit)
  (FarSide landmarks are hidden -- they're behind Bennu.)

Usage:  python plot/plot_landmarks.py            # interactive
        python plot/plot_landmarks.py --save     # write mp4/gif to out/mission/
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

# ── Paths / constants ─────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / "mission"

R_BENNU_M       = 262.0                       # m (must match config.rs)
CAMERA_FOV_HALF = np.deg2rad(15.0)            # rad (must match config.rs)

STATE_FARSIDE, STATE_DARK, STATE_OFFFOV, STATE_TRACKED = 0, 1, 2, 3
STATE_COLOR = {STATE_DARK: "#3a3a3a", STATE_OFFFOV: "#9e9e9e", STATE_TRACKED: "#33dd55"}

PHASE_COLORS = {
    "Capture": "#f06292", "Survey": "#4fc3f7", "CloseOrbit": "#ab47bc",
    "Flyover": "#ff7043", "ScienceHold": "#66bb6a",
}

# ── Load ──────────────────────────────────────────────────────────────────────

def load(name):
    p = DATA / name
    if not p.exists():
        sys.exit(f"[ERROR] Missing {p}\nRun: cargo run --bin proximity_mission --release")
    return np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8")

frames = load("landmark_frames.csv")
obs    = load("landmark_obs.csv")

phases = np.array([s.decode() if isinstance(s, bytes) else s for s in frames["phase"]])
frame_ids = frames["frame"].astype(int)
N = len(frame_ids)

# Group landmark observations by frame for O(1) lookup during animation.
obs_frame = obs["frame"].astype(int)
obs_by_frame = {fid: np.where(obs_frame == fid)[0] for fid in frame_ids}

# ── Bennu sphere mesh (drawn once, static at origin) ──────────────────────────

_u = np.linspace(0, 2 * np.pi, 30)
_v = np.linspace(0, np.pi, 15)
BX = R_BENNU_M * np.outer(np.cos(_u), np.sin(_v))
BY = R_BENNU_M * np.outer(np.sin(_u), np.sin(_v))
BZ = R_BENNU_M * np.outer(np.ones_like(_u), np.cos(_v))

# ── Figure layout ─────────────────────────────────────────────────────────────

plt.rcParams["figure.facecolor"] = "#0a0a12"
fig = plt.figure(figsize=(15, 7.5))
fig.patch.set_facecolor("#0a0a12")
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
axcam = fig.add_subplot(1, 2, 2)

# Axis limits from the spacecraft excursion (so Bennu + S/C both always fit).
sc = np.column_stack([frames["scx"], frames["scy"], frames["scz"]])
lim = max(np.abs(sc).max(), R_BENNU_M) * 1.15


def boresight_cone(apex, axis, half_angle, length, n=24):
    """Return line segments outlining a cone from `apex` along unit `axis`."""
    axis = axis / np.linalg.norm(axis)
    # Build an orthonormal basis perpendicular to the axis.
    tmp = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    r = length * np.tan(half_angle)
    rim = [apex + length * axis + r * (np.cos(t) * e1 + np.sin(t) * e2)
           for t in np.linspace(0, 2 * np.pi, n)]
    return np.array(rim)


def draw_frame(k):
    fid = frame_ids[k]
    ax3d.clear(); axcam.clear()
    phase = phases[k]
    pcol = PHASE_COLORS.get(phase, "white")

    # ── 3-D panel ─────────────────────────────────────────────────────────────
    ax3d.set_facecolor("#0a0a12")
    ax3d.plot_surface(BX, BY, BZ, color="#5a5048", alpha=0.30,
                      linewidth=0, antialiased=True, shade=True, zorder=1)

    idx = obs_by_frame[fid]
    ox, oy, oz = obs["x"][idx], obs["y"][idx], obs["z"][idx]
    oid = obs["id"][idx].astype(int)
    ost = obs["state"][idx].astype(int)
    for st in (STATE_DARK, STATE_OFFFOV, STATE_TRACKED):
        m = ost == st
        if not m.any():
            continue
        ax3d.scatter(ox[m], oy[m], oz[m], s=(34 if st == STATE_TRACKED else 16),
                     color=STATE_COLOR[st], depthshade=False,
                     edgecolors="white" if st == STATE_TRACKED else "none",
                     linewidths=0.4, zorder=5)
    # Label tracked landmarks with their IDs so they can be matched 1:1 with the
    # NavCam panel (same green circles, same numbers) and tracked across frames.
    mt = ost == STATE_TRACKED
    for xx, yy, zz, lid in zip(ox[mt], oy[mt], oz[mt], oid[mt]):
        ax3d.text(xx, yy, zz, f" {lid}", color="#a8ffa8", fontsize=6, zorder=11)

    # Spacecraft + boresight cone toward Bennu.
    p = sc[k]
    ax3d.scatter(*p, s=90, marker="^", color=pcol, edgecolors="white",
                 linewidths=0.8, zorder=10)
    axis = np.array([frames["bx"][k], frames["by"][k], frames["bz"][k]])
    rim = boresight_cone(p, axis, CAMERA_FOV_HALF, np.linalg.norm(p))
    for r in rim[::3]:
        ax3d.plot([p[0], r[0]], [p[1], r[1]], [p[2], r[2]],
                  color=pcol, alpha=0.25, lw=0.6, zorder=3)
    ax3d.plot(rim[:, 0], rim[:, 1], rim[:, 2], color=pcol, alpha=0.5, lw=0.8)

    # Sun direction arrow (scaled to the plot).
    s = np.array([frames["sunx"][k], frames["suny"][k], frames["sunz"][k]])
    ax3d.quiver(0, 0, 0, s[0], s[1], s[2], length=lim * 0.9, color="#ffd54f",
                arrow_length_ratio=0.08, lw=1.5, zorder=4)

    ax3d.set_xlim(-lim, lim); ax3d.set_ylim(-lim, lim); ax3d.set_zlim(-lim, lim)
    ax3d.set_box_aspect((1, 1, 1))
    for a in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        a.pane.set_facecolor("#0a0a12"); a.pane.set_alpha(1.0)
        a.label.set_color("#888")
    ax3d.tick_params(colors="#666", labelsize=7)
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    t_h = frames["time_s"][k] / 3600.0
    n_tr = int(frames["n_tracked"][k])
    err = frames["nav_err_m"][k]
    ax3d.set_title(f"{phase}   t={t_h:6.1f} h   tracked={n_tr:2d}   nav err={err:6.1f} m",
                   color=pcol, fontsize=11, pad=4)

    # ── NavCam panel ────────────────────────────────────────────────────────────
    axcam.set_facecolor("#04040a")
    fov_deg = np.rad2deg(CAMERA_FOV_HALF)
    circ = plt.Circle((0, 0), fov_deg, fill=False, color="#4fc3f7", lw=1.5)
    axcam.add_patch(circ)
    axcam.plot(0, 0, "+", color="#4fc3f7", ms=10, mew=1.0)  # boresight

    tracked = idx[ost == STATE_TRACKED]
    if len(tracked):
        # Camera convention: the boresight (+x_body) looks INTO the scene toward
        # Bennu, with +z_body up.  For a right-handed frame that makes screen-right
        # = -Y_body, so we negate the horizontal axis (a +Y_body landmark appears on
        # the left).  Vertical is +Z_body (up), unchanged.
        cy = -np.rad2deg(obs["cam_y"][tracked])
        cz =  np.rad2deg(obs["cam_z"][tracked])
        axcam.scatter(cy, cz, s=70, facecolors="none", edgecolors="#33dd55", lw=1.4)
        axcam.scatter(cy, cz, s=6, color="#33dd55")
        for yy, zz, lid in zip(cy, cz, obs["id"][tracked]):
            axcam.text(yy + 0.4, zz + 0.4, str(int(lid)), color="#8f8", fontsize=6)

    axcam.set_xlim(-fov_deg * 1.05, fov_deg * 1.05)
    axcam.set_ylim(-fov_deg * 1.05, fov_deg * 1.05)
    axcam.set_aspect("equal")
    axcam.set_xlabel("screen right  (-Y_body) [deg]", color="#888")
    axcam.set_ylabel("screen up  (+Z_body) [deg]", color="#888")
    axcam.tick_params(colors="#666", labelsize=7)
    axcam.set_title(f"NavCam — recognising {len(tracked)} landmarks",
                    color="#33dd55", fontsize=11, pad=4)
    for sp in axcam.spines.values():
        sp.set_color("#333")


def main():
    save = "--save" in sys.argv
    anim = FuncAnimation(fig, draw_frame, frames=N, interval=80, repeat=True)
    plt.tight_layout()
    if save:
        out = DATA / "landmark_tracking.mp4"
        try:
            anim.save(out, fps=12, dpi=110)
            print(f"Saved {out}")
        except Exception as e:                      # noqa: BLE001
            out = DATA / "landmark_tracking.gif"
            anim.save(out, fps=12, writer="pillow")
            print(f"(mp4 failed: {e})\nSaved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
