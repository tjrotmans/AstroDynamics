"""
Solar radiation pressure: panel geometry + Gauss-Markov stochastic-acceleration.

Reads out/mission/srp.csv and out/mission/nav.csv (written by proximity_mission).

TOP-LEFT    -- 3-D spacecraft geometry oriented by the true attitude each frame:
               gold bus box and two dark solar panels, Sun direction (yellow),
               and the truth SRP force vector (red).  As the spacecraft slews to
               stay nadir-pointed, watch the lit faces — and the SRP force — swing
               around: this attitude dependence is what the cannonball filter misses.
BOTTOM-LEFT -- 2-D orbit top-down view (inertial x-y).  Full trajectory coloured
               by mission phase; bright dot = current position; white trail = recent
               30 nav steps.  Bennu shown as a brown disk (~245 m radius).
TOP-RIGHT   -- |truth SRP| force magnitude over the mission.
MID-RIGHT   -- filter C_R: drops 1.40 → ~0.68 during the green radio-science arc
               because the solar panels spend most of the orbit edge-on to the Sun
               (nadir-pointing), so the effective cannonball area is much smaller
               than the initial 1.4 × 4 m² guess.
BOT-RIGHT   -- SRP residual |truth − filter| vs EKF Gauss-Markov |a_stoch|.
               a_stoch stays ~0 under dense OpNav (weak observability); it activates
               during prediction gaps when the force accumulates unobserved.

Usage:  python plot/plot_srp.py            # interactive
        python plot/plot_srp.py --save     # write mp4/gif
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / "mission"

BUS = np.array([2.0, 2.0, 0.63])
PANEL_SPAN, PANEL_CHORD = 2.5, 0.8

PHASE_COLORS = {
    "Capture":      "#f06292",
    "Survey":       "#4fc3f7",
    "RadioScience": "#80cbc4",
    "CloseOrbit":   "#ab47bc",
    "Flyover":      "#ff7043",
    "ScienceHold":  "#66bb6a",
}

BENNU_R_KM  = 0.245   # mean equatorial radius [km]
TRAIL_STEPS = 60      # nav points shown as bright trail


def load(name):
    p = DATA / name
    if not p.exists():
        sys.exit(f"[ERROR] Missing {p}\nRun: cargo run --bin proximity_mission --release")
    return np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8")


# ── SRP data ──────────────────────────────────────────────────────────────────
d = load("srp.csv")
phase = np.array([s.decode() if isinstance(s, bytes) else s for s in d["phase"]])
t_h   = d["time_s"] / 3600.0
N     = len(t_h)

q       = np.column_stack([d["qw"], d["qx"], d["qy"], d["qz"]])
sun     = np.column_stack([d["sunx"], d["suny"], d["sunz"]])
a_truth = np.column_stack([d["atx"], d["aty"], d["atz"]])
resid   = np.column_stack([d["rx"], d["ry"], d["rz"]])
est     = np.column_stack([d["ex"], d["ey"], d["ez"]])
a_truth_mag = np.linalg.norm(a_truth, axis=1) * 1e9
resid_mag   = np.linalg.norm(resid,   axis=1) * 1e9
est_mag     = np.linalg.norm(est,     axis=1) * 1e9
cr          = d["cr"]

# ── Nav / orbit data ──────────────────────────────────────────────────────────
nav = load("nav.csv")
nav_t   = nav["time_s"].astype(float)
nav_x   = nav["tx_m"].astype(float) / 1e3   # km
nav_y   = nav["ty_m"].astype(float) / 1e3
nav_z   = nav["tz_m"].astype(float) / 1e3
nav_ph  = np.array([s.decode() if isinstance(s, bytes) else s for s in nav["phase"]])

# For each SRP frame k find the closest nav row by time.
nav_idx = np.searchsorted(nav_t, d["time_s"].astype(float))
nav_idx = np.clip(nav_idx, 0, len(nav_t) - 1)

# Pre-build per-phase scatter arrays for the orbit background (drawn once, kept).
_ph_xy = {ph: (nav_x[nav_ph == ph], nav_y[nav_ph == ph])
          for ph in PHASE_COLORS if (nav_ph == ph).any()}


# ── Spacecraft geometry ───────────────────────────────────────────────────────
def quat_to_R(qq):
    w, x, y, z = qq
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def bus_faces():
    hx, hy, hz = BUS / 2
    def v(sx, sy, sz):
        return np.array([sx*hx, sy*hy, sz*hz])
    return [
        [v(1,-1,-1), v(1, 1,-1), v(1, 1, 1), v(1,-1, 1)],
        [v(-1,-1,-1), v(-1,1,-1), v(-1,1,1), v(-1,-1,1)],
        [v(-1,1,-1), v(1,1,-1), v(1,1,1), v(-1,1,1)],
        [v(-1,-1,-1), v(1,-1,-1), v(1,-1,1), v(-1,-1,1)],
        [v(-1,-1,1), v(1,-1,1), v(1,1,1), v(-1,1,1)],
        [v(-1,-1,-1), v(1,-1,-1), v(1,1,-1), v(-1,1,-1)],
    ]


def panel_faces():
    hx = PANEL_CHORD / 2
    y0 = BUS[1] / 2
    y1 = y0 + PANEL_SPAN
    return [
        [[-hx, y0, 0], [hx, y0, 0], [hx, y1, 0], [-hx, y1, 0]],
        [[-hx, -y0, 0], [hx, -y0, 0], [hx, -y1, 0], [-hx, -y1, 0]],
    ]


BUS_F   = bus_faces()
PANEL_F = [np.array(f) for f in panel_faces()]

# Bennu disk for orbit panel
_theta   = np.linspace(0, 2*np.pi, 120)
_bennu_x = BENNU_R_KM * np.cos(_theta)
_bennu_y = BENNU_R_KM * np.sin(_theta)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor="#0a0a12")
outer = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.0], wspace=0.24, figure=fig)

gs_left  = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[0], height_ratios=[1.7, 1.0], hspace=0.30)
gs_right = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=outer[1], hspace=0.52)

ax3d   = fig.add_subplot(gs_left[0], projection="3d")
ax_orb = fig.add_subplot(gs_left[1])
axm    = fig.add_subplot(gs_right[0])
axcr   = fig.add_subplot(gs_right[1])
axr    = fig.add_subplot(gs_right[2])

LIM = 4.0


# ── Draw function (called each animation frame) ───────────────────────────────
def draw(k):
    pcol = PHASE_COLORS.get(phase[k], "white")

    # ── Spacecraft 3D ────────────────────────────────────────────────────────
    ax3d.clear()
    ax3d.set_facecolor("#0a0a12")
    R = quat_to_R(q[k])

    bus_poly = [(R @ np.array(f).T).T for f in BUS_F]
    ax3d.add_collection3d(Poly3DCollection(bus_poly, facecolor="#b8902f",
                          edgecolor="#5a4510", alpha=0.85))
    pan_poly = [(R @ f.T).T for f in PANEL_F]
    ax3d.add_collection3d(Poly3DCollection(pan_poly, facecolor="#1b3a6b",
                          edgecolor="#0d1d38", alpha=0.92))

    s    = sun[k]
    fhat = a_truth[k] / (np.linalg.norm(a_truth[k]) + 1e-30)
    ax3d.quiver(0,0,0, s[0],s[1],s[2], length=LIM*0.85,
                color="#ffd54f", arrow_length_ratio=0.12, lw=2.0)
    ax3d.text(*(s * LIM * 0.9), "Sun", color="#ffd54f", fontsize=9)
    ax3d.quiver(0,0,0, fhat[0],fhat[1],fhat[2], length=LIM*0.7,
                color="#ff5252", arrow_length_ratio=0.14, lw=2.0)
    ax3d.text(*(fhat * LIM * 0.78), "SRP", color="#ff5252", fontsize=9)

    ax3d.set_xlim(-LIM, LIM); ax3d.set_ylim(-LIM, LIM); ax3d.set_zlim(-LIM, LIM)
    ax3d.set_box_aspect((1, 1, 1))
    for a in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        a.pane.set_facecolor("#0a0a12"); a.label.set_color("#888")
    ax3d.tick_params(colors="#666", labelsize=7)
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title(f"{phase[k]}   t={t_h[k]:.1f} h   |SRP|={a_truth_mag[k]:.1f} nm/s²",
                   color=pcol, fontsize=10)

    # ── Orbit top-down ───────────────────────────────────────────────────────
    ax_orb.clear()
    ax_orb.set_facecolor("#04040a")

    # Full trajectory coloured by phase (faint background).
    for ph, (px, py) in _ph_xy.items():
        ax_orb.scatter(px, py, c=PHASE_COLORS[ph], s=0.6, alpha=0.45, linewidths=0)

    # Bennu disk.
    ax_orb.fill(_bennu_x, _bennu_y, color="#6d4c2a", alpha=0.80, zorder=3)
    ax_orb.plot(_bennu_x, _bennu_y, color="#8d6e40", lw=0.6, zorder=4)

    # White trail of recent positions.
    ni = nav_idx[k]
    ts = max(0, ni - TRAIL_STEPS)
    ax_orb.plot(nav_x[ts:ni+1], nav_y[ts:ni+1],
                color="white", lw=1.0, alpha=0.55, zorder=5)

    # Current position marker.
    ax_orb.scatter([nav_x[ni]], [nav_y[ni]], c=pcol, s=55,
                   zorder=6, edgecolors="white", linewidths=0.5)
    r_km = np.sqrt(nav_x[ni]**2 + nav_y[ni]**2 + nav_z[ni]**2)
    ax_orb.text(nav_x[ni] + 0.05, nav_y[ni] + 0.05,
                f"{r_km:.2f} km", color=pcol, fontsize=6.5, zorder=7)

    # Phase legend (compact).
    leg = [Line2D([0],[0], marker="o", color="w", markerfacecolor=c,
                  markersize=5, linestyle="none", label=ph)
           for ph, c in PHASE_COLORS.items() if ph in _ph_xy]
    ax_orb.legend(handles=leg, loc="upper right", fontsize=5.5,
                  facecolor="#0a0a12", edgecolor="#333", labelcolor="#ccc",
                  handletextpad=0.3, borderpad=0.4)

    ax_orb.set_aspect("equal")
    ax_orb.set_xlabel("x [km]", color="#888", fontsize=7)
    ax_orb.set_ylabel("y [km]", color="#888", fontsize=7)
    ax_orb.tick_params(colors="#666", labelsize=6)
    for sp in ax_orb.spines.values():
        sp.set_color("#333")
    ax_orb.set_title("Orbit top-down  (inertial x–y)", color="#aaa", fontsize=8)

    # ── Time-series panels ───────────────────────────────────────────────────
    for ax in (axm, axcr, axr):
        ax.clear(); ax.set_facecolor("#04040a")
        ax.tick_params(colors="#888", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#333")
        ax.axvline(t_h[k], color="white", lw=0.8, alpha=0.6)

    quiet = phase == "RadioScience"
    if quiet.any():
        t0, t1 = t_h[quiet].min(), t_h[quiet].max()
        for ax in (axm, axcr, axr):
            ax.axvspan(t0, t1, color="#80cbc4", alpha=0.10)

    axm.plot(t_h, a_truth_mag, color="#ff8a65", lw=1.0)
    axm.set_ylabel("|truth SRP|\n[nm/s²]", color="#888", fontsize=8)
    axm.set_title("Attitude-dependent SRP force magnitude", color="#ff8a65", fontsize=9)

    axcr.plot(t_h, cr, color="#ffd54f", lw=1.3)
    axcr.set_ylabel("filter C_R", color="#888", fontsize=8)
    axcr.set_title("C_R calibrated in radio-science arc (teal shade)",
                   color="#ffd54f", fontsize=9)

    axr.plot(t_h, resid_mag, color="#4fc3f7", lw=1.1,
             label="|truth − filter|  (unmodelled)")
    axr.plot(t_h, est_mag,   color="#80cbc4", lw=1.1,
             label="|EKF Gauss-Markov|")
    axr.set_ylabel("nongrav accel\n[nm/s²]", color="#888", fontsize=8)
    axr.set_xlabel("time [h]", color="#888", fontsize=7)
    axr.legend(loc="upper right", fontsize=7, facecolor="#0a0a12",
               edgecolor="#333", labelcolor="#ccc")
    axr.set_title("Residual collapses as C_R calibrates; Markov tracks the rest",
                  color="#4fc3f7", fontsize=9)


def main():
    save = "--save" in sys.argv
    anim = FuncAnimation(fig, draw, frames=N, interval=80, repeat=True)
    plt.tight_layout()
    if save:
        out = DATA / "srp_geometry.mp4"
        try:
            anim.save(out, fps=12, dpi=110); print(f"Saved {out}")
        except Exception as e:                          # noqa: BLE001
            out = DATA / "srp_geometry.gif"
            anim.save(out, fps=12, writer="pillow"); print(f"(mp4 failed: {e})\nSaved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
