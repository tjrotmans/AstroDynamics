"""
MissionPlanner sim_engine — proximity mission visualisation.

Mirrors the layout of GNC/AutonomousNavigation/plot/plot_mission.py so outputs
from the two sims can be compared side-by-side.  May be removed once Phase 4
cross-validation is complete.

Reads from:
  MissionPlanner/out/<mission>/simulate/nav.csv
  MissionPlanner/out/<mission>/simulate/attitude.csv
  MissionPlanner/out/<mission>/simulate/maneuvers.csv

Produces (into the same directory):
  sim_orbit.png       3-D Hill-frame trajectory coloured by phase
  sim_nav.png         range + nav error + 3-sigma + orbital energy
  sim_attitude.png    pointing error + wheel speeds + momentum
  sim_dv.png          maneuver timeline + cumulative dV budget

Usage (from MissionPlanner/ directory):
  python plot/plot_mission_sim.py
  python plot/plot_mission_sim.py bennu_landing   # different mission sub-dir
"""

import sys
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("mission", nargs="?", default="bennu_sample_return",
                    help="Mission sub-directory under out/  (default: bennu_sample_return)")
args = parser.parse_args()

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / args.mission / "simulate"
OUT  = DATA

MU_BENNU  = 4.89        # m³/s²  (matches config.rs)
R_BENNU_M = 262.0       # m
WHEEL_MAX = 628.3        # rad/s  (6 000 RPM — matches config.rs)

PHASE_COLORS = {
    "Capture":     "#f06292",
    "Survey":      "#4fc3f7",
    "CloseOrbit":  "#ab47bc",
    "Flyover":     "#ff7043",
    "ScienceHold": "#66bb6a",
    "RadioScience":"#ffe066",
    "Transfer":    "#ffffff",   # white arc so transfer trajectories are distinct
    "Landing":     "#ef5350",
    "Approach":    "#80cbc4",
    "Orbit":       "#4fc3f7",
}

# ── Loader ────────────────────────────────────────────────────────────────────

def load(name):
    p = DATA / name
    if not p.exists():
        sys.exit(f"[ERROR] Missing {p}\nRun: cargo run --release -- simulate config/<mission>.toml")
    return np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8")

def to_str(arr, field):
    v = arr[field]
    return np.array([s.decode() if isinstance(s, bytes) else s for s in v])

print(f"Loading data from {DATA} ...")
nav = load("nav.csv")
att = load("attitude.csv")
man = load("maneuvers.csv")

t0_s  = nav["time_s"][0]
t_nav = (nav["time_s"] - t0_s) / 3600.0
t_att = (att["time_s"] - t0_s) / 3600.0
t_man = (man["time_s"] - t0_s) / 3600.0 if len(man) > 0 else np.array([])

phase_nav = to_str(nav, "phase")
phase_att = to_str(att, "phase")

# Derived quantities
r_truth = np.sqrt(nav["tx_m"]**2 + nav["ty_m"]**2 + nav["tz_m"]**2)
r_est   = np.sqrt(nav["ex_m"]**2 + nav["ey_m"]**2 + nav["ez_m"]**2)

dr = np.sqrt((nav["tx_m"] - nav["ex_m"])**2 +
             (nav["ty_m"] - nav["ey_m"])**2 +
             (nav["tz_m"] - nav["ez_m"])**2)
dv = np.sqrt((nav["tvx"] - nav["evx"])**2 +
             (nav["tvy"] - nav["evy"])**2 +
             (nav["tvz"] - nav["evz"])**2)

sigma_r = np.maximum(nav["sigma_r_m"], 0.0)

# Specific orbital energy: ε = v²/2 - μ/r
v2 = nav["tvx"]**2 + nav["tvy"]**2 + nav["tvz"]**2
eps = 0.5 * v2 - MU_BENNU / np.maximum(r_truth, 1e-3)

# Wheel saturation
w_frac = np.column_stack([
    np.abs(att["w1_rads"]) / WHEEL_MAX,
    np.abs(att["w2_rads"]) / WHEEL_MAX,
    np.abs(att["w3_rads"]) / WHEEL_MAX,
    np.abs(att["w4_rads"]) / WHEEL_MAX,
])
desat_mask = att["desat"].astype(bool)

# Cumulative dV
dv_cum = np.cumsum(man["dv_mag_ms"]) if len(man) > 0 else np.array([])

# ── Helpers ───────────────────────────────────────────────────────────────────

def shade_phases(ax, t, phases, alpha=0.12):
    if len(t) < 2:
        return
    prev, t0 = phases[0], t[0]
    for i in range(1, len(t)):
        if phases[i] != prev or i == len(t) - 1:
            ax.axvspan(t0, t[i], alpha=alpha,
                       color=PHASE_COLORS.get(prev, "gray"), zorder=0)
            t0, prev = t[i], phases[i]

def phase_patches(phases_present):
    return [mpatches.Patch(color=PHASE_COLORS[ph], alpha=0.6, label=ph)
            for ph in PHASE_COLORS if ph in phases_present]

def set_xl(ax):
    ax.set_xlabel("Elapsed time [h]")

present = set(phase_nav)

# ── Figure 1: 3-D Hill-frame trajectory ──────────────────────────────────────

print("Plotting: sim_orbit.png ...")
fig1 = plt.figure(figsize=(12, 9))
ax3  = fig1.add_subplot(111, projection="3d")
ax3.set_title(
    f"MissionPlanner sim_engine — Hill-Frame Trajectory ({args.mission})\n"
    "Coloured by phase  |  red dots = maneuver events", fontsize=10)
fig1.patch.set_facecolor("#0a0a1a")
ax3.set_facecolor("#0a0a1a")

tx = nav["tx_m"] / 1e3
ty = nav["ty_m"] / 1e3
tz = nav["tz_m"] / 1e3
ex = nav["ex_m"] / 1e3
ey = nav["ey_m"] / 1e3
ez = nav["ez_m"] / 1e3

for ph in list(dict.fromkeys(phase_nav)):
    mask = phase_nav == ph
    col  = PHASE_COLORS.get(ph, "gray")
    ax3.plot(tx[mask], ty[mask], tz[mask], color=col, lw=1.6, label=f"Truth – {ph}")

ax3.plot(ex, ey, ez, color="white", lw=0.8, ls="--", alpha=0.5, label="EKF est.")

# Bennu sphere wireframe
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 15)
xs = (R_BENNU_M/1e3) * np.outer(np.cos(u), np.sin(v))
ys = (R_BENNU_M/1e3) * np.outer(np.sin(u), np.sin(v))
zs = (R_BENNU_M/1e3) * np.outer(np.ones_like(u), np.cos(v))
ax3.plot_wireframe(xs, ys, zs, color="tan", linewidth=0.4, alpha=0.4)

# Maneuver markers
if len(man) > 0:
    for i in range(len(man)):
        if man["dv_mag_ms"][i] > 1e-5:
            idx = np.argmin(np.abs(nav["time_s"] - man["time_s"][i]))
            ax3.scatter(tx[idx], ty[idx], tz[idx], s=18, color="red", zorder=10)

ax3.set_xlabel("x [km]"); ax3.set_ylabel("y [km]"); ax3.set_zlabel("z [km]")
for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor("gray")
ax3.tick_params(colors="white"); ax3.title.set_color("white")
for axis in [ax3.xaxis, ax3.yaxis, ax3.zaxis]:
    axis.label.set_color("white")
ax3.legend(loc="upper left", fontsize=8,
           facecolor="#111", edgecolor="gray", labelcolor="white")
plt.tight_layout()
fig1.savefig(OUT / "sim_orbit.png", dpi=150, bbox_inches="tight",
             facecolor=fig1.get_facecolor())

# ── Figure 2: Navigation performance ─────────────────────────────────────────

print("Plotting: sim_nav.png ...")
fig2, axes2 = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig2.suptitle(f"sim_engine — Navigation Performance ({args.mission})", fontsize=13)

ax = axes2[0]
ax.plot(t_nav, r_truth / 1e3, color="#4fc3f7", lw=1.4, label="Truth range")
ax.plot(t_nav, r_est / 1e3,   color="#ef9a9a", lw=0.9, ls="--", alpha=0.8, label="EKF range")
ax.axhline(R_BENNU_M/1e3, color="tan", ls="--", lw=0.8, alpha=0.6,
           label=f"Bennu r={R_BENNU_M:.0f} m")
ax.set_ylabel("Range [km]")
ax.set_title("Spacecraft range from Bennu (truth and EKF estimate)")
shade_phases(ax, t_nav, phase_nav)
ax.legend(fontsize=8)

ax = axes2[1]
ax.semilogy(t_nav, dr + 1e-6, color="#ef5350", lw=1.2, label="|dr| truth-EKF")
ax.semilogy(t_nav, 3*sigma_r + 1e-6, color="#ffa726", lw=1.0, ls="--", alpha=0.8,
            label="3-sigma_r bound")
ax.set_ylabel("Position error [m]")
ax.set_title("Position navigation error vs 3-sigma bound")
ax.legend(fontsize=8)
shade_phases(ax, t_nav, phase_nav)

ax = axes2[2]
ax.semilogy(t_nav, dv + 1e-9, color="#ab47bc", lw=1.2, label="|dv| truth-EKF")
ax.semilogy(t_nav, nav["sigma_v_mps"] + 1e-9, color="#ce93d8", lw=0.9, ls="--",
            alpha=0.8, label="sigma_v bound")
ax.set_ylabel("Velocity error [m/s]")
ax.set_title("Velocity navigation error")
ax.legend(fontsize=8)
shade_phases(ax, t_nav, phase_nav)

ax = axes2[3]
ax.plot(t_nav, eps, color="#4db6ac", lw=1.2, label="Specific energy eps")
ax.axhline(0.0, color="white", lw=0.6, ls=":", alpha=0.5, label="Capture boundary eps=0")
ax.set_ylabel("eps [J/kg]")
ax.set_title("Orbital energy (negative = bound orbit)")
ax.legend(fontsize=8)
shade_phases(ax, t_nav, phase_nav)
set_xl(ax)

axes2[0].legend(handles=phase_patches(present) +
                axes2[0].get_legend_handles_labels()[0],
                fontsize=7, loc="upper right")
plt.tight_layout()
fig2.savefig(OUT / "sim_nav.png", dpi=150, bbox_inches="tight")

# ── Figure 3: Attitude & reaction wheels ─────────────────────────────────────

print("Plotting: sim_attitude.png ...")
fig3, axes3 = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig3.suptitle(f"sim_engine — Attitude & Reaction Wheels ({args.mission})", fontsize=13)

WHEEL_COLORS = ["#ef5350", "#42a5f5", "#66bb6a", "#ffa726"]

ax = axes3[0]
ax.plot(t_att, att["pointing_err_mrad"], color="#f06292", lw=0.8, label="Pointing error")
ax.axhline(0.5 * 180.0/math.pi * 1000.0, color="orange", ls="--", lw=0.8, alpha=0.6,
           label="~28 mrad approx dead-zone")
ax.set_ylabel("Pointing error [mrad]")
ax.set_title("Attitude pointing error (truth vs desired quaternion)")
shade_phases(ax, t_att, phase_att)
ax.legend(fontsize=8)

ax = axes3[1]
for i in range(4):
    ax.plot(t_att, w_frac[:, i] * 100.0,
            color=WHEEL_COLORS[i], lw=0.7, alpha=0.85, label=f"W{i+1}")
ax.axhline(80.0, color="white", ls="--", lw=0.7, alpha=0.5, label="Desat 80%")
ax.set_ylabel("Speed [% of max]")
ax.set_title("Reaction wheel speeds (% of 628.3 rad/s max)")
ax.legend(fontsize=8, ncol=5)
shade_phases(ax, t_att, phase_att)

ax = axes3[2]
ax.plot(t_att, att["omega_norm_rads"] * 1e3, color="#80cbc4", lw=0.8,
        label="|omega| body frame")
ax.set_ylabel("|omega| [mrad/s]")
ax.set_title("Total angular rate magnitude")
ax.legend(fontsize=8)
shade_phases(ax, t_att, phase_att)

ax = axes3[3]
ax.plot(t_att, att["h_w_norm_nms"] * 1e3, color="#4db6ac", lw=0.8, label="|H_w|")
desat_t = t_att[desat_mask]
if len(desat_t) > 0:
    ax.scatter(desat_t, np.zeros(len(desat_t)),
               s=5, color="#ff7043", alpha=0.6, label="Desat event", zorder=4)
ax.set_ylabel("|H_w| [mN*m*s]")
ax.set_title("Total wheel angular momentum + desaturation events")
ax.legend(fontsize=8)
shade_phases(ax, t_att, phase_att)
set_xl(ax)

plt.tight_layout()
fig3.savefig(OUT / "sim_attitude.png", dpi=150, bbox_inches="tight")

# ── Figure 4: Maneuvers ───────────────────────────────────────────────────────

print("Plotting: sim_dv.png ...")
fig4, axes4 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig4.suptitle(f"sim_engine — Maneuvers & dV Budget ({args.mission})", fontsize=13)

ax = axes4[0]
if len(man) > 0:
    label_man = to_str(man, "label")
    for lbl, col in [("SK", "#66bb6a"), ("Burn1", "#ef5350"),
                     ("Burn2", "#ffa726"), ("LOI", "#ab47bc"), ("Deorbit", "#80cbc4")]:
        mask_m = label_man == lbl
        if mask_m.any():
            ax.stem(t_man[mask_m], man["dv_mag_ms"][mask_m] * 1e3,
                    linefmt=col, markerfmt="D", basefmt=" ", label=lbl)
ax.set_ylabel("dV [mm/s]")
ax.set_title("Maneuver dV events")
ax.legend(fontsize=8)
shade_phases(ax, t_nav, phase_nav)

ax = axes4[1]
if len(dv_cum) > 0:
    ax.step(t_man, dv_cum, color="#4fc3f7", lw=1.4, where="post", label="Cumulative dV")
total_dv_val = dv_cum[-1] if len(dv_cum) > 0 else 0.0
ax.set_ylabel("Cumulative dV [m/s]")
ax.set_title(f"Total dV budget: {total_dv_val:.4f} m/s over {len(man)} burns")
ax.legend(fontsize=8)
shade_phases(ax, t_nav, phase_nav)

ax = axes4[2]
# Per-phase dV breakdown as a bar chart
phase_dv = {}
if len(man) > 0:
    nav_phases = {float(nav["time_s"][i]): phase_nav[i] for i in range(len(nav))}
    nav_t = nav["time_s"]
    for i in range(len(man)):
        idx = np.argmin(np.abs(nav_t - man["time_s"][i]))
        ph  = phase_nav[idx]
        phase_dv[ph] = phase_dv.get(ph, 0.0) + man["dv_mag_ms"][i]
if phase_dv:
    labels = list(phase_dv.keys())
    vals   = [phase_dv[ph] * 1e3 for ph in labels]  # mm/s
    cols   = [PHASE_COLORS.get(ph, "gray") for ph in labels]
    bars   = ax.bar(labels, vals, color=cols, alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("dV per phase [mm/s]")
ax.set_title("Station-keeping dV budget breakdown by phase")
set_xl(axes4[1])

plt.tight_layout()
fig4.savefig(OUT / "sim_dv.png", dpi=150, bbox_inches="tight")

# ── Summary printout ──────────────────────────────────────────────────────────

print()
print("=" * 54)
print("  sim_engine MISSION SUMMARY")
print("=" * 54)
print(f"  Duration          : {t_nav[-1]:.1f} h  ({t_nav[-1]/24:.1f} days)")
print(f"  Final range       : {r_truth[-1]/1e3:.4f} km")
print(f"  Final nav error   : {dr[-1]:.3f} m")
print(f"  Final EKF sigma_r : {sigma_r[-1]:.3f} m")
print(f"  Total dV          : {total_dv_val:.5f} m/s  ({len(man)} burns)")
print(f"  Max pointing err  : {att['pointing_err_mrad'].max():.2f} mrad")
print(f"  Max wheel sat     : {w_frac.max()*100:.1f}%")
print(f"  Desat firings     : {int(desat_mask.sum())}")
print(f"  Phases            : {' -> '.join(list(dict.fromkeys(phase_nav)))}")
print("=" * 54)
print()
print("Saved:")
for f in ["sim_orbit.png", "sim_nav.png", "sim_attitude.png", "sim_dv.png"]:
    print(f"  {OUT / f}")

plt.show()
