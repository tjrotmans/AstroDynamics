"""
Bennu 5-phase proximity mission visualisation.

Reads outputs from `cargo run --bin proximity_mission`:
  out/mission/nav.csv          -- truth + EKF state (logged at meas_every)
  out/mission/attitude.csv     -- attitude + wheel dynamics (every step)
  out/mission/maneuvers.csv    -- delta-V log
  out/mission/dsn_updates.csv  -- DSN pass log

Produces:
  out/mission/mission_orbit.png      -- 3-D Hill-frame trajectory by phase
  out/mission/mission_nav.png        -- range + nav error + sigma + DSN
  out/mission/mission_attitude.png   -- pointing error + wheel speeds + torques
  out/mission/mission_dv.png         -- maneuver timeline + cumulative budget

Usage:  python plot/plot_mission.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import Normalize
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / "mission"
OUT  = DATA

# ── Constants (must match config.rs) ─────────────────────────────────────────

MU_BENNU  = 6.674e-11 * 7.329e10   # m^3/s^2
R_BENNU_M = 262.0                   # m
WHEEL_MAX  = 628.3                  # rad/s  (6000 RPM)

PHASE_COLORS = {
    "Capture":     "#f06292",  # pink
    "Survey":      "#4fc3f7",  # sky blue
    "CloseOrbit":  "#ab47bc",  # purple
    "Flyover":     "#ff7043",  # orange
    "ScienceHold": "#66bb6a",  # green
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load(name):
    p = DATA / name
    if not p.exists():
        sys.exit(f"[ERROR] Missing {p}\nRun: cargo run --bin proximity_mission --release")
    return np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8")

def to_str(arr, field):
    """Return field as array of Python str (handles bytes from older numpy)."""
    v = arr[field]
    return np.array([s.decode() if isinstance(s, bytes) else s for s in v])

print("Loading data ...")
nav = load("nav.csv")
att = load("attitude.csv")
man = load("maneuvers.csv")
dsn = load("dsn_updates.csv")

# Use elapsed time from mission start so x-axis is readable
t0_s   = nav["time_s"][0]
t_nav  = (nav["time_s"] - t0_s) / 3600.0   # hours from mission start
t_att  = (att["time_s"] - t0_s) / 3600.0
t_man  = (man["time_s"] - t0_s) / 3600.0 if len(man) > 0 else np.array([])
t_dsn  = (dsn["time_s"] - t0_s) / 3600.0 if len(dsn) > 0 else np.array([])

phase_nav = to_str(nav, "phase")
phase_att = to_str(att, "phase")
label_man = to_str(man, "label")

# Nav error (truth - EKF)
dr = np.sqrt((nav["tx_m"] - nav["ex_m"])**2 +
             (nav["ty_m"] - nav["ey_m"])**2 +
             (nav["tz_m"] - nav["ez_m"])**2)
dv = np.sqrt((nav["tvx"]  - nav["evx"])**2  +
             (nav["tvy"]  - nav["evy"])**2  +
             (nav["tvz"]  - nav["evz"])**2)

sigma_r = np.maximum(nav["sigma_r_m"], 0.0)

# Orbital energy from truth state
eps = (0.5 * (nav["tvx"]**2 + nav["tvy"]**2 + nav["tvz"]**2)
       - MU_BENNU / (nav["range_km"] * 1e3))

# Cumulative dV
dv_cum = np.cumsum(man["dv_mag_ms"]) if len(man) > 0 else np.array([])

# Wheel saturation fraction
for i in range(1, 5):
    att[f"w{i}_rads"]  # confirm field exists; genfromtxt names it w1_rads etc.
w_frac = np.column_stack([
    np.abs(att["w1_rads"]) / WHEEL_MAX,
    np.abs(att["w2_rads"]) / WHEEL_MAX,
    np.abs(att["w3_rads"]) / WHEEL_MAX,
    np.abs(att["w4_rads"]) / WHEEL_MAX,
])
desat_mask = att["desat"].astype(bool)

# ── Helpers ───────────────────────────────────────────────────────────────────

def shade_phases(ax, t, phases, alpha=0.12):
    """Shade axes background by mission phase."""
    if len(t) < 2:
        return
    prev, t0 = phases[0], t[0]
    for i in range(1, len(t)):
        if phases[i] != prev or i == len(t) - 1:
            ax.axvspan(t0, t[i], alpha=alpha,
                       color=PHASE_COLORS.get(prev, "gray"), zorder=0)
            t0, prev = t[i], phases[i]

def phase_patches(present_phases):
    return [mpatches.Patch(color=PHASE_COLORS[ph], alpha=0.6, label=ph)
            for ph in PHASE_COLORS if ph in present_phases]

def set_time_label(ax):
    ax.set_xlabel("Elapsed time [h]")

# ── Figure 1: 3-D Hill-frame trajectory ──────────────────────────────────────

print("Plotting: mission_orbit.png ...")
fig1 = plt.figure(figsize=(12, 9))
ax3  = fig1.add_subplot(111, projection="3d")
ax3.set_title("5-Phase Proximity Mission — Hill-Frame Trajectory\n"
              "Colour by phase  |  arrows = body-x boresight  "
              "(Nadir → –r̂,  VelocityAligned → v̂)", fontsize=10)

tx = nav["tx_m"] / 1e3
ty = nav["ty_m"] / 1e3
tz = nav["tz_m"] / 1e3
ex = nav["ex_m"] / 1e3
ey = nav["ey_m"] / 1e3
ez = nav["ez_m"] / 1e3

# Draw truth trajectory segment-by-segment coloured by phase
phase_names = list(dict.fromkeys(phase_nav))   # preserve order, remove dups
for ph in phase_names:
    mask = phase_nav == ph
    col  = PHASE_COLORS.get(ph, "gray")
    ax3.plot(tx[mask], ty[mask], tz[mask],
             color=col, lw=1.6, label=f"Truth – {ph}")

# EKF estimate (thin dashed white)
ax3.plot(ex, ey, ez, color="white", lw=0.8, ls="--", alpha=0.5, label="EKF est.")

# ── Pointing axis arrows (body-x boresight, approximate from pointing mode) ──
# Nadir: boresight → –r̂ (toward Bennu).  VelocityAligned: boresight → v̂.
# Arrow length scales with current range so they stay proportional at all phases.
_r_norms = np.maximum(
    np.sqrt(nav["tx_m"]**2 + nav["ty_m"]**2 + nav["tz_m"]**2), 1.0)
_v_norms = np.maximum(
    np.sqrt(nav["tvx"]**2 + nav["tvy"]**2 + nav["tvz"]**2), 1e-9)
_arr_stride = max(1, len(tx) // 45)   # ≈45 arrows over the full mission

for _i in range(0, len(tx), _arr_stride):
    _ph  = phase_nav[_i]
    _col = PHASE_COLORS.get(_ph, "gray")
    _len = nav["range_km"][_i] * 0.06   # 6 % of current range, in km
    if _ph == "ScienceHold":
        _ux = nav["tvx"][_i] / _v_norms[_i] * _len
        _uy = nav["tvy"][_i] / _v_norms[_i] * _len
        _uz = nav["tvz"][_i] / _v_norms[_i] * _len
    else:
        _ux = -nav["tx_m"][_i] / _r_norms[_i] * _len
        _uy = -nav["ty_m"][_i] / _r_norms[_i] * _len
        _uz = -nav["tz_m"][_i] / _r_norms[_i] * _len
    ax3.quiver(tx[_i], ty[_i], tz[_i], _ux, _uy, _uz,
               color=_col, alpha=0.70, arrow_length_ratio=0.35, linewidth=0.8)

# Bennu sphere wireframe
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 15)
xs = (R_BENNU_M/1e3) * np.outer(np.cos(u), np.sin(v))
ys = (R_BENNU_M/1e3) * np.outer(np.sin(u), np.sin(v))
zs = (R_BENNU_M/1e3) * np.outer(np.ones_like(u), np.cos(v))
ax3.plot_wireframe(xs, ys, zs, color="tan", linewidth=0.4, alpha=0.4)

# Manoeuvre markers
if len(man) > 0:
    for i in range(len(man)):
        if man["dv_mag_ms"][i] > 1e-5:
            idx = np.argmin(np.abs(nav["time_s"] - man["time_s"][i]))
            ax3.scatter(tx[idx], ty[idx], tz[idx],
                        s=18, color="red", zorder=10)

ax3.set_xlabel("x [km]"); ax3.set_ylabel("y [km]"); ax3.set_zlabel("z [km]")
ax3.set_facecolor("#0a0a1a")
fig1.patch.set_facecolor("#0a0a1a")
for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor("gray")
ax3.tick_params(colors="white"); ax3.title.set_color("white")
for axis in [ax3.xaxis, ax3.yaxis, ax3.zaxis]:
    axis.label.set_color("white")
ax3.legend(loc="upper left", fontsize=8,
           facecolor="#111", edgecolor="gray", labelcolor="white")
plt.tight_layout()
fig1.savefig(OUT / "mission_orbit.png", dpi=150, bbox_inches="tight",
             facecolor=fig1.get_facecolor())

# ── Figure 2: Navigation ──────────────────────────────────────────────────────

print("Plotting: mission_nav.png ...")
fig2, axes2 = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig2.suptitle("Proximity Mission — Navigation Performance", fontsize=13)

# (0) Range profile
ax = axes2[0]
ax.plot(t_nav, nav["range_km"], color="#4fc3f7", lw=1.4)
ax.axhline(R_BENNU_M/1e3, color="tan", ls="--", lw=0.8, alpha=0.6, label=f"Bennu r={R_BENNU_M:.0f} m")
ax.set_ylabel("Range [km]")
ax.set_title("Spacecraft range from Bennu")
shade_phases(ax, t_nav, phase_nav)
ax.legend(fontsize=8)

# (1) Position error vs 3-sigma
ax = axes2[1]
ax.semilogy(t_nav, dr, color="#ef5350", lw=1.2, label="|Δr| truth-EKF")
ax.semilogy(t_nav, 3*sigma_r, color="#ffa726", lw=1.0, ls="--", alpha=0.8, label="3σ_r bound")
for td in t_dsn:
    ax.axvline(td, color="#66bb6a", lw=0.5, alpha=0.6)
ax.plot([], [], color="#66bb6a", lw=0.8, label="DSN pass")
ax.set_ylabel("Position error [m]")
ax.set_title("Navigation accuracy (truth – EKF)")
ax.legend(fontsize=8)
shade_phases(ax, t_nav, phase_nav)

# (2) Velocity error
ax = axes2[2]
ax.semilogy(t_nav, dv + 1e-9, color="#ab47bc", lw=1.2, label="|Δv| truth-EKF")
ax.set_ylabel("Velocity error [m/s]")
ax.set_title("Velocity navigation accuracy")
ax.legend(fontsize=8)
shade_phases(ax, t_nav, phase_nav)

# (3) Orbital energy convergence
ax = axes2[3]
ax.plot(t_nav, eps, color="#4db6ac", lw=1.2, label="Specific energy ε")
ax.axhline(0.0, color="white", lw=0.6, ls=":", alpha=0.5, label="Capture boundary ε=0")
ax.set_ylabel("ε [J/kg]")
ax.set_title("Orbital energy (negative = captured)")
ax.legend(fontsize=8)
shade_phases(ax, t_nav, phase_nav)
set_time_label(ax)

# Phase legend
present = set(phase_nav)
for a in axes2:
    for pp in phase_patches(present):
        pass   # already shaded; add only to last axis
axes2[0].legend(handles=phase_patches(present) + axes2[0].get_legend_handles_labels()[0],
                fontsize=7, loc="upper right")

plt.tight_layout()
fig2.savefig(OUT / "mission_nav.png", dpi=150, bbox_inches="tight")

# ── Figure 3: Attitude, reaction wheels & torques ────────────────────────────

print("Plotting: mission_attitude.png ...")

# Load new torque/pointing columns (backward-compatible)
try:
    tau_pd_norm  = np.sqrt(att["tau_pd_x_nm"]**2  + att["tau_pd_y_nm"]**2  + att["tau_pd_z_nm"]**2)
    tau_srp_norm = np.sqrt(att["tau_srp_x_nm"]**2 + att["tau_srp_y_nm"]**2 + att["tau_srp_z_nm"]**2)
    tau_gg_norm  = np.sqrt(att["tau_gg_x_nm"]**2  + att["tau_gg_y_nm"]**2  + att["tau_gg_z_nm"]**2)
    has_torques  = True
except (ValueError, KeyError):
    has_torques  = False

try:
    pointing_mode_att = to_str(att, "pointing_mode")
except (ValueError, KeyError):
    pointing_mode_att = np.where(phase_att == "ScienceHold", "VelocityAligned", "Nadir")

fig3, axes3 = plt.subplots(5, 1, figsize=(14, 18), sharex=True,
                            gridspec_kw={"height_ratios": [3, 2.5, 0.8, 2.5, 2]})
fig3.suptitle("Proximity Mission — Attitude, Reaction Wheels & Torque Budget", fontsize=13)

WHEEL_COLORS = ["#ef5350", "#42a5f5", "#66bb6a", "#ffa726"]

# (0) Pointing error
ax = axes3[0]
ax.plot(t_att, att["pointing_err_mrad"], color="#f06292", lw=0.8, label="Pointing error")
ax.set_ylabel("Pointing error [mrad]")
ax.set_title("Attitude pointing error (truth vs desired quaternion)")
shade_phases(ax, t_att, phase_att)
ax.legend(fontsize=8)

# (1) Wheel speeds (4 wheels as fraction of max)
ax = axes3[1]
for i in range(4):
    ax.plot(t_att, w_frac[:, i] * 100.0,
            color=WHEEL_COLORS[i], lw=0.7, alpha=0.85, label=f"W{i+1}")
ax.axhline(80.0, color="white", ls="--", lw=0.7, alpha=0.5, label="Desat 80%")
ax.set_ylabel("Speed [% max]")
ax.set_title("Reaction wheel speeds")
ax.legend(fontsize=8, ncol=5)
shade_phases(ax, t_att, phase_att)

# (2) Pointing mode indicator (Nadir=0, VelocityAligned=1)
ax = axes3[2]
mode_signal = (pointing_mode_att == "VelocityAligned").astype(float)
ax.fill_between(t_att, mode_signal, alpha=0.75, color="#ffe066", step="post")
ax.set_yticks([0, 1])
ax.set_yticklabels(["Nadir", "VelocityAligned"], fontsize=8)
ax.set_ylabel("Mode", fontsize=8)
ax.set_title("Pointing mode", fontsize=9)
ax.tick_params(axis="y", labelsize=7)

# (3) Torque budget: commanded PD vs SRP vs gravity gradient
ax = axes3[3]
if has_torques:
    tau_rw = np.sqrt(att["rw_tau_x_nm"]**2 + att["rw_tau_y_nm"]**2 + att["rw_tau_z_nm"]**2)
    ax.semilogy(t_att, tau_pd_norm  + 1e-12, color="#ce93d8", lw=0.9, label="|τ_PD| commanded")
    ax.semilogy(t_att, tau_srp_norm + 1e-12, color="#ff8f00", lw=0.9, label="|τ_SRP| perturbation", alpha=0.9)
    ax.semilogy(t_att, tau_gg_norm  + 1e-12, color="#4db6ac", lw=0.9, label="|τ_GG| perturbation",  alpha=0.9)
    ax.semilogy(t_att, tau_rw       + 1e-12, color="#7e57c2", lw=0.6, ls="--", label="|τ_rw| actual", alpha=0.7)
else:
    tau_rw = np.sqrt(att["rw_tau_x_nm"]**2 + att["rw_tau_y_nm"]**2 + att["rw_tau_z_nm"]**2)
    ax.semilogy(t_att, tau_rw + 1e-12, color="#ce93d8", lw=0.8, label="|τ_rw|")
ax.set_ylabel("|τ| [N·m]")
ax.set_title("Torque budget: PD controller vs perturbations (SRP + gravity gradient)")
ax.legend(fontsize=8, ncol=2)
shade_phases(ax, t_att, phase_att)

# (4) Wheel angular momentum + desat events
ax = axes3[4]
ax.plot(t_att, att["h_w_norm_nms"] * 1e3, color="#80cbc4", lw=0.8, label="|H_w|")
desat_t = t_att[desat_mask]
if len(desat_t) > 0:
    ax.scatter(desat_t, np.zeros(len(desat_t)),
               s=5, color="#ff7043", alpha=0.6, label="Desat event", zorder=4)
ax.set_ylabel("|H_w| [mN·m·s]")
ax.set_title("Total wheel angular momentum + desaturation events")
ax.legend(fontsize=8)
shade_phases(ax, t_att, phase_att)
set_time_label(ax)

plt.tight_layout()
fig3.savefig(OUT / "mission_attitude.png", dpi=150, bbox_inches="tight")

# ── Figure 4: Maneuvers & DSN ─────────────────────────────────────────────────

print("Plotting: mission_dv.png ...")
fig4, axes4 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig4.suptitle("Proximity Mission — Maneuvers & DSN Tracking", fontsize=13)

# (0) Per-maneuver ΔV
ax = axes4[0]
if len(man) > 0:
    burn1 = label_man == "Burn1"
    burn2 = label_man == "Burn2"
    sk    = label_man == "SK"
    for mask_m, col, lbl in [(burn1,"#ef5350","Burn1"),
                              (burn2,"#ffa726","Burn2"),
                              (sk,   "#66bb6a","SK")]:
        if mask_m.any():
            ax.stem(t_man[mask_m], man["dv_mag_ms"][mask_m] * 1e3,
                    linefmt=col, markerfmt=f"D", basefmt=" ", label=lbl)
ax.set_ylabel("ΔV [mm/s]")
ax.set_title("Maneuver ΔV events")
ax.legend(fontsize=8)
shade_phases(ax, t_nav, phase_nav)

# (1) Cumulative ΔV
ax = axes4[1]
if len(dv_cum) > 0:
    ax.step(t_man, dv_cum, color="#4fc3f7", lw=1.4, where="post", label="Cumulative ΔV")
ax.set_ylabel("Cumulative ΔV [m/s]")
ax.set_title(f"Total ΔV budget  ({dv_cum[-1]:.3f} m/s)" if len(dv_cum) > 0 else "ΔV budget")
ax.legend(fontsize=8)
shade_phases(ax, t_nav, phase_nav)

# (2) DSN innovation + sigma
ax = axes4[2]
if len(dsn) > 0:
    ax.semilogy(t_dsn, dsn["innov_r_m"] + 1.0, color="#ef5350",
                marker="o", ms=4, ls="--", lw=0.8, label="Innovation |δr|")
    ax.semilogy(t_dsn, dsn["sigma_r_m"], color="#ffa726",
                marker="s", ms=3, lw=0.8, label="EKF σ_r (post-update)")
ax.set_ylabel("Position [m]")
ax.set_title("DSN ground OD uplinks — innovation vs EKF 1σ")
ax.legend(fontsize=8)
shade_phases(ax, t_nav, phase_nav)
set_time_label(ax)

plt.tight_layout()
fig4.savefig(OUT / "mission_dv.png", dpi=150, bbox_inches="tight")

# ── Summary printout ──────────────────────────────────────────────────────────

print()
print("=" * 54)
print("  MISSION SUMMARY")
print("=" * 54)
print(f"  Duration          : {t_nav[-1]:.1f} h  ({t_nav[-1]/24:.1f} days)")
print(f"  Final range       : {nav['range_km'][-1]:.2f} km")
print(f"  Final nav error   : {dr[-1]:.1f} m")
print(f"  Final EKF sigma   : {sigma_r[-1]:.1f} m")
total_dv = dv_cum[-1] if len(dv_cum) > 0 else 0.0
print(f"  Total DeltaV      : {total_dv:.3f} m/s  ({len(man)} burns)")
print(f"  Max pointing err  : {att['pointing_err_mrad'].max():.2f} mrad")
print(f"  Max wheel sat     : {w_frac.max()*100:.1f}%")
print(f"  Desat firings     : {desat_mask.sum()}")
print(f"  DSN passes        : {len(dsn)}")
print("=" * 54)
print()
print("Saved:")
for f in ["mission_orbit.png", "mission_nav.png",
          "mission_attitude.png", "mission_dv.png"]:
    print(f"  {OUT / f}")

plt.show()
