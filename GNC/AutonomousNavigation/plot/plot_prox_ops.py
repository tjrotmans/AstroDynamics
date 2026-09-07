"""
Bennu proximity operations -- results visualisation.

Reads outputs from `cargo run --bin proximity_ops`:
  out/prox_ops/truth.csv        -- ground-truth spacecraft state
  out/prox_ops/ekf_est.csv      -- EKF mean + 1-sigma covariance
  out/prox_ops/maneuvers.csv    -- manoeuvre log
  out/prox_ops/dsn_updates.csv  -- DSN ground tracking pass log

Produces:
  out/prox_ops/prox_orbit_3d.png      -- 3-D Hill-frame trajectory
  out/prox_ops/prox_range_phase.png   -- range + guidance phase timeline
  out/prox_ops/prox_nav_errors.png    -- nav accuracy vs EKF 3-sigma bounds
  out/prox_ops/prox_maneuvers.png     -- ΔV timeline + cumulative budget
  out/prox_ops/prox_dsn.png           -- DSN innovation + covariance
  out/prox_ops/prox_orbital_energy.png -- orbital energy convergence

Usage:  python plot/plot_prox_ops.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.lines import Line2D
from pathlib import Path

# ── File paths ────────────────────────────────────────────────────────────────

ROOT  = Path(__file__).parent.parent
DATA  = ROOT / "out" / "prox_ops"
OUT   = DATA

# ── Constants (must match config.rs) ─────────────────────────────────────────

MU_BENNU         = 6.674e-11 * 7.329e10   # m³/s²
R_BENNU_M        = 262.0                   # m
TARGET_ORBIT_R_M = 1_500.0                 # m

PHASE_COLORS = {
    "OrbInsertion": "#f06292",   # pink — large capture burn from hyperbolic approach
    "InitCoast":    "#4fc3f7",   # light blue
    "Burn1":        "#ef5350",   # red
    "TransferCoast":"#ab47bc",   # purple
    "Burn2":        "#ff7043",   # orange-red
    "OrbitHold":    "#66bb6a",   # green
}

# ── Load data ─────────────────────────────────────────────────────────────────

def load(name):
    p = DATA / name
    if not p.exists():
        raise FileNotFoundError(f"Run 'cargo run --bin proximity_ops' first.\nMissing: {p}")
    arr = np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8")
    # genfromtxt returns a 0-d array when there is exactly one data row; always make 1-d.
    return np.atleast_1d(arr)

truth = load("truth.csv")
est   = load("ekf_est.csv")
man   = load("maneuvers.csv")
dsn   = load("dsn_updates.csv")

t_h   = truth["time_s"] / 3600.0     # hours
t_d   = t_h / 24.0                   # days
t_man = man["time_s"]  / 3600.0
t_dsn = dsn["time_s"]  / 3600.0

# Numpy structured arrays store str as bytes on older numpy versions
def col_str(arr, field):
    v = arr[field]
    return np.array([s.decode() if isinstance(s, bytes) else s for s in v])

phase_truth = col_str(truth, "phase")
phase_man   = col_str(man,   "phase")

# Position error
dr = np.sqrt((truth["x_m"]  - est["x_m"])**2 +
             (truth["y_m"]  - est["y_m"])**2 +
             (truth["z_m"]  - est["z_m"])**2)
dv = np.sqrt((truth["vx_ms"] - est["vx_ms"])**2 +
             (truth["vy_ms"] - est["vy_ms"])**2 +
             (truth["vz_ms"] - est["vz_ms"])**2)

sig_r = np.where(np.isnan(est["sigma_r_m"]), 0.0, est["sigma_r_m"])
sig_v = np.where(np.isnan(est["sigma_v_mps"]), 0.0, est["sigma_v_mps"])

# Orbital energy from truth state
eps_truth  = 0.5*(truth["vx_ms"]**2 + truth["vy_ms"]**2 + truth["vz_ms"]**2) \
             - MU_BENNU / (truth["range_km"] * 1e3)
eps_target = -MU_BENNU / (2.0 * TARGET_ORBIT_R_M)

# Cumulative ΔV
dv_cum = np.cumsum(man["dv_mag_ms"])

# ── Helper: shade phases as background ───────────────────────────────────────

def shade_phases(ax, t_arr, phases):
    """Shade background by guidance phase. t_arr in hours."""
    if len(t_arr) < 2:
        return
    prev_ph = phases[0]
    t_start = t_arr[0]
    for i in range(1, len(t_arr)):
        if phases[i] != prev_ph or i == len(t_arr) - 1:
            ax.axvspan(t_start, t_arr[i], alpha=0.08,
                       color=PHASE_COLORS.get(prev_ph, "gray"), zorder=0)
            t_start = t_arr[i]
            prev_ph = phases[i]

# ── Unique phase legend patches ───────────────────────────────────────────────

def phase_legend_patches():
    return [mpatches.Patch(color=c, alpha=0.5, label=ph)
            for ph, c in PHASE_COLORS.items()
            if ph in phase_truth]

# =============================================================================
# FIGURE 1 — 3-D Hill-frame orbital trajectory
# =============================================================================

fig1 = plt.figure(figsize=(13, 10))
ax3d = fig1.add_subplot(111, projection="3d")
ax3d.set_title("Bennu Proximity Operations — Hill-Frame Trajectory\n"
               "Colour = elapsed time [h]", fontsize=11)

tx = truth["x_m"] / 1e3
ty = truth["y_m"] / 1e3
tz = truth["z_m"] / 1e3

norm_t = Normalize(vmin=t_h.min(), vmax=t_h.max())
cmap   = plt.cm.viridis

for i in range(len(t_h) - 1):
    c = cmap(norm_t(t_h[i]))
    ax3d.plot(tx[i:i+2], ty[i:i+2], tz[i:i+2], color=c, lw=1.5)

# EKF estimate
ax3d.plot(est["x_m"]/1e3, est["y_m"]/1e3, est["z_m"]/1e3,
          "r--", lw=0.7, alpha=0.5, label="EKF estimate")

# Bennu sphere
u_b, v_b = np.mgrid[0:2*np.pi:24j, 0:np.pi:12j]
br = R_BENNU_M / 1e3
bx = br * np.cos(u_b) * np.sin(v_b)
by = br * np.sin(u_b) * np.sin(v_b)
bz = br * np.cos(v_b)
ax3d.plot_surface(bx, by, bz, color="gold", alpha=0.75, zorder=1)

# Target orbit circle (1.5 km)
theta_circ = np.linspace(0, 2*np.pi, 200)
r_tgt_km   = TARGET_ORBIT_R_M / 1e3
ax3d.plot(r_tgt_km * np.cos(theta_circ),
          r_tgt_km * np.sin(theta_circ),
          np.zeros(200), "g--", lw=1.0, alpha=0.5, label=f"Target {r_tgt_km:.1f} km orbit")

# Manoeuvre markers
for ph, sym, col in [("Burn1","*","red"), ("Burn2","*","darkorange")]:
    mask = phase_man == ph
    if mask.any():
        # find closest truth time
        for i_m in np.where(mask)[0]:
            t_b = man["time_s"][i_m] / 3600.0
            idx = np.argmin(np.abs(t_h - t_b))
            ax3d.scatter([tx[idx]], [ty[idx]], [tz[idx]],
                         s=120, marker=sym, c=col, zorder=10,
                         label=f"{ph} @ {t_b:.2f} h" if i_m == np.where(mask)[0][0] else "")

ax3d.scatter([tx[0]],  [ty[0]],  [tz[0]],  s=80, c="lime",  marker="^", zorder=10, label="Start")
ax3d.scatter([tx[-1]], [ty[-1]], [tz[-1]], s=80, c="red",   marker="v", zorder=10, label="End")

sm1 = ScalarMappable(cmap=cmap, norm=norm_t)
sm1.set_array([])
cb1 = fig1.colorbar(sm1, ax=ax3d, shrink=0.55, pad=0.08, aspect=20)
cb1.set_label("Time [h]", fontsize=9)

ax3d.set_xlabel("x [km]"); ax3d.set_ylabel("y [km]"); ax3d.set_zlabel("z [km]")
ax3d.legend(loc="upper left", fontsize=7)
plt.tight_layout()
p = OUT / "prox_orbit_3d.png"
plt.savefig(p, dpi=150); print(f"Saved {p}")
plt.show(block=False)

# =============================================================================
# FIGURE 2 — Range profile + guidance phase
# =============================================================================

fig2, axes2 = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig2.suptitle("Bennu Proximity Operations — Range Profile and Guidance Phase", fontsize=12)

ax = axes2[0]
shade_phases(ax, t_h, phase_truth)
ax.plot(t_h, truth["range_km"],  color="teal",    lw=1.5, label="Truth range")
ax.plot(t_h, est["range_km"],    color="crimson", lw=1.0, ls="--", alpha=0.85,
        label="EKF range")
ax.fill_between(t_h,
                est["range_km"] - 3*sig_r/1e3,
                est["range_km"] + 3*sig_r/1e3,
                alpha=0.18, color="crimson", label="EKF 3-sigma")
ax.axhline(TARGET_ORBIT_R_M / 1e3, color="green", ls=":", lw=1.2,
           label=f"Target {TARGET_ORBIT_R_M/1e3:.1f} km")
ax.set_ylabel("Range from Bennu [km]")
ax.set_title("Range to Bennu — truth vs EKF estimate")
ax.legend(fontsize=8, loc="upper right")
ax.legend(handles=ax.get_legend_handles_labels()[0] + phase_legend_patches(),
          labels=ax.get_legend_handles_labels()[1]  + [p.get_label() for p in phase_legend_patches()],
          fontsize=7, loc="upper right", ncol=2)
ax.grid(True, alpha=0.3)

# Vertical lines at manoeuvre times
for mrow in man:
    ph = mrow["phase"].decode() if isinstance(mrow["phase"], bytes) else mrow["phase"]
    t_b = mrow["time_s"] / 3600.0
    col = PHASE_COLORS.get(ph, "black")
    ax.axvline(t_b, color=col, lw=1.2, alpha=0.8, ls="-")

ax = axes2[1]
shade_phases(ax, t_h, phase_truth)
# Guidance phase as a discrete colour map
phases_uniq = list(PHASE_COLORS.keys())
phase_idx   = np.array([phases_uniq.index(p) if p in phases_uniq else 0
                        for p in phase_truth], dtype=float)
ax.scatter(t_h, phase_idx, c=[PHASE_COLORS.get(p, "gray") for p in phase_truth],
           s=4, zorder=5)
ax.set_yticks(range(len(phases_uniq)))
ax.set_yticklabels(phases_uniq, fontsize=8)
ax.set_xlabel("Time [h]")
ax.set_title("Guidance Phase")
ax.grid(True, alpha=0.2, axis="x")

plt.tight_layout()
p = OUT / "prox_range_phase.png"
plt.savefig(p, dpi=150); print(f"Saved {p}")
plt.show(block=False)

# =============================================================================
# FIGURE 3 — Navigation errors vs EKF 3-sigma bounds
# =============================================================================

fig3, axes3 = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
fig3.suptitle("Navigation Accuracy — EKF Consistency\n"
              "(errors must stay below the 3-sigma band for a consistent filter)", fontsize=11)

ax = axes3[0]
shade_phases(ax, t_h, phase_truth)
ax.semilogy(t_h, dr,       color="steelblue", lw=1.5, label="|r_truth − r_EKF|")
ax.semilogy(t_h, 3*sig_r,  color="steelblue", lw=1.0, ls="--", alpha=0.7,
            label="EKF 3-sigma")
ax.fill_between(t_h, 1e-2, 3*sig_r, alpha=0.12, color="steelblue")
for mrow in man:
    t_b = mrow["time_s"] / 3600.0
    ph  = mrow["phase"].decode() if isinstance(mrow["phase"], bytes) else mrow["phase"]
    ax.axvline(t_b, color=PHASE_COLORS.get(ph, "black"), lw=1.1, alpha=0.7, ls="-")
for t_d_dsn in t_dsn:
    ax.axvline(t_d_dsn, color="magenta", lw=0.8, alpha=0.5, ls=":")
ax.set_ylabel("Position error [m]")
ax.set_title("Position error |Δr|")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes3[1]
shade_phases(ax, t_h, phase_truth)
ax.semilogy(t_h, dv,       color="darkorange", lw=1.5, label="|v_truth − v_EKF|")
ax.semilogy(t_h, 3*sig_v,  color="darkorange", lw=1.0, ls="--", alpha=0.7,
            label="EKF 3-sigma")
ax.fill_between(t_h, 1e-6, 3*sig_v, alpha=0.12, color="darkorange")
for mrow in man:
    t_b = mrow["time_s"] / 3600.0
    ph  = mrow["phase"].decode() if isinstance(mrow["phase"], bytes) else mrow["phase"]
    ax.axvline(t_b, color=PHASE_COLORS.get(ph, "black"), lw=1.1, alpha=0.7, ls="-")
for t_d_dsn in t_dsn:
    ax.axvline(t_d_dsn, color="magenta", lw=0.8, alpha=0.5, ls=":",
               label="DSN pass" if t_d_dsn == t_dsn[0] else "")
ax.set_xlabel("Time [h]"); ax.set_ylabel("Velocity error [m/s]")
ax.set_title("Velocity error |Δv|  — vertical magenta = DSN pass, coloured = burn")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
p = OUT / "prox_nav_errors.png"
plt.savefig(p, dpi=150); print(f"Saved {p}")
plt.show(block=False)

# =============================================================================
# FIGURE 4 — Manoeuvre timeline + ΔV budget
# =============================================================================

fig4, axes4 = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig4.suptitle("Manoeuvre Timeline and ΔV Budget", fontsize=12)

ax = axes4[0]
shade_phases(ax, t_h, phase_truth)
colors_m = [PHASE_COLORS.get(
    (row["phase"].decode() if isinstance(row["phase"], bytes) else row["phase"]),
    "gray") for row in man]
ax.bar(t_man, man["dv_mag_ms"] * 1e3, width=0.1, color=colors_m, zorder=5,
       label="ΔV per manoeuvre")
ax.set_ylabel("ΔV [mm/s]")
ax.set_title("Manoeuvre ΔV  (red = Burn 1, orange = Burn 2, green = station-keeping)")
ax.legend(handles=phase_legend_patches(), fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3, axis="y")

ax = axes4[1]
shade_phases(ax, t_h, phase_truth)
if len(dv_cum) > 0:
    ax.plot(t_man, dv_cum * 1e3, "k-o", ms=5, lw=1.5)
    ax.axhline(dv_cum[-1] * 1e3, color="gray", ls="--", lw=0.8,
               label=f"Total = {dv_cum[-1]*1e3:.2f} mm/s")
    ax.legend(fontsize=9)
ax.set_xlabel("Time [h]"); ax.set_ylabel("Cumulative ΔV [mm/s]")
ax.set_title("Cumulative ΔV budget")
ax.grid(True, alpha=0.3)

plt.tight_layout()
p = OUT / "prox_maneuvers.png"
plt.savefig(p, dpi=150); print(f"Saved {p}")
plt.show(block=False)

# =============================================================================
# FIGURE 5 — DSN pass quality
# =============================================================================

fig5, axes5 = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
fig5.suptitle("DSN Ground Tracking Passes", fontsize=12)

ax = axes5[0]
ax.bar(t_dsn, dsn["innov_r_m"], width=0.2, color="royalblue", alpha=0.8,
       label="Position innovation |Δr|")
ax.set_ylabel("Innovation magnitude [m]")
ax.set_title("DSN position innovation magnitude\n"
             "(dominated by Bennu ephemeris uncertainty ~5 km; should be ~O(5000 m))")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

ax = axes5[1]
ax.semilogy(t_dsn, dsn["sigma_r_m"], "s-", color="darkorange", ms=6, lw=1.3,
            label="EKF σ_r at DSN epoch")
ax.axhline(5000, color="gray", ls="--", lw=0.8, alpha=0.7,
           label="Bennu ephem. floor (~5000 m)")
ax.set_xlabel("Time [h]"); ax.set_ylabel("σ_r [m]")
ax.set_title("EKF position uncertainty at each DSN pass  "
             "(drives down until limited by Bennu ephemeris error)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
p = OUT / "prox_dsn.png"
plt.savefig(p, dpi=150); print(f"Saved {p}")
plt.show(block=False)

# =============================================================================
# FIGURE 6 — Orbital energy convergence
# =============================================================================

fig6, axes6 = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig6.suptitle("Orbital Energy — Hohmann Transfer and Station-Keeping", fontsize=12)

ax = axes6[0]
shade_phases(ax, t_h, phase_truth)
ax.plot(t_h, eps_truth,           color="steelblue", lw=1.5, label="Orbital energy ε [J/kg]")
ax.axhline(eps_target, color="green", ls="--", lw=1.3,
           label=f"Target ε = {eps_target:.4e} J/kg  ({TARGET_ORBIT_R_M/1e3:.1f} km circular)")
# Annotate burns
for mrow in man:
    ph  = mrow["phase"].decode() if isinstance(mrow["phase"], bytes) else mrow["phase"]
    t_b = mrow["time_s"] / 3600.0
    ax.axvline(t_b, color=PHASE_COLORS.get(ph, "k"), lw=1.2, alpha=0.8)
ax.set_ylabel("Specific energy [J/kg]")
ax.set_title("Orbital energy — converges to target value after Burn 2")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.legend(handles=ax.get_legend_handles_labels()[0] + phase_legend_patches(),
          labels=ax.get_legend_handles_labels()[1]  + [p.get_label() for p in phase_legend_patches()],
          fontsize=7, loc="upper right", ncol=2)

# Station-keeping energy deviation (last 5 days)
in_hold = phase_truth == "OrbitHold"
if in_hold.any():
    t_hold = t_h[in_hold]
    de_hold = eps_truth[in_hold] - eps_target
    ax = axes6[1]
    ax.plot(t_hold, de_hold, color="mediumpurple", lw=1.2, label="|ε − ε_target|")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.set_xlabel("Time [h]"); ax.set_ylabel("ΔE [J/kg]")
    ax.set_title("Station-keeping: energy deviation from target  (corrected each hour)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
else:
    axes6[1].text(0.5, 0.5, "No OrbitHold phase data yet",
                  ha="center", va="center", transform=axes6[1].transAxes)

plt.tight_layout()
p = OUT / "prox_orbital_energy.png"
plt.savefig(p, dpi=150); print(f"Saved {p}")
plt.show(block=False)

# ── Summary stats ─────────────────────────────────────────────────────────────

in_hold_f = phase_truth == "OrbitHold"
if in_hold_f.any():
    dr_hold = dr[in_hold_f]
    print(f"\n── Station-keeping nav accuracy ──")
    print(f"  Mean position error : {dr_hold.mean():.1f} m")
    print(f"  Max  position error : {dr_hold.max():.1f} m")
    print(f"  Mean σ_r            : {sig_r[in_hold_f].mean():.1f} m")

if len(dv_cum) > 0:
    print(f"\n── ΔV budget ──")
    for ph in ["Burn1", "Burn2", "SK"]:
        mask_p = np.array([
            (row["phase"].decode() if isinstance(row["phase"], bytes) else row["phase"]) == ph
            for row in man
        ])
        total = man["dv_mag_ms"][mask_p].sum() * 1e3
        if mask_p.any():
            print(f"  {ph:<16}: {total:.3f} mm/s  ({mask_p.sum()} burns)")
    print(f"  {'Total':<16}: {dv_cum[-1]*1e3:.3f} mm/s")

plt.show()
