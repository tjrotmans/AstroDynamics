"""
Cruise operations visualisation.
Three tracks: truth, ground OD, on-board EKF.
Run: python plot/plot_cruise_ops.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

AU = 1.495_978_707e11  # m

# -- Load data ----------------------------------------------------------------
truth  = pd.read_csv("out/cruise_ops/truth.csv")
od     = pd.read_csv("out/cruise_ops/ground_od.csv")
ekf    = pd.read_csv("out/cruise_ops/onboard_ekf.csv")
tcm    = pd.read_csv("out/cruise_ops/tcm_log.csv")
bp_h   = pd.read_csv("out/cruise_ops/bplane_history.csv")
bodies = pd.read_csv("out/cruise_ops/body_tracks.csv")

def au(col): return col / AU

days_tr = truth["time_s"] / 86400.0

# Position errors vs truth
err_od  = np.sqrt((truth.x_m-od.x_m)**2  + (truth.y_m-od.y_m)**2  + (truth.z_m-od.z_m)**2)  / 1e3
err_ekf = np.sqrt((truth.x_m-ekf.x_m)**2 + (truth.y_m-ekf.y_m)**2 + (truth.z_m-ekf.z_m)**2) / 1e3
err_od_ekf = np.sqrt((od.x_m-ekf.x_m)**2 + (od.y_m-ekf.y_m)**2 + (od.z_m-ekf.z_m)**2) / 1e3

dv_cum = np.cumsum(tcm["dv_mag_ms"].values) if len(tcm) else np.array([])

# ============================================================================
# Figure 1: 3D heliocentric trajectory
# ============================================================================
fig = plt.figure(figsize=(12, 9))
ax  = fig.add_subplot(111, projection="3d")

bx=au(bodies.bennu_x); by=au(bodies.bennu_y); bz=au(bodies.bennu_z)
ex=au(bodies.earth_x); ey=au(bodies.earth_y)

ax.plot(bx, by, bz,          color="tan",         lw=1.2, alpha=0.5, label="Bennu track")
ax.plot(ex, ey, np.zeros(len(ex)), color="deepskyblue", lw=1.2, alpha=0.5, label="Earth track")
ax.plot(au(truth.x_m), au(truth.y_m), au(truth.z_m),
        color="lime",    lw=2.0, label="Truth")
ax.plot(au(od.x_m),    au(od.y_m),    au(od.z_m),
        color="dodgerblue", lw=1.2, ls="--", label="Ground OD")
ax.plot(au(ekf.x_m),   au(ekf.y_m),   au(ekf.z_m),
        color="orange",  lw=1.0, ls=":", label="On-board EKF")

ax.scatter([0],[0],[0], s=150, color="gold", zorder=10, label="Sun")
ax.scatter([au(truth.x_m.iloc[0])],[au(truth.y_m.iloc[0])],[au(truth.z_m.iloc[0])],
           s=60, color="deepskyblue", marker="^", zorder=9, label="Departure")
ax.scatter([bx.iloc[-1]],[by.iloc[-1]],[bz.iloc[-1]],
           s=80, color="tan", marker="*", zorder=9, label="Bennu arrival")

arrow_scale = 4e9 / AU
if len(tcm):
    for _, row in tcm.iterrows():
        mag = row["dv_mag_ms"]
        if mag < 1e-9: continue
        x0,y0,z0 = row["x_m"]/AU, row["y_m"]/AU, row["z_m"]/AU
        dx,dy,dz  = row["dvx_exec"]/mag*arrow_scale, row["dvy_exec"]/mag*arrow_scale, row["dvz_exec"]/mag*arrow_scale
        ax.quiver(x0,y0,z0, dx,dy,dz, color="red", linewidth=2, arrow_length_ratio=0.3)
    ax.plot([],[],[], color="red", lw=2, label=f"TCMs (N={len(tcm)})")

ax.set_xlabel("X [AU]"); ax.set_ylabel("Y [AU]"); ax.set_zlabel("Z [AU]")
ax.set_title("Earth → Bennu cruise: truth / ground OD / on-board EKF / TCMs")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()

# ============================================================================
# Figure 2: Diagnostics (2x3 grid)
# ============================================================================
fig2, axes = plt.subplots(2, 3, figsize=(17, 8))
fig2.suptitle("Cruise operations diagnostics", fontsize=13)

# (0,0) Position errors
ax0 = axes[0,0]
ax0.semilogy(days_tr, err_od.clip(1e-2),    color="dodgerblue", lw=1.5, label="OD vs truth")
ax0.semilogy(days_tr, err_ekf.clip(1e-2),   color="orange",     lw=1.2, ls="--", label="EKF vs truth")
# ax0.semilogy(days_tr, err_od_ekf.clip(1e-2),color="purple",     lw=1.0, ls=":", label="EKF vs OD")
for _, row in tcm.iterrows(): ax0.axvline(row["day"], color="red", lw=0.7, alpha=0.5)
ax0.set_xlabel("Mission day"); ax0.set_ylabel("Position error [km]")
ax0.set_title("Position errors"); ax0.legend(fontsize=8); ax0.grid(True, which="both", alpha=0.3)

# (0,1) B-plane miss history
ax1 = axes[0,1]
ax1.semilogy(bp_h["day"], bp_h["miss_km"].clip(1e-1), color="coral", lw=1.5)
ax1.axhline(500, color="orange", ls="--", lw=1.2, label="TCM threshold 500 km")
if len(tcm):
    ax1.scatter(tcm["day"], tcm["miss_before_km"], s=60, color="red",
                zorder=5, label="TCM trigger miss")
    ax1.scatter(tcm["day"], tcm["miss_after_km"],  s=60, color="lime",
                zorder=5, marker="v", label="Predicted miss after TCM")
ax1.set_xlabel("Mission day"); ax1.set_ylabel("B-plane miss [km]")
ax1.set_title("B-plane miss (from OD estimate)"); ax1.legend(fontsize=8)
ax1.grid(True, which="both", alpha=0.3)

# (0,2) C_R convergence
ax2 = axes[0,2]
ax2.plot(days_tr, od.cr,  color="dodgerblue", lw=1.5, label="Ground OD C_R estimate")
ax2.plot(days_tr, ekf.cr, color="orange",     lw=1.2, ls="--", label="On-board EKF C_R estimate")
ax2.axhline(1.35, color="lime", ls="--", lw=1.5, label="True C_R = 1.35")
ax2.axhline(1.20, color="gray", ls=":", lw=1.0, label="Initial guess = 1.20")
ax2.set_xlabel("Mission day"); ax2.set_ylabel("C_R estimate")
ax2.set_title("SRP reflectivity convergence"); ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3); ax2.set_ylim(0.8, 2.0)

# (1,0) TCM bar chart
ax3 = axes[1,0]
if len(tcm):
    ax3.bar(tcm["day"], tcm["dv_mag_ms"], width=3, color="tomato",
            edgecolor="darkred", alpha=0.8)
    ax3.set_title(f"TCM magnitudes  (total = {dv_cum[-1]:.4f} m/s)")
else:
    ax3.text(0.5, 0.5, "No TCMs", ha="center", va="center",
             transform=ax3.transAxes, fontsize=12)
    ax3.set_title("TCM magnitudes")
ax3.set_xlabel("TCM day"); ax3.set_ylabel("|ΔV| [m/s]"); ax3.grid(True, alpha=0.3)

# (1,1) Cumulative ΔV
ax4 = axes[1,1]
if len(tcm):
    days_pad = np.concatenate([[0], tcm["day"].values, [days_tr.iloc[-1]]])
    dv_pad   = np.concatenate([[0], dv_cum, [dv_cum[-1]]])
    ax4.step(days_pad, dv_pad, where="post", color="crimson", lw=2)
    ax4.fill_between(days_pad, dv_pad, step="post", alpha=0.2, color="crimson")
else:
    ax4.text(0.5, 0.5, "No TCMs", ha="center", va="center",
             transform=ax4.transAxes, fontsize=12)
ax4.set_xlabel("Mission day"); ax4.set_ylabel("Cumulative ΔV [m/s]")
ax4.set_title("TCM ΔV budget"); ax4.grid(True, alpha=0.3)

# (1,2) EKF vs OD position diff zoomed
ax5 = axes[1,2]
ax5.plot(days_tr, err_od_ekf, color="purple", lw=1.2)
for _, row in tcm.iterrows(): ax5.axvline(row["day"], color="red", lw=0.7, alpha=0.5)
# Mark daily uplink times
uplink_days = np.arange(1, int(days_tr.iloc[-1]))
ax5.set_xlabel("Mission day"); ax5.set_ylabel("OD vs EKF position diff [km]")
ax5.set_title("On-board EKF drift between uplinks"); ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
