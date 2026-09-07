"""
Autonomous Navigation -- proximity phase results.

Produces:
  out/nav_3d_hill.png          -- 3-D Hill-frame trajectory with uncertainty ellipsoids
  out/nav_error_sigma.png      -- position / velocity error vs 3-sigma bands + C_SRP
  out/nav_residuals.png        -- normalized measurement residuals (bearing + angular size)
  out/nav_range_profile.png    -- range to Bennu: truth vs EKF estimate +/- 3-sigma
  out/nav_attitude.png         -- angular rates in body frame
  out/nav_rcs.png              -- RCS torque and delta-V budget
  out/nav_pointing.png         -- camera boresight pointing error vs dead-band

Usage:  python plot/plot_nav.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                      # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path

# -- Sensor noise constants (must match config.rs) --------------------------------

OPNAV_BEARING_SIGMA_RAD = 1e-4   # 1-sigma bearing noise [rad]
OPNAV_SIZE_SIGMA_RAD    = 2e-4   # 1-sigma angular-size noise [rad]

# -- Load CSV outputs -------------------------------------------------------------

root  = Path(__file__).parent.parent
NAV   = root / "out"
truth = np.genfromtxt(NAV / "truth.csv",           delimiter=",", names=True)
est   = np.genfromtxt(NAV / "ekf_est.csv",         delimiter=",", names=True)
resid = np.genfromtxt(NAV / "opnav_residuals.csv", delimiter=",", names=True)

t_h  = truth["time_s"] / 3600.0   # hours
t_rs = resid["time_s"] / 3600.0

# Position / velocity errors (truth - estimate)
dr = np.sqrt((truth["x_m"]  - est["x_m"] )**2 +
             (truth["y_m"]  - est["y_m"] )**2 +
             (truth["z_m"]  - est["z_m"] )**2)
dv = np.sqrt((truth["vx_ms"] - est["vx_ms"])**2 +
             (truth["vy_ms"] - est["vy_ms"])**2 +
             (truth["vz_ms"] - est["vz_ms"])**2)

# EKF 1-sigma from covariance diagonal (log written by main.rs)
sig_r  = est["sigma_r_m"]    # sqrt(P_xx + P_yy + P_zz)  [m]
sig_x  = est["sigma_x_m"]
sig_y  = est["sigma_y_m"]
sig_z  = est["sigma_z_m"]
sig_v  = est["sigma_v_mps"]  # RMS of diagonal velocity variances [m/s]
sig_cr = est["sigma_cr"]

# Range to Bennu [km]
range_truth_km = np.sqrt(truth["x_m"]**2 + truth["y_m"]**2 + truth["z_m"]**2) / 1e3
range_est_km   = np.sqrt(est["x_m"]**2   + est["y_m"]**2   + est["z_m"]**2  ) / 1e3

# -- Camera FOV parameters -------------------------------------------------------

CAMERA_FOV_HALF_DEG = 15.0
FOV_HALF = np.radians(CAMERA_FOV_HALF_DEG)
MAX_CONE_LEN_KM = 1.0

# ============================================================================
# FIGURE 1 -- 3-D Hill-frame trajectory with uncertainty ellipsoids
# ============================================================================

fig = plt.figure(figsize=(13, 10))
ax  = fig.add_subplot(111, projection="3d")
ax.set_title("Spacecraft Trajectory -- Bennu Hill Frame\n"
             "Ellipsoids: EKF 3-sigma position uncertainty at selected epochs",
             fontsize=11)

norm_t = Normalize(vmin=t_h.min(), vmax=t_h.max())
cmap   = plt.cm.plasma
tx = truth["x_m"] / 1e3
ty = truth["y_m"] / 1e3
tz = truth["z_m"] / 1e3

for i in range(len(t_h) - 1):
    c = cmap(norm_t(t_h[i]))
    ax.plot(tx[i:i+2], ty[i:i+2], tz[i:i+2], color=c, lw=1.5)

ax.plot(est["x_m"]/1e3, est["y_m"]/1e3, est["z_m"]/1e3,
        "r--", lw=0.8, alpha=0.6, label="EKF estimate")

# Bennu at origin
u_b, v_b = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
bennu_r_km = 0.262
bx = bennu_r_km * np.cos(u_b) * np.sin(v_b)
by = bennu_r_km * np.sin(u_b) * np.sin(v_b)
bz = bennu_r_km * np.cos(v_b)
ax.plot_surface(bx, by, bz, color="gold", alpha=0.7, zorder=1)
ax.scatter([0], [0], [0], s=5, c="goldenrod", zorder=10)

ax.scatter(tx[0],  ty[0],  tz[0],  s=80, c="lime", marker="^", zorder=10, label="Start")
ax.scatter(tx[-1], ty[-1], tz[-1], s=80, c="red",  marker="v", zorder=10, label="End")

# -- Uncertainty ellipsoids at ~6 equally-spaced epochs -----------------------

def draw_ellipsoid(ax, cx, cy, cz, sx, sy, sz, n=20, color="cyan", alpha=0.15):
    """Draw a wireframe ellipsoid centred at (cx,cy,cz) with semi-axes sx,sy,sz [km]."""
    u_e = np.linspace(0, 2*np.pi, n)
    v_e = np.linspace(0,   np.pi, n)
    xe = cx + sx * np.outer(np.cos(u_e), np.sin(v_e))
    ye = cy + sy * np.outer(np.sin(u_e), np.sin(v_e))
    ze = cz + sz * np.outer(np.ones_like(u_e), np.cos(v_e))
    ax.plot_wireframe(xe, ye, ze, color=color, alpha=alpha, lw=0.4, rstride=4, cstride=4)

N_ellipsoids = 6
ell_idx = np.linspace(0, len(t_h) - 1, N_ellipsoids, dtype=int)
for ii, idx in enumerate(ell_idx):
    draw_ellipsoid(
        ax,
        est["x_m"][idx]/1e3, est["y_m"][idx]/1e3, est["z_m"][idx]/1e3,
        3*sig_x[idx]/1e3, 3*sig_y[idx]/1e3, 3*sig_z[idx]/1e3,
        color="deepskyblue", alpha=0.18,
    )

# Colourbar (time)
sm = ScalarMappable(cmap=cmap, norm=norm_t)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, aspect=20)
cb.set_label("Time [h]", fontsize=9)

ax.set_xlabel("x [km]  (radial)")
ax.set_ylabel("y [km]  (along-track)")
ax.set_zlabel("z [km]  (cross-track)")
ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
out_3d = NAV / "nav_3d_hill.png"
plt.savefig(out_3d, dpi=150)
print(f"Saved {out_3d}")
plt.show(block=False)

# ============================================================================
# FIGURE 2 -- Error vs 3-sigma bands + C_SRP convergence
# ============================================================================

fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig2.suptitle("Navigation Performance -- EKF Consistency Check (errors vs 3-sigma bounds)",
              fontsize=12)

# Position error vs 3-sigma
ax = axes2[0]
ax.semilogy(t_h, dr,       color="steelblue",  lw=1.5, label="Position error |dr|")
ax.semilogy(t_h, 3*sig_r,  color="steelblue",  lw=1.0, ls="--", alpha=0.7,
            label="3-sigma bound")
ax.fill_between(t_h, 1e-3, 3*sig_r, alpha=0.12, color="steelblue")
ax.set_ylabel("Position error [m]")
ax.set_title("Position error vs 3-sigma  (error should stay below dashed line for a consistent EKF)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Velocity error vs 3-sigma
ax = axes2[1]
ax.semilogy(t_h, dv,       color="darkorange", lw=1.5, label="Velocity error |dv|")
ax.semilogy(t_h, 3*sig_v,  color="darkorange", lw=1.0, ls="--", alpha=0.7,
            label="3-sigma bound")
ax.fill_between(t_h, 1e-6, 3*sig_v, alpha=0.12, color="darkorange")
ax.set_ylabel("Velocity error [m/s]")
ax.set_title("Velocity error vs 3-sigma")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# C_SRP estimate +/- 1-sigma
ax = axes2[2]
cr_true = float(truth["c_r"][0])
ax.plot(t_h, est["c_r"],          color="mediumpurple", lw=1.5, label="EKF estimate")
ax.fill_between(t_h,
                est["c_r"] - sig_cr,
                est["c_r"] + sig_cr,
                alpha=0.25, color="mediumpurple", label="1-sigma band")
ax.axhline(cr_true, color="k", ls="--", lw=1, label=f"Truth = {cr_true:.3f}")
ax.set_xlabel("Time [h]"); ax.set_ylabel("C_SRP")
ax.set_title("Reflectivity coefficient -- EKF estimate +/- 1-sigma vs truth")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
out_err = NAV / "nav_error_sigma.png"
plt.savefig(out_err, dpi=150)
print(f"Saved {out_err}")
plt.show(block=False)

# ============================================================================
# FIGURE 3 -- Normalized measurement residuals
# ============================================================================

fig3, axes3 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
fig3.suptitle(
    "Measurement Residuals (pre-update: measured - predicted)\n"
    "Normalized by sensor 1-sigma -- should be ~N(0,1) if EKF is consistent",
    fontsize=11)

avail = resid["opnav_available"].astype(bool)
t_ok  = t_rs[avail]
t_no  = t_rs[~avail]

# Row 1: measurement availability
axes3[0].scatter(t_ok, np.ones_like(t_ok),  c="green", s=6, label="Measurement received")
axes3[0].scatter(t_no, np.zeros_like(t_no), c="red",   s=6, label="Missed (FOV / Sun)")
axes3[0].set_yticks([0, 1]); axes3[0].set_yticklabels(["Missed", "OK"])
axes3[0].set_title("OpNav Measurement Availability")
axes3[0].legend(fontsize=8); axes3[0].grid(True, alpha=0.3)

# Row 2: normalized bearing residuals
bear_norm = resid["bearing_residual_rad"][avail] / OPNAV_BEARING_SIGMA_RAD
axes3[1].plot(t_ok, bear_norm, "b.", ms=2, alpha=0.6, label="Normalized residual")
axes3[1].axhline( 1, color="k", ls="--", lw=0.8, alpha=0.6, label="+/- 1-sigma")
axes3[1].axhline(-1, color="k", ls="--", lw=0.8, alpha=0.6)
axes3[1].axhline( 3, color="r", ls=":",  lw=0.8, alpha=0.5, label="+/- 3-sigma")
axes3[1].axhline(-3, color="r", ls=":",  lw=0.8, alpha=0.5)
axes3[1].set_ylabel("Residual / sigma_bearing")
axes3[1].set_title(
    f"Normalized Bearing Residual  (sigma = {OPNAV_BEARING_SIGMA_RAD:.0e} rad)  "
    "-- includes phase-angle centroid bias (unmodelled in EKF)")
axes3[1].legend(fontsize=8); axes3[1].grid(True, alpha=0.3)

# Row 3: normalized angular-size residuals
size_norm = resid["size_residual_rad"][avail] / OPNAV_SIZE_SIGMA_RAD
axes3[2].plot(t_ok, size_norm, "r.", ms=2, alpha=0.6, label="Normalized residual")
axes3[2].axhline( 1, color="k", ls="--", lw=0.8, alpha=0.6, label="+/- 1-sigma")
axes3[2].axhline(-1, color="k", ls="--", lw=0.8, alpha=0.6)
axes3[2].axhline( 3, color="darkred", ls=":", lw=0.8, alpha=0.5, label="+/- 3-sigma")
axes3[2].axhline(-3, color="darkred", ls=":", lw=0.8, alpha=0.5)
axes3[2].set_ylabel("Residual / sigma_size")
axes3[2].set_title(
    f"Normalized Angular-Size Residual  (sigma = {OPNAV_SIZE_SIGMA_RAD:.0e} rad)  "
    "-- captures range estimation error")
axes3[2].legend(fontsize=8); axes3[2].grid(True, alpha=0.3)

# Row 4: phase angle (Sun-Bennu-spacecraft) at each measurement epoch
has_phase = "phase_angle_rad" in resid.dtype.names
if has_phase:
    phi_deg = np.degrees(resid["phase_angle_rad"][avail])
    axes3[3].plot(t_ok, phi_deg, ".", color="darkorange", ms=2, alpha=0.7,
                  label="Phase angle")
    axes3[3].axhline(90, color="gray", ls="--", lw=0.8, alpha=0.6,
                     label="90° (quarter phase)")
    axes3[3].set_ylabel("Phase angle [deg]")
    axes3[3].set_title(
        "Sun–Bennu–Spacecraft Phase Angle\n"
        "High phase → crescent illumination → centroid shifts toward sub-solar limb "
        "(systematic bearing bias not modelled in EKF)")
    axes3[3].set_ylim(0, 180)
    axes3[3].legend(fontsize=8)
else:
    axes3[3].text(0.5, 0.5, "phase_angle_rad not in residuals CSV",
                  ha="center", va="center", transform=axes3[3].transAxes)
axes3[3].set_xlabel("Time [h]")
axes3[3].grid(True, alpha=0.3)

plt.tight_layout()
out_res = NAV / "nav_residuals.png"
plt.savefig(out_res, dpi=150)
print(f"Saved {out_res}")
plt.show(block=False)

# ============================================================================
# FIGURE 4 -- Range profile: truth vs EKF +/- 3-sigma
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(12, 5))
fig4.suptitle("Range to Bennu -- Truth vs EKF Estimate", fontsize=12)

ax4.plot(t_h, range_truth_km,           color="teal",       lw=1.5, label="Truth range")
ax4.plot(t_h, range_est_km,             color="crimson",    lw=1.2, ls="--", alpha=0.85,
         label="EKF estimated range")
ax4.fill_between(t_h,
                 (range_est_km*1e3 - 3*sig_r)/1e3,
                 (range_est_km*1e3 + 3*sig_r)/1e3,
                 alpha=0.18, color="crimson", label="EKF 3-sigma band")
ax4.set_xlabel("Time [h]"); ax4.set_ylabel("Range [km]")
ax4.set_title("If the truth range stays inside the 3-sigma band, the EKF is consistent on range")
ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

plt.tight_layout()
out_rng = NAV / "nav_range_profile.png"
plt.savefig(out_rng, dpi=150)
print(f"Saved {out_rng}")
plt.show(block=False)

# ============================================================================
# FIGURE 5 -- Attitude dynamics: body angular rates
# ============================================================================

fig5, axes5 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig5.suptitle(
    "Attitude Dynamics -- Angular Rates in body frame\n"
    "(oscillation is expected: bang-bang thrusters cause a limit cycle around the nadir-pointing command)",
    fontsize=10)

omega_cols   = ["omega_x_rads", "omega_y_rads", "omega_z_rads"]
body_ax_lbls = [r"$x_B$  (camera boresight)", r"$y_B$", r"$z_B$"]
omega_colors = ["steelblue", "darkorange", "forestgreen"]

for i, (col, lbl, col_c) in enumerate(zip(omega_cols, body_ax_lbls, omega_colors)):
    axes5[i].plot(t_h, truth[col] * 1e3, color=col_c, lw=0.8)
    axes5[i].axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    axes5[i].set_ylabel("[mrad/s]")
    axes5[i].set_title(rf"$\omega$ about {lbl}  (body frame)")
    axes5[i].grid(True, alpha=0.3)

omega_mag = np.sqrt(sum(truth[c]**2 for c in omega_cols))
axes5[3].plot(t_h, omega_mag * 1e3, color="crimson", lw=0.8)
axes5[3].set_ylabel("[mrad/s]")
axes5[3].set_xlabel("Time [h]")
axes5[3].set_title(r"$|\omega|$ -- total angular rate magnitude (body frame)")
axes5[3].grid(True, alpha=0.3)

plt.tight_layout()
out_att = NAV / "nav_attitude.png"
plt.savefig(out_att, dpi=150)
print(f"Saved {out_att}")
plt.show(block=False)

# ============================================================================
# FIGURE 6 -- RCS performance: torque authority & delta-V budget
# ============================================================================

rcs_csv = NAV / "rcs.csv"
if rcs_csv.exists():
    rcs = np.genfromtxt(rcs_csv, delimiter=",", names=True)

    STRIDE = 60
    rcs_d  = rcs[::STRIDE]
    t_rcs  = rcs_d["time_s"] / 3600.0

    fig6, axes6 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig6.suptitle(
        "RCS Performance -- PD Attitude Controller Output  (body frame)\n"
        r"Note: pure torque couples -> net translational force ~= 0; "
        r"$\Delta V$ counts every individual thruster firing",
        fontsize=10)

    ax = axes6[0]
    ax.plot(t_rcs, rcs_d["tau_x_nm"], color="steelblue",   lw=0.7, alpha=0.85,
            label=r"$\tau_{x_B}$")
    ax.plot(t_rcs, rcs_d["tau_y_nm"], color="darkorange",  lw=0.7, alpha=0.85,
            label=r"$\tau_{y_B}$")
    ax.plot(t_rcs, rcs_d["tau_z_nm"], color="forestgreen", lw=0.7, alpha=0.85,
            label=r"$\tau_{z_B}$")
    ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.4)
    ax.set_ylabel("Torque [N·m]")
    ax.set_title("RCS torque per body axis  -- PD controller limit-cycles between "
                 r"$\pm$max torque values")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes6[1]
    tau_mag = np.sqrt(rcs_d["tau_x_nm"]**2 + rcs_d["tau_y_nm"]**2 + rcs_d["tau_z_nm"]**2)
    ax.plot(t_rcs, tau_mag, color="mediumpurple", lw=0.8)
    ax.set_ylabel(r"$|\tau|$  [N·m]")
    ax.set_title("Total torque magnitude  -- constant means thrusters fire every step "
                 "(steady limit cycle)")
    ax.grid(True, alpha=0.3)

    ax = axes6[2]
    ax.plot(rcs["time_s"] / 3600.0, rcs["dv_total_ms"], color="crimson", lw=1.0)
    ax.set_ylabel(r"Cumulative $\Delta V_\mathrm{RCS}$  [m/s]")
    ax.set_xlabel("Time [h]")
    ax.set_title(
        rf"RCS propellant budget -- total equivalent $\Delta V$ = "
        rf"{rcs['dv_total_ms'][-1]:.3f} m/s  "
        rf"(sum of all individual thruster impulses / mass)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_rcs = NAV / "nav_rcs.png"
    plt.savefig(out_rcs, dpi=150)
    print(f"Saved {out_rcs}")
    plt.show(block=False)
else:
    print("  [skip] out/rcs.csv not found -- run cargo first to generate it")

# ============================================================================
# Helper: quaternion -> rotation matrix  (body -> Hill/inertial)
# ============================================================================

def quat_to_rot(q):
    """q shape (..., 4) with [w, x, y, z]; returns R shape (..., 3, 3)."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        np.stack([1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ], axis=-1),
        np.stack([2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ], axis=-1),
        np.stack([2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)], axis=-1),
    ], axis=-2)
    return R

has_quat = all(c in truth.dtype.names for c in ("qw", "qx", "qy", "qz"))
if has_quat:
    mask_valid = ~np.isnan(truth["qw"])
    Q = np.stack([truth["qw"], truth["qx"], truth["qy"], truth["qz"]], axis=-1)
    R_all = quat_to_rot(Q)

# ============================================================================
# FIGURE 7 -- 3-D trajectory with body-frame attitude triads
# ============================================================================

if has_quat and mask_valid.any():
    fig_t = plt.figure(figsize=(13, 10))
    ax_t  = fig_t.add_subplot(111, projection="3d")
    ax_t.set_title("Spacecraft Trajectory -- Bennu Hill Frame\n"
                   "RGB arrows: body-frame axes at selected epochs "
                   r"($x_B$=red/camera, $y_B$=green, $z_B$=blue)",
                   fontsize=10)

    for i in range(len(t_h) - 1):
        c = cmap(norm_t(t_h[i]))
        ax_t.plot(tx[i:i+2], ty[i:i+2], tz[i:i+2], color=c, lw=1.2, alpha=0.8)

    ax_t.plot(est["x_m"]/1e3, est["y_m"]/1e3, est["z_m"]/1e3,
              "r--", lw=0.7, alpha=0.5, label="EKF estimate")
    ax_t.plot_surface(bx, by, bz, color="gold", alpha=0.6, zorder=1)
    ax_t.scatter([0], [0], [0], s=5, c="goldenrod", zorder=10)
    ax_t.scatter(tx[0],  ty[0],  tz[0],  s=80, c="lime", marker="^", zorder=10, label="Start")
    ax_t.scatter(tx[-1], ty[-1], tz[-1], s=80, c="red",  marker="v", zorder=10, label="End")

    N_triads  = 8
    triad_idx = np.linspace(0, len(t_h) - 1, N_triads, dtype=int)
    arrow_len = float(range_truth_km.mean()) * 0.08

    triad_colors = ["red", "limegreen", "deepskyblue"]
    triad_names  = [r"$x_B$ (camera)", r"$y_B$", r"$z_B$"]

    for ii, idx in enumerate(triad_idx):
        sc = np.array([tx[idx], ty[idx], tz[idx]])
        R  = R_all[idx]
        for j, (col_j, name_j) in enumerate(zip(triad_colors, triad_names)):
            axis_hill = R[:, j]
            ax_t.quiver(sc[0], sc[1], sc[2],
                        axis_hill[0] * arrow_len,
                        axis_hill[1] * arrow_len,
                        axis_hill[2] * arrow_len,
                        color=col_j, lw=1.2, alpha=0.85,
                        label=name_j if ii == 0 else None)

    handles, labels = ax_t.get_legend_handles_labels()
    seen = {}
    for h_item, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h_item
    ax_t.legend(seen.values(), seen.keys(), loc="upper left", fontsize=7)

    sm2 = ScalarMappable(cmap=cmap, norm=norm_t)
    sm2.set_array([])
    cb2 = fig_t.colorbar(sm2, ax=ax_t, shrink=0.55, pad=0.08, aspect=20)
    cb2.set_label("Time [h]", fontsize=9)

    ax_t.set_xlabel("x [km]  (radial, Hill)")
    ax_t.set_ylabel("y [km]  (along-track, Hill)")
    ax_t.set_zlabel("z [km]  (cross-track, Hill)")

    all_pts = np.concatenate([tx, ty, tz])
    half = max(tx.max()-tx.min(), ty.max()-ty.min(), tz.max()-tz.min()) / 2.0
    cx_m, cy_m, cz_m = (tx.max()+tx.min())/2, (ty.max()+ty.min())/2, (tz.max()+tz.min())/2
    ax_t.set_xlim(cx_m - half, cx_m + half)
    ax_t.set_ylim(cy_m - half, cy_m + half)
    ax_t.set_zlim(cz_m - half, cz_m + half)
    ax_t.set_box_aspect([1, 1, 1])

    try:
        ax_t.set_proj_type('ortho')
    except Exception:
        pass

    plt.tight_layout()
    out_3d_triads = NAV / "nav_3d_triads.png"
    plt.savefig(out_3d_triads, dpi=150)
    print(f"Saved {out_3d_triads}")
    plt.show(block=False)

# ============================================================================
# FIGURE 8 -- Pointing performance: camera boresight angle vs Bennu
# ============================================================================

has_boresight_truth = "bx" in truth.dtype.names and not np.all(np.isnan(truth["bx"]))
if has_boresight_truth:
    r_sc      = np.stack([truth["x_m"], truth["y_m"], truth["z_m"]], axis=-1)
    r_mag     = np.linalg.norm(r_sc, axis=-1, keepdims=True)
    to_bennu  = -r_sc / np.where(r_mag > 0, r_mag, 1.0)

    boresight = np.stack([truth["bx"], truth["by"], truth["bz"]], axis=-1)
    dot_prod  = np.clip(np.sum(boresight * to_bennu, axis=-1), -1.0, 1.0)
    pointing_err_deg = np.degrees(np.arccos(dot_prod))

    DEAD_BAND_DEG = 0.5
    FOV_HALF_DEG  = CAMERA_FOV_HALF_DEG

    fig8, axes8 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig8.suptitle(
        "Pointing Performance -- Camera Boresight vs Bennu Direction\n"
        "(orange dashed = RCS dead-band; red dashed = camera FOV edge)",
        fontsize=11)

    ax = axes8[0]
    ax.plot(t_h, pointing_err_deg, color="steelblue", lw=0.8, label="Pointing error")
    ax.axhline(DEAD_BAND_DEG, color="orange",  ls="--", lw=1.4,
               label=f"Dead-band ({DEAD_BAND_DEG})")
    ax.axhline(FOV_HALF_DEG,  color="crimson", ls="--", lw=1.4,
               label=f"FOV half-angle ({FOV_HALF_DEG})")
    ax.set_ylabel("Pointing error [deg]")
    ax.set_title("Full simulation -- pointing error angle")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    frac_inside = np.mean(pointing_err_deg < DEAD_BAND_DEG) * 100
    ax2 = axes8[1]
    ax2.plot(t_h, pointing_err_deg, color="steelblue", lw=0.8)
    ax2.axhline(DEAD_BAND_DEG, color="orange",  ls="--", lw=1.4,
                label=f"Dead-band ({DEAD_BAND_DEG})")
    ax2.axhline(FOV_HALF_DEG,  color="crimson", ls="--", lw=1.4,
                label=f"FOV half-angle ({FOV_HALF_DEG})")
    ax2.set_ylim(0, max(5.0, DEAD_BAND_DEG * 4))
    ax2.set_ylabel("Pointing error [deg]")
    ax2.set_xlabel("Time [h]")
    ax2.set_title(
        f"Zoomed 0-5 view  --  {frac_inside:.1f}% of time inside dead-band ({DEAD_BAND_DEG})")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_point = NAV / "nav_pointing.png"
    plt.savefig(out_point, dpi=150)
    print(f"Saved {out_point}")
    plt.show(block=False)
else:
    print("  [skip] nav_pointing.png -- no boresight data in truth.csv")

plt.show()   # block here so all windows stay open
