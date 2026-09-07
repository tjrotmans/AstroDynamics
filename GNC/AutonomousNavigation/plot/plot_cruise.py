"""
Earth → Bennu Cruise Phase — Visualisation.

Reads CSVs written by `cargo run -p autonomous_navigation --bin cruise_design`.

Outputs:
  out/cruise_porkchop.png    — departure ΔV and total ΔV contour maps
  out/cruise_transfer.png    — best transfer plotted in the ecliptic plane

Run from GNC/AutonomousNavigation/:
    python plot/plot_cruise.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "out" / "cruise"

AU = 1.495_978_707e11

# ── Load CSVs ─────────────────────────────────────────────────────────────────

pc   = np.genfromtxt(OUT / "porkchop.csv",       delimiter=",", names=True)
traj = np.genfromtxt(OUT / "best_transfer.csv",  delimiter=",", names=True)
sol  = np.genfromtxt(OUT / "best_solution.csv",  delimiter=",", names=True)
meta = np.genfromtxt(OUT / "transfer_meta.csv",  delimiter=",", names=True)

dep_day  = float(sol["dep_day"])
tof_day  = float(sol["tof_day"])
dv_dep   = float(sol["dv_dep_kms"])
dv_arr   = float(sol["dv_arr_kms"])
dv_total = float(sol["dv_total_kms"])
dep_year = float(sol["dep_year"])

print(f"Best refined solution:")
print(f"  Departure : J2000+{dep_day:.0f} d  ({dep_year:.2f} yr)")
print(f"  TOF       : {tof_day:.0f} d")
print(f"  ΔV_total  : {dv_total:.3f} km/s")

# ── Porkchop grid → 2-D arrays ────────────────────────────────────────────────

dep_vals = np.unique(pc["dep_day"])
tof_vals = np.unique(pc["tof_day"])
ND, NT   = len(dep_vals), len(tof_vals)

dv_dep_grid   = np.full((NT, ND), np.nan)
dv_total_grid = np.full((NT, ND), np.nan)

dep_idx = {d: i for i, d in enumerate(dep_vals)}
tof_idx = {t: i for i, t in enumerate(tof_vals)}

for row in pc:
    j = dep_idx.get(row["dep_day"])
    i = tof_idx.get(row["tof_day"])
    if j is not None and i is not None:
        dv_dep_grid[i, j]   = row["dv_dep_kms"]
        dv_total_grid[i, j] = row["dv_total_kms"]

# ── Figure 1: Porkchop ────────────────────────────────────────────────────────

fig1, axes = plt.subplots(1, 2, figsize=(15, 6))
fig1.suptitle("Earth → Bennu Porkchop  (heliocentric ecliptic J2000, JPL Bennu ephemeris)",
              fontsize=12, fontweight="bold")

for ax, grid, title, cmap, vmax, cblbl in [
    (axes[0], dv_dep_grid,   "Departure ΔV  [km/s]", "plasma",   6.0,
     "ΔV_dep  [km/s]"),
    (axes[1], dv_total_grid, "Total ΔV  [km/s]",     "RdYlGn_r", 10.0,
     "ΔV_total  [km/s]"),
]:
    clipped = np.clip(grid, 0, vmax)
    pcm = ax.contourf(dep_vals, tof_vals, clipped, levels=30, cmap=cmap)
    cs  = ax.contour(dep_vals, tof_vals, clipped,
                     levels=15, colors="white", linewidths=0.4, alpha=0.35)
    ax.clabel(cs, fmt="%.1f", fontsize=6, colors="white")
    fig1.colorbar(pcm, ax=ax, label=cblbl, fraction=0.04, pad=0.02)

    ax.axvline(dep_day, color="cyan",  lw=1.2, ls="--", alpha=0.7)
    ax.axhline(tof_day, color="cyan",  lw=1.2, ls="--", alpha=0.7)
    ax.scatter(dep_day, tof_day, marker="*", color="cyan", s=250, zorder=10,
               label=f"Best  ΔV={dv_total:.2f} km/s\n"
                     f"dep={dep_day:.0f} d ({dep_year:.1f} yr),  TOF={tof_day:.0f} d")

    # Mark OSIRIS-REx window (~2016 = yr 16 ≈ day 5844)
    ax.axvline(16 * 365.25, color="gold", lw=0.8, ls=":", alpha=0.6,
               label="OSIRIS-REx era (~yr 16)")

    ax.set_xlabel("Departure (days from J2000)")
    ax.set_ylabel("Time of Flight  [days]")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.12)

    # Secondary x-axis in years
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0] / 365.25, ax.get_xlim()[1] / 365.25)
    ax2.set_xlabel("Departure  [years from J2000]", fontsize=8)

plt.tight_layout()
out_pc = OUT / "cruise_porkchop.png"
plt.savefig(out_pc, dpi=150)
print(f"Saved {out_pc}")
plt.show(block=False)

# ── Figure 2: Transfer orbit — 3-D ───────────────────────────────────────────

from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers the projection)

MU   = 1.327_124_400_41e20
A_B, E_B = 1.1264 * AU, 0.2037
I_B      = 6.034  * np.pi / 180
RAAN_B   = 2.060  * np.pi / 180
AOP_B    = 66.22  * np.pi / 180

def _kep2r(a, e, i, raan, aop, nu):
    p  = a * (1 - e**2)
    r  = p / (1 + e * np.cos(nu))
    co, so = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i),    np.sin(i)
    ca, sa = np.cos(aop),  np.sin(aop)
    R = np.array([
        [co*ca - so*sa*ci,  -co*sa - so*ca*ci,  so*si],
        [so*ca + co*sa*ci,  -so*sa + co*ca*ci, -co*si],
        [      sa*si,               ca*si,          ci],
    ])
    return R @ (r * np.array([np.cos(nu), np.sin(nu), 0.0]))

nu_arr    = np.linspace(0, 2 * np.pi, 600)
earth_orb = np.array([[np.cos(v), np.sin(v), 0.0] for v in nu_arr])          # AU
bennu_orb = np.array([_kep2r(A_B, E_B, I_B, RAAN_B, AOP_B, v) / AU
                      for v in nu_arr])

# Spacecraft + body tracks from CSV (AU)
sc    = np.column_stack([traj["sc_x_m"],    traj["sc_y_m"],    traj["sc_z_m"]])    / AU
bn_tr = np.column_stack([traj["bennu_x_m"], traj["bennu_y_m"], traj["bennu_z_m"]]) / AU
ea_tr = np.column_stack([traj["earth_x_m"], traj["earth_y_m"], traj["earth_z_m"]]) / AU

dep_pos = sc[0]          # Earth at departure
arr_pos = bn_tr[-1]      # Bennu at arrival

# ΔV vectors (m/s → AU-normalised for arrows)
V_E = np.sqrt(MU / AU)
arrow_scale = 0.10 / V_E
dv_dep_vec = np.array([float(meta["dv_dep_x"]), float(meta["dv_dep_y"]), 0.0]) * arrow_scale
dv_arr_vec = np.array([float(meta["dv_arr_x"]), float(meta["dv_arr_y"]), 0.0]) * arrow_scale

fig2 = plt.figure(figsize=(11, 9))
fig2.patch.set_facecolor("#06060e")
ax3  = fig2.add_subplot(111, projection="3d")
ax3.set_facecolor("#06060e")

# Orbital rings
ax3.plot(earth_orb[:, 0], earth_orb[:, 1], earth_orb[:, 2],
         color="#4488ff", lw=0.9, alpha=0.35, label="Earth orbit")
ax3.plot(bennu_orb[:, 0], bennu_orb[:, 1], bennu_orb[:, 2],
         color="#c8a84b", lw=0.9, alpha=0.35, label="Bennu orbit")

# Body tracks during transfer
ax3.plot(ea_tr[:, 0], ea_tr[:, 1], ea_tr[:, 2],
         color="#4488ff", lw=1.0, ls="--", alpha=0.55, label="Earth (transfer window)")
ax3.plot(bn_tr[:, 0], bn_tr[:, 1], bn_tr[:, 2],
         color="#ffcc44", lw=1.0, ls="--", alpha=0.55, label="Bennu (transfer window)")

# Spacecraft trajectory — colour by mission time
n_seg = len(sc) - 1
cmap  = plt.get_cmap("plasma")
for k in range(n_seg):
    c = cmap(k / n_seg)
    ax3.plot(sc[k:k+2, 0], sc[k:k+2, 1], sc[k:k+2, 2],
             color=c, lw=2.2, alpha=0.92)
# Dummy line for legend colour-bar label
ax3.plot([], [], [], color=cmap(0.5), lw=2.2, label="Transfer trajectory (departure→arrival)")

# Sun, departure, arrival markers
ax3.scatter([0], [0], [0], color="yellow", s=300, zorder=10, label="Sun", depthshade=False)
ax3.scatter(*dep_pos, color="#6699ff", s=130, marker="^", zorder=9,
            label="Earth at departure", depthshade=False)
ax3.scatter(*arr_pos, color="#ffcc44", s=130, marker="v", zorder=9,
            label="Bennu at arrival",   depthshade=False)

# ΔV arrows (quiver)
ax3.quiver(*dep_pos, *dv_dep_vec, color="lime",   linewidth=2.0, arrow_length_ratio=0.25)
ax3.quiver(*arr_pos, *dv_arr_vec, color="tomato", linewidth=2.0, arrow_length_ratio=0.25)
ax3.text(*(dep_pos + dv_dep_vec * 1.4),
         f"ΔV_dep\n{dv_dep:.2f} km/s", color="lime",   fontsize=7.5, ha="center")
ax3.text(*(arr_pos + dv_arr_vec * 1.4),
         f"ΔV_arr\n{dv_arr:.2f} km/s", color="tomato", fontsize=7.5, ha="center")

# Equal-ish aspect: set all axes to the same range
all_pts = np.vstack([sc, bn_tr, ea_tr, earth_orb, bennu_orb])
for dim, setter in enumerate([ax3.set_xlim3d, ax3.set_ylim3d, ax3.set_zlim3d]):
    mid = (all_pts[:, dim].max() + all_pts[:, dim].min()) / 2
    half = (all_pts[:, :2].max() - all_pts[:, :2].min()) / 2 * 0.65  # keep x-y scale for z
    setter([mid - half, mid + half])

# Cosmetics
ax3.set_xlabel("x  [AU]", color="#aaaacc", labelpad=6)
ax3.set_ylabel("y  [AU]", color="#aaaacc", labelpad=6)
ax3.set_zlabel("z  [AU]", color="#aaaacc", labelpad=6)
ax3.tick_params(colors="#777799", labelsize=7)
ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
ax3.xaxis.pane.set_edgecolor("#1a1a30")
ax3.yaxis.pane.set_edgecolor("#1a1a30")
ax3.zaxis.pane.set_edgecolor("#1a1a30")
ax3.grid(True, color="#1a1a2e", lw=0.4, alpha=0.5)

ax3.set_title(
    f"Best Direct Earth → Bennu Transfer  (3-D ecliptic J2000)\n"
    f"Dep J2000+{dep_day:.0f} d ({dep_year:.2f} yr)  |  TOF {tof_day:.0f} d  |  "
    f"ΔV_total = {dv_total:.3f} km/s",
    color="white", fontsize=9, pad=10)
ax3.legend(fontsize=7.5, facecolor="#111130", labelcolor="white",
           edgecolor="#333355", framealpha=0.85, loc="upper left")

plt.tight_layout()
out_trj = OUT / "cruise_transfer_3d.png"
plt.savefig(out_trj, dpi=150, bbox_inches="tight")
print(f"Saved {out_trj}")
plt.show()
