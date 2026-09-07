"""
Optical Navigation â€” how it works.

Three focused figures:
  1. Simulated camera image at three ranges  (what the camera actually sees)
  2. From image to measurements              (how bearing and angular size are extracted)
  3. FOV coverage and measurement gaps       (when the camera loses Bennu)

Usage:  python plot/plot_opnav.py
        (run after `cargo run --bin autonav`)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# â”€â”€ Parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

R_BENNU_M      = 262.0          # Bennu radius [m]
CAM_FOV_DEG    = 30.0           # full FOV [deg]
CAM_HALF_RAD   = np.radians(CAM_FOV_DEG / 2)
IMAGE_PIX      = 512
PIX_SCALE_RAD  = np.radians(CAM_FOV_DEG) / IMAGE_PIX
N_STARS        = 60
SIGMA_BEAR_RAD = 1e-4           # bearing noise 1-Ïƒ [rad]
SIGMA_SIZE_RAD = 2e-4

root = Path(__file__).parent.parent
rng  = np.random.default_rng(42)

# â”€â”€ Load run data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

try:
    truth = np.genfromtxt(root / "out/truth.csv",       delimiter=",", names=True)
    innov = np.genfromtxt(root / "out/innovations.csv", delimiter=",", names=True)
    range_km = np.sqrt(truth["x_m"]**2 + truth["y_m"]**2 + truth["z_m"]**2) / 1e3
    t_h  = truth["time_s"] / 3600.0
    t_in = innov["time_s"] / 3600.0
    avail = innov["meas_available"].astype(bool)
    has_data = True
except Exception:
    has_data = False

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FIGURE 1 â€” What the camera sees at three different ranges
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

ranges_km = [5.0, 2.0, 0.7]
labels    = ["5.0 km â€” approach start", "2.0 km â€” mid approach", "0.7 km â€” close approach"]
rng_fig   = np.random.default_rng(7)

fig1, axes = plt.subplots(1, 3, figsize=(14, 5))
fig1.suptitle("What the OpNav Camera Sees at Different Ranges", fontsize=12)

for ax, r_km, label in zip(axes, ranges_km, labels):
    img = rng_fig.normal(3, 5, (IMAGE_PIX, IMAGE_PIX)).clip(0, None)

    # Background stars
    for _ in range(N_STARS):
        sx, sy = rng_fig.integers(0, IMAGE_PIX, 2)
        bright = rng_fig.uniform(60, 220)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px, py = sx + dx, sy + dy
                if 0 <= px < IMAGE_PIX and 0 <= py < IMAGE_PIX:
                    img[py, px] += bright * np.exp(-(dx**2 + dy**2) / 0.5)

    # Bennu disk â€” centred with a small offset to show the centroid isn't always centred
    cx = IMAGE_PIX / 2 + rng_fig.normal(0, 8)
    cy = IMAGE_PIX / 2 + rng_fig.normal(0, 8)
    r_pix = (R_BENNU_M / (r_km * 1e3)) / PIX_SCALE_RAD

    yg, xg = np.ogrid[:IMAGE_PIX, :IMAGE_PIX]
    d2 = (xg - cx)**2 + (yg - cy)**2
    mask = d2 <= r_pix**2
    limb = np.where(mask, np.sqrt(np.maximum(0, 1 - d2 / max(r_pix**2, 1e-9))), 0)
    img += 230 * limb
    img = img.clip(0, 255)

    ax.imshow(img, cmap="gray", vmin=0, vmax=255, origin="lower")
    ax.set_facecolor("black")

    # Annotate centroid (= bearing measurement source)
    ax.plot(cx, cy, "+", color="lime", ms=14, mew=2, label="Centroid â†’ bearing")

    # Annotate disk edge (= angular-size measurement source)
    if r_pix >= 1.5:
        circle = plt.Circle((cx, cy), r_pix, color="deepskyblue", fill=False,
                             lw=1.5, ls="--")
        ax.add_patch(circle)
        ax.annotate("", xy=(cx + r_pix, cy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="<->", color="deepskyblue", lw=1.2))
        ax.text(cx + r_pix / 2, cy + 8, f"Î± = {np.degrees(R_BENNU_M/(r_km*1e3)):.1f}Â°",
                color="deepskyblue", fontsize=7, ha="center")

    ax.set_title(f"{label}\nBennu = {r_pix:.1f} px radius", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if r_km == ranges_km[0]:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.5)

plt.tight_layout()
out1 = root / "out/opnav_camera_images.png"
plt.savefig(out1, dpi=150)
print(f"Saved {out1}")
plt.show()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FIGURE 2 â€” From image to EKF measurements
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("From Camera Image to EKF Measurements", fontsize=12)

# â”€â”€ Left: Bearing = centroid angle in camera frame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax = axes2[0]
ax.set_facecolor("#0a0a15")
ax.set_xlim(-0.06, 0.06); ax.set_ylim(-0.06, 0.06)
ax.set_aspect("equal")
ax.set_title("Bearing Measurement\n(centroid angle from boresight)", fontsize=10)
ax.set_xlabel("Camera y-angle [rad]"); ax.set_ylabel("Camera z-angle [rad]")
ax.tick_params(colors="white"); ax.spines[:].set_color("gray")
ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
ax.title.set_color("white")

# FOV boundary
fov_circle = plt.Circle((0, 0), CAM_HALF_RAD, color="deepskyblue",
                         fill=False, lw=1.5, ls="--", label=f"FOV edge (Â±{CAM_FOV_DEG/2:.0f}Â°)")
ax.add_patch(fov_circle)
ax.plot(0, 0, "+", color="white", ms=8, mew=1.5, label="Boresight (body +x)")

# A few simulated Bennu centroid positions with noise
true_angle_y = 0.015; true_angle_z = -0.010
for _ in range(30):
    ny = true_angle_y + rng.normal(0, SIGMA_BEAR_RAD)
    nz = true_angle_z + rng.normal(0, SIGMA_BEAR_RAD)
    ax.plot(ny, nz, ".", color="lime", ms=3, alpha=0.5)
ax.plot(true_angle_y, true_angle_z, "o", color="gold", ms=8, zorder=5,
        label="True centroid")
# Draw the bearing angle arrow from boresight to centroid
ax.annotate("", xy=(true_angle_y, true_angle_z), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="gold", lw=1.5))
bear_deg = np.degrees(np.sqrt(true_angle_y**2 + true_angle_z**2))
ax.text(0.008, -0.005, f"bearing\n= {bear_deg:.2f}Â°", color="gold", fontsize=8)

# Star tracker rotation label
ax.text(-0.055, -0.055, "Star tracker\nrotates camera\nâ†’ inertial frame",
        color="cyan", fontsize=7, va="bottom")

ax.legend(loc="upper right", fontsize=7)
ax.grid(True, alpha=0.2, color="gray")

# â”€â”€ Right: Angular size â†’ range â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax = axes2[1]
r_arr = np.linspace(0.3, 6.0, 300)
alpha_deg = np.degrees(R_BENNU_M / (r_arr * 1e3))
r_pix_arr = (R_BENNU_M / (r_arr * 1e3)) / PIX_SCALE_RAD

ax.plot(r_arr, alpha_deg, "deepskyblue", lw=2, label="True Î± = R_Bennu / range")

# Noise band (Â±1Ïƒ)
sigma_deg = np.degrees(SIGMA_SIZE_RAD)
ax.fill_between(r_arr, alpha_deg - sigma_deg, alpha_deg + sigma_deg,
                color="deepskyblue", alpha=0.2, label=f"Â±1Ïƒ noise ({sigma_deg*1000:.1f} mdeg)")

# Pixel resolution limit
ax.axhline(np.degrees(PIX_SCALE_RAD), color="gray", ls=":", lw=1,
           label=f"1 pixel = {np.degrees(PIX_SCALE_RAD)*60:.1f} arcmin")

# Mark selected ranges
for r_mark, col in [(5.0, "green"), (2.0, "orange"), (0.7, "red")]:
    a = np.degrees(R_BENNU_M / (r_mark * 1e3))
    ax.annotate(f"{r_mark} km\nâ†’ Î±={a:.2f}Â°", xy=(r_mark, a),
                xytext=(r_mark + 0.3, a + 0.3),
                arrowprops=dict(arrowstyle="->", color=col, lw=1),
                color=col, fontsize=8)

ax.set_xlabel("Range to Bennu [km]")
ax.set_ylabel("Angular radius Î± [deg]")
ax.set_title("Angular Size â†’ Range\nr = R_Bennu / Î±  (range from one number)", fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 6.5); ax.set_ylim(0)

plt.tight_layout()
out2 = root / "out/opnav_measurements.png"
plt.savefig(out2, dpi=150)
print(f"Saved {out2}")
plt.show()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FIGURE 3 â€” FOV coverage and innovation timeline (needs run data)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if not has_data:
    print("No run data found â€” skipping Figure 3. Run autonav first.")
else:
    fig3, axes3 = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig3.suptitle("Measurement Timeline â€” FOV Coverage and Innovation Quality", fontsize=12)

    # Panel 1: range + missed measurements
    ax = axes3[0]
    ax.plot(t_h, range_km, "steelblue", lw=1.5, label="Range to Bennu")
    if (~avail).any():
        ax.scatter(t_in[~avail], range_km[np.searchsorted(t_h, t_in[~avail], side="left").clip(0, len(range_km)-1)],
                   s=12, c="red", zorder=5, label="Bennu outside FOV")
    ax.set_ylabel("Range [km]")
    ax.set_title("Range to Bennu  (red = missed measurement epoch)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: bearing innovation
    ax = axes3[1]
    los_vals = np.abs(innov["innov_los_rad"])
    los_vals[~avail] = np.nan
    ax.semilogy(t_in, los_vals, "b.", ms=3, alpha=0.6, label="Bearing innovation")
    ax.axhline(SIGMA_BEAR_RAD, color="b", ls="--", lw=0.8, alpha=0.5,
               label=f"1-Ïƒ bearing noise ({SIGMA_BEAR_RAD:.0e} rad)")
    ax.set_ylabel("|Innovation| [rad]")
    ax.set_title("Bearing Innovation  (gaps = missed measurements; spikes = off-pointing + star tracker noise)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 3: angular-size innovation
    ax = axes3[2]
    sz_vals = np.abs(innov["innov_size_rad"])
    sz_vals[~avail] = np.nan
    ax.semilogy(t_in, sz_vals, "r.", ms=3, alpha=0.6, label="Angular-size innovation")
    ax.axhline(SIGMA_SIZE_RAD, color="r", ls="--", lw=0.8, alpha=0.5,
               label=f"1-Ïƒ size noise ({SIGMA_SIZE_RAD:.0e} rad)")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("|Innovation| [rad]")
    ax.set_title("Angular-Size Innovation  (encodes range estimation error)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out3 = root / "out/opnav_fov_timeline.png"
    plt.savefig(out3, dpi=150)
    print(f"Saved {out3}")
    plt.show()
