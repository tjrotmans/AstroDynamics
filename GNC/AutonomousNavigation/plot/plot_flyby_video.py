"""
Bennu flyby — physics-correct camera simulation.

Left panel  : OpNav camera (full ±15 deg FOV).
              Stars are fixed in the inertial frame and drift as attitude changes.
              Bennu grows/shrinks with real range.
              Lambert crescent illuminated by the actual Sun direction from the
              JPL Horizons Bennu ephemeris, rotated to the camera frame each frame.
              Lime +  = intensity-weighted centroid  (bearing measurement).
              Cyan x  = true Bennu disk centre  (truth state, not EKF estimate).

Top-right   : Hill-frame external view.
Bottom-right: Range and phase angle vs time with time cursor.

Auto-plays on open.  Run from GNC/AutonomousNavigation/:
    python plot/plot_flyby_video.py
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

R_BENNU_M    = 262.0
CAM_HALF_DEG = 15.0
CAM_HALF_RAD = np.radians(CAM_HALF_DEG)
TAN_HALF     = np.tan(CAM_HALF_RAD)
IMG_SIZE     = 256
FPS          = 30
TRAIL_LEN    = 60

root = Path(__file__).parent.parent

# ── Star catalogue — fixed unit vectors in inertial (Hill) frame ──────────────

N_STARS  = 250
_sr      = np.random.default_rng(7)
_th      = np.arccos(1.0 - 2.0 * _sr.uniform(0, 1, N_STARS))
_ph      = _sr.uniform(0, 2 * np.pi, N_STARS)
STAR_DIRS = np.column_stack([               # (N_STARS, 3) unit vectors
    np.sin(_th) * np.cos(_ph),
    np.sin(_th) * np.sin(_ph),
    np.cos(_th),
])
STAR_B = _sr.power(0.3, N_STARS) * 190 + 40   # brightness 40–230

# ── Load simulation output ────────────────────────────────────────────────────

truth = np.genfromtxt(root / "out/truth.csv",           delimiter=",", names=True)
ekf   = np.genfromtxt(root / "out/ekf_est.csv",         delimiter=",", names=True)
resid = np.genfromtxt(root / "out/opnav_residuals.csv", delimiter=",", names=True)

t_h      = truth["time_s"] / 3600.0
r_sc_m   = np.stack([truth["x_m"], truth["y_m"], truth["z_m"]], axis=-1)
tx       = truth["x_m"] / 1e3
ty       = truth["y_m"] / 1e3
ex       = ekf["x_m"]   / 1e3
ey       = ekf["y_m"]   / 1e3
range_km = np.linalg.norm(r_sc_m, axis=-1) / 1e3

# Quaternion [w, x, y, z] from truth log.
# Convention verified: R(q) maps body→inertial, so inertial→body = R.T
has_quat = all(c in truth.dtype.names for c in ("qw", "qx", "qy", "qz"))
Q_arr = (np.stack([truth["qw"], truth["qx"], truth["qy"], truth["qz"]], axis=-1)
         if has_quat else np.tile([1.0, 0.0, 0.0, 0.0], (len(t_h), 1)))

# Phase angle: interpolate from measurement cadence to truth grid
t_resid  = resid["time_s"] / 3600.0
phi_res  = (resid["phase_angle_rad"] if "phase_angle_rad" in resid.dtype.names
            else np.full(len(resid), np.pi / 2))
phase_all = np.interp(t_h, t_resid, phi_res)

N_DATA = len(t_h)

# ── Bennu heliocentric ephemeris → Sun direction in the Hill/inertial frame ───
#
# The Hill frame is just the inertial ecliptic J2000 frame centred at Bennu
# (no rotation).  The Sun direction from Bennu is −normalize(r_Bennu_helio).
# We load the same Horizons file the Rust simulation uses, interpolate to each
# truth timestep, and rotate the unit vector into the camera frame each frame.

def _load_bennu_ephem():
    """Parse a JPL Horizons state-vector file.  Returns (t_sec, r_m)."""
    for parent in Path(__file__).parents:
        p = parent / "kernels" / "horizons_results_bennu.txt"
        if p.exists():
            break
    else:
        return None, None

    lines = p.read_text().splitlines()
    try:
        soe = next(i for i, l in enumerate(lines) if l.strip() == "$$SOE")
        eoe = next(i for i, l in enumerate(lines) if l.strip() == "$$EOE")
    except StopIteration:
        return None, None

    data  = lines[soe + 1:eoe]
    t_lst, r_lst = [], []
    i = 0
    while i < len(data):
        ln = data[i].strip()
        if not ln or not ln[0].isdigit():
            i += 1
            continue
        jd  = float(ln.split()[0])
        t_s = (jd - 2_451_545.0) * 86_400.0
        xyz = data[i + 1] if i + 1 < len(data) else ""
        x   = float(re.search(r'X\s*=\s*([-+\d.E]+)', xyz).group(1)) * 1e3  # km→m
        y   = float(re.search(r'Y\s*=\s*([-+\d.E]+)', xyz).group(1)) * 1e3
        z   = float(re.search(r'Z\s*=\s*([-+\d.E]+)', xyz).group(1)) * 1e3
        t_lst.append(t_s)
        r_lst.append([x, y, z])
        i += 4   # JD line + XYZ + VXYZ + LT line
    return np.array(t_lst), np.array(r_lst)


_eph_t, _eph_r = _load_bennu_ephem()

if _eph_t is not None:
    # Interpolate Bennu's heliocentric position at every truth timestep
    _bx = np.interp(truth["time_s"], _eph_t, _eph_r[:, 0])
    _by = np.interp(truth["time_s"], _eph_t, _eph_r[:, 1])
    _bz = np.interp(truth["time_s"], _eph_t, _eph_r[:, 2])
    _br = np.stack([_bx, _by, _bz], axis=-1)
    # Unit vector from Bennu toward the Sun (opposite of heliocentric position)
    SUN_INERTIAL = -_br / np.linalg.norm(_br, axis=-1, keepdims=True)  # (N_DATA, 3)
    print(f"Sun direction loaded.  "
          f"Initial: [{SUN_INERTIAL[0,0]:.3f}, {SUN_INERTIAL[0,1]:.3f}, {SUN_INERTIAL[0,2]:.3f}]")
else:
    print("WARNING: Bennu ephemeris not found — using fixed approximate Sun direction.")
    SUN_INERTIAL = np.tile(np.array([-0.707, 0.707, 0.0]) / np.sqrt(2), (N_DATA, 1))

# ── Rotation helper ───────────────────────────────────────────────────────────

def quat_to_R(q: np.ndarray) -> np.ndarray:
    """q = [w, x, y, z].  Returns R where v_inertial = R @ v_body.
    Use R.T to rotate inertial → body."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])

# ── Camera renderer ───────────────────────────────────────────────────────────

_noise_rng = np.random.default_rng(0)

# Pixel angular coordinate grids (computed once)
_cols_g, _rows_g = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
_Y_PIX = (_cols_g / IMG_SIZE) * 2 * TAN_HALF - TAN_HALF   # camera-y angle per pixel [rad]
_Z_PIX = (_rows_g / IMG_SIZE) * 2 * TAN_HALF - TAN_HALF   # camera-z angle per pixel [rad]


def render_frame(data_idx: int) -> dict:
    """Render the full camera image at truth index `data_idx`.

    Returns dict with img (uint8 HxW), in_fov (bool),
    cen_x/cen_y (centroid in plot coords [-1,1]),
    geo_x/geo_y (true Bennu disc centre in plot coords).
    """
    q       = Q_arr[data_idx]
    R       = quat_to_R(q)       # body → inertial
    Rt      = R.T                # inertial → body  ← apply to all inertial vectors
    r_vec   = r_sc_m[data_idx]   # true spacecraft position in Hill frame [m]
    range_m = np.linalg.norm(r_vec)

    # Background sky (faint Gaussian shot noise)
    img = _noise_rng.standard_normal((IMG_SIZE, IMG_SIZE)).clip(0) * 2.5

    # ── Stars ────────────────────────────────────────────────────────────────
    # Project each star's inertial direction into the camera frame
    d_body  = (Rt @ STAR_DIRS.T).T              # (N_STARS, 3)
    ahead   = d_body[:, 0] > 1e-6
    fy      = np.where(ahead, d_body[:, 1] / d_body[:, 0], 1e9)
    fz      = np.where(ahead, d_body[:, 2] / d_body[:, 0], 1e9)
    in_fov_s = ahead & (np.abs(fy) < TAN_HALF) & (np.abs(fz) < TAN_HALF)
    s_col   = ((fy + TAN_HALF) / (2 * TAN_HALF) * IMG_SIZE).astype(int)
    s_row   = ((fz + TAN_HALF) / (2 * TAN_HALF) * IMG_SIZE).astype(int)

    for k in np.where(in_fov_s)[0]:
        sc, sr = s_col[k], s_row[k]
        if 0 <= sr < IMG_SIZE and 0 <= sc < IMG_SIZE:
            img[sr, sc] += STAR_B[k]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r2 = dr*dr + dc*dc
                    if r2 > 0:
                        nr, nc = sr + dr, sc + dc
                        if 0 <= nr < IMG_SIZE and 0 <= nc < IMG_SIZE:
                            img[nr, nc] += STAR_B[k] * np.exp(-r2 / 0.7)

    # ── Bennu disk ────────────────────────────────────────────────────────────
    los_inert    = -r_vec / range_m        # inertial unit vector: s/c → Bennu
    los_body     = Rt @ los_inert          # same in camera frame
    bennu_in_fov = bool(los_body[0] > np.cos(CAM_HALF_RAD))

    result = dict(img=None, in_fov=bennu_in_fov,
                  cen_x=None, cen_y=None, geo_x=None, geo_y=None)

    if bennu_in_fov:
        alpha  = R_BENNU_M / range_m      # apparent angular radius [rad]
        fy_ben = los_body[1] / los_body[0]
        fz_ben = los_body[2] / los_body[0]

        # True disc centre in pixel coordinates
        cx_col = (fy_ben + TAN_HALF) / (2 * TAN_HALF) * IMG_SIZE
        cx_row = (fz_ben + TAN_HALF) / (2 * TAN_HALF) * IMG_SIZE

        # Normalised angular displacement from disc centre for every pixel
        dy_n    = (_Y_PIX - fy_ben) / alpha
        dz_n    = (_Z_PIX - fz_ben) / alpha
        r2      = dy_n**2 + dz_n**2
        on_disk = r2 < 1.0

        cos_rho = np.where(on_disk, np.sqrt(np.maximum(0.0, 1.0 - r2)), 0.0)
        ny      = np.where(on_disk, dy_n, 0.0)
        nz      = np.where(on_disk, dz_n, 0.0)

        # Sun direction in camera frame — from real ephemeris, rotated each frame.
        # As attitude changes the crescent orientation rotates with it.
        sun_cam = Rt @ SUN_INERTIAL[data_idx]          # (3,) unit vector
        illum   = np.maximum(0.0,
                             cos_rho * sun_cam[0]
                             + ny    * sun_cam[1]
                             + nz    * sun_cam[2])
        bennu_px = illum * 240.0
        shot     = (_noise_rng.standard_normal((IMG_SIZE, IMG_SIZE))
                    * np.sqrt(bennu_px.clip(0)))
        bennu_px = (bennu_px + shot).clip(0.0)

        img = np.where(on_disk, bennu_px, img)

        total = bennu_px.sum()
        if total > 1.0:
            cen_col = (_cols_g * bennu_px).sum() / total
            cen_row = (_rows_g * bennu_px).sum() / total
        else:
            cen_col, cen_row = cx_col, cx_row

        # Convert pixel coords → plot coords (imshow extent=[-1,1,-1,1])
        result["cen_x"] = cen_col / IMG_SIZE * 2.0 - 1.0
        result["cen_y"] = cen_row / IMG_SIZE * 2.0 - 1.0
        result["geo_x"] = cx_col  / IMG_SIZE * 2.0 - 1.0
        result["geo_y"] = cx_row  / IMG_SIZE * 2.0 - 1.0

    result["img"] = img.clip(0, 255).astype(np.uint8)
    return result


# ── Pre-render all animation frames ──────────────────────────────────────────

N_FRAMES = 400
data_idx = np.round(np.linspace(0, N_DATA - 1, N_FRAMES)).astype(int)

print(f"Pre-rendering {N_FRAMES} frames …")
PRE = []
for fi, di in enumerate(data_idx):
    PRE.append(render_frame(di))
    if fi % 50 == 0:
        print(f"  {fi}/{N_FRAMES}")
print("Done.\n")

t_a   = t_h[data_idx]
rng_a = range_km[data_idx]
phi_a = np.degrees(phase_all[data_idx])
tx_a  = tx[data_idx];  ty_a = ty[data_idx]
ex_a  = ex[data_idx];  ey_a = ey[data_idx]

# ── Figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 7.2), facecolor="black")
fig.suptitle(
    f"Bennu Flyby  —  OpNav Camera  "
    f"(+/-{CAM_HALF_DEG} deg FOV, real attitude + ephemeris Sun, truth state)",
    color="white", fontsize=11, x=0.5, y=0.99,
)
gs = fig.add_gridspec(
    2, 3,
    width_ratios=[1.4, 1, 1], height_ratios=[1, 1],
    hspace=0.42, wspace=0.32,
    left=0.02, right=0.98, top=0.92, bottom=0.07,
)
ax_cam  = fig.add_subplot(gs[:, 0])
ax_traj = fig.add_subplot(gs[0, 1:])
ax_data = fig.add_subplot(gs[1, 1:])


def _style(ax):
    ax.set_facecolor("black")
    ax.tick_params(colors="gray", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.xaxis.label.set_color("gray")
    ax.yaxis.label.set_color("gray")
    ax.title.set_color("white")


for a in (ax_cam, ax_traj, ax_data):
    _style(a)

# Camera panel
ax_cam.set_title(f"OpNav Camera  (+/-{CAM_HALF_DEG} deg FOV)", fontsize=9, pad=4)
ax_cam.set_xticks([]);  ax_cam.set_yticks([])
ax_cam.set_xlim(-1, 1); ax_cam.set_ylim(-1, 1); ax_cam.set_aspect("equal")
ax_cam.axhline(0, color="cyan", lw=0.5, alpha=0.2)
ax_cam.axvline(0, color="cyan", lw=0.5, alpha=0.2)
ax_cam.plot(0, 0, "+", color="cyan", ms=6, mew=1.0, alpha=0.4)

cam_im = ax_cam.imshow(
    PRE[0]["img"], cmap="gray", vmin=0, vmax=255,
    origin="lower", extent=[-1, 1, -1, 1], aspect="auto",
)
f0 = PRE[0]
centroid_pt, = ax_cam.plot(
    [f0["cen_x"]] if f0["cen_x"] is not None else [],
    [f0["cen_y"]] if f0["cen_y"] is not None else [],
    "+", color="lime", ms=16, mew=2.5, label="Centroid (bearing meas.)",
)
geom_pt, = ax_cam.plot(
    [f0["geo_x"]] if f0["geo_x"] is not None else [],
    [f0["geo_y"]] if f0["geo_y"] is not None else [],
    "x", color="deepskyblue", ms=10, mew=1.8, label="True Bennu centre",
)
ax_cam.legend(fontsize=6.5, loc="lower left",
              facecolor="black", edgecolor="#555", labelcolor="white")
cam_txt = ax_cam.text(0.03, 0.98, "", transform=ax_cam.transAxes,
                      color="#FFD700", fontsize=7.5, va="top", family="monospace")

# Trajectory panel
ax_traj.set_title("Hill Frame — External View  (top-down)", fontsize=9, pad=4)
ax_traj.plot(tx, ty, lw=0.6, color="#3a3a3a", alpha=0.7)
ax_traj.plot(ex, ey, lw=0.6, color="#5a1a1a", alpha=0.4, ls="--")
_pb = np.linspace(0, 2 * np.pi, 60)
ax_traj.fill(2.62e-4 * np.cos(_pb), 2.62e-4 * np.sin(_pb),
             color="gold", alpha=0.9, zorder=5)
ax_traj.text(0, 6e-4, "Bennu", color="gold", fontsize=7, ha="center")
ax_traj.set_xlabel("x [km]  (radial)",       fontsize=7)
ax_traj.set_ylabel("y [km]  (along-track)",  fontsize=7)
ax_traj.grid(True, color="#1c1c1c")
ax_traj.set_xlim(tx.min() - 0.3, tx.max() + 0.3)
ax_traj.set_ylim(ty.min() - 0.3, ty.max() + 0.3)

trail_ln, = ax_traj.plot([], [], "-",  color="goldenrod", lw=1.2, alpha=0.75)
sc_dot,   = ax_traj.plot([], [], "D",  color="#FFD700",   ms=7,   zorder=10, label="S/C truth")
ekf_dot,  = ax_traj.plot([], [], "o",  color="crimson",   ms=5,   zorder=9,  label="EKF est.")
ax_traj.legend(fontsize=6.5, loc="upper right",
               facecolor="black", edgecolor="#555", labelcolor="white")
traj_txt = ax_traj.text(0.02, 0.04, "", transform=ax_traj.transAxes,
                        color="white", fontsize=7, family="monospace")

# Data panel
ax_data.set_title("Range & Phase Angle vs Time", fontsize=9, pad=4)
ax_d2 = ax_data.twinx()
ax_d2.set_facecolor("black")
ax_d2.tick_params(colors="#FF8C00", labelsize=7)
for sp in ax_d2.spines.values():
    sp.set_edgecolor("#444")
ax_d2.yaxis.label.set_color("#FF8C00")

ax_data.plot(t_h, range_km, color="deepskyblue", lw=1.3, alpha=0.7)
ax_data.set_ylabel("Range [km]", fontsize=7, color="deepskyblue")
ax_data.tick_params(axis="y", colors="deepskyblue", labelsize=7)
ax_data.set_xlabel("Time [h]", fontsize=7)
ax_data.grid(True, color="#1c1c1c")

ax_d2.plot(t_h, np.degrees(phase_all), color="#FF8C00", lw=1.0, alpha=0.65)
ax_d2.set_ylabel("Phase [deg]", fontsize=7)
ax_d2.set_ylim(0, 180)
ax_d2.axhline(90, color="#FF8C00", ls=":", lw=0.5, alpha=0.3)

cursor_line, = ax_data.plot([], [], color="white", lw=1.0, alpha=0.8)
cursor_dot,  = ax_data.plot([], [], "o", color="white", ms=5)

# ── Animation ─────────────────────────────────────────────────────────────────

def draw_frame(i: int):
    i  = int(i % N_FRAMES)
    fr = PRE[i]

    # Camera
    cam_im.set_data(fr["img"])
    if fr["in_fov"]:
        centroid_pt.set_data([fr["cen_x"]], [fr["cen_y"]])
        geom_pt.set_data([fr["geo_x"]],     [fr["geo_y"]])
    else:
        centroid_pt.set_data([], [])
        geom_pt.set_data([], [])

    alpha_deg = np.degrees(R_BENNU_M / (rng_a[i] * 1e3))
    cam_txt.set_text(
        f"Range:  {rng_a[i]:7.2f} km\n"
        f"Phase:  {phi_a[i]:7.1f} deg\n"
        f"alpha:  {alpha_deg:7.3f} deg\n"
        f"Bennu:  {'IN FOV' if fr['in_fov'] else 'OUTSIDE FOV'}\n"
        f"T+{t_a[i]:6.2f} h"
    )

    # Trajectory
    i0 = max(0, i - TRAIL_LEN)
    trail_ln.set_data(tx_a[i0:i + 1], ty_a[i0:i + 1])
    sc_dot.set_data([tx_a[i]],  [ty_a[i]])
    ekf_dot.set_data([ex_a[i]], [ey_a[i]])
    traj_txt.set_text(f"T+{t_a[i]:.2f} h  |  {rng_a[i]:.2f} km")

    # Time cursor
    cursor_line.set_data([t_a[i], t_a[i]], [0, rng_a[i]])
    cursor_dot.set_data([t_a[i]], [rng_a[i]])
    return []


anim = animation.FuncAnimation(
    fig, draw_frame,
    frames=N_FRAMES,
    interval=1000 // FPS,
    blit=False, repeat=True,
)

# ── Save MP4 ──────────────────────────────────────────────────────────────────

out_mp4 = root / "out" / "flyby_video.mp4"
try:
    w = animation.FFMpegWriter(fps=FPS, bitrate=3000,
                               extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    print("Saving MP4 …")
    anim.save(str(out_mp4), writer=w, dpi=130)
    print(f"Saved  {out_mp4}")
except Exception as exc:
    print(f"ffmpeg unavailable ({exc}) — skipping MP4 save")

plt.show()
