"""
RCS Thruster Animation -- realistic spacecraft in Hill frame.

Left panel  : 3-D scene. Spacecraft moves along approach trajectory toward Bennu
              (at Hill-frame origin). Body = bus + solar-panel wings + HGA + camera.
              12 RCS nozzles as cones. Exhaust plumes when firing.
Right panels: omega_x / omega_y / omega_z [mrad/s] + |tau| [N*m]

Run from GNC/AutonomousNavigation/:
    python plot/plot_rcs_animation.py
Saves:  out/rcs_animation.mp4  (ffmpeg required)
        out/rcs_animation.gif  (Pillow fallback)
"""

from __future__ import annotations
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

# -- Settings ------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
T_SHOW_H = 0.5       # hours to animate
STRIDE   = 8         # sim-seconds per animation frame
FPS      = 15
DPI      = 120
BG       = "#07070f"

# -- Spacecraft geometry (body frame, metres -> animation units at 1 m = 1 u) --
# Main bus: 1.0 x 0.55 x 0.40  (xB forward/camera, yB solar-panel axis, zB up)
BX, BY, BZ = 0.50, 0.275, 0.20   # half-extents of bus

# Bus vertices (8 corners)
BUS_V = np.array([
    [-BX, -BY, -BZ], [+BX, -BY, -BZ],
    [+BX, +BY, -BZ], [-BX, +BY, -BZ],
    [-BX, -BY, +BZ], [+BX, -BY, +BZ],
    [+BX, +BY, +BZ], [-BX, +BY, +BZ],
], dtype=float)

BUS_FACES = [
    [0,3,2,1],  # -z bottom
    [4,5,6,7],  # +z top
    [0,1,5,4],  # -y
    [2,3,7,6],  # +y
    [0,4,7,3],  # -x back (HGA side)
    [1,2,6,5],  # +x CAMERA face
]

def _rgba(h, a):
    return (int(h[1:3],16)/255, int(h[3:5],16)/255, int(h[5:7],16)/255, a)

BUS_RGBA = [
    _rgba("#0a0a20", 0.5),   # bottom
    _rgba("#12122a", 0.5),   # top
    _rgba("#101828", 0.6),   # -y
    _rgba("#101828", 0.6),   # +y
    _rgba("#0c0c22", 0.6),   # -x back
    _rgba("#b83200", 0.95),  # +x CAMERA (orange-red marker)
]
BUS_EC = "#252550"

# Solar panels: flat wings extending outward in +/-y from bus faces
WING_OUT = 0.65   # extent from bus edge outward
WING_X   = 0.45   # half-length along xB axis
WING_Z   = 0.12   # half-thickness in zB

def _panel_verts(sign):
    y_in  = sign * (BY + 0.005)
    y_out = sign * (BY + WING_OUT)
    return np.array([
        [-WING_X, y_in,  -WING_Z], [+WING_X, y_in,  -WING_Z],
        [+WING_X, y_out, -WING_Z], [-WING_X, y_out, -WING_Z],
    ], dtype=float)

PANEL_P  = _panel_verts(+1)   # +y panel
PANEL_N  = _panel_verts(-1)   # -y panel

# High-gain antenna disk on -x face: 8-gon approximation
HGA_R = 0.28
HGA_N = 12
HGA_X = -BX - 0.04   # just off the -x face
_ang = np.linspace(0, 2*np.pi, HGA_N, endpoint=False)
HGA_VERTS = np.stack(
    [np.full(HGA_N, HGA_X), HGA_R*np.cos(_ang), HGA_R*np.sin(_ang)], axis=-1
)

# Camera protrusion on +x face: small box
CAM_DX, CAM_DY, CAM_DZ = 0.10, 0.08, 0.08
CAM_X = BX
CAM_V = np.array([
    [CAM_X,      -CAM_DY, -CAM_DZ],
    [CAM_X+CAM_DX,-CAM_DY,-CAM_DZ],
    [CAM_X+CAM_DX,+CAM_DY,-CAM_DZ],
    [CAM_X,      +CAM_DY, -CAM_DZ],
    [CAM_X,      -CAM_DY, +CAM_DZ],
    [CAM_X+CAM_DX,-CAM_DY,+CAM_DZ],
    [CAM_X+CAM_DX,+CAM_DY,+CAM_DZ],
    [CAM_X,      +CAM_DY, +CAM_DZ],
])
CAM_FACES = [
    [0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]
]

# -- Thruster layout (12 thrusters, matching rcs.rs moment arms) ---------------
a = 0.9   # moment arm scale relative to bus half-extents
T_DATA = np.array([
    # torque about +x
    [ 0,  BY*a, 0,   0,  0,  1],   #  0  +x torque
    [ 0, -BY*a, 0,   0,  0, -1],   #  1  +x torque
    [ 0, -BY*a, 0,   0,  0,  1],   #  2  -x torque
    [ 0,  BY*a, 0,   0,  0, -1],   #  3  -x torque
    # torque about +y
    [ 0,  0,  BZ*a,  1,  0,  0],   #  4  +y torque
    [ 0,  0, -BZ*a, -1,  0,  0],   #  5  +y torque
    [ 0,  0, -BZ*a,  1,  0,  0],   #  6  -y torque
    [ 0,  0,  BZ*a, -1,  0,  0],   #  7  -y torque
    # torque about +z
    [ BX*a, 0,  0,   0,  1,  0],   #  8  +z torque
    [-BX*a, 0,  0,   0, -1,  0],   #  9  +z torque
    [-BX*a, 0,  0,   0,  1,  0],   # 10  -z torque
    [ BX*a, 0,  0,   0, -1,  0],   # 11  -z torque
], dtype=float)

T_POS = T_DATA[:, :3]
T_DIR = T_DATA[:, 3:]

def _normalise_rows(arr):
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(n > 0, n, 1.0)

T_DIR = _normalise_rows(T_DIR)

NOZZLE_OFF  = 0.07
NOZZLE_R    = 0.045
NOZZLE_CONE = 0.10

T_COL_ON = [
    "#FF3030", "#FF3030",
    "#FF9090", "#FF9090",
    "#30FF60", "#30FF60",
    "#90FFB0", "#90FFB0",
    "#3090FF", "#3090FF",
    "#99BBFF", "#99BBFF",
]
T_COL_OFF = "#1a2030"

THRESH = 1e-5   # N*m -- below this, thruster considered off

# -- Quaternion helpers --------------------------------------------------------
def q2R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

def slerp(q1, q2, t):
    d = float(np.clip(np.dot(q1, q2), -1.0, 1.0))
    if d < 0.0:
        q2, d = -q2, -d
    if d > 0.9995:
        q = q1 + t*(q2 - q1); return q / np.linalg.norm(q)
    th = np.arccos(d); s = np.sin(th)
    return np.sin((1-t)*th)/s * q1 + np.sin(t*th)/s * q2

# -- Nozzle cone geometry (body frame) ----------------------------------------
def _cone_polys(pos, exhaust_dir, n_sides=8):
    tip  = pos + exhaust_dir * (NOZZLE_OFF + NOZZLE_CONE)
    base = pos + exhaust_dir * NOZZLE_OFF
    u = exhaust_dir
    w = np.array([1,0,0]) if abs(u[0]) < 0.9 else np.array([0,1,0])
    v1 = np.cross(u, w); v1 /= np.linalg.norm(v1)
    v2 = np.cross(u, v1)
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    rim = base + NOZZLE_R * (np.outer(np.cos(angles), v1) +
                              np.outer(np.sin(angles), v2))
    polys = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        polys.append([tip, rim[i], rim[j]])
    for i in range(n_sides):
        j = (i + 1) % n_sides
        polys.append([base, rim[i], rim[j]])
    return polys

_CONES_BODY = [_cone_polys(T_POS[i], T_DIR[i]) for i in range(12)]

# -- Load data ----------------------------------------------------------------
print("Loading CSVs ...")
truth = np.genfromtxt(ROOT/"out/truth.csv", delimiter=",", names=True)
rcs   = np.genfromtxt(ROOT/"out/rcs.csv",   delimiter=",", names=True)

T_MAX  = T_SHOW_H * 3600.0
truth  = truth[truth["time_s"] <= T_MAX]
rcs    = rcs[rcs["time_s"]     <= T_MAX]

t_tr  = truth["time_s"]
t_rc  = rcs["time_s"]
Q_tr  = np.stack([truth["qw"], truth["qx"], truth["qy"], truth["qz"]], axis=-1)

# Trajectory in Hill frame (metres), normalised to animation units
traj_xyz = np.stack([truth["x_m"], truth["y_m"], truth["z_m"]], axis=-1)
SCENE_SCALE = 5500.0   # 1 animation unit = ~5500 m (5 km starts at ~0.9 u)
traj_u = traj_xyz / SCENE_SCALE   # (N,3) in scene units

# In the Hill frame Bennu IS the origin; spacecraft moves toward (0,0,0)
BENNU_POS = np.zeros(3)

# -- Animation time axis -------------------------------------------------------
t_anim   = np.arange(0.0, T_MAX, float(STRIDE))
n_frames = len(t_anim)

def interp_q(t_target):
    i = int(np.clip(np.searchsorted(t_tr, t_target, "right") - 1, 0, len(t_tr)-2))
    dt = t_tr[i+1] - t_tr[i]
    fr = float((t_target - t_tr[i]) / dt) if dt > 0 else 0.0
    return slerp(Q_tr[i], Q_tr[i+1], fr)

print(f"Pre-computing {n_frames} quaternion frames ...")
Q_anim = np.array([interp_q(t) for t in t_anim])

# -- Thruster firing: OR over the STRIDE window --------------------------------
def _window_fire(fi):
    t0 = t_anim[fi]
    t1 = t0 + STRIDE
    mask_rows = (t_rc >= t0) & (t_rc < t1)
    if not np.any(mask_rows):
        return np.zeros(12, dtype=bool)
    tx = rcs["tau_x_nm"][mask_rows]
    ty = rcs["tau_y_nm"][mask_rows]
    tz = rcs["tau_z_nm"][mask_rows]
    m = np.zeros(12, dtype=bool)
    if np.any(tx >  THRESH): m[[0,1]] = True
    if np.any(tx < -THRESH): m[[2,3]] = True
    if np.any(ty >  THRESH): m[[4,5]] = True
    if np.any(ty < -THRESH): m[[6,7]] = True
    if np.any(tz >  THRESH): m[[8,9]] = True
    if np.any(tz < -THRESH): m[[10,11]] = True
    return m

def _s(arr, ts):
    return np.interp(t_anim, ts, arr)

omega_x = _s(rcs["omega_x_rads"], t_rc) * 1e3
omega_y = _s(rcs["omega_y_rads"], t_rc) * 1e3
omega_z = _s(rcs["omega_z_rads"], t_rc) * 1e3
tau_x   = _s(rcs["tau_x_nm"],     t_rc)
tau_y   = _s(rcs["tau_y_nm"],     t_rc)
tau_z   = _s(rcs["tau_z_nm"],     t_rc)
tau_mag = np.sqrt(tau_x**2 + tau_y**2 + tau_z**2)

print("Pre-computing fire masks ...")
FIRE = [_window_fire(fi) for fi in range(n_frames)]

# -- Figure layout -------------------------------------------------------------
fig = plt.figure(figsize=(16, 8), facecolor=BG)
gs  = gridspec.GridSpec(4, 2, fig,
                        left=0.00, right=0.98, top=0.93, bottom=0.06,
                        wspace=0.28, hspace=0.45)
ax3   = fig.add_subplot(gs[:, 0], projection="3d")
ax_ts = [fig.add_subplot(gs[r, 1]) for r in range(4)]

TS_CFG = [
    (omega_x, "wx  [mrad/s]",  "#4FC3F7"),
    (omega_y, "wy  [mrad/s]",  "#FFB347"),
    (omega_z, "wz  [mrad/s]",  "#55FF88"),
    (tau_mag, "|tau|  [N*m]",  "#CC88FF"),
]
t_h     = t_anim / 3600.0
cursors = []
for ax, (data, lbl, col) in zip(ax_ts, TS_CFG):
    ax.set_facecolor("#0c0c1e")
    for sp in ax.spines.values(): sp.set_edgecolor("#252545")
    ax.tick_params(colors="#666688", labelsize=7)
    ax.grid(True, color="#18182e", lw=0.5)
    ax.plot(t_h, data, color=col, lw=0.9, alpha=0.9)
    ax.axhline(0, color="#252545", lw=0.6)
    cur = ax.axvline(0.0, color="white", lw=1.2, ls="--", alpha=0.75)
    ax.set_ylabel(lbl, fontsize=8, color="#aaaacc")
    cursors.append(cur)
ax_ts[-1].set_xlabel("Time [h]", fontsize=8, color="#aaaacc")
for ax in ax_ts[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)

fig.suptitle("RCS Attitude Control -- 6DOF Proximity Phase (Bennu Approach)",
             color="white", fontsize=11, y=0.97)

_leg = [
    mpatches.Patch(fc="#FF3030",   label="+tx  T0,T1"),
    mpatches.Patch(fc="#FF9090",   label="-tx  T2,T3"),
    mpatches.Patch(fc="#30FF60",   label="+ty  T4,T5"),
    mpatches.Patch(fc="#90FFB0",   label="-ty  T6,T7"),
    mpatches.Patch(fc="#3090FF",   label="+tz  T8,T9"),
    mpatches.Patch(fc="#99BBFF",   label="-tz  T10,T11"),
    mpatches.Patch(fc="#b83200",   label="Camera (+xB)"),
    mpatches.Patch(fc=T_COL_OFF,   label="Idle nozzle"),
]

# -- 3-D scene helpers ---------------------------------------------------------
LIM  = 1.55
AXIS = 0.55

def _style_3d(ax):
    ax.set_facecolor(BG)
    for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        p.fill = False; p.set_edgecolor("none")
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)

def _sphere_mesh(cx, cy, cz, r, nu=14, nv=10):
    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(0,   np.pi, nv)
    xs = cx + r * np.outer(np.cos(u), np.sin(v))
    ys = cy + r * np.outer(np.sin(u), np.sin(v))
    zs = cz + r * np.outer(np.ones(nu), np.cos(v))
    return xs, ys, zs

def _plume_mesh(tip, dir_, fi_on, length=0.25):
    r = 0.06; n = 10
    base = tip + dir_ * length
    u = dir_
    w = np.array([1,0,0]) if abs(u[0]) < 0.9 else np.array([0,1,0])
    v1 = np.cross(u, w); v1 /= np.linalg.norm(v1)
    v2 = np.cross(u, v1)
    ang = np.linspace(0, 2*np.pi, n)
    rim = base + r * (np.outer(np.cos(ang), v1) + np.outer(np.sin(ang), v2))
    xs = np.zeros((2, n)); ys = np.zeros((2, n)); zs = np.zeros((2, n))
    xs[0,:] = tip[0];   ys[0,:] = tip[1];   zs[0,:] = tip[2]
    xs[1,:] = rim[:,0]; ys[1,:] = rim[:,1]; zs[1,:] = rim[:,2]
    return xs, ys, zs

# -- Per-frame draw ------------------------------------------------------------
def draw(fi):
    ax3.cla()
    _style_3d(ax3)
    ax3.view_init(elev=22, azim=28 + fi * 0.20)

    R = q2R(Q_anim[fi])

    # Interpolate current spacecraft position along the approach trajectory
    sc_pos = np.array([
        np.interp(t_anim[fi], t_tr, traj_u[:, 0]),
        np.interp(t_anim[fi], t_tr, traj_u[:, 1]),
        np.interp(t_anim[fi], t_tr, traj_u[:, 2]),
    ])

    # Trajectory trail: visited portion bright, remaining path dim
    trail_end = min(int(np.searchsorted(t_tr, t_anim[fi], 'right')) + 1, len(traj_u))
    if trail_end > 1:
        ax3.plot(traj_u[:trail_end, 0], traj_u[:trail_end, 1], traj_u[:trail_end, 2],
                 color="#3366aa", lw=0.8, ls=":", alpha=0.75)
    ax3.plot(traj_u[:, 0], traj_u[:, 1], traj_u[:, 2],
             color="#1a2233", lw=0.5, ls=":", alpha=0.35)

    # Bennu at Hill-frame origin (0,0,0), shown large for visibility
    xs, ys, zs = _sphere_mesh(0, 0, 0, r=0.35, nu=22, nv=14)
    ax3.plot_surface(xs, ys, zs, color="#5a4020", alpha=0.88,
                     linewidth=0, antialiased=True)
    xs2, ys2, zs2 = _sphere_mesh(0, 0, 0, r=0.40, nu=16, nv=10)
    ax3.plot_surface(xs2, ys2, zs2, color="#8a6030", alpha=0.12,
                     linewidth=0, antialiased=False)
    ax3.text(0, 0, 0.46, "Bennu", color="#cc9955",
             fontsize=7, ha="center", va="bottom")

    # LOS line: spacecraft -> Bennu
    ax3.plot([sc_pos[0], 0], [sc_pos[1], 0], [sc_pos[2], 0],
             color="#443322", lw=0.5, ls="--", alpha=0.45)

    # Bus (translated to current spacecraft position)
    bv = (R @ BUS_V.T).T + sc_pos
    face_verts = [[bv[i] for i in f] for f in BUS_FACES]
    poly = Poly3DCollection(face_verts, zsort="average",
                            facecolors=BUS_RGBA,
                            edgecolors=BUS_EC, linewidths=0.5)
    ax3.add_collection3d(poly)

    # Solar panel wings (translated to sc_pos)
    for raw_panel in (PANEL_P, PANEL_N):
        pv = (R @ raw_panel.T).T + sc_pos
        panel_col = _rgba("#204080", 0.82)
        ppoly = Poly3DCollection([pv], zsort="average",
                                 facecolors=[panel_col],
                                 edgecolors="#4488cc", linewidths=0.7)
        ax3.add_collection3d(ppoly)
        # Solar cell grid
        for frac in np.linspace(0.15, 0.85, 5):
            p0 = pv[0] + frac * (pv[1] - pv[0])
            p1 = pv[3] + frac * (pv[2] - pv[3])
            ax3.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]],
                     color="#2255aa", lw=0.4, alpha=0.55)
            q0 = pv[0] + frac * (pv[3] - pv[0])
            q1 = pv[1] + frac * (pv[2] - pv[1])
            ax3.plot([q0[0],q1[0]], [q0[1],q1[1]], [q0[2],q1[2]],
                     color="#2255aa", lw=0.4, alpha=0.35)

    # HGA dish (translated to sc_pos)
    hga_r   = (R @ HGA_VERTS.T).T + sc_pos
    hga_cen = R @ np.array([HGA_X, 0.0, 0.0]) + sc_pos
    hga_polys = []
    for i in range(len(hga_r)):
        j = (i + 1) % len(hga_r)
        hga_polys.append([hga_cen, hga_r[i], hga_r[j]])
    hga_col = _rgba("#c0c0c0", 0.65)
    hga_poly = Poly3DCollection(hga_polys, zsort="average",
                                facecolors=[hga_col] * len(hga_polys),
                                edgecolors="#909090", linewidths=0.3)
    ax3.add_collection3d(hga_poly)

    # Camera head (translated to sc_pos)
    cv = (R @ CAM_V.T).T + sc_pos
    cam_face_verts = [[cv[i] for i in f] for f in CAM_FACES]
    cam_col = _rgba("#cc5500", 0.95)
    cpoly = Poly3DCollection(cam_face_verts, zsort="average",
                             facecolors=[cam_col] * 6,
                             edgecolors="#aa4400", linewidths=0.4)
    ax3.add_collection3d(cpoly)

    # Body-frame axes (from sc_pos)
    for j, (col, lbl) in enumerate(
            [("red", "xB"), ("limegreen", "yB"), ("deepskyblue", "zB")]):
        tip = R[:, j] * AXIS
        ax3.quiver(sc_pos[0], sc_pos[1], sc_pos[2],
                   tip[0], tip[1], tip[2],
                   color=col, lw=1.8, arrow_length_ratio=0.28)
        ax3.text(sc_pos[0] + tip[0]*1.12,
                 sc_pos[1] + tip[1]*1.12,
                 sc_pos[2] + tip[2]*1.12,
                 lbl, color=col, fontsize=7)

    # RCS nozzles (translated to sc_pos)
    fire  = FIRE[fi]
    pos_r = (R @ T_POS.T).T + sc_pos
    dir_r = (R @ T_DIR.T).T

    for ti in range(12):
        cone_polys_world = [
            [(R @ v) + sc_pos for v in tri] for tri in _CONES_BODY[ti]
        ]
        if fire[ti]:
            col = T_COL_ON[ti]
            noz = Poly3DCollection(cone_polys_world, zsort="average",
                                   facecolors=[col]*len(cone_polys_world),
                                   edgecolors=col, linewidths=0, alpha=0.95)
            ax3.add_collection3d(noz)
            tip_w = pos_r[ti] + dir_r[ti] * NOZZLE_OFF
            px, py, pz = _plume_mesh(tip_w, dir_r[ti], True)
            ax3.plot_surface(px, py, pz, color=col,
                             alpha=0.35, linewidth=0, antialiased=False)
        else:
            noz = Poly3DCollection(cone_polys_world, zsort="average",
                                   facecolors=[T_COL_OFF]*len(cone_polys_world),
                                   edgecolors=T_COL_OFF, linewidths=0, alpha=0.60)
            ax3.add_collection3d(noz)

    # Overlay text
    t_min  = t_anim[fi] / 60.0
    n_fire = int(fire.sum())
    r_km   = np.linalg.norm(traj_xyz[min(int(np.searchsorted(t_tr, t_anim[fi])),
                                          len(traj_xyz) - 1)]) / 1000.0
    ax3.set_title(f"T + {t_min:5.1f} min   |   range: {r_km:.2f} km   |   "
                  f"{n_fire}/12 nozzles firing",
                  color="white", fontsize=9, pad=3)
    ax3.text2D(0.01, 0.98,
               (f"wx={omega_x[fi]:+7.3f}  "
                f"wy={omega_y[fi]:+7.3f}  "
                f"wz={omega_z[fi]:+7.3f} mrad/s"),
               transform=ax3.transAxes,
               color="#aaccff", fontsize=8, va="top", family="monospace")
    ax3.legend(handles=_leg, loc="lower left", fontsize=6.0,
               facecolor="#111130", edgecolor="#303060",
               labelcolor="white", framealpha=0.85, ncol=1)

    t_cur = t_h[fi]
    for cur in cursors:
        cur.set_xdata([t_cur, t_cur])


# -- Render -------------------------------------------------------------------
print(f"Building animation: {n_frames} frames @ {FPS} fps ...")
anim = FuncAnimation(fig, draw, frames=n_frames,
                     interval=1000 // FPS, blit=False)

out_mp4 = ROOT / "out/rcs_animation.mp4"
out_gif  = ROOT / "out/rcs_animation.gif"

if shutil.which("ffmpeg"):
    print("Rendering MP4 (a few minutes) ...")
    anim.save(str(out_mp4), writer="ffmpeg", fps=FPS, dpi=DPI,
              savefig_kwargs={"facecolor": BG})
    print(f"Saved  {out_mp4}")
else:
    print("ffmpeg not found -- trying Pillow GIF ...")
    try:
        anim.save(str(out_gif), writer="pillow", fps=FPS, dpi=90)
        print(f"Saved  {out_gif}")
    except Exception as e:
        print(f"GIF failed ({e}) -- showing interactively.")
        plt.show()
