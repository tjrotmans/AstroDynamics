"""
Spacecraft body-frame geometry visualisation.

Shows the spacecraft in its body frame with:
  - Main bus (box)
  - Solar panels (flat plates, ±y axis)
  - RCS attitude thrusters (12 nozzles)
  - RCS translation thrusters (6 nozzles)
  - Reaction wheel spin-axis directions
  - Centre-of-Mass (CoM)
  - Centre-of-Pressure (CoP) for SRP
  - Inertia principal axes (arrows proportional to I_i)
  - Body-axis triad

All geometry matches config.rs constants so the figure can be used as a
physical-model sanity check.

Run:  python plot/sc_geometry.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# ── Constants (mirror config.rs) ──────────────────────────────────────────────

SC_BUS_DIMS     = np.array([2.0, 2.0, 0.63])   # (x, y, z) full dimensions [m]
SC_PANEL_SPAN   = 2.5        # panel span per side (along ±y) [m]
SC_PANEL_CHORD  = 0.8        # panel chord (along x) [m]

COM_OFFSET      = np.array([0.02, 0.0, -0.04])  # CoM from geometric centre [m]
COP_OFFSET      = np.array([0.05, 0.0,  0.22])  # CoP from CoM [m]

RCS_MOMENT_ARM  = 1.0        # m
INERTIA_DIAG    = np.array([366.67, 366.67, 666.67])   # kg·m²

# Reaction wheel axes (4-wheel pyramid, β = arctan(1/√2))
_S, _C = 1/np.sqrt(3), np.sqrt(2/3)
WHEEL_AXES = np.array([
    [ _S, 0,  _C],
    [0,   _S, _C],
    [-_S, 0,  _C],
    [0,  -_S, _C],
])

# ── Helpers ───────────────────────────────────────────────────────────────────

def box_faces(centre, half_extents):
    """Return list of (4,3) vertex arrays for the 6 faces of a box."""
    cx, cy, cz = centre
    hx, hy, hz = half_extents
    verts = np.array([
        [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
        [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
        [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
        [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz],
    ])
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],  # -z
        [verts[4], verts[5], verts[6], verts[7]],  # +z
        [verts[0], verts[1], verts[5], verts[4]],  # -y
        [verts[2], verts[3], verts[7], verts[6]],  # +y
        [verts[0], verts[3], verts[7], verts[4]],  # -x
        [verts[1], verts[2], verts[6], verts[5]],  # +x
    ]
    return faces

def cone(tip, direction, length=0.12, radius=0.04, n=12):
    """Return (verts, faces) for a simple cone (nozzle)."""
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d)
    # Orthogonal basis
    if abs(d[0]) < 0.9:
        u = np.cross(d, [1,0,0]); u /= np.linalg.norm(u)
    else:
        u = np.cross(d, [0,1,0]); u /= np.linalg.norm(u)
    w = np.cross(d, u)
    base = tip + length * d
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    ring = [base + radius * (np.cos(a)*u + np.sin(a)*w) for a in angles]
    faces = [[tip] + [ring[i], ring[(i+1)%n]] for i in range(n)]
    return faces

def arrow3d(ax, origin, direction, scale, color, lw=2, head_frac=0.25, label=None):
    """Draw a 3D arrow using quiver."""
    ax.quiver(*origin, *direction, length=scale, color=color, linewidth=lw,
              arrow_length_ratio=head_frac, label=label)

# ── Build figure ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 11), facecolor="#0d0d0d")
ax  = fig.add_subplot(111, projection="3d", facecolor="#0d0d0d")

geom_centre = np.array([0.0, 0.0, 0.0])  # geometric centre in body frame
com         = geom_centre + COM_OFFSET    # CoM = where the body frame ORIGIN is
cop         = com + COP_OFFSET

# ── Main bus ──────────────────────────────────────────────────────────────────

bus_faces = box_faces(geom_centre, SC_BUS_DIMS / 2)
bus_poly  = Poly3DCollection(bus_faces, alpha=0.15, linewidth=0.6,
                              edgecolor="#4488cc", facecolor="#1a2a3a")
ax.add_collection3d(bus_poly)

# ── Solar panels ─────────────────────────────────────────────────────────────

for side in [-1, 1]:
    # Panel starts at bus edge (±y = SC_BUS_DIMS[1]/2) and extends outward
    panel_y_inner = side * SC_BUS_DIMS[1] / 2
    panel_y_outer = panel_y_inner + side * SC_PANEL_SPAN
    panel_z       = geom_centre[2] + SC_BUS_DIMS[2] / 2  # flush with top (+z) face
    px            = geom_centre[0]
    panel_verts   = [
        [px - SC_PANEL_CHORD/2, panel_y_inner, panel_z],
        [px + SC_PANEL_CHORD/2, panel_y_inner, panel_z],
        [px + SC_PANEL_CHORD/2, panel_y_outer, panel_z],
        [px - SC_PANEL_CHORD/2, panel_y_outer, panel_z],
    ]
    panel_poly = Poly3DCollection([panel_verts], alpha=0.5, linewidth=0.8,
                                   edgecolor="#88aaff", facecolor="#1a3a6a")
    ax.add_collection3d(panel_poly)
    # Solar cell grid lines
    ny = 4
    for k in range(ny+1):
        frac = k / ny
        y_line = panel_y_inner + frac * (panel_y_outer - panel_y_inner)
        ax.plot([px-SC_PANEL_CHORD/2, px+SC_PANEL_CHORD/2], [y_line, y_line],
                [panel_z, panel_z], color="#3355aa", lw=0.4, alpha=0.5)

# ── RCS attitude thrusters (12 nozzles = 6 couples) ──────────────────────────

a = RCS_MOMENT_ARM
rcs_att_config = [
    # (thrust direction, moment-arm position)  — same as rcs.rs
    ([ 0, 0, 1], [0, a, 0]),   # +x torque couple
    ([ 0, 0,-1], [0,-a, 0]),
    ([ 0, 0, 1], [0,-a, 0]),   # -x torque couple
    ([ 0, 0,-1], [0, a, 0]),
    ([ 1, 0, 0], [0, 0, a]),   # +y torque couple
    ([-1, 0, 0], [0, 0,-a]),
    ([ 1, 0, 0], [0, 0,-a]),   # -y torque couple
    ([-1, 0, 0], [0, 0, a]),
    ([ 0, 1, 0], [a, 0, 0]),   # +z torque couple
    ([ 0,-1, 0], [-a,0, 0]),
    ([ 0, 1, 0], [-a,0, 0]),   # -z torque couple
    ([ 0,-1, 0], [a, 0, 0]),
]

for i, (thrust_dir, pos) in enumerate(rcs_att_config):
    tip = np.array(pos, dtype=float) + com
    for f in cone(tip, thrust_dir, length=0.10, radius=0.03):
        poly = Poly3DCollection([f], alpha=0.7, facecolor="#cc3300", edgecolor="#ff5500")
        ax.add_collection3d(poly)

# ── RCS translation thrusters (6 nozzles at CoM) ─────────────────────────────

trans_dirs = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
# Place on bus surface along the thrust direction from CoM
for d in trans_dirs:
    d = np.array(d, dtype=float)
    # surface offset: half of the bus dim along that axis
    surface_point = com + d * 0.5 * abs(np.dot(SC_BUS_DIMS, abs(d))) / 2
    for f in cone(surface_point, d, length=0.09, radius=0.025):
        poly = Poly3DCollection([f], alpha=0.8, facecolor="#ff9900", edgecolor="#ffcc00")
        ax.add_collection3d(poly)

# ── Reaction wheels (spin-axis arrows, positioned inside bus) ─────────────────

wheel_colours = ["#00ffaa", "#00ccff", "#aa88ff", "#ffaa00"]
for i, (axis, col) in enumerate(zip(WHEEL_AXES, wheel_colours)):
    # Place wheel at a small offset from CoM inside the bus
    offset = 0.2 * np.array(axis)
    origin = com + offset * np.array([-1,-1,0.5])  # stagger for visibility
    arrow3d(ax, origin, axis, scale=0.4, color=col, lw=2.5,
            label=f"Wheel {i+1} axis" if i == 0 else None)

# ── CoM and CoP markers ───────────────────────────────────────────────────────

ax.scatter(*com, color="#ff0000", s=120, zorder=10, label="CoM")
ax.scatter(*cop, color="#ffff00", s=100, marker="^", zorder=10, label="CoP (SRP)")
ax.scatter(*geom_centre, color="#aaaaaa", s=60, marker="+", zorder=10,
           label="Geometric centre")

# CoM → CoP lever arm
ax.plot([com[0], cop[0]], [com[1], cop[1]], [com[2], cop[2]],
        color="#ffff00", lw=1.2, linestyle=":", alpha=0.6)

# ── Principal inertia axes ────────────────────────────────────────────────────

I_scale = 0.8 / INERTIA_DIAG.max()   # normalise longest axis to 0.8 m
axis_colours = ["#ff4444", "#44ff44", "#4444ff"]
axis_labels  = [f"$\\hat{{x}}_b$  (I={INERTIA_DIAG[0]:.0f} kg·m²)",
                f"$\\hat{{y}}_b$  (I={INERTIA_DIAG[1]:.0f} kg·m²)",
                f"$\\hat{{z}}_b$  (I={INERTIA_DIAG[2]:.0f} kg·m²)"]
eye = np.eye(3)
for k in range(3):
    arrow3d(ax, com, eye[k], scale=INERTIA_DIAG[k]*I_scale,
            color=axis_colours[k], lw=3, head_frac=0.15, label=axis_labels[k])

# ── Inertia ellipsoid (wireframe) ─────────────────────────────────────────────

u = np.linspace(0, 2*np.pi, 24)
v = np.linspace(0,   np.pi, 16)
ell_scale = 0.5
a_ell = ell_scale * np.sqrt(1.0 / INERTIA_DIAG[0])
b_ell = ell_scale * np.sqrt(1.0 / INERTIA_DIAG[1])
c_ell = ell_scale * np.sqrt(1.0 / INERTIA_DIAG[2])
xe = a_ell * np.outer(np.cos(u), np.sin(v)) + com[0]
ye = b_ell * np.outer(np.sin(u), np.sin(v)) + com[1]
ze = c_ell * np.outer(np.ones(len(u)), np.cos(v)) + com[2]
ax.plot_wireframe(xe, ye, ze, color="#555555", linewidth=0.4, alpha=0.35,
                  label="Inertia ellipsoid (1/√I)")

# ── Annotations ──────────────────────────────────────────────────────────────

# Camera (boresight) on +x face
cam_tip = com + np.array([SC_BUS_DIMS[0]/2 + 0.05, 0, 0])
ax.scatter(*cam_tip, color="#00ffff", s=80, marker="s", zorder=10, label="Camera (+x)")

# HGA dish on -x face
hga_tip = com + np.array([-SC_BUS_DIMS[0]/2 - 0.05, 0, 0])
ax.scatter(*hga_tip, color="#cc88ff", s=80, marker="D", zorder=10, label="HGA (−x)")

# Text annotations
def lbl(ax, pt, txt, col="#cccccc", fs=7.5, offset=(0.05, 0.05, 0.05)):
    ax.text(pt[0]+offset[0], pt[1]+offset[1], pt[2]+offset[2], txt,
            color=col, fontsize=fs, ha="left", va="bottom")

lbl(ax, com, f"CoM\n({COM_OFFSET[0]:+.2f}, {COM_OFFSET[1]:+.2f}, {COM_OFFSET[2]:+.2f}) m\nfrom geom. centre",
    col="#ff7777", fs=7)
lbl(ax, cop, f"CoP\n+{COP_OFFSET[2]:.2f} m  z from CoM", col="#ffffaa", fs=7,
    offset=(0.05, 0.05, 0.08))

# Dimension annotation arrows on bus
# x-dimension
zfloor = geom_centre[2] - SC_BUS_DIMS[2]/2 - 0.2
ax.annotate("", xy=(geom_centre[0]+SC_BUS_DIMS[0]/2, geom_centre[1],),
            xytext=(geom_centre[0]-SC_BUS_DIMS[0]/2, geom_centre[1]))

# ── Formatting ────────────────────────────────────────────────────────────────

lim = 2.2
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
ax.set_xlabel("x_body  [m]", color="#aaaaaa", labelpad=6)
ax.set_ylabel("y_body  [m]", color="#aaaaaa", labelpad=6)
ax.set_zlabel("z_body  [m]", color="#aaaaaa", labelpad=6)
ax.tick_params(colors="#888888", labelsize=7)
for spine in ax.spines.values():
    spine.set_color("#444444")

ax.set_title(
    f"Spacecraft body-frame geometry\n"
    f"Bus: {SC_BUS_DIMS[0]:.2f} × {SC_BUS_DIMS[1]:.2f} × {SC_BUS_DIMS[2]:.2f} m    "
    f"Panels: {SC_PANEL_SPAN:.1f} m span × {SC_PANEL_CHORD:.1f} m chord    "
    f"Mass: 1000 kg\n"
    f"I_x={INERTIA_DIAG[0]:.1f}  I_y={INERTIA_DIAG[1]:.1f}  I_z={INERTIA_DIAG[2]:.1f} kg·m²    "
    f"CoM offset: ({COM_OFFSET[0]:+.2f}, {COM_OFFSET[1]:+.2f}, {COM_OFFSET[2]:+.2f}) m",
    color="#cccccc", fontsize=9, pad=14
)

# Legend — use handles for coloured patches
handles = [
    mpatches.Patch(fc="#1a2a3a", ec="#4488cc", label=f"Main bus  ({SC_BUS_DIMS[0]:.2f}×{SC_BUS_DIMS[1]:.2f}×{SC_BUS_DIMS[2]:.2f} m)"),
    mpatches.Patch(fc="#1a3a6a", ec="#88aaff", label=f"Solar panels  (×2, span {SC_PANEL_SPAN:.1f} m)"),
    mpatches.Patch(fc="#cc3300", ec="#ff5500", label="Attitude RCS nozzles (×12)"),
    mpatches.Patch(fc="#ff9900", ec="#ffcc00", label="Translation RCS nozzles (×6)"),
    plt.Line2D([0],[0], color="#ff0000",  lw=2, marker="o", label="CoM  (body-frame origin)"),
    plt.Line2D([0],[0], color="#ffff00",  lw=0, marker="^", ms=7, label="CoP  (SRP pressure centre)"),
    plt.Line2D([0],[0], color="#aaaaaa",  lw=0, marker="+", ms=7, label="Geometric centre"),
    plt.Line2D([0],[0], color="#ff4444",  lw=2, label="x_body axis (I_x)"),
    plt.Line2D([0],[0], color="#44ff44",  lw=2, label="y_body axis (I_y)"),
    plt.Line2D([0],[0], color="#4444ff",  lw=2, label="z_body axis (I_z)"),
    plt.Line2D([0],[0], color="#555555",  lw=1, linestyle="-", label="Inertia ellipsoid (1/√I)"),
    plt.Line2D([0],[0], color="#00ffaa",  lw=2, label="Reaction wheel axes"),
    plt.Line2D([0],[0], color="#00ffff",  lw=0, marker="s", ms=7, label="Camera boresight (+x face)"),
    plt.Line2D([0],[0], color="#cc88ff",  lw=0, marker="D", ms=7, label="HGA dish (−x face)"),
]
legend = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
                   fontsize=7, framealpha=0.3, facecolor="#111111",
                   edgecolor="#333333", labelcolor="#cccccc", ncol=2)

ax.view_init(elev=22, azim=-55)
ax.set_box_aspect([1,1,1])

plt.tight_layout(pad=0.5)
os.makedirs("out", exist_ok=True)
out_path = "out/sc_geometry.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved {out_path}")

# ── Also show inertia consistency check ──────────────────────────────────────

def box_inertia(mass, dims):
    """Solid box moments for full dimensions (lx, ly, lz)."""
    lx, ly, lz = dims
    ix = mass/12 * (ly**2 + lz**2)
    iy = mass/12 * (lx**2 + lz**2)
    iz = mass/12 * (lx**2 + ly**2)
    return np.array([ix, iy, iz])

mass = 1000.0
I_box = box_inertia(mass, SC_BUS_DIMS)
print("\n--- Inertia consistency check ---")
print(f"  Config  I_x, I_y, I_z = {INERTIA_DIAG}")
print(f"  Solid bus only I_x, I_y, I_z = {I_box.round(2)}")

# Add panel contribution (thin plate of mass ~ 5% total per panel)
panel_mass = 0.05 * mass  # 50 kg per panel
panel_y_c  = SC_BUS_DIMS[1]/2 + SC_PANEL_SPAN/2  # panel CoM from geom centre
I_panel_y  = panel_mass * SC_PANEL_SPAN**2 / 12   # about panel's own centre
I_panel_y += panel_mass * panel_y_c**2            # parallel axis
I_panels_x = 2 * I_panel_y   # two panels, symmetric → contributes to Ix and Iz
print(f"  Bus + panel (2x{panel_mass:.0f} kg, span {SC_PANEL_SPAN:.1f} m) contribution:")
print(f"    dI_x ~ {I_panels_x:.1f} kg*m^2  (from parallel axis)")
print(f"  Updated bus+panel I_x ~ {I_box[0]+I_panels_x:.1f}  (config = {INERTIA_DIAG[0]:.1f})")
print("  -> Discrepancy: config values include component placement effects.")
print("     Update SC_INERTIA_DIAG in config.rs if needed after CAD review.")
