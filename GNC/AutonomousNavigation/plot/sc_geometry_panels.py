"""
Spacecraft geometry visualisation — SRP plate centres and CoM.

Shows three orthographic views with every plate centre marked,
CoM position, and numerical plate-centre table (all in body frame,
relative to CoM).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Spacecraft parameters (must match config.rs) ─────────────────────────────
LX, LY, LZ          = 2.0, 2.0, 0.63      # bus dimensions [m]
PANEL_SPAN           = 2.5                  # each panel along ±y [m]
PANEL_CHORD          = 0.8                  # along x [m]
COM_OFFSET           = np.array([0.02, 0.0, -0.04])  # CoM from geometric centre [m]

# ── Plate centres relative to geometric centre ────────────────────────────────
_geom = {
    '+x bus': np.array([ LX/2,                    0,           0        ]),
    '-x bus': np.array([-LX/2,                    0,           0        ]),
    '+y bus': np.array([   0,                  LY/2,           0        ]),
    '-y bus': np.array([   0,                 -LY/2,           0        ]),
    '+z bus': np.array([   0,                    0,           LZ/2      ]),
    '-z bus': np.array([   0,                    0,          -LZ/2      ]),
    'Panel+y': np.array([  0,  LY/2 + PANEL_SPAN/2,           0        ]),
    'Panel-y': np.array([  0, -LY/2 - PANEL_SPAN/2,           0        ]),
}
# Shift from geometric centre to CoM
CENTERS = {k: v - COM_OFFSET for k, v in _geom.items()}

# ── Colours ───────────────────────────────────────────────────────────────────
BG      = '#0f0f1a'
PANEL_BG= '#16213e'
BUS_C   = '#4a90d9'
SOL_C   = '#2ecc71'
COM_C   = '#e74c3c'
CTR_C   = '#f39c12'
AX_C    = '#ecf0f1'
DIM_C   = '#95a5a6'
TXT_C   = '#ecf0f1'
GRID_C  = '#1e1e3a'

def draw_arrow(ax, origin, dx, dy, label, color=AX_C):
    ax.annotate('', xy=(origin[0]+dx, origin[1]+dy), xytext=origin,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    ax.text(origin[0]+dx*1.15, origin[1]+dy*1.15, label,
            color=color, fontsize=9, ha='center', va='center')

def dim_h(ax, x0, x1, y, label, color=DIM_C):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='<->', color=color, lw=0.9))
    ax.text((x0+x1)/2, y+0.08, label, ha='center', color=color, fontsize=7.5)

def dim_v(ax, x, y0, y1, label, color=DIM_C):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='<->', color=color, lw=0.9))
    ax.text(x+0.08, (y0+y1)/2, label, ha='left', va='center', color=color, fontsize=7.5)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor=BG)
gs  = fig.add_gridspec(1, 3, wspace=0.35, left=0.04, right=0.98,
                        top=0.88, bottom=0.22)
axes = [fig.add_subplot(gs[i], facecolor=PANEL_BG) for i in range(3)]

plt.suptitle('Spacecraft Geometry — SRP Plate Centres & CoM  (body frame)',
             color='white', fontsize=13, fontweight='bold', y=0.96)

# =============================================================================
# VIEW 1 — Top view: x–y plane, looking from +z downward
# =============================================================================
ax = axes[0]
ax.set_facecolor(PANEL_BG)
ax.set_title('Top view  (−z̃, x→, y↑)', color=TXT_C, fontsize=10, pad=8)

# Bus (centred on geometric origin)
bus = mpatches.Rectangle((-LX/2, -LY/2), LX, LY,
                          lw=1.5, ec=BUS_C, fc=BUS_C+'28', zorder=3)
ax.add_patch(bus)

# Solar panels
for sign, label in [(1, 'Panel+y'), (-1, 'Panel-y')]:
    py0 = sign * LY/2
    py1 = sign * (LY/2 + PANEL_SPAN)
    ylo, yhi = min(py0, py1), max(py0, py1)
    pan = mpatches.Rectangle((-PANEL_CHORD/2, ylo), PANEL_CHORD, PANEL_SPAN,
                              lw=1.5, ec=SOL_C, fc=SOL_C+'28', zorder=3)
    ax.add_patch(pan)

# Plate centres — in x–y (z collapses)
labels_xy = {
    '+x bus':  ( 0.10,  0.08),
    '-x bus':  (-0.42,  0.08),
    '+y bus':  ( 0.08,  0.10),
    '-y bus':  ( 0.08, -0.18),
    'Panel+y': ( 0.10,  0.05),
    'Panel-y': ( 0.10, -0.12),
}
for name, off in labels_xy.items():
    c = CENTERS[name]
    ax.plot(c[0], c[1], 'o', color=CTR_C, ms=7, zorder=6)
    ax.text(c[0]+off[0], c[1]+off[1], name, color=CTR_C, fontsize=7.5, zorder=7)

# ±z faces both project to (0,0) in x–y — mark with a different symbol
for name in ['+z bus', '-z bus']:
    c = CENTERS[name]
    ax.plot(c[0], c[1], 'D', color=CTR_C, ms=6, zorder=6, alpha=0.6)
ax.text(0.08, 0.18, '±z bus\n(both ≈ origin\nin x–y)', color=CTR_C, fontsize=6.5, alpha=0.8)

# CoM
ax.plot(COM_OFFSET[0], COM_OFFSET[1], '*', color=COM_C, ms=16, zorder=9)
ax.text(COM_OFFSET[0]+0.1, COM_OFFSET[1]-0.25,
        f'CoM\n({COM_OFFSET[0]:+.2f}, {COM_OFFSET[1]:+.2f})',
        color=COM_C, fontsize=8)

# Geometric centre
ax.plot(0, 0, '+', color='white', ms=11, mew=2, zorder=8)
ax.text(0.1, -0.18, 'Geom ctr', color='white', fontsize=7)

# Dimensions
dim_h(ax, -LX/2, LX/2,  -LY/2-0.35, f'lx = {LX} m')
dim_v(ax, LX/2+0.35, -LY/2, LY/2, f'ly = {LY} m')
dim_v(ax, PANEL_CHORD/2+0.25, LY/2, LY/2+PANEL_SPAN,
      f'span = {PANEL_SPAN} m', color=SOL_C)
dim_h(ax, -PANEL_CHORD/2, PANEL_CHORD/2, LY/2+PANEL_SPAN+0.2,
      f'chord = {PANEL_CHORD} m', color=SOL_C)

draw_arrow(ax, (-4.2, -4.0), 0.7, 0, '+x')
draw_arrow(ax, (-4.2, -4.0), 0, 0.7, '+y')

ax.set_xlim(-4.6, 4.6); ax.set_ylim(-5.8, 5.8)
ax.set_aspect('equal')
ax.grid(True, color=GRID_C, lw=0.3, zorder=0)
ax.set_xlabel('x  [m]', color=TXT_C, fontsize=8)
ax.set_ylabel('y  [m]', color=TXT_C, fontsize=8)
for sp in ax.spines.values(): sp.set_edgecolor(GRID_C)
ax.tick_params(colors=TXT_C, labelsize=7)

# =============================================================================
# VIEW 2 — Front view: y–z plane, looking from +x (camera boresight direction)
# =============================================================================
ax = axes[1]
ax.set_title('Front view  (+x̃, y→, z↑)', color=TXT_C, fontsize=10, pad=8)

# Bus face in y–z
bus_yz = mpatches.Rectangle((-LY/2, -LZ/2), LY, LZ,
                              lw=1.5, ec=BUS_C, fc=BUS_C+'28', zorder=3)
ax.add_patch(bus_yz)

# Solar panels — appear as thin strips at z = 0, extending in y
for sign in [1, -1]:
    y0 = sign * LY/2
    y1 = sign * (LY/2 + PANEL_SPAN)
    ylo, yhi = min(y0, y1), max(y0, y1)
    pan_yz = mpatches.Rectangle((ylo, -0.03), PANEL_SPAN, 0.06,
                                  lw=1.5, ec=SOL_C, fc=SOL_C+'40', zorder=4)
    ax.add_patch(pan_yz)

# Line showing panel plane
ax.plot([LY/2, LY/2+PANEL_SPAN], [0, 0], color=SOL_C, lw=2.5, zorder=5)
ax.plot([-LY/2, -(LY/2+PANEL_SPAN)], [0, 0], color=SOL_C, lw=2.5, zorder=5)
ax.text(LY/2+PANEL_SPAN/2, 0.07, 'Panel+y', color=SOL_C, fontsize=7.5, ha='center')
ax.text(-(LY/2+PANEL_SPAN/2), 0.07, 'Panel-y', color=SOL_C, fontsize=7.5, ha='center')

# Plate centres in y–z
labels_yz = {
    '+y bus':  ( 0.08, 0.04),
    '-y bus':  (-0.45, 0.04),
    '+z bus':  ( 0.08, 0.03),
    '-z bus':  ( 0.08,-0.09),
    'Panel+y': ( 0.08, 0.06),
    'Panel-y': (-0.55, 0.06),
}
for name, off in labels_yz.items():
    c = CENTERS[name]
    ax.plot(c[1], c[2], 'o', color=CTR_C, ms=7, zorder=6)
    ax.text(c[1]+off[0], c[2]+off[1], name, color=CTR_C, fontsize=7.5, zorder=7)

# ±x faces project to (0,0) in y–z
ax.plot(0, 0, 'D', color=CTR_C, ms=6, zorder=6, alpha=0.6)
ax.text(0.08, 0.04, '±x bus\n(≈ origin y–z)', color=CTR_C, fontsize=6.5, alpha=0.8)

# CoM
ax.plot(COM_OFFSET[1], COM_OFFSET[2], '*', color=COM_C, ms=16, zorder=9)
ax.text(COM_OFFSET[1]+0.08, COM_OFFSET[2]-0.07,
        f'CoM ({COM_OFFSET[1]:+.2f}, {COM_OFFSET[2]:+.2f})',
        color=COM_C, fontsize=8)

ax.plot(0, 0, '+', color='white', ms=11, mew=2, zorder=8)
ax.text(0.06, -0.1, 'Geom ctr', color='white', fontsize=7)

dim_h(ax, -LY/2, LY/2, -LZ/2-0.15, f'ly = {LY} m')
dim_v(ax, LY/2+0.25, -LZ/2, LZ/2, f'lz = {LZ} m')
dim_v(ax, LY/2+PANEL_SPAN+0.15, LY/2, LY/2+PANEL_SPAN,
      f'{PANEL_SPAN} m', color=SOL_C)

draw_arrow(ax, (-4.0, -0.65), 0.6, 0, '+y')
draw_arrow(ax, (-4.0, -0.65), 0, 0.35, '+z')

ax.set_xlim(-4.5, 4.5); ax.set_ylim(-0.75, 0.85)
ax.set_aspect('equal')
ax.grid(True, color=GRID_C, lw=0.3, zorder=0)
ax.set_xlabel('y  [m]', color=TXT_C, fontsize=8)
ax.set_ylabel('z  [m]', color=TXT_C, fontsize=8)
for sp in ax.spines.values(): sp.set_edgecolor(GRID_C)
ax.tick_params(colors=TXT_C, labelsize=7)

# =============================================================================
# VIEW 3 — Side view: x–z plane, looking from +y
# =============================================================================
ax = axes[2]
ax.set_title('Side view  (+ỹ, x→, z↑)', color=TXT_C, fontsize=10, pad=8)

# Bus in x–z
bus_xz = mpatches.Rectangle((-LX/2, -LZ/2), LX, LZ,
                              lw=1.5, ec=BUS_C, fc=BUS_C+'28', zorder=3)
ax.add_patch(bus_xz)

# Solar panels appear edge-on as a line at z = 0, x in [-chord/2, chord/2]
ax.plot([-PANEL_CHORD/2, PANEL_CHORD/2], [0, 0],
        color=SOL_C, lw=3, zorder=5, label='Panel (edge-on, ±y)')
ax.text(PANEL_CHORD/2+0.1, 0.04, 'Panels\n(edge-on)', color=SOL_C, fontsize=7.5)

# Plate centres in x–z
labels_xz = {
    '+x bus':  ( 0.08,  0.04),
    '-x bus':  (-0.45,  0.04),
    '+z bus':  ( 0.08,  0.03),
    '-z bus':  ( 0.08, -0.08),
    'Panel+y': ( 0.08,  0.05),
    'Panel-y': ( 0.08, -0.11),
}
for name, off in labels_xz.items():
    c = CENTERS[name]
    ax.plot(c[0], c[2], 'o', color=CTR_C, ms=7, zorder=6)
    ax.text(c[0]+off[0], c[2]+off[1], name, color=CTR_C, fontsize=7.5, zorder=7)

# ±y faces project to (0,0) in x–z
ax.plot(0, 0, 'D', color=CTR_C, ms=6, zorder=6, alpha=0.6)
ax.text(0.08, -0.12, '±y bus\n(≈ origin x–z)', color=CTR_C, fontsize=6.5, alpha=0.8)

# CoM
ax.plot(COM_OFFSET[0], COM_OFFSET[2], '*', color=COM_C, ms=16, zorder=9)
ax.text(COM_OFFSET[0]+0.08, COM_OFFSET[2]-0.07,
        f'CoM ({COM_OFFSET[0]:+.2f}, {COM_OFFSET[2]:+.2f})',
        color=COM_C, fontsize=8)

ax.plot(0, 0, '+', color='white', ms=11, mew=2, zorder=8)
ax.text(0.08, -0.09, 'Geom ctr', color='white', fontsize=7)

dim_h(ax, -LX/2, LX/2, -LZ/2-0.15, f'lx = {LX} m')
dim_v(ax, LX/2+0.22, -LZ/2, LZ/2, f'lz = {LZ} m')
dim_h(ax, -PANEL_CHORD/2, PANEL_CHORD/2, LZ/2+0.1,
      f'chord={PANEL_CHORD}m', color=SOL_C)

draw_arrow(ax, (-1.6, -0.62), 0.5, 0, '+x')
draw_arrow(ax, (-1.6, -0.62), 0, 0.35, '+z')

ax.set_xlim(-1.9, 2.2); ax.set_ylim(-0.75, 0.75)
ax.set_aspect('equal')
ax.grid(True, color=GRID_C, lw=0.3, zorder=0)
ax.set_xlabel('x  [m]', color=TXT_C, fontsize=8)
ax.set_ylabel('z  [m]', color=TXT_C, fontsize=8)
for sp in ax.spines.values(): sp.set_edgecolor(GRID_C)
ax.tick_params(colors=TXT_C, labelsize=7)

# =============================================================================
# Plate centre table (body frame, relative to CoM)
# =============================================================================
header = f"{'Plate':<10}  {'center_body [x, y, z] m':^36}  normal"
rows = [
    ('+x bus',  CENTERS['+x bus'],  '+x'),
    ('-x bus',  CENTERS['-x bus'],  '-x'),
    ('+y bus',  CENTERS['+y bus'],  '+y'),
    ('-y bus',  CENTERS['-y bus'],  '-y'),
    ('+z bus',  CENTERS['+z bus'],  '+z'),
    ('-z bus',  CENTERS['-z bus'],  '-z'),
    ('Panel+y', CENTERS['Panel+y'], '±z  (double-sided)'),
    ('Panel-y', CENTERS['Panel-y'], '±z  (double-sided)'),
]
table_lines = [header, '─' * 68]
for name, c, norm in rows:
    table_lines.append(f'{name:<10}  [{c[0]:+7.4f}, {c[1]:+7.4f}, {c[2]:+7.4f}]  {norm}')
table_lines.append(f'\nCoM offset from geometric centre: [{COM_OFFSET[0]:+.4f}, {COM_OFFSET[1]:+.4f}, {COM_OFFSET[2]:+.4f}] m')

fig.text(0.5, 0.005, '\n'.join(table_lines),
         ha='center', va='bottom', color=TXT_C, fontsize=7.8,
         fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.6', fc='#16213e', ec='#2d2d4e'))

# Legend
from matplotlib.lines import Line2D
legend_els = [
    mpatches.Patch(fc=BUS_C+'28', ec=BUS_C, label='Bus face'),
    mpatches.Patch(fc=SOL_C+'28', ec=SOL_C, label='Solar panel'),
    Line2D([0],[0], marker='*', color='w', mfc=COM_C, ms=12, lw=0, label='CoM'),
    Line2D([0],[0], marker='o', color='w', mfc=CTR_C, ms=8,  lw=0, label='Plate centre'),
    Line2D([0],[0], marker='+', color='white',        ms=10, lw=0, label='Geometric centre'),
]
fig.legend(handles=legend_els, loc='upper right', framealpha=0.2,
           labelcolor='white', fontsize=8, ncol=5,
           bbox_to_anchor=(0.98, 0.93))

import os
os.makedirs('out', exist_ok=True)
outpath = 'out/sc_geometry_panels.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'Saved: {outpath}')
plt.show()
