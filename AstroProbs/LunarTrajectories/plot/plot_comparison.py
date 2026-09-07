"""
Visualise all Earth-Moon CRTBP outputs — periodic orbits, manifolds, transfers.

Reads all CSV files from out/ and produces:
  - out/crtbp_overview.png   — static 2×2 matplotlib overview
  - out/crtbp_full.html      — interactive Plotly: XY full / XY zoom / XZ / 3D

All axes are in normalised (dimensionless) CRTBP units:
  length  → L* = Earth-Moon mean distance
  time    → T* = 1/n  (mean motion period / 2π)
  velocity → V* = L*/T*

Usage:
    python plot/plot_comparison.py          # from LunarTrajectories/
"""

import pathlib, sys, webbrowser
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

OUT = pathlib.Path("out")

# ── CRTBP constants (normalised) ─────────────────────────────────────────────
MU    = 0.01215565          # Earth-Moon mass ratio
X_L1  = 0.83689             # L1 x (normalised)
X_L2  = 1.15570             # L2 x (normalised)
X_E   = -MU                 # Earth x
X_M   = 1.0 - MU            # Moon x

# Moon radius in normalised units
R_MOON_ND = 1_737.4 / 384_400.0   # ≈ 0.00452

# ── Colours ───────────────────────────────────────────────────────────────────
C = {
    "lyap_l2":  "#ffd166",   # yellow
    "lyap_l1":  "#06d6a0",   # green
    "halo":     "#22d3ee",   # cyan
    "dro":      "#a78bfa",   # purple
    "unstable": "#f97316",   # orange
    "stable":   "#38bdf8",   # sky blue
    "transfer": "#ef4444",   # red
    "earth":    "#60a5fa",   # blue
    "moon":     "#9ca3af",   # gray
    "l1":       "#06d6a0",
    "l2":       "#ffd166",
    "bg":       "#0d1117",
    "panel":    "#0d1117",
    "grid":     "#1f2937",
    "text":     "#e5e7eb",
    "tick":     "#6b7280",
}

# ── Load helper (optional files) ─────────────────────────────────────────────
def load(name, required=True):
    p = OUT / name
    if not p.exists():
        if required:
            sys.exit(f"Missing {p} — run `cargo run -p lunar_trajectories` first.")
        return None
    return pd.read_csv(p)

lyap_l2 = load("l2_lyapunov.csv")
lyap_l1 = load("l1_lyapunov.csv", required=False)
halo    = load("l2_halo_north.csv", required=False)
dro     = load("dro.csv", required=False)
unstab  = load("unstable_manifold.csv")
stab    = load("stable_manifold.csv")
direct  = load("direct_transfer.csv")

# ── Helpers ───────────────────────────────────────────────────────────────────
def xy_segs(df):
    """List of (N×2) xy arrays, one per branch."""
    segs = []
    for _, g in df.groupby("branch", sort=False):
        segs.append(np.column_stack([g["x_nd"].values, g["y_nd"].values]))
    return segs

def xz_segs(df):
    segs = []
    for _, g in df.groupby("branch", sort=False):
        segs.append(np.column_stack([g["x_nd"].values, g["z_nd"].values]))
    return segs

def style_ax(ax, xlabel="x  [nd]", ylabel="y  [nd]"):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["tick"])
    ax.spines[:].set_color(C["grid"])
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(C["tick"])
    ax.set_xlabel(xlabel, color=C["tick"])
    ax.set_ylabel(ylabel, color=C["tick"])

# ═══════════════════════════════════════════════════════════════════════════════
# Static matplotlib overview (2 × 2)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor=C["bg"])
fig.suptitle("Earth-Moon CRTBP — Periodic Orbits, Manifolds & Transfers",
             color=C["text"], fontsize=14, y=0.98)

segs_u_xy = xy_segs(unstab)
segs_s_xy = xy_segs(stab)

def add_bodies_xy(ax, lagrange_markers=True):
    ax.scatter([X_E], [0], s=100, color=C["earth"], zorder=10, label="Earth")
    ax.scatter([X_M], [0], s=60,  color=C["moon"],  zorder=10, label="Moon")
    if lagrange_markers:
        ax.scatter([X_L1], [0], s=30, marker="x", color=C["l1"], zorder=10, label="L1")
        ax.scatter([X_L2], [0], s=30, marker="x", color=C["l2"], zorder=10, label="L2")

# ── (0,0): Full XY portrait ──────────────────────────────────────────────────
ax = axes[0, 0]
ax.set_title("Rotating Frame — XY  (full view)", color=C["text"])
ax.add_collection(LineCollection(segs_u_xy, lw=0.5, alpha=0.5, colors=C["unstable"]))
ax.add_collection(LineCollection(segs_s_xy, lw=0.5, alpha=0.5, colors=C["stable"]))
ax.plot(lyap_l2["x_nd"], lyap_l2["y_nd"], color=C["lyap_l2"], lw=1.8, label="L2 Lyapunov")
if lyap_l1 is not None:
    ax.plot(lyap_l1["x_nd"], lyap_l1["y_nd"], color=C["lyap_l1"], lw=1.8, label="L1 Lyapunov")
if halo is not None:
    ax.plot(halo["x_nd"],    halo["y_nd"],    color=C["halo"],    lw=1.8, label="L2 Halo (N)")
if dro is not None:
    ax.plot(dro["x_nd"],     dro["y_nd"],     color=C["dro"],     lw=1.8, label="DRO")
ax.plot(direct["x_nd"], direct["y_nd"], color=C["transfer"], lw=1.2, ls="--",
        alpha=0.8, label="Direct transfer")
add_bodies_xy(ax)
style_ax(ax, "x  [nd]", "y  [nd]")
ax.set_aspect("equal"); ax.autoscale(); ax.margins(0.05)
ax.legend(facecolor="#1a1a2e", edgecolor=C["grid"], labelcolor=C["text"],
          fontsize=7, loc="upper left")

# ── (0,1): Moon-region zoom ───────────────────────────────────────────────────
ax = axes[0, 1]
ax.set_title("Rotating Frame — Moon Region Zoom", color=C["text"])
ax.add_collection(LineCollection(segs_u_xy, lw=0.8, alpha=0.65, colors=C["unstable"]))
ax.add_collection(LineCollection(segs_s_xy, lw=0.8, alpha=0.65, colors=C["stable"]))
ax.plot(lyap_l2["x_nd"], lyap_l2["y_nd"], color=C["lyap_l2"], lw=2.0)
if halo is not None:
    ax.plot(halo["x_nd"], halo["y_nd"], color=C["halo"], lw=2.0, label="L2 Halo (N)")
if dro is not None:
    ax.plot(dro["x_nd"], dro["y_nd"], color=C["dro"], lw=2.0, label="DRO")
moon_circle = plt.Circle((X_M, 0), R_MOON_ND, color="#374151", zorder=6)
ax.add_patch(moon_circle)
ax.scatter([X_M],  [0], s=60,  color=C["moon"],  zorder=7, label="Moon")
ax.scatter([X_L2], [0], s=40, marker="x", color=C["l2"], zorder=7, label="L2")
zoom = 0.21
ax.set_xlim(X_M - zoom, X_M + zoom)
ax.set_ylim(-zoom, zoom)
style_ax(ax, "x  [nd]", "y  [nd]")
ax.set_aspect("equal")
ax.legend(facecolor="#1a1a2e", edgecolor=C["grid"], labelcolor=C["text"], fontsize=7)

# ── (1,0): XZ view (shows halo z-extent) ─────────────────────────────────────
ax = axes[1, 0]
ax.set_title("Rotating Frame — XZ View  (out-of-plane)", color=C["text"])
segs_u_xz = xz_segs(unstab)
segs_s_xz = xz_segs(stab)
ax.add_collection(LineCollection(segs_u_xz, lw=0.5, alpha=0.5, colors=C["unstable"]))
ax.add_collection(LineCollection(segs_s_xz, lw=0.5, alpha=0.5, colors=C["stable"]))
ax.plot(lyap_l2["x_nd"], lyap_l2["z_nd"], color=C["lyap_l2"], lw=1.8, label="L2 Lyapunov")
if lyap_l1 is not None:
    ax.plot(lyap_l1["x_nd"], lyap_l1["z_nd"], color=C["lyap_l1"], lw=1.8, label="L1 Lyapunov")
if halo is not None:
    ax.plot(halo["x_nd"],    halo["z_nd"],    color=C["halo"],    lw=1.8, label="L2 Halo (N)")
if dro is not None:
    ax.plot(dro["x_nd"],     dro["z_nd"],     color=C["dro"],     lw=1.8, label="DRO")
ax.plot(direct["x_nd"], direct["z_nd"], color=C["transfer"], lw=1.2, ls="--", alpha=0.8)
ax.scatter([X_E], [0], s=100, color=C["earth"], zorder=10, label="Earth")
ax.scatter([X_M], [0], s=60,  color=C["moon"],  zorder=10, label="Moon")
ax.scatter([X_L1], [0], s=30, marker="x", color=C["l1"], zorder=10, label="L1")
ax.scatter([X_L2], [0], s=30, marker="x", color=C["l2"], zorder=10, label="L2")
style_ax(ax, "x  [nd]", "z  [nd]")
ax.set_aspect("equal"); ax.autoscale(); ax.margins(0.05)
ax.legend(facecolor="#1a1a2e", edgecolor=C["grid"], labelcolor=C["text"], fontsize=7)

# ── (1,1): 3D view ────────────────────────────────────────────────────────────
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
ax3 = fig.add_subplot(2, 2, 4, projection="3d")
ax3.set_facecolor(C["bg"])
ax3.set_title("Rotating Frame — 3D", color=C["text"])
for _, g in unstab.groupby("branch", sort=False):
    ax3.plot(g["x_nd"], g["y_nd"], g["z_nd"], color=C["unstable"], lw=0.4, alpha=0.35)
for _, g in stab.groupby("branch", sort=False):
    ax3.plot(g["x_nd"], g["y_nd"], g["z_nd"], color=C["stable"],   lw=0.4, alpha=0.35)
ax3.plot(lyap_l2["x_nd"], lyap_l2["y_nd"], lyap_l2["z_nd"],
         color=C["lyap_l2"], lw=2.0, label="L2 Lyapunov")
if halo is not None:
    ax3.plot(halo["x_nd"], halo["y_nd"], halo["z_nd"],
             color=C["halo"], lw=2.0, label="L2 Halo (N)")
if dro is not None:
    ax3.plot(dro["x_nd"], dro["y_nd"], dro["z_nd"],
             color=C["dro"], lw=2.0, label="DRO")
ax3.plot(direct["x_nd"], direct["y_nd"], direct["z_nd"],
         color=C["transfer"], lw=1.2, ls="--", alpha=0.8)
ax3.scatter([X_E], [0], [0], s=80,  color=C["earth"], zorder=10)
ax3.scatter([X_M], [0], [0], s=50,  color=C["moon"],  zorder=10)
for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
    pane.fill = False; pane.set_edgecolor(C["grid"])
ax3.tick_params(colors=C["tick"])
ax3.set_xlabel("x [nd]", color=C["tick"])
ax3.set_ylabel("y [nd]", color=C["tick"])
ax3.set_zlabel("z [nd]", color=C["tick"])
ax3.legend(facecolor="#1a1a2e", edgecolor=C["grid"], labelcolor=C["text"], fontsize=7)

# Remove the blank 4th subplot that matplotlib adds (we replaced it with 3D)
axes[1, 1].remove()

plt.tight_layout()
out_png = OUT / "crtbp_overview.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=C["bg"])
print(f"Saved {out_png}")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Interactive Plotly HTML — 2×2 layout with 3D panel
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    DARK = "#0d1117"
    GRID = "#1f2937"

    fig_pl = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "xy"},    {"type": "xy"}],
            [{"type": "xy"},    {"type": "scene"}],
        ],
        subplot_titles=[
            "Rotating Frame — XY (full)",
            "Rotating Frame — Moon Zoom",
            "Rotating Frame — XZ  (out-of-plane)",
            "Rotating Frame — 3D",
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )

    # ── Helper: add branch collection ─────────────────────────────────────────
    def add_branch_traces(df, col3d, col2d_x, col2d_y, color, name, row, col,
                          show_leg=True):
        """Add one trace per branch (all same legend group)."""
        for i, (_, g) in enumerate(df.groupby("branch", sort=False)):
            fig_pl.add_trace(go.Scatter(
                x=g[col2d_x], y=g[col2d_y],
                mode="lines",
                line=dict(color=color, width=0.7),
                name=name, legendgroup=name,
                showlegend=(show_leg and i == 0),
                opacity=0.55,
            ), row=row, col=col)

    def add_branch_3d(df, color, name, show_leg=True):
        for i, (_, g) in enumerate(df.groupby("branch", sort=False)):
            fig_pl.add_trace(go.Scatter3d(
                x=g["x_nd"], y=g["y_nd"], z=g["z_nd"],
                mode="lines",
                line=dict(color=color, width=2),
                name=name, legendgroup=name,
                showlegend=(show_leg and i == 0),
                opacity=0.4,
            ), row=2, col=2)

    def add_orbit(df, color, name, row, col, dash="solid", show_leg=True):
        fig_pl.add_trace(go.Scatter(
            x=df["x_nd"], y=df["y_nd"],
            mode="lines", line=dict(color=color, width=2.5, dash=dash),
            name=name, legendgroup=name, showlegend=show_leg,
        ), row=row, col=col)

    def add_orbit_xz(df, color, name, row, col, dash="solid", show_leg=False):
        fig_pl.add_trace(go.Scatter(
            x=df["x_nd"], y=df["z_nd"],
            mode="lines", line=dict(color=color, width=2.5, dash=dash),
            name=name, legendgroup=name, showlegend=show_leg,
        ), row=row, col=col)

    def add_orbit_3d(df, color, name, show_leg=False):
        fig_pl.add_trace(go.Scatter3d(
            x=df["x_nd"], y=df["y_nd"], z=df["z_nd"],
            mode="lines", line=dict(color=color, width=4),
            name=name, legendgroup=name, showlegend=show_leg,
        ), row=2, col=2)

    def add_bodies(row, col, show_leg=True):
        for x, label, color, sym in [
            (X_E, "Earth", C["earth"], "circle"),
            (X_M, "Moon",  C["moon"],  "circle"),
            (X_L1, "L1",   C["l1"],    "x"),
            (X_L2, "L2",   C["l2"],    "x"),
        ]:
            fig_pl.add_trace(go.Scatter(
                x=[x], y=[0], mode="markers",
                marker=dict(size=9 if label in ("Earth","Moon") else 8,
                            color=color, symbol=sym),
                name=label, legendgroup=label, showlegend=show_leg,
            ), row=row, col=col)

    def add_bodies_xz(row, col):
        for x, label, color in [(X_E, "Earth", C["earth"]),
                                 (X_M, "Moon",  C["moon"])]:
            fig_pl.add_trace(go.Scatter(
                x=[x], y=[0], mode="markers",
                marker=dict(size=9, color=color),
                name=label, legendgroup=label, showlegend=False,
            ), row=row, col=col)

    def add_bodies_3d():
        for x, label, color, size in [
            (X_E, "Earth", C["earth"], 8),
            (X_M, "Moon",  C["moon"],  6),
        ]:
            fig_pl.add_trace(go.Scatter3d(
                x=[x], y=[0], z=[0], mode="markers",
                marker=dict(size=size, color=color),
                name=label, legendgroup=label, showlegend=False,
            ), row=2, col=2)

    # ── Panel (1,1): XY full ──────────────────────────────────────────────────
    add_branch_traces(unstab, "z_nd","x_nd","y_nd", C["unstable"],"Unstable manifold",1,1)
    add_branch_traces(stab,   "z_nd","x_nd","y_nd", C["stable"],  "Stable manifold",  1,1)
    add_orbit(lyap_l2, C["lyap_l2"], "L2 Lyapunov", 1, 1)
    if lyap_l1 is not None:
        add_orbit(lyap_l1, C["lyap_l1"], "L1 Lyapunov", 1, 1)
    if halo is not None:
        add_orbit(halo, C["halo"], "L2 Halo (N)", 1, 1)
    if dro is not None:
        add_orbit(dro, C["dro"], "DRO", 1, 1)
    add_orbit(direct, C["transfer"], "Direct transfer", 1, 1, dash="dash")
    add_bodies(1, 1)

    # ── Panel (1,2): Moon zoom ────────────────────────────────────────────────
    add_branch_traces(unstab, "z_nd","x_nd","y_nd", C["unstable"],"Unstable manifold",1,2,False)
    add_branch_traces(stab,   "z_nd","x_nd","y_nd", C["stable"],  "Stable manifold",  1,2,False)
    add_orbit(lyap_l2, C["lyap_l2"], "L2 Lyapunov", 1, 2, show_leg=False)
    if halo is not None:
        add_orbit(halo, C["halo"], "L2 Halo (N)", 1, 2, show_leg=False)
    if dro is not None:
        add_orbit(dro, C["dro"], "DRO", 1, 2, show_leg=False)
    add_bodies(1, 2, show_leg=False)

    # ── Panel (2,1): XZ view ──────────────────────────────────────────────────
    add_branch_traces(unstab, "z_nd","x_nd","z_nd", C["unstable"],"Unstable manifold",2,1,False)
    add_branch_traces(stab,   "z_nd","x_nd","z_nd", C["stable"],  "Stable manifold",  2,1,False)
    add_orbit_xz(lyap_l2, C["lyap_l2"], "L2 Lyapunov", 2, 1)
    if lyap_l1 is not None:
        add_orbit_xz(lyap_l1, C["lyap_l1"], "L1 Lyapunov", 2, 1)
    if halo is not None:
        add_orbit_xz(halo, C["halo"], "L2 Halo (N)", 2, 1)
    if dro is not None:
        add_orbit_xz(dro, C["dro"], "DRO", 2, 1)
    add_orbit_xz(direct, C["transfer"], "Direct transfer", 2, 1, dash="dash")
    add_bodies_xz(2, 1)

    # ── Panel (2,2): 3D ───────────────────────────────────────────────────────
    add_branch_3d(unstab, C["unstable"], "Unstable manifold", show_leg=False)
    add_branch_3d(stab,   C["stable"],   "Stable manifold",   show_leg=False)
    add_orbit_3d(lyap_l2, C["lyap_l2"], "L2 Lyapunov")
    if lyap_l1 is not None:
        add_orbit_3d(lyap_l1, C["lyap_l1"], "L1 Lyapunov")
    if halo is not None:
        add_orbit_3d(halo, C["halo"], "L2 Halo (N)")
    if dro is not None:
        add_orbit_3d(dro, C["dro"], "DRO")
    add_orbit_3d(direct, C["transfer"], "Direct transfer")
    add_bodies_3d()

    # ── Axis config ───────────────────────────────────────────────────────────
    ax_style = dict(
        showgrid=True, gridcolor=GRID,
        zeroline=True, zerolinecolor=GRID,
        color=C["tick"],
    )

    # XY full — equal aspect approximated via range
    fig_pl.update_xaxes(title_text="x  [nd]", **ax_style, row=1, col=1)
    fig_pl.update_yaxes(title_text="y  [nd]", **ax_style,
                        scaleanchor="x", scaleratio=1, row=1, col=1)

    # XY Moon zoom
    zoom_nd = 0.21
    fig_pl.update_xaxes(title_text="x  [nd]", range=[X_M - zoom_nd, X_M + zoom_nd],
                        **ax_style, row=1, col=2)
    fig_pl.update_yaxes(title_text="y  [nd]", range=[-zoom_nd, zoom_nd],
                        **ax_style, scaleanchor="x2", scaleratio=1, row=1, col=2)

    # XZ
    fig_pl.update_xaxes(title_text="x  [nd]", **ax_style, row=2, col=1)
    fig_pl.update_yaxes(title_text="z  [nd]", **ax_style, row=2, col=1)

    # 3D scene
    scene_style = dict(
        bgcolor=DARK,
        xaxis=dict(title="x [nd]", gridcolor=GRID, color=C["tick"]),
        yaxis=dict(title="y [nd]", gridcolor=GRID, color=C["tick"]),
        zaxis=dict(title="z [nd]", gridcolor=GRID, color=C["tick"]),
    )
    fig_pl.update_layout(scene=scene_style)

    # ── Global layout ─────────────────────────────────────────────────────────
    fig_pl.update_layout(
        title=dict(
            text="Earth-Moon CRTBP — Periodic Orbits · Manifolds · Transfers",
            font=dict(color=C["text"], size=15),
            x=0.5,
        ),
        paper_bgcolor=DARK,
        plot_bgcolor=DARK,
        font=dict(color=C["text"]),
        legend=dict(
            bgcolor="#111827",
            bordercolor=GRID,
            font=dict(size=11),
            itemsizing="constant",
        ),
        height=920,
        width=1400,
    )

    out_html = OUT / "crtbp_full.html"
    fig_pl.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"Saved {out_html}")

    # Auto-open in default browser
    webbrowser.open(out_html.resolve().as_uri())
    print("Opened in browser.")

except ImportError:
    print("plotly not installed — skipping HTML output.  Install with: pip install plotly")
