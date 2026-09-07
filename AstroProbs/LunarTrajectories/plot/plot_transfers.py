"""
plot_transfers.py — visualise transfers from find_transfers binary.

Run directly:  python plot/plot_transfers.py
Or via Rust:   cargo run -p lunar_trajectories --bin find_transfers

Reads: out/transfers/info.txt  (determines mode)
       out/transfers/*.csv     (depends on mode)
Saves: out/transfers/transfers.html  (auto-opens in browser)

Layout (EarthToOrbit mode):
  Row 1: Rotating XY (full)  |  Rotating XY (Moon zoom)
  Row 2: Inertial XY (full)  |  Inertial XY (Moon zoom)
  Row 3: Rotating XZ         |  3-D view
"""
import pathlib, webbrowser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
OUT  = ROOT / "out" / "transfers"

# ── System constants ──────────────────────────────────────────────────────────
MU   = 0.01215565
X_E  = -MU
X_M  = 1.0 - MU
X_L1 = 0.836891
X_L2 = 1.155682


# ── Helpers ───────────────────────────────────────────────────────────────────
def load(name: str):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


def load_info():
    p = OUT / "info.txt"
    if not p.exists():
        return {}
    return dict(line.split("=", 1) for line in p.read_text().splitlines() if "=" in line)


def to_inertial(df):
    """
    Convert a rotating-frame trajectory DataFrame to inertial (non-rotating) frame.

    CRTBP rotating frame spins at ω = 1 [nd/nd].  At time t, to recover the
    inertial position, rotate the rotating-frame vector by angle +t about z:

        x_i =  x_r * cos(t) − y_r * sin(t)
        y_i =  x_r * sin(t) + y_r * cos(t)
        z_i =  z_r

    Returns (x_i, y_i, z_i) as numpy arrays.
    """
    t   = df["time_nd"].values
    x_r = df["x_nd"].values
    y_r = df["y_nd"].values
    z_r = df["z_nd"].values
    x_i = x_r * np.cos(t) - y_r * np.sin(t)
    y_i = x_r * np.sin(t) + y_r * np.cos(t)
    return x_i, y_i, z_r


def add_bodies_rot(fig, row, col, xz=False, show_legend=False):
    """Add Earth, Moon, and Lagrange markers in the rotating frame."""
    fig.add_trace(go.Scatter(
        x=[X_E], y=[0], mode="markers",
        marker=dict(size=12, color="#3b82f6"), name="Earth",
        showlegend=show_legend), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=[X_M], y=[0], mode="markers",
        marker=dict(size=8, color="#94a3b8"), name="Moon",
        showlegend=show_legend), row=row, col=col)
    if not xz:
        fig.add_trace(go.Scatter(
            x=[X_L1, X_L2], y=[0, 0], mode="markers+text",
            text=["L1", "L2"], textposition="top center",
            marker=dict(size=7, color="#64748b", symbol="x"),
            showlegend=False), row=row, col=col)


def add_bodies_inertial(fig, row, col, t_max=20.0, show_legend=False):
    """
    Add Earth (at barycenter origin, nearly stationary) and Moon orbit in
    inertial frame.  Moon traces a circle of radius (1-mu) at ω=1.
    """
    # Earth stays at x_E * cos(t) ≈ 0 (barycenter nearly = Earth for small mu)
    fig.add_trace(go.Scatter(
        x=[X_E], y=[0], mode="markers",
        marker=dict(size=12, color="#3b82f6"), name="Earth (inertial)",
        showlegend=show_legend), row=row, col=col)
    # Moon orbit
    th = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(
        x=X_M * np.cos(th), y=X_M * np.sin(th), mode="lines",
        line=dict(color="#94a3b8", width=0.8, dash="dot"),
        name="Moon orbit", showlegend=show_legend), row=row, col=col)


def add_traj_rot(fig, df, name, color, width=1.8, opacity=1.0, showlegend=True,
                 rows_cols=None):
    """Add a rotating-frame trajectory.  First trace owns the legend entry."""
    if rows_cols is None:
        rows_cols = [(1, 1), (1, 2), (3, 1)]
    kw = dict(mode="lines", line=dict(color=color, width=width),
              name=name, opacity=opacity, legendgroup=name)
    for i, (r, c) in enumerate(rows_cols):
        if r == 3 and c == 1:   # XZ panel
            fig.add_trace(go.Scatter(
                x=df.x_nd, y=df.z_nd, **kw, showlegend=False), row=r, col=c)
        else:
            fig.add_trace(go.Scatter(
                x=df.x_nd, y=df.y_nd, **kw,
                showlegend=(showlegend and i == 0)), row=r, col=c)


def add_traj_inertial(fig, df, name, color, width=1.8, opacity=1.0,
                      rows_cols=None):
    """Add an inertial-frame trajectory.  Always showlegend=False — the
    rotating-frame trace already owns the legend entry for this name."""
    if rows_cols is None:
        rows_cols = [(2, 1), (2, 2)]
    xi, yi, _ = to_inertial(df)
    kw = dict(mode="lines", line=dict(color=color, width=width),
              name=name, opacity=opacity, legendgroup=name, showlegend=False)
    for r, c in rows_cols:
        fig.add_trace(go.Scatter(x=xi, y=yi, **kw), row=r, col=c)


def add_traj_3d(fig, df, name, color, width=2, opacity=1.0, row=3, col=2):
    fig.add_trace(go.Scatter3d(
        x=df.x_nd, y=df.y_nd, z=df.z_nd, mode="lines",
        line=dict(color=color, width=width), opacity=opacity,
        name=name, showlegend=False), row=row, col=col)


# ── Debug plot ────────────────────────────────────────────────────────────────
def build_debug_plot(info, dbg_dep, dbg_arr):
    """
    Show all candidate arcs that were tested during the corrector seed phase.

    Left panel  (XY full):  departure seed arcs + arrival manifold arcs + section
    Right panel (XY zoom):  Moon-region zoom of the same
    """
    x_sec = float(info.get("x_section", 0.8))

    # Colour palettes — one per seed/branch index
    dep_colours = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#a855f7"]
    arr_colours = ["#38bdf8", "#67e8f9", "#a5f3fc", "#0ea5e9", "#0284c7",
                   "#7dd3fc", "#bae6fd", "#e0f2fe", "#0369a1", "#075985"]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=["Debug: section crossings — full XY", "Moon-region zoom"],
        horizontal_spacing=0.1,
    )

    for r, c in [(1, 1), (1, 2)]:
        add_bodies_rot(fig, r, c, show_legend=(c == 1))

    # Poincaré section
    for r, c in [(1, 1), (1, 2)]:
        fig.add_trace(go.Scatter(
            x=[x_sec, x_sec], y=[-0.8, 0.8], mode="lines",
            line=dict(color="#64748b", width=1.2, dash="dash"),
            name="Poincaré section",
            showlegend=(c == 1)), row=r, col=c)

    # Departure arcs (all grid angles that reach the section)
    if dbg_dep is not None:
        for arc_idx, grp in dbg_dep.groupby("arc_idx"):
            theta_deg = float(arc_idx) / len(dbg_dep["arc_idx"].unique()) * 360.0
            colour = dep_colours[int(arc_idx) % len(dep_colours)]
            name   = f"dep θ≈{int(round(float(arc_idx)*360/len(dbg_dep['arc_idx'].unique())))}°"
            for r, c in [(1, 1), (1, 2)]:
                fig.add_trace(go.Scatter(
                    x=grp.x_nd, y=grp.y_nd, mode="lines",
                    line=dict(color=colour, width=1.4), opacity=0.85,
                    name=name, legendgroup=name,
                    showlegend=(c == 1)), row=r, col=c)
            # Mark section crossing (last point)
            last = grp.iloc[-1]
            for r, c in [(1, 1), (1, 2)]:
                fig.add_trace(go.Scatter(
                    x=[last.x_nd], y=[last.y_nd], mode="markers",
                    marker=dict(size=9, color=colour, symbol="circle"),
                    name=name, legendgroup=name,
                    showlegend=False), row=r, col=c)

    # Arrival manifold arcs (far → section)
    if dbg_arr is not None:
        plotted: set = set()
        for branch_idx, grp in dbg_arr.groupby("branch_idx"):
            colour = arr_colours[int(branch_idx) % len(arr_colours)]
            name   = f"arr branch {branch_idx}"
            first  = branch_idx not in plotted
            plotted.add(branch_idx)
            for r, c in [(1, 1), (1, 2)]:
                fig.add_trace(go.Scatter(
                    x=grp.x_nd, y=grp.y_nd, mode="lines",
                    line=dict(color=colour, width=1.0), opacity=0.7,
                    name=name, legendgroup=name,
                    showlegend=(first and c == 1)), row=r, col=c)
            # Mark section crossing (last point)
            last = grp.iloc[-1]
            for r, c in [(1, 1), (1, 2)]:
                fig.add_trace(go.Scatter(
                    x=[last.x_nd], y=[last.y_nd], mode="markers",
                    marker=dict(size=7, color=colour, symbol="diamond"),
                    name=name, legendgroup=name,
                    showlegend=False), row=r, col=c)

    fig.update_xaxes(title_text="x [nd]", row=1, col=1)
    fig.update_yaxes(title_text="y [nd]", row=1, col=1)
    fig.update_xaxes(title_text="x [nd]", range=[X_M - 0.25, X_M + 0.25], row=1, col=2)
    fig.update_yaxes(title_text="y [nd]", range=[-0.25, 0.25],             row=1, col=2)

    fig.update_layout(
        height=600, width=1200,
        template="plotly_dark",
        title="Debug: corrector seed arcs at Poincaré section",
        legend=dict(x=1.02, y=1.0, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    info = load_info()
    mode = info.get("mode", "EarthToOrbit")
    print(f"Transfer mode: {mode}")

    if not info:
        print("ERROR: out/transfers/info.txt not found.")
        print("  Run:  cargo run -p lunar_trajectories --bin find_transfers")
        return

    if mode == "EarthToOrbit":
        fig, title = build_earth_to_orbit(info)
    else:
        fig, title = build_manifold_intersect(info)

    fig.update_layout(title=title)

    out_html = OUT / "transfers.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    print(f"Saved {out_html}")
    webbrowser.open(out_html.resolve().as_uri())
    print("Opened in browser.")

    if mode == "EarthToOrbit":
        dbg_dep = load("debug_departure.csv")
        dbg_arr = load("debug_arrival.csv")
        if dbg_dep is not None or dbg_arr is not None:
            dbg_fig = build_debug_plot(info, dbg_dep, dbg_arr)
            out_dbg = OUT / "transfers_debug.html"
            dbg_fig.write_html(str(out_dbg))
            print(f"Saved {out_dbg}")
            webbrowser.open(out_dbg.resolve().as_uri())
            print("Opened debug plot in browser.")


# ── Mode A ────────────────────────────────────────────────────────────────────
def build_earth_to_orbit(info):
    target      = load("target_orbit.csv")
    arc         = load("transfer_arc.csv")
    stable_arc  = load("stable_arc.csv")
    ext_arc     = load("extended_arc.csv")
    patch       = load("patch_point.csv")
    stab_man    = load("stable_manifold.csv")

    # ── Layout: 3 rows × 2 cols ───────────────────────────────────────────────
    # Row 1: rotating XY full | rotating XY Moon zoom
    # Row 2: inertial XY full | inertial XY Moon zoom
    # Row 3: rotating XZ     | 3-D view
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=[
            "Rotating frame — XY (full)",
            "Rotating — Moon region zoom",
            "Inertial frame — XY (full)",
            "Inertial — Moon region zoom",
            "Rotating frame — XZ",
            "3-D view (rotating)",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.09,
    )

    # Body markers
    add_bodies_rot(fig, 1, 1, show_legend=True)
    add_bodies_rot(fig, 1, 2)
    add_bodies_rot(fig, 3, 1, xz=True)
    add_bodies_inertial(fig, 2, 1, show_legend=True)
    add_bodies_inertial(fig, 2, 2)
    fig.add_trace(go.Scatter3d(
        x=[X_E, X_M], y=[0, 0], z=[0, 0], mode="markers",
        marker=dict(size=[5, 4], color=["#3b82f6", "#94a3b8"]),
        showlegend=False), row=3, col=2)

    # Faint stable manifold (rotating only)
    if stab_man is not None:
        first = True
        for _, grp in stab_man.groupby("branch"):
            fig.add_trace(go.Scatter(
                x=grp.x_nd, y=grp.y_nd, mode="lines",
                line=dict(color="#38bdf8", width=0.5),
                name="stable manifold", legendgroup="stab_man",
                showlegend=first, opacity=0.20), row=1, col=1)
            first = False

    # Arrival orbit
    if target is not None:
        add_traj_rot(fig, target, "arrival orbit", "#ffd166", width=2.2,
                     rows_cols=[(1,1),(1,2),(3,1)])
        add_traj_inertial(fig, target, "arrival orbit", "#ffd166", width=2.2,
                          rows_cols=[(2,1),(2,2)])
        add_traj_3d(fig, target, "arrival orbit", "#ffd166", row=3, col=2)

    # Departure arc
    if arc is not None:
        add_traj_rot(fig, arc, "departure arc", "#ef4444", width=1.8,
                     rows_cols=[(1,1),(1,2),(3,1)])
        add_traj_inertial(fig, arc, "departure arc", "#ef4444", width=1.8,
                          rows_cols=[(2,1),(2,2)])
        add_traj_3d(fig, arc, "departure arc", "#ef4444", row=3, col=2)

        # Injection point: rotating panels
        inj = arc.iloc[0]
        for r, c in [(1,1),(1,2)]:
            fig.add_trace(go.Scatter(
                x=[inj.x_nd], y=[inj.y_nd], mode="markers",
                marker=dict(size=12, color="#ef4444", symbol="star"),
                name="injection burn", legendgroup="injection",
                showlegend=(r==1 and c==1)), row=r, col=c)
        # Injection point: inertial panels (at t=0, inertial = rotating)
        for r, c in [(2,1),(2,2)]:
            fig.add_trace(go.Scatter(
                x=[inj.x_nd], y=[inj.y_nd], mode="markers",
                marker=dict(size=12, color="#ef4444", symbol="star"),
                name="injection burn", legendgroup="injection",
                showlegend=False), row=r, col=c)

    # Stable arc (backward from arrival orbit to patch point)
    if stable_arc is not None:
        add_traj_rot(fig, stable_arc, "stable arc", "#38bdf8", width=1.8,
                     rows_cols=[(1,1),(1,2),(3,1)])
        add_traj_inertial(fig, stable_arc, "stable arc", "#38bdf8",
                          width=1.8, rows_cols=[(2,1),(2,2)])
        add_traj_3d(fig, stable_arc, "stable arc", "#38bdf8", row=3, col=2)

    # Extended arc (forward propagation from patch point — arrival verification)
    if ext_arc is not None:
        add_traj_rot(fig, ext_arc, "extended arc", "#a78bfa",
                     width=1.5, opacity=0.85, rows_cols=[(1,1),(1,2),(3,1)])
        add_traj_inertial(fig, ext_arc, "extended arc", "#a78bfa",
                          width=1.5, opacity=0.85, rows_cols=[(2,1),(2,2)])
        add_traj_3d(fig, ext_arc, "extended arc", "#a78bfa", row=3, col=2)

    # Poincaré section — show seed section (dashed) and converged patch x (dotted)
    x_sec = info.get("x_section")
    x_pat = info.get("x_patch")
    if x_sec:
        xv = float(x_sec)
        for r, c in [(1,1),(1,2)]:
            fig.add_trace(go.Scatter(
                x=[xv, xv], y=[-0.8, 0.8], mode="lines",
                line=dict(color="#64748b", width=1.2, dash="dash"),
                name="seed section",
                showlegend=(r==1 and c==1)), row=r, col=c)
    if x_pat and x_pat != x_sec:
        xp = float(x_pat)
        for r, c in [(1,1),(1,2)]:
            fig.add_trace(go.Scatter(
                x=[xp, xp], y=[-0.8, 0.8], mode="lines",
                line=dict(color="#a78bfa", width=1.0, dash="dot"),
                name="patch section",
                showlegend=(r==1 and c==1)), row=r, col=c)

    # Patch point markers
    if patch is not None:
        arc_side  = patch[patch.side == "arc"]
        stab_side = patch[patch.side == "stable"]
        converged = info.get("converged", "false") == "true"
        for r, c in [(1,1),(1,2)]:
            if not arc_side.empty:
                fig.add_trace(go.Scatter(
                    x=arc_side.x_nd, y=arc_side.y_nd, mode="markers",
                    marker=dict(size=13, color="#ffffff", symbol="diamond"),
                    name="patch point (arc→manifold)",
                    showlegend=(r==1 and c==1)), row=r, col=c)
            if not stab_side.empty and not converged:
                fig.add_trace(go.Scatter(
                    x=stab_side.x_nd, y=stab_side.y_nd, mode="markers",
                    marker=dict(size=10, color="#f97316", symbol="diamond-open"),
                    name="patch point (manifold side)",
                    showlegend=(r==1 and c==1)), row=r, col=c)

    # Axes
    for r, c, xl, yl in [(1,1,"x [nd]","y [nd]"),
                          (1,2,"x [nd]","y [nd]"),
                          (2,1,"x [nd]","y [nd]"),
                          (2,2,"x [nd]","y [nd]"),
                          (3,1,"x [nd]","z [nd]")]:
        fig.update_xaxes(title_text=xl, scaleanchor=None, row=r, col=c)
        fig.update_yaxes(title_text=yl, row=r, col=c)
    fig.update_scenes(
        xaxis_title="x [nd]", yaxis_title="y [nd]", zaxis_title="z [nd]")

    # Moon zoom for rotating and inertial panels
    for r in [1, 2]:
        fig.update_xaxes(range=[X_M - 0.25, X_M + 0.25], row=r, col=2)
        fig.update_yaxes(range=[-0.25, 0.25], row=r, col=2)

    gate       = info.get("gate", "?")
    theta      = info.get("theta_deg", "?")
    dv_inj     = info.get("dv_inject_km_s")
    dv_cap     = info.get("dv_capture_km_s")
    gap_ok     = info.get("gap_accepted", "true") == "true"
    res_km_val = info.get("residual_km", "?")
    max_gap    = info.get("max_gap_km", "?")
    parts  = [f"L{gate} gateway — manifold injection"]
    if theta != "?":
        parts.append(f"θ = {float(theta):.1f}° (parking orbit)")
    if dv_inj and float(dv_inj) > 0:
        parts.append(f"ΔV_depart = {float(dv_inj):.3f} km/s")
    if dv_cap and float(dv_cap) > 0:
        parts.append(f"ΔV_patch = {float(dv_cap):.3f} km/s")
    else:
        parts.append("ΔV_patch ≈ 0")
    if res_km_val != "?" and max_gap != "?":
        status = "✓ accepted" if gap_ok else f"✗ rejected (gap {float(res_km_val):.0f} km > {float(max_gap):.0f} km limit)"
        parts.append(status)
    title = "Earth → Arrival Orbit  |  " + "  |  ".join(parts)

    fig.update_layout(
        height=1100, width=1300,
        template="plotly_dark",
        legend=dict(x=1.02, y=1.0, bgcolor="rgba(0,0,0,0)"),
    )
    return fig, title


# ── Mode B ────────────────────────────────────────────────────────────────────
def build_manifold_intersect(info):
    source   = load("source_orbit.csv")
    target   = load("target_orbit.csv")
    arc_u    = load("arc_unstable.csv")
    arc_s    = load("arc_stable.csv")
    unstab   = load("unstable.csv")
    stab     = load("stable.csv")

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=[
            "Rotating frame — XY (full)",
            "Moon region — XY zoom",
            "Rotating frame — XZ",
            "3-D view",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    add_bodies_rot(fig, 1, 1, show_legend=True)
    add_bodies_rot(fig, 1, 2)
    add_bodies_rot(fig, 2, 1, xz=True)
    fig.add_trace(go.Scatter3d(
        x=[X_E, X_M], y=[0, 0], z=[0, 0], mode="markers",
        marker=dict(size=[5, 4], color=["#3b82f6", "#94a3b8"]),
        showlegend=False), row=2, col=2)

    rot_panels = [(1,1),(1,2),(2,1)]

    for df, name, color in [(unstab, "unstable manifold", "#f97316"),
                             (stab,   "stable manifold",   "#38bdf8")]:
        if df is None:
            continue
        first = True
        for _, grp in df.groupby("branch"):
            fig.add_trace(go.Scatter(
                x=grp.x_nd, y=grp.y_nd, mode="lines",
                line=dict(color=color, width=0.5),
                name=name, legendgroup=name,
                showlegend=first, opacity=0.30), row=1, col=1)
            first = False

    if source is not None:
        add_traj_rot(fig, source, "source orbit", "#ffd166", width=2.2,
                     rows_cols=rot_panels)
        add_traj_3d(fig, source, "source orbit", "#ffd166", row=2, col=2)
    if target is not None:
        add_traj_rot(fig, target, "target orbit", "#22d3ee", width=2.2,
                     rows_cols=rot_panels)
        add_traj_3d(fig, target, "target orbit", "#22d3ee", row=2, col=2)

    if arc_u is not None:
        add_traj_rot(fig, arc_u, "unstable arc", "#f97316", width=2.5,
                     rows_cols=rot_panels)
        add_traj_3d(fig, arc_u, "unstable arc", "#f97316", row=2, col=2)
    if arc_s is not None:
        add_traj_rot(fig, arc_s, "stable arc", "#38bdf8", width=2.5,
                     rows_cols=rot_panels)
        add_traj_3d(fig, arc_s, "stable arc", "#38bdf8", row=2, col=2)

    if arc_u is not None and not arc_u.empty:
        pt = arc_u.iloc[-1]
        for r, c in [(1,1),(1,2)]:
            fig.add_trace(go.Scatter(
                x=[pt.x_nd], y=[pt.y_nd], mode="markers",
                marker=dict(size=12, color="#ffffff", symbol="diamond"),
                name="ΔV patch point",
                showlegend=(r==1 and c==1)), row=r, col=c)

    x_sec = info.get("x_section")
    if x_sec:
        xv = float(x_sec)
        for r, c in [(1,1),(1,2)]:
            fig.add_trace(go.Scatter(
                x=[xv, xv], y=[-0.6, 0.6], mode="lines",
                line=dict(color="#64748b", width=1, dash="dash"),
                name="Poincaré section",
                showlegend=(r==1 and c==1)), row=r, col=c)

    for r, c, xl, yl in [(1,1,"x [nd]","y [nd]"),(1,2,"x [nd]","y [nd]"),
                          (2,1,"x [nd]","z [nd]")]:
        fig.update_xaxes(title_text=xl, row=r, col=c)
        fig.update_yaxes(title_text=yl, row=r, col=c)
    fig.update_scenes(
        xaxis_title="x [nd]", yaxis_title="y [nd]", zaxis_title="z [nd]")
    fig.update_xaxes(range=[X_M - 0.25, X_M + 0.25], row=1, col=2)
    fig.update_yaxes(range=[-0.25, 0.25], row=1, col=2)

    fig.update_layout(
        height=820, width=1200,
        template="plotly_dark",
        legend=dict(x=1.02, y=1.0, bgcolor="rgba(0,0,0,0)"),
    )

    dv = info.get("dv_mag_km_s")
    suffix = f"  |  |ΔV| = {float(dv):.4f} km/s" if dv else \
             ("  |  no transfer found" if info.get("transfer_found") == "false" else "")
    return fig, "Manifold Intersection Transfer" + suffix


if __name__ == "__main__":
    main()
