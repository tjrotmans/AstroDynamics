"""
plot_manifolds.py — visualise invariant manifolds from plot_manifolds binary.

Run directly:  python plot/plot_manifolds.py
Or via Rust:   cargo run -p lunar_trajectories --bin plot_manifolds

Reads: out/manifolds/orbit.csv, unstable.csv, stable.csv, meta.txt
Saves: out/manifolds/manifolds.html  (auto-opens in browser)
"""
import pathlib, webbrowser
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
OUT  = ROOT / "out" / "manifolds"

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


def load_meta():
    p = OUT / "meta.txt"
    if not p.exists():
        return {}
    return dict(line.split("=", 1) for line in p.read_text().splitlines() if "=" in line)


def add_bodies_2d(fig, row, col, xz=False, show_legend=False):
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    orbit  = load("orbit.csv")
    unstab = load("unstable.csv")
    stab   = load("stable.csv")
    meta   = load_meta()

    if orbit is None:
        print("ERROR: out/manifolds/orbit.csv not found.")
        print("  Run:  cargo run -p lunar_trajectories --bin plot_manifolds")
        return

    title_suffix = ""
    if meta:
        fam    = meta.get("family", "?")
        period = meta.get("period_days", "?")
        title_suffix = f" — {fam}  T={float(period):.1f} d" if period != "?" else f" — {fam}"

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=[
            "XY — full view",
            "XY — Moon zoom",
            "XZ — out-of-plane",
            "3-D view",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Body markers
    add_bodies_2d(fig, 1, 1, show_legend=True)
    add_bodies_2d(fig, 1, 2)
    add_bodies_2d(fig, 2, 1, xz=True)
    fig.add_trace(go.Scatter3d(
        x=[X_E, X_M], y=[0, 0], z=[0, 0], mode="markers",
        marker=dict(size=[5, 4], color=["#3b82f6", "#94a3b8"]),
        showlegend=False), row=2, col=2)

    # ── Orbit ─────────────────────────────────────────────────────────────────
    for r, c, ycol in [(1, 1, "y_nd"), (1, 2, "y_nd"), (2, 1, "z_nd")]:
        fig.add_trace(go.Scatter(
            x=orbit.x_nd, y=orbit[ycol], mode="lines",
            line=dict(color="#ffd166", width=2.5),
            name="orbit", showlegend=(r == 1 and c == 1)), row=r, col=c)
    fig.add_trace(go.Scatter3d(
        x=orbit.x_nd, y=orbit.y_nd, z=orbit.z_nd, mode="lines",
        line=dict(color="#ffd166", width=3),
        name="orbit", showlegend=False), row=2, col=2)

    # ── Manifold branches ─────────────────────────────────────────────────────
    for df, name, color in [(unstab, "unstable", "#f97316"),
                             (stab,   "stable",   "#38bdf8")]:
        if df is None:
            continue
        first = True
        for bi, grp in df.groupby("branch"):
            show = first
            first = False
            kw = dict(mode="lines", line=dict(color=color, width=0.7),
                      name=name, legendgroup=name,
                      showlegend=show, opacity=0.55)
            fig.add_trace(go.Scatter(x=grp.x_nd, y=grp.y_nd, **kw), row=1, col=1)
            fig.add_trace(go.Scatter(x=grp.x_nd, y=grp.y_nd,
                                     **{**kw, "showlegend": False}), row=1, col=2)
            fig.add_trace(go.Scatter(x=grp.x_nd, y=grp.z_nd,
                                     **{**kw, "showlegend": False}), row=2, col=1)
            fig.add_trace(go.Scatter3d(
                x=grp.x_nd, y=grp.y_nd, z=grp.z_nd, mode="lines",
                line=dict(color=color, width=1),
                name=name, legendgroup=name,
                showlegend=False, opacity=0.45), row=2, col=2)

    # Axes
    for r, c, xl, yl in [(1, 1, "x [nd]", "y [nd]"),
                          (1, 2, "x [nd]", "y [nd]"),
                          (2, 1, "x [nd]", "z [nd]")]:
        fig.update_xaxes(title_text=xl, row=r, col=c)
        fig.update_yaxes(title_text=yl, row=r, col=c)
    fig.update_scenes(
        xaxis_title="x [nd]", yaxis_title="y [nd]", zaxis_title="z [nd]")

    # Moon zoom
    fig.update_xaxes(range=[X_M - 0.22, X_M + 0.22], row=1, col=2)
    fig.update_yaxes(range=[-0.22, 0.22], row=1, col=2)

    fig.update_layout(
        title="Invariant Manifolds — Earth-Moon CRTBP" + title_suffix,
        height=820, width=1200,
        template="plotly_dark",
        legend=dict(x=1.02, y=1.0, bgcolor="rgba(0,0,0,0)"),
    )

    out_html = OUT / "manifolds.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    print(f"Saved {out_html}")
    webbrowser.open(out_html.resolve().as_uri())
    print("Opened in browser.")


if __name__ == "__main__":
    main()
