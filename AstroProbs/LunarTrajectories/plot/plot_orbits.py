"""
plot_orbits.py — visualise periodic orbits from find_orbits.

Run directly:  python plot/plot_orbits.py
Or via Rust:   cargo run -p lunar_trajectories --bin find_orbits

Reads: out/orbits/manifest.csv  +  out/orbits/{label}.csv
Saves: out/orbits/orbits.html   (auto-opens in browser)
"""
import pathlib, webbrowser
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
OUT  = ROOT / "out" / "orbits"

# ── System constants ──────────────────────────────────────────────────────────
MU   = 0.01215565
X_E  = -MU            # Earth
X_M  = 1.0 - MU       # Moon
X_L1 = 0.836891       # L1 (approximate, good to 6 d.p.)
X_L2 = 1.155682       # L2

# ── Color per family ──────────────────────────────────────────────────────────
PALETTE = {
    "Lyapunov":  "#ffd166",
    "HaloNorth": "#22d3ee",
    "HaloSouth": "#a78bfa",
    "DRO":       "#f97316",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load(name: str):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


def bodies_2d(show_legend=False):
    """Earth, Moon, L1, L2 traces for XY panels."""
    return [
        go.Scatter(x=[X_E], y=[0], mode="markers",
                   marker=dict(size=14, color="#3b82f6", symbol="circle"),
                   name="Earth", showlegend=show_legend),
        go.Scatter(x=[X_M], y=[0], mode="markers",
                   marker=dict(size=10, color="#94a3b8", symbol="circle"),
                   name="Moon",  showlegend=show_legend),
        go.Scatter(x=[X_L1, X_L2], y=[0, 0], mode="markers+text",
                   text=["L1", "L2"], textposition="top center",
                   marker=dict(size=8, color="#64748b", symbol="x"),
                   name="L-points", showlegend=show_legend),
    ]


def bodies_xz():
    return [
        go.Scatter(x=[X_E], y=[0], mode="markers",
                   marker=dict(size=14, color="#3b82f6"), showlegend=False),
        go.Scatter(x=[X_M], y=[0], mode="markers",
                   marker=dict(size=10, color="#94a3b8"), showlegend=False),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    mf = load("manifest.csv")
    if mf is None:
        print("ERROR: out/orbits/manifest.csv not found.")
        print("  Run:  cargo run -p lunar_trajectories --bin find_orbits")
        return

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=[
            "Rotating frame — XY (full)",
            "Moon region — XY zoom",
            "Rotating frame — XZ (out-of-plane)",
            "3-D view",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Body markers
    for tr in bodies_2d(show_legend=True):
        fig.add_trace(tr, row=1, col=1)
    for tr in bodies_2d(show_legend=False):
        fig.add_trace(tr, row=1, col=2)
    for tr in bodies_xz():
        fig.add_trace(tr, row=2, col=1)
    fig.add_trace(go.Scatter3d(
        x=[X_E, X_M], y=[0, 0], z=[0, 0], mode="markers",
        marker=dict(size=[6, 5], color=["#3b82f6", "#94a3b8"]),
        showlegend=False), row=2, col=2)

    seen_families = set()
    for _, row in mf.iterrows():
        df = load(f"{row['label']}.csv")
        if df is None:
            continue

        color  = PALETTE.get(row["family"], "#94a3b8")
        period = float(row["period_days"])
        name   = f"{row['label']}  (T={period:.1f} d)"
        first  = row["family"] not in seen_families
        seen_families.add(row["family"])

        kw = dict(mode="lines", line=dict(color=color, width=1.8),
                  name=name, legendgroup=row["label"])

        fig.add_trace(go.Scatter(x=df.x_nd, y=df.y_nd,
                                 **kw, showlegend=first), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.x_nd, y=df.y_nd,
                                 **kw, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=df.x_nd, y=df.z_nd,
                                 **kw, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter3d(
            x=df.x_nd, y=df.y_nd, z=df.z_nd, mode="lines",
            line=dict(color=color, width=2),
            name=name, legendgroup=row["label"], showlegend=False), row=2, col=2)

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
        title="Periodic Orbits — Earth-Moon CRTBP",
        height=820, width=1200,
        template="plotly_dark",
        legend=dict(x=1.02, y=1.0, bgcolor="rgba(0,0,0,0)"),
    )

    out_html = OUT / "orbits.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    print(f"Saved {out_html}")
    webbrowser.open(out_html.resolve().as_uri())
    print("Opened in browser.")


if __name__ == "__main__":
    main()
