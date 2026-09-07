"""
plot_improved.py — visualise improve_transfers results with 4 panels:

  Panel 1 (top-left):  Full rotating frame — all improved trajectories
  Panel 2 (top-right): Moon zoom — original (dashed) + improved (solid) on
                        same axes per rank, colour-matched for direct comparison
  Panel 3 (bot-left):  Convergence arcs for rank 1 — each corrector iteration
                        drawn as a partial arc, coloured grey→vivid to show
                        the patch point walking onto the manifold
  Panel 4 (bot-right): Residual vs iteration (log scale, all ranks)

Run:  python plot/plot_improved.py
"""

import pathlib, webbrowser, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT  = pathlib.Path(__file__).parent.parent
OUT   = ROOT / "out" / "transfers"
TOP_N = 5

MU   = 0.01215565
X_E  = -MU
X_M  = 1.0 - MU
X_L1 = 0.836891
X_L2 = 1.155682

COLOURS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#9b5de5"]


# ── helpers ───────────────────────────────────────────────────────────────────

def load(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


def load_info(rank):
    p = OUT / f"improved_info_{rank}.txt"
    if not p.exists():
        return {}
    return dict(ln.split("=", 1) for ln in p.read_text().splitlines() if "=" in ln)


def hex_to_rgba(hex_col, alpha):
    r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def lerp_color(t):
    """Grey (t=0) → vivid red (t=1) gradient for convergence arcs."""
    r = int(180 + (231 - 180) * t)
    g = int(180 + (57  - 180) * t)
    b = int(180 + (70  - 180) * t)
    return f"rgb({r},{g},{b})"


def add_bodies(fig, rows_cols):
    shown = False
    for r, c in rows_cols:
        sl = not shown
        fig.add_trace(go.Scatter(x=[X_E], y=[0], mode="markers",
            marker=dict(size=11, color="#3b82f6"), name="Earth",
            showlegend=sl, legendgroup="Earth"), row=r, col=c)
        fig.add_trace(go.Scatter(x=[X_M], y=[0], mode="markers",
            marker=dict(size=8, color="#94a3b8"), name="Moon",
            showlegend=sl, legendgroup="Moon"), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=[X_L1, X_L2], y=[0, 0], mode="markers+text",
            text=["L1", "L2"], textposition="top center",
            marker=dict(size=6, color="#64748b", symbol="x"),
            showlegend=False), row=r, col=c)
        shown = True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ranks = [r for r in range(1, TOP_N + 1)
             if (OUT / f"improved_arc_{r}.csv").exists()]
    if not ranks:
        print(f"[!] No improved_arc_*.csv in {OUT}")
        print("    Run:  cargo run -p lunar_trajectories --bin improve_transfers")
        return

    print(f"Found {len(ranks)} improved candidate(s): {ranks}")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Full rotating frame — all improved trajectories",
            "Moon zoom — original (dashed) vs improved (solid)",
            f"Corrector convergence — rank {ranks[0]} (grey → red = iteration 0 → final)",
            "Residual vs iteration (log scale)",
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
    )

    add_bodies(fig, [(1, 1), (1, 2), (2, 1)])

    # Target orbit (shown on panels 1 and 2)
    orbit_df = load(f"improved_orbit_{ranks[0]}.csv")
    if orbit_df is not None:
        for r, c in [(1, 1), (1, 2), (2, 1)]:
            fig.add_trace(go.Scatter(
                x=orbit_df["x_nd"], y=orbit_df["y_nd"], mode="lines",
                line=dict(color="black", width=1.0, dash="dot"),
                name="Target orbit", legendgroup="orbit",
                showlegend=(r == 1 and c == 1)), row=r, col=c)

    for rank in ranks:
        col  = COLOURS[(rank - 1) % len(COLOURS)]
        info = load_info(rank)
        arc  = load(f"improved_arc_{rank}.csv")
        orig = load(f"improved_arc_orig_{rank}.csv")
        stab = load(f"improved_stable_{rank}.csv")
        cont = load(f"improved_continuation_{rank}.csv")

        res_orig_km = float(info.get("residual_km_orig", "nan"))
        res_imp_km  = float(info.get("residual_km", "nan"))
        dv_km       = float(info.get("dv_patch_km_s", "nan"))
        conv        = info.get("converged", "?")
        th_deg      = float(info.get("theta_deg", "nan"))
        r_park      = float(info.get("r_park", "nan"))

        label = (
            f"Rank {rank}  θ={th_deg:.1f}°  r={r_park:.4f} | "
            f"gap {res_orig_km:.0f}→{res_imp_km:.1f} km  "
            f"ΔV={dv_km:.4f} km/s  {'✓' if conv == 'true' else '✗'}"
        )
        lg = f"rank{rank}"

        # ── Panel 1: full frame — improved arc + stable + continuation ─────────
        for df, dash, width, alpha in [
            (arc,  "solid", 2.0, 1.0),
            (stab, "dash",  1.4, 1.0),
        ]:
            if df is not None:
                fig.add_trace(go.Scatter(
                    x=df["x_nd"], y=df["y_nd"], mode="lines",
                    line=dict(color=col, width=width, dash=dash),
                    name=label, legendgroup=lg,
                    showlegend=(df is arc)), row=1, col=1)

        if cont is not None and len(cont) > 0:
            fig.add_trace(go.Scatter(
                x=cont["x_nd"], y=cont["y_nd"], mode="lines",
                line=dict(color=col, width=1.5, dash="dashdot"),
                opacity=0.7, name=label, legendgroup=lg,
                showlegend=False), row=1, col=1)

        # ── Panel 2: Moon zoom — original (dashed faint) + improved (solid) ───
        if orig is not None:
            fig.add_trace(go.Scatter(
                x=orig["x_nd"], y=orig["y_nd"], mode="lines",
                line=dict(color=hex_to_rgba(col, 0.4), width=1.5, dash="dash"),
                name=label, legendgroup=lg, showlegend=False), row=1, col=2)

        if arc is not None:
            fig.add_trace(go.Scatter(
                x=arc["x_nd"], y=arc["y_nd"], mode="lines",
                line=dict(color=col, width=2.2),
                name=label, legendgroup=lg, showlegend=False), row=1, col=2)

        if stab is not None:
            fig.add_trace(go.Scatter(
                x=stab["x_nd"], y=stab["y_nd"], mode="lines",
                line=dict(color=col, width=1.4, dash="dash"),
                legendgroup=lg, showlegend=False), row=1, col=2)

        if cont is not None and len(cont) > 0:
            fig.add_trace(go.Scatter(
                x=cont["x_nd"], y=cont["y_nd"], mode="lines",
                line=dict(color=col, width=1.5, dash="dashdot"),
                opacity=0.7, legendgroup=lg, showlegend=False), row=1, col=2)

        # Patch-point markers on panels 1 and 2
        if arc is not None and len(arc) > 0:
            px, py = float(arc["x_nd"].iloc[-1]), float(arc["y_nd"].iloc[-1])
            for r, c in [(1, 1), (1, 2)]:
                fig.add_trace(go.Scatter(
                    x=[px], y=[py], mode="markers",
                    marker=dict(size=9, color=col, symbol="circle-open",
                                line=dict(width=2)),
                    legendgroup=lg, showlegend=False), row=r, col=c)

        # ── Panel 4: residual vs iteration for all ranks ───────────────────────
        conv_df = load(f"improved_convergence_{rank}.csv")
        if conv_df is not None and len(conv_df) > 0:
            fig.add_trace(go.Scatter(
                x=conv_df["iter"], y=conv_df["residual_nd"],
                mode="lines+markers",
                line=dict(color=col, width=1.8),
                marker=dict(size=5),
                name=f"Rank {rank}", legendgroup=f"conv{rank}",
                showlegend=True), row=2, col=2)

        print(f"  rank {rank}: {res_orig_km:.0f} km → {res_imp_km:.2f} km  "
              f"ΔV={dv_km:.4f} km/s  conv={conv}")

    # ── Panel 3: convergence arcs for rank 1 ──────────────────────────────────
    rank1      = ranks[0]
    conv1_df   = load(f"improved_convergence_{rank1}.csv")
    arc1_full  = load(f"improved_arc_{rank1}.csv")
    stab1_full = load(f"improved_stable_{rank1}.csv")

    if orbit_df is not None:
        fig.add_trace(go.Scatter(
            x=orbit_df["x_nd"], y=orbit_df["y_nd"], mode="lines",
            line=dict(color="black", width=1.0, dash="dot"),
            legendgroup="orbit", showlegend=False), row=2, col=1)

    if conv1_df is not None and len(conv1_df) > 1 and arc1_full is not None:
        n_iter = len(conv1_df)

        # Full improved arc as faint background reference
        fig.add_trace(go.Scatter(
            x=arc1_full["x_nd"], y=arc1_full["y_nd"], mode="lines",
            line=dict(color="#cccccc", width=1.0, dash="dot"),
            showlegend=False), row=2, col=1)

        # Full stable manifold as faint background reference
        if stab1_full is not None:
            fig.add_trace(go.Scatter(
                x=stab1_full["x_nd"], y=stab1_full["y_nd"], mode="lines",
                line=dict(color="#cccccc", width=1.0),
                showlegend=False), row=2, col=1)

        # Draw arc patch-end marker and manifold point for each iteration
        for i, row_data in conv1_df.iterrows():
            t      = i / max(n_iter - 1, 1)
            c_iter = lerp_color(t)
            # Arc endpoint (open circle)
            fig.add_trace(go.Scatter(
                x=[row_data["x_arc"]], y=[row_data["y_arc"]],
                mode="markers",
                marker=dict(size=8 if i < n_iter - 1 else 12,
                            color=c_iter, symbol="circle-open",
                            line=dict(width=2)),
                name=f"Iter {i}", showlegend=False), row=2, col=1)
            # Manifold match point (x marker)
            fig.add_trace(go.Scatter(
                x=[row_data["x_man"]], y=[row_data["y_man"]],
                mode="markers",
                marker=dict(size=7 if i < n_iter - 1 else 11,
                            color=c_iter, symbol="x",
                            line=dict(width=2)),
                showlegend=False), row=2, col=1)
            # Line connecting the two (the gap at this iteration)
            fig.add_trace(go.Scatter(
                x=[row_data["x_arc"], row_data["x_man"]],
                y=[row_data["y_arc"], row_data["y_man"]],
                mode="lines",
                line=dict(color=c_iter, width=1.5, dash="dot"),
                showlegend=False), row=2, col=1)

        # Colorbar-style annotation
        fig.add_annotation(
            text="○ arc end  ✕ manifold match  —— gap<br>colour: grey=iter 0 → red=final",
            x=0.02, y=0.02, xref="paper", yref="paper",
            showarrow=False, font=dict(size=10, color="#555"),
            align="left",
        )

    # ── Axis configuration ────────────────────────────────────────────────────
    moon_x = [0.72, 1.02]
    moon_y = [-0.20, 0.20]
    for r, c in [(1, 2), (2, 1)]:
        fig.update_xaxes(range=moon_x, row=r, col=c)
        fig.update_yaxes(range=moon_y, row=r, col=c)

    fig.update_yaxes(type="log", title_text="Residual [nd]", row=2, col=2)
    fig.update_xaxes(title_text="Iteration", row=2, col=2)
    fig.update_xaxes(title_text="x [nd]")
    fig.update_yaxes(title_text="y [nd]", col=1)

    fig.update_layout(
        height=1000,
        title="Transfer improvement: find_transfers → improve_transfers",
        template="plotly_white",
        legend=dict(x=1.02, y=1, xanchor="left", font=dict(size=10)),
    )

    out_path = OUT / "improved_transfers.html"
    fig.write_html(str(out_path))
    print(f"\nSaved {out_path}")
    webbrowser.open(str(out_path))


if __name__ == "__main__":
    main()
