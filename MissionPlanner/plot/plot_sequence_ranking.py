"""
MGA sequence ranking bar chart.

Reads mga_sequence_ranking.csv (written by mga.rs when sequence_search is active)
and produces a bar chart of total optimized ΔV per sequence, colour-coded by
number of intermediate flybys (i.e. number of legs − 1).

Usage:
    python plot/plot_sequence_ranking.py
    python plot/plot_sequence_ranking.py --csv out/venus_saturn_auto/mga_sequence_ranking.csv

Expects mga_sequence_ranking.csv with columns:
    rank, sequence, n_legs, total_dv_ms, dv_departure_ms, dv_dsm_total_ms,
    dv_loi_ms, tisserand_score, estimated_vinf_arr_ms
"""

import argparse
import os
import sys

# Windows consoles default to cp1252, which cannot encode "Δ" in the printed
# summary table — force UTF-8 so the script doesn't need PYTHONIOENCODING set.
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── default CSV path ────────────────────────────────────────────────────────
DEFAULT_CSV = os.path.join(
    os.path.dirname(__file__), "..", "out", "mga_sequence_ranking.csv"
)

# ── colour palette by number of intermediate flybys ─────────────────────────
#  0 flybys = direct transfer (1 leg), 1 flyby = 2 legs, etc.
FLYBY_COLOURS = {
    0: "#4fc3f7",   # cyan — direct
    1: "#81c784",   # green
    2: "#ffb74d",   # orange
    3: "#ef5350",   # red
    4: "#ab47bc",   # purple
    5: "#8d6e63",   # brown
}
FALLBACK_COLOUR = "#90a4ae"  # grey for > 5 flybys

BG_COLOUR  = "#0f0f0f"
GRID_COLOUR = "#1e1e1e"
TEXT_COLOUR = "#e0e0e0"


def flyby_colour(n_flybys: int) -> str:
    return FLYBY_COLOURS.get(n_flybys, FALLBACK_COLOUR)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] CSV not found: {path}", file=sys.stderr)
        print("Run the MGA optimizer with sequence_search enabled to generate it.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"rank", "sequence", "n_legs", "total_dv_ms"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns in CSV: {missing}", file=sys.stderr)
        sys.exit(1)
    return df


def build_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Optimized Total ΔV by Sequence",
            "ΔV Breakdown per Sequence",
        ],
        vertical_spacing=0.15,
        row_heights=[0.55, 0.45],
    )

    # ── panel 1 — total ΔV bar chart ─────────────────────────────────────────
    n_seqs = len(df)
    labels = df["sequence"].astype(str).tolist()
    total_dv_kms = df["total_dv_ms"] / 1000.0

    # Group bars by n_flybys count for legend.
    legend_shown: set = set()
    for _, row in df.iterrows():
        n_flybys = int(row["n_legs"]) - 1
        colour = flyby_colour(n_flybys)
        show_legend = n_flybys not in legend_shown
        legend_shown.add(n_flybys)
        label = f"{n_flybys} flyby{'s' if n_flybys != 1 else ''}"
        fig.add_trace(
            go.Bar(
                x=[row["sequence"]],
                y=[row["total_dv_ms"] / 1000.0],
                name=label,
                marker_color=colour,
                legendgroup=label,
                showlegend=show_legend,
                text=[f"{row['total_dv_ms']/1000:.2f} km/s"],
                textposition="outside",
                textfont=dict(color=TEXT_COLOUR, size=10),
            ),
            row=1, col=1,
        )

    # Annotate rank numbers below each bar.
    for i, (_, row) in enumerate(df.iterrows()):
        fig.add_annotation(
            x=row["sequence"],
            y=-0.08 * total_dv_kms.max(),
            text=f"#{int(row['rank'])}",
            showarrow=False,
            font=dict(color="#888", size=9),
            xref="x1", yref="y1",
        )

    # ── panel 2 — stacked ΔV breakdown ───────────────────────────────────────
    has_breakdown = all(c in df.columns for c in
                        ["dv_departure_ms", "dv_dsm_total_ms", "dv_loi_ms"])

    if has_breakdown:
        components = [
            ("Departure burn", "dv_departure_ms",  "#00d4ff"),
            ("DSM total",      "dv_dsm_total_ms",  "#ffb74d"),
            ("LOI",            "dv_loi_ms",        "#ef5350"),
        ]
        for label, col, colour in components:
            vals = df[col] / 1000.0 if col in df.columns else [0.0] * n_seqs
            fig.add_trace(
                go.Bar(
                    x=df["sequence"].tolist(),
                    y=vals,
                    name=label,
                    marker_color=colour,
                    legendgroup=label,
                    showlegend=True,
                ),
                row=2, col=1,
            )
        fig.update_layout(barmode="stack")
    else:
        # Fall back to repeating the total bar in a muted colour.
        fig.add_trace(
            go.Bar(
                x=labels,
                y=total_dv_kms.tolist(),
                name="Total ΔV",
                marker_color="#4fc3f7",
                showlegend=False,
            ),
            row=2, col=1,
        )

    # ── layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=BG_COLOUR,
        plot_bgcolor=BG_COLOUR,
        font=dict(color=TEXT_COLOUR, family="monospace"),
        title=dict(
            text="MGA Sequence Ranking — Optimized ΔV",
            x=0.5,
            font=dict(color=TEXT_COLOUR, size=16),
        ),
        legend=dict(
            bgcolor="#111111",
            bordercolor="#333333",
            borderwidth=1,
            font=dict(size=11),
        ),
        bargap=0.25,
        bargroupgap=0.05,
    )
    for axis in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{axis: dict(
            gridcolor=GRID_COLOUR,
            zerolinecolor="#333333",
            color=TEXT_COLOUR,
            tickfont=dict(size=9),
        )})
    fig.update_layout(
        yaxis=dict(title="Total ΔV [km/s]"),
        yaxis2=dict(title="ΔV component [km/s]"),
        xaxis=dict(tickangle=-30),
        xaxis2=dict(tickangle=-30),
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MGA sequence ranking")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to mga_sequence_ranking.csv")
    parser.add_argument("--output", default=None, help="Save HTML to this path instead of showing")
    args = parser.parse_args()

    df = load_csv(args.csv)
    print(f"Loaded {len(df)} sequences from {args.csv}")

    # Print summary table.
    print(f"\n{'Rank':>4}  {'Sequence':<55}  {'Total ΔV':>10}  {'Legs':>5}")
    print("─" * 80)
    for _, row in df.iterrows():
        print(f"{int(row['rank']):>4}  {row['sequence']:<55}  "
              f"{row['total_dv_ms']/1000:>8.3f} km/s  {int(row['n_legs']):>5}")

    fig = build_figure(df)

    if args.output:
        fig.write_html(args.output)
        print(f"Saved: {args.output}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
