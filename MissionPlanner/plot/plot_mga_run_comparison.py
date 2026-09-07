"""
MGA multi-run comparison -- 2026-07-23.

Generic comparison across any set of labeled MGA runs: different seeds
(reliability check), different budgets (does more compute help), or
different algorithm/config variants (an A/B test) -- all three are really
the same question ("compare final/convergence behaviour across a set of
labeled runs"), so one script covers all of them rather than one-off
diffing of console logs by hand.

Two panels:
  1. Overlaid convergence curves (best fitness vs. iteration), one line per
     run, common legend.
  2. Final-best-fitness comparison: a sorted bar chart across runs, best
     result highlighted -- doubles as a rough reliability/spread check when
     the runs are repeated seeds of the same config (are results clustered
     tightly, or is one seed a lucky outlier), and as a budget-scaling
     curve when x-axis order is deliberately budget-labeled (e.g.
     "60 hops", "500 hops", "3000 hops").

Reads: one mga_convergence.csv per run (generation,best_fitness,...) from
each run's BASE mission output directory -- NOT the .../optimize/
subdirectory, which (for a run made via the `optimize` CLI subcommand)
holds a different, thinner file with no per-parameter data (see
plot_mga_convergence_diagnostics.py's docstring for the full story; this
script only needs the fitness column, which both files have, but the
base-dir one is the canonical MGA writer and is what's used here).

Usage (from MissionPlanner/ directory):
  # Multi-seed reliability check
  python plot/plot_mga_run_comparison.py \\
      --run "seed 42=veega_1989_mbh_branch_s42_scratch" \\
      --run "seed 1042=veega_1989_mbh_branch_s1042_scratch" \\
      --run "seed 9042=veega_1989_mbh_branch_s9042_scratch"

  # Budget-scaling A/B (pagmo-faithful budget investigation)
  python plot/plot_mga_run_comparison.py \\
      --run "baseline (60h)=cassini2_gtop_official_bounds_s42_smoke_baseline" \\
      --run "structural (60h)=cassini2_gtop_official_bounds_s42_smoke" \\
      --run "pagmo (60h)=cassini2_gtop_official_bounds_s42_smoke_pagmo" \\
      --run "pagmo (500h)=cassini2_gtop_official_bounds_s42_pagmo_morehops" \\
      --output out/pagmo_comparison.html

  # A run label can also point directly at a CSV file:
  python plot/plot_mga_run_comparison.py --run "run A=out/foo/mga_convergence.csv"
"""

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BG = "#0f0f0f"
GRID = "#222"
TEXT = "#e0e0e0"
PALETTE = [
    "#00d4ff", "#ff6b35", "#7fba00", "#bf5fff",
    "#ffcf00", "#ff4081", "#00e676", "#ff9800",
]


def resolve_csv(spec: str) -> Path:
    """`spec` is either a direct path to a CSV, a path to a directory
    containing mga_convergence.csv, or a mission name under out/."""
    p = Path(spec)
    if p.suffix == ".csv" and p.exists():
        return p
    if p.is_dir() and (p / "mga_convergence.csv").exists():
        return p / "mga_convergence.csv"
    mission_path = Path("out") / spec / "mga_convergence.csv"
    if mission_path.exists():
        return mission_path
    raise FileNotFoundError(f"Could not resolve '{spec}' to an mga_convergence.csv (tried: {p}, {p / 'mga_convergence.csv'}, {mission_path})")


def main():
    parser = argparse.ArgumentParser(description="Compare MGA runs (seeds, budgets, or config variants)")
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=PATH",
                         help="A labeled run: LABEL=PATH, where PATH is a CSV, a directory, or a mission name under out/. Repeat for each run.")
    parser.add_argument("--output", default="out/mga_run_comparison.html", help="Output HTML path")
    args = parser.parse_args()

    if not args.run:
        print("[error] pass at least one --run LABEL=PATH")
        sys.exit(1)

    runs = []
    for spec in args.run:
        if "=" not in spec:
            print(f"[error] --run must be LABEL=PATH, got: {spec}")
            sys.exit(1)
        label, path_spec = spec.split("=", 1)
        try:
            csv_path = resolve_csv(path_spec)
        except FileNotFoundError as e:
            print(f"[warn] {e} -- skipping '{label}'")
            continue
        df = pd.read_csv(csv_path)
        runs.append((label, df))

    if not runs:
        print("[error] no runs resolved")
        sys.exit(1)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Convergence (best fitness vs. iteration)", "Final best fitness by run"))

    finals = []
    for i, (label, df) in enumerate(runs):
        color = PALETTE[i % len(PALETTE)]
        it = df["generation"] if "generation" in df.columns else df.index
        fig.add_trace(
            go.Scatter(x=it, y=df["best_fitness"], mode="lines", name=label,
                       line=dict(color=color, width=2)),
            row=1, col=1,
        )
        finals.append((label, df["best_fitness"].iloc[-1], color))

    finals_sorted = sorted(finals, key=lambda t: t[1])
    best_label = finals_sorted[0][0]
    bar_colors = ["#7fba00" if lbl == best_label else col for lbl, _, col in finals_sorted]
    fig.add_trace(
        go.Bar(x=[lbl for lbl, _, _ in finals_sorted], y=[v for _, v, _ in finals_sorted],
               marker_color=bar_colors, showlegend=False,
               text=[f"{v:.1f}" for _, v, _ in finals_sorted], textposition="outside"),
        row=1, col=2,
    )

    fig.update_layout(
        title="MGA multi-run comparison",
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT),
        legend=dict(bgcolor=BG),
    )
    fig.update_xaxes(title_text="Iteration", gridcolor=GRID, row=1, col=1)
    fig.update_yaxes(title_text="Best fitness so far", type="log", gridcolor=GRID, row=1, col=1)
    fig.update_xaxes(title_text="Run", gridcolor=GRID, row=1, col=2)
    fig.update_yaxes(title_text="Final best fitness", gridcolor=GRID, row=1, col=2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"Saved: {out_path}")

    print("\nFinal best fitness by run (best first):")
    spread = finals_sorted[-1][1] - finals_sorted[0][1]
    for lbl, v, _ in finals_sorted:
        marker = " <-- best" if lbl == best_label else ""
        print(f"  {lbl}: {v:.4f}{marker}")
    if len(finals_sorted) > 1:
        pct = 100.0 * spread / finals_sorted[0][1]
        print(f"  spread (worst - best): {spread:.4f} ({pct:.1f}% of best)")

    import webbrowser
    webbrowser.open(str(out_path))


if __name__ == "__main__":
    main()
