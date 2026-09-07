"""
MGA optimizer convergence diagnostics -- 2026-07-23.

Answers "is the optimizer actually behaving like an optimizer" and, when a
known-true chromosome is available (e.g. a published GTOP reference vector),
"is it converging to the RIGHT answer" -- not just "is the number going
down." Three panels:

  1. Fitness vs. iteration, log-scale. If --truth-fitness is given, also
     plots |f_best - f_truth| (the fitness GAP) on the same log axis -- a
     straight line here means geometric convergence (healthy); a plateau
     means stagnation; a staircase (long flats punctuated by sudden drops)
     is the signature of basin-hopping actually finding better basins.
  2. Per-parameter small multiples: each chromosome parameter's raw value
     vs. iteration, one subplot per parameter. If --truth (a saved
     mga_best_chromosome.csv-format reference) is given, a dashed
     horizontal reference line marks that parameter's true value in each
     subplot -- shows at a glance which parameters lock onto the truth fast
     vs. which stay volatile or jump between discrete families (a resonant
     leg's TOF gene hopping between period-multiples looks like horizontal
     banding here, not smooth convergence).
  3. Cumulative improvement count vs. iteration (derived from the fitness
     column alone, no new logging needed): counts how many iterations so
     far strictly improved the running best. A curve that goes flat means
     the search has stopped finding anything new at that point (either
     genuinely converged, or stuck).

Reads: out/<mission>/mga_convergence.csv (generation,best_fitness,p0,...,pN)
-- written by `mga.rs`'s own convergence writer, directly at the mission's
BASE output directory. NOTE (found 2026-07-23): a run made via the
`optimize` CLI subcommand ALSO gets a second, DIFFERENT
out/<mission>/optimize/mga_convergence.csv (the shared GA/PSO-style
writer, schema `iteration,best_fitness_so_far`, NO per-parameter columns)
-- always point this script at the BASE mission directory (no "/optimize"
suffix), regardless of which CLI subcommand launched the run, to get the
per-parameter data this script needs.

Usage (from MissionPlanner/ directory):
  python plot/plot_mga_convergence_diagnostics.py evj_flyby
  python plot/plot_mga_convergence_diagnostics.py cassini2_gtop_official_bounds_s42_smoke
  python plot/plot_mga_convergence_diagnostics.py evj_flyby --truth out/evj_flyby/mga_best_chromosome.csv
  python plot/plot_mga_convergence_diagnostics.py evj_flyby --truth-values "1.2e0,3.4e2,..." --truth-fitness 4930.71
"""

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BG = "#0f0f0f"
GRID = "#222"
TEXT = "#e0e0e0"
ACCENT = "#00d4ff"
TRUTH_COLOR = "#ff6b35"


def load_truth_chromosome(path: str) -> list[float]:
    """Reads a saved `mga_best_chromosome.csv` (header `flyby_bodies,p0,...`,
    one data row: `bodies;semicolon;separated,v0,v1,...`) and returns the
    parameter values only."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{path}: expected a header row and a data row")
    data_row = lines[1].split(",")
    return [float(v) for v in data_row[1:]]  # skip the flyby_bodies field


def load_truth_values(csv_str: str) -> list[float]:
    return [float(v) for v in csv_str.split(",")]


def main():
    parser = argparse.ArgumentParser(description="MGA optimizer convergence diagnostics")
    parser.add_argument("mission", nargs="?", default="evj_flyby",
                         help="Mission output directory name under out/ (the BASE directory, not .../optimize -- see module docstring)")
    parser.add_argument("--csv", default=None, help="Explicit mga_convergence.csv path (overrides mission-name default)")
    parser.add_argument("--truth", default=None,
                         help="Path to a saved mga_best_chromosome.csv-format reference chromosome")
    parser.add_argument("--truth-values", default=None,
                         help="Comma-separated reference chromosome values (alternative to --truth)")
    parser.add_argument("--truth-fitness", type=float, default=None,
                         help="Known true objective value, for the fitness-gap trace")
    parser.add_argument("--truth-uncertain", default=None,
                         help="Comma-separated 0-based parameter indices whose truth value is NOT directly "
                              "comparable yet (e.g. a departure-frame or B-plane-basis mismatch between our "
                              "parameterization and an external reference) -- rendered in a distinct grey/dotted "
                              "style instead of the confident orange dash, so the plot doesn't silently imply a "
                              "trustworthy comparison where none exists yet")
    parser.add_argument("--output", default=None, help="Save HTML to this path instead of the mission-default location")
    args = parser.parse_args()

    out_dir = Path("out") / args.mission
    csv_path = Path(args.csv) if args.csv else out_dir / "mga_convergence.csv"
    if not csv_path.exists():
        print(f"[error] {csv_path} not found")
        sys.exit(1)

    conv = pd.read_csv(csv_path)
    param_cols = [c for c in conv.columns if c.startswith("p") and c[1:].isdigit()]
    param_cols.sort(key=lambda c: int(c[1:]))
    n_params = len(param_cols)

    truth: list[float] | None = None
    if args.truth:
        truth = load_truth_chromosome(args.truth)
    elif args.truth_values:
        truth = load_truth_values(args.truth_values)
    if truth is not None and len(truth) > n_params:
        print(f"[warn] truth chromosome has {len(truth)} values but the run only has {n_params} parameters -- truncating truth")
        truth = truth[:n_params]
    elif truth is not None and len(truth) < n_params:
        print(f"[info] truth chromosome has {len(truth)} values but the run has {n_params} parameters -- "
              f"overlaying only the first {len(truth)} (the rest, e.g. this project's own leg-model genes, "
              f"have no counterpart in the reference and are left unmarked)")

    uncertain_idx = set()
    if args.truth_uncertain:
        uncertain_idx = {int(v) for v in args.truth_uncertain.split(",")}

    it = conv["generation"] if "generation" in conv.columns else conv.index
    fitness = conv["best_fitness"]

    # ── Panel 1: fitness (+ gap-to-truth) vs iteration, log scale ──────────
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=it, y=fitness, mode="lines", name="Best fitness so far",
                               line=dict(color=ACCENT, width=2)))
    if args.truth_fitness is not None:
        gap = (fitness - args.truth_fitness).abs().clip(lower=1e-9)
        fig1.add_trace(go.Scatter(x=it, y=gap, mode="lines", name="|f_best - f_truth|",
                                   line=dict(color=TRUTH_COLOR, width=2, dash="dot")))
    fig1.update_layout(
        title="Fitness convergence" + (f" (truth = {args.truth_fitness:.2f})" if args.truth_fitness is not None else ""),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT),
        xaxis=dict(title="Iteration", gridcolor=GRID),
        yaxis=dict(title="Value", type="log", gridcolor=GRID),
        legend=dict(bgcolor=BG),
    )

    # ── Panel 2: per-parameter small multiples ─────────────────────────────
    fig2 = None
    if n_params > 0:
        ncols = int(np.ceil(np.sqrt(n_params)))
        nrows = int(np.ceil(n_params / ncols))
        fig2 = make_subplots(rows=nrows, cols=ncols, subplot_titles=param_cols)
        for i, col in enumerate(param_cols):
            r, c = divmod(i, ncols)
            fig2.add_trace(
                go.Scatter(x=it, y=conv[col], mode="lines", line=dict(color=ACCENT, width=1.5), showlegend=False),
                row=r + 1, col=c + 1,
            )
            if truth is not None and i < len(truth):
                if i in uncertain_idx:
                    fig2.add_hline(y=truth[i], line=dict(color="#888888", width=1.5, dash="dot"),
                                    row=r + 1, col=c + 1)
                else:
                    fig2.add_hline(y=truth[i], line=dict(color=TRUTH_COLOR, width=1.5, dash="dash"),
                                    row=r + 1, col=c + 1)
        fig2.update_layout(
            title="Per-parameter convergence" + (" (orange dash = known truth, grey dot = truth not yet directly comparable)" if truth is not None else ""),
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT, size=10),
            height=250 * nrows,
        )
        fig2.update_xaxes(gridcolor=GRID)
        fig2.update_yaxes(gridcolor=GRID)

    # ── Panel 3: cumulative improvement count ──────────────────────────────
    improved = (fitness.diff() < 0).fillna(False)
    cum_improvements = improved.cumsum()
    fig3 = go.Figure(data=go.Scatter(x=it, y=cum_improvements, mode="lines",
                                      line=dict(color="#7fba00", width=2),
                                      name="Cumulative improving iterations"))
    fig3.update_layout(
        title=f"Cumulative improvement count ({int(cum_improvements.iloc[-1])} improving iterations of {len(conv)})",
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT),
        xaxis=dict(title="Iteration", gridcolor=GRID),
        yaxis=dict(title="Cumulative # improving iterations", gridcolor=GRID),
    )

    out_html = Path(args.output) if args.output else out_dir / "mga_convergence_diagnostics.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("<html><head><title>MGA convergence diagnostics</title></head><body style='background:#0f0f0f'>\n")
        f.write(fig1.to_html(full_html=False, include_plotlyjs="cdn"))
        if fig2 is not None:
            f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
        f.write(fig3.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>\n")
    print(f"Saved: {out_html}")

    final = fitness.iloc[-1]
    print(f"Final best fitness: {final:.4f}" + (f"  (gap to truth: {abs(final - args.truth_fitness):.4f})" if args.truth_fitness is not None else ""))
    print(f"Improving iterations: {int(cum_improvements.iloc[-1])} / {len(conv)}")

    import webbrowser
    webbrowser.open(str(out_html))


if __name__ == "__main__":
    main()
