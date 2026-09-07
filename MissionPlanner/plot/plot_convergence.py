"""
MissionPlanner trajectory design — optimizer convergence plot (best-fitness-
so-far vs. generation/iteration). Applies to any solver that writes a
"<method>_convergence.csv" with a (generation_or_iteration, best_fitness_so_far)
schema — currently GA and PSO (Phase 8j), both real-valued global searches
over (departure offset, TOF) with an identical CSV shape, so one script
covers both rather than two near-duplicates.

Reads: MissionPlanner/out/<mission>/design/ga_convergence.csv
   or: MissionPlanner/out/<mission>/design/pso_convergence.csv
       (generation,best_fitness_so_far  or  iteration,best_fitness_so_far)

Produces: <method>_convergence.html — interactive Plotly line plot, dark theme.

Usage (from MissionPlanner/ directory):
  python plot/plot_convergence.py mars_ga
  python plot/plot_convergence.py mars_pso
"""

import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

mission = sys.argv[1] if len(sys.argv) > 1 else "mars_ga"

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / mission / "design"

candidates = [("GA", DATA / "ga_convergence.csv"), ("PSO", DATA / "pso_convergence.csv")]
method, csv_path = next(((m, p) for m, p in candidates if p.exists()), (None, None))

if method is None:
    print(f"Not found: {DATA / 'ga_convergence.csv'} or {DATA / 'pso_convergence.csv'}")
    print("(Only the GA and PSO solvers write a convergence history.)")
    sys.exit(1)

df = pd.read_csv(csv_path)
x_col = df.columns[0]  # "generation" (GA) or "iteration" (PSO)

BG = "#0f0f0f"

fig = go.Figure(data=go.Scatter(
    x=df[x_col], y=df.best_fitness_so_far,
    mode="lines", line=dict(color="#00d4ff", width=2),
    name="Best fitness so far",
))

final = df.best_fitness_so_far.iloc[-1]
fig.update_layout(
    title=f"{mission} — {method} convergence, {len(df)} {x_col}s — final best fitness: {final:.4f}",
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#dddddd"),
    xaxis=dict(title=x_col.capitalize(), gridcolor="#333333"),
    yaxis=dict(title="Best fitness so far", gridcolor="#333333"),
)

out_path = DATA / f"{method.lower()}_convergence.html"
fig.write_html(str(out_path))
print(f"Wrote {out_path}")
webbrowser.open(out_path.as_uri())
