"""
MissionPlanner Phase 9 optimizer — a sample of the GA/PSO's actually
evaluated trajectories, overlaid in one heliocentric 3D view (not just the
single cherry-picked best). Same "show the whole considered spread, not just
the winner" precedent as `plot_porkchop_samples.py`'s narrowing-stage
equivalent.

Trajectories are re-propagated from an already-saved
`<method>_population.csv` -- `mission-planner population <config.toml>`
samples N individuals evenly by index across the *feasible* population in
file order (phase 1 then phase 2, chronological within each), so early
(phase 1, exploring) and late (phase 2, refining) samples are both
represented. This does NOT re-run the GA/PSO search itself -- cheap,
independent of search cost.

Reads: MissionPlanner/out/<mission>/optimize/{ga,pso}_population_sample_trajectories.csv
       (sample_id, phase, generation, fitness, t_s, x_m, y_m, z_m)
Also overlays Sun/departure-body/target-body reference points from
{ga,pso}_departure_geometry_helio.csv, if present (run `geometry` first).

Produces: <method>_population_trajectories.html

Usage (from MissionPlanner/ directory):
  python plot/plot_population_trajectories.py scratch_optimize_repro
"""

import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

mission = sys.argv[1] if len(sys.argv) > 1 else "mars_flyby"

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / mission / "optimize"

method = None
for m in ("ga", "pso"):
    if (DATA / f"{m}_population_sample_trajectories.csv").exists():
        method = m
        break

if method is None:
    print(f"Not found: {DATA}/{{ga,pso}}_population_sample_trajectories.csv")
    print("Run: cargo run --bin mission-planner --release -- population config/<mission>.toml")
    sys.exit(1)

df = pd.read_csv(DATA / f"{method}_population_sample_trajectories.csv")
helio_path = DATA / f"{method}_departure_geometry_helio.csv"
helio = pd.read_csv(helio_path) if helio_path.exists() else None

AU = 1.495978707e11
BG = "#0f0f0f"

fig = go.Figure()

# Colored by phase, not sample index -- phase 1 (broad, from-scratch
# exploration -- expected to scatter widely, including the net-retrograde
# "smaller than Earth's own orbit" half of theta) and phase 2 (narrowed
# real-objective refinement -- expected to cluster tightly once a basin is
# found) should look visibly different; cycling colors by index alone hid
# that distinction.
PHASE_COLORS = {1: "#3a7bd5", 2: "#FF6B35"}
PHASE_WIDTHS = {1: 1, 2: 3}
sample_ids = sorted(df.sample_id.unique())

for sid in sample_ids:
    sub = df[df.sample_id == sid]
    phase = int(sub.phase.iloc[0])
    gen = sub.generation.iloc[0]
    fitness = sub.fitness.iloc[0]
    fig.add_trace(go.Scatter3d(
        x=sub.x_m / AU, y=sub.y_m / AU, z=sub.z_m / AU,
        mode="lines",
        line=dict(color=PHASE_COLORS.get(phase, "#dddddd"), width=PHASE_WIDTHS.get(phase, 2)),
        opacity=0.55 if phase == 1 else 0.9,
        name=f"#{sid}  phase {phase} gen {gen}  fitness={fitness:.3g}",
    ))

if helio is not None:
    helio_idx = helio.set_index("name")
    if "Sun" in helio_idx.index:
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode="markers", marker=dict(size=10, color="#FFD700"), name="Sun",
        ))
    point_colors = {"_at_departure": "#3a7bd5", "_at_arrival": "#FF6B35"}
    for name in helio_idx.index:
        if name == "Sun" or name.startswith("v_infinity_helio"):
            continue
        r = helio_idx.loc[name]
        color = "#dddddd"
        for suffix, c in point_colors.items():
            if name.endswith(suffix):
                color = c
        fig.add_trace(go.Scatter3d(
            x=[r.x_au], y=[r.y_au], z=[r.z_au], mode="markers+text",
            marker=dict(size=7, color=color), text=[name], textposition="top center", name=name,
        ))
else:
    fig.add_annotation(
        text="No Sun/body reference points (run 'geometry' first for those)", showarrow=False,
        x=0.5, y=0.02, xref="paper", yref="paper", font=dict(color="#888888"),
    )

n_phase1 = (df.groupby("sample_id").phase.first() == 1).sum()
n_phase2 = (df.groupby("sample_id").phase.first() == 2).sum()
fig.update_layout(
    title=f"{mission} — {len(sample_ids)} sampled GA/PSO trajectories (heliocentric, AU): "
          f"{n_phase1} phase-1/exploring (blue) vs {n_phase2} phase-2/refining (orange)",
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#dddddd"),
    scene=dict(
        xaxis_title="x [AU]", yaxis_title="y [AU]", zaxis_title="z [AU]",
        xaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        yaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        zaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        aspectmode="data",
    ),
)

out_path = DATA / f"{method}_population_trajectories.html"
fig.write_html(str(out_path))
print(f"Wrote {out_path}")
webbrowser.open(out_path.as_uri())
