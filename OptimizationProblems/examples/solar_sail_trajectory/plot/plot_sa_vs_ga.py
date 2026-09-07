#!/usr/bin/env python3
"""
SA vs GA Comparison Plots.

Visualizes the performance difference between Simulated Annealing and
Genetic Algorithm on the solar sail orbit-raising problem.

Plots:
  1. Convergence curve — best energy found vs total evaluations (fair x-axis)
  2. Energy distribution — all evaluated energies as a violin/box comparison
  3. Best solution angles — final control trajectories side by side
  4. SA acceptance rate over time — shows exploration vs exploitation phases

Usage:
    python plot/plot_sa_vs_ga.py                         # default DB
    DB_PATH=out/compare_sa_ga.db python plot/plot_sa_vs_ga.py
"""

import matplotlib
matplotlib.use('Agg')
import sqlite3
import re
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Load data ────────────────────────────────────────────────────────────

db_path = Path(os.environ.get('DB_PATH', 'out/compare_sa_ga.db'))
if not db_path.exists():
    print(f"Error: Database file not found: {db_path}")
    exit(1)

conn = sqlite3.connect(str(db_path))

# Get runs — expect exactly 2 (SA first, GA second)
runs = conn.execute(
    "SELECT run_id, generation_count FROM optimization_runs ORDER BY id"
).fetchall()

if len(runs) < 2:
    print(f"Expected 2 runs (SA + GA), found {len(runs)}. Run compare_sa_ga first.")
    exit(1)

sa_run_id = runs[0][0]
ga_run_id = runs[1][0]

# SA: all evaluations (accepted and rejected)
sa_rows = conn.execute(
    "SELECT generation, energy, accepted, temperature, input FROM generations WHERE run_id = ? ORDER BY generation",
    (sa_run_id,)
).fetchall()

# GA: best-of-generation entries
ga_rows = conn.execute(
    "SELECT generation, energy, input FROM generations WHERE run_id = ? ORDER BY generation",
    (ga_run_id,)
).fetchall()

conn.close()

sa_gens = np.array([r[0] for r in sa_rows])
sa_energies = np.array([float(r[1]) for r in sa_rows])
sa_accepted = np.array([r[2] == 'true' for r in sa_rows])
sa_temps = np.array([float(r[3]) for r in sa_rows])

ga_gens = np.array([r[0] for r in ga_rows])
ga_energies = np.array([float(r[1]) for r in ga_rows])

# Infer GA population size from the run config line in the output header
# Fallback: check if the optimization_runs table has extra info, otherwise
# count total evals = len(ga_rows) * pop_size? We only have best-per-gen logs.
# We'll estimate population_size from the ratio of total evals we expect.
# For now, read it from the GA config (hardcoded at 50 in compare_sa_ga.rs).
GA_POPULATION_SIZE = 100


# ── Helper: parse angles from debug repr ─────────────────────────────────

def parse_angles(input_str):
    """Parse '[Angles { cone: 0.5, clock: 0.1 }, ...]' into lists of (cone, clock)."""
    cones = [float(m) for m in re.findall(r'cone:\s*([-\d.]+)', input_str)]
    clocks = [float(m) for m in re.findall(r'clock:\s*([-\d.]+)', input_str)]
    return cones, clocks


# ── Figure setup ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SA vs GA — Solar Sail Orbit Raising', fontsize=14, fontweight='bold')

SA_COLOR = '#2196F3'
GA_COLOR = '#FF5722'
SA_LIGHT = '#90CAF9'
GA_LIGHT = '#FFAB91'

# ══════════════════════════════════════════════════════════════════════════
# Plot 1: Convergence (best-so-far vs cumulative evaluations)
# ══════════════════════════════════════════════════════════════════════════
ax = axes[0, 0]

# SA: cumulative best (1 eval per generation)
sa_best_so_far = np.minimum.accumulate(sa_energies)
sa_evals = np.arange(1, len(sa_energies) + 1)

# GA: cumulative best (population_size evals per generation)
ga_best_so_far = np.minimum.accumulate(ga_energies)
ga_evals = (ga_gens + 1) * GA_POPULATION_SIZE

# Convert energy to delta SMA (negate, since energy = -delta_mean_sma)
ax.plot(sa_evals, -sa_best_so_far, color=SA_COLOR, linewidth=1.5, label='SA', alpha=0.9)
ax.plot(ga_evals, -ga_best_so_far, color=GA_COLOR, linewidth=1.5, label='GA', alpha=0.9)

ax.set_xlabel('Total evaluations')
ax.set_ylabel('Best Δ mean SMA (m)')
ax.set_title('Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════════════════
# Plot 2: Energy distribution — SA all evals vs GA best-per-gen
# ══════════════════════════════════════════════════════════════════════════
ax = axes[0, 1]

# Filter out extreme outliers for readability (keep 1st-99th percentile)
sa_pct = np.percentile(sa_energies, [1, 99])
sa_filtered = sa_energies[(sa_energies >= sa_pct[0]) & (sa_energies <= sa_pct[1])]

data = [sa_filtered, ga_energies]
parts = ax.violinplot(data, positions=[0, 1], showmedians=True, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor([SA_LIGHT, GA_LIGHT][i])
    pc.set_edgecolor([SA_COLOR, GA_COLOR][i])
    pc.set_alpha(0.7)
parts['cmedians'].set_color('black')

# Overlay box plots for quartiles
bp = ax.boxplot(data, positions=[0, 1], widths=0.15, patch_artist=True,
                showfliers=False, zorder=3)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor([SA_COLOR, GA_COLOR][i])
    patch.set_alpha(0.5)

ax.set_xticks([0, 1])
ax.set_xticklabels(['SA\n(all evals)', 'GA\n(best/gen)'])
ax.set_ylabel('Energy (lower = better)')
ax.set_title('Energy Distribution')
ax.grid(True, alpha=0.3, axis='y')

# ══════════════════════════════════════════════════════════════════════════
# Plot 3: Best solution — cone & clock angles
# ══════════════════════════════════════════════════════════════════════════
ax = axes[1, 0]

# Get final (best) solution for each algorithm
sa_best_idx = np.argmin(sa_energies)
ga_best_idx = np.argmin(ga_energies)

sa_cones, sa_clocks = parse_angles(sa_rows[sa_best_idx][4])
ga_cones, ga_clocks = parse_angles(ga_rows[ga_best_idx][2])

n_cp = len(sa_cones)
x = np.arange(n_cp)

ax.plot(x, np.degrees(sa_cones), 'o-', color=SA_COLOR, label='SA cone', markersize=5)
ax.plot(x, np.degrees(sa_clocks), 's--', color=SA_COLOR, label='SA clock', markersize=5, alpha=0.6)
ax.plot(x, np.degrees(ga_cones), 'o-', color=GA_COLOR, label='GA cone', markersize=5)
ax.plot(x, np.degrees(ga_clocks), 's--', color=GA_COLOR, label='GA clock', markersize=5, alpha=0.6)

ax.set_xlabel('Control point index')
ax.set_ylabel('Angle (deg)')
ax.set_title('Best Solution — Control Angles')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════════════════
# Plot 4: SA exploration dynamics — acceptance rate + temperature
# ══════════════════════════════════════════════════════════════════════════
ax = axes[1, 1]

# Compute rolling acceptance rate (window of 50 evaluations)
window = min(50, len(sa_accepted) // 4) or 1
rolling_accepted = np.convolve(sa_accepted.astype(float), np.ones(window)/window, mode='valid')
rolling_x = np.arange(window - 1, len(sa_accepted))

ax.plot(rolling_x, rolling_accepted * 100, color=SA_COLOR, linewidth=1.2, label='Acceptance rate')
ax.set_ylabel('Acceptance rate (%)', color=SA_COLOR)
ax.set_ylim(0, 105)
ax.tick_params(axis='y', labelcolor=SA_COLOR)

# Temperature on secondary axis
ax2 = ax.twinx()
ax2.plot(sa_gens, sa_temps, color='gray', linewidth=1, alpha=0.5, label='Temperature')
ax2.set_ylabel('Temperature', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

ax.set_xlabel('SA evaluation')
ax.set_title('SA Exploration Dynamics')
ax.grid(True, alpha=0.3)

# Add legend combining both axes
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# ── Final layout ─────────────────────────────────────────────────────────

# Summary annotation
sa_best_energy = sa_energies[sa_best_idx]
ga_best_energy = ga_energies[ga_best_idx]
winner = 'SA' if sa_best_energy <= ga_best_energy else 'GA'
margin = abs(sa_best_energy - ga_best_energy)
summary = (
    f"SA best: {-sa_best_energy:+.2f} m  |  "
    f"GA best: {-ga_best_energy:+.2f} m  |  "
    f"Winner: {winner} (by {margin:.2f} m)"
)
fig.text(0.5, 0.01, summary, ha='center', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

out_path = Path('out/sa_vs_ga_comparison.png')
plt.savefig(str(out_path), dpi=150)
print(f"Saved: {out_path}")
plt.close()