#!/usr/bin/env python3
"""
Plot raw control point angles from the optimization database.
This shows the actual angle values at each control point before interpolation.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import sqlite3
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

db_path = Path("out/drag_sail_trajectory.db")
if not db_path.exists():
    print("Error: Database file not found")
    exit(1)

connection = sqlite3.connect(str(db_path))
cursor = connection.cursor()

# Get the latest run
cursor.execute("SELECT run_id FROM optimization_runs ORDER BY datetime DESC LIMIT 1")
run_id = cursor.fetchone()[0]

# Get several samples from different generations
cursor.execute("""
    SELECT generation, input, accepted, energy, eval_id
    FROM generations
    WHERE run_id = ?
    ORDER BY generation ASC
""", (run_id,))
all_data = cursor.fetchall()

# Parse angles from the Debug format string
def parse_angles(input_str):
    pattern = r'elevation:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),\s*direction:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
    matches = re.findall(pattern, input_str)
    elevations = [float(m[0]) for m in matches]
    directions = [float(m[1]) for m in matches]
    return np.array(elevations), np.array(directions)

# Select a few interesting samples to plot
samples_to_plot = []

# First sample (random initial)
if len(all_data) > 0:
    samples_to_plot.append(("Gen 0 (Initial)", all_data[0]))

# Sample from early generation
early_idx = len(all_data) // 4
if early_idx < len(all_data):
    samples_to_plot.append((f"Gen {all_data[early_idx][0]} (Early)", all_data[early_idx]))

# Sample from middle
mid_idx = len(all_data) // 2
if mid_idx < len(all_data):
    samples_to_plot.append((f"Gen {all_data[mid_idx][0]} (Middle)", all_data[mid_idx]))

# Best solution (lowest energy)
best_sample = min(all_data, key=lambda x: x[3])
samples_to_plot.append((f"Gen {best_sample[0]} (Best)", best_sample))

# Final accepted solution
accepted_samples = [d for d in all_data if d[2] == 'true']
if accepted_samples:
    final_accepted = accepted_samples[-1]
    samples_to_plot.append((f"Gen {final_accepted[0]} (Final)", final_accepted))

# Create plots
fig, axes = plt.subplots(len(samples_to_plot), 2, figsize=(14, 4*len(samples_to_plot)))
if len(samples_to_plot) == 1:
    axes = axes.reshape(1, -1)

for idx, (label, data) in enumerate(samples_to_plot):
    generation, input_str, accepted, energy, eval_id = data
    elevations, directions = parse_angles(input_str)

    num_control_points = len(elevations)
    control_points = np.arange(num_control_points)

    # Plot elevation
    axes[idx, 0].scatter(control_points, elevations, alpha=0.7, s=20)
    axes[idx, 0].plot(control_points, elevations, 'b-', alpha=0.3, linewidth=0.5)
    axes[idx, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    axes[idx, 0].set_xlabel('Control Point Index')
    axes[idx, 0].set_ylabel('Elevation (rad)')
    axes[idx, 0].set_title(f'{label} - Elevation\nEnergy: {energy:.4f}, CPs: {num_control_points}')
    axes[idx, 0].grid(True, alpha=0.3)
    axes[idx, 0].set_ylim(-np.pi/2, np.pi/2)

    # Plot direction
    axes[idx, 1].scatter(control_points, directions, alpha=0.7, s=20, color='orange')
    axes[idx, 1].plot(control_points, directions, 'orange', alpha=0.3, linewidth=0.5)
    axes[idx, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    axes[idx, 1].set_xlabel('Control Point Index')
    axes[idx, 1].set_ylabel('Direction (rad)')
    axes[idx, 1].set_title(f'{label} - Direction\nEnergy: {energy:.4f}, CPs: {num_control_points}')
    axes[idx, 1].grid(True, alpha=0.3)
    axes[idx, 1].set_ylim(-np.pi, np.pi)

plt.tight_layout()
plt.savefig('out/raw_control_points.png', dpi=150, bbox_inches='tight')
print(f"Saved plot to out/raw_control_points.png")
print(f"\nTotal control points defined: {num_control_points}")

# Print statistics for the best solution
best_gen, best_input, best_acc, best_energy, best_eval = best_sample
best_elev, best_dir = parse_angles(best_input)

print(f"\nBest solution statistics:")
print(f"  Energy (neg. avg altitude loss rate): {best_energy:.6f}")
print(f"  Control points: {len(best_elev)}")
print(f"  Elevation: mean={best_elev.mean():.3f}, std={best_elev.std():.3f}")
print(f"  Direction: mean={best_dir.mean():.3f}, std={best_dir.std():.3f}")

connection.close()
