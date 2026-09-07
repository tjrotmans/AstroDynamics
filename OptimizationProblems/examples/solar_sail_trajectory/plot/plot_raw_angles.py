#!/usr/bin/env python3
"""
Plot optimization performance and control point angle evolution.

This script visualizes:
1. Temperature schedule over iterations
2. Energy (SMA error) over iterations  
3. Acceptance rate over iterations
4. Best energy found over iterations
5. Control angle evolution from initial to final solution

Usage:
    DB_PATH=out/optimization.db python plot/plot_raw_angles.py
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import sqlite3
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

db_path = Path(os.environ.get('DB_PATH', 'out/optimization.db'))
if not db_path.exists():
    print(f"Error: Database file not found: {db_path}")
    exit(1)

connection = sqlite3.connect(str(db_path))
cursor = connection.cursor()

# Get the latest run
cursor.execute("SELECT run_id, initial_energy FROM optimization_runs ORDER BY datetime DESC LIMIT 1")
result = cursor.fetchone()
if result is None:
    print("No optimization runs found in database")
    exit(1)
run_id, initial_energy = result

# Get all generation data including temperature
cursor.execute("""
    SELECT generation, temperature, energy, accepted, input, eval_id
    FROM generations
    WHERE run_id = ?
    ORDER BY generation ASC
""", (run_id,))
all_data = cursor.fetchall()

if len(all_data) == 0:
    print("No generation data found")
    exit(1)

# Extract arrays
generations = np.array([d[0] for d in all_data])
temperatures = np.array([d[1] for d in all_data])
energies = np.array([d[2] for d in all_data])
accepted = np.array([d[3] == 'true' or d[3] == 1 for d in all_data])
inputs = [d[4] for d in all_data]

# Calculate derived metrics
# Best energy found so far (running minimum)
best_so_far = np.minimum.accumulate(energies)

# Acceptance rate (rolling window)
window_size = max(10, len(accepted) // 50)
acceptance_rate = np.convolve(accepted.astype(float), np.ones(window_size)/window_size, mode='valid')
acceptance_gen = generations[window_size-1:]

print(f"Run ID: {run_id}")
print(f"Total iterations: {len(all_data)}")
print(f"Initial energy: {initial_energy:.1f} m" if initial_energy else "Initial energy: N/A")
print(f"Final energy: {energies[-1]:.1f} m")
print(f"Best energy: {energies.min():.1f} m")
print(f"Overall acceptance rate: {accepted.mean()*100:.1f}%")

# Parse angles from the Debug format string (cone/clock format)
def parse_angles(input_str):
    pattern = r'cone:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),\s*clock:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
    matches = re.findall(pattern, input_str)
    cones = [float(m[0]) for m in matches]
    clocks = [float(m[1]) for m in matches]
    return np.array(cones), np.array(clocks)

# ============================================
# FIGURE 1: Optimization Performance Metrics
# ============================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Simulated Annealing Optimization Performance', fontsize=14, fontweight='bold')

# Plot 1: Temperature schedule
ax = axes1[0, 0]
ax.plot(generations, temperatures, 'r-', linewidth=1.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Temperature')
ax.set_title('Temperature Schedule')
ax.grid(True, alpha=0.3)
if temperatures.max() / max(temperatures.min(), 1e-10) > 100:
    ax.set_yscale('log')

# Plot 2: Energy over iterations (with best-so-far)
ax = axes1[0, 1]
ax.scatter(generations, energies, alpha=0.3, s=5, c='blue', label='All evaluations')
ax.plot(generations, best_so_far, 'g-', linewidth=2, label='Best so far')
ax.set_xlabel('Iteration')
ax.set_ylabel('Energy (SMA error in m)')
ax.set_title('Energy Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Acceptance rate over iterations
ax = axes1[1, 0]
ax.plot(acceptance_gen, acceptance_rate * 100, 'purple', linewidth=1.5)
ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% target')
ax.set_xlabel('Iteration')
ax.set_ylabel('Acceptance Rate (%)')
ax.set_title(f'Rolling Acceptance Rate (window={window_size})')
ax.set_ylim(0, 100)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Energy vs Temperature (phase diagram)
ax = axes1[1, 1]
scatter = ax.scatter(temperatures, energies, c=generations, cmap='viridis', alpha=0.5, s=10)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Iteration')
ax.set_xlabel('Temperature')
ax.set_ylabel('Energy (SMA error in m)')
ax.set_title('Energy vs Temperature (Phase Diagram)')
if temperatures.max() / max(temperatures.min(), 1e-10) > 100:
    ax.set_xscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig('out/optimization_performance.png', dpi=150, bbox_inches='tight')
print(f"\nSaved optimization performance plot to: out/optimization_performance.png")

# ============================================
# FIGURE 1b: SMA Evolution Over Iterations
# ============================================
# Get final SMA for each evaluation by querying the final timestep
MU_EARTH = 3.986004418e14  # m³/s²
EARTH_RADIUS = 6.371e6     # m

# Get eval_ids from all_data
eval_ids = [d[5] for d in all_data]

# Query final state for each evaluation (this may be slow for large datasets)
print("\nCalculating SMA for each evaluation...")
final_smas = []
initial_sma = None

for i, eval_id in enumerate(eval_ids):
    # Get the final timestep for this evaluation
    cursor.execute("""
        SELECT x, y, z, vx, vy, vz 
        FROM timesteps 
        WHERE eval_id = ? 
        ORDER BY simtime DESC 
        LIMIT 1
    """, (eval_id,))
    result = cursor.fetchone()
    
    if result:
        x, y, z, vx, vy, vz = result
        r = np.sqrt(x**2 + y**2 + z**2)
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        # Vis-viva equation: v² = μ(2/r - 1/a) → a = μ / (2μ/r - v²)
        sma = MU_EARTH / (2*MU_EARTH/r - v**2)
        final_smas.append(sma)
        
        # Also get initial SMA from first evaluation's first timestep
        if initial_sma is None:
            cursor.execute("""
                SELECT x, y, z, vx, vy, vz 
                FROM timesteps 
                WHERE eval_id = ? 
                ORDER BY simtime ASC 
                LIMIT 1
            """, (eval_id,))
            init_result = cursor.fetchone()
            if init_result:
                x0, y0, z0, vx0, vy0, vz0 = init_result
                r0 = np.sqrt(x0**2 + y0**2 + z0**2)
                v0 = np.sqrt(vx0**2 + vy0**2 + vz0**2)
                initial_sma = MU_EARTH / (2*MU_EARTH/r0 - v0**2)
    else:
        final_smas.append(np.nan)
    
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(eval_ids)} evaluations...")

final_smas = np.array(final_smas)
print(f"  Done! Calculated {np.sum(~np.isnan(final_smas))} SMA values")

# Calculate best SMA so far (running maximum for orbit raising)
best_sma_so_far = np.maximum.accumulate(np.nan_to_num(final_smas, nan=0))

# Create SMA evolution plot
fig1b, axes1b = plt.subplots(2, 2, figsize=(14, 10))
fig1b.suptitle('Semi-Major Axis Evolution During Optimization', fontsize=14, fontweight='bold')

# Plot 1: Final SMA over iterations
ax = axes1b[0, 0]
valid_mask = ~np.isnan(final_smas)
ax.scatter(generations[valid_mask], final_smas[valid_mask]/1e3, alpha=0.3, s=5, c='blue', label='All evaluations')
ax.plot(generations, best_sma_so_far/1e3, 'g-', linewidth=2, label='Best so far')
if initial_sma:
    ax.axhline(y=initial_sma/1e3, color='gray', linestyle='--', alpha=0.7, label=f'Initial: {initial_sma/1e3:.1f} km')
ax.set_xlabel('Iteration')
ax.set_ylabel('Final SMA (km)')
ax.set_title('Final Semi-Major Axis vs Iteration')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: SMA change (delta) over iterations
ax = axes1b[0, 1]
if initial_sma:
    delta_sma = final_smas - initial_sma
    ax.scatter(generations[valid_mask], delta_sma[valid_mask]/1e3, alpha=0.3, s=5, c='blue', label='All evaluations')
    best_delta = best_sma_so_far - initial_sma
    ax.plot(generations, best_delta/1e3, 'g-', linewidth=2, label='Best so far')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('ΔSMA (km)')
else:
    ax.scatter(generations[valid_mask], final_smas[valid_mask]/1e3, alpha=0.3, s=5, c='blue')
    ax.set_ylabel('Final SMA (km)')
ax.set_xlabel('Iteration')
ax.set_title('SMA Change from Initial Orbit')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Altitude (SMA - Earth radius) over iterations  
ax = axes1b[1, 0]
final_altitudes = (final_smas - EARTH_RADIUS) / 1e3  # km
best_altitude = (best_sma_so_far - EARTH_RADIUS) / 1e3
ax.scatter(generations[valid_mask], final_altitudes[valid_mask], alpha=0.3, s=5, c='blue', label='All evaluations')
ax.plot(generations, best_altitude, 'g-', linewidth=2, label='Best so far')
if initial_sma:
    ax.axhline(y=(initial_sma - EARTH_RADIUS)/1e3, color='gray', linestyle='--', alpha=0.7, 
               label=f'Initial: {(initial_sma-EARTH_RADIUS)/1e3:.1f} km')
ax.set_xlabel('Iteration')
ax.set_ylabel('Altitude (km)')
ax.set_title('Final Altitude (SMA - R_Earth)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: SMA vs Temperature
ax = axes1b[1, 1]
scatter = ax.scatter(temperatures[valid_mask], final_smas[valid_mask]/1e3, 
                     c=generations[valid_mask], cmap='viridis', alpha=0.5, s=10)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Iteration')
ax.set_xlabel('Temperature')
ax.set_ylabel('Final SMA (km)')
ax.set_title('SMA vs Temperature')
if temperatures.max() / max(temperatures.min(), 1e-10) > 100:
    ax.set_xscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig1b.savefig('out/sma_evolution.png', dpi=150, bbox_inches='tight')
print(f"Saved SMA evolution plot to: out/sma_evolution.png")

# Print SMA statistics
print(f"\nSMA Statistics:")
print(f"  Initial SMA: {initial_sma/1e3:.2f} km" if initial_sma else "  Initial SMA: N/A")
print(f"  Best final SMA: {np.nanmax(final_smas)/1e3:.2f} km")
print(f"  Best ΔSMA: {(np.nanmax(final_smas) - initial_sma)/1e3:.2f} km" if initial_sma else "")
print(f"  Best altitude gain: {(np.nanmax(final_smas) - initial_sma)/1e3:.2f} km" if initial_sma else "")

# ============================================
# FIGURE 2: Control Angle Evolution
# ============================================

# Select samples at different stages
samples_to_plot = []

# First sample
if len(all_data) > 0:
    samples_to_plot.append(("Initial (Gen 0)", all_data[0]))

# Sample at 25%
idx_25 = len(all_data) // 4
if idx_25 > 0:
    samples_to_plot.append((f"25% (Gen {all_data[idx_25][0]})", all_data[idx_25]))

# Sample at 50%
idx_50 = len(all_data) // 2
if idx_50 > 0:
    samples_to_plot.append((f"50% (Gen {all_data[idx_50][0]})", all_data[idx_50]))

# Best solution
best_idx = np.argmin(energies)
best_sample = all_data[best_idx]
samples_to_plot.append((f"Best (Gen {best_sample[0]}, E={best_sample[2]:.0f}m)", best_sample))

# Final solution
final_sample = all_data[-1]
if final_sample != best_sample:
    samples_to_plot.append((f"Final (Gen {final_sample[0]})", final_sample))

fig2, axes2 = plt.subplots(len(samples_to_plot), 2, figsize=(14, 4*len(samples_to_plot)))
fig2.suptitle('Control Angle Evolution During Optimization', fontsize=14, fontweight='bold')

if len(samples_to_plot) == 1:
    axes2 = axes2.reshape(1, -1)

for idx, (label, data) in enumerate(samples_to_plot):
    generation, temp, energy, acc, input_str, eval_id = data
    cones, clocks = parse_angles(input_str)
    
    if len(cones) == 0:
        print(f"Warning: Could not parse angles for {label}")
        continue
    
    num_control_points = len(cones)
    control_points = np.arange(num_control_points)
    
    # Plot cone angle
    axes2[idx, 0].scatter(control_points, np.degrees(cones), alpha=0.7, s=30)
    axes2[idx, 0].plot(control_points, np.degrees(cones), 'b-', alpha=0.5, linewidth=1)
    axes2[idx, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    axes2[idx, 0].axhline(y=35.26, color='g', linestyle=':', alpha=0.7, linewidth=2, label='Optimal ~35°')
    axes2[idx, 0].set_xlabel('Control Point Index')
    axes2[idx, 0].set_ylabel('Cone Angle (degrees)')
    axes2[idx, 0].set_title(f'{label} - Cone Angle (α)')
    axes2[idx, 0].grid(True, alpha=0.3)
    axes2[idx, 0].set_ylim(-90, 90)
    axes2[idx, 0].legend(loc='upper right')
    
    # Plot clock angle
    axes2[idx, 1].scatter(control_points, np.degrees(clocks), alpha=0.7, s=30, color='orange')
    axes2[idx, 1].plot(control_points, np.degrees(clocks), 'orange', alpha=0.5, linewidth=1)
    axes2[idx, 1].axhline(y=90, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    axes2[idx, 1].set_xlabel('Control Point Index')
    axes2[idx, 1].set_ylabel('Clock Angle (degrees)')
    axes2[idx, 1].set_title(f'{label} - Clock Angle (δ)')
    axes2[idx, 1].grid(True, alpha=0.3)
    axes2[idx, 1].set_ylim(0, 180)

plt.tight_layout()
fig2.savefig('out/control_angle_evolution.png', dpi=150, bbox_inches='tight')
print(f"Saved control angle evolution plot to: out/control_angle_evolution.png")

# ============================================
# FIGURE 3: Energy Histogram & Statistics
# ============================================
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle('Energy Distribution Analysis', fontsize=14, fontweight='bold')

# Histogram of all energies
ax = axes3[0]
ax.hist(energies, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax.axvline(x=energies.min(), color='g', linestyle='--', linewidth=2, label=f'Best: {energies.min():.0f}m')
ax.axvline(x=energies.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {energies.mean():.0f}m')
ax.set_xlabel('Energy (SMA error in m)')
ax.set_ylabel('Frequency')
ax.set_title('Energy Distribution (All Evaluations)')
ax.legend()
ax.grid(True, alpha=0.3)

# Histogram of accepted energies only
ax = axes3[1]
accepted_energies = energies[accepted]
if len(accepted_energies) > 0:
    ax.hist(accepted_energies, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(x=accepted_energies.min(), color='darkgreen', linestyle='--', linewidth=2, 
               label=f'Best: {accepted_energies.min():.0f}m')
    ax.axvline(x=accepted_energies.mean(), color='r', linestyle='--', linewidth=2, 
               label=f'Mean: {accepted_energies.mean():.0f}m')
ax.set_xlabel('Energy (SMA error in m)')
ax.set_ylabel('Frequency')
ax.set_title('Energy Distribution (Accepted Only)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig3.savefig('out/energy_distribution.png', dpi=150, bbox_inches='tight')
print(f"Saved energy distribution plot to: out/energy_distribution.png")

# ============================================
# Print Summary Statistics
# ============================================
print("\n" + "="*50)
print("OPTIMIZATION SUMMARY")
print("="*50)
print(f"Total iterations: {len(generations)}")
print(f"Temperature range: {temperatures.min():.4f} - {temperatures.max():.4f}")
print(f"Energy range: {energies.min():.1f} - {energies.max():.1f} m")
print(f"Improvement: {energies[0]:.1f} -> {best_so_far[-1]:.1f} m ({(1-best_so_far[-1]/energies[0])*100:.1f}% reduction)")
print(f"Acceptance rate: {accepted.mean()*100:.1f}%")

# Best solution details
best_cones, best_clocks = parse_angles(best_sample[4])
if len(best_cones) > 0:
    print(f"\nBest Solution (Gen {best_sample[0]}):")
    print(f"  Energy: {best_sample[2]:.1f} m")
    print(f"  Cone angle:  mean={np.degrees(best_cones.mean()):.1f}°, std={np.degrees(best_cones.std()):.1f}°")
    print(f"  Clock angle: mean={np.degrees(best_clocks.mean()):.1f}°, std={np.degrees(best_clocks.std()):.1f}°")
    print(f"  (Theoretical optimal cone for max radial thrust: ~35.26°)")

connection.close()
