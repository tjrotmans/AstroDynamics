import sqlite3
import re
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

db_path = Path("out/drag_sail_trajectory.db")
if not db_path.exists():
    print("Error: Database file not found at 'out/drag_sail_trajectory.db'")
    print("Please run 'cargo run --release' first to generate optimization results.")
    exit(1)

connection = sqlite3.connect(str(db_path))
cursor = connection.cursor()

cursor.execute("SELECT run_id, generation_count FROM optimization_runs ORDER BY datetime DESC LIMIT 1")
run_result = cursor.fetchone()
if not run_result:
    print("Error: No optimization runs found in database.")
    exit(1)
run_id, gen_count = run_result

cursor.execute("""
    SELECT generation, temperature, energy, accepted
    FROM generations
    WHERE run_id = ?
    ORDER BY generation ASC
""", (run_id,))
gen_data = cursor.fetchall()

cursor.execute(
    """
    SELECT generation, energy, input
    FROM generations
    WHERE run_id = ? AND accepted = 'true'
    ORDER BY CAST(generation AS INTEGER) ASC
    """,
    (run_id,),
)
gen_inputs = cursor.fetchall()

cursor.execute("""
    SELECT eval_id, input, energy
    FROM generations
    WHERE run_id = ?
    ORDER BY energy ASC
    LIMIT 1
""", (run_id,))
best_solution = cursor.fetchone()
if not best_solution:
    print("Error: No generations found for latest run.")
    exit(1)
best_eval_id, best_input_str, best_energy = best_solution

cursor.execute("""
    SELECT simtime, x, y, z, vx, vy, vz
    FROM timesteps
    WHERE eval_id = ?
    ORDER BY simtime ASC
""", (best_eval_id,))
traj_data = cursor.fetchall()
connection.close()

if not gen_data or not traj_data:
    print("Error: No data found for optimization run.")
    exit(1)

def parse_angles(input_str):
    pattern = r'elevation:\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?),\s*direction:\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
    matches = re.findall(pattern, input_str)
    elevations = [float(m[0]) for m in matches]
    directions = [float(m[2]) for m in matches]
    return elevations, directions


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode='same')


def wrap_pi(arr: np.ndarray) -> np.ndarray:
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def sample_angles_over_time(
    elevations_cp: np.ndarray,
    directions_cp: np.ndarray,
    sim_times_s: np.ndarray,
    time_cutoff_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if elevations_cp.size < 2 or directions_cp.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    n = elevations_cp.size

    idx_f = (sim_times_s * float(n - 2)) / float(time_cutoff_s)
    idx_f = np.clip(idx_f, 0.0, float(n - 2) - 1e-12)

    i0 = np.floor(idx_f).astype(int)
    i1 = i0 + 1
    frac = idx_f - i0

    elev = elevations_cp[i0] + (elevations_cp[i1] - elevations_cp[i0]) * frac

    direc_unwrapped = np.unwrap(directions_cp)
    direc = direc_unwrapped[i0] + (direc_unwrapped[i1] - direc_unwrapped[i0]) * frac
    direc = wrap_pi(direc)

    return elev, direc

generations = [row[0] for row in gen_data]
temperatures = [row[1] for row in gen_data]
energies = np.array([row[2] for row in gen_data])

times = np.array([row[0] for row in traj_data])
positions = np.array([[row[1], row[2], row[3]] for row in traj_data])
velocities = np.array([[row[4], row[5], row[6]] for row in traj_data])

earth_radius = 6.371e6
altitudes = np.linalg.norm(positions, axis=1) - earth_radius
deorbit_cutoff = 300e3

# Derive actual sim time from the best trajectory's timestep data
best_sim_time = float(times[-1]) if len(times) > 0 else 0.0

TIME_CUTOFF_SECONDS = 60.0 * 60.0 * 24.0 * 10.0  # 10 days (must match main.rs)

if best_input_str:
    best_elevations, best_directions = parse_angles(best_input_str)
    best_elevations = np.asarray(best_elevations, dtype=float)
    best_directions = np.asarray(best_directions, dtype=float)
    elev_used, direc_used = sample_angles_over_time(
        best_elevations, best_directions, times, TIME_CUTOFF_SECONDS
    )
else:
    best_elevations = np.asarray([], dtype=float)
    best_directions = np.asarray([], dtype=float)
    elev_used = np.asarray([], dtype=float)
    direc_used = np.asarray([], dtype=float)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

axes[0, 0].plot(generations, temperatures, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Generation')
axes[0, 0].set_ylabel('Temperature')
axes[0, 0].set_title('Temperature Schedule')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(generations, energies, 'r-', linewidth=1, alpha=0.7)
best_e = min(energies)
axes[0, 1].axhline(y=best_e, color='g', linestyle='--', label=f'Best: {best_e:.6f}')
axes[0, 1].set_xlabel('Generation')
axes[0, 1].set_ylabel('Energy (neg. avg alt loss rate, m/s)')
axes[0, 1].set_title('Energy Optimization')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(times / 3600, altitudes / 1e3, 'b*', linewidth=2)
axes[1, 0].axhline(y=deorbit_cutoff / 1e3, color='r', linestyle='--', label=f'Deorbit cutoff: {deorbit_cutoff/1e3:.0f} km')
axes[1, 0].set_xlabel('Time (hours)')
axes[1, 0].set_ylabel('Altitude (km)')
axes[1, 0].set_title(f'Best Trajectory (Sim time: {best_sim_time/3600:.2f} hours, Energy: {best_energy:.6f})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

speeds = np.linalg.norm(velocities, axis=1)
axes[1, 1].plot(times / 3600, speeds / 1e3, 'g-', linewidth=2)
axes[1, 1].set_xlabel('Time (hours)')
axes[1, 1].set_ylabel('Speed (km/s)')
axes[1, 1].set_title('Orbital Speed vs Time')
axes[1, 1].grid(True, alpha=0.3)

if elev_used.size > 0:
    axes[2, 0].plot(times / 3600, np.degrees(elev_used), 'purple', linewidth=1)
    axes[2, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2, 0].set_xlabel('Time (hours)')
    axes[2, 0].set_ylabel('Elevation (degrees)')
    axes[2, 0].set_title('Sail Elevation Angle')
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(times / 3600, np.degrees(direc_used), 'orange', linewidth=1)
    axes[2, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2, 1].set_xlabel('Time (hours)')
    axes[2, 1].set_ylabel('Direction (degrees)')
    axes[2, 1].set_title('Sail Direction Angle')
    axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()

avg_elev_deg = []
avg_dir_deg = []
iter_idx = []

samples = 800
for (gen, energy, input_str) in gen_inputs:
    elev_cp, dir_cp = parse_angles(input_str)
    elev_cp = np.asarray(elev_cp, dtype=float)
    dir_cp = np.asarray(dir_cp, dtype=float)
    if elev_cp.size == 0:
        continue

    # Sample angles over the full time cutoff (all control points are used)
    sim_t = np.linspace(0.0, TIME_CUTOFF_SECONDS, samples, dtype=float)
    elev_t, dir_t = sample_angles_over_time(elev_cp, dir_cp, sim_t, TIME_CUTOFF_SECONDS)
    if elev_t.size == 0:
        continue

    avg_elev = float(np.mean(elev_t))
    s = float(np.mean(np.sin(dir_t)))
    c = float(np.mean(np.cos(dir_t)))
    avg_dir = float(np.arctan2(s, c))

    try:
        gen_num = int(gen)
    except (ValueError, TypeError):
        gen_num = None
    if gen_num is None:
        continue
    iter_idx.append(gen_num)
    avg_elev_deg.append(np.degrees(avg_elev))
    avg_dir_deg.append(np.degrees(avg_dir))

fig_avg, ax_avg = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
ax_avg[0].plot(iter_idx, avg_elev_deg, linewidth=1.2)
ax_avg[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax_avg[0].set_ylabel('Avg Elevation (deg)')
ax_avg[0].set_title('Average Sail Angles per Iteration')
ax_avg[0].grid(True, alpha=0.3)

ax_avg[1].plot(iter_idx, avg_dir_deg, linewidth=1.2)
ax_avg[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax_avg[1].set_xlabel('Generation')
ax_avg[1].set_ylabel('Avg Direction (deg)')
ax_avg[1].grid(True, alpha=0.3)

plt.tight_layout()

plt.show()