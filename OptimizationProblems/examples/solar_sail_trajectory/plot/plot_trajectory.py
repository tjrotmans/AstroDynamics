#!/usr/bin/env python3
"""
Plot solar sail orbit raising trajectory results.

This script visualizes the optimization results showing:
- 3D trajectory visualization
- Semi-major axis evolution over time
- Altitude evolution over time  
- Control angles (elevation and direction) over time

Usage:
    python plot_trajectory.py [--plots PLOT1 PLOT2 ...] [--reference]
    
Available plots:
    scatter     - 3D scatter plot colored by time (commented out by default)
    kepler      - 6-panel Keplerian orbital elements evolution
    angles      - Control angles (elevation and direction) over time
    sail3d      - 3D trajectory with sail normal vectors (first, middle, last orbit)
    trajectory  - SMA and altitude time series (commented out by default)
    
Options:
    --reference - Overlay the locally optimal reference trajectory (from out/reference.db)
    
Examples:
    python plot_trajectory.py                    # Plot all enabled plots
    python plot_trajectory.py --plots kepler angles     # Kepler and control angles
    python plot_trajectory.py --plots kepler sail3d angles  # Multiple plots
    python plot_trajectory.py --plots kepler --reference  # Compare with reference
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional

# Database connection
DB_PATH = os.environ.get('DB_PATH', "../out/optimization.db")
REFERENCE_DB_PATH = os.environ.get('REFERENCE_DB_PATH', "out/reference.db")

# Global flag for reference plotting
PLOT_REFERENCE = False

def get_solution_eval_id(label: str) -> Optional[str]:
    """Get the eval_id for a labeled solution (sa_best or refined_best)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT eval_id FROM solutions WHERE label = ? ORDER BY id DESC LIMIT 1
        """, (label,))
        result = cursor.fetchone()
    except sqlite3.OperationalError:
        result = None
    conn.close()
    return result[0] if result else None

def has_refinement() -> bool:
    """Check if a refined solution exists in the database."""
    return get_solution_eval_id('refined_best') is not None

def get_trajectory_by_eval_id(eval_id: str):
    """Load trajectory data for a specific eval_id. Returns same format as get_best_trajectory."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT simtime, x, y, z, vx, vy, vz, cone, clock
        FROM (
            SELECT *, ROW_NUMBER() OVER (ORDER BY simtime) as row_num
            FROM timesteps
            WHERE eval_id = ?
        )
        WHERE row_num % 1 = 0
        ORDER BY simtime
    """, (eval_id,))
    
    data = np.array(cursor.fetchall())
    conn.close()
    
    if len(data) == 0:
        return None
    
    times = data[:, 0]
    positions = data[:, 1:4]
    velocities = data[:, 4:7]
    cones = data[:, 7]
    clocks = data[:, 8]
    
    MU_EARTH = 3.986004418e14
    EARTH_RADIUS = 6.371e6
    
    n_points = len(positions)
    orbital_elements = np.zeros((n_points, 6))
    
    for idx in range(n_points):
        r_vec = positions[idx]
        v_vec = velocities[idx]
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        a = -MU_EARTH / (v**2 - 2*MU_EARTH/r)
        
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        
        e_vec = (1/MU_EARTH) * ((v**2 - MU_EARTH/r) * r_vec - np.dot(r_vec, v_vec) * v_vec)
        e = np.linalg.norm(e_vec)
        
        inc = np.arccos(np.clip(h_vec[2] / max(h, 1e-10), -1, 1))
        
        n_vec = np.cross([0, 0, 1], h_vec)
        n = np.linalg.norm(n_vec)
        
        if n > 1e-10:
            o = np.arccos(np.clip(n_vec[0] / n, -1, 1))
            if n_vec[1] < 0:
                o = 2 * np.pi - o
        else:
            o = 0.0
        
        if n > 1e-10 and e > 1e-10:
            w = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1))
            if e_vec[2] < 0:
                w = 2 * np.pi - w
        else:
            w = 0.0
        
        if e > 1e-10:
            nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1))
            if np.dot(r_vec, v_vec) < 0:
                nu = 2 * np.pi - nu
        else:
            nu = 0.0
        
        orbital_elements[idx] = [a, e, inc, w, o, nu]
    
    sma = orbital_elements[:, 0]
    altitude = np.linalg.norm(positions, axis=1) - EARTH_RADIUS
    
    return times, sma, altitude, positions, velocities, orbital_elements, cones, clocks

def get_sa_trajectory():
    """Get the SA-best trajectory (before refinement)."""
    eval_id = get_solution_eval_id('sa_best')
    if eval_id is None:
        return None
    return get_trajectory_by_eval_id(eval_id)

def get_refined_trajectory():
    """Get the refined trajectory (after local refinement)."""
    eval_id = get_solution_eval_id('refined_best')
    if eval_id is None:
        return None
    return get_trajectory_by_eval_id(eval_id)


def get_best_trajectory():
    """Get the best trajectory — refined if available, otherwise SA best, otherwise best from generations."""
    # Try refined first
    if has_refinement():
        result = get_refined_trajectory()
        if result is not None:
            print("Using refined trajectory as best")
            return result
    
    # Try SA best
    sa = get_sa_trajectory()
    if sa is not None:
        print("Using SA-best trajectory as best")
        return sa
    
    # Fallback: best energy from generations table
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT eval_id, energy FROM generations 
        WHERE accepted = 'true'
        ORDER BY CAST(energy AS REAL) ASC 
        LIMIT 1
    """)
    result = cursor.fetchone()
    conn.close()
    
    if result is None:
        raise RuntimeError("No trajectory data found in database!")
    
    eval_id, energy = result
    print(f"Using best generation trajectory: eval_id={eval_id}, energy={energy}")
    data = get_trajectory_by_eval_id(eval_id)
    if data is None:
        raise RuntimeError(f"No timestep data found for eval_id={eval_id}. Was logging enabled?")
    return data


def get_reference_trajectory() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Get the reference trajectory from the reference database (locally optimal steering)."""
    if not os.path.exists(REFERENCE_DB_PATH):
        print(f"Reference database not found: {REFERENCE_DB_PATH}")
        print("Run 'cargo run --release --bin quick_reference' to generate it.")
        return None
    
    conn = sqlite3.connect(REFERENCE_DB_PATH)
    cursor = conn.cursor()
    
    # Get the reference evaluation
    cursor.execute("""
        SELECT eval_id, energy 
        FROM generations 
        LIMIT 1
    """)
    result = cursor.fetchone()
    if result is None:
        print("No reference trajectory found in database!")
        conn.close()
        return None
        
    ref_eval_id, ref_energy = result
    
    print(f"\nReference trajectory (locally optimal): {ref_eval_id}")
    print(f"Reference Δmean SMA: {-ref_energy:.2f} m")
    
    # Get trajectory timesteps with control angles
    cursor.execute("""
        SELECT simtime, x, y, z, vx, vy, vz, cone, clock
        FROM timesteps
        WHERE eval_id = ?
        ORDER BY simtime
    """, (ref_eval_id,))
    
    data = np.array(cursor.fetchall())
    conn.close()
    
    if len(data) == 0:
        print(f"No trajectory data found for reference eval_id: {ref_eval_id}")
        return None
    
    print(f"Loaded {len(data)} reference timesteps")
    
    times = data[:, 0]
    positions = data[:, 1:4]
    velocities = data[:, 4:7]
    cones = data[:, 7]
    clocks = data[:, 8]
    
    # Calculate orbital elements
    MU_EARTH = 3.986004418e14  # m³/s²
    EARTH_RADIUS = 6.371e6     # m
    
    n_points = len(positions)
    orbital_elements = np.zeros((n_points, 6))  # [a, e, i, w, o, nu]
    
    for idx in range(n_points):
        r_vec = positions[idx]
        v_vec = velocities[idx]
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        
        # Semi-major axis
        a = -MU_EARTH / (v**2 - 2*MU_EARTH/r)
        
        # Angular momentum vector
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        
        # Eccentricity vector
        e_vec = np.cross(v_vec, h_vec) / MU_EARTH - r_vec / r
        e = np.linalg.norm(e_vec)
        
        # Inclination
        i = np.arccos(h_vec[2] / h)
        
        # Node vector
        n_vec = np.cross([0, 0, 1], h_vec)
        n = np.linalg.norm(n_vec)
        
        # Longitude of ascending node
        if n > 1e-10:
            o = np.arccos(n_vec[0] / n)
            if n_vec[1] < 0:
                o = 2 * np.pi - o
        else:
            o = 0.0
        
        # Argument of periapsis
        if n > 1e-10 and e > 1e-10:
            w = np.arccos(np.dot(n_vec, e_vec) / (n * e))
            if e_vec[2] < 0:
                w = 2 * np.pi - w
        else:
            w = 0.0
        
        # True anomaly
        if e > 1e-10:
            nu = np.arccos(np.dot(e_vec, r_vec) / (e * r))
            if np.dot(r_vec, v_vec) < 0:
                nu = 2 * np.pi - nu
        else:
            nu = 0.0
        
        orbital_elements[idx] = [a, e, i, w, o, nu]
    
    sma = orbital_elements[:, 0]
    altitude = np.linalg.norm(positions, axis=1) - EARTH_RADIUS
    
    return times, sma, altitude, positions, velocities, orbital_elements, cones, clocks


def plot_3d_trajectory():
    """Create 3D scatter plot of the orbit trajectory colored by time."""
    times, sma, altitude, positions, velocities, orbital_elements, cones, clocks = get_best_trajectory()
    
    # Downsample for 3D scatter plot (target ~5000 points)
    downsample_factor = max(1, len(times) // 5000)
    times = times[::downsample_factor]
    positions = positions[::downsample_factor]
    
    # Convert positions to km
    positions_km = positions / 1e3
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Earth as a sphere
    EARTH_RADIUS_KM = 6371
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_earth = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y_earth = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z_earth = EARTH_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth, color='cyan', alpha=0.3, edgecolor='none')
    
    # Plot trajectory as scatter points colored by time
    scatter = ax.scatter(positions_km[:, 0], positions_km[:, 1], positions_km[:, 2],
                        c=times, cmap='plasma', s=20, alpha=0.6)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Time (seconds)', fontsize=11)
    
    # Mark start and end points
    ax.scatter([positions_km[0, 0]], [positions_km[0, 1]], [positions_km[0, 2]],
              color='green', s=150, marker='o', label='Start', edgecolors='black', linewidths=2)
    ax.scatter([positions_km[-1, 0]], [positions_km[-1, 1]], [positions_km[-1, 2]],
              color='red', s=150, marker='s', label='End', edgecolors='black', linewidths=2)
    
    ax.legend(fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_zlabel('Z (km)', fontsize=12)
    ax.set_title('Solar Sail Trajectory (3D)', fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = np.array([positions_km[:, 0].max()-positions_km[:, 0].min(),
                         positions_km[:, 1].max()-positions_km[:, 1].min(),
                         positions_km[:, 2].max()-positions_km[:, 2].min()]).max() / 2.0
    mid_x = (positions_km[:, 0].max()+positions_km[:, 0].min()) * 0.5
    mid_y = (positions_km[:, 1].max()+positions_km[:, 1].min()) * 0.5
    mid_z = (positions_km[:, 2].max()+positions_km[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('out/trajectory_3d_scatter.png', dpi=150, bbox_inches='tight')
    print("\n3D scatter plot saved to: out/trajectory_3d_scatter.png")


def plot_orbital_elements():
    """Create plots of all 6 Keplerian orbital elements over time."""
    times, sma, altitude, positions, velocities, orbital_elements, cones, clocks = get_best_trajectory()
    
    # Downsample for Keplerian element plots (too many points otherwise)
    downsample_factor = max(1, len(times) // 10000)  # Target ~10k points max
    times = times[::downsample_factor]
    orbital_elements = orbital_elements[::downsample_factor]
    
    # Convert time to days
    times_days = times / (60 * 60 * 24)
    
    # Extract orbital elements
    a = orbital_elements[:, 0]  # Semi-major axis (m)
    e = orbital_elements[:, 1]  # Eccentricity
    i = orbital_elements[:, 2]  # Inclination (rad)
    w = orbital_elements[:, 3]  # Argument of periapsis (rad)
    o = orbital_elements[:, 4]  # Longitude of ascending node (rad)
    nu = orbital_elements[:, 5] # True anomaly (rad)
    
    # Compute running mean SMA
    mean_sma = np.cumsum(a) / np.arange(1, len(a) + 1)
    
    # Load reference trajectory if requested
    ref_data = None
    if PLOT_REFERENCE:
        ref_data = get_reference_trajectory()
    
    # Load SA-only trajectory for comparison (if refinement was used)
    sa_data = None
    if has_refinement():
        sa_data = get_sa_trajectory()
        if sa_data and len(sa_data[0]) > 10:
            print("  Overlaying SA-best trajectory (before refinement)")
    
    # Create figure with 6 subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    title_suffix = ' (with Reference)' if ref_data else ''
    if sa_data:
        title_suffix += ' (SA vs Refined)'
    fig.suptitle(f'Keplerian Orbital Elements Evolution{title_suffix}', fontsize=16, fontweight='bold')
    
    # Plot 1: Semi-major axis with running mean
    ax = axes[0, 0]
    ax.plot(times_days, a / 1e3, 'b-', linewidth=1, alpha=0.6, label='Instantaneous')
    ax.plot(times_days, mean_sma / 1e3, 'r-', linewidth=2, label='Running Mean')
    
    if sa_data and len(sa_data[0]) > 10:
        sa_times_days = sa_data[0] / (60*60*24)
        sa_ds = max(1, len(sa_times_days) // 10000)
        sa_a = sa_data[5][::sa_ds, 0]
        sa_mean = np.cumsum(sa_a) / np.arange(1, len(sa_a) + 1)
        ax.plot(sa_times_days[::sa_ds], sa_mean / 1e3, 'm--', linewidth=2, alpha=0.7, 
                label=f'SA Best Mean (Δ={((sa_mean[-1]-sa_a[0])/1e3):+.2f} km)')
    
    if ref_data:
        ref_times, ref_sma, _, _, _, ref_oe, _, _ = ref_data
        ref_times_days = ref_times / (60 * 60 * 24)
        ref_a = ref_oe[:, 0]
        ref_mean_sma = np.cumsum(ref_a) / np.arange(1, len(ref_a) + 1)
        ax.plot(ref_times_days, ref_a / 1e3, 'g-', linewidth=1, alpha=0.4, label='Ref Instantaneous')
        ax.plot(ref_times_days, ref_mean_sma / 1e3, 'g--', linewidth=2, label=f'Ref Mean (Δ={((ref_mean_sma[-1]-ref_a[0])/1e3):+.2f} km)')
    
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Semi-Major Axis (km)', fontsize=11)
    ax.set_title(f'(a) Semi-Major Axis (Δmean={((mean_sma[-1]-a[0])/1e3):+.2f} km)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Eccentricity
    ax = axes[0, 1]
    ax.plot(times_days, e, 'r-', linewidth=2, label='Optimized')
    if sa_data and len(sa_data[0]) > 10:
        sa_e = sa_data[5][::sa_ds, 1]
        ax.plot(sa_times_days[::sa_ds], sa_e, 'm--', linewidth=2, alpha=0.7, label='SA Best')
    if ref_data:
        ax.plot(ref_times_days, ref_oe[:, 1], 'g--', linewidth=2, label='Reference')
        ax.legend(loc='best', fontsize=9)
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Eccentricity', fontsize=11)
    ax.set_title('(e) Eccentricity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Inclination
    ax = axes[1, 0]
    ax.plot(times_days, np.degrees(i), 'g-', linewidth=2, label='Optimized')
    if sa_data and len(sa_data[0]) > 10:
        sa_i = np.degrees(sa_data[5][::sa_ds, 2])
        ax.plot(sa_times_days[::sa_ds], sa_i, 'm--', linewidth=2, alpha=0.7, label='SA Best')
    if ref_data:
        ax.plot(ref_times_days, np.degrees(ref_oe[:, 2]), 'orange', linestyle='--', linewidth=2, label='Reference')
        ax.legend(loc='best', fontsize=9)
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Inclination (degrees)', fontsize=11)
    ax.set_title('(i) Inclination', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Argument of periapsis
    ax = axes[1, 1]
    ax.plot(times_days, np.degrees(w), 'm-', linewidth=2, label='Optimized')
    if sa_data and len(sa_data[0]) > 10:
        sa_w = np.degrees(sa_data[5][::sa_ds, 3])
        ax.plot(sa_times_days[::sa_ds], sa_w, 'm--', linewidth=2, alpha=0.7, label='SA Best')
    if ref_data:
        ax.plot(ref_times_days, np.degrees(ref_oe[:, 3]), 'g--', linewidth=2, label='Reference')
        ax.legend(loc='best', fontsize=9)
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Arg. of Periapsis (degrees)', fontsize=11)
    ax.set_title('(ω) Argument of Periapsis', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Longitude of ascending node
    ax = axes[2, 0]
    ax.plot(times_days, np.degrees(o), 'c-', linewidth=2, label='Optimized')
    if sa_data and len(sa_data[0]) > 10:
        sa_o = np.degrees(sa_data[5][::sa_ds, 4])
        ax.plot(sa_times_days[::sa_ds], sa_o, 'm--', linewidth=2, alpha=0.7, label='SA Best')
    if ref_data:
        ax.plot(ref_times_days, np.degrees(ref_oe[:, 4]), 'g--', linewidth=2, label='Reference')
        ax.legend(loc='best', fontsize=9)
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Long. of Asc. Node (degrees)', fontsize=11)
    ax.set_title('(Ω) Longitude of Ascending Node', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: True anomaly
    ax = axes[2, 1]
    ax.plot(times_days, np.degrees(nu), 'orange', linewidth=2, label='Optimized')
    if sa_data and len(sa_data[0]) > 10:
        sa_nu = np.degrees(sa_data[5][::sa_ds, 5])
        ax.plot(sa_times_days[::sa_ds], sa_nu, 'm--', linewidth=2, alpha=0.7, label='SA Best')
    if ref_data:
        ax.plot(ref_times_days, np.degrees(ref_oe[:, 5]), 'g--', linewidth=2, label='Reference')
        ax.legend(loc='best', fontsize=9)
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('True Anomaly (degrees)', fontsize=11)
    ax.set_title('(ν) True Anomaly', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('out/orbital_elements.png', dpi=150, bbox_inches='tight')
    print("\nOrbital elements plot saved to: out/orbital_elements.png")


def plot_control_angles():
    """Create plots of control angles (cone and clock) over time."""
    times, sma, altitude, positions, velocities, orbital_elements, cones, clocks = get_best_trajectory()
    
    # Downsample for control angle plots
    downsample_factor = max(1, len(times) // 10000)  # Target ~10k points max
    times = times[::downsample_factor]
    cones = cones[::downsample_factor]
    clocks = clocks[::downsample_factor]
    
    # Convert time to days
    times_days = times / (60 * 60 * 24)
    
    # Load reference trajectory if requested
    ref_data = None
    if PLOT_REFERENCE:
        ref_data = get_reference_trajectory()
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    title_suffix = ' (with Reference)' if ref_data else ''
    fig.suptitle(f'Solar Sail Control Angles Evolution (Cone-Clock Frame){title_suffix}', fontsize=16, fontweight='bold')
    
    # Plot 1: Cone angle
    ax = axes[0]
    ax.plot(times_days, np.degrees(cones), 'b-', linewidth=1.5, label='Optimized')
    if ref_data:
        ref_times, _, _, _, _, _, ref_cones, _ = ref_data
        ref_times_days = ref_times / (60 * 60 * 24)
        ax.plot(ref_times_days, np.degrees(ref_cones), 'g--', linewidth=2, label='Reference (Locally Optimal)')
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Cone Angle α (degrees)', fontsize=11)
    ax.set_title('Cone Angle: Angle Between Sail Normal and Sun Vector', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='α=0° (max thrust)')
    ax.axhline(y=90, color='r', linestyle=':', alpha=0.5, linewidth=1, label='α=90° (zero thrust)')
    ax.legend(loc='best', fontsize=9)
    
    # Plot 2: Clock angle
    ax = axes[1]
    ax.plot(times_days, np.degrees(clocks), 'b-', linewidth=1.5, label='Optimized')
    if ref_data:
        _, _, _, _, _, _, _, ref_clocks = ref_data
        ax.plot(ref_times_days, np.degrees(ref_clocks), 'g--', linewidth=2, label='Reference (Locally Optimal)')
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Clock Angle δ (degrees)', fontsize=11)
    ax.set_title('Clock Angle: Rotation Around Sun Vector', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle=':', alpha=0.5, linewidth=1, label='δ=0° (in-plane)')
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('out/control_angles.png', dpi=150, bbox_inches='tight')
    print("\nControl angles plot saved to: out/control_angles.png")


def plot_3d_with_sail_normals():
    """Create 3D plot showing first orbit, middle orbit, and last orbit with sail normal vectors."""
    times, sma, altitude, positions, velocities, orbital_elements, cones, clocks = get_best_trajectory()
    
    # Load reference trajectory if requested
    ref_data = None
    if PLOT_REFERENCE:
        ref_data = get_reference_trajectory()
        if ref_data is None:
            print("Warning: Reference trajectory not available for 3D plot")
    
    # Check if we have enough data for trajectory plotting
    if len(times) < 10:
        print(f"\nERROR: Not enough timesteps ({len(times)}) for trajectory plotting!")
        print("  The sail3d plot requires a full trajectory with many timesteps.")
        print("  Your database was likely created with FinalStateOnly logging.")
        print("\n  To get trajectory plots, run quick_sim.rs which uses AllDownsampled logging,")
        print("  or change the logging strategy in main.rs to AllDownsampled(10).")
        return
    
    print(f"\nDiagnostics:")
    print(f"  Total timesteps: {len(times)}")
    print(f"  Time span: {times[0]:.1f} to {times[-1]:.1f} seconds ({times[-1]/3600:.2f} hours)")
    print(f"  Initial position: [{positions[0, 0]/1e3:.1f}, {positions[0, 1]/1e3:.1f}, {positions[0, 2]/1e3:.1f}] km")
    print(f"  Final position: [{positions[-1, 0]/1e3:.1f}, {positions[-1, 1]/1e3:.1f}, {positions[-1, 2]/1e3:.1f}] km")
    print(f"  Position range X: [{positions[:, 0].min()/1e3:.1f}, {positions[:, 0].max()/1e3:.1f}] km")
    print(f"  Position range Y: [{positions[:, 1].min()/1e3:.1f}, {positions[:, 1].max()/1e3:.1f}] km")
    print(f"  Position range Z: [{positions[:, 2].min()/1e3:.1f}, {positions[:, 2].max()/1e3:.1f}] km")
    print(f"  Cone angle range: [{np.degrees(cones.min()):.1f}, {np.degrees(cones.max()):.1f}] deg")
    print(f"  Clock angle range: [{np.degrees(clocks.min()):.1f}, {np.degrees(clocks.max()):.1f}] deg")
    print(f"\n  First 5 positions:")
    for i in range(min(5, len(positions))):
        print(f"    t={times[i]:.1f}s: pos=[{positions[i,0]/1e3:.1f}, {positions[i,1]/1e3:.1f}, {positions[i,2]/1e3:.1f}] km")
    
    # Print eccentricity info to debug revolution detection
    eccentricity = orbital_elements[:, 1]
    print(f"\n  Eccentricity range: [{eccentricity.min():.6f}, {eccentricity.max():.6f}]")
    
    # For near-circular orbits, true anomaly is unstable. Use argument of latitude instead.
    # Argument of latitude = arctan2(y/r, x/r) for equatorial orbits
    # This directly measures angle in the orbital plane from X-axis
    argument_of_latitude = np.arctan2(positions[:, 1], positions[:, 0])
    # Convert to [0, 2π] range
    argument_of_latitude = np.mod(argument_of_latitude, 2 * np.pi)
    
    print(f"  Argument of latitude range: [{np.degrees(argument_of_latitude.min()):.1f}, {np.degrees(argument_of_latitude.max()):.1f}] deg")
    print(f"  First 5 arg of lat: {np.degrees(argument_of_latitude[:5])}")
    
    # Find revolution boundaries (where argument of latitude wraps from ~2π to ~0)
    revolutions = [0]  # Start of first revolution
    for i in range(1, len(argument_of_latitude)):
        # Detect wrap-around: angle decreases by more than π (crosses 0°/360° boundary)
        delta = argument_of_latitude[i] - argument_of_latitude[i-1]
        # For prograde orbit, we expect increasing angle. Wrap happens when delta < -π
        if delta < -np.pi:
            revolutions.append(i)
    revolutions.append(len(times) - 1)  # End of last revolution
    
    print(f"\nFound {len(revolutions)-1} orbital revolutions")
    print(f"  Revolution boundaries at indices: {revolutions[:5]}..." if len(revolutions) > 5 else f"  Revolution boundaries: {revolutions}")
    
    # Select first orbit, middle orbit, and last orbit
    if len(revolutions) < 3:
        print("Warning: Less than 2 complete orbits, showing all available data")
        orbit_indices = [(0, len(times)-1)]
    else:
        # First orbit: from start to first wrap
        first_orbit = (0, revolutions[1])
        # Last orbit: from second-to-last wrap to end
        last_orbit = (revolutions[-2], revolutions[-1])
        # Middle orbit: find one roughly in the middle
        mid_rev = len(revolutions) // 2
        if mid_rev < len(revolutions) - 1:
            middle_orbit = (revolutions[mid_rev], revolutions[mid_rev + 1])
        else:
            middle_orbit = (revolutions[-3], revolutions[-2]) if len(revolutions) > 3 else None
        
        orbit_indices = [first_orbit]
        if middle_orbit:
            orbit_indices.append(middle_orbit)
        orbit_indices.append(last_orbit)
        
        print(f"Plotting orbits: first (0-{revolutions[1]}), ", end="")
        if middle_orbit:
            print(f"middle ({middle_orbit[0]}-{middle_orbit[1]}), ", end="")
        print(f"last ({last_orbit[0]}-{last_orbit[1]})")
    
    # Convert positions to km
    positions_km = positions / 1e3
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Earth as a sphere
    EARTH_RADIUS_KM = 6371
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_earth = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y_earth = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z_earth = EARTH_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth, color='cyan', alpha=0.3, edgecolor='none')
    
    # Colors for different orbits
    colors = ['blue', 'green', 'red']
    labels = ['First Orbit', 'Middle Orbit', 'Last Orbit']
    
    # Plot reference trajectory first (so it appears behind optimized)
    if ref_data is not None:
        ref_times, ref_sma, ref_altitude, ref_positions, ref_velocities, ref_orbital_elements, ref_cones, ref_clocks = ref_data
        ref_positions_km = ref_positions / 1e3
        ax.plot(ref_positions_km[:, 0], ref_positions_km[:, 1], ref_positions_km[:, 2],
               color='gray', linewidth=1.5, alpha=0.5, linestyle='--', label='Reference (Locally Optimal)')
        
        # Plot reference sail normal vectors
        vector_scale = 2000  # km
        step = 1 if len(ref_positions) <= 100 else max(1, len(ref_positions) // 100)
        for i in range(0, len(ref_positions), step):
            r_vec_m = ref_positions[i]  # Position in meters
            
            # Build sun-frame
            sun_position = np.array([150.0e9, 0.0, 0.0])  # meters
            z_inertial = np.array([0.0, 0.0, 1.0])
            
            s_vec = -(sun_position - r_vec_m)
            s_hat = s_vec / np.linalg.norm(s_vec)
            
            theta_hat = np.cross(z_inertial, s_hat)
            theta_hat = theta_hat / np.linalg.norm(theta_hat)
            
            phi_hat = np.cross(s_hat, theta_hat)
            
            R_sun_to_inertial = np.column_stack([s_hat, theta_hat, phi_hat])
            
            cone = ref_cones[i]
            clock = ref_clocks[i]
            
            sail_normal_sun_frame = np.array([
                np.cos(cone),
                np.sin(cone) * np.sin(clock),
                np.sin(cone) * np.cos(clock)
            ])
            
            sail_normal_inertial = R_sun_to_inertial @ sail_normal_sun_frame
            
            start_pos = ref_positions_km[i]
            ax.quiver(start_pos[0], start_pos[1], start_pos[2],
                     sail_normal_inertial[0]*vector_scale, 
                     sail_normal_inertial[1]*vector_scale, 
                     sail_normal_inertial[2]*vector_scale,
                     color='gray', alpha=0.3, arrow_length_ratio=0.3, linewidth=1.0)
    
    # Plot each selected orbit
    for idx, (start, end) in enumerate(orbit_indices):
        orbit_pos = positions_km[start:end+1]
        orbit_vel = velocities[start:end+1]
        orbit_cone = cones[start:end+1]
        orbit_clock = clocks[start:end+1]
        
        # Plot trajectory line
        ax.plot(orbit_pos[:, 0], orbit_pos[:, 1], orbit_pos[:, 2],
               color=colors[idx], linewidth=2, alpha=0.8, label=labels[idx])
        
        # Mark start point
        ax.scatter([orbit_pos[0, 0]], [orbit_pos[0, 1]], [orbit_pos[0, 2]],
                  color=colors[idx], s=150, marker='o', edgecolors='black', linewidths=2)
        
        # Plot sail normal vectors every 20th point
        vector_scale = 2000  # km
        step = 1 if len(orbit_pos) <= 100 else max(1, len(orbit_pos) // 10)
        for i in range(0, len(orbit_pos), step):
            # Get spacecraft position
            pos_idx = start + i
            r_vec_m = positions[pos_idx]  # Position in meters
            
            # Build sun-frame with proper axes:
            # Sun position: 150 million km in +X direction
            sun_position = np.array([150.0e9, 0.0, 0.0])  # meters
            z_inertial = np.array([0.0, 0.0, 1.0])  # Global Z-axis
            
            # ŝ (s_hat): vector from sun to spacecraft (matches physics code)
            s_vec = -(sun_position - r_vec_m)  # Points from sun to s/c
            s_hat = s_vec / np.linalg.norm(s_vec)
            
            # θ̂ (theta_hat): ẑ_inertial × ŝ
            theta_hat = np.cross(z_inertial, s_hat)
            theta_hat = theta_hat / np.linalg.norm(theta_hat)
            
            # φ̂ (phi_hat): ŝ × θ̂ (completes right-handed frame)
            phi_hat = np.cross(s_hat, theta_hat)
            
            # Build rotation matrix from sun-frame to inertial: R = [ŝ | θ̂ | φ̂]
            R_sun_to_inertial = np.column_stack([s_hat, theta_hat, phi_hat])
            
            # Get control angles for this timestep
            cone = orbit_cone[i]
            clock = orbit_clock[i]
            
            # Sail normal in sun-frame (cone-clock parameterization)
            # From reference: n̂|_ŝ = [cos(α), sin(α)sin(δ), sin(α)cos(δ)]
            sail_normal_sun_frame = np.array([
                np.cos(cone),
                np.sin(cone) * np.sin(clock),
                np.sin(cone) * np.cos(clock)
            ])
            
            # Transform to inertial frame
            sail_normal_inertial = R_sun_to_inertial @ sail_normal_sun_frame
            
            # Plot sail normal vector
            start_pos = orbit_pos[i]
            ax.quiver(start_pos[0], start_pos[1], start_pos[2],
                     sail_normal_inertial[0]*vector_scale, 
                     sail_normal_inertial[1]*vector_scale, 
                     sail_normal_inertial[2]*vector_scale,
                     color=colors[idx], alpha=0.4, arrow_length_ratio=0.3, linewidth=1.5)
    
    ax.legend(fontsize=12, loc='upper right')
    
    # Set labels and title
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_zlabel('Z (km)', fontsize=12)
    ax.set_title('Solar Sail Trajectory: First, Middle, and Last Orbits with Sail Normals', 
                fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = np.array([positions_km[:, 0].max()-positions_km[:, 0].min(),
                         positions_km[:, 1].max()-positions_km[:, 1].min(),
                         positions_km[:, 2].max()-positions_km[:, 2].min()]).max() / 2.0
    mid_x = (positions_km[:, 0].max()+positions_km[:, 0].min()) * 0.5
    mid_y = (positions_km[:, 1].max()+positions_km[:, 1].min()) * 0.5
    mid_z = (positions_km[:, 2].max()+positions_km[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('out/trajectory_3d_orbits_with_sail_normals.png', dpi=150, bbox_inches='tight')
    print("\n3D orbits plot with sail normals saved to: out/trajectory_3d_orbits_with_sail_normals.png")


def plot_first_day():
    """Plot the first day of trajectory continuously to investigate behavior."""
    times, sma, altitude, positions, velocities, orbital_elements, cones, clocks = get_best_trajectory()
    
    # Filter to first day only (86400 seconds)
    one_day = 20.0*86400.0
    mask = times <= one_day
    times = times[mask]
    positions = positions[mask]
    velocities = velocities[mask]
    orbital_elements = orbital_elements[mask]
    cones = cones[mask]
    clocks = clocks[mask]
    
    print(f"\n=== FIRST DAY ANALYSIS ===")
    print(f"Total timesteps in first day: {len(times)}")
    print(f"Time span: {times[0]:.1f} to {times[-1]:.1f} seconds ({times[-1]/3600:.2f} hours)")
    
    # Extract orbital elements
    a = orbital_elements[:, 0]
    e = orbital_elements[:, 1]
    i = orbital_elements[:, 2]
    w = orbital_elements[:, 3]
    
    print(f"\nOrbital element evolution:")
    print(f"  SMA:   {a[0]/1e3:.2f} -> {a[-1]/1e3:.2f} km (Δ = {(a[-1]-a[0])/1e3:.2f} km)")
    print(f"  Ecc:   {e[0]:.6f} -> {e[-1]:.6f}")
    print(f"  Inc:   {np.degrees(i[0]):.4f} -> {np.degrees(i[-1]):.4f} deg")
    print(f"  AoP:   {np.degrees(w[0]):.2f} -> {np.degrees(w[-1]):.2f} deg")
    
    # Calculate periapsis and apoapsis altitude
    r_p = a * (1 - e)  # Periapsis distance
    r_a = a * (1 + e)  # Apoapsis distance
    EARTH_RADIUS = 6.371e6
    alt_p = r_p - EARTH_RADIUS
    alt_a = r_a - EARTH_RADIUS
    
    print(f"\nPeriapsis altitude: {alt_p[0]/1e3:.1f} -> {alt_p[-1]/1e3:.1f} km")
    print(f"Apoapsis altitude:  {alt_a[0]/1e3:.1f} -> {alt_a[-1]/1e3:.1f} km")
    
    # Calculate current altitude
    r_mag = np.linalg.norm(positions, axis=1)
    altitude = r_mag - EARTH_RADIUS
    print(f"Current altitude range: {altitude.min()/1e3:.1f} to {altitude.max()/1e3:.1f} km")
    
    # Check: for e=0.19, the altitude variation should be significant
    expected_variation = 2 * a[-1] * e[-1]  # Apoapsis - Periapsis = 2ae
    print(f"\nExpected altitude variation for e={e[-1]:.3f}: {expected_variation/1e3:.1f} km")
    print(f"Actual altitude variation observed: {(altitude.max()-altitude.min())/1e3:.1f} km")
    
    # Convert positions to km
    positions_km = positions / 1e3
    
    # Create a 2x2 figure
    fig = plt.figure(figsize=(16, 14))
    
    # Plot 1: 3D trajectory (continuous first day)
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot Earth
    EARTH_RADIUS_KM = 6371
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_earth = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y_earth = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z_earth = EARTH_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_earth, y_earth, z_earth, color='cyan', alpha=0.3, edgecolor='none')
    
    # Plot trajectory colored by time
    times_hours = times / 3600
    scatter = ax1.scatter(positions_km[:, 0], positions_km[:, 1], positions_km[:, 2],
                         c=times_hours, cmap='viridis', s=2, alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.7)
    cbar.set_label('Time (hours)', fontsize=10)
    
    # Mark start/end
    ax1.scatter([positions_km[0, 0]], [positions_km[0, 1]], [positions_km[0, 2]],
               color='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter([positions_km[-1, 0]], [positions_km[-1, 1]], [positions_km[-1, 2]],
               color='red', s=100, marker='s', label='End (1 day)', zorder=5)
    
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_title('First Day Trajectory (3D)', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # Set equal aspect
    max_range = max(positions_km[:, 0].max()-positions_km[:, 0].min(),
                   positions_km[:, 1].max()-positions_km[:, 1].min(),
                   positions_km[:, 2].max()-positions_km[:, 2].min()) / 2.0
    mid_x = (positions_km[:, 0].max()+positions_km[:, 0].min()) * 0.5
    mid_y = (positions_km[:, 1].max()+positions_km[:, 1].min()) * 0.5
    mid_z = (positions_km[:, 2].max()+positions_km[:, 2].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Plot 2: XY view (orbital plane)
    ax2 = fig.add_subplot(222)
    scatter2 = ax2.scatter(positions_km[:, 0], positions_km[:, 1], c=times_hours, cmap='viridis', s=2)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Time (hours)', fontsize=10)
    
    # Draw Earth circle
    theta_earth = np.linspace(0, 2*np.pi, 100)
    ax2.plot(EARTH_RADIUS_KM * np.cos(theta_earth), EARTH_RADIUS_KM * np.sin(theta_earth), 
            'c-', linewidth=2, label='Earth')
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    ax2.set_title('XY Plane View (First Day)', fontsize=12, fontweight='bold')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Orbital elements over first day
    ax3 = fig.add_subplot(223)
    times_hours_arr = times / 3600
    ax3.plot(times_hours_arr, a/1e3, 'b-', label=f'SMA (km)', linewidth=1.5)
    ax3.set_ylabel('SMA (km)', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.set_xlabel('Time (hours)')
    ax3.set_title('Orbital Elements (First Day)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    ax3b = ax3.twinx()
    ax3b.plot(times_hours_arr, e, 'r-', label=f'Eccentricity', linewidth=1.5)
    ax3b.set_ylabel('Eccentricity', color='r')
    ax3b.tick_params(axis='y', labelcolor='r')
    ax3b.legend(loc='upper right')
    
    # Plot 4: Altitude over first day (to see if orbit is actually changing shape)
    ax4 = fig.add_subplot(224)
    ax4.plot(times_hours_arr, altitude/1e3, 'g-', linewidth=1, alpha=0.7, label='Current altitude')
    ax4.plot(times_hours_arr, alt_a/1e3, 'r--', linewidth=1.5, label='Apoapsis altitude')
    ax4.plot(times_hours_arr, alt_p/1e3, 'b--', linewidth=1.5, label='Periapsis altitude')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Altitude (km)')
    ax4.set_title('Altitude Evolution (First Day)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('out/first_day_trajectory.png', dpi=150, bbox_inches='tight')
    print(f"\nFirst day trajectory plot saved to: out/first_day_trajectory.png")


def plot_trajectory():
    """Create plots of the orbit raising trajectory."""
    times, sma, altitude, positions, velocities, orbital_elements, cones, clocks = get_best_trajectory()
    
    # Downsample for time-series plots
    downsample_factor = max(1, len(times) // 10000)  # Target ~10k points max
    times_ds = times[::downsample_factor]
    sma_ds = sma[::downsample_factor]
    altitude_ds = altitude[::downsample_factor]
    
    # Compute running mean SMA (cumulative mean up to each point)
    mean_sma_ds = np.cumsum(sma_ds) / np.arange(1, len(sma_ds) + 1)
    
    # Convert time to days
    times_days = times_ds / (60 * 60 * 24)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Semi-major axis evolution
    ax = axes[0]
    ax.plot(times_days, sma_ds / 1e3, 'b-', linewidth=1, alpha=0.6, label='Instantaneous SMA')
    ax.plot(times_days, mean_sma_ds / 1e3, 'r-', linewidth=2, label='Running Mean SMA')
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Semi-Major Axis (km)', fontsize=12)
    ax.set_title('Solar Sail Orbit Raising: SMA Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Add statistics
    initial_sma = sma_ds[0]
    final_sma = sma_ds[-1]
    final_mean_sma = mean_sma_ds[-1]
    delta_sma = final_sma - initial_sma
    delta_mean_sma = final_mean_sma - initial_sma
    ax.text(0.02, 0.98, 
            f'Initial: {initial_sma/1e3:.1f} km\n'
            f'Final: {final_sma/1e3:.1f} km (Δ={delta_sma/1e3:+.1f} km)\n'
            f'Mean: {final_mean_sma/1e3:.1f} km (Δ={delta_mean_sma/1e3:+.1f} km)',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Altitude evolution
    ax = axes[1]
    ax.plot(times_days, altitude_ds / 1e3, 'g-', linewidth=2)
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Altitude (km)', fontsize=12)
    ax.set_title('Altitude Above Earth Surface', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    initial_alt = altitude_ds[0]
    final_alt = altitude_ds[-1]
    delta_alt = final_alt - initial_alt
    ax.text(0.02, 0.98,
            f'Initial: {initial_alt/1e3:.1f} km\nFinal: {final_alt/1e3:.1f} km\nΔAltitude: {delta_alt/1e3:.1f} km',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('out/orbit_raising_trajectory.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: out/orbit_raising_trajectory.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot solar sail trajectory optimization results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available plots:
  scatter     - 3D scatter plot colored by time (commented out)
  kepler      - 6-panel Keplerian orbital elements evolution
  angles      - Control angles (elevation and direction) over time
  sail3d      - 3D trajectory with sail normal vectors (first, middle, last orbit)
  trajectory  - SMA and altitude time series (commented out)
  firstday    - First day continuous trajectory with detailed analysis

Options:
  --reference - Overlay the locally optimal reference trajectory (from out/reference.db)
               Run 'cargo run --release --bin quick_reference' first to generate it.

Examples:
  python plot_trajectory.py                              # Plot all enabled plots
  python plot_trajectory.py --plots kepler angles        # Kepler and control angles
  python plot_trajectory.py --plots kepler --reference   # Compare with reference
  python plot_trajectory.py --plots firstday             # First day analysis
        """
    )
    parser.add_argument('--plots', nargs='+', 
                       choices=['scatter', 'kepler', 'angles', 'sail3d', 'trajectory', 'firstday'],
                       help='Select which plots to generate (default: kepler sail3d)')
    parser.add_argument('--reference', action='store_true',
                       help='Overlay the locally optimal reference trajectory (from out/reference.db)')
    
    args = parser.parse_args()
    
    # Set global reference flag
    PLOT_REFERENCE = args.reference
    
    # Default plots if none specified
    if args.plots is None:
        args.plots = ['kepler', 'sail3d']
    
    # Commented out: 3D scatter plot
    # if 'scatter' in args.plots:
    #     print("Plotting 3D trajectory (scatter)...")
    #     plot_3d_trajectory()
    
    if 'kepler' in args.plots:
        print("\nPlotting orbital elements...")
        plot_orbital_elements()
    
    if 'angles' in args.plots:
        print("\nPlotting control angles...")
        plot_control_angles()
    
    if 'sail3d' in args.plots:
        print("\nPlotting 3D trajectory with sail normals (first, middle, last orbit)...")
        plot_3d_with_sail_normals()
    
    if 'firstday' in args.plots:
        print("\nPlotting first day trajectory analysis...")
        plot_first_day()
    
    # Commented out: trajectory time series
    # if 'trajectory' in args.plots:
    #     print("\nPlotting altitude/SMA evolution...")
    #     plot_trajectory()
    
    # Show all plots together (non-blocking until all are closed)
    plt.show()

