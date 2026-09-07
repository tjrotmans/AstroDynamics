# Solar Sail Orbit Raising Optimization

This example optimizes a solar sail trajectory to maximize the semi-major axis (raise the orbit) over a fixed time period.

## Problem Description

**Objective**: Maximize the semi-major axis of a spacecraft orbit using solar radiation pressure

**Physics**:
- Solar radiation pressure provides continuous low-thrust acceleration
- Optimal strategy for circular orbits: point sail perpendicular to velocity
- This maximizes tangential acceleration → increases orbital velocity → raises orbit

**Parameters**:
- Initial altitude: 500 km (circular orbit)
- Sail area: 100 m²
- Spacecraft mass: 10 kg (area-to-mass ratio: 10 m²/kg)
- Mission duration: 30 days
- Control points: 20 angle pairs

## Theory & Validation

### Edelbaum's Solution

For continuous low-thrust orbit raising, the optimal control law is well-known:
- **Thrust direction**: Always tangential to velocity (in-track direction)
- **Delta-V requirement**: Δv ≈ √(μ/r₁) - √(μ/r₂)

For our case (500 km → target):
- This provides an analytical benchmark to validate the optimizer

### Expected Behavior

1. **SMA should increase monotonically** over the mission
2. **Optimal angles** should keep sail normal ~perpendicular to velocity
3. **Rate of SMA growth** should be approximately constant (for constant thrust)

### Solar Radiation Pressure Model

- Pressure at 1 AU: P = 4.56 × 10⁻⁶ N/m²
- Force: F = P × A × (1 + ρ) × cos²(α)
  - A = sail area
  - ρ = reflectivity (1.6 for aluminized Kapton)
  - α = angle between sail normal and sun direction
- Acceleration: a = F / m

For our 100 m² sail with 10 kg mass:
- Maximum acceleration ≈ 1.2 × 10⁻⁴ m/s²
- Over 30 days: Δv ≈ 311 m/s

## Running the Example

### Build and Run Optimization

```bash
cargo run --release
```

The optimization will:
1. Initialize with random control angles
2. Run simulated annealing for 100 generations
3. Save results to `out/solar_sail_trajectory.db`

### Re-run Best Solution and Plot

After running an optimization, you can re-simulate the best solution with full trajectory logging and plot it:

```bash
cargo run --release --bin quick_sim -- --best && DB_PATH=out/quick_sim.db python plot/plot_trajectory.py --plots sail3d
```

This will:
1. Load the best control angles from the optimization database
2. Re-run the simulation with full timestep logging
3. Plot the 3D trajectory with sail normal vectors

### Visualize Optimization Progress

```bash
DB_PATH=out/solar_sail_trajectory.db python plot/plot_raw_angles.py
```

This generates plots showing:
- Temperature and energy evolution
- SMA evolution over iterations
- Acceptance rate

### Visualize Results

```bash
python plot/plot_trajectory.py
```

This generates plots showing:
- Semi-major axis evolution
- Altitude evolution
- Performance statistics

## Files

- `src/main.rs` - Entry point and problem setup
- `src/problem.rs` - Orbit raising problem definition
- `src/models/acceleration.rs` - SRP, gravity, and drag models
- `src/models/orbital.rs` - Orbital element conversions
- `src/models/environment.rs` - Atmospheric density model
- `src/control.rs` - Angle control representation
- `src/logging.rs` - Database logging strategies

## Simplifications

This is a simplified model for educational/validation purposes:

**Simplifications**:
1. Sun is fixed in +X direction (doesn't move relative to Earth)
2. Earth is always at 1 AU (ignores orbital eccentricity)
3. No eclipses (Earth shadow not modeled)
4. Idealized sail (no wrinkles, degradation, or optical imperfections)
5. Circular initial orbit (e = 0)

**For production use, add**:
- JPL ephemeris for accurate Sun position
- Eclipse detection and handling
- Non-Keplerian perturbations (J2, solar/lunar gravity)
- Sail degradation models
- Attitude dynamics and control constraints

## Comparison to Drag Sail

| Aspect | Drag Sail | Solar Sail |
|--------|-----------|------------|
| **Objective** | Minimize deorbit time | Maximize SMA |
| **Primary Force** | Atmospheric drag | Solar radiation pressure |
| **Altitude Range** | 300-500 km | 500+ km (any) |
| **Direction** | Opposes velocity | Toward/away from Sun |
| **Optimization** | Find fastest descent | Find highest orbit |
| **Complexity** | Simple (1 force) | Moderate (Sun geometry) |

## References

1. Edelbaum, T. N. (1961). "Propulsion Requirements for Controllable Satellites"
2. McInnes, C. R. (1999). "Solar Sailing: Technology, Dynamics and Mission Applications"
3. Wright, J. L. (1992). "Space Sailing"

## Next Steps

To extend this example:
1. Add realistic Sun ephemeris
2. Implement eclipse detection
3. Compare against analytical Edelbaum solution
4. Try different optimization algorithms (GA, etc.)
5. Add attitude dynamics constraints
6. Model sail degradation over time
