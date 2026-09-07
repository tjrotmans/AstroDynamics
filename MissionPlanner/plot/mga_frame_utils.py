"""mga_frame_utils.py -- shared helpers for MGA plot scripts.

All MGA plots are rendered in the ECLIPTIC frame, not ANISE's native ICRF/
equatorial-J2000 frame that `body_state_heliocentric()` actually returns.
Real planetary orbits lie close to the ecliptic, so this is the natural frame
for a trajectory visualization -- no tilted reference rings, no spurious-
looking "inclination" from axis convention alone.

The backend (design.rs / mga.rs, the propagator, the optimizer) stays in
ICRF/equatorial-J2000 -- untouched. This module converts ICRF -> ecliptic at
the plotting layer only, via a single FIXED rotation about the x-axis by
Earth's obliquity eps = 23.43929111 deg. This is not an approximation: ICRF
and ECLIPJ2000 are both frozen inertial orientations pinned to the single
epoch J2000.0, so the angle between them is an exact constant -- the same
value ANISE itself uses internally (`J2000_TO_ECLIPJ2000_ANGLE_RAD` in the
`anise` crate's `constants.rs`). All MGA plot scripts must use these helpers
for the rotation -- never redefine the obliquity or the rotation formulas
locally (mirrors the project's `crates/python/astrodynamics.py`
single-source-of-truth rule).
"""

import math
import numpy as np

AU = 1.495_978_707e11          # m
MU_SUN = 1.327_124_400_18e20   # m^3/s^2 (IAU 2012)
OBLIQUITY_RAD = math.radians(23.43929111)  # exact ICRF<->ECLIPJ2000 angle at J2000.0

BODY_SMA_AU = {
    "Mercury": 0.387, "Venus": 0.723, "Earth": 1.000, "Mars": 1.524,
    "Jupiter": 5.203, "Saturn": 9.537, "Uranus": 19.19, "Neptune": 30.07,
}
BODY_COLORS = {
    "Sun": "#ffffc0", "Mercury": "#aaaaaa", "Venus": "#e8c050",
    "Earth": "#4080ff", "Mars": "#cc4422", "Jupiter": "#d4a060",
    "Saturn": "#c8b870", "Uranus": "#80d4d4", "Neptune": "#4040cc",
}


def body_color(name: str) -> str:
    return BODY_COLORS.get(name, "#ffffff")


def circular_omega(sma_m: float) -> float:
    return math.sqrt(MU_SUN / sma_m**3)


def icrf_to_ecliptic(x, y, z):
    """Exact fixed rotation: ICRF/equatorial-J2000 -> ECLIPJ2000, about the x-axis."""
    y_ecl = y * np.cos(OBLIQUITY_RAD) + z * np.sin(OBLIQUITY_RAD)
    z_ecl = -y * np.sin(OBLIQUITY_RAD) + z * np.cos(OBLIQUITY_RAD)
    return x, y_ecl, z_ecl


def add_ecliptic_columns(df):
    """
    Add x_ecl_au / y_ecl_au / z_ecl_au columns (ecliptic frame, AU) to a
    dataframe loaded from an MGA arc CSV (which stores ICRF x_m/y_m/z_m).
    Mutates and returns `df`.
    """
    x_ecl, y_ecl, z_ecl = icrf_to_ecliptic(df["x_m"], df["y_m"], df["z_m"])
    df["x_ecl_au"] = x_ecl / AU
    df["y_ecl_au"] = y_ecl / AU
    df["z_ecl_au"] = z_ecl / AU
    return df


def icrf_point_to_ecliptic_au(x_m, y_m, z_m):
    """Convert a single ICRF position [m] to ecliptic [AU]."""
    x, y, z = icrf_to_ecliptic(x_m, y_m, z_m)
    return x / AU, y / AU, z / AU


def infer_theta0(x_m, y_m, z_m, sma_m, t_days):
    """Back-project a body's real ICRF position at t_days to its circular-orbit angle at t=0."""
    _, y_ecl, _ = icrf_to_ecliptic(x_m, y_m, z_m)
    theta_now = math.atan2(y_ecl, x_m)
    omega = circular_omega(sma_m)
    return theta_now - omega * t_days * 86_400.0


def planet_position_ecliptic_au(theta0_rad, sma_au, t_days):
    """Approximate circular-orbit position [AU] in the ecliptic frame at t_days."""
    sma_m = sma_au * AU
    omega = circular_omega(sma_m)
    theta = theta0_rad + omega * t_days * 86_400.0
    return sma_au * math.cos(theta), sma_au * math.sin(theta), 0.0


def ring_xyz(sma_au: float, n_points: int = 200):
    """Return (x, y, z) arrays [AU] for a circular orbit ring, flat in the ecliptic (z=0)."""
    theta_ring = np.linspace(0, 2 * math.pi, n_points)
    return sma_au * np.cos(theta_ring), sma_au * np.sin(theta_ring), np.zeros(n_points)


def find_dsm_point(arc_df, leg_idx: int, t_dep_days: float, eta: float, tof_days: float):
    """
    Locate the DSM point on a propagated arc: the nearest sampled row to the
    analytic patch time t_dep_days + eta*tof_days within the given leg. Works
    for both the analytic and the multiple-shooting-corrected arc, since both
    share the same (fixed) eta/tof geometry -- only the DSM's applied delta-v
    differs between them, not its timing.

    Returns (t_days, x_m, y_m, z_m) of the nearest row, or None if the leg has
    no rows in `arc_df`.
    """
    seg = arc_df[arc_df["leg_idx"] == leg_idx]
    if seg.empty:
        return None
    t_dsm = t_dep_days + eta * tof_days
    idx = (seg["t_days"] - t_dsm).abs().idxmin()
    row = seg.loc[idx]
    return float(row["t_days"]), float(row["x_m"]), float(row["y_m"]), float(row["z_m"])
