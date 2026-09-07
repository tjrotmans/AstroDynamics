"""
astrodynamics.py — Python mirror of the Rust orbital-mechanics crates.

This module is the single source of truth for all physical constants and
canonical frame-rotation utilities used across Python visualisation scripts
in this workspace.  Every value and every formula here is a direct translation
of the corresponding definition in the Rust crates; the Rust source is cited
next to each entry.  If a value changes in the Rust crates, change it here too
and vice-versa.

Rust source files this mirrors:
  crates/orbital_models/src/constants.rs  — physical constants
  AstroProbs/LunarTrajectories/src/crtbp.rs — CRTBP normalisation
  (future) crates/orbital_models/src/frames.rs — frame rotations

Usage (from any plot script):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[N] / "crates" / "python"))
    from astrodynamics import *
"""

import numpy as np

# ── Physical constants ─────────────────────────────────────────────────────────
# Source: crates/orbital_models/src/constants.rs
# All values in SI (metres, kilograms, seconds) unless noted.

MU_EARTH              = 3.986_004_418e14   # m³/s²   Standard grav. param. Earth (EGM2008)
MU_MOON               = 4.904_869_5e12     # m³/s²   Standard grav. param. Moon
MU_SUN                = 1.327_124_400_18e20# m³/s²   Standard grav. param. Sun  (JPL DE430)
MU_VENUS              = 3.248_598_96e14    # m³/s²   (JPL DE430)
MU_MARS               = 4.282_837_62e13    # m³/s²   (JPL DE430)
MU_JUPITER            = 1.267_127_678e17   # m³/s²   (JPL DE430)

G                     = 6.674_30e-11       # m³/(kg·s²)  Gravitational constant
M_EARTH               = 5.972e24           # kg

EARTH_RADIUS          = 6.371e6            # m   Mean equatorial radius
EARTH_EQUATORIAL_RADIUS = 6_378_136.6      # m   EGM2008 equatorial radius (for J2 etc.)
MOON_RADIUS           = 1.737_4e6          # m

AU                    = 1.496e11           # m   Astronomical unit
P_SRP                 = 4.56e-6            # N/m²  Solar radiation pressure at 1 AU

G0                    = 9.806_65           # m/s²  Standard gravity (Isp → thrust)

J2                    = 1.082_626_68e-3    # EGM2008
J3                    = -2.532_661_2e-6    # EGM2008
J4                    = -1.619_898_5e-6    # EGM2008

# ── Convenient unit conversions ────────────────────────────────────────────────
EARTH_RADIUS_KM       = EARTH_RADIUS / 1e3         # 6371.0 km
MOON_RADIUS_KM        = MOON_RADIUS  / 1e3         # 1737.4 km
MU_EARTH_KM3          = MU_EARTH / 1e9             # km³/s²
MU_MOON_KM3           = MU_MOON  / 1e9             # km³/s²

# ── Earth-Moon CRTBP normalisation ────────────────────────────────────────────
# Source: AstroProbs/LunarTrajectories/src/crtbp.rs  CrtbpParams::earth_moon()
# Non-dimensional (nd) units: L_KM = 1, T_STAR = 1 nd-time, V_STAR = 1 nd-vel.

MU_ND    = 0.012_155_65          # Earth-Moon mass ratio  m_M / (m_E + m_M)
X_M      = 1.0 - MU_ND          # Moon x-coordinate in rotating frame [nd]
L_KM     = 384_400.0            # km   Earth-Moon mean distance (1 length unit)
T_STAR   = 375_700.0            # s    Non-dimensional time unit
V_STAR   = L_KM * 1e3 / T_STAR  # m/s  Non-dimensional velocity unit  (≈ 1023 m/s)
V_STAR_KM = V_STAR / 1e3        # km/s

R_HILL_ND = (MU_ND / 3.0) ** (1.0 / 3.0)   # Moon Hill sphere radius [nd]
R_HILL_KM = R_HILL_ND * L_KM               # ≈ 61 282 km

# ── Moon sidereal / synodic periods ───────────────────────────────────────────
# Source: AstroProbs/LunarTrajectories/src/bin/wsb/wsb_circularize.rs
MOON_SIDEREAL_DAYS = 27.321_661
SYNODIC_DAYS       = 1.0 / (1.0 / MOON_SIDEREAL_DAYS - 1.0 / 365.25)


# ══════════════════════════════════════════════════════════════════════════════
# Frame rotation utilities
# ══════════════════════════════════════════════════════════════════════════════
# These are exact Python translations of the Rust implementations.  If the Rust
# source changes, update the corresponding function here.
# Source: AstroProbs/LunarTrajectories/src/bin/wsb/wsb_circularize.rs
#         AstroProbs/LunarTrajectories/plot/wsb_style.py  (canonical Python)


def rot_em_to_eci(
    x_nd: np.ndarray,
    y_nd: np.ndarray,
    z_nd: np.ndarray,
    t_nd: np.ndarray,
    R0:   np.ndarray,
) -> tuple:
    """
    BCR4BP rotating (EM barycentric) frame → Earth-centred ECI [km].

    Exact translation of wsb_style.rot_em_to_eci (which mirrors
    wsb_circularize.rs:rot_to_eci).

    Steps:
      1. Earth-centre shift:  x_ec = x_nd + MU_ND
      2. Unrotate by t_nd (BCR4BP frame rotates at ω = 1 nd/nd)
      3. Scale by L_KM, apply R0 tilt into true ECI

    Parameters
    ----------
    x_nd, y_nd, z_nd : arrays  Position in BCR4BP rotating frame [nd]
    t_nd             : array   Non-dimensional time
    R0               : (3,3)   Rotation from EM-inertial to ECI (from epoch_info.txt)

    Returns
    -------
    (x_eci_km, y_eci_km, z_eci_km) : arrays
    """
    ec_x = x_nd + MU_ND
    ec_y = y_nd
    ec_z = z_nd
    xi_em = (ec_x * np.cos(t_nd) - ec_y * np.sin(t_nd)) * L_KM
    yi_em = (ec_x * np.sin(t_nd) + ec_y * np.cos(t_nd)) * L_KM
    zi_em = ec_z * L_KM
    pos_em  = np.array([xi_em, yi_em, zi_em])
    pos_eci = R0 @ pos_em
    return pos_eci[0], pos_eci[1], pos_eci[2]


def moon_em_to_eci(t_nd: np.ndarray, R0: np.ndarray) -> tuple:
    """
    BCR4BP circular Moon position → ECI [km].

    Exact translation of wsb_style.moon_em_to_eci.
    Moon is at (L_KM, 0, 0) in EM-inertial at t=0; rotates at ω = 1 nd/nd.

    Parameters
    ----------
    t_nd : array   Non-dimensional time
    R0   : (3,3)   EM-inertial → ECI rotation

    Returns
    -------
    (mx_eci_km, my_eci_km, mz_eci_km) : arrays
    """
    mx_em = np.cos(t_nd) * L_KM
    my_em = np.sin(t_nd) * L_KM
    mz_em = np.zeros_like(t_nd)
    moon_em  = np.array([mx_em, my_em, mz_em])
    moon_eci = R0 @ moon_em
    return moon_eci[0], moon_eci[1], moon_eci[2]


def orbital_plane_r3d(moon_pos_km: np.ndarray, moon_vel_km_s: np.ndarray) -> np.ndarray:
    """
    Build the 3×3 rotation matrix mapping the BCR4BP inertial frame to ECI,
    accounting for the Moon's orbital inclination.

    Exact translation of wsb_continuation_corrected.rs:orbital_plane_r3d.

    Columns:  x̂ = Moon direction,
              ŷ = in-plane perpendicular (Moon velocity direction),
              ẑ = orbital angular momentum  h = r × v.

    Parameters
    ----------
    moon_pos_km   : (3,) array   Moon ECI position [km]
    moon_vel_km_s : (3,) array   Moon ECI velocity [km/s]

    Returns
    -------
    R3D : (3,3) ndarray
    """
    x_hat = moon_pos_km / np.linalg.norm(moon_pos_km)
    h     = np.cross(moon_pos_km, moon_vel_km_s)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)
    return np.column_stack([x_hat, y_hat, z_hat])


def eci_to_rot_em(
    x_eci_km: np.ndarray,
    y_eci_km: np.ndarray,
    t_nd:     np.ndarray,
    psi_m0:   float,
) -> tuple:
    """
    Earth-centred ECI [km] → BCR4BP barycentric rotating frame [nd].

    Inverse of rot_em_to_eci (2-D, z ignored).

    Parameters
    ----------
    x_eci_km, y_eci_km : arrays  ECI position [km]
    t_nd               : array   Non-dimensional time
    psi_m0             : float   Moon ECI angle at epoch [rad]

    Returns
    -------
    (x_nd, y_nd) : arrays in BCR4BP rotating frame
    """
    theta    = psi_m0 + t_nd          # BCR4BP rotates at ω = 1 nd/nd
    x_ec_rot = (np.cos(theta) * x_eci_km + np.sin(theta) * y_eci_km) / L_KM
    y_ec_rot = (-np.sin(theta) * x_eci_km + np.cos(theta) * y_eci_km) / L_KM
    return x_ec_rot - MU_ND, y_ec_rot
