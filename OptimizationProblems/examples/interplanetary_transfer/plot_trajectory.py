"""
3D trajectory visualization for the solar sail Earth-Mars transfer.

Reads from out/interplanetary_transfer.db and plots:
  - Spacecraft trajectory in J2000 ecliptic frame
  - Sun at the origin
  - Earth orbit and Mars orbit as reference rings in the ecliptic XY plane
  - Sail normal vector at each control point along the trajectory

FRAME NOTE:
  ANISE outputs positions in the J2000 equatorial (ICRF) frame, where Z points
  toward the celestial north pole.  The ecliptic (planetary orbital plane) is
  inclined ~23.44° from this frame.  All data is rotated to the J2000 ecliptic
  frame before plotting so that Earth/Mars orbits lie in the XY plane and the
  orbit rings are meaningful reference circles.

Run from the workspace root after cargo run:
    python examples/interplanetary_transfer/plot_trajectory.py

Requirements:
    pip install matplotlib numpy
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Constants ──────────────────────────────────────────────────────────────────
AU      = 1.496e11              # metres per AU
GM_SUN  = 1.32712440018e20      # m³/s²
DB_PATH = "out/interplanetary_transfer.db"
N_NORMAL_ARROWS = 80   # sail normal quivers shown along the trajectory

# Position penalty threshold [AU] — must match POS_PENALTY_AU in problem.rs.
# Solutions with pos_err > this value incur a steep energy penalty in the optimizer.
# The circle drawn around Mars arrival shows this threshold in position space.
# NOTE: a true rendezvous requires both position AND velocity proximity to Mars.
#       The vel penalty threshold (VEL_PENALTY_NORM = 1.5 × V_MARS) is not shown here
#       since it lives in velocity space, not in the position plot.
POS_PENALTY_AU = 0.1

# Mean obliquity of the ecliptic at J2000 [rad].
# This is the tilt between the J2000 equatorial (ICRF) and ecliptic planes.
OBLIQUITY = np.radians(23.4392911)

# Spacecraft / sail parameters — must match config.rs optimization_defaults()
SAIL_AREA_M2      = 350.0    # m²  — must match config.rs sail_area
SAIL_MASS_KG      = 10.0     # kg
SAIL_REFLECTIVITY = 0.9
P_SRP_1AU         = 4.56e-6  # N/m²  at 1 AU


# ── Frame conversion ───────────────────────────────────────────────────────────

def j2000_to_ecliptic(xyz):
    """
    Rotate 3D coordinates from J2000 equatorial (ICRF) to J2000 ecliptic frame.

    Applies Rx(-ε) where ε is the mean obliquity of the ecliptic:
        x_ecl =  x_eq
        y_ecl =  cos(ε) * y_eq + sin(ε) * z_eq
        z_ecl = -sin(ε) * y_eq + cos(ε) * z_eq

    After this rotation, Earth and Mars (which orbit in the ecliptic plane)
    have z ≈ 0, so the orbit rings drawn at z = 0 are the correct references.

    Parameters
    ----------
    xyz : array-like, shape (..., 3)

    Returns
    -------
    np.ndarray, same shape as input
    """
    xyz = np.asarray(xyz, dtype=float)
    c, s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    x =  xyz[..., 0]
    y =  c * xyz[..., 1] + s * xyz[..., 2]
    z = -s * xyz[..., 1] + c * xyz[..., 2]
    return np.stack([x, y, z], axis=-1)


# ── Physics ────────────────────────────────────────────────────────────────────

def compute_sail_normal(cone, clock, pos, vel):
    """
    Sail normal in the heliocentric inertial (ecliptic) frame.

    Mirrors SunPointingFrame::sail_normal_inertial in orbital_models/frames.rs exactly:
      - s_hat = pos / |pos|  (Sun → SC, i.e. away from Sun)
      - theta_hat = (z_inertial × s_hat) / |...|   (uses Z-axis, NOT velocity)
      - phi_hat   = s_hat × theta_hat
      - n = cos(α)·s_hat + sin(α)·(sin(δ)·theta_hat + cos(δ)·phi_hat)

    cone=0  → n points directly away from the Sun (maximum SRP, force away from Sun).
    cone=π/2 → edge-on, zero SRP.

    vel is unused here (kept in signature for call-site compatibility).
    """
    s_hat = pos / np.linalg.norm(pos)  # away from Sun

    z_inertial = np.array([0., 0., 1.])
    perp = np.cross(z_inertial, s_hat)
    perp_norm = np.linalg.norm(perp)
    if perp_norm > 1e-10:
        theta_hat = perp / perp_norm
    else:
        # s_hat is nearly parallel to Z — fall back to X axis
        x_inertial = np.array([1., 0., 0.])
        p = np.cross(x_inertial, s_hat)
        theta_hat = p / np.linalg.norm(p)

    phi_hat = np.cross(s_hat, theta_hat)

    return (np.cos(cone) * s_hat
            + np.sin(cone) * (np.sin(clock) * theta_hat + np.cos(clock) * phi_hat))


# ── SRP physics (mirrors srp_acceleration() in problem.rs) ────────────────────

def compute_srp_vector(pos, vel, cone, clock):
    """
    SRP acceleration vector [m/s²] in the heliocentric inertial frame.

    Mirrors srp_acceleration() in problem.rs exactly:
      pressure = P_SRP_1AU * (AU / r)²
      n_hat    = compute_sail_normal(cone, clock, pos, vel)
      cos_α    = dot(n_hat, s_hat).max(0)
      a_srp    = pressure * A * (1+ρ) * cos²α / m  * n_hat
    """
    r = np.linalg.norm(pos)
    pressure = P_SRP_1AU * (AU / r) ** 2
    n_hat = compute_sail_normal(cone, clock, pos, vel)
    s_hat = pos / r
    cos_alpha = max(0.0, float(np.dot(n_hat, s_hat)))
    force_per_mass = pressure * SAIL_AREA_M2 * (1.0 + SAIL_REFLECTIVITY) * cos_alpha ** 2 / SAIL_MASS_KG
    return force_per_mass * n_hat




# ── Physics panels ─────────────────────────────────────────────────────────────

def plot_physics_panels(traj, planets, planet_track):
    """
    2×2 figure of physical quantities along the best trajectory.

      (0,0) Heliocentric distance [AU]
      (0,1) Heliocentric speed + radial/transverse breakdown [km/s]
      (1,0) SRP acceleration magnitude [mm/s²]
      (1,1) Sail control angles over the transfer [deg]

    Saved to out/trajectory_physics.png.
    Positions and velocities from the DB are in J2000 ICRF (equatorial).
    All quantities computed here are scalar (frame-independent).
    """
    days  = traj["simtime"] / 86_400.0
    pos   = traj["pos"]    # ICRF metres
    vel   = traj["vel"]    # ICRF m/s
    cone  = traj["cone"]   # rad
    clock = traj["clock"]  # rad

    r_mag = np.linalg.norm(pos, axis=1) / AU          # [AU]
    speed = np.linalg.norm(vel, axis=1) / 1e3          # [km/s]
    r_hat = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    v_radial = np.einsum("ij,ij->i", vel, r_hat) / 1e3 # [km/s]
    v_perp   = np.sqrt(np.maximum(0.0, speed**2 - v_radial**2))  # [km/s]

    # Full SRP vectors [mm/s²] in ICRF
    srp_vecs = np.array([
        compute_srp_vector(pos[i], vel[i], cone[i], clock[i]) * 1e3
        for i in range(len(days))
    ])  # shape (N, 3)
    srp_total = np.linalg.norm(srp_vecs, axis=1)   # magnitude [mm/s²]

    # Unit velocity direction — prograde
    v_hat = vel / np.linalg.norm(vel, axis=1, keepdims=True)
    # Signed prograde component: + = accelerating along orbit, − = braking
    srp_prograde = np.einsum("ij,ij->i", srp_vecs, v_hat)   # [mm/s²]
    # Signed radial component: + = pushing away from Sun (raises aphelion), − = sunward
    srp_radial   = np.einsum("ij,ij->i", srp_vecs, r_hat)   # [mm/s²]
    # Sanity check: prograde² + radial² + residual² ≈ total²

    tof_d = days[-1]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"Solar Sail Earth\u2192Mars \u2014 Physical Quantities along Trajectory"
        f"  (TOF = {tof_d:.0f} d)",
        fontsize=11,
    )

    # ── (0,0) Heliocentric distance ──
    ax = axes[0, 0]
    ax.plot(days, r_mag, color="steelblue", lw=1.8, label="Spacecraft")
    ax.axhline(1.0, color="deepskyblue", lw=1.0, ls="--", alpha=0.7, label="Earth (1.00 AU)")
    ax.axhspan(1.381, 1.666, color="tomato", alpha=0.10, label="Mars range (1.38–1.67 AU)")
    if "mars_arrival" in planets:
        mars_r = np.linalg.norm(planets["mars_arrival"]["pos"]) / AU
        ax.axhline(mars_r, color="tomato", lw=1.2, ls=":",
                   label=f"Mars target ({mars_r:.3f} AU)")
    ax.set_xlabel("Mission elapsed time [days]")
    ax.set_ylabel("Heliocentric distance [AU]")
    ax.set_title("Heliocentric Distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (0,1) Velocity components ──
    ax = axes[0, 1]
    ax.plot(days, speed,    color="navy",       lw=1.8, label="Total |v|")
    ax.plot(days, v_perp,   color="seagreen",   lw=1.4, ls="--", label="Transverse v\u22a5 (orbital)")
    ax.plot(days, v_radial, color="darkorange",  lw=1.2, ls=":",  label="Radial v\u1d63 (+ = outbound)")

    # ANISE-sampled orbital speeds along the transfer (from planet_track table).
    # Falls back to flat reference lines when the table is absent (old DB).
    if "earth" in planet_track:
        pt = planet_track["earth"]
        ax.plot(pt["days"], pt["speed"], color="deepskyblue", lw=1.0, ls="--", alpha=0.65,
                label=f"Earth orbital speed (dep: {pt['speed'][0]:.2f} km/s)")
    else:
        ax.axhline(29.78, color="deepskyblue", lw=1.0, ls="--", alpha=0.5,
                   label="Earth (~29.78 km/s)")

    if "mars" in planet_track:
        pt = planet_track["mars"]
        ax.plot(pt["days"], pt["speed"], color="tomato", lw=1.0, ls="--", alpha=0.65,
                label=f"Mars orbital speed (arr: {pt['speed'][-1]:.2f} km/s)")
    else:
        ax.axhline(24.13, color="tomato", lw=1.0, ls="--", alpha=0.5,
                   label="Mars (~24.13 km/s)")
    ax.set_xlabel("Mission elapsed time [days]")
    ax.set_ylabel("Speed [km/s]")
    ax.set_title("Heliocentric Velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,0) SRP acceleration — prograde/retrograde decomposition ──
    ax = axes[1, 0]
    srp_max_1au = P_SRP_1AU * SAIL_AREA_M2 * (1.0 + SAIL_REFLECTIVITY) / SAIL_MASS_KG * 1e3
    # Filled areas: green = prograde thrust (speeding up), red = retrograde (braking)
    ax.fill_between(days, 0, srp_prograde, where=(srp_prograde >= 0),
                    color="seagreen", alpha=0.25, label="Prograde (orbital accel)")
    ax.fill_between(days, srp_prograde, 0, where=(srp_prograde < 0),
                    color="tomato",   alpha=0.25, label="Retrograde (orbital decel)")
    ax.plot(days, srp_prograde, color="steelblue", lw=1.5,
            label="Prograde (along v)")
    ax.plot(days, srp_radial,  color="mediumpurple", lw=1.2, ls=(0, (3, 1)),
            label="Radial (+ = away from Sun)")
    ax.plot(days, srp_total,   color="gray", lw=1.0, ls="--", alpha=0.7,
            label="|a\u209b\u1d63\u209a| total  (\u2248\u221aprograde\u00b2+radial\u00b2)")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline( srp_max_1au, color="gray", lw=0.8, ls=":", alpha=0.5,
               label=f"Max at 1 AU ({srp_max_1au:.3f} mm/s\u00b2)")
    ax.axhline(-srp_max_1au, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Mission elapsed time [days]")
    ax.set_ylabel("SRP acceleration [mm/s\u00b2]")
    ax.set_title("SRP — Prograde/Retrograde Component")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,1) Control angles ──
    ax = axes[1, 1]
    ax.plot(days, np.degrees(cone),  color="purple", lw=1.5, label="Cone \u03b1")
    ax.plot(days, np.degrees(clock), color="teal",   lw=1.2, ls="--", alpha=0.85, label="Clock \u03b4")
    ax.axhline(0,  color="gray", lw=0.7, ls=":")
    ax.axhline(90, color="gray", lw=0.7, ls=":", alpha=0.5, label="90\u00b0 (edge-on)")
    ax.set_xlabel("Mission elapsed time [days]")
    ax.set_ylabel("Angle [deg]")
    ax.set_title("Sail Control Angles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "out/trajectory_physics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


# ── Geometry helpers ───────────────────────────────────────────────────────────

def orbital_ellipse_3d(a_au, e, i_deg, lan_deg, aop_deg, n_pts=500):
    """
    Compute x, y, z coordinates of a Keplerian orbital ellipse in the J2000 ecliptic frame.

    Uses the standard Gauss Q/E vector transform (3-1-3 Euler sequence):
        r_ecliptic = xp * Q_hat + yp * E_hat
    where xp, yp are in the orbital (perifocal) plane.

    Parameters
    ----------
    a_au    : semi-major axis [AU]
    e       : eccentricity
    i_deg   : inclination w.r.t. J2000 ecliptic [degrees]
    lan_deg : longitude of ascending node [degrees]
    aop_deg : argument of periapsis (measured from ascending node) [degrees]
    """
    i   = np.radians(i_deg)
    lan = np.radians(lan_deg)
    aop = np.radians(aop_deg)

    ci, si = np.cos(i), np.sin(i)
    cO, sO = np.cos(lan), np.sin(lan)
    co, so = np.cos(aop), np.sin(aop)

    # Q-hat: direction of periapsis in ecliptic frame
    Q = np.array([cO*co - sO*so*ci, sO*co + cO*so*ci, so*si])
    # E-hat: 90° from periapsis in the orbital plane, in ecliptic frame
    E = np.array([-cO*so - sO*co*ci, -sO*so + cO*co*ci, co*si])

    # Sweep eccentric anomaly
    E_anom = np.linspace(0, 2 * np.pi, n_pts)
    xp = a_au * (np.cos(E_anom) - e)
    yp = a_au * np.sqrt(1 - e**2) * np.sin(E_anom)

    # Transform to ecliptic
    pos = np.outer(xp, Q) + np.outer(yp, E)  # shape (n_pts, 3)
    return pos[:, 0], pos[:, 1], pos[:, 2]


# ── Database ───────────────────────────────────────────────────────────────────

def load_planet_positions(db_path):
    """
    Load Earth and Mars positions at departure and arrival from the DB.

    Returns a dict keyed by e.g. 'earth_departure', 'mars_arrival', each value
    being {'pos': np.array([x, y, z]), 'vel': np.array([vx, vy, vz])} in metres
    and m/s in the J2000 equatorial (ICRF) frame as output by ANISE.

    Returns an empty dict if the table does not exist (older DB).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT body, epoch_label, x, y, z, vx, vy, vz FROM planet_positions"
        )
        rows = cur.fetchall()
    except Exception:
        conn.close()
        return {}
    conn.close()

    result = {}
    for body, label, x, y, z, vx, vy, vz in rows:
        key = f"{body.lower()}_{label}"   # e.g. "earth_departure", "mars_arrival"
        result[key] = {
            "pos": np.array([x, y, z]),
            "vel": np.array([vx, vy, vz]),
        }
    return result


def load_planet_track(db_path):
    """
    Load Earth and Mars positions/velocities sampled along the transfer from the DB.

    Returns a dict keyed by lowercase body name ('earth', 'mars'), each value:
        {'days': np.array, 'speed': np.array [km/s]}
    Returns an empty dict if the planet_track table is absent (older DB).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT body, day_offset, vx, vy, vz FROM planet_track ORDER BY body, day_offset"
        )
        rows = cur.fetchall()
    except Exception:
        conn.close()
        return {}
    conn.close()

    result = {}
    for body, day_offset, vx, vy, vz in rows:
        key = body.lower()
        if key not in result:
            result[key] = {"days": [], "speed": []}
        speed_kms = np.linalg.norm([vx, vy, vz]) / 1e3
        result[key]["days"].append(day_offset)
        result[key]["speed"].append(speed_kms)

    for key in result:
        result[key]["days"]  = np.array(result[key]["days"])
        result[key]["speed"] = np.array(result[key]["speed"])
    return result


def load_best_trajectory(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Prefer the eval_id stored by main.rs in the solutions table
    try:
        cur.execute("SELECT eval_id, energy, tof_days FROM solutions WHERE label='best' ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            eval_id, energy, tof_days = row
            print(f"Best solution: eval_id={eval_id}  energy={energy:.6f}  TOF={tof_days:.1f} days")
        else:
            raise ValueError("solutions table empty")
    except Exception:
        # Fallback: most recently logged eval_id
        cur.execute("SELECT eval_id FROM timesteps ORDER BY rowid DESC LIMIT 1")
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("No trajectory data found in the database.")
        eval_id = row[0]
        print(f"Fallback: using most recent eval_id={eval_id}")

    cur.execute("""
        SELECT simtime, x, y, z, vx, vy, vz, cone, clock
        FROM   timesteps
        WHERE  eval_id = ?
        ORDER  BY simtime
    """, (eval_id,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError(f"No timesteps found for eval_id={eval_id}")

    data = np.array(rows, dtype=float)
    return {
        "simtime": data[:, 0],   # seconds
        "pos":     data[:, 1:4], # metres, J2000 equatorial (ICRF) — raw from ANISE
        "vel":     data[:, 4:7], # m/s,   J2000 equatorial (ICRF)
        "cone":    data[:, 7],   # radians
        "clock":   data[:, 8],   # radians
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    traj         = load_best_trajectory(DB_PATH)
    planets      = load_planet_positions(DB_PATH)
    planet_track = load_planet_track(DB_PATH)

    # Rotate from J2000 equatorial (ANISE output) to J2000 ecliptic frame.
    # In the ecliptic frame Earth and Mars orbit in the XY plane (z ≈ 0),
    # making the orbit rings meaningful reference circles.
    pos_ecl = j2000_to_ecliptic(traj["pos"])   # metres, ecliptic frame
    vel_ecl = j2000_to_ecliptic(traj["vel"])   # m/s,   ecliptic frame
    pos_au  = pos_ecl / AU

    tof_days = traj["simtime"][-1] / 86_400.0

    # Transform planet positions to ecliptic frame
    planet_ecl = {k: j2000_to_ecliptic(v["pos"]) / AU for k, v in planets.items()}

    print(f"Trajectory: {len(traj['simtime'])} points over {tof_days:.1f} days")
    print(f"  SC departure: ({pos_au[0,0]:+.3f}, {pos_au[0,1]:+.3f}, {pos_au[0,2]:+.3f}) AU  (ecliptic frame)")
    print(f"  SC arrival:   ({pos_au[-1,0]:+.3f}, {pos_au[-1,1]:+.3f}, {pos_au[-1,2]:+.3f}) AU  (ecliptic frame)")
    if "mars_arrival" in planet_ecl:
        mp = planet_ecl["mars_arrival"]
        print(f"  Mars target:  ({mp[0]:+.3f}, {mp[1]:+.3f}, {mp[2]:+.3f}) AU")
        pos_err_au = np.linalg.norm(pos_au[-1] - mp)
        print(f"  Position error to Mars: {pos_err_au:.4f} AU  (penalty threshold: {POS_PENALTY_AU} AU)")

    r_dep = np.linalg.norm(pos_au[0])
    r_arr = np.linalg.norm(pos_au[-1])
    print(f"  Departure radius: {r_dep:.3f} AU  (expect ~1.0)")
    print(f"  Arrival radius:   {r_arr:.3f} AU  (expect ~1.38-1.67)")

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")

    # ── Sun ──
    ax.scatter([0], [0], [0], color="gold", s=300, marker="*", zorder=10, label="Sun")

    # ── Reference orbits — actual Keplerian ellipses (J2000 ecliptic, Standish 1992) ──
    # Earth: i≈0° so orbit lies in the ecliptic XY plane; Mars: i=1.85° gives ~0.05 AU z-offset.
    # The Mars planet markers should lie on (or very near) its ellipse.
    ax.plot(*orbital_ellipse_3d(1.000001, 0.016710, 0.00005, -11.261, 114.208),
            color="deepskyblue", lw=0.8, ls="--", alpha=0.6, label="Earth orbit")
    ax.plot(*orbital_ellipse_3d(1.523662, 0.093412, 1.85061, 49.719, 286.328),
            color="tomato",      lw=0.8, ls="--", alpha=0.6, label="Mars orbit")

    # ── Trajectory — colour-coded by elapsed time ──
    n = len(pos_au)
    cmap = plt.cm.viridis
    for i in range(n - 1):
        frac = i / (n - 1)
        ax.plot(pos_au[i:i+2, 0], pos_au[i:i+2, 1], pos_au[i:i+2, 2],
                color=cmap(frac), lw=1.8, alpha=0.9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, tof_days))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.12, label="Mission elapsed time [days]")

    # ── Planet positions at departure and arrival ──
    # Earth at departure: spacecraft starts here (sanity check — should coincide with SC departure)
    if "earth_departure" in planet_ecl:
        ep = planet_ecl["earth_departure"]
        ax.scatter(*ep, s=120, color="deepskyblue", marker="o", zorder=9, label="Earth (departure)")
    # Earth at arrival: where Earth is when the SC arrives (context)
    if "earth_arrival" in planet_ecl:
        ep = planet_ecl["earth_arrival"]
        ax.scatter(*ep, s=50, color="deepskyblue", marker="o", alpha=0.4, zorder=8, label="Earth (arrival)")
    # Mars at departure: where Mars is at launch (context)
    if "mars_departure" in planet_ecl:
        mp = planet_ecl["mars_departure"]
        ax.scatter(*mp, s=50, color="tomato", marker="o", alpha=0.4, zorder=8, label="Mars (departure)")
    # Mars at arrival: the rendezvous target
    if "mars_arrival" in planet_ecl:
        mp = planet_ecl["mars_arrival"]
        ax.scatter(*mp, s=150, color="tomato", marker="D", zorder=9, label="Mars target (arrival)")

        # ── Threshold circle around Mars at arrival ──
        # Radius = POS_PENALTY_AU: solutions outside this incur a steep energy penalty.
        # Note: a true rendezvous also requires matching Mars's velocity (vel penalty
        # threshold = 1.5 × V_MARS), which is not visible in this position-space plot.
        theta = np.linspace(0, 2 * np.pi, 200)
        cx, cy, cz = mp[0], mp[1], mp[2]
        ax.plot(
            cx + POS_PENALTY_AU * np.cos(theta),
            cy + POS_PENALTY_AU * np.sin(theta),
            np.full_like(theta, cz),
            color="tomato", lw=1.2, ls=":", alpha=0.8,
            label=f"Pos. penalty threshold ({POS_PENALTY_AU} AU)",
        )

    # ── Spacecraft departure and arrival markers ──
    ax.scatter(*pos_au[0],  s=60, color="white", edgecolors="deepskyblue", lw=1.5,
               zorder=10, label="SC departure")
    ax.scatter(*pos_au[-1], s=80, color="yellow", marker="^", edgecolors="black", lw=0.8,
               zorder=10, label="SC arrival")

    # ── Sail normal vectors at N evenly-spaced points (ecliptic frame) ──
    arrow_scale = 0.08   # AU
    indices = np.linspace(0, n - 1, N_NORMAL_ARROWS, dtype=int)
    first = True
    for idx in indices:
        p = pos_ecl[idx]
        v = vel_ecl[idx]
        # Compute sail normal using ecliptic-frame pos/vel — result is also in ecliptic frame
        n_hat = compute_sail_normal(traj["cone"][idx], traj["clock"][idx], p, v)

        p_au = p / AU
        d_au = n_hat * arrow_scale
        label = "Sail normal" if first else None
        first = False
        ax.quiver(p_au[0], p_au[1], p_au[2],
                  d_au[0], d_au[1], d_au[2],
                  color="orange", linewidth=1.5, arrow_length_ratio=0.35,
                  label=label)

    # ── Formatting ──
    ax.set_xlabel("X [AU]  (ecliptic J2000)", labelpad=8)
    ax.set_ylabel("Y [AU]  (ecliptic J2000)", labelpad=8)
    ax.set_zlabel("Z [AU]", labelpad=8)
    ax.set_title(
        f"Solar Sail Earth\u2192Mars Transfer\n"
        f"TOF = {tof_days:.0f} days   (ecliptic J2000 frame)",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=8)

    lim = 1.9
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-0.3, 0.3)   # ecliptic frame: Earth/Mars z ≈ 0, small deviations only

    # View slightly above the ecliptic plane
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    out_path = "out/trajectory_3d.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    plot_physics_panels(traj, planets, planet_track)

    plt.show()


if __name__ == "__main__":
    main()