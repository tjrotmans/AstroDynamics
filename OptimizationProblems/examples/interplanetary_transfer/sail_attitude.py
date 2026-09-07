#!/usr/bin/env python3
"""
Solar sail attitude visualiser — validation tool.

Uses the SAME compute_sail_normal() as plot_trajectory.py, which mirrors
SunPointingFrame::sail_normal_inertial from orbital_models/frames.rs exactly.
This lets you verify that a given cone/clock produces the expected sail
orientation and SRP thrust direction.

Inputs
------
--cone  α   Cone angle  [deg]  0 = faces Sun (max SRP), 90 = edge-on (zero SRP)
--clock δ   Clock angle [deg]  90 = max prograde, 0/180 = out-of-plane
--ta    ν   True anomaly of the spacecraft in the ecliptic plane [deg]
            Controls which direction is "prograde" and "radially outward".
            0° → SC at +X from Sun (same as perihelion for a circular orbit)

Usage (from workspace root)
---------------------------
    python examples/interplanetary_transfer/sail_attitude.py
    python examples/interplanetary_transfer/sail_attitude.py --cone 30 --clock 90 --ta 45
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Import the shared sail-normal function from plot_trajectory.py ────────────
# This is the Python mirror of SunPointingFrame::sail_normal_inertial (Rust).
# Using the same function here ensures this tool validates the actual implementation.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_trajectory import compute_sail_normal   # noqa: E402


# ── Geometry helpers ──────────────────────────────────────────────────────────

def frame_axes(pos: np.ndarray):
    """
    Return (s_hat, theta_hat, phi_hat) for a given position vector.
    Exact copy of the basis construction inside compute_sail_normal().
    """
    s_hat = pos / np.linalg.norm(pos)
    z_inertial = np.array([0., 0., 1.])
    perp = np.cross(z_inertial, s_hat)
    perp_norm = np.linalg.norm(perp)
    if perp_norm > 1e-10:
        theta_hat = perp / perp_norm
    else:
        x_inertial = np.array([1., 0., 0.])
        p = np.cross(x_inertial, s_hat)
        theta_hat = p / np.linalg.norm(p)
    phi_hat = np.cross(s_hat, theta_hat)
    return s_hat, theta_hat, phi_hat


def sail_patch_verts(n_hat: np.ndarray, half_size: float = 0.45) -> list:
    """Four corners of a square sail centred at the origin, lying in the plane ⊥ n_hat."""
    ref = np.array([0., 1., 0.]) if abs(n_hat[1]) < 0.9 else np.array([1., 0., 0.])
    u = ref - np.dot(ref, n_hat) * n_hat
    u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)
    h = half_size
    return [h*u + h*v, -h*u + h*v, -h*u - h*v, h*u - h*v]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Solar sail attitude visualiser (uses same frame logic as Rust/plot_trajectory.py)"
    )
    parser.add_argument("--cone",  type=float, default=30.0, metavar="DEG",
                        help="Cone angle α [deg]  (0 = faces Sun, 90 = edge-on)")
    parser.add_argument("--clock", type=float, default=90.0, metavar="DEG",
                        help="Clock angle δ [deg]  (90 ≈ prograde, 0/180 = out-of-plane)")
    parser.add_argument("--ta",    type=float, default=0.0,  metavar="DEG",
                        help="Spacecraft true anomaly in ecliptic plane [deg]  "
                             "(sets the 'prograde' direction; 0° = SC at +X)")
    args = parser.parse_args()

    cone_rad  = np.radians(args.cone)
    clock_rad = np.radians(args.clock)
    ta_rad    = np.radians(args.ta)

    # Spacecraft position and velocity in the ecliptic frame (unit-normalised).
    # Circular orbit assumed: vel ⊥ pos, pointing prograde.
    pos = np.array([np.cos(ta_rad),  np.sin(ta_rad), 0.])   # s_hat direction
    vel = np.array([-np.sin(ta_rad), np.cos(ta_rad), 0.])   # prograde direction

    # ── Compute sail normal via the shared function (mirrors Rust) ────────────
    n_hat = compute_sail_normal(cone_rad, clock_rad, pos, vel)

    # ── Decompose the frame axes for display ──────────────────────────────────
    s_hat, theta_hat, phi_hat = frame_axes(pos)

    cos_a = max(0.0, float(np.dot(n_hat, s_hat)))
    srp   = cos_a ** 2   # SRP force ∝ cos²α, normalised

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # Sun at origin
    ax.scatter([0], [0], [0], color="gold", s=700, marker="*", zorder=10, label="Sun")

    # Incoming photon rays along -s_hat (from Sun toward SC)
    for dy, dz in [(0., 0.), (.20, .12), (-.20, -.12), (.0, .20), (.0, -.20)]:
        ray_start = -s_hat * 1.7 + dy * theta_hat + dz * phi_hat
        ax.quiver(*ray_start, *s_hat, length=1.2,
                  color="gold", alpha=0.28, linewidth=1, arrow_length_ratio=0.08)

    # Frame axes: s_hat, theta_hat, phi_hat (computed from pos, same as Rust)
    FRAME = [
        (s_hat,     "tomato",    r"$\hat{s}$  (away from Sun)"),
        (theta_hat, "seagreen",  r"$\hat{\theta}$  (tangential / prograde)"),
        (phi_hat,   "steelblue", r"$\hat{\phi}$  (out-of-plane)"),
    ]
    for vec, col, lbl in FRAME:
        ax.quiver(0, 0, 0, *vec, length=0.72, color=col, linewidth=1.5,
                  arrow_length_ratio=0.13, alpha=0.52, label=lbl)

    # Velocity direction (prograde reference for the given true anomaly)
    ax.quiver(0, 0, 0, *vel, length=0.55, color="seagreen", linewidth=1.0,
              arrow_length_ratio=0.13, alpha=0.30)  # lighter duplicate just for reference

    # Sail surface (front = silver, back = dark)
    verts = sail_patch_verts(n_hat)
    ax.add_collection3d(
        Poly3DCollection([verts], alpha=0.60,
                         facecolor="silver", edgecolor="gray", linewidth=0.8)
    )
    ax.add_collection3d(
        Poly3DCollection([verts[::-1]], alpha=0.20,
                         facecolor="dimgray", edgecolor="none")
    )

    # Sail normal = SRP acceleration direction
    ax.quiver(0, 0, 0, *n_hat, length=1.1, color="darkorange", linewidth=2.5,
              arrow_length_ratio=0.10,
              label=r"Sail normal $\hat{n}$ / SRP direction")

    # ── Axis labels and title ──────────────────────────────────────────────────
    LIM = 1.35
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
    ax.set_xlabel("X  (ecliptic)")
    ax.set_ylabel("Y  (ecliptic)")
    ax.set_zlabel("Z  (ecliptic north)")
    ax.set_title(
        f"Solar Sail Attitude   (same frame logic as Rust / plot_trajectory.py)\n"
        f"cone α = {args.cone:.1f}°     clock δ = {args.clock:.1f}°     "
        f"true anomaly ν = {args.ta:.1f}°\n"
        f"cos α = {cos_a:.3f}     SRP efficiency = cos²α = {srp:.3f}",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=20, azim=-55)

    # ── Info box: frame axes + n_hat decomposition ────────────────────────────
    def fmt(v):
        return f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"

    n_s = float(np.dot(n_hat, s_hat))
    n_t = float(np.dot(n_hat, theta_hat))
    n_p = float(np.dot(n_hat, phi_hat))

    info = (
        f"Frame axes (ecliptic XYZ)\n"
        f"  ŝ       = {fmt(s_hat)}\n"
        f"  θ̂       = {fmt(theta_hat)}\n"
        f"  φ̂       = {fmt(phi_hat)}\n"
        f"\n"
        f"n̂ = {fmt(n_hat)}\n"
        f"  · ŝ  (radial)       = {n_s:+.3f}\n"
        f"  · θ̂  (prograde)     = {n_t:+.3f}\n"
        f"  · φ̂  (out-of-plane) = {n_p:+.3f}\n"
        f"\n"
        f"SRP force ∝ cos²α = {srp:.3f}"
    )
    ax.text2D(0.01, 0.01, info, transform=ax.transAxes, fontsize=8,
              verticalalignment="bottom", family="monospace",
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.88))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
