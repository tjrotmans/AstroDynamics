"""
SA optimizer performance analysis for the Earth-Mars interplanetary transfer.

Reads from out/interplanetary_transfer.db and produces a 2×2 figure:

  (0,0) Convergence  — best-so-far energy vs generation, with accepted/rejected scatter
  (0,1) Energy distribution — histogram of all evaluated energies in the run
  (1,0) Exploration dynamics — rolling acceptance rate + temperature (dual axis)
  (1,1) TOF evolution — best TOF found so far at each generation

Run from the workspace root:
    python examples/interplanetary_transfer/plot_optimizer.py

Requirements:
    pip install matplotlib numpy
"""

import sqlite3
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

DB_PATH = "out/interplanetary_transfer.db"

COLOR_SA    = "#2196F3"
COLOR_LIGHT = "#90CAF9"
COLOR_ACC   = "#4CAF50"
COLOR_REJ   = "#F44336"

# ── Load data ──────────────────────────────────────────────────────────────────

db_path = Path(DB_PATH)
if not db_path.exists():
    print(f"Error: {db_path} not found — run the optimizer first.")
    raise SystemExit(1)

conn = sqlite3.connect(str(db_path))

run = conn.execute(
    "SELECT run_id, generation_count FROM optimization_runs ORDER BY id DESC LIMIT 1"
).fetchone()
if run is None:
    print("No optimization_runs found in database.")
    raise SystemExit(1)

run_id, gen_count = run
print(f"Run: {run_id}  (up to {gen_count} generations recorded)")

rows = conn.execute(
    "SELECT g.generation, g.energy, g.accepted, g.temperature, "
    "       ec.tof_days, ec.pos_err_au, ec.vel_err_kms "
    "FROM generations g "
    "LEFT JOIN energy_components ec ON g.eval_id = ec.eval_id "
    "WHERE g.run_id = ? ORDER BY g.generation",
    (run_id,),
).fetchall()
conn.close()

if not rows:
    print("No generation data found.")
    raise SystemExit(1)

gens        = np.array([r[0] for r in rows], dtype=float)
energies    = np.array([float(r[1]) for r in rows])
accepted    = np.array([r[2] == "true" for r in rows])
temps       = np.array([float(r[3]) for r in rows])
tof_vals    = np.array([float(r[4]) if r[4] is not None else float("nan") for r in rows])
pos_err_au  = np.array([float(r[5]) if r[5] is not None else float("nan") for r in rows])
vel_err_kms = np.array([float(r[6]) if r[6] is not None else float("nan") for r in rows])

have_components = not np.all(np.isnan(tof_vals))

best_idx = int(np.argmin(energies))
print(f"  {len(rows)} evaluations loaded")
print(f"  Best energy : {energies.min():.4f}")
print(f"  Acceptance  : {accepted.mean()*100:.1f}%")
if have_components:
    print(f"  Best TOF    : {tof_vals[best_idx]:.0f} days")
    print(f"  Best pos err: {pos_err_au[best_idx]:.4f} AU")
    print(f"  Best vel err: {vel_err_kms[best_idx]:.2f} km/s")

# ── Build best-so-far traces ───────────────────────────────────────────────────

best_idx_trace = []
best_i = 0
for i in range(len(energies)):
    if energies[i] < energies[best_i]:
        best_i = i
    best_idx_trace.append(best_i)

best_so_far         = energies[best_idx_trace]
best_tof_trace      = tof_vals[best_idx_trace]
best_pos_err_trace  = pos_err_au[best_idx_trace]
best_vel_err_trace  = vel_err_kms[best_idx_trace]

# ── Figure ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    "SA Optimizer Performance — Earth\u2192Mars Solar Sail Transfer",
    fontsize=13, fontweight="bold",
)

# ══════════════════════════════════════════════════════════════════════════
# (0,0) Convergence
# ══════════════════════════════════════════════════════════════════════════
ax = axes[0, 0]

ax.scatter(gens[~accepted], energies[~accepted],
           s=2, color=COLOR_REJ, alpha=0.12, label="Rejected", rasterized=True)
ax.scatter(gens[accepted],  energies[accepted],
           s=2, color=COLOR_ACC, alpha=0.25, label="Accepted", rasterized=True)
ax.plot(gens, best_so_far, color=COLOR_SA, lw=2.0, label="Best-so-far", zorder=5)

ax.set_xlabel("Generation")
ax.set_ylabel("Energy (lower = better)")
ax.set_title("Convergence")
ax.legend(fontsize=8, markerscale=4)
ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════════════════
# (0,1) Energy distribution
# ══════════════════════════════════════════════════════════════════════════
ax = axes[0, 1]

p1, p99 = np.percentile(energies, [1, 99])
clipped = energies[(energies >= p1) & (energies <= p99)]

ax.hist(clipped, bins=60, color=COLOR_LIGHT, edgecolor=COLOR_SA,
        linewidth=0.4, alpha=0.85, label="All evaluations (1st–99th pct)")
ax.axvline(energies.min(), color="gold", lw=2.0, ls="--",
           label=f"Best: {energies.min():.4f}")
ax.axvline(float(np.median(energies)), color="gray", lw=1.2, ls=":",
           label=f"Median: {np.median(energies):.4f}")

ax.set_xlabel("Energy")
ax.set_ylabel("Count")
ax.set_title("Energy Distribution")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# ══════════════════════════════════════════════════════════════════════════
# (1,0) SA exploration dynamics — acceptance rate + temperature
# ══════════════════════════════════════════════════════════════════════════
ax = axes[1, 0]

window = max(1, min(100, len(accepted) // 20))
rolling_acc = np.convolve(
    accepted.astype(float), np.ones(window) / window, mode="valid"
)
rolling_x = np.arange(window - 1, len(accepted))

ax.plot(rolling_x, rolling_acc * 100,
        color=COLOR_ACC, lw=1.4, label=f"Acceptance rate (window = {window})")
ax.set_ylabel("Acceptance rate [%]", color=COLOR_ACC)
ax.set_ylim(0, 105)
ax.tick_params(axis="y", labelcolor=COLOR_ACC)

ax2 = ax.twinx()
ax2.plot(gens, temps, color="gray", lw=1.0, alpha=0.5, label="Temperature")
ax2.set_ylabel("Temperature", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")

ax.set_xlabel("Generation")
ax.set_title("SA Exploration Dynamics")
ax.grid(True, alpha=0.3)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# ══════════════════════════════════════════════════════════════════════════
# (1,1) Combined: TOF + pos_err + vel_err convergence
#
# All three are normalized to [0, 1] so they share one y-axis.
# Normalization: divide each best-so-far trace by its own starting value,
# so y=1 is "where we started" and y=0 is "perfect".
# ══════════════════════════════════════════════════════════════════════════
ax = axes[1, 1]

if have_components:
    def safe_normalize(arr):
        """Normalize trace to [0, 1] relative to its maximum, guarding against zeros."""
        m = np.nanmax(arr)
        return arr / m if m > 0 else arr

    pos_norm = safe_normalize(best_pos_err_trace)
    vel_norm = safe_normalize(best_vel_err_trace)

    # TOF: normalize to [0, 1] over the search range
    tof_min_val = np.nanmin(tof_vals)
    tof_max_val = np.nanmax(tof_vals)
    tof_range = tof_max_val - tof_min_val
    if tof_range > 0:
        tof_norm = (best_tof_trace - tof_min_val) / tof_range
    else:
        tof_norm = np.zeros_like(best_tof_trace)

    final_pos = pos_err_au[best_idx]
    final_vel = vel_err_kms[best_idx]
    final_tof = tof_vals[best_idx]

    ax.plot(gens, pos_norm, color="steelblue", lw=2.0,
            label=f"pos err (best: {final_pos:.3f} AU)")
    ax.plot(gens, vel_norm, color="darkorange", lw=2.0, ls="--",
            label=f"vel err (best: {final_vel:.1f} km/s)")
    ax.plot(gens, tof_norm, color="gray", lw=1.5, ls=":",
            label=f"TOF rel. range (best: {final_tof:.0f} d)")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_ylabel("Normalised value (0 = min, 1 = max in run)")
else:
    ax.text(0.5, 0.5, "Requires energy_components table.\nRe-run optimizer to populate.",
            transform=ax.transAxes, ha="center", va="center", fontsize=9)

ax.set_xlabel("Generation")
ax.set_title("Best-so-far: TOF / pos err / vel err")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Summary footer ─────────────────────────────────────────────────────────────

n_total  = len(energies)
n_accept = accepted.sum()
if have_components:
    summary = (
        f"Evaluations: {n_total}  |  Best energy: {energies.min():.4f}  |  "
        f"Best TOF: {tof_vals[best_idx]:.0f} d  |  "
        f"pos err: {pos_err_au[best_idx]:.4f} AU  |  "
        f"vel err: {vel_err_kms[best_idx]:.1f} km/s  |  "
        f"Acceptance: {n_accept/n_total*100:.1f}%"
    )
else:
    summary = (
        f"Evaluations: {n_total}  |  Best energy: {energies.min():.4f}  |  "
        f"Acceptance: {n_accept}/{n_total} ({n_accept/n_total*100:.1f}%)"
    )
fig.text(
    0.5, 0.01, summary, ha="center", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"),
)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

out_path = "out/optimizer_performance.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
