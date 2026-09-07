"""Phase 03 verification dashboard (13j, 2026-09-01).

Reads a saved `/api/simulate/{id}/steps` JSON (and optionally the
`/result` JSON) of a cruise-seeded run and draws the four things Phase 03
is judged on, each against its own criterion:

  1. Pointing budget   — pointing error vs. time, mode/TCM-phase bands,
                         the burn-ignition gate (0.5°) and a knowledge
                         floor (star-tracker 1-σ) for scale.
  2. Rates and wheels  — |ω|, wheel momentum, wheel speed- and torque-
                         saturation fractions (1.0 = saturated).
  3. Trajectory-following — position dispersion dr (log), velocity
                         dispersion dv, TCM threshold line, planned-burn
                         epochs, reference interpolation floor.
  4. Propellant        — remaining, TCM and RCS cumulative use, main-engine
                         burn windows.

Usage (from the repo root):
    python MissionPlanner/plot/plot_phase03_verification.py <steps.json> [cruise_result.json] [--out fig.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STAR_TRACKER_1SIGMA_DEG = 9.7e-5 * 180.0 / np.pi   # ~20 arcsec, config default
BURN_GATE_DEG = 0.5                                # cruise::DEFAULT_SETTLE_THRESHOLD_DEG

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--out"), "out/phase03_verification.png")
if not args:
    print(__doc__)
    sys.exit(2)
steps = json.load(open(args[0], encoding="utf-8"))
steps = steps["steps"] if isinstance(steps, dict) else steps
result = None
if len(args) > 1:
    result = json.load(open(args[1], encoding="utf-8"))
    result = result.get("result", result)

t_d = np.array([s["t_s"] for s in steps]) / 86400.0
err = np.array([s["pointing_error_deg"] for s in steps])
omega = np.array([np.linalg.norm(s["omega_radps"]) for s in steps]) * 180.0 / np.pi
h_w = np.array([s["wheel_momentum_nms"] for s in steps])
sat = np.array([s["wheel_sat_frac"] for s in steps])
tsat = np.array([s.get("wheel_torque_sat_frac", 0.0) for s in steps])
dr = np.array([s["dr_m"] for s in steps])
dv = np.array([s["dv_mps"] for s in steps])
prop = np.array([s["propellant_remaining_kg"] for s in steps])
tcm_prop = np.array([s["tcm_propellant_kg_cum"] for s in steps])
rcs_prop = np.array([s["rcs_propellant_kg_cum"] for s in steps])
phase = [s.get("tcm_phase") for s in steps]
mode = [s.get("active_mode") for s in steps]

fig, axes = plt.subplots(4, 1, figsize=(13, 15), sharex=True)
fig.suptitle("Phase 03 verification — cruise replay", fontsize=14)


def shade_phases(ax):
    """Shade Slewing / Burning / RcsCorrecting windows."""
    colors = {"Slewing": "#ffd54f", "Burning": "#ef5350", "RcsCorrecting": "#4fc3f7"}
    start = None
    for i in range(len(steps) + 1):
        ph = phase[i] if i < len(steps) else None
        if start is None and ph in colors:
            start = i
        elif start is not None and ph != phase[start]:
            ax.axvspan(t_d[start], t_d[min(i, len(steps) - 1)], color=colors[phase[start]], alpha=0.35, lw=0)
            start = i if ph in colors else None
    for name, c in colors.items():
        ax.plot([], [], color=c, lw=6, alpha=0.5, label=name)


def mode_bands(ax):
    """Label mode transitions along the top of the axis."""
    prev = None
    for i, m in enumerate(mode):
        if m != prev:
            ax.axvline(t_d[i], color="gray", ls=":", lw=0.8)
            ax.text(t_d[i], ax.get_ylim()[1], f" {m}", va="top", fontsize=7, color="gray", rotation=90)
            prev = m


# 1. Pointing budget
ax = axes[0]
ax.semilogy(t_d, np.maximum(err, 1e-6), lw=0.8, color="k", label="pointing error")
ax.axhline(BURN_GATE_DEG, color="r", ls="--", lw=1, label=f"burn ignition gate {BURN_GATE_DEG}°")
ax.axhline(STAR_TRACKER_1SIGMA_DEG, color="b", ls="--", lw=1, label=f"star tracker 1σ {STAR_TRACKER_1SIGMA_DEG:.3f}°")
shade_phases(ax)
ax.set_ylabel("pointing error [deg]")
ax.set_title("1. Pointing budget", loc="left", fontsize=10)
ax.legend(fontsize=7, ncol=3, loc="upper right")
ax.grid(alpha=0.3)
mode_bands(ax)

# 2. Rates and wheels
ax = axes[1]
ax.semilogy(t_d, np.maximum(omega, 1e-6), lw=0.8, color="k", label="|ω| [deg/s]")
ax2 = ax.twinx()
ax2.plot(t_d, sat, color="tab:orange", lw=0.8, label="wheel speed sat. frac")
ax2.plot(t_d, tsat, color="tab:purple", lw=0.8, label="wheel torque sat. frac")
ax2.axhline(1.0, color="tab:red", ls="--", lw=1, label="saturation")
ax2.set_ylabel("saturation fraction")
ax2.set_ylim(0, max(1.2, float(np.nanmax(np.concatenate([sat, tsat])) * 1.1) if len(sat) else 1.2))
ax.set_ylabel("|ω| [deg/s]")
ax.set_title(f"2. Rates and wheels (max momentum {h_w.max():.2f} N·m·s)", loc="left", fontsize=10)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
ax.grid(alpha=0.3)

# 3. Trajectory following
ax = axes[2]
ax.semilogy(t_d, np.maximum(dr, 1.0), lw=0.9, color="k", label="dr (position dispersion) [m]")
ax3 = ax.twinx()
ax3.semilogy(t_d, np.maximum(dv, 1e-6), lw=0.8, color="tab:green", label="dv (velocity dispersion) [m/s]")
ax3.set_ylabel("dv [m/s]")
if result:
    fl = result.get("reference_interpolation_floor_m")
    if fl:
        ax.axhline(fl, color="tab:blue", ls=":", lw=1, label=f"reference interpolation floor {fl:.0f} m")
    for rep in result.get("planned_burn_reports", []) or []:
        e = rep["configured_epoch_s"] / 86400.0
        ax.axvline(e, color="tab:red", ls="--", lw=1)
        ax.text(e, ax.get_ylim()[0] * 3, f" {rep['label']}\n {rep['status']}", fontsize=7, color="tab:red")
shade_phases(ax)
ax.set_ylabel("dr [m]")
ax.set_title(f"3. Trajectory following (final dr {dr[-1]:.3e} m, max {dr.max():.3e} m)", loc="left", fontsize=10)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax3.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
ax.grid(alpha=0.3)

# 4. Propellant
ax = axes[3]
ax.plot(t_d, prop, color="k", lw=1, label="propellant remaining [kg]")
ax.plot(t_d, tcm_prop, color="tab:red", lw=1, label="main-engine (TCM + planned) used [kg]")
ax.plot(t_d, rcs_prop, color="tab:blue", lw=1, label="RCS used [kg]")
shade_phases(ax)
ax.set_ylabel("kg")
ax.set_xlabel("mission time [days]")
ax.set_title(f"4. Propellant (remaining {prop[-1]:.1f} kg, TCM {tcm_prop[-1]:.1f} kg, RCS {rcs_prop[-1]:.2f} kg)", loc="left", fontsize=10)
ax.legend(fontsize=7, loc="upper right")
ax.grid(alpha=0.3)

fig.tight_layout(rect=(0, 0, 1, 0.98))
import os  # noqa: E402
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fig.savefig(out, dpi=130)
print(f"wrote {out}")

# Console summary the plot is judged with.
burn = [i for i, p in enumerate(phase) if p == "Burning"]
coast = [i for i, p in enumerate(phase) if p is None or p == "Coast"]
print(f"pointing: coast mean {err[coast].mean():.3f}° max {err[coast].max():.2f}°" if coast else "pointing: no coast ticks")
if burn:
    print(f"pointing during burns: mean {err[burn].mean():.3f}° max {err[burn].max():.2f}° (gate {BURN_GATE_DEG}°)")
print(f"rates: |ω| max {omega.max():.4f} deg/s; wheel momentum max {h_w.max():.2f} N·m·s; speed sat max {sat.max():.2f}; torque sat max {tsat.max():.2f}")
print(f"trajectory: dr final {dr[-1]:.3e} m max {dr.max():.3e} m; dv final {dv[-1]:.3e} m/s")
print(f"propellant: remaining {prop[-1]:.2f} kg, main-engine used {tcm_prop[-1]:.2f} kg, RCS used {rcs_prop[-1]:.3f} kg")
