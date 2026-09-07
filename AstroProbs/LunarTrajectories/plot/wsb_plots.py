"""
wsb_plots.py — unified WSB plotting script.

Single entry point for all WSB visualisations.  Toggle each group on/off
in the CONFIG block below, then run:

    python plot/wsb_plots.py

Each plot group calls the corresponding script as a subprocess (same approach
as wsb_pipeline.rs for the Rust binaries), so all existing scripts stay intact
and can still be run individually.

# Config reference
──────────────────────────────────────────────────────────────────────────────
Plot groups:
    search      — search-result plots  (trajectories, Pareto front, heatmap)
    phases      — phase-by-phase diagnostics  (backward arcs, α screen, fwd)
    solutions   — MC + GA Pareto front + SOI-reaching trajectories
    sensitivity — sensitivity ensemble  (run wsb_sensitivity first)
    anim        — animations  (run wsb_refine --reprop first for solution anim)
    artemis     — Artemis II comparison  (needs ephemeris files)
    multi_anim  — 10 fastest MC refined solutions, dual-panel animation
    param_sens  — parameter sensitivity: how θ / θ_sun / r_apogee affect ΔV
──────────────────────────────────────────────────────────────────────────────
"""

import pathlib
import subprocess
import sys
from typing import List

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      USER CONFIGURATION — edit here                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CONFIG = {
    # ── Plot groups ───────────────────────────────────────────────────────────
    # Pipeline order: wsb_search → wsb_optimize (GA) → wsb_refine (MC) → wsb_maxhifi
    "solutions":   True,  # plot_wsb_solutions.py   — GA global + MC refined Pareto + trajectories
    "sensitivity": True,  # plot_wsb_final_refinement.py — sensitivity ensemble
    "anim":        True,   # plot_wsb_anim.py + plot_wsb_solution_anim.py
    "artemis":     False,  # plot_wsb_vs_artemis.py  (needs ephemeris files)
    "multi_anim":  True,   # plot_wsb_multi_anim.py  — 5 fastest MC solutions, dual-panel
    "param_sens":  True,   # plot_wsb_covariance.py  — parameter sensitivity (θ, θ_sun, r_apo, α)
}

# ════════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = pathlib.Path(__file__).parent
ROOT       = SCRIPT_DIR.parent


def run(script_name: str, args: List[str] = ()) -> bool:
    """Run a sibling plot script. Returns True on success, non-fatal on failure."""
    script = SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script)] + list(args)
    print(f"  → {script_name}" + (f"  {' '.join(args)}" if args else ""))
    try:
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"  [wsb_plots] {script_name} exited with code {result.returncode} "
                  f"(non-fatal, continuing)")
            return False
        return True
    except FileNotFoundError:
        print(f"  [wsb_plots] {script_name} not found (skipping)")
        return False


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║             WSB Plots — unified plotter              ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    groups_on = [k for k, v in CONFIG.items() if isinstance(v, bool) and v]
    print(f"  Enabled : {', '.join(groups_on) if groups_on else 'none'}")
    print()

    # ── MC + GA solution comparison ───────────────────────────────────────────
    if CONFIG["solutions"]:
        print("── MC + GA solutions ──────────────────────────────────────────")
        run("plot_wsb_solutions.py")
        print()

    # ── Sensitivity ensemble ──────────────────────────────────────────────────
    if CONFIG["sensitivity"]:
        print("── Sensitivity ensemble ───────────────────────────────────────")
        run("plot_wsb_final_refinement.py")
        print()

    # ── Animations ───────────────────────────────────────────────────────────
    if CONFIG["anim"]:
        print("── Animations ─────────────────────────────────────────────────")
        run("plot_wsb_anim.py")
        run("plot_wsb_solution_anim.py")
        print()

    # ── Artemis comparison ────────────────────────────────────────────────────
    if CONFIG["artemis"]:
        print("── Artemis II comparison ──────────────────────────────────────")
        run("plot_wsb_vs_artemis.py")
        run("plot_wsb_vs_artemis_circularize.py")
        print()

    # ── Multi-candidate animation ─────────────────────────────────────────────
    if CONFIG["multi_anim"]:
        print("── Multi-candidate animation ───────────────────────────────────")
        run("plot_wsb_multi_anim.py")
        print()

    # ── Parameter sensitivity ─────────────────────────────────────────────────
    if CONFIG["param_sens"]:
        print("── Parameter sensitivity ───────────────────────────────────────")
        run("plot_wsb_covariance.py")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    out = ROOT / "out" / "wsb"
    htmls = sorted(out.glob("*.html")) if out.exists() else []
    print("═" * 60)
    print(f"  Done.  {len(htmls)} HTML files in {out}/")
    for h in htmls:
        print(f"    {h.name}")


if __name__ == "__main__":
    main()
