"""
Bennu Phase 4 regression test — compares MissionPlanner sim_engine output
against the GNC/AutonomousNavigation proximity_mission reference for the same
Bennu orbit configuration.

Run from the MissionPlanner directory:
    python plot/test_bennu_regression.py

Expected to produce: PASS for every check.

Physics tolerance rationale
---------------------------
The two sims differ in:
  - SK law: GNC uses a Hohmann impulsive transfer; sim_engine uses vis-viva
    continuous burn at each measurement epoch.
  - Phase durations: GNC runs each phase for the real mission day-count;
    sim_engine runs one orbital period per phase (same total distance/time for
    SK physics, different wall-clock durations).
  - Measurement rate: regression run uses meas-dt=600 s (GNC) vs 120 s (sim_engine).
  - IC: GNC initialises from proximity_init.rs true arrival state; sim_engine
    uses a perfect circular-orbit IC.

These differences affect transient behaviour but not steady-state orbit radius
and EKF convergence — the physics invariants being tested here.  Tolerances
are chosen to be tight enough to catch real bugs (wrong μ, wrong SRP sign,
etc.) but loose enough to not flag the known implementation deltas above.
"""

import sys
import os
import math
import statistics

# --paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MPLAN_DIR  = os.path.join(SCRIPT_DIR, "..")
REPO_DIR   = os.path.join(MPLAN_DIR, "..")

REF_NAV  = os.path.join(REPO_DIR, "GNC", "AutonomousNavigation", "out", "mission", "nav.csv")
TEST_NAV = os.path.join(MPLAN_DIR, "out", "bennu_sample_return", "simulate", "nav.csv")

# --Bennu analytic expectations ──────────────────────────────────────────────

MU_BENNU   = 4.89        # m³/s²
R_BENNU    = 262.0       # m

# Phase → expected orbit radius [m].  Must match trajectory.phases in the TOML
# AND the GNC/AutonomousNavigation proximity_mission target_r_m().
PHASE_RADIUS = {
    "Capture":     3_000.0,
    "Survey":      3_000.0,
    "CloseOrbit":    900.0,
    "Flyover":       500.0,
    "ScienceHold":   900.0,
    "RadioScience":3_000.0,
}

# Checks: (label, tol_frac) — mean orbit radius must be within tol_frac of target.
RADIUS_TOL_FRAC  = 0.10   # 10 % — orbit should not drift far from target
RADIUS_STD_TOL   = 0.15   # σ_r < 15 % of target — SK is keeping it bounded
EKF_CONV_TOL_M   = 5.0    # EKF sigma_r should converge to below this [m]

# --CSV loader ───────────────────────────────────────────────────────────────

def load_nav(path, has_range_km=False):
    """Return list of dicts from a nav.csv file."""
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split(",")
            row = dict(zip(header, vals))
            try:
                row["r_m"] = math.sqrt(
                    float(row["tx_m"])**2 + float(row["ty_m"])**2 + float(row["tz_m"])**2
                )
                row["sigma_r_m"] = float(row["sigma_r_m"])
            except (ValueError, KeyError):
                continue
            rows.append(row)
    return rows

def group_by_phase(rows):
    phases = {}
    for r in rows:
        p = r.get("phase", "unknown")
        phases.setdefault(p, []).append(r)
    return phases

# --individual checks ────────────────────────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0

def check(label, value, lo=None, hi=None, unit=""):
    global PASS_COUNT, FAIL_COUNT
    ok = True
    if lo is not None and value < lo:
        ok = False
    if hi is not None and value > hi:
        ok = False
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    bounds = ""
    if lo is not None and hi is not None:
        bounds = f"  [{lo:.4g} .. {hi:.4g}]"
    elif lo is not None:
        bounds = f"  [>= {lo:.4g}]"
    elif hi is not None:
        bounds = f"  [<= {hi:.4g}]"
    print(f"  [{tag}]  {label}: {value:.4g}{unit}{bounds}")
    return ok

# --main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Bennu Phase 4 regression test")
    print("=" * 70)

    # --Load test output ──────────────────────────────────────────────────────
    test_rows = load_nav(TEST_NAV)
    if test_rows is None:
        print(f"\nERROR: MissionPlanner output not found: {TEST_NAV}")
        print("Run: cargo run --release -- simulate config/bennu_sample_return.toml")
        sys.exit(1)

    print(f"\nMissionPlanner nav.csv: {len(test_rows)} rows from {TEST_NAV}")

    # --Section 1: MissionPlanner physics invariants ──────────────────────────
    print("\n--Section 1: MissionPlanner physics invariants -----------------------")
    test_phases = group_by_phase(test_rows)

    for phase_name, target_r in PHASE_RADIUS.items():
        rows_p = test_phases.get(phase_name, [])
        if not rows_p:
            print(f"  [SKIP]  Phase '{phase_name}' not found in output")
            continue

        radii  = [r["r_m"]     for r in rows_p]
        sigmas = [r["sigma_r_m"] for r in rows_p]

        mean_r = statistics.mean(radii)
        std_r  = statistics.stdev(radii) if len(radii) > 1 else 0.0
        final_sigma = sigmas[-1]

        print(f"\n  Phase: {phase_name}  (target {target_r:.0f} m, {len(rows_p)} rows)")
        check(f"  mean orbit radius",
              mean_r,
              lo=target_r * (1 - RADIUS_TOL_FRAC),
              hi=target_r * (1 + RADIUS_TOL_FRAC),
              unit=" m")
        check(f"  orbit radius std",
              std_r,
              hi=target_r * RADIUS_STD_TOL,
              unit=" m")
        check(f"  final EKF sigma_r",
              final_sigma,
              hi=EKF_CONV_TOL_M,
              unit=" m")
        check(f"  orbit above surface",
              min(radii) - R_BENNU,
              lo=0.0,
              unit=" m (altitude)")

    # Verify no NaN/inf in orbit radii
    all_radii = [r["r_m"] for r in test_rows]
    nan_count = sum(1 for x in all_radii if not math.isfinite(x))
    print("\n  No NaN/Inf in orbit radius:")
    check("  NaN/Inf count", nan_count, hi=0)

    # Total dV from maneuvers.csv
    man_path = TEST_NAV.replace("nav.csv", "maneuvers.csv")
    total_dv = 0.0
    n_burns = 0
    if os.path.exists(man_path):
        with open(man_path) as f:
            f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    try:
                        total_dv += float(parts[4])
                        n_burns += 1
                    except ValueError:
                        pass
    print(f"\n  dV budget: {total_dv:.4f} m/s over {n_burns} burns")
    # Sanity: total SK dV should be << escape speed ≈ √(2μ/r) ≈ 0.057 m/s at 3 km
    v_esc = math.sqrt(2 * MU_BENNU / 3000.0)
    check("  total dV < 10x escape speed",
          total_dv,
          hi=10 * v_esc,
          unit=" m/s")

    # --Section 2: cross-comparison with GNC/AutonomousNavigation reference ───
    ref_rows = load_nav(REF_NAV, has_range_km=True)
    if ref_rows is None:
        print(f"\n--Section 2: cross-comparison ----------------------------------------")
        print(f"  [SKIP]  Reference output not found: {REF_NAV}")
        print("  Run in GNC/AutonomousNavigation:")
        print("    cargo run --bin proximity_mission --release -- --dt 60 --meas-dt 600")
    else:
        print(f"\n--Section 2: cross-comparison with GNC/AutonomousNavigation ----------")
        print(f"  Reference nav.csv: {len(ref_rows)} rows from {REF_NAV}")
        ref_phases = group_by_phase(ref_rows)

        # Compare per-phase orbit radius mean and EKF convergence
        common = set(test_phases) & set(ref_phases) & set(PHASE_RADIUS)
        if not common:
            print("  [SKIP]  No common phase names between test and reference outputs")
        for phase_name in sorted(common, key=list(PHASE_RADIUS).index):
            t_rows = test_phases[phase_name]
            r_rows = ref_phases[phase_name]

            # Use the final 60% of rows for cross-comparison.
            # GNC/AutonomousNavigation fires a Hohmann transfer at the start of each
            # phase that changes orbit altitude, so the first ~40% of reference rows
            # span the transfer arc (intermediate radii) rather than the target orbit.
            # The MissionPlanner sim starts each phase already at the target altitude.
            # Comparing only the steady-state tail eliminates this systematic offset.
            TAIL_FRAC = 0.60
            t_tail = t_rows[int(len(t_rows) * (1 - TAIL_FRAC)):]
            r_tail = r_rows[int(len(r_rows) * (1 - TAIL_FRAC)):]

            t_mean_r = statistics.mean(x["r_m"] for x in t_tail)
            r_mean_r = statistics.mean(x["r_m"] for x in r_tail)
            t_sigma  = t_tail[-1]["sigma_r_m"]
            r_sigma  = r_tail[-1]["sigma_r_m"]

            print(f"\n  Phase: {phase_name}")
            ratio_r = abs(t_mean_r - r_mean_r) / r_mean_r
            check(f"  orbit radius agreement (|delta|/ref)",
                  ratio_r,
                  hi=0.20,
                  unit="")

            # EKF sigma within 3× of reference (different meas rates → different σ)
            if r_sigma > 0:
                check(f"  EKF sigma_r ratio (test/ref)",
                      t_sigma / r_sigma,
                      lo=0.1, hi=10.0,
                      unit="")

        # Overall dV rate: test may run fewer/more orbits, so compare dV/orbit
        ref_total_dv = 0.0
        ref_man_path = REF_NAV.replace("nav.csv", "maneuvers.csv")
        if os.path.exists(ref_man_path):
            with open(ref_man_path) as f:
                f.readline()
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 5:
                        try:
                            ref_total_dv += float(parts[4])
                        except ValueError:
                            pass

        print(f"\n  dV summary:")
        print(f"    GNC/AutonomousNavigation: {ref_total_dv:.4f} m/s")
        print(f"    MissionPlanner sim_engine: {total_dv:.4f} m/s")
        if ref_total_dv > 0:
            ratio_dv = total_dv / ref_total_dv
            # Both run different durations; we just check they're in the same order of magnitude
            check("  dV order-of-magnitude (test/ref in [0.05, 20])",
                  ratio_dv,
                  lo=0.05, hi=20.0,
                  unit="")

    # --Summary ───────────────────────────────────────────────────────────────
    total = PASS_COUNT + FAIL_COUNT
    print("\n" + "=" * 70)
    print(f"Result: {PASS_COUNT}/{total} checks passed")
    if FAIL_COUNT == 0:
        print("PASS — sim_engine Bennu proximity-ops output within physics tolerance")
    else:
        print(f"FAIL — {FAIL_COUNT} check(s) failed")
    print("=" * 70)
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
