"""Coarse timeline + per-burn table for a Phase 03 cruise job (2026-09-01).

Prints dispersion (dr, dv), third-body acceleration, propellant and TCM
phase every ~2 days, then one line per main-engine burn: when it fired,
the dispersion it was chasing, the propellant it consumed, and the
dispersion/velocity error it left behind — the fastest way to tell a
correct small correction from a runaway one.

Usage:
    py -3 MissionPlanner/scripts/phase03_timeline.py <steps.json> [cadence_days]
"""
import json
import sys

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(2)
s = json.load(open(sys.argv[1], encoding="utf-8"))
s = s["steps"] if isinstance(s, dict) else s
cadence_s = float(sys.argv[2]) * 86400.0 if len(sys.argv) > 2 else 172800.0

last = -1e18
print(f"coarse timeline (every ~{cadence_s/86400:.1f} days):")
for r in s:
    if r["t_s"] - last >= cadence_s:
        print(f"  t={r['t_s']/86400:6.1f} d dr={r['dr_m']:.3e} dv={r['dv_mps']:.3e} a3b={r['accel_third_body_mps2']:.2e} prop={r['propellant_remaining_kg']:.1f} ph={r.get('tcm_phase')}")
        last = r["t_s"]

print("burns:")
prev_ph, start = None, None
for r in s:
    ph = r.get("tcm_phase")
    if ph == "Burning" and prev_ph != "Burning":
        start = r
    if ph != "Burning" and prev_ph == "Burning" and start is not None:
        print(f"  burn {start['t_s']/86400:6.2f} d -> {r['t_s']/86400:6.2f} d  dr_at_start={start['dr_m']:.3e} dv_at_start={start['dv_mps']:.2f} prop {start['propellant_remaining_kg']:.1f} -> {r['propellant_remaining_kg']:.1f} kg, dr_after={r['dr_m']:.3e} dv_after={r['dv_mps']:.2f}")
    prev_ph = ph
