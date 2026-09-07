"""Summarize a Phase 03 cruise job's step history (2026-09-01).

Reads `/api/simulate/{id}/steps` saved to a file and prints: every
gain-schedule point (which law/mode/activity flew, with kp/kd/ki/ω_n/ζ),
every TCM-phase transition with pointing error at the transition,
burn-phase pointing quality, dispersion by mission thirds, propellant,
wheel momentum, and planned-burn faults.

Usage:
    py -3 MissionPlanner/scripts/analyze_phase03_steps.py <steps.json>
"""
import json
import sys

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(2)
steps = json.load(open(sys.argv[1], encoding="utf-8"))
if isinstance(steps, dict):
    steps = steps.get("steps", steps)
print(f"n={len(steps)} t {steps[0]['t_s']:.0f}..{steps[-1]['t_s']:.0f} s")

for s in steps:
    g = s.get("gain_schedule_point")
    if g:
        print(f"SCHED t={g['t_s']:.0f} {g['control_mode']}/{g['activity']} law={g['law']} kp={g.get('kp')} kd={g.get('kd')} ki={g.get('ki')} wn={g.get('omega_n_radps')} z={g.get('zeta')}")

prev = None
for s in steps:
    ph = s.get("tcm_phase")
    if ph != prev:
        print(f"t={s['t_s']:.0f} {prev}->{ph} err={s['pointing_error_deg']:.2f} act={s.get('control_activity')} law={s.get('controller_law')} mode={s.get('control_mode')} idx={s.get('planned_burn_idx')} fault={s.get('planned_burn_fault')}")
        prev = ph

burn = [s for s in steps if s.get("tcm_phase") == "Burning"]
if burn:
    errs = [s["pointing_error_deg"] for s in burn]
    print(f"BURN ticks(reported)={len(burn)} t {burn[0]['t_s']:.0f}..{burn[-1]['t_s']:.0f} err max={max(errs):.2f} mean={sum(errs)/len(errs):.2f} last={errs[-1]:.2f}")
    oms = [(s["omega_radps"][0] ** 2 + s["omega_radps"][1] ** 2 + s["omega_radps"][2] ** 2) ** 0.5 for s in burn]
    print(f"BURN |omega| max={max(oms):.5f} mean={sum(oms)/len(oms):.5f} rad/s")

n = len(steps)
for name, seg in [("first 10%", steps[: n // 10]), ("mid", steps[n // 3: 2 * n // 3]), ("last 10%", steps[-n // 10:])]:
    e = [s["pointing_error_deg"] for s in seg]
    d = [s["dr_m"] for s in seg]
    print(f"{name}: err max={max(e):.2f} mean={sum(e)/len(e):.3f} | dr max={max(d):.3e} last={d[-1]:.3e}")
last = steps[-1]
print(f"final: dr={last['dr_m']:.3e} m, prop_remaining={last['propellant_remaining_kg']:.2f} kg, tcm_prop={last['tcm_propellant_kg_cum']:.2f} kg, rcs_prop={last['rcs_propellant_kg_cum']:.3f} kg")
print(f"wheel momentum max={max(s['wheel_momentum_nms'] for s in steps):.2f} N*m*s, wheel_sat max={max(s['wheel_sat_frac'] for s in steps):.2f}")
faults = [(s["t_s"], s["planned_burn_fault"]) for s in steps if s.get("planned_burn_fault")]
print("faults:", faults if faults else "none")
