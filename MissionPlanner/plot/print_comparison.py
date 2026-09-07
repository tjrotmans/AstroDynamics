"""Print side-by-side per-phase comparison of GNC/AutonomousNavigation vs sim_engine."""
import math, statistics, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MPLAN_DIR  = os.path.join(SCRIPT_DIR, "..")
REPO_DIR   = os.path.join(MPLAN_DIR, "..")

REF_NAV  = os.path.join(REPO_DIR, "GNC", "AutonomousNavigation", "out", "mission", "nav.csv")
TEST_NAV = os.path.join(MPLAN_DIR, "out", "bennu_sample_return", "simulate", "nav.csv")
REF_MAN  = REF_NAV.replace("nav.csv", "maneuvers.csv")
TEST_MAN = TEST_NAV.replace("nav.csv", "maneuvers.csv")

PHASES = ["Capture", "Survey", "CloseOrbit", "Flyover", "ScienceHold", "RadioScience"]
PHASE_TARGET = {"Capture":3000, "Survey":3000, "CloseOrbit":900, "Flyover":500,
                "ScienceHold":900, "RadioScience":3000}


def load_nav(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        hdr = f.readline().strip().split(",")
        for line in f:
            v = line.strip().split(",")
            if len(v) < len(hdr):
                continue
            d = dict(zip(hdr, v))
            try:
                d["r"] = math.sqrt(float(d["tx_m"])**2 + float(d["ty_m"])**2 + float(d["tz_m"])**2)
                d["sig"] = float(d["sigma_r_m"])
                d["t"] = float(d["time_s"])
                rows.append(d)
            except (ValueError, KeyError):
                pass
    return rows


def group_by_phase(rows):
    g = {}
    for r in rows:
        g.setdefault(r.get("phase","?"), []).append(r)
    return g


def phase_stats(rows):
    if not rows:
        return None
    rs = [r["r"] for r in rows]
    return {
        "n":    len(rows),
        "dur":  (rows[-1]["t"] - rows[0]["t"]) / 86400.0,
        "mean": statistics.mean(rs),
        "std":  statistics.stdev(rs) if len(rs) > 1 else 0.0,
        "min":  min(rs),
        "max":  max(rs),
        "sig0": rows[0]["sig"],
        "sig1": rows[-1]["sig"],
    }


def dv_total(path):
    if not os.path.exists(path):
        return None, 0
    total, n = 0.0, 0
    with open(path) as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 5:
                try:
                    total += float(p[4]); n += 1
                except ValueError:
                    pass
    return total, n


ref_rows  = load_nav(REF_NAV)
test_rows = load_nav(TEST_NAV)

if ref_rows is None:
    print("GNC/AutonomousNavigation output not found:", REF_NAV)
    sys.exit(1)
if test_rows is None:
    print("MissionPlanner output not found:", TEST_NAV)
    sys.exit(1)

ref_ph  = group_by_phase(ref_rows)
test_ph = group_by_phase(test_rows)

C1, C2, C3, C4 = 22, 16, 16, 10


def hline(ch="="):
    print(ch * (C1 + C2 + C3 + C4 + 6))


def row(label, gv, mv, fmt=".1f", unit=""):
    def fmt_val(v):
        if v is None:
            return "n/a".rjust(C2)
        if fmt == ".0f":
            return f"{v:.0f}{unit}".rjust(C2)
        return f"{v:{fmt}}{unit}".rjust(C2)
    d = ""
    if gv is not None and mv is not None and gv != 0:
        pct = (mv - gv) / abs(gv) * 100.0
        d = f"{pct:+.1f}%".rjust(C4)
    print(f"  {label:<{C1}} {fmt_val(gv)} {fmt_val(mv)} {d}")


hline()
print(f"  Bennu proximity-ops comparison: GNC/AutoNav vs MissionPlanner sim_engine")
print(f"  GNC ref : {len(ref_rows)} rows   MissionPlanner: {len(test_rows)} rows")
hline()
print(f"  {'Metric':<{C1}} {'GNC/AutoNav':>{C2}} {'MissionPlanner':>{C2}} {'delta%':>{C4}}")
hline("-")

for ph in PHASES:
    gs = phase_stats(ref_ph.get(ph))
    ms = phase_stats(test_ph.get(ph))
    tgt = PHASE_TARGET.get(ph, 0)

    print(f"\n  Phase: {ph}  (target {tgt} m)")
    row("rows (meas)",       gs["n"]    if gs else None, ms["n"]    if ms else None, ".0f")
    row("duration [days]",   gs["dur"]  if gs else None, ms["dur"]  if ms else None, ".3f")
    row("mean radius [m]",   gs["mean"] if gs else None, ms["mean"] if ms else None, ".2f")
    row("std radius [m]",    gs["std"]  if gs else None, ms["std"]  if ms else None, ".2f")
    row("min radius [m]",    gs["min"]  if gs else None, ms["min"]  if ms else None, ".2f")
    row("max radius [m]",    gs["max"]  if gs else None, ms["max"]  if ms else None, ".2f")
    row("EKF sigma_r t0 [m]",gs["sig0"] if gs else None, ms["sig0"] if ms else None, ".4f")
    row("EKF sigma_r tf [m]",gs["sig1"] if gs else None, ms["sig1"] if ms else None, ".4f")

    # altitude above Bennu (262 m radius)
    if gs:
        print(f"  {'min altitude [m]':<{C1}} {gs['min']-262:>{C2}.1f} {ms['min']-262 if ms else 'n/a':>{C2}} ")
    hline("-")

ref_dv, ref_nb  = dv_total(REF_MAN)
test_dv, test_nb = dv_total(TEST_MAN)

print(f"\n  Total dV budget:")
print(f"    GNC/AutoNav     : {ref_dv:.5f} m/s  ({ref_nb} burns)")
print(f"    MissionPlanner  : {test_dv:.5f} m/s  ({test_nb} burns)")
if ref_dv and ref_dv > 0:
    print(f"    ratio test/ref  : {test_dv/ref_dv:.3f}")

hline()
print()
