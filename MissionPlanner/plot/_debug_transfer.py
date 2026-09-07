"""Debug: show unique phases in nav and attitude, and rows around Survey→CloseOrbit boundary."""
import csv, math

def show_boundary(fname, before_phase, after_phase, context=5):
    rows = list(csv.DictReader(open(fname)))
    print(f"\n--- {fname} ---")
    print(f"Unique phases: {list(dict.fromkeys(r['phase'] for r in rows))}")
    # Find transition
    for i in range(1, len(rows)):
        if rows[i-1]["phase"] == before_phase and rows[i]["phase"] != before_phase:
            lo = max(0, i - context)
            hi = min(len(rows), i + context)
            print(f"Transition at index {i}:")
            for j in range(lo, hi):
                r = rows[j]
                try:
                    radius = math.sqrt(float(r["tx_m"])**2+float(r["ty_m"])**2+float(r["tz_m"])**2)
                    print(f"  [{j}] t={float(r['time_s'])/3600:.4f}h  ph={r['phase']:<14}  r={radius:.1f}m")
                except KeyError:
                    print(f"  [{j}] t={float(r['time_s'])/3600:.4f}h  ph={r['phase']}")
            break

show_boundary("out/bennu_sample_return/simulate/nav.csv", "Survey", "CloseOrbit")

# Also check attitude CSV which has every truth step
rows = list(csv.DictReader(open("out/bennu_sample_return/simulate/attitude.csv")))
print(f"\n--- attitude.csv unique phases: {list(dict.fromkeys(r['phase'] for r in rows))}")
# Find Transfer block
in_transfer = False
first_trans = None
last_trans = None
for r in rows:
    if r["phase"] == "Transfer" and not in_transfer:
        in_transfer = True
        first_trans = r
    if r["phase"] != "Transfer" and in_transfer:
        last_trans_prev = r
        break
    if r["phase"] == "Transfer":
        last_trans = r

if first_trans:
    dur = (float(last_trans["time_s"]) - float(first_trans["time_s"])) / 3600.0
    print(f"First Transfer block: t={float(first_trans['time_s'])/3600:.3f}h to {float(last_trans['time_s'])/3600:.3f}h  ({dur:.3f}h)")
else:
    print("No Transfer rows found in attitude.csv")
