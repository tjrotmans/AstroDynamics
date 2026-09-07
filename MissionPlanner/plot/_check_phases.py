import csv, math

rows = list(csv.DictReader(open("out/bennu_sample_return/simulate/nav.csv")))

phase_data = {}
for r in rows:
    ph = r["phase"]
    t = float(r["time_s"])
    radius = math.sqrt(float(r["tx_m"])**2 + float(r["ty_m"])**2 + float(r["tz_m"])**2)
    if ph not in phase_data:
        phase_data[ph] = {"times": [], "radii": []}
    phase_data[ph]["times"].append(t)
    phase_data[ph]["radii"].append(radius)

print(f"{'Phase':<15} {'rows':>6}  {'t_start[h]':>11}  {'t_end[h]':>10}  {'mean_r[m]':>10}  {'min_r[m]':>9}  {'max_r[m]':>9}")
for ph, d in phase_data.items():
    ts = d["times"]
    rs = d["radii"]
    print(f"{ph:<15} {len(ts):>6}  {ts[0]/3600:>11.2f}  {ts[-1]/3600:>10.2f}  {sum(rs)/len(rs):>10.1f}  {min(rs):>9.1f}  {max(rs):>9.1f}")

total_dur = float(rows[-1]["time_s"]) / 3600.0
print(f"\nTotal rows: {len(rows)}   Total duration: {total_dur:.1f} h  ({total_dur/24:.1f} days)")
