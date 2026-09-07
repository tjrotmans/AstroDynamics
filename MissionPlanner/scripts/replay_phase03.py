"""Server-side Phase 03 replay of an adopted Phase 01 result (2026-09-01).

Assembles a `cruise_seed` EXACTLY the way the frontend's
`AstroDynamics-UI/src/lib/cruiseSeed.ts` does and POSTs it to a running
mission-server, so a mission can be flown and inspected without the UI:

- reference = pre_departure_orbit_arc (shifted so its last point lands on
  arc[0]) + arc clamped to achieved_tof_days + post_capture_orbit_arc
  (shifted onto achieved_tof), rebased so reference[0].t_s == 0, with the
  real vx/vy/vz velocities;
- planned burns: departure (real departure_dv_inertial_mps, marked
  `external_stage` — a launcher-provided injection) and capture
  (capture_dv_inertial_mps, `capture_body` = target);
- body tracks: departure/target bodies as soi_capture + every AlwaysThirdBody
  of the Phase 01 force model, SERVER-RESOLVED (empty track) and anchored at
  the result's `dep_jd` (NOT the config's nominal departure epoch — the
  optimizer's dep_offset_days can be weeks);
- modes: SolarPanel -> SunPointing, first StarTracker/OpNavCamera ->
  TargetPointing, 0.9 split; tick 10 s; report stride from a 20k budget.

Usage (from the repo root, server on :8000):
    py -3 MissionPlanner/scripts/replay_phase03.py <config.json> <optimize_result.json> [target_body]
        [--window START_D END_D] [--tcm-threshold-m X] [--tick-s T]
`config.json` is the full MissionConfig JSON the frontend sends to
/api/optimize; `optimize_result.json` is `/api/optimize/{id}/result` (or a
presetSnapshots/*-optimize.json). Prints the job id; then poll
/api/simulate/{id}/steps and analyze with analyze_phase03_steps.py /
phase03_timeline.py / plot/plot_phase03_verification.py.
`--window` (days on the rebased cruise clock; END_D may be given relative
to the end as a negative number, e.g. `--window -30 0` = the last 30 days)
flies only that segment starting from the reference state — the fast way
to re-test an approach/burn change (2026-09-01, `cruise_seed.window`).
"""
import datetime
import json
import sys
import urllib.request


def _opt(name, n, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        vals = sys.argv[i + 1: i + 1 + n]
        del sys.argv[i: i + 1 + n]
        return vals if n > 1 else vals[0]
    return default


window_arg = _opt("--window", 2, None)
initial_dr = _opt("--initial-dr-m", 3, None)       # [x y z] m added to the reference state at the window start
initial_dv = _opt("--initial-dv-mps", 3, None)     # [x y z] m/s
tcm_threshold_m = float(_opt("--tcm-threshold-m", 1, 50000.0))
tick_s = float(_opt("--tick-s", 1, 10.0))

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(2)
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
opt = json.load(open(sys.argv[2], encoding="utf-8"))
if isinstance(opt, dict) and "result" in opt:
    opt = opt["result"]
target = sys.argv[3] if len(sys.argv) > 3 else cfg["target_body"]["name"]
departure_body = cfg["trajectory"].get("departure_body", "Earth")

achieved_s = opt["achieved_tof_days"] * 86400.0
arc = [p for p in opt["arc"] if p["t_s"] <= achieved_s]
pre = opt.get("pre_departure_orbit_arc") or []
post = opt.get("post_capture_orbit_arc") or []


def shift(points, anchor_local, anchor_abs):
    d = anchor_abs - anchor_local
    return [dict(p, t_s=p["t_s"] + d) for p in points]


full = (shift(pre, pre[-1]["t_s"], arc[0]["t_s"]) if pre else []) + arc
if post:
    full = full + shift(post, post[0]["t_s"], achieved_s)
t0 = full[0]["t_s"]
assert all(p.get("vx_mps") is not None for p in full), "expected real velocities on every arc point"
reference = [
    {"t_s": p["t_s"] - t0, "r_m": [p["x_m"], p["y_m"], p["z_m"]], "v_mps": [p["vx_mps"], p["vy_mps"], p["vz_mps"]]}
    for p in full
]
duration_s = reference[-1]["t_s"]

first = arc[0]
dep_dv = opt.get("departure_dv_inertial_mps")
if not (dep_dv and len(dep_dv) == 3):
    last_pre = pre[-1]
    dep_dv = [first["vx_mps"] - last_pre["vx_mps"], first["vy_mps"] - last_pre["vy_mps"], first["vz_mps"] - last_pre["vz_mps"]]
    print("departure dv: interim derivation (no departure_dv_inertial_mps in result)")
else:
    print("departure dv: real departure_dv_inertial_mps")
burns = [{
    "epoch_s": first["t_s"] - t0,
    "dv_inertial_mps": dep_dv,
    "label": "departure burn (launch vehicle upper stage)",
    "external_stage": True,
}]
cap_t = opt.get("capture_time_s")
if cap_t is None:
    cap_t = achieved_s
if opt.get("capture_dv_inertial_mps"):
    burns.append({
        "epoch_s": cap_t - t0,
        "dv_inertial_mps": opt["capture_dv_inertial_mps"],
        "label": "arrival/capture burn",
        "capture_body": target,
    })

dep = datetime.datetime.strptime(cfg["trajectory"]["departure_epoch"].replace(" UTC", ""), "%Y-%m-%dT%H:%M:%S")
jd_nominal = 2440587.5 + (dep - datetime.datetime(1970, 1, 1)).total_seconds() / 86400.0
jd_dep = opt.get("dep_jd") or jd_nominal
epoch_jd = jd_dep + t0 / 86400.0
print(f"track anchor: dep_jd {jd_dep:.4f} (nominal {jd_nominal:.4f}, offset {jd_dep - jd_nominal:.2f} d), epoch_jd {epoch_jd:.4f}")

tracks = [
    {"name": departure_body, "track": [], "epoch_jd": epoch_jd, "soi_capture": True},
    {"name": target, "track": [], "epoch_jd": epoch_jd, "soi_capture": True},
]
for b in (cfg.get("optimization", {}).get("force_model", {}) or {}).get("bodies", []):
    if b.get("role") == "AlwaysThirdBody" and b["name"] not in (departure_body, target, "Sun"):
        tracks.append({"name": b["name"], "track": [], "epoch_jd": epoch_jd, "soi_capture": False})
print("body tracks:", [t["name"] for t in tracks])

hw = cfg["spacecraft"]["hardware"]
modes, schedule = [], []
panel_i = next((i for i, h in enumerate(hw) if h["type"] == "SolarPanel"), None)
sensor_i = next((i for i, h in enumerate(hw) if h["type"] in ("StarTracker", "OpNavCamera")), None)
if panel_i is not None:
    modes.append({"name": "SunPointing", "rules": [{"hardware_index": panel_i, "target": {"type": "Sun"}}], "pointing_locked": False})
if sensor_i is not None:
    modes.append({"name": "TargetPointing", "rules": [{"hardware_index": sensor_i, "target": {"type": "Body", "name": target}}], "pointing_locked": False})
approach = duration_s * 0.9
if panel_i is not None and sensor_i is not None:
    schedule = [{"start_s": 0, "end_s": approach, "mode": "SunPointing"}, {"start_s": approach, "end_s": duration_s, "mode": "TargetPointing"}]
elif panel_i is not None:
    schedule = [{"start_s": 0, "end_s": duration_s, "mode": "SunPointing"}]
elif sensor_i is not None:
    schedule = [{"start_s": 0, "end_s": duration_s, "mode": "TargetPointing"}]

window = None
if window_arg:
    start_d, end_d = float(window_arg[0]), float(window_arg[1])
    start_s = duration_s + start_d * 86400.0 if start_d < 0 else start_d * 86400.0
    end_s = duration_s + end_d * 86400.0 if end_d <= 0 else end_d * 86400.0
    window = {"start_s": max(0.0, start_s), "end_s": min(duration_s, end_s)}
    if initial_dr:
        window["initial_dr_m"] = [float(x) for x in initial_dr]
    if initial_dv:
        window["initial_dv_mps"] = [float(x) for x in initial_dv]
    print(f"window: {window['start_s']/86400:.2f} .. {window['end_s']/86400:.2f} d of {duration_s/86400:.2f} d, "
          f"initial dr {window.get('initial_dr_m')} dv {window.get('initial_dv_mps')}")
flown_s = (window["end_s"] - window["start_s"]) if window else duration_s
n_ticks = max(1, int(-(-flown_s // tick_s)))
stride = max(1, -(-n_ticks // 20000))

cfg["simulation"]["monte_carlo_runs"] = 0
cfg["cruise_seed"] = {
    "r0_m": reference[0]["r_m"],
    "v0_m": reference[0]["v_mps"],
    "reference": reference,
    "duration_s": duration_s,
    "tick_s": tick_s,
    "modes": modes,
    "mode_schedule": schedule,
    "safe_mode": "SunPointing" if panel_i is not None else None,
    "body_tracks": tracks,
    "report_stride": stride,
    "tcm_dr_threshold_m": tcm_threshold_m,
    "planned_burns": burns,
    "window": window,
}

print(f"duration {duration_s:.0f} s ({duration_s/86400:.2f} d), {len(reference)} ref pts, t0 {t0:.1f}, stride {stride}, tcm threshold {tcm_threshold_m:.0f} m")
print(f"burn epochs (rebased): {[round(b['epoch_s'], 1) for b in burns]}")
req = urllib.request.Request("http://localhost:8000/api/simulate", data=json.dumps(cfg).encode(),
                             headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=120) as r:
    print("HTTP", r.status, r.read().decode()[:400])
