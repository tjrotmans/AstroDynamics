"""
Attitude-commander verification plot — the priority-ordered multi-rule
attitude commander (mission_planner::cruise::GncCommander), flown against a
real ANISE Earth->Mars leg with a real periodic comm-pass schedule
(Cruise: SolarPanel normal -> Sun; Comm: CommAntenna boresight -> a real
ANISE-queried Earth position track).

Reads (written by `cargo run -p mission_planner --bin cruise_commander_demo
--release`, run from `MissionPlanner/`):
  out/cruise_commander_demo/cruise_commander_demo.csv
  out/cruise_commander_demo/mode_transitions.csv

Two real bugs this demo surfaced and fixed (not hidden):
  1. The first run (no wheel-speed clamp at all) showed wheel momentum
     running away to >400 N*m*s -- a pre-existing bug in every wheel-speed
     integration site in this codebase (none clamped speed to the wheel's
     rated max). Fixed by clamping integrated speed to +/-max_speed.
  2. That clamp alone was still WRONG: `sim_engine::control::allocate`'s
     WheelsPrimary branch computed the reaction torque applied to the
     BODY from the commanded (pre-saturation) motor torque, with no idea
     a wheel had hit its clamp -- a real violation of Newton's third law
     (the body was torqued as if a wheel kept absorbing momentum it
     physically could not once pinned at max_speed). This produced
     erratic, non-convergent pointing error once wheels first saturated,
     NOT explainable by slow gains alone. Fixed by zeroing the commanded
     torque component for any wheel already at its limit in that
     direction, before it feeds into both the body's reaction torque and
     the wheel's own speed integration -- see control.rs's own doc
     comment on the fix for the full derivation.
This plot reflects the state AFTER both fixes.

Six panels, matching "match the plot to what's being verified":
  1. Pointing error vs. time, with vertical lines at each detected mode
     transition (from mode_transitions.csv) -- shows real convergence
     behavior (or lack of full settling) around each schedule switch.
  2. Active mode as a colored timeline strip -- confirms the schedule is
     actually being followed tick-by-tick.
  3. Wheel cluster momentum vs. time, with the physical ceiling
     (4 wheels * wheel_inertia * max_speed) as a reference line -- confirms
     the post-fix clamp keeps AGGREGATE momentum physically bounded.
  4. Per-wheel speed vs. time, with +/-max_speed reference lines -- the
     aggregate in panel 3 can stay well under the naive 4-wheel ceiling
     even while ONE wheel is individually pinned at its own limit (the
     4-wheel pyramid's momenta partially cancel in the vector sum), so
     this is the panel that actually shows whether/when saturation
     happened, not panel 3 alone.
  5. Disturbance torque breakdown (gravity-gradient, SRP) vs. time.
  6. Control torque magnitude: commanded (what the quaternion-PD law wanted
     from the wheels alone) vs. delivered (what was actually applied to the
     body: wheels post-saturation-zeroing, plus RCS desat torque whenever
     that's firing). The two diverge for two DIFFERENT reasons -- a wheel
     pinned at its limit (the narrow bug-#2 case, rare) or RCS desaturation
     legitimately adding its own torque (common, intended) -- see the
     script's own inline comment for the breakdown; don't read every
     divergence as saturation.

Usage (from the repo root):
  python MissionPlanner/plot/plot_cruise_commander_demo.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MISSION_PLANNER_DIR = Path(__file__).parent.parent
OUT_DIR = MISSION_PLANNER_DIR / "out" / "cruise_commander_demo"
TRAJ_CSV = OUT_DIR / "cruise_commander_demo.csv"
TRANSITIONS_CSV = OUT_DIR / "mode_transitions.csv"

df = pd.read_csv(TRAJ_CSV)
transitions = pd.read_csv(TRANSITIONS_CSV)

# Same default four-wheel-pyramid spec used throughout this codebase's
# demos (ReactionWheelCluster::four_wheel_pyramid(0.012, 628.3, 0.12, 0.8)) --
# physical momentum ceiling per wheel = inertia * max_speed.
WHEEL_INERTIA_KGM2 = 0.012
WHEEL_MAX_SPEED_RADPS = 628.3
MAX_CLUSTER_MOMENTUM_NMS = 4 * WHEEL_INERTIA_KGM2 * WHEEL_MAX_SPEED_RADPS

MODE_COLORS = {"Cruise": "#00d4ff", "Comm": "#ffaa00", "": "#555555"}

fig = make_subplots(
    rows=6, cols=1,
    subplot_titles=[
        "Pointing error (vertical lines = detected mode transitions)",
        "Active mode",
        f"Wheel cluster momentum, aggregate (physical ceiling: {MAX_CLUSTER_MOMENTUM_NMS:.1f} N*m*s)",
        f"Per-wheel speed (individual limit: +/-{WHEEL_MAX_SPEED_RADPS:.1f} rad/s)",
        "Disturbance torque breakdown",
        "Control torque: commanded (wheels alone) vs. delivered (wheels post-saturation + RCS desat)",
    ],
    shared_xaxes=True,
    vertical_spacing=0.04,
)

fig.add_trace(go.Scatter(x=df["t_s"], y=df["pointing_error_deg"], mode="lines",
                          line=dict(color="#77dd77", width=1.5), showlegend=False),
              row=1, col=1)
for _, t in transitions.iterrows():
    fig.add_vline(x=t["t_s"], line=dict(color="#ff4444", width=1, dash="dot"), row=1, col=1)
fig.update_yaxes(title_text="error [deg]", row=1, col=1)

for mode_name, color in MODE_COLORS.items():
    if mode_name == "":
        continue
    mask = df["active_mode"] == mode_name
    if mask.any():
        fig.add_trace(go.Scatter(
            x=df.loc[mask, "t_s"], y=[mode_name] * mask.sum(), mode="markers",
            marker=dict(color=color, size=4, symbol="square"), name=mode_name,
        ), row=2, col=1)
fig.update_yaxes(title_text="mode", row=2, col=1)

fig.add_trace(go.Scatter(x=df["t_s"], y=df["wheel_momentum_nms"], mode="lines",
                          line=dict(color="#00d4ff", width=1.5), name="|H_wheel|"),
              row=3, col=1)
fig.add_trace(go.Scatter(x=df["t_s"], y=[MAX_CLUSTER_MOMENTUM_NMS] * len(df), mode="lines",
                          line=dict(color="#ff4444", width=1.2, dash="dash"), name="physical ceiling"),
              row=3, col=1)
fig.update_yaxes(title_text="momentum [N*m*s]", row=3, col=1)

wheel_colors = ["#ff6688", "#88ff66", "#6688ff", "#ffcc44"]
n_saturation_ticks = 0
for i, color in zip(range(1, 5), wheel_colors):
    col = f"w{i}"
    fig.add_trace(go.Scatter(x=df["t_s"], y=df[col], mode="lines",
                              line=dict(color=color, width=1), name=col),
                  row=4, col=1)
    n_saturation_ticks += int((df[col].abs() >= WHEEL_MAX_SPEED_RADPS - 1e-6).sum())
fig.add_trace(go.Scatter(x=df["t_s"], y=[WHEEL_MAX_SPEED_RADPS] * len(df), mode="lines",
                          line=dict(color="#ff4444", width=1.0, dash="dash"), showlegend=False),
              row=4, col=1)
fig.add_trace(go.Scatter(x=df["t_s"], y=[-WHEEL_MAX_SPEED_RADPS] * len(df), mode="lines",
                          line=dict(color="#ff4444", width=1.0, dash="dash"), showlegend=False),
              row=4, col=1)
fig.update_yaxes(title_text="wheel speed [rad/s]", row=4, col=1)

fig.add_trace(go.Scatter(x=df["t_s"], y=df["torque_gravity_gradient_nm"], mode="lines",
                          line=dict(color="#ff8844", width=1.3), name="gravity-gradient"),
              row=5, col=1)
fig.add_trace(go.Scatter(x=df["t_s"], y=df["torque_srp_nm"], mode="lines",
                          line=dict(color="#dd77ff", width=1.3), name="SRP"),
              row=5, col=1)
fig.update_yaxes(title_text="torque [N*m]", row=5, col=1)

tau_cmd_mag = (df["tau_cmd_x"] ** 2 + df["tau_cmd_y"] ** 2 + df["tau_cmd_z"] ** 2) ** 0.5
tau_del_mag = (df["tau_del_x"] ** 2 + df["tau_del_y"] ** 2 + df["tau_del_z"] ** 2) ** 0.5
fig.add_trace(go.Scatter(x=df["t_s"], y=tau_cmd_mag, mode="lines",
                          line=dict(color="#aaaaaa", width=1.2), name="|tau_cmd| (wanted)"),
              row=6, col=1)
fig.add_trace(go.Scatter(x=df["t_s"], y=tau_del_mag, mode="lines",
                          line=dict(color="#ff6688", width=1.2), name="|tau_delivered| (applied)"),
              row=6, col=1)
fig.update_yaxes(title_text="|torque| [N*m]", row=6, col=1)
fig.update_xaxes(title_text="time [s]", row=6, col=1)

# NOTE: commanded and delivered diverge for TWO different reasons, not one:
#  (1) a wheel is pinned at its speed limit and its commanded torque
#      component is zeroed -- the real Newton's-third-law bug this pair of
#      fields exists to make visible (control.rs's WheelsPrimary saturation
#      zeroing). This is the narrow, "something is wrong" case: only 276
#      ticks in this run had ANY wheel actually at its limit.
#  (2) RCS desaturation is actively firing (any wheel > 80% of max_speed)
#      and adds its own torque contribution on top of the wheels -- this is
#      INTENDED behavior, not a shortfall, and is a much more common
#      condition than outright saturation. The gap count below includes
#      both, so it is an upper bound on (1), not a measurement of it alone.
torque_gap = (tau_cmd_mag - tau_del_mag).abs()
n_gap_ticks = int((torque_gap > 1e-9).sum())

n_never_settled = transitions["settling_time_s"].isna().sum()
print(f"Mode transitions: {len(transitions)} total, {n_never_settled} never fully settled within their window")
print(f"Max wheel momentum observed (aggregate): {df['wheel_momentum_nms'].max():.3f} N*m*s (physical ceiling: {MAX_CLUSTER_MOMENTUM_NMS:.1f} N*m*s)")
for i in range(1, 5):
    col = f"w{i}"
    print(f"  wheel {i}: max |speed| = {df[col].abs().max():.2f} rad/s (limit {WHEEL_MAX_SPEED_RADPS:.1f}), ticks at/above limit: {int((df[col].abs() >= WHEEL_MAX_SPEED_RADPS - 1e-6).sum())}")
print(f"Total ticks with ANY wheel at its individual speed limit: {n_saturation_ticks}")
print(f"Ticks where commanded torque != delivered torque (wheel saturation zeroing OR RCS desat adding torque, not just the former -- see panel 6's own comment): {n_gap_ticks}")
print(f"Final position dispersion: {df['t_s'].iloc[-1]:.0f} s into the leg -- see console output from the Rust binary for the exact value")

fig.update_layout(
    title="Attitude commander with a real periodic comm-pass schedule",
    template="plotly_dark",
    paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
    font=dict(color="#dddddd"),
    height=1600, width=1300,
    legend=dict(bgcolor="rgba(0,0,0,0.5)"),
)
fig.update_xaxes(gridcolor="#333333")
fig.update_yaxes(gridcolor="#333333")

out_path = OUT_DIR / "cruise_commander_demo.html"
fig.write_html(str(out_path))
print(f"Saved {out_path}")
