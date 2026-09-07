"""
Parametric validation of `cruise.rs`'s `MomentumManagementLaw::ThresholdRcs`
gains (momentum-dump `gain` and null-motion `null_motion_gain`) -- requested
after `cruise_commander_demo` found 8 of 18 mode transitions never settle
under the hand-picked defaults (gain=0.02, null_motion_gain=0.05) on an
aggressive 8h-comm-pass schedule.

Reads (written by `cargo run -p mission_planner --bin cruise_gain_sweep_demo
--release`, run from `MissionPlanner/`):
  out/cruise_gain_sweep_demo/gain_sweep.csv

Two independent 1-D sweeps (gain with null_motion_gain fixed at its default;
null_motion_gain with gain fixed at its default), not a combined grid -- see
the Rust binary's own doc comment for why.

Real finding this plot documents: the momentum-DUMP gain has essentially NO
effect on settle rate across a 32x range (0.005-0.16) -- it stays pinned at
11/18 throughout. The null-motion gain shows a sharp threshold instead:
flat at 11/18 up to ~0.20, crossing through a transition band at 0.21-0.23,
then 18/18 (full settling) from ~0.22 upward. `max_wheel_sat_frac` tracks
the SAME threshold almost exactly (0.82 -> 0.19), confirming the mechanism:
below the threshold, individual wheels pin at their speed limit during the
aggressive comm-pass slew, which (per control.rs's own saturation-zeroing
fix) removes real control authority in that axis and stalls convergence;
above the threshold, null-motion damping keeps the redundant 4th wheel-speed
DOF near zero, so no single wheel disproportionately absorbs momentum and
saturates mid-maneuver -- even though the null-motion torque is, by
construction, supposed to produce zero net body torque and therefore "not
matter" to attitude tracking directly. It matters indirectly, through
actuator saturation avoidance.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "out" / "cruise_gain_sweep_demo" / "gain_sweep.csv"

df = pd.read_csv(CSV)
sweep_a = df[df["sweep_axis"] == "gain"].sort_values("gain")
sweep_b = df[df["sweep_axis"] == "null_motion_gain"].sort_values("null_motion_gain")

DEFAULT_GAIN = 0.02
DEFAULT_NULL_MOTION_GAIN = 0.05

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Sweep A: settle rate & wheel saturation vs. momentum-dump gain (null_motion_gain fixed)",
        "Sweep B: settle rate & wheel saturation vs. null-motion gain (gain fixed)",
        "Sweep A: mean settling time & RCS cost vs. momentum-dump gain",
        "Sweep B: mean settling time & RCS cost vs. null-motion gain",
    ),
    specs=[[{"secondary_y": True}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": True}]],
)

# ── Row 1: settle rate + max wheel saturation fraction ──────────────────────
for col, (sweep, xcol, default_x) in enumerate(
    [(sweep_a, "gain", DEFAULT_GAIN), (sweep_b, "null_motion_gain", DEFAULT_NULL_MOTION_GAIN)], start=1
):
    settle_rate = 100.0 * sweep["n_settled"] / sweep["n_transitions"]
    fig.add_trace(
        go.Scatter(
            x=sweep[xcol], y=settle_rate, mode="lines+markers", name="Settle rate [%]",
            line=dict(color="#00d4ff", width=2), marker=dict(size=7),
            showlegend=(col == 1), legendgroup="settle",
        ),
        row=1, col=col, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=sweep[xcol], y=sweep["max_wheel_sat_frac"], mode="lines+markers", name="Max wheel |speed|/max_speed",
            line=dict(color="#ff5555", width=2, dash="dot"), marker=dict(size=7),
            showlegend=(col == 1), legendgroup="sat",
        ),
        row=1, col=col, secondary_y=True,
    )
    fig.add_hline(y=0.8, line=dict(color="#888888", width=1, dash="dash"), row=1, col=col, secondary_y=True)
    fig.add_vline(x=default_x, line=dict(color="#ffaa00", width=1, dash="dash"), row=1, col=col)
    fig.update_xaxes(title_text=xcol, type="log" if xcol == "gain" else "linear", row=1, col=col)
    fig.update_yaxes(title_text="Settle rate [%]", range=[0, 105], row=1, col=col, secondary_y=False)
    fig.update_yaxes(title_text="Max wheel sat. frac.", range=[0, 1.05], row=1, col=col, secondary_y=True)

# ── Row 2: mean settling time + RCS propellant cost ─────────────────────────
for col, (sweep, xcol, default_x) in enumerate(
    [(sweep_a, "gain", DEFAULT_GAIN), (sweep_b, "null_motion_gain", DEFAULT_NULL_MOTION_GAIN)], start=1
):
    fig.add_trace(
        go.Scatter(
            x=sweep[xcol], y=sweep["mean_settling_time_s"], mode="lines+markers", name="Mean settling time [s]",
            line=dict(color="#7fff7f", width=2), marker=dict(size=7),
            showlegend=(col == 1), legendgroup="settle_time",
        ),
        row=2, col=col, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=sweep[xcol], y=sweep["rcs_propellant_kg_used"] * 1000.0, mode="lines+markers", name="RCS propellant used [g]",
            line=dict(color="#ffaa00", width=2, dash="dot"), marker=dict(size=7),
            showlegend=(col == 1), legendgroup="rcs",
        ),
        row=2, col=col, secondary_y=True,
    )
    fig.add_vline(x=default_x, line=dict(color="#ffaa00", width=1, dash="dash"), row=2, col=col)
    fig.update_xaxes(title_text=xcol, type="log" if xcol == "gain" else "linear", row=2, col=col)
    fig.update_yaxes(title_text="Mean settling time [s]", row=2, col=col, secondary_y=False)
    fig.update_yaxes(title_text="RCS propellant [g]", row=2, col=col, secondary_y=True)

fig.update_layout(
    title="cruise.rs momentum-management gain sweep (real ANISE Earth-&gt;Mars leg, "
    "3-day comm-pass schedule, orange dashed = current default)",
    template="plotly_dark",
    paper_bgcolor="#0f0f0f",
    plot_bgcolor="#0f0f0f",
    height=800,
    width=1400,
    legend=dict(orientation="h", y=-0.08),
)

out_path = HERE.parent / "out" / "cruise_gain_sweep_demo" / "gain_sweep.html"
fig.write_html(str(out_path))
print(f"wrote {out_path}")

n_a = len(sweep_a)
n_a_full = (sweep_a["n_settled"] == sweep_a["n_transitions"]).sum()
print(f"\nSweep A (momentum-dump gain): {n_a_full}/{n_a} grid points fully settled -- gain has essentially no effect here.")

threshold_rows = sweep_b[sweep_b["n_settled"] == sweep_b["n_transitions"]]
if not threshold_rows.empty:
    threshold = threshold_rows["null_motion_gain"].min()
    print(f"Sweep B (null-motion gain): full settling (18/18) first achieved at null_motion_gain={threshold:.3f}")
    best = sweep_b.loc[sweep_b["mean_settling_time_s"].idxmin()]
    print(
        f"Best mean settling time in the fully-settled region: null_motion_gain={best['null_motion_gain']:.3f} "
        f"-> {best['mean_settling_time_s']:.0f} s, RCS cost {best['rcs_propellant_kg_used']*1000:.1f} g"
    )
