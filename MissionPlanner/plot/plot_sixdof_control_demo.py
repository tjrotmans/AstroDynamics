"""
Phase 13d/13e verification plot — closed-loop attitude control.

Reads (written by `cargo run -p mission_planner --bin sixdof_control_demo
--release`, run from the repo root):
  out/sixdof_control_demo/sixdof_control_demo.csv

Confirms the cascaded controller + allocation loop (quaternion PD -> wheel
allocation, WheelsPrimary mode) actually converges a real 90 degree initial
pointing error toward zero, that wheel momentum grows to absorb it (no RCS
needed while wheels stay under the desaturation threshold), and that
individual wheel speeds stay well inside their saturation limit throughout.

Note: for this demo's initial condition (a pure body +x -> +y realignment,
i.e. a rotation purely about body +z), the commanded torque is purely along
z the whole time, and the wheel pyramid's z-axis allocation coefficient is
identical for all four wheels -- so all four wheel-speed traces are
numerically EQUAL and will visually overlap into what looks like one line.
That is the correct allocation result, not a rendering bug (verified
directly against the raw CSV).

Usage (from the repo root):
  python MissionPlanner/plot/plot_sixdof_control_demo.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent.parent
CSV = ROOT / "out" / "sixdof_control_demo" / "sixdof_control_demo.csv"

df = pd.read_csv(CSV)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Pointing error convergence", "Body angular rate",
        "Wheel cluster momentum (absorbs the spacecraft's angular momentum change)",
        "Individual wheel speeds vs saturation limit",
    ],
)

fig.add_trace(go.Scatter(x=df["t_s"], y=df["pointing_err_deg"], mode="lines",
                          line=dict(color="#00d4ff", width=2), name="pointing error", showlegend=False),
              row=1, col=1)
fig.update_xaxes(title_text="time [s]", row=1, col=1)
fig.update_yaxes(title_text="pointing error [deg]", row=1, col=1)

fig.add_trace(go.Scatter(x=df["t_s"], y=df["omega_norm_mrads"], mode="lines",
                          line=dict(color="#ff8844", width=1.5), name="|omega|", showlegend=False),
              row=1, col=2)
fig.update_xaxes(title_text="time [s]", row=1, col=2)
fig.update_yaxes(title_text="|omega| [mrad/s]", row=1, col=2)

max_h = 0.012 * 628.3  # wheel_inertia * max_speed, matches the demo's cluster
fig.add_trace(go.Scatter(x=df["t_s"], y=df["wheel_momentum_nms"], mode="lines",
                          line=dict(color="#77dd77", width=2), name="total |H_w|"),
              row=2, col=1)
fig.add_trace(go.Scatter(x=df["t_s"], y=[max_h] * len(df), mode="lines",
                          line=dict(color="#ff4444", width=1.2, dash="dash"),
                          name=f"per-wheel max ({max_h:.2f} N*m*s)"),
              row=2, col=1)
fig.update_xaxes(title_text="time [s]", row=2, col=1)
fig.update_yaxes(title_text="momentum [N*m*s]", row=2, col=1)

for col, color in zip(["w1", "w2", "w3", "w4"], ["#ff6688", "#88ff66", "#6688ff", "#ffcc44"]):
    fig.add_trace(go.Scatter(x=df["t_s"], y=df[col], mode="lines", line=dict(color=color, width=1.3), name=col),
                  row=2, col=2)
fig.add_trace(go.Scatter(x=df["t_s"], y=[628.3] * len(df), mode="lines",
                          line=dict(color="#ff4444", width=1.0, dash="dash"), name="wheel max speed", showlegend=False),
              row=2, col=2)
fig.add_trace(go.Scatter(x=df["t_s"], y=[-628.3] * len(df), mode="lines",
                          line=dict(color="#ff4444", width=1.0, dash="dash"), showlegend=False),
              row=2, col=2)
fig.update_xaxes(title_text="time [s]", row=2, col=2)
fig.update_yaxes(title_text="wheel speed [rad/s]", row=2, col=2)

max_rcs = df["rcs_duty_cycle"].max()
total_prop = df["rcs_propellant_kg_cum"].iloc[-1]
print(f"Max RCS duty cycle observed: {max_rcs:.3f} (should be 0.0 -- wheels never saturate in this run)")
print(f"Total RCS propellant used: {total_prop:.6e} kg")
print("Note: w1..w4 are numerically identical for this pure-z-torque case (see module docstring) -- they overlap.")

fig.update_layout(
    title="Phase 13d/13e — closed-loop pointing control (quaternion PD + wheel allocation)",
    template="plotly_dark",
    paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
    font=dict(color="#dddddd"),
    height=800, width=1200,
    legend=dict(bgcolor="rgba(0,0,0,0.5)"),
)
fig.update_xaxes(gridcolor="#333333")
fig.update_yaxes(gridcolor="#333333")

out_path = ROOT / "out" / "sixdof_control_demo" / "sixdof_control_demo_verification.html"
fig.write_html(str(out_path))
print(f"Saved {out_path}")
fig.show()
