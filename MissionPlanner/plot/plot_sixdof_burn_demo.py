"""
Phase 13f verification plot — finite-burn propulsion with thrust
misalignment, fully coupled with attitude control.

Reads (written by `cargo run -p mission_planner --bin sixdof_burn_demo
--release`, run from the repo root):
  out/sixdof_burn_demo/sixdof_burn_demo.csv

Confirms:
  - propellant mass depletes linearly at the Tsiolkovsky rate throughout
    the burn (a deterministic bookkeeping check -- the printed console
    output already confirms it matches to 1e-6 kg; this plot shows the
    linear trend visually),
  - delta-v accumulates as the burn proceeds, tracking close to (but
    slightly under, due to the small residual pointing error reducing the
    effective thrust component) the theoretical maximum F/m integrated
    over the burn,
  - the misalignment torque produces a real, bounded steady-state pointing
    error under pure PD control (no integral term -- a textbook proportional
    control offset against a constant disturbance, not divergence),
  - wheel momentum grows to hold that error against a CONTINUOUS
    disturbance (unlike the transient-only case in
    plot_sixdof_control_demo.py), and RCS duty cycle should stay at/near
    zero unless the wheels approach saturation.

Usage (from the repo root):
  python MissionPlanner/plot/plot_sixdof_burn_demo.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent.parent
CSV = ROOT / "out" / "sixdof_burn_demo" / "sixdof_burn_demo.csv"

df = pd.read_csv(CSV)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Propellant mass depletion (Tsiolkovsky)", "Delta-v accumulation",
        "Pointing error under continuous misalignment torque", "Wheel momentum + RCS duty cycle",
    ],
)

fig.add_trace(go.Scatter(x=df["t_s"], y=df["mass_kg"], mode="lines",
                          line=dict(color="#00d4ff", width=2), name="mass", showlegend=False),
              row=1, col=1)
fig.update_xaxes(title_text="time [s]", row=1, col=1)
fig.update_yaxes(title_text="mass [kg]", row=1, col=1)

fig.add_trace(go.Scatter(x=df["t_s"], y=df["dv_mps"], mode="lines",
                          line=dict(color="#77dd77", width=2), name="delta-v", showlegend=False),
              row=1, col=2)
fig.update_xaxes(title_text="time [s]", row=1, col=2)
fig.update_yaxes(title_text="delta-v [m/s]", row=1, col=2)

fig.add_trace(go.Scatter(x=df["t_s"], y=df["pointing_err_deg"], mode="lines",
                          line=dict(color="#ff8844", width=2), name="pointing error", showlegend=False),
              row=2, col=1)
fig.update_xaxes(title_text="time [s]", row=2, col=1)
fig.update_yaxes(title_text="pointing error [deg]", row=2, col=1)

fig.add_trace(go.Scatter(x=df["t_s"], y=df["wheel_momentum_nms"], mode="lines",
                          line=dict(color="#ffcc44", width=2), name="wheel momentum"),
              row=2, col=2)
fig.add_trace(go.Scatter(x=df["t_s"], y=df["rcs_duty_cycle"] * df["wheel_momentum_nms"].max(), mode="lines",
                          line=dict(color="#ff4444", width=1.5, dash="dot"),
                          name="RCS duty (scaled for visibility)"),
              row=2, col=2)
fig.update_xaxes(title_text="time [s]", row=2, col=2)
fig.update_yaxes(title_text="momentum [N*m*s]", row=2, col=2)

mass_flow_rate = (df["mass_kg"].iloc[0] - df["mass_kg"].iloc[-1]) / (df["t_s"].iloc[-1] - df["t_s"].iloc[0])
print(f"Observed mass flow rate: {mass_flow_rate:.6f} kg/s")
print(f"Final delta-v: {df['dv_mps'].iloc[-1]:.3f} m/s")
print(f"Steady-state pointing error: {df['pointing_err_deg'].iloc[-1]:.3f} deg")
print(f"Max RCS duty cycle: {df['rcs_duty_cycle'].max():.3f}")

fig.update_layout(
    title="Phase 13f — finite burn + misalignment torque + closed-loop attitude control",
    template="plotly_dark",
    paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
    font=dict(color="#dddddd"),
    height=800, width=1200,
    legend=dict(bgcolor="rgba(0,0,0,0.5)"),
)
fig.update_xaxes(gridcolor="#333333")
fig.update_yaxes(gridcolor="#333333")

out_path = ROOT / "out" / "sixdof_burn_demo" / "sixdof_burn_demo_verification.html"
fig.write_html(str(out_path))
print(f"Saved {out_path}")
fig.show()
