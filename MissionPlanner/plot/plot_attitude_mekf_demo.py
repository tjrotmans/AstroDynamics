"""
Phase 13h verification plot — attitude MEKF.

Reads (written by `cargo run -p mission_planner --bin attitude_mekf_demo
--release`, run from the repo root):
  out/attitude_mekf_demo/attitude_mekf_demo.csv

Shows the real closed-loop scenario the filter was run against: truth
pointing error converging under PD + wheel control (panel 1), the MEKF's
estimation error (truth vs q_hat) against its own reported 1-sigma bound
(panel 2, log scale), the gyro bias truth vs estimate (panel 3), and the
error/sigma consistency ratio over time (panel 4) -- this last panel is the
direct visual of the known, documented limitation in attitude_ekf.rs's own
module doc comment: the ratio spikes during aggressive slews (first-order
Phi/Q optimism under fast rotation) and settles to a modest ~2-6x during
quiescent tracking. Star-tracker update ticks are marked as vertical lines
on every panel.

Usage (from the repo root):
  python MissionPlanner/plot/plot_attitude_mekf_demo.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent.parent
CSV = ROOT / "out" / "attitude_mekf_demo" / "attitude_mekf_demo.csv"

df = pd.read_csv(CSV)
update_times = df.loc[df["star_tracker_update"] == 1, "t_s"].tolist()

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Truth pointing error (under PD + wheel control)",
        "MEKF estimation error vs. reported 1-sigma bound",
        "Gyro bias: truth vs. estimate",
        "Consistency ratio (error / sigma)",
    ],
)

fig.add_trace(go.Scatter(x=df["t_s"], y=df["truth_pointing_err_deg"], mode="lines", name="truth error", line=dict(color="#e74c3c")), row=1, col=1)

fig.add_trace(go.Scatter(x=df["t_s"], y=df["mekf_attitude_err_deg"], mode="lines", name="MEKF error (truth vs q_hat)", line=dict(color="#3498db")), row=1, col=2)
fig.add_trace(go.Scatter(x=df["t_s"], y=df["mekf_sigma_alpha_deg"], mode="lines", name="MEKF 1-sigma", line=dict(color="#95a5a6", dash="dash")), row=1, col=2)
fig.update_yaxes(type="log", row=1, col=2)

fig.add_trace(go.Scatter(x=df["t_s"], y=df["gyro_bias_truth_x_dps"], mode="lines", name="truth bias (x)", line=dict(color="#e67e22")), row=2, col=1)
fig.add_trace(go.Scatter(x=df["t_s"], y=df["gyro_bias_hat_x_dps"], mode="lines", name="estimated bias (x)", line=dict(color="#2ecc71")), row=2, col=1)

ratio = df["mekf_attitude_err_deg"] / df["mekf_sigma_alpha_deg"].clip(lower=1e-9)
fig.add_trace(go.Scatter(x=df["t_s"], y=ratio, mode="lines", name="error / sigma", line=dict(color="#9b59b6")), row=2, col=2)
fig.add_hline(y=3.0, line_dash="dot", line_color="gray", row=2, col=2, annotation_text="3-sigma")

for t in update_times:
    for r in (1, 2):
        for c in (1, 2):
            fig.add_vline(x=t, line_width=1, line_color="rgba(150,150,150,0.25)", row=r, col=c)

fig.update_layout(
    template="plotly_dark",
    title="Phase 13h — Attitude MEKF verification (real closed-loop slew + real gyro/star-tracker noise)",
    height=800,
    showlegend=True,
)
fig.update_xaxes(title_text="time [s]")

out_path = ROOT / "out" / "attitude_mekf_demo" / "attitude_mekf_demo.html"
fig.write_html(out_path)
print(f"Saved {out_path}")
