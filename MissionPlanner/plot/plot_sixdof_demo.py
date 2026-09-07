"""
Phase 13c verification plot — the single 6DOF propagator, diagnostic panels.

Reads (written by `cargo run -p mission_planner --bin sixdof_demo --release`,
run from the repo root):
  out/sixdof_demo/sixdof_demo.csv

This is what actually proves 13c works, not just that its unit tests pass in
isolation: with ZERO commanded control torque, an asymmetric bus-plus-panels
spacecraft (Izz != Ixx=Iyy, flat-plate SRP model) in a circular LEO-like
orbit should show
  (a) a translational orbit whose altitude never drifts (tick-based
      composition of the SOI-patched propagator must reproduce a clean
      circle -- the tick-splitting regression test checks this numerically,
      this plot is the visual confirmation),
  (b) real, physical gravity-gradient AND SRP disturbance torque components
      (both nonzero, gravity-gradient dominating at this altitude -- SRP
      torque several orders of magnitude smaller, as expected for a compact
      LEO bus far from any relative-torque authority comparison; this is
      what actually separates the two disturbance sources this module
      assembles, not just a combined number),
  (c) a real libration in the angular-rate components driven by those
      torques, and
  (d) |q| staying pinned to 1.0 (integration/renormalization health).

See also plot_sixdof_demo_3d.py for an interactive 3D view of the orbit with
the spacecraft's body axes and the inertial reference frame.

Usage (from the repo root):
  python MissionPlanner/plot/plot_sixdof_demo.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent.parent
CSV = ROOT / "out" / "sixdof_demo" / "sixdof_demo.csv"

df = pd.read_csv(CSV)
t_min = df["t_s"] / 60.0

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Translational orbit (body-centered)", "Altitude vs time (tick-composed SOI propagator)",
        "Angular rate components — torque-driven, no control", "Quaternion norm error (renormalization health)",
        "Gravity-gradient torque (body frame)", "Flat-plate SRP torque (body frame)",
    ],
)

# ── (a) Translational orbit stays circular (body-centered, km) ──────────────
fig.add_trace(go.Scatter(x=df["r_x"] / 1e3, y=df["r_y"] / 1e3, mode="lines",
                          line=dict(color="#00d4ff", width=2), name="orbit", showlegend=False),
              row=1, col=1)
fig.update_xaxes(title_text="x [km]", row=1, col=1, scaleanchor="y1", scaleratio=1)
fig.update_yaxes(title_text="y [km]", row=1, col=1)

# ── (b) Altitude vs time -- should be dead flat ─────────────────────────────
fig.add_trace(go.Scatter(x=t_min, y=df["alt_km"], mode="lines",
                          line=dict(color="#00d4ff", width=1.5), name="altitude", showlegend=False),
              row=1, col=2)
alt_spread = df["alt_km"].max() - df["alt_km"].min()
fig.add_annotation(text=f"peak-to-peak drift: {alt_spread:.4f} km", xref="x2 domain", yref="y2 domain",
                    x=0.02, y=0.05, showarrow=False, font=dict(color="#ffaa00", size=11))
fig.update_xaxes(title_text="time [min]", row=1, col=2)
fig.update_yaxes(title_text="altitude [km]", row=1, col=2)

# ── (c) omega components vs time -- gravity-gradient + SRP libration ────────
for col, color, name in [("omega_x", "#ff6688", "omega_x"), ("omega_y", "#88ff66", "omega_y"), ("omega_z", "#6688ff", "omega_z")]:
    fig.add_trace(go.Scatter(x=t_min, y=df[col] * 1e3, mode="lines", line=dict(color=color, width=1.3), name=name),
                  row=2, col=1)
fig.update_xaxes(title_text="time [min]", row=2, col=1)
fig.update_yaxes(title_text="omega [mrad/s]", row=2, col=1)

# ── (d) quaternion norm vs time -- integration health check ─────────────────
fig.add_trace(go.Scatter(x=t_min, y=df["q_norm"] - 1.0, mode="lines",
                          line=dict(color="#77dd77", width=1.5), name="|q|-1", showlegend=False),
              row=2, col=2)
fig.update_xaxes(title_text="time [min]", row=2, col=2)
fig.update_yaxes(title_text="|q| - 1.0", tickformat=".2e", row=2, col=2)

# ── (e) gravity-gradient torque components ───────────────────────────────────
for col, color, name in [("tau_gg_x", "#ff6688", "tau_x"), ("tau_gg_y", "#88ff66", "tau_y"), ("tau_gg_z", "#6688ff", "tau_z")]:
    fig.add_trace(go.Scatter(x=t_min, y=df[col], mode="lines", line=dict(color=color, width=1.3), name=f"gg_{name}"),
                  row=3, col=1)
fig.update_xaxes(title_text="time [min]", row=3, col=1)
fig.update_yaxes(title_text="torque [N*m]", tickformat=".2e", row=3, col=1)

# ── (f) SRP torque components ────────────────────────────────────────────────
for col, color, name in [("tau_srp_x", "#ff6688", "tau_x"), ("tau_srp_y", "#88ff66", "tau_y"), ("tau_srp_z", "#6688ff", "tau_z")]:
    fig.add_trace(go.Scatter(x=t_min, y=df[col], mode="lines", line=dict(color=color, width=1.3), name=f"srp_{name}"),
                  row=3, col=2)
fig.update_xaxes(title_text="time [min]", row=3, col=2)
fig.update_yaxes(title_text="torque [N*m]", tickformat=".2e", row=3, col=2)

gg_peak = df[["tau_gg_x", "tau_gg_y", "tau_gg_z"]].abs().to_numpy().max()
srp_peak = df[["tau_srp_x", "tau_srp_y", "tau_srp_z"]].abs().to_numpy().max()
print(f"Peak |tau_gg| component: {gg_peak:.3e} N*m")
print(f"Peak |tau_srp| component: {srp_peak:.3e} N*m")
print(f"Ratio gg/srp: {gg_peak / srp_peak:.1f}x")

fig.update_layout(
    title="Phase 13c — single 6DOF propagator verification (zero control torque)",
    template="plotly_dark",
    paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
    font=dict(color="#dddddd"),
    height=1100, width=1300,
    legend=dict(bgcolor="rgba(0,0,0,0.5)"),
)
fig.update_xaxes(gridcolor="#333333")
fig.update_yaxes(gridcolor="#333333")

out_path = ROOT / "out" / "sixdof_demo" / "sixdof_demo_verification.html"
fig.write_html(str(out_path))
print(f"Saved {out_path}")
fig.show()
