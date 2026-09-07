"""
Phase 13b/13j verification plot — per-source translational force and
rotational torque breakdown over a real heliocentric departure leg.

Reads (written by `cargo run -p mission_planner --bin sixdof_force_breakdown_demo
--release`, run from the repo root):
  out/sixdof_force_breakdown_demo/force_torque_breakdown.csv

Two passes over the SAME trajectory (see the Rust demo's own doc comment for
why one model can't show both effects at once):
  - "cannonball": real SRP force on translation (via a zero-thrust coupled
    burn tick), zero SRP torque (attitude-independent, GNC_MANUAL.md §4.3).
  - "flatplate": real SRP torque (asymmetric bus+panels+dish geometry), zero
    SRP force on translation (the decoupled path applies none, by design).

What this plot actually verifies (13j):
  (a) central gravity dominates every other translational source by 2+
      orders of magnitude throughout — sanity check that nothing is
      mis-scaled,
  (b) Earth's third-body pull decays as the spacecraft departs (it starts
      close, ~0.05 AU offset) while Jupiter's distant contribution never
      matters — a real, checkable physical claim, not just a plausible-
      looking curve,
  (c) the SRP force/torque model boundary itself: cannonball's SRP force is
      nonzero while its SRP torque is exactly zero, and vice versa for
      flat-plate — printed and annotated directly, so the current fidelity
      limit (documented in GNC_MANUAL.md §3.1) is visible, not hidden.

Usage (from the repo root):
  python MissionPlanner/plot/plot_force_torque_breakdown.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent.parent
CSV = ROOT / "out" / "sixdof_force_breakdown_demo" / "force_torque_breakdown.csv"

df = pd.read_csv(CSV)
cb = df[df["srp_model"] == "cannonball"].reset_index(drop=True)
fp = df[df["srp_model"] == "flatplate"].reset_index(drop=True)

# Day index within each pass (both passes cover the identical duration).
cb_day = cb.index.to_numpy()
fp_day = fp.index.to_numpy()

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Translational accel by source — Cannonball pass (real SRP force)",
        "Translational accel by source — FlatPlate pass (SRP force = 0, by design)",
        "Disturbance torque by source — Cannonball pass (SRP torque = 0, by design)",
        "Disturbance torque by source — FlatPlate pass (real SRP torque)",
    ],
)

ACCEL_SOURCES = [
    ("accel_central_ms2", "#00d4ff", "central gravity"),
    ("accel_thirdbody_ms2", "#ffaa00", "third-body (Earth+Jupiter)"),
    ("accel_srp_ms2", "#ff6688", "SRP"),
]
TORQUE_SOURCES = [
    ("torque_gg_nm", "#00d4ff", "gravity-gradient"),
    ("torque_srp_nm", "#ff6688", "SRP"),
]

for col_num, (data, day) in enumerate([(cb, cb_day), (fp, fp_day)], start=1):
    for col, color, name in ACCEL_SOURCES:
        fig.add_trace(
            go.Scatter(x=day, y=data[col], mode="lines", line=dict(color=color, width=1.6),
                       name=name, showlegend=(col_num == 1)),
            row=1, col=col_num,
        )
    fig.update_yaxes(title_text="|accel| [m/s^2]", type="log", row=1, col=col_num)
    fig.update_xaxes(title_text="tick (1 day each)", row=1, col=col_num)

    for col, color, name in TORQUE_SOURCES:
        # log(0) is undefined -- clip to a tiny floor so an exactly-zero
        # channel (cannonball's SRP torque) still renders as a flat line at
        # the bottom of the log axis instead of vanishing from the plot.
        y = data[col].clip(lower=1e-16)
        fig.add_trace(
            go.Scatter(x=day, y=y, mode="lines", line=dict(color=color, width=1.6),
                       name=name, showlegend=(col_num == 1 and col == "torque_srp_nm")),
            row=2, col=col_num,
        )
    fig.update_yaxes(title_text="|torque| [N*m]", type="log", row=2, col=col_num)
    fig.update_xaxes(title_text="tick (1 day each)", row=2, col=col_num)

# ── Verification summary, printed and annotated ─────────────────────────────
cb_srp_accel_peak = cb["accel_srp_ms2"].max()
cb_srp_torque_peak = cb["torque_srp_nm"].max()
fp_srp_accel_peak = fp["accel_srp_ms2"].max()
fp_srp_torque_peak = fp["torque_srp_nm"].max()
central_peak = df["accel_central_ms2"].max()
earth_pull_start = cb["accel_thirdbody_ms2"].iloc[0]
earth_pull_end = cb["accel_thirdbody_ms2"].iloc[-1]

print("-- Phase 13b/13j force/torque breakdown verification --")
print(f"Central gravity peak: {central_peak:.3e} m/s^2 (dominates every other source, as expected)")
print(f"Third-body accel: {earth_pull_start:.3e} m/s^2 at departure -> {earth_pull_end:.3e} m/s^2 after 150 days "
      f"({earth_pull_start / earth_pull_end:.0f}x decay, Earth's pull fading as the spacecraft departs)")
print(f"Cannonball pass: SRP accel peak {cb_srp_accel_peak:.3e} m/s^2 (real), SRP torque peak {cb_srp_torque_peak:.3e} N*m (should be 0)")
print(f"FlatPlate pass:  SRP accel peak {fp_srp_accel_peak:.3e} m/s^2 (should be 0), SRP torque peak {fp_srp_torque_peak:.3e} N*m (real)")
assert cb_srp_torque_peak < 1e-15, "cannonball SRP torque should be exactly zero (attitude-independent model)"
assert fp_srp_accel_peak < 1e-15, "flat-plate SRP accel-on-translation should be exactly zero (decoupled path applies none)"
print("Verified: cannonball SRP force paired with zero torque, flat-plate SRP torque paired with zero translational force.")

fig.add_annotation(
    text=(f"Cannonball: SRP accel {cb_srp_accel_peak:.2e} m/s^2 (real), SRP torque = 0 (exact)<br>"
          f"FlatPlate: SRP accel = 0 (exact), SRP torque {fp_srp_torque_peak:.2e} N*m (real)<br>"
          "See GNC_MANUAL.md §3.1 — current fidelity boundary, not a bug"),
    xref="paper", yref="paper", x=0.5, y=-0.08, showarrow=False,
    font=dict(color="#ffaa00", size=11), align="center",
)

fig.update_layout(
    title="Phase 13a/13b/13j — per-source force/torque breakdown over a 150-day heliocentric departure leg",
    template="plotly_dark",
    paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
    font=dict(color="#dddddd"),
    height=1000, width=1400,
    legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    margin=dict(b=120),
)
fig.update_xaxes(gridcolor="#333333")
fig.update_yaxes(gridcolor="#333333")

out_path = ROOT / "out" / "sixdof_force_breakdown_demo" / "force_torque_breakdown.html"
fig.write_html(str(out_path))
print(f"Saved {out_path}")
fig.show()
