"""
Phase 5.1 verification plot — the cruise mission-loop composition
(mission_planner::cruise::run_cruise_leg): a real Earth->Mars interplanetary
coast flown under step_tick's translation+attitude propagation with a real
quaternion-PD/wheel attitude-hold controller, while reference_guidance::
dispersion reports drift from the Layer-1 reference trajectory.

Reads (written by `cargo run -p mission_planner --bin cruise_demo --release`,
run from `MissionPlanner/`):
  out/cruise_demo/cruise_demo.csv          (flown trajectory + telemetry)
  out/cruise_demo/reference_trajectory.csv (Layer-1 reference, uniform samples)

Three figures, matching this repo's "match the plot to what's being verified"
convention (different plot types for different questions):
  1. Telemetry panel (2x2): position/velocity dispersion vs. time (is the
     flown state tracking the Layer-1 reference), pointing error vs. time
     (does the attitude controller hold SunPointing over a multi-day coast),
     wheel momentum vs. time (does it stay bounded, no runaway/saturation).
  2. 3D overview (local, meters): flown vs. reference path overlay -- a
     geometric sanity check that the two visually coincide (they should, to
     within a few meters at this scale -- see the telemetry panel for the
     actual number).
  3. 3D solar-system context (AU): Sun + Earth/Mars real heliocentric tracks
     over the FULL transfer window (not just the 3-day flown segment) --
     answers a different question than #2 (is the flown arc a physically
     sane departure from Earth's real orbit, at real interplanetary scale),
     which #2's meter-scale local view can't show at all.

Usage (from the repo root, matching cruise_demo's own "run from
MissionPlanner/" doc comment -- output lands in MissionPlanner/out/, not
the repo-root out/ some sibling demos use):
  python MissionPlanner/plot/plot_cruise_demo.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MISSION_PLANNER_DIR = Path(__file__).parent.parent
OUT_DIR = MISSION_PLANNER_DIR / "out" / "cruise_demo"
FLOWN_CSV = OUT_DIR / "cruise_demo.csv"
REF_CSV = OUT_DIR / "reference_trajectory.csv"

flown = pd.read_csv(FLOWN_CSV)
ref = pd.read_csv(REF_CSV)

# ── Figure 1: telemetry panel ────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Position dispersion vs. Layer-1 reference", "Velocity dispersion vs. Layer-1 reference",
        "Pointing error (SunPointing attitude hold)", "Wheel cluster momentum",
    ],
)

fig.add_trace(go.Scatter(x=flown["t_s"], y=flown["dr_m"], mode="lines",
                          line=dict(color="#00d4ff", width=2), showlegend=False),
              row=1, col=1)
fig.update_xaxes(title_text="time [s]", row=1, col=1)
fig.update_yaxes(title_text="|dr| [m]", row=1, col=1)

fig.add_trace(go.Scatter(x=flown["t_s"], y=flown["dv_mps"], mode="lines",
                          line=dict(color="#ff8844", width=2), showlegend=False),
              row=1, col=2)
fig.update_xaxes(title_text="time [s]", row=1, col=2)
fig.update_yaxes(title_text="|dv| [m/s]", row=1, col=2)

fig.add_trace(go.Scatter(x=flown["t_s"], y=flown["pointing_error_deg"], mode="lines",
                          line=dict(color="#77dd77", width=2), showlegend=False),
              row=2, col=1)
fig.update_xaxes(title_text="time [s]", row=2, col=1)
fig.update_yaxes(title_text="pointing error [deg]", row=2, col=1)

fig.add_trace(go.Scatter(x=flown["t_s"], y=flown["wheel_momentum_nms"], mode="lines",
                          line=dict(color="#ffcc44", width=2), showlegend=False),
              row=2, col=2)
fig.update_xaxes(title_text="time [s]", row=2, col=2)
fig.update_yaxes(title_text="|H_wheel| [N*m*s]", row=2, col=2)

fig.update_layout(
    title="Phase 5.1 — cruise mission-loop composition (translation + attitude + reference tracking)",
    template="plotly_dark",
    paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
    font=dict(color="#dddddd"),
    height=800, width=1200,
)
fig.update_xaxes(gridcolor="#333333")
fig.update_yaxes(gridcolor="#333333")

telemetry_path = OUT_DIR / "cruise_demo_telemetry.html"
fig.write_html(str(telemetry_path))
print(f"Saved {telemetry_path}")

print(f"[from decimated CSV, {len(flown)} of the full tick count -- may understate transient peaks; "
      f"see cruise_demo's own console output for the true full-resolution max]")
print(f"Max position dispersion: {flown['dr_m'].max():.3f} m")
print(f"Max velocity dispersion: {flown['dv_mps'].max():.6f} m/s")
print(f"Max pointing error: {flown['pointing_error_deg'].max():.4f} deg")
print(f"Max wheel momentum: {flown['wheel_momentum_nms'].max():.4f} N*m*s")
print(f"Total RCS propellant used: {flown['rcs_propellant_kg_cum'].iloc[-1]:.6e} kg")

# ── Figure 2: 3D overview -- flown vs. reference path ───────────────────
fig3d = go.Figure()
fig3d.add_trace(go.Scatter3d(
    x=ref["x_m"], y=ref["y_m"], z=ref["z_m"], mode="lines",
    line=dict(color="#00d4ff", width=4), name="Layer-1 reference",
))
fig3d.add_trace(go.Scatter3d(
    x=flown["x_m"], y=flown["y_m"], z=flown["z_m"], mode="lines",
    line=dict(color="#ff4444", width=2, dash="dash"), name="flown (step_tick)",
))
fig3d.update_layout(
    title="Flown trajectory vs. Layer-1 reference (should visually coincide -- see telemetry panel for the real gap)",
    template="plotly_dark",
    paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
    font=dict(color="#dddddd"),
    scene=dict(
        xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
        aspectmode="data",
    ),
    height=800, width=1000,
)
overview_path = OUT_DIR / "cruise_demo_overview_3d.html"
fig3d.write_html(str(overview_path))
print(f"Saved {overview_path}")

# ── Figure 3: solar-system context -- Sun + Earth/Mars real tracks (AU) ──
BODIES_CSV = OUT_DIR / "bodies.csv"
AU = 1.495978707e11

if BODIES_CSV.exists():
    bodies = pd.read_csv(BODIES_CSV)

    figsys = go.Figure()
    figsys.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=10, color="#FFD700"), name="Sun",
    ))
    figsys.add_trace(go.Scatter3d(
        x=bodies["earth_x_m"] / AU, y=bodies["earth_y_m"] / AU, z=bodies["earth_z_m"] / AU,
        mode="lines", line=dict(color="#4FC3F7", width=2, dash="dot"),
        name="Earth orbit (over transfer window)",
    ))
    figsys.add_trace(go.Scatter3d(
        x=bodies["mars_x_m"] / AU, y=bodies["mars_y_m"] / AU, z=bodies["mars_z_m"] / AU,
        mode="lines", line=dict(color="#FF6B35", width=2, dash="dot"),
        name="Mars orbit (over transfer window)",
    ))
    figsys.add_trace(go.Scatter3d(
        x=[bodies["earth_x_m"].iloc[0] / AU], y=[bodies["earth_y_m"].iloc[0] / AU], z=[bodies["earth_z_m"].iloc[0] / AU],
        mode="markers+text", marker=dict(size=6, color="#4FC3F7"),
        text=["Earth @ departure"], textposition="top center", name="Earth @ departure",
    ))
    figsys.add_trace(go.Scatter3d(
        x=[bodies["mars_x_m"].iloc[-1] / AU], y=[bodies["mars_y_m"].iloc[-1] / AU], z=[bodies["mars_z_m"].iloc[-1] / AU],
        mode="markers+text", marker=dict(size=6, color="#FF6B35"),
        text=["Mars @ arrival"], textposition="top center", name="Mars @ arrival",
    ))
    # The flown 3-day segment, at real scale -- a short arc hugging Earth's
    # orbit right at departure, since it's only ~3 days of a ~291-day transfer.
    figsys.add_trace(go.Scatter3d(
        x=flown["x_m"] / AU, y=flown["y_m"] / AU, z=flown["z_m"] / AU,
        mode="lines", line=dict(color="#ff4444", width=5),
        name="Flown segment (this demo)",
    ))

    figsys.update_layout(
        title="Solar-system context: real Earth/Mars tracks over the full transfer, flown segment at real scale",
        template="plotly_dark",
        paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
        font=dict(color="#dddddd"),
        scene=dict(
            xaxis_title="x [AU]", yaxis_title="y [AU]", zaxis_title="z [AU]",
            xaxis=dict(backgroundcolor="#0f0f0f", gridcolor="#333333"),
            yaxis=dict(backgroundcolor="#0f0f0f", gridcolor="#333333"),
            zaxis=dict(backgroundcolor="#0f0f0f", gridcolor="#333333"),
            aspectmode="data",
        ),
        height=800, width=1000,
    )
    system_path = OUT_DIR / "cruise_demo_solar_system.html"
    figsys.write_html(str(system_path))
    print(f"Saved {system_path}")
else:
    print(f"Note: {BODIES_CSV} not found -- run cruise_demo again to regenerate it (rebuild needed for the new bodies.csv output)")
