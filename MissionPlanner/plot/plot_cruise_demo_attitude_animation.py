"""
Phase 5.1 verification — spacecraft attitude ANIMATED along the real flown
position, not fixed at the origin (see plot_cruise_commander_attitude_
animation.py for the origin-fixed body-axis-only variant this borrows its
quaternion-to-DCM/Play-Pause-slider conventions from). Answers a different
question than plot_cruise_demo.py's plots: not "does the flown path track
the reference" or "where does this sit in the solar system," but "is the
attitude controller actually holding the commanded SunPointing orientation
while the spacecraft moves along its real trajectory."

Reads (written by `cargo run -p mission_planner --bin cruise_demo --release`,
run from `MissionPlanner/`):
  out/cruise_demo/cruise_demo.csv   (x_m,y_m,z_m, qw,qx,qy,qz true attitude,
                                      qcw,qcx,qcy,qcz commanded attitude)

Body-frame convention, per `sim_engine::reference_guidance::
desired_quaternion_cruise`'s SunPointing branch: body +z is the axis driven
toward the Sun; +x/+y are drawn too for orientation context. Solid = true
achieved attitude; dashed/dim = commanded attitude (the gap between the two
is the pointing error being driven to zero by the quaternion-PD loop).

Usage (from the repo root):
  python MissionPlanner/plot/plot_cruise_demo_attitude_animation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MISSION_PLANNER_DIR = Path(__file__).parent.parent
OUT_DIR = MISSION_PLANNER_DIR / "out" / "cruise_demo"
TRAJ_CSV = OUT_DIR / "cruise_demo.csv"

AU = 1.495978707e11

df = pd.read_csv(TRAJ_CSV)

N_FRAMES = 250
frame_idx = np.unique(np.linspace(0, len(df) - 1, N_FRAMES).astype(int))

x_au = (df["x_m"] / AU).to_numpy()
y_au = (df["y_m"] / AU).to_numpy()
z_au = (df["z_m"] / AU).to_numpy()

# Body-axis arrow length: a fixed fraction of the flown path's own extent,
# so the triad stays visible at real interplanetary (AU) scale without
# dominating the plot.
extent = max(x_au.max() - x_au.min(), y_au.max() - y_au.min(), z_au.max() - z_au.min(), 1e-6)
AXIS_LEN = 0.035 * extent
TARGET_LEN = AXIS_LEN * 1.6


def quat_to_rotmat(q):
    """Hamilton [w,x,y,z], body->inertial. Standard quaternion-to-DCM."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


BODY_X = np.array([1.0, 0.0, 0.0])
BODY_Y = np.array([0.0, 1.0, 0.0])
BODY_Z = np.array([0.0, 0.0, 1.0])  # SunPointing target axis

path_trace = go.Scatter3d(
    x=x_au, y=y_au, z=z_au, mode="lines",
    line=dict(color="#666666", width=2), name="Flown path", showlegend=False,
)
sun_marker = go.Scatter3d(
    x=[0], y=[0], z=[0], mode="markers",
    marker=dict(size=10, color="#FFD700"), name="Sun", showlegend=False,
)


def axis_trace(origin, vec, color, name):
    return go.Scatter3d(
        x=[origin[0], origin[0] + vec[0]], y=[origin[1], origin[1] + vec[1]], z=[origin[2], origin[2] + vec[2]],
        mode="lines", line=dict(color=color, width=8), name=name, showlegend=False,
    )


def ghost_trace(origin, vec, color, name):
    return go.Scatter3d(
        x=[origin[0], origin[0] + vec[0] * 0.85], y=[origin[1], origin[1] + vec[1] * 0.85], z=[origin[2], origin[2] + vec[2] * 0.85],
        mode="lines", line=dict(color=color, width=5, dash="dash"), name=name, showlegend=False, opacity=0.55,
    )


def target_trace(origin, vec, color, name):
    return go.Scatter3d(
        x=[origin[0], origin[0] + vec[0]], y=[origin[1], origin[1] + vec[1]], z=[origin[2], origin[2] + vec[2]],
        mode="lines", line=dict(color=color, width=3, dash="dot"), name=name, showlegend=False,
    )


frames = []
t_list, err_list = [], []
for i in frame_idx:
    row = df.iloc[i]
    origin = np.array([x_au[i], y_au[i], z_au[i]])

    q = np.array([row["qw"], row["qx"], row["qy"], row["qz"]])
    q = q / np.linalg.norm(q)
    R = quat_to_rotmat(q)

    q_cmd = np.array([row["qcw"], row["qcx"], row["qcy"], row["qcz"]])
    q_cmd = q_cmd / np.linalg.norm(q_cmd)
    R_cmd = quat_to_rotmat(q_cmd)

    sun_dir = -origin / np.linalg.norm(origin) * TARGET_LEN

    sc_marker = go.Scatter3d(
        x=[origin[0]], y=[origin[1]], z=[origin[2]], mode="markers",
        marker=dict(size=5, color="#00d4ff"), name="Spacecraft", showlegend=False,
    )

    # Only the traces that actually change per frame -- the static path and
    # Sun-marker traces (indices 0, 1 in the initial `data` below) are NOT
    # repeated here, via `traces=` targeting -- redrawing all ~4000 path
    # points x 250 frames bloated the first cut of this file to ~46 MB for
    # no visual benefit (the path never changes). This keeps it a few MB.
    body_traces = [
        sc_marker,
        axis_trace(origin, R @ BODY_X * AXIS_LEN, "#ffcc44", "body +x (true)"),
        axis_trace(origin, R @ BODY_Y * AXIS_LEN, "#66ffcc", "body +y (true)"),
        axis_trace(origin, R @ BODY_Z * AXIS_LEN, "#ff4488", "body +z (Sun-pointing, true)"),
        ghost_trace(origin, R_cmd @ BODY_X * AXIS_LEN, "#ffcc44", "body +x (commanded)"),
        ghost_trace(origin, R_cmd @ BODY_Z * AXIS_LEN, "#ff4488", "body +z (commanded)"),
        target_trace(origin, sun_dir, "#ffee55", "Sun direction"),
    ]
    frames.append(go.Frame(data=body_traces, name=str(i), traces=list(range(2, 9))))
    t_list.append(row["t_s"])
    err_list.append(row["pointing_error_deg"])

initial_frame_traces = frames[0].data
frames[0] = go.Frame(data=initial_frame_traces, name=frames[0].name, traces=list(range(2, 9)))
fig = go.Figure(data=[path_trace, sun_marker, *initial_frame_traces], frames=frames)

fig.update_layout(
    title="Spacecraft position + attitude animation (full transfer) — solid axes = true, dashed = commanded",
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
    width=1100, height=900,
    updatemenus=[dict(
        type="buttons", showactive=False,
        buttons=[
            dict(label="Play", method="animate",
                 args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ],
    )],
    sliders=[dict(
        steps=[
            dict(method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                 label=f"t={t_list[k]/86400:.1f}d err={err_list[k]:.2f}deg")
            for k, i in enumerate(frame_idx)
        ],
        currentvalue=dict(prefix="frame: "),
    )],
)

fig.add_annotation(
    text=(
        "solid yellow=body+x TRUE | solid teal=body+y TRUE | solid pink=body+z TRUE (Sun-pointing axis)<br>"
        "dashed/dim yellow+pink=COMMANDED +x/+z | dotted yellow=Sun direction<br>"
        "Pink solid axis should stay aligned with the dotted Sun-direction ray throughout the run."
    ),
    xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False, font=dict(size=11, color="#aaaaaa"),
)

out_path = OUT_DIR / "cruise_demo_attitude_animation.html"
fig.write_html(str(out_path))
print(f"Saved {out_path} ({len(frame_idx)} frames)")
