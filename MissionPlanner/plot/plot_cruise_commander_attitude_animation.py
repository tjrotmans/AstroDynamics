"""
Attitude-commander verification — a real 3D spacecraft attitude animation,
showing how the body actually rotates over time under the priority-ordered
GncCommander (cruise_commander_demo's real ANISE Earth->Mars leg + comm-pass
schedule), driven by the TRUE attitude quaternion the closed-loop control
system achieved (not yet the commanded/target quaternion -- that field
doesn't exist in the CSV yet, see the module's own known-limitations note).

Reads (written by `cargo run -p mission_planner --bin cruise_commander_demo
--release`, run from `MissionPlanner/`):
  out/cruise_commander_demo/cruise_commander_demo.csv   (qw,qx,qy,qz columns)
  out/cruise_commander_demo/earth_track.csv

Body-frame axes drawn, per this demo's own hardware placement
(cruise_commander_demo.rs): body +x = SolarPanel normal (hardware_index=1,
targeted at the Sun in "Cruise" mode); body +z = CommAntenna boresight
(hardware_index=0, targeted at Earth in "Comm" mode). Both are rotated into
the inertial frame at each frame via the true quaternion (Hamilton, [w,x,y,z]
convention, body->inertial) so the animation can show whether each boresight
is actually pointed at its real target -- Sun direction is -r_hat (from the
spacecraft toward the Sun, since the CSV's x_m/y_m/z_m are the spacecraft's
own heliocentric position); Earth direction is (earth_pos - sc_pos), Earth's
position linearly interpolated from earth_track.csv at each sample time.

Now shows BOTH: solid, saturated-color axes for the TRUE achieved attitude
(qw,qx,qy,qz) and dashed, dim/ghost axes for the COMMANDED/target attitude
(qcw,qcx,qcy,qcz) the quaternion-PD law is chasing -- the gap between solid
and ghost at any frame IS the pointing error being driven by the controller.
This is what makes the 180 deg antipodal-chattering episodes visible as a
real geometric event: the ghost snaps to a new target and the solid axes
oscillate trying (and briefly failing) to catch up, rather than smoothly
tracking.

Usage (from the repo root):
  python MissionPlanner/plot/plot_cruise_commander_attitude_animation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MISSION_PLANNER_DIR = Path(__file__).parent.parent
OUT_DIR = MISSION_PLANNER_DIR / "out" / "cruise_commander_demo"
TRAJ_CSV = OUT_DIR / "cruise_commander_demo.csv"
EARTH_CSV = OUT_DIR / "earth_track.csv"

df = pd.read_csv(TRAJ_CSV)
earth = pd.read_csv(EARTH_CSV)

# Decimate to a manageable number of animation frames -- 25920 raw ticks is
# far too many for a browser-side Plotly animation; keep the cadence coarse
# enough to see the real chattering episodes without a multi-hundred-MB file.
N_FRAMES = 400
frame_idx = np.linspace(0, len(df) - 1, N_FRAMES).astype(int)
frame_idx = np.unique(frame_idx)


def quat_to_rotmat(q):
    """Hamilton [w,x,y,z], body->inertial. Standard quaternion-to-DCM."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


BODY_X = np.array([1.0, 0.0, 0.0])  # SolarPanel normal (Cruise target: Sun)
BODY_Y = np.array([0.0, 1.0, 0.0])
BODY_Z = np.array([0.0, 0.0, 1.0])  # CommAntenna boresight (Comm target: Earth)

earth_t = earth["t_s"].to_numpy()
earth_pos = earth[["x_m", "y_m", "z_m"]].to_numpy()


def earth_pos_at(t_s):
    x = np.interp(t_s, earth_t, earth_pos[:, 0])
    y = np.interp(t_s, earth_t, earth_pos[:, 1])
    z = np.interp(t_s, earth_t, earth_pos[:, 2])
    return np.array([x, y, z])


# Fixed-size body-axis arrows in a local display frame -- absolute AU-scale
# translation doesn't matter for showing rotation, so each frame is drawn
# centered at the origin with unit-length axis arrows and unit-length
# target-direction rays for the Sun/Earth pointing check.
AXIS_LEN = 1.0
TARGET_LEN = 1.6

frames = []
mode_list = []
err_list = []
for i in frame_idx:
    row = df.iloc[i]
    q = np.array([row["qw"], row["qx"], row["qy"], row["qz"]])
    q = q / np.linalg.norm(q)
    R = quat_to_rotmat(q)

    q_cmd = np.array([row["qcw"], row["qcx"], row["qcy"], row["qcz"]])
    q_cmd = q_cmd / np.linalg.norm(q_cmd)
    R_cmd = quat_to_rotmat(q_cmd)

    x_axis = R @ BODY_X
    y_axis = R @ BODY_Y
    z_axis = R @ BODY_Z

    x_axis_cmd = R_cmd @ BODY_X
    z_axis_cmd = R_cmd @ BODY_Z

    sc_pos = np.array([row["x_m"], row["y_m"], row["z_m"]])
    sun_dir = -sc_pos / np.linalg.norm(sc_pos)
    e_pos = earth_pos_at(row["t_s"])
    earth_dir = (e_pos - sc_pos)
    earth_dir_norm = np.linalg.norm(earth_dir)
    # At t=0 the spacecraft departs FROM Earth, so the direction to Earth is
    # genuinely undefined right at that instant -- not a bug, a real
    # coincidence of departure. Fall back to the previous frame's direction
    # (or +x if there is none yet) rather than dividing by ~0.
    if earth_dir_norm < 1.0:
        earth_dir = frames[-1].data[4].x if frames else None
        earth_dir = np.array([1.0, 0.0, 0.0]) if earth_dir is None else np.array(
            [earth_dir[1] / TARGET_LEN, frames[-1].data[4].y[1] / TARGET_LEN, frames[-1].data[4].z[1] / TARGET_LEN]
        )
    else:
        earth_dir = earth_dir / earth_dir_norm

    def axis_trace(vec, color, name):
        return go.Scatter3d(
            x=[0, vec[0] * AXIS_LEN], y=[0, vec[1] * AXIS_LEN], z=[0, vec[2] * AXIS_LEN],
            mode="lines", line=dict(color=color, width=8), name=name, showlegend=False,
        )

    def target_trace(vec, color, name):
        return go.Scatter3d(
            x=[0, vec[0] * TARGET_LEN], y=[0, vec[1] * TARGET_LEN], z=[0, vec[2] * TARGET_LEN],
            mode="lines", line=dict(color=color, width=3, dash="dot"), name=name, showlegend=False,
        )

    def ghost_trace(vec, color, name):
        # Commanded/target attitude's body axis -- dashed, dimmer than the
        # solid TRUE-attitude axes, drawn slightly shorter so it never
        # fully occludes the solid axis when the two nearly coincide
        # (the well-tracked case, which should be the common one).
        return go.Scatter3d(
            x=[0, vec[0] * AXIS_LEN * 0.85], y=[0, vec[1] * AXIS_LEN * 0.85], z=[0, vec[2] * AXIS_LEN * 0.85],
            mode="lines", line=dict(color=color, width=5, dash="dash"), name=name, showlegend=False,
            opacity=0.55,
        )

    body_traces = [
        axis_trace(x_axis, "#ffcc44", "body +x (panel normal, true)"),
        axis_trace(y_axis, "#66ffcc", "body +y (true)"),
        axis_trace(z_axis, "#ff4488", "body +z (comm boresight, true)"),
        target_trace(sun_dir, "#ffee55", "Sun direction"),
        target_trace(earth_dir, "#55aaff", "Earth direction"),
        ghost_trace(x_axis_cmd, "#ffcc44", "body +x commanded (ghost)"),
        ghost_trace(z_axis_cmd, "#ff4488", "body +z commanded (ghost)"),
    ]

    frames.append(go.Frame(data=body_traces, name=str(i)))
    mode_list.append(row["active_mode"] if isinstance(row["active_mode"], str) else "")
    err_list.append(row["pointing_error_deg"])

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
)

lim = 1.8
fig.update_layout(
    title="Spacecraft attitude animation — true body axes vs. Sun/Earth direction",
    template="plotly_dark",
    paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
    font=dict(color="#dddddd"),
    scene=dict(
        xaxis=dict(range=[-lim, lim], title="x", backgroundcolor="#0f0f0f", gridcolor="#333333"),
        yaxis=dict(range=[-lim, lim], title="y", backgroundcolor="#0f0f0f", gridcolor="#333333"),
        zaxis=dict(range=[-lim, lim], title="z", backgroundcolor="#0f0f0f", gridcolor="#333333"),
        aspectmode="cube",
    ),
    width=1000, height=900,
    updatemenus=[dict(
        type="buttons", showactive=False,
        buttons=[
            dict(label="Play", method="animate",
                 args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ],
    )],
    sliders=[dict(
        steps=[
            dict(method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                 label=f"t={df.iloc[i]['t_s']:.0f}s [{mode_list[k]}] err={err_list[k]:.1f}deg")
            for k, i in enumerate(frame_idx)
        ],
        currentvalue=dict(prefix="frame: "),
    )],
)

legend_note = (
    "solid yellow=body+x TRUE (panel normal) | solid teal=body+y TRUE | solid pink=body+z TRUE (comm boresight)<br>"
    "dashed/dim yellow+pink=COMMANDED +x/+z (what the controller is chasing) | "
    "dotted yellow=Sun direction | dotted blue=Earth direction<br>"
    "Cruise mode: panel normal (+x) should track the Sun ray, ghost should coincide with solid. "
    "Comm mode: boresight (+z) should track the Earth ray. "
    "Solid lagging/oscillating around ghost = the controller chasing, not disturbances."
)
fig.add_annotation(
    text=legend_note, xref="paper", yref="paper", x=0.5, y=1.06,
    showarrow=False, font=dict(size=11, color="#aaaaaa"),
)

out_path = OUT_DIR / "cruise_commander_attitude_animation.html"
fig.write_html(str(out_path))
print(f"Saved {out_path} ({len(frame_idx)} frames)")
print("Solid axes = true achieved attitude; dashed/dim axes = commanded/target attitude.")
print("Commanded-vs-delivered TORQUE is not shown in this 3D view -- see")
print("plot_cruise_commander_demo.py's panel 6 for that comparison over time.")
