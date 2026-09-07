"""
plot_srp_anim.py — Plotly HTML animation: panel-SRP spacecraft body + orbit + EKF calibration.

Layout (3 rows × 2 cols):
  Rows 1-2, Col 1 : 3-D spacecraft body — gold bus box + dark solar panels,
                    rotating with the true attitude quaternion each frame.
                    Yellow arrow = Sun direction; red arrow = truth SRP force.
  Row 3,    Col 1 : Orbit top-down (inertial x-y).  Full trajectory coloured
                    by mission phase; bright dot = current position; white trail.
  Row 1,    Col 2 : |truth SRP| force magnitude over time.
  Row 2,    Col 2 : filter C_R — drops 1.40 → ~0.68 during the radio-science arc
                    because the solar panels spend most of the orbit edge-on to the
                    Sun (nadir-pointing), so the effective cannonball area is far
                    less than the initial 1.4 × 4 m² guess.
  Row 3,    Col 2 : |SRP residual| vs |Gauss-Markov stochastic-accel estimate|.

Reads:  out/mission/srp.csv
        out/mission/nav.csv
Writes: out/mission/srp_animation.html

Run from GNC/AutonomousNavigation/:
    py plot/plot_srp_anim.py
"""

import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / "mission"
OUT  = DATA / "srp_animation.html"


def load(name):
    p = DATA / name
    if not p.exists():
        sys.exit(f"[ERROR] Missing {p}\nRun: cargo run --bin proximity_mission --release")
    return np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8")


def dec(arr):
    return [s.decode() if isinstance(s, bytes) else str(s) for s in arr]


# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading data...", end=" ", flush=True)
srp = load("srp.csv")
nav = load("nav.csv")
print("done.")

N     = len(srp)
t_h   = srp["time_s"].astype(float) / 3600.0
phase = np.array(dec(srp["phase"]))

qw = srp["qw"].astype(float); qx = srp["qx"].astype(float)
qy = srp["qy"].astype(float); qz = srp["qz"].astype(float)
sun_arr = np.column_stack([srp["sunx"], srp["suny"], srp["sunz"]]).astype(float)

at       = np.column_stack([srp["atx"], srp["aty"], srp["atz"]]).astype(float)
srp_nm   = np.linalg.norm(at, axis=1) * 1e9
srp_hat  = at / (np.linalg.norm(at, axis=1, keepdims=True) + 1e-30)

resid    = np.column_stack([srp["rx"], srp["ry"], srp["rz"]]).astype(float)
est      = np.column_stack([srp["ex"], srp["ey"], srp["ez"]]).astype(float)
resid_nm = np.linalg.norm(resid, axis=1) * 1e9
est_nm   = np.linalg.norm(est,   axis=1) * 1e9
cr       = srp["cr"].astype(float)

# Nav / orbit
nav_t  = nav["time_s"].astype(float)
nav_x  = nav["tx_m"].astype(float) / 1e3
nav_y  = nav["ty_m"].astype(float) / 1e3
nav_z  = nav["tz_m"].astype(float) / 1e3
nav_ph = np.array(dec(nav["phase"]))

nidx   = np.clip(np.searchsorted(nav_t, srp["time_s"].astype(float)), 0, len(nav_t) - 1)
curr_x = nav_x[nidx]
curr_y = nav_y[nidx]
curr_z = nav_z[nidx]
curr_r = np.sqrt(curr_x**2 + curr_y**2 + curr_z**2)

bg     = slice(None, None, 8)          # downsample nav for background scatter
bg_x   = nav_x[bg]; bg_y = nav_y[bg]; bg_z = nav_z[bg]; bg_ph = nav_ph[bg]

rs_mask = phase == "RadioScience"
rs_t0   = float(t_h[rs_mask].min()) if rs_mask.any() else None
rs_t1   = float(t_h[rs_mask].max()) if rs_mask.any() else None

# ── Spacecraft geometry (matches config.rs / srp.rs) ──────────────────────────
BHX, BHY, BHZ = 1.0, 1.0, 0.315          # bus half-dims [m]
PSPAN, PCHORD  = 2.5, 0.8                  # panel span, chord [m]
PHX = PCHORD / 2; PY0 = BHY; PY1 = PY0 + PSPAN

# 8 corners of the bus (body frame)
BOX_V = np.array([
    [-BHX,-BHY,-BHZ], [ BHX,-BHY,-BHZ], [ BHX, BHY,-BHZ], [-BHX, BHY,-BHZ],
    [-BHX,-BHY, BHZ], [ BHX,-BHY, BHZ], [ BHX, BHY, BHZ], [-BHX, BHY, BHZ],
], dtype=float)
# 12 triangles (2 per rectangular face)
BOX_I = [0,0, 4,4, 0,0, 2,2, 0,0, 1,1]
BOX_J = [1,2, 5,6, 1,5, 3,7, 3,7, 2,6]
BOX_K = [2,3, 6,7, 5,4, 7,6, 7,4, 6,5]
# Face normals in body frame (one per triangle)
BOX_N = np.array([
    [0,0,-1],[0,0,-1],  # -z bottom
    [0,0, 1],[0,0, 1],  # +z top
    [0,-1,0],[0,-1,0],  # -y front
    [0, 1,0],[0, 1,0],  # +y back
    [-1,0,0],[-1,0,0],  # -x left
    [ 1,0,0],[ 1,0,0],  # +x right
], dtype=float)

# Solar panels in x-y plane (body frame, normal = +z)
P1_V = np.array([[-PHX, PY0,0],[ PHX, PY0,0],[ PHX, PY1,0],[-PHX, PY1,0]], dtype=float)
P2_V = np.array([[-PHX,-PY0,0],[ PHX,-PY0,0],[ PHX,-PY1,0],[-PHX,-PY1,0]], dtype=float)
PAN_I = [0,0]; PAN_J = [1,2]; PAN_K = [2,3]
PAN_N = np.array([[0,0,1],[0,0,1]], dtype=float)

BENNU_R_KM = 0.245

PHASE_COLORS = {
    "Capture":      "#f06292",
    "Survey":       "#4fc3f7",
    "RadioScience": "#80cbc4",
    "CloseOrbit":   "#ab47bc",
    "Flyover":      "#ff7043",
    "ScienceHold":  "#66bb6a",
}

# ── Precompute all rotated vertices + face intensities ────────────────────────
print(f"Precomputing {N} frames...", end=" ", flush=True)


def build_R(w, x, y, z):
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


box_v   = np.zeros((N, 8, 3))
p1_v    = np.zeros((N, 4, 3))
p2_v    = np.zeros((N, 4, 3))
box_int = np.zeros((N, 12))
p1_int  = np.zeros((N, 2))
p2_int  = np.zeros((N, 2))
sun_tip = np.zeros((N, 3))
srp_tip = np.zeros((N, 3))

for i in range(N):
    R   = build_R(qw[i], qx[i], qy[i], qz[i])
    sun = sun_arr[i]

    box_v[i] = (R @ BOX_V.T).T
    p1_v[i]  = (R @ P1_V.T).T
    p2_v[i]  = (R @ P2_V.T).T

    box_n_r    = (R @ BOX_N.T).T
    box_int[i] = np.clip(0.15 + 0.85 * np.maximum(0.0, box_n_r @ sun), 0.15, 1.0)

    pan_n_r  = R @ PAN_N[0]
    pan_b    = float(np.clip(0.12 + 0.88 * max(0.0, float(np.dot(pan_n_r, sun))), 0.12, 1.0))
    p1_int[i] = p2_int[i] = [pan_b, pan_b]

    sun_tip[i] = sun * 3.2
    srp_tip[i] = srp_hat[i] * 2.6

print("done.")

pos_km    = np.column_stack([curr_x, curr_y, curr_z])
bennu_tip = -(pos_km / (np.linalg.norm(pos_km, axis=1, keepdims=True) + 1e-30)) * 2.6

# ── Helpers to build per-frame trace dicts ─────────────────────────────────────

def _mesh(v, i_idx, j_idx, k_idx, intensity, colorscale, name):
    return go.Mesh3d(
        x=v[:,0].round(4).tolist(),
        y=v[:,1].round(4).tolist(),
        z=v[:,2].round(4).tolist(),
        i=i_idx, j=j_idx, k=k_idx,
        intensity=intensity.round(4).tolist(),
        intensitymode="cell",
        colorscale=colorscale,
        cmin=0.0, cmax=1.0,
        showscale=False, hoverinfo="skip",
        name=name, showlegend=False,
    )


def _arrow(tip, color, label):
    t = tip.round(3)
    return go.Scatter3d(
        x=[0.0, float(t[0])], y=[0.0, float(t[1])], z=[0.0, float(t[2])],
        mode="lines+text",
        text=["", label],
        textposition=["bottom center", "top center"],
        textfont=dict(color=color, size=10),
        line=dict(color=color, width=2.5),
        hoverinfo="skip", showlegend=False,
    )


def _bennu(i):
    t = bennu_tip[i].round(3)
    return go.Scatter3d(
        x=[0.0, float(t[0])], y=[0.0, float(t[1])], z=[0.0, float(t[2])],
        mode="lines+text",
        text=["", "Bennu"],
        textposition=["bottom center", "top center"],
        textfont=dict(color="#C8A84B", size=10),
        line=dict(color="#C8A84B", width=3.0),
        hoverinfo="skip", showlegend=False,
    )


TRAIL = 40

def _trail(i):
    s = max(0, i - TRAIL)
    return go.Scatter3d(
        x=curr_x[s:i+1].round(3).tolist(),
        y=curr_y[s:i+1].round(3).tolist(),
        z=curr_z[s:i+1].round(3).tolist(),
        mode="lines",
        line=dict(color="rgba(255,255,255,0.5)", width=2),
        hoverinfo="skip", showlegend=False,
    )


def _pos(i):
    col = PHASE_COLORS.get(phase[i], "white")
    return go.Scatter3d(
        x=[round(float(curr_x[i]), 3)],
        y=[round(float(curr_y[i]), 3)],
        z=[round(float(curr_z[i]), 3)],
        mode="markers+text",
        marker=dict(color=col, size=5, line=dict(color="white", width=0.7)),
        text=[f"  {curr_r[i]:.2f} km"],
        textposition="middle right",
        textfont=dict(color=col, size=9),
        hoverinfo="skip", showlegend=False,
    )


def _cursor(i, ylo, yhi):
    return go.Scatter(
        x=[round(float(t_h[i]), 2)] * 2,
        y=[ylo, yhi],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.55)", width=1.0, dash="dot"),
        hoverinfo="skip", showlegend=False,
    )


BUS_CS   = [[0, "#1c1403"], [1, "#b8902f"]]    # dark to gold
PANEL_CS = [[0, "#04090e"], [1, "#1b3a6b"]]    # dark to dark blue

# ── Figure ───────────────────────────────────────────────────────────────────
PAPER_BG = "#06060e"
BG_3D    = "#08080f"
BG_2D    = "#04040a"

fig = make_subplots(
    rows=3, cols=2,
    specs=[
        [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
        [None,                             {"type": "xy"}],
        [{"type": "scene"},               {"type": "xy"}],
    ],
    subplot_titles=[
        "",
        "SRP Force Magnitude [nm/s²]",
        "Filter C_R  (radio-science calibration arc shaded)",
        "Orbit — 3-D ecliptic frame  (drag to rotate)",
        "SRP Residual vs Gauss-Markov Estimate [nm/s²]",
    ],
    row_heights=[0.35, 0.32, 0.33],
    column_widths=[0.42, 0.58],
    horizontal_spacing=0.06,
    vertical_spacing=0.09,
)

# ── Static traces ─────────────────────────────────────────────────────────────

fig.add_trace(go.Scatter(x=t_h.tolist(), y=srp_nm.tolist(),
    mode="lines", line=dict(color="#ff8a65", width=1.0),
    name="|truth SRP|"), row=1, col=2)

fig.add_trace(go.Scatter(x=t_h.tolist(), y=cr.tolist(),
    mode="lines", line=dict(color="#ffd54f", width=1.3),
    name="filter C_R"), row=2, col=2)

fig.add_trace(go.Scatter(x=t_h.tolist(), y=resid_nm.tolist(),
    mode="lines", line=dict(color="#4fc3f7", width=1.0),
    name="|residual|"), row=3, col=2)

fig.add_trace(go.Scatter(x=t_h.tolist(), y=est_nm.tolist(),
    mode="lines", line=dict(color="#80cbc4", width=1.0),
    name="|Gauss-Markov|"), row=3, col=2)

bg_colors = [PHASE_COLORS.get(p, "#666666") for p in bg_ph]
fig.add_trace(go.Scatter3d(
    x=bg_x.tolist(), y=bg_y.tolist(), z=bg_z.tolist(), mode="markers",
    marker=dict(color=bg_colors, size=1, opacity=0.40),
    hoverinfo="skip", showlegend=False,
), row=3, col=1)

fig.add_trace(go.Scatter3d(
    x=[0.0], y=[0.0], z=[0.0], mode="markers",
    marker=dict(color="#C8A84B", size=8, symbol="circle",
                line=dict(color="#8d6e40", width=1)),
    hoverinfo="skip", showlegend=False,
), row=3, col=1)

N_STATIC = len(fig.data)

# ── Initial animated traces (frame 0) ─────────────────────────────────────────
fig.add_trace(_mesh(box_v[0], BOX_I, BOX_J, BOX_K, box_int[0], BUS_CS, "Bus"),
              row=1, col=1)
fig.add_trace(_mesh(p1_v[0], PAN_I, PAN_J, PAN_K, p1_int[0], PANEL_CS, "+y panel"),
              row=1, col=1)
fig.add_trace(_mesh(p2_v[0], PAN_I, PAN_J, PAN_K, p2_int[0], PANEL_CS, "-y panel"),
              row=1, col=1)
fig.add_trace(_arrow(sun_tip[0], "#ffd54f", "☀ Sun"), row=1, col=1)
fig.add_trace(_arrow(srp_tip[0], "#ff5252", "SRP"),   row=1, col=1)
fig.add_trace(_bennu(0), row=1, col=1)
fig.add_trace(_trail(0), row=3, col=1)
fig.add_trace(_pos(0),   row=3, col=1)

ylo_srp, yhi_srp = float(srp_nm.min()), float(srp_nm.max())
ylo_cr,  yhi_cr  = float(cr.min()) * 0.95, float(cr.max()) * 1.02
ylo_res, yhi_res = 0.0, float(max(resid_nm.max(), est_nm.max())) * 1.05

fig.add_trace(_cursor(0, ylo_srp, yhi_srp), row=1, col=2)
fig.add_trace(_cursor(0, ylo_cr,  yhi_cr),  row=2, col=2)
fig.add_trace(_cursor(0, ylo_res, yhi_res), row=3, col=2)

ANIM_TRACES = list(range(N_STATIC, N_STATIC + 11))

# ── Build frames ─────────────────────────────────────────────────────────────
print(f"Building {N} animation frames", end="", flush=True)

frames = []
for fi in range(N):
    if fi % 150 == 0:
        print(".", end="", flush=True)

    frames.append(go.Frame(
        data=[
            _mesh(box_v[fi], BOX_I, BOX_J, BOX_K, box_int[fi], BUS_CS, "Bus"),
            _mesh(p1_v[fi], PAN_I, PAN_J, PAN_K, p1_int[fi], PANEL_CS, "+y panel"),
            _mesh(p2_v[fi], PAN_I, PAN_J, PAN_K, p2_int[fi], PANEL_CS, "-y panel"),
            _arrow(sun_tip[fi], "#ffd54f", "☀ Sun"),
            _arrow(srp_tip[fi], "#ff5252", "SRP"),
            _bennu(fi),
            _trail(fi),
            _pos(fi),
            _cursor(fi, ylo_srp, yhi_srp),
            _cursor(fi, ylo_cr,  yhi_cr),
            _cursor(fi, ylo_res, yhi_res),
        ],
        traces=ANIM_TRACES,
        name=str(fi),
        layout=go.Layout(title_text=(
            f"<b style='color:{PHASE_COLORS.get(phase[fi], '#fff')}'>{phase[fi]}</b>"
            f"  ·  t = {t_h[fi]:.1f} h"
            f"  ·  |SRP| = {srp_nm[fi]:.1f} nm/s²"
            f"  ·  C_R = {cr[fi]:.3f}"
            f"  ·  r = {curr_r[fi]:.2f} km"
        )),
    ))

fig.frames = frames
print(" done.")

# ── Slider + buttons ──────────────────────────────────────────────────────────
steps = [
    dict(
        args=[[str(fi)], dict(frame=dict(duration=0, redraw=True),
                              mode="immediate", transition=dict(duration=0))],
        label=f"{t_h[fi]:.0f}h" if fi % 120 == 0 else "",
        method="animate",
    )
    for fi in range(N)
]

play_pause = {
    "type": "buttons", "showactive": False,
    "x": 0.5, "xanchor": "center", "y": -0.06, "yanchor": "top",
    "buttons": [
        {"label": "▶ Play", "method": "animate",
         "args": [None, {"frame": {"duration": 60, "redraw": True},
                         "fromcurrent": True, "transition": {"duration": 0}}]},
        {"label": "⏸ Pause", "method": "animate",
         "args": [[None], {"frame": {"duration": 0, "redraw": False},
                            "mode": "immediate", "transition": {"duration": 0}}]},
    ],
    "font": {"color": "white"},
    "bgcolor": "rgba(40,40,60,0.9)",
    "bordercolor": "rgba(100,100,150,0.5)",
}

slider = {
    "active": 0, "x": 0.05, "len": 0.90,
    "y": -0.02, "yanchor": "top",
    "currentvalue": {
        "prefix": "t = ", "visible": True, "xanchor": "center",
        "font": {"color": "white", "size": 11},
    },
    "transition": {"duration": 0},
    "bgcolor": "rgba(40,40,60,0.8)",
    "bordercolor": "rgba(100,100,150,0.4)",
    "tickcolor": "rgba(200,200,255,0.5)",
    "font": {"color": "white", "size": 9},
    "steps": steps,
}

# ── Layout ────────────────────────────────────────────────────────────────────
_ax2d = dict(color="#888", gridcolor="#1a1a2e", showgrid=True,
             zeroline=False, linecolor="#333")

# RadioScience shading on each time-series panel via layout shapes.
# Axis refs (confirmed): row1col2→x/y, row2col2→x2/y2, row3col2→x4/y4
shapes = []
if rs_t0 is not None:
    for xr, yr in [("x", "y"), ("x2", "y2"), ("x3", "y3")]:
        shapes.append(dict(
            type="rect", xref=xr, yref=f"{yr} domain",
            x0=rs_t0, x1=rs_t1, y0=0, y1=1,
            fillcolor="rgba(128,203,196,0.10)", line_width=0, layer="below",
        ))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor=PAPER_BG,
    font=dict(color="white", size=10),
    title=dict(
        text=(
            "Panel SRP — spacecraft body + orbit + C_R calibration + Gauss-Markov"
            "<br><sup>Nadir-pointing causes solar panels to be edge-on to Sun → "
            "effective C_R calibrates down from 1.40 to ~0.68 in the radio-science arc</sup>"
        ),
        x=0.01, xanchor="left", font=dict(size=11),
    ),
    height=840,
    margin=dict(l=20, r=20, t=90, b=110),
    updatemenus=[play_pause],
    sliders=[slider],
    shapes=shapes,
    legend=dict(x=0.58, y=0.98, bgcolor="rgba(10,10,20,0.7)",
                bordercolor="rgba(80,80,120,0.4)", font=dict(size=10)),
    # 3-D scene
    scene=dict(
        bgcolor=BG_3D,
        xaxis=dict(title="x [m]", **_ax2d, range=[-4.5, 4.5]),
        yaxis=dict(title="y [m]", **_ax2d, range=[-4.5, 4.5]),
        zaxis=dict(title="z [m]", **_ax2d, range=[-4.5, 4.5]),
        camera=dict(eye=dict(x=0.85, y=-1.55, z=0.85)),
        aspectmode="cube",
    ),
    # 2-D subplots — confirmed axis refs: row1col2→xaxis, row2col2→xaxis2,
    #                                      row3col1→xaxis3, row3col2→xaxis4
    xaxis= {**_ax2d, "title": "time [h]"},  yaxis= {**_ax2d},
    xaxis2={**_ax2d, "title": "time [h]"},  yaxis2={**_ax2d},
    xaxis3={**_ax2d, "title": "time [h]"},  yaxis3={**_ax2d},
    scene2=dict(
        bgcolor=BG_3D,
        xaxis=dict(title="x [km]", color="#888", gridcolor="#1a1a2e", showbackground=True, backgroundcolor=BG_3D),
        yaxis=dict(title="y [km]", color="#888", gridcolor="#1a1a2e", showbackground=True, backgroundcolor=BG_3D),
        zaxis=dict(title="z [km]", color="#888", gridcolor="#1a1a2e", showbackground=True, backgroundcolor=BG_3D),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    ),
    plot_bgcolor=BG_2D,
)

# Freeze y-ranges on time-series panels so the cursor doesn't rescale them
fig.update_yaxes(range=[ylo_srp, yhi_srp], row=1, col=2)
fig.update_yaxes(range=[ylo_cr,  yhi_cr],  row=2, col=2)
fig.update_yaxes(range=[ylo_res, yhi_res], row=3, col=2)

# ── Write ─────────────────────────────────────────────────────────────────────
print("Writing HTML...", end=" ", flush=True)
fig.write_html(str(OUT), auto_play=False)
sz = OUT.stat().st_size / 1e6
print(f"done.\n\nSaved {OUT}  ({sz:.1f} MB)")
