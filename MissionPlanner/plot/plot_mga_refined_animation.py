"""plot_mga_refined_animation.py -- 3D time-animation of an MGA-1DSM arc,
comparing the analytic Keplerian solution against the multiple-shooting
N-body-corrected solution, with real body positions rotating over time.

Always rendered in the ECLIPTIC frame — ANISE's native output is J2000
EQUATORIAL (ICRF); see mga_frame_utils.py for the exact fixed rotation used to
convert. The backend (propagation, optimization) is untouched — this is a
display-layer-only transform.

Animated traces use FIXED indices across all frames (via go.Frame(traces=...))
so Plotly always replaces the same trace slots -- a variable trace count per
frame silently leaves stale traces on screen (duplicate/ghost body markers,
legs that stop updating).

Usage (from MissionPlanner/ directory):
    python plot/plot_mga_refined_animation.py evj_flyby

Reads (from out/<mission>/):
    mga_best.csv           -- analytic Keplerian arc points (t_days, x/y/z_m, leg_idx)
    mga_refined.csv        -- multiple-shooting corrected arc points
    mga_legs.csv           -- per-leg body_dep/body_arr/tof_days/eta/dv_dsm_ms/...

Output:
    out/<mission>/mga_refined_animation.html
"""

import sys
import argparse
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import mga_frame_utils as mfu

AU = mfu.AU

BG    = "#0a0a14"
GRID  = "#222233"
TEXT  = "#cccccc"
LEG_COLORS = [
    "#00d4ff", "#ff6b35", "#a8ff3e", "#ff3ea8", "#ffd700", "#cf9fff",
]

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="3D animated MGA arc: analytic vs. multiple-shooting")
parser.add_argument("mission", nargs="?", default="evj_flyby")
parser.add_argument("--frames", type=int, default=200)
args = parser.parse_args()

mission = args.mission
out_dir = Path("out") / mission
n_frames = max(30, args.frames)

analytic_path = out_dir / "mga_best.csv"
refined_path  = out_dir / "mga_refined.csv"
reprop_path   = out_dir / "mga_repropagated.csv"
legs_path     = out_dir / "mga_legs.csv"

for p in (analytic_path, legs_path):
    if not p.exists():
        print(f"[ERROR] {p} not found.")
        sys.exit(1)

analytic = mfu.add_ecliptic_columns(pd.read_csv(analytic_path))
# Numeric arc: prefer the multiple-shooting corrected trajectory when present;
# fall back to the plain Dopri5 forward re-propagation (analytic DSMs applied
# under real N-body dynamics, no Newton correction) — always written by the
# optimizer, and the honest numeric arc to show when mga-refine diverged or
# wasn't run.
if refined_path.exists():
    numeric = mfu.add_ecliptic_columns(pd.read_csv(refined_path))
    numeric_label = "Refined"
elif reprop_path.exists():
    numeric = mfu.add_ecliptic_columns(pd.read_csv(reprop_path))
    numeric_label = "Numeric (Dopri5)"
else:
    numeric = None
    numeric_label = ""
refined = numeric
legs     = pd.read_csv(legs_path)

t_min = float(analytic["t_days"].min())
t_max = float(analytic["t_days"].max())
n_legs = int(analytic["leg_idx"].max()) + 1

body_seq = [legs.iloc[0]["body_dep"].strip()]
for _, r in legs.iterrows():
    body_seq.append(r["body_arr"].strip())

ARCS = [("Analytic", analytic, "dash")] + ([(numeric_label, refined, None)] if refined is not None else [])

# ── Infer theta0 for each body in the sequence from real ANISE positions ────

body_theta0 = {}

dep_row = analytic.iloc[0]
dep_body = body_seq[0]
if dep_body in mfu.BODY_SMA_AU:
    body_theta0[dep_body] = mfu.infer_theta0(
        dep_row["x_m"], dep_row["y_m"], dep_row["z_m"],
        mfu.BODY_SMA_AU[dep_body] * AU, dep_row["t_days"],
    )

for i, (_, leg) in enumerate(legs.iterrows()):
    barr = leg["body_arr"].strip()
    leg_arc = analytic[analytic["leg_idx"] == i]
    if leg_arc.empty or barr not in mfu.BODY_SMA_AU:
        continue
    last = leg_arc.iloc[-1]
    body_theta0[barr] = mfu.infer_theta0(
        last["x_m"], last["y_m"], last["z_m"],
        mfu.BODY_SMA_AU[barr] * AU, last["t_days"],
    )

ring_bodies = sorted(set(b for b in body_seq if b in mfu.BODY_SMA_AU))

# ── Static traces (added once; never touched by frames) ─────────────────────

static_traces = []

for body in ring_bodies:
    xs, ys, zs = mfu.ring_xyz(mfu.BODY_SMA_AU[body])
    static_traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=mfu.body_color(body), width=1),
        opacity=0.35, name=f"{body} orbit", hoverinfo="skip",
    ))

static_traces.append(go.Scatter3d(
    x=[0], y=[0], z=[0], mode="markers",
    marker=dict(color="#ffffc0", size=9, symbol="circle"), name="Sun",
))

# Faded full-arc backdrop, so the whole planned route is visible up front.
for label, df, dash in ARCS:
    for k in range(n_legs):
        seg = df[df["leg_idx"] == k]
        if seg.empty:
            continue
        col = LEG_COLORS[k % len(LEG_COLORS)]
        line = dict(color=col, width=1)
        if dash:
            line["dash"] = dash
        static_traces.append(go.Scatter3d(
            x=seg["x_ecl_au"], y=seg["y_ecl_au"], z=seg["z_ecl_au"],
            mode="lines", line=line, opacity=0.22,
            name=f"{label} leg {k} (full)", showlegend=False, hoverinfo="skip",
        ))

# DSM markers -- analytic (dim) and refined (bright), both shown.
for k in range(n_legs):
    row = legs[legs["leg_idx"] == k].iloc[0]
    col = LEG_COLORS[k % len(LEG_COLORS)]
    xe, ye, ze = mfu.icrf_point_to_ecliptic_au(row["x_dsm_m"], row["y_dsm_m"], row["z_dsm_m"])
    static_traces.append(go.Scatter3d(
        x=[xe], y=[ye], z=[ze],
        mode="markers", marker=dict(symbol="x", color=col, size=5, opacity=0.45),
        name=f"Analytic DSM leg {k}",
        hovertemplate="Analytic DSM leg %d<br>ΔV=%.1f m/s<extra></extra>" % (k, row["dv_dsm_ms"]),
    ))
    if refined is not None:
        leg_start = refined[refined["leg_idx"] == k]
        if not leg_start.empty:
            t_dep_days = float(leg_start["t_days"].min())
            dsm_ref = mfu.find_dsm_point(refined, k, t_dep_days, row["eta"], row["tof_days"])
            if dsm_ref is not None:
                t_dsm, xr, yr, zr = dsm_ref
                xe, ye, ze = mfu.icrf_point_to_ecliptic_au(xr, yr, zr)
                static_traces.append(go.Scatter3d(
                    x=[xe], y=[ye], z=[ze],
                    mode="markers",
                    marker=dict(symbol="x", color=col, size=9, line=dict(color="white", width=1.5)),
                    name=f"Refined DSM leg {k}",
                    hovertemplate=f"Refined DSM leg {k}<br>T+{t_dsm:.1f} days<extra></extra>",
                ))

# Flyby periapsis markers at intermediate flyby bodies, on the refined arc
# only -- the multiple-shooting constraint drives each intermediate leg's
# endpoint to the flyby body's periapsis position, so the refined leg-k
# endpoint IS the periapsis, by construction. Its distance to the analytic
# leg-k endpoint (= the body's true ANISE center at encounter) is the real
# periapsis radius achieved.
if refined is not None:
    for k in range(n_legs - 1):
        body_row = legs[legs["leg_idx"] == k].iloc[0]
        body_name = body_row["body_arr"]
        seg_a_arc = analytic[analytic["leg_idx"] == k]
        seg_r_arc = refined[refined["leg_idx"] == k]
        if seg_a_arc.empty or seg_r_arc.empty:
            continue
        body_center = seg_a_arc.iloc[-1]
        peri = seg_r_arc.iloc[-1]
        dist_m = ((peri["x_m"] - body_center["x_m"])**2
                  + (peri["y_m"] - body_center["y_m"])**2
                  + (peri["z_m"] - body_center["z_m"])**2) ** 0.5
        col = LEG_COLORS[k % len(LEG_COLORS)]
        static_traces.append(go.Scatter3d(
            x=[peri["x_ecl_au"]], y=[peri["y_ecl_au"]], z=[peri["z_ecl_au"]],
            mode="markers+text",
            marker=dict(color="#ffffff", size=7, symbol="diamond", line=dict(color=col, width=2)),
            text=[f"Flyby periapsis<br>{dist_m/1000:.0f} km"],
            textposition="bottom center", textfont=dict(color="#ffffff", size=9),
            name=f"Refined periapsis @ {body_name}",
            hovertemplate=(f"Refined flyby periapsis @ {body_name}<br>"
                            f"T+{float(peri['t_days']):.1f} days<br>"
                            f"distance from body center = {dist_m/1000:.0f} km<extra></extra>"),
        ))

# SOI passages on the refined arc: the N-body propagator records which body
# was gravitationally central at each sample (central_body column, empty =
# Sun). These open markers show exactly where the gravity assist acts on the
# trajectory, vs. the X markers where we command DSM burns.
if refined is not None and "central_body" in refined.columns:
    soi_rows = refined[refined["central_body"].notna() & (refined["central_body"] != "")]
    for body_name, grp in soi_rows.groupby("central_body"):
        static_traces.append(go.Scatter3d(
            x=grp["x_ecl_au"], y=grp["y_ecl_au"], z=grp["z_ecl_au"],
            mode="markers",
            marker=dict(color="#ffffff", size=4, symbol="circle-open",
                        line=dict(color="#ffffff", width=2)),
            name=f"Inside {body_name} SOI",
            hovertemplate=f"Inside {body_name} SOI<extra></extra>",
        ))

n_static = len(static_traces)

# ── Dynamic (animated) traces: FIXED count and order every frame ────────────
#
# Order: for each arc (Analytic, [Refined]): n_legs trail traces, then 1
# current-position marker. Then 1 marker per ring body. This order/count is
# identical for every frame -- required so go.Frame(traces=...) replaces the
# correct slots and no stale trace is ever left on screen.

def build_dynamic(t_days: float):
    traces = []
    for label, df, dash in ARCS:
        so_far = df[df["t_days"] <= t_days]
        for k in range(n_legs):
            seg = so_far[so_far["leg_idx"] == k]
            col = LEG_COLORS[k % len(LEG_COLORS)]
            line = dict(color=col, width=3)
            if dash:
                line["dash"] = dash
            traces.append(go.Scatter3d(
                x=seg["x_ecl_au"] if not seg.empty else [],
                y=seg["y_ecl_au"] if not seg.empty else [],
                z=seg["z_ecl_au"] if not seg.empty else [],
                mode="lines", line=line,
                name=f"{label} leg {k}",
                showlegend=bool(t_days == t_min),
            ))

        if df.empty:
            traces.append(go.Scatter3d(x=[], y=[], z=[], mode="markers", showlegend=False))
            continue
        idx = (df["t_days"] - t_days).abs().idxmin()
        cur = df.loc[idx]
        leg_k = int(cur["leg_idx"])
        symbol = "diamond" if label != "Analytic" else "circle-open"
        colr = "#ffffff" if label != "Analytic" else "#888888"
        traces.append(go.Scatter3d(
            x=[cur["x_ecl_au"]], y=[cur["y_ecl_au"]], z=[cur["z_ecl_au"]],
            mode="markers",
            marker=dict(color=colr, size=6 if label != "Analytic" else 4, symbol=symbol,
                        line=dict(color=LEG_COLORS[leg_k % len(LEG_COLORS)], width=2)),
            name=f"{label} spacecraft",
            showlegend=bool(t_days == t_min),
        ))

    for body in ring_bodies:
        if body not in body_theta0:
            traces.append(go.Scatter3d(x=[], y=[], z=[], mode="markers", showlegend=False))
            continue
        xi, yi, zi = mfu.planet_position_ecliptic_au(body_theta0[body], mfu.BODY_SMA_AU[body], t_days)
        traces.append(go.Scatter3d(
            x=[xi], y=[yi], z=[zi],
            mode="markers+text",
            marker=dict(color=mfu.body_color(body), size=8, symbol="circle",
                        line=dict(color="white", width=1)),
            text=[body], textposition="top center",
            textfont=dict(color=mfu.body_color(body), size=10),
            name=body, showlegend=False,
        ))

    return traces


dynamic_trace_indices = list(range(n_static, n_static + len(build_dynamic(t_min))))

t_sample = np.linspace(t_min, t_max, n_frames)

frames = []
for t in t_sample:
    frames.append(go.Frame(
        data=build_dynamic(t),
        traces=dynamic_trace_indices,
        name=f"{t:.1f}d",
        layout=go.Layout(title=dict(text=f"MGA-1DSM refined -- {mission}  |  T+{t:.0f} days")),
    ))

ext = max(
    analytic["x_ecl_au"].abs().max(), analytic["y_ecl_au"].abs().max(), analytic["z_ecl_au"].abs().max(),
) * 1.15

sequence_str = " -> ".join(body_seq)

fig = go.Figure(
    data=static_traces + build_dynamic(t_min),
    layout=go.Layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="monospace", size=12),
        title=dict(
            text=f"MGA-1DSM refined -- {sequence_str}  |  T+0 days"
                 "<br><sup>Dashed = analytic Keplerian | Solid = multiple-shooting corrected"
                 " | Ecliptic frame</sup>",
            font=dict(color="#00d4ff", size=15), x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="X [AU]", range=[-ext, ext], color="#aaaacc",
                       backgroundcolor=BG, gridcolor=GRID),
            yaxis=dict(title="Y [AU]", range=[-ext, ext], color="#aaaacc",
                       backgroundcolor=BG, gridcolor=GRID),
            zaxis=dict(title="Z [AU]", range=[-ext, ext], color="#aaaacc",
                       backgroundcolor=BG, gridcolor=GRID),
            bgcolor=BG,
            aspectmode="cube",
        ),
        height=850, width=1100,
        legend=dict(bgcolor="rgba(10,10,20,0.8)", bordercolor="#444466",
                    borderwidth=1, font=dict(color="#ccccee", size=10)),
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.0, x=1.02,
            xanchor="left", yanchor="top",
            buttons=[
                dict(label="Play", method="animate", args=[None, {
                    "frame": {"duration": max(30, 8000 // n_frames), "redraw": True},
                    "fromcurrent": True, "transition": {"duration": 0},
                }]),
                dict(label="Pause", method="animate", args=[[None], {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate", "transition": {"duration": 0},
                }]),
            ],
        )],
        sliders=[dict(
            active=0, pad={"t": 50},
            steps=[
                dict(method="animate", label=f"{t:.0f}d",
                     args=[[f"{t:.1f}d"], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate", "transition": {"duration": 0}}])
                for t in t_sample[::max(1, n_frames // 30)]
            ],
        )],
    ),
    frames=frames,
)

out_path = str(out_dir / "mga_refined_animation.html")
fig.write_html(out_path, include_plotlyjs=True)
print(f"Saved: {out_path}")
webbrowser.open(out_path)
