"""plot_mga_refined.py — overlay analytic MGA arc vs. multiple-shooting corrected arc.

Usage:
    python plot/plot_mga_refined.py <mission>

Example:
    python plot/plot_mga_refined.py evj_flyby

Reads:
    out/<mission>/mga_best.csv           — analytic Keplerian arc
    out/<mission>/mga_refined.csv        — multiple-shooting corrected arc
    out/<mission>/mga_legs.csv           — leg/body info for orbit rings
    out/<mission>/mga_refined_legs.csv   — corrected DSM ΔV magnitudes

Produces an interactive HTML file:
    out/<mission>/mga_refined.html

Frame: always plotted in the ECLIPTIC frame — ANISE's native output is J2000
EQUATORIAL (ICRF); see mga_frame_utils.py for the exact fixed rotation used to
convert. The backend (propagation, optimization) is untouched — this is a
display-layer-only transform.
"""

import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import mga_frame_utils as mfu

# ── Paths ─────────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python plot/plot_mga_refined.py <mission>")
    sys.exit(1)

mission   = sys.argv[1]
out_dir   = os.path.join("out", mission)

analytic_csv  = os.path.join(out_dir, "mga_best.csv")
refined_csv   = os.path.join(out_dir, "mga_refined.csv")
legs_csv      = os.path.join(out_dir, "mga_legs.csv")
ref_legs_csv  = os.path.join(out_dir, "mga_refined_legs.csv")
html_out      = os.path.join(out_dir, "mga_refined.html")

for p in (analytic_csv, refined_csv, legs_csv):
    if not os.path.exists(p):
        print(f"Missing file: {p}")
        sys.exit(1)

# ── Load data (converted to ecliptic AU columns immediately) ────────────────

analytic = mfu.add_ecliptic_columns(pd.read_csv(analytic_csv))
refined  = mfu.add_ecliptic_columns(pd.read_csv(refined_csv))
legs     = pd.read_csv(legs_csv)

# ── Colour palette ────────────────────────────────────────────────────────────

DARK_BG   = "#0a0a14"
LEG_COLS  = ["#00d4ff", "#ff6b35", "#a8ff3e", "#ff3ea8", "#ffd700", "#cf9fff"]

# ── Build figure ──────────────────────────────────────────────────────────────

fig = go.Figure()

n_legs = legs["leg_idx"].nunique()

# ── Analytic arc (dashed, per leg) ───────────────────────────────────────────

for k in range(n_legs):
    seg = analytic[analytic["leg_idx"] == k]
    if seg.empty:
        continue
    col = LEG_COLS[k % len(LEG_COLS)]
    body_arr = legs[legs["leg_idx"] == k]["body_arr"].iloc[0]
    fig.add_trace(go.Scatter3d(
        x=seg["x_ecl_au"], y=seg["y_ecl_au"], z=seg["z_ecl_au"],
        mode="lines",
        line=dict(color=col, width=2, dash="dash"),
        name=f"Analytic leg {k} (→{body_arr})",
        legendgroup=f"leg{k}_analytic",
        showlegend=True,
        hovertemplate="Analytic leg %d<br>x=%%{x:.3f} AU<br>y=%%{y:.3f} AU<extra></extra>" % k,
    ))

# ── DSM markers — analytic (dim) and refined (bright), both shown ───────────
# Both use the same (fixed) eta/tof geometry, so the DSM occurs at the same
# *time* for both; only the applied delta-v (and therefore the exact position
# after N-body perturbation) differs, which is why the refined marker can sit
# at a slightly different point than the analytic one.

for k in range(n_legs):
    row = legs[legs["leg_idx"] == k].iloc[0]
    col = LEG_COLS[k % len(LEG_COLS)]

    xe, ye, ze = mfu.icrf_point_to_ecliptic_au(row["x_dsm_m"], row["y_dsm_m"], row["z_dsm_m"])
    fig.add_trace(go.Scatter3d(
        x=[xe], y=[ye], z=[ze],
        mode="markers",
        marker=dict(symbol="x", color=col, size=5, opacity=0.45),
        name=f"Analytic DSM leg {k}",
        legendgroup=f"leg{k}_analytic",
        showlegend=True,
        hovertemplate="Analytic DSM leg %d<br>ΔV=%.1f m/s<extra></extra>" % (k, row["dv_dsm_ms"]),
    ))

    # t_dep for this leg = first sample of this leg in the refined arc (avoids
    # needing a JD<->t_days conversion here).
    leg_start = refined[refined["leg_idx"] == k]
    if not leg_start.empty:
        t_dep_days = float(leg_start["t_days"].min())
        dsm_ref = mfu.find_dsm_point(refined, k, t_dep_days, row["eta"], row["tof_days"])
        if dsm_ref is not None:
            t_dsm_days, xr, yr, zr = dsm_ref
            xe, ye, ze = mfu.icrf_point_to_ecliptic_au(xr, yr, zr)
            fig.add_trace(go.Scatter3d(
                x=[xe], y=[ye], z=[ze],
                mode="markers",
                marker=dict(symbol="x", color=col, size=9,
                            line=dict(color="white", width=1.5)),
                name=f"Refined DSM leg {k}",
                legendgroup=f"leg{k}_refined",
                showlegend=True,
                hovertemplate=f"Refined DSM leg {k}<br>T+{t_dsm_days:.1f} days<extra></extra>",
            ))

# ── Refined arc (solid, per leg) ─────────────────────────────────────────────

for k in range(n_legs):
    seg = refined[refined["leg_idx"] == k]
    if seg.empty:
        continue
    col = LEG_COLS[k % len(LEG_COLS)]
    body_arr = legs[legs["leg_idx"] == k]["body_arr"].iloc[0]
    fig.add_trace(go.Scatter3d(
        x=seg["x_ecl_au"], y=seg["y_ecl_au"], z=seg["z_ecl_au"],
        mode="lines",
        line=dict(color=col, width=4),
        name=f"Refined leg {k} (→{body_arr})",
        legendgroup=f"leg{k}_refined",
        showlegend=True,
        hovertemplate="Refined leg %d<br>x=%%{x:.3f} AU<br>y=%%{y:.3f} AU<extra></extra>" % k,
    ))

# ── Body orbit rings (flat in the ecliptic — no tilt needed) ───────────────

BODY_ORBIT_AU = mfu.BODY_SMA_AU

bodies_shown = set()
all_bodies = (
    list(legs["body_dep"].str.title().unique())
    + list(legs["body_arr"].str.title().unique())
)
for bname in all_bodies:
    if bname in bodies_shown or bname not in BODY_ORBIT_AU:
        continue
    bodies_shown.add(bname)
    a = BODY_ORBIT_AU[bname]
    col = mfu.body_color(bname)
    xs, ys, zs = mfu.ring_xyz(a)
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=col, width=1),
        name=bname + " orbit",
        hovertemplate=bname + " orbit ring<extra></extra>",
    ))

# ── Real body positions at each flyby / arrival epoch ────────────────────────
# These are exact ANISE positions, not the circular-orbit approximation above:
# the analytic Lambert arc targets the body's true center exactly, so the last
# point of each leg *is* that body's real heliocentric position at arrival.

for k in range(n_legs):
    seg = analytic[analytic["leg_idx"] == k]
    if seg.empty:
        continue
    last = seg.iloc[-1]
    row = legs[legs["leg_idx"] == k].iloc[0]
    body_arr = row["body_arr"]
    t_arr_days = float(last["t_days"])
    col = LEG_COLS[k % len(LEG_COLS)]
    fig.add_trace(go.Scatter3d(
        x=[last["x_ecl_au"]], y=[last["y_ecl_au"]], z=[last["z_ecl_au"]],
        mode="markers+text",
        marker=dict(color=col, size=8, symbol="circle",
                    line=dict(color="white", width=1.5)),
        text=[f"{body_arr}<br>T+{t_arr_days:.0f}d"],
        textposition="top center",
        textfont=dict(color=col, size=10),
        name=f"{body_arr} at encounter (leg {k})",
        showlegend=True,
        hovertemplate=f"{body_arr} at encounter<br>T+{t_arr_days:.1f} days<extra></extra>",
    ))

# ── Refined-arc flyby periapsis markers at intermediate flyby bodies ─────────
# The multiple-shooting constraint drives each intermediate leg's endpoint to
# the flyby body's periapsis position — so the refined leg-k endpoint IS the
# flyby periapsis, exactly, by construction. Its distance to the analytic
# leg-k endpoint (= the body's true ANISE center at encounter) is the real
# periapsis radius achieved. (Distance computed in ICRF meters — a rotation
# doesn't change distances, so this is frame-independent either way.)

for k in range(n_legs - 1):  # intermediate flyby bodies only, not the final target
    body_row = legs[legs["leg_idx"] == k].iloc[0]
    body_name = body_row["body_arr"]
    seg_a_arc = analytic[analytic["leg_idx"] == k]
    seg_r_arc = refined[refined["leg_idx"] == k]
    if seg_a_arc.empty or seg_r_arc.empty:
        continue
    body_center = seg_a_arc.iloc[-1]     # true ANISE body position at encounter
    peri = seg_r_arc.iloc[-1]            # refined arc's periapsis patch point
    dist_m = np.sqrt((peri["x_m"] - body_center["x_m"])**2
                     + (peri["y_m"] - body_center["y_m"])**2
                     + (peri["z_m"] - body_center["z_m"])**2)
    col = LEG_COLS[k % len(LEG_COLS)]
    fig.add_trace(go.Scatter3d(
        x=[peri["x_ecl_au"]], y=[peri["y_ecl_au"]], z=[peri["z_ecl_au"]],
        mode="markers+text",
        marker=dict(color="#ffffff", size=7, symbol="diamond",
                    line=dict(color=col, width=2)),
        text=[f"Flyby periapsis<br>{dist_m/1000:.0f} km"],
        textposition="bottom center",
        textfont=dict(color="#ffffff", size=9),
        name=f"Refined periapsis @ {body_name}",
        showlegend=True,
        hovertemplate=(f"Refined flyby periapsis @ {body_name}<br>"
                        f"T+{float(peri['t_days']):.1f} days<br>"
                        f"distance from body center = {dist_m/1000:.0f} km<extra></extra>"),
    ))

# ── SOI passages on the refined arc ──────────────────────────────────────────
# The N-body propagator records which body was gravitationally central at each
# sample (central_body column, empty = Sun). Highlight the samples where the
# spacecraft is inside a flyby body's SOI — this is where the gravity assist
# actually acts on the trajectory, as opposed to the DSM burns we command.

if "central_body" in refined.columns:
    soi_rows = refined[refined["central_body"].notna() & (refined["central_body"] != "")]
    for body_name, grp in soi_rows.groupby("central_body"):
        fig.add_trace(go.Scatter3d(
            x=grp["x_ecl_au"], y=grp["y_ecl_au"], z=grp["z_ecl_au"],
            mode="markers",
            marker=dict(color="#ffffff", size=4, symbol="circle-open",
                        line=dict(color="#ffffff", width=2)),
            name=f"Inside {body_name} SOI",
            showlegend=True,
            hovertemplate=f"Inside {body_name} SOI<br>T+%{{text}} days<extra></extra>",
            text=[f"{t:.1f}" for t in grp["t_days"]],
        ))

# ── Sun ───────────────────────────────────────────────────────────────────────

fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode="markers",
    marker=dict(color="#ffffc0", size=10, symbol="circle"),
    name="Sun",
    hovertemplate="Sun<extra></extra>",
))

# ── Layout ────────────────────────────────────────────────────────────────────

fig.update_layout(
    title=dict(
        text=f"MGA refined arc — {mission.replace('_', ' ').title()}"
             "<br><sup>Dashed = analytic Keplerian  |  Solid = multiple-shooting corrected"
             "  |  Ecliptic frame</sup>",
        font=dict(color="#e0e8ff", size=16),
    ),
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    scene=dict(
        xaxis=dict(title="X [AU]", color="#aaaacc", backgroundcolor=DARK_BG, gridcolor="#222233"),
        yaxis=dict(title="Y [AU]", color="#aaaacc", backgroundcolor=DARK_BG, gridcolor="#222233"),
        zaxis=dict(title="Z [AU]", color="#aaaacc", backgroundcolor=DARK_BG, gridcolor="#222233"),
        bgcolor=DARK_BG,
        aspectmode="data",
    ),
    legend=dict(
        bgcolor="rgba(10,10,20,0.8)",
        font=dict(color="#ccccee", size=11),
    ),
    margin=dict(l=0, r=0, t=80, b=0),
)

# ── Corrected DSM ΔV annotation ───────────────────────────────────────────────

if os.path.exists(ref_legs_csv):
    ref_legs = pd.read_csv(ref_legs_csv)
    lines = ["<b>Corrected DSM ΔVs</b>"]
    for _, row in ref_legs.iterrows():
        lines.append(f"Leg {int(row['leg_idx'])}: {row['dv_mag_ms']:.1f} m/s  "
                     f"(dx={row['dv_x_ms']:.1f}, dy={row['dv_y_ms']:.1f}, dz={row['dv_z_ms']:.1f})")
    fig.add_annotation(
        text="<br>".join(lines),
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        align="left",
        showarrow=False,
        font=dict(color="#ccccee", size=11),
        bgcolor="rgba(10,10,20,0.75)",
        bordercolor="#444466",
        borderwidth=1,
    )

# ── Write HTML ────────────────────────────────────────────────────────────────

fig.write_html(html_out, include_plotlyjs="cdn")
print(f"Wrote {html_out}")
