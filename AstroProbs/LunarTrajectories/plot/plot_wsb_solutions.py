"""
plot_wsb_solutions.py — WSB solution comparison: MC vs GA.

Reads  : out/wsb/mc_solutions.csv   (Monte Carlo results)
         out/wsb/ga_solutions.csv   (Genetic Algorithm results)
         out/wsb/mc_traj.csv        (top N trajectories from MC)

Saves  : out/wsb/wsb_solutions.html  (5-panel interactive figure)

Row 1
-----
1. Pareto front  : DV_total vs transfer time for all SOI-reaching solutions.
                   Pareto-optimal points starred and connected.
2. DV breakdown  : scatter DV_TLI vs DV_LOI, sized by est_capture_orbits.
                   Only SOI-reaching solutions shown.
3. Rotating frame: MC trajectories that entered the Hill sphere.

Row 2
-----
4. Inertial full : full Earth->Moon arcs in inertial frame.
5. Moon-centred  : Hill sphere approach in Moon-centred inertial frame.

Run:  python plot/plot_wsb_solutions.py
"""

import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wsb_style import (
    MU, X_M, L_KM, T_STAR, R_MOON, R_HILL_ND, R_HILL_KM,
    COL_WSB, COL_ART, COL_EARTH, COL_HILL,
    MULTI_COLORS, GRID_COLOR, ZERO_COLOR, PAPER_BG, BG_3D,
)

ROOT    = pathlib.Path(__file__).parent.parent
OUT_DIR = ROOT / "out" / "wsb"
MC_CSV   = OUT_DIR / "mc_solutions.csv"
GA_CSV   = OUT_DIR / "ga_solutions.csv"
TRAJ_CSV = OUT_DIR / "mc_traj.csv"

# Solutions with dv_loi_kms above this never entered the Hill sphere
LOI_PENALTY_THRESHOLD = 1.8   # km/s

VIEW_FULL = X_M * L_KM * 1.55
VIEW_MOON = R_HILL_KM * 2.8

# ---- Load data ---------------------------------------------------------------
mc_df_raw = pd.read_csv(MC_CSV) if MC_CSV.exists() else pd.DataFrame()
ga_df_raw = pd.read_csv(GA_CSV) if GA_CSV.exists() else pd.DataFrame()

if mc_df_raw.empty and ga_df_raw.empty:
    print("No solution files found. Run wsb_refine and wsb_optimize first.")
    raise SystemExit(1)

mc_df = (mc_df_raw[mc_df_raw["dv_loi_kms"] < LOI_PENALTY_THRESHOLD].copy()
         if not mc_df_raw.empty else pd.DataFrame())
ga_df = (ga_df_raw[ga_df_raw["dv_loi_kms"] < LOI_PENALTY_THRESHOLD].copy()
         if not ga_df_raw.empty else pd.DataFrame())

print(f"MC: {len(mc_df_raw)} total -> {len(mc_df)} reached SOI")
print(f"GA: {len(ga_df_raw)} total -> {len(ga_df)} reached SOI")

# ---- Colour scheme -----------------------------------------------------------
COLORS = {
    "MC":        COL_WSB,              # cyan  — MC refined
    "GA":        COL_ART,              # orange — GA global
    "MC_pareto": "#0099BB",
    "GA_pareto": "#BB5500",
}
LABELS = {"MC": "MC refined", "GA": "GA global"}

# ---- Build figure (2-row layout) ---------------------------------------------
fig = make_subplots(
    rows=2, cols=3,
    specs=[
        [{}, {}, {}],
        [{"colspan": 2}, None, {}],
    ],
    subplot_titles=[
        "Pareto front — SOI-reaching solutions",
        "DV breakdown (TLI vs LOI)",
        "SOI-reaching trajectories (rotating frame)",
        "Inertial frame — full transfer arcs",
        None,
        "Moon-centred inertial — Hill sphere approach",
    ],
    column_widths=[0.26, 0.26, 0.48],
    row_heights=[0.45, 0.55],
    horizontal_spacing=0.07,
    vertical_spacing=0.14,
)

# ---- Panel 1: Pareto front ---------------------------------------------------
for df, key in [(mc_df, "MC"), (ga_df, "GA")]:
    if df.empty:
        continue
    color = COLORS[key]; label = LABELS[key]
    pareto_mask = df.get("pareto_optimal", pd.Series(0, index=df.index)).astype(bool)

    fig.add_trace(go.Scatter(
        x=df["t_transfer_days"],
        y=df["dv_total_kms"],
        mode="markers",
        marker=dict(color=color, size=6, opacity=0.6),
        name=f"{label} all",
        legendgroup=key,
    ), row=1, col=1)

    if pareto_mask.any():
        sub = df[pareto_mask].sort_values("t_transfer_days")
        fig.add_trace(go.Scatter(
            x=sub["t_transfer_days"],
            y=sub["dv_total_kms"],
            mode="markers+lines",
            marker=dict(
                symbol="star", size=12,
                color=COLORS[f"{key}_pareto"],
                line=dict(width=1, color="white"),
            ),
            line=dict(color=COLORS[f"{key}_pareto"], width=1.5, dash="dot"),
            name=f"{label} Pareto",
            legendgroup=key,
        ), row=1, col=1)

fig.update_xaxes(title_text="Transfer time [days]", row=1, col=1)
fig.update_yaxes(title_text="DV_total [km/s]",      row=1, col=1)

# ---- Panel 2: DV breakdown ---------------------------------------------------
for df, key in [(mc_df, "MC"), (ga_df, "GA")]:
    if df.empty:
        continue
    color = COLORS[key]; label = LABELS[key]
    orbs = df["est_capture_orbits"].clip(0, 20)
    fig.add_trace(go.Scatter(
        x=df["dv_tli_kms"],
        y=df["dv_loi_kms"],
        mode="markers",
        marker=dict(
            color=orbs, colorscale="Plasma",
            size=7, opacity=0.7,
            cmin=0, cmax=10,
            colorbar=dict(title="Capture orbits", thickness=12,
                          x=0.53 if key == "MC" else None),
            showscale=(key == "MC"),
            symbol="circle" if key == "MC" else "diamond",
            line=dict(width=0.5, color="rgba(150,150,150,0.4)"),
        ),
        name=f"{label} DV",
        legendgroup=key,
        showlegend=False,
    ), row=1, col=2)

fig.update_xaxes(title_text="DV_TLI [km/s]", row=1, col=2)
fig.update_yaxes(title_text="DV_LOI [km/s]", row=1, col=2)

# ---- Panel 3: SOI-reaching trajectories in rotating frame --------------------
traj_df         = None
soi_ranks       = set()
soi_ranks_sorted: list = []

if TRAJ_CSV.exists():
    traj_df = pd.read_csv(TRAJ_CSV)
    traj_df = traj_df[traj_df["x_nd"].notna()].copy()

    if "dv_loi_kms" in traj_df.columns:
        for rank in traj_df["rank"].unique():
            sub = traj_df[traj_df["rank"] == rank]
            if sub["dv_loi_kms"].iloc[0] < LOI_PENALTY_THRESHOLD:
                soi_ranks.add(rank)
    else:
        moon_dist = np.sqrt((traj_df["x_nd"] - X_M)**2 + traj_df["y_nd"]**2)
        for rank in traj_df["rank"].unique():
            if (moon_dist[traj_df["rank"] == rank] < R_HILL_ND).any():
                soi_ranks.add(rank)

    soi_ranks_sorted = sorted(soi_ranks)
    for k, rank in enumerate(soi_ranks_sorted[:10]):
        sub  = traj_df[traj_df["rank"] == rank]
        col  = MULTI_COLORS[k % len(MULTI_COLORS)]
        dv_t = sub["dv_total_kms"].iloc[0]
        orbs = sub["est_capture_orbits"].iloc[0]
        fig.add_trace(go.Scatter(
            x=sub["x_nd"] * L_KM,
            y=sub["y_nd"] * L_KM,
            mode="lines",
            line=dict(color=col, width=1.2),
            opacity=0.85,
            name=f"MC #{rank}  DV={dv_t:.3f}  {orbs:.1f}orb",
        ), row=1, col=3)

# Moon and Hill sphere in rotating frame
a = np.linspace(0, 2 * np.pi, 120)
xm_rot = X_M * L_KM + R_MOON * np.cos(a)
ym_rot = R_MOON * np.sin(a)
fig.add_trace(go.Scatter(
    x=xm_rot, y=ym_rot, mode="lines", fill="toself",
    fillcolor="rgba(160,160,160,0.3)",
    line=dict(color="#888888", width=1),
    name="Moon", showlegend=(not bool(soi_ranks)),
), row=1, col=3)
fig.add_trace(go.Scatter(
    x=X_M * L_KM + R_HILL_KM * np.cos(a),
    y=R_HILL_KM * np.sin(a),
    mode="lines",
    line=dict(color=COL_HILL, width=1.2, dash="dash"),
    name="Hill sphere", showlegend=True,
), row=1, col=3)

VIEW_ROT = R_HILL_KM * 3.5
fig.update_xaxes(
    title_text="x [km] (rotating)", scaleanchor="y3", scaleratio=1,
    range=[X_M * L_KM - VIEW_ROT, X_M * L_KM + VIEW_ROT], row=1, col=3,
)
fig.update_yaxes(title_text="y [km] (rotating)", range=[-VIEW_ROT, VIEW_ROT], row=1, col=3)

# ---- Panels 4 & 5: Inertial frame (row 2) ------------------------------------
if traj_df is not None and "time_nd" in traj_df.columns and soi_ranks_sorted:
    xe  = 6_371 * np.cos(a);  ye  = 6_371 * np.sin(a)
    xmo = X_M * L_KM * np.cos(a);  ymo = X_M * L_KM * np.sin(a)
    fig.add_trace(go.Scatter(
        x=xe/1e3, y=ye/1e3, fill="toself",
        fillcolor=COL_EARTH,
        line=dict(color=COL_EARTH, width=0),
        name="Earth", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=xmo/1e3, y=ymo/1e3, mode="lines",
        line=dict(color="rgba(100,100,180,0.25)", width=1, dash="dot"),
        name="Moon orbit", showlegend=False,
    ), row=2, col=1)

    xmd = R_MOON * np.cos(a);  ymd = R_MOON * np.sin(a)
    xhs = R_HILL_KM * np.cos(a);  yhs = R_HILL_KM * np.sin(a)
    fig.add_trace(go.Scatter(
        x=xmd/1e3, y=ymd/1e3, fill="toself",
        fillcolor="rgba(160,160,160,0.4)",
        line=dict(color="#888888", width=1),
        name="Moon", showlegend=False,
    ), row=2, col=3)
    fig.add_trace(go.Scatter(
        x=xhs/1e3, y=yhs/1e3, mode="lines",
        line=dict(color=COL_HILL, width=1.5, dash="dash"),
        name="Hill sphere", showlegend=False,
    ), row=2, col=3)

    for k, rank in enumerate(soi_ranks_sorted[:10]):
        sub  = traj_df[traj_df["rank"] == rank]
        col  = MULTI_COLORS[k % len(MULTI_COLORS)]
        t_nd = sub["time_nd"].values
        x_nd = sub["x_nd"].values
        y_nd = sub["y_nd"].values

        xi  = (x_nd * np.cos(t_nd) - y_nd * np.sin(t_nd)) * L_KM
        yi  = (x_nd * np.sin(t_nd) + y_nd * np.cos(t_nd)) * L_KM
        xm_i = X_M * np.cos(t_nd) * L_KM
        ym_i = X_M * np.sin(t_nd) * L_KM
        xi_mc = xi - xm_i
        yi_mc = yi - ym_i

        dv_t = sub["dv_total_kms"].iloc[0]
        orbs = sub["est_capture_orbits"].iloc[0]

        fig.add_trace(go.Scatter(
            x=xi/1e3, y=yi/1e3, mode="lines",
            line=dict(color=col, width=1.2),
            opacity=0.85,
            name=f"MC #{rank}  DV={dv_t:.3f}  {orbs:.1f}orb",
            legendgroup=f"mc_{rank}",
            showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=xi_mc/1e3, y=yi_mc/1e3, mode="lines",
            line=dict(color=col, width=1.2),
            opacity=0.85,
            name=f"MC #{rank}",
            legendgroup=f"mc_{rank}",
            showlegend=False,
        ), row=2, col=3)

    fig.update_xaxes(
        title_text="x [x10^3 km]",
        range=[-VIEW_FULL/1e3, VIEW_FULL/1e3],
        scaleanchor="y4", scaleratio=1,
        row=2, col=1,
    )
    fig.update_yaxes(
        title_text="y [x10^3 km]",
        range=[-VIEW_FULL/1e3, VIEW_FULL/1e3],
        row=2, col=1,
    )
    fig.update_xaxes(
        title_text="x from Moon [x10^3 km]",
        range=[-VIEW_MOON/1e3, VIEW_MOON/1e3],
        scaleanchor="y5", scaleratio=1,
        row=2, col=3,
    )
    fig.update_yaxes(
        title_text="y from Moon [x10^3 km]",
        range=[-VIEW_MOON/1e3, VIEW_MOON/1e3],
        row=2, col=3,
    )

# ---- Layout ------------------------------------------------------------------
n_mc = len(mc_df)
n_ga = len(ga_df)
n_pareto_mc = int(mc_df.get("pareto_optimal", pd.Series(0)).sum()) if not mc_df.empty else 0
n_pareto_ga = int(ga_df.get("pareto_optimal", pd.Series(0)).sum()) if not ga_df.empty else 0

fig.update_layout(
    title=dict(
        text=(
            "WSB Solutions: GA global search + MC refined — SOI-reaching only<br>"
            f"<sup>GA global: {n_ga} | MC refined: {n_mc} | "
            f"Pareto: {n_pareto_mc} MC + {n_pareto_ga} GA | "
            f"Objective: DV_TLI + DV_LOI_min</sup>"
        ),
        font=dict(size=13),
        x=0.01, xanchor="left",
    ),
    height=1000,
    template="plotly_dark",
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=BG_3D,
    font=dict(color="white"),
    legend=dict(x=1.01, y=1, font=dict(size=9),
                bgcolor="rgba(15,15,25,0.7)"),
)

# Apply dark grid to all 2D subplots
fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=ZERO_COLOR)
fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=ZERO_COLOR)

out_path = OUT_DIR / "wsb_solutions.html"
fig.write_html(str(out_path))
print(f"Saved {out_path}")
