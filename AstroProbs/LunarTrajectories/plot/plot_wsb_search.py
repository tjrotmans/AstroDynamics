"""
plot_wsb_search.py — inertial frame, real units (km), plus search-space analytics.

Run:  python plot/plot_wsb_search.py
Reads: out/wsb_search/blt_candidates.csv
       out/wsb_search/blt_info.txt
       out/wsb_search/family_analysis.csv
Saves: out/wsb_search/wsb_search.html      — inertial-frame trajectory plot
       out/wsb_search/wsb_pareto.html      — ΔV vs transfer time, Pareto front
       out/wsb_search/wsb_heatmap.html     — capture quality heatmap (θ vs θ_sun)
       out/wsb_search/wsb_scatter.html     — orbits vs transfer time, crash-annotated
"""
import re
import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = pathlib.Path(__file__).parent.parent
OUT  = ROOT / "out" / "wsb"

# ── constants ─────────────────────────────────────────────────────────────────
MU      = 0.01215565
X_E     = -MU
X_M     = 1.0 - MU
R_HILL  = (MU / 3.0) ** (1.0 / 3.0)
OMEGA_S = 27.321661 / 365.25 - 1.0
R_SEL1  = 3.91
L_KM    = 384_400.0
T_STAR  = 375_700.0

cand_path = OUT / "blt_candidates.csv"
info_path  = OUT / "blt_info.txt"
if not cand_path.exists():
    print("Run: cargo run -p lunar_trajectories --bin wsb_search --release")
    raise SystemExit(1)

cand_df = pd.read_csv(cand_path)

# Parse theta_sun per candidate from blt_info.txt
theta_sun_map = {}
if info_path.exists():
    text = info_path.read_text(encoding="utf-8")
    parts = re.split(r"--- Candidate (\d+) ---", text)
    for i in range(1, len(parts), 2):
        cid  = int(parts[i])
        body = parts[i + 1]
        m = re.search(r"θ_sun.*?:\s*([\d.]+)°", body)
        if m:
            theta_sun_map[cid] = float(m.group(1)) * np.pi / 180.0

print(f"Candidates: {cand_df['cand_id'].nunique()}")
print(f"theta_sun map: { {k: f'{np.degrees(v):.1f}°' for k,v in theta_sun_map.items()} }")

# ── helpers ───────────────────────────────────────────────────────────────────

def to_inertial_km(t_nd, x_nd, y_nd):
    xi = (x_nd * np.cos(t_nd) - y_nd * np.sin(t_nd)) * L_KM
    yi = (x_nd * np.sin(t_nd) + y_nd * np.cos(t_nd)) * L_KM
    return xi, yi

def sel1_path_km(theta_sun0, t_start, t_end, n=300):
    t = np.linspace(t_start, t_end, n)
    phi = theta_sun0 + (OMEGA_S + 1.0) * t
    xi = R_SEL1 * np.cos(phi) * L_KM
    yi = R_SEL1 * np.sin(phi) * L_KM
    return xi, yi, t

def circle_km(cx_km, cy_km, r_km, n=200):
    t = np.linspace(0, 2 * np.pi, n)
    return cx_km + r_km * np.cos(t), cy_km + r_km * np.sin(t)

# ════════════════════════════════════════════════════════════════════════════════
# Plot 1 — Inertial-frame trajectories
# ════════════════════════════════════════════════════════════════════════════════

fig = go.Figure()

COLORS  = ["#e63946", "#2a9d8f", "#e9c46a", "#457b9d", "#f4a261"]
TOP5    = sorted(cand_df["cand_id"].dropna().unique().astype(int))[:5]

for k, cid in enumerate(TOP5):
    sub = cand_df[cand_df["cand_id"] == cid].dropna(subset=["x_nd", "y_nd"])
    col = COLORS[k]

    t_nd = sub["time_nd"].values
    t_max = t_nd.max()
    t_days = t_max * T_STAR / 86400.0

    xi, yi = to_inertial_km(t_nd, sub["x_nd"].values, sub["y_nd"].values)
    theta_sun0 = theta_sun_map.get(cid, 0.0)

    fig.add_trace(go.Scatter(
        x=xi / 1e3, y=yi / 1e3,
        mode="lines",
        line=dict(color=col, width=2.0),
        name=f"Candidate {k+1}  ({t_days:.0f} d)",
        legendgroup=f"c{cid}",
    ))

    xs, ys, _ = sel1_path_km(theta_sun0, 0.0, t_max)
    fig.add_trace(go.Scatter(
        x=xs / 1e3, y=ys / 1e3,
        mode="lines",
        line=dict(color=col, width=1.0, dash="dash"),
        name=f"SE-L1 drift {k+1}",
        legendgroup=f"c{cid}",
        showlegend=False,
    ))

xmo, ymo = circle_km(0, 0, X_M * L_KM)
fig.add_trace(go.Scatter(x=xmo/1e3, y=ymo/1e3, mode="lines",
    line=dict(color="silver", width=1.0, dash="dot"), name="Moon orbit"))

xe, ye = circle_km(0, 0, 6_371)
fig.add_trace(go.Scatter(x=xe/1e3, y=ye/1e3, fill="toself",
    fillcolor="steelblue", line=dict(color="steelblue", width=0), name="Earth"))

xm0, ym0 = circle_km(X_M * L_KM, 0, 1_737.4)
fig.add_trace(go.Scatter(x=xm0/1e3, y=ym0/1e3, fill="toself",
    fillcolor="silver", line=dict(color="grey", width=1), name="Moon (t=0)"))

xh, yh = circle_km(X_M * L_KM, 0, R_HILL * L_KM)
fig.add_trace(go.Scatter(x=xh/1e3, y=yh/1e3, mode="lines",
    line=dict(color="orange", width=1.5, dash="dash"), name="Hill sphere (t=0)"))

VIEW = 2_200
fig.update_layout(
    title=dict(
        text=(
            "WSB Ballistic Lunar Transfer — Inertial Frame  [×10³ km]<br>"
            "<sup>Dotted = Moon orbit  |  Dashed = SE-L1 path  |  ★ = Hill sphere entry</sup>"
        ),
        font=dict(size=14),
    ),
    xaxis=dict(title="x [×10³ km]", range=[-VIEW, VIEW], scaleanchor="y", scaleratio=1),
    yaxis=dict(title="y [×10³ km]", range=[-VIEW, VIEW]),
    height=750,
    legend=dict(x=1.01, y=1, font=dict(size=9)),
    template="plotly_white",
)
fig.write_html(str(OUT / "wsb_search.html"))
print(f"Saved {OUT / 'wsb_search.html'}")

# ════════════════════════════════════════════════════════════════════════════════
# Load family_analysis.csv for all remaining plots
# ════════════════════════════════════════════════════════════════════════════════

fam_path = OUT / "family_analysis.csv"
if not fam_path.exists():
    print("family_analysis.csv not found — run wsb_search first")
    raise SystemExit(0)

fam = pd.read_csv(fam_path)
has_crash = "crashed_moon" in fam.columns
if has_crash:
    fam["crashed_moon"] = fam["crashed_moon"].fillna(0).astype(bool)
else:
    fam["crashed_moon"] = False

print(f"  {len(fam)} solutions loaded  ({fam['crashed_moon'].sum()} moon-crash flagged)")

FAMILY_COLORS = {
    "Q2-fast":   "#e63946",
    "Q2-medium": "#f4a261",
    "Q2-slow":   "#e9c46a",
    "Q4-fast":   "#2a9d8f",
    "Q4-medium": "#457b9d",
    "Q4-slow":   "#8338ec",
}

# ════════════════════════════════════════════════════════════════════════════════
# Plot 2 — Pareto front: ΔV vs transfer time
# ════════════════════════════════════════════════════════════════════════════════

fig2 = go.Figure()

for fam_name, col in FAMILY_COLORS.items():
    sub = fam[fam["family"] == fam_name]
    if sub.empty:
        continue
    # Normal solutions
    normal = sub[~sub["crashed_moon"]]
    crashed = sub[sub["crashed_moon"]]

    if not normal.empty:
        fig2.add_trace(go.Scatter(
            x=normal["t_transfer_days"], y=normal["dv_kms"],
            mode="markers",
            marker=dict(color=col, size=5, opacity=0.5),
            name=fam_name, legendgroup=fam_name,
            hovertemplate=(
                f"<b>{fam_name}</b><br>"
                "t = %{x:.1f} d<br>ΔV = %{y:.4f} km/s<br>"
                "orbits=%{customdata[0]:.1f}<extra></extra>"
            ),
            customdata=normal[["est_capture_orbits"]].values,
        ))

    if not crashed.empty:
        fig2.add_trace(go.Scatter(
            x=crashed["t_transfer_days"], y=crashed["dv_kms"],
            mode="markers",
            marker=dict(color=col, size=7, opacity=0.6, symbol="x",
                        line=dict(color="black", width=1)),
            name=f"{fam_name} ☠", legendgroup=fam_name, showlegend=True,
            hovertemplate=(
                f"<b>{fam_name} — MOON CRASH</b><br>"
                "t = %{x:.1f} d<br>ΔV = %{y:.4f} km/s<br>"
                "alt=%{customdata[0]:.0f} km<extra></extra>"
            ),
            customdata=crashed[["min_alt_km"]].values,
        ))

pareto = fam[fam["pareto_rank"] == 1].sort_values("t_transfer_days")
if not pareto.empty:
    fig2.add_trace(go.Scatter(
        x=pareto["t_transfer_days"], y=pareto["dv_kms"],
        mode="markers+lines",
        marker=dict(color="white", size=10, symbol="star",
                    line=dict(color="black", width=1.5)),
        line=dict(color="black", width=1.5, dash="dot"),
        name="Pareto front (rank 1)",
        hovertemplate=(
            "<b>Pareto optimal</b><br>t = %{x:.1f} d<br>"
            "ΔV = %{y:.4f} km/s<br>family=%{customdata}<extra></extra>"
        ),
        customdata=pareto["family"].values,
    ))

# 80-day preference line
fig2.add_vline(x=80, line_dash="dash", line_color="rgba(200,50,50,0.5)",
               annotation_text="80 d target", annotation_position="top right")

fig2.update_layout(
    title=dict(
        text=(
            "WSB Transfer Family Analysis — ΔV vs Transfer Time<br>"
            "<sup>★ = Pareto-optimal  |  ✕ = Moon-crash flagged  |  "
            "Red dashed = 80-day preference boundary<br>"
            "Q2 = apogee toward Sun  |  Q4 = apogee away from Sun</sup>"
        ),
        font=dict(size=13),
    ),
    xaxis=dict(title="Transfer time [days]"),
    yaxis=dict(title="TLI ΔV [km/s]"),
    height=620,
    legend=dict(x=1.01, y=1, font=dict(size=9)),
    template="plotly_white",
)
fig2.write_html(str(OUT / "wsb_pareto.html"))
print(f"Saved {OUT / 'wsb_pareto.html'}")
print(f"  {len(pareto)} Pareto-optimal solutions across {fam['family'].nunique()} families")

# ════════════════════════════════════════════════════════════════════════════════
# Plot 3 — Heatmap: capture quality in (θ, θ_sun) parameter space
# ════════════════════════════════════════════════════════════════════════════════

# Only plot families separately if there are both Q2 and Q4 solutions
families_present = fam["family"].unique().tolist()
has_q2 = any("Q2" in f for f in families_present)
has_q4 = any("Q4" in f for f in families_present)
n_panels = (1 if has_q2 else 0) + (1 if has_q4 else 0)

if n_panels == 0:
    print("No solutions for heatmap — skipping")
else:
    subplot_titles = []
    if has_q2: subplot_titles.append("Q2 family (apogee toward Sun)")
    if has_q4: subplot_titles.append("Q4 family (apogee away from Sun)")

    fig3 = make_subplots(
        rows=1, cols=n_panels,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
    )

    panel = 1
    for q, q_label in [("Q2", has_q2), ("Q4", has_q4)]:
        if not q_label:
            continue
        sub = fam[fam["family"].str.startswith(q)].copy()

        # Round to nearest degree for binning
        sub["theta_bin"]     = sub["theta_deg"].round(0)
        sub["theta_sun_bin"] = sub["theta_sun_deg"].round(0)

        # Pivot: max est_capture_orbits per (θ, θ_sun) bin
        pivot = sub.groupby(["theta_bin", "theta_sun_bin"])["est_capture_orbits"].max().reset_index()
        pivot_crash = sub.groupby(["theta_bin", "theta_sun_bin"])["crashed_moon"].any().reset_index()

        # Build 2D grid
        theta_vals   = sorted(pivot["theta_bin"].unique())
        sun_vals     = sorted(pivot["theta_sun_bin"].unique())
        z_mat        = np.full((len(sun_vals), len(theta_vals)), np.nan)
        crash_mat    = np.zeros((len(sun_vals), len(theta_vals)), dtype=bool)
        t2i = {t: i for i, t in enumerate(theta_vals)}
        s2i = {s: i for i, s in enumerate(sun_vals)}

        for _, row in pivot.iterrows():
            ti = t2i[row["theta_bin"]]
            si = s2i[row["theta_sun_bin"]]
            z_mat[si, ti] = row["est_capture_orbits"]

        for _, row in pivot_crash.iterrows():
            ti = t2i[row["theta_bin"]]
            si = s2i[row["theta_sun_bin"]]
            crash_mat[si, ti] = row["crashed_moon"]

        fig3.add_trace(go.Heatmap(
            x=theta_vals, y=sun_vals, z=z_mat,
            colorscale="Viridis", zmin=0, zmax=max(6, float(np.nanmax(z_mat))),
            colorbar=dict(title="Est. orbits", x=1.02 if panel == n_panels else 0.45,
                          len=0.9),
            hovertemplate="θ=%{x:.0f}°  θ_sun=%{y:.0f}°<br>orbits=%{z:.2f}<extra></extra>",
            name=f"{q} quality",
        ), row=1, col=panel)

        # Overlay crash markers
        crash_rows_idx, crash_cols_idx = np.where(crash_mat)
        if len(crash_rows_idx):
            cx = [theta_vals[ci] for ci in crash_cols_idx]
            cy = [sun_vals[ri]   for ri in crash_rows_idx]
            fig3.add_trace(go.Scatter(
                x=cx, y=cy, mode="markers",
                marker=dict(symbol="x", size=8, color="red",
                            line=dict(color="red", width=2)),
                name="Moon crash",
                showlegend=(panel == 1),
                hovertemplate="θ=%{x:.0f}°  θ_sun=%{y:.0f}°<br>☠ Moon crash<extra></extra>",
            ), row=1, col=panel)

        panel += 1

    fig3.update_layout(
        title=dict(
            text=(
                "WSB Search Space — Capture Quality Heatmap  (θ vs θ_sun)<br>"
                "<sup>Colour = max estimated capture orbits per grid cell  |  "
                "✕ = moon-crash flagged  |  "
                "Bright clusters = good seed regions for refinement</sup>"
            ),
            font=dict(size=13),
        ),
        height=560,
        template="plotly_white",
    )
    for col_idx in range(1, n_panels + 1):
        fig3.update_xaxes(title_text="θ injection [°]", row=1, col=col_idx)
        fig3.update_yaxes(title_text="θ_sun [°]", row=1, col=col_idx)

    fig3.write_html(str(OUT / "wsb_heatmap.html"))
    print(f"Saved {OUT / 'wsb_heatmap.html'}")

# ════════════════════════════════════════════════════════════════════════════════
# Plot 4 — Scatter: capture orbits vs transfer time (crash-annotated, <80d focus)
# ════════════════════════════════════════════════════════════════════════════════

fig4 = go.Figure()

for fam_name, col in FAMILY_COLORS.items():
    sub = fam[fam["family"] == fam_name]
    if sub.empty:
        continue
    normal  = sub[~sub["crashed_moon"]]
    crashed = sub[sub["crashed_moon"]]

    if not normal.empty:
        fig4.add_trace(go.Scatter(
            x=normal["t_transfer_days"], y=normal["est_capture_orbits"],
            mode="markers",
            marker=dict(color=col, size=6, opacity=0.55),
            name=fam_name, legendgroup=fam_name,
            hovertemplate=(
                f"<b>{fam_name}</b><br>"
                "t=%{x:.1f} d  orbits=%{y:.2f}<br>"
                "ΔV=%{customdata[0]:.4f} km/s  θ=%{customdata[1]:.1f}°<extra></extra>"
            ),
            customdata=normal[["dv_kms", "theta_deg"]].values,
        ))

    if not crashed.empty:
        fig4.add_trace(go.Scatter(
            x=crashed["t_transfer_days"], y=crashed["est_capture_orbits"],
            mode="markers",
            marker=dict(color=col, size=8, symbol="x-open",
                        line=dict(color=col, width=2)),
            name=f"{fam_name} ☠", legendgroup=fam_name,
            hovertemplate=(
                f"<b>{fam_name} — MOON CRASH</b><br>"
                "t=%{x:.1f} d  orbits=%{y:.2f}<br>"
                "alt=%{customdata[0]:.0f} km — refine neighbourhood!<extra></extra>"
            ),
            customdata=crashed[["min_alt_km"]].values,
        ))

# 3-orbit minimum line
fig4.add_hline(y=3, line_dash="dash", line_color="rgba(100,100,100,0.6)",
               annotation_text="3-orbit minimum", annotation_position="right")
# 80-day preference line
fig4.add_vline(x=80, line_dash="dash", line_color="rgba(200,50,50,0.5)",
               annotation_text="80 d target", annotation_position="top right")

fig4.update_layout(
    title=dict(
        text=(
            "WSB Search Space — Capture Orbits vs Transfer Time<br>"
            "<sup>✕ = moon-crash (refine neighbourhood for safe solution)  |  "
            "Red dashed = 80-day preference  |  Grey dashed = 3-orbit minimum</sup>"
        ),
        font=dict(size=13),
    ),
    xaxis=dict(title="Transfer time [days]"),
    yaxis=dict(title="Estimated capture orbits"),
    height=580,
    legend=dict(x=1.01, y=1, font=dict(size=9)),
    template="plotly_white",
)
fig4.write_html(str(OUT / "wsb_scatter.html"))
print(f"Saved {OUT / 'wsb_scatter.html'}")

# ── Console summary ────────────────────────────────────────────────────────────
fast = fam[fam["t_transfer_days"] < 80]
print(f"\n-- Search-space summary ----------------------------------")
print(f"  Total solutions   : {len(fam)}")
print(f"  Moon-crash flagged: {fam['crashed_moon'].sum()}")
print(f"  >=3 orbit captures: {(fam['est_capture_orbits'] >= 3).sum()}")
print(f"  <80 day transfers : {len(fast)}  ({(fast['est_capture_orbits'] >= 3).sum()} meet >=3 orbits)")
print(f"  Pareto rank-1     : {(fam['pareto_rank'] == 1).sum()}")
if not fast.empty:
    best_fast = fast.loc[fast["est_capture_orbits"].idxmax()]
    crashed = "[CRASH]" if best_fast['crashed_moon'] else ""
    print(f"  Best fast capture : {best_fast['est_capture_orbits']:.2f} orbits  "
          f"({best_fast['t_transfer_days']:.0f} d)  "
          f"th={best_fast['theta_deg']:.1f}deg  th_sun={best_fast['theta_sun_deg']:.1f}deg  "
          f"{crashed}")