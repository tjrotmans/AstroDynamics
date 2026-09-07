"""
plot_wsb_covariance.py — parameter sensitivity analysis.

Shows how each of the three optimisation parameters (θ, θ_sun, r_apogee)
affects the total ΔV budget, using the combined MC + GA solution set.
Only solutions that actually reached the Hill sphere are shown.

Reads : out/wsb/mc_solutions.csv
        out/wsb/ga_solutions.csv
Saves : out/wsb/wsb_param_sensitivity.html

Run:  python plot/plot_wsb_covariance.py
"""

import pathlib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT    = pathlib.Path(__file__).parent.parent
OUT_DIR = ROOT / "out" / "wsb"

LOI_PENALTY = 1.8   # km/s — solutions above this never entered the Hill sphere

# ── Load GA + MC data ─────────────────────────────────────────────────────────
mc_df = pd.read_csv(OUT_DIR / "mc_solutions.csv") if (OUT_DIR / "mc_solutions.csv").exists() else pd.DataFrame()
ga_df = pd.read_csv(OUT_DIR / "ga_solutions.csv") if (OUT_DIR / "ga_solutions.csv").exists() else pd.DataFrame()

if mc_df.empty and ga_df.empty:
    print("No solution files found. Run wsb_refine and wsb_optimize first.")
    raise SystemExit(1)

if not mc_df.empty: mc_df["source"] = "MC refined"
if not ga_df.empty: ga_df["source"] = "GA global"
df = pd.concat([mc_df, ga_df], ignore_index=True)

# Keep only solutions that actually entered the Hill sphere
df = df[df["dv_loi_kms"] < LOI_PENALTY].copy()
print(f"SOI-reaching solutions: {len(df)}  (MC refined={len(df[df['source']=='MC refined'])}, GA global={len(df[df['source']=='GA global'])})")

if df.empty:
    print("No SOI-reaching solutions found.")
    raise SystemExit(1)

pareto = df.get("pareto_optimal", pd.Series(0, index=df.index)).astype(bool)

SOURCE_SYMBOL = {"MC refined": "circle", "GA global": "diamond"}
SOURCE_COLOR  = {"MC refined": "#1A73E8", "GA global": "#E63946"}


# ── Figure: 2×2 panels ────────────────────────────────────────────────────────
PANELS = [
    (1, 1, "theta_deg",     "θ [°]",          "Injection angle θ  →  ΔV_total"),
    (1, 2, "theta_sun_deg", "θ_sun [°]",       "Sun angle θ_sun  →  ΔV_total"),
    (2, 1, "r_apogee_nd",   "r_apogee [nd]",   "Apogee distance r_apo  →  ΔV_total"),
    (2, 2, "alpha_deg",     "α at apogee [°]", "α at apogee  →  ΔV_total"),
]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[p[4] for p in PANELS],
    vertical_spacing=0.14,
    horizontal_spacing=0.09,
)

for panel_idx, (row, col, xcol, xlabel, _) in enumerate(PANELS):
    if xcol not in df.columns:
        continue
    for src in ["MC refined", "GA global"]:
        sub = df[df["source"] == src]
        if sub.empty:
            continue
        sub_par  = sub[pareto[sub.index]]
        sub_rest = sub[~pareto[sub.index]]

        if not sub_rest.empty:
            fig.add_trace(go.Scatter(
                x=sub_rest[xcol],
                y=sub_rest["dv_total_kms"],
                mode="markers",
                marker=dict(
                    color=sub_rest["t_transfer_days"],
                    colorscale="Viridis",
                    size=7, opacity=0.65,
                    cmin=df["t_transfer_days"].min(),
                    cmax=df["t_transfer_days"].max(),
                    showscale=(panel_idx == 3 and src == "MC refined"),
                    colorbar=dict(title="Transfer<br>time [days]", thickness=12, x=1.02),
                    symbol=SOURCE_SYMBOL[src],
                    line=dict(width=0.5, color="grey"),
                ),
                name=src,
                legendgroup=src,
                showlegend=(panel_idx == 0),
            ), row=row, col=col)

        if not sub_par.empty:
            fig.add_trace(go.Scatter(
                x=sub_par[xcol],
                y=sub_par["dv_total_kms"],
                mode="markers",
                marker=dict(
                    symbol="star", size=13,
                    color=SOURCE_COLOR[src],
                    line=dict(width=1, color="white"),
                ),
                name=f"{src} Pareto",
                legendgroup=f"{src}_pareto",
                showlegend=(panel_idx == 0),
            ), row=row, col=col)

    fig.update_xaxes(title_text=xlabel, row=row, col=col)
    fig.update_yaxes(title_text="ΔV_total [km/s]", row=row, col=col)

n_mc  = len(df[df["source"] == "MC refined"])
n_ga  = len(df[df["source"] == "GA global"])
n_par = int(pareto.sum())

fig.update_layout(
    title=dict(
        text=(
            "WSB Parameter Sensitivity — SOI-reaching solutions only<br>"
            f"<sup>GA global: {n_ga} | MC refined: {n_mc} | {n_par} Pareto-optimal  |  "
            "Stars = Pareto-optimal  |  Colour = transfer time</sup>"
        ),
        font=dict(size=13),
    ),
    height=820,
    template="plotly_white",
    legend=dict(x=1.05, y=1, font=dict(size=9)),
)

out_path = OUT_DIR / "wsb_param_sensitivity.html"
fig.write_html(str(out_path))
print(f"Saved {out_path}")
