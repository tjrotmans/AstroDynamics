"""
MGA-1DSM trajectory visualisation -- Phase 9g-v.

Four panels:
  1. 3D heliocentric view -- all legs as arcs, DSM points, body positions at
     each encounter epoch (ecliptic X–Y plane shown with Z as depth).
  2. Convergence -- Phase 1 (sum-of-DSM ΔVs [m/s], left axis) and Phase 2
     (total ΔV [m/s], right axis) vs generation, two Y-axes clearly labelled.
  3. Per-leg ΔV breakdown -- departure escape burn, each DSM, optional LOI.
  4. Flyby summary table -- intermediate body name, v∞_in, v∞_out, turn angle,
     periapsis altitude for each gravity assist.

Reads (from out/<mission>/):
  mga_best.csv          -- arc sample points: t_days, x_m, y_m, z_m, leg_idx
  mga_convergence.csv   -- generation, best_fitness (Phase 1 = phase<0, Phase 2 = phase≥0)
  mga_params.csv        -- scalar result: dv_total_ms, dv_departure_ms, dv_arrival_ms,
                           leg_0_dv_dsm_ms, leg_1_dv_dsm_ms, ... tof_total_days
  mga_legs.csv          -- per-leg: leg_idx, body_dep, body_arr, t_start_jd,
                           t_end_jd, tof_days, eta, dv_dsm_ms, vinf_in_ms, vinf_out_ms,
                           turn_deg, rp_km

Usage (from MissionPlanner/ directory):
  python plot/plot_mga.py evj_flyby
  python plot/plot_mga.py veega_flyby
"""

import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ────────────────────────────────────────────────────────────────────

mission = sys.argv[1] if len(sys.argv) > 1 else "evj_flyby"
out_dir = Path("out") / mission

BG     = "#0f0f0f"
ACCENT = "#00d4ff"
GRID   = "#222"
LEG_COLORS = [
    "#00d4ff", "#ff6b35", "#7fba00", "#bf5fff",
    "#ffcf00", "#ff4081", "#00e676", "#ff9800",
]

# ── Load data ─────────────────────────────────────────────────────────────────

def load_csv(name: str) -> pd.DataFrame | None:
    p = out_dir / name
    if not p.exists():
        print(f"[warn] {p} not found — skipping")
        return None
    return pd.read_csv(p)

arc   = load_csv("mga_best.csv")
conv  = load_csv("mga_convergence.csv")
params_df = load_csv("mga_params.csv")
legs  = load_csv("mga_legs.csv")

if arc is None:
    print(f"No mga_best.csv in {out_dir} — run: cargo run -p mission_planner optimize config/{mission}.toml")
    sys.exit(1)

# Convert positions to AU for plotting.
AU = 1.495978707e11  # m
arc["x_au"] = arc["x_m"] / AU
arc["y_au"] = arc["y_m"] / AU
arc["z_au"] = arc["z_m"] / AU

# ── Fig layout ────────────────────────────────────────────────────────────────

fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"type": "scatter3d", "colspan": 2}, None],
        [{"type": "scatter"},                  {"type": "table"}],
    ],
    subplot_titles=[
        "Heliocentric trajectory (ecliptic)",
        "ΔV convergence",
        "Flyby summary",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.08,
)

fig.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color="#ccc", size=12),
    title=dict(
        text=f"MGA-1DSM: {mission}",
        font=dict(color=ACCENT, size=18),
        x=0.5,
    ),
    height=900,
    showlegend=True,
    legend=dict(bgcolor="#111", bordercolor="#333", borderwidth=1),
)

# ── Panel 1: 3D trajectory ────────────────────────────────────────────────────

n_legs = int(arc["leg_idx"].max()) + 1
for k in range(n_legs):
    seg = arc[arc["leg_idx"] == k]
    col = LEG_COLORS[k % len(LEG_COLORS)]
    fig.add_trace(go.Scatter3d(
        x=seg["x_au"], y=seg["y_au"], z=seg["z_au"],
        mode="lines",
        line=dict(color=col, width=3),
        name=f"Leg {k}",
        legendgroup=f"leg{k}",
    ), row=1, col=1)

# DSM points from legs csv.
if legs is not None and "x_dsm_m" in legs.columns:
    for _, r in legs.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[r["x_dsm_m"] / AU], y=[r["y_dsm_m"] / AU], z=[r.get("z_dsm_m", 0.0) / AU],
            mode="markers",
            marker=dict(color="white", size=5, symbol="x"),
            name=f"DSM {int(r['leg_idx'])}",
            showlegend=False,
        ), row=1, col=1)

# Sun at origin.
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode="markers",
    marker=dict(color="yellow", size=10, symbol="circle"),
    name="Sun",
), row=1, col=1)

fig.update_scenes(
    dict(
        bgcolor=BG,
        xaxis=dict(title="X [AU]", gridcolor=GRID, color="#aaa"),
        yaxis=dict(title="Y [AU]", gridcolor=GRID, color="#aaa"),
        zaxis=dict(title="Z [AU]", gridcolor=GRID, color="#aaa"),
        aspectmode="data",
    )
)

# ── Panel 2: Convergence ──────────────────────────────────────────────────────

if conv is not None and not conv.empty:
    # Detect phase column; fall back to single-phase if absent.
    has_phase = "phase" in conv.columns
    if has_phase:
        p1 = conv[conv["phase"] < 0].reset_index(drop=True)
        p2 = conv[conv["phase"] >= 0].reset_index(drop=True)
    else:
        p1 = pd.DataFrame()
        p2 = conv.copy().reset_index(drop=True)
    p2["gen_idx"] = range(len(p2))

    if not p1.empty:
        p1["gen_idx"] = range(len(p1))
        fig.add_trace(go.Scatter(
            x=p1["gen_idx"], y=p1["best_fitness"],
            mode="lines", name="Phase 1 DSM ΔV",
            line=dict(color="#ff9800", width=1.5, dash="dot"),
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=p2["gen_idx"], y=p2["best_fitness"],
        mode="lines", name="Phase 2 total ΔV",
        line=dict(color=ACCENT, width=2),
    ), row=2, col=1)

    fig.update_xaxes(title_text="Generation", row=2, col=1,
                     gridcolor=GRID, color="#aaa")
    fig.update_yaxes(title_text="ΔV [m/s]", row=2, col=1,
                     gridcolor=GRID, color="#aaa")

# ── Panel 3: Flyby / ΔV table ─────────────────────────────────────────────────

if legs is not None and not legs.empty:
    cols = ["body_dep", "body_arr", "tof_days", "dv_dsm_ms", "vinf_in_ms", "vinf_out_ms",
            "turn_deg", "rp_km"]
    present = [c for c in cols if c in legs.columns]
    headers = {
        "body_dep":    "From",
        "body_arr":    "To",
        "tof_days":    "TOF [d]",
        "dv_dsm_ms":   "DSM ΔV [m/s]",
        "vinf_in_ms":  "v∞_in [m/s]",
        "vinf_out_ms": "v∞_out [m/s]",
        "turn_deg":    "Turn [°]",
        "rp_km":       "rp [km]",
    }
    header_vals  = [headers.get(c, c) for c in present]
    cell_vals    = []
    for c in present:
        col_data = legs[c]
        if col_data.dtype == float:
            cell_vals.append([f"{v:.2f}" for v in col_data])
        else:
            cell_vals.append(list(col_data.astype(str)))

    fig.add_trace(go.Table(
        header=dict(
            values=header_vals,
            fill_color="#1a1a2e",
            font=dict(color=ACCENT, size=11),
            align="center",
        ),
        cells=dict(
            values=cell_vals,
            fill_color="#0f0f1a",
            font=dict(color="#ccc", size=10),
            align="center",
        ),
    ), row=2, col=2)

# ── Save ──────────────────────────────────────────────────────────────────────

out_html = out_dir / "mga_trajectory.html"
fig.write_html(str(out_html))
print(f"Saved: {out_html}")
webbrowser.open(str(out_html))
