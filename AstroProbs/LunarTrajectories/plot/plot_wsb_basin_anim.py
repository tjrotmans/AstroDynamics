"""
plot_wsb_basin_anim.py — capture basin scatter map.

Supports two data sources, selected via --source flag or auto-detected:

  basin   (default if basin_sweep.csv exists)
      out/wsb/basin_sweep.csv — uniform grid from wsb_basin.
      Outcome column written directly by Rust — no reclassification needed.
      Columns: r_apogee_nd, theta_deg, theta_sun_deg, outcome,
               est_capture_orbits, min_loi_kms

  family  (fallback, or force with --source family)
      out/wsb/family_analysis.csv — Phase 3/4 envelope from wsb_search.
      Outcome reconstructed from crashed_moon + est_capture_orbits.
      Capturable reclassified via proximity to mc_solutions.csv.
      Note: coverage is non-uniform (envelope patches around found islands).

Optional overlay (both sources):
      out/wsb/mc_solutions.csv — gold star markers for best refined solutions.

Usage:
  python plot/plot_wsb_basin_anim.py                  # auto-detect source
  python plot/plot_wsb_basin_anim.py --source basin   # force basin_sweep.csv
  python plot/plot_wsb_basin_anim.py --source family  # force family_analysis.csv

Saves: out/wsb/wsb_basin_anim.html
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────────────────────

MIN_CAPTURE_ORBITS = 0.1    # for family source outcome reconstruction
LOI_CAP_KMS        = 0.4    # km/s — capturable threshold (both sources)
MATCH_DEG          = 1.0    # deg — proximity window for family capturable

ALPHA_BANDS = [(20, 100), (200, 290)]   # θ_sun ranges for Q2/Q4 α windows

OUTCOME_ORDER = ["escaped", "moon_crash", "capturable", "captured"]
OUTCOME_COLOR = {
    "captured":   "#1A73E8",
    "capturable": "#FF9800",
    "moon_crash": "#E63946",
    "escaped":    "#AAAAAA",
}
OUTCOME_LABEL = {
    "captured":   "Captured (ballistic)",
    "capturable": f"Capturable (LOI \u2264 {LOI_CAP_KMS} km/s)",
    "moon_crash": "Moon crash",
    "escaped":    "Escaped",
}
MARKER_OPACITY = {
    "escaped":    0.20,
    "moon_crash": 0.75,
    "capturable": 0.88,
    "captured":   0.92,
}
MARKER_SIZE      = 4
N_DATA_SLOTS     = 5       # escaped, crash, capturable, captured, mc-stars
MAX_ESCAPED_PTS  = 6_000   # cap escaped points per slice — they're just background noise

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT       = pathlib.Path(__file__).parent.parent
OUT_DIR    = ROOT / "out" / "wsb"
BASIN_CSV  = OUT_DIR / "basin_sweep.csv"
FAMILY_CSV = OUT_DIR / "family_analysis.csv"
MC_CSV     = OUT_DIR / "mc_solutions.csv"

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--source", choices=["basin", "family"], default=None,
    help="Data source: 'basin' = basin_sweep.csv, 'family' = family_analysis.csv. "
         "Auto-detected if omitted (basin preferred).",
)
args = parser.parse_args()

if args.source is None:
    args.source = "basin" if BASIN_CSV.exists() else "family"
    print(f"  Auto-detected source: {args.source}")

# ── Load & normalise data ─────────────────────────────────────────────────────

mc = pd.read_csv(MC_CSV) if MC_CSV.exists() else None
if mc is not None:
    print(f"Loaded {len(mc)} MC solutions from {MC_CSV.name}")

if args.source == "basin":
    # ── Basin sweep — outcomes from Rust, no reclassification needed ──────────
    if not BASIN_CSV.exists():
        print(f"Not found: {BASIN_CSV}")
        print("Run: cargo run -p lunar_trajectories --bin wsb_basin --release")
        print("Or use: --source family")
        raise SystemExit(1)

    df = pd.read_csv(BASIN_CSV)
    print(f"Loaded {len(df)} rows from {BASIN_CSV.name}  [source=basin]")

    # Normalise any unrecognised labels
    df.loc[~df["outcome"].isin(OUTCOME_ORDER), "outcome"] = "escaped"

    # Use r_apogee_nd directly as the animation axis
    apo_col = "r_apogee_nd"
    source_label = "basin_sweep.csv (uniform grid)"

else:
    # ── Family analysis — reconstruct outcomes from Rust columns ──────────────
    if not FAMILY_CSV.exists():
        print(f"Not found: {FAMILY_CSV}")
        print("Run: cargo run -p lunar_trajectories --bin wsb_search --release")
        raise SystemExit(1)

    df = pd.read_csv(FAMILY_CSV)
    print(f"Loaded {len(df)} rows from {FAMILY_CSV.name}  [source=family]")

    # Normalise apogee column name
    if "r_apogee_target_nd" in df.columns:
        df = df.rename(columns={"r_apogee_target_nd": "r_apogee_nd"})
    elif "apogee_nd" in df.columns:
        df = df.rename(columns={"apogee_nd": "r_apogee_nd"})

    # Reconstruct outcome — fully vectorised, no row-by-row apply
    crashed  = df.get("crashed_moon", pd.Series(0, index=df.index)).astype(bool)
    captured = (~crashed) & (df["est_capture_orbits"] >= MIN_CAPTURE_ORBITS)
    df["outcome"] = "escaped"
    df.loc[captured, "outcome"] = "captured"
    df.loc[crashed,  "outcome"] = "moon_crash"
    print(f"  Before capturable reclassification: {dict(df['outcome'].value_counts())}")

    # Reclassify capturable via proximity to MC solutions with viable LOI.
    # Done per-apogee-slice to avoid an N_family × N_mc broadcast that can
    # be huge if family_analysis has many apogee values.
    if mc is not None:
        mc_cap = mc[mc["dv_loi_kms"] < LOI_CAP_KMS]
        if len(mc_cap):
            n_reclass = 0
            for apo in df["r_apogee_nd"].unique():
                fam_mask = df["r_apogee_nd"].round(3) == round(apo, 3)
                mc_mask  = (mc_cap["r_apogee_nd"] - apo).abs() < 0.15
                mc_here  = mc_cap[mc_mask]
                if mc_here.empty:
                    continue
                sub = df[fam_mask]
                dth = np.abs(sub["theta_deg"].values[:, None]
                             - mc_here["theta_deg"].values[None, :])
                dth = np.minimum(dth, 360 - dth)
                dsn = np.abs(sub["theta_sun_deg"].values[:, None]
                             - mc_here["theta_sun_deg"].values[None, :])
                dsn = np.minimum(dsn, 360 - dsn)
                near = ((dth < MATCH_DEG) & (dsn < MATCH_DEG)).any(axis=1)
                reclass = fam_mask & fam_mask  # same-shape mask scaffolding
                reclass = pd.Series(False, index=df.index)
                reclass[sub.index[near]] = True
                reclass &= df["outcome"].isin(["escaped", "moon_crash"])
                df.loc[reclass, "outcome"] = "capturable"
                n_reclass += reclass.sum()
            print(f"  Reclassified {n_reclass} rows -> capturable (proximity to MC)")

    apo_col = "r_apogee_nd"
    source_label = "family_analysis.csv (envelope patches)"

print(f"  Outcomes: {dict(df['outcome'].value_counts())}")
apo_values = sorted(df[apo_col].unique())
print(f"  r_apogee slices ({len(apo_values)}): {[f'{v:.2f}' for v in apo_values]}")

# Pre-split into per-(apogee, outcome) point lists once — O(N) total,
# avoids repeated boolean filtering inside the frame-building loop.
print("  Pre-splitting data by apogee slice…")
_apo_key = df[apo_col].round(3)
_slices: dict = {}   # apo_key -> {outcome -> (x_list, y_list)}
for apo in apo_values:
    key = round(apo, 3)
    sub = df[_apo_key == key]
    _slices[key] = {
        outcome: (
            sub.loc[sub["outcome"] == outcome, "theta_deg"].tolist(),
            sub.loc[sub["outcome"] == outcome, "theta_sun_deg"].tolist(),
        )
        for outcome in OUTCOME_ORDER
    }
    _slices[key]["_sub"] = sub   # keep for title counts

# ── Static layout elements ────────────────────────────────────────────────────

alpha_shapes = [
    dict(type="rect", xref="x", yref="y",
         x0=-5, x1=365, y0=lo, y1=hi,
         fillcolor="rgba(26,115,232,0.07)", line=dict(width=0), layer="below")
    for lo, hi in ALPHA_BANDS
]

# ── Trace builders ────────────────────────────────────────────────────────────

def make_data_traces(apo: float) -> list:
    """Exactly N_DATA_SLOTS traces, showlegend=False.
    Uses pre-split _slices cache — no dataframe filtering at call time."""
    key    = round(apo, 3)
    cache  = _slices[key]
    traces = []
    rng    = np.random.default_rng(0)
    for outcome in OUTCOME_ORDER:      # slots 0-3
        x, y = cache[outcome]
        if outcome == "escaped" and len(x) > MAX_ESCAPED_PTS:
            idx = rng.choice(len(x), MAX_ESCAPED_PTS, replace=False)
            x = [x[i] for i in idx]
            y = [y[i] for i in idx]
        traces.append(go.Scattergl(
            x=x, y=y,
            mode="markers",
            marker=dict(
                size=MARKER_SIZE,
                color=OUTCOME_COLOR[outcome],
                opacity=MARKER_OPACITY[outcome],
            ),
            showlegend=False,
        ))

    # Slot 4: MC refined stars
    mc_x, mc_y = [], []
    if mc is not None:
        mc_sub = mc[(mc["r_apogee_nd"] - apo).abs() < 0.15]
        mc_x   = mc_sub["theta_deg"].tolist()
        mc_y   = mc_sub["theta_sun_deg"].tolist()
    traces.append(go.Scattergl(
        x=mc_x, y=mc_y,
        mode="markers",
        marker=dict(symbol="star", size=14,
                    color="#FFD600", line=dict(color="#333333", width=0.8)),
        showlegend=False,
    ))

    assert len(traces) == N_DATA_SLOTS
    return traces


def make_legend_traces() -> list:
    """Invisible dummy traces that own legend entries permanently.
    Appended after data traces; never referenced by frame updates."""
    dummies = []
    for outcome in OUTCOME_ORDER:
        dummies.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=9, color=OUTCOME_COLOR[outcome], opacity=0.9),
            name=OUTCOME_LABEL[outcome],
            legendgroup=outcome,
            showlegend=True,
        ))
    if mc is not None:
        dummies.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="star", size=12, color="#FFD600",
                        line=dict(color="#333333", width=0.8)),
            name="MC refined (best solutions)",
            legendgroup="mc",
            showlegend=True,
        ))
    return dummies


def frame_title(apo: float) -> str:
    cache  = _slices[round(apo, 3)]
    n_cap  = len(cache["captured"][0])
    n_cap2 = len(cache["capturable"][0])
    n_cr   = len(cache["moon_crash"][0])
    n_esc  = len(cache["escaped"][0])
    return (
        f"WSB Capture Basin \u2014 r_apogee = {apo:.2f} nd "
        f"({apo * 384_400:.0f} km)  [{args.source}]<br>"
        f"<sup>Captured: {n_cap}  |  Capturable: {n_cap2}  |  "
        f"Crash: {n_cr}  |  Escaped: {n_esc}  |  "
        f"Blue bands = \u03b1 Q2/Q4 windows</sup>"
    )


# ── Build frames ──────────────────────────────────────────────────────────────

frames       = []
slider_steps = []

for apo in apo_values:
    frames.append(go.Frame(
        data=make_data_traces(apo),
        traces=list(range(N_DATA_SLOTS)),
        name=f"{apo:.2f}",
        layout=go.Layout(title_text=frame_title(apo)),
    ))
    slider_steps.append(dict(
        args=[[f"{apo:.2f}"], {"frame": {"duration": 0}, "mode": "immediate"}],
        label=f"{apo:.2f} nd",
        method="animate",
    ))

# ── Initial figure ────────────────────────────────────────────────────────────

init_traces = make_data_traces(apo_values[0]) + make_legend_traces()

fig = go.Figure(data=init_traces, frames=frames)

fig.update_layout(
    title=dict(text=frame_title(apo_values[0]), font=dict(size=13)),
    xaxis=dict(
        title="\u03b8_inject [deg]",
        range=[-5, 365],
        tickvals=list(range(0, 361, 60)),
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
    ),
    yaxis=dict(
        title="\u03b8_sun [deg]",
        range=[-5, 365],
        tickvals=list(range(0, 361, 60)),
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
    ),
    shapes=alpha_shapes,
    legend=dict(
        x=0.01, y=0.99,
        xanchor="left", yanchor="top",
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="rgba(0,0,0,0.20)",
        borderwidth=1,
    ),
    height=700,
    template="plotly_white",
    plot_bgcolor="rgba(245,244,240,1)",
    updatemenus=[dict(
        type="buttons", showactive=False,
        y=1.12, x=0.0, xanchor="left",
        buttons=[
            dict(label="\u25b6 Play", method="animate",
                 args=[None, {"frame": {"duration": 700}, "fromcurrent": True,
                              "transition": {"duration": 150}}]),
            dict(label="\u23f8 Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0}, "mode": "immediate",
                                "transition": {"duration": 0}}]),
        ],
    )],
    sliders=[dict(
        active=0, steps=slider_steps, y=0,
        currentvalue=dict(prefix="r_apogee: ", font=dict(size=13)),
        pad=dict(t=50),
    )] if len(apo_values) > 1 else [],
)

out_path = OUT_DIR / "wsb_basin_anim.html"
fig.write_html(str(out_path), include_plotlyjs="cdn")
print(f"Saved {out_path}  [source={args.source}]")

# ── PNG export ────────────────────────────────────────────────────────────────
# Requires: pip install kaleido
# Exports the apogee slice with the most captures (or --png-apo override).

import sys

png_apo_arg = None
for i, a in enumerate(sys.argv):
    if a == "--png-apo" and i + 1 < len(sys.argv):
        try:
            png_apo_arg = float(sys.argv[i + 1])
        except ValueError:
            pass

# Pick best slice for PNG
if png_apo_arg is not None:
    png_apo = min(apo_values, key=lambda v: abs(v - png_apo_arg))
    print(f"PNG slice: {png_apo:.2f} nd  (requested {png_apo_arg:.2f})")
else:
    # Slice with the most captures — use pre-split cache
    best_apo = max(apo_values,
                   key=lambda a: len(_slices[round(a, 3)]["captured"][0]))
    best_n   = len(_slices[round(best_apo, 3)]["captured"][0])
    png_apo  = best_apo
    print(f"PNG slice: {png_apo:.2f} nd  ({best_n} captures — auto-selected)")

# Build a clean static figure for the PNG (no animation controls)
png_traces  = make_data_traces(png_apo) + make_legend_traces()
fig_png     = go.Figure(data=png_traces)

fig_png.update_layout(
    title=dict(text=frame_title(png_apo), font=dict(size=13)),
    xaxis=dict(
        title="\u03b8_inject [deg]",
        range=[-5, 365],
        tickvals=list(range(0, 361, 60)),
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
    ),
    yaxis=dict(
        title="\u03b8_sun [deg]",
        range=[-5, 365],
        tickvals=list(range(0, 361, 60)),
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
    ),
    shapes=alpha_shapes,
    legend=dict(
        x=0.01, y=0.99,
        xanchor="left", yanchor="top",
        font=dict(size=13),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(0,0,0,0.20)",
        borderwidth=1,
    ),
    width=1400,
    height=900,
    template="plotly_white",
    plot_bgcolor="rgba(245,244,240,1)",
    margin=dict(l=70, r=40, t=80, b=60),
)

png_path = OUT_DIR / f"wsb_basin_{args.source}_{png_apo:.2f}nd.png"
try:
    fig_png.write_image(str(png_path), scale=2)
    print(f"Saved {png_path}  (2× scale)")
except Exception as e:
    print(f"PNG export failed: {e}")
    print("Install kaleido with: pip install kaleido")