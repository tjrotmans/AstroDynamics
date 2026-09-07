#!/usr/bin/env python3
"""
Artemis 2 Monte Carlo Uncertainty Propagation Analysis

Reads out/mc_all_solutions.csv (all N runs written by `cargo run --bin mc`).

Produces two HTML dashboards:
  out/mc_uncertainty_dashboard.html  — 9-panel UP analysis
  out/mc_correlation_heatmap.html    — input × output Pearson-r heatmap

Panels in the main dashboard:
  1. Lunar CA altitude histogram + Gaussian fit + ±1σ/2σ lines + target
  2. Lunar CA time histogram (hours from nominal April 6 19:05 UTC)
  3. Earth return altitude histogram + reentry corridor annotation
  4. Earth return time histogram (days after TLI)
  5. Tornado chart — Spearman |ρ| per input vs lunar CA altitude
  6. Pitch × yaw joint scatter coloured by lunar CA altitude + 1σ/2σ ellipses
  7. ΔV execution error vs lunar CA altitude
  8. Burn timing offset vs Earth return time
  9. Success-probability CDF of lunar CA altitude

Run from AstroProbs/Artemis/:
  cargo run --bin mc --release
  python plot/plot_uncertainty.py
"""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
ALL_CSV   = os.path.join(_HERE, "..", "out", "mc_all_solutions.csv")
OUT_DIR   = os.path.join(_HERE, "..", "out")
DASH_HTML = os.path.join(OUT_DIR, "mc_uncertainty_dashboard.html")
HEAT_HTML = os.path.join(OUT_DIR, "mc_correlation_heatmap.html")

# ── Mission constants ─────────────────────────────────────────────────────────
NOMINAL_LUNAR_ALT_KM   = 6_513.0   # target flyby altitude [km]
NOMINAL_LUNAR_CA_DAYS  = 3.80      # nominal CA time T+days (April 6 19:05 from April 2 23:49)
NOMINAL_EARTH_RETURN_DAYS = 9.0    # nominal free-return [days]
REENTRY_MAX_ALT_KM     = 130.0     # above this → skip, below → reentry corridor

# ── Style ─────────────────────────────────────────────────────────────────────
DARK: dict[str, Any] = {
    "template":        "plotly_dark",
    "paper_bgcolor":   "rgb(15,15,25)",
    "plot_bgcolor":    "rgb(15,15,25)",
    "font":            {"color": "white", "size": 11},
    "legend":          {"bgcolor": "rgba(20,20,35,0.8)",
                        "bordercolor": "rgba(255,255,255,0.15)"},
}

COL_INPUTS = [
    "pitch_rad", "yaw_rad", "mass_kg", "isp_s",
    "dv_ms", "burn_time_offset_s", "area_m2", "reflectivity",
]
COL_OUTPUTS = [
    "lunar_alt_km", "lunar_ca_time_days",
    "earth_alt_km", "earth_ca_time_days", "ei_dist_km",
]
LABEL_INPUTS = {
    "pitch_rad":           "Burn pitch [rad]",
    "yaw_rad":             "Burn yaw [rad]",
    "mass_kg":             "TLI mass [kg]",
    "isp_s":               "Isp [s]",
    "dv_ms":               "ΔV [m/s]",
    "burn_time_offset_s":  "Burn timing offset [s]",
    "area_m2":             "SRP area [m²]",
    "reflectivity":        "Reflectivity",
}
LABEL_OUTPUTS = {
    "lunar_alt_km":        "Lunar CA alt [km]",
    "lunar_ca_time_days":  "Lunar CA time [days]",
    "earth_alt_km":        "Earth return alt [km]",
    "earth_ca_time_days":  "Earth return time [days]",
    "ei_dist_km":          "EI distance [km]",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def gaussian_curve(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def confidence_ellipse_xy(
    x: np.ndarray, y: np.ndarray, n_std: float = 1.0, n_points: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ex, ey) for a covariance ellipse at n_std standard deviations."""
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.arctan2(*vecs[:, 0][::-1])
    width, height = 2 * n_std * np.sqrt(vals)
    t = np.linspace(0, 2 * np.pi, n_points)
    ex = width / 2 * np.cos(t)
    ey = height / 2 * np.sin(t)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    ex, ey = R @ np.vstack([ex, ey])
    return ex + x.mean(), ey + y.mean()


def add_histogram_panel(
    fig: go.Figure,
    values: np.ndarray,
    row: int, col: int,
    *,
    title: str,
    xunit: str,
    target: float | None = None,
    target_label: str = "Target",
    vlines: list[tuple[float, str, str]] | None = None,   # (x, color, label)
    nbins: int = 40,
) -> None:
    mu, sigma = values.mean(), values.std()
    counts, edges = np.histogram(values, bins=nbins)
    bin_w = edges[1] - edges[0]
    # Normalise so Gaussian overlay matches histogram count scale
    scale = counts.sum() * bin_w

    fig.add_trace(go.Histogram(
        x=values, nbinsx=nbins,
        marker_color="rgba(100,180,255,0.6)",
        name=title, showlegend=False,
    ), row=row, col=col)

    # Gaussian overlay
    xs = np.linspace(values.min(), values.max(), 300)
    fig.add_trace(go.Scatter(
        x=xs, y=gaussian_curve(xs, mu, sigma) * scale,
        mode="lines", line={"color": "#69F0AE", "width": 2},
        name="Gaussian fit", showlegend=False,
    ), row=row, col=col)

    # ±1σ / ±2σ bands
    for n, alpha in ((1, 0.12), (2, 0.06)):
        fig.add_vrect(
            x0=mu - n * sigma, x1=mu + n * sigma,
            fillcolor=f"rgba(100,180,255,{alpha})", line_width=0,
            row=row, col=col,
        )

    # ±1σ / ±2σ boundary lines
    for n, dash in ((1, "dot"), (2, "dash")):
        for sign in (-1, 1):
            fig.add_vline(
                x=mu + sign * n * sigma,
                line={"color": "rgba(100,180,255,0.5)", "dash": dash, "width": 1},
                row=row, col=col,
            )

    # Target / reference lines
    if target is not None:
        fig.add_vline(
            x=target,
            line={"color": "#FFD700", "dash": "dashdot", "width": 2},
            annotation_text=target_label,
            annotation_font_color="#FFD700",
            row=row, col=col,
        )

    if vlines:
        for x_val, color, label in vlines:
            fig.add_vline(
                x=x_val,
                line={"color": color, "dash": "dash", "width": 1.5},
                annotation_text=label, annotation_font_color=color,
                row=row, col=col,
            )

    fig.update_xaxes(title_text=xunit, row=row, col=col)
    fig.update_yaxes(title_text="Count",  row=row, col=col)


# ── Main dashboard ────────────────────────────────────────────────────────────

def build_dashboard(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            "Lunar CA Altitude",
            "Lunar CA Time (days after TLI)",
            "Earth Return Altitude",
            "Earth Return Time (days after TLI)",
            "Sensitivity — Spearman |ρ| vs Lunar CA Alt",
            "Burn Angle Dispersion vs Lunar CA Alt",
            "ΔV Execution Error vs Lunar CA Alt",
            "Burn Timing Offset vs Earth Return Time",
            "Success Probability — P(lunar alt < x)",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    lunar_alt   = df["lunar_alt_km"].values
    lunar_t     = df["lunar_ca_time_days"].values
    earth_alt   = df["earth_alt_km"].values
    earth_t     = df["earth_ca_time_days"].values
    pitch       = np.degrees(df["pitch_rad"].values)
    yaw         = np.degrees(df["yaw_rad"].values)
    dv          = df["dv_ms"].values
    timing      = df["burn_time_offset_s"].values
    nominal_dv  = dv.mean()  # centre of dispersion = config value

    # ── Panel 1: Lunar CA altitude ────────────────────────────────────────────
    add_histogram_panel(
        fig, lunar_alt, 1, 1,
        title="Lunar CA Altitude",
        xunit="Altitude [km]",
        target=NOMINAL_LUNAR_ALT_KM,
        target_label="Target 6513 km",
    )

    # ── Panel 2: Lunar CA time ────────────────────────────────────────────────
    # Express as hours offset from nominal
    nominal_h = NOMINAL_LUNAR_CA_DAYS * 24.0
    lunar_t_h = lunar_t * 24.0
    add_histogram_panel(
        fig, lunar_t_h, 1, 2,
        title="Lunar CA Time",
        xunit="Hours after TLI",
        target=nominal_h,
        target_label="Nominal",
    )

    # ── Panel 3: Earth return altitude ───────────────────────────────────────
    add_histogram_panel(
        fig, earth_alt, 1, 3,
        title="Earth Return Altitude",
        xunit="Altitude [km]",
        vlines=[(REENTRY_MAX_ALT_KM, "#EF5350", "Reentry limit 130 km"),
                (60.0, "#FF6F00", "Target 60 km")],
    )

    # ── Panel 4: Earth return time ────────────────────────────────────────────
    add_histogram_panel(
        fig, earth_t, 2, 1,
        title="Earth Return Time",
        xunit="Days after TLI",
        target=NOMINAL_EARTH_RETURN_DAYS,
        target_label="Nominal",
    )

    # ── Panel 5: Tornado chart (Spearman |ρ| vs lunar CA alt) ────────────────
    rhos = {}
    for col in COL_INPUTS:
        rho, _ = stats.spearmanr(df[col].values, lunar_alt)
        rhos[LABEL_INPUTS[col]] = rho

    labels = list(rhos.keys())
    values = list(rhos.values())
    # Sort by absolute value descending
    order  = np.argsort(np.abs(values))[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colors = ["#EF5350" if v < 0 else "#69F0AE" for v in values]

    fig.add_trace(go.Bar(
        x=[abs(v) for v in values], y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        showlegend=False,
    ), row=2, col=2)
    fig.update_xaxes(title_text="Spearman |ρ|", range=[0, 1.05], row=2, col=2)

    # ── Panel 6: Pitch × Yaw scatter coloured by lunar CA alt ────────────────
    fig.add_trace(go.Scatter(
        x=pitch, y=yaw,
        mode="markers",
        marker={
            "color":     lunar_alt,
            "colorscale": "Viridis",
            "size":      5,
            "opacity":   0.6,
            "colorbar":  {
                "title": "Lunar alt [km]",
                "x": 1.02, "len": 0.33, "y": 0.17,
                "thickness": 12,
            },
        },
        showlegend=False,
    ), row=2, col=3)

    # 1σ and 2σ ellipses
    for n_std, color, dash in ((1, "#69F0AE", "solid"), (2, "rgba(105,240,174,0.5)", "dash")):
        ex, ey = confidence_ellipse_xy(pitch, yaw, n_std=n_std)
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line={"color": color, "dash": dash, "width": 1.5},
            name=f"{n_std}σ ellipse", showlegend=False,
        ), row=2, col=3)

    # Nominal point
    fig.add_trace(go.Scatter(
        x=[pitch.mean()], y=[yaw.mean()],
        mode="markers",
        marker={"color": "#FFD700", "size": 10, "symbol": "cross"},
        name="Nominal", showlegend=False,
    ), row=2, col=3)
    fig.update_xaxes(title_text="Pitch [°]", row=2, col=3)
    fig.update_yaxes(title_text="Yaw [°]",   row=2, col=3)

    # ── Panel 7: ΔV execution error vs lunar CA alt ───────────────────────────
    dv_err = dv - nominal_dv
    fig.add_trace(go.Scatter(
        x=dv_err, y=lunar_alt,
        mode="markers",
        marker={"color": "rgba(100,180,255,0.5)", "size": 4},
        showlegend=False,
    ), row=3, col=1)

    # Linear regression line
    slope, intercept, r, *_ = stats.linregress(dv_err, lunar_alt)
    xs_dv = np.linspace(dv_err.min(), dv_err.max(), 100)
    fig.add_trace(go.Scatter(
        x=xs_dv, y=slope * xs_dv + intercept,
        mode="lines",
        line={"color": "#FF6F00", "width": 2},
        name=f"Linear fit  r={r:.3f}", showlegend=False,
    ), row=3, col=1)
    fig.add_hline(
        y=NOMINAL_LUNAR_ALT_KM,
        line={"color": "#FFD700", "dash": "dashdot", "width": 1.5},
        row=3, col=1,
    )
    fig.update_xaxes(title_text="ΔV error [m/s]",  row=3, col=1)
    fig.update_yaxes(title_text="Lunar CA alt [km]", row=3, col=1)

    # ── Panel 8: Burn timing offset vs Earth return time ──────────────────────
    fig.add_trace(go.Scatter(
        x=timing, y=earth_t,
        mode="markers",
        marker={"color": "rgba(240,100,180,0.5)", "size": 4},
        showlegend=False,
    ), row=3, col=2)

    slope2, intercept2, r2, *_ = stats.linregress(timing, earth_t)
    xs_t = np.linspace(timing.min(), timing.max(), 100)
    fig.add_trace(go.Scatter(
        x=xs_t, y=slope2 * xs_t + intercept2,
        mode="lines",
        line={"color": "#FF6F00", "width": 2},
        name=f"Linear fit  r={r2:.3f}", showlegend=False,
    ), row=3, col=2)
    fig.update_xaxes(title_text="Burn timing offset [s]",  row=3, col=2)
    fig.update_yaxes(title_text="Earth return time [days]", row=3, col=2)

    # ── Panel 9: CDF of lunar CA altitude ────────────────────────────────────
    sorted_alt = np.sort(lunar_alt)
    cdf        = np.arange(1, len(sorted_alt) + 1) / len(sorted_alt)

    fig.add_trace(go.Scatter(
        x=sorted_alt, y=cdf * 100,
        mode="lines",
        line={"color": "#69F0AE", "width": 2},
        fill="tozeroy", fillcolor="rgba(105,240,174,0.08)",
        showlegend=False,
    ), row=3, col=3)

    # Annotate probability at target altitude
    p_at_target = float(np.interp(NOMINAL_LUNAR_ALT_KM, sorted_alt, cdf) * 100)
    fig.add_vline(
        x=NOMINAL_LUNAR_ALT_KM,
        line={"color": "#FFD700", "dash": "dashdot", "width": 2},
        annotation_text=f"Target  P={p_at_target:.0f}%",
        annotation_font_color="#FFD700",
        row=3, col=3,
    )
    fig.update_xaxes(title_text="Lunar CA altitude [km]",    row=3, col=3)
    fig.update_yaxes(title_text="Cumulative probability [%]", row=3, col=3)

    # ── Layout ────────────────────────────────────────────────────────────────
    n = len(df)
    mu_alt, s_alt = lunar_alt.mean(), lunar_alt.std()
    fig.update_layout(
        **DARK,
        title={
            "text": (f"Artemis 2 — Monte Carlo Uncertainty Propagation  "
                     f"(N={n},  lunar CA: μ={mu_alt:.0f} km  σ={s_alt:.0f} km)"),
            "x": 0.01, "xanchor": "left",
        },
        height=1100,
        margin={"l": 60, "r": 60, "t": 80, "b": 40},
    )

    return fig


# ── Correlation heatmap ───────────────────────────────────────────────────────

def build_heatmap(df: pd.DataFrame) -> go.Figure:
    # Pearson r between every input and every output
    r_matrix = np.zeros((len(COL_INPUTS), len(COL_OUTPUTS)))
    for i, inp in enumerate(COL_INPUTS):
        for j, out in enumerate(COL_OUTPUTS):
            r_matrix[i, j], _ = stats.pearsonr(df[inp].values, df[out].values)

    xlabels = [LABEL_OUTPUTS[c] for c in COL_OUTPUTS]
    ylabels = [LABEL_INPUTS[c]  for c in COL_INPUTS]

    fig = go.Figure(go.Heatmap(
        z=r_matrix,
        x=xlabels,
        y=ylabels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(r_matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar={"title": "Pearson r"},
    ))

    # Highlight cells with |r| > 0.3
    for i in range(r_matrix.shape[0]):
        for j in range(r_matrix.shape[1]):
            if abs(r_matrix[i, j]) > 0.3:
                fig.add_shape(
                    type="rect",
                    x0=j - 0.5, x1=j + 0.5,
                    y0=i - 0.5, y1=i + 0.5,
                    line={"color": "white", "width": 2},
                )

    fig.update_layout(
        **DARK,
        title={"text": "Input × Output Pearson Correlation Matrix", "x": 0.01, "xanchor": "left"},
        xaxis={"side": "bottom"},
        height=500,
        margin={"l": 180, "r": 60, "t": 60, "b": 100},
    )

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(ALL_CSV):
        print(f"Not found: {ALL_CSV}")
        print("Run:  cargo run --bin mc --release")
        return

    df = pd.read_csv(ALL_CSV)
    print(f"Loaded {len(df)} solutions from {ALL_CSV}")

    mu  = df["lunar_alt_km"].mean()
    sig = df["lunar_alt_km"].std()
    print(f"Lunar CA alt:  μ = {mu:.0f} km   σ = {sig:.0f} km")
    print(f"Lunar CA time: μ = {df['lunar_ca_time_days'].mean()*24:.2f} h after TLI   "
          f"σ = {df['lunar_ca_time_days'].std()*60:.1f} min")
    print(f"Earth return:  μ = {df['earth_ca_time_days'].mean():.2f} days   "
          f"σ = {df['earth_ca_time_days'].std()*24:.2f} h")

    os.makedirs(OUT_DIR, exist_ok=True)

    dash = build_dashboard(df)
    dash.write_html(DASH_HTML)
    print(f"Saved dashboard  → {DASH_HTML}")

    heat = build_heatmap(df)
    heat.write_html(HEAT_HTML)
    print(f"Saved heatmap    → {HEAT_HTML}")

    for path in (DASH_HTML, HEAT_HTML):
        webbrowser.open(Path(path).resolve().as_uri())


if __name__ == "__main__":
    main()
