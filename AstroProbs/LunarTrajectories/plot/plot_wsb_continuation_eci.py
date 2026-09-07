"""
plot_wsb_continuation_eci.py — ECI continuation steps with moving Moon.

One panel per lambda step.  Each panel shows:
  - Moon orbiting Earth (dotted circle = full orbit)
  - Moon disc at the moment of closest approach
  - Hill-sphere bubble at that moment
  - Dashed line = "before correction" trajectory (may miss)
  - Solid line  = "after correction" trajectory (should enter SOI)

Reads : out/wsb/continuation_corrected.csv
Saves : out/wsb/wsb_continuation_eci.html
"""
import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wsb_style import (
    MU, L_KM, T_STAR, R_EARTH, R_MOON, R_HILL_KM,
    BG, PAPER_BG, TICK_COLOR, GRID_COLOR, ZERO_COLOR,
    COL_EARTH, COL_MOON, COL_HILL,
)

ROOT = pathlib.Path(__file__).parent.parent
CSV  = ROOT / "out" / "wsb" / "continuation_corrected.csv"
OUT  = ROOT / "out" / "wsb" / "wsb_continuation_eci.html"

LAM_COLORS = {
    0.00: "#00E5FF",
    0.25: "#4FC3F7",
    0.50: "#69FF47",
    0.75: "#FFD166",
    1.00: "#FF8C00",
}

AXIS = dict(
    showgrid=True, gridcolor=GRID_COLOR,
    zeroline=True, zerolinecolor=ZERO_COLOR,
    tickfont=dict(size=8, color=TICK_COLOR),
    title_font=dict(color=TICK_COLOR, size=10),
)


def hex_rgba(h, a):
    h = h.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"rgba({r},{g},{b},{a})"


def disc(cx, cy, r, col, n=120):
    t = np.linspace(0, 2*np.pi, n)
    return cx + r*np.cos(t), cy + r*np.sin(t)


def add_panel(fig, row, col, traj_before, traj_after, lam, col_color, first_col):
    """Draw one λ-step panel."""
    moon_bx = traj_after["moon_x_km"].values
    moon_by = traj_after["moon_y_km"].values
    sc_ax   = traj_after["x_km"].values
    sc_ay   = traj_after["y_km"].values
    r_after = np.sqrt((sc_ax - moon_bx)**2 + (sc_ay - moon_by)**2)
    i_ca    = int(np.argmin(r_after))
    t_ca    = traj_after["t_s"].values[i_ca] / 86400

    # Earth
    ex, ey = disc(0, 0, R_EARTH, COL_EARTH)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
        line=dict(color=COL_EARTH, width=1),
        fill="toself", fillcolor=hex_rgba(COL_EARTH, 0.3),
        showlegend=False, hoverinfo="skip"), row=row, col=col)

    # Moon path through ECI over the propagation (actual track, not just circle)
    fig.add_trace(go.Scatter(
        x=moon_bx, y=moon_by,
        mode="lines",
        line=dict(color=COL_MOON, width=1.0, dash="dot"),
        showlegend=first_col,
        name="Moon path" if first_col else None,
        opacity=0.5, hoverinfo="skip",
    ), row=row, col=col)

    # Moon disc at closest approach
    mc_x, mc_y = moon_bx[i_ca], moon_by[i_ca]
    mdx, mdy = disc(mc_x, mc_y, R_MOON, COL_MOON)
    fig.add_trace(go.Scatter(x=mdx, y=mdy, mode="lines",
        line=dict(color=COL_MOON, width=1.5),
        fill="toself", fillcolor=hex_rgba(COL_MOON, 0.4),
        name=f"Moon @ {t_ca:.1f} d" if first_col else None,
        showlegend=first_col, hoverinfo="skip"), row=row, col=col)

    # Hill sphere at closest approach
    hx, hy = disc(mc_x, mc_y, R_HILL_KM, COL_HILL)
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines",
        line=dict(color=COL_HILL, width=1, dash="dot"),
        name="Hill sphere" if first_col else None,
        showlegend=first_col, hoverinfo="skip"), row=row, col=col)

    # "Before" trajectory (dashed, muted)
    if traj_before is not None:
        fig.add_trace(go.Scatter(
            x=traj_before["x_km"], y=traj_before["y_km"],
            mode="lines",
            line=dict(color=col_color, width=1.2, dash="dash"),
            opacity=0.45,
            name="Before correction" if first_col else None,
            showlegend=first_col,
            hoverinfo="skip",
        ), row=row, col=col)

    # "After" trajectory (solid, bright)
    fig.add_trace(go.Scatter(
        x=sc_ax, y=sc_ay,
        mode="lines",
        line=dict(color=col_color, width=2.2),
        name="After correction" if first_col else None,
        showlegend=first_col,
        hovertemplate=f"λ={lam:.2f}<br>x=%{{x:.0f}} km<br>y=%{{y:.0f}} km<extra></extra>",
    ), row=row, col=col)


def main():
    df = pd.read_csv(CSV)
    lambdas = sorted(df["lambda"].unique())
    n_lam   = len(lambdas)

    # Auto-detect psi_m0 (not needed here — ECI positions are already correct)
    run_id    = int(df["run_id"].iloc[0])

    # Layout: two panels per row (before/after for first lambda, then next lambdas)
    n_cols = 2
    n_rows = int(np.ceil(n_lam / n_cols))

    titles = []
    for lam in lambdas:
        entered_after = False
        sub_a = df[(df["lambda"] == lam) & (df["phase"] == "after")]
        if len(sub_a):
            r_moon = sub_a["r_moon_km"].values
            entered_after = np.any(r_moon < R_HILL_KM)
        status = "✓ enters SOI" if entered_after else "✗ misses SOI"
        titles.append(f"λ={lam:.2f}  {status}")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.07,
        vertical_spacing=0.12,
    )

    for idx, lam in enumerate(lambdas):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        first_col = (idx == 0)
        col_color = LAM_COLORS.get(lam, "#aaaaaa")

        sub_b = df[(df["lambda"] == lam) & (df["phase"] == "before")]
        sub_a = df[(df["lambda"] == lam) & (df["phase"] == "after")]

        if len(sub_a) == 0:
            continue

        # Only show "before" when it differs from "after" (when correction happened)
        before_differs = (
            len(sub_b) > 0 and
            abs(sub_b["theta_deg"].iloc[0] - sub_a["theta_deg"].iloc[0]) > 1e-6
        )
        traj_before = sub_b if before_differs else None

        add_panel(fig, row, col, traj_before, sub_a, lam, col_color, first_col)

        # Zoom to Moon-orbit region — fixed range so all panels are comparable.
        # Moon orbits at ~384,400 km; add 20% margin so the orbit arc is visible.
        moon_r = np.sqrt(sub_a["moon_x_km"].iloc[0]**2 + sub_a["moon_y_km"].iloc[0]**2)
        v = moon_r * 1.20

        xax      = f"xaxis{'' if idx == 0 else idx+1}"
        yax_key  = f"yaxis{'' if idx == 0 else idx+1}"
        yax_ref  = f"y{'' if idx == 0 else idx+1}"
        fig.layout[xax].update(
            title_text="x ECI [km]", range=[-v, v],
            scaleanchor=yax_ref, scaleratio=1, **AXIS)
        fig.layout[yax_key].update(
            title_text="y ECI [km]", range=[-v, v], **AXIS)

    for ann in fig.layout.annotations:
        ann.update(font=dict(color="#ccccdd", size=10))

    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=BG,
        font=dict(color="white", family="monospace"),
        height=360 * n_rows,
        title=dict(
            text=(
                f"Step 2: ECI continuation — run {run_id}  ·  "
                "Dashed = before shooting  ·  Solid = corrected IC"
            ),
            font=dict(size=13, color="white"), x=0.5,
        ),
        legend=dict(
            bgcolor="rgba(12,12,28,0.88)",
            bordercolor="rgba(100,100,180,0.35)", borderwidth=1,
            font=dict(size=10), x=1.01, y=0.9,
        ),
        margin=dict(l=55, r=140, t=65, b=50),
    )

    fig.write_html(str(OUT), include_plotlyjs="cdn")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
