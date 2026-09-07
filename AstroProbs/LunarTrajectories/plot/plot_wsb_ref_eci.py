"""
plot_wsb_ref_eci.py — BCR4BP reference trajectory converted to ECI J2000.

Shows the original WSB solution in the inertial frame:
  - Earth fixed at origin
  - Moon orbiting in a circle (L_KM radius)
  - Spacecraft trajectory coloured by elapsed days
  - Hill-sphere bubble shown at the moment of closest Moon approach

Reads : out/wsb/dense_traj.csv
Saves : out/wsb/wsb_ref_eci.html
"""
import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wsb_style import (
    MU, X_M, L_KM, T_STAR, R_EARTH, R_MOON, R_HILL_KM,
    BG, PAPER_BG, TICK_COLOR, GRID_COLOR, ZERO_COLOR,
    COL_EARTH, COL_MOON, COL_HILL, MULTI_COLORS,
)

ROOT      = pathlib.Path(__file__).parent.parent
DENSE_CSV = ROOT / "out" / "wsb" / "dense_traj.csv"
OUT       = ROOT / "out" / "wsb" / "wsb_ref_eci.html"

# Which run to plot (first unique run_id found if None)
RUN_ID = None

AXIS = dict(
    showgrid=True, gridcolor=GRID_COLOR,
    zeroline=True,  zerolinecolor=ZERO_COLOR,
    tickfont=dict(size=9, color=TICK_COLOR),
    title_font=dict(color=TICK_COLOR, size=11),
)


def hex_rgba(h, a):
    h = h.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"rgba({r},{g},{b},{a})"


def circle(cx, cy, r, n=200):
    t = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def bcr4bp_to_eci(x_nd, y_nd, t_nd):
    """BCR4BP barycenter-centred rotating frame → ECI [km]."""
    x_ec = x_nd + MU            # Earth-centred [nd]
    y_ec = y_nd
    x_eci = (np.cos(t_nd) * x_ec - np.sin(t_nd) * y_ec) * L_KM
    y_eci = (np.sin(t_nd) * x_ec + np.cos(t_nd) * y_ec) * L_KM
    return x_eci, y_eci


def main():
    df = pd.read_csv(DENSE_CSV)

    run_id = RUN_ID or int(df["run_id"].iloc[0])
    sub = df[(df["run_id"] == run_id) & df["time_nd"].notna()].copy()
    sub = sub.dropna(subset=["x_nd"])

    meta = sub.iloc[0]
    theta_deg  = meta["theta_deg"]
    t_sun_deg  = meta["theta_sun_deg"]
    r_apo_nd   = meta["r_apogee_nd"]
    n_orbits   = meta["est_capture_orbits"]

    t_nd = sub["time_nd"].values
    x_nd = sub["x_nd"].values
    y_nd = sub["y_nd"].values

    x_eci, y_eci = bcr4bp_to_eci(x_nd, y_nd, t_nd)

    # Moon position in ECI at each step (BCR4BP: Moon on circle of radius L_KM from Earth)
    moon_x = np.cos(t_nd) * L_KM
    moon_y = np.sin(t_nd) * L_KM

    # Distance to Moon
    r_moon = np.sqrt((x_eci - moon_x)**2 + (y_eci - moon_y)**2)
    i_close = int(np.argmin(r_moon))

    t_days = t_nd * T_STAR / 86400

    # ── Subplots: full ECI view  +  Hill-sphere zoom ──────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.60, 0.40],
        subplot_titles=[
            f"BCR4BP trajectory in ECI — run {run_id}  "
            f"(θ={theta_deg:.1f}°, θ_sun={t_sun_deg:.1f}°, "
            f"r_apo={r_apo_nd:.1f}, ~{n_orbits:.1f} orbits)",
            "Moon Hill-sphere zoom",
        ],
        horizontal_spacing=0.08,
    )

    # ── Bodies ────────────────────────────────────────────────────────────────
    # Full view
    ex, ey = circle(0, 0, R_EARTH)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
        line=dict(color=COL_EARTH, width=1.5),
        fill="toself", fillcolor=hex_rgba(COL_EARTH, 0.25),
        name="Earth", showlegend=False, hoverinfo="skip"), row=1, col=1)

    # Moon orbit circle
    mx_orb, my_orb = circle(0, 0, L_KM)
    fig.add_trace(go.Scatter(x=mx_orb, y=my_orb, mode="lines",
        line=dict(color=COL_MOON, width=0.8, dash="dot"),
        name="Moon orbit", showlegend=True, opacity=0.4, hoverinfo="skip"), row=1, col=1)

    # Moon disc at closest approach time
    mc_x, mc_y = moon_x[i_close], moon_y[i_close]
    mx_c, my_c = circle(mc_x, mc_y, R_MOON)
    fig.add_trace(go.Scatter(x=mx_c, y=my_c, mode="lines",
        line=dict(color=COL_MOON, width=1.5),
        fill="toself", fillcolor=hex_rgba(COL_MOON, 0.35),
        name=f"Moon @ t={t_days[i_close]:.1f} d", showlegend=True,
        hoverinfo="skip"), row=1, col=1)

    # Hill sphere at closest approach
    hx, hy = circle(mc_x, mc_y, R_HILL_KM)
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines",
        line=dict(color=COL_HILL, width=1, dash="dot"),
        name="Hill sphere", showlegend=True, hoverinfo="skip"), row=1, col=1)

    # Trajectory coloured by time — use markers (line.color doesn't accept arrays)
    fig.add_trace(go.Scatter(
        x=x_eci, y=y_eci, mode="markers",
        marker=dict(
            color=t_days,
            colorscale="Plasma",
            size=2,
            colorbar=dict(
                title=dict(text="Days", font=dict(color=TICK_COLOR, size=10)),
                tickfont=dict(color=TICK_COLOR, size=9),
                len=0.5, y=0.5, x=0.58,
                thickness=12,
            ),
        ),
        name="Spacecraft (BCR4BP→ECI)",
        hovertemplate="t=%{text:.1f} d<br>x=%{x:.0f} km<br>y=%{y:.0f} km<extra></extra>",
        text=t_days,
    ), row=1, col=1)

    # ── Hill-sphere zoom (Moon-centred) ───────────────────────────────────────
    # Show a window around closest approach
    i_enter = np.where(r_moon < R_HILL_KM * 1.5)[0]
    if len(i_enter):
        i0, i1 = max(0, i_enter[0] - 20), min(len(x_eci)-1, i_enter[-1] + 20)
    else:
        w = max(50, len(x_eci)//10)
        i0, i1 = max(0, i_close - w), min(len(x_eci)-1, i_close + w)

    # Moon positions in zoom window (Moon-centred coords)
    mx_z = moon_x[i0:i1] - moon_x[i0:i1]   # Moon at origin in Moon-centred
    my_z = moon_y[i0:i1] - moon_y[i0:i1]

    # Spacecraft in Moon-centred coords
    sc_xz = x_eci[i0:i1] - moon_x[i0:i1]
    sc_yz = y_eci[i0:i1] - moon_y[i0:i1]

    # Moon disc (centred at origin)
    zm_x, zm_y = circle(0, 0, R_MOON)
    fig.add_trace(go.Scatter(x=zm_x, y=zm_y, mode="lines",
        line=dict(color=COL_MOON, width=1.5),
        fill="toself", fillcolor=hex_rgba(COL_MOON, 0.35),
        name="Moon", showlegend=False, hoverinfo="skip"), row=1, col=2)

    zh_x, zh_y = circle(0, 0, R_HILL_KM)
    fig.add_trace(go.Scatter(x=zh_x, y=zh_y, mode="lines",
        line=dict(color=COL_HILL, width=1, dash="dot"),
        name="Hill sphere", showlegend=False, hoverinfo="skip"), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=sc_xz, y=sc_yz, mode="markers",
        marker=dict(color=t_days[i0:i1], colorscale="Plasma", size=3),
        showlegend=False,
        hovertemplate="t=%{text:.1f} d<extra></extra>",
        text=t_days[i0:i1],
    ), row=1, col=2)

    # ── Axes ──────────────────────────────────────────────────────────────────
    v = max(abs(x_eci).max(), abs(y_eci).max()) * 1.05
    fig.update_xaxes(title_text="x ECI [km]", range=[-v, v],
                     scaleanchor="y", scaleratio=1, **AXIS, row=1, col=1)
    fig.update_yaxes(title_text="y ECI [km]", range=[-v, v], **AXIS, row=1, col=1)

    zoom = R_HILL_KM * 1.3
    fig.update_xaxes(title_text="x − Moon [km]", range=[-zoom, zoom],
                     scaleanchor="y2", scaleratio=1, **AXIS, row=1, col=2)
    fig.update_yaxes(title_text="y − Moon [km]", range=[-zoom, zoom], **AXIS, row=1, col=2)

    for ann in fig.layout.annotations:
        ann.update(font=dict(color="#ccccdd", size=11))

    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=BG,
        font=dict(color="white", family="monospace"),
        height=620,
        title=dict(
            text="Step 1: BCR4BP reference in ECI  ·  Moon orbits Earth, spacecraft enters SOI",
            font=dict(size=13, color="white"), x=0.5,
        ),
        legend=dict(
            bgcolor="rgba(12,12,28,0.88)",
            bordercolor="rgba(100,100,180,0.35)", borderwidth=1,
            font=dict(size=10), x=0.01, y=0.99,
        ),
        margin=dict(l=60, r=40, t=65, b=50),
    )

    fig.write_html(str(OUT), include_plotlyjs="cdn")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
