"""
MGA-1DSM v-infinity budget and gravity-assist effectiveness -- Phase 9 backend vis.

Four panels:
  1. v-inf budget: v-inf at each encounter in the sequence -- how gravity
     assists redistribute orbital energy leg by leg. Encounter labels are
     unique per flyby (repeated bodies like Cassini-2's Venus-Venus no longer
     collapse onto one categorical tick).
  2. Delta-V accounting per flyby -- the "free vs bought" picture:
       gained    = 2 * v_inf * sin(delta/2)        (heliocentric impulse the
                                                     turn actually extracted)
       available = 2 * v_inf * sin(delta_max/2)    (at a surface-graze pass)
       DSM paid  = the deep-space burn on the following leg
     A healthy gravity assist has gained close to available and a small DSM;
     a "switched-off" flyby (huge periapsis, tiny turn) shows near-zero
     gained with a large DSM bought propulsively instead.
  3. Turn angle per flyby: achieved vs the maximum possible at surface graze.
     All intermediate flybys are shown, including near-zero turns (the old
     turn_deg > 0.1 filter silently dropped exactly the bars that expose an
     unused flyby).
  4. Leg time-of-flight with the DSM position (eta fraction) inside each leg.

Usage (from MissionPlanner/ directory):
  python plot/plot_mga_vinf.py evj_flyby
  python plot/plot_mga_vinf.py cassini2_gtop

Reads (from out/<mission>/):
  mga_legs.csv      -- per-leg details written by the Rust backend
"""

import sys
import math
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ────────────────────────────────────────────────────────────────────

mission = sys.argv[1] if len(sys.argv) > 1 else "evj_flyby"
out_dir = Path("out") / mission

BG     = "#0f0f0f"
GRID   = "#222"
TEXT   = "#cccccc"
MUTED  = "#888888"
ACCENT = "#00d4ff"

# Series colors validated (dataviz six-checks) against the dark surface:
# blue/orange is CVD-safe (protan/deutan dE ~84); gray is the neutral
# context bar, not a categorical series.
C_GAINED  = "#1f8fc0"   # blue  -- what the flyby / arrival actually delivers
C_PAID    = "#c9661a"   # orange -- what is bought propulsively (DSM, dep burn)
C_CONTEXT = "#555555"   # neutral -- theoretical maximum (context, not a series)

LEG_COLORS = [
    "#00d4ff", "#ff6b35", "#7fba00", "#bf5fff",
    "#ffcf00", "#ff4081", "#00e676", "#ff9800",
]

# ── Body physical constants (for max turn calculation) ────────────────────────
BODY_MU = {
    "Mercury": 2.203e13, "Venus":   3.249e14, "Earth":  3.986e14,
    "Mars":    4.283e13, "Jupiter": 1.267e17, "Saturn": 3.793e16,
    "Uranus":  5.794e15, "Neptune": 6.836e15,
}
BODY_RADIUS_M = {
    "Mercury": 2.440e6, "Venus":  6.052e6, "Earth":  6.371e6,
    "Mars":    3.396e6, "Jupiter": 7.149e7, "Saturn": 6.027e7,
    "Uranus":  2.556e7, "Neptune": 2.476e7,
}


def turn_deg_at(vinf_ms: float, rp_m: float, mu_body: float) -> float:
    """Gravity-assist turn angle [deg] at periapsis rp_m with given v-inf.
    delta = 2*asin(1/e), e = 1 + rp*vinf^2/mu (Battin 6.3)."""
    if vinf_ms <= 0 or rp_m <= 0:
        return 0.0
    e = 1.0 + rp_m * vinf_ms**2 / mu_body
    return math.degrees(2.0 * math.asin(1.0 / e))


def dv_from_turn_ms(vinf_ms: float, turn_deg: float) -> float:
    """Heliocentric delta-V extracted by rotating v-inf through turn_deg:
    |dv| = 2*vinf*sin(delta/2)."""
    return 2.0 * vinf_ms * math.sin(math.radians(turn_deg) / 2.0)


# ── Load data ─────────────────────────────────────────────────────────────────

legs_path = out_dir / "mga_legs.csv"
if not legs_path.exists():
    print(f"[ERROR] {legs_path} not found.")
    print(f"Run: cargo run -p mission_planner --bin mission-planner optimize config/{mission}.toml")
    sys.exit(1)

legs = pd.read_csv(legs_path)
n_legs = len(legs)

# ── Encounter table with UNIQUE labels ────────────────────────────────────────
# Repeated body names (Venus-Venus) must not share a categorical tick.

dep_body = legs.iloc[0]["body_dep"].strip()
encounter_labels = [f"{dep_body} (dep)"]
for i, (_, r) in enumerate(legs.iterrows()):
    body = r["body_arr"].strip()
    tag  = "arr" if i == n_legs - 1 else f"F{i + 1}"
    encounter_labels.append(f"{body} ({tag})")

# Flybys = arrivals of every leg except the last (the final body is the
# target, not a flyby).
flybys = []   # dicts: label, body, vinf_ms, turn_deg, rp_km, rp_norm, dsm_next_ms
for j in range(n_legs - 1):
    r = legs.iloc[j]
    body = r["body_arr"].strip()
    flybys.append(dict(
        label=encounter_labels[j + 1],
        body=body,
        vinf_ms=float(r["vinf_arr_ms"]),
        turn_deg=float(r["turn_deg"]),
        rp_km=float(r["rp_km"]),
        rp_norm=float(r["rp_norm"]),
        dsm_next_ms=float(legs.iloc[j + 1]["dv_dsm_ms"]),
    ))

for fb in flybys:
    mu  = BODY_MU.get(fb["body"], 1e14)
    r_b = BODY_RADIUS_M.get(fb["body"], 6.4e6)
    fb["max_turn_deg"] = turn_deg_at(fb["vinf_ms"], r_b, mu)
    fb["dv_gained_ms"] = dv_from_turn_ms(fb["vinf_ms"], fb["turn_deg"])
    fb["dv_avail_ms"]  = dv_from_turn_ms(fb["vinf_ms"], fb["max_turn_deg"])

# ── Figure ────────────────────────────────────────────────────────────────────

fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=[
        "v∞ at each encounter",
        "ΔV accounting per flyby — gained free vs available vs DSM paid on next leg",
        "Gravity-assist turn angle per flyby — achieved vs max possible (surface graze)",
        "Transfer leg time-of-flight and DSM position η",
    ],
    vertical_spacing=0.09,
    row_heights=[0.24, 0.28, 0.24, 0.24],
)

fig.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=TEXT, family="monospace", size=12),
    title=dict(
        text=f"MGA gravity-assist analysis — {mission}",
        font=dict(color=ACCENT, size=18),
        x=0.5,
    ),
    height=1350,
    showlegend=True,
    legend=dict(bgcolor="#111", bordercolor="#333", borderwidth=1),
    barmode="group",
    bargap=0.30,
    bargroupgap=0.08,
)

# ── Panel 1: v∞ budget ────────────────────────────────────────────────────────
# One bar per encounter, unique tick each. Departure bar is orange (a burn
# the mission pays for); arrivals are blue.

vinf_kms = [float(legs.iloc[0]["vinf_dep_ms"]) / 1000.0]
vinf_kms += [float(r["vinf_arr_ms"]) / 1000.0 for _, r in legs.iterrows()]
vinf_colors = [C_PAID] + [C_GAINED] * n_legs

fig.add_trace(go.Bar(
    x=encounter_labels,
    y=vinf_kms,
    name="v∞ at encounter [km/s]",
    marker_color=vinf_colors,
    text=[f"{v:.2f}" for v in vinf_kms],
    textposition="outside",
    textfont=dict(color=TEXT, size=10),
    hovertemplate="%{x}<br>v∞ = %{y:.2f} km/s<extra></extra>",
    showlegend=False,
), row=1, col=1)

fig.update_xaxes(gridcolor=GRID, color="#aaa", row=1, col=1)
fig.update_yaxes(title_text="v∞ [km/s]", gridcolor=GRID, color="#aaa", row=1, col=1)

# ── Panel 2: ΔV accounting per flyby ─────────────────────────────────────────
# Grouped bars: available (context gray), gained (blue), DSM paid (orange).
# This is the panel that answers "was the flyby doing work, or did we buy
# the velocity change with propellant?"

if flybys:
    fb_x       = [fb["label"] for fb in flybys]
    fb_avail   = [fb["dv_avail_ms"]  / 1000.0 for fb in flybys]
    fb_gained  = [fb["dv_gained_ms"] / 1000.0 for fb in flybys]
    fb_paid    = [fb["dsm_next_ms"]  / 1000.0 for fb in flybys]
    fb_hover   = [
        (f"{fb['label']}<br>rp = {fb['rp_km']:,.0f} km ({fb['rp_norm']:.1f} radii)"
         f"<br>turn = {fb['turn_deg']:.2f}° of {fb['max_turn_deg']:.1f}° max"
         f"<br>ΔV gained = {fb['dv_gained_ms']:,.0f} m/s"
         f"<br>ΔV available = {fb['dv_avail_ms']:,.0f} m/s"
         f"<br>DSM on next leg = {fb['dsm_next_ms']:,.0f} m/s")
        for fb in flybys
    ]

    fig.add_trace(go.Bar(
        x=fb_x, y=fb_avail,
        name="ΔV available at surface graze [km/s]",
        marker_color=C_CONTEXT,
        text=[f"{v:.2f}" for v in fb_avail],
        textposition="outside",
        textfont=dict(color=MUTED, size=10),
        hovertext=fb_hover, hoverinfo="text",
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=fb_x, y=fb_gained,
        name="ΔV gained from turn [km/s]",
        marker_color=C_GAINED,
        text=[f"{v:.2f}" for v in fb_gained],
        textposition="outside",
        textfont=dict(color=TEXT, size=10),
        hovertext=fb_hover, hoverinfo="text",
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=fb_x, y=fb_paid,
        name="DSM paid on next leg [km/s]",
        marker_color=C_PAID,
        text=[f"{v:.2f}" for v in fb_paid],
        textposition="outside",
        textfont=dict(color=TEXT, size=10),
        hovertext=fb_hover, hoverinfo="text",
    ), row=2, col=1)

    total_gained = sum(fb["dv_gained_ms"] for fb in flybys)
    total_dsm    = float(legs["dv_dsm_ms"].sum())
    fig.add_annotation(
        text=(f"totals: gained free {total_gained:,.0f} m/s · "
              f"all DSMs {total_dsm:,.0f} m/s (incl. leg-0 DSM "
              f"{float(legs.iloc[0]['dv_dsm_ms']):,.0f} m/s, no flyby attached)"),
        xref="x2 domain", yref="y2 domain", x=0.99, y=0.98,
        xanchor="right", yanchor="top", showarrow=False,
        font=dict(color=MUTED, size=10),
    )

fig.update_xaxes(gridcolor=GRID, color="#aaa", row=2, col=1)
fig.update_yaxes(title_text="ΔV [km/s]", gridcolor=GRID, color="#aaa", row=2, col=1)

# ── Panel 3: Turn angle per flyby ─────────────────────────────────────────────
# ALL intermediate flybys shown, including near-zero turns.

if flybys:
    fb_turn     = [fb["turn_deg"] for fb in flybys]
    fb_max_turn = [fb["max_turn_deg"] for fb in flybys]
    fb_eff      = [100.0 * t / m if m > 0 else 0.0
                   for t, m in zip(fb_turn, fb_max_turn)]

    fig.add_trace(go.Bar(
        x=fb_x, y=fb_max_turn,
        name="Max possible turn at surface [°]",
        marker_color=C_CONTEXT,
        text=[f"{t:.1f}°" for t in fb_max_turn],
        textposition="outside",
        textfont=dict(color=MUTED, size=10),
        hovertemplate="%{x}<br>max turn = %{y:.1f}°<extra></extra>",
    ), row=3, col=1)

    fig.add_trace(go.Bar(
        x=fb_x, y=fb_turn,
        name="Achieved turn angle [°]",
        marker_color=C_GAINED,
        text=[f"{t:.1f}° (eff {e:.0f}%)" for t, e in zip(fb_turn, fb_eff)],
        textposition="outside",
        textfont=dict(color=TEXT, size=10),
        hovertemplate="%{x}<br>achieved turn = %{y:.2f}°<extra></extra>",
    ), row=3, col=1)

fig.update_xaxes(gridcolor=GRID, color="#aaa", row=3, col=1)
fig.update_yaxes(title_text="Turn angle [°]", gridcolor=GRID, color="#aaa", row=3, col=1)

# ── Panel 4: TOF and DSM timing ───────────────────────────────────────────────
# Stacked manually via `base=` so the global barmode stays "group" (a global
# barmode="stack" was what previously stacked panel 2/3's grouped bars too).

leg_labels = [f"L{i}: {r['body_dep'].strip()}→{r['body_arr'].strip()}"
              for i, (_, r) in enumerate(legs.iterrows())]
tof_days = [float(r["tof_days"]) for _, r in legs.iterrows()]
eta_vals = [float(r["eta"])      for _, r in legs.iterrows()]
dsm_days = [t * e for t, e in zip(tof_days, eta_vals)]
post_dsm = [t * (1 - e) for t, e in zip(tof_days, eta_vals)]
leg_colors = [LEG_COLORS[i % len(LEG_COLORS)] for i in range(n_legs)]

fig.add_trace(go.Bar(
    x=leg_labels, y=dsm_days,
    name="Coast to DSM [days]",
    marker_color=leg_colors,
    text=[f"η={e:.2f}" for e in eta_vals],
    textposition="inside",
    textfont=dict(color="white", size=10),
    offsetgroup="tof",
    hovertemplate="%{x}<br>coast to DSM: %{y:.0f} d<extra></extra>",
), row=4, col=1)

fig.add_trace(go.Bar(
    x=leg_labels, y=post_dsm,
    base=dsm_days,
    name="Lambert arc [days]",
    marker_color=["rgba(255,255,255,0.15)"] * n_legs,
    text=[f"{d:.0f}d" for d in tof_days],
    textposition="outside",
    textfont=dict(color=TEXT, size=10),
    offsetgroup="tof",
    hovertemplate="%{x}<br>post-DSM Lambert arc: %{y:.0f} d<extra></extra>",
), row=4, col=1)

fig.update_xaxes(gridcolor=GRID, color="#aaa", row=4, col=1)
fig.update_yaxes(title_text="Days", gridcolor=GRID, color="#aaa", row=4, col=1)

# ── Save ──────────────────────────────────────────────────────────────────────

out_html = out_dir / "mga_vinf.html"
fig.write_html(str(out_html))
print(f"Saved: {out_html}")
webbrowser.open(str(out_html))
