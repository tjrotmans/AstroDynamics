"""
3D Tisserand graph -- Phase 9 trajectory optimisation visualisation.

Full Tisserand relation (Strange & Longuski 2002):

    T_P = a_P/a + 2 * sqrt(a/a_P * (1 - e^2)) * cos(i)

For each body in the sequence (departure, flyby, arrival), this script
draws Tisserand surfaces at several v_inf values.  Each surface is a 2-D
manifold in (Rp, Ra, inclination) space.

Looking straight down the inclination axis (top view) gives you the
classical 2-D Tisserand graph -- the floor lines at i=0 are exactly the
2-D contours.  The trajectory orbit-state dots float above the floor at
their actual inclinations; both pre- and post-flyby dots lie on the
SAME Venus surface (Tisserand conservation).

Axes:
    X = Rp  [AU]   perihelion distance
    Y = Ra  [AU]   aphelion distance
    Z = i   [deg]  ecliptic inclination  (0 = prograde in ecliptic plane)

Usage (from MissionPlanner/ directory):
    python plot/plot_tisserand_3d.py evj_2033
    python plot/plot_tisserand_3d.py evj_2033 --i-max 60

Reads:  out/<mission>/mga_legs.csv
Output: out/<mission>/tisserand_3d.html
"""

import sys
import math
import argparse
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Constants ─────────────────────────────────────────────────────────────────

MU_SUN = 1.327_124_400_18e20   # m^3/s^2  (IAU 2012)
AU     = 1.495_978_707e11       # m

PLANET_SMA_AU = {
    "Mercury": 0.38710, "Venus": 0.72333, "Earth": 1.00000,
    "Mars":    1.52366, "Jupiter": 5.20336, "Saturn": 9.53707,
    "Uranus":  19.1914, "Neptune": 30.0690,
}
PLANET_COLOR = {
    "Mercury": "#aaaaaa", "Venus": "#e8c97e", "Earth": "#4fc3f7",
    "Mars":    "#ef5350",  "Jupiter": "#ffb74d", "Saturn": "#f9a825",
    "Uranus":  "#80cbc4",  "Neptune": "#1565c0",
}
LEG_COLORS = [
    "#ff4081", "#00e676", "#ffcf00", "#bf5fff",
    "#ff6b35", "#00d4ff", "#7fba00", "#ffffff",
]

BG   = "#0f0f0f"
GRID = "#222222"
TEXT = "#cccccc"

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="3D Tisserand graph")
parser.add_argument("mission", nargs="?", default="evj_flyby")
parser.add_argument("--legs-csv",  default=None)
parser.add_argument("--i-max",    type=float, default=70.0,
                    help="Max inclination on z-axis (default 70 deg)")
parser.add_argument("--n-surf",   type=int,   default=60,
                    help="Grid resolution per surface (default 60)")
parser.add_argument("--no-browser", action="store_true")
args = parser.parse_args()

mission  = args.mission
out_dir  = Path("out") / mission
legs_csv = Path(args.legs_csv) if args.legs_csv else out_dir / "mga_legs.csv"
I_MAX    = args.i_max
N_SURF   = args.n_surf

if not legs_csv.exists():
    print(f"[ERROR] {legs_csv} not found.")
    print("Run:  cargo run --bin mission-planner --release -- "
          "mga-geometry config/<mission>.toml")
    sys.exit(1)

legs = pd.read_csv(legs_csv)

if "i_dep_deg" not in legs.columns:
    print("[ERROR] legs CSV missing i_dep_deg / i_lambert_deg. "
          "Re-run mga-geometry with the updated binary.")
    sys.exit(1)

n_legs   = len(legs)
body_seq = [legs.iloc[0]["body_dep"].strip()]
for _, row in legs.iterrows():
    body_seq.append(row["body_arr"].strip())
n_bodies = len(body_seq)

print(f"Sequence: {' -> '.join(body_seq)}")

# ── Tisserand helpers ─────────────────────────────────────────────────────────

def v_circular_ms(a_p_au: float) -> float:
    return math.sqrt(MU_SUN / (a_p_au * AU))

def tp_from_vinf(vinf_ms: float, a_p_au: float) -> float:
    v_p = v_circular_ms(a_p_au)
    return 3.0 - (vinf_ms / v_p) ** 2

def vinf_from_tp(T_P: float, a_p_au: float) -> float:
    if T_P >= 3.0:
        return 0.0
    return v_circular_ms(a_p_au) * math.sqrt(3.0 - T_P)

def tisserand_param(rp_au: float, ra_au: float, i_deg: float,
                    a_p_au: float) -> float:
    """Exact 3-D Tisserand parameter from orbital elements."""
    a = (rp_au + ra_au) / 2.0
    e = (ra_au - rp_au) / (ra_au + rp_au)
    if a <= 0 or e >= 1.0 or e < 0.0:
        return float("nan")
    cos_i = math.cos(math.radians(i_deg))
    return a_p_au / a + 2.0 * math.sqrt(a / a_p_au * (1.0 - e ** 2)) * cos_i

def surface_i_grid(a_p_au: float, T_P: float,
                   rp_grid: np.ndarray, ra_grid: np.ndarray) -> np.ndarray:
    """
    Vectorised: for each (Rp, Ra) pair compute the ecliptic inclination [deg]
    that satisfies T_P = a_P/a + 2*sqrt(a/a_P*(1-e^2))*cos(i).
    Returns NaN where no valid real inclination exists (|cos_i| > 1)
    or where Ra < Rp.
    """
    valid = ra_grid >= rp_grid
    a     = np.where(valid, (rp_grid + ra_grid) / 2.0, np.nan)
    e     = np.where(valid, (ra_grid - rp_grid) / (ra_grid + rp_grid), np.nan)
    e2    = np.where(valid & (e >= 0), 1.0 - e ** 2, np.nan)
    denom = 2.0 * np.sqrt(np.where(e2 > 0, a / a_p_au * e2, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_i = (T_P - a_p_au / a) / denom
    cos_i = np.where(np.abs(cos_i) <= 1.0, cos_i, np.nan)
    return np.degrees(np.arccos(cos_i))

# ── Collect body encounter v_inf from legs CSV ────────────────────────────────

# For each body in the sequence build:
#   body_vinf_ms[b] = v_inf at that body on the trajectory
#   body_tp[b]      = exact 3-D Tisserand param for that body
body_vinf_ms: dict[str, float] = {}
body_tp:      dict[str, float] = {}

# Departure body
dep_row = legs.iloc[0]
dep = body_seq[0]
body_vinf_ms[dep] = float(dep_row["vinf_dep_ms"])
body_tp[dep]      = tisserand_param(
    float(dep_row["rp_dep_m"]), float(dep_row["ra_dep_m"]),
    float(dep_row["i_dep_deg"]), PLANET_SMA_AU.get(dep, 1.0)
)

# Intermediate flyby bodies
for k in range(n_legs - 1):
    row  = legs.iloc[k]
    body = row["body_arr"].strip()
    body_vinf_ms[body] = float(row["vinf_arr_ms"])
    body_tp[body]      = tisserand_param(
        float(row["rp_lambert_m"]), float(row["ra_lambert_m"]),
        float(row["i_lambert_deg"]), PLANET_SMA_AU.get(body, 1.0)
    )

# Arrival body
arr_row = legs.iloc[-1]
arr     = body_seq[-1]
body_vinf_ms[arr] = float(arr_row["vinf_arr_ms"])
body_tp[arr]      = tisserand_param(
    float(arr_row["rp_lambert_m"]), float(arr_row["ra_lambert_m"]),
    float(arr_row["i_lambert_deg"]), PLANET_SMA_AU.get(arr, 1.0)
)

# ── Reference v_inf levels per body ──────────────────────────────────────────
# Choose 5 reference v_inf values per body so surfaces span 0-I_MAX degrees.
# Anchor to the trajectory v_inf; add lower and higher reference levels.

def ref_vinfs_for_body(body: str, traj_vinf_ms: float) -> list[float]:
    """Return ~5 v_inf values [m/s] for reference surface lines, including
    the trajectory v_inf."""
    sma = PLANET_SMA_AU.get(body)
    if sma is None:
        return [traj_vinf_ms]
    v_p = v_circular_ms(sma)
    # Max v_inf where a prograde equatorial orbit can still reach that body
    # (T_P must be >= 0 for the orbit to cross the body's SMA)
    # T_P = 0 when v_inf = v_p*sqrt(3) (parabolic)
    v_max = v_p * math.sqrt(3.0) * 0.90   # 90% of parabolic speed
    step  = max(1000.0, round(traj_vinf_ms / 4.0 / 1000.0) * 1000.0)
    levels = sorted(set([
        max(500.0, traj_vinf_ms - 2 * step),
        max(500.0, traj_vinf_ms - step),
        traj_vinf_ms,
        min(v_max, traj_vinf_ms + step),
        min(v_max, traj_vinf_ms + 2 * step),
    ]))
    return levels

# ── Global axis limits ────────────────────────────────────────────────────────

all_rp, all_ra = [], []
for _, row in legs.iterrows():
    for rp_col, ra_col in [("rp_dep_m", "ra_dep_m"),
                            ("rp_lambert_m", "ra_lambert_m")]:
        rp = float(row[rp_col]); ra = float(row[ra_col])
        if rp > 0 and ra >= rp and math.isfinite(ra):
            all_rp.append(rp); all_ra.append(ra)

rp_min = max(0.05, min(all_rp) * 0.80)
rp_max = max(all_rp) * 1.30
ra_max = max(all_ra) * 1.30

# Widen to cover all SMA values of bodies in the sequence
for b in body_seq:
    sma = PLANET_SMA_AU.get(b)
    if sma:
        rp_max = max(rp_max, sma * 1.1)
        ra_max = max(ra_max, sma * 1.4)

rp_grid_vals = np.linspace(rp_min, rp_max,   N_SURF)
ra_grid_vals = np.linspace(rp_min, ra_max,   N_SURF)
rp_g, ra_g   = np.meshgrid(rp_grid_vals, ra_grid_vals)

# ── Build figure ──────────────────────────────────────────────────────────────

fig = go.Figure()

# ── Tisserand surfaces -- one set per body ────────────────────────────────────

for body in body_seq:
    sma = PLANET_SMA_AU.get(body)
    if sma is None:
        continue
    col          = PLANET_COLOR.get(body, "#ffffff")
    traj_vinf    = body_vinf_ms.get(body, 0.0)
    ref_levels   = ref_vinfs_for_body(body, traj_vinf)
    for vinf_ms in ref_levels:
        T_P        = tp_from_vinf(vinf_ms, sma)
        is_traj    = abs(vinf_ms - traj_vinf) < 200.0  # within 200 m/s
        opacity    = 0.30 if is_traj else 0.09
        line_width = 3    if is_traj else 1
        vinf_km    = vinf_ms / 1000.0

        i_g = surface_i_grid(sma, T_P, rp_g, ra_g)

        # Mask cells where inclination exceeds the plot ceiling
        i_g = np.where(i_g <= I_MAX, i_g, np.nan)

        # Skip surfaces where no grid cell has a valid i
        if np.all(np.isnan(i_g)):
            continue

        fig.add_trace(go.Surface(
            x=rp_g, y=ra_g, z=i_g,
            colorscale=[[0, col], [1, col]],
            showscale=False,
            opacity=opacity,
            name=f"{body}  v_inf={vinf_km:.1f} km/s",
            showlegend=is_traj,   # one legend entry per body, for the traj surface
            legendgroup=body,
            hovertemplate=(
                f"<b>{body}</b>  v_inf={vinf_km:.1f} km/s<br>"
                "Rp=%{x:.3f} AU<br>Ra=%{y:.3f} AU<br>i=%{z:.1f} deg"
                "<extra></extra>"
            ),
        ))

        # Floor contour line at i=0 (= the classical 2-D Tisserand contour).
        # This is what you see when looking straight down the inclination axis.
        i0_g = surface_i_grid(sma, T_P, rp_g, ra_g)
        # At i=0 the surface equation reduces to:
        #   T_P = a_P/a + 2*sqrt(a/a_P*(1-e^2))
        # Compute Ra(Rp) at i=0 analytically: solve via the grid at very small i.
        # Practical: pick the i=0 row from the grid where Ra varies and i=0 is exact.
        # We use a dense Rp scan at i=0 separately.
        rp_1d = np.linspace(rp_min, rp_max, 200)
        # For each Rp find Ra such that T_P = a_P/((Rp+Ra)/2) +
        #   2*sqrt((Rp+Ra)/(2*a_P) * 4*Rp*Ra/(Rp+Ra)^2)
        # = a_P/((Rp+Ra)/2) + 4*sqrt(Rp*Ra/((Rp+Ra)*a_P))
        # Numerically via vectorised scan:
        ra_1d_test = np.linspace(rp_1d, np.full_like(rp_1d, ra_max), 500).T
        rp_exp = rp_1d[:, None]
        a_t   = (rp_exp + ra_1d_test) / 2.0
        e_t   = np.where(ra_1d_test > rp_exp,
                         (ra_1d_test - rp_exp) / (ra_1d_test + rp_exp), np.nan)
        tp_t  = sma / a_t + 2.0 * np.sqrt(np.where(
            e_t >= 0, a_t / sma * (1.0 - e_t**2), np.nan))
        # Find where tp_t crosses T_P (sign change)
        above = (tp_t >= T_P)
        cross_idx = np.argmax(np.diff(above.astype(int), axis=1) != 0, axis=1)
        ra_floor = np.where(cross_idx > 0,
                            ra_1d_test[np.arange(len(rp_1d)), cross_idx],
                            np.nan)
        valid_floor = ~np.isnan(ra_floor) & (ra_floor > rp_1d)

        if valid_floor.sum() >= 2:
            fig.add_trace(go.Scatter3d(
                x=rp_1d[valid_floor],
                y=ra_floor[valid_floor],
                z=np.zeros(valid_floor.sum()),
                mode="lines",
                line=dict(color=col,
                          width=line_width + (2 if is_traj else 0)),
                opacity=0.6 if is_traj else 0.25,
                showlegend=False,
                legendgroup=body,
                hovertemplate=(
                    f"<b>{body}</b> floor contour  "
                    f"v_inf={vinf_km:.1f} km/s<br>"
                    "Rp=%{x:.3f} AU  Ra=%{y:.3f} AU  i=0 deg<extra></extra>"
                ),
                name=f"{body} i=0 contour  v_inf={vinf_km:.1f}",
            ))


# ── Orbit-state dots ──────────────────────────────────────────────────────────

for k, row in legs.iterrows():
    dep_body = row["body_dep"].strip()
    arr_body = row["body_arr"].strip()
    col      = LEG_COLORS[k % len(LEG_COLORS)]

    for (rp_col, ra_col, i_col, label, sym, sz) in [
        ("rp_dep_m",     "ra_dep_m",     "i_dep_deg",     "dep", "diamond", 10),
        ("rp_lambert_m", "ra_lambert_m", "i_lambert_deg", "arr", "circle",  12),
    ]:
        rp  = float(row[rp_col])
        ra  = float(row[ra_col])
        inc = float(row[i_col])
        body_label = dep_body if label == "dep" else arr_body

        if rp <= 0 or ra < rp or not math.isfinite(inc):
            continue

        hover = (f"<b>L{k} {label} -- {body_label}</b><br>"
                 f"Rp={rp:.3f} AU  Ra={ra:.3f} AU  i={inc:.1f} deg<extra></extra>")
        fig.add_trace(go.Scatter3d(
            x=[rp], y=[ra], z=[inc],
            mode="markers",
            marker=dict(size=sz, symbol=sym, color=col,
                        line=dict(color="white", width=1.5)),
            name=f"L{k} {label} ({body_label})",
            hovertemplate=hover,
        ))
        # Vertical projection line to the floor (visual aid)
        fig.add_trace(go.Scatter3d(
            x=[rp, rp], y=[ra, ra], z=[inc, 0.0],
            mode="lines",
            line=dict(color=col, width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))

# ── DSM burn jumps ────────────────────────────────────────────────────────────

BURN_COL = "#ff5555"
for k, row in legs.iterrows():
    rp0, ra0, i0 = float(row["rp_dep_m"]),     float(row["ra_dep_m"]),     float(row["i_dep_deg"])
    rp1, ra1, i1 = float(row["rp_lambert_m"]), float(row["ra_lambert_m"]), float(row["i_lambert_deg"])
    dv           = float(row["dv_dsm_ms"])
    if not all(math.isfinite(v) for v in [rp0, ra0, rp1, ra1]):
        continue
    fig.add_trace(go.Scatter3d(
        x=[rp0, rp1], y=[ra0, ra1], z=[i0, i1],
        mode="lines",
        line=dict(color=BURN_COL, width=5, dash="dash"),
        name=f"DSM {k}  {dv:.0f} m/s",
        hovertemplate=(
            f"<b>DSM {k}</b>  dv={dv:.0f} m/s<br>"
            f"Rp {rp0:.3f}->{rp1:.3f} AU  Ra {ra0:.3f}->{ra1:.3f} AU  "
            f"i {i0:.1f}->{i1:.1f} deg<extra></extra>"
        ),
    ))

# ── Flyby connections (on the Tisserand surface) ──────────────────────────────

for k in range(n_legs - 1):
    row0 = legs.iloc[k];  row1 = legs.iloc[k + 1]
    fb   = row0["body_arr"].strip()
    col  = PLANET_COLOR.get(fb, "#ffffff")

    rp_pre, ra_pre, i_pre = (float(row0["rp_lambert_m"]),
                              float(row0["ra_lambert_m"]),
                              float(row0["i_lambert_deg"]))
    rp_post, ra_post, i_post = (float(row1["rp_dep_m"]),
                                 float(row1["ra_dep_m"]),
                                 float(row1["i_dep_deg"]))

    if not all(math.isfinite(v)
               for v in [rp_pre, ra_pre, i_pre, rp_post, ra_post, i_post]):
        continue

    fig.add_trace(go.Scatter3d(
        x=[rp_pre, rp_post], y=[ra_pre, ra_post], z=[i_pre, i_post],
        mode="lines",
        line=dict(color=col, width=8),
        name=f"{fb} flyby (Tisserand conserved)",
        hovertemplate=(
            f"<b>{fb} flyby</b><br>"
            f"Rp {rp_pre:.3f}->{rp_post:.3f} AU<br>"
            f"Ra {ra_pre:.3f}->{ra_post:.3f} AU<br>"
            f"i  {i_pre:.1f}->{i_post:.1f} deg<extra></extra>"
        ),
    ))

# ── Body SMA reference lines (vertical at Rp=Ra=SMA, i in [0, I_MAX]) ────────

for body in body_seq:
    sma = PLANET_SMA_AU.get(body)
    if sma is None:
        continue
    col = PLANET_COLOR.get(body, "#888")
    i_line = np.linspace(0, I_MAX, 40)
    fig.add_trace(go.Scatter3d(
        x=[sma] * 40, y=[sma] * 40, z=list(i_line),
        mode="lines",
        line=dict(color=col, width=2, dash="dot"),
        name=f"{body}  a={sma:.3f} AU",
        showlegend=True,
        hoverinfo="skip",
    ))

# ── Legend body label annotations ────────────────────────────────────────────
# Print T_P info for each body
for body in body_seq:
    tp = body_tp.get(body)
    vi = body_vinf_ms.get(body, 0.0)
    if tp:
        sma = PLANET_SMA_AU.get(body)
        tp_coplanar = tp_from_vinf(vi, sma) if sma else float("nan")
        print(f"  {body:10s}  v_inf = {vi/1000:.2f} km/s  "
              f"T_P(3D) = {tp:.4f}  T_P(coplanar approx) = {tp_coplanar:.4f}")

# ── Layout ────────────────────────────────────────────────────────────────────

sequence_str = " -> ".join(body_seq)

fig.update_layout(
    paper_bgcolor=BG,
    font=dict(color=TEXT, family="monospace", size=11),
    title=dict(
        text=f"3D Tisserand Graph -- {sequence_str}",
        font=dict(color="#00d4ff", size=16),
        x=0.5,
    ),
    scene=dict(
        bgcolor=BG,
        xaxis=dict(
            title=dict(text="Rp [AU]", font=dict(color=TEXT)),
            gridcolor=GRID, zerolinecolor=GRID,
            tickfont=dict(color=TEXT),
            range=[rp_min, rp_max],
        ),
        yaxis=dict(
            title=dict(text="Ra [AU]", font=dict(color=TEXT)),
            gridcolor=GRID, zerolinecolor=GRID,
            tickfont=dict(color=TEXT),
            range=[rp_min, ra_max],
        ),
        zaxis=dict(
            title=dict(text="Inclination [deg]", font=dict(color=TEXT)),
            gridcolor=GRID, zerolinecolor=GRID,
            tickfont=dict(color=TEXT),
            range=[0, I_MAX],
        ),
        # Start with a view angle that shows the 3D structure clearly.
        # Rotate to look straight down (eye.z >> x,y) to see the 2D collapse.
        camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        aspectmode="manual",
        aspectratio=dict(x=1.0, y=1.5, z=0.8),
    ),
    height=820,
    legend=dict(
        bgcolor="#111", bordercolor="#333", borderwidth=1,
        font=dict(size=10), x=0.01, y=0.99,
    ),
)

# ── Save ──────────────────────────────────────────────────────────────────────

out_path = out_dir / "tisserand_3d.html"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(out_path), include_plotlyjs=True)
print(f"Saved: {out_path}")
if not args.no_browser:
    webbrowser.open(str(out_path))
