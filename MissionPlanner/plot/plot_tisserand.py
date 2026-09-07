"""
Tisserand graph -- apsis form (Ra vs Rp [AU], log-log).

The standard tool for MGA sequence design. For each planet, plots families of
heliocentric orbits accessible at different v∞ values -- each curve in a family
holds the Tisserand parameter T_P constant (= 3 - (v∞/v_P)²), so a *gravity
assist at planet P keeps the spacecraft on the same curve* (only the direction
of the v∞ vector changes; the orbit shape within the family is fixed by T_P).

Where a T_P = k curve for planet P intersects a T_Q = k curve for planet Q,
an unpowered link between P and Q is possible at that energy level.

Reference: Strange & Longuski (2002), "Graphical Method for Gravity-Assist
Trajectory Design", JGCD 25(6):1154-1159.

Usage (from MissionPlanner/ directory):
  python plot/plot_tisserand.py
  python plot/plot_tisserand.py --bodies Venus Earth Jupiter Saturn
  python plot/plot_tisserand.py --legs-csv out/evj_flyby/mga_legs.csv

Overlay a mission sequence:
  python plot/plot_tisserand.py --legs-csv out/evj_flyby/mga_legs.csv
"""

import argparse
import csv
import math
import sys
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# ── Constants ─────────────────────────────────────────────────────────────────

MU_SUN = 1.327_124_400_18e20   # m³/s²  (IAU 2012)
AU     = 1.495_978_707e11       # m

BG   = "#0f0f0f"
GRID = "#1c1c1c"
TEXT = "#cccccc"

# Per-planet display properties
PLANET_PROPS = {
    "Mercury": dict(sma_au=0.387,  color="#b0b0b0", vinf_km=[1, 2, 3, 5, 8]),
    "Venus":   dict(sma_au=0.723,  color="#e8c97e", vinf_km=[1, 2, 3, 5, 8, 10]),
    "Earth":   dict(sma_au=1.000,  color="#4fc3f7", vinf_km=[1, 2, 3, 5, 8, 10]),
    "Mars":    dict(sma_au=1.524,  color="#ef5350", vinf_km=[1, 2, 3, 5, 8, 10]),
    "Jupiter": dict(sma_au=5.203,  color="#ffb74d", vinf_km=[1, 2, 3, 5, 8, 10, 15]),
    "Saturn":  dict(sma_au=9.537,  color="#f9a825", vinf_km=[1, 2, 3, 5, 8, 10, 15, 20]),
    "Uranus":  dict(sma_au=19.19,  color="#80cbc4", vinf_km=[2, 5, 10, 15, 20]),
    "Neptune": dict(sma_au=30.07,  color="#1976d2", vinf_km=[2, 5, 10, 15, 20]),
}


# ── Tisserand maths ───────────────────────────────────────────────────────────

def v_circular_ms(a_m: float) -> float:
    return math.sqrt(MU_SUN / a_m)


def tisserand_parameter(a_m: float, e: float, a_p_m: float) -> float:
    """T_P = a_P/a + 2*sqrt(a/a_P * (1-e²))  (coplanar, Strange & Longuski eq.1)."""
    return a_p_m / a_m + 2.0 * math.sqrt((a_m / a_p_m) * (1.0 - e * e))


def tisserand_contour_apsis(a_p_m: float, vinf_ms: float, n_pts: int = 300):
    """
    Return (rp_au, ra_au) arrays for the Tisserand contour at planet P with
    the given v∞.

    Derivation: for fixed Rp, find Ra by solving the quadratic that results
    from squaring the Tisserand equation:

      T_P = 2*a_P/(Rp+Ra) + 2*sqrt(2*Rp*Ra / (a_P*(Rp+Ra)))

    Coefficients (with S = Rp + Ra, f = Rp fixed):
      A = T_P² - 8*f/a_P
      B = -4*T_P*a_P + 8*f²/a_P
      C = 4*a_P²

    This is a valid elliptic orbit intersecting the planet's distance when
    Ra >= a_P and Ra >= Rp.

    Returns: sorted (rp_au, ra_au) lists, empty if no valid contour.
    """
    v_p  = v_circular_ms(a_p_m)
    T_P  = 3.0 - (vinf_ms / v_p) ** 2

    # T_P <= 1 means the spacecraft can't form a bound orbit passing the planet
    # with this v∞ (would need too much energy).
    if T_P <= 1.0:
        return [], []

    points = []
    rp_max = a_p_m * 0.99999   # perihelion can't exceed planet's SMA

    for rp_m in np.linspace(0.02 * AU, rp_max, n_pts):
        A = T_P ** 2 - 8.0 * rp_m / a_p_m
        B = -4.0 * T_P * a_p_m + 8.0 * rp_m ** 2 / a_p_m
        C = 4.0 * a_p_m ** 2

        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            continue

        sqrt_d = math.sqrt(disc)
        for sign in (1, -1):
            denom = 2.0 * A
            if abs(denom) < 1e-20:
                continue
            S = (-B + sign * sqrt_d) / denom
            Ra = S - rp_m
            # Filter: valid ellipse that actually reaches the planet
            if Ra < a_p_m * 0.98 or Ra < rp_m or S <= 0:
                continue
            # Back-verify T_P (catches solutions from the squaring step that
            # don't satisfy the original equation)
            a_sc = S / 2.0
            e_sc = (Ra - rp_m) / S
            if e_sc < 0.0 or e_sc >= 1.0:
                continue
            T_check = tisserand_parameter(a_sc, e_sc, a_p_m)
            if abs(T_check - T_P) > 0.05 * abs(T_P):
                continue
            points.append((rp_m / AU, Ra / AU))

    if not points:
        return [], []
    points.sort()
    rp_vals, ra_vals = zip(*points)
    return list(rp_vals), list(ra_vals)


# ── Sequence overlay helpers ──────────────────────────────────────────────────

def load_legs_csv(path: str):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "leg_idx":        int(row["leg_idx"]),
                "body_dep":       row["body_dep"].strip(),
                "body_arr":       row["body_arr"].strip(),
                "vinf_dep_ms":    float(row["vinf_dep_ms"]),
                "vinf_arr_ms":    float(row["vinf_arr_ms"]),
                "tof_days":       float(row["tof_days"]),
                "dv_dsm_ms":      float(row["dv_dsm_ms"]),
                "turn_deg":       float(row.get("turn_deg", 0)),
                "rp_km":          float(row.get("rp_km", 0)),
                # Exact orbital elements written by Rust (AU). Present only
                # in CSVs produced after the orbit-element fix.
                "rp_dep_au":      float(row["rp_dep_m"])      if "rp_dep_m"      in row else None,
                "ra_dep_au":      float(row["ra_dep_m"])      if "ra_dep_m"      in row else None,
                "rp_lambert_au":  float(row["rp_lambert_m"])  if "rp_lambert_m"  in row else None,
                "ra_lambert_au":  float(row["ra_lambert_m"])  if "ra_lambert_m"  in row else None,
            })
    return rows


def approximate_orbit_point(vinf_ms: float, body_name: str, is_arrival: bool,
                            prev_body: str = None, next_body: str = None):
    """
    Estimate (Rp_au, Ra_au) for the spacecraft orbit at an encounter with body_name.

    Convention: if departing from an inner body (prev_body SMA < body SMA),
    the spacecraft approaches the planet from below → Ra ≈ a_body (aphelion).
    If departing from an outer body, Rp ≈ a_body (perihelion).
    """
    if body_name not in PLANET_PROPS:
        return None

    a_p_m = PLANET_PROPS[body_name]["sma_au"] * AU
    v_p   = v_circular_ms(a_p_m)
    T_P   = 3.0 - (vinf_ms / v_p) ** 2
    if T_P <= 1.0:
        return None

    # Determine whether body is at aphelion or perihelion of the encounter orbit
    ref_body = prev_body if is_arrival else next_body
    aphelion_encounter = True  # default: outbound (body is near aphelion)
    if ref_body and ref_body in PLANET_PROPS:
        ref_sma = PLANET_PROPS[ref_body]["sma_au"]
        body_sma = PLANET_PROPS[body_name]["sma_au"]
        aphelion_encounter = (ref_sma < body_sma)

    # Fix the known apsis = a_P, solve for the other using the same quadratic
    # (the formula is symmetric in Rp and Ra)
    f_m = a_p_m  # the known apsis

    A = T_P ** 2 - 8.0 * f_m / a_p_m
    B = -4.0 * T_P * a_p_m + 8.0 * f_m ** 2 / a_p_m
    C = 4.0 * a_p_m ** 2

    disc = B * B - 4.0 * A * C
    if disc < 0:
        return None

    sqrt_d = math.sqrt(disc)
    best = None
    for sign in (1, -1):
        denom = 2.0 * A
        if abs(denom) < 1e-20:
            continue
        S = (-B + sign * sqrt_d) / denom
        other = S - f_m
        if other <= 0 or S <= 0:
            continue
        if aphelion_encounter:
            rp_m, ra_m = other, f_m    # other < a_P, f is Ra=a_P
        else:
            rp_m, ra_m = f_m, other    # f is Rp=a_P, other > a_P

        if ra_m < rp_m:
            rp_m, ra_m = ra_m, rp_m  # swap if needed

        # Verify
        a_sc = (rp_m + ra_m) / 2.0
        e_sc = (ra_m - rp_m) / (rp_m + ra_m)
        if e_sc < 0 or e_sc >= 1:
            continue
        T_check = tisserand_parameter(a_sc, e_sc, a_p_m)
        if abs(T_check - T_P) > 0.1 * abs(T_P):
            continue
        if best is None or abs(T_check - T_P) < abs(best[2] - T_P):
            best = (rp_m / AU, ra_m / AU, T_check)

    return (best[0], best[1]) if best else None


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Tisserand apsis graph (Ra vs Rp, log-log)")
parser.add_argument("--bodies", nargs="+", default=[],
                    help="Planets to plot (default: auto from legs-csv or Venus/Earth/Jupiter)")
parser.add_argument("--vinf-max", type=float, default=None,
                    help="Maximum v∞ contour to draw [km/s] (auto if omitted)")
parser.add_argument("--legs-csv", default=None,
                    help="Path to mga_legs.csv — overlay the optimized sequence")
parser.add_argument("--rp-min", type=float, default=0.05,
                    help="Minimum Rp axis [AU]")
parser.add_argument("--rp-max", type=float, default=None,
                    help="Maximum Rp axis [AU] (auto)")
parser.add_argument("--ra-min", type=float, default=None,
                    help="Minimum Ra axis [AU] (auto)")
parser.add_argument("--ra-max", type=float, default=None,
                    help="Maximum Ra axis [AU] (auto)")
parser.add_argument("--output", default=None,
                    help="Save HTML to this path instead of auto-naming")
args = parser.parse_args()

# Load sequence data
legs_data  = []
seq_bodies = []
if args.legs_csv:
    if not Path(args.legs_csv).exists():
        print(f"[ERROR] {args.legs_csv} not found", file=sys.stderr)
        sys.exit(1)
    legs_data = load_legs_csv(args.legs_csv)
    seen = {}
    for leg in legs_data:
        for b in (leg["body_dep"], leg["body_arr"]):
            if b not in seen:
                seen[b] = True
    seq_bodies = list(seen.keys())
    seq_print = " -> ".join(
        [legs_data[0]["body_dep"]] + [l["body_arr"] for l in legs_data])
    print(f"Loaded {len(legs_data)} legs -- sequence: {seq_print}")

# Determine bodies to plot
if args.bodies:
    bodies = args.bodies
elif seq_bodies:
    bodies = [b for b in seq_bodies if b in PLANET_PROPS]
else:
    bodies = ["Venus", "Earth", "Jupiter"]

for b in bodies:
    if b not in PLANET_PROPS:
        print(f"Unknown body '{b}'. Available: {', '.join(PLANET_PROPS)}")
        sys.exit(1)

# Auto v∞ max: largest planet's highest contour level
vinf_max_km = args.vinf_max or max(max(PLANET_PROPS[b]["vinf_km"]) for b in bodies)

# Axis ranges
sma_vals = [PLANET_PROPS[b]["sma_au"] for b in bodies]
rp_min = args.rp_min
rp_max = args.rp_max or max(sma_vals) * 1.05
ra_min = args.ra_min or rp_min * 0.8
ra_max = args.ra_max or max(sma_vals) * 12.0

# ── Build figure ──────────────────────────────────────────────────────────────

fig = go.Figure()

# Diagonal Ra = Rp (circular orbits)
diag = np.logspace(math.log10(rp_min), math.log10(rp_max), 200)
fig.add_trace(go.Scatter(
    x=list(diag), y=list(diag),
    mode="lines",
    line=dict(color="#3a3a3a", width=1.5, dash="dot"),
    name="Ra = Rp (circular)",
    showlegend=True,
))

# ── Contour families (one per planet) ────────────────────────────────────────

label_every = 3   # label every N-th contour level for clarity

for body in bodies:
    props    = PLANET_PROPS[body]
    a_p_m    = props["sma_au"] * AU
    base_col = props["color"]
    levels   = [v for v in props["vinf_km"] if v <= vinf_max_km]

    n_levels = len(levels)
    first_body_trace = True

    for k, vinf_km in enumerate(levels):
        rp_vals, ra_vals = tisserand_contour_apsis(a_p_m, vinf_km * 1e3)
        if not rp_vals:
            continue

        # Opacity: lower v∞ = more transparent
        opacity = 0.35 + 0.65 * (k / max(1, n_levels - 1))

        fig.add_trace(go.Scatter(
            x=rp_vals, y=ra_vals,
            mode="lines",
            line=dict(color=base_col, width=1.8),
            opacity=opacity,
            name=f"{body}  v∞={vinf_km} km/s" if first_body_trace else "",
            legendgroup=body,
            showlegend=first_body_trace,
            hovertemplate=(
                f"<b>{body}</b>  v∞={vinf_km} km/s<br>"
                "Rp=%{x:.3f} AU<br>Ra=%{y:.3f} AU<extra></extra>"
            ),
        ))
        first_body_trace = False

        # Label contour: place text at the rightmost (largest Rp) valid point
        if rp_vals:
            label_x = rp_vals[-1]
            label_y = ra_vals[-1]
            fig.add_annotation(
                x=math.log10(label_x),
                y=math.log10(label_y),
                xref="x", yref="y",
                text=f"{vinf_km}",
                showarrow=False,
                font=dict(color=base_col, size=9),
                xanchor="left",
                opacity=opacity,
            )

# Vertical dashed lines at planet SMAs
for body in bodies:
    sma = PLANET_PROPS[body]["sma_au"]
    col = PLANET_PROPS[body]["color"]
    fig.add_vline(
        x=math.log10(sma),
        line=dict(color=col, width=1, dash="dash"),
        opacity=0.4,
        annotation_text=body,
        annotation_position="top",
        annotation_font=dict(color=col, size=10),
    )

# ── Sequence overlay from legs_csv ────────────────────────────────────────────

if legs_data:
    has_exact = legs_data[0]["rp_dep_au"] is not None
    if not has_exact:
        print("[WARN] legs CSV missing rp/ra columns -- re-run mga-geometry to get exact overlay.")

    OVERLAY_COLORS = [
        "#ff4081", "#00e676", "#ffcf00",
        "#bf5fff", "#ff6b35", "#00d4ff", "#7fba00", "#ffffff",
    ]

    # Helper: 2D Tisserand-equivalent v∞ from orbit (Rp, Ra).
    # Tisserand is coplanar -- out-of-plane velocity isn't captured.  Computing
    # v∞ from (Rp, Ra) rather than the 3D encounter v∞ ensures the flyby
    # contour segment passes exactly through each orbit dot.
    def vinf_2d_from_orbit(rp_au, ra_au, body):
        if body not in PLANET_PROPS:
            return None
        a_p_m = PLANET_PROPS[body]["sma_au"] * AU
        a_m   = (rp_au + ra_au) / 2.0 * AU
        e     = (ra_au - rp_au) / (ra_au + rp_au)
        if e < 0.0 or e >= 1.0 or a_m <= 0:
            return None
        T_P = a_p_m / a_m + 2.0 * math.sqrt(a_m / a_p_m * (1.0 - e * e))
        if T_P >= 3.0:
            return None
        return math.sqrt(MU_SUN / a_p_m) * math.sqrt(3.0 - T_P)

    # ── Step 2: flyby paths ──────────────────────────────────────────────────────
    # For a coplanar flyby the pre and post-flyby orbits lie on the same
    # Tisserand contour (|v∞| is conserved, inclination unchanged).  For an
    # inclined flyby the 2D Tisserand-equivalent v∞ (computed from Rp/Ra, which
    # ignores inclination) changes, so pre and post-flyby dots land on different
    # 2D contours even though 3D |v∞| is conserved.  In that case drawing a
    # Tisserand contour segment through an "average" v∞ produces a curve that
    # passes through NEITHER dot.  The honest representation is a straight line
    # between the two orbit dots labelled "flyby (inclined)".
    if has_exact:
        for k in range(len(legs_data) - 1):
            leg      = legs_data[k]
            leg_next = legs_data[k + 1]
            fb_body  = leg["body_arr"]

            rp_pre  = leg["rp_lambert_au"]
            ra_pre  = leg["ra_lambert_au"]
            rp_post = leg_next["rp_dep_au"]
            ra_post = leg_next["ra_dep_au"]

            if not all(v is not None and v > 0 and math.isfinite(v)
                       for v in [rp_pre, ra_pre, rp_post, ra_post]):
                continue
            if fb_body not in PLANET_PROPS:
                continue

            col = PLANET_PROPS[fb_body]["color"]
            drp = abs(rp_post - rp_pre)
            dra = abs(ra_post - ra_pre)

            v2d_pre  = vinf_2d_from_orbit(rp_pre,  ra_pre,  fb_body)
            v2d_post = vinf_2d_from_orbit(rp_post, ra_post, fb_body)

            # Decide: coplanar contour arc, or inclined straight line?
            use_contour = False
            v2d_flyby   = None
            if v2d_pre and v2d_post:
                avg = (v2d_pre + v2d_post) / 2.0
                if abs(v2d_pre - v2d_post) / avg < 0.10:
                    # < 10% difference: treat as coplanar, anchor contour to
                    # the post-flyby dot so L(k+1)-dep lies exactly on it.
                    use_contour = True
                    v2d_flyby   = v2d_post
            elif v2d_post or v2d_pre:
                use_contour = True
                v2d_flyby   = v2d_post or v2d_pre

            if drp < 0.05 and dra < 0.3:
                # Gravity assist was essentially an inclination change only;
                # the dots nearly overlap, so show a text annotation instead.
                fig.add_annotation(
                    x=math.log10((rp_pre + rp_post) / 2),
                    y=math.log10((ra_pre + ra_post) / 2 * 0.85),
                    xref="x", yref="y",
                    text=f"Flyby {fb_body}<br>inclination change<br>(2D Tisserand unchanged)",
                    showarrow=True, arrowhead=2, arrowwidth=1.5,
                    arrowcolor=col, ax=50, ay=50,
                    font=dict(color=col, size=9),
                    bgcolor="#0f0f0f", bordercolor=col, borderwidth=1,
                    opacity=0.9,
                )
            elif use_contour and v2d_flyby:
                # Coplanar flyby: draw the highlighted Tisserand contour segment.
                rp_c, ra_c = tisserand_contour_apsis(
                    PLANET_PROPS[fb_body]["sma_au"] * AU, v2d_flyby, n_pts=1000)
                if not rp_c:
                    continue
                rp_arr = np.array(rp_c)
                ra_arr = np.array(ra_c)
                rp_lo = min(rp_pre, rp_post)
                rp_hi = max(rp_pre, rp_post)
                tol   = max(0.04, (rp_hi - rp_lo) * 0.15)
                mask  = (rp_arr >= rp_lo - tol) & (rp_arr <= rp_hi + tol)
                seg_rp = rp_arr[mask]
                seg_ra = ra_arr[mask]
                if len(seg_rp) >= 2:
                    fig.add_trace(go.Scatter(
                        x=list(seg_rp), y=list(seg_ra), mode="lines",
                        line=dict(color=col, width=6), opacity=1.0,
                        name=f"{fb_body} flyby (coplanar)",
                        legendgroup=f"flyby_{fb_body}", showlegend=True,
                    ))
                    if rp_pre < rp_post:
                        ax_rp, ax_ra = seg_rp[-4], seg_ra[-4]
                        x_rp, x_ra  = seg_rp[-1], seg_ra[-1]
                    else:
                        ax_rp, ax_ra = seg_rp[3],  seg_ra[3]
                        x_rp, x_ra  = seg_rp[0],  seg_ra[0]
                    fig.add_annotation(
                        x=math.log10(x_rp), y=math.log10(x_ra),
                        ax=math.log10(ax_rp), ay=math.log10(ax_ra),
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=3,
                        arrowsize=2.0, arrowwidth=3.5, arrowcolor=col,
                    )
            else:
                # Inclined flyby: 2D Tisserand-equivalent v∞ changes (inclination
                # absorbed the energy).  Draw a straight line between the two orbit
                # dots — the only honest representation on a 2D coplanar graph.
                fig.add_trace(go.Scatter(
                    x=[rp_pre, rp_post], y=[ra_pre, ra_post],
                    mode="lines",
                    line=dict(color=col, width=6),
                    name=f"{fb_body} flyby (inclined)",
                    legendgroup=f"flyby_{fb_body}", showlegend=True,
                ))
                # Arrow pointing to the post-flyby dot
                fig.add_annotation(
                    x=math.log10(rp_post), y=math.log10(ra_post),
                    ax=math.log10(rp_pre),  ay=math.log10(ra_pre),
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=3,
                    arrowsize=2.0, arrowwidth=3.5, arrowcolor=col,
                    text="",
                )

    # ── Step 3: orbit point markers ──────────────────────────────────────────────
    # Each leg contributes two orbit states:
    #   triangle (dep): sub-arc 1 departure orbit, lies on departure body's contour
    #   circle   (arr): Lambert arc, lies on arrival body's contour
    orbit_pts = []  # (rp, ra, label, col, sym)

    for k, leg in enumerate(legs_data):
        dep = leg["body_dep"]
        arr = leg["body_arr"]
        col = OVERLAY_COLORS[k % len(OVERLAY_COLORS)]

        if has_exact:
            rp_dep = leg["rp_dep_au"]
            ra_dep = leg["ra_dep_au"]
            rp_lam = leg["rp_lambert_au"]
            ra_lam = leg["ra_lambert_au"]
        else:
            prev_b = legs_data[k - 1]["body_dep"] if k > 0 else None
            next_b = legs_data[k + 1]["body_arr"] if k < len(legs_data) - 1 else None
            dep_pt = approximate_orbit_point(leg["vinf_dep_ms"], dep,
                                             is_arrival=False, next_body=arr)
            arr_pt = approximate_orbit_point(leg["vinf_arr_ms"], arr,
                                             is_arrival=True, prev_body=dep, next_body=next_b)
            rp_dep, ra_dep = dep_pt if dep_pt else (None, None)
            rp_lam, ra_lam = arr_pt if arr_pt else (None, None)

        if rp_dep and ra_dep and math.isfinite(ra_dep) and ra_dep > rp_dep > 0:
            orbit_pts.append((rp_dep, ra_dep,
                               f"L{k} dep ({dep}) Rp={rp_dep:.3f} Ra={ra_dep:.3f} AU",
                               col, "triangle-up"))
        if rp_lam and ra_lam and math.isfinite(ra_lam) and ra_lam > rp_lam > 0:
            orbit_pts.append((rp_lam, ra_lam,
                               f"L{k} arr ({arr}) Rp={rp_lam:.3f} Ra={ra_lam:.3f} AU",
                               col, "circle"))

    for (rp, ra, label, col, sym) in orbit_pts:
        size = 13 if sym == "circle" else 11
        fig.add_trace(go.Scatter(
            x=[rp], y=[ra],
            mode="markers",
            marker=dict(color=col, size=size, symbol=sym,
                        line=dict(color="white", width=2.0)),
            name=label,
            showlegend=True,
            hovertemplate=f"Rp={rp:.4f} AU, Ra={ra:.4f} AU<extra></extra>",
        ))

    # ── Step 4: DSM jump lines (propulsive burns) ────────────────────────────────
    # Each leg has a DSM that jumps the orbit from the Keplerian departure arc
    # (rp_dep, ra_dep) to the Lambert arc (rp_lambert, ra_lambert).
    # go.Scatter dashed lines render reliably on Plotly log axes; pure
    # fig.add_annotation arrow-only calls without text can silently not render.
    if has_exact:
        for k, leg in enumerate(legs_data):
            rp0 = leg["rp_dep_au"]
            ra0 = leg["ra_dep_au"]
            rp1 = leg["rp_lambert_au"]
            ra1 = leg["ra_lambert_au"]
            dv  = leg["dv_dsm_ms"]

            if not all(v and v > 0 and math.isfinite(v) for v in [rp0, ra0, rp1, ra1]):
                continue

            burn_col = "#ff5555"

            # Dashed line from departure orbit → Lambert arc orbit.
            # No separate endpoint marker: the Lambert-arc end is already
            # shown as the L{k} arr circle in Step 3.
            fig.add_trace(go.Scatter(
                x=[rp0, rp1], y=[ra0, ra1],
                mode="lines",
                line=dict(color=burn_col, width=3, dash="dash"),
                name=f"DSM {k}  {dv:.0f} m/s",
                showlegend=True,
                hovertemplate=(
                    f"<b>DSM {k}</b>  ΔV = {dv:.0f} m/s<br>"
                    f"Rp {rp0:.3f}→{rp1:.3f} AU<br>"
                    f"Ra {ra0:.3f}→{ra1:.3f} AU<extra></extra>"
                ),
            ))
            # ΔV label: annotation with log10 coords (matches log-axis range space)
            mid_rp = math.sqrt(rp0 * rp1)
            mid_ra = math.sqrt(ra0 * ra1)
            fig.add_annotation(
                x=math.log10(mid_rp), y=math.log10(mid_ra),
                xref="x", yref="y",
                text=f"DSM {k}  {dv:.0f} m/s",
                showarrow=False,
                font=dict(color=burn_col, size=11, family="monospace"),
                bgcolor="#111111",
                bordercolor=burn_col,
                borderwidth=1,
                xanchor="left",
                yanchor="middle",
            )

# ── Auto-adjust axis range to show the full trajectory when legs data present ──
if legs_data and legs_data[0]["rp_dep_au"] is not None:
    all_ra = [leg["ra_dep_au"]      for leg in legs_data if leg["ra_dep_au"]      and math.isfinite(leg["ra_dep_au"])]
    all_ra += [leg["ra_lambert_au"] for leg in legs_data if leg["ra_lambert_au"]  and math.isfinite(leg["ra_lambert_au"])]
    all_rp = [leg["rp_dep_au"]      for leg in legs_data if leg["rp_dep_au"]      and leg["rp_dep_au"] > 0]
    all_rp += [leg["rp_lambert_au"] for leg in legs_data if leg["rp_lambert_au"]  and leg["rp_lambert_au"] > 0]
    if all_ra and not args.ra_max:
        ra_max = max(all_ra) * 1.4   # 40% headroom above the highest orbit state
    if all_rp and not args.rp_min:
        rp_min = min(all_rp) * 0.80  # 20% margin below the lowest perihelion

# ── Layout ────────────────────────────────────────────────────────────────────

body_str = " / ".join(bodies)
title_str = f"Tisserand Graph — {body_str}"
if legs_data:
    seq_str = " -> ".join(
        [legs_data[0]["body_dep"]] + [l["body_arr"] for l in legs_data]
    )
    title_str += f"  --  {seq_str}"

fig.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=TEXT, family="monospace", size=12),
    title=dict(text=title_str, font=dict(color="#00d4ff", size=17), x=0.5),
    xaxis=dict(
        title="Perihelion Radius Rp [AU]",
        type="log",
        range=[math.log10(rp_min), math.log10(rp_max)],
        gridcolor=GRID, color="#888",
        tickformat=".2g",
    ),
    yaxis=dict(
        title="Aphelion Radius Ra [AU]",
        type="log",
        range=[math.log10(max(ra_min, 0.04)), math.log10(ra_max)],
        gridcolor=GRID, color="#888",
        tickformat=".2g",
    ),
    height=780, width=950,
    legend=dict(
        bgcolor="#111", bordercolor="#333", borderwidth=1,
        font=dict(size=10), x=1.01, y=1.0,
        xanchor="left", yanchor="top",
    ),
    annotations=[
        dict(
            x=0.01, y=0.01, xref="paper", yref="paper",
            text="Contour labels = v∞ [km/s]  ·  diagonal = circular orbits",
            showarrow=False,
            font=dict(color="#555", size=10),
            xanchor="left",
        )
    ],
)

# ── Save ──────────────────────────────────────────────────────────────────────

out_path = (args.output if args.output else
            str(Path("out") / "tisserand_graph.html"))
if args.legs_csv:
    mission = Path(args.legs_csv).parent.name
    out_path = str(Path("out") / mission / "tisserand.html")

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
fig.write_html(out_path, include_plotlyjs=True)
print(f"Saved: {out_path}")
webbrowser.open(out_path)
