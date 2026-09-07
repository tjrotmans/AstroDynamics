"""
plot_wsb_phases.py — Phase-by-phase diagnostic visualisation of wsb_search.

Produces one HTML file per phase, all in out/wsb/.

  wsb_phase1_capture.html  — Phase 1: backward arcs from Moon, coloured by
                              β-filter pass/fail and beta angle.  Shows up to
                              MAX_TRAJ trajectories for readability.
                              Earth-Moon rotating frame.

  wsb_phase2_alpha.html    — Phase 2: α-angle geometry.  One point per surviving
                              backward solution projected to its Earth-perigee IC,
                              coloured by α quadrant (Q2/Q4 valid, Q1/Q3 rejected).

  wsb_phase3_captures.html — Phase 3+4: forward transfer arcs from Earth to Moon,
                              coloured by estimated capture orbits.

Reads:
  out/wsb/backward_solutions.csv   — written by wsb_search Phase 1 output
  out/wsb/forward_screen.csv       — written by wsb_search Phase 2 output
  out/wsb/blt_candidates.csv       — written by wsb_search Phase 3+4 output
  out/wsb/family_analysis.csv      — Pareto-ranked candidates

Run:  python plot/plot_wsb_phases.py
"""

import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = pathlib.Path(__file__).parent.parent
OUT  = ROOT / "out" / "wsb"

# ── Constants ──────────────────────────────────────────────────────────────────
MU      = 0.01215565
X_E     = -MU
X_M     = 1.0 - MU
R_HILL  = (MU / 3.0) ** (1.0 / 3.0)
L_KM    = 384_400.0
T_STAR  = 375_700.0
R_MOON  = 1_737.4 / L_KM  # nd

# Maximum backward trajectories to show (clarity over completeness)
MAX_BWD_TRAJ   = 80
MAX_FWD_TRAJ   = 60

# ── Helpers ────────────────────────────────────────────────────────────────────

def nd_to_km(x): return x * L_KM

def circle_xy(cx, cy, r, n=120):
    t = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(t), cy + r * np.sin(t)

def body_traces(show_hill=True):
    """Return Earth, Moon, Hill-sphere traces in rotating frame [km]."""
    traces = []
    # Earth
    ex, ey = circle_xy(X_E * L_KM, 0, 6_371.0)
    traces.append(go.Scatter(x=ex, y=ey, mode="lines",
        line=dict(color="#1a73e8", width=2), name="Earth", showlegend=True))
    # Moon
    mx, my = circle_xy(X_M * L_KM, 0, 1_737.4)
    traces.append(go.Scatter(x=mx, y=my, mode="lines",
        line=dict(color="#aaaaaa", width=2), name="Moon", showlegend=True))
    if show_hill:
        hx, hy = circle_xy(X_M * L_KM, 0, R_HILL * L_KM)
        traces.append(go.Scatter(x=hx, y=hy, mode="lines",
            line=dict(color="#aaaaaa", width=1, dash="dot"),
            name="Hill sphere", showlegend=True))
    return traces

BETA_COLOR = {
    True:  "#2ecc71",   # pass — green
    False: "#e74c3c",   # fail — red
}

QUAD_COLOR = {
    "Q2": "#f39c12",   # valid
    "Q4": "#9b59b6",   # valid
    "Q1": "#95a5a6",   # rejected
    "Q3": "#95a5a6",   # rejected
}

def alpha_to_quad(alpha_deg):
    a = alpha_deg % 360.0
    if 70 <= a <= 180:   return "Q2"
    if 180 < a <= 360:   return "Q4"
    if a < 70:           return "Q1"
    return "Q3"

def _write_placeholder(path: pathlib.Path, title: str, body: str):
    """Write a minimal informative HTML page when data is absent."""
    path.write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        f"<style>body{{font-family:sans-serif;background:#0e1117;color:#e0e0e0;"
        f"display:flex;align-items:center;justify-content:center;height:100vh;margin:0}}"
        f"div{{text-align:center;max-width:600px}}"
        f"h2{{color:#aaa}}p{{color:#888;line-height:1.6}}</style></head><body>"
        f"<div><h2>{title}</h2><p>{body}</p></div></body></html>"
    )
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — backward arcs from Moon
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase1():
    bwd_path = OUT / "backward_solutions.csv"
    if not bwd_path.exists():
        print(f"  Phase 1: {bwd_path} not found — run wsb_search first"); return

    df = pd.read_csv(bwd_path)
    # Drop sentinel NaN rows (end-of-arc markers)
    df = df.dropna(subset=["x_nd"])
    sol_ids = df["sol_id"].unique()
    n_total = len(sol_ids)

    if n_total == 0:
        msg = ("wsb_search Phase 1 found 0 backward solutions.<br>"
               "The β-angle pre-filter or Earth perigee filter rejected all backward arcs.<br>"
               "Re-run <code>wsb_search</code> (or the full pipeline) to regenerate.")
        print(f"  Phase 1: no data — writing placeholder")
        _write_placeholder(OUT / "wsb_phase1_capture.html",
                           "Phase 1: Backward propagation — no data", msg)
        return

    # Sample up to MAX_BWD_TRAJ solutions for readability
    rng = np.random.default_rng(42)
    if n_total > MAX_BWD_TRAJ:
        chosen_ids = rng.choice(sol_ids, MAX_BWD_TRAJ, replace=False)
    else:
        chosen_ids = sol_ids

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Rotating frame [km] — Phase 1 backward arcs (sample)",
            "β angle distribution",
        ),
        column_widths=[0.65, 0.35],
    )

    # ── Left panel: trajectories ──────────────────────────────────────────────
    for tr in body_traces(show_hill=True):
        fig.add_trace(tr, row=1, col=1)

    beta_pass_shown = set()
    for sid in chosen_ids:
        arc = df[df["sol_id"] == sid]
        # beta pass/fail determined by beta_deg being in valid window
        beta = arc["beta_deg"].iloc[0]
        passed = (60 <= beta <= 150) or (240 <= beta <= 330)
        key = passed
        color = BETA_COLOR[passed]
        label = "β pass" if passed else "β fail"
        show = label not in beta_pass_shown
        if show: beta_pass_shown.add(label)
        x_km = arc["x_nd"] * L_KM
        y_km = arc["y_nd"] * L_KM
        fig.add_trace(go.Scatter(
            x=x_km, y=y_km, mode="lines",
            line=dict(color=color, width=0.8),
            opacity=0.55,
            name=label, showlegend=show,
            hovertemplate=(
                f"sol={sid}  β={beta:.1f}°<br>"
                "x=%{x:.0f} km  y=%{y:.0f} km<extra></extra>"
            ),
        ), row=1, col=1)

    fig.update_xaxes(title_text="x [km] (rotating frame)", row=1, col=1)
    fig.update_yaxes(title_text="y [km]", scaleanchor="x", scaleratio=1, row=1, col=1)

    # ── Right panel: β histogram ──────────────────────────────────────────────
    all_beta = df.drop_duplicates("sol_id")["beta_deg"].values
    fig.add_trace(go.Histogram(
        x=all_beta, nbinsx=36,
        marker_color="#3498db", opacity=0.8,
        name="β distribution", showlegend=False,
    ), row=1, col=2)
    # Valid windows shading
    for lo, hi in [(60, 150), (240, 330)]:
        fig.add_vrect(x0=lo, x1=hi, fillcolor="green", opacity=0.1,
                      line_width=0, row=1, col=2)
    fig.update_xaxes(title_text="β angle [deg]", row=1, col=2)
    fig.update_yaxes(title_text="count", row=1, col=2)

    fig.update_layout(
        title=dict(
            text=(
                f"Phase 1: Backward propagation from Moon "
                f"({n_total} arcs total, showing {len(chosen_ids)})<br>"
                "<sub>Green shading = valid β windows [60°–150°] ∪ [240°–330°]  "
                "| Rotating frame, Earth-Moon normalised units</sub>"
            ),
            x=0.5,
        ),
        height=620,
        legend=dict(x=0.01, y=0.99),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
    )
    out_path = OUT / "wsb_phase1_capture.html"
    fig.write_html(str(out_path))
    print(f"  Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — α-angle screen
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase2():
    scr_path = OUT / "forward_screen.csv"
    if not scr_path.exists():
        print(f"  Phase 2: {scr_path} not found — run wsb_search first"); return

    df = pd.read_csv(scr_path)

    if df.empty or len(df) == 0:
        msg = ("Phase 1 produced 0 backward solutions, so Phase 2 had nothing to screen.<br>"
               "Re-run <code>wsb_search</code> after fixing Phase 1.")
        print(f"  Phase 2: no data — writing placeholder")
        _write_placeholder(OUT / "wsb_phase2_alpha.html",
                           "Phase 2: α-angle screen — no data", msg)
        return

    # Assign quadrant
    df["quad"] = df["alpha_deg"].apply(alpha_to_quad)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "α-angle geometry — Earth-perigee injection points",
            "α vs β scatter",
        ),
        column_widths=[0.55, 0.45],
    )

    # ── Left: injection points in rotating frame ──────────────────────────────
    for tr in body_traces(show_hill=False):
        fig.add_trace(tr, row=1, col=1)

    for quad, grp in df.groupby("quad"):
        valid = quad in ("Q2", "Q4")
        fig.add_trace(go.Scatter(
            x=grp["x_nd"] * L_KM,
            y=grp["y_nd"] * L_KM,
            mode="markers",
            marker=dict(color=QUAD_COLOR[quad], size=7,
                        symbol="circle" if valid else "x",
                        opacity=0.75),
            name=f"{quad} ({'valid' if valid else 'rejected'})",
            hovertemplate=(
                f"quad={quad}<br>"
                "α=%{customdata[0]:.1f}°  β=%{customdata[1]:.1f}°<extra></extra>"
            ),
            customdata=grp[["alpha_deg", "beta_deg"]].values,
        ), row=1, col=1)

    fig.update_xaxes(title_text="x [km] (rotating frame)", row=1, col=1)
    fig.update_yaxes(title_text="y [km]", row=1, col=1)

    # ── Right: α vs β ─────────────────────────────────────────────────────────
    for quad, grp in df.groupby("quad"):
        valid = quad in ("Q2", "Q4")
        fig.add_trace(go.Scatter(
            x=grp["beta_deg"], y=grp["alpha_deg"],
            mode="markers",
            marker=dict(color=QUAD_COLOR[quad], size=7, opacity=0.7,
                        symbol="circle" if valid else "x"),
            name=quad, showlegend=False,
            hovertemplate="β=%{x:.1f}°  α=%{y:.1f}°<extra></extra>",
        ), row=1, col=2)

    # Valid α regions
    for lo, hi in [(70, 180), (180, 360)]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor="green", opacity=0.08,
                      line_width=0, row=1, col=2)
    fig.update_xaxes(title_text="β angle [deg]", row=1, col=2)
    fig.update_yaxes(title_text="α angle [deg]", row=1, col=2)

    n_valid = (df["quad"].isin(["Q2", "Q4"])).sum()
    fig.update_layout(
        title=dict(
            text=(
                f"Phase 2: α-angle screen — {len(df)} backward solutions, "
                f"{n_valid} pass (Q2∪Q4)<br>"
                "<sub>Valid: α ∈ [70°,180°] (Q2) ∪ (180°,360°] (Q4) "
                "— apogee direction relative to Sun</sub>"
            ),
            x=0.5,
        ),
        height=580,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
    )
    out_path = OUT / "wsb_phase2_alpha.html"
    fig.write_html(str(out_path))
    print(f"  Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3+4 — forward transfer arcs
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase3():
    cand_path = OUT / "blt_candidates.csv"
    if not cand_path.exists():
        print(f"  Phase 3: {cand_path} not found — run wsb_search first"); return

    traj_df = pd.read_csv(cand_path)
    traj_df = traj_df.dropna(subset=["x_nd"])

    if traj_df.empty:
        print(f"  Phase 3: {cand_path} has no trajectory data"); return

    # blt_candidates.csv holds the top-N by SCORE (not by Pareto rank) —
    # all of them are the "best" solutions found by the search.
    # Sort by cand_id so #1 is always the top-scored solution.
    cand_ids = np.sort(traj_df["cand_id"].unique())
    n_total  = len(cand_ids)
    chosen_ids = cand_ids  # always show all (typically only 5)

    # Score-rank colours: candidate 1 = best score, 2 = second-best, etc.
    RANK_COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]

    def score_rank_color(position):
        if position < len(RANK_COLORS):
            return RANK_COLORS[position]
        return "#555555"

    fig = go.Figure()
    for tr in body_traces(show_hill=True):
        fig.add_trace(tr)

    # Draw transfers — truncate each arc at Hill sphere entry for cleaner view
    moon_x_nd = X_M
    for pos, cid in enumerate(chosen_ids):
        arc = traj_df[traj_df["cand_id"] == cid].copy()
        fam = arc["family"].iloc[0] if "family" in arc.columns else ""

        # Truncate at Hill sphere entry
        dx = arc["x_nd"] - moon_x_nd
        r_moon = np.sqrt(dx**2 + arc["y_nd"]**2)
        inside = r_moon < R_HILL
        if inside.any():
            first_in = inside.values.argmax()
            arc = arc.iloc[:first_in + 1]

        color = score_rank_color(pos)
        label = f"cand {cid} ({fam})"
        fig.add_trace(go.Scatter(
            x=arc["x_nd"] * L_KM,
            y=arc["y_nd"] * L_KM,
            mode="lines",
            line=dict(color=color, width=1.5),
            opacity=0.85,
            name=label,
            hovertemplate=(
                f"cand={cid}  family={fam}<br>"
                "x=%{x:.0f} km<br>y=%{y:.0f} km<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"Phase 3+4: Best WSB transfer arcs — "
                f"top {n_total} candidates by score (truncated at Hill sphere entry)<br>"
                "<sub>Colour = candidate rank by score  "
                "| Earth-Moon rotating frame [km]</sub>"
            ),
            x=0.5,
        ),
        xaxis_title="x [km] (rotating frame)",
        yaxis=dict(title="y [km]", scaleanchor="x", scaleratio=1),
        height=650,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        legend=dict(x=0.01, y=0.99),
    )
    out_path = OUT / "wsb_phase3_captures.html"
    fig.write_html(str(out_path))
    print(f"  Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("── Phase diagnostic plots ─────────────────────────────────────────")
    plot_phase1()
    plot_phase2()
    plot_phase3()
    print("Done.")
