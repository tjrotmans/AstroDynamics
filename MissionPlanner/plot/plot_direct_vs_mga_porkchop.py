"""
Direct-transfer vs. best-ballistic-MGA porkchop -- the Phase 9w decision gate.

Reads two mga_window_scan.csv files (see plot_mga_porkchop.py / mga-scan CLI):
a direct (zero-flyby, single-leg) scan and a multi-flyby MGA scan, both over
the SAME departure horizon and A->B bodies. Renders two aligned panels --
same departure-date x-axis, same ΔV colour scale -- so the visual gap between
panels at any date IS the MGA benefit. Where the gap is small, run the
simpler direct GA/PSO/DiffCorrection search instead of the MGA pipeline; this
is the physics-driven version of the per-mission-type method-gating the
Product Vision calls for.

Usage (from MissionPlanner/ directory):
  python plot/plot_direct_vs_mga_porkchop.py veega_direct_scratch veega_scan_scratch
  python plot/plot_direct_vs_mga_porkchop.py --direct-csv out/a/mga_window_scan.csv --mga-csv out/b/mga_window_scan.csv
"""

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BG = "#0f0f0f"
GRID = "#222"
TEXT = "#e0e0e0"

MJD2000_EPOCH_UNIX_S = (2_451_544.5 - 2_440_587.5) * 86_400.0


def mjd2000_to_datetime(mjd2000: pd.Series) -> pd.Series:
    unix_s = mjd2000 * 86_400.0 + MJD2000_EPOCH_UNIX_S
    return pd.to_datetime(unix_s, unit="s")


def load_csv(path: Path, label: str, objective: str = "flyby") -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] {label} CSV not found: {path}", file=sys.stderr)
        print("Run: cargo run -p mission_planner -- mga-scan config/<mission>.toml", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"dep_mjd2000", "vinf_dep_ms", "sum_flyby_dv_ms"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns in {label} CSV: {missing}", file=sys.stderr)
        sys.exit(1)
    df = df.copy()
    base = df["vinf_dep_ms"] + df["sum_flyby_dv_ms"]
    if objective == "orbit":
        if "capture_dv_ms" not in df.columns or df["capture_dv_ms"].isna().all():
            print(
                f"[ERROR] --objective orbit requires a capture_dv_ms column with real values in the "
                f"{label} CSV -- rerun mga-scan against a config with "
                f"[trajectory.capture].target_orbit_radius_m set.",
                file=sys.stderr,
            )
            sys.exit(1)
        base = base + df["capture_dv_ms"].fillna(0.0)
    df["cost_ms"] = base
    df["dep_date"] = mjd2000_to_datetime(df["dep_mjd2000"])
    return df


def window_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to best cost per departure date (min over all TOF splits)."""
    w = df.groupby("dep_mjd2000")["cost_ms"].min().reset_index()
    w["dep_date"] = mjd2000_to_datetime(w["dep_mjd2000"])
    return w.sort_values("dep_mjd2000")


def build_figure(direct: pd.DataFrame, mga: pd.DataFrame, direct_label: str, mga_label: str) -> go.Figure:
    direct_w = window_curve(direct)
    mga_w = window_curve(mga)

    # Shared colour scale across both panels, per the decision-gate design --
    # the visual gap at a fixed colour range is what makes "is MGA worth it"
    # readable at a glance.
    vmin = min(direct_w["cost_ms"].min(), mga_w["cost_ms"].min()) / 1000.0
    vmax = max(direct_w["cost_ms"].max(), mga_w["cost_ms"].max()) / 1000.0

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.35, 0.35, 0.30],
        vertical_spacing=0.10,
        subplot_titles=[
            f"Direct transfer — {direct_label}",
            f"Best MGA chain — {mga_label}",
            "Gap = MGA benefit (direct − best MGA, positive = MGA cheaper)",
        ],
        shared_xaxes=True,
    )

    def add_window_trace(w: pd.DataFrame, row: int, color: str):
        fig.add_trace(
            go.Scatter(
                x=w["dep_date"], y=w["cost_ms"] / 1000.0,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(
                    size=6, color=w["cost_ms"] / 1000.0,
                    colorscale="Viridis", cmin=vmin, cmax=vmax,
                    showscale=(row == 1),
                    colorbar=dict(title="ΔV [km/s]", x=1.02) if row == 1 else None,
                ),
                showlegend=False,
            ),
            row=row, col=1,
        )

    add_window_trace(direct_w, 1, "#ff6b35")
    add_window_trace(mga_w, 2, "#00d4ff")

    # ── gap panel: interpolate MGA's window curve onto direct's date grid ──
    # A strict nearest-match-within-tolerance join (the original approach)
    # can drop almost everything: direct and MGA scans satisfy different
    # feasibility constraints (MGA adds a powered-flyby ΔV cap), so their
    # feasible-date sets can end up largely disjoint even over the same
    # window/grid resolution -- confirmed on a real EVJ/Venus-flyby run
    #, where the tolerance join found zero matches at all.
    # Real linear interpolation instead: for every direct departure date,
    # interpolate MGA's cost from its two nearest bracketing dates.
    #
    # `np.interp`'s left/right=NaN only guards the OUTER edges (before MGA's
    # first date / after its last) -- it happily draws a straight line across
    # any INTERNAL gap in MGA's coverage, no matter how wide, and reports a
    # value there too. Found a date deep
    # inside a real MGA infeasibility gap (e.g. no feasible MGA branch for
    # months) still got a fabricated "comparison" value as long as it fell
    # between MGA's overall min and max dates. Fixed: after interpolating,
    # also compute the width of each point's actual bracketing pair of real
    # MGA dates and NaN out anything whose bracket is wider than a few native
    # grid steps -- a genuine gap, not a legitimate local interpolation.
    direct_sorted = direct_w.sort_values("dep_mjd2000").reset_index(drop=True)
    mga_sorted = mga_w.sort_values("dep_mjd2000").reset_index(drop=True)
    mga_dep = mga_sorted["dep_mjd2000"].to_numpy()
    direct_dep = direct_sorted["dep_mjd2000"].to_numpy()

    mga_interp_ms = np.interp(direct_dep, mga_dep, mga_sorted["cost_ms"], left=np.nan, right=np.nan)

    # Bracket width at each direct date: distance between the two real MGA
    # dates immediately surrounding it (0 if it coincides with a real MGA
    # date; undefined/huge at the outer edges, already NaN from above).
    mga_native_step = np.diff(mga_dep)
    mga_native_step = mga_native_step[mga_native_step > 1e-6].min() if len(mga_native_step) else 1.0
    max_bracket = 3.0 * mga_native_step  # generous but not "connect any two islands"
    hi_idx = np.clip(np.searchsorted(mga_dep, direct_dep), 1, len(mga_dep) - 1)
    bracket_width = mga_dep[hi_idx] - mga_dep[hi_idx - 1]

    gap_kms_all = (direct_sorted["cost_ms"].to_numpy() - mga_interp_ms) / 1000.0
    valid = ~np.isnan(gap_kms_all) & (bracket_width <= max_bracket)
    gap_dates = direct_sorted["dep_date"][valid]
    gap_kms = gap_kms_all[valid]

    fig.add_trace(
        go.Bar(
            x=gap_dates, y=gap_kms,
            marker_color=np.where(gap_kms >= 0, "#7fba00", "#ef5350"),
            showlegend=False,
            name="gap (linearly interpolated within MGA's date coverage)",
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line_color="#666", line_dash="dot", row=3, col=1)
    n_dropped = int((~valid).sum())
    if n_dropped:
        fig.add_annotation(
            text=f"{n_dropped}/{len(valid)} direct dates outside MGA's date coverage -- gap not computed there",
            xref="paper", yref="paper", x=0.5, y=-0.08, showarrow=False,
            font=dict(color="#888", size=9),
            row=3, col=1,
        )

    if len(gap_kms):
        best_gap_i = int(np.argmax(gap_kms))
        fig.add_annotation(
            x=gap_dates.iloc[best_gap_i], y=gap_kms[best_gap_i],
            text=f"max MGA benefit: {gap_kms[best_gap_i]:.2f} km/s",
            showarrow=True, arrowhead=2, arrowcolor="#7fba00",
            font=dict(color="#7fba00", size=10),
            ax=0, ay=-30,
            row=3, col=1,
        )

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="monospace"),
        title=dict(
            text="Direct vs. MGA — the Phase 9w decision gate",
            x=0.5, font=dict(size=16),
        ),
        height=950,
    )
    for axis in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
        fig.update_layout(**{axis: dict(gridcolor=GRID, zerolinecolor="#333", color=TEXT)})
    fig.update_yaxes(title_text="Cost [km/s]", row=1, col=1)
    fig.update_yaxes(title_text="Cost [km/s]", row=2, col=1)
    fig.update_yaxes(title_text="Gap [km/s]", row=3, col=1)
    fig.update_xaxes(title_text="Departure date", row=3, col=1)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct-vs-MGA porkchop decision gate (Phase 9w-v)"
    )
    parser.add_argument("direct_mission", nargs="?", default=None,
                         help="Mission name for the direct (zero-flyby) scan")
    parser.add_argument("mga_mission", nargs="?", default=None,
                         help="Mission name for the MGA (multi-flyby) scan")
    parser.add_argument("--direct-csv", default=None, help="Explicit direct-scan CSV path")
    parser.add_argument("--mga-csv", default=None, help="Explicit MGA-scan CSV path")
    parser.add_argument("--objective", choices=["flyby", "orbit"], default="flyby",
                         help="flyby (default): cost = departure v_inf + flyby burns, no arrival burn. "
                              "orbit: adds the real capture burn (requires capture_dv_ms in both CSVs).")
    parser.add_argument("--output", default=None, help="Save HTML to this path instead of showing")
    args = parser.parse_args()

    if args.direct_csv:
        direct_path = Path(args.direct_csv)
    elif args.direct_mission:
        direct_path = Path("out") / args.direct_mission / "mga_window_scan.csv"
    else:
        print("[ERROR] provide either direct_mission or --direct-csv", file=sys.stderr)
        sys.exit(1)

    if args.mga_csv:
        mga_path = Path(args.mga_csv)
    elif args.mga_mission:
        mga_path = Path("out") / args.mga_mission / "mga_window_scan.csv"
    else:
        print("[ERROR] provide either mga_mission or --mga-csv", file=sys.stderr)
        sys.exit(1)

    direct = load_csv(direct_path, "direct", args.objective)
    mga = load_csv(mga_path, "MGA", args.objective)
    print(f"Direct: {len(direct)} feasible branches from {direct_path}")
    print(f"MGA:    {len(mga)} feasible branches from {mga_path}")

    fig = build_figure(direct, mga, args.direct_mission or direct_path.parent.name,
                        args.mga_mission or mga_path.parent.name)

    if args.output:
        fig.write_html(args.output)
        print(f"Saved: {args.output}")
    else:
        out_html = mga_path.parent / "direct_vs_mga_porkchop.html"
        fig.write_html(out_html)
        print(f"Saved: {out_html}")
        fig.show()


if __name__ == "__main__":
    main()
