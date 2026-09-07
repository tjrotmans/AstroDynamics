"""
MGA ballistic grid-scan porkchop -- Phase 9w-iv.

Reads mga_window_scan.csv (written by the `mga-scan` CLI subcommand,
see mga_scan_run.rs) and produces two panels:

  1. 2D porkchop: departure date (x) x total time-of-flight (y), coloured by
     the best (minimum) total ballistic cost -- raw departure v_inf + sum of
     powered-flyby matching burns + arrival v_inf -- among all TOF splits
     found feasible for that (departure date, total TOF) cell. Arrival v_inf
     is reported separately (not priced into a burn here -- see
     plot_direct_vs_mga_porkchop.py for the full accounting used in the
     decision-gate comparison), so the colour scale is v_inf_dep + flyby burns
     only, annotated with the arrival v_inf at the best cell.
  2. Collapsed 1D window curve: best cost vs. departure date only (minimum
     over every TOF found for that date), with local minima annotated --
     this is the actual "launch window" a mission planner reads off.

Usage (from MissionPlanner/ directory):
  python plot/plot_mga_porkchop.py veega_scan_scratch
  python plot/plot_mga_porkchop.py veega_scan_scratch --csv out/other/mga_window_scan.csv
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
ACCENT = "#00d4ff"

# JD 2451544.5 = MJD2000 epoch = 2000-01-01 00:00 UTC.
MJD2000_EPOCH_UNIX_S = (2_451_544.5 - 2_440_587.5) * 86_400.0


def mjd2000_to_datetime(mjd2000: pd.Series) -> pd.Series:
    unix_s = mjd2000 * 86_400.0 + MJD2000_EPOCH_UNIX_S
    return pd.to_datetime(unix_s, unit="s")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] CSV not found: {path}", file=sys.stderr)
        print("Run: cargo run -p mission_planner -- mga-scan config/<mission>.toml", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"dep_mjd2000", "vinf_dep_ms", "vinf_arr_ms", "sum_flyby_dv_ms", "total_tof_days"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns in CSV: {missing}", file=sys.stderr)
        sys.exit(1)
    return df


def compute_cost(df: pd.DataFrame, objective: str) -> pd.Series:
    """cost_ms per the mission-type convention: Flyby has no arrival burn
    (departure v_inf + flyby matching burns only); Orbit adds the real
    vis-viva capture burn (capture_dv_ms, written by mga-scan only when
    [trajectory.capture].target_orbit_radius_m is configured)."""
    base = df["vinf_dep_ms"] + df["sum_flyby_dv_ms"]
    if objective == "flyby":
        return base
    if "capture_dv_ms" not in df.columns or df["capture_dv_ms"].isna().all():
        print(
            "[ERROR] --objective orbit requires a capture_dv_ms column with real values -- "
            "rerun mga-scan against a config with [trajectory.capture].target_orbit_radius_m set.",
            file=sys.stderr,
        )
        sys.exit(1)
    return base + df["capture_dv_ms"].fillna(0.0)


def build_figure(df: pd.DataFrame, mission: str, tof_bin_days: float, objective: str = "flyby") -> go.Figure:
    df = df.copy()
    df["cost_ms"] = compute_cost(df, objective)
    df["dep_date"] = mjd2000_to_datetime(df["dep_mjd2000"])

    # Bin TOF for the 2D grid -- the scan's TOF grid points don't line up
    # across different departure dates (each is an independent per-leg grid
    # in physical days), so a coarse common bin turns the scatter into a
    # regular porkchop grid.
    df["tof_bin_days"] = (df["total_tof_days"] / tof_bin_days).round() * tof_bin_days

    grid = (
        df.groupby(["dep_mjd2000", "tof_bin_days"])
        .agg(cost_ms=("cost_ms", "min"), vinf_arr_ms=("vinf_arr_ms", "min"))
        .reset_index()
    )
    dep_dates_feasible = np.sort(grid["dep_mjd2000"].unique())
    tof_bins_unique = np.sort(grid["tof_bin_days"].unique())

    # Reconstruct the TRUE uniform departure-date grid (not just the dates
    # that happen to have a feasible branch) so Plotly's Heatmap renders
    # real gaps as gaps, not as one wide solid block stretched to the next
    # feasible date. Plotly spaces heatmap columns by treating each x value's
    # cell as extending to the midpoint with its neighbours -- with only the
    # sparse feasible dates as x, a single feasible date surrounded by mostly
    # infeasible ones renders as a multi-week/month solid colour block,
    # falsely implying continuous coverage (found, real user-
    # visible artifact on a VEEGA scan with ~3% date-feasibility). The scan's
    # own configured step isn't in the CSV, so it's inferred as the smallest
    # positive gap between consecutive feasible dates -- exact when at least
    # two feasible dates are adjacent on the native grid (the common case),
    # a slight over-estimate otherwise (still far better than not gridding
    # at all).
    if len(dep_dates_feasible) > 1:
        gaps = np.diff(dep_dates_feasible)
        step_days = gaps[gaps > 1e-6].min() if (gaps > 1e-6).any() else 1.0
    else:
        step_days = 1.0
    n_steps = int(round((dep_dates_feasible[-1] - dep_dates_feasible[0]) / step_days)) + 1
    dep_dates_full = dep_dates_feasible[0] + np.arange(n_steps) * step_days

    z = np.full((len(tof_bins_unique), len(dep_dates_full)), np.nan)
    # Snap each feasible date to its nearest slot on the reconstructed grid
    # (handles float round-off from the step-inference above).
    dep_idx_map = {v: int(round((v - dep_dates_feasible[0]) / step_days)) for v in dep_dates_feasible}
    tof_idx = {v: i for i, v in enumerate(tof_bins_unique)}
    for _, row in grid.iterrows():
        z[tof_idx[row["tof_bin_days"]], dep_idx_map[row["dep_mjd2000"]]] = row["cost_ms"] / 1000.0

    dep_dates_dt = mjd2000_to_datetime(pd.Series(dep_dates_full))

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.12,
        shared_xaxes=True,
        subplot_titles=[
            f"{mission}: departure v∞ + flyby burns"
            + (" + capture burn" if objective == "orbit" else "")
            + " [km/s] (colour) vs. departure date x total TOF",
            "Best cost vs. departure date (minimum over all TOF splits found)",
        ],
    )

    fig.add_trace(
        go.Heatmap(
            x=dep_dates_dt,
            y=tof_bins_unique,
            z=z,
            colorscale="Viridis",
            colorbar=dict(title="ΔV [km/s]", x=1.02),
            hovertemplate="dep %{x|%Y-%m-%d}<br>TOF %{y:.0f} d<br>cost %{z:.3f} km/s<extra></extra>",
        ),
        row=1, col=1,
    )

    # Best point overall.
    best = df.loc[df["cost_ms"].idxmin()]
    fig.add_trace(
        go.Scatter(
            x=[best["dep_date"]], y=[best["total_tof_days"]],
            mode="markers+text",
            marker=dict(color="red", size=12, symbol="star"),
            text=[f"best: {best['cost_ms']/1000:.2f} km/s<br>v∞_arr {best['vinf_arr_ms']/1000:.2f} km/s"],
            textposition="top center",
            textfont=dict(color="red", size=10),
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ── panel 2: collapsed 1D window curve ──────────────────────────────────
    window = df.groupby("dep_mjd2000")["cost_ms"].min().reset_index()
    window["dep_date"] = mjd2000_to_datetime(window["dep_mjd2000"])
    window = window.sort_values("dep_mjd2000")
    y_kms = window["cost_ms"].to_numpy() / 1000.0

    fig.add_trace(
        go.Scatter(
            x=window["dep_date"], y=y_kms,
            mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=4),
            name="best cost",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Local minima: points lower than both neighbours.
    minima_idx = [
        i for i in range(1, len(y_kms) - 1)
        if y_kms[i] < y_kms[i - 1] and y_kms[i] < y_kms[i + 1]
    ]
    if len(y_kms) > 0:
        # Endpoints count as minima only if they're the global minimum region --
        # skip edge-pinning ambiguity per the project's porkchop-window lesson
        # (an edge-pinned minimum usually means the scan window is mis-centered,
        # not a real optimum) and just annotate interior minima plus the global best.
        for i in minima_idx:
            fig.add_annotation(
                x=window["dep_date"].iloc[i], y=y_kms[i],
                text=f"{y_kms[i]:.2f}",
                showarrow=True, arrowhead=2, arrowcolor="#888",
                font=dict(color="#ffcf00", size=9),
                ax=0, ay=-25,
                row=2, col=1,
            )
    global_best_i = int(np.argmin(y_kms)) if len(y_kms) else None
    if global_best_i is not None and global_best_i not in minima_idx:
        fig.add_annotation(
            x=window["dep_date"].iloc[global_best_i], y=y_kms[global_best_i],
            text=f"best {y_kms[global_best_i]:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="red",
            font=dict(color="red", size=10),
            ax=0, ay=-30,
            row=2, col=1,
        )

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="monospace"),
        title=dict(text=f"MGA Ballistic Grid Scan — {mission}", x=0.5, font=dict(size=16)),
        height=850,
    )
    for axis in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{axis: dict(gridcolor=GRID, zerolinecolor="#333", color=TEXT)})
    fig.update_yaxes(title_text="Total TOF [days]", row=1, col=1)
    fig.update_xaxes(title_text="Departure date", row=2, col=1)
    fig.update_yaxes(title_text="Best cost [km/s]", row=2, col=1)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the Phase 9w MGA ballistic grid-scan porkchop")
    parser.add_argument("mission", nargs="?", default=None,
                         help="Mission name (reads out/<mission>/mga_window_scan.csv); used as the "
                              "plot's own title. If omitted, derived from --csv's parent directory name "
                              "instead of a fixed default -- a hardcoded default here previously caused "
                              "every plot generated via --csv to be mistitled with an unrelated mission "
                              "name (found 2026-07-28), since the title always came from this argument "
                              "regardless of which CSV was actually loaded.")
    parser.add_argument("--csv", default=None, help="Explicit CSV path (overrides mission-name default)")
    parser.add_argument("--tof-bin-days", type=float, default=30.0,
                         help="TOF bin width [days] for the 2D grid (default 30)")
    parser.add_argument("--objective", choices=["flyby", "orbit"], default="flyby",
                         help="flyby (default): cost = departure v_inf + flyby burns, no arrival burn. "
                              "orbit: adds the real capture burn (requires capture_dv_ms in the CSV, "
                              "i.e. the scan config had [trajectory.capture].target_orbit_radius_m set).")
    parser.add_argument("--output", default=None, help="Save HTML to this path instead of showing")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        mission = args.mission or csv_path.parent.name
    else:
        mission = args.mission or "veega_scan_scratch"
        csv_path = Path("out") / mission / "mga_window_scan.csv"
    df = load_csv(csv_path)
    print(f"Loaded {len(df)} feasible branches from {csv_path}  (mission label: {mission})")

    fig = build_figure(df, mission, args.tof_bin_days, args.objective)

    if args.output:
        fig.write_html(args.output)
        print(f"Saved: {args.output}")
    else:
        out_html = csv_path.parent / "mga_porkchop.html"
        fig.write_html(out_html)
        print(f"Saved: {out_html}")
        fig.show()


if __name__ == "__main__":
    main()
