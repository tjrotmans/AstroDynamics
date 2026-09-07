"""
MissionPlanner Phase 9 optimization stage -- two outputs per run:

1. `<method>_optimize_trajectory.html` ("Screen 1") -- three trajectory
   panels: departure leg + parking-orbit ring (left), full heliocentric
   transfer (middle), arrival/capture leg (right) -- including the resulting
   captured orbit ring + capture burn vectors when this is an Orbit-targeting
   run that actually achieved a capture (nothing extra drawn for Flyby, or
   for an Orbit run whose best point only got close without capturing).

2. `<method>_optimize_stats.html` ("Screen 2") -- convergence (phase 1
   flyby-only, phase 2 real-objective) plus the population scatter: 2 panels
   (theta_burn/phi colored by departure dv, and by generation) for a Flyby
   run; 4 panels (those two, plus theta_arr/phi_arr -- the arrival-side
   mirror, see optimize.rs::arrival_burn_angles -- colored by total dv and by
   generation) for an Orbit run with real captured samples to show.

Phase 1 always searches for the best flyby/closest-approach first
(`flyby_only_fitness` in optimize.rs), regardless of `optimization.objective`
or whether the mission needs a real capture. Phase 2 then refines under the
real configured objective. PSO is not split into phases yet, so its stats
screen shows only a single convergence panel and no population scatter.

Reads: MissionPlanner/out/<mission>/optimize/{ga,pso}_convergence.csv
       MissionPlanner/out/<mission>/optimize/ga_phase1_convergence.csv        (GA only)
       MissionPlanner/out/<mission>/optimize/ga_population.csv               (GA only)
       MissionPlanner/out/<mission>/optimize/ga_population_arrival_sample.csv (GA + Orbit only, optional)
       MissionPlanner/out/<mission>/optimize/{ga,pso}_best.csv
       MissionPlanner/out/<mission>/optimize/{ga,pso}_trajectory.csv
       MissionPlanner/out/<mission>/optimize/{ga,pso}_departure_track.csv         (optional)
       MissionPlanner/out/<mission>/optimize/{ga,pso}_target_track.csv           (optional)
       MissionPlanner/out/<mission>/optimize/{ga,pso}_departure_geometry_orbit.csv  (optional)
       MissionPlanner/out/<mission>/optimize/{ga,pso}_arrival_geometry_orbit.csv    (optional)
       MissionPlanner/out/<mission>/optimize/{ga,pso}_arrival_geometry_vectors.csv  (optional)

Produces (into the same directory): <method>_optimize_trajectory.html, <method>_optimize_stats.html

Usage (from MissionPlanner/ directory):
  python plot/plot_optimize.py scratch_optimize_repro
"""

import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

mission = sys.argv[1] if len(sys.argv) > 1 else "mars_flyby"

ROOT = Path(__file__).parent.parent
DATA = ROOT / "out" / mission / "optimize"

candidates = [("GA", "generation"), ("PSO", "iteration")]
method, step_col, conv_path = (None,) * 3
for m, col in candidates:
    p = DATA / f"{m.lower()}_convergence.csv"
    if p.exists():
        method, step_col, conv_path = m, col, p
        break

if method is None:
    print(f"Not found: {DATA / 'ga_convergence.csv'} or {DATA / 'pso_convergence.csv'}")
    print("Run: cargo run --bin mission-planner --release -- optimize config/<mission>.toml")
    sys.exit(1)

prefix = method.lower()


def opt_csv(name):
    path = DATA / f"{prefix}_{name}.csv"
    return pd.read_csv(path) if path.exists() else None


conv = pd.read_csv(conv_path)
best = opt_csv("best").iloc[0]
traj = opt_csv("trajectory")
dep_track = opt_csv("departure_track")
target_track = opt_csv("target_track")
phase1_conv = opt_csv("phase1_convergence")
population = opt_csv("population")
if population is not None:
    population["fitness"] = pd.to_numeric(population.fitness, errors="coerce")  # "inf" -> NaN
dep_orbit_ring = opt_csv("departure_geometry_orbit")
arr_orbit_ring = opt_csv("arrival_geometry_orbit")
arr_vectors = opt_csv("arrival_geometry_vectors")
arrival_sample = opt_csv("population_arrival_sample")
if arrival_sample is not None:
    for c in ("dv_arrival_ms", "theta_arr_rad", "phi_arr_rad", "dv_total_ms"):
        arrival_sample[c] = pd.to_numeric(arrival_sample[c], errors="coerce")

# Orbit-mode stats panels only make sense if at least one sampled individual
# actually achieved a real capture -- otherwise theta_arr/phi_arr is all NaN
# and a 4-panel layout would just show two empty plots.
captured_sample = (
    arrival_sample[arrival_sample.theta_arr_rad.notna()] if arrival_sample is not None else None
)
is_orbit_mode = captured_sample is not None and not captured_sample.empty

AU = 1.495978707e11
BG = "#0f0f0f"
ACCENT = "#00d4ff"
DEPARTURE_COLOR = "#FFA94D"
ARRIVAL_COLOR = "#7ED957"


def scene_layout(xyz_unit):
    return dict(
        xaxis_title=f"x [{xyz_unit}]", yaxis_title=f"y [{xyz_unit}]", zaxis_title=f"z [{xyz_unit}]",
        xaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        yaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        zaxis=dict(backgroundcolor=BG, gridcolor="#333333"),
        aspectmode="data",
    )


has_phase_col = "central_body" in traj.columns
if has_phase_col and traj.central_body.iloc[0] != "Sun":
    first_body = traj.central_body.iloc[0]
    not_first_body = traj.central_body != first_body
    change_idx = not_first_body.idxmax() if not_first_body.any() else len(traj)
    is_departure_leg = traj.index < change_idx
else:
    is_departure_leg = pd.Series(False, index=traj.index)

if has_phase_col and traj.central_body.iloc[-1] != "Sun":
    last_body = traj.central_body.iloc[-1]
    not_last_body = traj.central_body != last_body
    change_idx_end = not_last_body[::-1].idxmax() if not_last_body.any() else -1
    is_arrival_leg = traj.index > change_idx_end
else:
    is_arrival_leg = pd.Series(False, index=traj.index)

capture_note = f", capture dV={best.dv_arrival_ms:.0f} m/s" if best.get("dv_arrival_ms", 0.0) > 0 else ""
title_summary = (
    f"{mission} -- Phase 9 {method} (real dynamics): achieved TOF={best.achieved_tof_days:.1f}d, "
    f"fitness={best.fitness:.4f}, real arrival miss={best.miss_km:.0f} km{capture_note}"
)

# ── Screen 1: trajectory views (departure / heliocentric / arrival) ───────────

fig1 = make_subplots(
    rows=1, cols=3,
    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
    subplot_titles=(
        "Departure leg + parking orbit (body-centered, km)",
        "Full transfer (heliocentric, AU)",
        "Arrival/capture leg" + (" + captured orbit (body-centered, km)" if is_orbit_mode else " (body-centered, km)"),
    ),
)

# Left: departure leg close-up, recentered against the departure body's own
# track, with the parking-orbit ring (theta=0 reference, same convention as
# plot_departure_geometry.py) overlaid for scale/orientation context.
if dep_track is not None and is_departure_leg.any():
    dep_x_km = (traj.x_m[is_departure_leg].values - dep_track.x_m[is_departure_leg].values) / 1e3
    dep_y_km = (traj.y_m[is_departure_leg].values - dep_track.y_m[is_departure_leg].values) / 1e3
    dep_z_km = (traj.z_m[is_departure_leg].values - dep_track.z_m[is_departure_leg].values) / 1e3
    fig1.add_trace(go.Scatter3d(
        x=dep_x_km, y=dep_y_km, z=dep_z_km, mode="lines",
        line=dict(color=DEPARTURE_COLOR, width=5), name="Departure leg",
    ), row=1, col=1)
    fig1.add_trace(go.Scatter3d(
        x=[dep_x_km[0]], y=[dep_y_km[0]], z=[dep_z_km[0]],
        mode="markers+text", marker=dict(size=5, color="#FFD700"),
        text=["Injection"], textposition="top center", name="Injection burn",
    ), row=1, col=1)
    fig1.add_trace(go.Scatter3d(
        x=[dep_x_km[-1]], y=[dep_y_km[-1]], z=[dep_z_km[-1]],
        mode="markers+text", marker=dict(size=5, color="#FF6B35"),
        text=["SOI exit"], textposition="top center", name="SOI exit",
    ), row=1, col=1)
    fig1.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers", marker=dict(size=8, color="#3a7bd5"), name="Departure body",
    ), row=1, col=1)
if dep_orbit_ring is not None:
    fig1.add_trace(go.Scatter3d(
        x=dep_orbit_ring.x_km, y=dep_orbit_ring.y_km, z=dep_orbit_ring.z_km, mode="lines",
        line=dict(color="#555555", width=3), name="Parking orbit",
    ), row=1, col=1)

# Middle: full heliocentric transfer -- unchanged from the previous layout.
x_au, y_au, z_au = traj.x_m / AU, traj.y_m / AU, traj.z_m / AU
if is_departure_leg.any():
    fig1.add_trace(go.Scatter3d(
        x=x_au[is_departure_leg], y=y_au[is_departure_leg], z=z_au[is_departure_leg], mode="lines",
        line=dict(color=DEPARTURE_COLOR, width=5), name="Departure leg (helio)", showlegend=False,
    ), row=1, col=2)
if is_arrival_leg.any():
    fig1.add_trace(go.Scatter3d(
        x=x_au[is_arrival_leg], y=y_au[is_arrival_leg], z=z_au[is_arrival_leg], mode="lines",
        line=dict(color=ARRIVAL_COLOR, width=5), name="Arrival/capture leg (helio)", showlegend=False,
    ), row=1, col=2)
cruise_only = ~is_departure_leg & ~is_arrival_leg
fig1.add_trace(go.Scatter3d(
    x=x_au[cruise_only], y=y_au[cruise_only], z=z_au[cruise_only], mode="lines",
    line=dict(color=ACCENT, width=4), name="Cruise leg",
), row=1, col=2)
fig1.add_trace(go.Scatter3d(
    x=[x_au.iloc[0]], y=[y_au.iloc[0]], z=[z_au.iloc[0]],
    mode="markers+text", marker=dict(size=6, color="#4FC3F7"),
    text=["Departure"], textposition="top center", name="Departure",
), row=1, col=2)
fig1.add_trace(go.Scatter3d(
    x=[x_au.iloc[-1]], y=[y_au.iloc[-1]], z=[z_au.iloc[-1]],
    mode="markers+text", marker=dict(size=6, color="#FF6B35"),
    text=["Spacecraft arrival"], textposition="top center", name="Spacecraft arrival",
), row=1, col=2)
tx_au, ty_au, tz_au = best.target_x_m / AU, best.target_y_m / AU, best.target_z_m / AU
fig1.add_trace(go.Scatter3d(
    x=[tx_au], y=[ty_au], z=[tz_au],
    mode="markers+text", marker=dict(size=6, color="#FFD700", symbol="diamond"),
    text=["Target (real position)"], textposition="bottom center", name="Target (real position)",
), row=1, col=2)
fig1.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode="markers", marker=dict(size=10, color="#FFD700"), name="Sun",
), row=1, col=2)

# Right: arrival/capture leg close-up, recentered against the target body's
# own track, with the captured-orbit ring + capture burn vectors overlaid
# when this run actually achieved a capture (Orbit mode).
if target_track is not None and is_arrival_leg.any():
    arr_x_km = (traj.x_m[is_arrival_leg].values - target_track.x_m[is_arrival_leg].values) / 1e3
    arr_y_km = (traj.y_m[is_arrival_leg].values - target_track.y_m[is_arrival_leg].values) / 1e3
    arr_z_km = (traj.z_m[is_arrival_leg].values - target_track.z_m[is_arrival_leg].values) / 1e3
    fig1.add_trace(go.Scatter3d(
        x=arr_x_km, y=arr_y_km, z=arr_z_km, mode="lines",
        line=dict(color=ARRIVAL_COLOR, width=5), name="Arrival leg", showlegend=False,
    ), row=1, col=3)
    fig1.add_trace(go.Scatter3d(
        x=[arr_x_km[0]], y=[arr_y_km[0]], z=[arr_z_km[0]],
        mode="markers+text", marker=dict(size=5, color="#4FC3F7"),
        text=["Target SOI entry"], textposition="top center", name="Target SOI entry",
    ), row=1, col=3)
    fig1.add_trace(go.Scatter3d(
        x=[arr_x_km[-1]], y=[arr_y_km[-1]], z=[arr_z_km[-1]],
        mode="markers+text", marker=dict(size=5, color="#FF6B35"),
        text=["End of arc"], textposition="top center", name="End of arc",
    ), row=1, col=3)
    fig1.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers", marker=dict(size=8, color="#FFD700"), name="Target body",
    ), row=1, col=3)
else:
    fig1.add_annotation(
        text="No arrival/capture leg in this arc (never entered target SOI)", showarrow=False,
        x=0.87, y=0.45, xref="paper", yref="paper", font=dict(color="#888888"),
    )
if arr_orbit_ring is not None:
    fig1.add_trace(go.Scatter3d(
        x=arr_orbit_ring.x_km, y=arr_orbit_ring.y_km, z=arr_orbit_ring.z_km, mode="lines",
        line=dict(color="#7ED957", width=3, dash="dot"), name="Captured orbit",
    ), row=1, col=3)
if arr_vectors is not None:
    vec_colors = {"v_incoming_relative": "#00d4ff", "v_circular_post_burn": "#FFD700"}
    for _, row in arr_vectors.iterrows():
        if row["name"] == "capture_position":
            fig1.add_trace(go.Scatter3d(
                x=[row.x1_km], y=[row.y1_km], z=[row.z1_km], mode="markers+text",
                marker=dict(size=6, color="#FF3366"), text=["Capture burn"], textposition="top center",
                name="Capture burn position",
            ), row=1, col=3)
            continue
        color = vec_colors.get(row["name"], "#dddddd")
        fig1.add_trace(go.Scatter3d(
            x=[row.x0_km, row.x1_km], y=[row.y0_km, row.y1_km], z=[row.z0_km, row.z1_km],
            mode="lines+markers+text", line=dict(color=color, width=6), marker=dict(size=[2, 5], color=color),
            text=["", row["name"]], textposition="top center", name=row["name"],
        ), row=1, col=3)

fig1.update_layout(
    title=title_summary,
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#dddddd"),
    height=700,
    scene=scene_layout("km"),
    scene2=scene_layout("AU"),
    scene3=scene_layout("km"),
)

traj_out_path = DATA / f"{prefix}_optimize_trajectory.html"
fig1.write_html(str(traj_out_path))
print(f"Wrote {traj_out_path}")

# ── Screen 2: stats (convergence + population scatter) ───────────────────────

has_population_plot = population is not None and not population.empty
n_stat_rows = 3 if is_orbit_mode else 2
row_titles = [
    "Phase 1: flyby-only convergence (km)",
    f"Phase 2: {method} convergence (real objective)",
]
if has_population_plot:
    row_titles += [
        "Departure theta/phi, colored by departure dV",
        "Departure theta/phi, colored by generation",
    ]
else:
    row_titles += ["(no population log)", "(no population log)"]
if is_orbit_mode:
    row_titles += [
        "Arrival theta_arr/phi_arr, colored by total dV",
        "Arrival theta_arr/phi_arr, colored by generation",
    ]

fig2 = make_subplots(
    rows=n_stat_rows, cols=2,
    specs=[[{"type": "xy"}, {"type": "xy"}] for _ in range(n_stat_rows)],
    subplot_titles=tuple(row_titles),
    vertical_spacing=0.10,
)

if phase1_conv is not None:
    fig2.add_trace(
        go.Scatter(x=phase1_conv.generation, y=phase1_conv.best_closest_approach_km_so_far, mode="lines",
                   line=dict(color=DEPARTURE_COLOR, width=2), name="Phase 1 best (km)"),
        row=1, col=1,
    )
else:
    fig2.add_annotation(text="No phase-1 data (PSO is not phase-split)", showarrow=False,
                         x=0.21, y=0.97, xref="paper", yref="paper", font=dict(color="#888888"))
fig2.add_trace(
    go.Scatter(x=conv[step_col], y=conv.best_fitness_so_far, mode="lines",
               line=dict(color=ACCENT, width=2), name="Phase 2 best fitness"),
    row=1, col=2,
)
fig2.update_xaxes(title_text="Generation", row=1, col=1)
fig2.update_yaxes(title_text="Best closest approach so far [km]", row=1, col=1)
fig2.update_xaxes(title_text=step_col.capitalize(), row=1, col=2)
fig2.update_yaxes(title_text="Best fitness so far", row=1, col=2)

if has_population_plot:
    feasible = population[population.fitness.notna()]
    infeasible = population[population.fitness.isna()]

    def scatter_panel(row, col, color_vals, color_title, colorbar_y):
        if not infeasible.empty:
            fig2.add_trace(go.Scatter(
                x=infeasible.theta_burn_rad, y=infeasible.phi_rad, mode="markers",
                marker=dict(size=3, color="#444444"), name=f"Infeasible ({len(infeasible)})", showlegend=(row == 2 and col == 1),
            ), row=row, col=col)
        fig2.add_trace(go.Scatter(
            x=feasible.theta_burn_rad, y=feasible.phi_rad, mode="markers",
            marker=dict(
                size=4, color=color_vals, colorscale="Viridis", showscale=True,
                colorbar=dict(title=color_title, x=1.0, len=1.0 / n_stat_rows, y=colorbar_y),
                symbol=["circle" if p == 1 else "diamond" for p in feasible.phase],
            ),
            name=f"Feasible ({len(feasible)})", showlegend=(row == 2 and col == 1),
            text=[f"phase {p}, gen {g}, fitness={f:.3g}, dv={d:.0f} m/s"
                  for p, g, f, d in zip(feasible.phase, feasible.generation, feasible.fitness, feasible.dv_mps)],
        ), row=row, col=col)
        fig2.update_xaxes(title_text="theta_burn [rad]", row=row, col=col)
        fig2.update_yaxes(title_text="phi_out_of_plane [rad]", row=row, col=col)

    dep_colorbar_y = 1.0 - (1.5 / n_stat_rows)
    scatter_panel(2, 1, feasible.dv_mps, "dv_departure [m/s]", dep_colorbar_y)
    scatter_panel(2, 2, feasible.generation + feasible.phase * (population.generation.max() + 1), "gen (phase-offset)", dep_colorbar_y)

if is_orbit_mode:
    arr_colorbar_y = 1.0 - (2.5 / n_stat_rows)

    def arrival_panel(row, col, color_vals, color_title):
        fig2.add_trace(go.Scatter(
            x=captured_sample.theta_arr_rad, y=captured_sample.phi_arr_rad, mode="markers",
            marker=dict(
                size=5, color=color_vals, colorscale="Plasma", showscale=True,
                colorbar=dict(title=color_title, x=1.0, len=1.0 / n_stat_rows, y=arr_colorbar_y),
                symbol=["circle" if p == 1 else "diamond" for p in captured_sample.phase],
            ),
            name=f"Captured sample ({len(captured_sample)})", showlegend=(row == 3 and col == 1),
            text=[f"phase {p}, gen {g}, dv_total={d:.0f} m/s"
                  for p, g, d in zip(captured_sample.phase, captured_sample.generation, captured_sample.dv_total_ms)],
        ), row=row, col=col)
        fig2.update_xaxes(title_text="theta_arr [rad]", row=row, col=col)
        fig2.update_yaxes(title_text="phi_arr [rad]", row=row, col=col)

    arrival_panel(3, 1, captured_sample.dv_total_ms, "dv_total [m/s]")
    arrival_panel(3, 2, captured_sample.generation + captured_sample.phase * (captured_sample.generation.max() + 1), "gen (phase-offset)")

fig2.update_layout(
    title=title_summary,
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#dddddd"),
    height=380 * n_stat_rows,
)
fig2.update_xaxes(gridcolor="#333333")
fig2.update_yaxes(gridcolor="#333333")

stats_out_path = DATA / f"{prefix}_optimize_stats.html"
fig2.write_html(str(stats_out_path))
print(f"Wrote {stats_out_path}")

webbrowser.open(traj_out_path.as_uri())
webbrowser.open(stats_out_path.as_uri())
