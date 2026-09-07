"""
plot_wsb_final_refinement.py — sensitivity ensemble visualisation.

Trajectories are reclassified: if a "moon_crash" trajectory had a viable LOI
burn opportunity (min ΔV_LOI < LOI_CAPTURABLE_KMS) inside the Hill sphere
before crashing, it is relabelled "capturable" — the spacecraft could have
been saved with a timely burn.

Reads : out/wsb/sensitivity_ensemble.csv
        out/wsb/sensitivity_summary.txt
Saves : out/wsb/wsb_sensitivity.html

Run:  python plot/plot_wsb_final_refinement.py
"""
import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = pathlib.Path(__file__).parent.parent
SEN  = ROOT / "out" / "wsb"

MU        = 0.01215565
X_M       = 1.0 - MU
R_HILL    = (MU / 3.0) ** (1.0 / 3.0)
L_KM      = 384_400.0
T_STAR    = 375_700.0
R_MOON_KM = 1_737.4
V_STAR    = L_KM / T_STAR          # km/s per nd

# Trajectories with min LOI ΔV below this threshold inside the Hill sphere
# are reclassified from "moon_crash" → "capturable"
LOI_CAPTURABLE_KMS = 0.4           # km/s — a reasonable single burn

N_SUBSAMPLE = 4                    # keep every N-th point to reduce file size

OUTCOME_COLOR = {
    "capturable":  "#FF9800",      # orange — was going to crash, but LOI viable
    "captured":    "#1A73E8",
    "moon_crash":  "#E63946",
    "escaped":     "#AAAAAA",
}
OUTCOME_LABEL = {
    "capturable":  "Capturable (viable LOI burn inside Hill)",
    "captured":    "Captured",
    "moon_crash":  "Moon crash (no viable LOI)",
    "escaped":     "Escaped",
}

# ── Load CSV ──────────────────────────────────────────────────────────────────
csv_path = SEN / "sensitivity_ensemble.csv"
if not csv_path.exists():
    print(f"Not found: {csv_path}")
    print("Run: cargo run … --bin wsb_sensitivity --release")
    raise SystemExit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows, {df['run_id'].nunique()} trajectories")

# ── LOI ΔV computation in the rotating frame ─────────────────────────────────
# Mirrors transfers::loi_dv from Rust, using ND units then converting to km/s.
def compute_loi_dv_nd(rows):
    """Return minimum LOI ΔV [km/s] for trajectory rows that are inside the Hill sphere."""
    # Moon-relative position in rotating frame
    rx = rows["x_nd"].values - X_M
    ry = rows["y_nd"].values
    rz = rows["z_nd"].values if "z_nd" in rows.columns else np.zeros(len(rows))
    r_mag = np.sqrt(rx**2 + ry**2 + rz**2)

    # Moon-relative inertial velocity (Coriolis correction)
    vrx = rows["vx_nd"].values - rows["y_nd"].values
    vry = rows["vy_nd"].values + rows["x_nd"].values - X_M
    vrz = rows["vz_nd"].values if "vz_nd" in rows.columns else np.zeros(len(rows))

    v_circ = np.sqrt(MU / np.maximum(r_mag, 1e-10))

    # Angular momentum vector h = r × v_rel
    hx = ry*vrz - rz*vry
    hy = rz*vrx - rx*vrz
    hz = rx*vry - ry*vrx
    h_mag = np.sqrt(hx**2 + hy**2 + hz**2)

    dv = np.full(len(rows), np.nan)
    ok = (h_mag > 1e-14) & (r_mag > 1e-10)
    hxn = np.where(ok, hx/h_mag, 0)
    hyn = np.where(ok, hy/h_mag, 0)
    hzn = np.where(ok, hz/h_mag, 0)
    rxn = np.where(ok, rx/r_mag, 0)
    ryn = np.where(ok, ry/r_mag, 0)
    rzn = np.where(ok, rz/r_mag, 0)
    # tangent = h_hat × r_hat
    tx = hyn*rzn - hzn*ryn
    ty = hzn*rxn - hxn*rzn
    tz = hxn*ryn - hyn*rxn
    dvx = v_circ*tx - vrx
    dvy = v_circ*ty - vry
    dvz = v_circ*tz - vrz
    dv_ok = np.sqrt(dvx**2 + dvy**2 + dvz**2) * V_STAR
    dv[ok] = dv_ok[ok]
    return dv   # [km/s], NaN outside Hill or degenerate

# ── Build per-run metadata with reclassification ──────────────────────────────
print("Computing LOI ΔV and reclassifying outcomes …")
traj_df = df[df["x_nd"].notna()].copy()

# Compute moon-relative distance and Hill sphere mask
traj_df["r_moon"] = np.sqrt((traj_df["x_nd"] - X_M)**2 + traj_df["y_nd"]**2)
traj_df["in_hill"] = traj_df["r_moon"] < R_HILL

run_meta_raw = (df[df["x_nd"].isna()]
                  .groupby("run_id").first().reset_index()
                  [["run_id", "outcome", "est_orbits", "min_alt_km", "is_nominal"]])

reclassified = {}
for run_id, grp in traj_df[traj_df["in_hill"]].groupby("run_id"):
    if len(grp) == 0:
        continue
    loi_dvs = compute_loi_dv_nd(grp)
    min_dv = np.nanmin(loi_dvs) if not np.all(np.isnan(loi_dvs)) else np.nan
    reclassified[run_id] = min_dv

run_meta = run_meta_raw.copy()
run_meta["min_loi_kms"] = run_meta["run_id"].map(reclassified)

# Reclassify: crash OR escape trajectories with viable LOI inside Hill sphere → "capturable"
mask_reclass = (
    (run_meta["outcome"].isin(["moon_crash", "escaped"])) &
    (run_meta["min_loi_kms"] < LOI_CAPTURABLE_KMS)
)
run_meta.loc[mask_reclass, "outcome"] = "capturable"

outcome_counts = run_meta["outcome"].value_counts()
print("Outcome breakdown (after reclassification):")
for k, v in outcome_counts.items():
    print(f"  {k}: {v}")

nominal_outcome = run_meta[run_meta["is_nominal"] == 1]["outcome"].values[0]
nom_col = OUTCOME_COLOR.get(nominal_outcome, "#000000")
print(f"Nominal outcome (reclassified): {nominal_outcome}")

# ── Summary metadata ──────────────────────────────────────────────────────────
summary_path = SEN / "sensitivity_summary.txt"
title_meta = ""
hit_id = 0
if summary_path.exists():
    for line in summary_path.read_text().splitlines():
        if line.startswith("Hit ID"):
            try: hit_id = int(line.split(":")[1].strip())
            except: pass
        if "θ_inject" in line or "θ_sun" in line or "est_orbits" in line:
            title_meta += line.strip() + "  |  "
title_meta = title_meta.rstrip("  |  ")

# ── Hill entry of nominal — clip perturbed arcs here ──────────────────────────
sub_nom = traj_df[traj_df["run_id"] == 0].sort_values("time_nd")
_in = sub_nom["r_moon"].values < R_HILL
t_hill_nom = float(sub_nom["time_nd"].values[_in][0]) if _in.any() else None
if t_hill_nom is not None:
    print(f"Nominal Hill entry: {t_hill_nom * T_STAR / 86400.0:.1f} d — clipping perturbed arcs")

# ── Per-trajectory coordinate helper ─────────────────────────────────────────
def get_traj(run_id, t_clip=None):
    sub = traj_df[traj_df["run_id"] == run_id].copy()
    if t_clip is not None:
        sub = sub[sub["time_nd"] <= t_clip]
    sub = sub.iloc[::N_SUBSAMPLE]
    t = sub["time_nd"].values
    x = sub["x_nd"].values
    y = sub["y_nd"].values
    xi    = (x * np.cos(t) - y * np.sin(t)) * L_KM
    yi    = (x * np.sin(t) + y * np.cos(t)) * L_KM
    xm    = X_M * np.cos(t) * L_KM
    ym    = X_M * np.sin(t) * L_KM
    xi_mc = xi - xm
    yi_mc = yi - ym
    return xi, yi, xi_mc, yi_mc

def circle(cx, cy, r, n=180):
    a = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(a), cy + r * np.sin(a)

def _s(x, y, **kw):
    return go.Scatter(x=x, y=y, **kw)

# Pre-compute nominal
xi_n, yi_n, xi_mc_n, yi_mc_n = get_traj(0)

# ── Static geometry ───────────────────────────────────────────────────────────
xe,  ye  = circle(0, 0, 6_371)
xmo, ymo = circle(0, 0, X_M * L_KM)
xmd, ymd = circle(0, 0, R_MOON_KM)
xhs, yhs = circle(0, 0, R_HILL * L_KM)

VIEW_FULL = X_M * L_KM * 1.55
VIEW_MOON = R_HILL * L_KM * 2.8

n_cap  = outcome_counts.get("captured",   0)
n_cap2 = outcome_counts.get("capturable", 0)
n_crash = outcome_counts.get("moon_crash", 0)
n_esc   = outcome_counts.get("escaped",    0)

# ── Build figure ──────────────────────────────────────────────────────────────
print("Building sensitivity plot …")
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Inertial frame — full transfer",
                    "Moon-centred inertial — Hill sphere region"],
    column_widths=[0.52, 0.48],
    horizontal_spacing=0.07,
)

fig.add_trace(_s(xe/1e3, ye/1e3, fill="toself", fillcolor="#378ADD",
    line=dict(color="#378ADD", width=0), name="Earth"), row=1, col=1)
fig.add_trace(_s(xmo/1e3, ymo/1e3, mode="lines",
    line=dict(color="rgba(160,160,160,0.3)", width=1, dash="dot"),
    name="Moon orbit", showlegend=False), row=1, col=1)
fig.add_trace(_s(xmd/1e3, ymd/1e3, fill="toself",
    fillcolor="rgba(160,160,160,0.5)", line=dict(color="grey", width=1),
    name="Moon", showlegend=False), row=1, col=2)
fig.add_trace(_s(xhs/1e3, yhs/1e3, mode="lines",
    line=dict(color="#BA7517", width=1.5, dash="dash"),
    name="Hill sphere", showlegend=False), row=1, col=2)

# Draw in order: escaped (grey, behind) → crash → capturable → captured
legend_shown = set()
for outcome in ["escaped", "moon_crash", "capturable", "captured"]:
    if outcome not in OUTCOME_COLOR:
        continue
    col   = OUTCOME_COLOR[outcome]
    label = OUTCOME_LABEL[outcome]
    rids  = run_meta[(run_meta["outcome"] == outcome) & (run_meta["is_nominal"] == 0)]["run_id"].values
    for rid in rids:
        xi, yi, xi_mc, yi_mc = get_traj(rid, t_clip=t_hill_nom)
        show_leg = outcome not in legend_shown
        if show_leg: legend_shown.add(outcome)
        fig.add_trace(_s(xi/1e3, yi/1e3, mode="lines",
            line=dict(color=col, width=0.8), opacity=0.4,
            name=label, legendgroup=outcome, showlegend=show_leg,
        ), row=1, col=1)
        fig.add_trace(_s(xi_mc/1e3, yi_mc/1e3, mode="lines",
            line=dict(color=col, width=0.8), opacity=0.4,
            name=label, legendgroup=outcome, showlegend=False,
        ), row=1, col=2)

# Nominal on top
fig.add_trace(_s(xi_n/1e3, yi_n/1e3, mode="lines",
    line=dict(color=nom_col, width=3.0),
    name=f"Nominal ({nominal_outcome})", legendgroup="nominal",
), row=1, col=1)
fig.add_trace(_s(xi_mc_n/1e3, yi_mc_n/1e3, mode="lines",
    line=dict(color=nom_col, width=3.0),
    name=f"Nominal ({nominal_outcome})", legendgroup="nominal", showlegend=False,
), row=1, col=2)

fig.update_layout(
    title=dict(text=(
        f"WSB Sensitivity Ensemble — Hit {hit_id}  |  N={len(run_meta)} trajectories<br>"
        f"<sup>{title_meta}<br>"
        f"Capturable (LOI&lt;{LOI_CAPTURABLE_KMS} km/s): {n_cap2}  |  "
        f"Captured: {n_cap}  |  No viable LOI (crash/escape): {n_crash + n_esc}  |  "
        f"Nominal: <b>{nominal_outcome}</b></sup>"
    ), font=dict(size=12)),
    height=680, template="plotly_white",
    legend=dict(x=1.01, y=1, font=dict(size=10)),
)
fig.update_xaxes(title_text="x [×10³ km]", range=[-VIEW_FULL/1e3, VIEW_FULL/1e3],
                 scaleanchor="y", scaleratio=1, row=1, col=1)
fig.update_yaxes(title_text="y [×10³ km]", range=[-VIEW_FULL/1e3, VIEW_FULL/1e3], row=1, col=1)
fig.update_xaxes(title_text="x from Moon [×10³ km]", range=[-VIEW_MOON/1e3, VIEW_MOON/1e3],
                 scaleanchor="y2", scaleratio=1, row=1, col=2)
fig.update_yaxes(title_text="y from Moon [×10³ km]", range=[-VIEW_MOON/1e3, VIEW_MOON/1e3], row=1, col=2)

out_path = SEN / "wsb_sensitivity.html"
fig.write_html(str(out_path))
print(f"Saved {out_path}")
