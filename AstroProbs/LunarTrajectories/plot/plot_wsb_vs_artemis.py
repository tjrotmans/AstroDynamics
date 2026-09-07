#!/usr/bin/env python3
"""
plot_wsb_vs_artemis.py -- 3-D Earth-inertial (ECI) comparison
  WSB ballistic capture vs Artemis II trajectory, including post-LOI circular orbits.

Prerequisites -- run these Rust binaries first
----------------------------------------------
From AstroProbs/LunarTrajectories/:

  1. cargo run --bin wsb_circularize --release
     Reads:  out/wsb/maxhifi.csv, out/wsb/solution_hifi.csv,
             ../Artemis/out/artemis2_trajectory.csv, ../Artemis/out/moon_ephem.csv
     Writes: out/wsb/capture.csv        -- BCR4BP transfer trajectory to periapsis
             out/wsb/loi_orbit.csv      -- post-LOI orbit in ECI with DE440S Moon
             out/wsb/epoch_info.txt     -- dep_offset_days + R0_wsb matrix
             out/wsb/circularize_info.txt

  2. cargo run --bin artemis_circularize --release  (from AstroProbs/LunarTrajectories/)
     Writes: out/artemis_circularize/loi_orbit.csv  -- Artemis LOI orbit in ECI

  3. The Artemis trajectory must exist:
             ../Artemis/out/artemis2_trajectory.csv  (run the Artemis binary separately)

Then run from AstroProbs/LunarTrajectories/:
  python plot/plot_wsb_vs_artemis.py

Fallback behaviour (if circularized files are missing)
-------------------------------------------------------
  - Without out/wsb/capture.csv + epoch_info.txt: falls back to maxhifi.csv /
    solution_hifi.csv (no LOI orbit shown for WSB, departure epoch computed from
    theta_sun_deg rather than read from epoch_info.txt).
  - Without out/artemis_circularize/loi_orbit.csv: Artemis is clipped at its
    return Earth flyby (~T+10 d) with no lunar orbit shown.
  - MP4 output requires ffmpeg installed and on PATH.

Frame notes
-----------
  WSB transfer (capture.csv): stored in BCR4BP rotating frame; Python converts to
  ECI via rot_em_to_eci() using a circular Moon approximation (unavoidable since
  the BCR4BP integrator assumes circular Moon orbit).

  WSB LOI orbit (loi_orbit.csv): already in ECI with DE440S Moon, written by
  wsb_circularize.rs -- Python loads directly, no conversion.

  Artemis trajectory and LOI orbit: both already in ECI with DE440S Moon,
  loaded directly from CSV.

  The animated Moon dot uses the Artemis DE440S Moon (art_mx/my/mz) throughout,
  which is more accurate than the BCR4BP circular approximation used for wsb_mx
  during the WSB transfer phase.

Outputs (saved to out/wsb/):
  wsb_vs_artemis_3d.html        -- animated Plotly 3D (ECI frame)
  wsb_vs_artemis_portrait.png   -- static matplotlib 3D portrait (ECI frame)
  wsb_vs_artemis.mp4            -- LinkedIn video (requires ffmpeg)
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Shared constants and frame utilities ──────────────────────────────────────
from wsb_style import (
    MU, X_M, L_KM, T_STAR, V_STAR_KM, R_EARTH, R_MOON,
    R_HILL_ND, R_HILL_KM,
    BG, COL_WSB, COL_ART, COL_EARTH, COL_MOON, COL_PARK, COL_HILL,
    rot_em_to_eci, moon_em_to_eci, orbital_plane_r3d,
    load_epoch_info,
)
from astrodynamics import MOON_SIDEREAL_DAYS, SYNODIC_DAYS

R_PARK_ND = (6_371.0 + 378.0) / L_KM   # ISS-altitude parking orbit [nd]
R_PARK_KM = R_PARK_ND * L_KM           # [km]
V_STAR_KMS = V_STAR_KM                  # alias used in this script

# ---- Paths -------------------------------------------------------------------
_HERE       = pathlib.Path(__file__).parent
ROOT        = _HERE.parent
WSB_DIR     = ROOT / "out" / "wsb"
ART_DIR     = ROOT.parent / "Artemis" / "out"
ART_CSV     = ART_DIR / "artemis2_trajectory.csv"
WSB_HIF_CSV   = WSB_DIR / "solution_hifi.csv"
WSB_MAXI_CSV  = WSB_DIR / "maxhifi.csv"
WSB_ENS_CSV   = WSB_DIR / "sensitivity_ensemble.csv"
WSB_CAPT_CSV  = WSB_DIR / "capture.csv"
WSB_LOI_CSV   = WSB_DIR / "loi_orbit.csv"
WSB_EPOCH_TXT = WSB_DIR / "epoch_info.txt"
ART_LOI_CSV   = ROOT / "out" / "artemis_circularize" / "loi_orbit.csv"
OUT_HTML    = WSB_DIR / "wsb_vs_artemis_3d.html"
OUT_PNG     = WSB_DIR / "wsb_vs_artemis_portrait.png"
OUT_MP4     = WSB_DIR / "wsb_vs_artemis.mp4"

def compute_eci_frame(moon_x_m: np.ndarray,
                      moon_y_m: np.ndarray,
                      moon_z_m: np.ndarray,
                      time_s:   np.ndarray) -> np.ndarray:
    """
    Build R0: 3×3 rotation matrix  EM-inertial → ECI.
    Thin wrapper around orbital_plane_r3d (from astrodynamics.py).
    """
    m0_km    = np.array([moon_x_m[0], moon_y_m[0], moon_z_m[0]]) / 1e3
    m1_km    = np.array([moon_x_m[1], moon_y_m[1], moon_z_m[1]]) / 1e3
    dt       = float(time_s[1] - time_s[0])
    m_vel_km = (m1_km - m0_km) / dt
    return orbital_plane_r3d(m0_km, m_vel_km)


def compute_wsb_epoch(
    theta_sun_deg: float,
    R0:            np.ndarray,
    u_sun_eci:     np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Find the WSB departure-date offset from the Artemis departure (2026-04-02)
    and the corrected EM→ECI rotation matrix at that date.

    theta_sun in the BCR4BP is the Sun's angle in the EM rotating frame —
    i.e. the angle from the Earth-Moon line to the Earth-Sun line in the EM
    orbital plane.  It drifts at the synodic rate (-360°/29.53 days).
    Inverting that rate gives the calendar offset that places the real Sun
    at theta_sun_deg.

    The corrected R0_wsb accounts for the Moon having moved by
    2π * offset / MOON_SIDEREAL_DAYS radians in its orbit during the offset.

    Returns (departure_offset_days, R0_wsb).  If theta_sun_deg is NaN
    (metadata unavailable), returns (0.0, R0) unchanged.
    """
    if np.isnan(theta_sun_deg):
        return 0.0, R0.copy()

    # Project real Sun direction into EM-inertial frame at Artemis departure.
    # At BCR4BP t=0 the rotating frame and the EM-inertial frame coincide,
    # so this projection gives theta_sun at the Artemis epoch directly.
    sun_in_em         = R0.T @ u_sun_eci
    theta_sun_artemis = np.degrees(np.arctan2(sun_in_em[1], sun_in_em[0]))

    # Wrap difference to the nearest synodic window ([-180°, +180°])
    delta_theta          = ((theta_sun_deg - theta_sun_artemis) + 180.0) % 360.0 - 180.0
    departure_offset_days = delta_theta / (-360.0 / SYNODIC_DAYS)

    # Corrected R0 at the WSB departure epoch:
    # the Moon has rotated by moon_angle_rad in the EM orbital plane.
    moon_angle_rad = 2.0 * np.pi * departure_offset_days / MOON_SIDEREAL_DAYS
    ca, sa  = np.cos(moon_angle_rad), np.sin(moon_angle_rad)
    new_x   = R0 @ np.array([ca, sa, 0.0])          # new Earth→Moon unit vec
    new_z   = R0[:, 2]                               # orbital plane normal (fixed)
    new_y   = np.cross(new_z, new_x)
    new_y  /= np.linalg.norm(new_y)
    R0_wsb  = np.column_stack([new_x, new_y, new_z])

    return departure_offset_days, R0_wsb


# ==============================================================================
# Data loading
# ==============================================================================

def load_epoch_info(path: pathlib.Path) -> tuple[float, np.ndarray]:
    """Read epoch_info.txt from wsb_circularize. Returns (dep_offset_days, R0_wsb)."""
    info: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                key, _, val = line.partition(":")
                info[key.strip()] = val.strip()
    dep_offset = float(info["dep_offset_days"])
    c0 = np.array([float(v) for v in info["R0_wsb_col0"].split()])
    c1 = np.array([float(v) for v in info["R0_wsb_col1"].split()])
    c2 = np.array([float(v) for v in info["R0_wsb_col2"].split()])
    return dep_offset, np.column_stack([c0, c1, c2])


def load_artemis(
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Artemis II trajectory, clip at return Earth closest-approach,
    and derive R0 and Sun direction from the anise-sourced CSV columns.

    Returns:
      R0           -- 3x3 ECI frame rotation matrix
      u_sun_eci    -- unit vector Earth→Sun at departure, true ECI (from anise)
      r_sun_km     -- Earth-Sun distance at departure [km]
      t_days       -- time [days from start]
      x,y,z        -- ECI spacecraft position [km]
      mx,my,mz     -- ECI Moon position [km]
    """
    if not ART_CSV.exists():
        raise FileNotFoundError(f"Missing {ART_CSV}")

    df  = pd.read_csv(ART_CSV)
    t_s = df["time_s"].values
    sc  = df[["x_m", "y_m", "z_m"]].values * 1e-3        # m -> km
    mn  = df[["moon_x_m", "moon_y_m", "moon_z_m"]].values * 1e-3

    # Sun direction from the anise ephemeris columns written by the Rust binary.
    # Use the first row (departure) as the reference direction for SE-L1.
    sn  = df[["sun_x_m", "sun_y_m", "sun_z_m"]].values * 1e-3   # m -> km
    sun0      = sn[0]
    r_sun_km  = float(np.linalg.norm(sun0))
    u_sun_eci = sun0 / r_sun_km

    # R0 derived from raw (m) Moon positions and raw time_s
    R0 = compute_eci_frame(
        df["moon_x_m"].values, df["moon_y_m"].values, df["moon_z_m"].values,
        t_s,
    )

    t_rel = t_s - t_s[0]

    if ART_LOI_CSV.exists():
        df_loi = pd.read_csv(ART_LOI_CSV)
        loi_sc = df_loi[["x_km", "y_km", "z_km"]].values          # ECI km
        loi_mn = df_loi[["moon_x_km", "moon_y_km", "moon_z_km"]].values
        loi_t  = df_loi["time_s"].values

        # Clip transfer exactly at the LOI burn time (same absolute epoch as Artemis CSV)
        clip_idx = int(np.searchsorted(t_s, loi_t[0]))
        sc, mn, t_s = sc[:clip_idx], mn[:clip_idx], t_s[:clip_idx]

        # Cap LOI to ~6.5 days so animation time scale matches the WSB LOI
        LOI_LIMIT_S = 23.0 * 86_400.0   # extend to match WSB LOI end (~day 27)
        loi_mask = (loi_t - loi_t[0]) <= LOI_LIMIT_S
        loi_sc, loi_mn, loi_t = loi_sc[loi_mask], loi_mn[loi_mask], loi_t[loi_mask]

        sc  = np.vstack([sc,  loi_sc])
        mn  = np.vstack([mn,  loi_mn])
        t_s = np.concatenate([t_s, loi_t])
        print(f"  Artemis: appended LOI orbit ({loi_mask.sum()} pts)")
    else:
        # Fallback: clip at Earth closest-approach after T+6 days
        mask = t_rel >= 6 * 86_400.0
        if mask.any():
            offset = int(np.where(mask)[0][0])
            end    = offset + int(np.argmin(np.linalg.norm(sc[mask], axis=1)))
            sc, mn, t_s = sc[:end+1], mn[:end+1], t_s[:end+1]

    t_days = (t_s - t_s[0]) / 86_400.0
    return (R0, u_sun_eci, r_sun_km,
            t_days, sc[:, 0], sc[:, 1], sc[:, 2],
            mn[:, 0], mn[:, 1], mn[:, 2])


def load_wsb(R0: np.ndarray, u_sun_eci: np.ndarray):
    """
    Load WSB trajectory in ECI, epoch-corrected to the true departure date.

    Priority:
      1. out/wsb_maxhifi/maxhifi.csv        (max-fidelity, every integrator step)
      2. out/wsb_refine/solution_hifi.csv   (repropagated at fixed log interval)
      3. out/wsb_sensitivity/sensitivity_ensemble.csv  (fallback)

    The BCR4BP theta_sun_deg is matched to the real Sun position from the
    Artemis ephemeris to find the actual departure date.  t_days is shifted
    so that t=0 is still the Artemis departure (2026-04-02); a positive
    departure_offset_days means the WSB departs after Artemis.

    Returns:
      t_days, x,y,z [km ECI], moon_x,y,z [km ECI], ic_info,
      departure_offset_days [days from Artemis departure], R0_wsb
    """
    # ---- Preferred: capture.csv + epoch_info.txt (wsb_circularize output) ------
    if WSB_CAPT_CSV.exists() and WSB_EPOCH_TXT.exists():
        df = pd.read_csv(WSB_CAPT_CSV)
        departure_offset_days, R0_wsb = load_epoch_info(WSB_EPOCH_TXT)
        ic_info = {"source": "capture", "hit_id": 0, "seed_id": 0,
                   "theta_deg": float("nan"), "theta_sun_deg": float("nan"),
                   "r_apogee_nd": float("nan"), "est_orbits": float("nan"),
                   "dtheta_deg": 0.0, "dsun_deg": 0.0}
        if WSB_HIF_CSV.exists():
            meta = pd.read_csv(WSB_HIF_CSV, nrows=1).iloc[0]
            ic_info.update({
                "hit_id":       int(meta["hit_id"]),
                "seed_id":      int(meta["seed_id"]),
                "theta_deg":    float(meta["theta_deg"]),
                "theta_sun_deg":float(meta["theta_sun_deg"]),
                "r_apogee_nd":  float(meta["r_apogee_nd"]),
                "est_orbits":   float(meta["est_capture_orbits"]),
                "dtheta_deg":   float(meta["dtheta_deg"]),
                "dsun_deg":     float(meta["dsun_deg"]),
            })
        t = df["time_nd"].values
        x = df["x_nd"].values
        y = df["y_nd"].values
        z = df["z_nd"].values
        xi, yi, zi = rot_em_to_eci(x, y, z, t, R0_wsb)
        mx, my, mz = moon_em_to_eci(t, R0_wsb)
        t_days     = t * T_STAR / 86_400.0 + departure_offset_days
        print(f"  WSB (capture): {len(df)} steps, T+{t[-1] * T_STAR / 86400:.1f} d flight")
        if WSB_LOI_CSV.exists():
            loi_df   = pd.read_csv(WSB_LOI_CSV)
            loi_t_s  = loi_df["time_s"].values          # absolute [s], same epoch as art CSV
            LOI_CAP_S = 8.0 * 70_377.0                  # all 8 available orbits (~6.5 d)
            loi_mask  = loi_t_s <= loi_t_s[0] + LOI_CAP_S
            loi_td    = t_days[-1] + (loi_t_s[loi_mask] - loi_t_s[0]) / 86_400.0
            t_days = np.concatenate([t_days, loi_td[1:]])
            xi     = np.concatenate([xi,     loi_df["x_km"].values[loi_mask][1:]])
            yi     = np.concatenate([yi,     loi_df["y_km"].values[loi_mask][1:]])
            zi     = np.concatenate([zi,     loi_df["z_km"].values[loi_mask][1:]])
            mx     = np.concatenate([mx,     loi_df["moon_x_km"].values[loi_mask][1:]])
            my     = np.concatenate([my,     loi_df["moon_y_km"].values[loi_mask][1:]])
            mz     = np.concatenate([mz,     loi_df["moon_z_km"].values[loi_mask][1:]])
            print(f"  WSB LOI orbit appended: {loi_mask.sum() - 1} rows")
        label = f"Hit {ic_info['hit_id']} (capture+LOI)" if ic_info["hit_id"] else "WSB (capture+LOI)"
        ic_info["label"] = label
        return t_days, xi, yi, zi, mx, my, mz, ic_info, departure_offset_days, R0_wsb

    # ---- Fallback: maxhifi / solution_hifi / ensemble -------------------------
    if WSB_MAXI_CSV.exists():
        df = pd.read_csv(WSB_MAXI_CSV)
        ic_info = {"source": "maxhifi", "hit_id": 0, "seed_id": 0,
                   "theta_deg": float("nan"), "theta_sun_deg": float("nan"),
                   "r_apogee_nd": float("nan"), "est_orbits": float("nan"),
                   "dtheta_deg": 0.0, "dsun_deg": 0.0}
        if WSB_HIF_CSV.exists():
            meta = pd.read_csv(WSB_HIF_CSV, nrows=1).iloc[0]
            ic_info.update({
                "hit_id":       int(meta["hit_id"]),
                "seed_id":      int(meta["seed_id"]),
                "theta_deg":    float(meta["theta_deg"]),
                "theta_sun_deg":float(meta["theta_sun_deg"]),
                "r_apogee_nd":  float(meta["r_apogee_nd"]),
                "est_orbits":   float(meta["est_capture_orbits"]),
                "dtheta_deg":   float(meta["dtheta_deg"]),
                "dsun_deg":     float(meta["dsun_deg"]),
            })
        label = f"Hit {ic_info['hit_id']} (maxhifi)"
        n_steps = len(df)
        print(f"  WSB (maxhifi): {n_steps} steps, T+{df['time_nd'].iloc[-1] * T_STAR / 86400:.1f} d")
    elif WSB_HIF_CSV.exists():
        df   = pd.read_csv(WSB_HIF_CSV)
        meta = df.iloc[0]
        ic_info = {
            "source":       "solution_hifi",
            "hit_id":       int(meta["hit_id"]),
            "seed_id":      int(meta["seed_id"]),
            "theta_deg":    float(meta["theta_deg"]),
            "theta_sun_deg":float(meta["theta_sun_deg"]),
            "r_apogee_nd":  float(meta["r_apogee_nd"]),
            "est_orbits":   float(meta["est_capture_orbits"]),
            "dtheta_deg":   float(meta["dtheta_deg"]),
            "dsun_deg":     float(meta["dsun_deg"]),
        }
        label = f"Hit {ic_info['hit_id']}"
    else:
        print(f"  solution_hifi.csv not found, falling back to sensitivity ensemble")
        df   = pd.read_csv(WSB_ENS_CSV)
        df   = df[(df["is_nominal"] == 1) & df["x_nd"].notna()].copy()
        ic_info = {"source": "sensitivity_ensemble", "hit_id": 0, "seed_id": 0,
                   "theta_deg": float("nan"), "theta_sun_deg": float("nan"),
                   "r_apogee_nd": float("nan"), "est_orbits": float("nan"),
                   "dtheta_deg": 0.0, "dsun_deg": 0.0}
        label = "WSB nominal"

    t  = df["time_nd"].values
    x  = df["x_nd"].values
    y  = df["y_nd"].values
    z  = df["z_nd"].values

    departure_offset_days, R0_wsb = compute_wsb_epoch(
        ic_info.get("theta_sun_deg", float("nan")), R0, u_sun_eci
    )

    xi, yi, zi = rot_em_to_eci(x, y, z, t, R0_wsb)
    mx, my, mz = moon_em_to_eci(t, R0_wsb)
    t_days     = t * T_STAR / 86_400.0 + departure_offset_days

    ic_info["label"] = label
    return t_days, xi, yi, zi, mx, my, mz, ic_info, departure_offset_days, R0_wsb


# ==============================================================================
# IC overlay: parking orbit + TLI injection point in ECI
# ==============================================================================

def compute_ic_overlay(ic_info: dict, R0: np.ndarray) -> dict | None:
    """
    Compute the parking orbit circle and TLI injection geometry in ECI.

    Returns None if ic_info is missing the required fields (fallback mode).
    """
    if np.isnan(ic_info.get("theta_deg", float("nan"))):
        return None

    theta    = np.radians(ic_info["theta_deg"])
    r_apo_nd = ic_info["r_apogee_nd"]

    # -- Parking orbit circle (ISS altitude, in EM orbital plane) --------------
    phi      = np.linspace(0, 2 * np.pi, 361)
    park_em  = np.array([R_PARK_ND * np.cos(phi),
                         R_PARK_ND * np.sin(phi),
                         np.zeros(361)]) * L_KM    # km, 3 x 361
    park_eci = R0 @ park_em                         # km, 3 x 361

    # -- TLI injection point ---------------------------------------------------
    inj_em  = np.array([[R_PARK_ND * np.cos(theta)],
                        [R_PARK_ND * np.sin(theta)],
                        [0.0]]) * L_KM
    inj_eci = (R0 @ inj_em).flatten()

    # -- TLI delta-V -----------------------------------------------------------
    mu_e     = 1.0 - MU
    v_circ   = np.sqrt(mu_e / R_PARK_ND)
    a        = (R_PARK_ND + r_apo_nd) / 2.0
    v_inj    = np.sqrt(mu_e * (2.0 / R_PARK_ND - 1.0 / a))
    dv_nd    = v_inj - v_circ
    dv_kms   = dv_nd * V_STAR_KMS

    # -- ΔV direction (prograde = tangent to parking orbit at θ) ---------------
    # In EM inertial, tangent at angle theta is (-sin θ, cos θ, 0)
    dv_dir_em  = np.array([[-np.sin(theta)], [np.cos(theta)], [0.0]])
    dv_dir_eci = (R0 @ dv_dir_em).flatten()

    return {
        "park_x": park_eci[0], "park_y": park_eci[1], "park_z": park_eci[2],
        "inj_eci": inj_eci,
        "dv_dir_eci": dv_dir_eci,
        "dv_kms": dv_kms,
        "theta_deg": ic_info["theta_deg"],
        "theta_sun_deg": ic_info["theta_sun_deg"],
        "r_apogee_km": r_apo_nd * L_KM,
        "est_orbits": ic_info["est_orbits"],
        "park_alt_km": R_PARK_KM - R_EARTH,
    }


# ==============================================================================
# Geometry helpers
# ==============================================================================

def sphere_surface(r: float, nu: int = 40, nv: int = 20):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    return (r * np.outer(np.cos(u), np.sin(v)),
            r * np.outer(np.sin(u), np.sin(v)),
            r * np.outer(np.ones(nu), np.cos(v)))


def downsample(arr: np.ndarray, n: int) -> np.ndarray:
    idx = np.unique(np.round(np.linspace(0, len(arr) - 1, n)).astype(int))
    return arr[idx]


def interp_pos(t_arr, x, y, z, t_query):
    t_q = np.clip(t_query, t_arr[0], t_arr[-1])
    return np.interp(t_q, t_arr, x), np.interp(t_q, t_arr, y), np.interp(t_q, t_arr, z)


# ==============================================================================
# Plot 1 -- interactive animated Plotly 3D HTML  (two panels: ECI + SOI)
# ==============================================================================

def build_plotly(
    wsb_t, wsb_x, wsb_y, wsb_z, wsb_mx, wsb_my, wsb_mz,
    art_t, art_x, art_y, art_z, art_mx, art_my, art_mz,
    ic_overlay:  dict | None,
    R0:          np.ndarray,
) -> go.Figure:

    N_FRAMES = 500
    FRAME_MS = 30

    t_min    = min(wsb_t[0],  art_t[0])
    t_max    = max(wsb_t[-1], art_t[-1])
    frames_t = np.linspace(t_min, t_max, N_FRAMES)

    # -- Moon-centred positions (XY top-down, ignore Z) ------------------------
    wsb_x_mc = wsb_x - wsb_mx
    wsb_y_mc = wsb_y - wsb_my
    art_x_mc = art_x - art_mx
    art_y_mc = art_y - art_my

    # -- ECI helpers -----------------------------------------------------------
    sx, sy, sz = sphere_surface(R_EARTH * 3.5)        # display radius (3.5x physical)
    # Clean analytic Moon orbit circle (evenly-spaced, one revolution)
    _t_orb  = np.linspace(0, 2 * np.pi, 361)
    _morb   = R0 @ (np.array([np.cos(_t_orb), np.sin(_t_orb), np.zeros(361)]) * L_KM)
    moon_cx, moon_cy, moon_cz = _morb[0], _morb[1], _morb[2]

    r_wsb   = np.sqrt(wsb_x**2 + wsb_y**2 + wsb_z**2)
    apo_idx = int(np.argmax(r_wsb))
    apo_km  = float(r_wsb[apo_idx])
    r_eci_lim = float(np.max(np.abs([wsb_x, wsb_y, wsb_z]))) * 1.08

    # -- 2D SOI helpers --------------------------------------------------------
    phi2d = np.linspace(0, 2 * np.pi, 180)
    hill_x2d = R_HILL_KM * np.cos(phi2d)
    hill_y2d = R_HILL_KM * np.sin(phi2d)
    moon_r_disp = R_MOON * 5           # enlarged for visibility
    moon_x2d = moon_r_disp * np.cos(phi2d)
    moon_y2d = moon_r_disp * np.sin(phi2d)

    # == Build two-panel figure: LEFT=3D ECI, RIGHT=2D Moon-centred ============
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=["ECI Frame — Full Transfer",
                        "Moon-Centred — Top-Down View (SOI)"],
        column_widths=[0.56, 0.44],
        horizontal_spacing=0.06,
    )

    # ---- LEFT panel: ECI 3D ---------------------------------------------------

    fig.add_trace(go.Surface(
        x=sx, y=sy, z=sz,
        colorscale=[[0, "#1A237E"], [0.5, COL_EARTH], [1, "#64B5F6"]],
        showscale=False, opacity=0.9,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.4),
        name="Earth",
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=moon_cx, y=moon_cy, z=moon_cz, mode="lines",
        line=dict(color="rgba(180,180,180,0.25)", width=2.5),
        name="Moon orbit", showlegend=True, hoverinfo="skip",
    ), row=1, col=1)

    if ic_overlay is not None:
        fig.add_trace(go.Scatter3d(
            x=ic_overlay["park_x"], y=ic_overlay["park_y"], z=ic_overlay["park_z"],
            mode="lines",
            line=dict(color="rgba(255,215,0,0.6)", width=2, dash="dot"),
            name=f"Parking orbit ({ic_overlay['park_alt_km']:.0f} km)",
            showlegend=False,
        ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[wsb_x[apo_idx]], y=[wsb_y[apo_idx]], z=[wsb_z[apo_idx]],
        mode="markers+text",
        marker=dict(color=COL_WSB, size=6, symbol="diamond"),
        text=[f"Apogee {apo_km/1e3:.1f}e3 km"],
        textposition="top center", textfont=dict(color=COL_WSB, size=9),
        name="WSB apogee", showlegend=False,
    ), row=1, col=1)

    if ic_overlay is not None:
        inj = ic_overlay["inj_eci"]
        dv  = ic_overlay["dv_dir_eci"]
        arrow_end = inj + dv * 80_000
        fig.add_trace(go.Scatter3d(
            x=[float(inj[0])], y=[float(inj[1])], z=[float(inj[2])],
            mode="markers+text",
            marker=dict(color=COL_PARK, size=8, symbol="circle"),
            text=["TLI"], textposition="top center",
            textfont=dict(color=COL_PARK, size=9),
            name="WSB TLI", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=[float(inj[0]), float(arrow_end[0])],
            y=[float(inj[1]), float(arrow_end[1])],
            z=[float(inj[2]), float(arrow_end[2])],
            mode="lines", line=dict(color=COL_PARK, width=3),
            name=f"TLI DV={ic_overlay['dv_kms']:.3f} km/s", showlegend=False,
        ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[art_x[0]], y=[art_y[0]], z=[art_z[0]],
        mode="markers+text",
        marker=dict(color=COL_PARK, size=7, symbol="circle-open"),
        text=["Artemis start"], textposition="top center",
        textfont=dict(color=COL_PARK, size=8),
        name="Artemis start", showlegend=False,
    ), row=1, col=1)

    # ECI animated traces
    n0_eci = len(fig.data)
    fig.add_trace(go.Scatter3d(
        x=[wsb_x[0]], y=[wsb_y[0]], z=[wsb_z[0]], mode="markers",
        marker=dict(color=COL_WSB, size=5, symbol="circle",
                    line=dict(color="white", width=0.8)),
        name="WSB s/c", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[art_x[0]], y=[art_y[0]], z=[art_z[0]], mode="markers",
        marker=dict(color=COL_ART, size=5, symbol="circle",
                    line=dict(color="white", width=0.8)),
        name="Artemis s/c", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[art_mx[0]], y=[art_my[0]], z=[art_mz[0]],
        mode="markers+text", marker=dict(color=COL_MOON, size=6),
        text=["Moon"], textposition="top center",
        textfont=dict(color=COL_MOON, size=9),
        name="Moon", showlegend=False,
    ), row=1, col=1)
    # Animated thick trails for 3D panel (grow as s/c flies)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="lines",
        line=dict(color=COL_WSB, width=5),
        name="BC trail 3D", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="lines",
        line=dict(color=COL_ART, width=5),
        name="Artemis trail 3D", showlegend=False,
    ), row=1, col=1)
    anim_eci = [n0_eci, n0_eci + 1, n0_eci + 2, n0_eci + 3, n0_eci + 4]

    # ---- RIGHT panel: Moon-centred 2D top-down --------------------------------

    # Hill sphere circle
    fig.add_trace(go.Scatter(
        x=hill_x2d.tolist(), y=hill_y2d.tolist(), mode="lines",
        line=dict(color=COL_HILL, width=2, dash="dot"),
        name=f"Hill sphere ({R_HILL_KM/1e3:.0f}e3 km)",
        showlegend=True,
    ), row=1, col=2)

    # Moon disc
    fig.add_trace(go.Scatter(
        x=moon_x2d.tolist(), y=moon_y2d.tolist(), mode="lines",
        fill="toself", fillcolor="rgba(160,160,160,0.5)",
        line=dict(color="#AAAAAA", width=1),
        name="Moon", showlegend=False,
    ), row=1, col=2)

    # SOI animated traces (2D): trails start empty, filled per-frame; dots follow s/c
    n0_soi = len(fig.data)
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines",
        line=dict(color=COL_WSB, width=4),
        name="Ballistic Capture trail", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines",
        line=dict(color=COL_ART, width=4),
        name="Artemis trail", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[float(wsb_x_mc[0])], y=[float(wsb_y_mc[0])], mode="markers",
        marker=dict(color=COL_WSB, size=7, symbol="circle",
                    line=dict(color="white", width=0.8)),
        name="WSB s/c SOI", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[float(art_x_mc[0])], y=[float(art_y_mc[0])], mode="markers",
        marker=dict(color=COL_ART, size=7, symbol="circle",
                    line=dict(color="white", width=0.8)),
        name="Artemis s/c SOI", showlegend=False,
    ), row=1, col=2)
    anim_soi = [n0_soi, n0_soi + 1, n0_soi + 2, n0_soi + 3]

    ANIM_TRACES = anim_eci + anim_soi

    # == Animation frames =====================================================
    MAX_TRAIL = 1500
    frames = []
    for fi, ft in enumerate(frames_t):
        wx,  wy,  wz  = interp_pos(wsb_t, wsb_x,  wsb_y,  wsb_z,  ft)
        ax,  ay,  az  = interp_pos(art_t, art_x,  art_y,  art_z,  ft)
        mxw, myw, _   = interp_pos(wsb_t, wsb_mx, wsb_my, wsb_mz, ft)  # WSB Moon (for WSB SOI)
        mxa, mya, mza = interp_pos(art_t, art_mx, art_my, art_mz, ft)  # Artemis Moon (DE440S)

        # Accumulating trails up to current frame time
        w_mask = wsb_t <= ft
        a_mask = art_t <= ft

        # 3D ECI thick trail
        wtx3 = wsb_x[w_mask]; wty3 = wsb_y[w_mask]; wtz3 = wsb_z[w_mask]
        atx3 = art_x[a_mask]; aty3 = art_y[a_mask]; atz3 = art_z[a_mask]
        if len(wtx3) > MAX_TRAIL:
            ti = downsample(np.arange(len(wtx3)), MAX_TRAIL)
            wtx3, wty3, wtz3 = wtx3[ti], wty3[ti], wtz3[ti]
        if len(atx3) > MAX_TRAIL:
            ti = downsample(np.arange(len(atx3)), MAX_TRAIL)
            atx3, aty3, atz3 = atx3[ti], aty3[ti], atz3[ti]

        # SOI 2D thick trail
        wtx = wsb_x_mc[w_mask]; wty = wsb_y_mc[w_mask]
        atx = art_x_mc[a_mask]; aty = art_y_mc[a_mask]
        if len(wtx) > MAX_TRAIL:
            ti = downsample(np.arange(len(wtx)), MAX_TRAIL)
            wtx, wty = wtx[ti], wty[ti]
        if len(atx) > MAX_TRAIL:
            ti = downsample(np.arange(len(atx)), MAX_TRAIL)
            atx, aty = atx[ti], aty[ti]

        frames.append(go.Frame(
            data=[
                # ECI 3D: WSB s/c, Artemis s/c, Moon
                go.Scatter3d(x=[float(wx)], y=[float(wy)], z=[float(wz)]),
                go.Scatter3d(x=[float(ax)], y=[float(ay)], z=[float(az)]),
                go.Scatter3d(x=[float(mxa)], y=[float(mya)], z=[float(mza)],
                             text=["Moon"], textposition="top center"),
                # ECI 3D: thick trails growing behind s/c
                go.Scatter3d(x=wtx3.tolist(), y=wty3.tolist(), z=wtz3.tolist()),
                go.Scatter3d(x=atx3.tolist(), y=aty3.tolist(), z=atz3.tolist()),
                # SOI 2D: trails up to current time
                go.Scatter(x=wtx.tolist(), y=wty.tolist()),
                go.Scatter(x=atx.tolist(), y=aty.tolist()),
                # SOI 2D: spacecraft dots
                go.Scatter(x=[float(wx - mxw)], y=[float(wy - myw)]),
                go.Scatter(x=[float(ax - mxa)], y=[float(ay - mya)]),
            ],
            traces=ANIM_TRACES, name=str(fi),
        ))
    fig.frames = frames

    slider_steps = []
    for fi in range(0, N_FRAMES, max(1, N_FRAMES // 60)):
        ft = frames_t[fi]
        slider_steps.append({
            "args": [[str(fi)], {"frame": {"duration": FRAME_MS, "redraw": True},
                                 "mode": "immediate", "transition": {"duration": 0}}],
            "label": f"{ft:.0f}", "method": "animate",
        })

    # IC annotation — keep only DV
    dv_text = (f"TLI \u0394V = {ic_overlay['dv_kms']:.3f} km/s"
               if ic_overlay is not None else "")

    def _3d_axis(title, rng):
        return dict(showgrid=True, gridcolor="rgba(80,80,160,0.18)", zeroline=False,
                    tickfont=dict(size=8), title=title, range=rng)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgb(10,10,18)",
        # LEFT: 3D ECI scene
        scene=dict(
            bgcolor="rgb(12,12,22)",
            xaxis=_3d_axis("X [km]", [-r_eci_lim, r_eci_lim]),
            yaxis=_3d_axis("Y [km]", [-r_eci_lim, r_eci_lim]),
            zaxis=_3d_axis("Z [km]", [-r_eci_lim * 0.4, r_eci_lim * 0.4]),
            camera=dict(eye=dict(x=0.9, y=-1.4, z=0.7)),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.4),
        ),
        # RIGHT: 2D axes zoomed to Hill sphere (dark background)
        xaxis=dict(
            title="X from Moon [km]", range=[-R_HILL_KM * 1.15, R_HILL_KM * 1.15],
            scaleanchor="y", scaleratio=1,
            showgrid=True, gridcolor="rgba(80,80,160,0.18)",
            zeroline=True, zerolinecolor="rgba(150,150,255,0.25)",
            tickfont=dict(size=9, color="#aaa"),
        ),
        yaxis=dict(
            title="Y from Moon [km]", range=[-R_HILL_KM * 1.15, R_HILL_KM * 1.15],
            showgrid=True, gridcolor="rgba(80,80,160,0.18)",
            zeroline=True, zerolinecolor="rgba(150,150,255,0.25)",
            tickfont=dict(size=9, color="#aaa"),
        ),
        plot_bgcolor="rgb(12,12,22)",
        font=dict(color="white", size=11),
        title=dict(
            text=(
                "WSB vs Artemis II — ECI (left)  |  Moon-Centred top-down (right)"
                + (f"<br><sup>{dv_text}</sup>" if dv_text else "")
            ),
            x=0.01, xanchor="left",
        ),
        legend=dict(x=0.01, y=0.98, font=dict(size=10),
                    bgcolor="rgba(15,15,25,0.7)"),
        height=740,
        margin=dict(l=0, r=0, t=95, b=80),
        updatemenus=[{
            "type": "buttons", "showactive": False,
            "x": 0.5, "xanchor": "center", "y": -0.08, "yanchor": "top",
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": FRAME_MS, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate", "transition": {"duration": 0}}]},
            ],
            "font": {"color": "white"},
            "bgcolor": "rgba(40,40,60,0.9)",
            "bordercolor": "rgba(100,100,150,0.5)",
        }],
        sliders=[{
            "active": 0, "x": 0.05, "len": 0.90, "y": -0.02, "yanchor": "top",
            "currentvalue": {"prefix": "T=", "suffix": " days (0 = Artemis dep.)",
                             "visible": True, "xanchor": "center",
                             "font": {"color": "white", "size": 11}},
            "transition": {"duration": 0},
            "bgcolor": "rgba(40,40,60,0.8)",
            "bordercolor": "rgba(100,100,150,0.4)",
            "tickcolor": "rgba(200,200,255,0.5)",
            "font": {"color": "white", "size": 9},
            "steps": slider_steps,
        }],
    )
    return fig


# ==============================================================================
# Plot 2 -- static matplotlib 3D portrait
# ==============================================================================

def build_portrait(
    wsb_t, wsb_x, wsb_y, wsb_z, wsb_mx, wsb_my, wsb_mz,
    art_t, art_x, art_y, art_z, art_mx, art_my, art_mz,
    ic_overlay: dict | None,
    ic_info:    dict,
    R0:         np.ndarray,
) -> None:

    PANE_EDGE = "#252540"

    PATH_PTS  = 266723

    ws_idx = downsample(np.arange(len(wsb_x)), PATH_PTS)
    at_idx = downsample(np.arange(len(art_x)), PATH_PTS)

    sx, sy, sz = sphere_surface(R_EARTH, nu=48, nv=24)

    # Moon sphere at T=0 WSB position
    msx, msy, msz = sphere_surface(R_MOON * 0.6, nu=24, nv=12)
    m0x, m0y, m0z = float(wsb_mx[0]), float(wsb_my[0]), float(wsb_mz[0])

    r_wsb   = np.sqrt(wsb_x**2 + wsb_y**2 + wsb_z**2)
    apo_idx = int(np.argmax(r_wsb))

    r_moon_wsb = np.sqrt((wsb_x - wsb_mx)**2 + (wsb_y - wsb_my)**2 +
                         (wsb_z - wsb_mz)**2)
    ca_idx = int(np.argmin(r_moon_wsb))

    # ---- Figure ---------------------------------------------------------------
    fig = plt.figure(figsize=(9, 16), dpi=200, facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d")

    ax.set_facecolor(BG)
    for attr in ("xaxis", "yaxis", "zaxis"):
        pane = getattr(ax, attr).pane
        pane.fill = False
        pane.set_edgecolor(PANE_EDGE)
        getattr(ax, attr)._axinfo["grid"]["color"] = (0.30, 0.30, 0.60, 0.20)
        getattr(ax, attr)._axinfo["grid"]["linewidth"] = 0.5
        getattr(ax, attr).label.set_color("white")
        getattr(ax, attr).label.set_fontsize(8)
    ax.tick_params(colors="white", labelsize=5.5, pad=0)

    # Earth
    ax.plot_surface(sx, sy, sz, color=COL_EARTH, alpha=0.90,
                    linewidth=0, zorder=2, shade=True)
    ax.text(0, 0, R_EARTH * 1.8, "Earth", color="white", fontsize=9,
            ha="center", va="bottom", fontweight="bold")

    # Moon orbit reference — analytic clean circle, one revolution
    _t_orb = np.linspace(0, 2 * np.pi, 361)
    _morb  = R0 @ (np.array([np.cos(_t_orb), np.sin(_t_orb), np.zeros(361)]) * L_KM)
    ax.plot(_morb[0], _morb[1], _morb[2],
            color=COL_MOON, alpha=0.20, linewidth=0.8, linestyle="--", zorder=1)

    # Moon sphere at T=0
    ax.plot_surface(msx + m0x, msy + m0y, msz + m0z,
                    color=COL_MOON, alpha=0.55, linewidth=0, zorder=2)

    # Parking orbit (no legend label)
    if ic_overlay is not None:
        ax.plot(ic_overlay["park_x"], ic_overlay["park_y"], ic_overlay["park_z"],
                color=COL_PARK, alpha=0.60, linewidth=1.0, linestyle=":", zorder=6)

    # Artemis (orange)
    ax.plot(art_x[at_idx], art_y[at_idx], art_z[at_idx],
            color=COL_ART, linewidth=2.5, alpha=0.85, zorder=5,
            label=f"Artemis II  (T+{art_t[-1]:.0f} d)")

    # WSB (cyan)
    ax.plot(wsb_x[ws_idx], wsb_y[ws_idx], wsb_z[ws_idx],
            color=COL_WSB, linewidth=2.5, alpha=0.80, zorder=5,
            label=f"Ballistic Capture  (T+{wsb_t[-1]:.0f} d)")

    # TLI injection marker + DV arrow
    if ic_overlay is not None:
        inj = ic_overlay["inj_eci"]
        dv  = ic_overlay["dv_dir_eci"]
        arrow_end = inj + dv * 60_000
        ax.scatter([inj[0]], [inj[1]], [inj[2]],
                   color=COL_PARK, s=40, zorder=10, marker="*")
        ax.plot([inj[0], arrow_end[0]], [inj[1], arrow_end[1]],
                [inj[2], arrow_end[2]], color=COL_PARK, linewidth=1.5, zorder=10)
        ax.text(arrow_end[0], arrow_end[1], arrow_end[2],
                f"TLI\n{ic_overlay['dv_kms']:.3f} km/s",
                color=COL_PARK, fontsize=6, ha="center", va="center")

    # Apogee annotation
    apo_pt  = np.array([wsb_x[apo_idx], wsb_y[apo_idx], wsb_z[apo_idx]])
    apo_r   = float(np.linalg.norm(apo_pt))
    apo_off = apo_pt / apo_r * 120_000 + np.array([0, 0, 60_000])
    lp      = apo_pt + apo_off
    ax.plot([apo_pt[0], lp[0]], [apo_pt[1], lp[1]], [apo_pt[2], lp[2]],
            color=COL_WSB, linewidth=0.8, alpha=0.8)
    ax.text(lp[0], lp[1], lp[2], f"WSB apogee\n{apo_r/1e3:.1f}e3 km",
            color=COL_WSB, fontsize=7, ha="center", va="center")

    # Moon CA annotation (WSB)
    ca_pt  = np.array([wsb_x[ca_idx], wsb_y[ca_idx], wsb_z[ca_idx]])
    ca_off = np.array([-80_000, 0, 70_000])
    ca_lp  = ca_pt + ca_off
    ax.plot([ca_pt[0], ca_lp[0]], [ca_pt[1], ca_lp[1]], [ca_pt[2], ca_lp[2]],
            color=COL_WSB, linewidth=0.7, alpha=0.75)
    ax.text(ca_lp[0], ca_lp[1], ca_lp[2],
            f"WSB Moon CA\n(T+{wsb_t[ca_idx]:.0f} d)",
            color=COL_WSB, fontsize=6.5, ha="center", va="center")

    # Artemis Moon CA
    r_art_moon = np.sqrt((art_x - art_mx)**2 + (art_y - art_my)**2 +
                         (art_z - art_mz)**2)
    art_ca_idx = int(np.argmin(r_art_moon))
    art_ca_pt  = np.array([art_x[art_ca_idx], art_y[art_ca_idx],
                            art_z[art_ca_idx]])
    art_ca_off = np.array([40_000, 40_000, 50_000])
    art_ca_lp  = art_ca_pt + art_ca_off
    ax.plot([art_ca_pt[0], art_ca_lp[0]], [art_ca_pt[1], art_ca_lp[1]],
            [art_ca_pt[2], art_ca_lp[2]], color=COL_ART, linewidth=0.7, alpha=0.75)
    ax.text(art_ca_lp[0], art_ca_lp[1], art_ca_lp[2],
            f"Artemis Moon flyby\n(T+{art_t[art_ca_idx]:.0f} d)",
            color=COL_ART, fontsize=6.5, ha="center", va="center")

    # ---- IC text box ----------------------------------------------------------
    if ic_overlay is not None:
        ic_lines = (
            f"WSB Initial Conditions (ECI)\n"
            f"Parking orbit alt : {ic_overlay['park_alt_km']:.0f} km\n"
            f"TLI DeltaV        : {ic_overlay['dv_kms']:.4f} km/s\n"
            f"Inj. angle (EM)   : {ic_overlay['theta_deg']:.2f} deg\n"
            f"Sun angle (EM)    : {ic_overlay['theta_sun_deg']:.2f} deg\n"
            f"Apogee radius     : {ic_overlay['r_apogee_km']/1e3:.2f}e3 km\n"
            f"Est. capture      : {ic_overlay['est_orbits']:.2f} lunar orbits"
        )
        fig.text(0.015, 0.12, ic_lines,
                 color="white", fontsize=6.5, va="bottom",
                 fontfamily="monospace",
                 bbox=dict(facecolor="#14141E", edgecolor="#333355",
                           alpha=0.75, boxstyle="round,pad=0.5"))

    # ---- Legend + title -------------------------------------------------------
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COL_WSB, linewidth=2.5,
               label=f"Ballistic Capture  ({wsb_t[-1]:.0f} d)"),
        Line2D([0], [0], color=COL_ART, linewidth=2.5,
               label=f"Artemis II  ({art_t[-1]:.0f} d)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975),
               ncol=2, fontsize=8, framealpha=0.45,
               facecolor="#14141E", edgecolor="#333355",
               labelcolor="white", handlelength=1.5, columnspacing=1.0)

    dv_str = (f"  |  TLI \u0394V = {ic_overlay['dv_kms']:.3f} km/s"
              if ic_overlay is not None else "")
    fig.text(0.5, 0.992, f"WSB vs Artemis II -- ECI{dv_str}",
             color="white", fontsize=10, ha="center", va="top", fontweight="bold")

    # ---- Axis limits ----------------------------------------------------------
    r_lim = float(np.max(np.abs([wsb_x, wsb_y]))) * 1.08
    z_lim = max(float(np.max(np.abs(wsb_z))),
                float(np.max(np.abs(art_z)))) + 50_000
    z_lim = max(z_lim, r_lim * 0.20)

    ax.set_xlim(-r_lim, r_lim)
    ax.set_ylim(-r_lim, r_lim)
    ax.set_zlim(-z_lim, z_lim)
    ax.set_xlabel("X [km]", labelpad=1)
    ax.set_ylabel("Y [km]", labelpad=1)
    ax.set_zlabel("Z [km]", labelpad=1)
    ax.set_box_aspect([1.8, 1.8, max(0.35, z_lim / r_lim * 1.8)])
    ax.view_init(elev=22, azim=-50)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.02)
    fig.savefig(str(OUT_PNG), dpi=200, facecolor=BG)
    jpg_path = str(OUT_PNG).replace(".png", ".jpg")
    fig.savefig(jpg_path, dpi=200, facecolor=BG, format="jpeg")
    print(f"Saved portrait -> {OUT_PNG}")
    print(f"Saved portrait -> {jpg_path}")
    plt.close(fig)


# ==============================================================================
# Plot 3 -- MP4 video for LinkedIn (matplotlib FuncAnimation)
# ==============================================================================

def build_video(
    wsb_t, wsb_x, wsb_y, wsb_z, wsb_mx, wsb_my, wsb_mz,
    art_t, art_x, art_y, art_z, art_mx, art_my, art_mz,
    ic_overlay: dict | None,
    R0:         np.ndarray,
) -> None:
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import matplotlib.gridspec as gridspec

    N_FRAMES  = 400
    FPS       = 30
    MAX_TRAIL = 600

    t_min    = wsb_t[0]
    t_max    = max(wsb_t[-1], art_t[-1])
    frames_t = np.linspace(t_min, t_max, N_FRAMES)

    wsb_x_mc = wsb_x - wsb_mx
    wsb_y_mc = wsb_y - wsb_my
    art_x_mc = art_x - art_mx
    art_y_mc = art_y - art_my

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                            width_ratios=[1.3, 1.0], wspace=0.03,
                            left=0.02, right=0.99, top=0.92, bottom=0.06)
    ax3 = fig.add_subplot(gs[0], projection="3d")
    ax2 = fig.add_subplot(gs[1])

    # ---- 3D axes style --------------------------------------------------------
    ax3.set_facecolor(BG)
    for attr in ("xaxis", "yaxis", "zaxis"):
        pane = getattr(ax3, attr).pane
        pane.fill = False
        pane.set_edgecolor("#252540")
        getattr(ax3, attr)._axinfo["grid"]["color"] = (0.30, 0.30, 0.60, 0.18)
        getattr(ax3, attr)._axinfo["grid"]["linewidth"] = 0.5
        getattr(ax3, attr).label.set_color("white")
        getattr(ax3, attr).label.set_fontsize(7)
    ax3.tick_params(colors="white", labelsize=5, pad=0)

    r_lim = float(np.max(np.abs([wsb_x, wsb_y]))) * 1.08
    z_lim = max(float(np.max(np.abs(wsb_z))), float(np.max(np.abs(art_z)))) + 50_000
    z_lim = max(z_lim, r_lim * 0.20)
    ax3.set_xlim(-r_lim, r_lim)
    ax3.set_ylim(-r_lim, r_lim)
    ax3.set_zlim(-z_lim, z_lim)
    ax3.set_xlabel("X [km]", labelpad=1, fontsize=7)
    ax3.set_ylabel("Y [km]", labelpad=1, fontsize=7)
    ax3.set_zlabel("Z [km]", labelpad=1, fontsize=7)
    ax3.set_box_aspect([1.8, 1.8, max(0.35, z_lim / r_lim * 1.8)])
    ax3.view_init(elev=22, azim=-50)

    # ---- Static 3D: Earth, Moon orbit, ghost paths ----------------------------
    sx, sy, sz = sphere_surface(R_EARTH, nu=48, nv=24)
    ax3.plot_surface(sx, sy, sz, color=COL_EARTH, alpha=0.85,
                     linewidth=0, zorder=2, shade=True)
    ax3.text(0, 0, R_EARTH * 1.8, "Earth", color="white", fontsize=8,
             ha="center", va="bottom", fontweight="bold")

    _t_orb = np.linspace(0, 2 * np.pi, 361)
    _morb  = R0 @ (np.array([np.cos(_t_orb), np.sin(_t_orb), np.zeros(361)]) * L_KM)
    ax3.plot(_morb[0], _morb[1], _morb[2],
             color=COL_MOON, alpha=0.18, linewidth=0.8, linestyle="--")

    if ic_overlay is not None:
        ax3.plot(ic_overlay["park_x"], ic_overlay["park_y"], ic_overlay["park_z"],
                 color=COL_PARK, alpha=0.45, linewidth=0.8, linestyle=":")

    # ---- Animated 3D objects --------------------------------------------------
    trail_w3, = ax3.plot([], [], [], color=COL_WSB, linewidth=2.5, alpha=0.90, zorder=5)
    trail_a3, = ax3.plot([], [], [], color=COL_ART, linewidth=2.5, alpha=0.90, zorder=5)
    dot_w3,   = ax3.plot([], [], [], "o", color=COL_WSB, markersize=6, zorder=10,
                         markeredgecolor="white", markeredgewidth=0.6)
    dot_a3,   = ax3.plot([], [], [], "o", color=COL_ART, markersize=6, zorder=10,
                         markeredgecolor="white", markeredgewidth=0.6)
    dot_m3,   = ax3.plot([], [], [], "o", color=COL_MOON, markersize=5, zorder=8)

    # ---- 2D axes style --------------------------------------------------------
    ax2.set_facecolor(BG)
    ax2.set_aspect("equal")
    ax2.tick_params(colors="white", labelsize=7)
    ax2.set_xlabel("X from Moon [km]", color="white", fontsize=8)
    ax2.set_ylabel("Y from Moon [km]", color="white", fontsize=8)
    ax2.set_xlim(-R_HILL_KM * 1.15, R_HILL_KM * 1.15)
    ax2.set_ylim(-R_HILL_KM * 1.15, R_HILL_KM * 1.15)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#252540")
    ax2.grid(True, color=(0.30, 0.30, 0.60, 0.15), linewidth=0.5)

    phi2d = np.linspace(0, 2 * np.pi, 180)
    ax2.plot(R_HILL_KM * np.cos(phi2d), R_HILL_KM * np.sin(phi2d),
             color=COL_HILL, linewidth=1.5, linestyle=":",
             label=f"Hill sphere ({R_HILL_KM/1e3:.0f}e3 km)")
    moon_r_disp = R_MOON * 3
    ax2.fill(moon_r_disp * np.cos(phi2d), moon_r_disp * np.sin(phi2d),
             color=(0.63, 0.63, 0.63, 0.5), zorder=5)
    ax2.plot(moon_r_disp * np.cos(phi2d), moon_r_disp * np.sin(phi2d),
             color="#AAAAAA", linewidth=1, zorder=5)


    # ---- Animated 2D objects --------------------------------------------------
    trail_w2, = ax2.plot([], [], color=COL_WSB, linewidth=3, alpha=0.90, zorder=6)
    trail_a2, = ax2.plot([], [], color=COL_ART, linewidth=3, alpha=0.90, zorder=6)
    dot_w2,   = ax2.plot([], [], "o", color=COL_WSB, markersize=9, zorder=10,
                         markeredgecolor="white", markeredgewidth=0.8)
    dot_a2,   = ax2.plot([], [], "o", color=COL_ART, markersize=9, zorder=10,
                         markeredgecolor="white", markeredgewidth=0.8)

    # ---- Titles / legend ------------------------------------------------------
    from matplotlib.lines import Line2D
    fig.text(0.5, 0.98, "WSB Ballistic Capture vs Artemis II",
             color="white", fontsize=13, ha="center", va="top", fontweight="bold")
    fig.text(0.5, 0.955, "ECI frame (left)  |  Moon-centred SOI (right)",
             color="white", fontsize=9, ha="center", va="top")
    time_txt = fig.text(0.5, 0.928, "", color="#CCCCEE", fontsize=9,
                        ha="center", va="top")

    legend_handles = [
        Line2D([0], [0], color=COL_WSB, linewidth=2.5, label="Ballistic Capture"),
        Line2D([0], [0], color=COL_ART, linewidth=2.5, label="Artemis II"),
        Line2D([0], [0], color=COL_HILL, linewidth=1.5, linestyle=":", label="Hill sphere"),
    ]
    ax2.legend(handles=legend_handles, loc="upper right", fontsize=8,
               framealpha=0.5, facecolor="#14141E", edgecolor="#333355",
               labelcolor="white")

    # ---- Update function -------------------------------------------------------
    def update(fi: int):
        ft = frames_t[fi]
        wx,  wy,  wz  = interp_pos(wsb_t, wsb_x,  wsb_y,  wsb_z,  ft)
        ax_, ay_, az_ = interp_pos(art_t, art_x,  art_y,  art_z,  ft)
        mxw, myw, _   = interp_pos(wsb_t, wsb_mx, wsb_my, wsb_mz, ft)  # WSB Moon (for WSB SOI)
        mxa, mya, mza = interp_pos(art_t, art_mx, art_my, art_mz, ft)  # Artemis Moon (DE440S)

        w_mask = wsb_t <= ft
        a_mask = art_t <= ft

        wtx3, wty3, wtz3 = wsb_x[w_mask], wsb_y[w_mask], wsb_z[w_mask]
        atx3, aty3, atz3 = art_x[a_mask], art_y[a_mask], art_z[a_mask]
        if len(wtx3) > MAX_TRAIL:
            ti = downsample(np.arange(len(wtx3)), MAX_TRAIL)
            wtx3, wty3, wtz3 = wtx3[ti], wty3[ti], wtz3[ti]
        if len(atx3) > MAX_TRAIL:
            ti = downsample(np.arange(len(atx3)), MAX_TRAIL)
            atx3, aty3, atz3 = atx3[ti], aty3[ti], atz3[ti]
        trail_w3.set_data_3d(wtx3, wty3, wtz3)
        trail_a3.set_data_3d(atx3, aty3, atz3)
        dot_w3.set_data_3d([wx], [wy], [wz])
        dot_a3.set_data_3d([ax_], [ay_], [az_])
        dot_m3.set_data_3d([mxa], [mya], [mza])

        wtx, wty = wsb_x_mc[w_mask], wsb_y_mc[w_mask]
        atx, aty = art_x_mc[a_mask], art_y_mc[a_mask]
        if len(wtx) > MAX_TRAIL:
            ti = downsample(np.arange(len(wtx)), MAX_TRAIL)
            wtx, wty = wtx[ti], wty[ti]
        if len(atx) > MAX_TRAIL:
            ti = downsample(np.arange(len(atx)), MAX_TRAIL)
            atx, aty = atx[ti], aty[ti]
        trail_w2.set_data(wtx, wty)
        trail_a2.set_data(atx, aty)
        dot_w2.set_data([wx - mxw], [wy - myw])
        dot_a2.set_data([ax_ - mxa], [ay_ - mya])

        time_txt.set_text(f"T = {ft - t_min:+.1f} days  (T=0 = WSB departure)")
        return (trail_w3, trail_a3, dot_w3, dot_a3, dot_m3,
                trail_w2, trail_a2, dot_w2, dot_a2, time_txt)

    anim = FuncAnimation(fig, update, frames=N_FRAMES, blit=False,
                         interval=1000 // FPS)
    writer = FFMpegWriter(fps=FPS, bitrate=8000,
                          metadata=dict(title="WSB Ballistic Capture vs Artemis II"))
    print(f"Rendering {N_FRAMES} frames at {FPS} fps -> {OUT_MP4}  (this may take a few minutes) ...")
    anim.save(str(OUT_MP4), writer=writer, dpi=100)
    print(f"Saved video -> {OUT_MP4}")
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    print("Loading Artemis II trajectory + deriving ECI frame ...")
    (R0, u_sun_eci, _r_sun_km, art_t, art_x, art_y, art_z,
     art_mx, art_my, art_mz) = load_artemis()

    # Print frame orientation
    moon_inc_deg = np.degrees(np.arctan2(R0[2, 0], np.sqrt(R0[0,0]**2 + R0[1,0]**2)))
    print(f"  R0 (EM->ECI) -- Moon inclination component: {moon_inc_deg:.2f} deg from equatorial X-axis")
    print(f"  Artemis: {len(art_t)} points, T+{art_t[-1]:.1f} d")
    print(f"  Sun ECI (from anise CSV): [{u_sun_eci[0]:.4f}, {u_sun_eci[1]:.4f}, {u_sun_eci[2]:.4f}]")

    print("Loading WSB trajectory + epoch correction ...")
    (wsb_t, wsb_x, wsb_y, wsb_z,
     wsb_mx, wsb_my, wsb_mz, ic_info,
     wsb_departure_days, R0_wsb) = load_wsb(R0, u_sun_eci)

    apo_r = float(np.max(np.sqrt(wsb_x**2 + wsb_y**2 + wsb_z**2)))
    print(f"  WSB ({ic_info['source']}): {len(wsb_t)} points, "
          f"T+{wsb_t[-1] - wsb_departure_days:.1f} d flight, apogee {apo_r/1e3:.0f}e3 km")
    if not np.isnan(ic_info.get("theta_sun_deg", float("nan"))):
        sun_in_em = R0.T @ u_sun_eci
        theta_sun_artemis = np.degrees(np.arctan2(sun_in_em[1], sun_in_em[0]))
        print(f"  theta_sun (Artemis departure 2026-04-02): {theta_sun_artemis:.2f} deg")
        print(f"  theta_sun (WSB solution):                 {ic_info['theta_sun_deg']:.2f} deg")
        print(f"  WSB departure offset:                     {wsb_departure_days:+.2f} days")

    print("Computing IC overlay ...")
    ic_overlay = compute_ic_overlay(ic_info, R0_wsb)
    if ic_overlay is not None:
        print(f"  TLI DeltaV  = {ic_overlay['dv_kms']:.4f} km/s")
        print(f"  Parking alt = {ic_overlay['park_alt_km']:.0f} km")
        print(f"  theta_inj   = {ic_overlay['theta_deg']:.2f} deg  "
              f"(EM rotating frame)")
        print(f"  theta_sun   = {ic_overlay['theta_sun_deg']:.2f} deg")
        print(f"  r_apogee    = {ic_overlay['r_apogee_km']/1e3:.2f}e3 km")
        print(f"  est_orbits  = {ic_overlay['est_orbits']:.2f}")

    WSB_DIR.mkdir(parents=True, exist_ok=True)

    print("Building interactive 3D animation ...")
    fig = build_plotly(
        wsb_t, wsb_x, wsb_y, wsb_z, wsb_mx, wsb_my, wsb_mz,
        art_t, art_x, art_y, art_z, art_mx, art_my, art_mz,
        ic_overlay, R0,
    )
    fig.write_html(str(OUT_HTML), auto_play=False)
    print(f"Saved 3D animation -> {OUT_HTML}")

    print("Building static portrait ...")
    build_portrait(
        wsb_t, wsb_x, wsb_y, wsb_z, wsb_mx, wsb_my, wsb_mz,
        art_t, art_x, art_y, art_z, art_mx, art_my, art_mz,
        ic_overlay, ic_info, R0,
    )

    print("Building LinkedIn MP4 video ...")
    build_video(
        wsb_t, wsb_x, wsb_y, wsb_z, wsb_mx, wsb_my, wsb_mz,
        art_t, art_x, art_y, art_z, art_mx, art_my, art_mz,
        ic_overlay, R0,
    )


if __name__ == "__main__":
    main()
