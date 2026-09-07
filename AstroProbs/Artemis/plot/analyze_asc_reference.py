#!/usr/bin/env python3
"""Analyze CCSDS OEM ASC reference states near the Artemis TLI epoch."""

from __future__ import annotations

import argparse
from datetime import datetime
from math import acos, atan2, cos, sqrt, sin, tau
from pathlib import Path
from typing import Iterable, NamedTuple

MU_EARTH = 3.986004418e14  # m^3 / s^2


class AscState(NamedTuple):
    epoch: datetime
    position_m: tuple[float, float, float]
    velocity_ms: tuple[float, float, float]


def parse_asc_file(path: Path) -> tuple[str, list[AscState]]:
    frame = "UNKNOWN"
    states: list[AscState] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("COMMENT"):
                continue
            if "REF_FRAME" in line:
                frame = line.split("=", 1)[1].strip()
                continue
            if line[0].isdigit():
                parts = line.split()
                epoch = datetime.fromisoformat(parts[0])
                coords = [float(x) for x in parts[1:7]]
                # ASC OEM stores kilometers and kilometers per second
                pos = tuple(coord * 1000.0 for coord in coords[:3])
                vel = tuple(coord * 1000.0 for coord in coords[3:6])
                states.append(AscState(epoch=epoch, position_m=pos, velocity_ms=vel))

    return frame, states


def closest_state(states: Iterable[AscState], target: datetime) -> AscState:
    return min(states, key=lambda state: abs((state.epoch - target).total_seconds()))


def unit_vector(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    return (vec[0] / norm, vec[1] / norm, vec[2] / norm)


def vector_norm(vec: tuple[float, float, float]) -> float:
    return sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


def cross(u: tuple[float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def dot(u: tuple[float, float, float], v: tuple[float, float, float]) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def angle_between(u: tuple[float, float, float], v: tuple[float, float, float]) -> float:
    cos_a = dot(u, v) / (vector_norm(u) * vector_norm(v))
    return acos(max(-1.0, min(1.0, cos_a)))


def normalize_angle(rad: float) -> float:
    return rad % tau


def orbital_elements_from_state(position: tuple[float, float, float], velocity: tuple[float, float, float]) -> dict[str, float]:
    r = position
    v = velocity
    r_norm = vector_norm(r)
    v_norm = vector_norm(v)
    h = cross(r, v)
    h_norm = vector_norm(h)
    rv_dot = dot(r, v)

    e_vec = tuple(((v_norm * v_norm - MU_EARTH / r_norm) * r[i] - rv_dot * v[i]) / MU_EARTH for i in range(3))
    e = vector_norm(e_vec)
    a = 1.0 / (2.0 / r_norm - v_norm * v_norm / MU_EARTH)

    inclination = acos(max(-1.0, min(1.0, h[2] / h_norm)))
    n = cross((0.0, 0.0, 1.0), h)
    n_norm = vector_norm(n)

    if n_norm < 1e-12:
        raan = 0.0
    else:
        raan = atan2(n[1], n[0])
        raan = normalize_angle(raan)

    if e < 1e-12 or n_norm < 1e-12:
        arg_pe = 0.0
    else:
        cos_arg_pe = max(-1.0, min(1.0, dot(n, e_vec) / (n_norm * e)))
        arg_pe = acos(cos_arg_pe)
        if e_vec[2] < 0.0:
            arg_pe = tau - arg_pe
        arg_pe = normalize_angle(arg_pe)

    if e < 1e-12:
        true_anomaly = 0.0
    else:
        cos_nu = max(-1.0, min(1.0, dot(e_vec, r) / (e * r_norm)))
        true_anomaly = acos(cos_nu)
        if rv_dot < 0.0:
            true_anomaly = tau - true_anomaly
        true_anomaly = normalize_angle(true_anomaly)

    return {
        "a_m": a,
        "e": e,
        "i_rad": inclination,
        "raan_rad": raan,
        "arg_pe_rad": arg_pe,
        "nu_rad": true_anomaly,
        "r_norm_m": r_norm,
        "v_norm_ms": v_norm,
        "h_norm": h_norm,
    }


def format_deg(rad: float) -> str:
    return f"{rad * 180.0 / 3.141592653589793:.8f}°"


def print_summary(frame: str, state: AscState, elements: dict[str, float], target: datetime) -> None:
    print("ASC reference analysis")
    print("----------------------")
    print(f"REF_FRAME: {frame}")
    print(f"Target firing epoch: {target.isoformat()}")
    print(f"Matched epoch:        {state.epoch.isoformat()}")
    print(f"Epoch offset:         {abs((state.epoch - target).total_seconds()):.3f} seconds")
    print()
    print("State vector (EME2000 / ECI)")
    print(f"  Position: {state.position_m[0]:.3f}, {state.position_m[1]:.3f}, {state.position_m[2]:.3f} m")
    print(f"  Velocity: {state.velocity_ms[0]:.9f}, {state.velocity_ms[1]:.9f}, {state.velocity_ms[2]:.9f} m/s")
    print(f"  |r| = {elements['r_norm_m'] / 1000.0:.6f} km")
    print(f"  |v| = {elements['v_norm_ms']:.9f} km/s")
    print()
    print("Keplerian elements")
    print(f"  Semi-major axis (a):      {elements['a_m'] / 1000.0:.6f} km")
    print(f"  Eccentricity (e):         {elements['e']:.9f}")
    print(f"  Inclination (i):          {format_deg(elements['i_rad'])}")
    print(f"  RAAN (Ω):                 {format_deg(elements['raan_rad'])}")
    print(f"  Argument of periapsis (ω): {format_deg(elements['arg_pe_rad'])}")
    print(f"  True anomaly (ν):         {format_deg(elements['nu_rad'])}")
    print(f"  Specific angular momentum: {elements['h_norm']:.6e} m^2/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a CCSDS OEM ASC reference trajectory near the Artemis TLI epoch.")
    parser.add_argument(
        "--asc",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "input/Artemis_II_OEM_2026_04_04_to_EI.asc",
        help="Path to the ASC reference file",
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default="2026-04-02T23:49:00",
        help="Target firing epoch in ISO format (UTC)",
    )
    args = parser.parse_args()

    target_epoch = datetime.fromisoformat(args.epoch)
    frame, states = parse_asc_file(args.asc)
    if not states:
        raise SystemExit(f"No state rows found in ASC file: {args.asc}")

    state = closest_state(states, target_epoch)
    elements = orbital_elements_from_state(state.position_m, state.velocity_ms)
    print_summary(frame, state, elements, target_epoch)


if __name__ == "__main__":
    main()
