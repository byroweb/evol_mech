"""Parameter specification for the fixed-topology, crank-driven biped.

The genome is a flat vector in [0, 1]. `decode` maps it to physical values via
the bounds below. Keeping everything normalised makes mutation sigma mean the
same thing for a 2 cm crank and a 15 N fan.

Topology is fixed: hip (steppers inside) -> two 4-bar linkage legs, each driven
by a stepper on the crank; the core on top (optionally on a yaw pivot, see the
run config's "waist_yaw"); two pitch-vectored fans on the core. Everything
else is a number in this table.

Masses are NOT genes. Structure mass follows from dimensions and the payload
from a fixed electronics allowance plus the battery capacity
(`body/hardware.py`), otherwise evolution would simply dial every mass to its
lower bound. Parts people buy in whole units (prop inches, blades, stator mm,
battery mAh) carry a `step` and are rounded on decode.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    default: float
    group: str  # "geometry" | "linkage" | "drivetrain" | "fan_spec" | "gait" | "fan_ctrl"
    step: float | None = None   # catalogue granularity; decode rounds to it


DEG = math.pi / 180.0

PARAMS: list[Param] = [
    # ---- body geometry (metres) ----------------------------------------------
    Param("hip_width",       0.08, 0.30, 0.16, "geometry"),   # between leg planes
    Param("core_height",     0.05, 0.30, 0.14, "geometry"),
    # payload = 4S LiPo + electronics allowance, a dense block placed inside the core
    Param("battery_mah",     1000, 6000, 2200, "geometry", 100),
    Param("payload_x",      -0.06, 0.06, 0.00, "geometry"),   # fore/aft inside core
    Param("payload_zf",     -0.45, 0.45, -0.20, "geometry"),  # fraction of core height
    Param("fan_x",          -0.12, 0.12, 0.00, "geometry"),   # fore/aft on core
    Param("fan_y",           0.06, 0.30, 0.14, "geometry"),   # half-spread
    Param("fan_z",           0.00, 0.30, 0.12, "geometry"),   # above core centre
    Param("foot_type",       0,    1,    0,    "geometry", 1),  # 0 = flat pad, 1 = ball end
    Param("foot_radius",     0.008, 0.030, 0.015, "geometry"), # pad thickness or ball radius
    Param("foot_half_len",   0.005, 0.070, 0.035, "geometry"), # pad fore/aft half-length (pad only)
    # ---- 4-bar linkage, per leg (both legs identical), labels as in the sketch:
    # A crank pivot (hip origin), D crank tip, B rocker pivot = A + (pivot_b_dx, pivot_b_dz),
    # C coupler/rocker joint, P foot rigid to coupler CD.
    Param("crank_len",       0.015, 0.080, 0.035, "linkage"),   # |AD|
    Param("coupler_len",     0.050, 0.260, 0.110, "linkage"),   # |CD|
    Param("rocker_len",      0.050, 0.260, 0.140, "linkage"),   # |BC|
    Param("pivot_b_dx",     -0.150, 0.100, -0.050, "linkage"),  # B behind A
    Param("pivot_b_dz",     -0.050, 0.180, 0.100, "linkage"),   # B above A, on a post
    # P = D + foot_frac*(C-D) + foot_height*n, n perpendicular to CD, away from A
    Param("foot_frac",       0.00, 1.00, 0.45, "linkage"),
    Param("foot_height",     0.02, 0.32, 0.16, "linkage"),
    # ---- drivetrain ----------------------------------------------------------
    Param("stepper_torque",  0.02, 0.60, 0.20, "drivetrain"),  # N·m holding, NEMA 8..17
    Param("gear_ratio",      1.0,  8.0,  3.0,  "drivetrain"),
    Param("kp_yaw",          2.0,  30.0, 8.0,  "drivetrain"),  # inert unless waist_yaw
    # ---- fan hardware spec (catalogue steps) ---------------------------------
    Param("fan_diam_in",     2,    6,    3,    "fan_spec", 1),  # prop diameter, whole inches
    Param("fan_blades",      2,    6,    3,    "fan_spec", 1),
    Param("stator_d_mm",     12,   28,   18,   "fan_spec", 1),  # 1408 -> 14, 2106 -> 21
    Param("stator_h_mm",     4,    10,   6,    "fan_spec", 1),
    # ---- gait ----------------------------------------------------------------
    Param("crank_speed_hz", -3.0,  3.0,  1.0,  "gait"),        # sign = direction
    Param("crank_phase_r",   0.0,  2*math.pi, math.pi, "gait"),  # right crank offset
    Param("yaw_amp",       -30*DEG, 30*DEG, 0.0, "gait"),      # inert unless waist_yaw
    # ---- fan attitude controller --------------------------------------------
    Param("fan_base",        0.0,  1.0,  0.2,  "fan_ctrl"),      # throttle 0..1
    Param("fan_kp_roll",    -4.0,  4.0,  0.5,  "fan_ctrl"),
    Param("fan_kd_roll",    -1.0,  1.0,  0.05, "fan_ctrl"),
    Param("fan_pitch_bias", -60*DEG, 60*DEG, 10*DEG, "fan_ctrl"),  # forward tilt
    Param("fan_kp_pitch",   -4.0,  4.0,  1.0,  "fan_ctrl"),
    Param("fan_kd_pitch",   -1.0,  1.0,  0.1,  "fan_ctrl"),
]

NAMES = [p.name for p in PARAMS]
INDEX = {p.name: i for i, p in enumerate(PARAMS)}
LO = np.array([p.lo for p in PARAMS], dtype=float)
HI = np.array([p.hi for p in PARAMS], dtype=float)
DEFAULT_NORM = (np.array([p.default for p in PARAMS], dtype=float) - LO) / (HI - LO)
N_PARAMS = len(PARAMS)


def snap(name: str, v: float) -> float:
    """Round a physical value to its catalogue step, if it has one."""
    s = PARAMS[INDEX[name]].step
    return float(v) if s is None else float(round(v / s) * s)


def decode(genome: np.ndarray) -> dict[str, float]:
    """Normalised [0,1] vector -> dict of physical values (catalogue values snapped)."""
    g = np.clip(np.asarray(genome, dtype=float), 0.0, 1.0)
    phys = LO + g * (HI - LO)
    return {n: snap(n, v) for n, v in zip(NAMES, phys)}


def encode(phys: dict[str, float]) -> np.ndarray:
    """Dict of physical values -> normalised vector (missing keys use defaults)."""
    g = DEFAULT_NORM.copy()
    for k, v in phys.items():
        i = INDEX[k]
        g[i] = (v - LO[i]) / (HI[i] - LO[i])
    return np.clip(g, 0.0, 1.0)
