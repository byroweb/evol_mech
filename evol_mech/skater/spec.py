"""Genome for the stage-1 skater mech (docs/skater_design.md).

Same conventions as `genome/spec.py`: a flat vector in [0, 1], decoded through
the bounds below; catalogue parts carry a `step`; masses are never genes.

Topology is fixed: hip (all ten leg-joint motors inside) -> per leg hip yaw,
hip roll, hip pitch, knee, ankle pitch -> flat sole with a powered tread (or a
heel and toe wheel pair). The core sits on the hip and carries the battery and
two fans vectored in pitch and roll.
"""
from __future__ import annotations

import numpy as np

from ..genome.spec import Param, DEG

PARAMS: list[Param] = [
    # ---- body geometry (metres) ----------------------------------------------
    Param("hip_width",        0.08, 0.32, 0.18, "geometry"),   # between leg planes
    Param("core_height",      0.05, 0.30, 0.14, "geometry"),
    Param("battery_mah",      1000, 6000, 2200, "geometry", 100),
    Param("payload_x",       -0.06, 0.06, 0.00, "geometry"),
    Param("payload_zf",      -0.45, 0.45, -0.20, "geometry"),
    Param("fan_x",           -0.12, 0.12, 0.00, "geometry"),
    Param("fan_gap",          0.00, 0.15, 0.03, "geometry"),   # prop inboard edge to core side
    Param("fan_zf",           0.00, 1.00, 0.50, "geometry"),   # height, fraction of core height
    # ---- leg (both identical) -------------------------------------------------
    Param("thigh_len",        0.08, 0.26, 0.15, "leg"),
    Param("shin_len",         0.08, 0.26, 0.15, "leg"),
    Param("crouch",           10*DEG, 100*DEG, 40*DEG, "leg"),  # nominal knee flexion
    Param("ankle_lever",      0.5,  2.0,  1.0,  "leg"),         # ankle pushrod lever ratio: torque x, speed /
    # ---- foot -----------------------------------------------------------------
    Param("foot_type",        0,    1,    0,    "foot", 1),     # 0 = tread (roller row), 1 = heel + toe wheels
    Param("sole_len",         0.08, 0.26, 0.14, "foot"),
    Param("pulley_radius",    0.010, 0.035, 0.018, "foot"),     # tread drive pulley = roller radius
    Param("worm_ratio",       8.0,  25.0, 15.0, "foot"),
    # ---- drivetrain -----------------------------------------------------------
    Param("leg_motor_torque", 0.5,  6.0,  2.0,  "drivetrain"),  # Nm peak at the joint (QDD)
    Param("leg_kp",           10.0, 300.0, 30.0, "drivetrain"), # joint position stiffness Nm/rad
    Param("tread_motor_w",    20.0, 150.0, 60.0, "drivetrain"), # rated electrical power
    # ---- fan hardware spec (catalogue steps) ---------------------------------
    Param("fan_diam_in",      2,    6,    3,    "fan_spec", 1),
    Param("fan_blades",       2,    6,    3,    "fan_spec", 1),
    Param("stator_d_mm",      12,   28,   18,   "fan_spec", 1),
    Param("stator_h_mm",      4,    10,   6,    "fan_spec", 1),
    # ---- suspension and leveling ---------------------------------------------
    Param("kp_height",        0.0,  40.0, 5.0,  "suspension"),  # knee rad per m of height error
    Param("kd_height",        0.0,  6.0,  0.2,  "suspension"),
    Param("kp_roll_lvl",     -3.0,  3.0,  1.0,  "suspension"),  # hip roll per rad of body roll
    Param("kd_roll_lvl",     -0.5,  0.5,  0.1,  "suspension"),
    Param("kp_pitch_lvl",    -3.0,  3.0,  0.0,  "suspension"),  # hip pitch per rad of body pitch error
    Param("kd_pitch_lvl",    -0.5,  0.5,  0.0,  "suspension"),
    Param("kp_ankle_lvl",    -3.0,  3.0,  0.3,  "suspension"),  # ankle (toe-down) per rad of body pitch error
    Param("kd_ankle_lvl",    -0.5,  0.5,  0.05, "suspension"),
    Param("lean_gain",        0.0,  0.30, 0.06, "suspension"),  # hip pitch (feet back) per m/s^2 demanded: lean into acceleration
    # ---- force allocation and steering ---------------------------------------
    Param("vel_kp",           0.3,  6.0,  2.0,  "alloc"),       # demanded accel per (m/s) of velocity error
    Param("tread_share",      0.0,  1.0,  0.8,  "alloc"),       # fraction of in-line force asked of the treads first
    Param("friction_margin",  0.2,  1.0,  0.6,  "alloc"),       # tread force cap as a fraction of mu*N
    Param("steer_kp",         0.3,  6.0,  2.0,  "steer"),       # yaw rate per rad of heading error
    Param("path_kp",          0.3,  6.0,  1.5,  "steer"),       # heading per m of lateral path error
    # ---- fan attitude controller --------------------------------------------
    Param("fan_base",         0.0,  1.0,  0.15, "fan_ctrl"),
    Param("fan_kp_roll",     -4.0,  4.0,  0.5,  "fan_ctrl"),
    Param("fan_kd_roll",     -1.0,  1.0,  0.05, "fan_ctrl"),
    Param("fan_pitch_bias",  -60*DEG, 60*DEG, 0.0, "fan_ctrl"),
    Param("fan_kp_pitch",    -4.0,  4.0,  1.0,  "fan_ctrl"),
    Param("fan_kd_pitch",    -1.0,  1.0,  0.1,  "fan_ctrl"),
]

NAMES = [p.name for p in PARAMS]
INDEX = {p.name: i for i, p in enumerate(PARAMS)}
GROUPS = sorted({p.group for p in PARAMS})
LO = np.array([p.lo for p in PARAMS], dtype=float)
HI = np.array([p.hi for p in PARAMS], dtype=float)
DEFAULT_NORM = (np.array([p.default for p in PARAMS], dtype=float) - LO) / (HI - LO)
N_PARAMS = len(PARAMS)


def snap(name: str, v: float) -> float:
    s = PARAMS[INDEX[name]].step
    return float(v) if s is None else float(round(v / s) * s)


def decode(genome: np.ndarray) -> dict[str, float]:
    g = np.clip(np.asarray(genome, dtype=float), 0.0, 1.0)
    phys = LO + g * (HI - LO)
    return {n: snap(n, v) for n, v in zip(NAMES, phys)}


def encode(phys: dict[str, float]) -> np.ndarray:
    g = DEFAULT_NORM.copy()
    for k, v in phys.items():
        i = INDEX[k]
        g[i] = (v - LO[i]) / (HI[i] - LO[i])
    return np.clip(g, 0.0, 1.0)
