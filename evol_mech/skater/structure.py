"""Skater mech: derived masses, leg kinematics and foot layout. Never genes.

Mass rules from docs/skater_design.md:
  - every leg-joint motor sits in the hip cluster (ten QDD motors);
  - below the hip are tubes, pushrods and belts, plus the tread drive (motor
    and worm) on the shin at the knee, plus the sole, rollers and rubber;
  - each leg's distal mass should stay at or below ~10-12 % of total mass;
    `leg_mass_fraction` reports it and the rollout penalises the excess.

The core, battery and fan placement rules are shared with the biped
(`body/structure.py`): the core follows the hip width, the battery sits along
x inside it, the fans hang off the core sides by `fan_gap`.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ..body.hardware import (fan_from_params, leg_motor_from_params, tread_drive_from_params,
                             PITCH_SERVO_KG, TREAD_RUBBER_KG_PER_M2, ROLLER_KG_PER_M,
                             SOLE_PLATE_KG_PER_M2)
from ..body.structure import (HIP_HALF_X, HIP_HALF_Z, SHELL_KG_PER_M2, HIP_HARDWARE_KG,
                              FAN_ARM_KG_PER_M, ELECTRONICS_KG, BATTERY_KG_BASE, BATTERY_KG_PER_MAH,
                              hip_half_y, core_half_x, core_half_y, payload_z, fan_center_y,
                              fan_z_core, battery_half, _box_area)

LINK_RADIUS = 0.008             # thigh / shin tube radius (16 mm CF tube)
LINK_KG_PER_M = 0.10            # tube + pushrod + belt run + bearings, per metre
JOINT_BRACKET_KG = 0.025        # knee and ankle brackets, each
SOLE_HALF_W = 0.020             # sole plate half-width
SOLE_HALF_T = 0.004             # sole plate half-thickness
ANKLE_PLATE_GAP = 0.006         # ankle axis above the plate top (the coaxial pulley sits there)
ROLLER_HALF_LEN = 0.015         # tread width 30 mm
ROLLER_GAP = 0.002              # clearance between adjacent rollers
MAX_ROLLERS = 6
WHEEL_RADIUS_FACTOR = 1.5       # foot_type 1: heel and toe wheels, larger than the tread rollers
LEG_MASS_FRACTION_TARGET = 0.12

HIP_YAW_RANGE = math.radians(90)
HIP_ROLL_RANGE = math.radians(35)
HIP_PITCH_RANGE = (math.radians(-120), math.radians(60))   # negative = thigh forward
KNEE_RANGE = (0.0, math.radians(150))                      # positive = shin back
ANKLE_RANGE = math.radians(70)


class InvalidBody(ValueError):
    """Raised when the genome decodes to a body that cannot be built."""


# ---- foot layout ----------------------------------------------------------------

@dataclass(frozen=True)
class FootLayout:
    roller_r: float
    roller_x: tuple[float, ...]     # roller centres along the sole, ankle at x = 0
    plate_z: float                  # sole plate centre below the ankle axis
    roller_z: float                 # roller axis below the ankle axis
    bottom_z: float                 # lowest point below the ankle axis


def foot_layout(p: dict[str, float]) -> FootLayout:
    L, r = p["sole_len"], p["pulley_radius"]
    if p["foot_type"] >= 0.5:                      # heel + toe wheels
        rw = min(WHEEL_RADIUS_FACTOR * r, (L - ROLLER_GAP) / 4)
        xs = (-(L / 2 - rw), L / 2 - rw)
        r = rw
    else:                                          # row of tread rollers
        if L < 4 * r + ROLLER_GAP:
            raise InvalidBody(f"sole {L*1e3:.0f} mm too short for {r*1e3:.0f} mm rollers")
        n = min(MAX_ROLLERS, int((L - 2 * r) / (2 * r + ROLLER_GAP)) + 1)
        xs = tuple(-(L / 2 - r) + (L - 2 * r) * k / (n - 1) for k in range(n))
    plate_z = -(ANKLE_PLATE_GAP + SOLE_HALF_T)
    roller_z = plate_z - SOLE_HALF_T - r
    return FootLayout(r, xs, plate_z, roller_z, roller_z - r)


# ---- leg kinematics (sagittal plane) -------------------------------------------
# Joint sign convention (hinge about +y): a positive angle swings the distal
# link backward. Nominal crouch: knee = crouch, hip pitch chosen so the ankle
# sits directly below the hip, ankle pitch keeps the sole level.

def nominal_angles(p: dict[str, float]) -> tuple[float, float, float]:
    """(hip_pitch, knee, ankle) of the parked crouch."""
    L1, L2, k = p["thigh_len"], p["shin_len"], p["crouch"]
    hip = -math.atan2(L2 * math.sin(k), L1 + L2 * math.cos(k))
    return hip, k, -(hip + k)


def ankle_drop(p: dict[str, float], hip: float, knee: float) -> float:
    """Vertical distance from hip joint to ankle axis for given joint angles."""
    return p["thigh_len"] * math.cos(hip) + p["shin_len"] * math.cos(hip + knee)


def nominal_height(p: dict[str, float]) -> float:
    """Hip-joint height above the floor in the parked crouch."""
    hip, knee, _ = nominal_angles(p)
    return ankle_drop(p, hip, knee) - foot_layout(p).bottom_z


def max_height(p: dict[str, float]) -> float:
    return p["thigh_len"] + p["shin_len"] - foot_layout(p).bottom_z


def spawn_height(p: dict[str, float], terrain_zmax: float = 0.0) -> float:
    return nominal_height(p) + 0.01 + terrain_zmax


# ---- masses ---------------------------------------------------------------------

@dataclass(frozen=True)
class SkaterStructure:
    hip_kg: float               # shell + hardware + ten leg motors
    core_kg: float              # shell + fan arms
    core_half_x: float
    core_half_y: float
    payload_kg: float
    payload_half: tuple[float, float, float]
    thigh_kg: float
    shin_kg: float              # tube + bracket + tread drive (motor, worm, belt)
    foot_kg: float              # plate + rubber + rollers
    fan_kg: float               # each, incl. pitch and roll servos
    total_kg: float
    leg_fraction: float         # (thigh + shin + foot) / total, per leg


def structure_from_params(p: dict[str, float]) -> SkaterStructure:
    motor = leg_motor_from_params(p)
    tread = tread_drive_from_params(p)
    fan = fan_from_params(p)
    lay = foot_layout(p)

    hip = (SHELL_KG_PER_M2 * _box_area(HIP_HALF_X, hip_half_y(p), HIP_HALF_Z)
           + HIP_HARDWARE_KG + 10 * motor.mass_kg)
    arm = math.sqrt(p["fan_x"] ** 2 + fan_center_y(p) ** 2 + fan_z_core(p) ** 2)
    cx, cy = core_half_x(p), core_half_y(p)
    core = SHELL_KG_PER_M2 * _box_area(cx, cy, p["core_height"] / 2) + 2 * FAN_ARM_KG_PER_M * arm
    battery = BATTERY_KG_BASE + BATTERY_KG_PER_MAH * p["battery_mah"]
    payload = battery + ELECTRONICS_KG

    thigh = LINK_KG_PER_M * p["thigh_len"] + JOINT_BRACKET_KG
    shin = LINK_KG_PER_M * p["shin_len"] + JOINT_BRACKET_KG + tread.mass_kg
    sole_area = p["sole_len"] * 2 * SOLE_HALF_W
    foot = (SOLE_PLATE_KG_PER_M2 * sole_area + TREAD_RUBBER_KG_PER_M2 * 2 * sole_area
            + ROLLER_KG_PER_M * 2 * ROLLER_HALF_LEN * len(lay.roller_x))
    fan_kg = fan.mass_kg + PITCH_SERVO_KG          # second vectoring servo (roll)
    total = hip + core + payload + 2 * (thigh + shin + foot + fan_kg)
    return SkaterStructure(hip, core, cx, cy, payload, battery_half(p), thigh, shin, foot,
                           fan_kg, total, (thigh + shin + foot) / total)


def leg_mass_fraction(p: dict[str, float]) -> float:
    return structure_from_params(p).leg_fraction


def check_buildable(p: dict[str, float]) -> None:
    """Raise InvalidBody for geometry that cannot be assembled."""
    foot_layout(p)
    if nominal_height(p) < 0.05:
        raise InvalidBody("crouch leaves the hip on the floor")
    # feet must not meet under the hip when parallel (sole width + roller stick-out)
    if p["hip_width"] / 2 < SOLE_HALF_W + 0.005:
        raise InvalidBody("hip too narrow for the soles")
    if core_half_y(p) < battery_half(p)[1]:
        raise InvalidBody("battery wider than the core")


def is_buildable(p: dict[str, float]) -> bool:
    try:
        check_buildable(p)
    except InvalidBody:
        return False
    return True
