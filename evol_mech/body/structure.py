"""Structure and payload masses, derived from dimensions. Never genes.

If masses were free parameters evolution would dial every one to its lower
bound. Instead the hip and core shells weigh what a printed box of their size
weighs, the bracket posts and fan arms weigh by length, and the payload is a
fixed electronics allowance plus a 4S LiPo whose mass follows its capacity.
Constants are rough catalogue values; replace with weighed parts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

HIP_HALF_X = 0.05               # hip box (drivetrain housing), fore/aft half-size
HIP_HALF_Z = 0.035              # hip box half-height; pivot B sits on a post above it
CORE_HALF_XY = 0.05             # core box footprint half-size (x grows to fit the battery)
BATTERY_CLEARANCE = 0.005
LINK_RADIUS = 0.006
LEG_SIDE_GAP = 0.002            # leg plane clearance outside the hip side face
SHELL_KG_PER_M2 = 2.4           # ~2 mm printed walls with ribs, per m^2 of box surface
HIP_HARDWARE_KG = 0.06          # bearings, shafts, couplers
POST_KG_PER_M = 0.08            # bracket post A->B, each leg (10 mm Al/CF section)
FAN_ARM_KG_PER_M = 0.06         # core -> fan arm, each
ELECTRONICS_KG = 0.25           # flight controller, SBC, ESCs, drivers, wiring
# 4S LiPo: ~20 g pack overhead + 0.1 g/mAh (1500 -> 170 g, 5000 -> 520 g)
BATTERY_KG_BASE = 0.020
BATTERY_KG_PER_MAH = 1.0e-4
BATTERY_M3_PER_MAH = 4.9e-8     # 2200 mAh ~ 105 x 34 x 30 mm
BATTERY_ASPECT = (3.5, 1.13, 1.0)   # long : mid : short; mounted long axis along x (fore/aft), flat


def hip_half_y(p: dict[str, float]) -> float:
    return p["hip_width"] / 2 - LINK_RADIUS - LEG_SIDE_GAP


def _box_area(hx: float, hy: float, hz: float) -> float:
    return 8 * (hx * hy + hy * hz + hx * hz)


@dataclass(frozen=True)
class StructureSpec:
    hip_kg: float               # shell + hardware + both posts (excl. steppers)
    core_kg: float              # shell + both fan arms
    core_half_x: float          # core footprint fore/aft half-length (>= CORE_HALF_XY)
    battery_kg: float
    payload_kg: float           # battery + electronics
    payload_half: tuple[float, float, float]


def battery_half(p: dict[str, float]) -> tuple[float, float, float]:
    """Half-sizes (x, y, z): long axis fore/aft (x), thinnest axis vertical."""
    vol = BATTERY_M3_PER_MAH * p["battery_mah"]
    long_, mid, short = BATTERY_ASPECT
    unit = (vol / (8 * long_ * mid * short)) ** (1 / 3)
    return long_ * unit, mid * unit, short * unit


def core_half_x(p: dict[str, float]) -> float:
    """Core lengthens to enclose the battery wherever payload_x puts it."""
    return max(CORE_HALF_XY, battery_half(p)[0] + abs(p["payload_x"]) + BATTERY_CLEARANCE)


def payload_z(p: dict[str, float]) -> float:
    """Battery centre height relative to the core centre: payload_zf of the
    core height, clamped so the pack stays inside the shell."""
    zmax = max(0.0, p["core_height"] / 2 - battery_half(p)[2] - BATTERY_CLEARANCE)
    return float(min(zmax, max(-zmax, p["payload_zf"] * p["core_height"])))


def core_half_y(p: dict[str, float]) -> float:
    return max(CORE_HALF_XY, battery_half(p)[1] + BATTERY_CLEARANCE)


def fan_center_y(p: dict[str, float]) -> float:
    """Lateral fan centre: core side face + gap + prop radius, so widening the
    core or the prop moves the fan outward instead of into something."""
    return core_half_y(p) + p["fan_gap"] + p["fan_diam_in"] * 0.0254 / 2


def structure_from_params(p: dict[str, float]) -> StructureSpec:
    post = math.hypot(p["pivot_b_dx"], p["pivot_b_dz"])
    hip = (SHELL_KG_PER_M2 * _box_area(HIP_HALF_X, hip_half_y(p), HIP_HALF_Z)
           + HIP_HARDWARE_KG + 2 * POST_KG_PER_M * post)
    arm = math.sqrt(p["fan_x"] ** 2 + fan_center_y(p) ** 2 + (p["fan_z"] - p["core_height"] / 2) ** 2)  # from core centre
    cx = core_half_x(p)
    core = (SHELL_KG_PER_M2 * _box_area(cx, core_half_y(p), p["core_height"] / 2)
            + 2 * FAN_ARM_KG_PER_M * arm)
    battery = BATTERY_KG_BASE + BATTERY_KG_PER_MAH * p["battery_mah"]
    return StructureSpec(hip, core, cx, battery, battery + ELECTRONICS_KG, battery_half(p))
