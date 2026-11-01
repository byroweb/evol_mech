"""Physical models for off-the-shelf parts: small BLDC fans and steppers.

These are deliberately simple, documented formulas with tunable constants, so
that a fan spec in the genome (diameter, blades, stator size) maps to thrust,
mass and spin-up lag the simulator can use. Replace constants with measured
values once real parts are on the bench.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

RHO_AIR = 1.225                 # kg/m^3

# BLDC outrunner continuous power ~ stator volume. 1408 (~130 W) and 2306
# (~300 W continuous) both sit near 0.09 W/mm^3.
MOTOR_W_PER_MM3 = 0.09
MOTOR_KG_PER_MM3 = 8.0e-6       # 1408 ~ 12 g, 2306 ~ 28 g
PROP_G_PER_INCH_PER_SQRT_BLADE = 0.7
DUCT_MOUNT_KG = 0.008
PITCH_SERVO_KG = 0.010          # micro servo that vectors the fan
FAN_PITCH_SERVO_KP = 2.0        # N·m/rad, SG90-class
FAN_PITCH_SERVO_TORQUE = 0.18   # N·m

# Steppers: holding torque -> mass, roughly linear across NEMA 8..17
STEPPER_KG_BASE = 0.05
STEPPER_KG_PER_NM = 0.45
STEPPER_MAX_RAD_S = 60.0        # ~570 rpm before torque collapses
GEARBOX_KG_PER_RATIO = 0.004


@dataclass(frozen=True)
class FanSpec:
    diam_m: float
    blades: int
    power_w: float
    max_thrust_n: float
    mass_kg: float
    tau_s: float                # spin-up time constant
    figure_of_merit: float


def fan_from_params(p: dict[str, float]) -> FanSpec:
    d = p["fan_diam_in"] * 0.0254
    blades = int(p["fan_blades"])
    area = math.pi * d * d / 4
    power = MOTOR_W_PER_MM3 * p["stator_d_mm"] ** 2 * p["stator_h_mm"]
    # more blades on a small prop trade efficiency for thrust density
    fom = min(0.70, max(0.45, 0.68 - 0.03 * (blades - 2)))
    # ideal hover: P = T^1.5 / (FoM * sqrt(2 rho A))  ->  T = (FoM P sqrt(2 rho A))^(2/3)
    t_max = (fom * power * math.sqrt(2 * RHO_AIR * area)) ** (2.0 / 3.0)
    mass = (MOTOR_KG_PER_MM3 * p["stator_d_mm"] ** 2 * p["stator_h_mm"]
            + PROP_G_PER_INCH_PER_SQRT_BLADE * p["fan_diam_in"] * math.sqrt(blades) * 1e-3
            + DUCT_MOUNT_KG + PITCH_SERVO_KG)
    tau = 0.02 + 0.015 * p["fan_diam_in"]
    return FanSpec(d, blades, power, t_max, mass, tau, fom)


@dataclass(frozen=True)
class StepperSpec:
    torque_nm: float            # at the motor
    gear: float
    mass_kg: float              # motor + gearbox
    crank_torque_nm: float
    crank_max_rad_s: float


def stepper_from_params(p: dict[str, float]) -> StepperSpec:
    t = p["stepper_torque"]
    g = p["gear_ratio"]
    mass = STEPPER_KG_BASE + STEPPER_KG_PER_NM * t + GEARBOX_KG_PER_RATIO * g
    return StepperSpec(t, g, mass, t * g * 0.9, STEPPER_MAX_RAD_S / g)
