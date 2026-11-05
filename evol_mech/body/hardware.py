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


# ---- skater mech parts (docs/skater_design.md) ----------------------------------
# Quasi-direct-drive leg joint: small outrunner + low-ratio planetary, sized by
# peak joint torque. GIM3505-class (1.2 Nm, 90 g) to GIM6010-class (6 Nm, 300 g).
QDD_KG_BASE = 0.045
QDD_KG_PER_NM = 0.042
QDD_MAX_RAD_S = 25.0            # joint speed where peak torque falls off (~240 rpm)
# Tread drive: small high-kv BLDC (power rating sets mass), self-locking worm,
# belt down the shin to a pulley coaxial with the ankle.
TREAD_MOTOR_KG_BASE = 0.012
TREAD_MOTOR_KG_PER_W = 0.0004   # 50 W -> 32 g
TREAD_MOTOR_MAX_RAD_S = 2000.0  # ~19 000 rpm, high-kv small outrunner
WORM_KG = 0.030                 # worm, wheel, housing, bearings
WORM_EFFICIENCY = 0.45          # self-locking lead angle; the design doc accepts 40-50 %
SHIN_BELT_KG = 0.012            # belt + two pulleys
TREAD_RUBBER_KG_PER_M2 = 1.2    # 2 mm rubber band, both runs, per m2 of sole area x2
ROLLER_KG_PER_M = 0.25          # roller + axle per metre of roller length
SOLE_PLATE_KG_PER_M2 = 3.0      # 3 mm CF/aluminium plate


@dataclass(frozen=True)
class LegMotorSpec:
    torque_nm: float            # peak joint torque after the planetary
    mass_kg: float              # per motor, lives in the hip cluster
    max_rad_s: float


def leg_motor_from_params(p: dict[str, float]) -> LegMotorSpec:
    t = p["leg_motor_torque"]
    return LegMotorSpec(t, QDD_KG_BASE + QDD_KG_PER_NM * t, QDD_MAX_RAD_S)


@dataclass(frozen=True)
class TreadDriveSpec:
    motor_w: float
    ratio: float
    pulley_r: float
    mass_kg: float              # motor + worm + belt, mounted at the knee on the shin
    max_speed_m_s: float        # tread speed at max motor rpm
    max_force_n: float          # tread force at rated power through the worm
    pulley_torque_nm: float


def tread_drive_from_params(p: dict[str, float]) -> TreadDriveSpec:
    pw, ratio, r = p["tread_motor_w"], p["worm_ratio"], p["pulley_radius"]
    omega_pulley = TREAD_MOTOR_MAX_RAD_S / ratio
    v_max = omega_pulley * r
    # torque at rated power and max rpm, multiplied by the worm, less its losses
    tau_pulley = pw / TREAD_MOTOR_MAX_RAD_S * ratio * WORM_EFFICIENCY
    mass = TREAD_MOTOR_KG_BASE + TREAD_MOTOR_KG_PER_W * pw + WORM_KG + SHIN_BELT_KG
    return TreadDriveSpec(pw, ratio, r, mass, v_max, tau_pulley / r, tau_pulley)
