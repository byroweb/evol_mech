"""Parametric MJCF for the crank-driven, fan-assisted biped.

Topology (fixed):
    hip (free, steppers inside)
      ├─ core (yaw hinge ±30°) ── payload mass, fan_l, fan_r (pitch hinges, thrust sites)
      ├─ crank_l (stepper hinge) ── coupler_l (passive hinge) ── foot pad
      ├─ rocker_l (passive hinge)      ⟵ connect constraint closes loop at B
      └─ same for the right leg

Conventions: +x forward, +y left, +z up. Positive fan pitch tilts thrust
forward. Linkage members do not collide with anything; only foot pads, hip,
and core touch the floor.
"""
from __future__ import annotations

import math

from .hardware import (fan_from_params, stepper_from_params,
                       FAN_PITCH_SERVO_KP, FAN_PITCH_SERVO_TORQUE)
from . import linkage

SIM_DT = 0.002          # physics step
YAW_RANGE = math.radians(30)
FAN_PITCH_RANGE = math.radians(90)
HIP_HALF_X = 0.05
HIP_HALF_Z = 0.035      # hip box extends this far below the crank pivot A
PIVOT_MARGIN = 0.015    # hip frame extends this far past pivot B


def hip_extent(p: dict[str, float]) -> tuple[float, float]:
    """(half-width in x, top z) of the hip frame, relative to crank pivot A.

    Both fixed pivots A and B are mounted on the hip, so the hip frame grows
    to enclose B; the core is seated on top of that frame.
    """
    hx = max(HIP_HALF_X, abs(p["pivot_b_dx"]) + PIVOT_MARGIN)
    top = max(HIP_HALF_Z, p["pivot_b_dz"] + PIVOT_MARGIN)
    return hx, top
LINK_RADIUS = 0.006
NOCOL = 'contype="0" conaffinity="0"'

ACTUATOR_NAMES = [
    "crank_l", "crank_r",
    "core_yaw",
    "fan_pitch_l", "fan_pitch_r",
    "fan_thrust_l", "fan_thrust_r",
]


class InvalidBody(ValueError):
    """Raised when the genome decodes to a body that cannot be built."""


def nominal_leg_len(p: dict[str, float]) -> float:
    """Distance from crank pivot down to the lowest foot position."""
    return -linkage.lowest_foot_z(p) + p["foot_radius"]


def spawn_height(p: dict[str, float], terrain_zmax: float = 0.0, theta_l: float = 0.0) -> float:
    """Hip height so the lower of the two feet in the initial pose just clears the floor."""
    a = linkage.solve(p, theta_l)
    b = linkage.solve(p, theta_l + p["crank_phase_r"])
    if a is None or b is None:
        raise InvalidBody("linkage does not close in the initial pose")
    return -min(a.P[1], b.P[1]) + p["foot_radius"] + 0.01 + terrain_zmax


def _leg(side: str, sign: float, p: dict[str, float], theta0: float, crank_torque: float) -> str:
    pose = linkage.solve(p, theta0)
    if pose is None:
        raise InvalidBody(f"linkage does not close at theta0={theta0:.2f}")
    A, B, F = pose.D, pose.C, pose.P           # crank tip, coupler joint, foot
    Bp = (p["pivot_b_dx"], p["pivot_b_dz"])
    y = sign * p["hip_width"] / 2
    bx, bz = B[0] - A[0], B[1] - A[1]          # C in coupler frame (origin D)
    fx, fz = F[0] - A[0], F[1] - A[1]          # P in coupler frame
    rx, rz = B[0] - Bp[0], B[1] - Bp[1]        # C in rocker frame (origin B)
    return f"""
      <body name="crank_{side}" pos="0 {y:.4f} 0">
        <joint name="crank_{side}" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 {A[0]:.4f} 0 {A[1]:.4f}" size="{LINK_RADIUS}"
              rgba="0.9 0.3 0.3 1" {NOCOL}/>
        <body name="coupler_{side}" pos="{A[0]:.4f} 0 {A[1]:.4f}">
          <joint name="coupler_{side}" type="hinge" axis="0 1 0" damping="0.01"/>
          <geom type="capsule" fromto="0 0 0 {bx:.4f} 0 {bz:.4f}" size="{LINK_RADIUS}"
                rgba="0.3 0.5 0.9 1" {NOCOL}/>
          <geom type="capsule" fromto="0 0 0 {fx:.4f} 0 {fz:.4f}" size="{LINK_RADIUS}"
                rgba="0.3 0.5 0.9 1" {NOCOL}/>
          <geom type="capsule" fromto="{bx:.4f} 0 {bz:.4f} {fx:.4f} 0 {fz:.4f}" size="{LINK_RADIUS}"
                rgba="0.3 0.5 0.9 1" {NOCOL}/>
          <geom name="foot_{side}" type="capsule" size="{p['foot_radius']:.4f}"
                fromto="{fx-p['foot_half_len']:.4f} 0 {fz:.4f} {fx+p['foot_half_len']:.4f} 0 {fz:.4f}"
                friction="1.2 0.005 0.0001" rgba="0.15 0.15 0.15 1"/>
        </body>
      </body>
      <geom type="capsule" fromto="0 {y:.4f} 0 {Bp[0]:.4f} {y:.4f} {Bp[1]:.4f}" size="0.004"
            rgba="0.5 0.5 0.5 1" {NOCOL}/>
      <body name="rocker_{side}" pos="{Bp[0]:.4f} {y:.4f} {Bp[1]:.4f}">
        <joint name="rocker_{side}" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 {rx:.4f} 0 {rz:.4f}" size="{LINK_RADIUS}"
              rgba="0.3 0.8 0.4 1" {NOCOL}/>
      </body>"""


def _fan(side: str, sign: float, p: dict[str, float], mass: float, diam: float) -> str:
    return f"""
        <body name="fan_{side}" pos="{p['fan_x']:.4f} {sign*p['fan_y']:.4f} {p['fan_z']:.4f}">
          <joint name="fan_pitch_{side}" type="hinge" axis="0 1 0"
                 range="{-FAN_PITCH_RANGE:.4f} {FAN_PITCH_RANGE:.4f}" damping="0.02"/>
          <geom type="cylinder" size="{diam/2:.4f} 0.008" mass="{mass:.4f}"
                rgba="0.9 0.6 0.2 0.7" {NOCOL}/>
          <site name="fan_thrust_{side}" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
        </body>"""


def build_mjcf(p: dict[str, float], terrain_asset: str, terrain_geom: str,
               terrain_zmax: float = 0.0, theta_l: float = 0.0) -> str:
    """theta_l: initial left crank angle; the right crank is theta_l + crank_phase_r."""
    if not linkage.is_valid(p):
        raise InvalidBody("linkage is not a rotatable crank-rocker")
    fan = fan_from_params(p)
    step = stepper_from_params(p)
    hip_mass = p["hip_struct_mass"] + 2 * step.mass_kg
    hy = p["hip_width"] / 2 + 0.01      # hip spans the two leg planes
    hx, hip_top = hip_extent(p)
    hip_cz = (hip_top - HIP_HALF_Z) / 2
    hip_hz = (hip_top + HIP_HALF_Z) / 2
    core_z = hip_top + p["core_height"] / 2
    payload_z = p["payload_zf"] * p["core_height"]
    theta_r = theta_l + p["crank_phase_r"]
    z0 = spawn_height(p, terrain_zmax, theta_l)

    return f"""<mujoco model="evol_mech_biped">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{SIM_DT}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <joint armature="0.001"/>
    <geom condim="3" friction="1.0 0.005 0.0001" density="900"/>
    <equality solref="0.002 1"/>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.7 0.9" rgb2="0.9 0.9 0.9" width="32" height="32"/>
    {terrain_asset}
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    {terrain_geom}
    <body name="hip" pos="0 0 {z0:.4f}">
      <freejoint name="root"/>
      <geom name="hip" type="box" pos="0 0 {hip_cz:.4f}" size="{hx:.4f} {hy:.4f} {hip_hz:.4f}" mass="{hip_mass:.4f}"
            rgba="0.6 0.6 0.6 1"/>
      <body name="core" pos="0 0 {core_z:.4f}">
        <joint name="core_yaw" type="hinge" axis="0 0 1" range="{-YAW_RANGE:.4f} {YAW_RANGE:.4f}" damping="0.2"/>
        <geom name="core" type="box" size="0.05 0.05 {p['core_height']/2:.4f}" mass="{p['core_shell_mass']:.4f}"
              rgba="0.7 0.3 0.3 0.6"/>
        <geom name="payload" type="box" pos="{p['payload_x']:.4f} 0 {payload_z:.4f}" size="0.03 0.035 0.015"
              mass="{p['payload_mass']:.4f}" rgba="0.2 0.2 0.6 1" {NOCOL}/>
        {_fan('l', +1.0, p, fan.mass_kg, fan.diam_m)}
        {_fan('r', -1.0, p, fan.mass_kg, fan.diam_m)}
      </body>
      {_leg('l', +1.0, p, theta_l, step.crank_torque_nm)}
      {_leg('r', -1.0, p, theta_r, step.crank_torque_nm)}
    </body>
  </worldbody>
  <equality>
    <connect name="loop_l" body1="rocker_l" body2="coupler_l"
             anchor="{linkage.solve(p, theta_l).C[0]-p['pivot_b_dx']:.4f} 0 {linkage.solve(p, theta_l).C[1]-p['pivot_b_dz']:.4f}"/>
    <connect name="loop_r" body1="rocker_r" body2="coupler_r"
             anchor="{linkage.solve(p, theta_r).C[0]-p['pivot_b_dx']:.4f} 0 {linkage.solve(p, theta_r).C[1]-p['pivot_b_dz']:.4f}"/>
  </equality>
  <actuator>
    <velocity name="crank_l" joint="crank_l" kv="2.0"
              forcerange="{-step.crank_torque_nm:.4f} {step.crank_torque_nm:.4f}"/>
    <velocity name="crank_r" joint="crank_r" kv="2.0"
              forcerange="{-step.crank_torque_nm:.4f} {step.crank_torque_nm:.4f}"/>
    <position name="core_yaw" joint="core_yaw" kp="{p['kp_yaw']:.3f}" kv="{p['kp_yaw']*0.05:.3f}" forcerange="-3 3"/>
    <position name="fan_pitch_l" joint="fan_pitch_l" kp="{FAN_PITCH_SERVO_KP}" kv="0.1"
              forcerange="{-FAN_PITCH_SERVO_TORQUE} {FAN_PITCH_SERVO_TORQUE}"/>
    <position name="fan_pitch_r" joint="fan_pitch_r" kp="{FAN_PITCH_SERVO_KP}" kv="0.1"
              forcerange="{-FAN_PITCH_SERVO_TORQUE} {FAN_PITCH_SERVO_TORQUE}"/>
    <general name="fan_thrust_l" site="fan_thrust_l" gear="0 0 {fan.max_thrust_n:.3f} 0 0 0"
             ctrlrange="0 1" dyntype="filter" dynprm="{fan.tau_s:.3f}"
             gaintype="fixed" gainprm="1" biastype="none"/>
    <general name="fan_thrust_r" site="fan_thrust_r" gear="0 0 {fan.max_thrust_n:.3f} 0 0 0"
             ctrlrange="0 1" dyntype="filter" dynprm="{fan.tau_s:.3f}"
             gaintype="fixed" gainprm="1" biastype="none"/>
  </actuator>
</mujoco>
"""
