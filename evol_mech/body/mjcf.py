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
from .clearance import interference
from .structure import (structure_from_params, hip_half_y, fan_center_y, payload_z as _payload_z, HIP_HALF_X, HIP_HALF_Z,
                        CORE_HALF_XY, LINK_RADIUS)

SIM_DT = 0.002          # physics step
YAW_RANGE = math.radians(30)
FAN_PITCH_RANGE = math.radians(90)
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
          {_foot(side, p, fx, fz)}
        </body>
      </body>
      <geom type="capsule" fromto="0 {y:.4f} 0 {Bp[0]:.4f} {y:.4f} {Bp[1]:.4f}" size="0.004"
            rgba="0.5 0.5 0.5 1" {NOCOL}/>
      <body name="rocker_{side}" pos="{Bp[0]:.4f} {y:.4f} {Bp[1]:.4f}">
        <joint name="rocker_{side}" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 {rx:.4f} 0 {rz:.4f}" size="{LINK_RADIUS}"
              rgba="0.3 0.8 0.4 1" {NOCOL}/>
      </body>"""


def _foot(side: str, p: dict[str, float], fx: float, fz: float) -> str:
    """Ball end (rotation-invariant contact) or flat pad along the coupler."""
    if p["foot_type"] >= 0.5:
        return (f'<geom name="foot_{side}" type="sphere" size="{p["foot_radius"]:.4f}" pos="{fx:.4f} 0 {fz:.4f}" '
                f'friction="1.2 0.005 0.0001" rgba="0.15 0.15 0.15 1"/>')
    return (f'<geom name="foot_{side}" type="capsule" size="{p["foot_radius"]:.4f}" '
            f'fromto="{fx-p["foot_half_len"]:.4f} 0 {fz:.4f} {fx+p["foot_half_len"]:.4f} 0 {fz:.4f}" '
            f'friction="1.2 0.005 0.0001" rgba="0.15 0.15 0.15 1"/>')


def _fan(side: str, sign: float, p: dict[str, float], mass: float, diam: float) -> str:
    fz = p["fan_z"] - p["core_height"] / 2        # fan_z is measured from the core bottom
    fy = sign * fan_center_y(p)                   # fan_gap is prop edge to core side
    return f"""
        <body name="fan_{side}" pos="{p['fan_x']:.4f} {fy:.4f} {fz:.4f}">
          <joint name="fan_pitch_{side}" type="hinge" axis="0 1 0"
                 range="{-FAN_PITCH_RANGE:.4f} {FAN_PITCH_RANGE:.4f}" damping="0.02"/>
          <geom type="cylinder" size="{diam/2:.4f} 0.008" mass="{mass:.4f}"
                rgba="0.9 0.6 0.2 0.7" {NOCOL}/>
          <site name="fan_thrust_{side}" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
        </body>"""


def build_mjcf(p: dict[str, float], terrain_asset: str, terrain_geom: str,
               terrain_zmax: float = 0.0, theta_l: float = 0.0, waist_yaw: bool = False) -> str:
    """theta_l: initial left crank angle; the right crank is theta_l + crank_phase_r.
    waist_yaw: put the core on a +-30 deg yaw hinge (else rigid to the hip)."""
    if not linkage.is_valid(p):
        raise InvalidBody("linkage is not a rotatable crank-rocker")
    hit = interference(p, YAW_RANGE if waist_yaw else 0.0)
    if hit is not None:
        raise InvalidBody(hit[0])
    fan = fan_from_params(p)
    step = stepper_from_params(p)
    struct = structure_from_params(p)
    hip_mass = struct.hip_kg + 2 * step.mass_kg
    hy = hip_half_y(p)                  # legs run just outside the hip's side faces
    core_z = HIP_HALF_Z + p["core_height"] / 2
    px, py, pz = struct.payload_half
    yaw_joint = (f'<joint name="core_yaw" type="hinge" axis="0 0 1" '
                 f'range="{-YAW_RANGE:.4f} {YAW_RANGE:.4f}" damping="0.2"/>' if waist_yaw else "")
    yaw_act = (f'<position name="core_yaw" joint="core_yaw" kp="{p["kp_yaw"]:.3f}" '
               f'kv="{p["kp_yaw"]*0.05:.3f}" forcerange="-3 3"/>' if waist_yaw else "")
    payload_z = _payload_z(p)
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
      <geom name="hip" type="box" size="{HIP_HALF_X} {hy:.4f} {HIP_HALF_Z}" mass="{hip_mass:.4f}"
            rgba="0.6 0.6 0.6 1"/>
      <body name="core" pos="0 0 {core_z:.4f}">
        {yaw_joint}
        <geom name="core" type="box" size="{struct.core_half_x:.4f} {CORE_HALF_XY} {p['core_height']/2:.4f}" mass="{struct.core_kg:.4f}"
              rgba="0.7 0.3 0.3 0.6"/>
        <geom name="payload" type="box" pos="{p['payload_x']:.4f} 0 {payload_z:.4f}" size="{px:.4f} {py:.4f} {pz:.4f}"
              mass="{struct.payload_kg:.4f}" rgba="0.2 0.2 0.6 1" {NOCOL}/>
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
    {yaw_act}
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
