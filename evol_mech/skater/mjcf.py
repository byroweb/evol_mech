"""Parametric MJCF for the skater mech, archetype B (docs/skater_design.md,
docs/skater_review.md).

Topology (fixed):
    hip (free; six leg servos inside)
      ├─ core ── payload, fan_l / fan_r (own pitch hinge each -> thrust site)
      ├─ leg_l: hip_yaw (z) -> thigh (hip_pitch, y) -> knee (y) -> shin
      │           (+ tread drive at the knee) -> ankle (y, passive spring)
      │           -> sole plate + roller row + raised front idler (tread)
      │              or heel / toe wheels
      └─ leg_r: same

The parallelogram knee drive is not built as links: the knee hinge carries a
position actuator whose target is set by the controller as (shin command -
thigh command), which is what the parallelogram does geometrically. The
coupler's mass is in the thigh.

Conventions: +x forward, +y left, +z up. Hinges about +y swing the distal
link backward for positive angles. All model joints are zero in the straight
leg; `nominal_qpos` gives the parked crouch.

Tread model (MuJoCo has no tracks): the rollers and the idler are hinged
cylinders tied together by joint equalities and driven by one velocity
actuator on the first roller, with the worm's pulley torque as its limit. The
controller holds the first roller's angle when the tread command is zero, so a
stopped tread is a locked sole (non-back-drivable worm).

Contacts: only sole plates, rollers and idlers (named foot_*) may touch the
floor. Links, the tread drive, fans and payload carry contype 0 /
conaffinity 2 so the floor (contype 3) still hits them and the rollout can
call it a fall.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..body.hardware import (fan_from_params, servo_from_params, tread_drive_from_params,
                             FAN_PITCH_SERVO_KP, FAN_PITCH_SERVO_TORQUE)
from ..body.structure import (hip_half_y, fan_center_y, fan_z_core, payload_z as _payload_z,
                              HIP_HALF_X, HIP_HALF_Z)
from .structure import (structure_from_params, foot_layout, nominal_angles, spawn_height,
                        check_buildable, InvalidBody, LINK_RADIUS, SOLE_HALF_W, SOLE_HALF_T,
                        ROLLER_HALF_LEN, HIP_YAW_RANGE, THIGH_RANGE, KNEE_RANGE, ANKLE_RANGE)

SIM_DT = 0.002
FAN_PITCH_RANGE = math.radians(90)
NOCOL = 'contype="0" conaffinity="2"'
FOOT_ATTR = 'friction="1.2 0.005 0.0001" rgba="0.15 0.15 0.15 1"'
TREAD_TRACK_KV = 0.1            # velocity actuator gain on the first roller (Nm s/rad)
SERVO_KV_FRAC = 0.02            # servo damping as a fraction of its kp (s)

LEG_JOINTS = ["hip_yaw", "hip_pitch", "knee"]           # the three servos; the ankle is passive
ACTUATOR_NAMES = ([f"{j}_{s}" for s in ("l", "r") for j in LEG_JOINTS]
                  + ["tread_l", "tread_r", "fan_pitch_l", "fan_pitch_r", "fan_thrust_l", "fan_thrust_r"])


def _foot(side: str, p: dict[str, float], mass_kg: float) -> str:
    lay = foot_layout(p)
    rollers = [(k, x, lay.roller_z) for k, x in enumerate(lay.roller_x)]
    if lay.idler:
        rollers.append((len(rollers), lay.idler[0], lay.roller_z + lay.idler[1]))
    n = len(rollers)
    plate_mass = mass_kg * 0.5
    roller_mass = mass_kg * 0.5 / n
    body = "".join(
        f"""
            <body name="roller_{side}{k}" pos="{x:.4f} 0 {z:.4f}">
              <joint name="roller_{side}{k}" type="hinge" axis="0 1 0" damping="0.0005" armature="0.00002"/>
              <geom name="foot_{side}_r{k}" type="cylinder" zaxis="0 1 0"
                    size="{lay.roller_r:.4f} {ROLLER_HALF_LEN}" mass="{roller_mass:.4f}" {FOOT_ATTR}/>
            </body>"""
        for k, x, z in rollers)
    return f"""
            <geom name="foot_{side}_sole" type="box" pos="0 0 {lay.plate_z:.4f}"
                  size="{p['sole_len']/2:.4f} {SOLE_HALF_W} {SOLE_HALF_T}" mass="{plate_mass:.4f}"
                  friction="1.2 0.005 0.0001" rgba="0.25 0.25 0.28 1"/>{body}"""


def n_rollers(p: dict[str, float]) -> int:
    lay = foot_layout(p)
    return len(lay.roller_x) + (1 if lay.idler else 0)


def _leg(side: str, sign: float, p: dict[str, float], st) -> str:
    y = sign * p["hip_width"] / 2
    L1, L2 = p["thigh_len"], p["shin_len"]
    tread = tread_drive_from_params(p)
    # passive ankle: spring rest = sole level at the parked crouch, toe up by ankle_rest
    # (positive ankle angle = toe down, so the rest angle is -shin - ankle_rest)
    _, shin_nom = nominal_angles(p)
    ankle_rest = -shin_nom - p["ankle_rest"]
    return f"""
      <body name="hip_yaw_{side}" pos="0 {y:.4f} 0">
        <joint name="hip_yaw_{side}" type="hinge" axis="0 0 1" range="{-HIP_YAW_RANGE:.4f} {HIP_YAW_RANGE:.4f}" damping="0.05"/>
        <geom type="sphere" size="0.012" mass="0.001" rgba="0.5 0.5 0.5 1" {NOCOL}/>
        <body name="thigh_{side}">
          <joint name="hip_pitch_{side}" type="hinge" axis="0 1 0" range="{THIGH_RANGE[0]:.4f} {THIGH_RANGE[1]:.4f}" damping="0.05"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-L1:.4f}" size="{LINK_RADIUS}" mass="{st.thigh_kg*0.6:.4f}"
                rgba="0.3 0.5 0.9 1" {NOCOL}/>
          <geom name="coupler_{side}" type="capsule" fromto="-0.03 0 0 -0.03 0 {-L1:.4f}" size="0.003" mass="{st.thigh_kg*0.4:.4f}"
                rgba="0.9 0.3 0.3 1" {NOCOL}/>
          <body name="shin_{side}" pos="0 0 {-L1:.4f}">
            <joint name="knee_{side}" type="hinge" axis="0 1 0" range="{KNEE_RANGE[0]:.4f} {KNEE_RANGE[1]:.4f}" damping="0.05"/>
            <geom type="capsule" fromto="0 0 0 0 0 {-L2:.4f}" size="{LINK_RADIUS}"
                  mass="{st.shin_kg - tread.mass_kg:.4f}" rgba="0.3 0.8 0.4 1" {NOCOL}/>
            <geom name="tread_drive_{side}" type="box" pos="0.018 0 -0.03" size="0.012 0.012 0.02"
                  mass="{tread.mass_kg:.4f}" rgba="0.9 0.3 0.3 1" {NOCOL}/>
            <body name="foot_{side}" pos="0 0 {-L2:.4f}">
              <joint name="ankle_{side}" type="hinge" axis="0 1 0" range="{ankle_rest-ANKLE_RANGE:.4f} {ankle_rest+ANKLE_RANGE:.4f}"
                     stiffness="{p['ankle_stiffness']:.3f}" springref="{ankle_rest:.4f}"
                     damping="{0.1 * p['ankle_stiffness']:.4f}"/>
              {_foot(side, p, st.foot_kg)}
            </body>
          </body>
        </body>
      </body>"""


def _fan(side: str, sign: float, p: dict[str, float], mass: float, diam: float) -> str:
    fz = fan_z_core(p)
    fy = sign * fan_center_y(p)
    return f"""
        <body name="fan_{side}" pos="{p['fan_x']:.4f} {fy:.4f} {fz:.4f}">
          <joint name="fan_pitch_{side}" type="hinge" axis="0 1 0"
                 range="{-FAN_PITCH_RANGE:.4f} {FAN_PITCH_RANGE:.4f}" damping="0.02"/>
          <geom type="cylinder" size="{diam/2:.4f} 0.008" mass="{mass:.4f}"
                rgba="0.9 0.6 0.2 0.7" {NOCOL}/>
          <site name="fan_thrust_{side}" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
        </body>"""


def _leg_actuators(side: str, p: dict[str, float]) -> str:
    servo = servo_from_params(p)
    tread = tread_drive_from_params(p)
    t, kp, kv = servo.torque_nm, servo.kp, servo.kp * SERVO_KV_FRAC
    rows = [f'<position name="{j}_{side}" joint="{j}_{side}" kp="{kp:.2f}" kv="{kv:.3f}" '
            f'forcerange="{-t:.3f} {t:.3f}"/>' for j in LEG_JOINTS]
    rows.append(f'<velocity name="tread_{side}" joint="roller_{side}0" kv="{TREAD_TRACK_KV}" '
                f'forcerange="{-tread.pulley_torque_nm:.4f} {tread.pulley_torque_nm:.4f}"/>')
    return "\n    ".join(rows)


def _roller_couplings(side: str, p: dict[str, float]) -> str:
    return "\n    ".join(
        f'<joint name="tread_{side}{k}" joint1="roller_{side}{k}" joint2="roller_{side}0" polycoef="0 1 0 0 0"/>'
        for k in range(1, n_rollers(p)))


def build_mjcf(p: dict[str, float], terrain_asset: str, terrain_geom: str,
               terrain_zmax: float = 0.0) -> str:
    check_buildable(p)
    fan = fan_from_params(p)
    st = structure_from_params(p)
    hy = hip_half_y(p)
    core_z = HIP_HALF_Z + p["core_height"] / 2
    px, py, pz = st.payload_half
    z0 = spawn_height(p, terrain_zmax)
    return f"""<mujoco model="evol_mech_skater">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{SIM_DT}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="3"/>
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
      <geom name="hip" type="box" size="{HIP_HALF_X} {hy:.4f} {HIP_HALF_Z}" mass="{st.hip_kg:.4f}"
            rgba="0.6 0.6 0.6 1"/>
      <body name="core" pos="0 0 {core_z:.4f}">
        <geom name="core" type="box" size="{st.core_half_x:.4f} {st.core_half_y:.4f} {p['core_height']/2:.4f}"
              mass="{st.core_kg:.4f}" rgba="0.7 0.3 0.3 0.6"/>
        <geom name="payload" type="box" pos="{p['payload_x']:.4f} 0 {_payload_z(p):.4f}" size="{px:.4f} {py:.4f} {pz:.4f}"
              mass="{st.payload_kg:.4f}" rgba="0.2 0.2 0.6 1" {NOCOL}/>
        {_fan('l', +1.0, p, st.fan_kg, fan.diam_m)}
        {_fan('r', -1.0, p, st.fan_kg, fan.diam_m)}
      </body>
      {_leg('l', +1.0, p, st)}
      {_leg('r', -1.0, p, st)}
    </body>
  </worldbody>
  <equality>
    {_roller_couplings('l', p)}
    {_roller_couplings('r', p)}
  </equality>
  <actuator>
    {_leg_actuators('l', p)}
    {_leg_actuators('r', p)}
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


def nominal_qpos(model: mujoco.MjModel, p: dict[str, float]) -> np.ndarray:
    """qpos of the parked crouch: root as built, thigh and knee at the nominal
    angles, ankle at zero (sole square to the shin; the spring settles it),
    treads and fans at zero."""
    q = model.qpos0.copy()
    thigh, shin = nominal_angles(p)
    for s in ("l", "r"):
        for name, val in (("hip_pitch", thigh), ("knee", shin - thigh), ("ankle", -shin)):
            adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_{s}")]
            q[adr] = val
    return q
