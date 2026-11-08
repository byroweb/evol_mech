"""Parametric MJCF for the skater mech, archetype B (docs/skater_design.md,
docs/skater_review.md).

Topology (fixed):
    hip (free; six leg servos inside)
      ├─ core ── payload, fan_l / fan_r (own pitch hinge each -> thrust site)
      ├─ leg_l: hip_roll (x) -> thigh (hip_pitch, y) -> knee (y) -> shin
      │           (+ tread drive at the knee) -> ankle (y and x, passive springs)
      │           -> sole plate + roller row + raised front idler (tread)
      │              or heel / toe wheels
      └─ leg_r: same

The parallelogram knee drive is not built as links: the knee hinge carries a
position actuator whose target is set by the controller as (shin command -
thigh command), which is what the parallelogram does geometrically. The
coupler's mass is in the thigh. The six servos are drawn (pink) where they
sit, the two roll servos at the hip centre with their axes forward, one behind
the other, each with its horn and a link down to its leg's bracket (a light
passive loop that follows the roll), and each leg's two servos on a bracket
(purple) on the roll body, so they swing with the leg plane as the sketch's
front view has it; the servo mass stays in the hip box (see _bracket), which
is drawn translucent so the cluster shows.

Pantograph leg (leg_drive 1, structure.py "Pantograph leg"): built as the real
closed chain. Under the roll bracket, horn_f (joint hip_pitch, the front servo)
-> rod_f -> link_c -> foot, and horn_r (joint knee, the rear servo) -> rod_r ->
link_b; a connect equality pins the rods together at the knee node and another
pins link_c to link_b at the ankle node. Every body is placed at the parked
crouch with its joint `ref` set to that angle, so qpos reads the same
body-relative angles the kinematics use and the loops close at qpos0. The two
servo actuators sit on the horn hinges, so the load split between the servos
is whatever the linkage makes it. Its roll pivots are at the hip centre
(ROLL_PIVOT_Y) and the bracket runs out to the leg plane. The hip roll lever is
not built as a load path: its ratio (structure.roll_ratio: the bracket arm over
the horn on the pantograph, the roll_ratio gene on the four-bar) is reflected
onto the roll hinge's actuator (stiffness x ratio^2, torque x ratio, range /
ratio); the drawn horn and link are a light passive loop. Flexing horns (the horn_k_log10 gene, one
part for all three) are the servo's position stiffness in series with the
horn's spring.

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

from ..body.hardware import (fan_from_params, servo_from_params, roll_servo_from_params, tread_drive_from_params,
                             FAN_PITCH_SERVO_KP, FAN_PITCH_SERVO_TORQUE)
from ..body.structure import (hip_half_y, fan_center_y, fan_z_core, payload_z as _payload_z,
                              HIP_HALF_X, HIP_HALF_Z)
from .structure import (structure_from_params, foot_layout, nominal_angles, spawn_height,
                        check_buildable, InvalidBody, LINK_RADIUS, SOLE_HALF_W, SOLE_HALF_T,
                        roller_half_len, sole_half_w, HIP_ROLL_RANGE, THIGH_RANGE, KNEE_RANGE, ANKLE_RANGE,
                        ANKLE_ROLL_RANGE, ANKLE_ROLL_STIFF, HORN_RANGE, HORN_KG, PIN_KG, JOINT_BRACKET_KG,
                        LINK_KG_PER_M, is_pantograph, panto_nominal, panto_points, hip_roll_range,
                        horn_stiffness, series_kp, SHIN_DRIVE, roll_arm, roll_ratio, ROLL_PIVOT_Y,
                        BRACKET_Z, ROLL_LINK_LEN)

SIM_DT = 0.002
FAN_PITCH_RANGE = math.radians(90)
NOCOL = 'contype="0" conaffinity="2"'
FOOT_ATTR = 'friction="1.2 0.005 0.0001" rgba="0.15 0.15 0.15 1"'
TREAD_TRACK_KV = 0.1            # velocity actuator gain on the first roller (Nm s/rad)
SERVO_KV_FRAC = 0.02            # servo damping as a fraction of its kp (s)

LEG_JOINTS = ["hip_roll", "hip_pitch", "knee"]          # the three servos; the ankle is passive
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
                    size="{lay.roller_r:.4f} {roller_half_len(p):.4f}" mass="{roller_mass:.4f}" {FOOT_ATTR}/>
            </body>"""
        for k, x, z in rollers)
    return f"""
            <geom name="foot_{side}_sole" type="box" pos="0 0 {lay.plate_z:.4f}"
                  size="{p['sole_len']/2:.4f} {sole_half_w(p):.4f} {SOLE_HALF_T}" mass="{plate_mass:.4f}"
                  friction="1.2 0.005 0.0001" rgba="0.25 0.25 0.28 1"/>{body}"""


def n_rollers(p: dict[str, float]) -> int:
    lay = foot_layout(p)
    return len(lay.roller_x) + (1 if lay.idler else 0)


def _ankle(side: str, p: dict[str, float], st, ankle_rest: float, ref: float = 0.0, euler: str = "") -> str:
    """The passive two-axis sprung ankle and the foot under it. `ref` / `euler` place the
    foot level at the parked crouch when its parent is not the shin (pantograph)."""
    return f"""
            <body name="foot_{side}" {euler}>
              <joint name="ankle_{side}" type="hinge" axis="0 1 0" ref="{ref:.4f}" range="{ankle_rest-ANKLE_RANGE:.4f} {ankle_rest+ANKLE_RANGE:.4f}"
                     stiffness="{p['ankle_stiffness']:.3f}" springref="{ankle_rest:.4f}"
                     damping="{0.1 * p['ankle_stiffness']:.4f}"/>
              <joint name="ankle_roll_{side}" type="hinge" axis="1 0 0" range="{-ANKLE_ROLL_RANGE:.4f} {ANKLE_ROLL_RANGE:.4f}"
                     stiffness="{ANKLE_ROLL_STIFF * p['ankle_stiffness']:.3f}" springref="0"
                     damping="{0.1 * ANKLE_ROLL_STIFF * p['ankle_stiffness']:.4f}"/>
              {_foot(side, p, st.foot_kg)}
            </body>"""


SERVO_HALF = (0.020, 0.010, 0.020)      # a 40 x 20 x 40 mm hobby servo, its output shaft at the bottom of its long face


VIS_KG = 1e-4                           # the roll drive's drawn horn and link (they must move, so they must weigh something)


def _bracket(side: str, sign: float, p: dict[str, float], st) -> str:
    """The leg bracket on the roll body: the two leg servos at the leg plane and the bar
    they sit on. Pantograph body: the pivot is at the hip centre and the bar runs out to
    the leg plane (roll_arm), the lever the roll drive's vertical link pulls on at its end,
    the sketch's front view. Four-bar body: the pivot is at the leg plane, so a tab hangs
    roll_ratio horn lengths below it for a horizontal link. Drawn, not weighed: the servo
    mass stays lumped in the hip box as every evolved gait was tuned to (moving 4 x 65 g
    onto the brackets dropped the walker, runner and courser), so moving it is a body
    change for a re-evolution, not a rendering fix."""
    xs = (p["servo_dx"] / 2, -p["servo_dx"] / 2) if is_pantograph(p) else (0.0, SHIN_DRIVE["dx"])
    arm = roll_arm(p)
    yl = sign * arm                                     # the leg plane in the roll body
    yi = yl - sign * 0.014                              # servo bodies just inboard of it, the horns in it
    servos = "".join(
        f"""
        <geom name="servo_{side}{k}" type="box" pos="{x:.4f} {yi:.4f} {SERVO_HALF[2]:.4f}" size="{SERVO_HALF[0]} {SERVO_HALF[1]} {SERVO_HALF[2]}"
              mass="0" rgba="1 0.6 0.75 1" {NOCOL}/>"""
        for k, x in enumerate(xs))
    half_x = max(0.03, abs(xs[0] - xs[1]) / 2 + SERVO_HALF[0])
    if is_pantograph(p):
        return f"""
        <geom name="bracket_{side}" type="box" pos="{(xs[0] + xs[1]) / 2:.4f} {yl / 2:.4f} {BRACKET_Z}" size="{half_x:.4f} {arm / 2 + 0.012:.4f} 0.004"
              mass="0" rgba="0.55 0.2 0.6 1" {NOCOL}/>{servos}"""
    tab = float(p.get("roll_ratio", 1.0)) * p["horn_len"]
    return f"""
        <geom name="bracket_{side}" type="box" pos="{(xs[0] + xs[1]) / 2:.4f} {yi / 2:.4f} {BRACKET_Z}" size="{half_x:.4f} 0.012 0.004"
              mass="0" rgba="0.55 0.2 0.6 1" {NOCOL}/>
        <geom name="tab_{side}" type="box" pos="{sign * 0.022:.4f} 0 {-tab / 2:.4f}" size="0.006 0.004 {tab / 2:.4f}"
              mass="0" rgba="0.55 0.2 0.6 1" {NOCOL}/>{servos}"""


def _roll_drive(side: str, sign: float, p: dict[str, float]) -> str:
    """The roll servo on the hip (the two sit near the hip centre, one behind the other on
    their common forward axis) with its horn and link, a passive, light loop closed by a
    connect equality so it follows the leg's roll: the real lever's motion, drawn. Its
    force on the leg is the actuator's business (reflected on the roll hinge), not this
    linkage's. Pantograph body: horn horizontal, pointing outboard, and a vertical link
    from its tip down to the bracket's outboard end, so a degree of roll turns the horn
    by arm / horn_len. Four-bar body: horn vertical and a horizontal link to the tab."""
    h, x_s = p["horn_len"], sign * 0.022
    if is_pantograph(p):
        arm = roll_arm(p)
        y_s = sign * (ROLL_PIVOT_Y + arm - h)           # horn tip above the bracket end
        z_s = BRACKET_Z + ROLL_LINK_LEN
        return f"""
      <geom name="roll_servo_{side}" type="box" pos="{x_s:.4f} {y_s:.4f} {z_s + 0.01:.4f}" size="0.02 0.01 0.02" mass="0" rgba="1 0.6 0.75 1" {NOCOL}/>
      <body name="roll_horn_{side}" pos="{x_s:.4f} {y_s:.4f} {z_s:.4f}">
        <joint name="roll_horn_{side}" type="hinge" axis="1 0 0" damping="0.001"/>
        <geom type="capsule" fromto="0 0 0 0 {sign * h:.4f} 0" size="0.004" mass="{VIS_KG}" rgba="0.95 0.85 0.1 1" {NOCOL}/>
        <body name="roll_link_{side}" pos="0 {sign * h:.4f} 0">
          <joint name="roll_link_{side}" type="hinge" axis="1 0 0" damping="0.001"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-ROLL_LINK_LEN}" size="0.003" mass="{VIS_KG}" rgba="0.9 0.2 0.2 1" {NOCOL}/>
        </body>
      </body>"""
    tab, hw = float(p.get("roll_ratio", 1.0)) * h, p["hip_width"] / 2
    z_s = h - tab                                       # horn tip level with the tab's end
    return f"""
      <geom name="roll_servo_{side}" type="box" pos="{x_s:.4f} 0 {z_s + 0.01:.4f}" size="0.02 0.01 0.02" mass="0" rgba="1 0.6 0.75 1" {NOCOL}/>
      <body name="roll_horn_{side}" pos="{x_s:.4f} 0 {z_s:.4f}">
        <joint name="roll_horn_{side}" type="hinge" axis="1 0 0" damping="0.001"/>
        <geom type="capsule" fromto="0 0 0 0 0 {-h:.4f}" size="0.004" mass="{VIS_KG}" rgba="0.95 0.85 0.1 1" {NOCOL}/>
        <body name="roll_link_{side}" pos="0 0 {-h:.4f}">
          <joint name="roll_link_{side}" type="hinge" axis="1 0 0" damping="0.001"/>
          <geom type="capsule" fromto="0 0 0 0 {sign * hw:.4f} 0" size="0.003" mass="{VIS_KG}" rgba="0.9 0.2 0.2 1" {NOCOL}/>
        </body>
      </body>"""


def _roll_loops(side: str, sign: float, p: dict[str, float]) -> str:
    """The roll drive's loop closure: the link's far end on the bracket (its outboard end on
    the pantograph body, the tab's end on the four-bar)."""
    if is_pantograph(p):
        return f'<connect name="roll_{side}" body1="roll_link_{side}" body2="hip_roll_{side}" anchor="0 0 {-ROLL_LINK_LEN}"/>'
    return f'<connect name="roll_{side}" body1="roll_link_{side}" body2="hip_roll_{side}" anchor="0 {sign * p["hip_width"] / 2:.4f} 0"/>'


def _panto_leg(side: str, sign: float, p: dict[str, float], st) -> str:
    """The pantograph leg under its roll bracket (see the module docstring): the roll
    pivot at the hip centre, the leg hanging from the bracket's end, roll_arm out."""
    y, yl = sign * ROLL_PIVOT_Y, sign * roll_arm(p)
    d, h, r, e, s = p["servo_dx"], p["horn_len"], p["rod_len"], p["rod_ext"], p["strut_len"]
    tread = tread_drive_from_params(p)
    hf, hr = panto_nominal(p)
    k = panto_points(p, hf, hr)
    rf, rr = k["rod_f"], k["rod_r"]                    # the rods' body-relative angles at the crouch
    ankle_rest = -rr - p["ankle_rest"]                 # the foot hangs from link_c, which lies along the rear rod
    rod_kg = LINK_KG_PER_M * (r + e) + PIN_KG / 2
    link_b_kg = LINK_KG_PER_M * e + PIN_KG
    link_c_kg = LINK_KG_PER_M * (e + s) + 2 * PIN_KG + JOINT_BRACKET_KG
    roll = hip_roll_range(p)
    return f"""
      <body name="hip_roll_{side}" pos="0 {y:.4f} 0">
        <joint name="hip_roll_{side}" type="hinge" axis="1 0 0" range="{-roll:.4f} {roll:.4f}" damping="0.05"/>
        <geom type="sphere" size="0.012" mass="0.001" rgba="0.5 0.5 0.5 1" {NOCOL}/>{_bracket(side, sign, p, st)}
        <body name="horn_f_{side}" pos="{d/2:.4f} {yl:.4f} 0" euler="0 {hf:.5f} 0">
          <joint name="hip_pitch_{side}" type="hinge" axis="0 1 0" ref="{hf:.5f}" range="{HORN_RANGE[0]:.4f} {HORN_RANGE[1]:.4f}" damping="0.05"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-h:.4f}" size="0.004" mass="{HORN_KG:.4f}" rgba="0.95 0.85 0.1 1" {NOCOL}/>
          <body name="rod_f_{side}" pos="0 0 {-h:.4f}" euler="0 {rf-hf:.5f} 0">
            <joint name="rod_f_{side}" type="hinge" axis="0 1 0" ref="{rf-hf:.5f}" damping="0.01"/>
            <geom type="capsule" fromto="0 0 0 0 0 {-(r+e):.4f}" size="{LINK_RADIUS}" mass="{rod_kg:.4f}" rgba="0.9 0.2 0.2 1" {NOCOL}/>
            <body name="link_c_{side}" pos="0 0 {-(r+e):.4f}" euler="0 {rr-rf:.5f} 0">
              <joint name="link_c_{side}" type="hinge" axis="0 1 0" ref="{rr-rf:.5f}" damping="0.01"/>
              <geom type="capsule" fromto="0 0 0 0 0 {-(e+s):.4f}" size="{LINK_RADIUS}" mass="{link_c_kg:.4f}" rgba="0.9 0.2 0.2 1" {NOCOL}/>
              <geom name="tread_drive_{side}" type="box" pos="0.018 0 -0.03" size="0.012 0.012 0.02"
                    mass="{tread.mass_kg:.4f}" rgba="0.9 0.3 0.3 1" {NOCOL}/>
              {_ankle(side, p, st, ankle_rest, ref=-rr, euler=f'pos="0 0 {-(e+s):.4f}" euler="0 {-rr:.5f} 0"')}
            </body>
          </body>
        </body>
        <body name="horn_r_{side}" pos="{-d/2:.4f} {yl:.4f} 0" euler="0 {hr:.5f} 0">
          <joint name="knee_{side}" type="hinge" axis="0 1 0" ref="{hr:.5f}" range="{HORN_RANGE[0]:.4f} {HORN_RANGE[1]:.4f}" damping="0.05"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-h:.4f}" size="0.004" mass="{HORN_KG:.4f}" rgba="0.95 0.85 0.1 1" {NOCOL}/>
          <body name="rod_r_{side}" pos="0 0 {-h:.4f}" euler="0 {rr-hr:.5f} 0">
            <joint name="rod_r_{side}" type="hinge" axis="0 1 0" ref="{rr-hr:.5f}" damping="0.01"/>
            <geom type="capsule" fromto="0 0 0 0 0 {-(r+e):.4f}" size="{LINK_RADIUS}" mass="{rod_kg:.4f}" rgba="0.9 0.2 0.2 1" {NOCOL}/>
            <body name="link_b_{side}" pos="0 0 {-(r+e):.4f}" euler="0 {rf-rr:.5f} 0">
              <joint name="link_b_{side}" type="hinge" axis="0 1 0" ref="{rf-rr:.5f}" damping="0.01"/>
              <geom type="capsule" fromto="0 0 0 0 0 {-e:.4f}" size="{LINK_RADIUS}" mass="{link_b_kg:.4f}" rgba="0.9 0.2 0.2 1" {NOCOL}/>
            </body>
          </body>
        </body>
      </body>"""


def _panto_loops(side: str, p: dict[str, float]) -> str:
    """The two pins that close the pantograph: rods at the knee node, links at the ankle node."""
    if not is_pantograph(p):
        return ""
    return (f'<connect name="pin_{side}" body1="rod_f_{side}" body2="rod_r_{side}" anchor="0 0 {-p["rod_len"]:.4f}"/>\n    '
            f'<connect name="node_{side}" body1="link_c_{side}" body2="link_b_{side}" anchor="0 0 {-p["rod_ext"]:.4f}"/>')


def _leg(side: str, sign: float, p: dict[str, float], st) -> str:
    if is_pantograph(p):
        return _panto_leg(side, sign, p, st)
    y = sign * p["hip_width"] / 2
    L1, L2 = p["thigh_len"], p["shin_len"]
    tread = tread_drive_from_params(p)
    # passive ankle: spring rest = sole level at the parked crouch, toe up by ankle_rest
    # (positive ankle angle = toe down, so the rest angle is -shin - ankle_rest)
    _, shin_nom = nominal_angles(p)
    ankle_rest = -shin_nom - p["ankle_rest"]
    roll = hip_roll_range(p)
    return f"""
      <body name="hip_roll_{side}" pos="0 {y:.4f} 0">
        <joint name="hip_roll_{side}" type="hinge" axis="1 0 0" range="{-roll:.4f} {roll:.4f}" damping="0.05"/>
        <geom type="sphere" size="0.012" mass="0.001" rgba="0.5 0.5 0.5 1" {NOCOL}/>{_bracket(side, sign, p, st)}
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
            {_ankle(side, p, st, ankle_rest, euler=f'pos="0 0 {-L2:.4f}"')}
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
    servo, roll_servo = servo_from_params(p), roll_servo_from_params(p)
    tread = tread_drive_from_params(p)
    # each servo's position stiffness in series with its flexing horn; the roll servo (its own
    # spec) works through the bracket lever (structure.roll_ratio), reflected onto the roll hinge
    ratio = roll_ratio(p)
    kp_leg = series_kp(servo.kp, horn_stiffness(p))
    kp_roll = series_kp(roll_servo.kp, horn_stiffness(p)) * ratio ** 2
    gains = {"hip_roll": (kp_roll, roll_servo.torque_nm * ratio), "hip_pitch": (kp_leg, servo.torque_nm),
             "knee": (kp_leg, servo.torque_nm)}
    rows = [f'<position name="{j}_{side}" joint="{j}_{side}" kp="{gains[j][0]:.2f}" kv="{gains[j][0] * SERVO_KV_FRAC:.3f}" '
            f'forcerange="{-gains[j][1]:.3f} {gains[j][1]:.3f}"/>' for j in LEG_JOINTS]
    rows.append(f'<velocity name="tread_{side}" joint="roller_{side}0" kv="{TREAD_TRACK_KV}" '
                f'forcerange="{-tread.pulley_torque_nm:.4f} {tread.pulley_torque_nm:.4f}"/>')
    return "\n    ".join(rows)


def _roller_couplings(side: str, p: dict[str, float]) -> str:
    return "\n    ".join(
        f'<joint name="tread_{side}{k}" joint1="roller_{side}{k}" joint2="roller_{side}0" polycoef="0 1 0 0 0"/>'
        for k in range(1, n_rollers(p)))


def build_mjcf(p: dict[str, float], terrain_asset: str, terrain_geom: str,
               terrain_zmax: float = 0.0, scene_geoms: str = "") -> str:
    """scene_geoms: extra static worldbody geoms (task obstacles such as slalom cones)."""
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
    {scene_geoms}
    <body name="hip" pos="0 0 {z0:.4f}">
      <freejoint name="root"/>
      <geom name="hip" type="box" size="{HIP_HALF_X} {hy:.4f} {HIP_HALF_Z}" mass="{st.hip_kg:.4f}"
            rgba="0.6 0.6 0.6 0.35"/>{_roll_drive('l', +1.0, p)}{_roll_drive('r', -1.0, p)}
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
    {_panto_loops('l', p)}
    {_panto_loops('r', p)}
    {_roll_loops('l', +1.0, p)}
    {_roll_loops('r', -1.0, p)}
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
    if is_pantograph(p):            # built at the crouch: every loop joint's ref is its parked angle
        return q
    thigh, shin = nominal_angles(p)
    for s in ("l", "r"):
        for name, val in (("hip_pitch", thigh), ("knee", shin - thigh), ("ankle", -shin)):
            adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_{s}")]
            q[adr] = val
    return q
