"""Stage-1 skater controller, archetype B (docs/skater_design.md "Controller",
docs/skater_review.md).

Layers, all gains from the genome, structure fixed:

1. Body planner: desired world velocity (vx, vy) and yaw rate from the task
   (straight skate, slalom path, or hold position for stand).
2. Force allocation: demanded horizontal force = m * vel_kp * velocity error.
   Treads take `tread_share` of the in-line component first, capped at
   `friction_margin * mu * N` per foot and at the tip-over limit; the fans take
   the remainder, all of it when airborne.
3. Steering: one turning centre. Each foot's commanded ground velocity is
   v + omega x r_foot; hip yaw points the tread along it (+-45 deg) and the
   tread speed is that velocity's magnitude. Straight ahead both feet are
   exactly parallel.
4. Legs: each leg is two servos in the hip setting the thigh and shin angles
   relative to the body (the shin through the parallelogram). The controller
   picks a foot target (x, z) under the hip and solves the two-link inverse
   kinematics, so height, lean and roll levelling are foot positions, not
   joint angles. Servo commands are slew-limited to the servo's rate.
5. Balance: the ankle is passive, so pitch is held like a Segway: the treads
   are driven toward the fall (kp_bal, kd_bal) and the feet are shifted under
   the body (kp_foot_x, kd_foot_x). Fans add their pitch PD on top.
6. Fans: each on its own pitch servo. Common pitch = allocation tilt +
   attitude PD; opposite pitch = yaw torque toward the demanded yaw rate;
   differential throttle = roll.

Treads are position-tracked (velocity command plus angle-error term) so a zero
command is a locked sole: the worm does not back-drive.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..body.hardware import tread_drive_from_params, servo_from_params, fan_from_params
from ..sim.controller import quat_to_roll_pitch
from .mjcf import ACTUATOR_NAMES, FAN_PITCH_RANGE
from .structure import (nominal_angles, nominal_height, foot_from_angles, angles_from_foot,
                        HIP_YAW_RANGE)

MU_FOOT = 1.2
G = 9.81
A_MAX = 4.0                 # m/s^2 cap on demanded body acceleration
TIP_MARGIN = 0.5            # treads may use this fraction of the tip-over acceleration
COM_ABOVE_HIP = 0.08        # rough centre-of-mass height above the hip joint (core + battery)
TREAD_TRACK_GAIN = 10.0     # 1/s, angle-error term of the tread position hold
AIRBORNE_Z = 0.03           # hip higher than nominal by this -> treat as airborne
CONTROL_DT = 0.01


def quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class SkaterController:
    def __init__(self, model: mujoco.MjModel, p: dict[str, float], task: dict,
                 fans_enabled: bool = True):
        self.p, self.task = p, task
        self.fans_enabled = fans_enabled
        self.idx = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES}
        self.ctrl = np.zeros(model.nu)
        hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip")
        self.mass = float(model.body_subtreemass[hip_id])
        self.tread = tread_drive_from_params(p)
        self.servo = servo_from_params(p)
        self.t_max = fan_from_params(p).max_thrust_n
        self.roller_q = {s: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"roller_{s}0")]
                         for s in ("l", "r")}
        self.tread_ref = {"l": 0.0, "r": 0.0}       # tracked roller angle
        self.z_ref = nominal_height(p)
        thigh, shin = nominal_angles(p)
        self.x_nom, z_nom = foot_from_angles(p, thigh, shin)
        self.h_nom = -z_nom                          # hip-to-ankle drop in the parked crouch
        self.servo_cmd = {f"{j}_{s}": (thigh if j == "hip_pitch" else shin - thigh if j == "knee" else 0.0)
                          for s in ("l", "r") for j in ("hip_yaw", "hip_pitch", "knee")}
        reach = max(0.005, p["sole_len"] / 2 - p["pulley_radius"])
        self.a_tip = TIP_MARGIN * G * reach / (self.z_ref + COM_ABOVE_HIP)
        self.objective = task["objective"]
        self.speed = float(task.get("speed", 1.0))
        self.p0 = None
        self.last_throttle = (0.0, 0.0)
        self.last_fan_force = (0.0, 0.0)
        self.foot_yaw = {"l": 0.0, "r": 0.0}

    # ---- 1. planner ------------------------------------------------------------
    def plan(self, pos: np.ndarray, yaw: float) -> tuple[np.ndarray, float]:
        p, obj = self.p, self.objective
        if obj == "skate":
            v, heading = np.array([self.speed, 0.0]), 0.0
        elif obj == "crab":                          # kept for later stages; not in the stage-1 task set
            v, heading = np.array([0.0, self.speed]), 0.0
        elif obj == "slalom":
            A, lam = float(self.task.get("amplitude", 0.5)), float(self.task.get("wavelength", 3.0))
            k = 2 * math.pi / lam
            y_ref = A * math.sin(k * pos[0])
            slope = A * k * math.cos(k * pos[0])
            heading = math.atan(slope) - math.atan(p["path_kp"] * (pos[1] - y_ref))
            v = self.speed * np.array([math.cos(heading), math.sin(heading)])
        else:                                       # stand: hold the spawn point
            v = np.clip(-0.5 * (pos[:2] - self.p0[:2]), -0.3, 0.3)
            heading = 0.0
        return v, p["steer_kp"] * wrap(heading - yaw)

    def _slew(self, name: str, target: float) -> float:
        step = self.servo.max_rad_s * CONTROL_DT
        cur = self.servo_cmd[name]
        cur += max(-step, min(step, target - cur))
        self.servo_cmd[name] = cur
        return cur

    # ---- main --------------------------------------------------------------------
    def act(self, data: mujoco.MjData, t: float) -> np.ndarray:
        p, c, i = self.p, self.ctrl, self.idx
        pos = data.qpos[:3].copy()
        if self.p0 is None:
            self.p0 = pos.copy()
        quat = data.qpos[3:7]
        roll, pitch = quat_to_roll_pitch(quat)
        yaw = quat_to_yaw(quat)
        v_world = data.qvel[:3]
        roll_rate, pitch_rate, yaw_rate = float(data.qvel[3]), float(data.qvel[4]), float(data.qvel[5])
        cy, sy = math.cos(yaw), math.sin(yaw)

        # 1. planner (world) -> body yaw frame
        v_des, omega_des = self.plan(pos, yaw)
        a_des_w = np.clip(p["vel_kp"] * (v_des - v_world[:2]), -A_MAX, A_MAX)
        rot = lambda u: np.array([cy * u[0] + sy * u[1], -sy * u[0] + cy * u[1]])
        a_des, v_des_b, v_b = rot(a_des_w), rot(v_des), rot(v_world[:2])
        f_des = self.mass * a_des

        # 2. allocation. The ankle is passive, so the treads can only push by
        # leaning: the in-line acceleration they are asked for (capped by the
        # tip-over limit) becomes a lean, split between shifting the feet back
        # under the body (lean_gain) and pitching the body (the rest). The fans
        # take whatever acceleration is left, and all of the lateral demand.
        airborne = pos[2] > self.z_ref + AIRBORNE_Z
        a_cap = 0.0 if airborne else min(self.a_tip, p["friction_margin"] * MU_FOOT * G)
        a_tread = p["tread_share"] * max(-a_cap, min(a_cap, float(a_des[0])))
        h_com = float(pos[2]) + COM_ABOVE_HIP
        d_lean = a_tread * h_com / G                      # CoM ahead of the contact for this acceleration
        d_feet = max(-0.06, min(0.06, p["lean_gain"] * a_tread))
        pitch_ref = (d_lean - d_feet) / h_com
        f_fan = self.mass * np.array([float(a_des[0]) - a_tread, float(a_des[1])])
        self.last_fan_force = (float(f_fan[0]), float(f_fan[1]))

        # 3 + 5. steering and Segway balance through the treads: each tread runs
        # at its own ground speed plus a term driving toward the fall, plus the
        # differential that turns the body at the demanded yaw rate.
        w = p["hip_width"] / 2
        v_bal = p["kp_bal"] * (pitch - pitch_ref) + p["kd_bal"] * pitch_rate
        for s, y_f in (("l", w), ("r", -w)):
            u = np.array([v_des_b[0] - omega_des * y_f, v_des_b[1]])
            if np.hypot(*u) < 0.02:
                delta = self.foot_yaw[s]
            else:
                delta = math.atan2(u[1], u[0])
                if abs(delta) > math.pi / 2:
                    delta = wrap(delta + math.pi)
            delta = max(-HIP_YAW_RANGE, min(HIP_YAW_RANGE, delta))
            self.foot_yaw[s] = delta
            e = np.array([math.cos(delta), math.sin(delta)])
            v_ground = float(np.array([v_b[0] - yaw_rate * y_f, v_b[1]]) @ e)
            v_turn = -(omega_des - yaw_rate) * y_f
            v_cmd = v_ground + v_bal + v_turn
            v_cmd = max(-self.tread.max_speed_m_s, min(self.tread.max_speed_m_s, v_cmd))
            om_cmd = v_cmd / self.tread.pulley_r
            self.tread_ref[s] += om_cmd * CONTROL_DT
            q = float(data.qpos[self.roller_q[s]])
            self.tread_ref[s] = q + max(-0.5, min(0.5, self.tread_ref[s] - q))
            c[i[f"tread_{s}"]] = om_cmd + TREAD_TRACK_GAIN * (self.tread_ref[s] - q)
            c[i[f"hip_yaw_{s}"]] = self._slew(f"hip_yaw_{s}", delta)

        # 4 + 5. legs: foot targets under the hip -> two-servo inverse kinematics.
        # Feet shift back for the lean and forward under a nose-down body.
        h = self.h_nom + p["kp_height"] * (self.z_ref - float(pos[2])) - p["kd_height"] * float(v_world[2])
        x_f = (self.x_nom - d_feet
               + p["kp_foot_x"] * (pitch - pitch_ref) + p["kd_foot_x"] * pitch_rate)
        dh_roll = p["kp_roll_lvl"] * roll + p["kd_roll_lvl"] * roll_rate   # left side up -> shorten left
        for s, sign in (("l", +1.0), ("r", -1.0)):
            thigh, shin = angles_from_foot(p, x_f, -(h - sign * dh_roll))
            c[i[f"hip_pitch_{s}"]] = self._slew(f"hip_pitch_{s}", thigh)
            c[i[f"knee_{s}"]] = self._slew(f"knee_{s}", shin - thigh)

        # 6. fans: two pitch channels
        if self.fans_enabled:
            diff = p["fan_kp_roll"] * roll + p["fan_kd_roll"] * roll_rate
            thr_l = max(0.0, min(1.0, p["fan_base"] - diff))
            thr_r = max(0.0, min(1.0, p["fan_base"] + diff))
            t_vert = max(0.05, p["fan_base"]) * self.t_max
            common = (math.atan2(f_fan[0] / 2, t_vert) + p["fan_pitch_bias"]
                      - p["fan_kp_pitch"] * (pitch - pitch_ref) - p["fan_kd_pitch"] * pitch_rate)
            d_yaw = p["fan_kp_yaw"] * (omega_des - yaw_rate)
            fan_l = max(-FAN_PITCH_RANGE, min(FAN_PITCH_RANGE, common - d_yaw))
            fan_r = max(-FAN_PITCH_RANGE, min(FAN_PITCH_RANGE, common + d_yaw))
        else:
            thr_l = thr_r = fan_l = fan_r = 0.0
        self.dbg = {"pitch": pitch, "pitch_ref": pitch_ref, "a_des": float(a_des[0]), "a_tread": a_tread,
                    "d_feet": d_feet, "v_bal": v_bal, "x_f": x_f, "h": h, "f_fan_x": float(f_fan[0]),
                    "fan_l": fan_l, "fan_r": fan_r, "thr": (thr_l, thr_r)}
        c[i["fan_thrust_l"]], c[i["fan_thrust_r"]] = thr_l, thr_r
        c[i["fan_pitch_l"]], c[i["fan_pitch_r"]] = fan_l, fan_r
        self.last_throttle = (thr_l, thr_r)
        return c
