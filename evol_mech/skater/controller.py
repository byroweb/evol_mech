"""Stage-1 skater controller (docs/skater_design.md, "Controller").

Five layers, all gains from the genome, structure fixed:

1. Body planner: desired world velocity (vx, vy), yaw rate and height from the
   task (straight skate, crab, slalom path, or hold position for stand).
2. Force allocation: demanded horizontal force = m * vel_kp * velocity error.
   Treads take `tread_share` of the in-line component first, capped at
   `friction_margin * mu * N` per foot; the fans take the remainder, all of
   the lateral component the treads cannot give, and all of it when airborne.
3. Steering: one turning centre. Each foot's commanded ground velocity is
   v + omega x r_foot; its hip yaw points the tread along it (Ackermann for
   two side-by-side feet) and its tread speed is that velocity's magnitude.
   Straight ahead both feet are exactly parallel. Yaw is limited to +-45 deg in
   stage 1 (steering only); the "crab" objective still exists but is not in the
   stage-1 task set.
4. Suspension and leveling: knee PD on body height, hip pitch PD on body
   pitch about a lean reference (lean into demanded acceleration), hip roll
   PD on body roll, ankle keeps the sole level with the ground.
5. Fan attitude: the biped's PD loops on top of the allocation tilt.

Treads are position-tracked (velocity command plus angle-error term) so a zero
command is a locked sole: the worm does not back-drive.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..body.hardware import tread_drive_from_params
from ..sim.controller import quat_to_roll_pitch
from .mjcf import ACTUATOR_NAMES, FAN_PITCH_RANGE, FAN_ROLL_RANGE
from .structure import nominal_angles, nominal_height, HIP_YAW_RANGE

MU_FOOT = 1.2
G = 9.81
A_MAX = 4.0                 # m/s^2 cap on demanded body acceleration
TIP_MARGIN = 0.5            # treads may use this fraction of the tip-over acceleration
COM_ABOVE_HIP = 0.08        # rough centre-of-mass height above the hip joint (core + battery)
TREAD_TRACK_GAIN = 10.0     # 1/s, angle-error term of the tread position hold
TREAD_LEAD_S = 0.15         # force demand expressed as a speed lead on the tread
AIRBORNE_Z = 0.03           # hip higher than nominal by this -> treat as airborne


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
        self.roller_q = {s: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"roller_{s}0")]
                         for s in ("l", "r")}
        self.tread_ref = {"l": 0.0, "r": 0.0}       # tracked roller angle
        self.z_ref = nominal_height(p)
        # tip-over limit of force through the feet: beyond g * (front roller
        # reach) / (hip height) the front rollers lift. The fans take the excess.
        reach = max(0.005, p["sole_len"] / 2 - p["pulley_radius"])
        self.a_tip = TIP_MARGIN * G * reach / (self.z_ref + COM_ABOVE_HIP)
        self.hip_nom, self.knee_nom, _ = nominal_angles(p)
        self.objective = task["objective"]
        self.speed = float(task.get("speed", 1.0))
        self.p0 = None
        self.last_throttle = (0.0, 0.0)
        self.last_fan_force = (0.0, 0.0)
        self.foot_yaw = {"l": 0.0, "r": 0.0}

    # ---- 1. planner ------------------------------------------------------------
    def plan(self, pos: np.ndarray, yaw: float) -> tuple[np.ndarray, float]:
        """Desired world velocity and yaw rate."""
        p, obj = self.p, self.objective
        if obj == "skate":
            v = np.array([self.speed, 0.0])
            heading = 0.0
        elif obj == "crab":
            v = np.array([0.0, self.speed])
            heading = 0.0                           # body keeps facing +x
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
        omega = p["steer_kp"] * wrap(heading - yaw)
        return v, omega

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
        roll_rate, pitch_rate = float(data.qvel[3]), float(data.qvel[4])
        cy, sy = math.cos(yaw), math.sin(yaw)

        # 1. planner (world) -> body yaw frame
        v_des, omega_des = self.plan(pos, yaw)
        v_err = v_des - v_world[:2]
        a_des_w = np.clip(p["vel_kp"] * v_err, -A_MAX, A_MAX)
        a_des = np.array([cy * a_des_w[0] + sy * a_des_w[1], -sy * a_des_w[0] + cy * a_des_w[1]])
        v_des_b = np.array([cy * v_des[0] + sy * v_des[1], -sy * v_des[0] + cy * v_des[1]])
        v_b = np.array([cy * v_world[0] + sy * v_world[1], -sy * v_world[0] + cy * v_world[1]])
        yaw_rate = float(data.qvel[5])
        f_des = self.mass * a_des

        # 3. steering: one turning centre -> foot yaw and tread speeds
        w = p["hip_width"] / 2
        airborne = pos[2] > self.z_ref + AIRBORNE_Z
        n_foot = 0.0 if airborne else self.mass * G / 2
        f_cap = min(p["friction_margin"] * MU_FOOT * n_foot, self.mass * self.a_tip / 2)
        f_tread_total = np.zeros(2)
        for s, y_f in (("l", w), ("r", -w)):
            u = np.array([v_des_b[0] - omega_des * y_f, v_des_b[1]])
            if np.hypot(*u) < 0.02:
                delta, speed = self.foot_yaw[s], 0.0
            else:
                delta = math.atan2(u[1], u[0])
                speed = float(np.hypot(*u))
                if abs(delta) > math.pi / 2:        # drive the tread backwards instead of over-yawing
                    delta, speed = wrap(delta + math.pi), -speed
            # stage 1: yaw is for steering only (+-45 deg); a demand beyond that is
            # clipped and the tread only gets the component along the clipped heading
            delta = max(-HIP_YAW_RANGE, min(HIP_YAW_RANGE, delta))
            if speed:
                speed = float(np.hypot(*u)) * math.cos(math.atan2(u[1], u[0]) - delta) * (1 if speed > 0 else -1)
            self.foot_yaw[s] = delta
            e = np.array([math.cos(delta), math.sin(delta)])
            # 2. allocation: in-line force this foot can give
            f_inline = p["tread_share"] * float(f_des @ e) / 2
            f_inline = max(-f_cap, min(f_cap, f_inline))
            f_tread_total += f_inline * e
            # tread speed = the ground speed this foot already has along its tread,
            # plus a lead proportional to the force asked of it: the tread is a
            # force actuator with slip control, never a step to the target speed.
            # Worm-locked (position hold) when both are zero.
            v_ground = float(np.array([v_b[0] - yaw_rate * y_f, v_b[1]]) @ e)
            v_cmd = v_ground + TREAD_LEAD_S * f_inline / max(self.mass / 2, 1e-6)
            v_cmd = max(-self.tread.max_speed_m_s, min(self.tread.max_speed_m_s, v_cmd))
            om_cmd = v_cmd / self.tread.pulley_r
            self.tread_ref[s] += om_cmd * 0.01
            q = float(data.qpos[self.roller_q[s]])
            # keep the reference within reach so a stalled tread does not wind up
            self.tread_ref[s] = q + max(-0.5, min(0.5, self.tread_ref[s] - q))
            c[i[f"tread_{s}"]] = om_cmd + TREAD_TRACK_GAIN * (self.tread_ref[s] - q)
            c[i[f"hip_yaw_{s}"]] = delta
        f_fan = f_des - f_tread_total
        self.last_fan_force = (float(f_fan[0]), float(f_fan[1]))

        # 4. suspension and leveling
        z_err = float(pos[2] - self.z_ref)
        knee = self.knee_nom + p["kp_height"] * z_err + p["kd_height"] * float(v_world[2])
        knee = max(0.05, min(2.5, knee))
        L1, L2 = p["thigh_len"], p["shin_len"]
        hip_under = -math.atan2(L2 * math.sin(knee), L1 + L2 * math.cos(knee))
        # lean into the demanded acceleration: swing the feet back (positive hip
        # pitch) so the inertial force lands the centre of pressure mid-sole
        lean = p["lean_gain"] * float(a_des[0])
        pitch_ref = 0.0
        hip = hip_under + lean - p["kp_pitch_lvl"] * (pitch - pitch_ref) - p["kd_pitch_lvl"] * pitch_rate
        hip_roll = -p["kp_roll_lvl"] * roll - p["kd_roll_lvl"] * roll_rate
        # sole square to the leg plus the ankle balance strategy (toe down when nose
        # down). No world-level feed-forward: with the sole planted that term pushes
        # the body the wrong way; on slopes the ankle complies through its PD instead.
        ankle = -(hip + knee) + p["kp_ankle_lvl"] * (pitch - pitch_ref) + p["kd_ankle_lvl"] * pitch_rate
        for s in ("l", "r"):
            c[i[f"hip_roll_{s}"]] = hip_roll
            c[i[f"hip_pitch_{s}"]] = hip
            c[i[f"knee_{s}"]] = knee
            c[i[f"ankle_{s}"]] = ankle

        # 5. fans: allocation tilt + attitude PD
        if self.fans_enabled:
            diff = p["fan_kp_roll"] * roll + p["fan_kd_roll"] * roll_rate
            thr_l = max(0.0, min(1.0, p["fan_base"] - diff))
            thr_r = max(0.0, min(1.0, p["fan_base"] + diff))
            t_vert = max(0.05, p["fan_base"]) * self._t_max(data)
            tilt_pitch = math.atan2(f_fan[0] / 2, t_vert)
            tilt_roll = -math.atan2(f_fan[1] / 2, t_vert)
            fan_pitch = tilt_pitch + p["fan_pitch_bias"] - p["fan_kp_pitch"] * pitch - p["fan_kd_pitch"] * pitch_rate
            fan_pitch = max(-FAN_PITCH_RANGE, min(FAN_PITCH_RANGE, fan_pitch))
            fan_roll = max(-FAN_ROLL_RANGE, min(FAN_ROLL_RANGE, tilt_roll))
        else:
            thr_l = thr_r = 0.0
            fan_pitch = fan_roll = 0.0
        c[i["fan_thrust_l"]], c[i["fan_thrust_r"]] = thr_l, thr_r
        c[i["fan_pitch_l"]] = c[i["fan_pitch_r"]] = fan_pitch
        c[i["fan_roll_l"]] = c[i["fan_roll_r"]] = fan_roll
        self.last_throttle = (thr_l, thr_r)
        return c

    _t_max_cache: float | None = None

    def _t_max(self, data: mujoco.MjData) -> float:
        if self._t_max_cache is None:
            from ..body.hardware import fan_from_params
            self._t_max_cache = fan_from_params(self.p).max_thrust_n
        return self._t_max_cache
