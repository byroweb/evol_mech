"""Two-layer controller.

Gait layer: each stepper spins its crank at a constant commanded angular
velocity (the 4-bar linkage turns that into a foot path). The core yaw pivot
oscillates in step with the left crank. Fan layer: PD attitude loop mapping
body roll to differential throttle and body pitch to fan pitch. All gains live
in the genome; the structure is fixed.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..body.mjcf import ACTUATOR_NAMES, FAN_PITCH_RANGE
from ..body.hardware import stepper_from_params


STEPPER_TRACK_GAIN = 40.0   # 1/s; velocity cmd per rad of angle error


def quat_to_roll_pitch(q: np.ndarray) -> tuple[float, float]:
    """MuJoCo quaternion (w, x, y, z) -> (roll about x, pitch about y)."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, s)))
    return roll, pitch


class Controller:
    def __init__(self, model: mujoco.MjModel, p: dict[str, float],
                 fans_enabled: bool = True, gait_enabled: bool = True):
        self.p = p
        self.fans_enabled = fans_enabled
        self.gait_enabled = gait_enabled
        self.idx = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                    for n in ACTUATOR_NAMES}
        self.ctrl = np.zeros(model.nu)
        self.last_throttle = (0.0, 0.0)
        self.q_adr = {s: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"crank_{s}")]
                      for s in ("l", "r")}
        step = stepper_from_params(p)
        omega = 2 * math.pi * p["crank_speed_hz"]
        self.omega = max(-step.crank_max_rad_s, min(step.crank_max_rad_s, omega))

    def act(self, data: mujoco.MjData, t: float) -> np.ndarray:
        p, c, i = self.p, self.ctrl, self.idx

        # --- gait layer ---------------------------------------------------------
        # Steppers follow a commanded angle: velocity actuator + angle error term
        # gives a stiff position hold that stalls at the torque limit, like a
        # real stepper. Reference starts at the built pose (qpos=0).
        omega = self.omega if self.gait_enabled else 0.0
        q_ref = omega * t
        for s in ("l", "r"):
            err = q_ref - float(data.qpos[self.q_adr[s]])
            c[i[f"crank_{s}"]] = omega + STEPPER_TRACK_GAIN * err
        if i["core_yaw"] >= 0:   # waist yaw is optional (config "waist_yaw")
            c[i["core_yaw"]] = p["yaw_amp"] * math.sin(omega * t) if self.gait_enabled else 0.0

        # --- fan layer: PD attitude loop -------------------------------------
        if self.fans_enabled:
            roll, pitch = quat_to_roll_pitch(data.qpos[3:7])
            roll_rate, pitch_rate = float(data.qvel[3]), float(data.qvel[4])
            diff = p["fan_kp_roll"] * roll + p["fan_kd_roll"] * roll_rate
            thr_l = max(0.0, min(1.0, p["fan_base"] - diff))
            thr_r = max(0.0, min(1.0, p["fan_base"] + diff))
            fan_pitch = p["fan_pitch_bias"] - p["fan_kp_pitch"] * pitch - p["fan_kd_pitch"] * pitch_rate
            fan_pitch = max(-FAN_PITCH_RANGE, min(FAN_PITCH_RANGE, fan_pitch))
        else:
            thr_l = thr_r = 0.0
            fan_pitch = 0.0
        c[i["fan_thrust_l"]] = thr_l
        c[i["fan_thrust_r"]] = thr_r
        c[i["fan_pitch_l"]] = fan_pitch
        c[i["fan_pitch_r"]] = fan_pitch
        self.last_throttle = (thr_l, thr_r)
        return c
