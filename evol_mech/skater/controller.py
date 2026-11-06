"""Stage-1 skater controller, archetype B (docs/skater_design.md "Controller",
docs/skater_review.md).

Layers, all gains from the genome, structure fixed:

1. Body planner: desired world velocity (vx, vy) and yaw rate from the task
   (straight skate, slalom path, or hold position for stand).
2. Force allocation: demanded horizontal force = m * vel_kp * velocity error.
   Treads take `tread_share` of the in-line component first, capped at
   `friction_margin * mu * N` per foot and at the tip-over limit; the fans take
   the remainder, all of it when airborne.
3. Steering: there is no hip yaw, so both treads always point along the
   body. Each tread runs at its own ground speed along x; the yaw rate the
   planner demands comes from the tread speed differential (a tank turn, so
   the rollers scrub) and from opposite fan pitch (fan_kp_yaw). The lateral
   velocity demand goes to the fans.
4. Legs: each leg is two servos in the hip setting the thigh angle and, through
   the shin drive four-bar (horn -> coupler -> tab on the shin), the shin angle
   relative to the body. The controller picks a foot target (x, z) under the
   hip, solves the two-link inverse kinematics, converts the shin angle to a
   horn angle for that thigh (structure.horn_from_shin), clamps the pair to
   the drive's hard stops (structure.drive_stops), slew-limits the thigh and
   horn servos, and sets the knee actuator to the shin angle the four-bar
   gives for the commanded thigh and horn. So a thigh command with a fixed
   horn swings the shin as the linkage would. The third hip servo is roll
   (LEO's abduction axis): the controller picks a lateral foot offset too and
   rolls the leg plane so the two-link IK, solved in that plane, lands the
   ankle there.
5. Balance: the ankle is passive, so pitch is held like a Segway: the treads
   are driven toward the fall (kp_bal, kd_bal) and the feet are shifted under
   the body (kp_foot_x, kd_foot_x). Roll is held the same way with the hip
   roll servos: the feet shift sideways toward the fall (kp_roll_y,
   kd_roll_y) and toward the lateral body motion (kv_foot_y, the capture
   point), and unequal leg height levels the body (kp_roll_lvl). A wider
   track (stance_w) is an evolved standing / landing margin. Fans add their
   pitch PD and roll differential on top.
6. Fans: each on its own pitch servo. Common pitch = allocation tilt +
   attitude PD; opposite pitch = yaw torque toward the demanded yaw rate;
   differential throttle = roll.

Treads are position-tracked (velocity command plus angle-error term) so a zero
command is a locked sole: the worm does not back-drive.

Legged objectives (walk, hop, broad_jump) lock the treads and move the foot
targets instead: a walk cycle (stance foot travels back under the hip at
step_hz for step_duty of the cycle, swing foot returns lifted by step_lift;
legs half a cycle apart) or
a jump cycle (crouch by jump_crouch, thrust to jump_ext above nominal with the
feet jump_lean behind the hip on the broad jump, tuck,
settle). Nothing is asked of the treads then (no lean reference); the
passive ankle is the only pitch pivot and its spring is softer than gravity,
so pitch is balanced by shifting the feet under the body (kp_foot_leg,
kd_foot_leg, with ki_foot_leg integrating out steady moments such as a fan
pitch bias or a forward payload) and ahead of it by kv_foot_x per m/s of
forward speed. In a
walk the body is also shifted sideways over the stance foot
(walk_shift, a sinusoid peaking at shift_phase of the step) through the hip
roll servos,
and the fan over the swing foot gets step_fan extra throttle at full lift so
that side is carried while the foot is off the ground.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..body.hardware import tread_drive_from_params, servo_from_params, fan_from_params
from ..sim.controller import quat_to_roll_pitch
from .mjcf import ACTUATOR_NAMES, FAN_PITCH_RANGE
from .structure import (nominal_angles, nominal_height, foot_from_angles, angles_from_foot,
                        horn_from_shin, shin_from_horn, drive_stops, HIP_ROLL_RANGE, KNEE_RANGE)

MU_FOOT = 1.2
G = 9.81
A_MAX = 4.0                 # m/s^2 cap on demanded body acceleration
TIP_MARGIN = 0.5            # treads may use this fraction of the tip-over acceleration
COM_ABOVE_HIP = 0.08        # rough centre-of-mass height above the hip joint (core + battery)
TREAD_TRACK_GAIN = 10.0     # 1/s, angle-error term of the tread position hold
AIRBORNE_Z = 0.03           # hip higher than nominal by this -> treat as airborne
CONTROL_DT = 0.01
LEGGED = ("walk", "hop", "broad_jump")      # objectives with locked treads and moving feet
# walk cycle: stance duty (`step_duty`) and the cycle fraction at which the body is
# furthest left (`shift_phase`) are genes; the left foot's phase is the cycle reference
JUMP_PHASES = (0.45, 0.60, 0.80)            # cycle fractions: crouch | thrust | tuck | settle
KNEE_FROM_MEASURED_THIGH = False  # knee target: four-bar at the commanded thigh. True = at the measured thigh,
                                  # closer to a rigid link but the two position loops then chatter (rms accel x3)


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
        # the shin drive: horn hard stops per thigh angle, and the horn that parks the shin
        self.stops = drive_stops(p)
        self.thigh_lim = self.stops.thigh_range()
        horn = horn_from_shin(p, thigh, shin)
        self.horn_nom = thigh + p["crouch"] if horn is None else horn
        self.servo_cmd = {f"{j}_{s}": (thigh if j == "hip_pitch" else self.horn_nom if j == "knee" else 0.0)
                          for s in ("l", "r") for j in ("hip_roll", "hip_pitch", "knee")}
        self.thigh_q = {s: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"hip_pitch_{s}")]
                        for s in ("l", "r")}
        reach = max(0.005, p["sole_len"] / 2 - p["pulley_radius"])
        self.a_tip = TIP_MARGIN * G * reach / (self.z_ref + COM_ABOVE_HIP)
        self.objective = task["objective"]
        self.legged = self.objective in LEGGED
        self.speed = float(task.get("speed", 1.0))
        self.p0 = None
        self.last_throttle = (0.0, 0.0)
        self.last_fan_force = (0.0, 0.0)
        self.x_int = 0.0                             # legged: integrated foot shift (steady moments)

    # ---- 1. planner ------------------------------------------------------------
    def plan(self, pos: np.ndarray, yaw: float) -> tuple[np.ndarray, float]:
        p, obj = self.p, self.objective
        if obj in ("skate", "walk", "broad_jump"):
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
        else:                                       # stand, hop: hold the spawn point
            v = np.clip(-0.5 * (pos[:2] - self.p0[:2]), -0.3, 0.3)
            heading = 0.0
        return v, p["steer_kp"] * wrap(heading - yaw)

    # ---- legged cycles -------------------------------------------------------------
    def gait_offsets(self, t: float) -> dict[str, tuple[float, float, bool]]:
        """Per side: (foot x offset, foot lift, height feedback on) for the legged
        objectives; zero for the others."""
        p = self.p
        if self.objective == "walk":
            out = {}
            for s, ph in (("l", 0.0), ("r", 0.5)):
                u = (p["step_hz"] * t + ph) % 1.0
                L = p["step_len"]
                duty = p["step_duty"]
                if u < duty:                         # stance: foot travels back under the hip
                    out[s] = (L / 2 - L * u / duty, 0.0, True)
                else:                                # swing: lifted return
                    w = (u - duty) / (1 - duty)
                    out[s] = (-L / 2 + L * w, p["step_lift"] * math.sin(math.pi * w), True)
            return out
        if self.objective in ("hop", "broad_jump"):
            u = (p["jump_hz"] * t) % 1.0
            c, th, tk = JUMP_PHASES
            lean = p["jump_lean"] if self.objective == "broad_jump" else 0.0
            if u < c:                                # crouch: ease down
                off = (0.0, -p["jump_crouch"] * min(1.0, u / (0.5 * c)), True)
            elif u < th:                             # thrust: full extension, feet back for a forward jump
                off = (-lean, p["jump_ext"], False)
            elif u < tk:                             # flight: tuck
                off = (0.0, -0.5 * p["jump_crouch"], False)
            else:                                    # settle
                off = (0.0, 0.0, True)
            return {"l": off, "r": off}
        return {"l": (0.0, 0.0, True), "r": (0.0, 0.0, True)}

    def body_shift(self, t: float) -> float:
        """Walk: lateral body offset (+ = left) over the feet, furthest toward the
        stance side mid-swing of the other foot; zero for the other objectives."""
        if self.objective != "walk":
            return 0.0
        u = (self.p["step_hz"] * t) % 1.0
        amp = self.p["walk_shift"] * self.p["hip_width"] / 2          # 1.0 = hip over the stance foot
        return amp * math.cos(2 * math.pi * (u - self.p["shift_phase"]))

    def _slew(self, name: str, target: float) -> float:
        step = self.servo.max_rad_s * CONTROL_DT
        cur = self.servo_cmd[name]
        cur += max(-step, min(step, target - cur))
        self.servo_cmd[name] = cur
        return cur

    def _drive_commands(self, s: str, thigh: float, shin: float) -> tuple[float, float]:
        """Thigh and horn servo commands for a desired (thigh, shin): the thigh is
        kept where the drive has a working band, the shin is converted to a horn
        angle through the four-bar and clamped to the band's hard stops, then
        both are slew-limited."""
        p = self.p
        thigh = max(self.thigh_lim[0], min(self.thigh_lim[1], thigh))
        horn = horn_from_shin(p, thigh, shin)
        band = self.stops.band(thigh)
        if band is not None:
            if horn is None:            # out of reach: the band end whose shin is nearest the request
                ends = [(abs((r[0] if (r := shin_from_horn(p, thigh, h)) else 1e9) - shin), h) for h in band]
                horn = min(ends)[1]
            horn = max(band[0], min(band[1], horn))
        elif horn is None:
            horn = self.servo_cmd[f"knee_{s}"]
        thigh_cmd = self._slew(f"hip_pitch_{s}", thigh)
        horn_cmd = self._slew(f"knee_{s}", horn)
        return thigh_cmd, horn_cmd

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
        # legged: the soles are locked, so nothing is asked of the treads and there is no lean
        a_cap = 0.0 if (airborne or self.legged) else min(self.a_tip, p["friction_margin"] * MU_FOOT * G)
        a_tread = p["tread_share"] * max(-a_cap, min(a_cap, float(a_des[0])))
        h_com = float(pos[2]) + COM_ABOVE_HIP
        d_lean = a_tread * h_com / G                      # CoM ahead of the contact for this acceleration
        d_feet = max(-0.06, min(0.06, p["lean_gain"] * a_tread))
        pitch_ref = (d_lean - d_feet) / h_com
        # legged on the ground: the fans do not push forward against locked soles (that
        # just topples the body); progress comes from the steps. Airborne they take it all.
        push = 0.0 if (self.legged and not airborne) else 1.0
        f_fan = self.mass * np.array([push * (float(a_des[0]) - a_tread), float(a_des[1])])
        self.last_fan_force = (float(f_fan[0]), float(f_fan[1]))

        # 3 + 5. steering and Segway balance through the treads: each tread runs
        # at its own ground speed (along the body, there is no hip yaw) plus a
        # term driving toward the fall, plus the differential that turns the
        # body at the demanded yaw rate.
        w = p["hip_width"] / 2
        v_bal = p["kp_bal"] * (pitch - pitch_ref) + p["kd_bal"] * pitch_rate
        for s, y_f in (("l", w), ("r", -w)):
            v_ground = float(v_b[0]) - yaw_rate * y_f
            v_turn = -(omega_des - yaw_rate) * y_f
            v_cmd = 0.0 if self.legged else v_ground + v_bal + v_turn      # legged: soles locked
            v_cmd = max(-self.tread.max_speed_m_s, min(self.tread.max_speed_m_s, v_cmd))
            om_cmd = v_cmd / self.tread.pulley_r
            self.tread_ref[s] += om_cmd * CONTROL_DT
            q = float(data.qpos[self.roller_q[s]])
            self.tread_ref[s] = q + max(-0.5, min(0.5, self.tread_ref[s] - q))
            c[i[f"tread_{s}"]] = om_cmd + TREAD_TRACK_GAIN * (self.tread_ref[s] - q)

        # 4 + 5. legs: foot targets under the hip -> hip roll + two-servo inverse
        # kinematics in the rolled leg plane. Feet shift back for the lean and
        # forward under a nose-down body; sideways toward a roll (positive roll
        # = right side down) and away from the body's walk shift.
        h_fb = p["kp_height"] * (self.z_ref - float(pos[2])) - p["kd_height"] * float(v_world[2])
        if self.legged:
            # on locked soles the passive ankle is the only pitch pivot and its spring is
            # softer than gravity, so the feet are the balance: shifted under the body by
            # the legged pitch PD and ahead of it when it runs away forward
            self.x_int = max(-0.05, min(0.05, self.x_int + p["ki_foot_leg"] * pitch * CONTROL_DT))
            x_f = (self.x_nom + p["kp_foot_leg"] * pitch + p["kd_foot_leg"] * pitch_rate + self.x_int
                   + p["kv_foot_x"] * float(v_b[0]))
        else:
            x_f = (self.x_nom - d_feet
                   + p["kp_foot_x"] * (pitch - pitch_ref) + p["kd_foot_x"] * pitch_rate)
        dh_roll = p["kp_roll_lvl"] * roll + p["kd_roll_lvl"] * roll_rate   # left side up -> shorten left
        # feet toward the fall and toward the lateral motion: with kp_roll_y ~ leg drop and
        # kv_foot_y ~ sqrt(drop / g) the swing foot lands on the capture point
        dy_f = (-self.body_shift(t) - p["kp_roll_y"] * roll - p["kd_roll_y"] * roll_rate
                + p["kv_foot_y"] * float(v_b[1]))
        gait = self.gait_offsets(t)
        for s, sign in (("l", +1.0), ("r", -1.0)):
            gx, gz, fb = gait[s]
            h = max(0.02, self.h_nom + (h_fb if fb else 0.0) + gz - sign * dh_roll)
            dy = dy_f + sign * p["stance_w"]                      # + = foot to the left, both legs
            roll_cmd = max(-HIP_ROLL_RANGE, min(HIP_ROLL_RANGE, math.atan2(dy, h)))
            c[i[f"hip_roll_{s}"]] = self._slew(f"hip_roll_{s}", roll_cmd)
            thigh, shin = angles_from_foot(p, x_f + gx, -math.hypot(h, dy))
            thigh_cmd, horn_cmd = self._drive_commands(s, thigh, shin)
            c[i[f"hip_pitch_{s}"]] = thigh_cmd
            # the knee follows the four-bar for the thigh as it actually is
            thigh_ref = float(data.qpos[self.thigh_q[s]]) if KNEE_FROM_MEASURED_THIGH else thigh_cmd
            r = shin_from_horn(p, thigh_ref, horn_cmd)
            if r is None:
                r = shin_from_horn(p, thigh_cmd, horn_cmd)
            shin_act = r[0] if r is not None else thigh_cmd + p["crouch"]
            c[i[f"knee_{s}"]] = max(KNEE_RANGE[0], min(KNEE_RANGE[1], shin_act - thigh_ref))

        # 6. fans: two pitch channels
        if self.fans_enabled:
            diff = p["fan_kp_roll"] * roll + p["fan_kd_roll"] * roll_rate
            # swing-side assist: the fan over a lifted foot carries that side (feedforward
            # on the gait phase, scaled by how far the foot is lifted), the roll PD does the rest
            lift = p["step_lift"] if self.objective == "walk" else 1.0
            assist_l, assist_r = (p["step_fan"] * max(0.0, g[1]) / lift for g in (gait["l"], gait["r"]))
            thr_l = max(0.0, min(1.0, p["fan_base"] - diff + assist_l))
            thr_r = max(0.0, min(1.0, p["fan_base"] + diff + assist_r))
            t_vert = max(0.05, p["fan_base"]) * self.t_max
            common = (math.atan2(f_fan[0] / 2, t_vert) + p["fan_pitch_bias"]
                      - p["fan_kp_pitch"] * (pitch - pitch_ref) - p["fan_kd_pitch"] * pitch_rate)
            d_yaw = p["fan_kp_yaw"] * (omega_des - yaw_rate)
            fan_l = max(-FAN_PITCH_RANGE, min(FAN_PITCH_RANGE, common - d_yaw))
            fan_r = max(-FAN_PITCH_RANGE, min(FAN_PITCH_RANGE, common + d_yaw))
        else:
            thr_l = thr_r = fan_l = fan_r = 0.0
        self.dbg = {"pitch": pitch, "pitch_ref": pitch_ref, "a_des": float(a_des[0]), "a_tread": a_tread,
                    "d_feet": d_feet, "v_bal": v_bal, "x_f": x_f, "h": self.h_nom + h_fb, "dy_f": dy_f,
                    "f_fan_x": float(f_fan[0]),
                    "fan_l": fan_l, "fan_r": fan_r, "thr": (thr_l, thr_r)}
        c[i["fan_thrust_l"]], c[i["fan_thrust_r"]] = thr_l, thr_r
        c[i["fan_pitch_l"]], c[i["fan_pitch_r"]] = fan_l, fan_r
        self.last_throttle = (thr_l, thr_r)
        return c
